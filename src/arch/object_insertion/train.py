import argparse
import logging
import os
import torch

from accelerate.utils import set_seed

from ...utils.config_loader import load_training_config
from ...utils.model_utils import ModelTrainManager
from ...utils.dataset_utils import get_dtype_training, collate_fn
from ...core.dataset import AbstractDataset
from .model import ObjectInsertionDiffusionModel

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Object insertion training entrypoint")
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config file")
    args_cli, unknown = parser.parse_known_args()

   
    # Load YAML config merged over defaults for backward compatibility.
    # If no config is provided, fall back to full CLI parsing from args.py
    if args_cli.config:
        args, model_cfg = load_training_config(config_path=args_cli.config)
    else:
        from ...utils import args as legacy_args
        args = legacy_args.parse_args()
        from ...utils.model_utils import ModelConfig
        model_cfg = ModelConfig(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            unet_model_name_or_path=args.unet_model_name_or_path,
            vae_model_name_or_path=args.vae_model_name_or_path,
            is_small_vae=args.is_small_vae,
        )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args_cli.config:
        logger.info("Loaded config from %s", args_cli.config)
    else:
        logger.info("Using legacy CLI arguments (no YAML config provided)")

    # Logging wandb 
    wandb_logger = WandbLogger(
        project=args.wandb_name,
        log_model=False)
    
    # Initialize models according to config
    manager = ModelTrainManager(args=args, model_config=model_cfg)

    # Check file embedding data exist or not
    if not os.path.exists(os.path.join(args.embedding_dir, 'train_embeddings')):
        logger.info("Not found embedding data , begin create new embedding: %s", args.data_csv_path)
        try:
            vae, tokenizer, text_encoder = manager.run_load_embedding_model()
            model = ObjectInsertionDiffusionModel(
                args=args,
                unet=None,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=None,
            )

            # Load dataset 
            train_dataset = AbstractDataset(data_csv_path=args.train_data_csv_path,
                                            data_embedding_path=args.embedding_dir,
                                            tensor_dtype=get_dtype_training(args.mixed_precision),
                                            image_size=(args.image_size, args.image_size) if hasattr(args, 'image_size') else (args.resolution, args.resolution)                          
                                            )
            
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=1, # must batchsize is 1 to save embedding
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.dataloader_num_workers,
            )


            model.preprocess_data(dataloader=train_dataloader,
                                  pytorch_dtype=get_dtype_training(args.mixed_precision),
                                  is_train=True)
            

            valid_dataset = AbstractDataset(data_csv_path=args.valid_data_csv_path,
                                            data_embedding_path=args.embedding_dir,
                                            tensor_dtype=get_dtype_training(args.mixed_precision),
                                            image_size=(args.image_size, args.image_size) if hasattr(args, 'image_size') else (args.resolution, args.resolution)                          
                                            )

            valid_dataloader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.dataloader_num_workers,
            )
           
            model.preprocess_data(dataloader=valid_dataloader,
                                  pytorch_dtype=get_dtype_training(args.mixed_precision),
                                  is_train=False)
            
            del model
            del vae
            del tokenizer
            del text_encoder

        
        except Exception as e:
            logger.error("Error during embedding creation: %s", str(e))
            import traceback
            logger.error("Full traceback: %s", traceback.format_exc())

    unet, noise_scheduler, vae  = manager.run_load_trainable_model()

    # freeze VAE weights 
    for param in vae.parameters():
        param.requires_grad = False
    # Freeze unet weights and train only attn1 layers 

    for name, param in unet.named_parameters():
        if "attn1" in name:  
            param.requires_grad = True  
        else:
            param.requires_grad = False 



    ptln_model = ObjectInsertionDiffusionModel(
        args=args,
        unet=unet,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        noise_scheduler=noise_scheduler,
    )

    del unet 
    del vae 

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="object-insertion-diffusion-{epoch:02d}-{valid_loss:.4f}",
        save_top_k=3,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(
            max_epochs=args.num_train_epochs,         
            accelerator= 'gpu',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=wandb_logger,
            log_every_n_steps=1,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            
        )
    
    train_dataset = AbstractDataset(data_csv_path=args.train_data_csv_path,
                                        data_embedding_path=os.path.join(args.output_dir,'train_embeddings'),
                                        tensor_dtype=get_dtype_training(args.mixed_precision),
                                        image_size=(args.image_size, args.image_size) if hasattr(args, 'image_size') else (args.resolution, args.resolution)                          
                                        )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    valid_dataset = AbstractDataset(data_csv_path=args.valid_data_csv_path,
                                        data_embedding_path=os.path.join(args.embedding_dir,'valid_embeddings'),
                                        tensor_dtype=get_dtype_training(args.mixed_precision),
                                        image_size=(args.image_size, args.image_size) if hasattr(args, 'image_size') else (args.resolution, args.resolution)                          
                                        )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    trainer.fit(ptln_model, train_dataloader, valid_dataloader)





if __name__ == "__main__":
    main()
