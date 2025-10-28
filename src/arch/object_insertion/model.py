from ...core.model import AbstractDiffusionModel
from ...core.loss import LPIPS, Discriminator
from ...utils.model_utils import compute_dream_and_update_latents_for_inpaint

import os 

import torch 
from diffusers.training_utils import compute_snr
import numpy as np
import bitsandbytes as bnb 
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import pandas as pd 

from ...utils.dataset_utils import (
    
    get_dtype_training)

from ...utils.model_utils import (
    compute_dream_and_update_latents_for_inpaint,
    remove_noise
)

from ...core.loss import masked_mse_loss


ENCODER_HIDDEN_STATES_PATH = os.getenv("ENCODER_HIDDEN_STATES_PATH")
CONCAT_DIM = int(os.getenv("CONCAT_DIM")) 
SAVE_IMAGE = int(os.getenv("SAVE_IMAGE", "100"))  # Default to every 100 steps if not set
# init lpips loss
lpips_model = LPIPS().eval().to("cuda")
discriminator = Discriminator(im_channels=3).to("cuda")
disc_criterion = torch.nn.MSELoss()

class ObjectInsertionDiffusionModel(AbstractDiffusionModel):

    def __init__(self, args, unet, vae, text_encoder, tokenizer, noise_scheduler):
        super().__init__(args, unet, vae, text_encoder, tokenizer, noise_scheduler)
        self.encoder_hidden_states = torch.load(ENCODER_HIDDEN_STATES_PATH).to('cuda')

    def forward(self, noisy_latents, time_steps, encoder_hidden_states):
        
        return self.forward_unet(
            unet=self.unet,
            noisy_latents=noisy_latents,
            time_steps=time_steps,
            encoder_hidden_states=encoder_hidden_states,
        )
    
    def step(self, batch):

        device = 'cuda'
        dtype = get_dtype_training(self.args.mixed_precision)
        # dtype= torch.float32 # For debugging purpose

        # Ensure encoder_hidden_states has correct dtype
        if self.encoder_hidden_states.dtype != dtype:
            self.encoder_hidden_states = self.encoder_hidden_states.to(dtype=dtype)

        # extract batch data
        latents_target = batch["latents_target"].to(dtype =dtype )
        latent_masked = batch['latents_masked'].to(dtype = dtype)
        mask_pixel_values = batch['mask_pixel_values'].to(dtype = dtype)
        
        # Ensure mask has channel dimension [B, C, H, W]
        if mask_pixel_values.dim() == 3:  # [B, H, W] -> [B, 1, H, W] 
            mask_pixel_values = mask_pixel_values.unsqueeze(1)
        
        target_pixel_value = batch["target_pixel_value"].to(dtype=dtype, device = 'cuda')
        object_image = batch['object_image'].to(dtype = dtype)

        # create time steps 
        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,)).to(device=device, dtype=torch.long)

        # Concatenate conditional 
        masked_latent_concat = torch.cat([latent_masked, object_image], dim=CONCAT_DIM)

        # Get size tensor 
        height_mask, width_mask = mask_pixel_values.shape[2], mask_pixel_values.shape[3]        
        mask_latent=  torch.nn.functional.interpolate(mask_pixel_values, size=( height_mask// 8, width_mask // 8)) # devide 8 for matching with vae encoder shape
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=CONCAT_DIM)

        # Padding latent target to fit shape masked_latent and mask
        latent_target_concat = torch.cat((latents_target, object_image), dim=CONCAT_DIM).to(dtype= dtype)

        # Add noise to latents target 
        noise = torch.randn_like(latent_target_concat).to(dtype= dtype)
        noisy_latents_target = self.noise_scheduler.add_noise(latent_target_concat, noise, timesteps)
        inpainting_latent_unet_input = torch.cat([noisy_latents_target, mask_latent_concat, masked_latent_concat], dim=1).to(dtype= dtype)
        # DREAM interation 
        self.unet.to(device = device , dtype = dtype) # make unet ensure on correct device and dtype
        inpainting_latent_unet_input, latent_target = compute_dream_and_update_latents_for_inpaint(
            unet = self.unet,
            noise_scheduler = self.noise_scheduler,
            timesteps = timesteps,
            noise = noise,
            noisy_latents = inpainting_latent_unet_input,
            target = latent_target_concat,
            encoder_hidden_states = self.encoder_hidden_states,
            dream_detail_preservation=1.0 # set to 1.0 to fully preserve details
        )

        # calculate model prediction
        model_pred = self(
            noisy_latents=inpainting_latent_unet_input,
            time_steps=timesteps,
            encoder_hidden_states=self.encoder_hidden_states,
        )

        # Using epsilon to caculate noise afficiently with SNR and Dream
        # target = noise 

        # 1. Caculate SNR weighting loss 
        snr = compute_snr(self.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        mse_loss_weights = mse_loss_weights / snr # best SNR hyperparameter from https://arxiv.org/pdf/2303.09556 


        # 2. Caculate perceptual loss (LPIPS)

        # 2.1 split the model prediction to get predicted latents
        model_pred_split  = model_pred.split(model_pred.shape[CONCAT_DIM] // 2, dim=CONCAT_DIM)[0] 
        # noise_no_cat = noise.split(noise.shape[float(CONCAT_DIM)] // 2, dim=float(CONCAT_DIM))[0] # get grounth truth noise without padding 

        # 2.2 Remove noise from noisy latents to get reconstructed latents
        # Split noisy_latents_target to match model_pred_split shape
        noisy_latents_target_split = noisy_latents_target.split(noisy_latents_target.shape[CONCAT_DIM] // 2, dim=CONCAT_DIM)[0]
        noise_remove_padding = noise.split(noise.shape[CONCAT_DIM] // 2, dim=CONCAT_DIM)[0] # get grounth truth noise without padding 
        latent_rescontructed = remove_noise(
            noise_scheduler=self.noise_scheduler,
            noisy_latent= noisy_latents_target_split,
            noise= model_pred_split,
            timestep= timesteps
        )

        # 2.3 Decode the reconstructed latents and target latents
        # latent_no_cat_pred  = latent_rescontructed.split(latent_rescontructed.shape[CONCAT_DIM] // 2, dim=CONCAT_DIM)[0]
        latent_no_cat_pred = latent_rescontructed / self.vae.config.scaling_factor # scale back to vae latent space
        decoded_image = self.vae.decode(latent_no_cat_pred.to("cuda")).sample
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1) # scale to fit with target pixel value range [0, 1]
        target_pixel_value= (target_pixel_value /2 + 0.5).clamp(0,1)

        if getattr(self, "save_image", 0) % int(SAVE_IMAGE) == 0:
            
            image_predict = to_pil_image(decoded_image[0])
            image_predict.save(os.path.join(self.args.output_dir, 'image_train.jpg'))
            pil_mask = to_pil_image(mask_pixel_values[0])
            pil_mask.save(os.path.join(self.args.output_dir, 'mask_train.jpg'))
    
            self.save_image= 0

        lpips_loss = torch.mean(lpips_model(decoded_image, target_pixel_value))
        perceptual_losses = lpips_loss.item()


        # Caculate MSE loss

        masked_loss = masked_mse_loss(model_pred_split.float(), noise_remove_padding.float(), mask_latent)
        masked_loss = (masked_loss * mse_loss_weights).mean()


        self.save_image = getattr(self, "save_image", 0) + 1
        # loss =  perceptual_losses 
        loss = 0.5 * masked_loss + perceptual_losses
        self.log("train_loss", loss, prog_bar=True, on_step=True,on_epoch=True,logger=True, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        # Convert to tensor if it's a float
        if isinstance(step_loss, float):
            step_loss = torch.tensor(step_loss, device='cuda')
            
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05 # exponential moving average for loss smoothing
        )

    def validation_step(self, batch, batch_idx):
            step_loss = self.step(batch)
            # Convert to tensor if it's a float
            if isinstance(step_loss, float):
                step_loss = torch.tensor(step_loss, device='cuda')
            
            self.log_loss = (
                step_loss.item()
                if not hasattr(self, "log_loss")
                else self.log_loss * 0.95 + step_loss.item() * 0.05
            )
            self.log('valid_loss', step_loss, prog_bar= True, on_epoch= True, on_step= True, logger= True, sync_dist= True)
            return step_loss
    

    def preprocess_data(self, dataloader, pytorch_dtype, is_train: bool = False):
         
        npz_data = {}
        metadata_records = []
        self.vae = self.vae.to(device='cuda', dtype=torch.bfloat16) # Need to convert to bfloat16 because some VAE when convert FP16 , it will harn the latent quality


        if is_train == True:
            save_embedding_dir = os.path.join(self.args.output_dir, "train_embeddings")
            os.makedirs(save_embedding_dir, exist_ok=True)
        else:
            save_embedding_dir = os.path.join(self.args.output_dir, "valid_embeddings")
            os.makedirs(save_embedding_dir, exist_ok=True)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Preprocessing data"):
            with torch.no_grad():

                # Extract batch data
                pixel_values = batch["pixel_values"].to(device='cuda', dtype=torch.bfloat16)
                object_pixel_values = batch['object_pixel_values'].to(device='cuda', dtype=torch.bfloat16)
                mask_pixel_values = batch['mask_pixel_values'].to(device='cuda', dtype=torch.bfloat16)
                masked_values = batch['latents_masked'].to(device='cuda', dtype=torch.bfloat16)
                
                # Encode with vae
                vae_output = self.vae.encode(pixel_values)
                if hasattr(vae_output, 'latent_dist'):
                    # Standard VAE
                    latents_target = vae_output.latent_dist.sample() * self.vae.config.scaling_factor
                else:
                    # Tiny VAE
                    latents_target = vae_output.latents * self.vae.config.scaling_factor
                
                vae_output_obj = self.vae.encode(object_pixel_values)
                if hasattr(vae_output_obj, 'latent_dist'):
                    object_latent = vae_output_obj.latent_dist.sample() * self.vae.config.scaling_factor
                else:
                    object_latent = vae_output_obj.latents * self.vae.config.scaling_factor
                
                vae_output_masked = self.vae.encode(masked_values)
                if hasattr(vae_output_masked, 'latent_dist'):
                    latent_masked = vae_output_masked.latent_dist.sample() * self.vae.config.scaling_factor
                else:
                    latent_masked = vae_output_masked.latents * self.vae.config.scaling_factor

                # Build metadata record - save each item in batch separately
                bsz = latents_target.shape[0]
                for b in range(bsz):
                    item_idx = i * bsz + b  # Global item index
                    
                    npz_data[f"latents_target_{item_idx}"] = latents_target[b].to(self.dtype).detach().cpu().numpy()
                    npz_data[f"latents_masked_{item_idx}"] = latent_masked[b].to(self.dtype).detach().cpu().numpy()
                    npz_data[f"mask_pixel_values_{item_idx}"] = mask_pixel_values[b].to(self.dtype).detach().cpu().numpy()
                    npz_data[f"target_pixel_value_{item_idx}"] = pixel_values[b].to(self.dtype).detach().cpu().numpy()
                    npz_data[f"object_image_{item_idx}"] = object_latent[b].to(self.dtype).detach().cpu().numpy()

                    metadata_records.append(
                        {
                            "latents_target_key": f"latents_target_{item_idx}",
                            "latents_masked_key": f"latents_masked_{item_idx}",
                            "mask_pixel_values_key": f"mask_pixel_values_{item_idx}",
                            "target_pixel_value_key": f"target_pixel_value_{item_idx}",
                            "object_image_key": f"object_image_{item_idx}",
                        }
                    )
        # Save to npz file
        np.savez_compressed(
            os.path.join(save_embedding_dir, "embeddings_data.npz"),
            **npz_data,
        )
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(os.path.join(save_embedding_dir, "metadata.parquet"))
        