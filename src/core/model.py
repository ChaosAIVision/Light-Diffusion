from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
import numpy as np
import bitsandbytes as bnb 
from torchvision.transforms.functional import to_pil_image
from .loss import  LPIPS, Discriminator

import os
import pandas as pd 
from tqdm import tqdm 
from dotenv import load_dotenv


load_dotenv()

ENCODER_HIDDEN_STATES_PATH = os.getenv("ENCODER_HIDDEN_STATES_PATH")


# init lpips loss
lpips_model = LPIPS().eval().to("cuda")
discriminator = Discriminator(im_channels=3).to("cuda")
disc_criterion = torch.nn.MSELoss()


class AbstractDiffusionModel(LightningModule):
    
    def __init__(self, args, unet, vae, text_encoder, tokenizer, noise_scheduler):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.args = args

    def forward_vae_encoder(self, vae, pixel_values):

        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        return latents
    
    def forward_add_noise(self, noise_scheduler, timesteps):

        noise = torch.randn_like(timesteps)
        noisy_latents = noise_scheduler.add_noise(timesteps, noise, timesteps)

        return noisy_latents, noise
    
    def forward_create_timesteps(self, noise_scheduler, latents_target):

        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
        return timesteps
    
    def forward_unet(self,unet, noisy_latents, time_steps, encoder_hidden_states, **kwargs):
        
        model_pred = unet(
            sample=noisy_latents,
            timestep=time_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        return model_pred
    
    def configure_optimizers(self):

        if self.args.use_adam8bit == True:
            optimizer= bnb.optim.Adam8bit(
            self.unet.parameters(),
            lr = self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay= self.args.adam_weight_decay,
            eps= self.args.adam_epsilon

        )
        
        else:
            optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.adam_weight_decay
        )
        return optimizer
    
    def step(self, batch):
        pass 

    def training_step(self, batch, batch_idx):
        pass 

    def validation_step(self, batch, batch_idx):
        pass 

    def preprocess_data(self, dataloader, weight_dtype, is_train):
        pass







