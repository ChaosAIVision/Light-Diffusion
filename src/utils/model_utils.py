import torch
from typing import Optional, Tuple
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import cv2
from pydantic import BaseModel
import logging


def get_dtype_training(dtype):
    if dtype == 'bf16':
        return torch.bfloat16
    if dtype == 'fp16':
        return torch.float16
    else:
        return torch.float32
    
    
def compute_dream_and_update_latents_for_inpaint(
    unet ,
    noise_scheduler ,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
    **kwang
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    conditional_controls = kwang.get('conditional_controls', None)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None  # b, 4, h, w
    with torch.no_grad():
        if conditional_controls is not None:
            pred = unet(sample=noisy_latents, timestep = timesteps, encoder_hidden_states=  encoder_hidden_states,
            conditional_controls = conditional_controls).sample
            
        else:
            pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    noisy_latents_no_condition = noisy_latents[:, :4]
    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents_no_condition.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    _noisy_latents = torch.cat([_noisy_latents, noisy_latents[:, 4:]], dim=1)
    # del unet
    return _noisy_latents, _target


def remove_noise(noise_scheduler, noisy_latent, noise, timestep):
    """Remove noise from a noisy latent based on DDIM's formulation.

    Ensures `alpha_t` and `sqrt(1 - alpha_t)` broadcast across [B, 4, H, W]
    by expanding to [B, 1, 1, 1].
    """
    # Align device and dtype to the latent for safe math ops
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy_latent.device, dtype=noisy_latent.dtype)

    # Gather per-sample alpha values and expand for broadcasting
    alpha_t = alphas_cumprod[timestep].sqrt().view(-1, 1, 1, 1)
    one_minus_alpha_t = (1 - alphas_cumprod[timestep]).sqrt().view(-1, 1, 1, 1)

    # DDIM reconstruction: x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
    latent_reconstructed = (noisy_latent - one_minus_alpha_t * noise) / alpha_t

    return latent_reconstructed



class ModelConfig(BaseModel):

    pretrained_model_name_or_path:str
    unet_model_name_or_path:str 
    vae_model_name_or_path:str 
    is_small_vae:bool = False


class ModelTrainManager:

    def __init__(self, args, model_config: ModelConfig):

        self.model_config = model_config
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.dtype = get_dtype_training(self.args.mixed_precision)


    
    def load_unet(self):
        try:
            from diffusers import UNet2DConditionModel
            # Try loading from unet_model_name_or_path first
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    self.model_config.unet_model_name_or_path,
                ).to(dtype = self.dtype)
            except (OSError, Exception):
                # Fallback: load from pretrained model subfolder
                self.unet = UNet2DConditionModel.from_pretrained(
                    self.model_config.pretrained_model_name_or_path,
                    subfolder="unet"
                ).to(dtype = self.dtype)
            
            self.logger.info("\033[92mSuccessfully initialized UNet model\033[0m")
        except Exception as e:
            self.logger.error(f"\033[91mFailed to initialize UNet model\033[0m")
            self.logger.exception(e)


    def load_vae(self):
        try:
            from diffusers import AutoencoderKL, AutoencoderTiny
            if self.model_config.is_small_vae:
                self.vae = AutoencoderTiny.from_pretrained(
                    self.model_config.vae_model_name_or_path,
                ).to(dtype = self.dtype)
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    self.model_config.vae_model_name_or_path,
                ).to(dtype = self.dtype)
            self.logger.info("\033[92mSuccessfully initialized VAE model\033[0m")
        except Exception as e:
            self.logger.error(f"\033[91mFailed to initialize VAE model\033[0m")
            self.logger.exception(e)

    def load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                subfolder="tokenizer", use_fast=False
            )
            self.logger.info("\033[92mSuccessfully initialized tokenizer\033[0m")
        except Exception as e:
            self.logger.error(f"\033[91mFailed to initialize tokenizer\033[0m")
            self.logger.exception(e)

    def load_noise_scheduler(self):
        try:
            from diffusers import DDIMScheduler
            self.noise_scheduler = DDIMScheduler.from_pretrained(
                self.model_config.pretrained_model_name_or_path, subfolder="scheduler"
            )
            self.logger.info("\033[92mSuccessfully initialized noise scheduler\033[0m")
        except Exception as e:
            self.logger.error(f"\033[91mFailed to initialize noise scheduler\033[0m")
            self.logger.exception(e)

    def load_text_encoder(self):
        try:
            from transformers import PretrainedConfig
            text_encoder_config = PretrainedConfig.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                subfolder="text_encoder",
            )
            model_class = text_encoder_config.architectures[0]

            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.model_config.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                ).to(device = 'cuda', dtype = self.dtype)
                self.logger.info("\033[92mSuccessfully initialized text encoder: CLIPTextModel\033[0m")
            else:
                raise ValueError(f"{model_class} is not supported.")
        except Exception as e:
            self.logger.error(f"\033[91mFailed to initialize text encoder\033[0m")
            self.logger.exception(e)


    def run_load_embedding_model(self):
        self.load_vae()
        self.load_tokenizer()
        self.load_text_encoder()

        return self.vae, self.tokenizer, self.text_encoder
    
    def run_load_trainable_model(self):
        self.load_unet()
        if not hasattr(self, 'unet'):
            raise RuntimeError("Failed to load UNet model")
        
        self.load_noise_scheduler()
        if not hasattr(self, 'noise_scheduler'):
            raise RuntimeError("Failed to load noise scheduler")
            
        self.load_vae()
        if not hasattr(self, 'vae'):
            raise RuntimeError("Failed to load VAE model")

        return self.unet, self.noise_scheduler, self.vae