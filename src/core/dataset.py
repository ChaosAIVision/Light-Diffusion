from typing import Tuple
import torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from diffusers.image_processor import VaeImageProcessor
from colorama import Fore, init
from pathlib import Path

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")

from ..utils.dataset_utils import (
    TransformImage,
    get_dtype_training,
    transform_pil_to_tensor,
    numpy_to_pil,
    resize_and_padding,
    mask_to_bbox_image,
    string_to_list,
    get_bboxes_annotations,
    get_bbox_mask,
    get_masked_image_bbox,
    apply_mask
)


class AbstractDataset(Dataset):

    def __init__(self, data_csv_path:str, data_embedding_path:str, tensor_dtype:str ,image_size: Tuple[int, int]):
        
        """
        AbstractDataset class for custom datasets.

        Args:
            data_csv_path (str): Path to the CSV file containing dataset information.
            data_embedding_path (str): Path to the directory containing embeddings.
            tensor_dtype (str): Data type for tensors (e.g., 'float32', 'float16').
            image_size (Tuple[int, int]): Desired image size as (height, width).
        """
        self.tensor_dtype = tensor_dtype
        self.image_size = image_size
        self.transform_image = TransformImage()
        self.logger = logger

        # valid embedding path is exits
        if data_embedding_path is None or not Path(data_embedding_path).exists() or not Path(data_embedding_path + '/metadata.parquet').exists():
            if data_embedding_path is None:
                self.logger.warning(Fore.RED + "Embedding path is None. Generating new embeddings...")
            else:
                self.logger.warning(Fore.RED + f"Embedding metadata.parquet not found in {data_embedding_path}. Generating new embeddings...")
            self.data_csv = pd.read_csv(data_csv_path)
            self.need_generate_embedding = True
        
        else:
            self.data_embedding_path = Path(data_embedding_path)
            self.data_csv = None
            self.metadata = pd.read_parquet(self.data_embedding_path / 'metadata.parquet')
            self.need_generate_embedding = False


    def __len__(self):
        if self.data_csv is not None:
            return len(self.data_csv)
        else:
            return len(self.metadata)
            

    def make_data(self, image_path, object_image_path, bbox_str, mask_path=None):

        pixel_values = Image.open(image_path).convert("RGB").resize(self.image_size)
        
        # Handle mask - either from separate file or create from bbox
        if mask_path and os.path.exists(mask_path):
            # Use the actual mask file
            mask = Image.open(mask_path).convert("RGB").resize(self.image_size)
            # mask = mask_to_bbox_image(mask_image=mask)
        # else:
        #     # If no mask file, create from bbox if available
        #     if bbox_str:
        #         bbox = get_bboxes_annotations(bbox_str)
        #         mask = get_bbox_mask(pixel_values, bbox)
        latents_masked = apply_mask(image=pixel_values, mask=mask)
        object_image = Image.open(object_image_path).convert("RGB").resize(self.image_size)


        return {
            "pixel_values": self.transform_image.vae_image_transform(image=pixel_values, height= self.image_size[0], width= self.image_size[1]),
            "latents_masked": self.transform_image.vae_image_transform(image=latents_masked, height= self.image_size[0], width= self.image_size[1]),
            "object_pixel_values": self.transform_image.vae_image_transform(image=object_image, height= self.image_size[0], width= self.image_size[1]),
            "target_pixel_values": self.transform_image.vae_image_transform(image=pixel_values, height= self.image_size[0], width= self.image_size[1]),
            "mask_pixel_values": self.transform_image.vae_mask_transform(image=mask, height= self.image_size[0], width= self.image_size[1])
        }
    
    def load_saved_embeddings(self, idx):
        npz_file = np.load(os.path.join( self.data_embedding_path, "embeddings_data.npz"))
        metadata_df = pd.read_parquet(os.path.join( self.data_embedding_path, "metadata.parquet"))

        if idx is not None:
            if idx < 0 or idx >= len(metadata_df):
                    raise ValueError(f"Index {idx} out of range. Must be between 0 and {len(metadata_df) - 1}.")            
           
            row = metadata_df.iloc[idx]

            return {
                "latents_target": npz_file[row["latents_target_key"]],
                "latents_masked": npz_file[row["latents_masked_key"]],
                "object_image": npz_file[row["object_image_key"]],
                "mask_pixel_values": npz_file[row["mask_pixel_values_key"]],
                "target_pixel_value": npz_file[row["target_pixel_value_key"]],
                            
            }
        

    def __getitem__(self , index):

        if self.need_generate_embedding:
            item = self.data_csv.iloc[index]
            # Map CSV columns to expected format
            image_path = item.get('image_path', item.get('target_image'))
            object_image_path = item.get('object_image_path', item.get('object_image'))
            bbox_str = item.get('bbox', '')  # bbox might not exist in all datasets
            mask_path = item.get('mask', None)  # mask column from CSV
            batch = self.make_data(image_path, object_image_path, bbox_str, mask_path)

            return batch
        else:
            data= self.load_saved_embeddings(index)
            return {
                "latents_target": torch.tensor(data['latents_target']).to(dtype= self.tensor_dtype).squeeze(0),
                "latents_masked": torch.tensor(data['latents_masked']).to(dtype= self.tensor_dtype).squeeze(0),
                "object_image": torch.tensor(data['object_image']).to(dtype= self.tensor_dtype).squeeze(0),
                "mask_pixel_values": torch.tensor(data['mask_pixel_values']).to(dtype= self.tensor_dtype).squeeze(0),
                "target_pixel_value": torch.tensor(data['target_pixel_value']).to(dtype= self.tensor_dtype).squeeze(0)
            }
