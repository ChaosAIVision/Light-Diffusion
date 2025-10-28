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
import cv2
from PIL import ImageDraw, Image, ImageOps, ImageChops


class TransformImage():

    """
    TransformImage class for preprocessing images 
    """
    
    def __init__(self):
        
        self.vae_image_processor = VaeImageProcessor(vae_scale_factor= 8)
        
        self.vae_mask_processor =  VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        
        
        
    def vae_image_transform(self, image, height, width):
        
        return  self.vae_image_processor.preprocess(image,height, width)[0] 
          
    def vae_mask_transform(self, image, height, width):
        mask_tensor = self.vae_mask_processor.preprocess(image, height, width)[0]
        # Ensure mask has channel dimension [C, H, W]
        if mask_tensor.dim() == 2:  # [H, W] -> [1, H, W]
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
    
    def pytorch_image_transform(self, image, height, width):
        
        return transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # normalize with config of stable diffusion normalize in Huggingface
        ])


def transform_pil_to_tensor(image: Image.Image, size=(512, 512)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),  
        transforms.ToTensor(),   
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo VGG
    ])
    return transform(image)


def get_dtype_training(dtype):
    if dtype == 'bf16':
        return torch.bfloat16
    if dtype == 'fp16':
        return torch.float16
    else:
        return torch.float32
    
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def resize_and_padding(image, size):
    # Padding to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # padding
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def mask_to_bbox_image(mask_image):
    mask_image = mask_image.convert("L")
    mask_array = np.array(mask_image)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = (x, y, x + w, y + h)
    bbox_image = Image.new("L", mask_image.size, 0)
    draw = ImageDraw.Draw(bbox_image)
    draw.rectangle(bbox, fill=255)
    return bbox_image


def string_to_list(str_annotation):
        """
        Converts a string of annotations to a list of integers.

        Args:
            str_annotation: 'x1,y1,x2,y2' to list [x1,y1,x2,y2,x3, y3]
        """
        
        if isinstance(str_annotation, float):
            str_annotation = str(int(str_annotation))
        
        str_values = str_annotation.split(',')        
        result = []
        for value in str_values:
            if '.' in value:
                result.append(int(round(float(value)))) 
            else:
                result.append(int(value)) 
        
        return result


def get_bboxes_annotations(bbox_string):
        """
        Convert bbox string to boundingbox list

        Args:
            bbox_string: 'x1,y1,x2,y2' to list
        """

        return string_to_list(bbox_string)


def get_bbox_mask(image, bbox):
    
    """
    Creates a black-and-white mask where the region inside the bbox is white,
    and the region outside is black.

    Args:
        image : PIL image
        bbox : List Bbox

    """
    mask = Image.new('L', image.size, color = 0)
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2):
        for y in range(y1, y2):
            mask.putpixel((x,y), 225)
    return mask


def get_masked_image_bbox( image ,bbox):
        """

        Args:
            image : PIL image target
            bbox : list bbox
            
        """
        image_array = np.array(image) 
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_array.shape[1], x2), min(image_array.shape[0], y2)
        black_overlay = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)  
        masked_image_array = image_array.copy()
        masked_image_array[y1:y2, x1:x2] = black_overlay
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image



def apply_mask(image, mask):
    """
    Apply a binary mask to an image, setting the masked area to black.
    """
    # Convert mask to binary (0 and 255)
    mask = mask.convert("L")
    inverted_mask = ImageOps.invert(mask) # reverse mask: range 0 -> 255, 255 -> 0
    inverted_mask_rgb = Image.merge("RGB", (inverted_mask, inverted_mask, inverted_mask))

    masked_image = ImageChops.multiply(image, inverted_mask_rgb)

    # convert to PIL image

    return masked_image




def collate_fn(examples):
    # Check if we're dealing with raw data (for preprocessing) or saved embeddings
    first_example = examples[0]
    
    if "pixel_values" in first_example:
        # Raw data case - for preprocessing
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format)
        
        latents_masked = torch.stack([example['latents_masked'] for example in examples])
        latents_masked = latents_masked.to(memory_format=torch.contiguous_format)

        object_pixel_values = torch.stack([example['object_pixel_values'] for example in examples])
        object_pixel_values = object_pixel_values.to(memory_format=torch.contiguous_format)

        mask_pixel_values = torch.stack([example['mask_pixel_values'] for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format)
        
        return {
            "pixel_values": pixel_values,
            "latents_masked": latents_masked,
            'object_pixel_values': object_pixel_values,
            'mask_pixel_values': mask_pixel_values,
        }
    else:
        # Saved embeddings case - for training
        latents_target = torch.stack([example["latents_target"] for example in examples])
        latents_target = latents_target.to(memory_format=torch.contiguous_format)
        
        latents_masked = torch.stack([example['latents_masked'] for example in examples])
        latents_masked = latents_masked.to(memory_format=torch.contiguous_format)

        object_image = torch.stack([example['object_image'] for example in examples])
        object_image = object_image.to(memory_format=torch.contiguous_format)

        mask_pixel_values = torch.stack([example['mask_pixel_values'] for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format)
        
        target_pixel_value = torch.stack([example['target_pixel_value'] for example in examples])
        target_pixel_value = target_pixel_value.to(memory_format=torch.contiguous_format)
        
        return {
            "latents_target": latents_target,
            "latents_masked": latents_masked,
            'object_image': object_image,
            'mask_pixel_values': mask_pixel_values,
            'target_pixel_value': target_pixel_value,
        }
