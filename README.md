# Light-Diffusion: Object Insertion Training Framework

Light-Diffusion is a training framework for object insertion using diffusion models. This framework supports embedding preprocessing, training with PyTorch Lightning, and configuration through YAML files.

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training Configuration](#training-configuration)
- [Training Workflow](#training-workflow)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### System Requirements
- Python 3.10+
- CUDA-capable GPU (RTX 3060+ recommended)
- Minimum 16GB RAM
- At least 50GB free storage

### Step 1: Clone repository
```bash
git clone https://github.com/ChaosAIVision/Light-Diffusion.git
cd Light-Diffusion
```

### Step 2: Create virtual environment
```bash
conda create -n light-diffusion python=3.10
conda activate light-diffusion
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install additional packages (if needed)
```bash
pip install hf_transfer  # Accelerate model downloads from HuggingFace
```

## 📁 Project Structure

```
Light-Diffusion/
├── configs/
│   └── object_insertion/
│       └── default.yaml          # Main configuration file
├── dataset/
│   └── object_insertion/
│       └── train_dataset.csv     # Dataset CSV
├── outputs/
│   └── object_insertion/
│       ├── train_embeddings/     # Preprocessed embeddings
│       ├── valid_embeddings/
│       └── checkpoints/          # Model checkpoints
├── src/
│   ├── arch/
│   │   └── object_insertion/
│   │       ├── train.py          # Main training script
│   │       └── model.py          # Model architecture
│   ├── core/
│   │   ├── dataset.py           # Dataset class
│   │   └── model.py             # Base model
│   └── utils/
│       ├── args.py              # Legacy argument parser
│       ├── config_loader.py     # YAML config loader
│       ├── model_utils.py       # Model utilities
│       └── dataset_utils.py     # Dataset utilities
└── wandb/                       # W&B logging directory
```

## 📊 Data Preparation

### Step 1: Prepare CSV dataset
Create a CSV file with the following format: `dataset/object_insertion/train_dataset.csv`

```csv
target_image,object_image,mask
/path/to/target_image_0.png,/path/to/object_image_0.png,/path/to/mask_0.png
/path/to/target_image_1.png,/path/to/object_image_1.png,/path/to/mask_1.png
...
```

**Data Requirements:**
- **target_image**: Background image where object will be inserted
- **object_image**: Object image to be inserted
- **mask**: Mask defining insertion region (L mode, 0=background, 255=foreground)

### Step 2: Validate file paths
Ensure all paths in the CSV file exist:
```bash
python -c "
import pandas as pd
import os
df = pd.read_csv('dataset/object_insertion/train_dataset.csv')
for _, row in df.iterrows():
    for col in ['target_image', 'object_image', 'mask']:
        if not os.path.exists(row[col]):
            print(f'Missing: {row[col]}')
"
```

## ⚙️ Training Configuration

### Main config file: `configs/object_insertion/default.yaml`

```yaml
# Model configuration
model:
  pretrained_model_name_or_path: botp/stable-diffusion-v1-5-inpainting
  unet_model_name_or_path: "botp/stable-diffusion-v1-5-inpainting"
  vae_model_name_or_path: "madebyollin/taesd"
  is_small_vae: true

# Paths
output_dir: outputs/object_insertion
embedding_dir: 

# Training parameters
seed: 42
image_size: 512
train_batch_size: 4
num_train_epochs: 1

# Optimization
use_adam8bit: true
learning_rate: 5.0e-5
mixed_precision: bf16

# Data paths
train_data_csv_path: /path/to/your/train_dataset.csv
valid_data_csv_path: /path/to/your/valid_dataset.csv

# W&B monitoring
wandb_name: 'Your Project Name'
```

### Key Parameters:

**Model Configuration:**
- `pretrained_model_name_or_path`: Base diffusion model
- `vae_model_name_or_path`: VAE model (tiny VAE recommended for memory efficiency)
- `is_small_vae`: true if using tiny VAE

**Training Parameters:**
- `image_size`: Image resolution (512x512 or 1024x1024)
- `train_batch_size`: Batch size (reduce if OOM occurs)
- `mixed_precision`: bf16 or fp16 for memory efficiency

**Paths:**
- `embedding_dir`: null to generate new, or path to use existing embeddings
- `train_data_csv_path`: Path to training CSV data
- `valid_data_csv_path`: Path to validation CSV data

## 🏋️ Training Workflow

### Step 1: Start training
```bash
cd Light-Diffusion
python -m src.arch.object_insertion.train --config configs/object_insertion/default.yaml
```

### Step 2: Automatic process

**Phase 1: Embedding Preprocessing**
- If `embedding_dir: path to save embedding`, the system will automatically:
  1. Load VAE, tokenizer, text encoder
  2. Process each batch of data from CSV
  3. Generate latents from images
  4. Save embeddings to `outputs/object_insertion/train_embeddings/` and `valid_embeddings/`

**Phase 2: Model Training**
- Load UNet and noise scheduler
- Freeze VAE weights
- Train only attention layers in UNet
- Use saved embeddings for faster training

### Step 3: Monitoring

**PyTorch Lightning progress:**
```
Epoch 1/1: 100%|████████| 50/50 [02:30<00:00, 0.33it/s, loss=0.123, v_num=abc123]
```

**W&B dashboard:**
- Training/validation loss
- Learning rate schedule
- Model checkpoints
- System metrics

## 🧠 VRAM Optimization

This framework implements several techniques to minimize VRAM usage during training:

### 1. Preprocessed Data Ready
- **Benefit**: Training process doesn't waste VRAM on models like text encoder, tokenize, etc.
- **How it works**: All embeddings are precomputed and saved to disk before training starts
- **Implementation**: Set `embedding_dir` in config to save/load preprocessed embeddings

### 2. Partial Model Training
- **Benefit**: Only trains part of the model, not the full model, still effective
- **How it works**: Freezes VAE weights and trains only attention layers in UNet

### 3. Adam8bit Optimizer
- **Benefit**: Uses 8-bit precision for optimizer states, reducing memory footprint
- **How it works**: Quantizes optimizer parameters to 8 bits instead of 32 bits
- **Implementation**: Set `use_adam8bit: true` in config (requires bitsandbytes)

### 4. Tiny VAE
- **Benefit**: Smaller VAE model consumes significantly less VRAM
- **How it works**: Uses compressed VAE architecture (madebyollin/taesd)
- **Implementation**: Set `vae_model_name_or_path: "madebyollin/taesd"` and `is_small_vae: true`

### Combined Effect
With all optimizations enabled, VRAM usage can be reduced by **50-70%** compared to standard diffusion training, enabling training on consumer GPUs with 8GB+ VRAM.

## 📈 Monitoring

### W&B Integration
Training automatically logs to Weights & Biases:

1. **Login to W&B:**
```bash
wandb login
```

2. **View logs:**
- Training loss: `train_loss`
- Validation loss: `valid_loss`
- Learning rate: `lr-AdamW`

### Checkpoints
Model checkpoints are saved at:
```
outputs/object_insertion/checkpoints/
├── object-insertion-diffusion-epoch=00-valid_loss=0.1234.ckpt
├── object-insertion-diffusion-epoch=01-valid_loss=0.1150.ckpt
└── ...
```

### Log Files
```
wandb/
└── run-YYYYMMDD_HHMMSS-{run_id}/
    ├── files/
    ├── logs/
    └── ...
```

## 🔧 Troubleshooting

### Common Issues:

**1. OOM (Out of Memory)**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `train_batch_size` from 4 to 2 or 1
- Use `mixed_precision: fp16`
- Set `dataloader_num_workers: 0`

**2. Model not found**
```
OSError: botp/stable-diffusion-v1-5-inpainting does not appear to have a file named diffusion_pytorch_model.bin
```
**Solutions:**
- Check internet connection
- Try alternative model: `runwayml/stable-diffusion-v1-5`

**3. Dataset path does not exist**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solutions:**
- Check paths in CSV file
- Use absolute paths
- Ensure file read permissions

**4. W&B authentication**
```
wandb: ERROR Unable to authenticate
```
**Solution:**
```bash
wandb login --relogin
```

### Performance Optimization:

**1. Use saved embeddings:**
```yaml
embedding_dir: outputs/object_insertion/train_embeddings
```

**2. Increase workers:**
```yaml
dataloader_num_workers: 4  # Increase from 0
```

**3. Gradient accumulation:**
```yaml
gradient_accumulation_steps: 2  # Equivalent to batch_size x2
```

## 🎯 Advanced Usage

### Custom Environment Variables
```bash
export ENCODER_HIDDEN_STATES_PATH="/path/to/encoder_states.pt"
export CONCAT_DIM="1"
export SAVE_IMAGE="100"
```

### Resume from checkpoint
```yaml
resume_from_checkpoint: "outputs/object_insertion/checkpoints/last.ckpt"
```

### Mixed Training Modes
```yaml
# Fast preprocessing with batch_size=1
train_batch_size: 1  # For preprocessing
# Training with larger batch_size
train_batch_size: 4  # For actual training
```

## 📞 Support

If you encounter issues:
1. Check [Troubleshooting](#troubleshooting)
2. Review logs in terminal and W&B
3. Monitor GPU memory usage: `nvidia-smi`
4. Create an issue with complete logs

---

**Happy Training! 🚀**
