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
- [Example Training Results](#example-training-results)
- [Roadmap](#roadmap)

## 🚀 Installation

### System Requirements
- Python 3.10+
- CUDA-capable GPU with **minimum 8GB VRAM** (RTX 3060+ recommended)
- Minimum 16GB RAM

### Minimum GPU Configuration (8GB VRAM)
The framework is optimized to train on GPUs with as little as 8GB VRAM:
- **Model size**: 49.1M parameters
- **Image size**: 512 x 512
- **Batch size**: 1
- **Gradient accumulation**: 4 (equivalent to batch size 4)
- **Optimizer**: AdamW8bit
- **Mixed precision**: bf16 or fp16

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
- `image_size`: Image resolution (512x512 recommended for 8GB VRAM)
- `train_batch_size`: Batch size (1 recommended for 8GB VRAM)
- `gradient_accumulation_steps`: 4 (to simulate batch size 4)
- `mixed_precision`: bf16 or fp16 for memory efficiency
- `use_adam8bit`: true (required for 8GB VRAM training)

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
With all optimizations enabled, VRAM usage can be reduced by **50-70%** compared to standard diffusion training, enabling training on consumer GPUs with **8GB VRAM**.

### Example Configuration for 8GB VRAM
```yaml
image_size: 512
train_batch_size: 1
gradient_accumulation_steps: 4
use_adam8bit: true
mixed_precision: bf16
# Model has ~49.1M trainable parameters
```

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
- Use recommended 8GB VRAM config: `train_batch_size: 1`, `gradient_accumulation_steps: 4`
- Use `mixed_precision: fp16` or `bf16`
- Set `dataloader_num_workers: 0`
- Enable `use_adam8bit: true`
- Use tiny VAE: `vae_model_name_or_path: "madebyollin/taesd"`

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

## 🎨 Example Training Results

### Stable Diffusion Inpainting v1.5 Training Example

This example demonstrates object insertion training using Stable Diffusion Inpainting 1.5 with a real-world dataset.

**Input Images:**
- **Mask**: `assert/mask_train.jpg` - Defines the insertion region
- **Object Image**: `assert/object_image_6.png` - Object to be inserted into the scene
- **Target Image**: `assert/image_train.jpg` - Final composite result (ground truth)

**Training Configuration:**
```yaml
model:
  pretrained_model_name_or_path: botp/stable-diffusion-v1-5-inpainting
  vae_model_name_or_path: "madebyollin/taesd"
  is_small_vae: true

image_size: 512
train_batch_size: 1
gradient_accumulation_steps: 4
use_adam8bit: true
mixed_precision: bf16
learning_rate: 5.0e-5
```

**Dataset Format:**
```csv
target_image,object_image,mask
/home/chaos/Documents/chaos/repo/Light-Diffusion/assert/image_train.jpg,/home/chaos/Documents/chaos/repo/Light-Diffusion/assert/object_image_6.png,/home/chaos/Documents/chaos/repo/Light-Diffusion/assert/mask_train.jpg
```

**Training Process:**
1. **Preprocessing**: VAE encodes images to latents, text encoder processes prompts
2. **Training**: UNet learns to insert objects into masked regions
3. **Validation**: Model generates predictions and compares with target images

**Expected Results:**
- Model learns to seamlessly blend objects into background scenes
- Maintains lighting and perspective consistency
- Preserves object details while adapting to scene context

**Visualization:**
The training process produces intermediate results showing:
- Input mask overlay
- Object placement
- Final composite output
- Loss curves tracking training progress

You can monitor training progress through W&B dashboard or checkpoint outputs.

## 🗺️ Roadmap

This section outlines the planned features and improvements for the Light-Diffusion framework.

### Phase 1: Enhanced VRAM-Safe Training Framework (Done)

**Goal**: Develop a comprehensive PyTorch Lightning training framework with ultra-safe VRAM management through advanced configuration options.

**Features:**
- **Config-based VRAM optimization**: Special configuration flag (`safe_vram_mode: 1`) for maximum memory efficiency
  - Automatic gradient checkpointing
  - Dynamic batch size adjustment
  - Progressive model loading/unloading
- **Adaptive memory management**: Real-time VRAM monitoring and automatic adjustments
- **Multi-GPU support**: Distributed training with efficient memory allocation
- **Training resumption**: Smart checkpoint loading with memory optimization

**Expected Benefits:**
- Train on GPUs with 8GB+ VRAM
- Reduced OOM errors through intelligent memory management

### Phase 2: Extended Conditional Training Tasks

**Goal**: Extend framework to support multiple conditional image editing tasks beyond object insertion.

#### 2.1 White Balance Correction
- **Task**: Automatically correct white balance in images
- **Input**: Image with incorrect white balance
- **Output**: Image with corrected color temperature
- **Use cases**: Photography enhancement, post-processing automation
- **Dataset format**: `(input_image, target_image, white_balance_params)`

#### 2.2 Object Removal
- **Task**: Remove unwanted objects from images seamlessly
- **Input**: Image with mask indicating objects to remove
- **Output**: Image with objects removed and background inpainting
- **Use cases**: Photo editing, content moderation, privacy protection
- **Dataset format**: `(input_image, mask, target_image)`

#### 2.3 Paint-to-Image
- **Task**: Convert simple sketches/paintings to photorealistic images
- **Input**: Sketch image with optional color hints
- **Output**: High-quality rendered image
- **Use cases**: Concept art visualization, design prototyping
- **Dataset format**: `(sketch_image, target_image, optional_prompt)`

**Implementation Plan:**
- Unified architecture supporting multiple task types
- Task-specific loss functions and data loaders
- Configurable training pipelines per task
- Cross-task knowledge transfer capabilities

### Phase 3: Diffusion Transformer (DIT) Architecture Support

**Goal**: Enable efficient training for Diffusion Transformer architectures like Flux and similar models.

#### 3.1 Flux Model Training
- **Architecture**: DiT (Diffusion Transformer) based models
- **Features**:
  - Support for transformer-based diffusion models
  - Efficient attention mechanisms (Flash Attention, SDPA)
  - Sequence-based training pipeline
  - Multi-resolution training support
- **Optimizations**:
  - Token-based gradient accumulation
  - Transformer-specific memory optimizations
  - Efficient positional encoding handling
  - Support for variable sequence lengths

#### 3.2 Architecture Adaptations
- **Modular design**: Easy integration of different DIT variants

**Expected Configuration:**
```yaml
model:
  architecture: dit  # or "flux"
  pretrained_model_name_or_path: black-forest-labs/FLUX.1-dev
```

### Phase 4: Qwen-Image-Edit Model Training

**Goal**: Integrate and support training for Qwen-Image-Edit models and similar vision-language editing models.

#### 4.1 Qwen-Image-Edit Integration
- **Model**: Qwen/Qwen-Image-Edit

**Expected Configuration:**
```yaml
model:
  architecture: qwen-image-edit
  pretrained_model_name_or_path: Qwen/Qwen-Image-Edit
  
```

### Implementation Timeline

**Phase 1** :
- ✅ Basic VRAM optimization (current)
- 🔄 Enhanced safe VRAM mode with config flag
- 🔄 Adaptive memory management

**Phase 2** :
- 📅 White balance correction task
- 📅 Object removal task
- 📅 Paint-to-image task

**Phase 3** :
- 📅 Flux/DIT architecture support
- 📅 Transformer-specific optimizations
- 📅 Multi-resolution training

**Phase 4** :
- 📅 Qwen-Image-Edit integration
- 📅 Multi-modal training pipeline
- 📅 Instruction tuning support

### Contributing

We welcome contributions to help accelerate the roadmap! Areas where contributions are especially valuable:

- **VRAM optimization techniques**: Novel memory-efficient training methods
- **New task implementations**: Additional conditional training tasks
- **Architecture support**: Integration of new diffusion model architectures
- **Documentation**: Tutorials and examples for new features
- **Testing**: Comprehensive test suites for all features

### Feedback and Suggestions

If you have ideas, feature requests, or want to contribute to any of these roadmap items, please:

1. Open an issue with the `roadmap` label
2. Discuss in discussions section
3. Submit a pull request for implementations

---

## 📞 Support

If you encounter issues:
1. Check [Troubleshooting](#troubleshooting)
2. Review logs in terminal and W&B
3. Monitor GPU memory usage: `nvidia-smi`
4. Create an issue with complete logs

---

**Happy Training! 🚀**
