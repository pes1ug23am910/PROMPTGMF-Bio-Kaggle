# Environment Setup Instructions for PromptGFM-Bio

## Quick Setup (Windows)

### Option 1: Automated Setup (Recommended)

Run the setup script:
```powershell
.\setup_environment.ps1
```

### Option 2: Manual Setup

#### Step 1: Create Conda Environment
```powershell
conda create -n promptgfm python=3.10 -y
conda activate promptgfm
```

#### Step 2: Install PyTorch with CUDA 11.8
```powershell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Install PyTorch Geometric
```powershell
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

#### Step 4: Install Remaining Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 5: Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

## Linux/MacOS Setup

### Step 1: Create Conda Environment
```bash
conda create -n promptgfm python=3.10 -y
conda activate promptgfm
```

### Step 2: Install PyTorch
**For CUDA 11.8 (Linux with GPU):**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only (MacOS/Linux without GPU):**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install PyTorch Geometric
**For CUDA 11.8:**
```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**For CPU:**
```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Step 4: Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

## Troubleshooting

### CUDA Not Available
If `torch.cuda.is_available()` returns `False`, check:
1. NVIDIA GPU drivers are installed
2. CUDA 11.8 toolkit is installed
3. PyTorch was installed with the correct CUDA version

### PyTorch Geometric Installation Issues
If PyG extensions fail to install:
```bash
# Try installing from conda-forge instead
conda install pyg -c pyg
```

### Memory Issues During Installation
If you encounter memory errors:
```bash
# Install packages one at a time
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

## Verifying Your Setup

Run the verification script:
```powershell
python scripts/verify_setup.py
```

This will check:
- Python version (should be 3.10.x)
- PyTorch installation and CUDA availability
- PyTorch Geometric installation
- All required packages

## Environment Variables

For Weights & Biases logging (optional):
```powershell
# Windows
$env:WANDB_API_KEY="your_api_key_here"

# Linux/MacOS
export WANDB_API_KEY="your_api_key_here"
```

## Next Steps

After setup is complete:

1. **Download Data:**
   ```bash
   bash scripts/download_data.sh
   ```

2. **Preprocess Graphs:**
   ```bash
   python scripts/preprocess_all.py
   ```

3. **Run Tests:**
   ```bash
   pytest tests/
   ```

4. **Start Training:**
   ```bash
   python scripts/train_promptgfm.py --config configs/base_config.yaml
   ```

## Development Setup

For development work, also install:
```bash
pip install black flake8 mypy pre-commit
```

Configure pre-commit hooks:
```bash
pre-commit install
```

## GPU Requirements

- **Minimum:** NVIDIA GPU with 8GB VRAM
- **Recommended:** NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)
- **CUDA:** Version 11.8 or compatible

For CPU-only usage, modify configs to set `device: 'cpu'`.
