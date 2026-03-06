# Troubleshooting Guide

Common issues and solutions for PromptGFM-Bio.

---

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Data Download Problems](#data-download-problems)
3. [Training Errors](#training-errors)
4. [GPU and Memory Issues](#gpu-and-memory-issues)
5. [Evaluation Problems](#evaluation-problems)
6. [Performance Issues](#performance-issues)

---

## Installation Issues

### PyTorch Geometric Installation Fails

**Problem**: Error installing `torch-scatter`, `torch-sparse`, or `torch-cluster`

**Solution**:
```bash
# Ensure PyTorch is installed first
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install PyG with correct CUDA version
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**Alternative**: Install from conda
```bash
conda install pytorch-geometric -c pyg
```

### CUDA Version Mismatch

**Problem**: `RuntimeError: CUDA error: no kernel image is available for execution`

**Solution**:
```bash
# Check your GPU's CUDA version
nvidia-smi

# Uninstall PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with matching CUDA version
# For CUDA 11.8:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### BioBERT Download Fails

**Problem**: `OSError: Can't load tokenizer for 'dmis-lab/biobert-base-cased-v1.1'`

**Solution**:
```bash
# Download manually
python -c "from transformers import AutoTokenizer, AutoModel; \
           AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1'); \
           AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')"

# Or set offline mode with local cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

---

## Data Download Problems

### STRING Database Download Fails

**Problem**: Connection timeout or incomplete download

**Solution**:
```bash
# Download manually
wget https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
gunzip 9606.protein.links.v12.0.txt.gz
mv 9606.protein.links.v12.0.txt data/raw/string/

# Then run preprocessing
python scripts/preprocess_all.py
```

### DisGeNET Requires Registration

**Problem**: DisGeNET download requires API credentials

**Solution**:
1. Register at https://www.disgenet.org/signup/
2. Get API key
3. Add to `.env`:
   ```bash
   DISGENET_API_KEY=your_api_key_here
   ```
4. Re-run download script

### Orphanet XML Parsing Error

**Problem**: `xml.etree.ElementTree.ParseError: mismatched tag`

**Solution**:
```bash
# Re-download Orphanet files
rm -rf data/raw/orphanet/
python scripts/download_data.py --datasets orphanet

# Or use backup URLs in the script
```

### Insufficient Disk Space

**Problem**: `OSError: [Errno 28] No space left on device`

**Solution**:
```bash
# Check disk usage
df -h

# Clear cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/pip/

# Download only essential datasets
python scripts/download_data.py --datasets string disgenet hpo
```

---

## Training Errors

### Out of Memory (OOM)

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**
   ```yaml
   # In config file
   training:
     batch_size: 16  # Reduce from 32
   ```

2. **Enable gradient checkpointing**
   ```python
   model = PromptGFM(
       ...,
       gradient_checkpointing=True
   )
   ```

3. **Use mixed precision**
   ```bash
   python scripts/train.py --config configs/base_config.yaml --fp16
   ```

4. **Reduce model size**
   ```yaml
   model:
     gnn:
       hidden_dim: 256  # Reduce from 512
       num_layers: 2    # Reduce from 3
   ```

### NaN Loss

**Problem**: Loss becomes NaN during training

**Solutions**:

1. **Lower learning rate**
   ```yaml
   training:
     learning_rate: 0.00005  # Reduce from 0.0001
   ```

2. **Enable gradient clipping**
   ```yaml
   training:
     gradient_clip: 0.5  # Lower threshold
   ```

3. **Check data for NaN values**
   ```python
   # In preprocess.py
   df = df.dropna()
   df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
   ```

4. **Use more stable loss function**
   ```yaml
   training:
     loss: 'bce'  # Instead of 'ranking' or 'contrastive'
   ```

### Checkpoint Not Saving

**Problem**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Check directory permissions
chmod 755 checkpoints/

# Ensure directory exists
mkdir -p checkpoints/promptgfm_film/

# Check disk space
df -h checkpoints/
```

### Weights & Biases Login Error

**Problem**: `wandb.errors.UsageError: api_key not configured`

**Solution**:
```bash
# Login to W&B
wandb login

# Or set API key in environment
export WANDB_API_KEY=your_api_key_here

# Or disable W&B
export WANDB_MODE=disabled
```

### KeyError: 'gene-gene' Edge Type

**Problem**: `KeyError: ('gene', 'interacts', 'gene')`

**Solution**:
```python
# Check available edge types
graph = torch.load('data/processed/biomedical_graph.pt')
print(graph.edge_types)

# Ensure preprocessing completed successfully
python scripts/preprocess_all.py --force
```

---

## GPU and Memory Issues

### No GPU Detected

**Problem**: Model training on CPU despite GPU availability

**Solution**:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Explicitly set device
export CUDA_VISIBLE_DEVICES=0

# Or in Python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Multiple GPUs Not Utilized

**Problem**: Only one GPU is being used

**Solution**:
```python
# Use DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Or DistributedDataParallel (recommended)
torchrun --nproc_per_node=4 scripts/train.py --config configs/base_config.yaml
```

### CUDA Memory Fragmentation

**Problem**: `RuntimeError: CUDA error: out of memory` even with sufficient memory

**Solution**:
```python
# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# Use memory-efficient settings
model.eval()  # During inference
with torch.no_grad():  # Disable gradient tracking
    predictions = model(batch)
```

### Slow Data Loading

**Problem**: Training bottlenecked by data loading

**Solution**:
```python
# Increase DataLoader workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Increase from default
    pin_memory=True,  # Faster GPU transfers
    prefetch_factor=2
)

# Preprocess and cache data
python scripts/preprocess_all.py --cache
```

---

## Evaluation Problems

### Metrics Not Computing

**Problem**: `ValueError: No positive samples in batch`

**Solution**:
```python
# Ensure test set has positive examples
print(f"Positive rate: {df['label'].mean()}")

# Use stratified sampling
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, stratify=df['label'], test_size=0.15)

# Check minimum sample count
assert test['label'].sum() > 0, "No positive samples in test set"
```

### Case Study Genes Not Found

**Problem**: `KeyError: 'UBE3A' not in gene vocabulary`

**Solution**:
```python
# Check gene name mapping
print(gene_to_idx.keys())

# Use Ensembl IDs instead of symbols
gene = 'ENSG00000114062'  # Instead of 'UBE3A'

# Or update ID mapping
python src/data/preprocess.py --update-mapping
```

### AUROC Score is 0.5

**Problem**: Model performs at random chance

**Possible causes**:
1. **Model not trained**: Load correct checkpoint
2. **Data leakage**: Check train/test split
3. **Feature mismatch**: Verify input preprocessing
4. **Wrong mode**: Ensure `model.eval()` is called

**Debugging**:
```python
# Check predictions are not constant
print(predictions.min(), predictions.max(), predictions.std())

# Verify labels
print(labels.unique(), labels.mean())

# Check if model is loaded correctly
print(model.state_dict()['gnn.layers.0.weight'].mean())
```

---

## Performance Issues

### Training is Very Slow

**Solutions**:

1. **Profile bottlenecks**
   ```python
   with torch.profiler.profile(activities=[
       torch.profiler.ProfilerActivity.CPU,
       torch.profiler.ProfilerActivity.CUDA
   ]) as prof:
       for batch in dataloader:
           model(batch)
   
   print(prof.key_averages().table())
   ```

2. **Optimize data loading** (see above)

3. **Reduce validation frequency**
   ```yaml
   training:
     validate_every: 5  # Every 5 epochs instead of every epoch
   ```

4. **Use gradient accumulation**
   ```yaml
   training:
     gradient_accumulation_steps: 4
     batch_size: 8  # Effective batch size: 8 * 4 = 32
   ```

### High CPU Usage

**Problem**: CPU at 100% during training

**Solution**:
```python
# Limit PyTorch threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Reduce DataLoader workers
num_workers=2  # Instead of 8

# Disable unnecessary logging
logging.getLogger('transformers').setLevel(logging.WARNING)
```

### Model Overfitting

**Problem**: Training accuracy high, validation accuracy low

**Solutions**:

1. **Increase dropout**
   ```yaml
   model:
     gnn:
       dropout: 0.5  # Increase from 0.2
   ```

2. **Add weight decay**
   ```yaml
   training:
     weight_decay: 0.01
   ```

3. **Early stopping**
   ```yaml
   training:
     early_stopping_patience: 5
   ```

4. **More regularization**
   ```yaml
   training:
     label_smoothing: 0.1
   ```

### Underfitting

**Problem**: Both training and validation accuracy low

**Solutions**:

1. **Increase model capacity**
   ```yaml
   model:
     gnn:
       hidden_dim: 1024
       num_layers: 4
   ```

2. **Train longer**
   ```yaml
   training:
     num_epochs: 200
   ```

3. **Increase learning rate**
   ```yaml
   training:
     learning_rate: 0.0005
   ```

4. **Remove regularization**
   ```yaml
   model:
     gnn:
       dropout: 0.1
   training:
     weight_decay: 0.001
   ```

---

## Common Error Messages

### `ImportError: cannot import name 'MessagePassing' from 'torch_geometric.nn'`

**Cause**: PyTorch Geometric not installed correctly

**Solution**: Reinstall PyG (see [Installation Issues](#installation-issues))

### `TypeError: forward() got an unexpected keyword argument 'edge_index'`

**Cause**: Model signature mismatch

**Solution**: Check model definition matches PyG conventions

### `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Tensors on different devices (CPU vs GPU)

**Solution**:
```python
# Move all inputs to same device
batch = batch.to(device)
labels = labels.to(device)
```

### `FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/biomedical_graph.pt'`

**Cause**: Data not preprocessed

**Solution**:
```bash
python scripts/preprocess_all.py
```

---

## Getting Help

If your issue isn't covered here:

1. **Check logs**: Review training logs in `logs/` directory
2. **Enable debug mode**: `export CUDA_LAUNCH_BLOCKING=1`
3. **Minimal reproducible example**: Isolate the problem
4. **GitHub Issues**: Open an issue with:
   - Error message and full traceback
   - System information (`nvidia-smi`, `pip list`)
   - Steps to reproduce
   - Configuration file used

---

## Useful Debugging Commands

```bash
# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi  # Monitor continuously

# Check Python environment
pip list | grep torch
python -c "import torch; print(torch.__version__)"

# Check CUDA
python -c "import torch; print(torch.version.cuda)"

# Check PyG
python -c "import torch_geometric; print(torch_geometric.__version__)"

# Profile memory
python -m torch.utils.bottleneck scripts/train.py

# Check disk usage
df -h
du -sh data/ checkpoints/

# Monitor training
tail -f logs/training.log
watch -n 10 'ls -lh checkpoints/promptgfm_film/'
```

---

For architecture questions, see [ARCHITECTURE.md](ARCHITECTURE.md).
For deployment issues, see [DEPLOYMENT.md](DEPLOYMENT.md).
