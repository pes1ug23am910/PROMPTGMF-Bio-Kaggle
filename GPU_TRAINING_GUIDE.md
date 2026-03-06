# GPU Training Guide for RTX 4060

## Hardware Configuration

**Your GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM**: 8 GB
- **Compute Capability**: 8.9 (Ada Lovelace)
- **CUDA Cores**: 3,072
- **Performance**: ~8-9x speedup vs CPU

## ✅ GPU Status: FULLY OPERATIONAL

All tests passed! Your environment is ready for GPU-accelerated training.

---

## Optimal Training Settings for 8GB VRAM

### Memory Budget Breakdown (Approximate)

For your 8GB VRAM:
- **Available for training**: ~7GB (1GB reserved for OS/display)
- **Model parameters**: ~500MB (with hidden_dim=512, 3 layers)
- **Optimizer state**: ~500MB (AdamW doubles memory)
- **Activations & gradients**: ~2-3GB (depends on batch size)
- **Graph data**: ~2-3GB (depends on graph size)
- **Buffer**: Keep ~1-2GB free for safety

### Recommended Batch Sizes

Based on your 8GB VRAM and typical GNN memory requirements:

**Pretraining (Large Graph)**
```yaml
batch_size: 32-48  # Start with 32, increase if stable
gradient_accumulation_steps: 2  # Effective batch size = 64-96
```

**Finetuning (Disease-specific)**
```yaml
batch_size: 32-64  # Finetuning needs less memory
```

**If you get OOM (Out of Memory) errors:**
```yaml
batch_size: 16-24
gradient_accumulation_steps: 4
mixed_precision: true  # Already enabled
```

---

## Configuration Files Updated

All config files now have GPU-optimized settings:

### ✅ base_config.yaml
- `device: 'cuda'`
- `mixed_precision: true` (Automatic Mixed Precision for 30-40% speedup)
- `cudnn_benchmark: true` (Optimizes convolutions)

### ✅ pretrain_config.yaml
- Optimized for pretraining phase
- Batch size: 32 (conservative for large graphs)

### ✅ finetune_config.yaml
- Optimized for finetuning phase
- Batch size: 32 (safe default)

---

## Performance Optimization Tips

### 1. **Use Mixed Precision Training (AMP)**
Already enabled in configs. This gives you:
- 30-40% faster training
- ~50% memory savings
- Minimal accuracy loss

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    with autocast():  # Automatic mixed precision
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. **Gradient Accumulation**
If batch size is too large for memory:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. **Monitor GPU Memory**
Use these commands during training:

```python
import torch

# Check current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### 4. **Graph Sampling for Large Graphs**
If your PPI graph is too large:
```python
from torch_geometric.loader import NeighborLoader

# Sample k-hop neighborhoods instead of full graph
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10, 10],  # 3-layer GNN
    batch_size=32,
    shuffle=True
)
```

---

## Expected Training Times (Estimates)

With your RTX 4060:

**Pretraining** (50 epochs on large PPI graph)
- Per epoch: ~10-15 minutes
- Total: ~8-12 hours

**Finetuning** (100 epochs on gene-disease associations)
- Per epoch: ~2-5 minutes
- Total: ~3-8 hours

**Baselines**
- GNN-only: ~2-3 hours
- Static text concat: ~4-6 hours

---

## Memory Optimization Strategies

### If You Get OOM Errors:

**1. Reduce batch size**
```yaml
batch_size: 16  # From 32
```

**2. Use gradient checkpointing**
```python
model.gradient_checkpointing_enable()
```

**3. Reduce model size**
```yaml
hidden_dim: 384  # From 512
num_layers: 2     # From 3
```

**4. Use GraphSAINT sampling**
```python
from torch_geometric.loader import GraphSAINTRandomWalkSampler

loader = GraphSAINTRandomWalkSampler(
    data,
    batch_size=6000,  # Number of nodes per batch
    walk_length=3,
    num_steps=5,
)
```

---

## Monitoring GPU During Training

### Option 1: nvidia-smi (Terminal)
```bash
# Watch GPU usage in real-time
nvidia-smi -l 1
```

### Option 2: Python (In Training Script)
```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Call during training
print_gpu_memory()
```

### Option 3: Weights & Biases
Your config already has W&B enabled. It automatically tracks:
- GPU utilization
- GPU memory usage
- GPU temperature
- Training speed

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce `batch_size` in config
2. Enable `gradient_accumulation`
3. Use graph sampling
4. Reduce `hidden_dim`
5. Clear cache: `torch.cuda.empty_cache()`

### Issue: "GPU not being used"
**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(model.device)  # Should be 'cuda:0'
```

**Force GPU:**
```python
model = model.cuda()
data = data.cuda()
```

### Issue: "Training too slow"
**Optimizations:**
1. Ensure `pin_memory=True` in DataLoader
2. Increase `num_workers` (try 4-8)
3. Use `cudnn_benchmark=True`
4. Enable AMP (already enabled)
5. Reduce logging frequency

---

## Testing Your Setup

Run the GPU test script:
```bash
conda activate promptgfm
python scripts/test_gpu.py
```

Expected output:
- ✓ CUDA Available: True
- ✓ GPU Memory: 8.00 GB
- ✓ Speedup: ~8-9x

---

## Quick Reference: Config Values

**For 8GB VRAM (Your Setup)**
```yaml
# Safe defaults
batch_size: 32
hidden_dim: 512
num_layers: 3
gradient_accumulation_steps: 2
mixed_precision: true

# If OOM:
batch_size: 16
hidden_dim: 384
num_layers: 2
gradient_accumulation_steps: 4
```

**For larger GPUs (reference)**
```yaml
# 16GB VRAM
batch_size: 64-128
hidden_dim: 768

# 24GB+ VRAM
batch_size: 128-256
hidden_dim: 1024
```

---

## Summary

✅ **Your RTX 4060 is ready for training!**

**Key Points:**
- All configs updated for GPU training
- Mixed precision enabled (30-40% faster)
- Batch size set conservatively (32)
- 8.89x speedup confirmed

**Next Steps:**
1. Start with Phase 2: Data Pipeline
2. Once data is ready, test training with small dataset
3. Monitor GPU memory with `nvidia-smi -l 1`
4. Adjust batch size based on actual memory usage

**Expected Performance:**
- Pretraining: ~10-15 min/epoch
- Finetuning: ~2-5 min/epoch
- Total project training: ~15-20 hours

Happy training! 🚀
