# PromptGFM-Bio: Complete Training & Resume Guide

**Comprehensive guide for running training manually in PowerShell with checkpoint management and resume functionality.**

Last Updated: February 17, 2026  
Version: 2.0 (with checkpoint resume support)

---

## 🎯 Quick Start Commands

### Start New Training
```powershell
cd E:\Lab\DLG\PromptGMF-Bio
conda activate promptgfm
python scripts/train.py --config configs/finetune_config.yaml
```

### Resume Training (Interactive - Recommended)
```powershell
python scripts/resume_training.py --interactive
```

### Resume from Last Checkpoint
```powershell
python scripts/resume_training.py
```

---

## 📋 Table of Contents

1. [Initial Setup](#initial-setup)
2. [Starting Training](#starting-training)
3. [Understanding Output](#understanding-output)
4. [Checkpoint Management](#checkpoint-management)
5. [Resume Options](#resume-options)
6. [Troubleshooting](#troubleshooting)
7. [Complete Workflow Examples](#complete-workflow-examples)

---

## 1. Initial Setup

### Step 1.1: Activate Environment

```powershell
# Navigate to project directory
cd E:\Lab\DLG\PromptGMF-Bio

# Activate conda environment
conda activate promptgfm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.1.0+cu118
CUDA Available: True
```

### Step 1.2: Check GPU Status

```powershell
# Check NVIDIA GPU (open new PowerShell window if training is running)
nvidia-smi

# Or in Python
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected:**
```
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

### Step 1.3: Verify Data Files

```powershell
# Check preprocessed graph
Test-Path data/processed/biomedical_graph.pt

# Check edge file
Test-Path data/processed/hpo_gene_disease_edges.csv

# View statistics (optional)
Get-Content data/processed/biomedical_graph_stats.txt
```

**All should return `True`. If not, run preprocessing:**
```powershell
python scripts/preprocess_all.py
```

---

## 2. Starting Training

### Option A: Fresh Training (First Time)

```powershell
# Make sure you're in the project directory
cd E:\Lab\DLG\PromptGMF-Bio

# Activate environment
conda activate promptgfm

# Start training
python scripts/train.py --config configs/finetune_config.yaml
```

**What Happens:**
- Loads biomedical graph
- Splits data (80% train, 10% val, 10% test)
- Creates PromptGFM model with BioBERT
- Starts training from Epoch 1
- Saves checkpoint after every epoch

### Option B: Resume Previous Training

**Interactive Mode (Easiest):**
```powershell
python scripts/resume_training.py --interactive
```

**You'll see an interactive menu:**
```
======================================================================
  PROMPTGFM-BIO TRAINING RESUME
======================================================================

Enter config file path [configs/finetune_config.yaml]: ▌

Found 5 checkpoint(s) in checkpoints\promptgfm_film:

----------------------------------------------------------------------
  Epoch   1 | AUROC: 0.5234 | AUPR: 0.6123 | Loss: 0.000234
  Epoch   2 | AUROC: 0.5456 | AUPR: 0.6345 | Loss: 0.000198
  Epoch   3 | AUROC: 0.5678 | AUPR: 0.6567 | Loss: 0.000167
  Epoch   4 | AUROC: 0.5823 | AUPR: 0.6712 | Loss: 0.000149
  Epoch   5 | AUROC: 0.5891 | AUPR: 0.6789 | Loss: 0.000143
----------------------------------------------------------------------

Choose an option:
  A) Resume from last checkpoint (Epoch 5)
  B) Start fresh (archive current checkpoints)
  C) Resume from custom epoch
  Q) Quit

Your choice [A/B/C/Q]: ▌
```

**Command-Line Options:**
```powershell
# Auto-resume from last checkpoint
python scripts/resume_training.py

# Resume from specific epoch
python scripts/resume_training.py --epoch 3

# Resume from specific checkpoint file
python scripts/resume_training.py --checkpoint checkpoints/promptgfm_film/checkpoint_epoch_5.pt

# Archive old checkpoints and start completely fresh
python scripts/resume_training.py --archive
```

---

## 3. Understanding Output

### 3.1 Startup Messages

When training starts, you'll see:
```
============================================================
Starting Supervised Fine-tuning
============================================================
INFO: Creating dataloaders...
INFO: Loading graph from data\processed\biomedical_graph.pt
INFO: Graph loaded: gene=5251, disease=12985, phenotype=11794
INFO: Split sizes: train=936114, val=117014, test=117015
INFO: Training PromptGFM
INFO: PromptEncoder initialized - hidden_size: 768
INFO: GNN Backbone: graphsage, 3 layers, 128→256→256
INFO: PromptGFM initialized
```

### 3.2 Training Progress (Per Epoch)

**During Each Epoch:**
```
Epoch 5/100: 100%|████████████| 29254/29254 [2:41:30<00:00, 3.02it/s, loss=0.0002, avg=0.0003, eta=0m0s]
```

**Explanation:**
- `Epoch 5/100`: Current epoch / total epochs
- `29254/29254`: Batches completed / total batches
- `[2:41:30<00:00]`: Time elapsed / time remaining
- `3.02it/s`: Processing speed (batches per second)
- `loss=0.0002`: Current batch loss
- `avg=0.0003`: Average loss for this epoch
- `eta=0m0s`: Estimated time remaining for this epoch

### 3.3 Epoch Summary

**After Each Epoch:**
```
======================================================================
Epoch 5/100 Complete
======================================================================
  Time: 9724.3s (Avg: 9724.3s/epoch)
  ETA: 26h 44m (for 95 epochs)
  Train Loss: 0.000234
  Val Loss:   0.000198
  Val AUROC:  0.5678
  Val AUPR:   0.6345
  ✓ New best auroc: 0.5678 (saved as best_model.pt)
  💾 Checkpoint saved: checkpoint_epoch_5.pt
======================================================================
```

**Key Metrics Explained:**

| Metric | Range | Description | Target |
|--------|-------|-------------|--------|
| **Train Loss** | 0-1 | Training error | < 0.001 |
| **Val Loss** | 0-1 | Validation error | < 0.001 |
| **AUROC** | 0-1 | Area Under ROC Curve | > 0.80 |
| **AUPR** | 0-1 | Area Under Precision-Recall | > 0.75 |
| **Time** | seconds | Epoch duration | Varies |
| **ETA** | h:m | Time to completion | Varies |

**What's Good:**
- ✅ Train Loss decreasing each epoch
- ✅ Val AUROC/AUPR increasing
- ✅ "New best" messages appearing
- ✅ Consistent processing speed (it/s stable)

**What's Problematic:**
- ⚠️ Val AUROC/AUPR = `nan` (missing negative samples)
- ⚠️ Train Loss not decreasing (learning issue)
- ⚠️ Val metrics decreasing (overfitting)
- ⚠️ Very slow speed (< 1 it/s)

### 3.4 Validation Phase

```
Validating: 100%|██████████████████████████| 3657/3657 [02:53<00:00, 21.04it/s]
```

This runs after each training epoch to assess model performance on held-out data.

---

## 4. Checkpoint Management

### 4.1 Automatic Checkpoint Saving

**Every Epoch:**
- File: `checkpoint_epoch_N.pt` (e.g., `checkpoint_epoch_5.pt`)
- Contains: Full training state (model, optimizer, history, metrics)
- Size: ~50-100 MB each
- Purpose: Resume training from any epoch

**Best Model:**
- File: `best_model.pt`
- Contains: Best performing model so far
- Updated: When validation metric improves
- Purpose: Use this for final predictions

### 4.2 What's in Each Checkpoint

```python
{
    'epoch': 5,                      # Completed epoch number
    'global_step': 146270,           # Total training steps
    'model_state_dict': {...},       # Model weights (5M parameters)
    'optimizer_state_dict': {...},   # Optimizer state (Adam momentum, etc.)
    'best_val_metric': 0.5678,       # Best validation score so far
    'epochs_without_improvement': 2, # Early stopping counter
    'train_losses': [0.01, ...],     # Loss history
    'val_metrics': [{...}, ...],     # Validation metrics history
    'current_metrics': {             # This epoch's results
        'auroc': 0.5678,
        'aupr': 0.6345,
        'loss': 0.000234
    }
}
```

### 4.3 Listing Checkpoints

```powershell
# List all checkpoints with timestamps
Get-ChildItem checkpoints\promptgfm_film\*.pt | Sort-Object LastWriteTime | Format-Table Name, LastWriteTime, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}

# Count checkpoints
(Get-ChildItem checkpoints\promptgfm_film\checkpoint_epoch_*.pt).Count

# View latest checkpoint metrics
Get-Content checkpoints\promptgfm_film\checkpoint_epoch_5.json
```

**Example Output:**
```
Name                        LastWriteTime          Size(MB)
----                        -------------          --------
checkpoint_epoch_1.pt       2/17/2026 10:23:45 PM     52.34
checkpoint_epoch_2.pt       2/17/2026 11:46:12 PM     52.34
checkpoint_epoch_3.pt       2/18/2026 1:08:39 AM      52.34
checkpoint_epoch_4.pt       2/18/2026 2:31:06 AM      52.34
checkpoint_epoch_5.pt       2/18/2026 3:53:33 AM      52.34
best_model.pt               2/18/2026 3:53:33 AM      52.34
```

### 4.4 Managing Disk Space

**Delete Old Checkpoints (Keep Best Only):**
```powershell
# Keep only best model and last 3 checkpoints
$keep = 3
Get-ChildItem checkpoints\promptgfm_film\checkpoint_epoch_*.pt | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -Skip $keep | 
    Remove-Item -Verbose
```

**Archive to External Drive:**
```powershell
# Archive to external/network drive
$archive = "D:\Backups\PromptGFM\$(Get-Date -Format 'yyyyMMdd')"
New-Item -ItemType Directory -Path $archive -Force
Copy-Item checkpoints\promptgfm_film\*.pt $archive\
```

---

## 5. Resume Options

### Option A: Resume from Last Checkpoint

**Use Case:** Training was interrupted (power loss, Ctrl+C, etc.)

**Command:**
```powershell
python scripts/resume_training.py --interactive
# Then choose option 'A'
```

**Or:**
```powershell
python scripts/resume_training.py
```

**What Happens:**
1. Loads `checkpoint_epoch_N.pt` (highest N)
2. Restores model weights
3. Restores optimizer state (learning rate, momentum)
4. Continues from Epoch N+1
5. Preserves training history and early stopping counter

**Example:**
```
✓ Resumed from epoch 5
  Best val metric: 0.5678
  Global step: 146270

Starting training from epoch 6 to 100
```

### Option B: Start Fresh (Archive Old Checkpoints)

**Use Case:** Want to try different hyperparameters but keep old results

**Command:**
```powershell
python scripts/resume_training.py --interactive
# Then choose option 'B'
```

**Or:**
```powershell
python scripts/resume_training.py --archive
```

**What Happens:**
1. Creates archive folder: `checkpoints/promptgfm_film_archive_20260217_120000/`
2. Moves all `.pt` and `.json` files to archive
3. Starts training from Epoch 1
4. Creates new checkpoints in original folder

**Example:**
```
⚠ Archive checkpoints and start fresh? [y/N]: y
✓ Archived 5 files to: checkpoints/promptgfm_film_archive_20260217_154523

✓ Starting fresh training...
Epoch 1/100: ...
```

### Option C: Resume from Custom Epoch

**Use Case:** Later epochs showed overfitting, want to resume from earlier checkpoint

**Command:**
```powershell
python scripts/resume_training.py --interactive
# Then choose option 'C' and enter epoch number
```

**Or:**
```powershell
python scripts/resume_training.py --epoch 3
```

**What Happens:**
1. Shows list of available epochs
2. Loads specified checkpoint (e.g., epoch 3)
3. Continues from that epoch (epoch 4, 5, 6...)
4. **Note:** Will overwrite epoch_4.pt, epoch_5.pt if they exist

**Example:**
```
Enter epoch number to resume from: 3

✓ Resuming from Epoch 3
  AUROC: 0.5678, AUPR: 0.6567

Starting training from epoch 4 to 100
```

**When to Use:**
- Validation metrics peaked at epoch 3, dropped after
- Want to try different learning rate from epoch 3 onward
- Experimental branching from earlier state

---

## 6. Troubleshooting

### Issue 1: Training Very Slow

**Symptom:** < 1 it/s (batches per second)

**Solutions:**

1. **Verify GPU is being used:**
```powershell
python -c "import torch; print(f'Using GPU: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```

2. **Check GPU utilization:**
```powershell
nvidia-smi

# Should show:
# GPU Memory Usage: 3000-4000 MB / 8192 MB
# GPU Utilization: 80-100%
```

3. **Close unnecessary programs:**
- Browsers (especially with many tabs)
- Other GPU-using applications
- Background processes

4. **Reduce batch size if swapping to CPU:**
Edit `configs/finetune_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduced from 32
```

### Issue 2: CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

**Quick Fix:**
```powershell
# 1. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# 2. Reduce batch size
# Edit configs/finetune_config.yaml
#   batch_size: 32 → 16
```

**Configuration Changes:**
```yaml
# configs/finetune_config.yaml

training:
  batch_size: 16           # Reduced from 32
  
model:
  hidden_dim: 128          # Reduced from 256
  num_layers: 2            # Reduced from 3
```

**Restart Training:**
```powershell
python scripts/train.py --config configs/finetune_config.yaml
```

### Issue 3: Validation Metrics = NaN

**Symptom:**
```
Val AUROC: nan
Val AUPR:  nan
WARNING: Only one class present in y_true. AUROC is undefined.
```

**Cause:** Validation batches contain only positive or only negative samples

**Impact:**
- Training loss still works (model is learning)
- Early stopping won't trigger (no valid metric to compare)
- Training will run for all epochs

**Solutions:**

1. **Wait and Monitor:** Sometimes resolves after a few epochs
2. **Check data distribution:**
```powershell
python -c "import pandas as pd; edges = pd.read_csv('data/processed/hpo_gene_disease_edges.csv'); print(edges['score'].describe())"
```

3. **Add negative sampling** (requires code modification):
   - Edit `src/data/dataset.py` to include negative edges
   - Typically not needed if using ranking loss

### Issue 4: Checkpoint Not Found

**Symptom:**
```
ERROR: Checkpoint not found: checkpoints/promptgfm_film/checkpoint_epoch_5.pt
```

**Solutions:**

1. **List available checkpoints:**
```powershell
Get-ChildItem checkpoints\promptgfm_film\*.pt
```

2. **Use interactive mode:**
```powershell
python scripts/resume_training.py --interactive
# Shows all available checkpoints
```

3. **Check checkpoint directory in config:**
Edit `configs/finetune_config.yaml`:
```yaml
training:
  checkpoint_dir: checkpoints/promptgfm_film  # Verify this path
```

### Issue 5: Training Hangs on Data Loading

**Symptom:** Stuck at "Creating dataloaders..." or "Loading graph..."

**Solutions:**

1. **Check if files exist:**
```powershell
Test-Path data/processed/biomedical_graph.pt
Test-Path data/processed/hpo_gene_disease_edges.csv
```

2. **Check file sizes (should be reasonable):**
```powershell
Get-ChildItem data\processed\*.pt, data\processed\*.csv | Format-Table Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
```

3. **Try loading manually to see error:**
```powershell
python -c "import torch; graph = torch.load('data/processed/biomedical_graph.pt'); print('Graph loaded:', graph)"
```

4. **Regenerate if corrupted:**
```powershell
# Backup old file
Move-Item data\processed\biomedical_graph.pt data\processed\biomedical_graph_old.pt

# Regenerate
python scripts/preprocess_all.py
```

---

## 7. Complete Workflow Examples

### Scenario A: Overnight Training

**Evening (6:00 PM):**
```powershell
# 1. Setup
cd E:\Lab\DLG\PromptGMF-Bio
conda activate promptgfm

# 2. Quick GPU check
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 3. Start training
python scripts/train.py --config configs/finetune_config.yaml

# 4. Let it run overnight...
```

**Morning (8:00 AM):**

Training still running? Check progress in PowerShell output (automatically displays).

**Afternoon (2:00 PM):**

Need to use computer for something else? Stop training:
```powershell
# Press Ctrl+C in training window

# You'll see:
# KeyboardInterrupt
# Training interrupted at Epoch 7
# Checkpoint saved: checkpoint_epoch_7.pt
```

**Evening (6:00 PM) - Resume:**
```powershell
cd E:\Lab\DLG\PromptGMF-Bio
conda activate promptgfm

# Resume from where you left off
python scripts/resume_training.py

# Training continues from Epoch 8...
```

### Scenario B: Hyperparameter Tuning

**First Run:**
```powershell
# Train with default parameters
python scripts/train.py --config configs/finetune_config.yaml

# After 10 epochs, not converging well
# Stop with Ctrl+C
```

**Archive and Try Different Parameters:**
```powershell
# Archive first attempt
python scripts/resume_training.py --archive

# Edit configs/finetune_config.yaml:
#   learning_rate: 0.0005 → 0.001 (higher learning rate)
#   num_layers: 3 → 2 (simpler model)

# Start new training
python scripts/train.py --config configs/finetune_config.yaml
```

**Compare Results:**
```powershell
# Check archived run
Get-Content checkpoints\promptgfm_film_archive_20260217_154523\checkpoint_epoch_10.json

# Check new run
Get-Content checkpoints\promptgfm_film\checkpoint_epoch_10.json
```

### Scenario C: Recovering from Overfitting

**Problem:** Validation AUROC peaked at epoch 15, then declined

**Check History:**
```powershell
# View all checkpoint metrics
Get-ChildItem checkpoints\promptgfm_film\checkpoint_epoch_*.json | 
    ForEach-Object { Get-Content $_ | ConvertFrom-Json } | 
    Format-Table epoch, @{Label="AUROC";Expression={$_.metrics.auroc}}, @{Label="AUPR";Expression={$_.metrics.aupr}}
```

**Example Output:**
```
Epoch  AUROC   AUPR
-----  -----   ----
1      0.5234  0.6123
5      0.6456  0.7234
10     0.7123  0.7856
15     0.7891  0.8234  ← Peak
20     0.7654  0.8123  ← Starting to decline
25     0.7234  0.7912
```

**Solution: Resume from Epoch 15**
```powershell
# Backup current checkpoints
New-Item -ItemType Directory -Path "checkpoints\promptgfm_film_overfit" -Force
Copy-Item checkpoints\promptgfm_film\checkpoint_epoch_*.pt checkpoints\promptgfm_film_overfit\

# Resume from epoch 15
python scripts/resume_training.py --epoch 15

# Modify training (optional):
# - Lower learning rate
# - Add regularization
# - Increase dropout
```

---

## 8. Configuration Reference

### Key Configuration Options

Edit `configs/finetune_config.yaml`:

```yaml
# Training
training:
  num_epochs: 100                    # Total epochs
  batch_size: 32                     # Samples per batch (reduce if OOM)
  learning_rate: 0.0005              # Learning rate
  weight_decay: 0.0001               # L2 regularization
  gradient_clip: 1.0                 # Gradient clipping value
  early_stopping_patience: 15        # Stop if no improvement for N epochs
  val_metric: auroc                  # Metric for early stopping (auroc or aupr)
  checkpoint_dir: checkpoints/promptgfm_film
  
# Model
model:
  gnn_type: graphsage                # GNN architecture
  hidden_dim: 256                    # Model dimension
  num_layers: 3                      # Number of GNN layers
  dropout: 0.3                       # Dropout rate
  conditioning_type: film            # Prompt conditioning (film, cross_attention)
  
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls            # How to pool BERT output
    max_length: 512                  # Max sequence length
    freeze_encoder: false            # Whether to freeze BERT
    
# Data
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  min_score: 0.3                     # Minimum edge confidence
  
# Hardware
hardware:
  device: cuda                       # cuda or cpu
  num_workers: 0                     # DataLoader workers
  
# Logging
logging:
  use_wandb: false                   # Weights & Biases logging
  log_interval: 100                  # Log every N steps
```

---

## 9. Performance Expectations

### Timing (RTX 4060 8GB)

| Stage | Time | Notes |
|-------|------|-------|
| Data Loading | 30s | One-time at startup |
| Epoch (Train) | ~2.5 hours | 29,254 batches @ ~3 it/s |
| Epoch (Val) | ~3 minutes | 3,657 batches @ ~21 it/s |
| **Total per Epoch** | **~2.5 hours** | |
| **Full Training (100 epochs)** | **~10 days** | With early stopping: 2-3 days |

### Expected Metrics

| Metric | Initial | Mid-Training | Final |
|--------|---------|--------------|-------|
| AUROC | 0.50-0.55 | 0.70-0.75 | 0.80-0.85 |
| AUPR | 0.60-0.65 | 0.75-0.80 | 0.85-0.90 |
| Train Loss | 0.01-0.05 | 0.001-0.005 | 0.0001-0.001 |

---

## 10. Quick Reference Card

**Print this section for easy access:**

```
═══════════════════════════════════════════════════════════════
  PROMPTGFM-BIO TRAINING CHEAT SHEET
═══════════════════════════════════════════════════════════════

SETUP
  cd E:\Lab\DLG\PromptGMF-Bio
  conda activate promptgfm

START FRESH
  python scripts/train.py --config configs/finetune_config.yaml

RESUME (INTERACTIVE)
  python scripts/resume_training.py --interactive
  
RESUME (AUTO FROM LAST)
  python scripts/resume_training.py

RESUME (SPECIFIC EPOCH)
  python scripts/resume_training.py --epoch 5

ARCHIVE & START FRESH
  python scripts/resume_training.py --archive

CHECK GPU
  nvidia-smi

LIST CHECKPOINTS
  Get-ChildItem checkpoints\promptgfm_film\*.pt

VIEW METRICS
  Get-Content checkpoints\promptgfm_film\checkpoint_epoch_5.json

STOP TRAINING
  Ctrl+C (checkpoint auto-saved)

═══════════════════════════════════════════════════════════════
```

---

## 📞 Support & Questions

**Common Questions:**

**Q: How long will training take?**  
A: ~2.5 hours per epoch. With early stopping (patience=15), expect 20-30 epochs = 2-3 days total.

**Q: Can I use my computer while training?**  
A: Yes for light tasks. Avoid other GPU-intensive applications. You can stop and resume anytime.

**Q: What if my computer crashes during training?**  
A: No problem! Checkpoints are saved after every epoch. Just run `python scripts/resume_training.py` to continue.

**Q: How do I know if training is working?**  
A: Watch for:
- Train Loss decreasing each epoch
- Val AUROC increasing (target: > 0.80)
- "New best" messages appearing

**Q: What's the best checkpoint to use for predictions?**  
A: Use `checkpoints/promptgfm_film/best_model.pt` - it's automatically saved when validation metrics are best.

---

**Last Updated:** February 17, 2026  
**Version:** 2.0 (Enhanced with resume functionality)  
**Compatible with:** PromptGFM-Bio v1.0+

**Happy Training! 🚀**
