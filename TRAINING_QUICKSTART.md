# PromptGFM-Bio Training - Quick Start

**Last Updated:** February 17, 2026

---

## ✅ What's New

**Enhanced Features:**
- ✅ **Checkpoint saved after EVERY epoch** (not just every 10)
- ✅ **Resume from any checkpoint** (A/B/C options)
- ✅ **Detailed progress statistics** (time, ETA, speed)
- ✅ **Resource-efficient logging** (optimized for GPU training)
- ✅ **Interactive resume script** (easy checkpoint management)

---

## 🚀 Training Commands

### Start Fresh Training
```powershell
cd E:\Lab\DLG\PromptGMF-Bio
conda activate promptgfm
python scripts/train.py --config configs/finetune_config.yaml
```

### Resume Training (Interactive - Easiest!)
```powershell
python scripts/resume_training.py --interactive
```

**You'll get:**
```
Choose an option:
  A) Resume from last checkpoint (Epoch 5)
  B) Start fresh (archive current checkpoints)
  C) Resume from custom epoch
  Q) Quit
```

### Resume Training (Command Line)
```powershell
# Auto-resume from last checkpoint
python scripts/resume_training.py

# Resume from specific epoch
python scripts/resume_training.py --epoch 3

# Archive and start fresh
python scripts/resume_training.py --archive
```

---

## 📊 What You'll See

### During Training (Per Batch)
```
Epoch 5/100: 47%|██████| 13844/29254 [1:48:35<48:11, 5.33it/s, loss=0.0002, avg=0.0003, eta=12m5s]
```

### After Each Epoch
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

---

## 💾 Checkpoint Files

**After every epoch, you'll get:**
- `checkpoint_epoch_1.pt` - Full training state from epoch 1
- `checkpoint_epoch_2.pt` - Full training state from epoch 2
- `checkpoint_epoch_3.pt` - Full training state from epoch 3
- ... (one per epoch)
- `best_model.pt` - Best performing model (auto-updated)

**Each checkpoint is ~50-100 MB**

---

## 🎯 Resume Options Explained

### Option A: Resume from Last Checkpoint
**When to use:** Training was interrupted, want to continue

```powershell
python scripts/resume_training.py
# Automatically resumes from last saved epoch
```

**Example:**
- You stopped at Epoch 5
- Resume → continues from Epoch 6
- Preserves optimizer, learning rate, early stopping counter

### Option B: Start Fresh (Archive Old)
**When to use:** Try new hyperparameters, keep old results

```powershell
python scripts/resume_training.py --archive
```

**What happens:**
- Old checkpoints moved to `checkpoints/promptgfm_film_archive_TIMESTAMP/`
- New training starts from Epoch 1
- Old results preserved for comparison

### Option C: Resume from Custom Epoch
**When to use:** Go back to earlier checkpoint (e.g., before overfitting)

```powershell
python scripts/resume_training.py --epoch 3
```

**Example:**
- Val AUROC peaked at Epoch 3, declined after
- Resume from 3 → continue with different settings

---

## ⏱️ Expected Performance (RTX 4060 8GB)

| Metric | Value |
|--------|-------|
| **Per Epoch** | ~2.5 hours |
| **Batches/sec** | ~3-5 it/s |
| **Full Training (100 epochs)** | ~10 days |
| **With Early Stopping (typical)** | ~2-3 days (20-30 epochs) |

---

## 🔍 Monitoring Training

### Check GPU Status (New Window)
```powershell
nvidia-smi -l 2  # Update every 2 seconds
```

### List Checkpoints
```powershell
Get-ChildItem checkpoints\promptgfm_film\*.pt
```

### View Checkpoint Metrics
```powershell
Get-Content checkpoints\promptgfm_film\checkpoint_epoch_5.json
```

---

## 🛑 Stopping & Resuming

### To Stop Training
**Press `Ctrl+C` in the training window**

**What happens:**
- Training stops gracefully
- Last completed epoch's checkpoint is saved
- Safe to stop anytime (no data loss)

### To Resume Later
```powershell
python scripts/resume_training.py --interactive
# Choose option A (Resume from last checkpoint)
```

---

## ⚙️ Configuration (Optional)

Edit `configs/finetune_config.yaml` to adjust:

```yaml
training:
  num_epochs: 100              # Total epochs
  batch_size: 32               # Reduce to 16 if OOM errors
  learning_rate: 0.0005        # Learning rate
  early_stopping_patience: 15  # Stop if no improvement

model:
  hidden_dim: 256              # Model size
  num_layers: 3                # GNN depth
  conditioning_type: film      # Prompt conditioning
```

---

## 🆘 Quick Fixes

### Training Too Slow (< 1 it/s)
```powershell
# Check GPU is active
python -c "import torch; print(f'GPU Active: {torch.cuda.is_available()}')"

# Close other programs
# Check nvidia-smi shows GPU usage
```

### Out of Memory
```yaml
# Edit configs/finetune_config.yaml
training:
  batch_size: 16  # Reduce from 32
```

### Checkpoint Not Found
```powershell
# Use interactive mode to see what's available
python scripts/resume_training.py --interactive
```

---

## 📋 Complete Workflow Example

### Day 1 Evening (Start Training)
```powershell
cd E:\Lab\DLG\PromptGMF-Bio
conda activate promptgfm
python scripts/train.py --config configs/finetune_config.yaml

# Let run overnight... zzz 💤
```

### Day 2 Morning (Check Progress)
```
# Training automatically shows progress
# Epoch 10/100 Complete  ← Still running
# Val AUROC: 0.6234
```

### Day 2 Afternoon (Need to Stop)
```
Press Ctrl+C

# Training stopped at Epoch 12
# Checkpoint saved: checkpoint_epoch_12.pt  ✓
```

### Day 2 Evening (Resume)
```powershell
python scripts/resume_training.py --interactive
# Choose option A (Resume from last checkpoint)

# Training resumes from Epoch 13... 🚀
```

---

## 📖 Full Documentation

For detailed information, see:
- **[TRAINING_RESUME_GUIDE.md](TRAINING_RESUME_GUIDE.md)** - Complete guide with all features
- **[GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)** - GPU-specific setup

---

## 🎉 Summary

**3 Simple Commands You Need:**

1. **Start Training:**
   ```powershell
   python scripts/train.py --config configs/finetune_config.yaml
   ```

2. **Resume Training:**
   ```powershell
   python scripts/resume_training.py --interactive
   ```

3. **Check Checkpoints:**
   ```powershell
   Get-ChildItem checkpoints\promptgfm_film\*.pt
   ```

**That's it! Training now:**
- ✅ Saves after every epoch
- ✅ Shows detailed progress (time, ETA, metrics)
- ✅ Can be stopped and resumed anytime
- ✅ Automatically manages checkpoints

**Happy Training! 🚀**
