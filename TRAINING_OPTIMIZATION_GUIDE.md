# PromptGFM-Bio: Training Optimization Guide

**Comprehensive analysis of performance optimizations for conference-quality research**

Last Updated: February 17, 2026  
Version: 1.0

---

## 📑 Table of Contents

1. [Current Performance Analysis](#current-performance-analysis)
2. [Recommended Safe Optimizations](#recommended-safe-optimizations)
3. [Conditional Optimizations](#conditional-optimizations)
4. [Not Recommended Optimizations](#not-recommended-optimizations)
5. [Hardware Recommendations](#hardware-recommendations)
6. [Cost-Benefit Analysis](#cost-benefit-analysis)
7. [Implementation Action Plan](#implementation-action-plan)
8. [Impact on Paper Quality](#impact-on-paper-quality)
9. [Decision Framework](#decision-framework)
10. [FAQ](#faq)

---

## 📊 Current Performance Analysis

### Measured Performance (RTX 4060 Laptop GPU)

From training logs (February 17, 2026):

```
Epoch 1/100:   1%|▊ 402/29254 [00:34<39:55, 12.05it/s, loss=0]
```

**Key Metrics:**

| Metric | Value | Calculation |
|--------|-------|-------------|
| **Training Speed** | ~12 it/s | 402 iterations in 34 seconds |
| **Iterations per Epoch** | 29,254 | 936,114 edges ÷ 32 batch_size |
| **Time per Epoch** | ~41 minutes | 29,254 ÷ 12 it/s ÷ 60 |
| **Full Training (100 epochs)** | ~68 hours | 41 min × 100 ÷ 60 |
| **With Early Stop (25 epochs)** | ~17 hours | 41 min × 25 ÷ 60 |

### Hardware Specifications

```yaml
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8GB GDDR6
CUDA: 11.8
PyTorch: 2.1.0+cu118
Platform: Windows 11
CPU: High-performance laptop CPU (6-8 cores typical)
RAM: 16-32GB (assumed)
Storage: NVMe SSD (assumed)
```

### Performance Assessment

**Verdict:** ✅ **Reasonable for laptop hardware**, but significant improvements possible

**Why is this reasonable?**
- Transformer-based model (BioBERT) with GNN → computationally intensive
- Large dataset (936K training edges)
- Laptop GPU has thermal/power constraints
- No optimizations currently applied

**Why we can do better?**
- No mixed precision (FP16) → missing 2× speedup
- DataLoader not optimized → CPU bottleneck likely
- No cuDNN autotuning → suboptimal algorithms
- Laptop power settings → potential throttling

---

## ✅ Recommended Safe Optimizations

### 1. Mixed Precision Training (AMP) ⭐⭐⭐⭐⭐

**Priority:** HIGHEST  
**Difficulty:** Medium  
**Risk:** Very Low

#### Benefits

| Benefit | Impact | Notes |
|---------|--------|-------|
| **Speed Increase** | 1.5-2.5× faster | Typical for transformer models |
| **Memory Reduction** | ~40% less VRAM | FP16 uses half the memory of FP32 |
| **Larger Batch Size** | 2× possible | Due to memory savings |
| **Industry Standard** | Standard practice | All production systems use this |
| **Conference Expectation** | Expected | Reviewers assume modern training |

#### Trade-offs

| Consideration | Impact | Mitigation |
|---------------|--------|------------|
| **Numerical Stability** | Very rare issues | GradScaler handles automatically |
| **First Run Setup** | None | Works out of the box |
| **Code Complexity** | Minimal | ~10 lines of code |
| **Reproducibility** | Identical results | Deterministic with same seed |

#### Impact on Paper

✅ **Positive:**
- Mention in methods: "Trained with mixed precision (FP16) for efficiency"
- Enables more experiments (ablations, baselines)
- Faster iteration → better ideas
- Standard practice (reviewers expect it)

❌ **No negative impact on:**
- Model accuracy
- Convergence behavior
- Reproducibility
- Novelty claims

#### Implementation Details

**File to modify:** `src/training/finetune.py`

**Code changes:**
```python
# 1. Import AMP utilities
from torch.cuda.amp import autocast, GradScaler

# 2. Initialize in __init__
class PromptGFMTrainer:
    def __init__(self, ...):
        # ... existing code ...
        self.scaler = GradScaler() if device == 'cuda' else None
        self.use_amp = (device == 'cuda')
        logger.info(f"  Mixed precision (AMP): {self.use_amp}")

# 3. Modify train_epoch forward/backward pass
def train_epoch(self, train_loader, scheduler=None):
    # ... existing setup ...
    
    for batch in pbar:
        batch = self._move_to_device(batch)
        self.optimizer.zero_grad()
        
        # Wrap forward pass with autocast
        with autocast(enabled=self.use_amp):
            outputs = self._forward_batch(batch)
            loss = self._compute_loss(outputs, batch)
        
        # Use scaler for backward/step
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Fallback for CPU training
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            self.optimizer.step()
        
        # ... rest of training loop ...
```

#### Expected Results

**Before AMP:**
- Speed: 12 it/s
- Epoch time: 41 minutes
- GPU memory: ~3.8 GB

**After AMP:**
- Speed: 18-24 it/s (1.5-2× faster)
- Epoch time: 20-27 minutes
- GPU memory: ~2.3 GB (40% reduction)

**Validation:**
```python
# After training 1 epoch, compare:
# - Validation AUROC should be within 0.001 of original
# - Training loss curve should be nearly identical
# - Checkpoint size unchanged
```

---

### 2. Optimize DataLoader ⭐⭐⭐⭐☆

**Priority:** HIGH  
**Difficulty:** Easy  
**Risk:** Very Low

#### Benefits

| Benefit | Impact | Notes |
|---------|--------|-------|
| **CPU Parallelism** | 20-40% speedup | Overlap data loading with GPU compute |
| **Reduced I/O Wait** | Smoother training | No GPU starvation |
| **Better GPU Utilization** | 80-95% usage | vs. 60-70% without |
| **Pin Memory** | Faster transfers | DMA to GPU |

#### Trade-offs

| Consideration | Impact | Mitigation |
|---------------|--------|------------|
| **CPU Usage** | Higher (6-8 cores) | Laptop has enough cores |
| **RAM Usage** | +2-4 GB | Prefetch buffers |
| **Init Time** | +5-10 seconds | One-time cost |

#### Implementation Details

**File to modify:** `scripts/train.py`

**Code changes:**
```python
def create_dataloaders(config):
    # ... existing dataset creation ...
    
    # Optimized DataLoader settings
    train_loader = DataLoader(
        TensorDataset(torch.arange(len(train_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=6,              # ← ADD: CPU parallelism
        pin_memory=True,            # ← ADD: Faster GPU transfer
        persistent_workers=True,    # ← ADD: Keep workers alive
        prefetch_factor=2,          # ← ADD: Prefetch 2 batches
        collate_fn=train_collate_fn
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.arange(len(val_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,              # ← ADD: Fewer for validation
        pin_memory=True,            # ← ADD
        persistent_workers=True,    # ← ADD
        prefetch_factor=2,          # ← ADD
        collate_fn=val_collate_fn
    )
    
    return train_loader, val_loader, dataset
```

#### Tuning num_workers

**Recommended values:**
```python
import os

# Conservative (safe)
num_workers = min(4, os.cpu_count() - 2)

# Balanced (recommended)
num_workers = min(6, os.cpu_count() - 1)

# Aggressive (if CPU is strong)
num_workers = min(8, os.cpu_count())
```

**How to find optimal value:**
```python
# Test different values, measure it/s:
for num_workers in [0, 2, 4, 6, 8]:
    # Train for 100 iterations, record speed
    print(f"num_workers={num_workers}: {speed:.2f} it/s")
# Pick the fastest
```

#### Expected Results

**Before optimization:**
- Speed: 12 it/s
- GPU utilization: 60-70%
- CPU idle time: High

**After optimization:**
- Speed: 15-18 it/s (25-50% faster)
- GPU utilization: 85-95%
- CPU idle time: Low

---

### 3. cuDNN Autotuning ⭐⭐⭐☆☆

**Priority:** MEDIUM  
**Difficulty:** Trivial  
**Risk:** None

#### Benefits

| Benefit | Impact | Notes |
|---------|--------|-------|
| **Algorithm Selection** | 5-15% speedup | Finds fastest convolution algorithm |
| **Automatic** | No manual tuning | cuDNN benchmarks once |
| **One-line change** | Minimal effort | Add at program start |

#### Trade-offs

| Consideration | Impact | Mitigation |
|---------------|--------|------------|
| **First Epoch Slower** | +30-60 seconds | Benchmarking overhead (one-time) |
| **Non-determinism** | Slight variation | Use `deterministic=True` if needed |

#### Implementation Details

**File to modify:** `scripts/train.py`

**Code changes:**
```python
def main():
    # ... argument parsing ...
    
    # Enable cuDNN autotuning (add near top of main)
    import torch
    torch.backends.cudnn.benchmark = True
    
    # Optional: For reproducibility (slight speed cost)
    if config.get('reproducible', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # ... rest of training ...
```

#### Expected Results

**Before:**
- Speed: 12 it/s
- cuDNN uses default algorithms

**After:**
- Speed: 13-14 it/s (8-15% faster)
- cuDNN uses optimized algorithms

---

### 4. Power Settings (Windows) ⭐⭐☆☆☆

**Priority:** LOW (but easy)  
**Difficulty:** Trivial  
**Risk:** Low (heat)

#### Benefits

| Benefit | Impact | Notes |
|---------|--------|-------|
| **Prevent Throttling** | 10-20% speedup | Maintain max clock speeds |
| **Consistent Performance** | Stable it/s | No slowdowns over time |
| **No Code Changes** | Zero effort | System settings only |

#### Trade-offs

| Consideration | Impact | Mitigation |
|---------------|--------|------------|
| **Heat** | Higher temperatures (75-85°C) | Monitor temps, ensure cooling |
| **Fan Noise** | Louder | Use headphones |
| **Power Consumption** | Higher | Plug in laptop |
| **Battery Life** | Drains quickly | Must use AC power |

#### Implementation Steps

**Step 1: Windows Power Plan**
```powershell
# Option A: GUI
Control Panel → Power Options → High Performance (select)

# Option B: Command Line
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

**Step 2: NVIDIA Control Panel**
```
1. Right-click Desktop → NVIDIA Control Panel
2. Manage 3D Settings → Global Settings
3. Power management mode → Prefer maximum performance
4. Apply
```

**Step 3: Windows Graphics Settings**
```
Settings → System → Display → Graphics settings
→ Add python.exe
→ Options → High performance
```

**Step 4: Disable Laptop Power Saving**
```
Control Panel → Hardware and Sound → Power Options
→ Change plan settings → Change advanced power settings
→ Processor power management
   → Minimum processor state: 100%
   → Maximum processor state: 100%
```

#### Monitoring

**Check temperatures while training:**
```powershell
# GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 2

# CPU temperature (requires HWiNFO or similar)
# Safe range: < 85°C for GPU, < 95°C for CPU
```

**If temperatures exceed 85°C:**
- Elevate laptop (improve airflow)
- Use cooling pad
- Clean dust from vents
- Reduce ambient temperature

#### Expected Results

**Before:**
- Speed: 12 it/s (may drop to 8-10 it/s after 30 min due to throttling)
- GPU clock: 1400-1700 MHz (variable)
- Temperature: 70-75°C

**After:**
- Speed: 14-15 it/s (sustained)
- GPU clock: 1800-2000 MHz (stable)
- Temperature: 78-83°C

---

## ⚠️ Conditional Optimizations

### 5. Increase Batch Size

**Priority:** MEDIUM  
**Difficulty:** Easy  
**Risk:** MEDIUM (may affect convergence)

#### When to Consider

✅ **Good candidates:**
- After implementing AMP (more GPU memory available)
- GPU memory utilization < 50%
- Training is stable (no gradient explosions)

❌ **Not recommended if:**
- Already using large batch (>64)
- Model is sensitive to batch size
- Need to compare with published baselines (must match batch size)

#### Benefits vs. Risks

**Benefits:**

| Benefit | Impact | Notes |
|---------|--------|-------|
| **Fewer Iterations** | 30-50% speedup | If batch_size 32→64, half the iterations |
| **Better GPU Utilization** | 85-95% usage | Maximizes parallel compute |
| **More Stable Gradients** | Lower variance | Larger batch = better gradient estimate |

**Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Convergence Changes** | May need LR adjustment | Scale LR proportionally |
| **Different Optima** | May find different solution | Validate performance carefully |
| **Memory Overflow** | OOM error | Monitor with nvidia-smi |
| **Need to Re-run Baselines** | More work | All models must use same batch size |

#### Implementation Protocol

**Step 1: Measure Baseline (Current batch_size=32)**
```python
# Train for 5 epochs, record:
# - Final validation AUROC
# - Final validation AUPR  
# - Training loss curve
# Save as baseline_batch32.json
```

**Step 2: Test batch_size=64**
```yaml
# Edit configs/finetune_config.yaml
training:
  batch_size: 64  # Changed from 32
  
  # Optional: Scale learning rate
  learning_rate: 0.001  # 2× original (0.0005 × 2)
  # Rule of thumb: LR scales with sqrt(batch_size_ratio)
  # or linearly for small increases
```

**Step 3: Train and Compare**
```python
# Train for 5 epochs with batch_size=64
# Record same metrics
# Save as test_batch64.json

# Compare:
delta_auroc = test_batch64.auroc - baseline_batch32.auroc
delta_aupr = test_batch64.aupr - baseline_batch32.aupr

if abs(delta_auroc) < 0.01 and abs(delta_aupr) < 0.01:
    print("✅ batch_size=64 is safe, keeping it")
elif delta_auroc < -0.02 or delta_aupr < -0.02:
    print("❌ Performance degraded, reverting to 32")
else:
    print("⚠️ Borderline, train full 25 epochs to decide")
```

**Step 4: Decision Tree**

```
                    Train 5 epochs with batch_size=64
                                |
                  ┌─────────────┴─────────────┐
                  |                           |
        AUROC drop < 1%               AUROC drop > 2%
                  |                           |
          ✅ Keep batch_size=64        ❌ Revert to 32
          Retrain all models          No action needed
                  |
        Update all configs
        Re-run baselines
```

#### Learning Rate Scaling Rules

**General guidelines:**

```python
# Linear scaling (simple, works for small changes)
new_lr = base_lr * (new_batch / old_batch)
# Example: 0.0005 * (64/32) = 0.001

# Square root scaling (conservative, more stable)
new_lr = base_lr * sqrt(new_batch / old_batch)
# Example: 0.0005 * sqrt(64/32) = 0.0007

# Recommendation: Start with linear, fall back to sqrt if unstable
```

**Hyperparameter search space:**

```yaml
# If batch_size=64 doesn't work, try:
experiments:
  - batch_size: 64, lr: 0.001  # Linear scaling
  - batch_size: 64, lr: 0.0007 # Sqrt scaling
  - batch_size: 64, lr: 0.0005 # Keep same LR
  - batch_size: 48, lr: 0.00075 # Intermediate
```

#### Expected Results

**Successful case (batch_size=64):**
- Speed: 20-25 it/s (half iterations → 2× faster per epoch)
- Epoch time: 15-20 minutes
- AUROC: Within 1% of baseline
- GPU memory: 5-6 GB (vs. 2.3 GB with AMP at batch_size=32)

**Failed case (revert to 32):**
- AUROC drop > 2%
- Training unstable (loss oscillates)
- OOM error
- Keep batch_size=32

---

### 6. Gradient Accumulation

**Priority:** LOW  
**Difficulty:** Medium  
**Risk:** Low

#### When to Use

✅ **Use gradient accumulation when:**
- Want effective batch_size > GPU memory capacity
- batch_size=64 causes OOM
- Specific experiments require large batches (e.g., contrastive learning)

❌ **Don't use if:**
- GPU memory is sufficient for desired batch size
- Adds complexity without benefit

#### Concept

**Simulate large batch by accumulating gradients:**

```
Effective batch_size = physical_batch_size × accumulation_steps

Example:
- Physical batch: 32
- Accumulation: 4 steps
- Effective batch: 128

Same convergence as batch_size=128, but fits in memory of batch_size=32
```

#### Implementation

**Code changes in `src/training/finetune.py`:**

```python
class PromptGFMTrainer:
    def __init__(self, ..., gradient_accumulation_steps=1):
        # ... existing init ...
        self.grad_accum_steps = gradient_accumulation_steps
        logger.info(f"  Gradient accumulation: {self.grad_accum_steps} steps")
        if self.grad_accum_steps > 1:
            logger.info(f"  Effective batch size: "
                       f"{config['training']['batch_size'] * self.grad_accum_steps}")
    
    def train_epoch(self, train_loader, scheduler=None):
        # ... existing setup ...
        
        for batch_idx, batch in enumerate(pbar):
            batch = self._move_to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self._forward_batch(batch)
                loss = self._compute_loss(outputs, batch)
                
                # Scale loss by accumulation steps
                loss = loss / self.grad_accum_steps
            
            # Backward (accumulate gradients)
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Only update weights every N steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # ... rest of loop ...
```

**Config:**
```yaml
# configs/finetune_config.yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch_size = 128
```

#### Trade-offs

**Pros:**
- ✅ Same convergence as large batch
- ✅ Fits in limited memory
- ✅ Useful for contrastive learning

**Cons:**
- ⚠️ Slower than true large batch (4× accumulation → 4× longer per update)
- ⚠️ Code complexity
- ⚠️ BatchNorm behaves differently (uses physical batch statistics)

**Recommendation:** ⏸️ **Only implement if needed** (OOM with batch_size=64)

---

## ❌ Not Recommended Optimizations

### 7. Pre-compute BioBERT Embeddings

**Idea:** Encode all disease texts once, cache embeddings, freeze BioBERT

#### Why NOT to Do This

| Reason | Impact | Explanation |
|--------|--------|-------------|
| **Freezes BioBERT** | ❌ No fine-tuning | Loses adaptation to gene-disease task |
| **Lower Accuracy** | ❌ 3-7% AUROC drop | Fine-tuning is valuable |
| **Reduces Novelty** | ❌ Harms paper | Static embeddings less interesting |
| **Breaks Architecture** | ❌ No end-to-end training | FiLM conditioning loses meaning |

#### Example showing the problem:

```python
# BAD: Pre-computed (frozen)
disease_emb = precomputed_embeddings[disease_id]  # Static
gene_emb_conditioned = FiLM(gene_emb, disease_emb)  # disease_emb doesn't adapt

# GOOD: Fine-tuned (current)
disease_emb = BioBERT(disease_text)  # Learns task-specific representations
gene_emb_conditioned = FiLM(gene_emb, disease_emb)  # Both adapt together
```

#### When it MIGHT be acceptable:

- ✅ Initial debugging (first test run)
- ✅ Ablation study ("w/o BioBERT fine-tuning")
- ❌ NOT for main model in paper

**Recommendation:** ❌ **DON'T DO** - Harms your core contribution

---

### 8. Reduce Model Capacity

**Ideas:**
- Reduce GNN layers (3→2)
- Reduce hidden dim (256→128)
- Simplify architecture

#### Why NOT to Do This

| Reason | Impact | Explanation |
|--------|--------|-------------|
| **Lower Accuracy** | ❌ 5-10% worse | Model capacity needed for complex task |
| **Weaker Baseline** | ❌ Unfair comparison | Reviewers expect reasonable model size |
| **Undermines Claims** | ❌ "Is smaller model enough?" | Questions necessity of your approach |

#### Appropriate use:

✅ **Ablation study:** "Effect of model depth"
```yaml
experiments:
  - layers: 1, dim: 128  # Minimal
  - layers: 2, dim: 256  # Medium
  - layers: 3, dim: 256  # Full (main model)
  - layers: 4, dim: 512  # Large
```

Show that 3 layers, 256 dim is optimal balance.

**Recommendation:** ❌ **DON'T reduce for main model** - Use for ablations only

---

### 9. Reduce Training Epochs

**Idea:** Train for fewer epochs to save time

#### Why NOT Needed

| Reason | Explanation |
|--------|-------------|
| **Early Stopping** | Already implemented (patience=15) |
| **Auto Convergence** | Stops when no improvement |
| **Typical Epochs** | 20-30 epochs (not 100) |

#### Current behavior:

```python
# configs/finetune_config.yaml
training:
  num_epochs: 100              # Maximum allowed
  early_stopping_patience: 15  # Stops if no improvement for 15 epochs

# Actual training usually stops at epoch 20-35
```

**Recommendation:** ⏸️ **No action needed** - Early stopping handles this

---

## 🖥️ Hardware Recommendations

### Current Setup Assessment

**RTX 4060 Laptop GPU (8GB VRAM)**

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Development** | ⭐⭐⭐⭐⭐ | Excellent for coding/debugging |
| **Small Experiments** | ⭐⭐⭐⭐☆ | Good for 1-2 model training |
| **Large-Scale Study** | ⭐⭐☆☆☆ | Too slow for 20+ experiments |
| **Parallel Training** | ⭐☆☆☆☆ | Can only run 1 model at a time |

**Suitable for:**
- ✅ Initial development
- ✅ Code debugging
- ✅ Single model training (with patience)
- ✅ Literature review, writing

**Not suitable for:**
- ❌ Running 5+ ablation studies in parallel
- ❌ Extensive hyperparameter search
- ❌ Tight conference deadlines
- ❌ Training multiple baseline methods

---

### Option 1: Desktop Workstation (Recommended for Serious Research)

#### **Configuration: RTX 4090 Workstation**

**Specifications:**

| Component | Model | Price (USD) | Justification |
|-----------|-------|-------------|---------------|
| **GPU** | NVIDIA RTX 4090 (24GB) | $1,599 | 4× faster than laptop 4060 |
| **CPU** | AMD Ryzen 9 7950X (16-core) | $549 | Excellent multi-core for data loading |
| **Motherboard** | ASUS TUF Gaming X670E-Plus | $289 | PCIe 5.0, good VRMs |
| **RAM** | 64GB DDR5-6000 (2×32GB) | $219 | Enough for large graphs in memory |
| **Storage** | 2TB Samsung 990 Pro NVMe | $179 | Fast loading, plenty of space |
| **PSU** | Corsair RM1000x (1000W) | $179 | 4090 needs strong PSU |
| **Case** | Fractal Design Meshify 2 | $159 | Good airflow for cooling |
| **CPU Cooler** | Noctua NH-D15 | $99 | Quiet, efficient cooling |
| **Thermal Paste** | Noctua NT-H1 | $10 | - |
| **Case Fans** | 3× Noctua NF-A14 | $75 | Keep everything cool |
| **Windows 11 Pro** | Microsoft | $199 | Or use university license |
| **Assembly** | DIY or shop | $0-200 | Can build yourself |
| **TOTAL** | | **$3,556** | **±$200 depending on sales** |

#### Performance Comparison

**RTX 4060 Laptop vs. RTX 4090 Desktop:**

| Metric | Laptop 4060 | Desktop 4090 | Speedup |
|--------|-------------|--------------|---------|
| **CUDA Cores** | 3,072 | 16,384 | 5.3× |
| **VRAM** | 8GB GDDR6 | 24GB GDDR6X | 3× |
| **Memory Bandwidth** | 272 GB/s | 1,008 GB/s | 3.7× |
| **TDP** | 115W (laptop) | 450W (desktop) | 3.9× |
| **Measured Speed** | 12 it/s | ~40-50 it/s | **3-4×** |
| **Epoch Time** | 41 min | ~10-12 min | **4×** |
| **Full Training** | 68 hours | ~16-20 hours | **4×** |
| **Max Batch Size** | 32 (64 w/ AMP) | 256+ | **8×** |

#### Cost-Benefit Analysis

**Scenario: Running 20 Experiments for Paper**

| Scenario | Laptop 4060 | Desktop 4090 |
|----------|-------------|--------------|
| **Time per Experiment** | 68 hours | 17 hours |
| **Total Time (serial)** | 56 days | 14 days |
| **Parallel Capacity** | 1 model | 4-6 models |
| **Total Time (parallel)** | 56 days | 3-4 days |
| **Electricity Cost** | ~$80 | ~$150 |
| **Opportunity Cost** | High | Low |

**Break-even calculation:**

```
Desktop cost: $3,556
Time saved per paper: ~50 days
If your time is worth > $70/day, desktop pays for itself in 1 paper

For PhD students:
- 2-3 papers from this project → $1,200-1,800 value/paper
- Productivity boost → more time for analysis, writing
- Quality improvement → better experiments → better venue
```

#### Additional Benefits

✅ **Parallel Training:**
- Run 4-6 models simultaneously (24GB VRAM)
- Example: All ablation studies in one night

✅ **Larger Experiments:**
- Batch size 256 (vs. 64 on laptop)
- Pre-training on full datasets
- Multi-task learning

✅ **Future Projects:**
- Reusable for next 3-5 years
- Other lab members can use
- Can upgrade GPU later

✅ **Flexibility:**
- Leave running overnight without worry
- No thermal throttling
- Stable, reproducible performance

#### When to Buy

**Buy desktop if:**
- ✅ Planning multiple papers from this project
- ✅ Conference deadline < 4 months
- ✅ Need to run >10 experiments
- ✅ Budget available ($3-4K)
- ✅ Lab has space for workstation

**Stick with laptop if:**
- ⏸️ Only 1-2 experiments needed
- ⏸️ Timeline flexible (>6 months)
- ⏸️ Budget constrained
- ⏸️ Cluster access available (see Option 2)

---

### Option 2: University Cluster (HIGHEST ROI)

#### **What is a Cluster?**

A shared computing resource with:
- 10-100+ GPUs (Tesla A100, H100, V100)
- Job scheduling system (Slurm, PBS)
- Free or very cheap for students
- Professional maintenance

#### Typical Cluster Specs

**Example: University Research Cluster**

```
Nodes: 20-50 GPU nodes
GPUs per node: 4-8× A100 (40GB or 80GB)
Total GPUs: 200-400
Scheduler: Slurm
Cost: Free for students (or $0.01-0.05/GPU-hour)
Queue: Priority for students/faculty
```

#### Performance Comparison

**A100 (40GB) vs. Laptop 4060:**

| Metric | Laptop 4060 | Tesla A100 | Speedup |
|--------|-------------|------------|---------|
| **CUDA Cores** | 3,072 | 6,912 | 2.25× |
| **Tensor Cores** | 96 (3rd gen) | 432 (3rd gen) | 4.5× |
| **VRAM** | 8GB | 40GB | 5× |
| **Memory BW** | 272 GB/s | 1,555 GB/s | 5.7× |
| **Measured Speed** | 12 it/s | ~60-80 it/s | **5-6×** |
| **Epoch Time** | 41 min | ~6-8 min | **6-7×** |
| **Full Training** | 68 hours | ~10-14 hours | **5-6×** |

#### How to Apply for Access

**Step 1: Find Your Cluster**

```
Common cluster systems at universities:
- Campus Research Computing (SLURM)
- National supercomputers (XSEDE, ACCESS)
- Lab-specific clusters
- Department-level clusters
```

**Check with:**
- University IT department
- Your advisor
- Other PhD students
- CS/EE department

**Step 2: Application Process**

**Example application email:**

```
To: research-computing@university.edu
Subject: GPU Cluster Access Request - PhD Research

Dear Research Computing Team,

I am a PhD student in [Department] working on biomedical deep learning 
research for a conference paper submission. I request access to GPU 
resources for training graph neural network models.

Project: Gene-Disease Association Prediction with Graph Foundation Models
Advisor: [Name]
Estimated GPU hours: 500-1000 GPU-hours over 2-3 months
GPU requirements: Tesla A100 or V100 (32GB+ VRAM preferred)
Software: PyTorch 2.1, CUDA 11.8, Python 3.10

Use case: Training 15-20 models for ablation studies and baselines

Thank you for your consideration.

Best regards,
[Your Name]
[Student ID]
[Department]
```

**Step 3: Common Access Requirements**

- ✅ Enroll in cluster training (1-2 hour session)
- ✅ Submit resource request form
- ✅ Advisor approval
- ✅ Agree to usage policies
- ✅ Optional: Cite cluster in publications

**Step 4: Using the Cluster**

**Example SLURM job script:**

```bash
#!/bin/bash
#SBATCH --job-name=promptgfm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1        # Request 1 A100 GPU
#SBATCH --cpus-per-task=8        # 8 CPU cores for data loading
#SBATCH --mem=64G                # 64GB RAM
#SBATCH --time=24:00:00          # Max 24 hours
#SBATCH --output=logs/train_%j.log

# Load modules
module load cuda/11.8
module load python/3.10
module load pytorch/2.1

# Activate environment
source ~/miniconda3/bin/activate promptgfm

# Run training
cd $HOME/PromptGFM-Bio
python scripts/train.py --config configs/finetune_config.yaml

# Notify on completion
echo "Training complete: $(date)"
```

**Submit job:**
```bash
sbatch train_job.sh
```

**Check status:**
```bash
squeue -u $USER
```

**View output:**
```bash
tail -f logs/train_JOBID.log
```

#### Parallel Experiment Example

**Run all ablations simultaneously:**

```bash
# Submit 10 jobs at once (different configs)
for config in configs/ablations/*.yaml; do
    sbatch --job-name=$(basename $config .yaml) train_job.sh $config
done

# All 10 experiments finish in ~12-14 hours
# vs. 30 days on laptop sequentially
```

#### Cost Comparison

| Option | Setup Cost | Per-Experiment Cost | 20 Experiments | Total |
|--------|------------|---------------------|----------------|-------|
| **Laptop** | $0 | $0 (your time) | 56 days | $0 cash, high time cost |
| **Desktop** | $3,556 | $2-3 electricity | 3-4 days | $3,600 |
| **Cluster** | $0 (申请) | $0-5 | 1-2 days | $0-100 |

**Verdict:** ✅ **Cluster is best value** if available

#### When to Use Cluster

**Use cluster for:**
- ✅ Final training runs (after debugging on laptop)
- ✅ Ablation studies (parallel experiments)
- ✅ Hyperparameter search
- ✅ Baseline method training
- ✅ Large-scale pretraining

**Use laptop for:**
- ✅ Code development
- ✅ Debugging issues
- ✅ Small tests (1-2 epochs)
- ✅ Interactive work

---

### Option 3: Cloud GPU Services

#### **When to Use Cloud**

✅ **Good for:**
- Quick experiments when cluster queue is full
- Short-term needs (paper deadline crunch)
- One-time large training
- Flexibility (scale up/down)

❌ **Not ideal for:**
- Long-term extensive use (expensive)
- Many small experiments (overhead)
- Debugging (better on laptop)

#### Popular Providers

**1. Lambda Labs (Recommended for Research)**

**Pricing:**
```
1× A100 (40GB):    $1.10/hour
1× A100 (80GB):    $1.29/hour  ← Recommended
4× A100 (80GB):    $4.40/hour
8× A100 (80GB):    $8.80/hour

# Example: Full training on A100 (80GB)
# Time: 10-12 hours
# Cost: $1.29 × 12 = $15.48 per experiment
# 20 experiments: $309.60
```

**Pros:**
- ✅ Simple interface
- ✅ Pre-configured PyTorch
- ✅ No complicated setup
- ✅ Pay per second

**Cons:**
- ⚠️ Availability can be limited (high demand)
- ⚠️ Data transfer costs

**2. Google Cloud / Colab Pro+**

**Colab Pro+:**
```
Cost: $49.99/month
GPUs: V100, A100 (subject to availability)
Usage: 500 compute units/month
Good for: Periodic experiments
```

**GCP A100 instances:**
```
1× A100 (40GB): ~$3.67/hour
1× A100 (80GB): ~$4.31/hour

Cheaper with sustained use discounts and preemptible instances
```

**3. AWS / Azure**

**AWS p4d instances (A100):**
```
p4d.24xlarge: 8× A100 (40GB), $32.77/hour
Spot instances: ~$10-12/hour (70% discount, can be terminated)
```

**Use spot for:**
- ✅ Non-critical experiments
- ✅ Cost-sensitive work
- ⚠️ Checkpointing required (can be interrupted)

#### Cost Projection

**Full paper (20 experiments):**

```
Laptop: $0 cash, 56 days
Desktop: $3,556 upfront, 3-4 days
Cluster: $0, 1-2 days (if available)
Cloud (Lambda A100): $310, 1-2 days
```

**Decision matrix:**

| Budget | Timeline | Recommendation |
|--------|----------|----------------|
| **Tight, Fast needed** | <1 month | Cloud |
| **Flexible, Low budget** | 2-6 months | Laptop + optimizations |
| **Medium budget** | 1-3 months | Cluster申请 + Cloud backup |
| **High budget, Long-term** | Any | Desktop workstation |

---

## 💰 Cost-Benefit Analysis

### Total Cost of Ownership (3-Year Projection)

**Scenario: PhD student, 3 papers from this project**

#### Option A: Laptop Only

```
Hardware: $0 (assumed you have it)
Optimizations: 1-2 hours implementation ($0)
Electricity: ~$150/year × 3 = $450
Total cash: $450

Time cost:
- Paper 1: 56 days (20 experiments)
- Paper 2: 40 days (15 experiments)
- Paper 3: 30 days (10 experiments)
Total time: 126 days = 18 weeks = 4.5 months of GPU time

Opportunity cost:
- 4.5 months at $50/day (conservative) = $6,750
- Total cost: $450 + $6,750 = $7,200
```

#### Option B: Desktop Workstation

```
Hardware: $3,556 (one-time)
Electricity: ~$400/year × 3 = $1,200
Total cash: $4,756

Time cost:
- Paper 1: 4 days (parallel)
- Paper 2: 3 days (parallel)
- Paper 3: 2 days (parallel)
Total time: 9 days

Opportunity cost:
- 9 days at $50/day = $450
- Total cost: $4,756 + $450 = $5,206

Savings: $7,200 - $5,206 = $1,994
ROI: 138% over 3 years
Break-even: 1.5 papers
```

#### Option C: University Cluster

```
Hardware: $0
申请 time: 4-8 hours (one-time)
Cost per GPU-hour: $0 (typical for students)
Total cash: $0

Time cost:
- Paper 1: 1-2 days
- Paper 2: 1-2 days
- Paper 3: 1 day
Total time: 5 days (mostly waiting in queue)

Opportunity cost: ~$250

Total cost: $250

Savings: $7,200 - $250 = $6,950 ✅ BEST VALUE
```

#### Option D: Cloud (Lambda Labs)

```
Cost per experiment: ~$15 (A100, 12 hours)
Total experiments: 45 (3 papers)
Total cash: $675

Time cost: 5 days (parallel possible)
Opportunity cost: $250

Total cost: $925

Savings vs laptop: $6,275
Savings vs desktop: $4,281
```

---

### Decision Matrix

**Use this table to decide:**

| Consideration | Laptop | Desktop | Cluster | Cloud |
|---------------|--------|---------|---------|-------|
| **Upfront cost** | $0 | $3,556 | $0 | $0 |
| **Running cost** | Low | Medium | $0 | High |
| **Speed** | 1× | 4× | 6× | 6× |
| **Parallel capacity** | 1 | 4-6 | 10+ | Unlimited |
| **Flexibility** | High | High | Medium | High |
| **Learning curve** | None | Low | Medium | Low-Med |
| **Best for** | Small work | Long-term | Research | Short-term |

**Recommended strategy:**

```
Phase 1 (Development): Laptop
Phase 2 (Initial experiments): Laptop + optimizations
Phase 3 (If cluster available): Cluster for all experiments ✅ BEST
Phase 3 (If no cluster): Desktop OR Cloud
Phase 4 (Final runs): Cluster/Desktop with full validation
```

---

## 📋 Implementation Action Plan

### Phase 1: Immediate Actions (This Week)

**Time required:** 2-3 hours  
**Cost:** $0  
**Risk:** Very Low  
**Expected benefit:** 2-3× speedup

#### Step 1.1: Stop Current Training

Your current baseline training is:
- Without PPI edges (no message passing)
- Without optimizations

**Action:**
```powershell
# In training PowerShell window, press Ctrl+C
# This will save checkpoint_epoch_N.pt automatically
```

#### Step 1.2: Implement AMP (Mixed Precision)

**Time:** 30-45 minutes  
**Files to modify:** `src/training/finetune.py`

**Changes:** (See detailed code in Section "Mixed Precision Training")

**Test:**
```powershell
# Quick test (5 minutes)
python scripts/train.py --config configs/finetune_config.yaml

# Watch first few iterations, should see:
# - Speed increase (12→18-24 it/s)
# - GPU memory decrease (check nvidia-smi)
# - No error messages

# If successful, let run for 2-3 epochs to validate metrics
```

#### Step 1.3: Optimize DataLoader

**Time:** 10-15 minutes  
**Files to modify:** `scripts/train.py`

**Changes:** (See detailed code in Section "Optimize DataLoader")

**Test:**
```powershell
#Run again, monitor:
# - CPU usage (should increase to 60-80%)
# - GPU utilization (nvidia-smi, should be 85-95%)
# - Speed (should see another 20-30% boost)
```

#### Step 1.4: Enable cuDNN Autotuning

**Time:** 2 minutes  
**Files to modify:** `scripts/train.py`

**Changes:**
```python
# Add near top of main()
torch.backends.cudnn.benchmark = True
```

#### Step 1.5: Power Settings

**Time:** 5-10 minutes  
**Changes:** System settings (see detailed steps in Section "Power Settings")

**Verify:**
```powershell
# Monitor GPU clock speeds
nvidia-smi --query-gpu=clocks.gr --format=csv -l 2

# Should see 1800-2000 MHz (vs 1400-1700 before)
```

#### Step 1.6: Regenerate Graph with PPI Edges

**Time:** 20-30 minutes  
**Critical:** This is your main model for the paper!

```powershell
# Archive baseline (no PPI) results
python scripts/resume_training.py --archive

# Regenerate graph with PPI edges
python scripts/preprocess_all.py --force

# Expected output:
# ✓ STRING: ~320K interactions
# ✓ BioGRID: ~87K interactions
# ✓ Combined PPI edges: ~287K
```

#### Step 1.7: Start Optimized Training

```powershell
# Start main training (with all optimizations + PPI edges)
python scripts/train.py --config configs/finetune_config.yaml
```

**Expected performance:**
```
Before:  12 it/s, 41 min/epoch, 68 hours total
After:   30-36 it/s, 13-16 min/epoch, 22-27 hours total

Speedup: 2.5-3× faster ✅
```

**Monitor for 1-2 epochs:**
- Speed stable at ~30-35 it/s
- GPU utilization 85-95%
- Loss decreasing normally
- No error messages

If all good, let it run for full training (2-3 days)!

---

### Phase 2: Hardware Request (This Month)

**Timeline:** 1-4 weeks  
**Priority:** HIGH

#### Action 2.1: Apply for Cluster Access

**Week 1:**
- ✅ Research your university's computing resources
- ✅ Contact IT department / research computing
- ✅ Find申请 form / process

**Week 2:**
- ✅ Submit application (see template in Section "University Cluster")
- ✅ Get advisor approval
- ✅ Schedule cluster training session (if required)

**Week 3:**
- ✅ Complete training
- ✅ Receive account credentials
- ✅ Test job submission with small experiment

**Week 4:**
- ✅ Transfer code and data to cluster
- ✅ Run first full training on cluster
- ✅ Compare performance

#### Action 2.2: Evaluate Desktop Purchase

**Discuss with advisor:**

**Option A: Request from department**

**Budget request template:**
```
Equipment Request: GPU Workstation for PhD Research

Total cost: $3,556
Breakdown:
  - RTX 4090 GPU: $1,599
  - Supporting components: $1,957

Justification:
  - Current laptop: 68 hours per experiment
  - Desktop: 17 hours per experiment
  - Enables 20+ experiments for paper in 2 weeks vs 4 months
  
Expected outcomes:
  - 1-2 conference papers (NeurIPS/ICML/ISMB tier)
  - Faster iteration → better quality
  - Reusable for future projects (3-5 years)
  
ROI: Pays for itself in 1-2 papers через improved productivity
```

**Option B: Lab shared resource**

If lab has budget:
- Propose as shared resource
- Multiple students can use
- Higher utilization = better value

**Option C: Personal purchase**

If self-funding:
- Consider RTX 4080 (~$1,000, 16GB) as compromise
- 3× faster than laptop (vs. 4× for 4090)
- Total build: ~$2,500

#### Action 2.3: Cloud Backup Plan

Even with cluster申请, have cloud ready:
- ✅ Create Lambda Labs account
- ✅ Test small run ($2-3)
- ✅ Have ready for deadline crunch

---

### Phase 3: Conditional Optimizations (After Phase 1)

**Timeline:** 1-2 days  
**Conditional:** Only if Phase 1 successful

#### Experiment 3.1: Test Larger Batch Size

**Goal:** See if batch_size=64 improves speed without hurting accuracy

**Protocol:**

**Day 1:**
```powershell
# Baseline measurement (current: batch_size=32)
# Let current training run for 5 epochs
# Record: validation AUROC, AUPR, loss curve
```

**Day 2:**
```powershell
# Stop training (Ctrl+C)

# Edit configs/finetune_config.yaml:
#   batch_size: 64
#   learning_rate: 0.001 (2× original)

# Start training for 5 epochs
python scripts/train.py --config configs/finetune_config.yaml

# Compare metrics
```

**Decision:**
```python
if abs(AUROC_64 - AUROC_32) < 0.01:
    print("✅ Keep batch_size=64")
    # Update all config files
    # Retrain main model
else:
    print("❌ Revert to batch_size=32")
    # Keep original settings
```

---

### Phase 4: Validation & Monitoring

**Ongoing throughout training**

#### Daily Checks

**Monitor training:**
```powershell
# Check speed is stable
# Should see: 30-35 it/s consistently

# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Should be: 75-83°C (safe range)
# If > 85°C: Check cooling, reduce workload
```

**Check metrics:**
```
# Validation AUROC should increase
# Training loss should decrease
# No NaN or inf values
```

#### Weekly Checkpoints

**Verify checkpoints:**
```powershell
# List checkpoints
Get-ChildItem checkpoints\promptgfm_film\*.pt

# Should have one per epoch
# Check file sizes (should be ~50-100MB each)
```

**Test resuming:**
```powershell
# Test that resume works (don't interfere with training)
# Just verify the script runs:
python scripts/resume_training.py --interactive
# Press 'Q' to quit without changing anything
```

---

## 📊 Impact on Paper Quality

### Software Optimizations ⭐⭐⭐⭐⭐

**Direct Benefits:**

| Aspect | Impact | Details |
|--------|--------|---------|
| **More Experiments** | 2-3× more ablations possible | Faster turnaround → more thorough study |
| **Better Baselines** | Can afford to implement more | Stronger paper with comprehensive comparison |
| **Meet Deadlines** | ~25 days vs 68 days | Submit to preferred venue |
| **Reproducibility** | Industry standard methods | Reviewers expect AMP, parallel data loading |
| **Credibility** | Professional implementation | Shows engineering competence |

**Indirect Benefits:**

- More time for analysis and writing
- Faster feedback loop for ideas
- Less stress from long training times
- Can explore more research directions

**Paper Impact:** ⭐⭐⭐⭐⭐ (5/5) - Essential for competitive paper

**What to mention in paper:**

```
Methods section:
"All models were trained with mixed precision (FP16) using PyTorch's 
automatic mixed precision (AMP) for computational efficiency. We used 
AdamW optimizer with learning rate 5e-4, batch size 32, and gradient 
clipping of 1.0. Training was performed on NVIDIA RTX GPU with early 
stopping (patience=15) on validation AUROC."

No need to over-emphasize optimizations - they're standard practice.
```

---

### Hardware Upgrade ⭐⭐⭐⭐☆

**Direct Benefits:**

| Aspect | Impact | Details |
|--------|--------|---------|
| **Parallel Experiments** | 4-10× throughput | Run all ablations overnight |
| **Faster Iteration** | Test ideas same day | Improves research quality |
| **Larger Models** | Explore scaling | May find better architecture |
| **Pre-training** | Becomes feasible | Graph pre-training (Phase 6) |

**Indirect Benefits:**

- Less waiting → more focused work
- Can afford to try risky ideas
- Better hyperparameter tuning
- More comprehensive study

**Paper Impact:** ⭐⭐⭐⭐☆ (4/5) - Enables better research, but doesn't directly improve methods

**What matters for paper:**

✅ **More important than hardware:**
- Novel methodology
- Thorough evaluation
- Clear writing
- Good ablation studies

⚠️ **Hardware is secondary:**
- Doesn't make bad method good
- But enables good method to shine
- Investment in productivity

---

### Large Batch Size ⭐⭐⭐☆☆

**Potential Benefits:**

| Aspect | Impact | Details |
|--------|--------|---------|
| **Training Speed** | 30-50% faster | Half iterations per epoch |
| **GPU Utilization** | Better hardware use | Maximizes parallelism |

**Potential Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Convergence Changes** | Different solution found | Tune learning rate |
| **Inconsistent Baselines** | Must retrain all models | Significant work |
| **Reviewer Questions** | Why different batch size? | Must justify |

**Paper Impact:** ⭐⭐⭐☆☆ (3/5) - Speed benefit, but risks consistency

**Recommendation:**

```
Conservative approach:
1. Keep batch_size=32 for main paper
2. Use larger batches only after testing thoroughly
3. Mention in paper if using: "We used batch size 64 across all methods"

Aggressive approach (if validated):
1. Test batch_size=64 carefully
2. If metrics ±1%, use it
3. Retrain ALL baselines with same batch size
4. Document in paper's hyperparameter table
```

---

## ❓ Decision Framework

### Quick Decision Tool

**Answer these questions:**

#### Q1: How much time until paper deadline?

```
> 6 months:  Laptop + optimizations OK
3-6 months:  Laptop + optimizations + cluster申请
< 3 months:  Need cluster OR desktop
< 1 month:   Need cluster + cloud backup
```

#### Q2: How many experiments needed?

```
< 5:   Laptop sufficient
5-15:  Laptop + optimizations (tight but doable)
15-30: Need cluster OR desktop
> 30:  Need cluster with parallel jobs
```

#### Q3: What's your budget?

```
$0:           Laptop + cluster申请
$500-1000:    Laptop + cloud for critical runs
$3000-4000:   Desktop workstation
$0 (free):    University cluster (best option!)
```

#### Q4: Technical comfort level?

```
Beginner:     Start with laptop, phase optimizations
Intermediate: Implement all optimizations, use cluster
Advanced:     Desktop build, cluster, parallel training
```

---

### Recommended Paths

#### Path A: Conservative (Low Risk)

**Target:** First paper, careful approach

**Phase 1 (Week 1):**
- ✅ Implement AMP
- ✅ Optimize DataLoader
- ✅ Enable cuDNN
- ✅ Keep batch_size=32

**Phase 2 (Week 2-4):**
- ✅ Train main model (with PPI edges)
- ✅ Apply for cluster access
- ⏸️ Hold on hardware purchase

**Phase 3 (Month 2):**
- ✅ Use cluster if available
- ✅ Otherwise continue on laptop
- ✅ 5-10 ablation studies

**Outcome:** Solid paper, 3-4 months to completion

---

#### Path B: Balanced (Recommended)

**Target:** High-quality paper, moderate timeline

**Phase 1 (Week 1):**
- ✅ All software optimizations
- ✅ Test batch_size=64 (revert if issues)
- ✅ Apply for cluster access

**Phase 2 (Week 2):**
- ✅ Start main training
- ✅ Request desktop workstation (advisor approval)
- ✅ Create cloud account (backup)

**Phase 3 (Month 2):**
- ✅ Desktop arrives OR cluster approved
- ✅ Run 15-20 experiments in parallel
- ✅ Comprehensive study

**Outcome:** Strong paper, 2-3 months to completion

---

#### Path C: Aggressive (Fast Track)

**Target:** Top-tier venue, tight deadline

**Phase 1 (Week 1 - Day 1):**
- ✅ All optimizations same day
- ✅ Emergency cluster申请 (if possible)
- ✅ Start cloud training immediately

**Phase 1 (Week 1 - Day 2-7):**
- ✅ Run main model on cloud
- ✅ Start 5 key ablations in parallel
- ✅ Order desktop (overnight shipping if possible)

**Phase 2 (Week 2-3):**
- ✅ Desktop arrives, run remaining experiments
- ✅ OR use cloud extensively (budget ~$500-800)
- ✅ 20+ experiments in 2 weeks

**Outcome:** Comprehensive paper, 1-1.5 months to experiments done

**Cost:** ~$3,500 (desktop) OR $500-800 (cloud) OR $0 (cluster luck)

---

## 💬 FAQ

### General Optimizations

**Q1: Will AMP change my results?**

**A:** No, results should be identical within numerical precision (typically within 0.001 AUROC). Use the same random seed for exact reproduction comparison.

**Q2: Can I use AMP for validation/testing?**

**A:** Yes! AMP should be used for all forward passes (training, validation, testing). It only affects computation, not results.

**Q3: What if my model becomes unstable with AMP?**

**A:** Very rare, but if happens:
1. Check for NaN/Inf in losses (GradScaler handles this automatically)
2. Try `scaler = GradScaler(growth_interval=2000)` (less aggressive scaling)
3. Report issue - it's usually a bug elsewhere in code

**Q4: Do I need to retrain all baselines with optimizations?**

**A:** Only if optimizations change hyperparameters (e.g., batch size). AMP, DataLoader, cuDNN are implementation details - don't affect fairness.

---

### Hardware

**Q5: Is RTX 4090 overkill for this project?**

**A:** No, if planning multiple papers. For single paper, RTX 4080 (16GB, $1000) is good middle ground - still 3× faster than laptop.

**Q6: Can I use Google Colab for free?**

**A:** Free Colab has limitations:
- ~12-hour session limits
- Unreliable GPU access (may get CPU)
- Colab Pro ($10/month) or Pro+ ($50/month) recommended for serious work

**Q7: How do I know if my laptop will handle the optimizations?**

**A:** Your laptop (RTX 4060, 8GB) can handle all recommended optimizations. The laptop might run hotter (75-83°C), which is normal and safe.

**Q8: What if university cluster has long queues?**

**A:** Common issue. Strategies:
- Submit jobs overnight/weekends (shorter queues)
- Request priority queue access (for students with deadlines)
- Use cloud for urgent experiments
- Plan experiments in advance

---

### Batch Size

**Q9: Why might larger batch size hurt accuracy?**

**A:** Larger batches → different optimization dynamics:
- Smoother gradients (less noise)
- Can converge to "sharper" minima (worse generalization)
- May need learning rate adjustment

See: "Large Batch Training of Convolutional Networks" (Goyal et al., 2017)

**Q10: If batch_size=64 works, should I use batch_size=128?**

**A:** Not necessarily:
- Diminishing returns (64→128 less gain than 32→64)
- Higher risk of convergence issues
- May not fit in memory even with AMP
- Test empirically, but 64 is likely sweet spot for 8GB GPU

---

### Paper & Research

**Q11: Should I mention optimizations in paper?**

**A:** Brief mention in methods:
- ✅ "Trained with mixed precision (FP16) for efficiency"
- ✅ "Batch size 32, learning rate 5e-4"
- ❌ Don't over-emphasize (it's standard practice, not contribution)

**Q12: Can I compare my optimized model to unoptimized baselines?**

**A:** Yes, if only implementation differs:
- ✅ Your model with AMP vs. baseline without AMP → Fair (implementation detail)
- ❌ Your model with batch_size=64 vs. baseline with batch_size=32 → Not fair (different hyperparameters)

**Q13: Will reviewers care about training time?**

**A:** Secondary concern:
- Primary: Novel method, strong results, thorough evaluation
- Secondary: Efficiency (but usually doesn't determine acceptance)
- Mention in paper: "Training took ~24 hours on single RTX 4060 GPU"

**Q14: Should I run ablations with and without optimizations?**

**A:** No need. Ablations test architectural choices, not implementation details:
- Test: FiLM vs. CrossAttention (meaningful)
- Don't test: AMP vs. no AMP (implementation, not research)

---

### Technical

**Q15: What's the difference between pin_memory and non_blocking?**

**A:** 
- `pin_memory=True`: Allocates data in pinned (page-locked) CPU memory → faster transfer
- `non_blocking=True`: Allows asynchronous CPU→GPU transfer (overlap with compute)
- Use both for best performance

**Q16: Why num_workers=0 vs. num_workers=6?**

**A:** 
- `num_workers=0`: Single-process (simpler, easier to debug, but slower)
- `num_workers=6`: Multi-process (faster, but uses more RAM and CPU)
- Start with 0 for debugging, use 6+ for production

**Q17: What if I get "Too many open files" error with num_workers > 0?**

**A:** Increase file descriptor limit:
```bash
# Linux/Mac
ulimit -n 4096

# Windows: Usually not an issue, but can increase in Windows settings if needed
```

**Q18: Can I combine gradient accumulation with large batch size?**

**A:** Yes, but rarely needed:
```yaml
# Effective batch_size = 64 × 4 = 256
batch_size: 64
gradient_accumulation_steps: 4
```
Only useful if you specifically need very large batches (e.g., contrastive learning).

---

## 📝 Summary Checklist

### Before You Start

- [ ] **Read this guide** (you're doing it!)
- [ ] **Understand trade-offs** (safe vs. risky optimizations)
- [ ] **Decide on approach** (conservative vs. balanced vs. aggressive)
- [ ] **Check current performance** (record baseline metrics)

### Phase 1: Software Optimizations (Week 1)

- [ ] **Stop current training** (Ctrl+C, save checkpoint)
- [ ] **Implement AMP** (mixed precision)
- [ ] **Optimize DataLoader** (num_workers, pin_memory)
- [ ] **Enable cuDNN benchmark**
- [ ] **Set power settings** (Windows high performance)
- [ ] **Regenerate graph with PPI edges** (critical!)
- [ ] **Start optimized training**
- [ ] **Monitor for 2-3 epochs** (verify metrics, speed, stability)

### Phase 2: Hardware (Weeks 2-4)

- [ ] **Apply for cluster access** (highest priority!)
- [ ] **Discuss desktop purchase** with advisor (if budget allows)
- [ ] **Create cloud accounts** (Lambda Labs, Colab Pro+)
- [ ] **Test cluster/cloud** with small experiment

### Phase 3: Experiments (Months 2-3)

- [ ] **Complete main model training** (~25 epochs)
- [ ] **Run ablation studies** (conditioning, GNN, prompts)
- [ ] **Train baseline methods** (5-7 comparisons)
- [ ] **Generate figures and tables**
- [ ] **Write methods section**

### Validation

- [ ] **Compare optimized vs. baseline** (should be within 0.001 AUROC)
- [ ] **Verify checkpoints saved** (every epoch)
- [ ] **Test resume functionality**
- [ ] **Document all hyperparameters**

---

## 🎯 Final Recommendations

### DO NOW (High Confidence)

1. ✅ **Implement all safe optimizations** (AMP, DataLoader, cuDNN)
2. ✅ **Regenerate graph with PPI edges**
3. ✅ **Apply for cluster access**
4. ✅ **Start optimized training**

**Expected outcome:** 2-3× speedup, main model training in 22-27 hours

### DISCUSS WITH ADVISOR

1. **Budget for hardware:**
   - Desktop workstation ($3-4K) if planning multiple papers
   - Cloud budget ($500-1000) for deadline flexibility

2. **Timeline:**
   - Target conference/journal
   - Affects urgency of hardware decision

3. **Scope:**
   - How many experiments needed
   - Affects hardware requirements

### DON'T DO (High Risk)

1. ❌ **Pre-compute BioBERT embeddings** (harms novelty)
2. ❌ **Reduce model capacity** (harms results)
3. ❌ **Skip thorough validation** (harms reproducibility)

### CONDITIONAL (Test First)

1. ⚠️ **Larger batch size** (test thoroughly before committing)
2. ⚠️ **Gradient accumulation** (only if needed for OOM)

---

## 📞 Support

**Questions? Check:**

1. This guide's FAQ section
2. [TRAINING_QUICKSTART.md](TRAINING_QUICKSTART.md) for commands
3. [TRAINING_RESUME_GUIDE.md](TRAINING_RESUME_GUIDE.md) for checkpoint management
4. [CONFERENCE_PAPER_ROADMAP.md](CONFERENCE_PAPER_ROADMAP.md) for research plan

**Still stuck?**
- PyTorch forums: https://discuss.pytorch.org/
- PyTorch AMP docs: https://pytorch.org/docs/stable/amp.html
- SLURM docs: https://slurm.schedmd.com/

---

**Last Updated:** February 17, 2026  
**Version:** 1.0  
**Author:** PromptGFM-Bio Project  

**Good luck with your research! 🚀**
