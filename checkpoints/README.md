# Checkpoints Directory

This directory stores trained model checkpoints.

**⚠️ Important**: Model checkpoint files (.pt, .pth) are excluded from version control due to their large size (50-200MB each).

---

## Directory Structure

```
checkpoints/
├── promptgfm_film/           # FiLM conditioning model
│   ├── best_model.pt         # Best validation checkpoint
│   ├── best_model.json       # Training metadata
│   ├── checkpoint_epoch_X.pt # Periodic checkpoints
│   └── checkpoint_epoch_X.json
├── promptgfm_crossattn/      # Cross-attention model
└── pretrained/               # Self-supervised pretrained models
```

---

## Checkpoint Contents

Each checkpoint file contains:
- **Model state_dict**: Trained model weights
- **Optimizer state_dict**: Optimizer state (for resuming training)
- **Epoch number**: Training progress
- **Best validation metrics**: AUROC, AUPR, etc.
- **Configuration**: Model and training hyperparameters

---

## Using Checkpoints

### Loading a Trained Model

```python
import torch
from src.models.promptgfm import PromptGFM

# Load checkpoint
checkpoint = torch.load('checkpoints/promptgfm_film/best_model.pt')

# Initialize model
model = PromptGFM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
predictions = model.predict(disease_description, candidate_genes)
```

### Resuming Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/promptgfm_film/checkpoint_epoch_10.pt')

# Restore model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
trainer.train(start_epoch=start_epoch)
```

### Using the Resume Script

```bash
# Resume training from last checkpoint
python scripts/resume_training.py \
    --checkpoint checkpoints/promptgfm_film/checkpoint_epoch_10.pt \
    --config configs/base_config.yaml
```

---

## Checkpoint Frequency

Checkpoints are saved:
- **Every epoch**: Full checkpoint with optimizer state
- **Best model**: When validation metric improves
- **Final epoch**: At end of training

Configure in your config file:
```yaml
training:
  save_every: 1           # Save every N epochs
  keep_last: 5            # Keep only last N checkpoints
  save_best_only: false   # Save only when validation improves
```

---

## Storage Considerations

**Checkpoint sizes**:
- Full checkpoint (model + optimizer): ~200MB
- Model only: ~100MB
- Scripted model (TorchScript): ~110MB

**Disk space management**:
```bash
# Clean old checkpoints (keep best and last 3)
python scripts/clean_checkpoints.py --keep 3

# Archive old experiments
mv checkpoints/promptgfm_film/ checkpoints/archive/promptgfm_film_$(date +%Y%m%d)/
```

---

## Downloading Pretrained Models

Pretrained checkpoints are available separately:

```bash
# Download from release (when available)
wget https://github.com/yourusername/PromptGFM-Bio/releases/download/v1.0/pretrained_model.pt
mv pretrained_model.pt checkpoints/pretrained/

# Or use provided script
python scripts/download_pretrained.py --model film --output checkpoints/
```

---

## Checkpoint Metadata

Each `.json` file contains training information:

```json
{
  "epoch": 25,
  "best_auroc": 0.857,
  "best_aupr": 0.423,
  "config": {
    "model": "promptgfm",
    "gnn_type": "graphsage",
    "conditioning": "film",
    "hidden_dim": 512
  },
  "training_time": "2h 34m",
  "timestamp": "2026-02-19 15:30:45"
}
```

---

## Version Control

Checkpoints are excluded from Git via `.gitignore`:
```
checkpoints/*.pt
checkpoints/*.pth
checkpoints/*_*/
```

**To share models**:
1. Upload to cloud storage (Google Drive, AWS S3)
2. Create release on GitHub with model files
3. Use model hosting (HuggingFace Hub, PyTorch Hub)

---

## Model Hosting (Advanced)

### HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/promptgfm_film/best_model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="username/promptgfm-bio",
    repo_type="model"
)
```

### PyTorch Hub

Create `hubconf.py` in repo root:
```python
dependencies = ['torch', 'torch_geometric']

def promptgfm(pretrained=True, **kwargs):
    model = PromptGFM(**kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/username/PromptGFM-Bio/releases/download/v1.0/model.pt',
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

---

## Troubleshooting

### "Checkpoint file not found"

Checkpoints are not included in the repository. You must:
1. Train your own model, or
2. Download pretrained models separately

### "RuntimeError: Error in loading state_dict"

Model architecture mismatch. Ensure:
- Same model configuration as checkpoint
- Correct PyTorch and PyTorch Geometric versions
- Load with `strict=False` for partial loading

### "CUDA out of memory" when loading

```python
# Load checkpoint to CPU first
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
# Then move to GPU if needed
model = model.cuda()
```

---

For more information:
- [Training Guide](../TRAINING_GUIDE.md) - How to train models
- [Training Resume Guide](../TRAINING_RESUME_GUIDE.md) - Resume from checkpoints
- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Deploy trained models
