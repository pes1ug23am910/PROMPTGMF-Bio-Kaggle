# Running PromptGFM-Bio on Kaggle & VS Code Cloud

## Quick Comparison

| Platform | Free GPU | VRAM | Session Limit | Best For |
|---|---|---|---|---|
| **Kaggle Notebooks** | ✅ T4 (16 GB) or P100 | 16 GB | 9 h/session, 30 h/week | Full training runs |
| **GitHub Codespaces** | ❌ (CPU only on free tier) | — | 60 h/month free (CPU) | Code editing, debugging |
| **Lightning AI Studio** | ✅ T4 (free tier) | 16 GB | ~22 h/month free | Training + VS Code UI |
| **Google Colab Free** | ✅ T4 (limited) | 16 GB | ~3–5 h, no guarantee | Quick experiments |
| **Google Colab Pro** | ✅ A100 (paid) | 40 GB | Extended | Heavy training |

**Recommendation**: Use **Kaggle** for training (best free GPU, no credit card needed) and **GitHub Codespaces** or your local VS Code for code editing.

---

## Option A — Kaggle Notebooks (Recommended for Training)

### Step 1: Push Your Code to GitHub

Kaggle will clone your repo directly.

```powershell
cd E:\Lab\DLG\PromptGMF-Bio
git init           # if not already a git repo
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/PromptGFM-Bio.git
git push -u origin main
```

### Step 2: Create a Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com) → **Create** → **New Notebook**
2. Click **File** → **Import Notebook** → upload `notebooks/kaggle_training.ipynb`
   - Or start a blank notebook and paste the cells manually.
3. In the right panel → **Settings**:
   - **Accelerator**: GPU T4 x2 *(or P100)*
   - **Internet**: ✅ On
4. In **Cell 3** set `GITHUB_URL` to your repo URL.

### Step 3: Run the Notebook

Run all cells in order:
1. Environment check (confirm T4 GPU + 16 GB VRAM)
2. Install PyTorch Geometric *(auto-detects correct wheel URL)*
3. Clone your repo
4. Download data (~1.5 GB, ~5–10 min)
5. Preprocess (build knowledge graph, ~10–20 min)
6. Train with `configs/kaggle_config.yaml`

### Step 4: Persist Checkpoints Between Sessions

Kaggle sessions expire after 9 h. The training config saves a checkpoint every epoch.
After each session:

1. Run **Cell 10** to copy checkpoints to `/kaggle/working/saved_checkpoints/`
2. The session output is automatically saved; go to **Output** tab → **New Dataset**
3. Name it `promptgfm-checkpoints`
4. Next session: add that dataset as input → **Cell 7** will restore it automatically

### Kaggle T4 vs Your Laptop (RTX 4060)

| | Laptop RTX 4060 | Kaggle T4 |
|---|---|---|
| VRAM | 8 GB | 16 GB |
| `batch_size` | 32 | 64 |
| `hidden_dim` | 256 | 512 |
| `accumulation_steps` | 2 | 1 |
| ~Training speed | baseline | ~1.5–2× faster |

`configs/kaggle_config.yaml` is pre-tuned for these settings.

---

## Option B — GitHub Codespaces (VS Code in the Browser)

GitHub Codespaces gives you a full VS Code environment in the cloud, but **only CPU** on the free tier.
Use it for code editing, debugging on small inputs, and running non-GPU scripts.

### Setup

1. Push code to GitHub (see Step 1 above).
2. On the GitHub repo page → green **Code** button → **Codespaces** → **Create codespace on main**.
3. VS Code opens in your browser with the repo already cloned.

### Install Dependencies in Codespaces

```bash
pip install -r requirements.txt
# PyTorch Geometric CPU-only build:
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

### Limitations

- No GPU → training is ~8–10× slower, impractical for 100 epochs.
- Use for: linting, small unit tests, editing configs, reviewing outputs.
- **Free quota**: 60 h/month on 2-core, 8 GB RAM machines; 30 h/month on 4-core.

---

## Option C — Lightning AI Studio (VS Code + Free GPU)

[Lightning AI](https://lightning.ai) offers a **VS Code-based IDE** with a free GPU tier (T4, ~22 h/month).

### Setup

1. Sign up at <https://lightning.ai>
2. Create a new **Studio**
3. Select **VS Code** as the IDE + **T4 GPU**
4. Open a terminal and clone your repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PromptGFM-Bio.git
   cd PromptGFM-Bio
   pip install -r requirements.txt
   pip install torch-scatter torch-sparse torch-cluster torch-geometric \
       -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
   ```
5. Run training exactly as on your laptop (GPU auto-detected):
   ```bash
   python scripts/train.py --config configs/kaggle_config.yaml
   ```

This is the closest experience to your local VS Code — same GPU VRAM (16 GB T4), same interface.

---

## Tips for All Cloud Platforms

### Handle the 9-Hour Session Limit (Kaggle)
Checkpoints are already saved every epoch in `configs/kaggle_config.yaml`.  
To resume after a timeout:
```python
# In notebook Cell 9, set:
RESUME = True
```

### Reduce Data Download Time (Subsequent Runs)
After preprocessing once, save `data/processed/biomedical_graph.pt` as a Kaggle Dataset and mount it directly — skips all download and preprocessing steps.

### Monitor VRAM Usage
```python
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
print(f'Allocated: {allocated:.2f} GB  |  Reserved: {reserved:.2f} GB')
```

### Disable W&B if Not Needed
In `configs/kaggle_config.yaml`, `use_wandb: false` is already set.  
To enable, call `wandb.login(key='YOUR_KEY')` in the notebook before training.

---

## File Reference

| File | Purpose |
|---|---|
| `notebooks/kaggle_training.ipynb` | Step-by-step Kaggle notebook |
| `configs/kaggle_config.yaml` | T4-optimized training config (16 GB VRAM) |
| `configs/finetune_config.yaml` | Laptop config (8 GB VRAM, RTX 4060) |
| `scripts/train.py` | Main training entry point |
| `scripts/resume_training.py` | Resume from checkpoint |
| `scripts/download_data.py` | Download biomedical datasets |
| `scripts/preprocess_all.py` | Build knowledge graph |
