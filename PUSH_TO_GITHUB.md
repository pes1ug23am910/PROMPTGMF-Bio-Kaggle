# Push to GitHub - Quick Guide

**Author**: Yash Verma (PES1UG23AM910)  
**GitHub**: pes1ug23am910

---

## ✅ All Files Updated!

Your personal information has been updated in all files:
- ✅ GitHub username: **pes1ug23am910**
- ✅ Name: **Yash Verma**
- ✅ Email: **yashverma.pes@gmail.com**
- ✅ College SRN: **PES1UG23AM910**

---

## 🔒 SECURITY ALERT

**Your GitHub token is stored in `.env` file** which is:
- ✅ Excluded from git (in .gitignore)
- ✅ Never committed to repository
- ⚠️ Keep this token secret!

**Token Safety**:
```bash
# Verify .env is ignored
git check-ignore .env
# Should output: .env

# If you accidentally expose it, immediately:
# 1. Go to GitHub → Settings → Developer settings → Personal access tokens
# 2. Delete the compromised token
# 3. Generate a new one
```

---

## 🚀 Push to GitHub - Step by Step

### Step 1: Initialize Git Repository

```powershell
# Navigate to project directory
cd "E:\Lab\DLG\PromptGMF-Bio"

# Initialize git (if not already done)
git init

# Configure git with your info
git config user.name "Yash Verma"
git config user.email "yashverma.pes@gmail.com"

# Check current status
git status
```

### Step 2: Verify Large Files Are Excluded

```powershell
# Check what will be committed (should be ~50MB, not 10GB+)
git add .
git status

# Verify large files are ignored
git ls-files | Select-String -Pattern "\.pt$|\.pth$"
# Should return NOTHING (these are excluded)

# If you see .pt files, check your .gitignore
```

### Step 3: Create Initial Commit

```powershell
# Add all files
git add .

# Create commit
git commit -m "Initial commit: PromptGFM-Bio v1.0.0

Complete implementation of prompt-conditioned graph foundation model for rare disease gene prediction.

Features:
- BioBERT integration with GNN backbones (GraphSAGE, GAT, GIN)
- FiLM and cross-attention conditioning mechanisms
- Comprehensive training and evaluation pipeline
- Full documentation and deployment guides
- Production-ready codebase (~3,500 lines)

Author: Yash Verma (PES1UG23AM910)"

# Verify commit
git log -1
```

### Step 4: Create GitHub Repository

**Option A: Via GitHub Website**

1. Go to: https://github.com/new
2. **Repository name**: `PromptGFM-Bio`
3. **Description**: "A Prompt-Conditioned Graph Foundation Model for Rare Disease Gene Prediction"
4. **Visibility**: ✅ Public (for placement showcase)
5. **DO NOT** initialize with README, .gitignore, or license (you already have them)
6. Click **Create repository**

**Option B: Via GitHub CLI** (if installed)

```powershell
gh repo create PromptGFM-Bio --public --description "A Prompt-Conditioned Graph Foundation Model for Rare Disease Gene Prediction" --source=. --remote=origin --push
```

### Step 5: Push to GitHub

```powershell
# Add remote repository
git remote add origin https://github.com/pes1ug23am910/PromptGFM-Bio.git

# Verify remote
git remote -v

# Set branch name to main
git branch -M main

# Push to GitHub (first time)
git push -u origin main
```

**If prompted for credentials**, use:
- **Username**: pes1ug23am910
- **Password**: <use token from .env file>

**Or use token in URL** (more secure):
```powershell
# Alternative: Use token in remote URL (get token from .env file)
git remote set-url origin https://YOUR_TOKEN@github.com/pes1ug23am910/PromptGFM-Bio.git

# Then push
git push -u origin main
```

### Step 6: Verify Upload

Visit: https://github.com/pes1ug23am910/PromptGFM-Bio

Check:
- ✅ All source files present
- ✅ README displays correctly
- ✅ No large .pt or .pth files
- ✅ No .env file (contains token!)
- ✅ All documentation visible

---

## 🎨 Configure GitHub Repository

### 1. Add Topics (Tags)

Go to: https://github.com/pes1ug23am910/PromptGFM-Bio

Click "⚙️ About" (top right) → Add Topics:

```
machine-learning
deep-learning
graph-neural-networks
bioinformatics
rare-diseases
pytorch
transformers
biomedical-ai
precision-medicine
gnn
pytorch-geometric
gene-prediction
```

### 2. Update About Section

- **Description**: "Prompt-conditioned GNN for rare disease gene prediction | BioBERT + GraphSAGE | 87K gene-disease associations"
- **Website**: (leave blank or add portfolio link)

### 3. Enable Features

Go to: Settings → Features

- ✅ Issues
- ✅ Projects (optional)
- ✅ Discussions (optional - for Q&A)

### 4. Security Settings

Go to: Settings → Security → Code security and analysis

- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates

---

## 🏷️ Create Release

```powershell
# Create version tag
git tag -a v1.0.0 -m "Release v1.0.0 - Initial Production Release

PromptGFM-Bio: Prompt-Conditioned Graph Foundation Model

Features:
- Complete GNN + BioBERT architecture
- Multiple conditioning mechanisms (FiLM, Cross-Attention)
- Comprehensive training pipeline
- Full documentation
- Production-ready deployment

Metrics:
- 3,500+ lines of code
- 9.7M protein-protein interactions
- 87K gene-disease associations
- Validated on rare disease case studies

Author: Yash Verma (PES1UG23AM910)
Date: March 7, 2026"

# Push tag
git push origin v1.0.0
```

Then on GitHub:
1. Go to: https://github.com/pes1ug23am910/PromptGFM-Bio/releases
2. Click "Draft a new release"
3. Choose tag: `v1.0.0`
4. Release title: "**PromptGFM-Bio v1.0.0** - Initial Release"
5. Description: Copy from CHANGELOG.md
6. Click "Publish release"

---

## 📸 Add Social Preview Image (Optional)

Create a banner (1280x640px) with:
- Project name: PromptGFM-Bio
- Tagline: "Prompt-Conditioned GNN for Rare Disease Gene Prediction"
- Your name: Yash Verma

Upload at: Settings → General → Social preview

---

## 📊 Verify Repository Quality

### Check List

- [ ] README renders correctly
- [ ] All documentation links work
- [ ] No large files uploaded (check repository size < 100MB)
- [ ] License file present
- [ ] Topics/tags added
- [ ] Description set
- [ ] No .env file visible
- [ ] CI workflow running (may take 5-10 minutes)

### View Repository Stats

```powershell
# Check repository size
git count-objects -vH

# Should show:
# size-pack: ~50-80M (mostly code)
# NOT 10-15GB (that would mean large files were included)
```

---

## 🔧 Troubleshooting

### Problem: Large Files Uploaded

```powershell
# If you accidentally committed large files:

# 1. Remove from tracking
git rm --cached checkpoints/*.pt
git rm --cached data/processed/*.pt
git rm --cached data/raw/**/*

# 2. Commit the removal
git commit -m "Remove large files"

# 3. Clean git history (if already pushed)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch checkpoints/*.pt data/**/*.pt" \
  --prune-empty --tag-name-filter cat -- --all

# 4. Force push
git push origin main --force
```

### Problem: Token Not Working

```powershell
# Check token scope at: https://github.com/settings/tokens
# Required scopes: repo, workflow

# Generate new token if expired:
# GitHub → Settings → Developer settings → Personal access tokens → Generate new token
```

### Problem: Push Rejected

```powershell
# If remote has changes:
git pull origin main --rebase
git push origin main
```

---

## 🎓 For Placement Interviews

### Quick Repository Tour Script

```
1. "Let me show you my GitHub profile: github.com/pes1ug23am910"

2. "This is PromptGFM-Bio, my capstone project combining GNNs and NLP"

3. "The README explains the problem: rare disease gene prediction with limited data"

4. "I integrated multiple biomedical databases - here's the data pipeline"

5. "The architecture uses BioBERT to encode disease descriptions as prompts that condition GNN layers via FiLM modulation"

6. "I've documented everything - architecture, deployment, troubleshooting"

7. "The codebase is production-ready with 3,500 lines, testing, and CI/CD"

8. "I validated it on real rare diseases: Angelman, Rett, Fragile X syndromes"
```

### Key Points to Emphasize

✅ **Technical Complexity**: Multi-modal (graph + text), large-scale (9.7M edges)  
✅ **Engineering Quality**: Documentation, testing, deployment-ready  
✅ **Real-World Impact**: Addresses actual biomedical problem  
✅ **System Design**: Scalable architecture, GPU optimization  

---

## 📞 Next Steps After Pushing

### 1. Add to Resume/Portfolio

```
Project: PromptGFM-Bio
- Developed prompt-conditioned GNN for rare disease gene prediction
- Integrated 5 biomedical databases (9.7M edges, 87K associations)
- Built end-to-end ML pipeline with PyTorch & PyTorch Geometric
- Deployed with Docker, documented for production use
- Tech: Python, PyTorch, BioBERT, GNNs, GPU optimization
Link: github.com/pes1ug23am910/PromptGFM-Bio
```

### 2. LinkedIn Post

```
🧬 Excited to share my latest project: PromptGFM-Bio!

Built a deep learning system that combines Graph Neural Networks with 
natural language processing to predict rare disease genes.

Key challenges solved:
• Handling massive biomedical graphs (9.7M interactions)
• Integrating text and graph modalities
• Effective prediction with limited labeled data

Tech stack: PyTorch, PyTorch Geometric, BioBERT, Docker

Check it out: github.com/pes1ug23am910/PromptGFM-Bio

#MachineLearning #DeepLearning #Bioinformatics #AIForGood
```

### 3. Add to College Portal

Update PESU profile/portal with:
- GitHub link
- Project description
- Technologies used

---

## ✅ Final Checklist

Before sharing repository:

- [x] All personal info updated (Yash Verma, PES1UG23AM910)
- [x] GitHub token secured in .env (excluded from git)
- [ ] Repository initialized and committed
- [ ] Pushed to GitHub
- [ ] README renders correctly on GitHub
- [ ] Topics/tags added
- [ ] Release v1.0.0 created
- [ ] Repository size verified (<100MB)
- [ ] Added to resume/portfolio
- [ ] Shared on LinkedIn (optional)

---

## 🎉 Congratulations!

Your project is now live and ready for placement showcases!

**Repository URL**: https://github.com/pes1ug23am910/PromptGFM-Bio

Good luck with placements! 🚀

---

**Author**: Yash Verma  
**SRN**: PES1UG23AM910  
**Date**: March 7, 2026
