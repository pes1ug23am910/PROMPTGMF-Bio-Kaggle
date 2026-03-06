# Changelog

All notable changes to PromptGFM-Bio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-task learning support
- Additional GNN architectures (PNA, GatedGCN)
- Explainability modules (GNNExplainer, attention visualization)
- REST API for production deployment
- Integration with newer protein language models (ESM-2)

---

## [1.0.0] - 2026-02-19

### Added
- **Core Architecture**
  - PromptGFM model with BioBERT prompt encoder
  - GNN backbones: GraphSAGE, GAT, GIN
  - FiLM conditioning mechanism
  - Cross-attention conditioning mechanism
  - Hybrid conditioning (FiLM + Cross-Attention)

- **Training System**
  - Supervised fine-tuning pipeline
  - Self-supervised pretraining (masked prediction, contrastive learning)
  - Multiple loss functions: BCE, Margin Ranking, ListNet, Contrastive, Focal
  - Early stopping with patience
  - Gradient clipping and mixed precision support
  - Weights & Biases integration
  - Checkpoint management with automatic saving

- **Data Pipeline**
  - Automated dataset downloaders for STRING, BioGRID, DisGeNET, HPO, Orphanet
  - Biomedical knowledge graph construction
  - Heterogeneous graph support (gene, disease, phenotype nodes)
  - Stratified train/val/test splitting by disease rarity
  - Negative sampling strategies (random, hard, disease-aware)

- **Evaluation**
  - Comprehensive metrics: AUROC, AUPR, Precision@K, MAP, MRR, NDCG
  - Stratified evaluation by disease rarity
  - Few-shot learning evaluation
  - Case study validation: Angelman, Rett, Fragile X syndromes

- **Scripts**
  - Training script with configuration support
  - Evaluation script with case studies
  - Data download and preprocessing automation
  - Resume training from checkpoints
  - GPU testing and verification

- **Documentation**
  - Comprehensive README with quickstart
  - Architecture documentation
  - Deployment guide (Docker, cloud, API)
  - Troubleshooting guide
  - Contributing guidelines
  - API documentation

- **Repository Health**
  - MIT License
  - Code of Conduct
  - Issue templates (bug report, feature request, question)
  - Pull request template
  - CI/CD pipeline (GitHub Actions)
  - Comprehensive .gitignore

### Technical Details
- **Total codebase**: ~3,500 lines of production-ready code
- **Supported Python**: 3.10+
- **PyTorch**: 2.1.0
- **PyTorch Geometric**: 2.4.0
- **Graph size**: ~9.7M edges (protein-protein interactions)
- **Gene-disease associations**: ~87K edges

---

## Development Guidelines

### Version Numbering

- **Major (X.0.0)**: Breaking API changes, major architectural changes
- **Minor (1.X.0)**: New features, backward-compatible changes
- **Patch (1.0.X)**: Bug fixes, documentation updates

### Changelog Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

### Example Entry

```markdown
## [1.1.0] - 2026-03-15

### Added
- New GNN backbone: PNA (Principal Neighbourhood Aggregation)
- Multi-task learning support for drug-target prediction

### Changed
- Improved negative sampling with hard negative mining
- Updated BioBERT to latest version

### Fixed
- Memory leak in data loader (#45)
- NaN loss during training with small batch sizes (#52)

### Deprecated
- Old configuration format (will be removed in 2.0.0)
```

---

## Migration Guides

### Migrating from Pre-1.0 to 1.0

No migrations needed - this is the initial release.

---

## Links

- [PyPI Package](https://pypi.org/project/promptgfm-bio/) (when published)
- [GitHub Releases](https://github.com/yourusername/PromptGFM-Bio/releases)
- [Documentation](https://github.com/yourusername/PromptGFM-Bio/tree/main/docs)

---

[Unreleased]: https://github.com/pes1ug23am910/PromptGFM-Bio/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pes1ug23am910/PromptGFM-Bio/releases/tag/v1.0.0
