# Documentation

Comprehensive documentation for PromptGFM-Bio.

---

## Quick Links

- **Main README**: [../README.md](../README.md) - Project overview and quickstart
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) - Technical deep dive
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

---

## Documentation Structure

### For Users

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Project overview, setup, quickstart | All users |
| [QUICKSTART.md](../QUICKSTART.md) | Fast setup for common use cases | New users |
| [SETUP.md](../SETUP.md) | Detailed installation instructions | All users |
| [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) | Training models from scratch | Researchers |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common problems and fixes | All users |

### For Developers

| Document | Purpose | Audience |
|----------|---------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design | Developers |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to contribute code | Contributors |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment options | DevOps, ML Engineers |

### For Researchers

| Document | Purpose | Audience |
|----------|---------|----------|
| [Project_Details.md](../Project_Details.md) | Research problem and methodology | Researchers |
| [CONFERENCE_PAPER_ROADMAP.md](../CONFERENCE_PAPER_ROADMAP.md) | Publication roadmap | Academic researchers |

---

## Getting Started

1. **New to the project?** Start with [README.md](../README.md)
2. **Want to train models?** See [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)
3. **Deploying to production?** Read [DEPLOYMENT.md](DEPLOYMENT.md)
4. **Encountering errors?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
5. **Want to contribute?** Read [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Documentation Sections

### Setup and Installation
- System requirements
- Environment setup
- Dependency installation
- Data download and preprocessing

### Usage
- Training models
- Evaluating performance
- Running inference
- Case studies

### Architecture
- Model components
- Data pipeline
- Training system
- Evaluation metrics

### Deployment
- Local deployment
- Docker containers
- Cloud platforms (AWS, GCP, Azure)
- API deployment
- Batch inference

### Development
- Code structure
- Testing guidelines
- Documentation standards
- Contribution workflow

---

## External Resources

### Papers and Publications
- BioBERT: [Lee et al., 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- PyTorch Geometric: [Fey & Lenssen, 2019](https://arxiv.org/abs/1903.02428)
- Graph Attention Networks: [Veličković et al., 2018](https://arxiv.org/abs/1710.10903)

### Datasets
- [STRING Database](https://string-db.org/)
- [BioGRID](https://thebiogrid.org/)
- [DisGeNET](https://www.disgenet.org/)
- [Human Phenotype Ontology](https://hpo.jax.org/)
- [Orphanet](https://www.orphadata.com/)

### Tools and Frameworks
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

## Contributing to Documentation

We welcome documentation improvements:

### What to Contribute
- Fix typos and errors
- Clarify confusing sections
- Add examples and tutorials
- Translate to other languages
- Update outdated information

### How to Contribute
1. **Small fixes**: Edit directly on GitHub (click "Edit this file")
2. **Large changes**: Fork, edit, and submit pull request
3. **New documentation**: Propose in GitHub issue first

### Documentation Standards
- Use Markdown format
- Include code examples
- Add links to related docs
- Keep language clear and concise
- Consider diverse audiences (beginners to experts)

---

## Building API Documentation

(For developers who want to generate API docs from code)

```bash
# Install documentation tools
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Generate documentation
cd docs/
sphinx-quickstart
make html

# View locally
python -m http.server 8000
# Open http://localhost:8000/_build/html/
```

---

## Documentation Versioning

Documentation is versioned along with the code:

- **main branch**: Latest stable documentation
- **dev branch**: In-progress features and updates
- **Release tags**: Documentation for specific versions

To view documentation for a specific version:
```bash
git checkout tags/v1.0.0
```

---

## Feedback

Found an issue with the documentation?

- **GitHub Issues**: Report documentation bugs
- **Pull Requests**: Submit fixes directly
- **Discussions**: Ask questions or suggest improvements

Your feedback helps improve this project for everyone!

---

## License

Documentation is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Code examples in documentation are licensed under the same license as the project (MIT).
