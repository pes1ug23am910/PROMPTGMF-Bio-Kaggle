# Contributing to PromptGFM-Bio

Thank you for your interest in contributing to PromptGFM-Bio! This document provides guidelines for contributing to the project.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat everyone with respect and professionalism
- **Be collaborative**: Work together constructively
- **Be inclusive**: Welcome diverse perspectives and backgrounds
- **Be constructive**: Provide helpful feedback
- **Focus on the science**: Keep discussions technical and objective

---

## Getting Started

### Ways to Contribute

We welcome various types of contributions:

1. **Bug Reports**: Identify and report issues
2. **Feature Requests**: Suggest new capabilities
3. **Code Contributions**: Fix bugs or implement features
4. **Documentation**: Improve or extend documentation
5. **Testing**: Write or improve tests
6. **Benchmarking**: Compare with other methods
7. **Examples**: Create tutorials or use case examples

### Before You Start

- **Check existing issues**: Look for related issues or discussions
- **Open an issue first**: For major changes, discuss your approach before coding
- **Ask questions**: If unsure, ask for clarification

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/PromptGFM-Bio.git
cd PromptGFM-Bio

# Add upstream remote
git remote add upstream https://github.com/pes1ug23am910/PromptGFM-Bio.git
```

### 2. Create Development Environment

```bash
# Create conda environment
conda create -n promptgfm-dev python=3.10
conda activate promptgfm-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

### 3. Download Test Data

```bash
# Download minimal dataset for testing
python scripts/download_data.py --datasets string hpo --sample
python scripts/preprocess_all.py --test-mode
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/ scripts/
black --check src/ scripts/

# Type checking
mypy src/
```

---

## Contribution Workflow

### 1. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-123
```

**Branch naming conventions**:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

### 2. Make Changes

- Write clear, concise code
- Follow existing code style
- Add comments for complex logic
- Update documentation as needed
- Write or update tests

### 3. Commit Changes

```bash
git add <changed files>
git commit -m "Brief description of changes"
```

**Commit message format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat: Add GAT support to GNN backbone

Implement Graph Attention Network as alternative to GraphSAGE.
Includes multi-head attention and edge feature support.

Closes #42
```

### 4. Keep Branch Updated

```bash
# Regularly sync with upstream
git fetch upstream
git rebase upstream/main

# Or merge if you prefer
git merge upstream/main
```

### 5. Push Changes

```bash
git push origin feature/your-feature-name
```

---

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black (with default settings)
- **Linter**: Flake8
- **Type hints**: Required for public functions
- **Docstrings**: Google style

### Code Formatting

```bash
# Format code with Black
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/
```

### Type Annotations

```python
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor

def predict_genes(
    disease_desc: str,
    candidate_genes: List[str],
    top_k: int = 10
) -> Tuple[List[str], Tensor]:
    """
    Predict top genes for a disease.
    
    Args:
        disease_desc: Natural language disease description
        candidate_genes: List of gene symbols to rank
        top_k: Number of top predictions to return
        
    Returns:
        Tuple of (gene_names, scores) for top K predictions
    """
    pass
```

### Documentation Style

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.
    
    Longer description with more details about the functionality,
    use cases, and any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        RuntimeError: When computation fails
        
    Examples:
        >>> result = example_function(10, "test")
        >>> print(result)
        True
    """
    pass
```

### Code Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch import nn

# Local imports
from src.models.gnn_backbone import GraphSAGE
from src.utils.config import load_config
```

---

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror source code structure
- Name test files `test_*.py`
- Name test functions `test_*`

**Example**:
```python
# tests/test_models/test_gnn_backbone.py
import pytest
import torch
from src.models.gnn_backbone import GraphSAGE

def test_graphsage_initialization():
    """Test GraphSAGE model initializes correctly."""
    model = GraphSAGE(
        input_dim=128,
        hidden_dim=256,
        output_dim=512,
        num_layers=3
    )
    assert model is not None
    assert model.num_layers == 3

def test_graphsage_forward_pass():
    """Test GraphSAGE forward pass produces correct output shape."""
    model = GraphSAGE(input_dim=128, hidden_dim=256, output_dim=512, num_layers=2)
    x = torch.randn(100, 128)  # 100 nodes, 128 features
    edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
    
    output = model(x, edge_index)
    
    assert output.shape == (100, 512)
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models/test_gnn_backbone.py

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models/test_gnn_backbone.py::test_graphsage_forward_pass

# Run with verbose output
pytest -v

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Test Markers

```python
import pytest

@pytest.mark.slow
def test_full_training_pipeline():
    """Slow integration test."""
    pass

@pytest.mark.gpu
def test_cuda_operations():
    """Test that requires GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pass
```

---

## Documentation

### Code Documentation

- **All public functions/classes**: Must have docstrings
- **Complex algorithms**: Add inline comments
- **Magic numbers**: Explain with comments or constants

### Updating Documentation

When making changes:

1. **Update docstrings** in code
2. **Update README.md** if adding features
3. **Update docs/** for major changes
4. **Add examples** for new functionality

### Building Documentation

```bash
# Generate API documentation
cd docs/
make html

# View locally
python -m http.server 8000
# Open http://localhost:8000/docs/
```

---

## Submitting Changes

### Creating a Pull Request

1. **Push your branch** to your fork
2. **Go to GitHub** and create a Pull Request
3. **Fill out the PR template**:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if UI changes)

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up-to-date with main
- [ ] No merge conflicts

### Review Process

1. **Automated checks** will run (tests, linting)
2. **Maintainers will review** your code
3. **Address feedback** by pushing new commits
4. **Once approved**, maintainers will merge

**Expected timeline**: 1-2 weeks for review

---

## Development Guidelines

### Project Structure

```
src/
  data/          # Data loading and preprocessing
  models/        # Model architectures
  training/      # Training loops and losses
  evaluation/    # Metrics and evaluation
  utils/         # Helper utilities

scripts/         # Standalone scripts
tests/           # Test files (mirrors src/)
configs/         # Configuration files
docs/            # Documentation
```

### Adding a New Model

1. **Create model file**: `src/models/your_model.py`
2. **Implement base class**: Inherit from `torch.nn.Module`
3. **Add configuration**: Update `configs/your_model_config.yaml`
4. **Write tests**: `tests/test_models/test_your_model.py`
5. **Update documentation**: Add to `docs/ARCHITECTURE.md`
6. **Add example**: Create usage example

### Adding a New Dataset

1. **Create downloader**: Add to `src/data/download.py`
2. **Implement preprocessor**: Add to `src/data/preprocess.py`
3. **Update dataset class**: Modify `src/data/dataset.py`
4. **Add tests**: Test data loading and preprocessing
5. **Update docs**: Document data format and source

### Adding a New Loss Function

1. **Implement loss**: Add to `src/training/losses.py`
2. **Add configuration**: Allow selection in config files
3. **Write tests**: Verify gradient flow and edge cases
4. **Document**: Explain mathematical formulation

---

## Performance Considerations

When contributing:

- **Profile code**: Use `torch.profiler` for bottlenecks
- **Optimize hot paths**: Focus on frequently called functions
- **Memory efficiency**: Consider GPU memory usage
- **Scalability**: Test with large graphs (>1M nodes)

---

## Release Process

(For maintainers)

1. **Update version**: In `setup.py` and `__init__.py`
2. **Update CHANGELOG**: Document all changes
3. **Create tag**: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. **Push tag**: `git push origin v1.0.0`
5. **Create GitHub release**: With release notes
6. **Update documentation**: Reflect new version

---

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: [maintainer email] for private inquiries

---

## Attribution

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to PromptGFM-Bio! 🧬🤖
