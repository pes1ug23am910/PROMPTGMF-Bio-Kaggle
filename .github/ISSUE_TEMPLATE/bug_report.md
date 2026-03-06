---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Go to '...'
2. Run command '...'
3. See error

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened instead.

## Error Messages

```
Paste any error messages, stack traces, or logs here
```

## Environment

**System Information:**
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python Version: [e.g., 3.10.8]
- CUDA Version: [e.g., 11.8, or "N/A" for CPU]
- GPU: [e.g., NVIDIA RTX 3090, or "N/A" for CPU]

**Package Versions:**
```bash
# Run and paste output:
pip list | grep -E "torch|geometric|transformers"
```

**PyTorch Info:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
```

## Configuration File

```yaml
# Paste relevant parts of your config file
```

## Additional Context

Add any other context about the problem here:
- Screenshots
- Related issues
- Possible cause
- Attempted solutions

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included relevant error messages and logs
- [ ] I have specified my environment details
