"""
Setup configuration for PromptGFM-Bio package.

Install with: pip install -e .
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

# Development requirements
dev_requirements_path = Path(__file__).parent / "requirements-dev.txt"
if dev_requirements_path.exists():
    dev_requirements = [
        line.strip()
        for line in dev_requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    dev_requirements = []

setup(
    name="promptgfm-bio",
    version="1.0.0",
    author="Yash Verma",
    author_email="yashverma.pes@gmail.com",
    description="A Prompt-Conditioned Graph Foundation Model for Rare Disease Gene Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pes1ug23am910/PromptGFM-Bio",
    project_urls={
        "Bug Tracker": "https://github.com/pes1ug23am910/PromptGFM-Bio/issues",
        "Documentation": "https://github.com/pes1ug23am910/PromptGFM-Bio/tree/main/docs",
        "Source Code": "https://github.com/pes1ug23am910/PromptGFM-Bio",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promptgfm-train=scripts.train:main",
            "promptgfm-eval=scripts.evaluate:main",
            "promptgfm-download=scripts.download_data:main",
            "promptgfm-preprocess=scripts.preprocess_all:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "graph neural networks",
        "bioinformatics",
        "rare diseases",
        "gene prediction",
        "precision medicine",
        "biomedical ai",
        "knowledge graphs",
        "transformers",
    ],
    zip_safe=False,
)
