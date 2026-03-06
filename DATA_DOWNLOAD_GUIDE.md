# Data Download Module - Quick Reference

## Overview
The download module (`src/data/download.py`) handles automated downloading of all biomedical datasets required for PromptGFM-Bio.

## Datasets

### 1. BioGRID (~500MB)
- **Source**: https://downloads.thebiogrid.org/
- **Version**: 4.4.224
- **Content**: Protein-protein interaction data
- **Format**: Tab-delimited (tab3.zip)
- **Usage**: `download_biogrid(force=False)`

### 2. STRING (~700MB)
- **Source**: https://string-db.org/
- **Version**: v12.0
- **Content**: Protein network for Homo sapiens (9606)
- **Format**: Gzipped text files
- **Usage**: `download_string(organism='9606', score_threshold=400, force=False)`

### 3. DisGeNET (~300MB)
- **Source**: https://www.disgenet.org/
- **Version**: v7.0
- **Content**: Curated gene-disease associations
- **Format**: TSV (gzipped)
- **Usage**: `download_disgenet(version='v7.0', force=False)`
- **Note**: Public dataset included. For full access, register at disgenet.org

### 4. HPO (~50MB)
- **Source**: https://hpo.jax.org/
- **Content**: Human Phenotype Ontology annotations
- **Format**: Text files
- **Files**: phenotype_to_genes.txt, genes_to_phenotype.txt, phenotype.hpoa
- **Usage**: `download_hpo(force=False)`

## Features

### ✓ Progress Tracking
- Real-time progress bars using `tqdm`
- Shows download speed and estimated time remaining

### ✓ Error Handling
- Automatic retry with exponential backoff (max 3 attempts)
- Network timeout handling (300 seconds default)
- Graceful failure with informative error messages

### ✓ File Integrity
- MD5/SHA256 checksum verification
- Validates downloaded files automatically

### ✓ Smart Caching
- Checks if files already exist before downloading
- Use `force=True` to override and re-download

### ✓ Archive Extraction
- Automatically extracts .zip and .gz archives
- Preserves original archives for verification

### ✓ Directory Management
- Auto-creates data/raw/ subdirectories
- Organizes files by dataset (biogrid/, string/, disgenet/, hpo/)

## Usage Examples

### Quick Start - Download Everything
```bash
python scripts/download_data.py
```

### Download Specific Dataset
```bash
python scripts/download_data.py --dataset string
python scripts/download_data.py --dataset hpo
```

### Force Re-download
```bash
python scripts/download_data.py --force
python scripts/download_data.py --dataset biogrid --force
```

### Python API
```python
from src.data.download import download_all, download_biogrid

# Download all datasets
results = download_all(force=False, skip_failing=True)

# Download specific dataset
biogrid_files = download_biogrid(force=False)
print(biogrid_files)  # {'biogrid_zip': Path(...), 'biogrid_dir': Path(...)}
```

## Command Line Interface

The module can be run directly:
```bash
python -m src.data.download --dataset all
python -m src.data.download --dataset biogrid --force
```

## Directory Structure After Download

```
data/
└── raw/
    ├── biogrid/
    │   ├── BIOGRID-ALL-4.4.224.tab3.zip
    │   └── BIOGRID-ALL-4.4.224.tab3.txt  (extracted)
    ├── string/
    │   ├── 9606.protein.links.v12.0.txt.gz
    │   ├── 9606.protein.links.v12.0.txt  (extracted)
    │   ├── 9606.protein.info.v12.0.txt.gz
    │   └── 9606.protein.info.v12.0.txt   (extracted)
    ├── disgenet/
    │   ├── curated_gene_disease_associations.tsv.gz
    │   └── curated_gene_disease_associations.tsv  (extracted)
    └── hpo/
        ├── phenotype_to_genes.txt
        ├── genes_to_phenotype.txt
        └── phenotype.hpoa
```

## Testing

Test the module without downloading:
```bash
python scripts/test_download.py
```

This verifies:
- All functions can be imported
- Directory creation works
- Function signatures are correct
- Module is ready to use

## Troubleshooting

### Download Fails
1. Check internet connection
2. Verify URLs are still valid (datasets may update)
3. Try downloading individual datasets: `--dataset <name>`
4. Check disk space (need ~2GB free)

### Authentication Issues (DisGeNET)
- Public dataset is included
- For full DisGeNET access:
  1. Register at https://www.disgenet.org/signup/
  2. Download manually
  3. Place in `data/raw/disgenet/`

### URL Updates
If dataset URLs change:
1. Check dataset websites for new URLs
2. Update URLs in `src/data/download.py`
3. Update version numbers if needed

## Next Steps

After downloading data:
1. Run preprocessing: `python scripts/preprocess_all.py`
2. Build heterogeneous graph
3. Create train/val/test splits

## Performance Notes

- **Total download size**: ~1.5 GB
- **Disk space needed**: ~2 GB (with extracted files)
- **Estimated time**: 10-30 minutes (depends on connection)
- **Network bandwidth**: Downloads use full available bandwidth
- **Memory usage**: Minimal (~100-200 MB during extraction)

## Implementation Details

### Key Functions

**`_download_file_with_progress(url, output_path, max_retries=3, timeout=300)`**
- Core download function with progress tracking
- Handles retries and timeouts
- Returns True/False for success/failure

**`_verify_checksum(file_path, expected_hash=None, algorithm='md5')`**
- Computes and verifies file checksums
- Logs hash values for validation
- Optional comparison with expected hash

**`_extract_archive(archive_path, extract_to)`**
- Extracts .zip and .gz files
- Creates target directories automatically
- Preserves archive files

### Error Handling Strategy
1. Network errors → Retry with exponential backoff
2. File errors → Log error and continue to next dataset
3. Authentication errors → Provide manual download instructions
4. Disk space errors → Fail with clear message

## Credits

Datasets are provided by:
- **BioGRID**: The Biological General Repository for Interaction Datasets
- **STRING**: Protein-Protein Interaction Networks Database
- **DisGeNET**: A comprehensive platform on human gene-disease associations
- **HPO**: Human Phenotype Ontology Consortium

Please cite these resources in your publications.
