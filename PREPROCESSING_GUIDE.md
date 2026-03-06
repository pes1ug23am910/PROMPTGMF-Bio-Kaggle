# Graph Preprocessing Module - Quick Reference

## Overview
The preprocessing module (`src/data/preprocess.py`) constructs heterogeneous biomedical knowledge graphs from raw datasets.

## Pipeline Flow

```
Raw Data               Parsing                  Graph Construction
--------               -------                  ------------------
BioGRID    ──┐
STRING     ──┼──>  PPI Edges        ─┐
             │                        │
DisGeNET   ──┼──>  Gene-Disease      ├──>  HeteroData Graph
             │      Edges             │     (PyTorch Geometric)
HPO        ──┘                        │
             └──>  Disease-Phenotype ─┘
                   Edges
```

## Graph Structure

### Node Types
1. **Gene** (`gene`)
   - Total: ~20,000 human genes
   - Attributes: Gene symbols (HGNC standard)
   - Features: Learned during model training

2. **Disease** (`disease`)
   - Total: ~1,000-4,000 rare diseases
   - Attributes: Disease IDs, descriptions
   - Filter: Diseases with ≤100 known genes

3. **Phenotype** (`phenotype`) [Optional]
   - Total: ~13,000 HPO terms
   - Attributes: HPO IDs, descriptions
   - Source: Human Phenotype Ontology

### Edge Types
1. **Gene-Gene** (`gene → interacts → gene`)
   - Source: BioGRID + STRING
   - Attributes: Confidence scores [0, 1]
   - Filter: Confidence ≥ 0.4 for STRING

2. **Gene-Disease** (`gene → associated_with → disease`)
   - Source: DisGeNET
   - Attributes: Association scores
   - Bidirectional: `disease → rev_associated_with → gene`

3. **Disease-Phenotype** (`disease → has_phenotype → phenotype`)
   - Source: HPO
   - Links diseases to phenotypic features

## Key Functions

### `parse_biogrid(filepath, organism='Homo sapiens')`
Parses BioGRID protein-protein interaction data.

**Input**: BioGRID tab3 file (tab-delimited)
**Output**: Edge DataFrame + set of unique genes
**Filtering**:
- Human-human interactions only
- Remove self-loops
- Normalize gene symbols to HGNC

**Usage**:
```python
from src.data.preprocess import parse_biogrid
edges, genes = parse_biogrid('data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.txt')
```

### `parse_string(filepath, info_filepath, min_score=400)`
Parses STRING protein network database.

**Input**: 
- STRING protein.links file (space-separated)
- STRING protein.info file (for gene mapping)

**Output**: Edge DataFrame + set of unique genes
**Filtering**:
- Combined score ≥ 400 (medium confidence)
- Maps protein IDs to gene symbols
- Normalizes scores to [0, 1]

**Usage**:
```python
from src.data.preprocess import parse_string
edges, genes = parse_string(
    'data/raw/string/9606.protein.links.v12.0.txt',
    'data/raw/string/9606.protein.info.v12.0.txt',
    min_score=400
)
```

### `parse_ppi_network(biogrid_path, string_path, string_info_path, min_confidence=0.4)`
Combines BioGRID and STRING networks.

**Output**: Combined edge DataFrame with deduplicated edges
**Deduplication**: Keeps highest confidence score per edge

**Usage**:
```python
from src.data.preprocess import parse_ppi_network
edges, genes = parse_ppi_network(
    biogrid_path='data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.txt',
    string_path='data/raw/string/9606.protein.links.v12.0.txt',
    string_info_path='data/raw/string/9606.protein.info.v12.0.txt',
    min_confidence=0.4
)
```

### `parse_disgenet(filepath, rare_only=True, max_known_genes=100)`
Parses DisGeNET gene-disease associations.

**Input**: DisGeNET TSV file (tab-separated)
**Output**: Edge DataFrame + disease info dict
**Filtering**:
- `rare_only=True`: Only diseases with ≤ max_known_genes
- Focus on rare/orphan diseases for project scope

**Usage**:
```python
from src.data.preprocess import parse_disgenet
edges, disease_info = parse_disgenet(
    'data/raw/disgenet/curated_gene_disease_associations.tsv',
    rare_only=True,
    max_known_genes=100
)
```

### `parse_hpo(phenotype_to_genes_path)`
Parses HPO phenotype annotations.

**Input**: phenotype_to_genes.txt
**Output**: Gene-phenotype edges + phenotype info dict

**Usage**:
```python
from src.data.preprocess import parse_hpo
edges, phenotype_info = parse_hpo(
    'data/raw/hpo/phenotype_to_genes.txt'
)
```

### `build_heterogeneous_graph(ppi_edges, gene_disease_edges, disease_info, ...)`
Constructs PyTorch Geometric HeteroData graph.

**Process**:
1. Create node ID mappings (gene, disease, phenotype)
2. Convert edge lists to PyG format (edge_index tensors)
3. Add node attributes (descriptions, IDs)
4. Add edge attributes (confidence scores)
5. Create bidirectional edges where needed

**Output**: HeteroData object with:
- `data['gene'].num_nodes`
- `data['disease'].num_nodes`
- `data['gene', 'interacts', 'gene'].edge_index`
- `data['gene', 'associated_with', 'disease'].edge_index`
- Edge attributes for confidence scores

**Usage**:
```python
from src.data.preprocess import build_heterogeneous_graph
graph = build_heterogeneous_graph(
    ppi_edges=ppi_df,
    gene_disease_edges=disgenet_df,
    disease_info=disease_dict
)
```

### `save_graph(graph, output_path)`
Saves processed graph to disk.

**Output**:
- `biomedical_graph.pt` - PyTorch serialized HeteroData
- `biomedical_graph_stats.txt` - Human-readable statistics

**Usage**:
```python
from src.data.preprocess import save_graph
save_graph(graph, 'data/processed/biomedical_graph.pt')
```

### `preprocess_all(force=False)`
Runs complete preprocessing pipeline.

**Process**:
1. Locate all raw data files
2. Parse BioGRID and STRING → PPI network
3. Parse DisGeNET → gene-disease associations
4. Parse HPO → phenotype annotations
5. Build heterogeneous graph
6. Save to `data/processed/biomedical_graph.pt`

**Usage**:
```bash
python scripts/preprocess_all.py
python scripts/preprocess_all.py --force  # Reprocess
```

Or in Python:
```python
from src.data.preprocess import preprocess_all
preprocess_all(force=False)
```

## Helper Functions

### `_normalize_gene_symbol(symbol)`
Normalizes gene symbols to HGNC standard.

**Transformations**:
- Convert to uppercase
- Strip whitespace
- Remove species prefixes (HUMAN_, Hs_, ENSP)

**Examples**:
```python
_normalize_gene_symbol("tp53")      # Returns "TP53"
_normalize_gene_symbol("  BRCA1 ") # Returns "BRCA1"
_normalize_gene_symbol("HUMAN_MYC") # Returns "MYC"
```

### `_get_data_dirs()`
Returns dictionary of data directory paths.

**Returns**:
```python
{
    'raw': Path('data/raw'),
    'processed': Path('data/processed')
}
```

## Usage Examples

### Complete Pipeline
```bash
# After downloading data
python scripts/preprocess_all.py
```

### Custom Processing
```python
from src.data.preprocess import *
from pathlib import Path

# Parse individual datasets
biogrid_edges, genes1 = parse_biogrid(
    Path('data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.txt')
)

string_edges, genes2 = parse_string(
    Path('data/raw/string/9606.protein.links.v12.0.txt'),
    Path('data/raw/string/9606.protein.info.v12.0.txt'),
    min_score=500  # High confidence only
)

# Combine PPI networks
import pandas as pd
all_ppi = pd.concat([biogrid_edges, string_edges])

# Parse gene-disease
gd_edges, disease_info = parse_disgenet(
    Path('data/raw/disgenet/curated_gene_disease_associations.tsv'),
    rare_only=True,
    max_known_genes=50  # Very rare diseases only
)

# Build graph
graph = build_heterogeneous_graph(
    ppi_edges=all_ppi,
    gene_disease_edges=gd_edges,
    disease_info=disease_info
)

# Save
save_graph(graph, Path('data/processed/my_custom_graph.pt'))
```

## Expected Output

### Graph Statistics
After preprocessing, you should see:

```
Graph Statistics:
  Gene nodes: ~20,000
  Disease nodes: ~1,000-4,000 (depending on filtering)
  Edge types: [
    ('gene', 'interacts', 'gene'),
    ('gene', 'associated_with', 'disease'),
    ('disease', 'rev_associated_with', 'gene')
  ]
  ('gene', 'interacts', 'gene'): ~400,000-500,000 edges
  ('gene', 'associated_with', 'disease'): ~5,000-50,000 edges
  ('disease', 'rev_associated_with', 'gene'): ~5,000-50,000 edges
```

### Files Created
```
data/processed/
├── biomedical_graph.pt           # PyG HeteroData (100-500MB)
└── biomedical_graph_stats.txt    # Human-readable stats
```

## Testing

Test the module without full datasets:
```bash
python scripts/test_preprocess.py
```

Tests verify:
- ✓ All functions can be imported
- ✓ Gene symbol normalization works
- ✓ Directory structure is correct
- ✓ Graph construction works with toy data

## Troubleshooting

### Missing Files
**Error**: `FileNotFoundError: data/raw/.../...`
**Solution**: Run download first: `python scripts/download_data.py`

### Memory Issues
**Error**: Out of memory during processing
**Solutions**:
1. Increase rare disease threshold (fewer diseases)
2. Increase PPI confidence threshold (fewer edges)
3. Process datasets separately

### Gene Symbol Mismatches
**Issue**: Some genes not mapping correctly
**Solutions**:
1. Check gene symbol normalization
2. Verify HGNC symbol usage
3. Update mapping tables if needed

### Empty Graph
**Issue**: Graph has very few nodes/edges
**Solutions**:
1. Check input file formats
2. Verify filtering thresholds aren't too strict
3. Check log messages for parsing errors

## Performance Notes

- **Processing time**: 5-15 minutes on typical hardware
- **Memory usage**: 2-4 GB RAM during processing
- **Output size**: 100-500 MB for processed graph
- **Bottlenecks**: File I/O, pandas operations

## Next Steps

After preprocessing:
1. **Create dataset splits**: `python -m src.data.dataset`
2. **Verify graph**: Load and inspect in notebook
3. **Begin training**: Model implementation in Phase 3

## Implementation Details

### Data Flow
1. **Raw files** → Pandas DataFrames
2. **DataFrames** → Filtered/normalized DataFrames
3. **DataFrames** → Node/edge index mappings
4. **Mappings** → PyTorch tensors
5. **Tensors** → HeteroData object

### Memory Optimization
- Streaming reads for large files
- Drop unnecessary columns early
- Use categorical types where applicable
- Clear intermediate DataFrames

### Type Conversions
- Gene symbols: `str` → `int` indices → `torch.long`
- Confidence scores: `float` → `torch.float`
- Edge indices: `List[int]` → `torch.tensor([src], [dst])`

## Credits

Implements preprocessing for:
- **BioGRID**: Protein interaction database
- **STRING**: Functional protein association networks
- **DisGeNET**: Gene-disease association database
- **HPO**: Human Phenotype Ontology

Please cite these resources in publications.
