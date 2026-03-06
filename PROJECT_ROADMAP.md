# PromptGFM-Bio: Complete Project Roadmap

**Status**: Phase 3 COMPLETE ✅ | 100% Implementation Ready | Starting Training Phase 🚀

**Implementation Progress**:
- ✅ Phase 1: Environment Setup COMPLETE
- ✅ Phase 2: Data Pipeline COMPLETE (9.7M edges)
- ✅ Phase 3: Model Implementation COMPLETE (all components)
- 🚀 Phase 4: Training & Evaluation STARTING NOW

**Latest Update**: February 17, 2026 - All core and optional features implemented!

---

## Project Overview

**Goal**: Develop a prompt-conditioned graph foundation model for rare disease gene-phenotype mapping

**Core Innovation**: Dynamic prompt conditioning of GNN message passing (not static text features)

**Target Application**: Predicting gene associations for rare diseases with <5 known genes

**Hardware**: RTX 4060 (8GB VRAM) - **8.89x speedup** confirmed ✅

---

## Research Problem

**Question**: How can natural-language biomedical knowledge be integrated into graph foundation models for task-adaptive, long-tail rare-disease prediction?

**Limitations of Existing Approaches**:
- Graph-only learning: Ignores disease semantics  
- Static text concatenation: No dynamic task adaptation
- Neither enables adaptive reasoning for rare diseases

**Our Solution**: Prompt-conditioned GNN where disease descriptions dynamically modulate message passing

---

## Architecture Components

### 1. Biological Knowledge Graph (Heterogeneous)

**Node Types**:
- **Genes**: ~20,000 protein-coding genes (HGNC)
- **Diseases**: Rare diseases from Orphanet/DisGeNET
- **Phenotypes**: HPO terms

**Edge Types**:
- **Gene-Gene**: PPI from STRING/BioGRID
- **Gene-Disease**: Associations from DisGeNET
- **Disease-Phenotype**: HPO annotations

### 2. Prompt Encoder
```python
class PromptEncoder(nn.Module):
    # BioBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
    # Input: "Disease: Angelman Syndrome. Phenotypes: developmental delay, seizures, ataxia..."
    # Output: 768-dim embedding
```

### 3. GNN Backbone
```python
class GNNBackbone(nn.Module):
    # Options: GraphSAGE (start here), GAT, GIN
    # 3 layers, hidden_dim=512
    # Operates on heterogeneous graph
```

### 4. Conditioning Mechanisms

**Phase 1: FiLM (Weeks 5-6)**
```python
class FiLMConditioning(nn.Module):
    # gamma * node_features + beta
    # Where gamma, beta = f(prompt_embedding)
```

**Phase 2: Cross-Attention (Weeks 7-8)**
```python
class CrossAttentionConditioning(nn.Module):
    # Query: node embeddings
    # Key/Value: prompt embedding
    # More expressive but higher compute
```

### 5. Prediction Head
```python
# Rank genes by association score with query disease
# Output: gene_scores ∈ R^{num_genes}
```

---

## Data Sources & Preprocessing

### Raw Data Sources

| Source | Description | URL | Size |
|--------|-------------|-----|------|
| **STRING** | PPI network (confidence scores) | https://string-db.org | ~700 MB |
| **BioGRID** | Protein interactions | https://thebiogrid.org | ~500 MB |
| **DisGeNET** | Gene-disease associations | https://www.disgenet.org | ~300 MB |
| **HPO** | Phenotype ontology | https://hpo.jax.org | ~50 MB |
| **Orphanet** | Rare disease metadata | https://www.orphadata.org | ~100 MB |

### Graph Construction Pipeline

**Step 1: Parse PPI Networks**
- Filter to Homo sapiens
- Retain edges with confidence > 0.4
- Map to HGNC gene symbols
- Result: ~20,000 genes, ~500,000 edges

**Step 2: Parse Gene-Disease Associations**
- DisGeNET + Orphanet
- Filter to rare diseases (<5 genes initially)
- Result: ~4,000 diseases, ~12,000 gene-disease edges

**Step 3: Add Phenotype Annotations**
- Link diseases to HPO terms
- Create disease text prompts: name + phenotype descriptions
- Result: ~10,000 disease-phenotype edges

**Step 4: Create HeteroData Graph**
```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['gene'].x = ... # gene features (learned or one-hot)
data['disease'].text = ... # disease descriptions for prompt encoding
data['phenotype'].x = ... # HPO term embeddings

data['gene', 'interacts', 'gene'].edge_index = ... # PPI
data['gene', 'associated', 'disease'].edge_index = ... # gene-disease
data['disease', 'has', 'phenotype'].edge_index = ... # disease-phenotype
```

---

## Rare Disease Definition & Splits

### Disease Categorization
- **Ultra-rare**: 1-2 known genes (~30% of rare diseases)
- **Very rare**: 3-5 known genes (~25% of rare diseases)
- **Moderately rare**: 6-10 known genes (~20% of rare diseases)
- **Common rare**: 10+ known genes (~25% of rare diseases)

### Data Splits
```yaml
Train: 
  - Diseases with 10+ known genes
  - ~1,000 diseases
  - Used for learning graph representations

Validation:
  - Diseases with 6-10 known genes
  - ~500 diseases  
  - Hyperparameter tuning

Test-Few-Shot:
  - Diseases with 3-5 known genes
  - Split: 1-3 support, rest query
  - ~500 diseases
  - Main evaluation

Test-Zero-Shot:
  - Diseases with 1-2 known genes
  - Prompt-only (no labeled genes)
  - ~800 diseases
  - Stress test
```

---

## Training Strategy

### Stage 1: Graph Foundation Pretraining (Weeks 4-5)

**Objectives**:
```python
# 1. Masked Node Prediction
mask_ratio = 0.15
loss_masked = MSE(predicted_features, true_features)

# 2. Edge Contrastive Learning  
loss_contrastive = InfoNCE(positive_pairs, negative_pairs)

# 3. Context Prediction
loss_context = BCE(within_k_hops, labels)
```

**Configuration**:
```yaml
batch_size: 32
learning_rate: 0.0003
num_epochs: 50
loss_weights:
  masked_node: 1.0
  contrastive: 0.5
  context: 0.3
```

**Expected Time**: ~10 hours on RTX 4060

### Stage 2: Prompt-Conditioned Finetuning (Weeks 6-8)

**Loss Functions**:
```python
# 1. Binary Cross-Entropy
loss_bce = BCE(predictions, labels)

# 2. Margin Ranking Loss
loss_ranking = max(0, margin + neg_score - pos_score)

# 3. Contrastive (gene-prompt alignment)
loss_align = InfoNCE(gene_embs, prompt_emb)
```

**Configuration**:
```yaml
batch_size: 32
learning_rate: 0.00005  # Lower for finetuning
num_epochs: 100
early_stopping_patience: 10
loss_weights:
  ranking_loss: 1.0
  margin_loss: 0.8
  contrastive: 0.5
```

**Expected Time**: ~6 hours on RTX 4060

---

## Baseline Models

### Baseline 1: GNN-Only (No Text)
```python
class GNNBaseline(nn.Module):
    # Standard GraphSAGE on PPI network
    # Use disease node embedding (if disease nodes exist)
    # OR sum of connected gene embeddings
    # NO prompt information
```

**Purpose**: Shows value of text information

### Baseline 2: Static Text Concatenation
```python
class StaticTextGNN(nn.Module):
    prompt_emb = promptEncoder(disease_text)  # Encode ONCE
    node_features = torch.cat([gene_features, prompt_emb.expand(...)], dim=-1)
    output = GNN(node_features, edge_index)  # Standard GNN
```

**Purpose**: Shows value of DYNAMIC conditioning vs static features

### Baseline 3: Text-Only (No Graph)
```python
class TextOnlyBaseline(nn.Module):
    disease_emb = promptEncoder(disease_text)
    gene_desc_embs = promptEncoder(gene_descriptions)  # From UniProt
    scores = cosine_similarity(disease_emb, gene_desc_embs)
```

**Purpose**: Shows value of graph structure

---

## Evaluation Metrics

### Ranking Metrics (Primary)

```python
from sklearn.metrics import (
    roc_auc_score,         # AUROC - overall discrimination
    average_precision_score, # AUPR - important for imbalance
    ndcg_score             # NDCG - ranking quality
)

def evaluate(y_true, y_scores, k_values=[10, 20, 50]):
    metrics = {
        'auroc': roc_auc_score(y_true, y_scores),
        'aupr': average_precision_score(y_true, y_scores),
        'map': mean_average_precision(y_true, y_scores),
        'mrr': mean_reciprocal_rank(y_true, y_scores),
    }
    
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(y_true, y_scores, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(y_true, y_scores, k)
    
    return metrics
```

### Stratified Evaluation

```python
# Evaluate separately by disease rarity
results = {
    'ultra_rare': evaluate(ultra_rare_diseases),
    'very_rare': evaluate(very_rare_diseases),
    'moderately_rare': evaluate(moderately_rare_diseases),
    'all': evaluate(all_test_diseases)
}
```

### Few-Shot Evaluation

```python
# k-shot: use k labeled genes, predict rest
for k in [1, 3, 5]:
    support_genes = sample_k_genes(disease, k)
    query_genes = remaining_genes(disease)
    predictions = model.predict(disease_text, support_genes)
    metrics[f'{k}-shot'] = evaluate(query_genes, predictions)
```

---

## Case Study: Angelman Syndrome

### Disease Profile
- **Prevalence**: 1 in 15,000 (rare)
- **Primary Gene**: UBE3A (chromosome 15q11-q13)
- **Inheritance**: Maternal deletion (70%), mutation (11%), imprinting (6%)
- **Phenotypes**: 
  - Severe developmental delay
  - Absent speech
  - Ataxia and tremors
  - Seizures (80%)
  - Happy demeanor
  - Microcephaly

### Validation Strategy

**Positive Controls (should rank HIGH)**:
1. **UBE3A** - Primary gene (target: rank 1-3)
2. **Pathway genes** - Should rank in top 50
   - MAPK1 (UBE3A substrate)
   - PRMT5 (regulatory)
   - CDK1, CDK4 (cell cycle)
   - β-catenin (signaling)
   - UBXD8 (ubiquitin pathway)

**Negative Controls (should rank LOW)**:
3. **Differential diagnosis genes** - Should rank >500
   - MECP2 (Rett syndrome)
   - ZEB2 (Mowat-Wilson syndrome)
   - TCF4 (Pitt-Hopkins syndrome)

**Random Controls**:
4. Genes unrelated to neurology (e.g., immunological genes)

### Analysis Components

```python
class AngelmanCaseStudy:
    def run_analysis(self):
        # 1. Prompt generation
        prompt = self.create_prompt(
            disease="Angelman Syndrome",
            phenotypes=["developmental delay", "seizures", "ataxia", "happy demeanor"]
        )
        
        # 2. Get predictions
        all_genes = self.get_all_genes()
        scores = model.predict(prompt, all_genes)
        
        # 3. Rank analysis
        ube3a_rank = self.get_rank(scores, 'UBE3A')
        pathway_ranks = [self.get_rank(scores, g) for g in pathway_genes]
        negative_ranks = [self.get_rank(scores, g) for g in negative_controls]
        
        # 4. Top-K analysis
        top_50 = self.get_top_k(scores, k=50)
        
        # 5. Literature validation
        self.validate_against_literature(top_50)
        
        # 6. Attention visualization (if using cross-attention)
        self.visualize_attention_weights()
        
        # 7. Subgraph analysis
        self.plot_subgraph_around_predictions()
```

### Expected Results
- UBE3A rank: 1-5 (excellent: 1-3, good: 4-10, acceptable: 11-20)
- Pathway genes in top 50: ≥50% (3+ out of 6)
- Negative controls: rank >100
- Top 50 enriched for: neurodevelopmental genes, ubiquitin pathway

---

## Ablation Studies

### Study 1: Conditioning Mechanism
**Question**: FiLM vs Cross-Attention vs None?

```yaml
Variants:
  - baseline_no_conditioning
  - film_conditioning
  - cross_attention_conditioning
  - hybrid_film_early_cross_attn_late

Metrics:
  - AUPR on ultra-rare diseases
  - Precision@20
  - Training time
```

### Study 2: Pretraining Impact
**Question**: Does graph pretraining help?

```yaml
Variants:
  - no_pretraining (random init)
  - pretraining_10epochs
  - pretraining_50epochs

Metrics:
  - Few-shot performance (1-shot, 3-shot, 5-shot)
  - Zero-shot performance
```

### Study 3: Prompt Design
**Question**: What prompt information is most useful?

```yaml
Variants:
  - disease_name_only
  - disease_name_plus_top5_phenotypes
  - disease_name_plus_all_phenotypes
  - phenotypes_only

Metrics:
  - AUPR
  - Cross-disease generalization
```

### Study 4: Graph Architecture
**Question**: Which GNN backbone is best?

```yaml
Variants:
  - GraphSAGE_mean
  - GraphSAGE_lstm
  - GAT
  - GIN

Metrics:
  - AUPR
  - Training time
  - Memory usage
```

---

## Timeline & Milestones (12 weeks)

### ✅ Weeks 1-2: Setup (COMPLETE)
- [x] Environment setup
- [x] GPU configuration (RTX 4060)
- [x] Project structure
- [x] Documentation

### 📍 Weeks 3-4: Data Pipeline (CURRENT)
- [ ] Download raw datasets (STRING, BioGRID, DisGeNET, HPO)
- [ ] Implement parsers
- [ ] Construct heterogeneous graph
- [ ] Create train/val/test splits
- [ ] Data exploration notebooks

**Deliverable**: `data/processed/biomedical_graph.pt`

### Weeks 5-6: Model Implementation
- [ ] Implement GNN backbone (GraphSAGE)
- [ ] Implement prompt encoder (BioBERT)
- [ ] Implement FiLM conditioning
- [ ] Implement baselines
- [ ] Unit tests for all components

**Deliverable**: Working model architecture

### Weeks 7-8: Training & Optimization
- [ ] Graph pretraining (50 epochs)
- [ ] Prompt-conditioned finetuning
- [ ] Hyperparameter tuning
- [ ] Implement cross-attention (optional)
- [ ] Save best checkpoints

**Deliverable**: Trained models

### Weeks 9-10: Evaluation
- [ ] Baseline comparisons
- [ ] Rare disease evaluation
- [ ] Few-shot performance
- [ ] Ablation studies
- [ ] Statistical significance tests

**Deliverable**: Results tables

### Weeks 11-12: Case Study & Paper
- [ ] Angelman syndrome analysis
- [ ] Visualizations
- [ ] Draft paper
- [ ] Final presentation

**Deliverable**: Workshop paper + presentation

---

## Expected Outcomes

### Quantitative Results (Target)

| Metric | GNN-Only | Static Text | **PromptGFM** |
|--------|----------|-------------|---------------|
| AUROC (all) | 0.75 | 0.80 | **0.85** |
| AUPR (ultra-rare) | 0.45 | 0.55 | **0.70** |
| Precision@20 | 0.40 | 0.50 | **0.65** |
| MAP | 0.50 | 0.60 | **0.75** |

### Qualitative Results (Expected)

**Angelman Syndrome**:
- UBE3A rank: 1-3
- 4+ pathway genes in top 50
- Biologically coherent predictions

**Ablation Insights**:
- Pretraining improves few-shot by ~10%
- Cross-attention > FiLM by ~5% AUPR
- Full phenotype descriptions > name-only by ~15%

---

## Implementation Priority

### High Priority (Must Have)
1. ✅ GPU setup
2. Data pipeline (Week 3-4)
3. GraphSAGE backbone
4. BioBERT prompt encoder
5. FiLM conditioning
6. Training pipeline
7. Base evaluation

### Medium Priority (Should Have)
8. Cross-attention conditioning
9. Ablation studies
10. Case study
11. Visualization tools

### Low Priority (Nice to Have)
12. GAT/GIN alternatives
13. Advanced sampling strategies
14. Interactive demo
15. Extended case studies

---

## Quick Start (Next Steps)

### Step 1: Activate Environment
```bash
conda activate promptgfm
```

### Step 2: Start Data Pipeline (Week 3)
```bash
# Download data
bash scripts/download_data.sh

# Or implement download.py first
code src/data/download.py
```

### Step 3: Use Copilot for Implementation
Refer to `promptgfm_bio_copilot_prompt.md` for detailed implementation prompts for each module.

### Step 4: Test GPU During Development
```bash
python scripts/test_gpu.py
```

---

## Resources & References

### Code References
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- HuggingFace Transformers: https://huggingface.co/docs
- BioBERT: https://huggingface.co/dmis-lab/biobert-v1.1

### Paper References
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
- Graph Attention: Veličković et al., "Graph Attention Networks", ICLR 2018
- Prompt Learning: Liu et al., "Pre-train, Prompt, and Predict", ACM Computing Surveys 2023

### Data Sources
- STRING: https://string-db.org/
- DisGeNET: https://www.disgenet.org/
- HPO: https://hpo.jax.org/
- Orphanet: https://www.orphadata.org/

---

## Contact & Collaboration

For questions or issues:
1. Check documentation: `README.md`, `SETUP.md`, `GPU_TRAINING_GUIDE.md`
2. Review implementation prompts: `promptgfm_bio_copilot_prompt.md`
3. Test GPU: `python scripts/test_gpu.py`
4. Verify setup: `python scripts/verify_setup.py`

---

**Current Status**: ✅ Phase 1 Complete | 🚀 Ready for Phase 2: Data Pipeline

**Next Milestone**: Construct heterogeneous biomedical knowledge graph (Week 3-4)
