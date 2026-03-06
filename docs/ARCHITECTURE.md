# PromptGFM-Bio Architecture

This document provides a detailed technical overview of the PromptGFM-Bio system architecture.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation System](#evaluation-system)

---

## System Overview

PromptGFM-Bio is designed as a modular, scalable deep learning system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                   │
│  • Raw Data Downloaders (STRING, BioGRID, DisGeNET, HPO, etc.) │
│  • Preprocessing Pipeline (Graph Construction)                  │
│  • Dataset Loaders (PyTorch Geometric Data Objects)            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL LAYER                                  │
│  • GNN Backbone (GraphSAGE / GAT / GIN)                         │
│  • Prompt Encoder (BioBERT)                                     │
│  • Conditioning Modules (FiLM / Cross-Attention)                │
│  • Prediction Head (Gene Scoring)                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING LAYER                                 │
│  • Loss Functions (BCE, Ranking, Contrastive, Combined)        │
│  • Optimizers (AdamW with Warmup + Cosine Scheduling)          │
│  • Training Loop (Gradient Accumulation, Mixed Precision)       │
│  • Checkpointing & Early Stopping                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                                │
│  • Metrics (AUROC, AUPR, P@K, MAP, MRR, NDCG)                  │
│  • Case Studies (Rare Disease Validation)                       │
│  • Stratified Analysis (By Disease Rarity)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Graph Neural Network Backbone

**Location**: `src/models/gnn_backbone.py`

Implements three GNN architectures:

#### GraphSAGE (Default)
- **Aggregation**: Mean/Max/LSTM pooling of neighbor features
- **Layers**: 3 layers with residual connections
- **Hidden Dimension**: 512
- **Advantages**: Scalable, inductive learning, handles large graphs efficiently

```python
Message Passing:
h_v^(k+1) = σ(W^(k) · CONCAT(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)})))
```

#### Graph Attention Networks (GAT)
- **Attention Mechanism**: Multi-head self-attention on graph edges
- **Heads**: 8 attention heads per layer
- **Advantages**: Learns edge importance, interpretable attention weights

```python
Attention Coefficient:
α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
h_i' = σ(Σ_j α_ij W·h_j)
```

#### Graph Isomorphism Network (GIN)
- **Aggregation**: Sum aggregation with learnable ε
- **Expressive Power**: Most powerful GNN for graph isomorphism testing
- **Advantages**: Maximum discriminative power for graph classification

```python
Message Passing:
h_v^(k+1) = MLP((1 + ε^(k)) · h_v^(k) + Σ_{u∈N(v)} h_u^(k))
```

**Common Features**:
- Dropout (0.2) for regularization
- Batch normalization between layers
- Residual connections to prevent vanishing gradients
- ReLU activation functions

---

### 2. Prompt Encoder

**Location**: `src/models/prompt_encoder.py`

Encodes disease descriptions into semantic embeddings using pretrained biomedical language models.

#### Architecture
```
Input Text → Tokenizer → BioBERT → Pooling → Projection → Prompt Embedding
                                     (CLS)      (Linear)     (dim: 512)
```

#### Pretrained Model Options
1. **BioBERT** (Default): `dmis-lab/biobert-base-cased-v1.1`
   - Pretrained on PubMed abstracts and PMC full-text articles
   - Optimized for biomedical entity recognition

2. **PubMedBERT**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
   - Trained from scratch on PubMed
   - Better domain adaptation than general BERT

3. **SciBERT**: `allenai/scibert_scivocab_uncased`
   - Trained on scientific papers
   - Broader scientific vocabulary

#### Pooling Strategies
- **CLS Token** (Default): Use [CLS] token embedding
- **Mean Pooling**: Average all token embeddings
- **Max Pooling**: Take max across token embeddings

#### Fine-tuning
- Model can be frozen (feature extraction) or fine-tuned end-to-end
- Default: Fine-tune last 2 layers, freeze rest (computational efficiency)

---

### 3. Conditioning Modules

**Location**: `src/models/conditioning.py`

Enables prompt embeddings to modulate GNN layers dynamically.

#### FiLM (Feature-wise Linear Modulation)
```python
# For each GNN layer l:
γ_l, β_l = MLP(prompt_embedding)  # Learn scale and shift
h_out = γ_l ⊙ GNN_l(h_in) + β_l    # Modulate node features
```

**Advantages**:
- Lightweight (few parameters)
- Effective for feature scaling
- Computationally efficient

#### Cross-Attention
```python
# Node features attend to prompt embedding
Q = Linear(node_features)          # Query: from graph
K, V = Linear(prompt_embedding)    # Key, Value: from prompt
attention = softmax(Q @ K^T / √d)
h_out = attention @ V + node_features  # Residual connection
```

**Advantages**:
- More expressive than FiLM
- Learns which nodes are relevant to the prompt
- Provides interpretability via attention weights

#### Hybrid (FiLM + Cross-Attention)
- Applies both mechanisms sequentially
- Captures both global (FiLM) and local (attention) context

---

### 4. Prediction Head

**Location**: `src/models/promptgfm.py`

Maps node embeddings to gene-disease association scores.

```python
# For candidate gene node v:
score = MLP([node_embedding_v || global_graph_embedding || prompt_embedding])
output = sigmoid(score)  # Binary classification [0, 1]
```

**Components**:
1. **Node Embedding**: Final GNN output for the gene node
2. **Global Graph Embedding**: Aggregated information from entire graph
3. **Prompt Embedding**: Disease-specific context
4. **MLP**: 2-layer feedforward network with dropout

---

## Data Pipeline

### Raw Data Sources

| Source | Size | Description | Format |
|--------|------|-------------|--------|
| **STRING** | ~4GB | Protein-protein interactions (human) | TSV |
| **BioGRID** | ~500MB | Experimentally validated interactions | TAB3 |
| **DisGeNET** | ~200MB | Gene-disease associations | TSV |
| **HPO** | ~50MB | Phenotype-gene mappings | OBO/TSV |
| **Orphanet** | ~100MB | Rare disease definitions | XML |

### Preprocessing Steps

1. **Download Raw Data** (`scripts/download_data.py`)
   - Fetches latest versions from public APIs
   - Validates checksums
   - Stores in `data/raw/`

2. **Parse and Clean** (`src/data/preprocess.py`)
   - Extract relevant fields
   - Map identifiers (Ensembl, UniProt, HGNC)
   - Filter low-confidence interactions (STRING score > 700)

3. **Graph Construction** (`src/data/preprocess.py`)
   - Build heterogeneous graph:
     - Nodes: Genes, Diseases, Phenotypes
     - Edges: Gene-Gene (PPI), Gene-Disease, Gene-Phenotype
   - Store as PyTorch Geometric `HeteroData` object

4. **Train/Val/Test Split** (`src/data/dataset.py`)
   - Stratified by disease rarity (rare vs. common)
   - 70% train / 15% validation / 15% test
   - Ensures no data leakage across splits

### Dataset Classes

#### `BiomedicalGraphDataset`
- Loads preprocessed graph structure
- Provides node feature initialization
- Handles graph sampling for mini-batching

#### `GeneDiseaseDataset`
- Generates gene-disease pairs for training
- Implements negative sampling strategies:
  - **Random**: Sample random genes
  - **Hard**: Sample genes from related diseases
  - **Disease-Aware**: Sample within same disease category

---

## Model Architecture

### Forward Pass

```python
def forward(disease_description, candidate_genes, graph):
    # 1. Encode disease description
    prompt_emb = prompt_encoder(disease_description)
    
    # 2. Initialize node features
    node_features = initialize_node_features(graph)
    
    # 3. GNN message passing with conditioning
    for layer in gnn_layers:
        # Apply conditioning (FiLM or cross-attention)
        node_features = condition(node_features, prompt_emb)
        
        # Message passing
        node_features = gnn_layer(node_features, graph.edge_index)
    
    # 4. Generate scores for candidate genes
    scores = prediction_head(
        node_features[candidate_genes],
        global_graph_pooling(node_features),
        prompt_emb
    )
    
    return scores
```

### Model Variants

1. **PromptGFM-BioBERT-FiLM** (Default)
   - BioBERT prompt encoder
   - GraphSAGE backbone
   - FiLM conditioning
   - ~50M parameters

2. **PromptGFM-CrossAttn**
   - Cross-attention conditioning
   - More interpretable
   - ~70M parameters

3. **GNN-Only Baseline**
   - No prompt conditioning
   - Used for ablation studies
   - ~30M parameters

---

## Training Pipeline

### Optimization Strategy

```python
Optimizer: AdamW
  - Learning Rate: 1e-4
  - Weight Decay: 0.01
  - Betas: (0.9, 0.999)

Scheduler: Cosine Annealing with Warmup
  - Warmup Epochs: 5
  - Total Epochs: 100
  - Min LR: 1e-6
```

### Loss Functions

#### Combined Loss (Default)
```python
L_total = λ_rank · L_ranking + λ_contrast · L_contrastive
```

**Ranking Loss** (Margin-based):
```
L_rank = Σ max(0, margin - (s_pos - s_neg))
```
Ensures positive pairs score higher than negative pairs.

**Contrastive Loss**:
```
L_contrast = -log(exp(sim(g_i, d_i)/τ) / Σ_j exp(sim(g_i, d_j)/τ))
```
Maximizes similarity between true gene-disease pairs.

#### Alternative Losses
- **BCE**: Standard binary cross-entropy
- **ListNet**: Listwise ranking loss
- **Focal Loss**: Handles class imbalance

### Training Loop

```python
for epoch in epochs:
    for batch in dataloader:
        # Forward pass
        scores = model(batch.disease_desc, batch.genes, batch.graph)
        
        # Compute loss
        loss = criterion(scores, batch.labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Log metrics
        wandb.log({'loss': loss, 'lr': scheduler.get_last_lr()})
    
    # Validation
    val_metrics = evaluate(model, val_loader)
    
    # Early stopping
    if val_metrics['auroc'] > best_auroc:
        best_auroc = val_metrics['auroc']
        save_checkpoint(model, 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > early_stopping_patience:
            break
```

---

## Evaluation System

### Ranking Metrics

1. **AUROC** (Area Under ROC Curve)
   - Measures discrimination ability
   - Range: [0, 1], higher is better

2. **AUPR** (Area Under Precision-Recall Curve)
   - Better for imbalanced datasets
   - More informative than AUROC for rare diseases

3. **Precision@K** (P@K)
   - Precision in top K predictions
   - K = {10, 20, 50}

4. **Mean Average Precision** (MAP)
   - Average precision across all disease queries

5. **Mean Reciprocal Rank** (MRR)
   - Average of reciprocal ranks of first relevant result

6. **NDCG** (Normalized Discounted Cumulative Gain)
   - Considers ranking quality and position

### Stratified Evaluation

Performance is analyzed separately for:
- **Ultra-Rare Diseases**: < 3 known genes
- **Rare Diseases**: 3-10 known genes
- **Common Diseases**: > 10 known genes

### Case Studies

Three rare disease syndromes are validated:

1. **Angelman Syndrome**
   - Known genes: UBE3A, GABRB3
   - Tests maternally inherited genetic imprinting

2. **Rett Syndrome**
   - Known genes: MECP2, CDKL5, FOXG1
   - Neurological disorder primarily affecting females

3. **Fragile X Syndrome**
   - Known genes: FMR1
   - Most common inherited cause of intellectual disability

---

## Scalability Considerations

### Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision (FP16)**: Reduce memory by 50%
- **Graph Sampling**: Mini-batch GNN training with neighbor sampling

### Computational Efficiency
- **Multi-GPU Training**: Data parallelism with DistributedDataParallel
- **Efficient Attention**: Flash Attention for cross-attention modules
- **Cached Embeddings**: Precompute and cache prompt embeddings

### Storage
- **Compressed Graphs**: Use sparse matrix formats (COO, CSR)
- **Incremental Loading**: Load subgraphs on-demand
- **Checkpointing**: Save only model state_dict, not optimizer states

---

## Extension Points

The architecture is designed for extensibility:

1. **New GNN Layers**: Add to `gnn_backbone.py`
2. **New Conditioning**: Implement in `conditioning.py`
3. **New Encoders**: Swap BioBERT for ESM, ProtBERT, etc.
4. **Multi-Task Learning**: Extend prediction head for auxiliary tasks
5. **Explainability**: Integrate GNNExplainer, attention visualization

---

## References

- Hu et al. (2020). "Open Graph Benchmark: Datasets for Machine Learning on Graphs"
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
- Lee et al. (2020). "BioBERT: a pre-trained biomedical language representation model"
- Köhler et al. (2021). "The Human Phenotype Ontology"
