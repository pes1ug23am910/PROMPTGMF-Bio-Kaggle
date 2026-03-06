# GitHub Copilot Setup Prompt for PromptGFM-Bio Project

## Project Context
I'm building **PromptGFM-Bio**, a prompt-conditioned graph foundation model for rare-disease gene-phenotype mapping. This is a semester-long deep learning project combining Graph Neural Networks (GNNs) with natural language processing for biomedical applications.

**Key Innovation:** Using disease descriptions as dynamic prompts to condition GNN message passing, rather than static text features, enabling task-adaptive gene discovery for rare diseases.

---

## Phase 1: Environment Setup & Project Structure

### 1.1 Create Python Environment
```bash
# Set up conda environment with Python 3.10
conda create -n promptgfm python=3.10
conda activate promptgfm

# Install core dependencies
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install transformers and NLP tools
pip install transformers==4.35.0 sentence-transformers==2.2.2
pip install datasets==2.14.0

# Install biomedical data tools
pip install biopython==1.81 pandas==2.1.0 numpy==1.24.0
pip install networkx==3.1 scipy==1.11.0

# Install utilities
pip install scikit-learn==1.3.0 matplotlib==3.8.0 seaborn==0.12.0
pip install tqdm==4.66.0 wandb==0.15.0 pyyaml==6.0

# Save requirements
pip freeze > requirements.txt
```

### 1.2 Project Directory Structure
Create this folder structure:
```
promptgfm-bio/
├── data/
│   ├── raw/                    # Downloaded datasets
│   │   ├── biogrid/
│   │   ├── string/
│   │   ├── disgenet/
│   │   └── hpo/
│   ├── processed/              # Preprocessed graphs
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # Data downloaders
│   │   ├── preprocess.py       # Graph construction
│   │   └── dataset.py          # PyG dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_backbone.py     # GraphSAGE/GAT/GIN
│   │   ├── prompt_encoder.py   # BioBERT encoder
│   │   ├── conditioning.py     # FiLM & Cross-attention
│   │   └── promptgfm.py        # Main model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py         # Self-supervised training
│   │   ├── finetune.py         # Supervised training
│   │   └── losses.py           # Loss functions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # AUROC, AUPR, MAP, etc.
│   │   └── case_study.py       # Angelman syndrome validation
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
├── configs/
│   ├── base_config.yaml
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_statistics.ipynb
│   └── 03_results_visualization.ipynb
├── scripts/
│   ├── download_data.sh
│   ├── preprocess_all.py
│   ├── train_baseline.py
│   └── train_promptgfm.py
├── tests/
│   └── test_models.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Phase 2: Data Pipeline Implementation

### 2.1 Data Download Script (`src/data/download.py`)

**Copilot Prompt:**
```
Create a Python script to download biomedical datasets for gene-disease mapping:

1. BioGRID protein-protein interactions (download from https://downloads.thebiogrid.org/)
2. STRING database PPI (download v11.5 from https://string-db.org/cgi/download)
3. DisGeNET gene-disease associations (download from https://www.disgenet.org/downloads)
4. Human Phenotype Ontology (HPO) from https://hpo.jax.org/app/download/annotation

Requirements:
- Use requests library for downloads
- Implement progress bars with tqdm
- Handle retries and errors gracefully
- Cache downloaded files to avoid re-downloading
- Verify file integrity with checksums
- Create data/raw/ subdirectories automatically

Include functions:
- download_biogrid()
- download_string()
- download_disgenet()
- download_hpo()
- download_all()

Add proper logging and error handling.
```

### 2.2 Graph Construction (`src/data/preprocess.py`)

**Copilot Prompt:**
```
Create a graph preprocessing pipeline that builds a heterogeneous biomedical knowledge graph:

Input data:
- BioGRID/STRING PPI networks (protein-protein edges)
- DisGeNET gene-disease associations
- HPO phenotype annotations

Output:
- PyTorch Geometric HeteroData graph with node types: [gene, disease, phenotype]
- Edge types: [gene-gene (PPI), gene-disease, disease-phenotype]

Requirements:
1. Parse BioGRID/STRING files to create gene-gene edges with confidence scores
2. Parse DisGeNET to create gene-disease edges with association scores
3. Parse HPO to create disease-phenotype edges
4. Map all gene IDs to HGNC symbols (canonical naming)
5. Create node feature matrices:
   - Gene nodes: one-hot encoding or random initialization (will be learned)
   - Disease nodes: store text descriptions for later prompt encoding
   - Phenotype nodes: HPO term embeddings
6. Filter to Homo sapiens only
7. Remove duplicate edges and self-loops
8. Save as PyTorch Geometric HeteroData object

Functions needed:
- parse_ppi_network(filepath) -> edge_index, edge_weights
- parse_disgenet(filepath) -> gene_disease_edges, scores
- parse_hpo(filepath) -> disease_phenotype_edges, phenotype_texts
- build_heterogeneous_graph(ppi_edges, gene_disease_edges, disease_pheno_edges) -> HeteroData
- save_graph(graph, output_path)

Add statistics logging (num nodes, num edges per type, graph density).
```

### 2.3 Dataset Class (`src/data/dataset.py`)

**Copilot Prompt:**
```
Create PyTorch Geometric dataset classes for gene-disease prediction:

1. BiomedicaGraphDataset (base class):
   - Load preprocessed HeteroData graph
   - Handle train/val/test splits based on disease rarity
   - Implement rare disease stratification:
     * Ultra-rare: 1-2 known genes
     * Very rare: 3-5 known genes
     * Moderately rare: 6-10 known genes
     * Common rare: 10+ known genes

2. GeneDiseaseDataset (for link prediction):
   - Returns (disease_idx, prompt_text, positive_genes, negative_genes)
   - Implements negative sampling (sample genes not associated with disease)
   - Supports few-shot splits (k=1,3,5 support examples)

3. Implement collate function for batch processing:
   - Batch disease prompts together
   - Create subgraphs around candidate genes
   - Handle variable-length positive/negative gene lists

Key methods:
- split_by_rarity(min_genes, max_genes) -> train/val/test indices
- create_few_shot_split(k_shot=3) -> support/query sets
- get_disease_prompt(disease_idx) -> str (disease name + phenotype descriptions)
- negative_sample(disease_idx, num_negatives) -> gene_indices

Include data augmentation for rare diseases (e.g., synonym replacement in prompts).
```

---

## Phase 3: Model Architecture

### 3.1 GNN Backbone (`src/models/gnn_backbone.py`)

**Copilot Prompt:**
```
Implement flexible GNN backbone supporting multiple architectures:

Create a modular GNN that supports:
1. GraphSAGE (mean/max/LSTM aggregation)
2. GAT (Graph Attention Networks with multi-head attention)
3. GIN (Graph Isomorphism Network)

For heterogeneous graphs, use PyG's HeteroConv wrapper.

Requirements:
- Support 2-4 layer configurations
- Hidden dimensions: 256/512
- Dropout: 0.1-0.3
- Residual connections between layers
- Layer normalization after each GNN layer
- Support both homogeneous and heterogeneous graphs

Class structure:
```python
class GNNBackbone(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_layers,
                 gnn_type='graphsage',  # or 'gat', 'gin'
                 dropout=0.2,
                 use_residual=True):
        # Initialize layers
        
    def forward(self, x, edge_index, edge_attr=None):
        # Multi-layer message passing
        # Return node embeddings
```

Add edge attribute handling for weighted graphs (PPI confidence scores).
Include BatchNorm and skip connections for deep GNNs.
```

### 3.2 Prompt Encoder (`src/models/prompt_encoder.py`)

**Copilot Prompt:**
```
Create a biomedical prompt encoder using pretrained language models:

Use: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

Requirements:
1. Encode disease descriptions and phenotype lists into fixed-size embeddings
2. Support prompt templates:
   - "Disease: {disease_name}. Phenotypes: {phenotype_list}. Associated genes:"
   - "Clinical presentation: {phenotypes}. Molecular cause:"
3. Implement pooling strategies:
   - CLS token embedding
   - Mean pooling over all tokens
   - Max pooling
4. Cache encoded prompts to avoid recomputing
5. Support batched encoding

Class structure:
```python
class PromptEncoder(nn.Module):
    def __init__(self, 
                 model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 pooling='cls',  # or 'mean', 'max'
                 max_length=128,
                 device='cuda'):
        # Load pretrained model
        
    def encode_disease_prompt(self, disease_name, phenotypes_list):
        # Create prompt template
        # Tokenize and encode
        # Return embedding (768-dim for BERT-base)
        
    def forward(self, prompt_texts):
        # Batch encoding
```

Add prompt augmentation (paraphrasing, phenotype shuffling) for robustness.
Implement gradient checkpointing for memory efficiency.
```

### 3.3 Conditioning Mechanisms (`src/models/conditioning.py`)

**Copilot Prompt:**
```
Implement two conditioning mechanisms to fuse prompt embeddings into GNN layers:

1. FiLM (Feature-wise Linear Modulation):
   - Takes prompt embedding and generates scale/shift parameters
   - Applies affine transformation to GNN node embeddings
   - Formula: FiLM(h) = γ(prompt) ⊙ h + β(prompt)
   
2. Cross-Attention Conditioning:
   - Queries: GNN node embeddings
   - Keys/Values: Prompt embedding (broadcast to all nodes)
   - Multi-head attention to fuse prompt information
   - Residual connection + LayerNorm

Class structure:
```python
class FiLMConditioning(nn.Module):
    def __init__(self, node_dim, prompt_dim):
        self.gamma_net = nn.Linear(prompt_dim, node_dim)  # scale
        self.beta_net = nn.Linear(prompt_dim, node_dim)   # shift
        
    def forward(self, node_features, prompt_embedding):
        gamma = self.gamma_net(prompt_embedding)
        beta = self.beta_net(prompt_embedding)
        return gamma * node_features + beta

class CrossAttentionConditioning(nn.Module):
    def __init__(self, node_dim, prompt_dim, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            kdim=prompt_dim,
            vdim=prompt_dim
        )
        self.layer_norm = nn.LayerNorm(node_dim)
        
    def forward(self, node_features, prompt_embedding):
        # node_features: [num_nodes, node_dim]
        # prompt_embedding: [1, prompt_dim] -> broadcast
        # Apply cross-attention
        # Add residual + normalize
```

Support both single-layer and hierarchical conditioning (different prompts per layer).
```

### 3.4 Complete PromptGFM Model (`src/models/promptgfm.py`)

**Copilot Prompt:**
```
Integrate all components into the complete PromptGFM-Bio model:

Architecture flow:
1. Encode disease prompt using PromptEncoder
2. Pass graph through GNN backbone with prompt conditioning at each layer
3. Extract gene node embeddings
4. Compute gene-disease association scores

Support three modes:
- Baseline GNN (no conditioning)
- Static text concatenation (concat prompt to node features once)
- Dynamic conditioning (FiLM or Cross-Attention at each layer)

Class structure:
```python
class PromptGFM(nn.Module):
    def __init__(self,
                 gnn_config,
                 prompt_encoder_config,
                 conditioning_type='film',  # or 'cross_attn', 'none'
                 num_gnn_layers=3,
                 hidden_dim=512,
                 output_dim=256):
        
        self.prompt_encoder = PromptEncoder(**prompt_encoder_config)
        self.gnn_layers = nn.ModuleList([
            GNNLayer + Conditioning for _ in range(num_gnn_layers)
        ])
        self.gene_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graph, disease_prompt_text, candidate_gene_indices):
        # 1. Encode prompt
        prompt_emb = self.prompt_encoder(disease_prompt_text)
        
        # 2. GNN message passing with conditioning
        node_embs = graph.x
        for layer, conditioning in zip(self.gnn_layers, self.conditionings):
            node_embs = layer(node_embs, graph.edge_index)
            node_embs = conditioning(node_embs, prompt_emb)
            
        # 3. Extract gene embeddings
        gene_embs = node_embs[candidate_gene_indices]
        
        # 4. Compute scores (dot product with prompt or MLP)
        scores = self.score_genes(gene_embs, prompt_emb)
        return scores
```

Add support for batch processing multiple diseases in parallel.
Implement efficient subgraph sampling for large graphs.
```

---

## Phase 4: Training Pipeline

### 4.1 Self-Supervised Pretraining (`src/training/pretrain.py`)

**Copilot Prompt:**
```
Implement self-supervised pretraining tasks for the graph backbone:

Three pretraining objectives:

1. Masked Node Prediction:
   - Randomly mask 15% of gene nodes
   - Predict masked node features from neighborhood
   - Loss: MSE or contrastive loss

2. Edge Contrastive Learning:
   - Sample positive pairs (connected nodes in PPI)
   - Sample negative pairs (random unconnected nodes)
   - Maximize agreement for positives, minimize for negatives
   - Use InfoNCE loss

3. Context Prediction:
   - Predict whether two nodes are within k-hop neighborhood
   - Binary classification task

Training setup:
```python
class GraphPretrainer:
    def __init__(self, model, graph, device='cuda'):
        self.model = model
        self.graph = graph
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
    def masked_node_task(self, mask_ratio=0.15):
        # Mask random nodes
        # Predict features
        # Return loss
        
    def edge_contrastive_task(self, num_negatives=5):
        # Sample positive/negative pairs
        # Compute embeddings
        # InfoNCE loss
        
    def train_epoch(self):
        # Combine all objectives
        # Return total loss
```

Use learning rate warmup and cosine annealing schedule.
Save checkpoints every 10 epochs.
Log to Weights & Biases.
```

### 4.2 Supervised Finetuning (`src/training/finetune.py`)

**Copilot Prompt:**
```
Implement supervised training for gene-disease link prediction:

Loss functions:
1. Binary Cross-Entropy for positive/negative gene pairs
2. Ranking loss (margin-based): push positive genes higher than negatives
3. ListNet/ListMLE for ranking quality

Training procedure:
```python
class PromptGFMTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
    def train_step(self, batch):
        # batch = (disease_prompt, positive_genes, negative_genes)
        # Forward pass
        # Compute ranking loss
        # Backward + optimize
        
    def validate(self):
        # Compute AUROC, AUPR, MAP on validation set
        # Return metrics dict
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            # Log metrics
            # Save best model
```

Implement early stopping based on validation AUPR.
Support gradient accumulation for large batch sizes.
Add gradient clipping to prevent exploding gradients.
```

### 4.3 Loss Functions (`src/training/losses.py`)

**Copilot Prompt:**
```
Implement specialized loss functions for gene ranking:

1. Margin Ranking Loss:
```python
def margin_ranking_loss(pos_scores, neg_scores, margin=0.5):
    """
    Ensure positive scores > negative scores + margin
    """
    loss = torch.clamp(margin + neg_scores - pos_scores, min=0)
    return loss.mean()
```

2. ListNet Loss (ranking loss based on probability distributions):
```python
def listnet_loss(pred_scores, true_relevance):
    """
    pred_scores: [batch, num_genes]
    true_relevance: [batch, num_genes] (1 for positive, 0 for negative)
    """
    # Compute softmax over scores
    # Compute softmax over true relevance
    # KL divergence between distributions
```

3. Contrastive Loss for prompt-gene alignment:
```python
def prompt_gene_contrastive_loss(gene_embs, prompt_emb, temperature=0.07):
    """
    InfoNCE-style loss: pull positive genes close to prompt,
    push negatives away
    """
```

Add loss weighting for rare diseases (upweight diseases with <5 genes).
```

---

## Phase 5: Evaluation & Baselines

### 5.1 Metrics Implementation (`src/evaluation/metrics.py`)

**Copilot Prompt:**
```
Implement comprehensive evaluation metrics for gene ranking:

Required metrics:
1. AUROC (Area Under ROC Curve)
2. AUPR (Area Under Precision-Recall Curve) - critical for imbalanced data
3. Precision@K (K=10, 20, 50)
4. Mean Average Precision (MAP)
5. Mean Reciprocal Rank (MRR)
6. Normalized Discounted Cumulative Gain (NDCG@K)

Implementation:
```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class GeneRankingEvaluator:
    def __init__(self):
        pass
        
    def compute_auroc(self, y_true, y_scores):
        """y_true: binary labels, y_scores: predicted scores"""
        return roc_auc_score(y_true, y_scores)
        
    def compute_aupr(self, y_true, y_scores):
        return average_precision_score(y_true, y_scores)
        
    def precision_at_k(self, y_true, y_scores, k=20):
        # Sort by scores descending
        # Check top-k for true positives
        # Return precision
        
    def mean_average_precision(self, y_true, y_scores):
        # Compute AP for each disease
        # Average across diseases
        
    def evaluate_all(self, predictions_dict):
        # predictions_dict: {disease_id: (y_true, y_scores)}
        # Compute all metrics
        # Return comprehensive results
```

Add stratified evaluation by disease rarity (ultra-rare vs common rare).
Support few-shot evaluation (metrics computed only on query genes).
```

### 5.2 Baseline Models (`scripts/train_baseline.py`)

**Copilot Prompt:**
```
Implement three baseline models for comparison:

1. GNN-Only Baseline (no text):
```python
class GNNBaseline(nn.Module):
    def __init__(self, gnn_config):
        self.gnn = GNNBackbone(**gnn_config)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, graph, disease_idx, candidate_genes):
        # Run GNN
        # Use disease node embedding to rank genes
        # No prompt information used
```

2. Static Text Concatenation Baseline:
```python
class StaticTextGNN(nn.Module):
    def __init__(self, gnn_config, prompt_encoder):
        self.prompt_encoder = prompt_encoder
        self.gnn = GNNBackbone(**gnn_config)
        
    def forward(self, graph, disease_prompt, candidate_genes):
        # Encode prompt ONCE
        # Concatenate to all gene node features
        # Run GNN on augmented features
        # No dynamic conditioning
```

3. Text-Only Baseline (no graph):
```python
class TextOnlyBaseline(nn.Module):
    def __init__(self, prompt_encoder):
        self.prompt_encoder = prompt_encoder
        self.gene_encoder = BioBERT()  # encode gene descriptions
        
    def forward(self, disease_prompt, gene_descriptions):
        # Encode disease prompt
        # Encode gene descriptions
        # Compute similarity scores (dot product)
        # Rank genes by similarity to disease
```

Create unified training script that trains all baselines + PromptGFM with same data splits.
```

### 5.3 Angelman Syndrome Case Study (`src/evaluation/case_study.py`)

**Copilot Prompt:**
```
Create detailed case study analysis for Angelman Syndrome validation:

Known information:
- Primary gene: UBE3A
- Pathway genes: MAPK1, PRMT5, CDK1, CDK4
- Negative controls: MECP2 (Rett), ZEB2 (Mowat-Wilson), TCF4 (Pitt-Hopkins)

Implementation:
```python
class AngelmanCaseStudy:
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.known_gene = 'UBE3A'
        self.pathway_genes = ['MAPK1', 'PRMT5', 'CDK1', 'CDK4']
        self.negative_controls = ['MECP2', 'ZEB2', 'TCF4']
        
    def run_case_study(self):
        # Create Angelman prompt from HPO phenotypes
        prompt = """
        Disease: Angelman Syndrome
        Phenotypes: Severe developmental delay, absent speech, 
        ataxia, seizures, happy demeanor, microcephaly
        """
        
        # Get predictions for all genes
        scores = self.model.predict(prompt, all_genes)
        
        # Analyze rankings
        ube3a_rank = self.get_rank(scores, 'UBE3A')
        pathway_ranks = [self.get_rank(scores, g) for g in self.pathway_genes]
        negative_ranks = [self.get_rank(scores, g) for g in self.negative_controls]
        
        # Get top 50 predicted genes
        top_genes = self.get_top_k(scores, k=50)
        
        # Validate against literature
        self.literature_validation(top_genes)
        
        # Visualize results
        self.plot_ranking_comparison()
```

Generate visualization:
- Rank distribution plot (UBE3A vs pathway vs negatives)
- Attention weights visualization (if using cross-attention)
- Subgraph around top predicted genes
- Comparison with baseline models

Export results as PDF report with interpretable explanations.
```

---

## Phase 6: Experiment Configuration

### 6.1 Base Configuration (`configs/base_config.yaml`)

**Copilot Prompt:**
```
Create YAML configuration file for hyperparameters:

```yaml
# configs/base_config.yaml
data:
  graph_path: 'data/processed/biomedical_graph.pt'
  splits_path: 'data/splits/'
  rare_disease_threshold: 5  # diseases with <5 genes
  few_shot_k: [1, 3, 5]
  negative_sample_ratio: 10  # 10 negatives per positive

model:
  gnn:
    type: 'graphsage'  # or 'gat', 'gin'
    num_layers: 3
    hidden_dim: 512
    dropout: 0.2
    use_residual: true
    
  prompt_encoder:
    model_name: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    pooling: 'cls'
    max_length: 128
    freeze_bert: false  # finetune BERT or freeze
    
  conditioning:
    type: 'film'  # or 'cross_attn', 'none'
    num_heads: 8  # for cross-attention
    
  output_dim: 256

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 100
  early_stopping_patience: 10
  gradient_clip: 1.0
  
  optimizer: 'adamw'
  scheduler: 'cosine'
  warmup_epochs: 5
  
  loss_weights:
    ranking_loss: 1.0
    contrastive_loss: 0.5

evaluation:
  metrics: ['auroc', 'aupr', 'precision@k', 'map', 'mrr']
  k_values: [10, 20, 50]
  stratify_by_rarity: true

logging:
  use_wandb: true
  wandb_project: 'promptgfm-bio'
  log_interval: 10  # log every 10 batches
  save_checkpoint_every: 5  # epochs

hardware:
  device: 'cuda'
  num_workers: 4
  pin_memory: true
```

Create separate configs for pretraining and finetuning stages.
```

---

## Phase 7: Main Training Scripts

### 7.1 Main Experiment Runner (`scripts/train_promptgfm.py`)

**Copilot Prompt:**
```
Create end-to-end training script that orchestrates the full pipeline:

```python
import yaml
import torch
from src.data.dataset import GeneDiseaseDataset
from src.models.promptgfm import PromptGFM
from src.training.finetune import PromptGFMTrainer
from src.evaluation.metrics import GeneRankingEvaluator
import wandb

def main():
    # 1. Load configuration
    with open('configs/base_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 2. Initialize wandb
    wandb.init(project=config['logging']['wandb_project'], config=config)
    
    # 3. Load data
    dataset = GeneDiseaseDataset(config['data']['graph_path'])
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config['training']['batch_size'],
        split_by_rarity=True
    )
    
    # 4. Initialize model
    model = PromptGFM(
        gnn_config=config['model']['gnn'],
        prompt_encoder_config=config['model']['prompt_encoder'],
        conditioning_type=config['model']['conditioning']['type']
    )
    
    # 5. Load pretrained weights if available
    if config.get('pretrained_checkpoint'):
        model.load_state_dict(torch.load(config['pretrained_checkpoint']))
    
    # 6. Train
    trainer = PromptGFMTrainer(model, train_loader, val_loader, config)
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # 7. Evaluate on test set
    evaluator = GeneRankingEvaluator()
    test_metrics = evaluator.evaluate_all(model, test_loader)
    
    # 8. Run case study
    from src.evaluation.case_study import AngelmanCaseStudy
    case_study = AngelmanCaseStudy(model, dataset.graph)
    case_study.run_case_study()
    
    # 9. Save final model
    torch.save(model.state_dict(), 'checkpoints/promptgfm_final.pt')
    
    print("Training complete!")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test AUPR: {test_metrics['aupr']:.4f}")

if __name__ == '__main__':
    main()
```

Add command-line arguments for config overrides.
Support resuming from checkpoints.
Include comprehensive error handling.
```

---

## Phase 8: Testing & Validation

### 8.1 Unit Tests (`tests/test_models.py`)

**Copilot Prompt:**
```
Create unit tests for all model components:

```python
import pytest
import torch
from src.models.gnn_backbone import GNNBackbone
from src.models.prompt_encoder import PromptEncoder
from src.models.conditioning import FiLMConditioning, CrossAttentionConditioning
from src.models.promptgfm import PromptGFM

def test_gnn_backbone():
    model = GNNBackbone(input_dim=128, hidden_dim=256, num_layers=3)
    x = torch.randn(100, 128)  # 100 nodes, 128 features
    edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
    
    out = model(x, edge_index)
    assert out.shape == (100, 256)  # Check output shape
    
def test_prompt_encoder():
    encoder = PromptEncoder()
    prompts = ["Disease: Angelman Syndrome. Phenotypes: developmental delay, seizures"]
    
    embeddings = encoder(prompts)
    assert embeddings.shape == (1, 768)  # BERT-base dimension
    
def test_film_conditioning():
    film = FiLMConditioning(node_dim=256, prompt_dim=768)
    node_features = torch.randn(100, 256)
    prompt_emb = torch.randn(1, 768)
    
    conditioned = film(node_features, prompt_emb)
    assert conditioned.shape == node_features.shape
    
def test_full_model_forward():
    model = PromptGFM(...)
    # Create dummy inputs
    # Test forward pass
    # Check output shapes and values
```

Add integration tests for full training loop.
Test with toy datasets for quick validation.
```

---

## Quick Start Commands

Once you have this setup, run:

```bash
# 1. Setup environment
conda create -n promptgfm python=3.10
conda activate promptgfm
pip install -r requirements.txt

# 2. Download data
python scripts/download_data.sh

# 3. Preprocess graphs
python scripts/preprocess_all.py

# 4. Train baseline
python scripts/train_baseline.py --model gnn_only

# 5. Train PromptGFM with FiLM
python scripts/train_promptgfm.py --config configs/base_config.yaml --conditioning film

# 6. Evaluate on rare diseases
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test_few_shot

# 7. Run Angelman case study
python scripts/run_case_study.py --disease angelman
```

---

## Key Implementation Tips for GitHub Copilot

When using Copilot in VS Code with Claude Sonnet 4.5:

1. **Start each file with detailed docstrings** describing the module's purpose
2. **Write function signatures first** with type hints - Copilot will auto-complete implementations
3. **Use descriptive variable names** - helps Copilot understand context
4. **Add inline comments** explaining tricky logic before writing code
5. **Test incrementally** - write tests alongside implementations
6. **Use the chat interface** to ask for explanations or alternative implementations

**Example Copilot workflow:**
```python
# File: src/models/conditioning.py
"""
Prompt conditioning mechanisms for PromptGFM-Bio.

Implements FiLM and Cross-Attention conditioning to dynamically
integrate disease prompts into GNN message passing.
"""

import torch
import torch.nn as nn

class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation conditioning.
    
    Takes a prompt embedding and generates scale (gamma) and shift (beta)
    parameters to modulate GNN node features.
    
    Args:
        node_dim: Dimension of GNN node embeddings
        prompt_dim: Dimension of prompt embeddings (768 for BERT-base)
    """
    
    def __init__(self, node_dim: int, prompt_dim: int):
        super().__init__()
        # [Copilot will suggest the linear layers here]
```

---

## Monitoring & Debugging

Use Weights & Biases for experiment tracking:
```python
# In training loop
wandb.log({
    'train_loss': loss.item(),
    'learning_rate': scheduler.get_last_lr()[0],
    'epoch': epoch
})

# Log model architecture
wandb.watch(model, log='all', log_freq=100)
```

---

## Expected Timeline Checkpoints

Week 2: Data pipeline complete, graph constructed ✓  
Week 4: GNN backbone + baselines working ✓  
Week 6: FiLM conditioning implemented ✓  
Week 8: Pretraining complete, initial finetuning results ✓  
Week 10: Cross-attention added, ablations run ✓  
Week 12: Case study done, paper drafted ✓

---

This prompt provides comprehensive scaffolding for your entire project. Copy relevant sections to Copilot chat as you work through each phase!
