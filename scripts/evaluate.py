"""
Evaluation script for PromptGFM-Bio.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset
from src.models.promptgfm import PromptGFM, GNNOnlyBaseline
from src.evaluation.metrics import GeneRankingEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path, config_path, device='cuda'):
    """Load model and data from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})
    
    # Load model
    if config['model'].get('baseline', False):
        model = GNNOnlyBaseline(**config['model'])
    else:
        model = PromptGFM(**config['model'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    graph_dataset = BiomedicalGraphDataset(config['data']['graph_file'])
    dataset = GeneDiseaseDataset(
        graph_dataset=graph_dataset,
        edge_file=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3)
    )
    
    return model, dataset, graph_dataset, config


def evaluate_model(model, dataset, graph_dataset, split='test', device='cuda'):
    """Evaluate model on specified split."""
    logger.info(f"\nEvaluating on {split} split...")
    
    # Get edges for split
    if split == 'test':
        edges = dataset.test_edges
    elif split == 'val':
        edges = dataset.val_edges
    else:
        edges = dataset.train_edges
    
    logger.info(f"  Number of edges: {len(edges)}")
    
    # Prepare graph data
    graph = graph_dataset.graph
    node_features = graph.x_dict['gene']  # Assuming gene nodes
    edge_index = graph.edge_index_dict[('gene', 'interacts', 'gene')]
    
    # Move to device
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    # Collect predictions
    all_scores = []
    all_labels = []
    
    # Sample negative edges
    neg_edges = dataset.sample_negative_edges(
        len(edges),
        exclude_edges=dataset.get_all_edges()
    )
    
    # Combine positive and negative
    all_test_edges = edges + neg_edges
    all_test_labels = [1] * len(edges) + [0] * len(neg_edges)
    
    logger.info(f"  Positive: {len(edges)}, Negative: {len(neg_edges)}")
    
    # Predict in batches
    batch_size = 256
    with torch.no_grad():
        for i in tqdm(range(0, len(all_test_edges), batch_size), desc="Predicting"):
            batch_edges = all_test_edges[i:i + batch_size]
            
            # Get gene indices
            gene_indices = torch.tensor(
                [edge[0] for edge in batch_edges],
                dtype=torch.long,
                device=device
            )
            
            # Create disease prompts
            disease_prompts = []
            for edge in batch_edges:
                disease_id = edge[1]
                disease_name = dataset.disease_id_to_name.get(disease_id, f"Disease_{disease_id}")
                prompt = f"Disease: {disease_name}. Associated genes:"
                disease_prompts.append(prompt)
            
            # Forward pass
            if hasattr(model, 'forward'):
                scores = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    disease_texts=disease_prompts,
                    gene_indices=gene_indices
                )
            else:
                # Baseline without prompts
                scores = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    gene_indices=gene_indices
                )
            
            scores = scores.squeeze(-1).cpu().numpy()
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_test_labels)
    
    # Compute metrics
    evaluator = GeneRankingEvaluator(k_values=[10, 20, 50, 100])
    metrics = evaluator.evaluate_all(all_labels, all_scores)
    
    return metrics, all_scores, all_labels


def evaluate_stratified(model, dataset, graph_dataset, device='cuda'):
    """Evaluate model stratified by disease rarity."""
    logger.info("\n" + "="*60)
    logger.info("Stratified Evaluation by Disease Rarity")
    logger.info("="*60)
    
    # Split by rarity
    rarity_splits = dataset.split_by_rarity()
    
    results = {}
    for rarity, edges in rarity_splits.items():
        logger.info(f"\nRarity: {rarity} ({len(edges)} edges)")
        
        # Create temporary dataset with only these edges
        temp_dataset = dataset
        temp_dataset.test_edges = edges
        
        # Evaluate
        metrics, _, _ = evaluate_model(
            model, temp_dataset, graph_dataset, split='test', device=device
        )
        
        results[rarity] = metrics
        
        # Print metrics
        evaluator = GeneRankingEvaluator()
        evaluator.print_metrics(metrics, prefix=f"{rarity}:")
    
    return results


def evaluate_few_shot(model, dataset, graph_dataset, k=5, device='cuda'):
    """Evaluate model on few-shot learning."""
    logger.info("\n" + "="*60)
    logger.info(f"Few-Shot Evaluation (K={k})")
    logger.info("="*60)
    
    # Create few-shot split
    support_set, query_set = dataset.create_few_shot_split(k_shot=k)
    
    logger.info(f"  Support set size: {len(support_set)}")
    logger.info(f"  Query set size: {len(query_set)}")
    
    # For simplicity, evaluate on query set
    # In a full implementation, you'd fine-tune on support set first
    temp_dataset = dataset
    temp_dataset.test_edges = query_set
    
    metrics, _, _ = evaluate_model(
        model, temp_dataset, graph_dataset, split='test', device=device
    )
    
    evaluator = GeneRankingEvaluator()
    evaluator.print_metrics(metrics, prefix=f"Few-Shot (K={k}):")
    
    return metrics


def save_results(results, output_path):
    """Save evaluation results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional, will try to load from checkpoint)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='test',
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Run stratified evaluation by disease rarity'
    )
    parser.add_argument(
        '--few-shot',
        type=int,
        nargs='+',
        help='Run few-shot evaluation with specified K values'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_results.json',
        help='Path to save results'
    )
    
    args = parser.parse_args()
    
    # Load model and data
    model, dataset, graph_dataset, config = load_model_and_data(
        args.checkpoint,
        args.config,
        device=args.device
    )
    
    all_results = {}
    
    # Standard evaluation
    if args.split == 'all':
        for split in ['train', 'val', 'test']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {split.upper()} split")
            logger.info(f"{'='*60}")
            
            metrics, _, _ = evaluate_model(
                model, dataset, graph_dataset, split=split, device=args.device
            )
            
            evaluator = GeneRankingEvaluator()
            evaluator.print_metrics(metrics, prefix=f"{split}:")
            
            all_results[split] = metrics
    else:
        metrics, _, _ = evaluate_model(
            model, dataset, graph_dataset, split=args.split, device=args.device
        )
        
        evaluator = GeneRankingEvaluator()
        evaluator.print_metrics(metrics, prefix=f"{args.split}:")
        
        all_results[args.split] = metrics
    
    # Stratified evaluation
    if args.stratified:
        stratified_results = evaluate_stratified(
            model, dataset, graph_dataset, device=args.device
        )
        all_results['stratified'] = stratified_results
    
    # Few-shot evaluation
    if args.few_shot:
        few_shot_results = {}
        for k in args.few_shot:
            metrics = evaluate_few_shot(
                model, dataset, graph_dataset, k=k, device=args.device
            )
            few_shot_results[f'k={k}'] = metrics
        all_results['few_shot'] = few_shot_results
    
    # Save results
    save_results(all_results, args.output)
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
