"""
HPO-Based Gene-Disease Bridge Implementation
==============================================

This module creates high-quality gene-disease associations by bridging
HPO gene→phenotype and phenotype→disease relationships.

Key features:
- Weighted phenotype scoring (IDF-based)
- Jaccard similarity with phenotype specificity
- Provenance tracking
- ID harmonization
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class HPOGeneDiseaseBuilder:
    """
    Build gene-disease associations from HPO phenotype bridge.
    
    Implements scoring methods:
    1. IDF-weighted phenotype overlap
    2. Weighted Jaccard similarity
    3. Provenance tracking for explainability
    """
    
    def __init__(self, min_score: float = 0.1, max_common_phenotype_freq: float = 0.5):
        """
        Args:
            min_score: Minimum edge score to include (0-1)
            max_common_phenotype_freq: Filter phenotypes appearing in >X of diseases
        """
        self.min_score = min_score
        self.max_common_phenotype_freq = max_common_phenotype_freq
        self.phenotype_idf = {}  # Inverse document frequency per phenotype
        
    def compute_phenotype_idf(self, 
                              disease_phenotypes: Dict[str, Set[str]],
                              gene_phenotypes: Dict[str, Set[str]]) -> None:
        """
        Compute IDF (inverse document frequency) for each phenotype.
        
        IDF = log(N / df) where:
        - N = total number of entities (diseases + genes)
        - df = number of entities with this phenotype
        
        Higher IDF = rarer, more specific phenotype
        """
        phenotype_counts = defaultdict(int)
        
        # Count phenotype occurrences across diseases
        for phenotypes in disease_phenotypes.values():
            for pheno in phenotypes:
                phenotype_counts[pheno] += 1
                
        # Count phenotype occurrences across genes
        for phenotypes in gene_phenotypes.values():
            for pheno in phenotypes:
                phenotype_counts[pheno] += 1
        
        total_entities = len(disease_phenotypes) + len(gene_phenotypes)
        
        # Compute IDF
        for pheno, count in phenotype_counts.items():
            self.phenotype_idf[pheno] = np.log(total_entities / count)
            
        logger.info(f"Computed IDF for {len(self.phenotype_idf)} phenotypes")
        
    def filter_common_phenotypes(self, 
                                 disease_phenotypes: Dict[str, Set[str]]) -> Set[str]:
        """
        Remove overly common phenotypes (e.g., 'fever', 'pain').
        
        Returns set of phenotypes to exclude.
        """
        total_diseases = len(disease_phenotypes)
        phenotype_freq = defaultdict(int)
        
        for phenotypes in disease_phenotypes.values():
            for pheno in phenotypes:
                phenotype_freq[pheno] += 1
        
        # Filter phenotypes appearing in >X% of diseases
        excluded = {
            pheno for pheno, count in phenotype_freq.items()
            if count / total_diseases > self.max_common_phenotype_freq
        }
        
        logger.info(f"Filtering {len(excluded)} overly common phenotypes")
        return excluded
    
    def weighted_phenotype_overlap_score(self,
                                         gene_phenotypes: Set[str],
                                         disease_phenotypes: Set[str],
                                         excluded_phenotypes: Set[str]) -> Tuple[float, List[str]]:
        """
        Score gene-disease association using IDF-weighted phenotype overlap.
        
        Score = sum_{pheno in intersection} IDF(pheno)
        Normalized by max possible score.
        
        Returns:
            score: Float in [0, 1]
            supporting_phenotypes: List of shared phenotypes
        """
        # Remove excluded phenotypes
        gene_pheno_filtered = gene_phenotypes - excluded_phenotypes
        disease_pheno_filtered = disease_phenotypes - excluded_phenotypes
        
        # Find intersection
        shared = gene_pheno_filtered & disease_pheno_filtered
        
        if not shared:
            return 0.0, []
        
        # Compute weighted score
        score = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in shared)
        
        # Normalize by maximum possible score (if all disease phenotypes matched)
        max_score = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in disease_pheno_filtered)
        
        if max_score > 0:
            normalized_score = score / max_score
        else:
            normalized_score = 0.0
            
        return normalized_score, list(shared)
    
    def weighted_jaccard_score(self,
                              gene_phenotypes: Set[str],
                              disease_phenotypes: Set[str],
                              excluded_phenotypes: Set[str]) -> Tuple[float, List[str]]:
        """
        Weighted Jaccard similarity.
        
        J = sum_{pheno in intersection} w(pheno) / sum_{pheno in union} w(pheno)
        where w(pheno) = IDF(pheno)
        """
        # Remove excluded
        gene_pheno_filtered = gene_phenotypes - excluded_phenotypes
        disease_pheno_filtered = disease_phenotypes - excluded_phenotypes
        
        shared = gene_pheno_filtered & disease_pheno_filtered
        union = gene_pheno_filtered | disease_pheno_filtered
        
        if not union:
            return 0.0, []
        
        # Weighted intersection
        weighted_intersection = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in shared)
        
        # Weighted union
        weighted_union = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in union)
        
        score = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
        
        return score, list(shared)
    
    def create_gene_disease_edges(self,
                                  gene_phenotypes_path: str,
                                  disease_phenotypes_path: str,
                                  scoring_method: str = 'weighted_overlap') -> pd.DataFrame:
        """
        Main pipeline: Create scored gene-disease edges from HPO.
        
        Args:
            gene_phenotypes_path: Path to genes_to_phenotype.txt
            disease_phenotypes_path: Path to phenotype.hpoa
            scoring_method: 'weighted_overlap' or 'weighted_jaccard'
            
        Returns:
            DataFrame with columns: [gene, disease, score, supporting_phenotypes, provenance]
        """
        # Step 1: Parse HPO files
        logger.info("Parsing HPO gene-phenotype annotations...")
        gene_to_phenotypes = self._parse_gene_phenotypes(gene_phenotypes_path)
        
        logger.info("Parsing HPO disease-phenotype annotations...")
        disease_to_phenotypes = self._parse_disease_phenotypes(disease_phenotypes_path)
        
        # Step 2: Compute IDF
        logger.info("Computing phenotype IDF scores...")
        self.compute_phenotype_idf(disease_to_phenotypes, gene_to_phenotypes)
        
        # Step 3: Filter common phenotypes
        excluded_phenotypes = self.filter_common_phenotypes(disease_to_phenotypes)
        
        # Step 4: Score all gene-disease pairs
        logger.info("Scoring gene-disease associations...")
        edges = []
        
        for gene, gene_phenos in gene_to_phenotypes.items():
            for disease, disease_phenos in disease_to_phenotypes.items():
                
                # Choose scoring method
                if scoring_method == 'weighted_overlap':
                    score, supporting = self.weighted_phenotype_overlap_score(
                        gene_phenos, disease_phenos, excluded_phenotypes
                    )
                elif scoring_method == 'weighted_jaccard':
                    score, supporting = self.weighted_jaccard_score(
                        gene_phenos, disease_phenos, excluded_phenotypes
                    )
                else:
                    raise ValueError(f"Unknown scoring method: {scoring_method}")
                
                # Filter by minimum score
                if score >= self.min_score:
                    edges.append({
                        'gene': gene,
                        'disease': disease,
                        'score': score,
                        'supporting_phenotypes': ';'.join(supporting),
                        'num_shared_phenotypes': len(supporting),
                        'provenance': 'HPO_phenotype_bridge'
                    })
        
        logger.info(f"Created {len(edges)} gene-disease edges (score >= {self.min_score})")
        
        return pd.DataFrame(edges)
    
    def _parse_gene_phenotypes(self, filepath: str) -> Dict[str, Set[str]]:
        """
        Parse genes_to_phenotype.txt
        
        Format:
        #gene_symbol  HPO_ID  ...
        TP53          HP:0001909
        """
        gene_to_phenotypes = defaultdict(set)
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                gene = parts[0]
                hpo_id = parts[1]
                
                gene_to_phenotypes[gene].add(hpo_id)
        
        return dict(gene_to_phenotypes)
    
    def _parse_disease_phenotypes(self, filepath: str) -> Dict[str, Set[str]]:
        """
        Parse phenotype.hpoa
        
        Format (tab-separated):
        database_id  disease_name  ...  HPO_ID  ...
        OMIM:154700  Marfan syndrome  ...  HP:0001166  ...
        """
        disease_to_phenotypes = defaultdict(set)
        
        with open(filepath, 'r') as f:
            header = f.readline()  # Skip header
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                
                # Extract disease ID and HPO term
                disease_id = parts[0]  # e.g., OMIM:154700
                hpo_id = parts[3]       # e.g., HP:0001166
                
                disease_to_phenotypes[disease_id].add(hpo_id)
        
        return dict(disease_to_phenotypes)


# ============================================================================
# Integration with existing preprocessing pipeline
# ============================================================================

def integrate_hpo_bridge_into_preprocess():
    """
    Example integration into src/data/preprocess.py
    """
    code_example = '''
# In src/data/preprocess.py, add new function:

from src.data.hpo_bridge import HPOGeneDiseaseBuilder

def create_gene_disease_from_hpo(
    gene_pheno_path: str,
    disease_pheno_path: str,
    output_path: str = 'data/processed/hpo_gene_disease_edges.csv',
    min_score: float = 0.1
) -> pd.DataFrame:
    """
    Create gene-disease associations from HPO phenotype bridge.
    
    Args:
        gene_pheno_path: Path to genes_to_phenotype.txt
        disease_pheno_path: Path to phenotype.hpoa
        output_path: Where to save edges
        min_score: Minimum edge score
        
    Returns:
        DataFrame of gene-disease edges with scores
    """
    builder = HPOGeneDiseaseBuilder(min_score=min_score)
    
    # Create edges with weighted overlap scoring
    edges_df = builder.create_gene_disease_edges(
        gene_pheno_path,
        disease_pheno_path,
        scoring_method='weighted_overlap'
    )
    
    # Save
    edges_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(edges_df)} HPO-derived gene-disease edges to {output_path}")
    
    return edges_df

# Then in preprocess_all(), use this instead of DisGeNET:

def preprocess_all():
    # ... existing BioGRID, STRING, HPO parsing ...
    
    # Create gene-disease edges from HPO bridge
    hpo_gene_disease = create_gene_disease_from_hpo(
        gene_pheno_path='data/raw/hpo/genes_to_phenotype.txt',
        disease_pheno_path='data/raw/hpo/phenotype.hpoa',
        min_score=0.1  # Tune this threshold
    )
    
    # Build graph with HPO-derived edges
    graph = build_heterogeneous_graph(
        ppi_edges=ppi_edges,
        gene_disease_edges=hpo_gene_disease,  # Use HPO bridge
        disease_phenotype_edges=disease_pheno_edges
    )
    '''
    return code_example


# ============================================================================
# Validation against Orphadata (once downloaded)
# ============================================================================

def validate_against_orphadata(hpo_edges_df: pd.DataFrame,
                               orphadata_path: str) -> Dict[str, float]:
    """
    Validate HPO-derived edges against Orphadata gold standard.
    
    Computes:
    - Precision: What fraction of HPO edges are in Orphadata?
    - Recall: What fraction of Orphadata edges are recovered?
    - F1 score
    
    Args:
        hpo_edges_df: DataFrame with 'gene' and 'disease' columns
        orphadata_path: Path to Orphadata gene-disease file
        
    Returns:
        Dict with precision, recall, F1
    """
    # Parse Orphadata (pseudo-code, adjust to actual format)
    orphadata_df = pd.read_csv(orphadata_path)  # Adjust parsing
    
    # Create sets of (gene, disease) tuples
    hpo_pairs = set(zip(hpo_edges_df['gene'], hpo_edges_df['disease']))
    orphadata_pairs = set(zip(orphadata_df['gene'], orphadata_df['disease']))
    
    # Compute metrics
    true_positives = len(hpo_pairs & orphadata_pairs)
    precision = true_positives / len(hpo_pairs) if hpo_pairs else 0
    recall = true_positives / len(orphadata_pairs) if orphadata_pairs else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hpo_edges': len(hpo_pairs),
        'orphadata_edges': len(orphadata_pairs),
        'overlap': true_positives
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    # Initialize builder
    builder = HPOGeneDiseaseBuilder(
        min_score=0.1,                    # Start with 0.1, tune based on results
        max_common_phenotype_freq=0.5     # Filter phenotypes in >50% of diseases
    )
    
    # Create edges
    edges_df = builder.create_gene_disease_edges(
        gene_phenotypes_path='data/raw/hpo/genes_to_phenotype.txt',
        disease_phenotypes_path='data/raw/hpo/phenotype.hpoa',
        scoring_method='weighted_overlap'
    )
    
    print(f"\nCreated {len(edges_df)} gene-disease edges")
    print(f"\nScore distribution:")
    print(edges_df['score'].describe())
    
    print(f"\nTop 10 edges:")
    print(edges_df.nlargest(10, 'score')[['gene', 'disease', 'score', 'num_shared_phenotypes']])
    
    # Save
    edges_df.to_csv('data/processed/hpo_gene_disease_edges.csv', index=False)
    print(f"\nSaved to data/processed/hpo_gene_disease_edges.csv")
    
    # If you have Orphadata, validate
    # metrics = validate_against_orphadata(edges_df, 'data/raw/orphanet/orphadata.csv')
    # print(f"\nValidation vs Orphadata: {metrics}")
