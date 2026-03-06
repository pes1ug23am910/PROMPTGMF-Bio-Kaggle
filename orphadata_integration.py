"""
Orphadata Integration Guide
============================

Orphadata provides authoritative rare disease gene associations.
This module handles download, parsing, and integration into the graph.

Orphadata Files:
- en_product6.xml: Gene-disease associations
- en_product1.xml: Rare disease classifications
- en_product4.xml: Disease prevalence data
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Download Orphadata
# ============================================================================

def download_orphadata(output_dir: str = 'data/raw/orphanet'):
    """
    Download Orphadata XML files.
    
    Files downloaded:
    1. en_product6.xml - Gene-disease associations (PRIMARY)
    2. en_product1.xml - Disease classifications
    3. en_product4.xml - Prevalence data
    
    Args:
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_to_download = {
        'en_product6.xml': 'http://www.orphadata.com/data/xml/en_product6.xml',  # Genes
        'en_product1.xml': 'http://www.orphadata.com/data/xml/en_product1.xml',  # Classifications
        'en_product4.xml': 'http://www.orphadata.com/data/xml/en_product4.xml',  # Prevalence
    }
    
    for filename, url in files_to_download.items():
        output_file = output_path / filename
        
        if output_file.exists():
            logger.info(f"✓ {filename} already exists, skipping")
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✓ Downloaded {filename} ({len(response.content) / 1024 / 1024:.1f} MB)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
    
    logger.info(f"\nOrphadata files saved to {output_dir}")


# ============================================================================
# Parse Orphadata XML
# ============================================================================

def parse_orphadata_gene_associations(xml_path: str) -> pd.DataFrame:
    """
    Parse en_product6.xml to extract gene-disease associations.
    
    XML structure (simplified):
    <JDBOR>
        <DisorderList>
            <Disorder id="...">
                <OrphaCode>ORPHA:166024</OrphaCode>
                <Name>Angelman syndrome</Name>
                <DisorderGeneAssociationList>
                    <DisorderGeneAssociation>
                        <Gene>
                            <Symbol>UBE3A</Symbol>
                            <ExternalReferenceList>
                                <ExternalReference>
                                    <Source>HGNC</Source>
                                    <Reference>12496</Reference>
                                </ExternalReference>
                            </ExternalReferenceList>
                        </Gene>
                        <DisorderGeneAssociationType>
                            <Name>Disease-causing germline mutation(s) in</Name>
                        </DisorderGeneAssociationType>
                        <DisorderGeneAssociationStatus>
                            <Name>Assessed</Name>
                        </DisorderGeneAssociationStatus>
                    </DisorderGeneAssociation>
                </DisorderGeneAssociationList>
            </Disorder>
        </DisorderList>
    </JDBOR>
    
    Returns:
        DataFrame with columns: [orpha_code, disease_name, gene_symbol, hgnc_id, 
                                 association_type, association_status]
    """
    logger.info(f"Parsing Orphadata gene associations from {xml_path}...")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    associations = []
    
    # Find all disorders
    for disorder in root.findall('.//Disorder'):
        # Extract disease info
        orpha_code = disorder.find('OrphaCode').text if disorder.find('OrphaCode') is not None else None
        disease_name = disorder.find('Name').text if disorder.find('Name') is not None else None
        
        # Extract gene associations
        gene_assoc_list = disorder.find('DisorderGeneAssociationList')
        if gene_assoc_list is None:
            continue
        
        for gene_assoc in gene_assoc_list.findall('DisorderGeneAssociation'):
            # Gene info
            gene = gene_assoc.find('Gene')
            if gene is None:
                continue
            
            gene_symbol = gene.find('Symbol').text if gene.find('Symbol') is not None else None
            
            # Extract HGNC ID
            hgnc_id = None
            ext_refs = gene.find('ExternalReferenceList')
            if ext_refs is not None:
                for ext_ref in ext_refs.findall('ExternalReference'):
                    source = ext_ref.find('Source')
                    if source is not None and source.text == 'HGNC':
                        hgnc_id = ext_ref.find('Reference').text
                        break
            
            # Association type
            assoc_type_elem = gene_assoc.find('.//DisorderGeneAssociationType/Name')
            assoc_type = assoc_type_elem.text if assoc_type_elem is not None else None
            
            # Association status
            assoc_status_elem = gene_assoc.find('.//DisorderGeneAssociationStatus/Name')
            assoc_status = assoc_status_elem.text if assoc_status_elem is not None else None
            
            associations.append({
                'orpha_code': orpha_code,
                'disease_name': disease_name,
                'gene_symbol': gene_symbol,
                'hgnc_id': hgnc_id,
                'association_type': assoc_type,
                'association_status': assoc_status
            })
    
    df = pd.DataFrame(associations)
    logger.info(f"Extracted {len(df)} gene-disease associations from Orphadata")
    
    return df


def filter_high_confidence_orphadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to high-confidence gene-disease associations.
    
    Criteria:
    - Association status = 'Assessed' or 'Validated'
    - Association type includes 'Disease-causing' or 'Major'
    
    Args:
        df: DataFrame from parse_orphadata_gene_associations
        
    Returns:
        Filtered DataFrame
    """
    # Filter by status
    valid_statuses = ['Assessed', 'Validated']
    df_filtered = df[df['association_status'].isin(valid_statuses)].copy()
    
    # Filter by type (disease-causing mutations)
    disease_causing_keywords = ['Disease-causing', 'disease-causing', 'Major', 'major']
    df_filtered = df_filtered[
        df_filtered['association_type'].str.contains('|'.join(disease_causing_keywords), na=False)
    ]
    
    logger.info(f"Filtered to {len(df_filtered)} high-confidence associations")
    
    return df_filtered


def parse_orphadata_classifications(xml_path: str) -> pd.DataFrame:
    """
    Parse en_product1.xml to get disease classifications and prevalence info.
    
    Returns:
        DataFrame with disease metadata (orpha_code, name, group, prevalence)
    """
    logger.info(f"Parsing Orphadata classifications from {xml_path}...")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    diseases = []
    
    for disorder in root.findall('.//Disorder'):
        orpha_code = disorder.find('OrphaCode').text if disorder.find('OrphaCode') is not None else None
        disease_name = disorder.find('Name').text if disorder.find('Name') is not None else None
        
        # Disease group/type
        disorder_type = disorder.find('.//DisorderType/Name')
        disorder_type_name = disorder_type.text if disorder_type is not None else None
        
        diseases.append({
            'orpha_code': orpha_code,
            'disease_name': disease_name,
            'disorder_type': disorder_type_name
        })
    
    df = pd.DataFrame(diseases)
    logger.info(f"Extracted {len(df)} disease classifications")
    
    return df


# ============================================================================
# Integration Functions
# ============================================================================

def merge_orphadata_with_hpo(orphadata_df: pd.DataFrame,
                             hpo_edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Orphadata and HPO-derived gene-disease edges.
    
    Strategy:
    1. Use Orphadata as gold standard (score = 1.0, provenance = 'Orphadata')
    2. Add HPO edges that are NOT in Orphadata (provenance = 'HPO_bridge')
    3. Tag overlapping edges (provenance = 'Orphadata+HPO')
    
    Args:
        orphadata_df: From parse_orphadata_gene_associations
        hpo_edges_df: From HPOGeneDiseaseBuilder
        
    Returns:
        Merged DataFrame with all edges
    """
    # Prepare Orphadata edges
    orpha_edges = orphadata_df[['gene_symbol', 'orpha_code', 'disease_name']].copy()
    orpha_edges.rename(columns={'gene_symbol': 'gene', 'orpha_code': 'disease'}, inplace=True)
    orpha_edges['score'] = 1.0  # Gold standard
    orpha_edges['provenance'] = 'Orphadata'
    orpha_edges['supporting_phenotypes'] = ''
    
    # Create lookup set for Orphadata pairs
    orpha_pairs = set(zip(orpha_edges['gene'], orpha_edges['disease']))
    
    # Tag HPO edges
    hpo_edges_df['in_orphadata'] = hpo_edges_df.apply(
        lambda row: (row['gene'], row['disease']) in orpha_pairs, axis=1
    )
    
    # Separate HPO edges
    hpo_only = hpo_edges_df[~hpo_edges_df['in_orphadata']].copy()
    hpo_overlap = hpo_edges_df[hpo_edges_df['in_orphadata']].copy()
    
    # Update provenance for overlapping edges in Orphadata
    for idx, row in hpo_overlap.iterrows():
        orpha_edges.loc[
            (orpha_edges['gene'] == row['gene']) & (orpha_edges['disease'] == row['disease']),
            'provenance'
        ] = 'Orphadata+HPO'
        
        # Add supporting phenotypes to Orphadata edge
        orpha_edges.loc[
            (orpha_edges['gene'] == row['gene']) & (orpha_edges['disease'] == row['disease']),
            'supporting_phenotypes'
        ] = row['supporting_phenotypes']
    
    # Combine all edges
    merged = pd.concat([orpha_edges, hpo_only], ignore_index=True)
    
    logger.info(f"Merged edges: {len(orpha_edges)} Orphadata + {len(hpo_only)} HPO-only = {len(merged)} total")
    logger.info(f"Overlap: {len(hpo_overlap)} edges confirmed by both sources")
    
    return merged


def create_orphadata_rare_disease_splits(merged_edges_df: pd.DataFrame,
                                         orphadata_classifications_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits stratified by disease rarity.
    
    Uses Orphadata gene counts to define rarity:
    - Ultra-rare: 1-2 genes
    - Very rare: 3-5 genes
    - Moderately rare: 6-10 genes
    - Common rare: 10+ genes
    
    Returns:
        Dict with keys: ['ultra_rare', 'very_rare', 'moderately_rare', 'common_rare']
    """
    # Count genes per disease
    gene_counts = merged_edges_df.groupby('disease').size().reset_index(name='num_genes')
    
    # Categorize
    gene_counts['rarity'] = pd.cut(
        gene_counts['num_genes'],
        bins=[0, 2, 5, 10, float('inf')],
        labels=['ultra_rare', 'very_rare', 'moderately_rare', 'common_rare']
    )
    
    # Split edges by rarity
    splits = {}
    for rarity in ['ultra_rare', 'very_rare', 'moderately_rare', 'common_rare']:
        diseases_in_category = gene_counts[gene_counts['rarity'] == rarity]['disease'].tolist()
        splits[rarity] = merged_edges_df[merged_edges_df['disease'].isin(diseases_in_category)]
        
        logger.info(f"{rarity}: {len(diseases_in_category)} diseases, {len(splits[rarity])} edges")
    
    return splits


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Complete Orphadata integration pipeline"""
    
    # Step 1: Download
    download_orphadata()
    
    # Step 2: Parse gene associations
    orphadata_genes = parse_orphadata_gene_associations('data/raw/orphanet/en_product6.xml')
    
    # Step 3: Filter to high confidence
    orphadata_filtered = filter_high_confidence_orphadata(orphadata_genes)
    
    # Step 4: Parse classifications
    orphadata_classes = parse_orphadata_classifications('data/raw/orphanet/en_product1.xml')
    
    # Step 5: Load HPO-derived edges (from previous step)
    hpo_edges = pd.read_csv('data/processed/hpo_gene_disease_edges.csv')
    
    # Step 6: Merge
    merged_edges = merge_orphadata_with_hpo(orphadata_filtered, hpo_edges)
    
    # Step 7: Create rarity splits
    splits = create_orphadata_rare_disease_splits(merged_edges, orphadata_classes)
    
    # Step 8: Save
    merged_edges.to_csv('data/processed/merged_gene_disease_edges.csv', index=False)
    
    for rarity, edges_df in splits.items():
        edges_df.to_csv(f'data/processed/{rarity}_gene_disease_edges.csv', index=False)
    
    logger.info("\n✓ Orphadata integration complete!")
    logger.info(f"Total edges: {len(merged_edges)}")
    logger.info(f"Orphadata gold standard: {len(orphadata_filtered)}")
    logger.info(f"HPO-only edges: {len(merged_edges[merged_edges['provenance'] == 'HPO_bridge'])}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
