#!/usr/bin/env python3
"""
Standalone Ablation Therapeutic Candidate Ranking Script

This is a standalone version that doesn't require the full robin package.
It provides basic ranking functionality using simple scoring methods.
"""

import argparse
import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def extract_hypothesis_from_file(file_path: Path) -> str | None:
    """
    Extract the hypothesis (drug name) from a therapeutic candidate file.
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        The hypothesis text or None if not found
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        
        # Extract text between "Proposal for" and "Overview"
        candidate_match = re.search(
            r"Proposal for(.*?)\s*Overview", content, re.DOTALL | re.IGNORECASE
        )
        candidate_text = (
            candidate_match.group(1).strip() if candidate_match else None
        )
        
        return candidate_text
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None


def create_candidate_dataframe(directory_a: str, directory_b: str) -> pd.DataFrame:
    """
    Create a DataFrame with therapeutic candidates from both directories.
    
    Args:
        directory_a: Path to directory A
        directory_b: Path to directory B
        
    Returns:
        DataFrame with columns: ['unique_id', 'file_url', 'hypothesis', 'answer', 'index']
    """
    candidates_data = []
    index_counter = 0
    
    # Process directory A
    dir_a_path = Path(directory_a)
    if dir_a_path.is_dir():
        for txt_file in sorted(dir_a_path.glob("*.txt")):
            hypothesis = extract_hypothesis_from_file(txt_file)
            if hypothesis:
                unique_id = f"drug_hypothesis_{index_counter + 1}_A"
                file_url = str(txt_file.absolute())
                
                # Read full content for answer
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()
                
                candidates_data.append({
                    "unique_id": unique_id,
                    "file_url": file_url,
                    "hypothesis": hypothesis,
                    "answer": content,
                    "index": index_counter
                })
                index_counter += 1
                logger.info(f"Added from directory A: {unique_id}")
    
    # Process directory B
    dir_b_path = Path(directory_b)
    if dir_b_path.is_dir():
        for txt_file in sorted(dir_b_path.glob("*.txt")):
            hypothesis = extract_hypothesis_from_file(txt_file)
            if hypothesis:
                unique_id = f"drug_hypothesis_{index_counter + 1}_B"
                file_url = str(txt_file.absolute())
                
                # Read full content for answer
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()
                
                candidates_data.append({
                    "unique_id": unique_id,
                    "file_url": file_url,
                    "hypothesis": hypothesis,
                    "answer": content,
                    "index": index_counter
                })
                index_counter += 1
                logger.info(f"Added from directory B: {unique_id}")
    
    return pd.DataFrame(candidates_data)


def calculate_simple_scores(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple scores based on content analysis.
    
    This is a simplified scoring method that doesn't require LLM comparisons.
    It uses basic heuristics like content length, keyword presence, etc.
    
    Args:
        candidates_df: DataFrame with candidate information
        
    Returns:
        DataFrame with added strength scores and ranks
    """
    scores = []
    
    for _, row in candidates_df.iterrows():
        content = row['answer']
        hypothesis = row['hypothesis']
        
        # Simple scoring based on content characteristics
        score = 0.0
        
        # Base score for having content
        if content:
            score += 1.0
        
        # Length bonus (longer, more detailed content gets higher score)
        content_length = len(content)
        if content_length > 1000:
            score += 2.0
        elif content_length > 500:
            score += 1.0
        
        # Keyword bonuses for scientific terms
        scientific_keywords = [
            'mechanism', 'pathway', 'clinical', 'preclinical', 'efficacy',
            'safety', 'pharmacokinetic', 'pharmacodynamic', 'therapeutic',
            'dose', 'administration', 'trial', 'study', 'evidence'
        ]
        
        keyword_count = sum(1 for keyword in scientific_keywords if keyword.lower() in content.lower())
        score += keyword_count * 0.1
        
        # Reference bonus (more references = higher score)
        reference_count = content.count('http') + content.count('doi:') + content.count('PMID')
        score += reference_count * 0.2
        
        # Directory bonus (slight preference for directory A)
        if row['unique_id'].endswith('_A'):
            score += 0.1
        
        scores.append(score)
    
    candidates_df = candidates_df.copy()
    candidates_df['strength_score'] = scores
    candidates_df['rank'] = candidates_df['strength_score'].rank(ascending=False, method="dense").astype(int)
    
    return candidates_df.sort_values("strength_score", ascending=False).reset_index(drop=True)


def run_standalone_ablation_ranking(
    directory_a: str,
    directory_b: str,
    output_file: str | None = None
) -> pd.DataFrame:
    """
    Run the standalone ablation ranking process.
    
    Args:
        directory_a: Path to directory A containing therapeutic candidate files
        directory_b: Path to directory B containing therapeutic candidate files
        output_file: Optional output file path for results
        
    Returns:
        DataFrame with rankings
    """
    logger.info("Starting standalone ablation therapeutic candidate ranking...")
    
    # Create candidate DataFrame
    logger.info("Extracting candidates from directories...")
    candidates_df = create_candidate_dataframe(directory_a, directory_b)
    
    if candidates_df.empty:
        logger.error("No candidates found in either directory")
        return pd.DataFrame()
    
    logger.info(f"Found {len(candidates_df)} candidates total")
    
    # Calculate simple scores
    logger.info("Calculating scores...")
    results_df = calculate_simple_scores(candidates_df)
    
    # Save results if output file specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    return results_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare and rank therapeutic candidates from two directories (standalone version)"
    )
    parser.add_argument("directory_a", help="Path to directory A containing therapeutic candidate files")
    parser.add_argument("directory_b", help="Path to directory B containing therapeutic candidate files")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the ranking
    results = run_standalone_ablation_ranking(
        directory_a=args.directory_a,
        directory_b=args.directory_b,
        output_file=args.output
    )
    
    # Print results
    print("\n" + "="*80)
    print("STANDALONE ABLATION THERAPEUTIC CANDIDATE RANKING RESULTS")
    print("="*80)
    print(f"Total candidates: {len(results)}")
    print(f"From directory A: {len(results[results['unique_id'].str.endswith('_A')])}")
    print(f"From directory B: {len(results[results['unique_id'].str.endswith('_B')])}")
    print("\nTop 10 ranked candidates:")
    print("-" * 80)
    
    for idx, row in results.head(10).iterrows():
        print(f"Rank {row['rank']:2d}: {row['unique_id']} (Score: {row['strength_score']:.4f})")
        print(f"         Hypothesis: {row['hypothesis']}")
        print(f"         File: {row['file_url']}")
        print()


if __name__ == "__main__":
    main()
