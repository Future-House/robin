#!/usr/bin/env python3
"""
Test script for ablation therapeutic candidate ranking.

This script demonstrates how to use the ablation ranking functionality
to compare therapeutic candidates from two directories.
"""

import asyncio
import logging
from pathlib import Path

from robin.ablation_therapeutic_candidate_ranking import run_ablation_ranking

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def test_ablation_ranking():
    """Test the ablation ranking with example directories."""
    
    # Example directories - replace with your actual directories
    directory_a = "robin_output/dry_age-related_macular_degeneration_2025-10-27_17-51-49/therapeutic_candidate_detailed_hypotheses"
    directory_b = "robin_output/dry_age-related_macular_degeneration_2025-10-27_17-49-21/therapeutic_candidate_detailed_hypotheses"
    
    # Check if directories exist
    if not Path(directory_a).exists():
        print(f"Directory A not found: {directory_a}")
        return
    
    if not Path(directory_b).exists():
        print(f"Directory B not found: {directory_b}")
        return
    
    print("Running ablation therapeutic candidate ranking...")
    print(f"Directory A: {directory_a}")
    print(f"Directory B: {directory_b}")
    print()
    
    # Run the ranking
    results = await run_ablation_ranking(
        directory_a=directory_a,
        directory_b=directory_b,
        output_file="ablation_ranking_results.csv",
        llm_model="gpt-4o-mini",
        max_concurrent_requests=5  # Lower for testing
    )
    
    if not results.empty:
        print("\n" + "="*80)
        print("RANKING RESULTS")
        print("="*80)
        print(f"Total candidates: {len(results)}")
        print(f"From directory A: {len(results[results['unique_id'].str.endswith('_A')])}")
        print(f"From directory B: {len(results[results['unique_id'].str.endswith('_B')])}")
        print("\nTop 10 ranked candidates:")
        print("-" * 80)
        
        for idx, row in results.head(10).iterrows():
            print(f"Rank {row['rank']:2d}: {row['unique_id']} (Score: {row['strength_score']:.4f})")
            print(f"         Hypothesis: {row['hypothesis']}")
            print()
    else:
        print("No results generated.")

if __name__ == "__main__":
    asyncio.run(test_ablation_ranking())
