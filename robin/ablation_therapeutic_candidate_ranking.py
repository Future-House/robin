#!/usr/bin/env python3
"""
Ablation Therapeutic Candidate Ranking Script

This script compares therapeutic candidate hypotheses from two directories (A and B)
and ranks them using pairwise comparisons with the Bradley-Terry model.

Usage:
    python ablation_therapeutic_candidate_ranking.py <directory_A> <directory_B> [output_file]

The script will:
1. Extract therapeutic candidate hypotheses from both directories
2. Create a data table with unique IDs and file URLs
3. Run pairwise comparisons using the existing ranking infrastructure
4. Calculate Bradley-Terry model scores using choix.ilsr_pairwise()
5. Output a complete data table with rankings
"""

import argparse
import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import choix
import pandas as pd

try:
    from lmi import LiteLLMModel
except ImportError:
    # Fallback for environments without lmi
    LiteLLMModel = None

try:
    from .configuration_CROW_ablation import RobinConfiguration, get_default_llm_config
    from .utils import (
        run_comparisons,
        processing_ranking_output,
        uniformly_random_pairs,
    )
except ImportError as e:
    # CROW ablation configuration is required - no fallback
    import sys
    print(
        f"ERROR: Failed to import configuration_CROW_ablation: {e}\n"
        "The ablation ranking script requires configuration_CROW_ablation.py. "
        "Please ensure the file exists and is properly configured.",
        file=sys.stderr
    )
    RobinConfiguration = None
    get_default_llm_config = None
    run_comparisons = None
    processing_ranking_output = None
    uniformly_random_pairs = None

logger = logging.getLogger(__name__)


def _find_candidate_directory(base_path: Path) -> Path | None:
    """
    Find the directory containing therapeutic candidate files.
    Checks the base path first, then looks in therapeutic_candidate_detailed_hypotheses subdirectory.
    
    Args:
        base_path: Base path to search
        
    Returns:
        Path to directory with .txt files, or None if not found
    """
    if not base_path.is_dir():
        return None
    
    # First, check if base_path itself contains .txt files
    txt_files = list(base_path.glob("*.txt"))
    if txt_files:
        logger.info(f"Found .txt files directly in: {base_path}")
        return base_path
    
    # If not, check for therapeutic_candidate_detailed_hypotheses subdirectory
    candidate_dir = base_path / "therapeutic_candidate_detailed_hypotheses"
    if candidate_dir.is_dir():
        txt_files = list(candidate_dir.glob("*.txt"))
        if txt_files:
            logger.info(f"Found .txt files in subdirectory: {candidate_dir}")
            return candidate_dir
    
    logger.warning(f"No .txt files found in {base_path} or its subdirectories")
    return None


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
    candidate_dir_a = _find_candidate_directory(dir_a_path)
    
    if candidate_dir_a:
        for txt_file in sorted(candidate_dir_a.glob("*.txt")):
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
    candidate_dir_b = _find_candidate_directory(dir_b_path)
    
    if candidate_dir_b:
        for txt_file in sorted(candidate_dir_b.glob("*.txt")):
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
    
    # Return DataFrame with proper columns even if empty
    if not candidates_data:
        return pd.DataFrame(columns=["unique_id", "file_url", "hypothesis", "answer", "index"])
    
    return pd.DataFrame(candidates_data)


async def run_ablation_ranking(
    directory_a: str,
    directory_b: str,
    output_file: str | None = None,
    llm_model: str = "anthropic/claude-3-7-sonnet-20250219",
    max_concurrent_requests: int = 10
) -> pd.DataFrame:
    """
    Run the ablation ranking process.
    
    Args:
        directory_a: Path to directory A containing therapeutic candidate files
        directory_b: Path to directory B containing therapeutic candidate files
        output_file: Optional output file path for results
        llm_model: LLM model to use for comparisons
        max_concurrent_requests: Maximum concurrent requests for comparisons
        
    Returns:
        DataFrame with rankings
    """
    logger.info("Starting ablation therapeutic candidate ranking...")
    
    # Check if required imports are available
    if LiteLLMModel is None:
        logger.error("LiteLLMModel not available. Please install lmi package.")
        return pd.DataFrame()
    
    if run_comparisons is None or processing_ranking_output is None or uniformly_random_pairs is None:
        logger.error("Required ranking functions not available. Please ensure robin package is properly installed.")
        return pd.DataFrame()
    
    # Create candidate DataFrame
    logger.info("Extracting candidates from directories...")
    candidates_df = create_candidate_dataframe(directory_a, directory_b)
    
    if candidates_df.empty:
        logger.error("No candidates found in either directory")
        return pd.DataFrame()
    
    logger.info(f"Found {len(candidates_df)} candidates total")
    
    # Set up configuration for ranking using RobinConfiguration from config file
    # This ensures proper timeout settings (400s for Claude) and other configuration are inherited
    if RobinConfiguration is None:
        logger.error("RobinConfiguration not available. Cannot create LLM client with proper config.")
        return candidates_df
    
    if get_default_llm_config is None:
        logger.error("get_default_llm_config not available. Cannot configure LLM properly.")
        return candidates_df
    
    # Create a RobinConfiguration instance with the specified model
    # This will use the timeout (400s) and other settings from get_default_llm_config()
    config = RobinConfiguration(
        disease_name="disease",
        llm_name=llm_model,
        llm_config=get_default_llm_config()
    )
    
    # Import prompts
    try:
        from .prompts import CANDIDATE_RANKING_SYSTEM_PROMPT, CANDIDATE_RANKING_PROMPT_FORMAT
    except ImportError:
        logger.error("Could not import ranking prompts. Please ensure robin package is properly installed.")
        return candidates_df
    
    # Generate pairs for comparison
    n_candidates = len(candidates_df)
    pairs_list = uniformly_random_pairs(n_hypotheses=n_candidates)
    logger.info(f"Generated {len(pairs_list)} pairs for comparison")
    
    # Create temporary output file for ranking results
    temp_output_file = "temp_ablation_ranking_results.csv"
    
    # Run comparisons
    logger.info("Running pairwise comparisons...")
    await run_comparisons(
        pairs_list=pairs_list,
        client=config.llm_client,
        system_prompt=CANDIDATE_RANKING_SYSTEM_PROMPT.format(disease_name=config.disease_name),
        ranking_prompt_format=CANDIDATE_RANKING_PROMPT_FORMAT,
        assay_hypothesis_df=candidates_df,
        output_filepath=temp_output_file,
        max_concurrent_requests=max_concurrent_requests
    )
    
    # Process ranking results
    logger.info("Processing ranking results...")
    ranking_df = processing_ranking_output(temp_output_file)
    
    if ranking_df.empty or "Game Score" not in ranking_df.columns:
        logger.error("No valid ranking results found")
        return candidates_df
    
    # Extract game data for choix
    raw_game_scores = ranking_df["Game Score"].to_list()
    games_data = []
    
    for game in raw_game_scores:
        if (
            game is not None
            and isinstance(game, (tuple, list))
            and len(game) == 2
        ):
            try:
                winner_id = int(game[0])
                loser_id = int(game[1])
                
                if (
                    0 <= winner_id < n_candidates
                    and 0 <= loser_id < n_candidates
                    and winner_id != loser_id
                ):
                    games_data.append((winner_id, loser_id))
            except (ValueError, TypeError):
                continue
    
    if not games_data:
        logger.error("No valid game data for ranking")
        return candidates_df
    
    logger.info(f"Using {len(games_data)} valid games for ranking")
    
    # Calculate Bradley-Terry model scores
    try:
        strength_scores = choix.ilsr_pairwise(n_candidates, games_data, alpha=0.1)
        logger.info("Successfully calculated Bradley-Terry model scores")
    except Exception as e:
        logger.error(f"Error calculating Bradley-Terry scores: {e}")
        strength_scores = [0.0] * n_candidates
    
    # Add rankings to the DataFrame
    candidates_df["strength_score"] = strength_scores
    candidates_df["rank"] = candidates_df["strength_score"].rank(ascending=False, method="dense").astype(int)
    
    # Sort by strength score (descending)
    candidates_df_sorted = candidates_df.sort_values("strength_score", ascending=False).reset_index(drop=True)
    
    # Clean up temporary file
    temp_path = Path(temp_output_file)
    if temp_path.exists():
        temp_path.unlink()
    
    # Save results if output file specified
    if output_file:
        candidates_df_sorted.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    return candidates_df_sorted


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare and rank therapeutic candidates from two directories"
    )
    parser.add_argument("directory_a", help="Path to directory A containing therapeutic candidate files")
    parser.add_argument("directory_b", help="Path to directory B containing therapeutic candidate files")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--model", "-m", default="anthropic/claude-3-7-sonnet-20250219", help="LLM model to use")
    parser.add_argument("--max-concurrent", "-c", type=int, default=10, help="Maximum concurrent requests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the ranking
    results = await run_ablation_ranking(
        directory_a=args.directory_a,
        directory_b=args.directory_b,
        output_file=args.output,
        llm_model=args.model,
        max_concurrent_requests=args.max_concurrent
    )
    
    # Print results
    print("\n" + "="*80)
    print("ABLATION THERAPEUTIC CANDIDATE RANKING RESULTS")
    print("="*80)
    print(f"Total candidates: {len(results)}")
    
    if results.empty or 'unique_id' not in results.columns:
        print("No candidates found or ranking failed.")
        return
    
    print(f"From directory A: {len(results[results['unique_id'].str.endswith('_A')])}")
    print(f"From directory B: {len(results[results['unique_id'].str.endswith('_B')])}")
    print("\nTop 10 ranked candidates:")
    print("-" * 80)
    
    for idx, row in results.head(10).iterrows():
        rank = row.get('rank', 'N/A')
        score = row.get('strength_score', 'N/A')
        unique_id = row.get('unique_id', 'N/A')
        hypothesis = row.get('hypothesis', 'N/A')
        file_url = row.get('file_url', 'N/A')
        
        print(f"Rank {rank:2d}: {unique_id} (Score: {score:.4f})")
        print(f"         Hypothesis: {hypothesis}")
        print(f"         File: {file_url}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
