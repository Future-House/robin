# Ablation Therapeutic Candidate Ranking

This module provides functionality to compare and rank therapeutic candidate hypotheses from two different directories using pairwise comparisons and the Bradley-Terry model.

## Overview

The `ablation_therapeutic_candidate_ranking.py` script is designed to compare the outputs of Robin with ablated versions of Robin. It takes therapeutic candidate hypotheses from two directories (A and B) and ranks them using the same ranking infrastructure as the main Robin system.

## Features

- **Dual Directory Processing**: Processes therapeutic candidate files from two separate directories
- **Unique ID Generation**: Assigns unique identifiers to each hypothesis (e.g., `drug_hypothesis_1_A`, `drug_hypothesis_3_B`)
- **Pairwise Comparisons**: Uses LLM-based pairwise comparisons with scientific evaluation criteria
- **Bradley-Terry Model Scoring**: Calculates strength scores using `choix.ilsr_pairwise()`
- **Comprehensive Output**: Generates a complete data table with rankings and file URLs

## File Structure

The script expects therapeutic candidate files in the following format:
- Each file should be a `.txt` file
- Files should contain "Proposal for [Drug Name]" followed by detailed hypothesis information
- The script extracts the drug name from the "Proposal for" section

## Usage

### Command Line Interface

```bash
python -m robin.ablation_therapeutic_candidate_ranking <directory_A> <directory_B> [options]
```

**Arguments:**
- `directory_A`: Path to directory A containing therapeutic candidate files
- `directory_B`: Path to directory B containing therapeutic candidate files

**Options:**
- `--output`, `-o`: Output CSV file path (optional)
- `--model`, `-m`: LLM model to use (default: "gpt-4o-mini")
- `--max-concurrent`, `-c`: Maximum concurrent requests (default: 10)
- `--verbose`, `-v`: Enable verbose logging

**Example:**
```bash
python -m robin.ablation_therapeutic_candidate_ranking \
    robin_output/run_A/therapeutic_candidate_detailed_hypotheses \
    robin_output/run_B/therapeutic_candidate_detailed_hypotheses \
    --output ablation_results.csv \
    --model gpt-4o \
    --max-concurrent 5
```

### Programmatic Usage

```python
import asyncio
from robin.ablation_therapeutic_candidate_ranking import run_ablation_ranking

async def main():
    results = await run_ablation_ranking(
        directory_a="path/to/directory_a",
        directory_b="path/to/directory_b",
        output_file="results.csv",
        llm_model="gpt-4o-mini",
        max_concurrent_requests=10
    )
    
    print(f"Ranked {len(results)} candidates")
    print(results.head())

asyncio.run(main())
```

## Output Format

The script generates a DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| `unique_id` | Unique identifier (e.g., `drug_hypothesis_1_A`) |
| `file_url` | Full path to the source .txt file |
| `hypothesis` | Extracted drug name/hypothesis |
| `answer` | Full content of the therapeutic candidate file |
| `index` | Numeric index for ranking |
| `strength_score` | Bradley-Terry model strength score |
| `rank` | Final ranking (1 = highest score) |

## Ranking Process

1. **File Processing**: Scans both directories for `.txt` files and extracts hypotheses
2. **Data Preparation**: Creates a unified DataFrame with unique IDs and file URLs
3. **Pair Generation**: Generates random pairs of candidates for comparison
4. **LLM Evaluation**: Uses scientific criteria to compare pairs via LLM
5. **Bradley-Terry Scoring**: Calculates strength scores using `choix.ilsr_pairwise()`
6. **Ranking**: Assigns final ranks based on strength scores

## Scientific Evaluation Criteria

The ranking uses the same scientific criteria as the main Robin system:

1. **Strength and Relevance of Supporting Evidence**
2. **Mechanism of Action Clarity and Plausibility**
3. **Safety Profile and Risk Assessment**
4. **Feasibility and Development Potential**

## Dependencies

- `choix`: For Bradley-Terry model calculations
- `pandas`: For data manipulation
- `lmi`: For LLM interactions
- `asyncio`: For concurrent processing

## Example Output

```
================================================================================
ABLATION THERAPEUTIC CANDIDATE RANKING RESULTS
================================================================================
Total candidates: 50
From directory A: 25
From directory B: 25

Top 10 ranked candidates:
--------------------------------------------------------------------------------
Rank  1: drug_hypothesis_15_A (Score: 0.8234)
         Hypothesis: SRT1720
         File: /path/to/therapeutic_candidate_1_srt1720.txt

Rank  2: drug_hypothesis_8_B (Score: 0.7891)
         Hypothesis: Metformin
         File: /path/to/therapeutic_candidate_10_metformin.txt
...
```

## Testing

Use the provided test script to verify functionality:

```bash
python test_ablation_ranking.py
```

This will run the ranking on example directories and display the results.

## Notes

- The script handles errors gracefully and logs issues for debugging
- Temporary files are automatically cleaned up
- The ranking process is designed to be robust with validation of game data
- Concurrent processing is used for efficiency with configurable limits
