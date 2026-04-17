# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robin is a multi-agent system for automating scientific discovery in therapeutics development. It generates experimental assays and therapeutic candidates for disease research by orchestrating LLMs and FutureHouse platform agents (Crow for literature search, Falcon for report generation, Finch for data analysis).

## Development Setup

```bash
# Install with uv (preferred)
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'

# Or with pip
pip install -e '.[dev]'
```

Required env vars: `OPENAI_API_KEY`, `FUTUREHOUSE_API_KEY`

## Common Commands

```bash
# Run all pre-commit checks (linting, formatting, type checking)
pre-commit run --all-files

# Individual tools
ruff check .
mypy robin/
pylint robin/
black .
```

There is no automated test suite — validation is done via notebooks (`robin_demo.ipynb`, `robin_full.ipynb`) and the `examples/` directory.

## Architecture

### Pipeline Flow

The three main entry points in `robin/__init__.py`:

1. **`experimental_assay(config)`** — Generates search queries → runs Crow (literature search) → ranks assays via pairwise comparison → returns top assay
2. **`therapeutic_candidates(goal, config)`** — Same pipeline structure for therapeutic candidates
3. **`data_analysis(data_path, analysis_type, goal, config)`** — Executes R/Python analysis steps on the FutureHouse Finch platform

### Key Abstractions

- **`RobinConfiguration`** (`configuration.py`) — Central config object with lazy-loaded LLM and FutureHouse clients. All pipeline functions accept this as their primary argument.
- **`Prompts`** (`configuration.py`) — Pydantic model that validates all prompt templates. Raises on missing `{placeholder}` substitutions — always fill all placeholders before passing to agents.
- **`Step` / `StepConfig` / `MultiTrajectoryRunner`** (`multitrajectory_runner.py`) — Abstractions for running agent tasks on the FutureHouse platform. Steps are executed in parallel via asyncio.
- **`prompts.py`** — 800+ lines of prompt templates for all pipeline stages. Edit here to change agent behavior.

### Agent Integration

FutureHouse agents accessed via `futurehouse_client`:
- **Crow** — Literature search (used in both assay and candidate pipelines)
- **Falcon** — Detailed report generation for candidates
- **Finch** — Code execution (R/Python) for data analysis — requires FutureHouse closed beta access

LLMs accessed via `lmi` (LiteLLM wrapper). Default model is `o4-mini`.

### Output Structure

All results saved to `robin_output/{disease_name}_{timestamp}/` with subdirectories for literature reviews, detailed hypotheses, ranking CSVs, and summaries.

## Code Conventions

- Python 3.12+ required
- Async/await throughout — all pipeline functions are `async`
- Line length: 97 characters (ruff/black configured)
- Docstrings: Google style
- Strict mypy with pydantic plugin enabled
- Pre-commit hooks enforce ruff, black, mypy, pylint, prettier, and typos — run before committing
