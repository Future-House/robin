# Robin: A multi-agent system for automating scientific discovery

See our [blog](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system) or [arXiv](https://arxiv.org/abs/2505.13400) preprint for more info.

## Prerequisites

- **Python:** Version 3.12 or higher.
- **API Keys:**
  - `FUTUREHOUSE_API_KEY`: For accessing FutureHouse platform agents (Crow, Falcon).
  - An API key for your chosen LLM provider (e.g., `OPENAI_API_KEY` if using OpenAI models). Robin uses LiteLLM, so it can support various providers.
  - The "Finch" (data analysis) portion of this repo needs access to the FutureHouse platform closed beta. To request access, visit https://platform.futurehouse.org/profile, and use the "Rate Limit Increase" form to request access to Finch. Without access, all the hypothesis and experiment generation code can still be run.

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Future-House/robin.git
    cd robin
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

    OR

    ```bash
    python3 -m venv .robin_env
    source .robin_env/bin/activate
    ```

3.  **Install Dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the base package and development dependencies (which include Jupyter):

    ```bash
    uv pip install -e '.[dev]'
    ```

    OR

    ```bash
    pip install -e '.[dev]'
    ```

4.  **Set API Keys:**
    It's highly recommended to set your API keys as environment variables. Create a `.env` file in the `robin` directory:
    ```
    FUTUREHOUSE_API_KEY="your_futurehouse_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    # etc. for other LLM providers
    ```
    The notebook and `RobinConfiguration` will attempt to load these. Alternatively, you can pass them directly when creating the `RobinConfiguration` object in the notebook.

## Running Robin via `robin_demo.ipynb`

1.  **Launch Jupyter Notebook or JupyterLab:**
    Navigate to the `robin` directory in your terminal (ensure your virtual environment is activated) and run:

    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```

2.  **Open the Notebook:**
    In the Jupyter interface, open `robin_demo.ipynb`.

3.  **Configure Robin:**
    Locate the cell where the `RobinConfiguration` object is created:

    ```python
    config = RobinConfiguration(
        disease_name="DISEASE_NAME",  # <-- Customize the disease name here
        # You can also explicitly set API keys here if not using environment variables:
        # futurehouse_api_key="your_futurehouse_api_key_here"
    )
    ```

    - **Modify `disease_name`**: Change `"DISEASE_NAME"` to your target disease.
    - **API Keys**: If you didn't set environment variables, you can provide the keys directly in the `RobinConfiguration` instantiation.
    - **LLM Choice**: The default is `o4-mini`. You can change `llm_name` and `llm_config` in `RobinConfiguration` if you wish to use a different model supported by LiteLLM (ensure you have the corresponding API key set).
    - Other parameters like `num_queries`, `num_assays`, `num_candidates` can also be adjusted here if needed.

4.  **Run the Notebook Cells:**
    Execute the cells in the notebook sequentially. The notebook is structured to guide you through:
    - **Experimental Assay Generation:** Generates and ranks potential experimental assays.
    - **Therapeutic Candidate Generation:** Based on the top assay, generates and ranks therapeutic candidates.
    - **(Optional) Experimental Data Analysis:** If you have experimental data, this section can analyze it and feed insights back into candidate generation. This currently requires access to the Finch closed beta.

## Expected Output

- **Logs:** Detailed logs will be printed in the notebook output and/or your console, showing the progress of each step (e.g., query generation, literature search, candidate proposal, ranking).

- **Files:** Results are saved in a new subdirectory within `robin_output/`, named after the `disease_name` and a timestamp (e.g., `robin_output/DISEASE_NAME_YYYY-MM-DD_HH-MM/`). This directory contains a structured set of outputs, including:
  - Folders for detailed hypotheses and literature reviews for both experimental assays and therapeutic candidates (e.g., `experimental_assay_detailed_hypotheses/`, `therapeutic_candidate_literature_reviews/`).
  - CSV files for ranking results and final ranked lists (e.g., `experimental_assay_ranking_results.csv`, `ranked_therapeutic_candidates.csv`).
  - Text summaries for proposed assays and candidates (e.g., `experimental_assay_summary.txt`, `therapeutic_candidates_summary.txt`).
  - If the optional data analysis step is run (using the `data_analysis` function), there will be an additional `data_analysis/` subfolder containing outputs from the Finch agent (e.g., `consensus_results.csv`). Correspondingly, some therapeutic candidate-related files generated after this step may have an `_experimental` suffix (e.g., `ranked_therapeutic_candidates_experimental.csv`, `therapeutic_candidate_detailed_hypotheses_experimental/`).

## Overview of `examples` Folder:

The `examples` folder provides practical usage demonstrations of pre-generated output directories from complete Robin runs for 10 diseases:

- Age-Related Hearing Loss
- Celiac Disease
- Charcot-Marie-Tooth Disease
- Chronic Kidney Disease
- Friedreich's Ataxia
- Glaucoma
- Idiopathic Pulmonary Fibrosis
- Non-alcoholic Steatohepatitis
- Polycystic Ovary Syndrome
- Sarcopenia

Each disease-specific subfolder mirrors the exact file and directory structure a user would obtain in their own `robin_output/` directory after a run:

- `experimental_assay_detailed_hypotheses/`: Text files containing detailed reports for each proposed experimental assay.
- `experimental_assay_literature_reviews/`: Text files of literature reviews generated from queries related to assay development.
- `experimental_assay_ranking_results.csv`: CSV file showing pairwise comparison results for assay ranking.
- `experimental_assay_summary.txt`: A textual summary of the proposed experimental assays.
- `ranked_therapeutic_candidates.csv`: CSV file listing the final ranked therapeutic candidates and their strength scores.
- `therapeutic_candidate_detailed_hypotheses/`: Text files with detailed reports for each proposed therapeutic candidate.
- `therapeutic_candidate_literature_reviews/`: Text files of literature reviews for therapeutic candidate queries.
- `therapeutic_candidate_ranking_results.csv`: CSV file of pairwise comparison results for candidate ranking.
- `therapeutic_candidates_summary.txt`: A textual summary of the proposed therapeutic candidates.

These example outputs are provided to help users to understand the depth, format, and typical errors seen in Robin runs across various diseases.

## Advanced Usage

A full example trajectory of both the initial therapeutic candidate generation and experimental data analysis can be found in the `robin_full.ipynb` notebook. This notebook includes the parameters and agents used in the paper. Note that the parameters used in this notebook exceeds the current free rate limits and data analysis functionality is currently in beta testing.

While this guide focuses on the `robin_demo.ipynb` notebook, the `robin` Python module (in the `robin/` directory) can be imported and its functions (`experimental_assay`, `therapeutic_candidates`, `data_analysis`) can be used programmatically in your own Python scripts for more customized workflows.
