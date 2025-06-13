import json
import logging
from pathlib import Path
from typing import Any, cast

import aiofiles
import choix
import pandas as pd
from aviary.core import Message

from .configuration import RobinConfiguration
from .utils import (
    call_platform,
    format_assay_ideas,
    output_to_string,
    processing_ranking_output,
    run_comparisons,
    save_crow_files,
    uniformly_random_pairs,
)
from .get_mesh_terms import get_mesh_terms

logger = logging.getLogger(__name__)


async def generate_assay_queries(
    configuration: RobinConfiguration,
) -> dict[str, str]:
    """
    Step 1: Formulate literature search queries for experimental assays.

    Uses an LLM to generate a set of queries for researching experimental
    assays relevant to the specified disease.

    Args:
        configuration: The RobinConfiguration object for the run.

    Returns:
        A dictionary of queries for the literature search, where keys and values
        are the query strings.
    """
    logger.info("\n\nStep 1: Formulating relevant queries for literature search...")

    assay_literature_system_message = (
        configuration.prompts.assay_literature_system_message.format(
            num_assays=configuration.num_assays
        )
    )


    mesh_terms = get_mesh_terms(configuration.disease_name)
    assay_literature_user_message = (
        configuration.prompts.assay_literature_user_message.format(
            num_queries=configuration.num_queries,
            disease_name=configuration.disease_name,
            mesh_terms=", ".join(mesh_terms),
        )
    )

    assay_literature_query_messages = [
        Message(role="system", content=assay_literature_system_message),
        Message(role="user", content=assay_literature_user_message),
    ]

    assay_literature_query_result = await configuration.llm_client.call_single(
        assay_literature_query_messages
    )

    assay_literature_query_result_text = cast(str, assay_literature_query_result.text)
    assay_literature_queries = assay_literature_query_result_text.split("<>")
    logger.info("Generated Queries:")
    for ia, aquery in enumerate(assay_literature_queries):
        logger.info(f"{ia + 1}. {aquery}")

    return {q: q for q in assay_literature_queries}


async def experimental_assay_lit_review(
    configuration: RobinConfiguration, experimental_assay_queries_dict: dict[str, str]
) -> str:
    """
    Step 2: Conduct a literature review for experimental assays.

    Uses the FutureHouse platform to run the generated queries and returns a
    summarized string of the results.

    Args:
        configuration: The RobinConfiguration object for the run.
        experimental_assay_queries_dict: The dictionary of queries to run.

    Returns:
        A string containing the summarized literature review.
    """
    logger.info("\n\nStep 2: Conducting literature search with FutureHouse platform...")

    assay_lit_review = await call_platform(
        queries=experimental_assay_queries_dict,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.assay_lit_search_agent,
    )

    assay_lit_review_results = assay_lit_review["results"]

    save_crow_files(
        assay_lit_review_results,
        run_dir=f"robin_output/{configuration.run_folder_name}/experimental_assay_literature_reviews",
        prefix="query",
    )

    return output_to_string(assay_lit_review_results)


async def propose_experimental_assay(
    configuration: RobinConfiguration, assay_lit_review_output: str
) -> list[str]:
    """
    Step 3: Propose experimental assays based on the literature review.

    Uses an LLM to generate a list of assay ideas based on the literature review,
    saves a summary, and returns the list of proposals.

    Args:
        configuration: The RobinConfiguration object for the run.
        assay_lit_review_output: A string containing the summarized literature review.

    Returns:
        A list of formatted strings, where each string represents a proposed assay.
    """
    logger.info("\n\nStep 3: Generating ideas for relevant experimental assays...")

    assay_proposal_system_message = (
        configuration.prompts.assay_proposal_system_message.format(
            num_assays=configuration.num_assays
        )
    )

    assay_proposal_user_message = (
        configuration.prompts.assay_proposal_user_message.format(
            num_assays=configuration.num_assays,
            disease_name=configuration.disease_name,
            assay_lit_review_output=assay_lit_review_output,
        )
    )

    assay_proposal_messages = [
        Message(role="system", content=assay_proposal_system_message),
        Message(role="user", content=assay_proposal_user_message),
    ]

    experimental_assay_ideas = await configuration.llm_client.call_single(
        assay_proposal_messages
    )

    assay_idea_json = json.loads(cast(str, experimental_assay_ideas.text))
    assay_idea_list = format_assay_ideas(assay_idea_json)

    for assay_idea in assay_idea_list:
        logger.info(f"{assay_idea[:100]}...")

    assay_list_export_file = (
        f"robin_output/{configuration.run_folder_name}/experimental_assay_summary.txt"
    )

    async with aiofiles.open(assay_list_export_file, "w") as f:
        for i, item in enumerate(assay_idea_list):
            parts = item.split("<|>")
            strategy = parts[0]
            reasoning = parts[1]

            await f.write(f"Assay Candidate {i + 1}:\n")
            await f.write(f"{strategy}\n")
            await f.write(f"{reasoning}\n\n")

    logger.info(f"Successfully exported to {assay_list_export_file}")

    return assay_idea_list


async def experimental_assay_detailed_reports(
    configuration: RobinConfiguration, assay_idea_list: list[str]
) -> dict[str, Any]:
    """
    Step 4: Generate detailed reports for all proposed experimental assays.

    For each proposed assay, this function uses the FutureHouse platform to generate
    a detailed report, which is then saved to disk.

    Args:
        configuration: The RobinConfiguration object for the run.
        assay_idea_list: The list of proposed assay strings.

    Returns:
        A dictionary containing the raw results from the platform call.
    """
    logger.info("\n\nStep 4: Detailed investigation and evaluation for each assay...")

    def create_assay_hypothesis_queries(assay_idea_list: list[str]) -> dict[str, str]:
        assay_hypothesis_system_prompt = (
            configuration.prompts.assay_hypothesis_system_prompt.format(
                disease_name=configuration.disease_name
            )
        )
        assay_hypothesis_format = configuration.prompts.assay_hypothesis_format.format(
            disease_name=configuration.disease_name
        )
        assay_hypothesis_queries = {}
        formatted_assay_idea_list = [
            item.replace("<|>", "\n") for item in assay_idea_list
        ]
        for assay in formatted_assay_idea_list:
            assay_name = assay.split("Strategy:")[1].split("\n")[0].strip()
            assay_hypothesis_queries[assay_name] = (
                assay_hypothesis_system_prompt + assay + assay_hypothesis_format
            )
        return assay_hypothesis_queries

    assay_hypothesis_queries = create_assay_hypothesis_queries(
        assay_idea_list=assay_idea_list
    )

    assay_hypotheses = await call_platform(
        queries=assay_hypothesis_queries,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.assay_hypothesis_report_agent,
    )

    save_crow_files(
        assay_hypotheses["results"],
        run_dir=f"robin_output/{configuration.run_folder_name}/experimental_assay_detailed_hypotheses",
        prefix="assay_hypothesis",
        has_hypothesis=True,
    )

    return assay_hypotheses


async def select_top_experimental_assay(
    configuration: RobinConfiguration,
    assay_hypotheses: dict[str, Any],
) -> str:
    """
    Step 5: Rank and select the top experimental assay.

    Uses pairwise comparison with an LLM to rank the detailed assay reports
    and returns the name of the top-ranked assay.

    Args:
        configuration: The RobinConfiguration object for the run.
        assay_hypotheses: The dictionary of detailed assay reports.

    Returns:
        The name (hypothesis) of the top-ranked experimental assay.
    """
    logger.info("\n\nStep 5: Selecting the top experimental assay...")

    assay_hypothesis_df = pd.DataFrame(assay_hypotheses["results"])
    assay_hypothesis_df["index"] = assay_hypothesis_df.index

    assay_ranking_system_prompt = (
        configuration.prompts.assay_ranking_system_prompt.format(
            disease_name=configuration.disease_name
        )
    )

    assay_ranking_prompt_format = configuration.prompts.assay_ranking_prompt_format

    assay_ranking_output_folder = f"robin_output/{configuration.run_folder_name}"
    assay_ranking_output_folder_path = Path(assay_ranking_output_folder)
    assay_ranking_output_filepath = (
        assay_ranking_output_folder_path / "experimental_assay_ranking_results.csv"
    )
    assay_ranking_output_folder_path.mkdir(parents=True, exist_ok=True)

    assay_pairs_list = uniformly_random_pairs(n_hypotheses=configuration.num_assays)

    await run_comparisons(
        pairs_list=assay_pairs_list,
        client=configuration.llm_client,
        system_prompt=assay_ranking_system_prompt,
        ranking_prompt_format=assay_ranking_prompt_format,
        assay_hypothesis_df=assay_hypothesis_df,
        output_filepath=str(assay_ranking_output_filepath),
    )

    assay_ranking_df = processing_ranking_output(str(assay_ranking_output_filepath))
    games_data = assay_ranking_df["Game Score"].to_list()
    params = choix.ilsr_pairwise(configuration.num_assays, games_data, alpha=0.1)

    assay_ranked_results = pd.DataFrame()
    assay_ranked_results["hypothesis"] = assay_hypothesis_df["hypothesis"]
    assay_ranked_results["answer"] = assay_hypothesis_df["answer"]
    assay_ranked_results["strength_score"] = params
    assay_ranked_results["index"] = assay_hypothesis_df["index"]
    assay_ranked_results_sorted = assay_ranked_results.sort_values(
        by="strength_score", ascending=False
    )

    top_experimental_assay = assay_ranked_results_sorted["hypothesis"].iloc[0]

    logger.info(f"Experimental Assay Selected: {top_experimental_assay}")

    return top_experimental_assay


async def synthesize_candidate_goal(
    configuration: RobinConfiguration, assay_name: str
) -> str:
    """
    Step 6: Synthesize a research goal for the next stage (candidate generation).

    Given the top-ranked assay, this function uses an LLM to generate a concise
    research goal for identifying therapeutic compounds.

    Args:
        configuration: The RobinConfiguration object for the run.
        assay_name: The name of the top-ranked assay.

    Returns:
        The synthesized research goal as a string.
    """
    client = configuration.llm_client
    synthesize_user_content = configuration.prompts.synthesize_user_content.format(
        assay_name=assay_name, disease_name=configuration.disease_name
    )

    synthesize_system_message_content = (
        configuration.prompts.synthesize_system_message_content.format(
            disease_name=configuration.disease_name
        )
    )

    messages = [
        Message(role="system", content=synthesize_system_message_content),
        Message(role="user", content=synthesize_user_content),
    ]

    response = await client.call_single(messages)
    return cast(str, response.text)


async def experimental_assay(configuration: RobinConfiguration) -> str:
    """
    Orchestrates the full workflow for identifying and selecting an experimental assay.

    This function coordinates the following steps:
    1. Generates literature search queries.
    2. Conducts the literature review.
    3. Proposes experimental assay ideas.
    4. Generates detailed reports on each proposal.
    5. Ranks the proposals to select the best one.
    6. Synthesizes a research goal for the next phase based on the top assay.

    Args:
        configuration: The RobinConfiguration object for the run.

    Returns:
        The synthesized candidate generation goal, or None if the process fails.
    """
    logger.info("Starting selection of a relevant experimental assay.")
    logger.info("————————————————————————————————————————————————————")

    # Step 1: Generate queries for Crow
    experimental_assay_queries_dict = await generate_assay_queries(configuration)

    # Step 2: Conduct literature review on cell culture assays
    assay_lit_review_output = await experimental_assay_lit_review(
        configuration, experimental_assay_queries_dict
    )

    # Step 3: Propose experimental assays
    assay_idea_list = await propose_experimental_assay(
        configuration, assay_lit_review_output
    )

    # Step 4: Generate detailed reports for all assays
    assay_hypotheses = await experimental_assay_detailed_reports(
        configuration, assay_idea_list
    )
    # Step 5: Select the top experimental assay
    top_experimental_assay = await select_top_experimental_assay(
        configuration, assay_hypotheses
    )

    # Step 6: Synthesize candidate generation goal
    candidate_generation_goal = await synthesize_candidate_goal(
        configuration, top_experimental_assay
    )
    logger.info(f"Candidate Generation Goal: {candidate_generation_goal}")

    return candidate_generation_goal
