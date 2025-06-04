import json
import logging
from pathlib import Path
from typing import cast

import aiofiles
import choix
import pandas as pd
from aviary.core import Message
from lmi import LiteLLMModel

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

logger = logging.getLogger(__name__)


async def experimental_assay(configuration: RobinConfiguration) -> str | None:

    logger.info("Starting selection of a relevant experimental assay.")
    logger.info("————————————————————————————————————————————————————")

    # Step 1: Generating queries for Crow

    logger.info("\n\nStep 1: Formulating relevant queries for literature search...")

    assay_literature_system_message = (
        configuration.prompts.assay_literature_system_message.format(
            num_assays=configuration.num_assays
        )
    )

    assay_literature_user_message = (
        configuration.prompts.assay_literature_user_message.format(
            num_queries=configuration.num_queries,
            disease_name=configuration.disease_name,
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

    experimental_assay_queries_dict = {}

    experimental_assay_queries_dict = {q: q for q in assay_literature_queries}

    # ### Step 2: Literature review on cell culture assays

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

    assay_lit_review_output = output_to_string(assay_lit_review_results)

    # ### Step 3: Proposing cell culture assays

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

    # ### Step 4: Generating reports for all assays

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

    # ### Step 5: Selecting the top experimental assay

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

    # ## Synthesizing goal for candidate generation using specified assay and disease

    async def synthesize_candidate_goal(
        assay_name: str, client: LiteLLMModel
    ) -> str | None:

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

    candidate_generation_goal = await synthesize_candidate_goal(
        top_experimental_assay, configuration.llm_client
    )

    logger.info(f"Candidate Generation Goal: {candidate_generation_goal}")

    return candidate_generation_goal
