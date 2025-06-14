import logging
import re
from pathlib import Path
from typing import cast

import aiofiles
import choix
import pandas as pd
from aviary.core import Message

from .configuration import RobinConfiguration
from .utils import (
    call_platform,
    extract_candidate_info_from_folder,
    format_candidate_ideas,
    format_final_report,
    output_to_string,
    processing_ranking_output,
    run_comparisons,
    save_crow_files,
    save_falcon_files,
    uniformly_random_pairs,
)

logger = logging.getLogger(__name__)

GAME_REQUIREMENT = 2


async def generate_candidate_queries(
    configuration: RobinConfiguration,
    candidate_generation_goal: str,
    experimental_insights: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Step 1: Formulate literature search queries for therapeutic candidates.

    Args:
        configuration: The RobinConfiguration object for the run.
        candidate_generation_goal: The specific research goal for candidate generation.
        experimental_insights: Optional insights from experimental data analysis.

    Returns:
        A dictionary of queries for the literature search.
    """
    logger.info("\n\nStep 1: Formulating relevant queries for literature search...")

    candidate_query_generation_system_message = (
        configuration.prompts.candidate_query_generation_system_message.format(
            disease_name=configuration.disease_name
        )
    )

    if experimental_insights:
        candidate_query_generation_system_message += (
            configuration.prompts.experimental_insights_appendage.format(
                candidate_generation_goal=candidate_generation_goal,
                experimental_insights_analysis_summary=experimental_insights[
                    "analysis_summary"
                ],
                experimental_insights_mechanistic_insights=experimental_insights[
                    "mechanistic_insights"
                ],
                experimental_insights_questions_raised=experimental_insights[
                    "questions_raised"
                ],
            )
        )

    candidate_query_generation_content_message = (
        configuration.prompts.candidate_query_generation_content_message.format(
            num_queries=configuration.num_queries,
            double_queries=2 * configuration.num_queries,
            candidate_generation_goal=candidate_generation_goal,
            disease_name=configuration.disease_name,
        )
    )

    candidate_query_generation_messages = [
        Message(role="system", content=candidate_query_generation_system_message),
        Message(role="user", content=candidate_query_generation_content_message),
    ]

    candidate_query_generation_result = await configuration.llm_client.call_single(
        candidate_query_generation_messages, temperature=1
    )

    candidate_query_generation_result_text = cast(
        str, candidate_query_generation_result.text
    )
    candidate_generation_queries = candidate_query_generation_result_text.split("<>")
    logger.info("Generated Queries:")
    for ic, cquery in enumerate(candidate_generation_queries):
        logger.info(f"{ic + 1}. {cquery}")

    return {q: q for q in candidate_generation_queries}


async def candidate_lit_review(
    configuration: RobinConfiguration,
    candidate_queries_dict: dict[str, str],
    experimental_insights: dict[str, str] | None = None,
) -> str:
    """
    Step 2: Conduct literature review on therapeutic candidates.

    Args:
        configuration: The RobinConfiguration object for the run.
        candidate_queries_dict: The dictionary of queries to run.
        experimental_insights: Optional insights from experimental data analysis.

    Returns:
        A string containing the summarized literature review.
    """
    logger.info("\n\nStep 2: Conducting literature search with FutureHouse platform...")

    run_folder_name = str(configuration.run_folder_name)

    therapeutic_candidate_review = await call_platform(
        queries=candidate_queries_dict,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.candidate_lit_search_agent,
    )

    if experimental_insights:
        save_crow_files(
            therapeutic_candidate_review["results"],
            run_dir=f"robin_output/{run_folder_name}/therapeutic_candidate_literature_reviews_experimental",
            prefix="query",
        )
    else:
        save_crow_files(
            therapeutic_candidate_review["results"],
            run_dir=f"robin_output/{run_folder_name}/therapeutic_candidate_literature_reviews",
            prefix="query",
        )

    return output_to_string(therapeutic_candidate_review["results"])


async def propose_therapeutic_candidates(  # noqa: PLR0912
    configuration: RobinConfiguration,
    candidate_generation_goal: str,
    therapeutic_candidate_review_output: str,
    experimental_insights: dict[str, str] | None = None,
) -> list[str]:
    """
    Step 3: Propose therapeutic candidates based on the literature review.

    Args:
        configuration: The RobinConfiguration object for the run.
        candidate_generation_goal: The specific research goal.
        therapeutic_candidate_review_output: The literature review summary string.
        experimental_insights: Optional insights from experimental data analysis.

    Returns:
        A list of formatted strings, each representing a proposed candidate.
    """
    logger.info(
        f"\n\nStep 3: Generating {configuration.num_candidates} ideas for therapeutic"
        " candidates..."
    )

    run_folder_name = str(configuration.run_folder_name)

    candidate_generation_system_message = (
        configuration.prompts.candidate_generation_system_message.format(
            disease_name=configuration.disease_name,
            num_candidates=configuration.num_candidates,
        )
    )

    candidate_generation_user_message = (
        configuration.prompts.candidate_generation_user_message.format(
            disease_name=configuration.disease_name,
            num_candidates=configuration.num_candidates,
            therapeutic_candidate_review_output=therapeutic_candidate_review_output,
        )
    )

    if experimental_insights:
        candidate_generation_user_message += (
            configuration.prompts.experimental_insights_for_candidate_generation.format(
                candidate_generation_goal=candidate_generation_goal,
                experimental_insights_analysis_summary=experimental_insights[
                    "analysis_summary"
                ],
                experimental_insights_mechanistic_insights=experimental_insights[
                    "mechanistic_insights"
                ],
                experimental_insights_questions_raised=experimental_insights[
                    "questions_raised"
                ],
            )
        )

    messages = [
        Message(role="system", content=candidate_generation_system_message),
        Message(role="user", content=candidate_generation_user_message),
    ]

    if "claude" in configuration.llm_name:
        candidate_generation_result = await configuration.llm_client.call_single(
            messages,
            timeout=600,
            temperature=1,
            max_tokens=32000,
            reasoning_effort="high",
        )
    else:
        candidate_generation_result = await configuration.llm_client.call_single(
            messages,
            temperature=1,
        )

    llm_raw_output = cast(str, candidate_generation_result.text)
    candidate_ideas_json = []

    raw_blocks = llm_raw_output.strip().split(r"<CANDIDATE END>")
    candidate_blocks_text = [block.strip() for block in raw_blocks if block.strip()]

    for block_content in candidate_blocks_text:
        if not block_content.startswith("<CANDIDATE START>"):
            logger.warning(
                "Skipping malformed block not starting with <CANDIDATE START>:"
                f" {block_content[:100]}..."
            )
            continue

        content_str = block_content.split("<CANDIDATE START>", 1)[1].strip()

        field_data = {}
        current_key = None
        accumulated_value = []

        for line in content_str.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            match = re.match(r"^([A-Z_]+):\s*(.*)", line)

            if match:
                if current_key and accumulated_value:
                    field_data[current_key] = "\n".join(accumulated_value).strip()

                current_key = match.group(1).strip()
                initial_value_part = match.group(2).strip()
                accumulated_value = [initial_value_part] if initial_value_part else []
            elif current_key:
                accumulated_value.append(line.strip())
            else:
                logger.warning(
                    "Orphan line in candidate block (before any key found):"
                    f" '{line_stripped}'"
                )

        if current_key and accumulated_value:
            field_data[current_key] = "\n".join(accumulated_value).strip()

        candidate_name = field_data.get("CANDIDATE")
        hypothesis_text = field_data.get("HYPOTHESIS")
        reasoning_text = field_data.get("REASONING")

        if not candidate_name or not hypothesis_text or not reasoning_text:
            logger.warning(
                f"Missing CANDIDATE or HYPOTHESIS in block: {block_content[:100]}..."
                " Skipping."
            )
            logger.debug(f"Parsed field_data for skipped block: {field_data}")
            continue

        current_candidate_data = {
            "candidate": candidate_name,
            "hypothesis": hypothesis_text,
            "reasoning": reasoning_text,
        }
        candidate_ideas_json.append(current_candidate_data)

    if not candidate_ideas_json:
        logger.error(
            f"No candidate ideas were parsed from LLM output:\n{llm_raw_output}"
        )

    candidate_idea_list = format_candidate_ideas(candidate_ideas_json)

    logger.info(f"\nSuccessfully parsed {len(candidate_idea_list)} candidate ideas.")
    for idea_str in candidate_idea_list:
        logger.info(f"{idea_str[:100]}...")

    if experimental_insights:
        export_file = f"robin_output/{run_folder_name}/therapeutic_candidates_summary_experimental.txt"
    else:
        export_file = (
            f"robin_output/{run_folder_name}/therapeutic_candidates_summary.txt"
        )

    async with aiofiles.open(export_file, "w") as f:
        for i, item in enumerate(candidate_idea_list):
            parts = item.split("<|>")
            await f.write(f"Therapeutic Candidate {i + 1}:\n")
            await f.write(f"{parts[0]}\n")  # Candidate
            await f.write(f"{parts[1]}\n")  # Hypothesis
            await f.write(f"{parts[2]}\n\n")  # Reasoning

    logger.info(f"Successfully exported to {export_file}")
    return candidate_idea_list


async def candidate_detailed_reports(
    configuration: RobinConfiguration,
    candidate_idea_list: list[str],
    experimental_insights: dict[str, str] | None = None,
) -> None:
    """
    Step 4: Generate detailed reports for all proposed candidates.

    Args:
        configuration: The RobinConfiguration object for the run.
        candidate_idea_list: The list of proposed candidate strings.
        experimental_insights: Optional insights from experimental data analysis.
    """
    logger.info("\n\nStep 4: Detailed investigation and evaluation for candidates...")
    run_folder_name = str(configuration.run_folder_name)

    def create_therapeutic_candidate_queries(
        idea_list: list[str],
    ) -> dict[str, str]:
        prompt = configuration.prompts.candidate_lit_review_direction_prompt.format(
            disease_name=configuration.disease_name
        )
        report_format = configuration.prompts.candidate_report_format.format(
            disease_name=configuration.disease_name
        )
        queries = {}
        formatted_list = [item.replace("<|>", "\n") for item in idea_list]
        for candidate in formatted_list:
            name = candidate.split("Candidate:")[1].split("\n")[0].strip()
            queries[name] = prompt + candidate + report_format
        return queries

    therapeutic_candidate_queries = create_therapeutic_candidate_queries(
        candidate_idea_list
    )

    therapeutic_candidate_hypotheses = await call_platform(
        queries=therapeutic_candidate_queries,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.candidate_hypothesis_report_agent,
    )

    print(therapeutic_candidate_hypotheses)

    if experimental_insights:
        save_dir = f"robin_output/{run_folder_name}/therapeutic_candidate_detailed_hypotheses_experimental"
    else:
        save_dir = (
            f"robin_output/{run_folder_name}/therapeutic_candidate_detailed_hypotheses"
        )

    final_therapeutic_candidate_hypotheses = await format_final_report(
        therapeutic_candidate_hypotheses["results"], configuration
    )

    save_falcon_files(
        final_therapeutic_candidate_hypotheses,
        run_dir=save_dir,
        prefix="therapeutic_candidate",
    )


async def rank_therapeutic_candidates(  # noqa: PLR0912
    configuration: RobinConfiguration,
    experimental_insights: dict[str, str] | None = None,
) -> None:
    """
    Step 5: Rank the therapeutic candidates using pairwise comparison.

    Args:
        configuration: The RobinConfiguration object for the run.
        experimental_insights: Optional insights from experimental data analysis.
    """
    logger.info("\n\nStep 5: Ranking the strength of the therapeutic candidates...")

    run_folder_name = str(configuration.run_folder_name)
    hypotheses_folder = (
        f"robin_output/{run_folder_name}/therapeutic_candidate_detailed_hypotheses"
    )
    if experimental_insights:
        hypotheses_folder += "_experimental"

    candidate_information_df = extract_candidate_info_from_folder(hypotheses_folder)

    system_prompt = configuration.prompts.candidate_ranking_system_prompt.format(
        disease_name=configuration.disease_name
    )
    prompt_format = configuration.prompts.candidate_ranking_prompt_format

    output_folder = f"robin_output/{run_folder_name}"
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    if experimental_insights:
        ranking_csv = "therapeutic_candidate_ranking_results_experimental.csv"
        final_csv = "ranked_therapeutic_candidates_experimental.csv"
    else:
        ranking_csv = "therapeutic_candidate_ranking_results.csv"
        final_csv = "ranked_therapeutic_candidates.csv"

    output_filepath = output_folder_path / ranking_csv
    pairs_list = uniformly_random_pairs(n_hypotheses=len(candidate_information_df))

    await run_comparisons(
        pairs_list=pairs_list,
        client=configuration.llm_client,
        system_prompt=system_prompt,
        ranking_prompt_format=prompt_format,
        assay_hypothesis_df=candidate_information_df,
        output_filepath=str(output_filepath),
    )

    logger.info(f"Processing ranking output from: {output_filepath}")
    ranking_df = processing_ranking_output(str(output_filepath))

    if ranking_df.empty or "Game Score" not in ranking_df.columns:
        logger.error(
            "Ranking DataFrame is empty or invalid. Cannot proceed with Choix."
        )
        return

    raw_scores = ranking_df["Game Score"].to_list()
    games_data: list[tuple[int, int]] = []
    valid_count, invalid_count = 0, 0

    for game in raw_scores:
        if game and isinstance(game, (tuple, list)) and len(game) == GAME_REQUIREMENT:
            try:
                w_id, l_id = int(game[0]), int(game[1])
                if (
                    0 <= w_id < len(candidate_information_df)
                    and 0 <= l_id < len(candidate_information_df)
                    and w_id != l_id
                ):
                    games_data.append((w_id, l_id))
                    valid_count += 1
                else:
                    invalid_count += 1
            except (ValueError, TypeError):
                invalid_count += 1
        else:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(
            f"Prepared {valid_count} valid games and skipped {invalid_count} invalid games for Choix."
        )

    if not games_data:
        logger.error("No valid game data for Choix. Aborting ranking.")
        return

    n_items = len(candidate_information_df)
    logger.info(
        f"Calling choix.ilsr_pairwise with n_items={n_items} and {len(games_data)} games."
    )

    try:
        params = choix.ilsr_pairwise(n_items, games_data, alpha=0.1)
    except Exception:
        logger.exception("Error during choix.ilsr_pairwise")
        # Handle error case by saving partial data if needed
        return

    ranked_results = pd.DataFrame()
    if not candidate_information_df.empty:
        ranked_results = candidate_information_df[
            ["hypothesis", "answer", "index"]
        ].copy()
        if len(params) == len(candidate_information_df):
            ranked_results["strength_score"] = params
        else:
            logger.error("Mismatch in length between Choix params and DataFrame.")
            ranked_results["strength_score"] = float("nan")
        ranked_results = ranked_results.sort_values(
            by="strength_score", ascending=False
        )
    else:
        logger.warning(
            "Candidate info DataFrame was empty, creating empty ranked results."
        )
        ranked_results = pd.DataFrame(
            columns=["hypothesis", "answer", "strength_score", "index"]
        )

    final_filepath = output_folder_path / final_csv
    ranked_results.to_csv(final_filepath, index=False)
    logger.info(f"Finished! Saved final rankings to {final_filepath}")


async def therapeutic_candidates(
    candidate_generation_goal: str,
    configuration: RobinConfiguration,
    experimental_insights: dict[str, str] | None = None,
) -> None:
    """
    Overall function for generating and ranking therapeutic candidates.

    Args:
        candidate_generation_goal: The specific research goal to guide candidate generation.
        configuration: The RobinConfiguration object for the run.
        experimental_insights: Optional insights from experimental data analysis to refine the process.
    """
    logger.info(
        f"Starting generation of {configuration.num_candidates} therapeutic candidates."
    )
    logger.info("———————————————————————————————————————————————————————————————")

    # Step 1: Generate queries for literature search
    candidate_queries_dict = await generate_candidate_queries(
        configuration, candidate_generation_goal, experimental_insights
    )

    # Step 2: Conduct literature review
    lit_review_output = await candidate_lit_review(
        configuration, candidate_queries_dict, experimental_insights
    )

    # Step 3: Propose therapeutic candidates
    candidate_idea_list = await propose_therapeutic_candidates(
        configuration,
        candidate_generation_goal,
        lit_review_output,
        experimental_insights,
    )

    # Step 4: Generate detailed reports for all candidates
    await candidate_detailed_reports(
        configuration, candidate_idea_list, experimental_insights
    )

    # Step 5: Rank the therapeutic candidates
    await rank_therapeutic_candidates(configuration, experimental_insights)
