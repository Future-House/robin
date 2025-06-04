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


async def therapeutic_candidates(  # noqa: PLR0912
    candidate_generation_goal: str,
    configuration: RobinConfiguration,
    experimental_insights: dict[str, str] | None = None,
) -> None:

    logger.info(
        f"Starting generation of {configuration.num_candidates} therapeutic candidates."
    )
    logger.info("———————————————————————————————————————————————————————————————")

    # ### Step 1: Generating queries for Crow

    logger.info("\n\nStep 1: Formulating relevant queries for literature search...")

    candidate_query_generation_system_message = (
        configuration.prompts.candidate_query_generation_system_message.format(
            disease_name=configuration.disease_name
        )
    )

    run_config_folder_name = str(configuration.run_folder_name)
    # if experimental_insights:
    #     run_config_folder_name += "_experimental"

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
        candidate_query_generation_messages
    )

    candidate_query_generation_result_text = cast(
        str, candidate_query_generation_result.text
    )
    candidate_generation_queries = candidate_query_generation_result_text.split("<>")
    logger.info("Generated Queries:")
    for ic, cquery in enumerate(candidate_generation_queries):
        logger.info(f"{ic + 1}. {cquery}")

    candidate_generation_queries_dict = {q: q for q in candidate_generation_queries}

    # ### Step 2: Literature review on therapeutic candidates

    logger.info("\n\nStep 2: Conducting literature search with FutureHouse platform...")

    therapeutic_candidate_review = await call_platform(
        queries=candidate_generation_queries_dict,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.candidate_lit_search_agent,
    )

    if experimental_insights:
        save_crow_files(
            therapeutic_candidate_review["results"],
            run_dir=f"robin_output/{run_config_folder_name}/therapeutic_candidate_literature_reviews_experimental",
            prefix="query",
        )
    else:
        save_crow_files(
            therapeutic_candidate_review["results"],
            run_dir=f"robin_output/{run_config_folder_name}/therapeutic_candidate_literature_reviews",
            prefix="query",
        )

    therapeutic_candidate_review_output = output_to_string(
        therapeutic_candidate_review["results"]
    )

    # ### Step 3: Proposing therapeutic candidates

    logger.info(
        f"\n\nStep 3: Generating {configuration.num_candidates} ideas for therapeutic"
        " candidates..."
    )

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

    candidate_generation_result = await configuration.llm_client.call_single(
        messages,
        timeout=600,
        temperature=1,
        max_tokens=32000,
        reasoning_effort="high")

    llm_raw_output = cast(str, candidate_generation_result.text)
    candidate_ideas_json = []

    # Split by "<CANDIDATE END>" and filter out any empty strings
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
        candidate_list_export_file = f"robin_output/{run_config_folder_name}/therapeutic_candidates_summary_experimental.txt"
    else:
        candidate_list_export_file = (
            f"robin_output/{run_config_folder_name}/therapeutic_candidates_summary.txt"
        )

    async with aiofiles.open(candidate_list_export_file, "w") as f:
        for i, item in enumerate(candidate_idea_list):
            parts = item.split("<|>")
            candidate = parts[0]
            hypothesis = parts[1]
            reasoning = parts[2]

            await f.write(f"Therapeutic Candidate {i + 1}:\n")
            await f.write(f"{candidate}\n")
            await f.write(f"{hypothesis}\n")
            await f.write(f"{reasoning}\n\n")

    logger.info(f"Successfully exported to {candidate_list_export_file}")

    # ### Step 4: Generating reports for all candidates

    logger.info("\n\nStep 4: Detailed investigation and evaluation for candidates...")

    def create_therapeutic_candidate_queries(
        candidate_idea_list: list[str],
    ) -> dict[str, str]:

        candidate_lit_review_direction_prompt = (
            configuration.prompts.candidate_lit_review_direction_prompt.format(
                disease_name=configuration.disease_name
            )
        )

        candidate_report_format = configuration.prompts.candidate_report_format.format(
            disease_name=configuration.disease_name
        )

        candidate_queries = {}

        formatted_candidate_idea_list = [
            item.replace("<|>", "\n") for item in candidate_idea_list
        ]

        for candidate in formatted_candidate_idea_list:
            candidate_name = candidate.split("Candidate:")[1].split("\n")[0].strip()
            candidate_queries[candidate_name] = (
                candidate_lit_review_direction_prompt
                + candidate
                + candidate_report_format
            )

        return candidate_queries

    therapeutic_candidate_queries = create_therapeutic_candidate_queries(
        candidate_idea_list=candidate_idea_list
    )

    therapeutic_candidate_hypotheses = await call_platform(
        queries=therapeutic_candidate_queries,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.candidate_hypothesis_report_agent,
    )

    final_therapeutic_candidate_hypotheses = await format_final_report(
        therapeutic_candidate_hypotheses["results"], configuration.llm_client
    )

    if experimental_insights:
        save_falcon_files(
            final_therapeutic_candidate_hypotheses,
            run_dir=f"robin_output/{run_config_folder_name}/therapeutic_candidate_detailed_hypotheses_experimental",
            prefix="therapeutic_candidate",
        )
    else:
        save_falcon_files(
            final_therapeutic_candidate_hypotheses,
            run_dir=f"robin_output/{run_config_folder_name}/therapeutic_candidate_detailed_hypotheses",
            prefix="therapeutic_candidate",
        )

    # ### Step 5: Ranking/selecting the therapeutic candidates

    logger.info("\n\nStep 5: Ranking the strength of the therapeutic candidates...")

    candidate_information_df = extract_candidate_info_from_folder(
        f"robin_output/{run_config_folder_name}/therapeutic_candidate_detailed_hypotheses"
    )

    candidate_ranking_system_prompt = (
        configuration.prompts.candidate_ranking_system_prompt.format(
            disease_name=configuration.disease_name
        )
    )

    candidate_ranking_prompt_format = (
        configuration.prompts.candidate_ranking_prompt_format
    )

    if experimental_insights:
        candidate_ranking_output_folder = f"robin_output/{run_config_folder_name}"
        candidate_ranking_output_folder_path = Path(candidate_ranking_output_folder)
        candidate_ranking_output_filepath = (
            candidate_ranking_output_folder_path
            / "therapeutic_candidate_ranking_results_experimental.csv"
        )
        candidate_ranking_output_folder_path.mkdir(parents=True, exist_ok=True)

        candidate_pairs_list = uniformly_random_pairs(
            n_hypotheses=len(candidate_information_df)
        )
    else:
        candidate_ranking_output_folder = f"robin_output/{run_config_folder_name}"
        candidate_ranking_output_folder_path = Path(candidate_ranking_output_folder)
        candidate_ranking_output_filepath = (
            candidate_ranking_output_folder_path
            / "therapeutic_candidate_ranking_results.csv"
        )
        candidate_ranking_output_folder_path.mkdir(parents=True, exist_ok=True)

        candidate_pairs_list = uniformly_random_pairs(
            n_hypotheses=len(candidate_information_df)
        )

    await run_comparisons(
        pairs_list=candidate_pairs_list,
        client=configuration.llm_client,
        system_prompt=candidate_ranking_system_prompt,
        ranking_prompt_format=candidate_ranking_prompt_format,
        assay_hypothesis_df=candidate_information_df,
        output_filepath=str(candidate_ranking_output_filepath),
    )

    logger.info(f"Processing ranking output from: {candidate_ranking_output_filepath}")
    therapeutic_candidate_ranking_df = processing_ranking_output(
        str(candidate_ranking_output_filepath)
    )

    if (
        therapeutic_candidate_ranking_df.empty
        or "Game Score" not in therapeutic_candidate_ranking_df.columns
    ):
        logger.error(
            "Ranking DataFrame is empty or missing 'Game Score' column. Cannot proceed"
            " with Choix."
        )
        return

    raw_game_scores_from_df = therapeutic_candidate_ranking_df["Game Score"].to_list()

    therapeutic_candidate_games_data = []
    valid_game_count = 0
    invalid_game_count = 0

    for game in raw_game_scores_from_df:
        if (
            game is not None
            and isinstance(game, (tuple, list))
            and len(game) == GAME_REQUIREMENT
        ):
            try:
                winner_id = int(game[0])
                loser_id = int(game[1])

                if (
                    0 <= winner_id < len(candidate_information_df)
                    and 0 <= loser_id < len(candidate_information_df)
                    and winner_id != loser_id
                ):
                    therapeutic_candidate_games_data.append((winner_id, loser_id))
                    valid_game_count += 1
                else:
                    logger.warning(
                        "Skipping game with out-of-range or identical IDs:"
                        f" {(winner_id, loser_id)}. Max index:"
                        f" {len(candidate_information_df) - 1}"
                    )
                    invalid_game_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Skipping game due to ID conversion error: {game}. Error: {e}"
                )
                invalid_game_count += 1
        else:
            logger.debug(
                f"Skipping malformed or None game score: {game} of type {type(game)}"
            )
            invalid_game_count += 1

    if invalid_game_count > 0:
        logger.warning(
            f"Prepared {valid_game_count} valid games and skipped"
            f" {invalid_game_count} invalid/malformed game scores for Choix."
        )

    if not therapeutic_candidate_games_data:
        logger.error(
            "No valid game data to pass to choix.ilsr_pairwise. Aborting candidate"
            " ranking scores computation."
        )
        candidate_ranked_results_sorted_empty = pd.DataFrame(
            columns=["hypothesis", "answer", "strength_score", "index"]
        )
        candidate_ranked_results_sorted_empty.to_csv(
            f"{candidate_ranking_output_folder}/ranked_therapeutic_candidates_empty.csv",
            index=False,
        )
        logger.info(
            "Saved an empty ranked_therapeutic_candidates_empty.csv due to no valid"
            " game data."
        )
        return

    n_items = len(candidate_information_df)
    logger.info(
        f"Calling choix.ilsr_pairwise with n_items={n_items} and"
        f" {len(therapeutic_candidate_games_data)} games."
    )

    try:
        therapeutic_candidate_params = choix.ilsr_pairwise(
            n_items, therapeutic_candidate_games_data, alpha=0.1
        )
    except Exception:
        logger.exception("Error during choix.ilsr_pairwise")
        logger.exception(f"  n_items: {n_items}")
        logger.exception(f"  Number of games: {len(therapeutic_candidate_games_data)}")
        if therapeutic_candidate_games_data:
            logger.exception(
                f"  Example game data: {therapeutic_candidate_games_data[:5]}"
            )
            all_ids_in_games = set()
            for (
                w,
                l_,
            ) in therapeutic_candidate_games_data:
                all_ids_in_games.add(w)
                all_ids_in_games.add(l_)
            if all_ids_in_games:
                logger.exception(
                    f"  Min ID in games: {min(all_ids_in_games)}, Max ID in games:"
                    f" {max(all_ids_in_games)}"
                )
        candidate_ranked_results_sorted_error = candidate_information_df[
            ["hypothesis", "answer", "index"]
        ].copy()
        candidate_ranked_results_sorted_error["strength_score"] = float("nan")
        candidate_ranked_results_sorted_error.to_csv(
            f"{candidate_ranking_output_folder}/ranked_therapeutic_candidates_choix_error.csv",
            index=False,
        )
        logger.info(
            "Saved ranked_therapeutic_candidates_choix_error.csv due to error in Choix."
        )
        return

    candidate_ranked_results = pd.DataFrame()
    if not candidate_information_df.empty:
        candidate_ranked_results["hypothesis"] = candidate_information_df["hypothesis"]
        candidate_ranked_results["answer"] = candidate_information_df["answer"]
        candidate_ranked_results["index"] = candidate_information_df["index"]
        if len(therapeutic_candidate_params) == len(candidate_information_df):
            candidate_ranked_results["strength_score"] = therapeutic_candidate_params
        else:
            logger.error(
                "Mismatch in length between Choix params and candidate_information_df."
                " Cannot assign strength scores."
            )
            candidate_ranked_results["strength_score"] = float("nan")

        candidate_ranked_results_sorted = candidate_ranked_results.sort_values(
            by="strength_score", ascending=False
        )
    else:
        logger.warning(
            "candidate_information_df was empty, creating an empty ranked results"
            " table."
        )
        candidate_ranked_results_sorted = pd.DataFrame(
            columns=["hypothesis", "answer", "strength_score", "index"]
        )

    if experimental_insights:
        candidate_ranked_results_sorted.to_csv(
            f"{candidate_ranking_output_folder}/ranked_therapeutic_candidates_experimental.csv",
            index=False,
        )
        logger.info(
            "Finished! Saved final rankings to"
            f" {candidate_ranking_output_folder}/ranked_therapeutic_candidates_experimental.csv"
        )
    else:
        candidate_ranked_results_sorted.to_csv(
            f"{candidate_ranking_output_folder}/ranked_therapeutic_candidates.csv",
            index=False,
        )
        logger.info(
            "Finished! Saved final rankings to"
            f" {candidate_ranking_output_folder}/ranked_therapeutic_candidates.csv"
        )
