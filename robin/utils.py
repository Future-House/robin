import ast
import asyncio
import csv
import io
import itertools
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, cast

import aiofiles
import pandas as pd
from aviary.core import Message
from futurehouse_client import FutureHouseClient, JobNames, TaskResponse
from lmi import LiteLLMModel
from tqdm.asyncio import tqdm_asyncio

from .prompts import (
    FINAL_REPORT_FORMATTING_SYSTEM_MESSAGE,
    FINAL_REPORT_FORMATTING_USER_MESSAGE,
)

logger = logging.getLogger(__name__)

POLLING_INTERVAL = 5  # seconds
OVERALL_TIMEOUT = 6000  # seconds


async def poll_for_task_completion(
    task_id: str, fh_client: FutureHouseClient
) -> TaskResponse | dict[str, str]:
    """Asynchronously polls a single task until it completes (success/failure)."""
    while True:
        try:
            task_response = fh_client.get_task(task_id)
            logger.debug(f"Polling task {task_id}: Status = {task_response.status}")

            if task_response.status == "success":
                return task_response
            if task_response.status in {"queued", "in progress"}:
                logger.debug(
                    f"Task {task_id} is still in progress (Status:"
                    f" {task_response.status}). Continuing poll..."
                )
                await asyncio.sleep(POLLING_INTERVAL)
                continue

        except Exception as e:
            logger.exception(f"Error polling task {task_id}")
            return {
                "status": "POLLING_ERROR",
                "error": f"Polling error: {e!s}",
                "task_id": task_id,
            }
        else:
            logger.warning(f"Task {task_id} failed with status: {task_response.status}")
            return task_response


async def gather_results(
    task_ids: list[str], fh_client: FutureHouseClient
) -> list[TaskResponse | dict[str, str]]:
    """Gathers results for multiple task IDs by polling concurrently."""
    tasks = [poll_for_task_completion(t_id, fh_client) for t_id in task_ids]
    return await asyncio.gather(*tasks)


async def call_platform(  # noqa: PLR0912
    queries: dict[str, str], fh_client: FutureHouseClient, job_name: JobNames
) -> dict[str, Any]:
    logger.info(
        f"Starting literature search for {len(queries)} queries using {job_name}."
    )
    task_id_to_context: dict[str, dict[str, str]] = {}
    submitted_ids: list[str] = []

    for hypothesis, q in queries.items():
        task_data = {
            "name": job_name,
            "query": q,
        }
        try:
            task_run_id = fh_client.create_task(task_data)
            if not isinstance(task_run_id, str):
                logger.warning(
                    "FutureHouseClient.create_task did not return a string ID for"
                    f" query '{q}'. Got: {type(task_run_id)}. Skipping."
                )
                continue

            task_id_to_context[task_run_id] = {"hypothesis": hypothesis, "query": q}
            submitted_ids.append(task_run_id)
        except Exception:
            logger.exception(f"Failed to submit task for query '{q}'")

    completed_tasks_results = []
    try:
        completed_tasks_results = await asyncio.wait_for(
            gather_results(submitted_ids, fh_client), timeout=OVERALL_TIMEOUT
        )
    except TimeoutError:
        logger.exception(
            f"Overall operation timed out after {OVERALL_TIMEOUT} seconds."
        )
        final_statuses = {}

        for task_id in submitted_ids:
            try:
                task_status_obj = await fh_client.get_task(task_id)
                final_statuses[task_id] = task_status_obj.status
            except Exception:
                final_statuses[task_id] = "unknown (timeout during final fetch)"
        return {
            "hypothesis": hypothesis,
            "error": f"Operation timed out after {OVERALL_TIMEOUT}s",
            "submitted_tasks": task_id_to_context,
            "final_statuses": final_statuses,
        }
    except Exception as gather_err:
        logger.exception("An error occurred while gathering results")
        return {
            "hypothesis": hypothesis,
            "error": f"Failed during result gathering: {gather_err!s}",
            "submitted_tasks": task_id_to_context,
        }

    all_results = []
    errors_occurred = False
    for task_result in completed_tasks_results:
        if (
            isinstance(task_result, dict)
            and task_result.get("status") == "POLLING_ERROR"
        ):
            errors_occurred = True
            current_task_id = str(task_result.get("task_id"))

            task_context = task_id_to_context.get(current_task_id, {})
            original_query = task_context.get(
                "query", f"Unknown Query (Polling Error for {current_task_id})"
            )
            original_hypothesis = task_context.get("hypothesis", "Unknown Hypothesis")

            error_message = task_result.get(
                "error", "Polling failed with unknown error"
            )
            logger.error(
                f"Polling failed for task {current_task_id} (Query:"
                f" '{original_query}'): {error_message}"
            )
            all_results.append(
                {
                    "hypothesis": original_hypothesis,
                    "query": original_query,
                    "error": error_message,
                    "status": "POLLING_ERROR",
                    "task_run_id": current_task_id,
                }
            )
        elif isinstance(
            task_result,
            (
                TaskResponse,
                fh_client.get_task.__annotations__.get("return", type(None)),
            ),
        ):
            current_task_id = str(task_result.task_id)

            task_context = task_id_to_context.get(current_task_id, {})
            original_query = task_context.get(
                "query", f"Unknown Query (Polling Error for {current_task_id})"
            )
            original_hypothesis = task_context.get("hypothesis", "Unknown Hypothesis")

            if task_result.status == "success":
                answer = task_result.answer
                verbose_task_result = fh_client.get_task(
                    task_result.task_id, verbose=True
                )
                sources = verbose_task_result.environment_frame["state"]["state"][
                    "response"
                ]["answer"]["references"]

                result_context = f"Query: {original_query}\nAnswer: {answer}"

                all_results.append(
                    {
                        "hypothesis": original_hypothesis,
                        "query": original_query,
                        "answer": answer,
                        "sources": sources,
                        "context": result_context,
                        "status": "success",
                        "task_run_id": current_task_id,
                    }
                )
            else:
                errors_occurred = True
                error_message = (
                    f"Task ended with status: {task_result.status}. Details:"
                    f" {getattr(task_result, 'error_details', 'N/A')}"
                )
                logger.error(
                    f"Task {current_task_id} for query '{original_query}' ended with"
                    f" status: {task_result.status}"
                )
                all_results.append(
                    {
                        "hypothesis": original_hypothesis,
                        "query": original_query,
                        "error": error_message,
                        "status": task_result.status,
                        "task_run_id": current_task_id,
                    }
                )
        else:
            errors_occurred = True
            logger.error(
                "Received unexpected result type during processing:"
                f" {type(task_result)}. Content: {task_result}"
            )
            all_results.append(
                {
                    "hypothesis": "Unknown Hypothesis (Result Error)",
                    "query": "Unknown Query (Result Error)",
                    "error": f"Received unexpected result type: {type(task_result)}",
                    "status": "PROCESSING_ERROR",
                    "task_run_id": getattr(task_result, "task_id", None),
                }
            )
    logger.info(f"Finished processing {len(completed_tasks_results)} tasks.")
    return {
        "results": all_results,
        "count": len(all_results),
        "has_errors": errors_occurred,
    }


def save_crow_files(
    data_list: list[dict[str, Any]],
    run_dir: str | Path,
    prefix: str,
    has_hypothesis: bool = False,
) -> None:
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(data_list):
        hypothesis_text = item.get("hypothesis", "").strip()
        query_text = item.get("query", "").strip()
        answer_text = item.get("answer", "").strip()
        sources_text = item.get("sources", "").strip()
        task_id_text = item.get("task_run_id", "").strip()

        file_number = i + 1

        first_word_from_hypothesis = "untitled"
        if hypothesis_text:
            match = re.match(r"([a-zA-Z0-9_]+)", hypothesis_text)
            if match:
                first_word_from_hypothesis = match.group(1).lower()
            elif query_text.split():
                first_word_from_hypothesis = re.sub(
                    r"\W+", "", hypothesis_text.split()[0]
                ).lower()
                if not first_word_from_hypothesis:
                    first_word_from_hypothesis = "untitled"

        filename = f"{prefix}_{file_number}_{first_word_from_hypothesis}.txt"
        filepath = run_dir_path / filename

        content = ""
        if has_hypothesis:
            content = f"Hypothesis: {hypothesis_text}\n\n"
        content += f"Query: {query_text}\n\n"
        content += f"{answer_text}\n\n"
        content += f"Full trajectory link: https://platform.futurehouse.org/trajectories/{task_id_text}\n\n"
        content += f"References:\n{sources_text}\n"

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Successfully wrote: {filename} to: {filepath}")
        except OSError as e:
            logger.info(f"Error writing {filename}: {e}")
        except Exception as e:
            logger.info(f"An unexpected error occurred while writing {filename}: {e}")


def save_falcon_files(
    data_list: list[dict[str, Any]],
    run_dir: str | Path,
    prefix: str,
) -> None:

    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(data_list):
        hypothesis_text = item.get("hypothesis", "").strip()
        formatted_output_text = item.get("formatted_output", "").strip()
        task_id_text = item.get("task_run_id", "").strip()

        file_number = i + 1

        first_word_from_hypothesis = "untitled"
        if hypothesis_text:
            match = re.match(r"([a-zA-Z0-9_]+)", hypothesis_text)
            if match:
                first_word_from_hypothesis = match.group(1).lower()
            elif hypothesis_text.split():
                first_word_from_hypothesis = re.sub(
                    r"\W+", "", hypothesis_text.split()[0]
                ).lower()
                if not first_word_from_hypothesis:
                    first_word_from_hypothesis = "untitled"

        filename = f"{prefix}_{file_number}_{first_word_from_hypothesis}.txt"
        filepath = run_dir_path / filename

        content = f"Proposal for {hypothesis_text}\n\n"
        content += f"{formatted_output_text}\n\n"
        content += f"Full trajectory link: https://platform.futurehouse.org/trajectories/{task_id_text}\n"

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Successfully wrote: {filename} to: {filepath}")
        except OSError as e:
            logger.info(f"Error writing {filename}: {e}")
        except Exception as e:
            logger.info(f"An unexpected error occurred while writing {filename}: {e}")


def output_to_string(data_list: list[dict[str, Any]]) -> str:
    full_output_string = ""
    for i, item in enumerate(data_list):
        number = i + 1

        query_text = item.get("query", "N/A").strip()
        answer_text = item.get("answer", "N/A").strip()
        sources_text = item.get("sources", "N/A").strip()

        full_output_string += f"Query {number}: {query_text}\n"
        full_output_string += f"Answer {number}: {answer_text}\n"
        full_output_string += f"References {number}:\n{sources_text}\n"

        if i < len(data_list) - 1:
            full_output_string += "\n---\n\n"

    return full_output_string


def format_assay_ideas(assay_data_list: list[dict[str, str]]) -> list[str]:
    """
    Converts a list of assay dictionaries into a list of formatted strings.

    Args:
        assay_data_list: A list of dictionaries, where each dictionary
                         must contain 'strategy_name' and 'reasoning' keys.

    Returns:
        A list of strings, with each string formatted as:
        "Strategy: {strategy_name}, Reasoning: {reasoning}"
    """
    formatted_ideas = []
    for item in assay_data_list:
        strategy_name = item.get("strategy_name", "N/A")
        reasoning = item.get("reasoning", "N/A")
        summary_string = f"Strategy: {strategy_name}<|>Reasoning: {reasoning}"
        formatted_ideas.append(summary_string)
    return formatted_ideas


def format_candidate_ideas(candidate_data_list: list[dict[str, str]]) -> list[str]:
    """
    Converts a list of candidate dictionaries into a list of formatted strings.

    Args:
        candidate_data_list: A list of dictionaries, where each dictionary
                         must contain 'strategy_name' and 'reasoning' keys.

    Returns:
        A list of strings, with each string formatted as:
        "Candidate: {candidate}, Hypothesis: {hypothesis}, Reasoning: {reasoning}"
    """
    formatted_ideas = []
    for item in candidate_data_list:
        candidate = item.get("candidate", "N/A")
        hypothesis = item.get("hypothesis", "N/A")
        reasoning = item.get("reasoning", "N/A")
        summary_string = (
            f"Candidate: {candidate}<|>Hypothesis: {hypothesis}<|>Reasoning:"
            f" {reasoning}"
        )
        formatted_ideas.append(summary_string)
    return formatted_ideas


def get_candidate_info(df: pd.DataFrame, id_num: int) -> dict:

    required_columns = ["hypothesis", "answer", "index"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"DataFrame is missing required columns: {missing_cols}")

    matching_rows = df[df["index"] == id_num]

    if matching_rows.empty:
        raise ValueError(f"ID {id_num} not found in the DataFrame's 'index' column.")
    if len(matching_rows) > 1:
        raise ValueError(f"Multiple rows found for ID {id_num}. ID should be unique.")

    target_row = matching_rows.iloc[0]

    return {
        "hypothesis": str(target_row["hypothesis"]),
        "answer": str(target_row["answer"]),
        "index": str(target_row["index"]),
    }


def uniformly_random_pairs(
    n_hypotheses: int, seed: int = 621, n_games: int | None = None
) -> list[tuple[int, int]]:

    random.seed(seed)

    MIN_HYPOTHESES = 2

    if n_hypotheses < MIN_HYPOTHESES:
        if n_games is not None and n_games > 0:
            raise ValueError(
                f"Cannot generate {n_games} pairs with distinct elements when"
                f" n_hypotheses={n_hypotheses} < 2."
            )
        return []

    max_possible_games = n_hypotheses * (n_hypotheses - 1) // 2

    if n_games is None:
        n_games = min(300, max_possible_games)
    elif n_games < 1:
        raise ValueError("n_games cannot be less than 1.")

    if n_games > max_possible_games:
        logger.warning(
            f"Warning: n_games ({n_games}) exceeds maximum possible"
            f" ({max_possible_games}). Setting n_games to {max_possible_games}."
        )
        n_games = max_possible_games

    all_possible_unordered_pairs = list(itertools.combinations(range(n_hypotheses), 2))
    return random.sample(all_possible_unordered_pairs, n_games)


def processing_ranking_output(ranking_csv_path: str) -> pd.DataFrame:
    try:
        ranking_output_df = pd.read_csv(ranking_csv_path)
    except FileNotFoundError:
        return pd.DataFrame(
            columns=["Winner", "Loser", "Winner ID", "Loser ID", "Game Score"]
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=["Winner", "Loser", "Winner ID", "Loser ID", "Game Score"]
        )

    def parse_custom_tuple_string(s):  # noqa: PLR0911
        if not isinstance(s, str):
            return (None, None)

        s_stripped = s.strip()
        # Regex to handle optional quotes around name and ID, and ensure ID is numeric
        match = re.fullmatch(
            r"\s*\(\s*"
            r'(?:"(?P<quoted_name>(?:[^"]|\\")*)"'
            r"|(?P<unquoted_name>[^,]+?))"
            r"\s*,\s*"
            r'(?:"(?P<quoted_id_val>\d+)"'
            r"|(?P<unquoted_id_val>\d+))"
            r"\s*\)\s*$",
            s_stripped,
        )

        if match:
            name_parts = match.group("quoted_name", "unquoted_name")
            id_parts = match.group("quoted_id_val", "unquoted_id_val")

            name = next((part for part in name_parts if part is not None), None)
            id_val_str = next((part for part in id_parts if part is not None), None)

            if name is not None and id_val_str is not None:
                try:
                    name = name.replace('\\"', '"')
                    id_val = int(id_val_str)
                    return (name.strip(), id_val)
                except ValueError:
                    return (name.strip(), None)
            else:
                return (None, None)
        else:
            try:
                EXPECTED_TUPLE_LENGTH = 2
                parsed_tuple = ast.literal_eval(s_stripped)
                if (
                    isinstance(parsed_tuple, tuple)
                    and len(parsed_tuple) == EXPECTED_TUPLE_LENGTH
                ):
                    name_val, id_val_raw = parsed_tuple
                    try:
                        id_val = int(id_val_raw)
                        return (str(name_val).strip(), id_val)
                    except (ValueError, TypeError):
                        return (str(name_val).strip(), None)
            except (ValueError, SyntaxError, TypeError):
                pass

            return (None, None)

    ranking_output_df["Winner_tuple_parsed"] = ranking_output_df["Winner"].apply(
        parse_custom_tuple_string
    )
    ranking_output_df["Loser_tuple_parsed"] = ranking_output_df["Loser"].apply(
        parse_custom_tuple_string
    )

    processed_ranking_results = pd.DataFrame()
    processed_ranking_results["Winner"] = ranking_output_df[
        "Winner_tuple_parsed"
    ].apply(lambda x: x[0] if x else None)
    processed_ranking_results["Loser"] = ranking_output_df["Loser_tuple_parsed"].apply(
        lambda x: x[0] if x else None
    )

    processed_ranking_results["Winner ID"] = ranking_output_df[
        "Winner_tuple_parsed"
    ].map(lambda x: x[1] if x and x[1] is not None else pd.NA)
    processed_ranking_results["Loser ID"] = ranking_output_df["Loser_tuple_parsed"].map(
        lambda x: x[1] if x and x[1] is not None else pd.NA
    )

    # Convert to Int64 which supports pd.NA
    processed_ranking_results["Winner ID"] = processed_ranking_results[
        "Winner ID"
    ].astype("Int64")
    processed_ranking_results["Loser ID"] = processed_ranking_results[
        "Loser ID"
    ].astype("Int64")

    game_scores: list[tuple[int, int] | None] = []
    for _, row in processed_ranking_results.iterrows():
        winner_id = row["Winner ID"]
        loser_id = row["Loser ID"]
        if pd.notna(winner_id) and pd.notna(loser_id):
            try:
                game_scores.append((int(winner_id), int(loser_id)))
            except (ValueError, TypeError):
                game_scores.append(None)
        else:
            game_scores.append(None)
    processed_ranking_results["Game Score"] = game_scores

    processed_ranking_results = processed_ranking_results.dropna(subset=["Game Score"])

    if processed_ranking_results.empty:
        return pd.DataFrame(
            columns=["Winner", "Loser", "Winner ID", "Loser ID", "Game Score"]
        )

    return processed_ranking_results


def _validate_llm_evaluation_keys(
    evaluation_result: dict[str, Any], expected_keys: list[str], raw_response: str
) -> None:
    if not all(key in evaluation_result for key in expected_keys):
        raise ValueError(
            "LLM response missing expected keys. Got:"
            f" {list(evaluation_result.keys())}. Parsed JSON: {evaluation_result}. Raw:"
            f" {raw_response[:200]}..."
        )


async def process_comparison_pair(
    pair: tuple[int, int],
    idx: int,
    semaphore: asyncio.Semaphore,
    client: LiteLLMModel,
    system_prompt: str,
    ranking_prompt_format: str,
    hypothesis_df: pd.DataFrame,
) -> dict[str, Any]:

    async with semaphore:
        pair_id_1, pair_id_2 = pair

        hypo_1_info = get_candidate_info(hypothesis_df, pair_id_1)
        hypo_2_info = get_candidate_info(hypothesis_df, pair_id_2)

        user_prompt = f"""Please evaluate and compare the following two candidates based on the criteria provided.

            **Candidate 1 (ID: {hypo_1_info['index']})**
            Name: {hypo_1_info['hypothesis']}
            Reasoning: {hypo_1_info['answer']}

            --- VERSUS ---

            **Candidate 2 (ID: {hypo_2_info['index']})**
            Name: {hypo_2_info['hypothesis']}
            Reasoning: {hypo_2_info['answer']}


            Provide your evaluation STRICTLY in the following JSON format. Do NOT include any text before or after the JSON object.
            Ensure the entire output is a single, valid JSON object

            {ranking_prompt_format}
            """

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = await client.call_single(messages)

        response_content = cast(str, response.text)

        # Attempt to find JSON within the response, even if there's extra text
        json_start = response_content.find("{")
        json_end = response_content.rfind("}")
        if json_start == -1 or json_end == -1 or json_start > json_end:
            raise json.JSONDecodeError(  # pylint: disable=W0715
                "Could not find JSON object markers '{' and '}' in response.",
                response_content,
                0,
            )

        json_string = response_content[json_start : json_end + 1]

        try:
            evaluation_result = json.loads(json_string)
            expected_keys = [
                "Analysis",
                "Reasoning",
                "Winner",
                "Loser",
            ]
            _validate_llm_evaluation_keys(
                evaluation_result, expected_keys, response_content
            )

            return {  # noqa: TRY300
                "status": "success",
                "data": {
                    "pair_index": idx,
                    "pair_ids": pair,
                    "input_hypo_1": hypo_1_info,
                    "input_hypo_2": hypo_2_info,
                    "llm_evaluation": evaluation_result,
                },
            }

        except json.JSONDecodeError as e:
            error_detail = (
                f"JSON Decode Error: {e}. Raw text before parse attempt:"
                f" '{json_string}'"
            )
            error_payload = {"raw_response": response_content}
            logger.exception(
                f"\nJSON Decode Error for pair {pair} (Index {idx})\nRaw Content:"
                f" {response_content}"
            )
            return {
                "status": "error",
                "pair": pair,
                "error": error_detail,
                "details": error_payload,
                "input_hypo_1": hypo_1_info,
                "input_hypo_2": hypo_2_info,
            }
        except ValueError as e:
            error_detail = f"Response Validation Error: {e}"
            error_payload = {"raw_response": response_content}
            logger.exception(
                f"\nValidation Error for pair {pair} (Index {idx})\nRaw Content:"
                f" {response_content}"
            )
            return {
                "status": "error",
                "pair": pair,
                "error": error_detail,
                "details": error_payload,
                "input_hypo_1": hypo_1_info,
                "input_hypo_2": hypo_2_info,
            }


async def run_comparisons(  # noqa: PLR0912
    pairs_list: list[tuple[int, int]],
    client: LiteLLMModel,
    system_prompt: str,
    ranking_prompt_format: str,
    assay_hypothesis_df: pd.DataFrame,
    output_filepath: str,
    max_concurrent_requests: int = 100,
) -> None:
    all_comparison_results = []
    error_log = []

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    logger.info(
        f"Starting comparisons for {len(pairs_list)} pairs with max concurrency"
        f" {max_concurrent_requests}..."
    )

    tasks = [
        process_comparison_pair(
            pair,
            idx,
            semaphore,
            client,
            system_prompt,
            ranking_prompt_format,
            assay_hypothesis_df,
        )
        for idx, pair in enumerate(pairs_list)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Comparing Hypotheses")

    # Process results
    for result in results:
        if result and result["status"] == "success":
            all_comparison_results.append(result["data"])
        elif result and result["status"] == "error":
            error_log.append(result)

    logger.info("\nFinished processing pairs.")
    logger.info(f" - Successful comparisons: {len(all_comparison_results)}")
    logger.info(f" - Errors encountered: {len(error_log)}")

    if not all_comparison_results:
        logger.error("No results to save. CSV file will not be created.")

    else:
        try:
            processed_results = []
            for res_dict in all_comparison_results:
                new_row = res_dict.copy()
                llm_eval_data = new_row.pop("llm_evaluation", {})

                if isinstance(llm_eval_data, str):
                    try:
                        llm_eval_data = ast.literal_eval(llm_eval_data)
                    except (ValueError, SyntaxError):
                        logger.exception(
                            "Warning: Could not parse llm_evaluation string:"
                            f" {llm_eval_data}. Skipping llm_evaluation fields for this"
                            " row."
                        )
                        llm_eval_data = {}

                new_row["Winner"] = llm_eval_data.get(
                    "Winner", ""
                )  # Use .get for safety
                new_row["Loser"] = llm_eval_data.get("Loser", "")
                new_row["Analysis"] = llm_eval_data.get("Analysis", "")
                new_row["Reasoning"] = llm_eval_data.get("Reasoning", "")

                processed_results.append(new_row)

            if not processed_results:
                logger.error(
                    "No results after processing. CSV file will not be created."
                )
                return

            desired_fieldnames = ["Winner", "Loser", "Analysis", "Reasoning"]

            if processed_results:
                original_keys = list(processed_results[0].keys())
                for key in original_keys:
                    if key not in desired_fieldnames:
                        desired_fieldnames.append(key)
            else:
                logger.warning(
                    "Warning: Processed results are empty. Cannot determine all"
                    " fieldnames automatically."
                )

            string_buffer = io.StringIO()
            writer = csv.DictWriter(
                string_buffer,
                fieldnames=desired_fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(processed_results)
            csv_content_string = string_buffer.getvalue()
            string_buffer.close()

            async with aiofiles.open(output_filepath, "w", encoding="utf-8") as csvfile:
                await csvfile.write(csv_content_string)

            logger.info(
                f"Successfully saved {len(all_comparison_results)} results to"
                f" {output_filepath}"
            )

        except IndexError:
            logger.exception(
                "Error: all_comparison_results (or processed_results) is empty, cannot"
                " determine CSV headers."
            )
        except Exception:
            logger.exception("Error saving results to CSV file.")


async def format_single_report(
    report: dict[str, Any], client: LiteLLMModel
) -> dict[str, str]:

    hypothesis_text = report.get("hypothesis", "").strip()
    answer_text = report.get("answer", "").strip()

    sources_data = report.get("sources", [])

    sources_for_prompt_string = ""
    if isinstance(sources_data, list):
        if sources_data:
            sources_for_prompt_string = "\n".join(str(s) for s in sources_data)
    elif isinstance(sources_data, str):
        sources_for_prompt_string = sources_data.strip()
    elif sources_data is not None:
        sources_for_prompt_string = str(sources_data)

    final_report_formatting_user_message = FINAL_REPORT_FORMATTING_USER_MESSAGE.format(
        answer_text=answer_text, sources_text=sources_for_prompt_string
    )

    final_report_formatting_system_message = FINAL_REPORT_FORMATTING_SYSTEM_MESSAGE

    formatting_messages = [
        Message(role="system", content=final_report_formatting_system_message),
        Message(role="user", content=final_report_formatting_user_message),
    ]

    final_report_formatted_result = await client.call_single(formatting_messages)

    return {
        "hypothesis": hypothesis_text,
        "formatted_output": cast(str, final_report_formatted_result.text),
    }


async def format_final_report(
    data_list: list[dict[str, Any]], client: LiteLLMModel
) -> list[dict[str, str]]:

    tasks = [format_single_report(item, client) for item in data_list]

    processed_results = await asyncio.gather(*tasks)
    return list(processed_results)


def extract_candidate_info_from_folder(folder_path: str) -> pd.DataFrame:

    candidate_information = []
    txt_files_found = False

    folder = Path(folder_path)

    if not folder.is_dir():
        logger.error(f"Error: Folder not found at '{folder_path}'")
        return pd.DataFrame(columns=["hypothesis", "answer"])

    txt_files_found = False

    for item in folder.iterdir():
        filename = item.name

        if item.is_file() and item.suffix == ".txt":
            txt_files_found = True
            filepath = item

            logger.info(f"Processing file: {filename}...")

            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()

                candidate_match = re.search(
                    r"Proposal for(.*?)\s*Overview", content, re.DOTALL | re.IGNORECASE
                )
                candidate_text = (
                    candidate_match.group(1).strip() if candidate_match else None
                )
                if candidate_text is None:
                    logger.error(
                        "  - Could not find text between 'Proposal for' and 'Overview'"
                        f" in {filename}"
                    )

                candidate_information.append(
                    {
                        "filename": filename,
                        "hypothesis": candidate_text,
                        "answer": content,
                    }
                )

            except Exception as e:
                logger.exception(f"Error processing file {filename}.")
                candidate_information.append(
                    {
                        "filename": filename,
                        "hypothesis": f"Error: {e}",
                        "answer": f"Error: {e}",
                    }
                )

    if not txt_files_found:
        logger.error(f"No .txt files found in '{folder_path}'")
        return pd.DataFrame(columns=["hypothesis", "answer"])

    candidate_information_df = pd.DataFrame(candidate_information)
    candidate_information_df["index"] = candidate_information_df.index

    return candidate_information_df


def _raise_if_file_not_found(csv_path_str: str) -> None:
    """
    Checks if a file exists at the given path and raises FileNotFoundError if not.

    Args:
        csv_path_str: The string path to the CSV file.
    """
    file_path = Path(csv_path_str)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")


def read_and_process_csv(csv_path: str) -> str | None:
    """Reads CSV, converts to simple HTML, and extracts drug names."""
    try:
        _raise_if_file_not_found(csv_path)
        csv_df = pd.read_csv(csv_path)

        html_output = csv_df.to_html(index=False, na_rep="NA")
        html_output = html_output.replace(
            '<table border="1" class="dataframe">',
            '<table style="border-collapse: collapse; border: 1px solid black;">',
        )
        html_output = html_output.replace('<tr style="text-align: right;">', "<tr>")
        MAX_HTML_OUTPUT_LENGTH = 30000

        if len(html_output) > MAX_HTML_OUTPUT_LENGTH:
            logger.info(
                f"Generated HTML from {csv_path} exceeds max length"
                f" ({len(html_output)} > 30,000). Analysis might be incomplete or fail."
            )

        return html_output  # noqa: TRY300

    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError:
        logger.exception(f"Data file is empty: {csv_path}")
        return None
    except Exception:
        logger.exception(f"Error reading or processing CSV {csv_path}.")
        return None
