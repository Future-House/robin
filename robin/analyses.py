import logging
from pathlib import Path
from typing import cast

from aviary.core import Message

from .configuration import RobinConfiguration
from .multitrajectory_runner import Step, StepConfig
from .utils import read_and_process_csv

logger = logging.getLogger(__name__)

PARALLEL_ANALYSIS = 5
EXPECTED_RESPONSE_LENGTH = 4


async def data_analysis(
    data_path: str,
    data_analysis_type: str,
    goal: str,
    configuration: RobinConfiguration,
) -> dict[str, str]:

    data_analyzer = configuration.get_da_client()

    analysis_query = configuration.prompts.analysis_queries[data_analysis_type]

    consensus_query = configuration.prompts.consensus_queries[data_analysis_type]

    CoT = configuration.prompts.cot

    guideline = configuration.prompts.guideline

    analysis_prompt = f"""\
    Here is the user query to address:
    {CoT}
    {guideline}
    <query>
    {analysis_query}
    </query>
    """

    consensus_prompt = f"""\
    Here is the user query to address:
    {CoT}
    {guideline}
    <query>
    {consensus_query}
    </query>
    """

    output_dir = f"robin_output/{configuration.run_folder_name}/data_analysis"

    # Step 1: Gating, MFI and statistical analysis
    analysis_step = Step(
        name="job-futurehouse-data-analysis-crow-high",
        prompt_template=analysis_prompt,
        cot_prompt=False,
        input_files={data_path: "flow_250508/"},  # change this to your input folder
        output_files={"flow_results.csv": "flow_results/flow_results.csv"},
        parallel=PARALLEL_ANALYSIS,
        config=StepConfig(language="R", max_steps=30, timeout=15 * 60),
    )
    data_analyzer.add_step(analysis_step)

    # Step 2: Consensus Analysis
    consensus_step = Step(
        name="job-futurehouse-data-analysis-crow-high",
        prompt_template=consensus_prompt,
        cot_prompt=False,
        input_files={
            f"{output_dir}/{analysis_step.step_id}/flow_results": "flow_results/"
        },
        output_files={"consensus_results.csv": "consensus_results.csv"},
        config=StepConfig(language="R", max_steps=30, timeout=15 * 60),
    )
    data_analyzer.add_step(consensus_step)

    await data_analyzer.run_pipeline(output_dir)

    logger.info(
        "View the final volcano plot at:"
        f" https://platform.edisonscientific.com/trajectories/{data_analyzer.results[consensus_step.step_id]['task_ids'][0]}"
    )

    consensus_output_filename = consensus_step.output_files["consensus_results.csv"]

    output_dir_path = Path(output_dir)
    relative_consensus_output_path = (
        output_dir_path / consensus_step.step_id / consensus_output_filename
    )
    full_consensus_output_path = relative_consensus_output_path.resolve()

    logger.info(
        f"The full path to the consensus results CSV is: {full_consensus_output_path}"
    )

    processed_data = read_and_process_csv(str(full_consensus_output_path))

    if processed_data is None:
        return {
            "status": "error",
            "message": (
                f"Failed to read or process data file: {full_consensus_output_path}."
                " Check logs for details."
            ),
        }

    data_html = processed_data

    # Check if HTML is empty (might happen if CSV was empty or only headers)
    if "<td>" not in data_html:
        logger.error(
            "No data rows found or converted to HTML from"
            f" {full_consensus_output_path}."
        )
        return {
            "analysis_summary": "No data rows found in the provided file to analyze.",
            "mechanistic_insights": (
                "No data rows found in the provided file to analyze."
            ),
            "questions_raised": "No data rows found in the provided file to analyze.",
            "followup_suggestions": (
                "No data rows found in the provided file to analyze."
            ),
        }

    logger.info("Sending data review prompt to LLM.")

    data_interpretation_system_message = (
        configuration.prompts.data_interpretation_system_message
    )
    data_interpretation_content_message = (
        configuration.prompts.data_interpretation_content_message.format(
            goal=goal, data_html=data_html
        )
    )

    data_interpretation_messages = [
        Message(role="system", content=data_interpretation_system_message),
        Message(role="user", content=data_interpretation_content_message),
    ]

    data_interpretation_result = await configuration.llm_client.call_single(
        data_interpretation_messages
    )

    data_interpretation_result_text = cast(str, data_interpretation_result.text)
    data_interpretation_response = data_interpretation_result_text.split("<>")

    logger.info(
        f"Received LLM response for data review: {data_interpretation_response}"
    )

    if len(data_interpretation_response) != EXPECTED_RESPONSE_LENGTH:
        logger.error(
            "Error in the LLM response parsing. Four items not found in the LLM"
            " response"
        )
        return {
            "analysis_summary": "Error in the LLM response parsing.",
            "mechanistic_insights": "Error in the LLM response parsing.",
            "questions_raised": "Error in the LLM response parsing.",
            "followup_suggestions": "Error in the LLM response parsing.",
        }

    drugs_in_data = data_interpretation_response[0]
    analysis_summary = data_interpretation_response[1]
    questions_raised = data_interpretation_response[2]
    mechanistic_insights = data_interpretation_response[3]

    followup_system_message = configuration.prompts.followup_system_message

    followup_content_message = configuration.prompts.followup_content_message.format(
        goal=goal,
        analysis_summary=analysis_summary,
        mechanistic_insights=mechanistic_insights,
        questions_raised=questions_raised,
    )

    followup_messages = [
        Message(role="system", content=followup_system_message),
        Message(role="user", content=followup_content_message),
    ]

    followup_result = await configuration.llm_client.call_single(followup_messages)

    followup_suggestions = cast(str, followup_result.text)

    analysis_summary = (
        analysis_summary
        + " These drugs have already been tested: "
        + drugs_in_data
        + " AS THEY HAVE BEEN TESTED, DO NOT SUGGEST THESE DRUGS AGAIN."
    )

    return {
        "analysis_summary": analysis_summary,
        "mechanistic_insights": mechanistic_insights,
        "questions_raised": questions_raised,
        "followup_suggestions": followup_suggestions,
    }
