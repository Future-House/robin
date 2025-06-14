
import logging
from typing import cast
from aviary.core import Message
import re

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

#______GENERATING HYPOTHESES______
def _split_out_hypotheses_and_rationale(hypotheses_output:list[str])-> list[str]:
    all_hypotheses = []
    # Only capturing Hypothesis & Rationale 
    # Ignoring mesh terms & Therapeutic Implications for now 
    pattern = r"\*\*\s*(Hypothesis|Rationale)\s*\d*\s*[:]*\s*\*\*\s*(.*?)(?=\*\*\s*(Hypothesis|Rationale|\w.*?)\s*\d*\s*[:]*\s*\*\*|$)"

    for h in hypotheses_output:
        try:
            matches = re.findall(pattern, h, flags=re.IGNORECASE | re.DOTALL)

            hypotheses_output = [match[1].strip() for match in matches if match[0].lower() == "hypothesis"]
            rationale_output = [match[1].strip() for match in matches if match[0].lower() == "rationale"]

            all_hypotheses.extend(f"Hypothesis: {hyp.strip()}\nRationale: {rationale.strip()}" for hyp, rationale in zip(hypotheses_output, rationale_output))
        except Exception as e:
            logger.error(f"Error processing hypothesis output: {e}")
            continue
    return all_hypotheses


async def generate_hypotheses_from_mesh(configuration: RobinConfiguration)-> dict[str, str] | None:
    """
    Generate hypotheses based on MeSH terms for the specified disease.

    Args:
        configuration: The RobinConfiguration object for the run.

    Returns:
        a dictionary of hypothesis: hypothesis (intentionally redundant).
    """
    mesh_terms = get_mesh_terms(configuration.disease_name, shuffle_terms=True)
    if not mesh_terms:
        #TODO: maybe have an LLM rejig the disease name -OR- skip this step
        logger.error("No MeSH terms found for the specified disease.")
        return None
    logger.info("\n\nStep 1: Generating hypotheses from MeSH terms...")

    mesh_term_system_message = (
        configuration.prompts.mesh_term_system_message.format(
            disease_name=configuration.disease_name,
            num_mesh_hypotheses=configuration.num_mesh_hypotheses,
        )
    )

    mesh_term_user_message = (
        configuration.prompts.mesh_term_user_message.format(
            disease_name=configuration.disease_name,
            mesh_terms=", ".join(mesh_terms),
        )
    )

    mesh_term_hypotheses_query_messages = [
        Message(role="system", content=mesh_term_system_message),
        Message(role="user", content=mesh_term_user_message),
    ]

    mesh_term_hypotheses_query_result = await configuration.llm_client.call_single(
        mesh_term_hypotheses_query_messages
    )

    mesh_term_hypotheses_query_result_text = cast(str, mesh_term_hypotheses_query_result.text)
    mesh_term_hypotheses = mesh_term_hypotheses_query_result_text.split("<>")

    logger.info("Hypotheses Generated from MeSH Terms:")
    for ia, hyp_gen in enumerate(mesh_term_hypotheses):
        logger.info(f"{ia + 1}. {hyp_gen}")
    
    mesh_term_hypotheses = _split_out_hypotheses_and_rationale(mesh_term_hypotheses)
    return {q: q for q in mesh_term_hypotheses}


# ______LIT REVIEW______

# TODO: move this to prompts & config
MESH_HYPOTHESIS_LITERATURE_REVIEW_PROMPT = "Given the following hypothesis (and rationale) about the pathogenesis of {disease_name}, find scientific evidence to support and/or refute this hypothesis.\nFormat your response with the following headings: **Supporting evidence**: <supporting info>, **Countering evidence**: <refutation>, **Conclusion**: <conclusion>\n\nHypothesis: {hyp_and_rationale}" 

def make_mesh_hypotheses_queries(hyp_and_rationale:list[str], configuration: RobinConfiguration)-> list[str]:
    all_queries = [MESH_HYPOTHESIS_LITERATURE_REVIEW_PROMPT.format(hyp_and_rationale=hyp, disease_name=configuration.disease_name) for hyp in hyp_and_rationale]
    return all_queries

async def mesh_hypotheses_lit_review(
    configuration: RobinConfiguration, mesh_term_hypotheses: dict[str, str]
) -> str:
    """
    Conduct a literature review for the MeSH generated hypotheses using the FutureHouse platform.

    Args:
        configuration: The RobinConfiguration object for the run.
        mesh_term_hypotheses: A dictionary of MeSH term hypotheses.

    Returns:
        A string containing the literature review results.
    """
    logger.info("\n\nStep 2: Conducting literature search with FutureHouse platform...")

    mesh_term_hypotheses_queries_list = make_mesh_hypotheses_queries(mesh_term_hypotheses, configuration)
    mesh_term_hypotheses_queries_dict = {q: q for q in mesh_term_hypotheses_queries_list}

    mesh_hyp_lit_review = await call_platform(
        queries=mesh_term_hypotheses_queries_dict,
        fh_client=configuration.fh_client,
        job_name=configuration.agent_settings.mesh_hyp_lit_search_agent, 
    )

    mesh_hyp_lit_review_results = mesh_hyp_lit_review["results"]

    save_crow_files(
        mesh_hyp_lit_review_results,
        run_dir=f"robin_output/{configuration.run_folder_name}/mesh_hypotheses_literature_reviews",
        prefix="query",
    )

    return output_to_string(mesh_hyp_lit_review_results)


# ______RANKING______
def get_hypothesis_from_query(text:str) -> str:
    return text.split("Hypothesis: Hypothesis:")[1].strip()

def get_conclusion(text: str) -> str:
    conclusion_pattern = r"\*{0,2}Conclusion\*{0,2}:?"
    parts = re.split(conclusion_pattern, text, maxsplit=1)
    return parts[1].strip() 

def extract_conclusions(text:str) -> list[str]:
    pattern = r"Query\s*\d+\s*:(.*?)Answer\s*\d+\s*:(.*?)(?=Query\s*\d+\s*:|References\s*\d+\s*:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    all_hyp_conclusions = []
    for q, a in matches:
        try:
            query = get_hypothesis_from_query(q)
            conclusion = get_conclusion(a)
            all_hyp_conclusions.append(f"Hypothesis: {query}\nConclusion: {conclusion}")
        except Exception as e:
            print(f"Error processing query and answer: {e}... skipping")
    return all_hyp_conclusions

async def mesh_hypotheses_ranking(configuration: RobinConfiguration, hypotheses_and_conclusions: list[str]) -> str:
    """
    Choose the best hypothesis based on the literature review results.

    Args:
        configuration: The RobinConfiguration object for the run.
        hypotheses_and_conclusions: A list of hypotheses and their conclusions.

    Returns:
        A string containing the top hypothesis & reasoning.
        
    """

    logger.info("\n\nStep 3: Choosing the best hypotheses based on literature review...")

    mesh_hypothesis_ranking_system_message = (
        configuration.prompts.mesh_hypothesis_ranking_system_message.format(
            disease_name=configuration.disease_name,
        )
    )

    mesh_hypothesis_ranking_user_message = (
        configuration.prompts.mesh_hypothesis_ranking_user_message.format(
            disease_name=configuration.disease_name,
            hypotheses_and_conclusions="\n".join(hypotheses_and_conclusions),
        )
    )

    mesh_hypothesis_ranking_query_messages = [
        Message(role="system", content=mesh_hypothesis_ranking_system_message),
        Message(role="user", content=mesh_hypothesis_ranking_user_message),
    ]
    mesh_hypothesis_ranking_query_result = await configuration.llm_client.call_single(
        mesh_hypothesis_ranking_query_messages
    )
    mesh_hypothesis_ranking_query_result_text = cast(str, mesh_hypothesis_ranking_query_result.text)
    return mesh_hypothesis_ranking_query_result_text