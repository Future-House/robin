import copy
import os
import re
from datetime import datetime

from futurehouse_client import FutureHouseClient, JobNames
from lmi import LiteLLMModel
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .prompts import (
    ANALYSIS_QUERIES,
    ASSAY_HYPOTHESIS_FORMAT,
    ASSAY_HYPOTHESIS_SYSTEM_PROMPT,
    ASSAY_LITERATURE_SYSTEM_MESSAGE,
    ASSAY_LITERATURE_USER_MESSAGE,
    ASSAY_PROPOSAL_SYSTEM_MESSAGE,
    ASSAY_PROPOSAL_USER_MESSAGE,
    ASSAY_RANKING_PROMPT_FORMAT,
    ASSAY_RANKING_SYSTEM_PROMPT,
    CANDIDATE_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_GENERATION_USER_MESSAGE,
    CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT,
    CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE,
    CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_RANKING_PROMPT_FORMAT,
    CANDIDATE_RANKING_SYSTEM_PROMPT,
    CANDIDATE_REPORT_FORMAT,
    CHAIN_OF_THOUGHT_AGNOSTIC,
    CONSENSUS_QUERIES,
    COT,
    DATA_INTERPRETATION_CONTENT_MESSAGE,
    DATA_INTERPRETATION_SYSTEM_MESSAGE,
    EXPERIMENTAL_INSIGHTS_APPENDAGE,
    EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION,
    FOLLOWUP_CONTENT_MESSAGE,
    FOLLOWUP_SYSTEM_MESSAGE,
    GENERAL_NOTEBOOK_GUIDELINES,
    GUIDELINE,
    R_SPECIFIC_GUIDELINES,
    SYNTHESIZE_SYSTEM_MESSAGE_CONTENT,
    SYNTHESIZE_USER_CONTENT,
)

_DEFAULT_LLM_CONFIG_DATA = {
    "model_list": [
        {
            "model_name": "o4-mini",
            "litellm_params": {
                "model": "o4-mini",
                "api_key": os.getenv("OPENAI_API_KEY", "insert_openai_key_here"),
                "timeout": 300,
            },
        }
    ]
}


def get_default_llm_config():
    return copy.deepcopy(_DEFAULT_LLM_CONFIG_DATA)


def _get_prompt_args(template_string: str) -> set[str]:
    """
    Extracts root variable names from f-string like placeholders (e.g., {variable})
    using a direct regex approach.
    """  # noqa: D205
    placeholders: set[str] = set()
    placeholders.update(
        match.group(1)
        for match in re.finditer(
            r"(?<!{){([a-zA-Z_][a-zA-Z0-9_]*)[^}]*}(?!})", template_string
        )
    )
    return placeholders


class Prompts(BaseModel):
    analysis_queries: dict[str, str] = Field(default_factory=lambda: ANALYSIS_QUERIES)
    consensus_queries: dict[str, str] = Field(default_factory=lambda: CONSENSUS_QUERIES)
    assay_literature_system_message: str = Field(
        default=ASSAY_LITERATURE_SYSTEM_MESSAGE
    )
    assay_literature_user_message: str = Field(default=ASSAY_LITERATURE_USER_MESSAGE)
    assay_proposal_system_message: str = Field(default=ASSAY_PROPOSAL_SYSTEM_MESSAGE)
    assay_proposal_user_message: str = Field(default=ASSAY_PROPOSAL_USER_MESSAGE)
    assay_hypothesis_system_prompt: str = Field(default=ASSAY_HYPOTHESIS_SYSTEM_PROMPT)
    assay_hypothesis_format: str = Field(default=ASSAY_HYPOTHESIS_FORMAT)
    assay_ranking_system_prompt: str = Field(default=ASSAY_RANKING_SYSTEM_PROMPT)
    assay_ranking_prompt_format: str = Field(default=ASSAY_RANKING_PROMPT_FORMAT)
    synthesize_user_content: str = Field(default=SYNTHESIZE_USER_CONTENT)
    synthesize_system_message_content: str = Field(
        default=SYNTHESIZE_SYSTEM_MESSAGE_CONTENT
    )
    candidate_query_generation_system_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE
    )
    experimental_insights_appendage: str = Field(
        default=EXPERIMENTAL_INSIGHTS_APPENDAGE
    )
    candidate_query_generation_content_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE
    )
    candidate_generation_system_message: str = Field(
        default=CANDIDATE_GENERATION_SYSTEM_MESSAGE
    )
    candidate_generation_user_message: str = Field(
        default=CANDIDATE_GENERATION_USER_MESSAGE
    )
    experimental_insights_for_candidate_generation: str = Field(
        default=EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION
    )
    candidate_lit_review_direction_prompt: str = Field(
        default=CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT
    )
    candidate_report_format: str = Field(default=CANDIDATE_REPORT_FORMAT)
    candidate_ranking_system_prompt: str = Field(
        default=CANDIDATE_RANKING_SYSTEM_PROMPT
    )
    candidate_ranking_prompt_format: str = Field(
        default=CANDIDATE_RANKING_PROMPT_FORMAT
    )
    cot: str = Field(default=COT)
    guideline: str = Field(default=GUIDELINE)
    data_interpretation_system_message: str = Field(
        default=DATA_INTERPRETATION_SYSTEM_MESSAGE
    )
    data_interpretation_content_message: str = Field(
        default=DATA_INTERPRETATION_CONTENT_MESSAGE
    )
    followup_system_message: str = Field(default=FOLLOWUP_SYSTEM_MESSAGE)
    followup_content_message: str = Field(default=FOLLOWUP_CONTENT_MESSAGE)
    general_notebook_guidelines: str = Field(default=GENERAL_NOTEBOOK_GUIDELINES)
    r_specific_guidelines: str = Field(default=R_SPECIFIC_GUIDELINES)
    cot_agnostic: str = Field(default=CHAIN_OF_THOUGHT_AGNOSTIC)

    @model_validator(mode="after")
    def validate_all_prompts(self) -> "Prompts":
        current_prompt_expectations: dict[str, set[str]] = {
            "data_interpretation_content_message": {"goal", "data_html"},
            "followup_content_message": {
                "goal",
                "analysis_summary",
                "mechanistic_insights",
                "questions_raised",
            },
            "assay_literature_system_message": {"num_assays"},
            "assay_literature_user_message": {"num_queries", "disease_name"},
            "assay_proposal_system_message": {"num_assays"},
            "assay_proposal_user_message": {
                "num_assays",
                "disease_name",
                "assay_lit_review_output",
            },
            "assay_hypothesis_system_prompt": {"disease_name"},
            "assay_hypothesis_format": {"disease_name"},
            "assay_ranking_system_prompt": {"disease_name"},
            "synthesize_user_content": {"assay_name", "disease_name"},
            "synthesize_system_message_content": {"disease_name"},
            "candidate_query_generation_system_message": {"disease_name"},
            "experimental_insights_appendage": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_query_generation_content_message": {
                "num_queries",
                "double_queries",
                "candidate_generation_goal",
                "disease_name",
            },
            "candidate_generation_system_message": {"disease_name", "num_candidates"},
            "candidate_generation_user_message": {
                "num_candidates",
                "disease_name",
                "therapeutic_candidate_review_output",
            },
            "experimental_insights_for_candidate_generation": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_lit_review_direction_prompt": {"disease_name"},
            "candidate_report_format": {"disease_name"},
            "candidate_ranking_system_prompt": {"disease_name"},
            "cot": set(),
            "guideline": set(),
            "assay_ranking_prompt_format": set(),
            "candidate_ranking_prompt_format": set(),
            "data_interpretation_system_message": set(),
            "followup_system_message": set(),
            "analysis_queries": set(),
            "consensus_queries": set(),
        }

        for field_name, expected_args in current_prompt_expectations.items():
            if not hasattr(self, field_name):
                raise ValueError(
                    f"Prompt field '{field_name}' defined in PROMPT_EXPECTATIONS but"
                    " not found in Prompts model."
                )

            prompt_template_value = getattr(self, field_name)

            if isinstance(prompt_template_value, dict):
                continue

            if not isinstance(prompt_template_value, str):
                raise TypeError(f"Prompt field '{field_name}' is not a string type.")

            actual_placeholders = _get_prompt_args(prompt_template_value)

            missing_in_template = expected_args - actual_placeholders
            if missing_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' is missing expected placeholders:"
                    f" {missing_in_template}. Expected: {sorted(expected_args)}, Found:"
                    f" {sorted(actual_placeholders)}"
                )

            unexpected_in_template = actual_placeholders - expected_args
            if unexpected_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' contains unexpected placeholders:"
                    f" {unexpected_in_template}. Expected: {sorted(expected_args)},"
                    f" Found: {sorted(actual_placeholders)}"
                )

        return self


class AgentConfig(BaseModel):
    assay_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for literature search during assay idea generation.",
    )
    assay_hypothesis_report_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for generating detailed reports on assay hypotheses.",
    )
    candidate_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description=(
            "Agent to use for literature search during therapeutic candidate idea"
            " generation."
        ),
    )
    candidate_hypothesis_report_agent: JobNames = Field(
        default=JobNames.FALCON,
        description=(
            "Agent to use for generating detailed reports on therapeutic candidates."
        ),
    )


class RobinConfiguration(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    prompts: Prompts = Field(default_factory=Prompts)
    num_queries: int = Field(
        default=3,
        description=(
            "Number of queries to generate for each step, more means more data but also"
            " more cost."
        ),
    )
    num_assays: int = Field(default=3, description="Number of assay to generate.")
    num_candidates: int = Field(
        default=5, description="Number of candidates to generate for each query."
    )
    disease_name: str = Field(
        default="input_disease", description="Name of the disease to focus on."
    )
    run_folder_name: str | None = Field(
        default=None,
        description=(
            "Name of the folder where results will be stored. "
            "If not provided or None, it will be auto-generated "
            "using the disease_name and the timestamp."
        ),
    )
    futurehouse_api_key: str = "insert_futurehouse_api_key_here"
    llm_name: str = "o4-mini"
    llm_config: dict | None = Field(default_factory=get_default_llm_config)
    agent_settings: AgentConfig = Field(default_factory=AgentConfig)
    _fh_client: FutureHouseClient | None = PrivateAttr(default=None)
    _llm_client: LiteLLMModel | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_run_folder_name_default(self) -> "RobinConfiguration":
        if self.run_folder_name is None:
            disease_part = self.disease_name[:70].replace(" ", "_")
            timestamp_part = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_folder_name = f"{disease_part}_{timestamp_part}"
        return self

    @property
    def fh_client(self) -> FutureHouseClient:
        if self._fh_client is None:
            api_key = os.getenv("FUTUREHOUSE_API_KEY") or self.futurehouse_api_key
            if not api_key:
                raise ValueError(
                    "FutureHouse API key is not set. Please provide it in the"
                    " configuration or set FUTUREHOUSE_API_KEY env variable."
                )
            self._fh_client = FutureHouseClient(api_key=api_key)
        return self._fh_client

    @property
    def llm_client(self) -> LiteLLMModel:
        if self._llm_client is None:
            self._llm_client = LiteLLMModel(name=self.llm_name, config=self.llm_config)
        return self._llm_client

    def get_da_client(self):
        from .multitrajectory_runner import MultiTrajectoryRunner

        return MultiTrajectoryRunner(configuration=self)
