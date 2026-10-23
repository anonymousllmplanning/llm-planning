"""
Pydantic Schema Definitions for LLM Planning Output.

These schemas define the expected output format for planning tasks and can be
converted to JSON Schema for SGLang guided decoding (xgrammar).

Usage:
    from src.inference.schemas import get_planning_schema, PlanningOutput

    # Get JSON Schema for SGLang guided decoding
    schema = get_planning_schema(dataset="gaia")

    # Validate output
    output = PlanningOutput.model_validate(parsed_json)

    # Get GAIA tool IDs
    from src.inference.schemas import GaiaToolId
    valid_tools = [t.value for t in GaiaToolId]
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class StepType(str, Enum):
    """Type of step in the plan DAG."""
    TOOL = "tool"
    THOUGHT = "thought"
    ACTION = "action"


class EdgeType(str, Enum):
    """Type of edge in the plan DAG."""
    DATA_DEP = "data_dep"
    CONTROL_DEP = "control_dep"
    DEPENDENCY = "dependency"


class AnswerType(str, Enum):
    """Type of final answer."""
    NONE = "none"
    STRING = "string"
    NUMBER = "number"


class GaiaToolId(str, Enum):
    """
    Enumeration of the 18 standard GAIA benchmark tools.

    These are the ONLY valid tool_ids for GAIA dataset tasks.
    Use this enum for strict schema validation in SGLang xgrammar decoding.
    """
    # Web & Search Tools
    WEB_SEARCH = "web_search"
    WEB_BROWSER = "web_browser"
    WIKIPEDIA_SEARCH = "wikipedia_search"

    # Document & File Tools
    PDF_READER = "pdf_reader"
    EXCEL_READER = "excel_reader"
    FILE_READER = "file_reader"
    PPTX_READER = "pptx_reader"
    ZIP_EXTRACTOR = "zip_extractor"

    # Media Analysis Tools
    IMAGE_RECOGNITION = "image_recognition"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VIDEO_ANALYSIS = "video_analysis"

    # Computation Tools
    PYTHON_EXECUTOR = "python_executor"
    CODE_INTERPRETER = "code_interpreter"
    CALCULATOR = "calculator"

    # Specialized Tools
    PDB_ANALYZER = "pdb_analyzer"
    DATE_CALCULATOR = "date_calculator"
    UNIT_CONVERTER = "unit_converter"

    # Reasoning (internal)
    REASONING = "reasoning"


# Export list of valid GAIA tool IDs for easy access
GAIA_TOOL_IDS: List[str] = [t.value for t in GaiaToolId]


# =============================================================================
# Core Schema Components
# =============================================================================

class PlanNode(BaseModel):
    """A single node in the plan DAG."""
    node_id: str = Field(description="Unique identifier for this node (e.g., 'n0', 'n1')")
    step_index: int = Field(ge=0, description="Zero-based index of this step")
    label: str = Field(min_length=1, description="Human-readable description of this step")
    step_type: StepType = Field(default=StepType.TOOL, description="Type of step")
    tool_id: Optional[str] = Field(default=None, description="Tool to use for this step")
    arguments: Union[Dict[str, Any], List[Any]] = Field(
        default_factory=dict,
        description="Arguments for the tool"
    )
    output_vars: List[str] = Field(
        default_factory=list,
        description="Output variable names from this step"
    )
    needs_tool: bool = Field(default=True, description="Whether this step requires a tool")
    needs_new_tool: bool = Field(default=False, description="Whether a new tool is needed")

    class Config:
        extra = "allow"  # Allow additional fields


class PlanNodeGaia(BaseModel):
    """
    A single node in the plan DAG for GAIA dataset.

    Enforces that tool_id must be one of the 18 standard GAIA tools.
    """
    node_id: str = Field(description="Unique identifier for this node (e.g., 'n0', 'n1')")
    step_index: int = Field(ge=0, description="Zero-based index of this step")
    label: str = Field(min_length=1, description="Human-readable description of this step")
    step_type: StepType = Field(default=StepType.TOOL, description="Type of step")
    tool_id: Optional[GaiaToolId] = Field(default=None, description="Tool to use (must be from GAIA tool set)")
    arguments: Union[Dict[str, Any], List[Any]] = Field(
        default_factory=dict,
        description="Arguments for the tool"
    )
    output_vars: List[str] = Field(
        default_factory=list,
        description="Output variable names from this step"
    )
    needs_tool: bool = Field(default=True, description="Whether this step requires a tool")
    needs_new_tool: bool = Field(default=False, description="Whether a new tool is needed")

    class Config:
        extra = "allow"


class PlanEdge(BaseModel):
    """An edge representing dependency between nodes."""
    source: Union[str, List[str]] = Field(description="Source node ID(s)")
    target: Union[str, List[str]] = Field(description="Target node ID(s)")
    edge_type: EdgeType = Field(default=EdgeType.DATA_DEP, description="Type of dependency")

    class Config:
        extra = "allow"


class PlanDAG(BaseModel):
    """Directed Acyclic Graph representing the execution plan."""
    nodes: List[PlanNode] = Field(default_factory=list, description="Plan steps")
    edges: List[PlanEdge] = Field(default_factory=list, description="Dependencies between steps")


class PlanDAGGaia(BaseModel):
    """
    Directed Acyclic Graph for GAIA dataset with strict tool validation.

    Uses PlanNodeGaia which enforces GAIA tool_id constraints.
    """
    nodes: List[PlanNodeGaia] = Field(default_factory=list, description="Plan steps (GAIA tools only)")
    edges: List[PlanEdge] = Field(default_factory=list, description="Dependencies between steps")


class ToolArgument(BaseModel):
    """A single argument for a tool call."""
    name: str = Field(description="Argument name")
    value: Any = Field(description="Argument value")


class ToolCall(BaseModel):
    """A single tool call in the execution plan."""
    tool_id: str = Field(description="ID of the tool to call")
    call_index: Optional[int] = Field(default=None, description="Order of this call")
    node_id: Optional[str] = Field(default=None, description="Associated plan node ID")
    arguments: List[ToolArgument] = Field(
        default_factory=list,
        description="Arguments for the tool"
    )

    class Config:
        extra = "allow"


class ToolCallGaia(BaseModel):
    """
    Tool call for GAIA dataset with strict tool_id validation.

    Enforces that tool_id must be one of the 18 standard GAIA tools.
    """
    tool_id: GaiaToolId = Field(description="ID of the tool to call (must be from GAIA tool set)")
    call_index: Optional[int] = Field(default=None, ge=0, description="Order of this call")
    node_id: Optional[str] = Field(default=None, description="Associated plan node ID")
    arguments: List[ToolArgument] = Field(
        default_factory=list,
        description="Arguments for the tool"
    )

    class Config:
        extra = "allow"


class ToolCallNoArgs(BaseModel):
    """Tool call without arguments (for Delta dataset)."""
    tool_id: str = Field(description="ID of the tool to call")
    call_index: Optional[int] = Field(default=None, description="Order of this call")
    node_id: Optional[str] = Field(default=None, description="Associated plan node ID")
    arguments: List[Any] = Field(
        default_factory=list,
        max_length=0,
        description="Arguments (must be empty for Delta)"
    )

    class Config:
        extra = "allow"


class FinalAnswer(BaseModel):
    """Final answer to the user's query."""
    answer_type: AnswerType = Field(default=AnswerType.NONE, description="Type of answer")
    answer: Optional[Union[str, int, float]] = Field(default=None, description="The answer value")
    aliases: List[str] = Field(default_factory=list, description="Alternative phrasings")
    tolerance: float = Field(default=0.0, ge=0.0, description="Numeric tolerance for matching")


class FinalAnswerRequired(BaseModel):
    """
    Final answer with required concrete value (for GAIA dataset).

    This class enforces strict validation to ensure:
    1. answer_type is not 'none' (GAIA requires concrete answers)
    2. answer is not a placeholder or template text
    3. answer has meaningful content
    """
    answer_type: AnswerType = Field(description="Type of answer (cannot be 'none')")
    answer: Union[str, int, float] = Field(description="The answer value (required)")
    aliases: List[str] = Field(default_factory=list, description="Alternative phrasings")
    tolerance: float = Field(default=0.0, ge=0.0, description="Numeric tolerance for matching")

    @field_validator("answer_type")
    @classmethod
    def answer_type_not_none(cls, v: AnswerType) -> AnswerType:
        if v == AnswerType.NONE:
            raise ValueError("answer_type cannot be 'none' for GAIA dataset - a concrete answer is required")
        return v

    @field_validator("answer")
    @classmethod
    def answer_not_placeholder(cls, v: Union[str, int, float]) -> Union[str, int, float]:
        """
        Reject placeholder answers with comprehensive detection.

        Strictly forbids:
        - Angle bracket templates: <PUT YOUR ANSWER>, <RESULT>, etc.
        - Placeholder text: YOUR_ANSWER, TOOL_NAME, etc.
        - Generic templates: "answer here", "computed result"
        - Empty or whitespace-only answers
        """
        if isinstance(v, str):
            # Check for empty/whitespace
            if not v.strip():
                raise ValueError("Answer cannot be empty or whitespace-only")

            v_upper = v.upper().strip()

            # Comprehensive placeholder pattern list
            placeholder_patterns = [
                # Angle bracket patterns (most common)
                "<",
                ">",
                # Explicit placeholder text
                "PUT YOUR",
                "YOUR_ANSWER",
                "YOUR_EXACT_ANSWER",
                "YOUR_COMPUTED",
                "TOOL_NAME",
                "EXAMPLE",
                "PLACEHOLDER",
                "COMPUTED RESULT",
                "ANSWER HERE",
                "RESULT HERE",
                "INSERT",
                "FILL IN",
                # Template indicators
                "[YOUR",
                "[ANSWER",
                "[RESULT",
                "{YOUR",
                "{ANSWER",
                "{RESULT",
                # Generic non-answers
                "TODO",
                "TBD",
                "N/A",
                "NOT AVAILABLE",
                "UNKNOWN",
                "UNABLE TO",
                "CANNOT DETERMINE",
            ]

            for pattern in placeholder_patterns:
                if pattern in v_upper:
                    raise ValueError(
                        f"Answer appears to be a placeholder (contains '{pattern}'): {v[:80]}..."
                        if len(v) > 80 else f"Answer appears to be a placeholder: {v}"
                    )

            # Check for suspicious short generic answers
            suspicious_generic = ["ANSWER", "RESULT", "OUTPUT", "VALUE", "RESPONSE"]
            if v_upper in suspicious_generic:
                raise ValueError(f"Answer is too generic: '{v}' - provide a specific answer")

        return v

    @model_validator(mode="after")
    def validate_answer_type_matches_value(self):
        """Ensure answer_type is consistent with the actual answer value."""
        if self.answer_type == AnswerType.NUMBER:
            if isinstance(self.answer, str):
                # Try to parse as number
                try:
                    float(self.answer)
                except ValueError:
                    raise ValueError(
                        f"answer_type is 'number' but answer '{self.answer}' is not numeric"
                    )
        return self


# =============================================================================
# Complete Planning Output Schemas
# =============================================================================

class PlanningOutput(BaseModel):
    """Complete planning output for general datasets (TaskBench, UltraTool)."""
    plan_dag: PlanDAG = Field(description="Execution plan DAG")
    tool_calls: List[ToolCall] = Field(description="List of tool calls")
    final_answer: FinalAnswer = Field(description="Final answer")

    # Optional metadata
    model_name: Optional[str] = Field(default=None, description="Model that generated this")

    class Config:
        extra = "allow"


class PlanningOutputGaia(BaseModel):
    """
    Planning output for GAIA dataset with strict validation.

    Key constraints:
    1. tool_calls must use only the 18 standard GAIA tools
    2. final_answer is required (no 'none' type, no placeholders)
    3. Validates answer consistency

    Use get_planning_schema("gaia") to export JSON schema for SGLang xgrammar.
    """
    plan_dag: PlanDAG = Field(description="Execution plan DAG")
    tool_calls: List[ToolCall] = Field(description="List of tool calls")
    final_answer: FinalAnswerRequired = Field(description="Final answer (required, no placeholders)")

    model_name: Optional[str] = Field(default=None, description="Model that generated this")

    class Config:
        extra = "allow"

    @field_validator("tool_calls")
    @classmethod
    def validate_gaia_tools(cls, v: List[ToolCall]) -> List[ToolCall]:
        """Validate that all tool_ids are from the GAIA tool set."""
        for tc in v:
            if tc.tool_id and tc.tool_id not in GAIA_TOOL_IDS:
                raise ValueError(
                    f"Invalid tool_id '{tc.tool_id}' - must be one of: {', '.join(GAIA_TOOL_IDS)}"
                )
        return v


class PlanningOutputGaiaStrict(BaseModel):
    """
    Strictly typed planning output for GAIA with enum-enforced tools.

    This version uses GaiaToolId enum for maximum type safety.
    Best for SGLang xgrammar decoding where tool_id must be constrained.
    """
    plan_dag: PlanDAGGaia = Field(description="Execution plan DAG (GAIA tools only)")
    tool_calls: List[ToolCallGaia] = Field(description="List of tool calls (GAIA tools only)")
    final_answer: FinalAnswerRequired = Field(description="Final answer (required, no placeholders)")

    model_name: Optional[str] = Field(default=None, description="Model that generated this")

    class Config:
        extra = "allow"


class PlanningOutputDelta(BaseModel):
    """Planning output for Delta dataset (no arguments)."""
    plan_dag: PlanDAG = Field(description="Execution plan DAG")
    tool_calls: List[ToolCallNoArgs] = Field(description="List of tool calls (no args)")
    final_answer: FinalAnswer = Field(description="Final answer")

    model_name: Optional[str] = Field(default=None, description="Model that generated this")

    class Config:
        extra = "allow"


# =============================================================================
# Multi-Stage Pipeline Schemas
# =============================================================================

class AbstractStep(BaseModel):
    """A step in the abstract plan (no tools)."""
    step_index: int = Field(ge=0, description="Zero-based index of this step")
    description: str = Field(description="Description of what needs to be done in this step")
    
    class Config:
        extra = "allow"


class AbstractPlan(BaseModel):
    """
    Abstract plan for Stage 1: Plan Annotation (ToDo Writer).
    Focuses on logical steps without specific tool assignments.
    """
    steps: List[AbstractStep] = Field(description="List of logical steps to complete the task")
    abs_plan_dag: Optional[PlanDAG] = Field(
        default=None,
        description="Abstract dependency DAG over the same logical steps, without tool assignments",
    )
    valid_execution_order: Optional[List[str]] = Field(
        default=None,
        description="One valid topological order over abs_plan_dag nodes",
    )
    reasoning: Optional[str] = Field(default=None, description="High-level reasoning for the plan")

    class Config:
        extra = "allow"


class ToolDefinition(BaseModel):
    """
    Definition of a new tool for Stage 2: Tool Creation.
    """
    tool_id: str = Field(description="Unique identifier for the tool")
    description: str = Field(description="Clear description of what the tool does")
    arguments: Dict[str, str] = Field(
        description="Dictionary of argument names and their type descriptions (e.g., {'query': 'string'})"
    )

    class Config:
        extra = "allow"


class ToolCreationOutput(BaseModel):
    """Output for Stage 2: Tool Creation."""
    reasoning: str = Field(description="Analysis of why existing tools are insufficient")
    new_tools: List[ToolDefinition] = Field(default_factory=list, description="List of newly created tools")

    class Config:
        extra = "allow"


# =============================================================================
# Schema Generation Functions
# =============================================================================

def get_planning_schema(dataset: str, strict: bool = False) -> Dict[str, Any]:
    """
    Get JSON Schema for a specific dataset.

    This returns a JSON Schema compatible with SGLang's guided decoding (xgrammar).

    Args:
        dataset: Dataset name ("gaia", "delta", "taskbench", "ultratool")
        strict: If True, use stricter enum-based schema for GAIA (enforces tool_id enum)

    Returns:
        JSON Schema dict suitable for SGLang xgrammar
    """
    dataset_lower = dataset.lower()

    if dataset_lower.startswith("gaia"):
        if strict:
            return PlanningOutputGaiaStrict.model_json_schema()
        return PlanningOutputGaia.model_json_schema()
    elif dataset_lower == "delta":
        return PlanningOutputDelta.model_json_schema()
    else:
        # Default to general schema (TaskBench, UltraTool)
        return PlanningOutput.model_json_schema()


def get_gaia_schema_for_sglang() -> Dict[str, Any]:
    """
    Get the GAIA JSON Schema optimized for SGLang xgrammar decoding.

    Returns a schema where:
    - tool_id is constrained to the 18 GAIA tools via enum
    - final_answer is required (no 'none' type)
    - Placeholder answers are rejected by validators

    Usage:
        schema = get_gaia_schema_for_sglang()
        # Pass to SGLang via guided_json parameter
    """
    return PlanningOutputGaiaStrict.model_json_schema()


def get_planning_model(dataset: str, strict: bool = False):
    """
    Get the Pydantic model class for a specific dataset.

    Args:
        dataset: Dataset name
        strict: If True, return stricter model for GAIA

    Returns:
        Pydantic model class
    """
    dataset_lower = dataset.lower()
 
    if dataset_lower.startswith("gaia"):
        if strict:
            return PlanningOutputGaiaStrict
        return PlanningOutputGaia
    elif dataset_lower == "delta":
        return PlanningOutputDelta
    else:
        return PlanningOutput


def validate_output(output: Dict[str, Any], dataset: str) -> tuple[bool, List[str]]:
    """
    Validate output against the schema for a dataset.

    Args:
        output: Parsed JSON output
        dataset: Dataset name

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    model_class = get_planning_model(dataset)
    errors = []

    try:
        model_class.model_validate(output)
        return True, []
    except Exception as e:
        errors.append(str(e))
        return False, errors


def create_empty_output(model_name: str = "") -> Dict[str, Any]:
    """
    Create an empty planning output structure.

    Args:
        model_name: Optional model name to include

    Returns:
        Dict with empty but valid structure
    """
    return {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [],
        "final_answer": {
            "answer_type": "none",
            "answer": None,
            "aliases": [],
            "tolerance": 0.0
        },
        "model_name": model_name,
        "_parse_error": True
    }


# =============================================================================
# Stop Sequences
# =============================================================================

# Standard stop sequences for planning tasks
PLANNING_STOP_SEQUENCES = [
    "Observation:",
    "User:",
    "###",
    "\n\nHuman:",
    "\n\nAssistant:",
    "<|endoftext|>",
    "<|im_end|>",
]


# =============================================================================
# Sampling Parameters
# =============================================================================

# Fixed sampling parameters for reproducibility
PLANNING_SAMPLING_PARAMS = {
    "temperature": 0.0,  # Greedy decoding
    "top_p": 1.0,        # No nucleus sampling
    "top_k": -1,         # No top-k sampling
}


# =============================================================================
# Main (self-test)
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("GAIA Schema:")
    print("=" * 60)
    print(json.dumps(get_planning_schema("gaia"), indent=2))

    print("\n" + "=" * 60)
    print("Delta Schema:")
    print("=" * 60)
    print(json.dumps(get_planning_schema("delta"), indent=2))

    print("\n" + "=" * 60)
    print("TaskBench Schema:")
    print("=" * 60)
    print(json.dumps(get_planning_schema("taskbench"), indent=2))

    # Test validation
    print("\n" + "=" * 60)
    print("Validation Test:")
    print("=" * 60)

    test_output = {
        "plan_dag": {
            "nodes": [
                {
                    "node_id": "n0",
                    "step_index": 0,
                    "label": "Search for information",
                    "step_type": "tool",
                    "tool_id": "web_search"
                }
            ],
            "edges": []
        },
        "tool_calls": [
            {
                "tool_id": "web_search",
                "arguments": [{"name": "query", "value": "test query"}]
            }
        ],
        "final_answer": {
            "answer_type": "string",
            "answer": "Test answer",
            "aliases": []
        }
    }

    is_valid, errors = validate_output(test_output, "gaia")
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    else:
        print("No errors!")
