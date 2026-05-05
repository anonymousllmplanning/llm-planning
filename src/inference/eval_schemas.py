"""
Evaluation-Aligned JSON Schemas for Grammar-Constrained Decoding (xgrammar).

These schemas are designed to produce outputs that EXACTLY match what
src/evaluation/metrics.py expects. The field names, types, and structures
are derived from metrics.py's actual field access patterns.

Key alignment points with metrics.py:
- PlanScores: uses plan_dag.nodes[].step_index, tool_id, label; plan_dag.edges[].source/target
- ToolScores: uses tool_calls[].tool_id, tool_calls[].arguments[].name/value
- AnswerScores: uses final_answer.answer, final_answer.answer_type, final_answer.aliases

Usage:
    from src.inference.eval_schemas import get_eval_schema
    schema = get_eval_schema(dataset="gaia")  # or "delta", "taskbench", "ultratool"
"""

from typing import Dict, Any, Optional


# =============================================================================
# Core Schema Definitions (JSON Schema format for xgrammar/SGLang)
# =============================================================================

# Schema for a single plan node - matches metrics.py evaluate_plan_dag()
PLAN_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "step_index": {"type": "integer"},
        "label": {"type": "string"},
        "step_type": {
            "type": "string",
            "enum": ["tool", "thought", "action"]
        },
        "tool_id": {"type": ["string", "null"]},
        "arguments": {
            "type": ["object", "array"],
            "default": {}
        },
        "output_vars": {
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },
        "needs_tool": {"type": "boolean", "default": True},
        "needs_new_tool": {"type": "boolean", "default": False}
    },
    "required": ["node_id", "step_index", "label"],
    "additionalProperties": True
}

# Schema for a single edge - matches metrics.py _edge_pairs()
PLAN_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "target": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "edge_type": {
            "type": "string",
            "enum": ["data_dep", "control_dep", "dependency"],
            "default": "data_dep"
        }
    },
    "required": ["source", "target"],
    "additionalProperties": True
}

# Schema for plan_dag - matches metrics.py evaluate_plan_dag()
PLAN_DAG_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": PLAN_NODE_SCHEMA,
            "default": []
        },
        "edges": {
            "type": "array",
            "items": PLAN_EDGE_SCHEMA,
            "default": []
        }
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False
}

# Schema for tool argument - matches metrics.py _extract_arguments()
TOOL_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "object"},
                {"type": "array"}
            ]
        }
    },
    "required": ["name", "value"],
    "additionalProperties": False
}

# Schema for a single tool call - matches metrics.py evaluate_tool_calls()
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "call_index": {"type": "integer"},
        "node_id": {"type": "string"},
        "tool_id": {"type": "string"},
        "arguments": {
            "type": "array",
            "items": TOOL_ARGUMENT_SCHEMA,
            "default": []
        }
    },
    "required": ["tool_id"],
    "additionalProperties": True
}

# Schema for tool call without arguments (Delta dataset)
TOOL_CALL_NO_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "call_index": {"type": "integer"},
        "node_id": {"type": "string"},
        "tool_id": {"type": "string"},
        "arguments": {
            "type": "array",
            "items": {},
            "maxItems": 0,
            "default": []
        }
    },
    "required": ["tool_id"],
    "additionalProperties": True
}

# Schema for final_answer - matches metrics.py evaluate_answer()
FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_type": {
            "type": "string",
            "enum": ["none", "string", "number"],
            "default": "none"
        },
        "answer": {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ]
        },
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },
        "tolerance": {
            "type": "number",
            "default": 0.0
        }
    },
    "required": ["answer_type", "answer"],
    "additionalProperties": False
}

# Schema for final_answer requiring actual answer (GAIA dataset)
# Enforces NO placeholder text by pattern matching
FINAL_ANSWER_REQUIRED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_type": {
            "type": "string",
            "enum": ["string", "number"]
        },
        "answer": {
            "oneOf": [
                {
                    "type": "string",
                    "minLength": 1
                },
                {"type": "number"}
            ]
        },
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },
        "tolerance": {
            "type": "number",
            "default": 0.0
        }
    },
    "required": ["answer_type", "answer"],
    "additionalProperties": False
}


# =============================================================================
# Complete Evaluation Schemas for Different Datasets
# =============================================================================

def get_eval_schema_gaia() -> Dict[str, Any]:
    """
    Schema for GAIA dataset.

    Key requirements:
    - final_answer.answer MUST have a concrete value (not null/none)
    - tool_calls and plan_dag are optional but keys must exist
    - Uses 18 predefined tools (web_search, web_browser, etc.)
    """
    return {
        "type": "object",
        "properties": {
            "plan_dag": PLAN_DAG_SCHEMA,
            "tool_calls": {
                "type": "array",
                "items": TOOL_CALL_SCHEMA,
                "default": []
            },
            "final_answer": FINAL_ANSWER_REQUIRED_SCHEMA
        },
        "required": ["plan_dag", "tool_calls", "final_answer"],
        "additionalProperties": False
    }


def get_eval_schema_delta() -> Dict[str, Any]:
    """
    Schema for Delta MCP dataset.

    Key requirements:
    - tool_calls[].arguments MUST be empty list (no parameters)
    - plan_dag.nodes uses tool_id for node_f1 matching (not step_index)
    - final_answer is typically none/null
    - Tool IDs use server_name::tool_name format
    """
    return {
        "type": "object",
        "properties": {
            "plan_dag": PLAN_DAG_SCHEMA,
            "tool_calls": {
                "type": "array",
                "items": TOOL_CALL_NO_ARGS_SCHEMA,
                "default": []
            },
            "final_answer": FINAL_ANSWER_SCHEMA
        },
        "required": ["plan_dag", "tool_calls", "final_answer"],
        "additionalProperties": False
    }


def get_eval_schema_taskbench() -> Dict[str, Any]:
    """
    Schema for TaskBench / UltraTool datasets.

    Key requirements:
    - tool_calls MUST have complete argument information
    - arguments format: [{"name": "param", "value": "val"}, ...]
    - plan_dag.nodes uses step_index for node_f1 matching
    - final_answer is typically none for these datasets
    """
    return {
        "type": "object",
        "properties": {
            "plan_dag": PLAN_DAG_SCHEMA,
            "tool_calls": {
                "type": "array",
                "items": TOOL_CALL_SCHEMA,
                "minItems": 1
            },
            "final_answer": FINAL_ANSWER_SCHEMA
        },
        "required": ["plan_dag", "tool_calls", "final_answer"],
        "additionalProperties": False
    }


def get_eval_schema(dataset: str) -> Dict[str, Any]:
    """
    Get the appropriate evaluation schema for a dataset.

    Args:
        dataset: Dataset name ("gaia", "delta", "taskbench", "ultratool")

    Returns:
        JSON Schema dict compatible with xgrammar/SGLang
    """
    dataset_lower = dataset.lower()

    if dataset_lower.startswith("gaia"):
        return get_eval_schema_gaia()
    elif dataset_lower == "delta":
        return get_eval_schema_delta()
    elif dataset_lower in ("taskbench", "ultratool"):
        return get_eval_schema_taskbench()
    else:
        # Default to taskbench schema (most complete)
        return get_eval_schema_taskbench()


# =============================================================================
# Schema Validation Utilities
# =============================================================================

def validate_schema_compliance(output: Dict[str, Any], dataset: str) -> tuple[bool, list[str]]:
    """
    Validate that an output dict complies with the expected schema.

    Args:
        output: The parsed JSON output from the model
        dataset: Dataset name for schema selection

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required top-level keys
    for key in ["plan_dag", "tool_calls", "final_answer"]:
        if key not in output:
            errors.append(f"Missing required key: {key}")

    if errors:
        return False, errors

    # Check plan_dag structure
    plan_dag = output.get("plan_dag", {})
    if not isinstance(plan_dag, dict):
        errors.append(f"plan_dag must be dict, got {type(plan_dag).__name__}")
    else:
        if "nodes" not in plan_dag:
            errors.append("plan_dag missing 'nodes' key")
        elif not isinstance(plan_dag["nodes"], list):
            errors.append(f"plan_dag.nodes must be list, got {type(plan_dag['nodes']).__name__}")

        if "edges" not in plan_dag:
            errors.append("plan_dag missing 'edges' key")
        elif not isinstance(plan_dag["edges"], list):
            errors.append(f"plan_dag.edges must be list, got {type(plan_dag['edges']).__name__}")

        # Validate individual nodes
        for i, node in enumerate(plan_dag.get("nodes", [])):
            if not isinstance(node, dict):
                errors.append(f"plan_dag.nodes[{i}] must be dict")
                continue
            if "step_index" in node and not isinstance(node["step_index"], int):
                errors.append(f"plan_dag.nodes[{i}].step_index must be int")

    # Check tool_calls structure
    tool_calls = output.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        errors.append(f"tool_calls must be list, got {type(tool_calls).__name__}")
    else:
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                errors.append(f"tool_calls[{i}] must be dict")
                continue
            if "tool_id" not in tc:
                errors.append(f"tool_calls[{i}] missing 'tool_id'")
            elif not isinstance(tc["tool_id"], str):
                errors.append(f"tool_calls[{i}].tool_id must be string")

            args = tc.get("arguments", [])
            if not isinstance(args, list):
                errors.append(f"tool_calls[{i}].arguments must be list, got {type(args).__name__}")
            else:
                for j, arg in enumerate(args):
                    if not isinstance(arg, dict):
                        errors.append(f"tool_calls[{i}].arguments[{j}] must be dict")
                    elif "name" not in arg or "value" not in arg:
                        errors.append(f"tool_calls[{i}].arguments[{j}] missing 'name' or 'value'")

    # Check final_answer structure
    final_answer = output.get("final_answer", {})
    if not isinstance(final_answer, dict):
        errors.append(f"final_answer must be dict, got {type(final_answer).__name__}")
    else:
        if "answer" not in final_answer:
            errors.append("final_answer missing 'answer' key")

        aliases = final_answer.get("aliases", [])
        if not isinstance(aliases, list):
            errors.append(f"final_answer.aliases must be list, got {type(aliases).__name__}")

    # Dataset-specific validation
    dataset_lower = dataset.lower()
    if dataset_lower == "gaia":
        # GAIA requires actual answer
        answer = final_answer.get("answer")
        answer_type = final_answer.get("answer_type", "none")
        if answer is None and answer_type != "none":
            errors.append("GAIA dataset requires non-null answer")
    elif dataset_lower == "delta":
        # Delta requires empty arguments
        for i, tc in enumerate(tool_calls):
            args = tc.get("arguments", [])
            if isinstance(args, list) and len(args) > 0:
                errors.append(f"Delta dataset: tool_calls[{i}].arguments should be empty")

    return len(errors) == 0, errors


# =============================================================================
# Schema as String (for debugging/dry-run)
# =============================================================================

def get_schema_description(dataset: str) -> str:
    """
    Get a human-readable description of the schema for a dataset.
    Useful for dry-run mode and debugging.
    """
    import json
    schema = get_eval_schema(dataset)
    return json.dumps(schema, indent=2)


if __name__ == "__main__":
    # Self-test: print schemas for each dataset
    import json

    print("=" * 60)
    print("GAIA Schema:")
    print("=" * 60)
    print(json.dumps(get_eval_schema("gaia"), indent=2))

    print("\n" + "=" * 60)
    print("Delta Schema:")
    print("=" * 60)
    print(json.dumps(get_eval_schema("delta"), indent=2))

    print("\n" + "=" * 60)
    print("TaskBench Schema:")
    print("=" * 60)
    print(json.dumps(get_eval_schema("taskbench"), indent=2))

    # Test validation
    print("\n" + "=" * 60)
    print("Validation Test:")
    print("=" * 60)

    test_output = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Test", "tool_id": "web_search"}
            ],
            "edges": []
        },
        "tool_calls": [
            {"tool_id": "web_search", "arguments": [{"name": "query", "value": "test"}]}
        ],
        "final_answer": {
            "answer_type": "string",
            "answer": "test answer",
            "aliases": []
        }
    }

    is_valid, errs = validate_schema_compliance(test_output, "gaia")
    print(f"Valid: {is_valid}")
    if errs:
        print(f"Errors: {errs}")
    else:
        print("No errors!")
