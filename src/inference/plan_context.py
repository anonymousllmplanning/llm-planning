"""Render versioned planning context at the S1/S3 and S3/S4 boundaries."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def build_stage3_abstract_context(abstract_plan: Dict[str, Any], *,
                                  handoff: str = "steps_only") -> str:
    """Expose S1 dependencies for an explicitly versioned S3 experiment.

    ``steps_only`` preserves the historical prompt. ``dag_v1`` transmits the
    original graph and correspondence hints; it never copies S1 edges into a
    different S3 node namespace or infers dependencies from array order.
    """
    if handoff not in ("steps_only", "dag_v1"):
        raise ValueError(f"Unsupported Stage-1 DAG handoff: {handoff}")
    if handoff == "steps_only":
        return ""
    if not isinstance(abstract_plan, dict):
        raise ValueError("Stage-1 abstract plan must be an object")

    steps = abstract_plan.get("steps", [])
    raw_dag = abstract_plan.get("abs_plan_dag")
    nodes = raw_dag.get("nodes", []) if isinstance(raw_dag, dict) else []
    nodes = nodes if isinstance(nodes, list) else []
    links = []
    for position, step in enumerate(steps if isinstance(steps, list) else []):
        step_index = step.get("step_index", position) if isinstance(step, dict) else position
        links.append({
            "step_position": position,
            "step_index": step_index,
            "mapping_basis": ("explicit_step_index" if isinstance(step, dict)
                              and "step_index" in step else "step_position"),
            "abstract_node_ids": [node.get("node_id", node.get("id"))
                                  for node in nodes if isinstance(node, dict)
                                  and node.get("step_index") == step_index],
        })
    payload = {
        "schema_version": "s1-s3-dag-v1",
        "steps": steps,
        "abs_plan_dag": raw_dag,
        "step_node_links": links,
    }
    if "valid_execution_order" in abstract_plan:
        payload["valid_execution_order"] = abstract_plan["valid_execution_order"]
    return (
        "\n\n## Stage-1 Dependency Handoff (s1-s3-dag-v1)\n"
        "The graph below is the model's proposed abstract plan, not ground truth "
        "or observed evidence. Preserve genuine prerequisites when refining it; "
        "do not copy abstract IDs as concrete edge endpoints.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\nRefinement rules:\n"
        "- A listed step order or valid_execution_order is one execution order; "
        "it does not imply a dependency chain. Do not add edges solely from array order.\n"
        "- Map each concrete node to its source abstract node IDs using an "
        "abstract_node_ids array. Split subtasks when needed and retain this mapping. "
        "Use [] for a genuinely added subtask; do not invent missing abstract IDs.\n"
        "- Express prerequisites between concrete nodes using their concrete IDs. "
        "Preserve independent branches. Check actual input/output needs when "
        "splitting or combining subtasks.\n"
        "- If an abstract dependency is mistaken, cyclic, or cannot be preserved, "
        "explain the revision in dependency_revision_reason on the affected "
        "concrete node. Do not silently stitch, drop, or reverse abstract edges.\n"
        "- Missing or empty abs_plan_dag supplies no explicit dependency constraints; "
        "infer genuine dependencies from the task rather than manufacturing a chain."
    )


def build_stage4_plan_context(final_pred: Dict[str, Any]) -> str:
    """Preserve the execution plan's content without treating it as evidence.

    Stage-3 nodes use ``label`` in the canonical schema. Some older outputs
    also use ``description`` or ``task``. Serializing the plan keeps all these
    fields, arguments, output variables, node identifiers, and dependency edges
    intact; picking one text alias would discard potentially distinct content.
    This function does not mutate the prediction or execute planned calls.
    """
    if not isinstance(final_pred, dict):
        return ""

    context = {}
    plan = final_pred.get("plan_dag")
    if isinstance(plan, dict) and (plan.get("nodes") or plan.get("edges")):
        context["plan_dag"] = plan
    calls = final_pred.get("tool_calls")
    if isinstance(calls, list) and calls:
        context["planned_tool_calls"] = calls
    if not context:
        return ""

    return (
        "Your execution plan (from planning stage):\n"
        "Use this as planning context and reassess it using tool observations. "
        "Planned arguments, output variables, and predicted values are not "
        "observed evidence.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def build_stage4_replay_snapshot(
    query_record, final_pred, new_tools, args, visible_tools, *, effective_max_turns
):
    """Copy the inputs needed to rerun Stage 4 without copying credentials.

    ``tool_calls`` is still the Stage-3 proposal at this boundary. Later changes
    to the prediction or its nested fields must not alter this snapshot.
    """
    meta = query_record.get("meta") or {}
    payload = {
        "schema_version": "stage4-input-v1",
        "record": {
            "meta": {k: meta[k] for k in (
                "id", "dataset", "subset", "split", "tool_profile_id",
            ) if k in meta},
            "query": query_record.get("query") or {},
            "tool_environment": query_record.get("tool_environment") or {},
        },
        "stage3_prediction": final_pred,
        "created_tools": new_tools or [],
        "visible_tool_specs": visible_tools,
        "effective_max_turns": effective_max_turns,
        "settings": {k: getattr(args, k, None) for k in (
            "model_name", "backend", "dataset", "tool_scope", "max_turns",
            "skill_mode", "document_guidance_mode",
        )},
        "plan_context_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).with_name("runner.py").read_bytes()).hexdigest(),
    }
    payload = copy.deepcopy(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["input_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return payload
