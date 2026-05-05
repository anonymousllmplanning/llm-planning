"""
Prompt Builders for Multi-Stage Planning Pipeline.

This module contains the logic for building prompts for the 3-stage pipeline:
1. Abstract Plan (ToDo Writer)
2. Tool Creation
3. Plan Refinement (Tool Calling)
"""

import inspect
import json
from typing import Dict, Any, List, Optional, get_args, get_origin, Union
from src.config.zh_profile import get_stage_prompt_suffix
from src.inference.gaia_utils import get_attachment_display_path
from src.inference.schemas import AbstractPlan, ToolCreationOutput
from src.inference.tools import TOOL_IMPLEMENTATIONS


_TOOL_DESCRIPTION_OVERRIDES = {
    "reasoning": "Use an internal reasoning helper for logic-only subproblems when no external information is needed.",
    "code_interpreter": "Use a fallback code interpreter for non-Python code or lightweight execution-style reasoning.",
    "video_analysis": "Analyze video inputs with the available video-analysis utility.",
    "submit_final_answer": "Reserved for final answer submission in Stage 5; do not include it as a normal Stage 3 planning node.",
}


def _annotation_to_json_type(annotation: Any) -> str:
    """Map Python annotations to lightweight JSON schema types for prompting."""
    if annotation is inspect._empty:
        return "string"

    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _annotation_to_json_type(args[0])

    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    if annotation in (dict, Dict):
        return "object"
    if annotation in (list, List):
        return "array"
    return "string"


def _schema_from_openai_parameters(parameters: Any) -> Dict[str, Dict[str, Any]]:
    """Convert an OpenAI function parameters object into the local tool schema."""
    if not isinstance(parameters, dict):
        return {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    if not isinstance(properties, dict):
        return {}

    schema: Dict[str, Dict[str, Any]] = {}
    for name, info in properties.items():
        if not isinstance(info, dict):
            info = {}
        entry = {
            "type": str(info.get("type") or "string"),
            "description": str(info.get("description") or ""),
            "required": str(name) in required,
        }
        if "enum" in info:
            entry["enum"] = info.get("enum")
        schema[str(name)] = entry
    return schema


def _normalize_tool_spec(tool: Any) -> Optional[Dict[str, Any]]:
    """Normalize supported unified tool-environment schemas to one shape."""
    if isinstance(tool, str):
        tool_id = tool.strip()
        return {"tool_id": tool_id, "description": "", "arguments_schema": {}} if tool_id else None
    if not isinstance(tool, dict):
        return None

    if isinstance(tool.get("function"), dict):
        fn = tool["function"]
        tool_id = str(fn.get("name") or tool.get("tool_id") or "").strip()
        if not tool_id:
            return None
        return {
            "tool_id": tool_id,
            "description": str(fn.get("description") or tool.get("description") or ""),
            "arguments_schema": _schema_from_openai_parameters(fn.get("parameters")),
        }

    tool_id = str(tool.get("tool_id") or tool.get("name") or "").strip()
    if not tool_id:
        return None
    arguments_schema = (
        tool.get("arguments_schema")
        or tool.get("parameters")
        or tool.get("arguments")
        or {}
    )
    if isinstance(arguments_schema, dict) and "properties" in arguments_schema:
        arguments_schema = _schema_from_openai_parameters(arguments_schema)
    if not isinstance(arguments_schema, dict):
        arguments_schema = {}
    return {
        **tool,
        "tool_id": tool_id,
        "description": str(tool.get("description") or ""),
        "arguments_schema": arguments_schema,
    }


def normalize_tool_environment(record_or_env: Any) -> List[Dict[str, Any]]:
    """
    Return record-visible tools across the supported unified-data schemas.

    TaskBench uses `tool_environment.tools`, while UltraTool records may expose
    OpenAI function specs under `tool_environment.openai_function_tools`.
    """
    if isinstance(record_or_env, dict) and "tool_environment" in record_or_env:
        env = record_or_env.get("tool_environment", {})
    else:
        env = record_or_env

    raw_tools: List[Any] = []
    if isinstance(env, list):
        raw_tools.extend(env)
    elif isinstance(env, dict):
        raw_tools.extend(env.get("tools") or [])
        raw_tools.extend(env.get("openai_function_tools") or [])
        raw_tools.extend(env.get("functions") or [])

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for tool in raw_tools:
        spec = _normalize_tool_spec(tool)
        if not spec:
            continue
        tool_id = spec["tool_id"]
        if tool_id in seen:
            continue
        normalized.append(spec)
        seen.add(tool_id)
    return normalized


def _format_abstract_step(step: Any, fallback_index: int) -> str:
    """Render a Stage-1 step whether the model returned a dict or a string."""
    if isinstance(step, dict):
        idx = step.get("step_index", fallback_index)
        desc = step.get("description") or step.get("label") or step.get("task") or ""
    else:
        idx = fallback_index
        desc = step
    return f"Step {idx}: {str(desc).strip()}"


def get_available_tool_list() -> List[Dict[str, Any]]:
    """
    Get the list of executable tool specs exposed during answer-mode refinement.
    This stays aligned with the actual tool implementations in tools.py.
    """
    tools: List[Dict[str, Any]] = []
    for tool_id, func in TOOL_IMPLEMENTATIONS.items():
        description = _TOOL_DESCRIPTION_OVERRIDES.get(tool_id)
        if not description:
            doc = inspect.getdoc(func) or ""
            description = doc.splitlines()[0].strip() if doc else f"{tool_id} tool"

        arguments_schema: Dict[str, Dict[str, str]] = {}
        for name, param in inspect.signature(func).parameters.items():
            if name in {"self", "kwargs"}:
                continue
            arguments_schema[name] = {"type": _annotation_to_json_type(param.annotation)}

        tools.append({
            "tool_id": tool_id,
            "description": description,
            "arguments_schema": arguments_schema,
        })

    return tools


def get_answer_mode_tool_context(
    record: Dict[str, Any],
    new_tools: Optional[List[Dict[str, Any]]] = None,
    tool_scope: str = "record",
) -> Dict[str, Any]:
    """
    Build the answer-mode executable tool context.

    The baseline comes from the record's tool_environment so that prompt
    exposure stays aligned with the annotated benchmark environment. We then
    merge only those Stage-2 created tools that map to actual runtime tools.
    When tool_scope="global", expose the full executable runtime library as an
    ablation while leaving the default record-level setup unchanged.
    """
    runtime_tools = get_available_tool_list()
    runtime_map = {tool["tool_id"]: tool for tool in runtime_tools}

    annotated_tools = normalize_tool_environment(record)

    selected_tools: List[Dict[str, Any]] = []
    selected_ids = set()
    merged_created_tool_ids: List[str] = []
    omitted_created_tool_ids: List[str] = []

    if tool_scope == "global":
        for tool in runtime_tools:
            tool_id = tool.get("tool_id")
            if tool_id and tool_id not in selected_ids:
                selected_tools.append(tool)
                selected_ids.add(tool_id)
    else:
        for tool in annotated_tools or []:
            tool_id = tool.get("tool_id") if isinstance(tool, dict) else str(tool)
            if not tool_id or tool_id not in runtime_map or tool_id in selected_ids:
                continue
            selected_tools.append(runtime_map[tool_id])
            selected_ids.add(tool_id)

    for tool in new_tools or []:
        tool_id = tool.get("tool_id") if isinstance(tool, dict) else None
        if not tool_id:
            continue
        if tool_id in runtime_map and tool_id not in selected_ids:
            selected_tools.append(runtime_map[tool_id])
            selected_ids.add(tool_id)
            merged_created_tool_ids.append(tool_id)
        elif tool_id not in runtime_map:
            omitted_created_tool_ids.append(tool_id)

    if "submit_final_answer" in runtime_map and "submit_final_answer" not in selected_ids:
        selected_tools.append(runtime_map["submit_final_answer"])
        selected_ids.add("submit_final_answer")

    if not selected_tools:
        fallback_ids = [
            "web_search",
            "web_browser",
            "calculator",
            "python_executor",
            "excel_reader",
            "file_reader",
            "pdf_reader",
            "submit_final_answer",
        ]
        for tool_id in fallback_ids:
            if tool_id in runtime_map and tool_id not in selected_ids:
                selected_tools.append(runtime_map[tool_id])
                selected_ids.add(tool_id)

    return {
        "tools": selected_tools,
        "merged_created_tool_ids": merged_created_tool_ids,
        "omitted_created_tool_ids": omitted_created_tool_ids,
    }


def build_abstract_plan_prompt(record: Dict[str, Any], dataset: Optional[str] = None) -> str:
    """
    Stage 1: Build prompt for Abstract Plan generation.
    Does NOT show tools to the model.
    """
    q = record.get("query", {})
    user_query = q.get("user_query", "")
    extra = q.get("extra_instruction", "")

    extra_block = f"\n\nAdditional Instructions:\n{extra}" if extra else ""
    cross_lingual_block = get_stage_prompt_suffix(dataset, "abstract")
    cross_lingual_block = f"\n\n{cross_lingual_block}" if cross_lingual_block else ""

    schema_json = AbstractPlan.model_json_schema()
    _ = schema_json  # Kept for schema drift checks during prompt maintenance.

    prompt = """[ABSTRACT PLANNING TASK]

## User Request
{user_query}{extra_block}

## Task
Create a high-level abstract plan to solve THIS SPECIFIC REQUEST.
Also express the same planning intent as an abstract dependency DAG.

⚠️ CRITICAL RULES:
1. **Length depends on complexity**: Simple tasks may only need 2-3 steps. Complex tasks may require 6-8 steps. DO NOT artificially limit or expand the plan.
2. **Concrete Descriptions**: Use specific terms from the request (e.g., "Finding Nemo main character"), not placeholders like "Entity A".
3. **No executable tool IDs or API names**: Natural planning actions such as retrieve, inspect, compare, calculate, or summarize are allowed; do not name concrete runtime tools.
4. **Keep steps and DAG aligned**: `steps` is the ordered planning sketch used by later stages. `abs_plan_dag.nodes` must describe the same logical subtasks.
5. **Edges mean true dependency**: Add an edge only when the target step needs the source step's fact, output, or established state. If two steps can be done independently, do not connect them.
6. **Use the right dependency shape**: Use a chain when each step truly depends on the previous step; expose independent branches when they exist. `valid_execution_order` should be one legal topological order of the DAG.
{cross_lingual_block}

## Examples

### Example 1: Simple Fact Retrieval (Short Plan - 3 Steps)
**Request**: "What is the capital of the country where the 2024 Olympics will be held?"
**Output shape**:
{{
  "steps": [
    {{"step_index": 0, "description": "Identify the country hosting the 2024 Olympics."}},
    {{"step_index": 1, "description": "Determine the capital city of that country."}},
    {{"step_index": 2, "description": "Use the capital city as the final answer."}}
  ],
  "abs_plan_dag": {{
    "nodes": [
      {{"node_id": "a0", "step_index": 0, "label": "Identify the country hosting the 2024 Olympics.", "step_type": "thought", "needs_tool": false}},
      {{"node_id": "a1", "step_index": 1, "label": "Determine the capital city of that country.", "step_type": "thought", "needs_tool": false}},
      {{"node_id": "a2", "step_index": 2, "label": "Use the capital city as the final answer.", "step_type": "thought", "needs_tool": false}}
    ],
    "edges": [
      {{"source": "a0", "target": "a1", "edge_type": "dependency"}},
      {{"source": "a1", "target": "a2", "edge_type": "dependency"}}
    ]
  }},
  "valid_execution_order": ["a0", "a1", "a2"],
  "reasoning": "The capital lookup depends on first identifying the host country."
}}

### Example 2: Complex Analysis (Long Plan - 8 Steps)
**Request**: "Compare the stock performance of Apple and Microsoft in 2023 and identify which had a higher volatility."
**Dependency idea**: Apple data and Microsoft data can be retrieved independently. Each return calculation depends only on its own company's data. The final comparison depends on both volatility values.

## MANDATORY OUTPUT FORMAT (STRICT JSON ONLY)
Your response must be a SINGLE VALID JSON OBJECT with this structure:
{{
  "steps": [
    {{"step_index": 0, "description": "..."}},
    {{"step_index": 1, "description": "..."}},
    ... (add more steps as needed based on complexity)
  ],
  "abs_plan_dag": {{
    "nodes": [
      {{"node_id": "a0", "step_index": 0, "label": "...", "step_type": "thought", "needs_tool": false}},
      {{"node_id": "a1", "step_index": 1, "label": "...", "step_type": "thought", "needs_tool": false}}
    ],
    "edges": [
      {{"source": "a0", "target": "a1", "edge_type": "dependency"}}
    ]
  }},
  "valid_execution_order": ["a0", "a1"],
  "reasoning": "Explain why you chose this number of steps and your decomposition logic."
}}

## OUTPUT JSON NOW:""".format(
        user_query=user_query,
        extra_block=extra_block,
        cross_lingual_block=cross_lingual_block,
    )
    return prompt


def build_tool_creation_prompt(
    record: Dict[str, Any],
    abstract_plan: Dict[str, Any],
    dataset: Optional[str] = None,
    tool_scope: str = "record",
) -> str:
    """
    Stage 2: Build prompt for Tool Creation.
    Shows Abstract Plan + Existing Tools.
    Asks model to propose NEW tools if needed.
    """
    q = record.get("query", {})
    user_query = q.get("user_query", "")
    cross_lingual_block = get_stage_prompt_suffix(dataset, "creation")
    cross_lingual_block = f"\n\n{cross_lingual_block}" if cross_lingual_block else ""
    tools = get_available_tool_list() if tool_scope == "global" else normalize_tool_environment(record)

    # Format existing tools
    tool_lines = []
    for t in tools:
        tool_id = t.get("tool_id", "")
        desc = (t.get("description") or "")[:100]
        tool_lines.append(f"- {tool_id}: {desc}")
    
    tools_block = "\n".join(tool_lines) if tool_lines else "No tools available."

    # Format abstract plan
    steps = abstract_plan.get("steps", [])
    plan_lines = []
    for i, s in enumerate(steps):
        plan_lines.append(_format_abstract_step(s, i))
    plan_block = "\n".join(plan_lines)

    # NOTE: Use empty new_tools as example to prevent models from copying example tool names
    # Previous issue: models were copying "calculate_X" from example verbatim (98.6% for 0.5b)
    example_json = """{
  "reasoning": "<Your analysis of tool sufficiency>",
  "new_tools": []
}"""

    prompt = """[TOOL CREATION TASK]

## User Request
{user_query}

## Abstract Plan
{plan_block}

## Existing Tools
{tools_block}

## Task
Analyze if the Existing Tools are sufficient to execute the Abstract Plan.

⚠️ IMPORTANT:
1. Check EACH step in the Abstract Plan
2. For EACH step, identify which Existing Tool can handle it
3. If ALL steps can be handled by Existing Tools → return EMPTY "new_tools" list
4. ONLY propose new tools if there is a clear gap
{cross_lingual_block}

✅ In most cases, existing tools ARE sufficient - return empty list!
❌ DO NOT create duplicate tools with slightly different names

## MANDATORY OUTPUT FORMAT (STRICT JSON ONLY)
Your response must be a SINGLE VALID JSON OBJECT with this structure:
{example_json}

⚠️ START YOUR RESPONSE DIRECTLY WITH "{{"
❌ NO text before the JSON
❌ NO markdown code blocks (```json)

## OUTPUT JSON NOW:""".format(
        user_query=user_query,
        plan_block=plan_block,
        tools_block=tools_block,
        example_json=example_json,
        cross_lingual_block=cross_lingual_block,
    )
    return prompt


def build_refinement_prompt(
    record: Dict[str, Any],
    abstract_plan: Dict[str, Any],
    new_tools: List[Dict[str, Any]],
    use_actual_tools: bool = False,
    dataset: Optional[str] = None,
    tool_scope: str = "record",
) -> str:
    """
    Stage 3: Build prompt for Plan Refinement (Tool Calling).
    Shows Abstract Plan + Existing Tools + New Tools.
    Same output format as standard runner (PlanningOutput).

    Args:
        record: Input record with query and tool_environment
        abstract_plan: Abstract plan from Stage 1
        new_tools: New tools proposed in Stage 2
        use_actual_tools: If True, use actual implemented tools from tools.py (for answer mode)
    """
    q = record.get("query", {})
    user_query = q.get("user_query", "")
    attachments = q.get("attachments", [])
    cross_lingual_block = get_stage_prompt_suffix(dataset, "refinement")
    cross_lingual_block = f"\n\n{cross_lingual_block}" if cross_lingual_block else ""
    async_structure_block = ""
    if dataset == "gaia_cat_A_async":
        async_structure_block = """

## IMPORTANT: Async Dependency Structure
- Preserve independent branches when two steps do not consume each other's outputs.
- Only add an edge when the target step truly depends on the source step's output.
- Do NOT serialize steps unless one step consumes another step's output.
- Do NOT collapse multiple abstract-plan steps into one coarse node if the abstract plan separates them.
- Prefer a dependency DAG that exposes parallelizable work over a purely linear chain.
"""

    # Use actual executable tools if in answer mode
    tool_creation_bridge_block = ""

    if use_actual_tools or tool_scope == "global":
        answer_mode_context = get_answer_mode_tool_context(record, new_tools, tool_scope=tool_scope)
        all_tools = answer_mode_context["tools"]
        merged_created_tool_ids = answer_mode_context["merged_created_tool_ids"]
        omitted_created_tool_ids = answer_mode_context["omitted_created_tool_ids"]
        bridge_lines = []
        if merged_created_tool_ids:
            merged = ", ".join(merged_created_tool_ids)
            bridge_lines.append(
                f"- Executable tool suggestions from Stage 2 were merged into the tool list above: {merged}."
            )
        if omitted_created_tool_ids:
            omitted = ", ".join(omitted_created_tool_ids)
            bridge_lines.append(
                f"- Other Stage-2 suggestions ({omitted}) are not executable in the current runtime, so do not use them directly."
            )
        if bridge_lines:
            tool_creation_bridge_block = "## Tool-Creation Handoff\n" + "\n".join(bridge_lines) + "\n\n"
    else:
        # In order mode (evaluation), use tools from environment and merge new_tools
        all_tools = list(normalize_tool_environment(record))

        # Merge new_tools
        for nt in new_tools:
            if not isinstance(nt, dict):
                continue
            # Check if already exists to avoid duplicates
            if not any(t.get("tool_id") == nt.get("tool_id") for t in all_tools):
                nt_formatted = {
                    "tool_id": nt.get("tool_id"),
                    "description": nt.get("description"),
                    "arguments_schema": {
                        arg: {"type": "string"} for arg in nt.get("arguments", {}).keys()
                    }
                }
                all_tools.append(nt_formatted)

    # Build attachment information block for GAIA dataset
    attachment_block = ""
    audio_guidance_block = ""
    if attachments:
        attachment_lines = ["## Available Files (Attachments)"]
        has_audio_attachment = False
        for att in attachments:
            file_name = att.get("file_name", "")
            file_path = get_attachment_display_path(att, record)
            file_type = att.get("file_type", "")
            file_name_lower = str(file_name).lower()
            if file_name_lower.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")):
                has_audio_attachment = True

            # For GAIA, the file_path might need to be resolved
            # But we'll show the file_name which the model should use
            attachment_lines.append(f"- File: {file_name} (Type: {file_type})")
            if file_path:
                attachment_lines.append(f"  Path: {file_path}")

        attachment_block = "\n".join(attachment_lines) + "\n\n"
        if has_audio_attachment:
            audio_guidance_block = """## IMPORTANT: Audio file usage
- For audio attachments (.mp3, .wav, .m4a, .ogg, .flac, .aac), start with audio_transcription.
- Do NOT use file_reader on audio attachments.
- If you provide the optional language argument, use a short language code like "en" or "zh", not locale variants like "en-US".

"""

    # Use standard prompt logic similar to runner.py build_prompt_for_tools
    # We will reimplement it here simplified to avoid importing runner.py (circular dependency)

    tool_lines = []
    for t in all_tools:
        tool_id = t.get("tool_id", "")
        desc = (t.get("description") or "")[:150]
        args_schema = t.get("arguments_schema", {})
        
        args_desc = []
        if isinstance(args_schema, dict):
            for arg_name, arg_info in args_schema.items():
                arg_type = arg_info.get("type", "string")
                args_desc.append(f"{arg_name} ({arg_type})")
        
        args_str = ", ".join(args_desc) if args_desc else "no args"
        tool_lines.append(f'- {tool_id}: {desc} | Args: {args_str}')

    tools_block = "\n".join(tool_lines)

    # Format abstract plan as context
    steps = abstract_plan.get("steps", [])
    plan_lines = []
    for i, s in enumerate(steps):
        plan_lines.append(_format_abstract_step(s, i))
    plan_block = "\n".join(plan_lines)

    # Universal generic example that applies to all datasets and modes
    # Uses placeholders like <TOOL_ID> to avoid leading the model to specific tools
    json_schema = """{
  "plan_dag": {
    "nodes": [
      {"node_id": "n0", "step_index": 0, "label": "Step A description", "step_type": "tool", "tool_id": "<TOOL_ID>", "arguments": {"<ARG_NAME>": "<ARG_VALUE>"}, "output_vars": ["<n0>"], "needs_tool": true},
      {"node_id": "n1", "step_index": 1, "label": "Step B description", "step_type": "tool", "tool_id": "<TOOL_ID>", "arguments": {"<ARG_NAME>": "<n0>"}, "output_vars": ["<n1>"], "needs_tool": true}
    ],
    "edges": [
      {"source": "n0", "target": "n1", "edge_type": "data_dep"}
    ]
  },
  "tool_calls": [
    {"call_index": 0, "node_id": "n0", "tool_id": "<TOOL_ID>", "arguments": [{"name": "<ARG_NAME>", "value": "<ARG_VALUE>"}]},
    {"call_index": 1, "node_id": "n1", "tool_id": "<TOOL_ID>", "arguments": [{"name": "<ARG_NAME>", "value": "<n0>"}]}
  ],
  "final_answer": {"answer_type": "string", "answer": null}
}"""

    prompt = """[EXECUTION PLANNING TASK]

## User Request
{user_query}

    {attachment_block}{tool_creation_bridge_block}## Proposed Abstract Plan (Reference)
{plan_block}

## Available Tools (Existing + Created)
{tools_block}

## Task
Create a CONCRETE execution plan as a Directed Acyclic Graph (DAG) to satisfy the User Request.
You MUST follow the Abstract Plan logic where possible, but mapped to specific tools.
{cross_lingual_block}
{async_structure_block}

## IMPORTANT: File-reading tools usage
- For tools that read files (pptx_reader, pdf_reader, excel_reader, etc.), you MUST provide the "file_path" argument
- Use the EXACT filename from "Available Files (Attachments)" section above
- Example: {{"file_path": "abc123.pptx"}}
{audio_guidance_block}

## IMPORTANT: Edges represent dependencies
- If step B uses the output of step A, add an edge: {{"source": "nA", "target": "nB", "edge_type": "data_dep"}}
- Use "<nX>" in arguments to reference output from node nX
- Every node that depends on another node MUST have a corresponding edge

## MANDATORY OUTPUT FORMAT (STRICT JSON ONLY)
Your response must be a SINGLE VALID JSON OBJECT with this EXACT structure:
{json_schema}

## CRITICAL RULES
1. Use ONLY tools from the list above.
2. For file-reading tools, ALWAYS provide file_path argument with exact filename.
3. Create edges for ALL dependencies between nodes.
4. Do NOT add `submit_final_answer` as a node or tool call in Stage 3; final answer synthesis is handled separately in Stage 5.
5. Output valid JSON starting directly with "{{".

## OUTPUT JSON NOW:""".format(
        user_query=user_query,
        attachment_block=attachment_block,
        tool_creation_bridge_block=tool_creation_bridge_block,
        plan_block=plan_block,
        tools_block=tools_block,
        json_schema=json_schema,
        cross_lingual_block=cross_lingual_block,
        async_structure_block=async_structure_block,
        audio_guidance_block=audio_guidance_block,
    )
    return prompt
