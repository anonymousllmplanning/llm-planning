#!/usr/bin/env python3
"""
DEPRECATED: Legacy Single-Stage Prompt Functions

These functions were part of the deprecated single-stage pipeline.
They are kept here for reference only and are NO LONGER USED in production.

The current multi-stage pipeline uses prompt builders from src.inference.prompts:
  - build_abstract_plan_prompt()  (Stage 1: Abstract Planning)
  - build_tool_creation_prompt()  (Stage 2: Tool Creation)
  - build_refinement_prompt()     (Stage 3: Refinement)

These legacy functions may be removed in a future cleanup.
"""
from __future__ import annotations
from typing import Dict, Any
from src.inference.gaia_utils import read_attachment_content
from src.inference.filtering import filter_relevant_tools


def build_prompt_for_gaia(record: Dict[str, Any], model_name: str = "assistant") -> str:
    """
    Build prompt specifically for GAIA dataset.

    GAIA is an answer-focused QA benchmark that may include attachments.
    The model MUST provide a concrete final answer.

    Key features:
    1. Read and embed attachment content when available
    2. Provide clear instructions for answer format
    3. Support both planning evaluation and answer evaluation
    """
    q = record["query"]
    user_query = q.get("user_query", "")
    extra = q.get("extra_instruction") or ""
    attachments = q.get("attachments", [])

    # Get gold plan info for better prompt
    gold = record.get("gold", {})
    gold_nodes = gold.get("plan_dag", {}).get("nodes", [])

    # GAIA 18 Standard Tool Descriptions (hard-coded for system prompt)
    GAIA_STANDARD_TOOLS = [
        ("web_search", "Search the web using search engines. Returns search results with titles, URLs, and snippets."),
        ("web_browser", "Navigate to URLs, browse web pages, and extract information from websites."),
        ("wikipedia_search", "Search and retrieve information from Wikipedia articles."),
        ("pdf_reader", "Read and extract text, tables, and images from PDF documents."),
        ("excel_reader", "Read and analyze Excel spreadsheets (.xlsx, .xls, .csv)."),
        ("file_reader", "Read and extract content from various file formats (txt, docx, pptx, json, etc.)."),
        ("pptx_reader", "Read and extract text, images, and slide content from PowerPoint presentations."),
        ("zip_extractor", "Extract and list contents from ZIP archive files."),
        ("image_recognition", "Analyze images to identify objects, text (OCR), colors, and other visual elements."),
        ("audio_transcription", "Transcribe audio files to text. Supports mp3, wav, and other formats."),
        ("video_analysis", "Analyze video files to extract frames, transcribe audio, and identify visual content."),
        ("python_executor", "Execute Python code for complex calculations, data processing, or custom logic."),
        ("code_interpreter", "Execute and interpret code in various programming languages (Python, JavaScript, Bash, etc.)."),
        ("calculator", "Perform mathematical calculations including arithmetic, algebra, and statistics."),
        ("pdb_analyzer", "Parse and analyze PDB (Protein Data Bank) molecular structure files."),
        ("date_calculator", "Perform date and time calculations including differences, adding/subtracting durations."),
        ("unit_converter", "Convert between different units of measurement (length, weight, volume, etc.)."),
        ("reasoning", "Apply logical reasoning, deduction, or inference without external tools."),
    ]

    # Read attachment contents with split-aware path resolution
    attachment_contents = []
    for att in attachments:
        content = read_attachment_content(att, record=record)
        if content:
            attachment_contents.append(content)

    # Build attachment block
    attachment_block = ""
    if attachment_contents:
        attachment_block = "\n\n## Attached Files\n"
        for content in attachment_contents:
            attachment_block += f"\n{content}\n"

    # Build tool list
    tool_lines = []
    for tool_name, tool_desc in GAIA_STANDARD_TOOLS:
        tool_lines.append(f"- {tool_name}: {tool_desc}")
    tools_block = "\n".join(tool_lines)

    # JSON schema
    json_schema = """{
  "plan_dag": {
    "nodes": [
      {"node_id": "n0", "step_index": 0, "label": "Describe step 0 action", "step_type": "tool", "tool_id": "web_search", "arguments": {"query": "example query"}, "output_vars": ["<n0>"], "needs_tool": true, "needs_new_tool": false}
    ],
    "edges": [
      {"source": "n0", "target": "n1", "edge_type": "control_dep"}
    ]
  },
  "tool_calls": [
    {"call_index": 0, "node_id": "n0", "tool_id": "web_search", "arguments": [{"name": "query", "value": "example query"}]}
  ],
  "final_answer": {
    "answer_type": "string",
    "answer": "42",
    "aliases": [],
    "tolerance": 0.0
  }
}"""

    extra_block = ""
    if extra:
        extra_block = f"\n\n## Additional Instructions\n{extra}"

    prompt = f"""[STRUCTURED PLANNING TASK - JSON OUTPUT ONLY]

## Question
{user_query}{attachment_block}{extra_block}

## Available Tools (USE ONLY THESE - tool_id must match EXACTLY)
{tools_block}

## Task Requirements
1. Analyze the question carefully
2. Create a step-by-step execution plan using the available tools
3. Provide a CONCRETE FINAL ANSWER

## MANDATORY OUTPUT FORMAT
Your response must be a SINGLE VALID JSON OBJECT with this exact structure:

{json_schema}

## CRITICAL RULES

### Rule 1: ANSWER FIELD (REQUIRED - NO PLACEHOLDERS!)
- "answer" must contain your ACTUAL COMPUTED ANSWER based on the question
- ⚠️ STRICTLY FORBIDDEN - DO NOT output these:
  * Any text containing "<" or ">" symbols (like "<PUT YOUR ANSWER>")
  * Placeholder text like "PUT YOUR COMPUTED ANSWER HERE"
  * Template text like "YOUR_EXACT_ANSWER" or "TOOL_NAME"
  * Generic text like "answer here" or "computed result"
- ✅ CORRECT answer format:
  * Numbers: provide exact value (e.g., 42, 3.14, 34689)
  * Text: provide exact string (e.g., "Paris", "backtick", "Time-Parking 2")
  * If unsure: provide your BEST ESTIMATE based on available information

### Rule 2: TOOL SELECTION (REQUIRED)
- Use ONLY tools from "Available Tools" above
- Copy tool_id EXACTLY (case-sensitive)
- Use step_type: "tool" for all data-gathering steps
- DO NOT invent new tools

### Rule 3: OUTPUT FORMAT (STRICTLY ENFORCED)
⚠️ YOUR RESPONSE MUST:
- START with {{ (the very first character)
- END with }} (the very last character)
- Be valid JSON that can be parsed directly
- Contain NO other text

❌ DO NOT OUTPUT:
- "Here is the JSON:" or similar phrases
- Markdown code blocks (```)
- Explanations or reasoning text
- "We need to..." or "Let me..." phrases
- Any text before or after the JSON

✅ CORRECT FORMAT EXAMPLE:
{{"plan_dag": {{"nodes": [...], "edges": [...]}}, "tool_calls": [...], "final_answer": {{...}}}}

## START YOUR JSON RESPONSE NOW (first character must be {{):"""

    return prompt


def build_prompt_for_tools(record: Dict[str, Any], model_name: str = "assistant") -> str:
    """
    Build prompt for tool-usage datasets (UltraTool, TaskBench).
    """
    q = record["query"]
    env = record["tool_environment"]

    user_query = q.get("user_query", "")
    extra = q.get("extra_instruction") or ""
    tools = env.get("tools") or []

    # Build detailed tool list
    tool_lines = []
    for t in tools:
        tool_id = t.get("tool_id", "")
        desc_raw = t.get("description") or ""
        desc = str(desc_raw)[:150]

        args_schema = t.get("arguments_schema") or {}
        arg_info = []

        if isinstance(args_schema, dict):
            for arg_name, arg_spec in args_schema.items():
                if not isinstance(arg_spec, dict):
                    continue

                arg_type = arg_spec.get("type") or "string"
                desc_val = arg_spec.get("description") or ""
                arg_desc = str(desc_val)[:50]
                required = "required" if arg_spec.get("required") else "optional"
                arg_info.append(f"{arg_name}({arg_type}, {required})")

        args_str = ", ".join(arg_info) if arg_info else "no args"
        tool_lines.append(f'- {tool_id}: {desc}\n  Arguments: {args_str}')

    tools_block = "\n".join(tool_lines) if tool_lines else "No tools available."

    json_schema = """{
  "model_name": "MODEL_NAME",
  "plan_dag": {
    "nodes": [
      {"node_id": "n0", "step_index": 0, "label": "Understand the task", "step_type": "thought", "tool_id": null, "arguments": {}, "output_vars": ["<n0>"], "needs_tool": false, "needs_new_tool": false},
      {"node_id": "n1", "step_index": 1, "label": "Execute tool X", "step_type": "tool", "tool_id": "tool_name", "arguments": {"param1": "value1"}, "output_vars": ["<n1>"], "needs_tool": true, "needs_new_tool": false}
    ],
    "edges": [
      {"source": "n0", "target": "n1", "edge_type": "control_dep"}
    ]
  },
  "tool_calls": [
    {"call_index": 0, "node_id": "n1", "tool_id": "tool_name", "arguments": [{"name": "param1", "value": "value1"}]}
  ],
  "final_answer": {"answer_type": "none", "answer": null, "aliases": [], "tolerance": 0.0}
}"""

    extra_line = f"Additional Instructions: {extra}" if extra else ""
    prompt = f"""[EXECUTION PLANNING TASK - JSON OUTPUT ONLY]

## Available Tools (USE ONLY THESE)
{tools_block}

## User Request
{user_query}
{extra_line}

## Task
Create a DAG (Directed Acyclic Graph) execution plan with:
1. Logical steps to complete the request
2. Tool assignments for each step
3. Dependencies between steps

## MANDATORY OUTPUT FORMAT
{json_schema}

## CRITICAL RULES

### Tool Selection
- Use ONLY tools from "Available Tools" above
- Copy tool_id EXACTLY (case-sensitive)
- DO NOT invent new tools

### Output Format (STRICTLY ENFORCED)
⚠️ YOUR RESPONSE MUST:
- START with {{ (first character)
- END with }} (last character)
- Be valid JSON only

❌ PROHIBITED:
- Text before or after JSON
- Markdown code blocks
- Explanations or reasoning

## OUTPUT JSON NOW (start with {{):"""

    return prompt


def build_prompt_for_delta(record: Dict[str, Any], model_name: str = "assistant") -> str:
    """
    Build prompt specifically for Delta MCP dataset.

    Key differences:
    1. Tool catalog may be large (old_delta: 2397 tools, new_delta: 17 tools)
    2. Tools use server_name::tool_name format
    3. No argument values needed
    4. Focus on correct tool_id matching and multi-step planning
    """
    q = record["query"]
    env = record["tool_environment"]
    meta = record.get("meta", {})

    user_query = q.get("user_query", "")
    extra = q.get("extra_instruction") or ""
    all_tools = env.get("tools") or []
    expected_tool_count = meta.get("expected_tool_count", 0)

    # Filter to relevant tools
    tools = filter_relevant_tools(all_tools, user_query, max_tools=30, min_score=0.3)

    if not tools and all_tools:
        tools = all_tools[:30]

    # Build tool list
    tool_lines = []
    for i, t in enumerate(tools):
        tool_id = t.get("tool_id", "")
        desc = (t.get("description") or "")[:150]
        tool_lines.append(f'{i+1}. {tool_id}\n   Description: {desc}')

    tools_block = "\n".join(tool_lines) if tool_lines else "No tools available."

    # Dynamic JSON example based on expected tool count
    if expected_tool_count >= 3:
        json_schema = """{
  "plan_dag": {
    "nodes": [
      {"node_id": "n0", "step_index": 0, "label": "Step 1 description", "step_type": "tool", "tool_id": "server::tool1", "arguments": {}, "output_vars": ["<n0>"], "needs_tool": true, "needs_new_tool": false},
      {"node_id": "n1", "step_index": 1, "label": "Step 2 description", "step_type": "tool", "tool_id": "server::tool2", "arguments": {}, "output_vars": ["<n1>"], "needs_tool": true, "needs_new_tool": false},
      {"node_id": "n2", "step_index": 2, "label": "Step 3 description", "step_type": "tool", "tool_id": "server::tool3", "arguments": {}, "output_vars": ["<n2>"], "needs_tool": true, "needs_new_tool": false}
    ],
    "edges": [
      {"source": "n0", "target": "n1", "edge_type": "data_dep"},
      {"source": "n1", "target": "n2", "edge_type": "data_dep"}
    ]
  },
  "tool_calls": [
    {"call_index": 0, "node_id": "n0", "tool_id": "server::tool1", "arguments": []},
    {"call_index": 1, "node_id": "n1", "tool_id": "server::tool2", "arguments": []},
    {"call_index": 2, "node_id": "n2", "tool_id": "server::tool3", "arguments": []}
  ],
  "final_answer": {"answer_type": "none", "answer": null, "aliases": [], "tolerance": 0.0}
}"""
    else:
        json_schema = """{
  "plan_dag": {
    "nodes": [
      {"node_id": "n0", "step_index": 0, "label": "Step 1 description", "step_type": "tool", "tool_id": "server::tool1", "arguments": {}, "output_vars": ["<n0>"], "needs_tool": true, "needs_new_tool": false},
      {"node_id": "n1", "step_index": 1, "label": "Step 2 description", "step_type": "tool", "tool_id": "server::tool2", "arguments": {}, "output_vars": ["<n1>"], "needs_tool": true, "needs_new_tool": false}
    ],
    "edges": [
      {"source": "n0", "target": "n1", "edge_type": "data_dep"}
    ]
  },
  "tool_calls": [
    {"call_index": 0, "node_id": "n0", "tool_id": "server::tool1", "arguments": []},
    {"call_index": 1, "node_id": "n1", "tool_id": "server::tool2", "arguments": []}
  ],
  "final_answer": {"answer_type": "none", "answer": null, "aliases": [], "tolerance": 0.0}
}"""

    extra_line = f"\nAdditional Instructions: {extra}" if extra else ""

    hint = ""
    if expected_tool_count > 1:
        hint = f"\n\n**IMPORTANT**: This task typically requires {expected_tool_count} sequential tool calls. Think carefully about ALL required steps before outputting."

    prompt = f"""[MCP TOOL PLANNING TASK - JSON OUTPUT ONLY]

## Available Tools (USE ONLY THESE - format: server_name::tool_name)
{tools_block}

## User Request
{user_query}{extra_line}{hint}

## Planning Guidelines
1. Identify ALL required sub-tasks
2. Most tasks need 2-4 tool calls
3. Consider data dependencies between steps
4. Use tool_id EXACTLY as shown

## MANDATORY OUTPUT FORMAT
{json_schema}

## CRITICAL RULES

### Tool Selection
- Use ONLY tools from list above
- Copy tool_id EXACTLY (case-sensitive, includes :: separator)
- DO NOT invent new tools
- Leave arguments empty: {{}} or []

### Output Format (STRICTLY ENFORCED)
⚠️ YOUR RESPONSE MUST:
- START with {{ (first character)
- END with }} (last character)
- Be valid JSON only

❌ PROHIBITED:
- Text before or after JSON
- Markdown code blocks
- Explanations or reasoning
- "We need to..." phrases

## OUTPUT JSON NOW (start with {{):"""

    return prompt


def build_prompt(record: Dict[str, Any], model_name: str = "assistant") -> str:
    """
    Build prompt based on dataset type.
    """
    dataset = record.get("meta", {}).get("dataset", "unknown")

    if dataset == "gaia":
        return build_prompt_for_gaia(record, model_name)

    if dataset == "delta":
        return build_prompt_for_delta(record, model_name)

    return build_prompt_for_tools(record, model_name)
