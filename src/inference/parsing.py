#!/usr/bin/env python3
"""
Robust JSON Parsing Utilities for LLM Outputs

This module provides comprehensive JSON extraction and normalization
for handling various LLM output formats with multiple fallback strategies.
"""
from __future__ import annotations
import re
import json
from typing import Dict, Any, Optional, Tuple, Set
from json import JSONDecodeError


# ============================================================================
# JSON Extraction
# ============================================================================

def extract_json_block(raw_output: str) -> str:
    """
    Extract JSON block from model output.
    Handles:
    - ```json ... ``` fences
    - ``` ... ``` fences
    - Raw JSON starting with {
    - DeepSeek/Qwen3 <think>...</think> tags (skip thinking, extract JSON after)
    - DeepSeek R1 reasoning models without tags (detect and skip reasoning preamble)
    - Multiple JSON blocks (take the most complete one)
    - Natural language followed by JSON (common in weak models)
    """
    txt = raw_output.strip()

    # STEP 0: Handle DeepSeek/Qwen3 thinking tags
    think_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'\[think\].*?\[/think\]',
        r'<reasoning>.*?</reasoning>',
        r'<analysis>.*?</analysis>',
    ]
    for pattern in think_patterns:
        txt = re.sub(pattern, '', txt, flags=re.DOTALL)
    txt = txt.strip()

    # STEP 0.05: Handle Llama3.1 multi-JSON response pattern
    if '# END OF JSON RESPONSE' in txt or '  # END OF JSON' in txt:
        parts = re.split(r'\s*#\s*END\s+OF\s+JSON(?:\s+RESPONSE)?', txt)
        for part in parts:
            part = part.strip()
            if part and '{' in part:
                if re.search(r'\{\s*["\']?(?:plan_dag|plan|tool_calls|final_answer|nodes)["\']?\s*:', part):
                    txt = part
                    break

    # STEP 0.1: Handle DeepSeek R1 style reasoning WITHOUT tags
    reasoning_start_patterns = [
        r'^First,?\s+(?:I\s+(?:need|should|will|am)|the\s+response|let\s+me)',
        r'^(?:Let\s+me|I\s+(?:need|should|will)|To\s+(?:answer|solve|complete))',
        r'^(?:Okay|OK|Alright|Now),?\s+(?:I|let)',
        r'^The\s+(?:user|question|task|problem)\s+(?:is|asks|wants)',
        r'^(?:Analyzing|Understanding|Considering|Looking\s+at)',
    ]
    for pattern in reasoning_start_patterns:
        if re.match(pattern, txt, re.IGNORECASE):
            json_block_patterns = [
                r'\{\s*"plan_dag"\s*:',
                r'\{\s*"plan"\s*:',
                r'\{\s*"tool_calls"\s*:',
                r'\{\s*"final_answer"\s*:',
                r'\{\s*"nodes"\s*:',
            ]
            best_json_start = -1
            for jp in json_block_patterns:
                matches = list(re.finditer(jp, txt))
                if matches:
                    best_json_start = matches[-1].start()
                    break

            if best_json_start > 0:
                txt = txt[best_json_start:]
            break

    # STEP 0.5: Handle models that output natural language before JSON
    json_intro_patterns = [
        r'(?:^|[.!?]\s*)(?:here\s+is|here\'s|the\s+(?:JSON|json|response|output|answer|plan)(?:\s+is)?)\s*[:\-]?\s*',
        r'(?:^|[.!?]\s*)(?:final\s+(?:JSON|json|answer|output))\s*[:\-]?\s*',
        r'(?:^|[.!?]\s*)(?:output|response)\s*[:\-]\s*',
        r'\*\*(?:JSON|Output|Response|Plan)\*\*\s*[:\-]?\s*',
    ]
    for pattern in json_intro_patterns:
        match = re.search(pattern, txt, re.IGNORECASE | re.MULTILINE)
        if match:
            after_match = txt[match.end():]
            if '{' in after_match:
                txt = after_match.strip()
                break

    # STEP 1: Try to extract from markdown code blocks first
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, txt, re.DOTALL)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            if '{' in match:
                txt = match.strip()
                break

    # STEP 2: Find the outermost { ... } block
    start = txt.find("{")
    if start == -1:
        return ""

    # Find matching closing brace by counting
    depth = 0
    end = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(txt[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        return txt[start:end+1]
    else:
        return txt[start:]


def balance_brackets(s: str) -> str:
    """Balance unclosed brackets in JSON string."""
    if not s:
        return s

    open_curly = 0
    open_square = 0
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_curly += 1
        elif ch == '}':
            open_curly -= 1
        elif ch == '[':
            open_square += 1
        elif ch == ']':
            open_square -= 1

    result = s
    if open_curly > 0:
        result += "}" * open_curly
    if open_square > 0:
        result += "]" * open_square

    return result


def fix_common_json_errors(s: str) -> str:
    """Fix common JSON syntax errors from LLM output."""
    if not s:
        return s

    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # Fix unquoted keys
    s = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)

    # Remove JavaScript-style comments
    s = re.sub(r'//.*?\n', '\n', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    # Fix single quotes to double quotes
    s = re.sub(r"'([^']*)'(\s*[:],}])", r'"\1"\2', s)

    # Fix missing quotes around string values after colons
    s = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*([,}\]])', r': "\1"\2', s)

    # Fix "None" -> null
    s = re.sub(r':\s*None\s*([,}\]])', r': null\1', s)

    # Fix "True" -> true and "False" -> false
    s = re.sub(r':\s*True\s*([,}\]])', r': true\1', s)
    s = re.sub(r':\s*False\s*([,}\]])', r': false\1', s)

    # Remove ellipsis placeholders
    s = re.sub(r'\.\.\.\s*,?', '', s)
    s = re.sub(r'…\s*,?', '', s)

    return s


def try_extract_answer_from_text(raw_output: str) -> Optional[Dict]:
    """
    Last resort: try to extract a final answer from natural language output.
    Also attempts to extract tool usage from natural language descriptions.
    """
    # Look for answer patterns
    answer_patterns = [
        # LaTeX boxed
        (r'\$\$\s*\\boxed\{([^}]+)\}\s*\$\$', 1),
        (r'\\boxed\{([^}]+)\}', 1),
        # Explicit answer statements with quoted values
        (r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+["\']([^"\']+)["\']', 1),
        (r'(?:the\s+)?(?:final\s+)?answer\s*:\s*["\']([^"\']+)["\']', 1),
        # Numeric values
        (r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(\d+(?:\.\d+)?)', 1),
        (r'(?:the\s+)?(?:final\s+)?answer\s*:\s*(\d+(?:\.\d+)?)', 1),
        # Text values
        (r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([A-Za-z][A-Za-z0-9\s\-_]+?)(?:\.|,|\n|$)', 1),
        # Result patterns
        (r'result\s+is[:\s]+["\']([^"\']+)["\']', 1),
        (r'result\s*:\s*["\']([^"\']+)["\']', 1),
        (r'result\s+is[:\s]+(\d+(?:\.\d+)?)', 1),
        # Answer = value
        (r'answer\s*=\s*["\']([^"\']+)["\']', 1),
        (r'answer\s*=\s*(\d+(?:\.\d+)?)', 1),
    ]

    answer = None
    answer_type = "none"

    for pattern, group in answer_patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            answer = match.group(group).strip()
            if not answer:
                continue

            # Determine answer type
            answer_type = "string"
            try:
                if re.match(r'^-?\d+$', answer):
                    answer_type = "number"
                    answer = int(answer)
                elif re.match(r'^-?\d+\.\d+$', answer):
                    answer_type = "number"
                    answer = float(answer)
            except:
                pass
            break

    # Try to extract tool calls from natural language
    tool_calls = []
    plan_nodes = []

    tool_patterns = [
        (r'(?:use|call|invoke|run)\s+(web_search|web_browser|python_executor|file_reader|calculator|pdf_reader|code_interpreter)', 1),
        (r'(web_search|web_browser)(?:\s+(?:for|with|to))?', 1),
        (r'(?:search|google|look\s+up)\s+(?:for\s+)?["\']?([^"\']+)["\']?', None, 'web_search'),
        (r'(?:browse|visit|open|navigate\s+to)\s+(?:the\s+)?(?:webpage?|url|site|link)', None, 'web_browser'),
    ]

    node_idx = 0
    for pattern_tuple in tool_patterns:
        if len(pattern_tuple) == 2:
            pattern, group = pattern_tuple
            tool_override = None
        else:
            pattern, group, tool_override = pattern_tuple

        for match in re.finditer(pattern, raw_output, re.IGNORECASE):
            if tool_override:
                tool_id = tool_override
            elif group:
                tool_id = match.group(group).lower()
            else:
                continue

            tool_id = tool_id.replace(' ', '_')

            existing_tools = {tc.get('tool_id') for tc in tool_calls}
            if tool_id not in existing_tools:
                node_id = f"n{node_idx}"
                tool_calls.append({
                    "call_index": node_idx,
                    "node_id": node_id,
                    "tool_id": tool_id,
                    "arguments": []
                })
                plan_nodes.append({
                    "node_id": node_id,
                    "step_index": node_idx,
                    "label": f"Use {tool_id}",
                    "step_type": "tool",
                    "tool_id": tool_id,
                    "arguments": {},
                    "output_vars": [f"<{node_id}>"],
                    "needs_tool": True,
                    "needs_new_tool": False
                })
                node_idx += 1

    # Only return if we found something useful
    if answer is not None or tool_calls:
        return {
            "plan_dag": {
                "nodes": plan_nodes,
                "edges": []
            },
            "tool_calls": tool_calls,
            "final_answer": {
                "answer_type": answer_type,
                "answer": answer,
                "aliases": [],
                "tolerance": 0.0
            },
            "_extracted_from_text": True
        }

    return None


def _parse_json_with_repairs(raw_output: str, json_str: str) -> Tuple[Optional[Dict], str]:
    """
    Parse a JSON object after extraction, with syntax-repair fallbacks only.

    This helper intentionally does NOT attempt any natural-language extraction.
    Callers that want text-based fallbacks should do so explicitly.
    """
    # Step 1: Try direct parse
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "Parsed JSON was not an object"
    except JSONDecodeError:
        pass

    # Step 2: Fix common errors and balance brackets
    json_str = fix_common_json_errors(json_str)
    json_str = balance_brackets(json_str)

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "Parsed JSON was not an object"
    except JSONDecodeError:
        pass

    # Step 3: Try truncation strategies
    for _ in range(10):
        last_valid_pos = -1
        for pattern in ['},', '},\n', '],', '],\n', '}"', ']"', '}\n', ']\n']:
            pos = json_str.rfind(pattern)
            if pos > last_valid_pos:
                last_valid_pos = pos + len(pattern) - 1

        if last_valid_pos <= 0:
            break

        json_str = json_str[:last_valid_pos + 1]
        json_str = balance_brackets(json_str)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed, ""
            return None, "Parsed JSON was not an object"
        except JSONDecodeError:
            continue

    # Step 4: Line-by-line truncation
    lines = raw_output.split('\n')
    for end_idx in range(len(lines), 0, -1):
        partial = '\n'.join(lines[:end_idx])
        partial = extract_json_block(partial)
        if not partial:
            continue
        partial = fix_common_json_errors(partial)
        partial = balance_brackets(partial)
        try:
            parsed = json.loads(partial)
            if isinstance(parsed, dict):
                return parsed, ""
            return None, "Parsed JSON was not an object"
        except JSONDecodeError:
            continue

    return None, f"Failed to parse JSON. Raw start: {raw_output[:200]}"


def parse_action_json(raw_output: str) -> Tuple[Optional[Dict], str]:
    """
    Parse Stage 4 action JSON only.

    Unlike robust_json_parse(), this function refuses to invent planning-style
    fallback objects from natural language. That keeps action parsing from
    accidentally treating free-form reasoning as a valid decision.
    """
    if not raw_output or not raw_output.strip():
        return None, "Empty output from model"

    json_str = extract_json_block(raw_output)
    if not json_str:
        return None, "No JSON block found in output"

    parsed, err = _parse_json_with_repairs(raw_output, json_str)
    if not parsed:
        return None, err

    # Some models wrap the action once inside an outer key.
    for wrapper_key in ("action", "next_action", "decision"):
        inner = parsed.get(wrapper_key)
        if len(parsed) == 1 and isinstance(inner, dict):
            parsed = inner
            break

    return parsed, ""


def robust_json_parse(raw_output: str, fallback_empty: bool = True) -> Tuple[Optional[Dict], str]:
    """
    Robustly parse JSON from LLM output with multiple fallback strategies.
    """
    if not raw_output or not raw_output.strip():
        return None, "Empty output from model"

    # Step 1: Extract JSON block
    json_str = extract_json_block(raw_output)
    if not json_str:
        # Step 1b: Try to extract answer from natural language
        extracted = try_extract_answer_from_text(raw_output)
        if extracted:
            return extracted, ""
        return None, "No JSON block found in output"

    parsed, err = _parse_json_with_repairs(raw_output, json_str)
    if parsed:
        return parsed, ""

    # Final fallback - try to extract answer from natural language
    extracted = try_extract_answer_from_text(raw_output)
    if extracted:
        return extracted, ""

    return None, err


# ============================================================================
# Structure Normalization
# ============================================================================

KNOWN_TOOL_IDS = {
    "web_search", "web_browser", "wikipedia_search", "pdf_reader",
    "excel_reader", "file_reader", "pptx_reader", "zip_extractor",
    "image_recognition", "audio_transcription", "video_analysis",
    "python_executor", "code_interpreter", "calculator", "pdb_analyzer",
    "date_calculator", "unit_converter", "reasoning"
}


def normalize_pred_structure(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize various LLM output formats to the expected structure.
    Handles common variations like:
    - 'plan' vs 'plan_dag'
    - 'steps' vs 'nodes'
    - Missing tool_calls
    - Various field name variations
    - Placeholder detection and removal
    """
    result = pred.copy()

    # STEP -1: Unwrap single-key root objects
    if len(result) == 1:
        single_key = list(result.keys())[0]
        inner = result[single_key]
        if isinstance(inner, dict):
            expected_fields = {"plan_dag", "plan", "tool_calls", "final_answer", "nodes", "edges"}
            if any(f in inner for f in expected_fields):
                result = inner.copy()

    # STEP 0: Detect flat node structure
    node_fields = {"node_id", "step_index", "label", "step_type", "tool_id",
                   "arguments", "output_vars", "needs_tool", "needs_new_tool"}
    top_level_node_fields = node_fields & set(result.keys())

    if top_level_node_fields and ("plan_dag" not in result or
        (isinstance(result.get("plan_dag"), dict) and not result["plan_dag"].get("nodes"))):
        single_node = {}
        for field in node_fields:
            if field in result:
                single_node[field] = result.pop(field)

        if "node_id" not in single_node:
            single_node["node_id"] = f"n{single_node.get('step_index', 0)}"

        if "plan_dag" not in result or not isinstance(result.get("plan_dag"), dict):
            result["plan_dag"] = {"nodes": [], "edges": []}

        if single_node:
            result["plan_dag"]["nodes"] = [single_node]
            result["plan_dag"]["edges"] = []

    # Normalize plan_dag
    if "plan_dag" not in result:
        if "plan" in result:
            plan = result.pop("plan")
            if isinstance(plan, dict):
                if "final_answer" in plan and "final_answer" not in result:
                    result["final_answer"] = plan.pop("final_answer")
                if "calls" in plan and "tool_calls" not in result:
                    result["tool_calls"] = plan.pop("calls")
                if "tool_calls" in plan and "tool_calls" not in result:
                    result["tool_calls"] = plan.pop("tool_calls")
            result["plan_dag"] = plan
        else:
            result["plan_dag"] = {"nodes": [], "edges": []}

    if "calls" in result and "tool_calls" not in result:
        result["tool_calls"] = result.pop("calls")

    dag = result["plan_dag"]

    # Handle case where plan_dag is a list
    if isinstance(dag, list):
        result["plan_dag"] = {"nodes": dag, "edges": []}
        dag = result["plan_dag"]

    if not isinstance(dag, dict):
        result["plan_dag"] = {"nodes": [], "edges": []}
        dag = result["plan_dag"]

    # Normalize nodes
    if "nodes" not in dag:
        if "steps" in dag:
            dag["nodes"] = dag.pop("steps")
        else:
            dag["nodes"] = []

    if "edges" not in dag:
        dag["edges"] = []

    # Normalize edge format
    normalized_edges = []
    for edge in dag.get("edges", []):
        normalized_edge = edge.copy()
        if "from_node" in normalized_edge and "source" not in normalized_edge:
            normalized_edge["source"] = normalized_edge.pop("from_node")
        if "to_node" in normalized_edge and "target" not in normalized_edge:
            normalized_edge["target"] = normalized_edge.pop("to_node")
        if "src" in normalized_edge and "source" not in normalized_edge:
            normalized_edge["source"] = normalized_edge.pop("src")
        if "dst" in normalized_edge and "target" not in normalized_edge:
            normalized_edge["target"] = normalized_edge.pop("dst")
        if "dest" in normalized_edge and "target" not in normalized_edge:
            normalized_edge["target"] = normalized_edge.pop("dest")
        normalized_edges.append(normalized_edge)
    dag["edges"] = normalized_edges

    # Normalize nodes
    for i, node in enumerate(dag.get("nodes", [])):
        # Field name normalization
        if "id" in node and "node_id" not in node:
            node["node_id"] = node.pop("id")
        if "step" in node and "step_index" not in node:
            node["step_index"] = node.pop("step")
        if "type" in node and "step_type" not in node:
            node["step_type"] = node.pop("type")
        if "tool" in node and "tool_id" not in node:
            node["tool_id"] = node.pop("tool")
        if "query" in node and "arguments" not in node:
            node["arguments"] = {"query": node.pop("query")}
        if "code" in node and "arguments" not in node:
            node["arguments"] = {"code": node.pop("code")}

        if "node_id" not in node:
            node["node_id"] = f"n{node.get('step_index', i)}"

        # Fix step_type if it contains a tool name
        step_type = node.get("step_type", "")
        if step_type and step_type.lower() in KNOWN_TOOL_IDS:
            if not node.get("tool_id"):
                node["tool_id"] = step_type.lower()
            node["step_type"] = "tool"
        elif step_type and step_type not in ("tool", "thought", "action", ""):
            if node.get("tool_id"):
                node["step_type"] = "tool"
            else:
                node["step_type"] = "thought"

    # Generate tool_calls from plan_dag if missing
    if "tool_calls" not in result or not result["tool_calls"]:
        tool_calls = []
        for i, node in enumerate(dag.get("nodes", [])):
            tool_id = node.get("tool_id")
            if tool_id:
                tool_calls.append({
                    "call_index": len(tool_calls),
                    "node_id": node.get("node_id", f"n{i}"),
                    "tool_id": tool_id,
                    "arguments": node.get("arguments", []) or []
                })
        result["tool_calls"] = tool_calls

    # Normalize tool_calls arguments
    for tc in result.get("tool_calls", []):
        args = tc.get("arguments")
        if isinstance(args, dict):
            tc["arguments"] = [{"name": k, "value": v} for k, v in args.items()]
        elif args is None:
            tc["arguments"] = []

    # Ensure final_answer exists
    if "final_answer" not in result:
        result["final_answer"] = {
            "answer_type": "none",
            "answer": None,
            "aliases": [],
            "tolerance": 0.0
        }
    elif isinstance(result["final_answer"], dict):
        fa = result["final_answer"]
        if "type" in fa and "answer_type" not in fa:
            fa["answer_type"] = fa.pop("type")
        if "value" in fa and "answer" not in fa:
            fa["answer"] = fa.pop("value")
        if "answer_type" not in fa:
            fa["answer_type"] = "string" if fa.get("answer") else "none"
        if "answer" not in fa:
            fa["answer"] = None
        if "aliases" not in fa:
            fa["aliases"] = []
        if "tolerance" not in fa:
            fa["tolerance"] = 0.0
    elif isinstance(result["final_answer"], (int, float)):
        result["final_answer"] = {
            "answer_type": "number",
            "answer": result["final_answer"],
            "aliases": [],
            "tolerance": 0.0
        }
    elif isinstance(result["final_answer"], str):
        ans_str = result["final_answer"]
        try:
            float(ans_str)
            ans_type = "number"
        except (ValueError, TypeError):
            ans_type = "string" if ans_str else "none"
        result["final_answer"] = {
            "answer_type": ans_type,
            "answer": ans_str if ans_str else None,
            "aliases": [],
            "tolerance": 0.0
        }

    # Detect and remove placeholder answers
    if "final_answer" in result and isinstance(result["final_answer"], dict):
        answer = result["final_answer"].get("answer")
        if isinstance(answer, str):
            placeholder_patterns = [
                r"<[^>]+>",
                r"PUT\s+YOUR\s+.*\s+HERE",
                r"YOUR_.*_ANSWER",
                r"TOOL_NAME",
                r"^\s*answer\s+here\s*$",
                r"^\s*computed\s+result\s*$",
                r"^\s*EXAMPLE\s*$",
                r"^\s*placeholder\s*$",
                r"^\s*unable\s+to\s+determine\s*$",
                r"^\s*cannot\s+determine\s*$",
                r"^\s*not\s+enough\s+information\s*$",
                r"^\s*N/?A\s*$",
                r"^\s*unknown\s*$",
                r"^\s*none\s*$",
                r"^\s*null\s*$",
                r"^\s*\[\s*\]\s*$",
                r"^\s*\{\s*\}\s*$",
            ]

            is_placeholder = False
            answer_upper = answer.upper()
            for pattern in placeholder_patterns:
                if re.search(pattern, answer_upper):
                    is_placeholder = True
                    break

            if is_placeholder:
                result["final_answer"]["answer"] = None
                result["final_answer"]["answer_type"] = "none"
                if "_warnings" not in result:
                    result["_warnings"] = []
                result["_warnings"].append(f"Placeholder detected and removed: {answer[:100]}")

    return result


def create_empty_pred(model_name: str) -> Dict[str, Any]:
    """Create an empty prediction structure."""
    return {
        "model_name": model_name,
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [],
        "final_answer": {
            "answer_type": "none",
            "answer": None,
            "aliases": [],
            "tolerance": 0.0,
        },
        "_parse_error": True,
    }
