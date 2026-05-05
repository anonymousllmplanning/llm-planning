"""Execution-normalized value matching for GAIA tool arguments.

This module is intentionally side-effect free.  For GAIA, ToolScores uses this
execution-normalized matcher as the main ParamValueF1 component and keeps the
older strict type-aware value score as an audit field.  The normalization is
conservative: it credits stable executable resources and information needs
without treating free-form prompts or raw code text as stable value slots.
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote, urlparse


TEXT_ARGS = {"query"}
IGNORED_TEXT_ARGS = {"problem", "custom_prompt"}
IGNORED_TOOLS = {"submit_final_answer"}
FILE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".jsonld",
    ".mp3",
    ".pdf",
    ".pdb",
    ".png",
    ".pptx",
    ".tif",
    ".tiff",
    ".txt",
    ".wav",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


@dataclass(frozen=True)
class CanonicalValueEntry:
    """A canonicalized tool argument value for normalized value-F1."""

    tool: str
    arg: str
    value: Any
    source_tool: str
    source_arg: str


def _extract_tool_name(tool_id: str) -> str:
    return str(tool_id or "").split("/")[-1]


def _iter_arguments(arguments: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(arguments, dict):
        yield from arguments.items()
        return
    for arg in arguments or []:
        if isinstance(arg, dict):
            name = str(arg.get("name") or "")
            if name:
                yield name, arg.get("value")
        elif isinstance(arg, (list, tuple)) and len(arg) >= 2:
            yield str(arg[0]), arg[1]


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9./:_+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> List[str]:
    text = _normalize_text(value)
    return text.split() if text else []


def _token_f1(a: Any, b: Any) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca: Dict[str, int] = {}
    cb: Dict[str, int] = {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    overlap = sum(min(ca.get(t, 0), cb.get(t, 0)) for t in set(ca) | set(cb))
    if overlap == 0:
        return 0.0
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    return 2 * precision * recall / (precision + recall)


def _char_ngram_jaccard(a: Any, b: Any, n: int = 3) -> float:
    sa = _normalize_text(a).replace(" ", "")
    sb = _normalize_text(b).replace(" ", "")
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    if len(sa) < n or len(sb) < n:
        return 1.0 if sa == sb else 0.0
    ga = {sa[i : i + n] for i in range(len(sa) - n + 1)}
    gb = {sb[i : i + n] for i in range(len(sb) - n + 1)}
    return len(ga & gb) / len(ga | gb) if ga or gb else 0.0


def _text_similarity(a: Any, b: Any) -> float:
    return 0.75 * _token_f1(a, b) + 0.25 * _char_ngram_jaccard(a, b)


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip().strip("'\"").replace("\\", "/")
    if not text:
        return ""
    text = text.replace("data.hg/GAIA/", "data/GAIA/")
    text = text.replace("data/Augmented/", "data/GAIA/")
    text = text.replace("data.hg/Augmented/", "data/GAIA/")
    text = os.path.normpath(text).replace("\\", "/")
    marker = "/data/GAIA/"
    if marker in text:
        text = "data/GAIA/" + text.split(marker, 1)[1]
    elif text.startswith("data/GAIA/"):
        pass
    return text


def _path_identity(value: Any) -> str:
    text = _normalize_path(value)
    if not text:
        return ""
    basename = os.path.basename(text)
    if basename:
        return basename.lower()
    return text.lower()


def _canonicalize_url(value: Any) -> str:
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else "https://" + text)
    host = parsed.netloc.lower()
    path = unquote(parsed.path or "/").rstrip("/") or "/"
    query = "&".join(
        f"{k}={v}"
        for k, v in sorted(parse_qsl(parsed.query, keep_blank_values=False))
        if not k.lower().startswith("utm_")
    )
    return f"{host}{path}" + (f"?{query}" if query else "")


def _url_similarity(a: Any, b: Any) -> float:
    ca, cb = _canonicalize_url(a), _canonicalize_url(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    pa = urlparse("https://" + ca)
    pb = urlparse("https://" + cb)
    if pa.netloc == pb.netloc and os.path.basename(pa.path) == os.path.basename(pb.path):
        return 1.0
    return 0.0


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
)
_SAFE_MATH_NAMES = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "round": round,
    "sqrt": math.sqrt,
}


def _safe_eval_expression(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in _SAFE_MATH_NAMES:
            return None
    try:
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_MATH_NAMES)
    except Exception:
        return None
    try:
        return float(result)
    except Exception:
        return None


def _expression_similarity(a: Any, b: Any) -> float:
    ea = _safe_eval_expression(a)
    eb = _safe_eval_expression(b)
    if ea is not None and eb is not None:
        return 1.0 if abs(ea - eb) <= 1e-9 * max(1.0, abs(ea), abs(eb)) else 0.0
    return 1.0 if _normalize_text(a) == _normalize_text(b) else 0.0


def _extract_paths_from_code(code: Any) -> List[str]:
    text = str(code or "")
    pattern = r"""['"]([^'"]+?(?:%s))['"]""" % "|".join(re.escape(ext) for ext in sorted(FILE_EXTENSIONS))
    paths: List[str] = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        value = match.group(1)
        if "/" in value or "\\" in value or "data" in value.lower():
            paths.append(value)
    return paths


def _route_file_reader(tool: str, args: Dict[str, Any]) -> str:
    if tool != "file_reader":
        return tool
    ext = os.path.splitext(str(args.get("file_path") or ""))[1].lower()
    if ext in {".xlsx", ".xls", ".csv"}:
        return "excel_reader"
    if ext == ".pdf":
        return "pdf_reader"
    if ext == ".pptx":
        return "pptx_reader"
    if ext == ".zip":
        return "zip_extractor"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return "image_recognition"
    return tool


def canonicalize_tool_call(
    call: Dict[str, Any],
    *,
    include_open_text: bool = True,
    include_reasoning_problem: bool = False,
    include_custom_prompt: bool = False,
    include_code_text: bool = False,
) -> List[CanonicalValueEntry]:
    """Canonicalize a single tool call into stable value entries."""

    source_tool = _extract_tool_name(call.get("tool_id") or call.get("name") or "")
    if source_tool in IGNORED_TOOLS:
        return []
    raw_args = {name: value for name, value in _iter_arguments(call.get("arguments", []))}

    if source_tool == "reasoning" and "problem" not in raw_args:
        for alias in ("query", "question", "task", "prompt", "text", "input"):
            if alias in raw_args:
                raw_args["problem"] = raw_args[alias]
                break
    if source_tool == "python_executor" and "code" not in raw_args:
        for alias in ("python", "python_code", "script", "program", "snippet"):
            if alias in raw_args:
                raw_args["code"] = raw_args[alias]
                break
        if "expression" in raw_args and "code" not in raw_args:
            raw_args["code"] = f"result = {raw_args['expression']}\nprint(result)"
    if "image_path" in raw_args and "file_path" not in raw_args:
        raw_args["file_path"] = raw_args["image_path"]
    if "audio_path" in raw_args and "file_path" not in raw_args:
        raw_args["file_path"] = raw_args["audio_path"]
    if source_tool == "web_browser" and "file_path" in raw_args and "url" not in raw_args:
        raw_args["url"] = raw_args["file_path"]

    tool = _route_file_reader(source_tool, raw_args)
    entries: List[CanonicalValueEntry] = []

    def add(c_tool: str, c_arg: str, value: Any, src_arg: str) -> None:
        entries.append(CanonicalValueEntry(c_tool, c_arg, value, source_tool, src_arg))

    file_value = raw_args.get("file_path")
    if file_value:
        add("resource", "file_path", _path_identity(file_value), "file_path")
        add(tool, "file_path", _path_identity(file_value), "file_path")

    if tool == "image_recognition" and raw_args.get("task"):
        add("image_recognition", "task", str(raw_args["task"]).strip().lower(), "task")

    if raw_args.get("url"):
        add("web_resource", "url", _canonicalize_url(raw_args["url"]), "url")
        add(tool, "url", _canonicalize_url(raw_args["url"]), "url")

    if tool == "web_search" and raw_args.get("query") and include_open_text:
        add("web_search", "query", raw_args["query"], "query")

    if tool == "calculator" and raw_args.get("expression") is not None:
        add("calculator", "expression", raw_args["expression"], "expression")

    if source_tool == "python_executor" and raw_args.get("code"):
        code = raw_args["code"]
        for path in _extract_paths_from_code(code):
            path_identity = _path_identity(path)
            add("resource", "file_path", path_identity, "code")
            routed_tool = _route_file_reader("file_reader", {"file_path": path})
            add(routed_tool, "file_path", path_identity, "code")
        if include_code_text:
            add("python_executor", "code", code, "code")

    if source_tool == "reasoning" and raw_args.get("problem") and include_reasoning_problem:
        add("reasoning", "problem", raw_args["problem"], "problem")

    if tool == "image_recognition" and raw_args.get("custom_prompt") and include_custom_prompt:
        add("image_recognition", "custom_prompt", raw_args["custom_prompt"], "custom_prompt")

    return entries


def canonicalize_tool_calls(calls: Sequence[Dict[str, Any]], **kwargs: Any) -> List[CanonicalValueEntry]:
    entries: List[CanonicalValueEntry] = []
    for call in calls or []:
        entries.extend(canonicalize_tool_call(call, **kwargs))
    return entries


def value_similarity(gold: CanonicalValueEntry, pred: CanonicalValueEntry) -> float:
    if gold.tool != pred.tool or gold.arg != pred.arg:
        return 0.0
    if gold.arg == "file_path":
        return 1.0 if str(gold.value or "") == str(pred.value or "") else 0.0
    if gold.arg == "url":
        return _url_similarity(gold.value, pred.value)
    if gold.arg == "expression":
        return _expression_similarity(gold.value, pred.value)
    if gold.arg == "query":
        return _text_similarity(gold.value, pred.value)
    if gold.arg in IGNORED_TEXT_ARGS or gold.arg == "code":
        return _text_similarity(gold.value, pred.value)
    return 1.0 if _normalize_text(gold.value) == _normalize_text(pred.value) else 0.0


def _best_matches(sim_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Greedy maximum matching; enough for audit-side value scoring."""

    candidates: List[Tuple[float, int, int]] = []
    for i, row in enumerate(sim_matrix):
        for j, score in enumerate(row):
            candidates.append((score, i, j))
    used_i = set()
    used_j = set()
    matches: List[Tuple[int, int]] = []
    for score, i, j in sorted(candidates, reverse=True):
        if score <= 0 or i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))
    return matches


def normalized_value_f1(
    gold_calls: Sequence[Dict[str, Any]],
    pred_calls: Sequence[Dict[str, Any]],
    *,
    threshold: float = 0.45,
    **canonicalize_kwargs: Any,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute execution-normalized value-F1 and return diagnostics."""

    gold_entries = canonicalize_tool_calls(gold_calls, **canonicalize_kwargs)
    pred_entries = canonicalize_tool_calls(pred_calls, **canonicalize_kwargs)
    if not gold_entries:
        return None, {"gold_entries": 0, "pred_entries": len(pred_entries), "matched": 0}
    if not pred_entries:
        return 0.0, {"gold_entries": len(gold_entries), "pred_entries": 0, "matched": 0}

    sim_matrix = [
        [value_similarity(gold, pred) for pred in pred_entries]
        for gold in gold_entries
    ]
    matches = _best_matches(sim_matrix)
    matched = sum(1 for i, j in matches if sim_matrix[i][j] >= threshold)
    precision = matched / len(pred_entries) if pred_entries else 0.0
    recall = matched / len(gold_entries) if gold_entries else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, {
        "gold_entries": len(gold_entries),
        "pred_entries": len(pred_entries),
        "matched": matched,
        "precision": precision,
        "recall": recall,
    }
