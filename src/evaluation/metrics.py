#!/usr/bin/env python3
"""
AST Evaluation System V2 - Fixed and Dataset-Aware Version

Key Improvements over V1:
1. Dataset-aware metrics (Delta vs TaskBench/UltraTool handling)
2. Fixed empty-set handling for param F1 and value F1
3. Proper tool F1 calculation with alternative tools support
4. Better handling of different tool_id formats
5. Explicit "not applicable" (N/A) for metrics that don't apply to certain datasets

Metric Definitions (dataset-aware):
- Node F1:
  * Main evaluator uses strict semantic one-to-one NodeF1.
  * Node alignment uses the active sentence-transformer node matcher by default.
  * Span-Node F1 is retained only as a diagnostic field.
- Link/Edge F1: strict and semantically re-anchored edge matching are both reported
- t-F1 (param_name_f1): F1 of "{tool_name}-{param_name}" strings
- v-F1 (type_aware_value_f1): value F1 used in ToolUsageScore.
  GAIA uses execution-normalized stable value entries; other benchmarks keep
  the strict type-aware value matcher.
- PlanningScore: (StrictNodeF1 + SemanticEdgeF1) / 2
- SSI: (node_label_similarity + edge_f1) / 2, where edge_f1 is
  the paper-facing semantic direct-edge score. raw_edge_f1 is retained as an
  index/step-position diagnostic.

For Delta dataset (no arguments):
- param_name_f1 and type_aware_value_f1 return None (not applicable)
- These should not be included in aggregate scores
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Set, Sequence
import math
import os
import re
import warnings
from collections import Counter
from urllib.parse import parse_qsl, unquote, urlparse
from src.config.models import get_verifier_api_base, get_verifier_api_key
from src.evaluation.value_normalization import normalized_value_f1
from src.inference.prompts import normalize_tool_environment

warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    linear_sum_assignment = None

def _float_env(name: str, default: float) -> float:
    """Read a floating-point environment setting with a safe fallback."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}; using {default}")
        return default


def _int_env(name: str, default: int) -> int:
    """Read an integer environment setting with a safe fallback."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}; using {default}")
        return default


# Lazy loading for sentence transformers
_EMBEDDING_MODEL = None
_EMBEDDING_CACHE: Dict[str, Any] = {}
_EMBEDDING_FALLBACK_WARNED = False
_EMBEDDING_MODEL_NAME = os.getenv(
    "LLM_PLANNING_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
_NODE_MATCH_BACKEND = os.getenv("LLM_PLANNING_NODE_MATCH_BACKEND", "embedding").strip().lower()

# Main SpanNodeF1 threshold. The default is calibrated for sentence-transformer
# cosine similarities. Set LLM_PLANNING_NODE_MATCH_BACKEND=token and
# LLM_PLANNING_NODE_MATCH_THRESHOLD=0.35 to reproduce the previous deterministic
# token/tool matcher.
_NODE_MATCH_THRESHOLD = _float_env("LLM_PLANNING_NODE_MATCH_THRESHOLD", 0.45)
_SPAN_ALIGNMENT_EXACT_CANDIDATE_LIMIT = _int_env(
    "LLM_PLANNING_SPAN_ALIGNMENT_EXACT_CANDIDATE_LIMIT",
    250,
)
_NODE_MATCH_STEP_TYPE_BONUS = 0.02
_NODE_MATCH_TOOL_ID_BONUS = 0.03
_NODE_MATCH_TOOL_NAME_BONUS = 0.015
_NODE_MATCH_CHAR_TIEBREAKER_SCALE = 0.02
_SPAN_NODE_MAX_SIZE = 3
_NODE_MATCH_STOPWORDS = {
    "the", "a", "an", "to", "and", "of", "for", "in", "on", "from", "with",
    "into", "page", "go", "through", "that", "is", "be", "find", "identify",
    "search", "examine", "compare", "note", "navigate", "enter", "select",
    "submit", "look", "use", "return", "back", "open", "read", "get",
    "compute", "calculate", "obtain", "extract", "paper", "article",
}


def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
            try:
                import torch
                if torch.cuda.is_available():
                    _EMBEDDING_MODEL = _EMBEDDING_MODEL.to('cuda')
            except:
                pass
            print(f"[INFO] Loaded embedding model: {_EMBEDDING_MODEL_NAME}")
        except ImportError:
            print("[WARN] sentence-transformers not installed. Using fallback string similarity.")
            _EMBEDDING_MODEL = "FALLBACK"
        except Exception as e:
            print(f"[WARN] Failed to load embedding model: {e}. Using fallback.")
            _EMBEDDING_MODEL = "FALLBACK"
    return _EMBEDDING_MODEL


def _compute_embeddings(texts: List[str]) -> Optional[Any]:
    """Compute embeddings for a list of texts."""
    if not HAS_NUMPY or not texts:
        return None

    model = _get_embedding_model()
    if model == "FALLBACK":
        return None

    try:
        normalized = [" ".join(str(text or "").split()) for text in texts]
        missing = [text for text in dict.fromkeys(normalized) if text not in _EMBEDDING_CACHE]
        if missing:
            try:
                vectors = model.encode(
                    missing,
                    batch_size=128,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except TypeError:
                vectors = model.encode(
                    missing,
                    batch_size=128,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            for text, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=float)
                norm = np.linalg.norm(vector)
                if norm > 1e-8:
                    vector = vector / norm
                _EMBEDDING_CACHE[text] = vector
        return np.asarray([_EMBEDDING_CACHE[text] for text in normalized], dtype=float)
    except Exception as e:
        print(f"[WARN] Embedding computation failed: {e}")
        return None


def _cosine_similarity(v1, v2) -> float:
    """Compute cosine similarity between two vectors."""
    if not HAS_NUMPY:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ============================================================================
# String Utilities
# ============================================================================

def _normalize_string(s: str) -> str:
    """Normalize a string for comparison."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_value_for_comparison(value: str) -> str:
    """
    Normalize a parameter value for more robust comparison.

    Handles:
    - Numeric values: "100.0" -> "100", "3.14159" -> "3.14"
    - File paths: "/path/to/file.txt" -> "file.txt"
    - URLs: "https://example.com/page" -> "example.com/page"
    - Boolean strings: "True" -> "true", "yes" -> "true"
    - List-like strings: "[1, 2, 3]" -> "1,2,3"
    """
    if not value:
        return ""

    value = _normalize_string(value)

    # Try to normalize as number
    try:
        num = float(value)
        if num == int(num):
            return str(int(num))  # "100.0" -> "100"
        else:
            # Round to 2 decimal places for comparison
            return f"{num:.2f}".rstrip('0').rstrip('.')
    except ValueError:
        pass

    # Normalize file paths - extract filename
    if "/" in value or "\\" in value:
        # Keep just the filename for paths
        parts = value.replace("\\", "/").split("/")
        if parts and parts[-1]:
            value = parts[-1]

    # Normalize URLs
    if value.startswith("http://") or value.startswith("https://"):
        value = re.sub(r"^https?://", "", value)

    # Normalize booleans
    if value in ["true", "yes", "1"]:
        return "true"
    if value in ["false", "no", "0"]:
        return "false"

    # Normalize list-like strings
    if value.startswith("[") and value.endswith("]"):
        # Remove brackets and extra spaces
        inner = value[1:-1].strip()
        # Split by comma and rejoin without spaces
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return ",".join(parts)

    # Remove quotes
    value = value.strip("'\"")

    return value


_URL_TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "session", "spm",
}

_TYPE_AWARE_TEXT_ARGS = {"query", "problem", "custom_prompt"}
_TYPE_AWARE_EXACT_ARGS = {"task", "language", "action", "sheet", "page", "slide_number", "answer_type"}
_TYPE_AWARE_THRESHOLD = 0.5


def _multiset_token_f1(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Token-level F1 over multisets."""
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    ca, cb = Counter(a_tokens), Counter(b_tokens)
    overlap = sum((ca & cb).values())
    precision = overlap / sum(cb.values())
    recall = overlap / sum(ca.values())
    return _f1_from_prec_recall(precision, recall)


def _text_value_similarity(a: str, b: str) -> float:
    """Lightweight semantic-ish similarity for open-ended textual arguments."""
    token_f1 = _multiset_token_f1(_tokenize(a), _tokenize(b))
    char_sim = _char_ngram_jaccard(a, b)
    return 0.8 * token_f1 + 0.2 * char_sim


def _code_tokens(text: str) -> List[str]:
    """Tokenize code-like strings without requiring parsing."""
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|[-+*/%=(){}\[\],.:]", text)


def _code_value_similarity(a: str, b: str) -> float:
    """Shallow similarity for code-valued arguments."""
    token_f1 = _multiset_token_f1(_code_tokens(a), _code_tokens(b))
    char_sim = _char_ngram_jaccard(a, b)
    return 0.7 * token_f1 + 0.3 * char_sim


def _normalize_file_path_for_match(value: str) -> str:
    """Normalize a file path while being robust to absolute/relative prefixes."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""
    text = text.replace("\\", "/")
    text = os.path.normpath(text).replace("\\", "/")
    return text


def _file_path_similarity(a: str, b: str) -> float:
    """Identifier-like matching for file paths."""
    a_norm = _normalize_file_path_for_match(a)
    b_norm = _normalize_file_path_for_match(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm.endswith("/" + b_norm) or b_norm.endswith("/" + a_norm):
        return 1.0
    a_base = os.path.basename(a_norm)
    b_base = os.path.basename(b_norm)
    if a_base and a_base == b_base:
        return 1.0
    return 0.0


def _canonicalize_url_for_match(value: str) -> str:
    """Normalize URL strings for conservative identifier-style matching."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""

    parsed = urlparse(text)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlparse("https://" + text)

    host = parsed.netloc.lower()
    path = unquote(parsed.path or "").rstrip("/")
    if not path:
        path = "/"

    if "arxiv.org" in host and path.startswith("/abs/"):
        paper_id = path.split("/abs/", 1)[1].strip("/")
        if paper_id:
            path = f"/pdf/{paper_id}.pdf"

    query_items = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in _URL_TRACKING_KEYS
    ]
    query = "&".join(f"{k}={v}" for k, v in sorted(query_items))
    return f"{host}{path}" + (f"?{query}" if query else "")


def _extract_url_identity(url: str) -> Tuple[str, str, Set[str]]:
    """Extract host, basename, and stable ID-like tokens from a canonical URL."""
    canon = _canonicalize_url_for_match(url)
    if not canon:
        return "", "", set()
    parsed = urlparse("https://" + canon if "://" not in canon else canon)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    basename = os.path.basename(path.rstrip("/"))
    stable_ids = set(re.findall(r"[A-Za-z]*\d{4,}[A-Za-z0-9._-]*", canon))
    return host, basename, stable_ids


def _url_similarity(a: str, b: str) -> float:
    """Conservative matching for URLs with light canonicalization."""
    a_canon = _canonicalize_url_for_match(a)
    b_canon = _canonicalize_url_for_match(b)
    if not a_canon and not b_canon:
        return 1.0
    if not a_canon or not b_canon:
        return 0.0
    if a_canon == b_canon:
        return 1.0

    a_host, a_base, a_ids = _extract_url_identity(a)
    b_host, b_base, b_ids = _extract_url_identity(b)
    if a_host and a_host == b_host:
        if a_base and a_base == b_base:
            return 1.0
        if a_ids and b_ids and (a_ids & b_ids):
            return 1.0
    return 0.0


def _expression_similarity(a: str, b: str) -> float:
    """Numeric-equivalence-first matching for calculator expressions."""
    a_norm = _normalize_value_for_comparison(a)
    b_norm = _normalize_value_for_comparison(b)
    if a_norm == b_norm:
        return 1.0
    try:
        return 1.0 if abs(float(a_norm) - float(b_norm)) < 1e-9 else 0.0
    except Exception:
        return 0.0


def _type_aware_value_similarity(tool_name: str, arg_name: str, gold_value: Any, pred_value: Any) -> float:
    """Argument-type-aware value similarity for heterogeneous tool parameters."""
    arg_name = str(arg_name or "").strip()
    gold_text = "" if gold_value is None else str(gold_value)
    pred_text = "" if pred_value is None else str(pred_value)

    if arg_name == "file_path":
        return _file_path_similarity(gold_text, pred_text)
    if arg_name == "url":
        return _url_similarity(gold_text, pred_text)
    if arg_name == "expression":
        return _expression_similarity(gold_text, pred_text)
    if arg_name in _TYPE_AWARE_TEXT_ARGS:
        return _text_value_similarity(gold_text, pred_text)
    if arg_name == "code":
        return _code_value_similarity(gold_text, pred_text)
    if arg_name in _TYPE_AWARE_EXACT_ARGS:
        return 1.0 if _normalize_value_for_comparison(gold_text) == _normalize_value_for_comparison(pred_text) else 0.0

    return 1.0 if _normalize_value_for_comparison(gold_text) == _normalize_value_for_comparison(pred_text) else 0.0


def _type_aware_value_f1(
    gold_entries: List[Tuple[str, str, Any]],
    pred_entries: List[Tuple[str, str, Any]],
    threshold: float = _TYPE_AWARE_THRESHOLD,
) -> Optional[float]:
    """
    Compute a discrete F1 after type-aware one-to-one alignment of argument values.

    Each entry is a (tool_name, arg_name, value) triple. Tool name and arg name
    must agree; value similarity is then computed with type-specific rules.
    """
    if not gold_entries:
        return None
    if not pred_entries:
        return 0.0

    sim_matrix: List[List[float]] = []
    for gold_tool, gold_arg, gold_value in gold_entries:
        row = []
        for pred_tool, pred_arg, pred_value in pred_entries:
            if gold_tool != pred_tool or gold_arg != pred_arg:
                row.append(0.0)
                continue
            row.append(_type_aware_value_similarity(gold_tool, gold_arg, gold_value, pred_value))
        sim_matrix.append(row)

    matches = _solve_max_weight_matching(sim_matrix)
    matched = sum(1 for i, j in matches if sim_matrix[i][j] >= threshold)
    precision = matched / len(pred_entries) if pred_entries else 0.0
    recall = matched / len(gold_entries) if gold_entries else 0.0
    return _f1_from_prec_recall(precision, recall)


def _tokenize(s: str) -> List[str]:
    """Tokenize a string into words."""
    s = _normalize_string(s)
    if not s:
        return []
    return s.split()


def _semantic_tokens(s: str) -> List[str]:
    """Tokenize a node label while dropping generic planning stopwords."""
    return [tok for tok in _tokenize(s) if tok not in _NODE_MATCH_STOPWORDS]


def _token_f1_similarity(a: str, b: str) -> float:
    """Token-level F1 similarity for semantic node matching."""
    ta = _semantic_tokens(a)
    tb = _semantic_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    precision = overlap / sum(cb.values())
    recall = overlap / sum(ca.values())
    return _f1_from_prec_recall(precision, recall)


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard for paraphrase-tolerant label matching."""
    a = _normalize_string(a)
    b = _normalize_string(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def grams(text: str) -> Set[str]:
        text = f" {text} "
        if len(text) < n:
            return {text}
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    ga = grams(a)
    gb = grams(b)
    return len(ga & gb) / len(ga | gb) if (ga or gb) else 0.0


def _infer_step_type(node: Dict[str, Any]) -> str:
    """Infer a coarse step type for node-level semantic alignment."""
    explicit = node.get("step_type")
    if explicit:
        return str(explicit)
    return "tool" if node.get("tool_id") else "thought"


def _embedding_node_similarity(gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> Optional[float]:
    """Sentence-transformer similarity between node labels."""
    gold_label = gold_node.get("label", "") or ""
    pred_label = pred_node.get("label", "") or ""
    gold_norm = " ".join(str(gold_label).split())
    pred_norm = " ".join(str(pred_label).split())

    if gold_norm == pred_norm:
        return 1.0
    if not gold_norm or not pred_norm:
        return 0.0

    embeddings = _compute_embeddings([gold_norm, pred_norm])
    if embeddings is None or len(embeddings) != 2:
        return None
    sim = _cosine_similarity(embeddings[0], embeddings[1])
    return max(0.0, min(1.0, sim))


def _token_node_similarity(gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> float:
    """
    Compute a lightweight semantic similarity between two nodes.

    The score is content-token-F1 dominant:
    - exact normalized-label matches are treated as perfect
    - otherwise, content-token F1 is the primary signal
    - character n-grams and structural compatibility only act as tiny tie-breakers

    This avoids the previous failure mode where shared tool identity could
    artificially inflate semantically different substeps into high-confidence
    matches.
    """
    gold_label = gold_node.get("label", "") or ""
    pred_label = pred_node.get("label", "") or ""
    gold_norm = _normalize_string(gold_label)
    pred_norm = _normalize_string(pred_label)

    if gold_norm == pred_norm:
        return 1.0

    token_f1 = _token_f1_similarity(gold_label, pred_label)
    if token_f1 <= 0.0:
        return 0.0

    base = token_f1
    base += _NODE_MATCH_CHAR_TIEBREAKER_SCALE * _char_ngram_jaccard(gold_label, pred_label)

    if _infer_step_type(gold_node) == _infer_step_type(pred_node):
        base += _NODE_MATCH_STEP_TYPE_BONUS

    pred_tool_id = pred_node.get("tool_id", "") or ""
    pred_tool_norm = _normalize_tool_id(pred_tool_id) if pred_tool_id else ""
    pred_tool_name = _extract_tool_name(pred_tool_id) if pred_tool_id else ""

    gold_tool_ids = _get_acceptable_tool_ids(gold_node)
    gold_tool_names = _get_acceptable_tool_names(gold_node)
    if pred_tool_norm and pred_tool_norm in gold_tool_ids:
        base += _NODE_MATCH_TOOL_ID_BONUS
    elif pred_tool_name and pred_tool_name in gold_tool_names:
        base += _NODE_MATCH_TOOL_NAME_BONUS

    return min(max(base, 0.0), 1.0)


def _semantic_node_similarity(gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> float:
    """
    Main node similarity used by SpanNodeF1 and DW-OrderF1 anchoring.

    By default this is a sentence-transformer cosine similarity over node labels,
    so compressed macro-plans and paraphrased intent nodes can match gold spans.
    A deterministic token/tool matcher is retained as an explicit fallback and
    for reproducibility experiments via LLM_PLANNING_NODE_MATCH_BACKEND=token.
    """
    if _NODE_MATCH_BACKEND in {"embedding", "sentence-transformer", "sentence-transformers", "auto"}:
        sim = _embedding_node_similarity(gold_node, pred_node)
        if sim is not None:
            return sim
        if _NODE_MATCH_BACKEND != "auto":
            global _EMBEDDING_FALLBACK_WARNED
            if not _EMBEDDING_FALLBACK_WARNED:
                _EMBEDDING_FALLBACK_WARNED = True
                print("[WARN] Embedding node similarity unavailable; falling back to token matcher.")

    return _token_node_similarity(gold_node, pred_node)


def _uses_embedding_node_backend() -> bool:
    return _NODE_MATCH_BACKEND in {"embedding", "sentence-transformer", "sentence-transformers", "auto"}


def _solve_max_weight_matching(sim_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Solve one-to-one maximum-weight matching with a SciPy or greedy fallback."""
    if not sim_matrix:
        return []

    num_rows = len(sim_matrix)
    num_cols = len(sim_matrix[0]) if sim_matrix else 0
    if num_rows == 0 or num_cols == 0:
        return []

    if HAS_SCIPY and HAS_NUMPY:
        matrix = np.array(sim_matrix, dtype=float)
        rows, cols = linear_sum_assignment(-matrix)
        return list(zip(rows.tolist(), cols.tolist()))

    candidates: List[Tuple[float, int, int]] = []
    for i, row in enumerate(sim_matrix):
        for j, value in enumerate(row):
            candidates.append((float(value), i, j))
    candidates.sort(reverse=True)

    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    matches: List[Tuple[int, int]] = []
    for value, i, j in candidates:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        matches.append((i, j))
        if len(used_rows) == num_rows or len(used_cols) == num_cols:
            break
    return matches


def _f1_from_prec_recall(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _set_f1(
    gold_set: Set,
    pred_set: Set,
    empty_is_perfect: bool = True
) -> Tuple[float, float, float]:
    """
    Compute set-based F1 score with configurable edge case handling.

    Args:
        gold_set: Ground truth set
        pred_set: Prediction set
        empty_is_perfect: If True, both-empty returns (1,1,1). If False, returns (0,0,0).

    Returns:
        Tuple of (precision, recall, f1)
    """
    # Both empty
    if not gold_set and not pred_set:
        if empty_is_perfect:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Gold empty, pred not empty → pred has extra items
    if not gold_set:
        return 0.0, 1.0, 0.0

    # Gold not empty, pred empty → pred missed everything
    if not pred_set:
        return 1.0, 0.0, 0.0

    inter = len(gold_set & pred_set)
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    f1 = _f1_from_prec_recall(precision, recall)
    return precision, recall, f1


def _gold_slot_tolerant_item_f1(
    gold_items: Sequence[Any],
    pred_items: Sequence[Any],
    empty_is_perfect: bool = True,
) -> Tuple[float, float, float, int]:
    """
    Compute F1 over repeated gold slots while tolerating repeated valid pred items.

    ReAct-style agents may call the same gold-relevant tool/parameter multiple
    times while gathering evidence. Those repeated in-vocabulary calls should not
    lower precision for tool cleanliness. Missing repeated gold slots still lower
    recall, and genuinely out-of-gold items still lower precision.
    """
    if not gold_items and not pred_items:
        if empty_is_perfect:
            return 1.0, 1.0, 1.0, 0
        return 0.0, 0.0, 0.0, 0
    if not gold_items:
        return 0.0, 1.0, 0.0, 0
    if not pred_items:
        return 1.0, 0.0, 0.0, 0

    gold_counter = Counter(gold_items)
    pred_counter = Counter(pred_items)
    matched = sum((gold_counter & pred_counter).values())
    extra_pred = sum(
        count
        for item, count in pred_counter.items()
        if item not in gold_counter
    )
    precision_denom = matched + extra_pred
    precision = matched / precision_denom if precision_denom > 0 else 0.0
    recall = matched / len(gold_items)
    f1 = _f1_from_prec_recall(precision, recall)
    return precision, recall, f1, matched


def _semantic_node_match_scores(
    gold_nodes: List[Dict[str, Any]],
    pred_nodes: List[Dict[str, Any]],
    threshold: float = _NODE_MATCH_THRESHOLD,
) -> Tuple[float, float, float]:
    """
    Compute discrete node P/R/F1 after semantic one-to-one alignment.

    This differs from soft semantic alignment diagnostics: we first find the
    best one-to-one node correspondences, then count a pair as a true positive
    only when the semantic similarity exceeds the threshold.
    """
    if not gold_nodes and not pred_nodes:
        return 1.0, 1.0, 1.0
    if not gold_nodes:
        return 0.0, 1.0, 0.0
    if not pred_nodes:
        return 1.0, 0.0, 0.0

    matched_pairs, _, _ = _semantic_node_alignment(
        gold_nodes=gold_nodes,
        pred_nodes=pred_nodes,
        threshold=threshold,
    )
    matched = len(matched_pairs)

    precision = matched / len(pred_nodes) if pred_nodes else 0.0
    recall = matched / len(gold_nodes) if gold_nodes else 0.0
    f1 = _f1_from_prec_recall(precision, recall)
    return precision, recall, f1


def _semantic_node_alignment(
    gold_nodes: List[Dict[str, Any]],
    pred_nodes: List[Dict[str, Any]],
    threshold: float = _NODE_MATCH_THRESHOLD,
) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    Build semantic one-to-one node alignment under the same threshold used by Node F1.

    Returns:
        - matched_pairs: list of (gold_idx, pred_idx) above threshold
        - gold_to_pred: mapping gold_idx -> pred_idx
        - pred_to_gold: mapping pred_idx -> gold_idx
    """
    if not gold_nodes or not pred_nodes:
        return [], {}, {}

    sim_matrix = [
        [_semantic_node_similarity(gold_node, pred_node) for pred_node in pred_nodes]
        for gold_node in gold_nodes
    ]
    matches = _solve_max_weight_matching(sim_matrix)
    matched_pairs = [(i, j) for i, j in matches if sim_matrix[i][j] >= threshold]
    gold_to_pred = {i: j for i, j in matched_pairs}
    pred_to_gold = {j: i for i, j in matched_pairs}
    return matched_pairs, gold_to_pred, pred_to_gold


def _sorted_gold_nodes_by_step_index(
    gold_nodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort gold nodes by step_index, falling back to original order."""
    indexed = list(enumerate(gold_nodes))
    indexed.sort(key=lambda item: (item[1].get("step_index", item[0]), item[0]))
    return [node for _, node in indexed]


def _build_undirected_adjacency_by_index(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> Dict[int, Set[int]]:
    """Build undirected adjacency over local node indices for connected-span checks."""
    id_to_idx = {n.get("node_id", ""): i for i, n in enumerate(nodes)}
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(nodes))}
    for e in edges or []:
        source = e.get("source")
        target = e.get("target")
        sources = source if isinstance(source, list) else ([source] if source else [])
        targets = target if isinstance(target, list) else ([target] if target else [])
        for s in sources:
            si = id_to_idx.get(s)
            if si is None:
                continue
            for t in targets:
                ti = id_to_idx.get(t)
                if ti is None or ti == si:
                    continue
                adj[si].add(ti)
                adj[ti].add(si)
    return adj


def _is_connected_dependency_span(
    indices: Sequence[int],
    adj: Dict[int, Set[int]],
) -> bool:
    """Check whether the induced gold-node span is connected after ignoring edge direction."""
    if len(indices) <= 1:
        return True
    node_set = set(indices)
    stack = [indices[0]]
    seen = {indices[0]}
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            if nxt in node_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(node_set)


def _make_span_pseudo_node(span_nodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse a short gold span into a pseudo-node for semantic matching."""
    labels = [
        str(n.get("label", "") or "").strip()
        for n in span_nodes
        if str(n.get("label", "") or "").strip()
    ]
    tool_ids: List[str] = []
    for n in span_nodes:
        tool_ids.extend(sorted(_get_acceptable_tool_ids(n)))
    tool_ids = list(dict.fromkeys(tool_ids))

    step_types = {_infer_step_type(n) for n in span_nodes}
    pseudo: Dict[str, Any] = {
        "label": " ; ".join(labels),
    }
    if len(step_types) == 1:
        pseudo["step_type"] = next(iter(step_types))
    if tool_ids:
        pseudo["tool_id"] = tool_ids[0]
        if len(tool_ids) > 1:
            pseudo["alternative_tools"] = tool_ids[1:]
    return pseudo


def _member_supported_count(
    span_nodes: Sequence[Dict[str, Any]],
    pred_node: Dict[str, Any],
    threshold: float,
) -> int:
    """Count how many member nodes are individually supported by the prediction."""
    return sum(1 for g in span_nodes if _semantic_node_similarity(g, pred_node) >= threshold)


def _span_node_match_scores(
    gold_nodes: Sequence[Dict[str, Any]],
    gold_edges: Sequence[Dict[str, Any]],
    pred_nodes: Sequence[Dict[str, Any]],
    threshold: float = _NODE_MATCH_THRESHOLD,
    max_span_size: int = _SPAN_NODE_MAX_SIZE,
) -> Tuple[float, float, float]:
    """
    Compute Span-Node precision / recall / F1.

    One predicted node may cover a short contiguous gold span if that induced
    span is connected in the dependency graph and semantically aligned. Precision counts matched
    predicted nodes; recall counts covered gold nodes.
    """
    precision, recall, f1, _ = _span_node_alignment(
        gold_nodes=gold_nodes,
        gold_edges=gold_edges,
        pred_nodes=pred_nodes,
        threshold=threshold,
        max_span_size=max_span_size,
    )
    return precision, recall, f1


def _span_node_alignment(
    gold_nodes: Sequence[Dict[str, Any]],
    gold_edges: Sequence[Dict[str, Any]],
    pred_nodes: Sequence[Dict[str, Any]],
    threshold: float = _NODE_MATCH_THRESHOLD,
    max_span_size: int = _SPAN_NODE_MAX_SIZE,
) -> Tuple[float, float, float, Dict[int, Tuple[int, ...]]]:
    """
    Align predicted nodes to non-overlapping local gold spans.

    Returns Span-Node precision / recall / F1 plus a mapping from predicted
    node index to the matched gold-span indices in step-index order. The mapping
    is reused by span-reanchored ordering metrics so compressed predictions are
    not evaluated against raw node positions.
    """
    if not gold_nodes and not pred_nodes:
        return 1.0, 1.0, 1.0, {}
    if not gold_nodes:
        return 0.0, 1.0, 0.0, {}
    if not pred_nodes:
        return 1.0, 0.0, 0.0, {}

    sorted_gold = _sorted_gold_nodes_by_step_index(gold_nodes)
    adj = _build_undirected_adjacency_by_index(sorted_gold, gold_edges or [])

    candidates_by_pred: Dict[int, List[Tuple[float, float, Tuple[int, ...]]]] = {}
    for pred_idx, pred_node in enumerate(pred_nodes):
        pred_candidates: List[Tuple[float, float, Tuple[int, ...]]] = []
        for start in range(len(sorted_gold)):
            for span_len in range(1, max_span_size + 1):
                end = start + span_len
                if end > len(sorted_gold):
                    break
                gold_indices = tuple(range(start, end))
                if span_len > 1 and not _is_connected_dependency_span(gold_indices, adj):
                    continue
                span_nodes = [sorted_gold[i] for i in gold_indices]
                pseudo = _make_span_pseudo_node(span_nodes)
                sim = _semantic_node_similarity(pseudo, pred_node)
                if sim < threshold:
                    continue
                if span_len > 1 and _member_supported_count(span_nodes, pred_node, threshold) < 2:
                    continue
                pred_candidates.append((sim * span_len, sim, gold_indices))
        if pred_candidates:
            pred_candidates.sort(reverse=True)
            candidates_by_pred[pred_idx] = pred_candidates

    def _greedy_alignment() -> Tuple[float, int, int, Tuple[Tuple[int, Tuple[int, ...]], ...]]:
        all_candidates: List[Tuple[float, float, int, Tuple[int, ...]]] = []
        for pred_idx, pred_candidates in candidates_by_pred.items():
            for objective, sim, gold_indices in pred_candidates:
                all_candidates.append((objective, sim, pred_idx, gold_indices))
        all_candidates.sort(key=lambda item: (item[0], item[1], len(item[3])), reverse=True)

        used_pred: Set[int] = set()
        used_gold: Set[int] = set()
        mapping: List[Tuple[int, Tuple[int, ...]]] = []
        score = 0.0
        for objective, _, pred_idx, gold_indices in all_candidates:
            if pred_idx in used_pred:
                continue
            if any(gold_idx in used_gold for gold_idx in gold_indices):
                continue
            used_pred.add(pred_idx)
            used_gold.update(gold_indices)
            mapping.append((pred_idx, gold_indices))
            score += objective
        return score, len(used_pred), len(used_gold), tuple(mapping)

    total_candidate_count = sum(len(cands) for cands in candidates_by_pred.values())
    if total_candidate_count > _SPAN_ALIGNMENT_EXACT_CANDIDATE_LIMIT:
        _, matched_pred_count, covered_gold_count, mapping = _greedy_alignment()
        precision = matched_pred_count / len(pred_nodes) if pred_nodes else 0.0
        recall = covered_gold_count / len(sorted_gold) if sorted_gold else 0.0
        f1 = _f1_from_prec_recall(precision, recall)
        pred_to_gold_span = {pred_idx: tuple(gold_indices) for pred_idx, gold_indices in mapping}
        return precision, recall, f1, pred_to_gold_span

    pred_order = sorted(
        candidates_by_pred.keys(),
        key=lambda idx: candidates_by_pred[idx][0][0],
        reverse=True,
    )

    def _alignment_f1(matched_pred_count: int, covered_gold_count: int) -> float:
        precision = matched_pred_count / len(pred_nodes) if pred_nodes else 0.0
        recall = covered_gold_count / len(sorted_gold) if sorted_gold else 0.0
        return _f1_from_prec_recall(precision, recall)

    def _is_better_alignment(
        cand: Tuple[float, int, int, Tuple[Tuple[int, Tuple[int, ...]], ...]],
        best: Tuple[float, int, int, Tuple[Tuple[int, Tuple[int, ...]], ...]],
    ) -> bool:
        cand_f1 = _alignment_f1(cand[1], cand[2])
        best_f1 = _alignment_f1(best[1], best[2])
        if cand_f1 > best_f1 + 1e-12:
            return True
        if best_f1 > cand_f1 + 1e-12:
            return False
        # Prefer the alignment with stronger semantic evidence only after the
        # final SpanNodeF1 objective is tied.
        if cand[0] > best[0] + 1e-12:
            return True
        if abs(cand[0] - best[0]) <= 1e-12:
            # Prefer covering more gold steps, then matching more predicted nodes.
            if cand[2] != best[2]:
                return cand[2] > best[2]
            if cand[1] != best[1]:
                return cand[1] > best[1]
        return False

    @lru_cache(maxsize=None)
    def _search(
        pos: int,
        used_gold_mask: int,
    ) -> Tuple[float, int, int, Tuple[Tuple[int, Tuple[int, ...]], ...]]:
        if pos >= len(pred_order):
            return (0.0, 0, 0, tuple())

        pred_idx = pred_order[pos]
        best = _search(pos + 1, used_gold_mask)
        for objective, _, gold_indices in candidates_by_pred.get(pred_idx, []):
            cand_mask = 0
            for gi in gold_indices:
                cand_mask |= 1 << gi
            if cand_mask & used_gold_mask:
                continue
            tail_score, tail_matched_preds, tail_covered_gold, tail_mapping = _search(
                pos + 1,
                used_gold_mask | cand_mask,
            )
            score = objective + tail_score
            matched_preds = 1 + tail_matched_preds
            covered_gold = len(gold_indices) + tail_covered_gold
            mapping = ((pred_idx, gold_indices),) + tail_mapping
            cand = (score, matched_preds, covered_gold, mapping)
            if _is_better_alignment(cand, best):
                best = cand
        return best

    _, matched_pred_count, covered_gold_count, mapping = _search(0, 0)
    precision = matched_pred_count / len(pred_nodes) if pred_nodes else 0.0
    recall = covered_gold_count / len(sorted_gold) if sorted_gold else 0.0
    f1 = _f1_from_prec_recall(precision, recall)
    pred_to_gold_span = {pred_idx: tuple(gold_indices) for pred_idx, gold_indices in mapping}
    return precision, recall, f1, pred_to_gold_span


def _string_em_f1(gold: str, pred: str) -> Tuple[float, float]:
    """Exact match + token-level F1 for strings."""
    g_norm = _normalize_string(gold)
    p_norm = _normalize_string(pred)

    # Handle empty strings consistently: both empty = no valid match
    if not g_norm and not p_norm:
        return 0.0, 0.0

    em = 1.0 if g_norm == p_norm else 0.0

    g_tokens = set(_tokenize(gold))
    p_tokens = set(_tokenize(pred))
    _, _, f1 = _set_f1(g_tokens, p_tokens, empty_is_perfect=False)
    return em, f1


def _simple_string_similarity(s1: str, s2: str) -> float:
    """Simple string similarity based on token overlap (Jaccard)."""
    t1 = set(_tokenize(s1))
    t2 = set(_tokenize(s2))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _normalize_tool_id(tool_id: str) -> str:
    """
    Normalize tool_id for matching.

    Handles:
    - "server_name::tool_name" -> lowercase
    - Extra whitespace removal
    - Case normalization
    - Space/underscore/dash normalization for cross-format matching

    Examples:
        "MCP Crypto Wallet EVM::wallet_send_transaction"
        -> "mcp_crypto_wallet_evm::wallet_send_transaction"

        "mcp-crypto-wallet-evm::wallet-send-transaction"
        -> "mcp_crypto_wallet_evm::wallet_send_transaction"
    """
    if not tool_id:
        return ""

    s = tool_id.strip().lower()
    # Normalize spaces and dashes to underscores for consistent matching
    s = re.sub(r"[\s\-]+", "_", s)
    # Remove consecutive underscores
    s = re.sub(r"_+", "_", s)
    return s


def _extract_tool_name(tool_id: str) -> str:
    """
    Extract tool_name from tool_id for TaskBench-style comparison.

    Handles formats:
    - "server_name::tool_name" -> "tool_name" (normalized)
    - "tool_name" -> "tool_name" (normalized)
    """
    if not tool_id:
        return ""

    # First normalize the tool_id
    normalized = _normalize_tool_id(tool_id)

    if "::" in normalized:
        parts = normalized.split("::", 1)
        return parts[1] if len(parts) > 1 else normalized
    return normalized


def _get_acceptable_tool_ids(node: Dict[str, Any]) -> Set[str]:
    """
    Get all acceptable tool IDs for a node, including alternatives.
    Returns normalized (lowercased) tool IDs.
    """
    tool_ids = set()
    primary_id = node.get("tool_id")
    if primary_id:
        tool_ids.add(_normalize_tool_id(primary_id))

    alternatives = node.get("alternative_tools", []) or []
    for alt_id in alternatives:
        if alt_id:
            tool_ids.add(_normalize_tool_id(alt_id))

    return tool_ids


def _get_acceptable_tool_names(node: Dict[str, Any]) -> Set[str]:
    """
    Get all acceptable tool NAMES for a node (extracted from tool_ids).

    This extracts only the tool name portion after '::' for flexible matching.
    Handles cases like:
    - "search_files::search_files" -> "search_files"
    - "server::search_files" -> "search_files"
    - "mcp_tools::read_file" -> "read_file"

    This allows matching between different server naming conventions.
    """
    tool_names = set()
    primary_id = node.get("tool_id")
    if primary_id:
        tool_names.add(_extract_tool_name(primary_id))

    alternatives = node.get("alternative_tools", []) or []
    for alt_id in alternatives:
        if alt_id:
            tool_names.add(_extract_tool_name(alt_id))

    return tool_names


# ============================================================================
# Score Dataclasses
# ============================================================================

@dataclass
class PlanScores:
    """Scores for plan DAG evaluation."""
    node_f1: float  # Main node-side metric: strict semantic one-to-one NodeF1
    edge_f1: float  # Paper-facing EdgeF1: semantic direct-edge overlap after node matching
    node_label_similarity: float
    ssi: float  # Structural Similarity Index = (node_label_sim + paper-facing edge_f1) / 2
    semantic_edge_f1: float  # Backward-compatible alias for edge_f1
    dw_order_f1: float  # Supplementary order metric; strict-node-reanchored and distance-weighted
    planning_score: float
    raw_edge_f1: float = 0.0  # Raw (source_step_index, target_step_index) overlap diagnostic
    span_node_f1: float = 0.0
    strict_node_f1: float = 0.0
    strict_reanchored_dw_order_f1: float = 0.0
    span_reanchored_dw_order_f1: float = 0.0
    index_dw_order_f1: float = 0.0

    # Detailed metrics
    node_precision: float = 0.0
    node_recall: float = 0.0
    strict_node_precision: float = 0.0
    strict_node_recall: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    raw_edge_precision: float = 0.0
    raw_edge_recall: float = 0.0
    semantic_edge_precision: float = 0.0
    semantic_edge_recall: float = 0.0
    dw_order_precision: float = 0.0
    dw_order_recall: float = 0.0
    strict_reanchored_dw_order_precision: float = 0.0
    strict_reanchored_dw_order_recall: float = 0.0
    index_dw_order_precision: float = 0.0
    index_dw_order_recall: float = 0.0
    order_precision: float = 0.0
    order_recall: float = 0.0
    order_f1: float = 0.0
    gold_node_count: int = 0
    pred_node_count: int = 0


@dataclass
class ToolScores:
    """
    Scores for tool usage evaluation.

    Metrics aligned with TaskBench evaluate.py:
    - tool_name_f1: Tool name F1 (primary metric)
    - param_name_f1: t-F1 = F1 of {tool_name}-{param_name} pairs
    - type_aware_value_f1: type-aware value F1 that adapts to heterogeneous argument semantics

    For datasets without arguments (Delta), the parameter/value metrics will be
    None to indicate "not applicable".
    """
    tool_name_f1: float
    param_name_f1: Optional[float]  # None means N/A (no arguments in dataset)
    type_aware_value_f1: Optional[float]  # Main value metric used by ToolUsageScore
    strict_type_aware_value_f1: Optional[float] = None
    normalized_type_aware_value_f1: Optional[float] = None

    # Detailed metrics
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    gold_tool_count: int = 0
    pred_tool_count: int = 0

    # Argument counts (for debugging)
    gold_param_count: int = 0
    pred_param_count: int = 0
    matched_param_count: int = 0
    normalized_value_gold_count: int = 0
    normalized_value_pred_count: int = 0
    normalized_value_matched_count: int = 0

    # Flag indicating whether this record has arguments
    has_arguments: bool = False


@dataclass
class AnswerScores:
    """Scores for answer evaluation."""
    has_answer: bool
    exact_match: Optional[float] = None
    token_f1: Optional[float] = None
    numeric_rel_error: Optional[float] = None
    alias_match: Optional[float] = None
    llm_judge_score: Optional[float] = None


# ============================================================================
# Main Evaluation Class
# ============================================================================

class ASTEvaluationSystem:
    """
    Evaluate a single (gold, pred) pair in unified schema.

    Key features:
    1. Dataset-aware metrics handling
    2. Proper empty-set handling for arguments
    3. Recursive argument extraction for nested formats
    4. Embedding-based node label similarity
    5. Alternative tools support for Delta dataset
    """

    def __init__(self, use_embeddings: bool = True, verifier_model: Optional[str] = None):
        """Initialize evaluator."""
        self.use_embeddings = use_embeddings
        self.verifier_model = verifier_model
        self._verifier_client = None
        self._verifier_available: Optional[bool] = None
        if use_embeddings:
            _get_embedding_model()

    def _ensure_verifier_client(self) -> bool:
        """Initialize an OpenAI-compatible answer verifier if available."""
        if self._verifier_available is not None:
            return self._verifier_available

        if not self.verifier_model:
            self._verifier_available = False
            return False

        api_base = get_verifier_api_base()
        api_key = get_verifier_api_key()
        if not api_base or not api_key:
            self._verifier_available = False
            return False

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                default_headers={"User-Agent": "curl/7.68.0"},
            )
            client.chat.completions.create(
                model=self.verifier_model,
                messages=[{"role": "user", "content": "Reply with 1"}],
                max_tokens=1,
                temperature=0,
            )
            self._verifier_client = client
            self._verifier_available = True
            print(f"[INFO] Answer verifier enabled: {self.verifier_model}")
            return True
        except Exception as e:
            self._verifier_available = False
            print(f"[WARN] Answer verifier unavailable for {self.verifier_model}: {e}")
            return False

    def _judge_answer_with_llm(
        self,
        query_text: str,
        gold_ans: Dict[str, Any],
        pred_value: Any,
    ) -> Optional[float]:
        """Use an LLM judge to decide whether a final answer should count as correct."""
        if not self._ensure_verifier_client():
            return None

        if pred_value is None:
            return 0.0

        gold_value = gold_ans.get("answer")
        aliases = gold_ans.get("aliases", []) or []
        tolerance = gold_ans.get("tolerance", 0.0)
        ans_type = gold_ans.get("answer_type", "none")

        prompt = (
            f"Question: {query_text}\n"
            f"Gold answer: {gold_value}\n"
            f"Aliases: {aliases}\n"
            f"Tolerance: {tolerance}\n"
            f"Answer type: {ans_type}\n"
            f"Predicted answer: {pred_value}\n"
        )

        is_gpt_oss_verifier = "gpt-oss" in _normalize_string(self.verifier_model or "")
        max_tokens = 128 if is_gpt_oss_verifier else 16

        try:
            response = self._verifier_client.chat.completions.create(
                model=self.verifier_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict answer verifier.\n"
                            "Decide whether the predicted answer should count as correct.\n"
                            "Consider semantic equivalence, aliases, and numeric tolerance.\n"
                            "Reply with exactly one line: FINAL_VERDICT=1 or FINAL_VERDICT=0"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            message = response.choices[0].message
            content = message.content or getattr(message, "reasoning_content", "") or ""
            match = re.search(r"FINAL_VERDICT\s*=\s*([01])", str(content))
            if not match:
                match = re.search(r"\b([01])\b", str(content).strip())
            if match:
                return float(match.group(1))
        except Exception as e:
            print(f"[WARN] LLM judge failed on one sample: {e}")

        return None

    def _is_anonymous_arg_name(self, name: str) -> bool:
        """Return True when a TaskBench slot is just an arg0/arg1 placeholder."""
        return bool(re.fullmatch(r"arg\d+", _normalize_string(name)))

    def _normalize_taskbench_type_name(self, type_name: str) -> str:
        """Normalize TaskBench content/resource type names."""
        if not type_name:
            return ""
        return _normalize_tool_id(type_name)

    def _infer_taskbench_literal_type(self, value: str) -> str:
        """Infer a canonical TaskBench resource type from a literal value."""
        if not value:
            return "text"

        value_norm = _normalize_string(value)
        if value_norm.startswith(("http://", "https://", "www.")):
            return "url"

        content_type = self._infer_content_type(value_norm)
        if content_type == "node_reference":
            return "text"
        return content_type

    def _build_tool_catalog(self, available_tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Index tool metadata by raw tool_id and normalized tool name."""
        catalog: Dict[str, Dict[str, Any]] = {}
        for tool in available_tools or []:
            tool_id = tool.get("tool_id", "") or ""
            if not tool_id:
                continue
            catalog[_normalize_tool_id(tool_id)] = tool
            catalog[_extract_tool_name(tool_id)] = tool
        return catalog

    def _resolve_taskbench_reference(
        self,
        value: str,
        call_index_to_call: Dict[int, Dict[str, Any]],
        node_id_to_call: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve TaskBench references like <node-0> or <n0> back to a call."""
        if not isinstance(value, str):
            return None

        raw = value.strip()
        raw_lower = raw.lower()

        match = re.fullmatch(r"<?node-(\d+)>?", raw_lower)
        if match:
            return call_index_to_call.get(int(match.group(1)))

        match = re.fullmatch(r"<?n(\d+)>?", raw_lower)
        if match:
            return node_id_to_call.get(f"n{match.group(1)}")

        if raw in node_id_to_call:
            return node_id_to_call.get(raw)

        if raw.startswith("<") and raw.endswith(">"):
            inner = raw[1:-1]
            if inner in node_id_to_call:
                return node_id_to_call.get(inner)

        return None

    def _normalize_taskbench_call_arguments(
        self,
        call: Dict[str, Any],
        all_calls: List[Dict[str, Any]],
        tool_catalog: Dict[str, Dict[str, Any]],
        force_type_based: bool,
    ) -> List[Tuple[str, str]]:
        """Normalize TaskBench arguments to resource-aware names and comparable values."""
        extracted_args = self._extract_arguments(call.get("arguments", []))
        if not extracted_args:
            return []

        call_index_to_call: Dict[int, Dict[str, Any]] = {}
        node_id_to_call: Dict[str, Dict[str, Any]] = {}
        for idx, candidate in enumerate(all_calls):
            call_index = candidate.get("call_index", idx)
            call_index_to_call[call_index] = candidate
            node_id = candidate.get("node_id")
            if node_id:
                node_id_to_call[str(node_id)] = candidate

        current_tool_id = call.get("tool_id", "") or ""
        current_tool_meta = (
            tool_catalog.get(_normalize_tool_id(current_tool_id))
            or tool_catalog.get(_extract_tool_name(current_tool_id))
            or {}
        )
        input_types = current_tool_meta.get("input_type", []) or []

        normalized: List[Tuple[str, str]] = []
        for arg_idx, (name, value) in enumerate(extracted_args):
            input_type = self._normalize_taskbench_type_name(input_types[arg_idx]) if arg_idx < len(input_types) else ""
            referenced_call = self._resolve_taskbench_reference(value, call_index_to_call, node_id_to_call)

            if referenced_call is not None:
                ref_tool_id = referenced_call.get("tool_id", "") or ""
                ref_tool_meta = (
                    tool_catalog.get(_normalize_tool_id(ref_tool_id))
                    or tool_catalog.get(_extract_tool_name(ref_tool_id))
                    or {}
                )
                output_types = ref_tool_meta.get("output_type", []) or []
                canonical_name = self._normalize_taskbench_type_name(output_types[0]) if output_types else (input_type or "other")
                canonical_value = _extract_tool_name(ref_tool_id)
            elif force_type_based or self._is_anonymous_arg_name(name) or not name:
                canonical_name = input_type or self._infer_taskbench_literal_type(value)
                canonical_value = _normalize_value_for_comparison(str(value))
            else:
                canonical_name = _normalize_tool_id(str(name))
                canonical_value = _normalize_value_for_comparison(str(value))

            if canonical_name:
                normalized.append((canonical_name, canonical_value))

        return normalized

    def _extract_arguments(self, args: Any) -> List[Tuple[str, str]]:
        """
        Extract (name, value) pairs from arguments, handling various formats.

        Handles:
        - List of dicts: [{"name": "x", "value": "y"}, ...] (TaskBench dailylifeapis)
        - Dict: {"x": "y", ...} (alternative format)
        - Nested formats: {"name": "arg0", "value": {"name": "title", "value": "X"}}
        - Raw string list: ["example.mp4", "example.wav"] (TaskBench multimedia/huggingface)
          - These are transformed to (content_type, value) pairs based on file extension
          - Node references like "<node-0>" are transformed to ("node_reference", "<node-0>")
        """
        results = []

        if isinstance(args, dict):
            for name, value in args.items():
                if isinstance(value, dict) and "name" in value and "value" in value:
                    # Nested format
                    results.append((str(value["name"]), str(value.get("value", ""))))
                else:
                    results.append((str(name), str(value) if value is not None else ""))
        elif isinstance(args, list):
            for arg in args:
                if isinstance(arg, dict):
                    name = arg.get("name", "")
                    value = arg.get("value", "")

                    # Check if value is nested
                    if isinstance(value, dict) and "name" in value and "value" in value:
                        results.append((str(value["name"]), str(value.get("value", ""))))
                    else:
                        if name:  # Only add if name is non-empty
                            results.append((str(name), str(value) if value is not None else ""))
                elif isinstance(arg, str):
                    # Raw string argument (TaskBench multimedia/huggingface style)
                    # Transform to (content_type, value) format following TaskBench evaluate.py
                    content_type = self._infer_content_type(arg)
                    results.append((content_type, arg))

        return results

    def _infer_content_type(self, value: str) -> str:
        """
        Infer content type from a raw string argument.

        Following TaskBench evaluate.py logic:
        - Node references (<node-X>) -> "node_reference"
        - Image extensions -> "image"
        - Audio extensions -> "audio"
        - Video extensions -> "video"
        - Otherwise -> "text"
        """
        if not value:
            return "text"

        value_lower = value.lower()

        # Check for node reference (e.g., "<node-0>", "<node-1>")
        if re.match(r"<node-\d+>", value_lower):
            return "node_reference"

        # Check file extensions
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
        audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg'}

        for ext in image_exts:
            if value_lower.endswith(ext):
                return "image"

        for ext in audio_exts:
            if value_lower.endswith(ext):
                return "audio"

        for ext in video_exts:
            if value_lower.endswith(ext):
                return "video"

        return "text"

    def compute_node_label_similarity(
        self,
        gold_nodes: List[Dict[str, Any]],
        pred_nodes: List[Dict[str, Any]]
    ) -> float:
        """
        Compute node label similarity using embeddings.

        Formula (from paper):
        NodeLabelSim = (1/|N_expected|) * Σ max(CosineSim(L_i, L_j))

        For each gold node i, find the max cosine similarity with any pred node j.
        Sum these max similarities and divide by number of gold nodes.
        """
        # Edge cases
        if not gold_nodes and not pred_nodes:
            return 1.0
        if not gold_nodes:
            return 1.0  # No expected nodes = trivially satisfied
        if not pred_nodes:
            return 0.0  # No actual nodes to match against

        # Extract labels
        gold_labels = [str(n.get("label", "")) for n in gold_nodes]
        pred_labels = [str(n.get("label", "")) for n in pred_nodes]

        # Compute similarities using embeddings or fallback
        if self.use_embeddings and HAS_NUMPY:
            gold_emb = _compute_embeddings(gold_labels)
            pred_emb = _compute_embeddings(pred_labels)

            if gold_emb is not None and pred_emb is not None:
                max_similarities = []
                for i in range(len(gold_nodes)):
                    # For each gold node, find max similarity with any pred node
                    max_sim = 0.0
                    for j in range(len(pred_nodes)):
                        sim = _cosine_similarity(gold_emb[i], pred_emb[j])
                        sim = max(0.0, min(1.0, sim))  # Clamp to [0, 1]
                        if sim > max_sim:
                            max_sim = sim
                    max_similarities.append(max_sim)

                # Average over gold nodes (|N_expected|)
                return sum(max_similarities) / len(gold_nodes)

        # Fallback: string similarity (Jaccard)
        max_similarities = []
        for g_label in gold_labels:
            max_sim = 0.0
            for p_label in pred_labels:
                sim = _simple_string_similarity(g_label, p_label)
                if sim > max_sim:
                    max_sim = sim
            max_similarities.append(max_sim)

        return sum(max_similarities) / len(gold_nodes)

    def evaluate_plan_dag(
        self,
        gold_dag: Dict[str, Any],
        pred_dag: Dict[str, Any],
        dataset: str = ""
    ) -> PlanScores:
        """
        Evaluate plan DAG structure.

        Node F1 calculation:
        - Main evaluator: Span-Node F1 over local, semantically aligned gold spans
        - Diagnostic baseline: stricter semantic one-to-one node alignment

        Edge F1 calculation:
        - Uses (step_index_source, step_index_target) pairs for matching
        """
        g_nodes = gold_dag.get("nodes") or []
        p_nodes = pred_dag.get("nodes") or []
        g_edges = gold_dag.get("edges") or []
        p_edges = pred_dag.get("edges") or []

        # === Node metrics ===
        strict_node_precision, strict_node_recall, strict_node_f1 = _semantic_node_match_scores(g_nodes, p_nodes)
        span_node_precision, span_node_recall, span_node_f1, pred_to_gold_span = _span_node_alignment(
            gold_nodes=g_nodes,
            gold_edges=g_edges,
            pred_nodes=p_nodes,
        )
        _, _, pred_to_gold = _semantic_node_alignment(g_nodes, p_nodes)

        # === Edge utilities ===
        def _edge_pairs(nodes, edges) -> Set[Tuple[int, int]]:
            """Strict edge pairs based on (source_step_index, target_step_index)."""
            id_to_idx = {n.get("node_id", ""): n.get("step_index", i) for i, n in enumerate(nodes)}
            pairs = set()
            for e in edges:
                source = e.get("source")
                target = e.get("target")

                # Handle source as list (multiple sources -> single target)
                if isinstance(source, list):
                    sources = source
                else:
                    sources = [source] if source else []

                # Handle target as list (single source -> multiple targets)
                if isinstance(target, list):
                    targets = target
                else:
                    targets = [target] if target else []

                # Create edge pairs for all combinations
                for src in sources:
                    s = id_to_idx.get(src)
                    for tgt in targets:
                        t = id_to_idx.get(tgt)
                        if s is not None and t is not None:
                            pairs.add((s, t))
            return pairs

        def _edge_pairs_by_node_index(nodes, edges) -> Set[Tuple[int, int]]:
            """Edge pairs using local node-list indices (for semantic re-anchoring)."""
            id_to_local_idx = {n.get("node_id", ""): i for i, n in enumerate(nodes)}
            pairs = set()
            for e in edges:
                source = e.get("source")
                target = e.get("target")

                if isinstance(source, list):
                    sources = source
                else:
                    sources = [source] if source else []

                if isinstance(target, list):
                    targets = target
                else:
                    targets = [target] if target else []

                for src in sources:
                    s = id_to_local_idx.get(src)
                    for tgt in targets:
                        t = id_to_local_idx.get(tgt)
                        if s is not None and t is not None:
                            pairs.add((s, t))
            return pairs

        def _transitive_pairs_with_shortest_dist(
            num_nodes: int,
            edge_pairs: Set[Tuple[int, int]],
        ) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int]]:
            """Build transitive order pairs and their shortest-path lengths."""
            adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
            for s, t in edge_pairs:
                if 0 <= s < num_nodes and 0 <= t < num_nodes and s != t:
                    adjacency[s].add(t)

            closure: Set[Tuple[int, int]] = set()
            shortest: Dict[Tuple[int, int], int] = {}

            for start in range(num_nodes):
                queue: List[Tuple[int, int]] = [(start, 0)]
                seen = {start}
                head = 0
                while head < len(queue):
                    node, dist = queue[head]
                    head += 1
                    for nxt in adjacency.get(node, set()):
                        if nxt in seen:
                            continue
                        seen.add(nxt)
                        queue.append((nxt, dist + 1))
                        if start != nxt:
                            pair = (start, nxt)
                            closure.add(pair)
                            prev = shortest.get(pair)
                            if prev is None or dist + 1 < prev:
                                shortest[pair] = dist + 1
            return closure, shortest

        def _span_reanchored_dw_order_scores() -> Tuple[float, float, float]:
            """
            Distance-weighted order F1 after mapping predicted nodes to gold spans.

            The legacy DW-OrderF1 compared raw numeric step indices. That is too
            strict for compressed plans: if one predicted node covers gold steps
            0-2 and another covers step 3, the predicted relation 0->1 should
            cover the gold relations from the first span to the second, not only
            the literal pair (0, 1). This metric uses the Span-Node alignment as
            the anchor, then evaluates order over the aligned gold spans.
            """
            sorted_gold = _sorted_gold_nodes_by_step_index(g_nodes)
            g_edge_pairs_local = _edge_pairs_by_node_index(sorted_gold, g_edges)
            p_edge_pairs_local = _edge_pairs_by_node_index(p_nodes, p_edges)
            g_order_pairs_local, g_order_dist_local = _transitive_pairs_with_shortest_dist(
                len(sorted_gold),
                g_edge_pairs_local,
            )
            p_order_pairs_local, p_order_dist_local = _transitive_pairs_with_shortest_dist(
                len(p_nodes),
                p_edge_pairs_local,
            )

            if not g_order_pairs_local and not p_order_pairs_local:
                return (1.0, 1.0, 1.0) if sorted_gold else (0.0, 0.0, 0.0)

            covered_gold_pairs: Set[Tuple[int, int]] = set()

            # A compressed predicted node carries the local order inside the
            # matched gold span. This is what prevents macro-step predictions
            # from being punished only because they did not spell out each
            # internal gold transition as a separate predicted node.
            for gold_span in pred_to_gold_span.values():
                span_set = set(gold_span)
                for pair in g_order_pairs_local:
                    if pair[0] in span_set and pair[1] in span_set:
                        covered_gold_pairs.add(pair)

            pred_weight_denom = sum(
                1.0 / max(p_order_dist_local[pair], 1)
                for pair in p_order_pairs_local
            )
            pred_weight_num = 0.0

            for pair in p_order_pairs_local:
                src_span = pred_to_gold_span.get(pair[0])
                tgt_span = pred_to_gold_span.get(pair[1])
                if not src_span or not tgt_span:
                    continue

                cross_pairs = {
                    (src_gold, tgt_gold)
                    for src_gold in src_span
                    for tgt_gold in tgt_span
                    if src_gold != tgt_gold
                }
                if not cross_pairs:
                    continue

                compatible_pairs = cross_pairs & g_order_pairs_local
                if compatible_pairs:
                    weight = 1.0 / max(p_order_dist_local[pair], 1)
                    pred_weight_num += weight * (len(compatible_pairs) / len(cross_pairs))
                    covered_gold_pairs.update(compatible_pairs)

            gold_weight_denom = sum(
                1.0 / max(g_order_dist_local[pair], 1)
                for pair in g_order_pairs_local
            )
            gold_weight_num = sum(
                1.0 / max(g_order_dist_local[pair], 1)
                for pair in covered_gold_pairs
                if pair in g_order_dist_local
            )

            span_dw_precision = (
                pred_weight_num / pred_weight_denom
                if pred_weight_denom > 0 else 1.0
            )
            span_dw_recall = (
                gold_weight_num / gold_weight_denom
                if gold_weight_denom > 0 else (1.0 if not p_order_pairs_local else 0.0)
            )
            span_dw_f1 = _f1_from_prec_recall(span_dw_precision, span_dw_recall)
            return span_dw_precision, span_dw_recall, span_dw_f1

        def _strict_reanchored_dw_order_scores() -> Tuple[float, float, float]:
            """
            Distance-weighted OrderF1 after strict one-to-one node alignment.

            This supplementary order metric first
            align predicted nodes to gold nodes with the same thresholded
            semantic matcher used by StrictNodeF1, then compare transitive
            precedence relations over the aligned gold-node indices.
            """
            g_edge_pairs_local = _edge_pairs_by_node_index(g_nodes, g_edges)
            p_edge_pairs_local = _edge_pairs_by_node_index(p_nodes, p_edges)
            g_order_pairs_local, g_order_dist_local = _transitive_pairs_with_shortest_dist(
                len(g_nodes),
                g_edge_pairs_local,
            )
            p_order_pairs_local, p_order_dist_local = _transitive_pairs_with_shortest_dist(
                len(p_nodes),
                p_edge_pairs_local,
            )

            if not g_order_pairs_local and not p_order_pairs_local:
                return (1.0, 1.0, 1.0) if g_nodes else (0.0, 0.0, 0.0)

            pred_weight_denom = sum(
                1.0 / max(p_order_dist_local[pair], 1)
                for pair in p_order_pairs_local
            )
            pred_weight_num = 0.0
            covered_gold_pairs: Set[Tuple[int, int]] = set()

            for s_pred, t_pred in p_order_pairs_local:
                s_gold = pred_to_gold.get(s_pred)
                t_gold = pred_to_gold.get(t_pred)
                if s_gold is None or t_gold is None or s_gold == t_gold:
                    continue
                mapped_pair = (s_gold, t_gold)
                if mapped_pair in g_order_pairs_local:
                    pred_weight_num += 1.0 / max(p_order_dist_local[(s_pred, t_pred)], 1)
                    covered_gold_pairs.add(mapped_pair)

            gold_weight_denom = sum(
                1.0 / max(g_order_dist_local[pair], 1)
                for pair in g_order_pairs_local
            )
            gold_weight_num = sum(
                1.0 / max(g_order_dist_local[pair], 1)
                for pair in covered_gold_pairs
                if pair in g_order_dist_local
            )

            precision = (
                pred_weight_num / pred_weight_denom
                if pred_weight_denom > 0 else (1.0 if not g_order_pairs_local else 0.0)
            )
            recall = (
                gold_weight_num / gold_weight_denom
                if gold_weight_denom > 0 else (1.0 if not p_order_pairs_local else 0.0)
            )
            return precision, recall, _f1_from_prec_recall(precision, recall)

        # === Raw Edge F1 ===
        # Diagnostic only: compares edges by their source/target step_index.
        # This can over-credit positionally similar chains even when the
        # predicted node labels do not semantically align to the gold nodes.
        g_edge_pairs = _edge_pairs(g_nodes, g_edges)
        p_edge_pairs = _edge_pairs(p_nodes, p_edges)
        # For edge F1: if gold has no nodes (empty plan), edges should not be "perfect"
        # Use empty_is_perfect=False to avoid inflating Edge F1 when gold is empty
        edge_empty_is_perfect = len(g_nodes) > 0  # Only "perfect" if gold has actual plan
        raw_edge_precision, raw_edge_recall, raw_edge_f1 = _set_f1(
            g_edge_pairs, p_edge_pairs, empty_is_perfect=edge_empty_is_perfect
        )

        # === Semantic Edge F1 ===
        # Re-anchor predicted edges to the semantically matched gold-node indices.
        # Unmatched predicted endpoints are mapped to out-of-gold sentinel IDs so they
        # remain counted as false positives instead of being silently dropped.
        g_edge_pairs_local = _edge_pairs_by_node_index(g_nodes, g_edges)
        p_edge_pairs_local = _edge_pairs_by_node_index(p_nodes, p_edges)
        unmatched_base = len(g_nodes) + (2 * len(p_nodes)) + 1
        p_edge_pairs_semantic: Set[Tuple[int, int]] = set()
        for s_pred, t_pred in p_edge_pairs_local:
            s_gold = pred_to_gold.get(s_pred, unmatched_base + s_pred)
            t_gold = pred_to_gold.get(t_pred, unmatched_base + len(p_nodes) + t_pred)
            p_edge_pairs_semantic.add((s_gold, t_gold))

        semantic_edge_precision, semantic_edge_recall, semantic_edge_f1 = _set_f1(
            g_edge_pairs_local, p_edge_pairs_semantic, empty_is_perfect=edge_empty_is_perfect
        )

        # === Global order metrics ===
        g_order_pairs, g_order_dist = _transitive_pairs_with_shortest_dist(len(g_nodes), g_edge_pairs)
        p_order_pairs, p_order_dist = _transitive_pairs_with_shortest_dist(len(p_nodes), p_edge_pairs)
        order_precision, order_recall, order_f1 = _set_f1(
            g_order_pairs, p_order_pairs, empty_is_perfect=edge_empty_is_perfect
        )

        order_overlap = g_order_pairs & p_order_pairs
        pred_weight_denom = sum(1.0 / max(p_order_dist[pair], 1) for pair in p_order_pairs)
        pred_weight_num = sum(1.0 / max(p_order_dist[pair], 1) for pair in order_overlap)
        gold_weight_denom = sum(1.0 / max(g_order_dist[pair], 1) for pair in g_order_pairs)
        gold_weight_num = sum(1.0 / max(g_order_dist[pair], 1) for pair in order_overlap)

        index_dw_precision = (
            pred_weight_num / pred_weight_denom
            if pred_weight_denom > 0 else (1.0 if not g_order_pairs else 0.0)
        )
        index_dw_recall = (
            gold_weight_num / gold_weight_denom
            if gold_weight_denom > 0 else (1.0 if not p_order_pairs else 0.0)
        )
        index_dw_order_f1 = _f1_from_prec_recall(index_dw_precision, index_dw_recall)
        span_dw_precision, span_dw_recall, span_dw_order_f1 = _span_reanchored_dw_order_scores()
        dw_precision, dw_recall, dw_order_f1 = _strict_reanchored_dw_order_scores()

        # === Node Label Similarity (embedding-based) ===
        node_label_sim = self.compute_node_label_similarity(g_nodes, p_nodes)

        # === SSI = (Node Label Similarity + paper-facing EdgeF1) / 2 ===
        edge_f1 = semantic_edge_f1
        ssi = (node_label_sim + edge_f1) / 2
        planning_score = (strict_node_f1 + semantic_edge_f1) / 2

        return PlanScores(
            node_f1=strict_node_f1,
            edge_f1=edge_f1,
            semantic_edge_f1=semantic_edge_f1,
            dw_order_f1=dw_order_f1,
            planning_score=planning_score,
            raw_edge_f1=raw_edge_f1,
            span_node_f1=span_node_f1,
            strict_node_f1=strict_node_f1,
            strict_reanchored_dw_order_f1=dw_order_f1,
            span_reanchored_dw_order_f1=span_dw_order_f1,
            index_dw_order_f1=index_dw_order_f1,
            node_label_similarity=node_label_sim,
            ssi=ssi,
            node_precision=strict_node_precision,
            node_recall=strict_node_recall,
            strict_node_precision=strict_node_precision,
            strict_node_recall=strict_node_recall,
            edge_precision=semantic_edge_precision,
            edge_recall=semantic_edge_recall,
            raw_edge_precision=raw_edge_precision,
            raw_edge_recall=raw_edge_recall,
            semantic_edge_precision=semantic_edge_precision,
            semantic_edge_recall=semantic_edge_recall,
            dw_order_precision=dw_precision,
            dw_order_recall=dw_recall,
            strict_reanchored_dw_order_precision=dw_precision,
            strict_reanchored_dw_order_recall=dw_recall,
            index_dw_order_precision=index_dw_precision,
            index_dw_order_recall=index_dw_recall,
            order_precision=order_precision,
            order_recall=order_recall,
            order_f1=order_f1,
            gold_node_count=len(g_nodes),
            pred_node_count=len(p_nodes),
        )

    def evaluate_tool_calls(
        self,
        gold_calls: List[Dict[str, Any]],
        pred_calls: List[Dict[str, Any]],
        has_arguments: bool = True,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        pred_tool_outputs: Optional[List[str]] = None,
        dataset: str = "",
        subset: str = "",
    ) -> ToolScores:
        """
        Evaluate tool usage.

        Args:
            gold_calls: Ground truth tool calls
            pred_calls: Predicted tool calls
            has_arguments: Whether this dataset has argument values.
                          If False, param_name_f1 and type_aware_value_f1
                          will be None.
            available_tools: List of available tools from the environment.
                            If provided, pred_calls using invented tools (not in this list)
                            will be filtered out before evaluation.
            pred_tool_outputs: Optional execution observations aligned with pred_calls.
                               When provided, value matching uses only the last
                               successful call per gold tool slot.

        Returns:
            ToolScores with all metrics
        """
        if pred_tool_outputs and len(pred_tool_outputs) == len(pred_calls):
            pred_call_output_pairs = list(zip(pred_calls, pred_tool_outputs))
        else:
            pred_call_output_pairs = [(c, None) for c in pred_calls]

        # Filter out invented tools if available_tools is provided
        if available_tools is not None:
            # Build set of available tool names (normalized)
            available_tool_names = set()
            for t in available_tools:
                tool_id = t.get("tool_id", "")
                if tool_id:
                    available_tool_names.add(_normalize_tool_id(tool_id))
                    available_tool_names.add(_extract_tool_name(tool_id))

            # Filter pred_calls to only include tools in available_tools
            filtered_pred_call_output_pairs = []
            for c, output in pred_call_output_pairs:
                tool_id = c.get("tool_id", "")
                if tool_id:
                    norm_id = _normalize_tool_id(tool_id)
                    tool_name = _extract_tool_name(tool_id)
                    if norm_id in available_tool_names or tool_name in available_tool_names:
                        filtered_pred_call_output_pairs.append((c, output))
            pred_call_output_pairs = filtered_pred_call_output_pairs

        # Exclude administrative tools from F1 scoring
        EXCLUDED_TOOL_IDS = {"submit_final_answer"}
        gold_calls = [c for c in gold_calls
                      if _extract_tool_name(c.get("tool_id", "")) not in EXCLUDED_TOOL_IDS]
        pred_call_output_pairs = [
            (c, output) for c, output in pred_call_output_pairs
            if _extract_tool_name(c.get("tool_id", "")) not in EXCLUDED_TOOL_IDS
        ]
        pred_calls = [c for c, _ in pred_call_output_pairs]
        pred_outputs = [output for _, output in pred_call_output_pairs]

        # ============================================================
        # Tool Name F1 (supports alternatives)
        # Use tool NAMES (after ::) for flexible matching
        # ============================================================

        # Build gold tool sets (including alternatives for each call)
        # Extract tool names for flexible matching across server naming conventions
        gold_tool_sets = []  # List of sets, one per call
        for c in gold_calls:
            tool_id = c.get("tool_id")
            if tool_id:
                tool_names = {_extract_tool_name(tool_id)}
                for alt in c.get("alternative_tools", []) or []:
                    if alt:
                        tool_names.add(_extract_tool_name(alt))
                gold_tool_sets.append(tool_names)

        pred_tools = [_extract_tool_name(c.get("tool_id", ""))
                      for c in pred_calls if c.get("tool_id")]

        # Tool name F1 with alternative support. Recall is gold-slot
        # count-sensitive, but repeated calls to a gold-relevant tool type do not
        # reduce precision: in GAIA/ReAct traces, repeated search/browser calls
        # can be legitimate evidence gathering rather than a tool-selection error.
        if not gold_tool_sets and not pred_tools:
            tool_precision, tool_recall, tool_name_f1 = 1.0, 1.0, 1.0
        elif not gold_tool_sets:
            tool_precision, tool_recall, tool_name_f1 = 0.0, 1.0, 0.0
        elif not pred_tools:
            tool_precision, tool_recall, tool_name_f1 = 1.0, 0.0, 0.0
        else:
            matched_pred_indices = set()
            matched_gold_indices = set()

            for pred_idx, p_tool in enumerate(pred_tools):
                for i, g_tool_set in enumerate(gold_tool_sets):
                    if p_tool in g_tool_set and i not in matched_gold_indices:
                        matched_pred_indices.add(pred_idx)
                        matched_gold_indices.add(i)
                        break

            extra_pred_count = sum(
                1
                for p_tool in pred_tools
                if not any(p_tool in g_tool_set for g_tool_set in gold_tool_sets)
            )
            precision_denom = len(matched_gold_indices) + extra_pred_count
            tool_precision = (
                len(matched_gold_indices) / precision_denom
                if precision_denom > 0 else 0.0
            )
            tool_recall = len(matched_gold_indices) / len(gold_tool_sets) if gold_tool_sets else 0.0
            tool_name_f1 = _f1_from_prec_recall(tool_precision, tool_recall)

        # ============================================================
        # Parameter metrics (t-F1 and v-F1) 
        # ============================================================

        # First, check if any arguments exist in gold
        gold_has_args = False
        for c in gold_calls:
            args = c.get("arguments", [])
            if args and (isinstance(args, list) and len(args) > 0) or (isinstance(args, dict) and len(args) > 0):
                gold_has_args = True
                break

        # If has_arguments is False OR gold has no actual arguments,
        # return None for param metrics to indicate "not applicable"
        if not has_arguments or not gold_has_args:
            return ToolScores(
                tool_name_f1=tool_name_f1,
                param_name_f1=None,  # N/A
                type_aware_value_f1=None,  # N/A
                strict_type_aware_value_f1=None,
                normalized_type_aware_value_f1=None,
                tool_precision=tool_precision,
                tool_recall=tool_recall,
                gold_tool_count=len(gold_tool_sets),
                pred_tool_count=len(pred_tools),
                gold_param_count=0,
                pred_param_count=0,
                matched_param_count=0,
                has_arguments=False,
            )

        # Calculate TaskBench-style t-F1 and v-F1
        gold_task_arg_names: List[str] = []      # "{tool_name}-{param_name}"
        gold_type_aware_entries: List[Tuple[str, str, Any]] = []
        pred_task_arg_names: List[str] = []
        pred_type_aware_entries: List[Tuple[str, str, Any]] = []

        use_taskbench_normalization = False
        if dataset == "taskbench":
            gold_arg_names: List[str] = []
            for c in gold_calls:
                gold_arg_names.extend(name for name, _ in self._extract_arguments(c.get("arguments", [])))
            use_taskbench_normalization = (
                subset in {"huggingface", "multimedia"}
                or any(self._is_anonymous_arg_name(name) for name in gold_arg_names)
            )

        tool_catalog = self._build_tool_catalog(available_tools)

        for c in gold_calls:
            tool_id = c.get("tool_id", "")
            if not tool_id:
                continue
            tool_name = _extract_tool_name(tool_id)

            if use_taskbench_normalization:
                normalized_args = self._normalize_taskbench_call_arguments(
                    c, gold_calls, tool_catalog, force_type_based=True
                )
            else:
                normalized_args = self._extract_arguments(c.get("arguments", []))

            for name, value in normalized_args:
                if name:
                    task_arg_key = f"{tool_name}-{name}"
                    gold_task_arg_names.append(task_arg_key)
                    gold_type_aware_entries.append((tool_name, name, value))

        pred_calls_for_value_metrics = pred_calls
        if pred_outputs and len(pred_outputs) == len(pred_calls):
            success_flags = []
            for output in pred_outputs:
                output_text = str(output or "")
                if "Output:\n" in output_text:
                    output_text = output_text.split("Output:\n", 1)[1]
                output_norm = _normalize_string(output_text)
                is_error = (
                    not output_norm
                    or output_norm.startswith("[error]")
                    or output_norm.startswith("error:")
                    or output_norm.startswith("execution failed:")
                )
                success_flags.append(not is_error)

            selected_indices: Set[int] = set()
            for gold_tool_set in gold_tool_sets:
                last_success_idx = None
                for idx, (call, is_success) in enumerate(zip(pred_calls, success_flags)):
                    if not is_success or idx in selected_indices:
                        continue
                    pred_tool_name = _extract_tool_name(call.get("tool_id", ""))
                    if pred_tool_name in gold_tool_set:
                        last_success_idx = idx
                if last_success_idx is not None:
                    selected_indices.add(last_success_idx)

            gold_tool_union = set().union(*gold_tool_sets) if gold_tool_sets else set()
            extra_indices = {
                idx for idx, (call, is_success) in enumerate(zip(pred_calls, success_flags))
                if is_success
                and idx not in selected_indices
                and _extract_tool_name(call.get("tool_id", "")) not in gold_tool_union
            }
            pred_calls_for_value_metrics = [pred_calls[idx] for idx in sorted(selected_indices | extra_indices)]

        for c in pred_calls:
            tool_id = c.get("tool_id", "")
            if not tool_id:
                continue
            tool_name = _extract_tool_name(tool_id)

            if use_taskbench_normalization:
                normalized_args = self._normalize_taskbench_call_arguments(
                    c, pred_calls, tool_catalog, force_type_based=True
                )
            else:
                normalized_args = self._extract_arguments(c.get("arguments", []))

            for name, value in normalized_args:
                if name:
                    pred_task_arg_names.append(f"{tool_name}-{name}")

        for c in pred_calls_for_value_metrics:
            tool_id = c.get("tool_id", "")
            if not tool_id:
                continue
            tool_name = _extract_tool_name(tool_id)

            if use_taskbench_normalization:
                normalized_args = self._normalize_taskbench_call_arguments(
                    c, pred_calls, tool_catalog, force_type_based=True
                )
            else:
                normalized_args = self._extract_arguments(c.get("arguments", []))

            for name, value in normalized_args:
                if not name:
                    continue
                pred_type_aware_entries.append((tool_name, name, value))

        # For param metrics with arguments, empty is NOT perfect (if gold has
        # args and pred is empty, that's a miss). Repeated gold parameter slots
        # are recall-sensitive, while repeated valid ReAct calls using the same
        # parameter name do not by themselves lower precision.
        _, _, param_name_f1, matched_param_count = _gold_slot_tolerant_item_f1(
            gold_task_arg_names, pred_task_arg_names, empty_is_perfect=False
        )
        strict_type_aware_value_f1 = _type_aware_value_f1(
            gold_type_aware_entries,
            pred_type_aware_entries,
        )
        normalized_type_aware_value_f1 = None
        normalized_diag = {"gold_entries": 0, "pred_entries": 0, "matched": 0}
        if str(dataset or "").strip().lower() == "gaia":
            # GAIA mixes file readers, typed attachment tools, web queries,
            # image/audio aliases, and Python snippets.  The official GAIA
            # value component therefore uses execution-normalized stable
            # value entries, while keeping the old strict value metric as an
            # audit field below.
            normalized_type_aware_value_f1, normalized_diag = normalized_value_f1(
                gold_calls,
                pred_calls,
                threshold=0.45,
                include_open_text=True,
                include_reasoning_problem=False,
                include_custom_prompt=False,
                include_code_text=False,
            )

        type_aware_value_f1 = (
            normalized_type_aware_value_f1
            if normalized_type_aware_value_f1 is not None
            else strict_type_aware_value_f1
        )

        return ToolScores(
            tool_name_f1=tool_name_f1,
            param_name_f1=param_name_f1,
            type_aware_value_f1=type_aware_value_f1,
            strict_type_aware_value_f1=strict_type_aware_value_f1,
            normalized_type_aware_value_f1=normalized_type_aware_value_f1,
            tool_precision=tool_precision,
            tool_recall=tool_recall,
            gold_tool_count=len(gold_tool_sets),
            pred_tool_count=len(pred_tools),
            gold_param_count=len(gold_task_arg_names),
            pred_param_count=len(pred_task_arg_names),
            matched_param_count=matched_param_count,
            normalized_value_gold_count=int(normalized_diag.get("gold_entries", 0) or 0),
            normalized_value_pred_count=int(normalized_diag.get("pred_entries", 0) or 0),
            normalized_value_matched_count=int(normalized_diag.get("matched", 0) or 0),
            has_arguments=True,
        )

    def _smart_match(self, gold: str, pred: str) -> bool:
        """
        Check if gold answer is wrapped by a lightweight answer phrase.
        This is intentionally conservative: we allow small scaffolding such as
        "the answer is X", but do not treat arbitrary supersets as exact match.
        """
        g_norm = _normalize_string(str(gold))
        p_norm = _normalize_string(str(pred))
        
        if not g_norm or not p_norm:
            return False
            
        # 1. Exact match (already checked but good for safety)
        if g_norm == p_norm:
            return True

        # 2. Conservative wrapper phrases only.
        wrapper_patterns = [
            rf"^(?:the answer is|answer is|answer:|final answer:|the final answer is)\s+{re.escape(g_norm)}[.!?]?$",
            rf"^(?:it is|it's|its)\s+{re.escape(g_norm)}[.!?]?$",
            rf"^(?:therefore|thus|so)[,:]?\s+{re.escape(g_norm)}[.!?]?$",
        ]
        if any(re.fullmatch(pattern, p_norm) for pattern in wrapper_patterns):
            return True

        return False

    def _is_pure_numeric_string(self, value: str) -> bool:
        norm = _normalize_string(value).replace(",", "")
        return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?", norm))

    def _normalized_exact_match(self, gold: str, pred: str) -> bool:
        """
        Conservative format-aware exact match.

        This is intentionally limited to formatting differences that do not
        change semantics, such as terminal punctuation and delimiter spacing.
        """
        def _canonical_text(value: str) -> str:
            norm = _normalize_string(value)
            norm = re.sub(r"[.!?]+$", "", norm).strip()
            norm = re.sub(r"\s*([,;:])\s*", r"\1", norm)
            norm = re.sub(r"\s+", " ", norm)
            return norm.strip()

        def _canonical_simple_list(value: str) -> str:
            norm = _canonical_text(value)
            if not norm:
                return norm
            candidate = re.sub(r"\s+(?:and|&)\s+", ",", norm)
            if "," not in candidate:
                return norm
            items = [item.strip() for item in candidate.split(",") if item.strip()]
            if len(items) < 2:
                return norm
            if not all(1 <= len(_tokenize(item)) <= 4 for item in items):
                return norm
            return ",".join(items)

        g_basic = _canonical_text(gold)
        p_basic = _canonical_text(pred)
        if g_basic and p_basic and g_basic == p_basic:
            return True

        g_list = _canonical_simple_list(gold)
        p_list = _canonical_simple_list(pred)
        return bool(g_list and p_list and g_list == p_list)

    def _is_short_numeric_wrapper(self, value: str) -> bool:
        norm = _normalize_string(value)
        tokens = _tokenize(value)
        if len(tokens) > 6:
            return False
        numeric_tokens = re.findall(r"[-+]?\d*\.?\d+", norm.replace(",", ""))
        return len(numeric_tokens) == 1

    def _safe_numeric_wrapper_match(
        self,
        gold: str,
        pred: str,
        tolerance: float = 0.0,
    ) -> bool:
        """
        Allow exact match when the prediction wraps the same numeric value in a
        lightweight count/currency/unit phrase, while rejecting magnitude
        changes such as "17 thousand hours".
        """
        g_norm = _normalize_string(str(gold))
        p_norm = _normalize_string(str(pred))
        if not g_norm or not p_norm or not self._is_pure_numeric_string(g_norm):
            return False

        num_pattern = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        gold_nums = re.findall(num_pattern, g_norm.replace(",", ""))
        pred_nums = re.findall(num_pattern, p_norm.replace(",", ""))
        if len(gold_nums) != 1 or len(pred_nums) != 1:
            return False

        try:
            g_num = float(gold_nums[0])
            p_num = float(pred_nums[0])
        except (ValueError, TypeError):
            return False

        if not (abs(g_num - p_num) <= tolerance or abs(g_num - p_num) / (abs(g_num) + 1e-8) < 1e-6):
            return False

        residue = p_norm.replace(pred_nums[0], " ", 1)
        residue = re.sub(r"[$€£¥₩₹]", " ", residue)
        residue = re.sub(r"[()%\\[\\]{}.,;:/\\\\]", " ", residue)
        residue = re.sub(r"\s+", " ", residue).strip()
        if not residue:
            return True

        tokens = residue.split()
        magnitude_words = {
            "hundred", "hundreds", "thousand", "thousands", "million", "millions",
            "billion", "billions", "trillion", "trillions", "k", "m", "bn",
        }
        if any(tok in magnitude_words for tok in tokens):
            return False

        safe_wrapper_tokens = {
            "participant", "participants", "person", "people", "user", "users",
            "item", "items", "record", "records", "entry", "entries",
            "task", "tasks", "ticket", "tickets", "times", "time",
            "hour", "hours", "minute", "minutes", "second", "seconds",
            "day", "days", "week", "weeks", "month", "months", "year", "years",
            "dollar", "dollars", "usd", "eur", "euro", "euros", "gbp",
            "pound", "pounds", "ntd", "twd", "hkd", "cad", "aud", "jpy",
            "yen", "cny", "rmb", "percent", "percentage", "percentages",
            "degree", "degrees",
        }

        if all(tok in safe_wrapper_tokens for tok in tokens):
            return True

        # Allow a single short unit token (e.g., "å", "km", "kg") but avoid
        # broader free-text wrappers that change semantics.
        if len(tokens) == 1 and len(tokens[0]) <= 5:
            return True

        return False

    def evaluate_answer(
        self,
        gold_ans: Dict[str, Any],
        pred_ans: Dict[str, Any],
        query_text: str = "",
    ) -> AnswerScores:
        """Evaluate final answer."""
        ans_type = gold_ans.get("answer_type", "none")
        gold_value = gold_ans.get("answer")
        aliases = gold_ans.get("aliases", []) or []
        tolerance = float(gold_ans.get("tolerance", 0.0) or 0.0)

        if ans_type == "none" or gold_value is None:
            return AnswerScores(has_answer=False)

        # Handle case where pred_ans is not a dict (string, int, float)
        if isinstance(pred_ans, (str, int, float)):
            pred_value = pred_ans
        elif isinstance(pred_ans, dict):
            pred_value = pred_ans.get("answer") if pred_ans else None
        else:
            pred_value = None

        if pred_value is None:
            return AnswerScores(
                has_answer=True,
                exact_match=0.0,
                token_f1=0.0,
                alias_match=0.0,
                llm_judge_score=self._judge_answer_with_llm(query_text, gold_ans, pred_value),
            )

        # Check alias match
        alias_match = 0.0
        
        # Numeric answer
        if ans_type == "number":
            try:
                g = float(gold_value)
                # Try to extract number from prediction if it's a string
                p_val = pred_value
                if isinstance(p_val, str):
                    if self._safe_numeric_wrapper_match(str(gold_value), p_val, tolerance=tolerance):
                        return AnswerScores(
                            has_answer=True,
                            exact_match=1.0,
                            token_f1=1.0,
                            numeric_rel_error=0.0,
                            alias_match=1.0,
                            llm_judge_score=self._judge_answer_with_llm(query_text, gold_ans, pred_value),
                        )
                
                # If extraction failed or didn't match, try direct conversion
                p = float(pred_value)
                em = 1.0 if abs(g - p) <= tolerance else 0.0
                rel_err = abs(g - p) / (abs(g) + 1e-8)
                return AnswerScores(
                    has_answer=True,
                    exact_match=em,
                    token_f1=em,
                    numeric_rel_error=rel_err,
                    alias_match=max(alias_match, em),
                    llm_judge_score=self._judge_answer_with_llm(query_text, gold_ans, pred_value),
                )
            except (ValueError, TypeError):
                pass

        # String answer
        em, f1 = _string_em_f1(str(gold_value), str(pred_value))

        if em < 1.0 and self._normalized_exact_match(str(gold_value), str(pred_value)):
            em = 1.0

        # Soft match for string answers (e.g. sentence wrapping)
        if em < 1.0 and self._smart_match(str(gold_value), str(pred_value)):
            em = 1.0

        # Conservative numeric fallback for string-typed answers that are
        # actually pure numeric strings. This avoids false positives on
        # alphanumeric IDs such as award codes or catalog numbers.
        if em < 1.0:
            try:
                if self._is_pure_numeric_string(str(gold_value)):
                    if self._safe_numeric_wrapper_match(str(gold_value), str(pred_value)):
                        em = 1.0
            except (ValueError, TypeError):
                pass

        # Check aliases for better EM
        if em < 1.0:
            for alias in aliases:
                if self._normalized_exact_match(str(alias), str(pred_value)):
                    em = 1.0
                    alias_match = 1.0
                    break

                # Check soft match for aliases too
                if self._smart_match(str(alias), str(pred_value)):
                    em = 1.0
                    alias_match = 1.0
                    break

                alias_em, _ = _string_em_f1(str(alias), str(pred_value))
                if alias_em > em:
                    em = alias_em
                    alias_match = max(alias_match, alias_em)
                    break

        return AnswerScores(
            has_answer=True,
            exact_match=em,
            token_f1=1.0 if em == 1.0 else f1,
            alias_match=alias_match,
            llm_judge_score=self._judge_answer_with_llm(query_text, gold_ans, pred_value),
        )

    def evaluate_record(
        self,
        gold: Dict[str, Any],
        pred: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single record.

        Args:
            gold: Ground truth from unified format
            pred: Prediction from model
            meta: Optional metadata (used to determine has_arguments)

        Returns:
            Dict with plan, tool, and answer scores
        """
        def _select_pred_plan_dag(pred_obj: Dict[str, Any], meta_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            pred_obj = pred_obj or {}
            plan_source = str((meta_obj or {}).get("plan_source") or "stage1").lower()

            if plan_source in {"stage3", "execution", "refined", "plan_dag"}:
                return pred_obj.get("plan_dag") or {}

            if plan_source not in {"stage1", "abs", "abstract", "abs_plan_dag"}:
                raise ValueError(
                    f"Unsupported plan_source='{plan_source}'. "
                    "Use one of: stage1, abs, abstract, abs_plan_dag, stage3."
                )

            trace = pred_obj.get("_trace") or {}
            abstract = trace.get("abstract_plan") or {}
            return (
                pred_obj.get("abs_plan_dag")
                or trace.get("abs_plan_dag")
                or abstract.get("abs_plan_dag")
                or abstract.get("abstract_plan_dag")
                # Backward-compatible fallback for older outputs that were
                # produced before abs_plan_dag existed.
                or pred_obj.get("plan_dag")
                or {}
            )

        # Determine if this record has arguments
        dataset = meta.get("dataset", "") if meta else ""
        subset = meta.get("subset", "") if meta else ""

        # Support both old schema (plan_dag, tool_calls) and new schema (canonical_plan_dag, minimal_tool_ast)
        gold_dag = gold.get("plan_dag") or gold.get("canonical_plan_dag") or {}
        gold_calls = gold.get("tool_calls") or gold.get("minimal_tool_ast") or []
        pred_dag = _select_pred_plan_dag(pred or {}, meta)
        pred_calls = (pred or {}).get("tool_calls") or []
        pred_tool_outputs = (pred or {}).get("tool_outputs") or []

        has_image_attachment = bool(meta.get("has_image_attachment")) if meta else False
        supports_native_vision = meta.get("supports_native_vision") if meta else None
        if supports_native_vision is False and has_image_attachment:
            has_non_admin_gold_tool = any(
                _extract_tool_name(c.get("tool_id", "")) != "submit_final_answer"
                for c in gold_calls
            )
            if not has_non_admin_gold_tool:
                gold_calls = list(gold_calls) + [{"tool_id": "image_recognition", "arguments": []}]

        # Check explicit flag first
        if meta and "has_arguments" in meta:
            has_arguments = meta.get("has_arguments", True)
        # For Delta dataset, assume no arguments
        elif dataset == "delta":
            has_arguments = False
        else:
            # Auto-detect by checking if gold has any actual arguments
            has_arguments = False
            for c in gold_calls:
                args = c.get("arguments", [])
                if args:
                    if isinstance(args, list) and len(args) > 0:
                        has_arguments = True
                        break
                    elif isinstance(args, dict) and len(args) > 0:
                        has_arguments = True
                        break

        def _planning_score_for_reference(scores: PlanScores) -> float:
            return (float(scores.strict_node_f1 or 0.0) + float(scores.edge_f1 or 0.0)) / 2.0

        def _pack_reference_scores(prefix: str, scores: PlanScores) -> Dict[str, Any]:
            packed = asdict(scores)
            return {
                f"{prefix}_node_f1": packed.get("strict_node_f1", packed.get("node_f1", 0.0)),
                f"{prefix}_dw_order_f1": packed.get("dw_order_f1", 0.0),
                f"{prefix}_semantic_edge_f1": packed.get("semantic_edge_f1", 0.0),
                f"{prefix}_edge_f1": packed.get("edge_f1", 0.0),
                f"{prefix}_raw_edge_f1": packed.get("raw_edge_f1", 0.0),
                f"{prefix}_node_label_similarity": packed.get("node_label_similarity", 0.0),
                f"{prefix}_planning_score": _planning_score_for_reference(scores),
            }

        reference_candidates: List[Tuple[str, Dict[str, Any]]] = []
        if meta:
            raw_candidates = meta.get("reference_plan_dags") or []
            if isinstance(raw_candidates, list):
                for item in raw_candidates:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("reference") or "").strip()
                    dag = item.get("dag") or item.get("plan_dag")
                    if name and isinstance(dag, dict):
                        reference_candidates.append((name, dag))

        if not reference_candidates:
            reference_candidates = [("chain", gold_dag)]

        scored_references: List[Tuple[str, PlanScores]] = [
            (
                name,
                self.evaluate_plan_dag(
                    gold_dag=dag,
                    pred_dag=pred_dag,
                    dataset=dataset,
                ),
            )
            for name, dag in reference_candidates
        ]

        # Prefer the original chain in exact ties. This keeps the legacy GAIA
        # reference primary unless the dependency DAG genuinely improves the
        # NodeF1+EdgeF1 planning score.
        ref_preference = {"chain": 0, "dag": 1, "dependency_dag": 1}
        selected_reference, plan_scores = max(
            scored_references,
            key=lambda item: (
                _planning_score_for_reference(item[1]),
                -ref_preference.get(item[0], 99),
            ),
        )

        plan_score_dict = asdict(plan_scores)
        plan_score_dict["reference_mode"] = "augmented_best" if len(scored_references) > 1 else "chain_only"
        plan_score_dict["reference_selected"] = selected_reference

        score_by_name = {name: scores for name, scores in scored_references}
        chain_scores = score_by_name.get("chain") or scored_references[0][1]
        plan_score_dict.update(_pack_reference_scores("chain_only", chain_scores))
        dependency_scores = score_by_name.get("dag") or score_by_name.get("dependency_dag")
        if dependency_scores is not None:
            plan_score_dict.update(_pack_reference_scores("dependency_dag", dependency_scores))
        else:
            for key in (
                "node_f1",
                "dw_order_f1",
                "semantic_edge_f1",
                "edge_f1",
                "raw_edge_f1",
                "node_label_similarity",
                "planning_score",
            ):
                plan_score_dict[f"dependency_dag_{key}"] = None
        plan_score_dict.update(_pack_reference_scores("augmented_best", plan_scores))

        # Extract available tools from metadata or gold record for filtering invented tools
        available_tools = None
        if meta and "available_tools" in meta:
            available_tools = meta.get("available_tools")
        elif "tool_environment" in gold:
            available_tools = normalize_tool_environment(gold.get("tool_environment", {}))

        tool_scores = self.evaluate_tool_calls(
            gold_calls=gold_calls,
            pred_calls=pred_calls,
            has_arguments=has_arguments,
            available_tools=available_tools,
            pred_tool_outputs=pred_tool_outputs,
            dataset=dataset,
            subset=subset,
        )

        ans_scores = self.evaluate_answer(
            gold_ans=gold.get("final_answer") or {},
            pred_ans=(pred or {}).get("final_answer") or {},
            query_text=str((query or {}).get("user_query", "") or ""),
        )

        return {
            "plan": plan_score_dict,
            "tool": asdict(tool_scores),
            "answer": asdict(ans_scores),
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

def convert_gaia_gold_format(raw_gold: dict) -> dict:
    """Convert GAIA original format to evaluation format."""
    import json as _json

    # If already in evaluation format, return as-is
    if "tool_calls" in raw_gold or "plan_dag" in raw_gold:
        return raw_gold

    # Convert GAIA format
    result = {
        "task_id": raw_gold.get("task_id"),
        "tool_calls": [],
        "plan_dag": {"nodes": [], "edges": []},
        "final_answer": {"answer": raw_gold.get("Final answer", "")}
    }

    # Parse Annotator Metadata
    meta = raw_gold.get("Annotator Metadata", {})

    # Parse Tools (may be JSON string or list)
    tools_raw = meta.get("Tools", "[]")
    if isinstance(tools_raw, str):
        try:
            tools = _json.loads(tools_raw)
        except:
            tools = []
    else:
        tools = tools_raw or []

    # Parse Tool Arguments
    tool_args = meta.get("Tool Arguments", [])

    # Build tool_calls
    tool_args_by_name = {ta.get("tool"): ta.get("arguments", []) for ta in tool_args}

    for i, tool_name in enumerate(tools):
        args = tool_args_by_name.get(tool_name, [])
        result["tool_calls"].append({
            "tool_id": tool_name,
            "arguments": args
        })
        # Add to plan_dag nodes
        result["plan_dag"]["nodes"].append({
            "node_id": f"n{i}",
            "step_index": i,
            "tool_id": tool_name,
            "label": f"Use {tool_name}"
        })

    # Add edges (sequential)
    for i in range(len(tools) - 1):
        result["plan_dag"]["edges"].append({
            "source": f"n{i}",
            "target": f"n{i+1}"
        })

    return result


def run_evaluation(
    gold_path: str,
    pred_path: str,
    dataset: str = "gaia",
    use_embeddings: bool = False,
    plan_source: str = "stage1",
):
    """Run evaluation on gold and prediction files."""
    import json
    import sys
    from pathlib import Path

    gold_path = Path(gold_path)
    pred_path = Path(pred_path)

    if not gold_path.exists():
        print(f"[ERROR] Gold file not found: {gold_path}")
        sys.exit(1)

    if not pred_path.exists():
        print(f"[ERROR] Prediction file not found: {pred_path}")
        sys.exit(1)

    print(f"Loading gold standard: {gold_path}")
    print(f"Plan source: {plan_source}")
    gold_records = []
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                # Auto-convert GAIA format
                gold_records.append(convert_gaia_gold_format(raw))

    print(f"Loading predictions: {pred_path}")
    pred_records = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pred_records.append(json.loads(line))

    print(f"\n[OK] Loaded {len(gold_records)} gold records, {len(pred_records)} predictions")

    # Create task_id lookup for predictions
    pred_by_task_id = {p.get("task_id"): p for p in pred_records}

    # Initialize evaluator
    evaluator = ASTEvaluationSystem(use_embeddings=use_embeddings)

    # Evaluate each gold record
    all_scores = []
    matched_count = 0

    for gold in gold_records:
        task_id = gold.get("task_id")
        pred = pred_by_task_id.get(task_id)

        if pred is None:
            print(f"[WARN] No prediction found for task_id: {task_id}")
            continue

        metadata = {"dataset": dataset, "task_id": task_id, "plan_source": plan_source}
        scores = evaluator.evaluate_record(gold, pred, metadata)
        all_scores.append(scores)
        matched_count += 1

    if matched_count == 0:
        print("[ERROR] No matching task_ids found between gold and prediction files")
        sys.exit(1)

    # Aggregate scores
    print(f"\n{'='*70}")
    print(f"Evaluation Results ({matched_count} tasks)")
    print(f"{'='*70}")

    # Compute averages
    def avg(key_path):
        """Average a nested key like 'plan.node_f1'"""
        parts = key_path.split('.')
        values = []
        for s in all_scores:
            val = s
            for part in parts:
                val = val.get(part, {})
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    val = None
                    break
            if val is not None and isinstance(val, (int, float)):
                values.append(val)
        return sum(values) / len(values) if values else 0.0

    print("\n--- Plan Metrics ---")
    print(f"Node F1:       {avg('plan.node_f1'):.3f}")
    print(f"Edge F1:       {avg('plan.edge_f1'):.3f}")
    print(f"Raw Edge F1:   {avg('plan.raw_edge_f1'):.3f}")
    print(f"Node Sim:      {avg('plan.node_label_similarity'):.3f}")
    print(f"SSI:           {avg('plan.ssi'):.3f}")

    print("\n--- Tool Metrics ---")
    print(f"Tool F1:       {avg('tool.tool_name_f1'):.3f}")

    # Only show param/value F1 if dataset has arguments
    if dataset not in ["delta"]:
        print(f"Param F1:      {avg('tool.param_name_f1'):.3f}")
        print(f"Value F1:      {avg('tool.type_aware_value_f1'):.3f}")
    else:
        print(f"Param F1:      N/A (Delta dataset)")
        print(f"Value F1:      N/A (Delta dataset)")

    print("\n--- Answer Metrics ---")
    print(f"Answer Match:  {avg('answer.exact_match'):.3f}")

    # Detailed per-task breakdown
    print(f"\n{'='*70}")
    print("Per-Task Breakdown")
    print(f"{'='*70}")
    print(f"{'Task ID':<40} {'Node F1':>8} {'Tool F1':>8} {'Answer':>8}")
    print("-" * 70)

    for i, scores in enumerate(all_scores):
        task_id = gold_records[i].get("task_id", f"task_{i}")[:38]
        node_f1 = scores['plan'].get('node_f1', 0.0) or 0.0
        tool_f1 = scores['tool'].get('tool_name_f1', 0.0) or 0.0
        answer = scores['answer'].get('exact_match', 0.0) or 0.0
        print(f"{task_id:<40} {node_f1:>8.3f} {tool_f1:>8.3f} {answer:>8.3f}")

    print(f"{'='*70}")
    print(f"Evaluation complete!")
    print(f"{'='*70}")


def run_tests():
    """Run built-in unit tests."""
    import json

    print("=" * 70)
    print("Running built-in unit tests...")
    print("=" * 70)
    evaluator = ASTEvaluationSystem(use_embeddings=False)

    gold = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Search files", "step_type": "tool", "tool_id": "search_files::search_files"},
                {"node_id": "n1", "step_index": 1, "label": "Read file", "step_type": "tool", "tool_id": "read_text_file::read_text_file"},
                {"node_id": "n2", "step_index": 2, "label": "Analyze", "step_type": "tool", "tool_id": "codefix_query_agent::codefix_query_agent"},
            ],
            "edges": [
                {"source": "n0", "target": "n1", "edge_type": "data_dep"},
                {"source": "n1", "target": "n2", "edge_type": "data_dep"},
            ]
        },
        "tool_calls": [
            {"tool_id": "search_files::search_files", "arguments": []},
            {"tool_id": "read_text_file::read_text_file", "arguments": []},
            {"tool_id": "codefix_query_agent::codefix_query_agent", "arguments": []},
        ],
        "final_answer": {"answer_type": "none"}
    }

    # Prediction only has 1 tool (partial match)
    pred = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Invoke MISRA agent", "step_type": "tool", "tool_id": "codefix_query_agent::codefix_query_agent"},
            ],
            "edges": []
        },
        "tool_calls": [
            {"tool_id": "codefix_query_agent::codefix_query_agent", "arguments": []},
        ],
        "final_answer": {"answer_type": "none"}
    }

    meta = {"dataset": "delta", "has_arguments": False}
    scores = evaluator.evaluate_record(gold, pred, meta)

    print(f"Node F1: {scores['plan']['node_f1']:.3f} (expected: ~0.0, only 1/3 match)")
    print(f"Edge F1: {scores['plan']['edge_f1']:.3f} (expected: 0.0, pred has no edges)")
    print(f"Tool F1: {scores['tool']['tool_name_f1']:.3f} (expected: 0.5, 1 match out of 3 gold / 1 pred)")
    print(f"Param F1: {scores['tool']['param_name_f1']} (expected: None - N/A for Delta)")
    print(f"Value F1: {scores['tool']['type_aware_value_f1']} (expected: None - N/A for Delta)")
    print(f"Has Arguments: {scores['tool']['has_arguments']} (expected: False)")

    print("\n" + "=" * 70)
    print("Test 2: TaskBench style with arguments")
    print("=" * 70)

    gold_args = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "play_movie", "arguments": [{"name": "title", "value": "Inception"}]},
            {"tool_id": "get_info", "arguments": [{"name": "query", "value": "cast"}]},
        ],
        "final_answer": {"answer_type": "none"}
    }

    pred_args = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "play_movie", "arguments": [{"name": "title", "value": "Inception"}]},
        ],
        "final_answer": {"answer_type": "none"}
    }

    meta_args = {"dataset": "taskbench", "has_arguments": True}
    scores = evaluator.evaluate_record(gold_args, pred_args, meta_args)

    print(f"Tool F1: {scores['tool']['tool_name_f1']:.3f} (expected: 0.667, 1 match / 2 gold)")
    print(f"Param F1: {scores['tool']['param_name_f1']:.3f} (expected: 0.667)")
    print(f"Value F1: {scores['tool']['type_aware_value_f1']:.3f} (type-aware value metric)")
    print(f"Has Arguments: {scores['tool']['has_arguments']} (expected: True)")
    print(f"Gold param count: {scores['tool']['gold_param_count']} (expected: 2)")
    print(f"Pred param count: {scores['tool']['pred_param_count']} (expected: 1)")

    print("\n" + "=" * 70)
    print("Test 3: Alternative tools (Delta style)")
    print("=" * 70)

    gold_alt = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Send tx", "step_type": "tool",
                 "tool_id": "ServerA::send_tx", "alternative_tools": ["ServerB::send_tx", "ServerC::send_tx"]},
            ],
            "edges": []
        },
        "tool_calls": [
            {"tool_id": "ServerA::send_tx", "alternative_tools": ["ServerB::send_tx", "ServerC::send_tx"], "arguments": []},
        ],
        "final_answer": {"answer_type": "none"}
    }

    pred_alt = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Send tx", "step_type": "tool", "tool_id": "ServerB::send_tx"},
            ],
            "edges": []
        },
        "tool_calls": [
            {"tool_id": "ServerB::send_tx", "arguments": []},
        ],
        "final_answer": {"answer_type": "none"}
    }

    meta_alt = {"dataset": "delta", "has_arguments": False}
    scores = evaluator.evaluate_record(gold_alt, pred_alt, meta_alt)

    print(f"Node F1: {scores['plan']['node_f1']:.3f} (expected: 1.0 - alternative match)")
    print(f"Tool F1: {scores['tool']['tool_name_f1']:.3f} (expected: 1.0 - alternative match)")

    print("\n" + "=" * 70)
    print("Test 4: Pred with extra arguments (TaskBench)")
    print("=" * 70)

    gold_extra = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "get_weather", "arguments": [{"name": "city", "value": "Tokyo"}]},
        ],
        "final_answer": {"answer_type": "none"}
    }

    pred_extra = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "get_weather", "arguments": [
                {"name": "city", "value": "Tokyo"},
                {"name": "units", "value": "celsius"},  # Extra argument
            ]},
        ],
        "final_answer": {"answer_type": "none"}
    }

    meta_extra = {"dataset": "taskbench", "has_arguments": True}
    scores = evaluator.evaluate_record(gold_extra, pred_extra, meta_extra)

    print(f"Tool F1: {scores['tool']['tool_name_f1']:.3f} (expected: 1.0)")
    print(f"Param F1: {scores['tool']['param_name_f1']:.3f} (expected: 0.667, 1/1 gold but 1/2 pred)")
    print(f"Value F1: {scores['tool']['type_aware_value_f1']:.3f}")

    print("\n" + "=" * 70)
    print("Test 5: SpanNodeF1 with strict one-to-one node diagnostic (TaskBench/UltraTool)")
    print("=" * 70)

    # TaskBench/UltraTool style: node_f1 is the backward-compatible SpanNodeF1 alias.
    gold_steps = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Search for info"},
                {"node_id": "n1", "step_index": 1, "label": "Parse results"},
                {"node_id": "n2", "step_index": 2, "label": "Summarize"},
                {"node_id": "n3", "step_index": 3, "label": "Format output"},
                {"node_id": "n4", "step_index": 4, "label": "Return answer"},
            ],
            "edges": [
                {"source": "n0", "target": "n1"},
                {"source": "n1", "target": "n2"},
                {"source": "n2", "target": "n3"},
                {"source": "n3", "target": "n4"},
            ]
        },
        "tool_calls": [],
        "final_answer": {"answer_type": "none"}
    }

    # Pred only has 3 steps (indices 0, 1, 2)
    pred_steps = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Search"},
                {"node_id": "n1", "step_index": 1, "label": "Parse"},
                {"node_id": "n2", "step_index": 2, "label": "Summarize"},
            ],
            "edges": [
                {"source": "n0", "target": "n1"},
                {"source": "n1", "target": "n2"},
            ]
        },
        "tool_calls": [],
        "final_answer": {"answer_type": "none"}
    }

    meta_steps = {"dataset": "taskbench"}
    scores = evaluator.evaluate_record(gold_steps, pred_steps, meta_steps)

    # Embedding SpanNodeF1 recovers the first three nodes and may align a short
    # missing tail span when it is semantically close to the returned answer.
    print(f"Node F1: {scores['plan']['node_f1']:.3f} (embedding SpanNodeF1 diagnostic)")
    print(f"Node Precision: {scores['plan']['node_precision']:.3f} (embedding SpanNodeF1 diagnostic)")
    print(f"Node Recall: {scores['plan']['node_recall']:.3f} (embedding SpanNodeF1 diagnostic)")
    # Edge F1: gold has 4 edges, pred has 2, intersection={(0,1),(1,2)}=2
    # precision=2/2=1.0, recall=2/4=0.5, F1=0.667
    print(f"Edge F1: {scores['plan']['edge_f1']:.3f} (paper-facing aligned edge score)")
    print(f"Raw Edge F1: {scores['plan']['raw_edge_f1']:.3f} (expected: 0.667)")

    print("\n" + "=" * 70)
    print("Test 6: Delta vs TaskBench semantic Node F1 comparison")
    print("=" * 70)

    # Same data but different dataset flag
    gold_compare = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Step A", "tool_id": "tool_a"},
                {"node_id": "n1", "step_index": 1, "label": "Step B", "tool_id": "tool_b"},
            ],
            "edges": []
        },
        "tool_calls": [{"tool_id": "tool_a"}, {"tool_id": "tool_b"}],
        "final_answer": {"answer_type": "none"}
    }

    pred_compare = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Step X", "tool_id": "tool_a"},  # tool matches
            ],
            "edges": []
        },
        "tool_calls": [{"tool_id": "tool_a"}],
        "final_answer": {"answer_type": "none"}
    }

    # Delta: semantic node F1 should still recover the tool_a node via tool identity
    meta_delta = {"dataset": "delta"}
    scores_delta = evaluator.evaluate_record(gold_compare, pred_compare, meta_delta)
    print(f"Delta Node F1: {scores_delta['plan']['node_f1']:.3f} (expected: ~0.667)")

    # TaskBench/UltraTool: semantic node F1 should behave similarly here
    meta_other = {"dataset": "ultratool"}
    scores_other = evaluator.evaluate_record(gold_compare, pred_compare, meta_other)
    print(f"UltraTool Node F1: {scores_other['plan']['node_f1']:.3f} (expected: ~0.667)")

    print("\n" + "=" * 70)
    print("Test 6b: GAIA semantic Node F1")
    print("=" * 70)

    gold_gaia = {
        "plan_dag": {
            "nodes": [
                {"node_id": "n0", "step_index": 0, "label": "Find the main character fish species in Finding Nemo"},
                {"node_id": "n1", "step_index": 1, "label": "Search the USGS database for nonnative sightings"},
                {"node_id": "n2", "step_index": 2, "label": "Return the ZIP codes"},
            ],
            "edges": []
        },
        "tool_calls": [],
        "final_answer": {"answer_type": "none"}
    }
    pred_gaia = {
        "plan_dag": {
            "nodes": [
                {"node_id": "p0", "step_index": 0, "label": "Identify the fish species of Nemo"},
                {"node_id": "p1", "step_index": 1, "label": "Look up nonnative locations in the USGS database"},
                {"node_id": "p2", "step_index": 2, "label": "Format the zip codes"},
                {"node_id": "p3", "step_index": 3, "label": "Write the answer nicely"},
            ],
            "edges": []
        },
        "tool_calls": [],
        "final_answer": {"answer_type": "none"}
    }
    meta_gaia = {"dataset": "gaia"}
    scores_gaia = evaluator.evaluate_record(gold_gaia, pred_gaia, meta_gaia)
    print(
        f"GAIA Node F1: {scores_gaia['plan']['node_f1']:.3f} "
        "(semantic alignment should match the first three substeps)"
    )

    print("\n" + "=" * 70)
    print("Test 7: Invented tool filtering (Option C)")
    print("=" * 70)

    # Gold has 2 tools
    gold_filter = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "web_search", "arguments": [{"name": "query", "value": "test"}]},
            {"tool_id": "file_reader", "arguments": [{"name": "path", "value": "test.txt"}]},
        ],
        "final_answer": {"answer_type": "none"},
        "tool_environment": {
            "tools": [
                {"tool_id": "web_search", "description": "Search the web"},
                {"tool_id": "file_reader", "description": "Read files"},
                {"tool_id": "calculator", "description": "Do math"},
            ]
        }
    }

    # Pred has 3 tools: 2 valid + 1 invented (not in available tools)
    pred_filter = {
        "plan_dag": {"nodes": [], "edges": []},
        "tool_calls": [
            {"tool_id": "web_search", "arguments": [{"name": "query", "value": "test"}]},
            {"tool_id": "file_reader", "arguments": [{"name": "path", "value": "wrong.txt"}]},
            {"tool_id": "invented_tool_xyz", "arguments": [{"name": "foo", "value": "bar"}]},  # INVENTED!
        ],
        "final_answer": {"answer_type": "none"}
    }

    meta_filter = {"dataset": "taskbench", "has_arguments": True}
    scores_filter = evaluator.evaluate_record(gold_filter, pred_filter, meta_filter)

    print(f"Tool F1: {scores_filter['tool']['tool_name_f1']:.3f} (expected: 1.0, invented tool filtered out)")
    print(f"Pred tool count: {scores_filter['tool']['pred_tool_count']} (expected: 2, not 3)")
    print(f"Param F1: {scores_filter['tool']['param_name_f1']:.3f} (expected: 1.0, both param names match)")
    # web_search-query-test matches, file_reader-path differs (test.txt vs wrong.txt)
    print(f"Value F1: {scores_filter['tool']['type_aware_value_f1']:.3f} (type-aware value metric)")

    # Test without filtering (no available_tools)
    meta_no_filter = {"dataset": "taskbench", "has_arguments": True}  # No available_tools
    gold_no_env = {k: v for k, v in gold_filter.items() if k != "tool_environment"}
    scores_no_filter = evaluator.evaluate_record(gold_no_env, pred_filter, meta_no_filter)
    print(f"\nWithout filtering (no available_tools):")
    print(f"Tool F1: {scores_no_filter['tool']['tool_name_f1']:.3f} (expected: 0.8, precision=2/3, recall=1.0)")
    print(f"Pred tool count: {scores_no_filter['tool']['pred_tool_count']} (expected: 3, includes invented)")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against gold standard"
    )
    parser.add_argument("--gold", type=str, help="Path to gold standard JSONL file")
    parser.add_argument("--pred", type=str, help="Path to prediction JSONL file")
    parser.add_argument("--dataset", type=str, default="gaia",
                       help="Dataset name (gaia, delta, ultratool, taskbench)")
    parser.add_argument("--use_embeddings", action="store_true",
                       help="Use embeddings for node label similarity")
    parser.add_argument("--plan_source", type=str, default="stage1",
                       choices=["stage1", "stage3", "abs", "abstract", "abs_plan_dag"],
                       help="Prediction plan field to evaluate. Default stage1 reads pred.abs_plan_dag; stage3 reads pred.plan_dag.")
    parser.add_argument("--test", action="store_true",
                       help="Run built-in unit tests instead of evaluation")

    args = parser.parse_args()

    if args.gold and args.pred:
        # Run evaluation
        run_evaluation(args.gold, args.pred, args.dataset, args.use_embeddings, args.plan_source)
    elif args.test:
        # Run unit tests
        run_tests()
    else:
        # Default: show help
        print("Usage:")
        print("  python -m src.evaluation.metrics --gold GOLD_FILE --pred PRED_FILE")
        print("  python -m src.evaluation.metrics --test  # Run unit tests")
        print("\nOptions:")
        print("  --gold       Path to gold standard JSONL file")
        print("  --pred       Path to prediction JSONL file")
        print("  --dataset    Dataset name (default: gaia)")
        print("  --test       Run built-in unit tests")
