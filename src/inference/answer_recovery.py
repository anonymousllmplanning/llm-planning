"""Bounded, gold-blind extraction of explicit final-answer values.

Never infer an answer from reasoning, fill an unfinished answer string, or use
the expected answer to select a value. A complete answer may survive a response
truncated later in its optional reasoning field; the recovery route is recorded.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple


MAX_RESPONSE_CHARS = 131072


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def _is_answer_scalar(value: Any) -> bool:
    return (
        isinstance(value, (str, int, float, bool))
        and not (isinstance(value, str) and not value.strip())
        and not (isinstance(value, float) and not math.isfinite(value))
    )


def extract_answer_scalar(payload: Any, *, max_depth: int = 4) -> Any:
    """Read explicit answer wrappers, preserving 0 and rejecting reasoning-only objects."""
    value = payload
    for _ in range(max_depth + 1):
        if _is_answer_scalar(value):
            return value
        if not isinstance(value, dict):
            return None
        if "answer" in value:
            value = value["answer"]
        elif "final_answer" in value:
            value = value["final_answer"]
        else:
            return None
    return None


def answer_payload_to_text(payload: Any) -> Tuple[str, Optional[str]]:
    """Normalize explicit scalar values without stringifying empty dictionaries."""
    value = extract_answer_scalar(payload)
    if value is None:
        return "", "no_nonempty_explicit_answer_scalar"
    if isinstance(value, str):
        return value.strip(), None
    return json.dumps(value, ensure_ascii=False, allow_nan=False), "scalar_to_text"


def _unwrap_outer_fence(raw: str) -> Tuple[str, bool]:
    text = raw.strip()
    match = re.match(r"^```(?:json)?[ \t]*\r?\n", text, re.IGNORECASE)
    if not match:
        return text, False
    text = text[match.end():]
    closing = re.search(r"\r?\n```[ \t]*$", text)
    if closing:
        text = text[:closing.start()]
    return text.strip(), True


def _complete_answer_prefix(raw: str) -> Any:
    """Read a closed scalar plus delimiter; never reconstruct answer text.

    Only the optional reasoning field may be damaged. Alternative answer keys,
    duplicate fields and additional answer objects make recovery ambiguous.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    position = 0

    def whitespace():
        nonlocal position
        while position < len(raw) and raw[position] in " \t\r\n":
            position += 1

    def symbol(expected):
        nonlocal position
        whitespace()
        if position >= len(raw) or raw[position] != expected:
            raise ValueError("Unexpected answer-prefix syntax")
        position += 1

    def read():
        nonlocal position
        whitespace()
        value, position = decoder.raw_decode(raw, position)
        return value

    symbol("{")
    first_key = read()
    symbol(":")
    if first_key == "final_answer":
        symbol("{")
        if read() != "answer":
            raise ValueError("No leading explicit answer field")
        symbol(":")
    elif first_key != "answer":
        raise ValueError("Not an explicit answer wrapper")
    value = read()
    if not _is_answer_scalar(value):
        raise ValueError("Answer is not a complete nonempty finite scalar")
    whitespace()
    if position == len(raw):
        raise ValueError("Answer has no following field delimiter")
    if raw[position] == "}":
        # Missing only the outer wrapper closing brace is unambiguous.
        position += 1
        whitespace()
        if first_key != "final_answer" or position != len(raw):
            raise ValueError("Unexpected trailing answer content")
        return value
    symbol(",")
    whitespace()
    if position == len(raw):
        return value  # closed scalar followed by a comma and EOF
    if read() != "reasoning":
        raise ValueError("Only optional reasoning may follow the answer")
    symbol(":")
    # Reasoning is neither interpreted nor reconstructed. Reject apparent new
    # fields / answer objects, including ambiguous examples in damaged text.
    remainder = raw[position:]
    if re.search(r'(?:\{|,)\s*(?<!\\)"(?:answer|final_answer|reasoning)"\s*:', remainder):
        raise ValueError("Ambiguous later answer or duplicate reasoning key")
    return value


_CANDIDATE_START = re.compile(r'\{\s*"(?:final_answer|answer)"\s*:')
_EXAMPLE_HINT = re.compile(
    r"(?:\bexamples?\b|\bsample\b|"
    r"\be\.g\.|示範|示例|範例)", re.IGNORECASE
)


def _explicit_wrapper(parsed):
    if not isinstance(parsed, dict):
        return None
    # A root carrying two alternative answer slots is never selected by order.
    if "final_answer" in parsed and "answer" in parsed:
        raise ValueError("Ambiguous root answer wrappers")
    if "final_answer" in parsed:
        final = parsed["final_answer"]
        if isinstance(final, dict) and "answer" in final and "final_answer" in final:
            raise ValueError("Ambiguous nested answer wrappers")
        return final
    if "answer" in parsed:
        return parsed
    return None


def parse_stage5_answer(raw_output: Any, *, max_chars: int = MAX_RESPONSE_CHARS) -> Dict[str, Any]:
    """Extract a unique explicit answer JSON object without consulting gold.

    A prose preamble is accepted only before a unique terminal answer object.
    Example-like preambles, duplicate keys, competing objects, unfinished answer
    scalars, nonfinite values and reasoning-only prose are rejected. Original raw
    output is always kept byte-for-byte as a Python string.
    """
    raw = raw_output if isinstance(raw_output, str) else ""
    result = {
        "final_answer": {"answer": "", "reasoning": ""},
        "raw_output": raw,
        "parsed_output": {},
        "parse_status": "unrecoverable_answer",
        "parse_error": None,
        "answer_parser_protocol": "gold-blind-explicit-answer-v2",
    }
    if len(raw) > max_chars:
        result["parse_error"] = "response_exceeds_answer_parser_limit"
        return result
    normalized, fenced = _unwrap_outer_fence(raw)
    offset = 0
    try:
        parsed = json.loads(normalized, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except _DuplicateKey:
        result["parse_error"] = "duplicate_json_key"
        return result
    except (ValueError, TypeError) as exc:
        result["parse_error"] = str(exc)
        match = _CANDIDATE_START.search(normalized)
        if not match:
            return result
        offset = match.start()
        preamble = normalized[:offset]
        if _EXAMPLE_HINT.search(preamble[-500:]):
            result["parse_error"] = "ambiguous_example_preamble"
            return result
        candidate = normalized[offset:]
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        try:
            parsed, end = decoder.raw_decode(candidate)
        except _DuplicateKey:
            result["parse_error"] = "duplicate_json_key"
            return result
        except (ValueError, TypeError):
            try:
                answer = _complete_answer_prefix(candidate)
            except (ValueError, TypeError):
                return result
            result.update({
                "final_answer": {"answer": answer, "reasoning": ""},
                "parse_status": "recovered_complete_answer_prefix_after_preamble" if offset else "recovered_complete_answer_prefix_after_fence" if fenced else "recovered_complete_answer_prefix",
                "answer_recovery_applied": True,
                "answer_json_start_offset": offset,
            })
            return result
        else:
            tail = candidate[end:].strip()
            if tail not in {"", "```"}:
                result["parse_error"] = "competing_json_or_trailing_content"
                return result
    try:
        final = _explicit_wrapper(parsed)
    except ValueError as exc:
        result["parse_error"] = str(exc)
        return result
    result["parsed_output"] = parsed if isinstance(parsed, dict) else {}
    answer = extract_answer_scalar(final)
    if answer is None:
        result["parse_status"] = "valid_json_empty_or_invalid_answer"
        result["parse_error"] = "no_nonempty_explicit_answer_scalar"
        return result
    reasoning = final.get("reasoning", "") if isinstance(final, dict) else ""
    root_wrapper = isinstance(parsed, dict) and "answer" in parsed
    status = "parsed_root_answer" if root_wrapper else "parsed_final_answer"
    if offset:
        status += "_after_preamble"
    elif fenced:
        status += "_after_outer_fence"
    result.update({
        "final_answer": {"answer": answer, "reasoning": reasoning if isinstance(reasoning, str) else ""},
        "parse_status": status,
        "parse_error": None,
    })
    if offset or root_wrapper:
        result["answer_recovery_applied"] = True
        result["answer_json_start_offset"] = offset
    return result
