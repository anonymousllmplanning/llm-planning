#!/usr/bin/env python3
"""
LLM Planning Agent Runner - Unified Multi-Stage Pipeline

ARCHITECTURE OVERVIEW
=====================
This module provides the inference pipeline for the LLM Planning framework.
It supports three backends with different trade-offs:

1. SGLang Backend (recommended):
   - JSON schema constraints via xgrammar for reliable structured output
   - Best for production experiments
   - Requires SGLang server running on localhost:30000

2. API Backend:
   - Remote inference via OpenAI-compatible APIs
   - No local GPU required
   - Models: phi4, deepseek-r1-8b, gemma3-27b, etc.

3. Local Backend (legacy):
   - Direct HuggingFace model loading
   - Uses Flash Attention 2 for efficiency
   - No JSON schema constraints

MULTI-STAGE PIPELINE
====================
Stage 1: Abstract Planning (tool-agnostic reasoning)
Stage 2: Tool Creation (evaluate tools, propose new ones)
Stage 3: Refinement (concrete plan with tool calls)
Stage 4: Execution (optional - execute tools)
Stage 5: Answer Generation (optional - synthesize final answer)

USAGE EXAMPLES
==============
    # SGLang backend (recommended)
    python -m src.inference.runner \\
        --unified_path data.jsonl \\
        --output out.jsonl \\
        --model_name qwen2.5-7b \\
        --backend sglang \\
        --sglang_url http://127.0.0.1:30000

    # API model
    python -m src.inference.runner \\
        --unified_path data.jsonl \\
        --output out.jsonl \\
        --model_name phi4 \\
        --backend api

    # Local HuggingFace model (legacy)
    python -m src.inference.runner \\
        --unified_path data.jsonl \\
        --output out.jsonl \\
        --model_name qwen2.5-7b \\
        --backend local
"""

from __future__ import annotations
import argparse
import json
import re
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple, List, Set
from tqdm import tqdm

# Note: torch and transformers are imported lazily in load_model() to allow
# API/SGLang-only usage without GPU dependencies
from json import JSONDecodeError
from src.config.models import (
    get_api_base,
    get_api_key,
    get_api_provider,
    supports_native_vision_input,
)
from src.config.zh_profile import get_system_prompt_suffix
from src.inference.gaia_utils import get_attachment_display_path, resolve_attachment_path

# Import parsing utilities from modular files
from src.inference.parsing import (
    parse_action_json,
    robust_json_parse,
    extract_json_block,
    normalize_pred_structure,
    create_empty_pred
)

# Import stop sequences and sampling params from schemas
try:
    from src.inference.schemas import (
        PLANNING_STOP_SEQUENCES,
        PLANNING_SAMPLING_PARAMS,
        get_planning_schema,
        validate_output,
        create_empty_output,
        AbstractPlan,
        ToolCreationOutput,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    PLANNING_STOP_SEQUENCES = ["Observation:", "User:", "###"]
    PLANNING_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0}

# Import multi-stage prompts and tools
try:
    from src.inference.prompts import (
        build_abstract_plan_prompt,
        build_tool_creation_prompt,
        build_refinement_prompt,
        get_answer_mode_tool_context,
        normalize_tool_environment,
    )
    from src.inference.tools import (
        execute_tool,
        normalize_submitted_answer,
        canonicalize_image_recognition_prompt_family,
    )
    MULTISTAGE_AVAILABLE = True
    MULTISTAGE_IMPORT_ERROR = None
except ImportError as e:
    MULTISTAGE_AVAILABLE = False
    MULTISTAGE_IMPORT_ERROR = e


def _openai_sampling_kwargs(model: str) -> Dict[str, float]:
    """Return sampling controls accepted by the selected OpenAI model.

    Newer OpenAI reasoning/chat models such as gpt-5.5 and o-series reject
    temperature/top_p on the Responses API. Keep explicit greedy controls for
    other providers and models where they are accepted.
    """
    m = (model or "").strip().lower()
    if m == "gpt-5" or m.startswith(("gpt-5-", "gpt-5.5", "o1", "o3", "o4")):
        return {}
    return {
        "temperature": PLANNING_SAMPLING_PARAMS["temperature"],
        "top_p": PLANNING_SAMPLING_PARAMS["top_p"],
    }


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIGS = {
    # =========================================================================
    # LOCAL MODELS (HuggingFace, run on GPU)
    # =========================================================================

    # ===== Small Models (for quick testing) =====
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "phi3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "max_new_tokens": 2048,
        "use_flash_attn": True,
        "backend": "local",
    },

    # ===== Medium Models (good balance) =====
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_new_tokens": 2048,
        "use_flash_attn": True,
        "backend": "local",
    },
    "vicuna-7b": {
        "hf_id": "lmsys/vicuna-7b-v1.5-16k",
        "max_new_tokens": 2048,
        "use_flash_attn": True,
        "backend": "local",
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 4096,
        "backend": "local",
    },
    # ===== Gemma Family (API) - Additional =====
    "gemma2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "max_new_tokens": 4096,
        "backend": "local",
    },

    # ===== Large Models (recommended for quality) =====
    "qwen2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "max_new_tokens": 4096,
        "use_flash_attn": True,
        "backend": "local",
    },
    "vicuna-13b": {
        "hf_id": "lmsys/vicuna-13b-v1.5-16k",
        "max_new_tokens": 2048,
        "use_flash_attn": True,
        "backend": "local",
    },

    # =========================================================================
    # API MODELS (OpenAI-compatible server)
    # These are recommended for testing - no local GPU needed!
    # =========================================================================

    # ===== Qwen Family (API) =====
    "qwen2.5-7b-api": {
        "api_model": "qwen2.5-7b",
        "max_new_tokens": 4096,
        "backend": "api",
    },
    "qwen3-4b": {
        "api_model": "qwen3:4b",
        "max_new_tokens": 4096,
        "backend": "api",
    },
    "qwen3-4b-thinking": {
        "api_model": "qwen3:4b-thinking",
        "max_new_tokens": 4096,
        "backend": "api",
    },

    # ===== Gemma Family (API) =====
    "gemma3-27b": {
        "api_model": "gemma3:27b",
        "max_new_tokens": 4096,
        "backend": "api",
    },
    "gemma3-vl-27b": {
        "api_model": "gemma3(vl):27b",
        "max_new_tokens": 4096,
        "backend": "api",
    },

    # ===== Phi Family (API) =====
    "phi4": {
        "api_model": "phi4",
        "max_new_tokens": 4096,
        "backend": "api",
    },
    "phi4-mini": {
        "api_model": "phi4-mini",
        "max_new_tokens": 4096,
        "backend": "api",
    },

    # ===== DeepSeek (API) =====
    "deepseek-r1-8b": {
        "api_model": "deepseek-r1:8b",
        "max_new_tokens": 4096,
        "backend": "api",
    },

    # ===== LLaMA (API) =====
    "llama3-8b": {
        "api_model": "llama3:8b",
        "max_new_tokens": 4096,
        "backend": "api",
    },

    # ===== GPT-OSS (API) =====
    "gpt-oss-20b": {
        "api_model": "gpt-oss:20b",
        "max_new_tokens": 4096,
        "backend": "api",
    },
}


# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_json(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Load records from a JSON or JSONL file.

    Automatically converts raw GAIA format to unified evaluation format
    if needed (fixes 0-score issue).
    """
    from src.inference.gaia_utils import is_gaia_format, convert_gaia_record_to_unified

    # Determine if JSON (array) or JSONL (lines)
    is_json_array = path.suffix.lower() == '.json'
    
    records_to_yield = []
    
    if is_json_array:
        with path.open(encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records_to_yield = data
            else:
                records_to_yield = [data]
    else:
        with path.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records_to_yield.append(json.loads(line))

    for record in records_to_yield:
        # Auto-convert GAIA format if detected
        if is_gaia_format(record):
            record = convert_gaia_record_to_unified(record)
        yield record


def load_model(model_name: str, cfg: Dict[str, Any]):
    """Load model with optimizations."""
    # Lazy imports for torch and transformers (only needed for local models)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = cfg["hf_id"]
    use_flash_attn = cfg.get("use_flash_attn", False)
    multi_gpu = cfg.get("multi_gpu", False)

    # Optional: require a minimum free memory (GB) for single-GPU selection
    min_free_gb = cfg.get("min_free_gb", 0.0)

    print(f"Loading model: {hf_id}")
    print(f"  Flash Attention: {use_flash_attn}")
    print(f"  Multi-GPU: {multi_gpu}")

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {name} ({mem:.1f} GB)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,  # BF16 is better for Blackwell
        "trust_remote_code": True,
    }

    # Flash Attention 2
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except ImportError:
            print("  Flash Attention not installed, using default attention")

    # Device mapping
    if multi_gpu and num_gpus > 1:
        model_kwargs["device_map"] = "auto"  # Distribute across GPUs
        print(f"  Distributing model across {num_gpus} GPUs")
    else:
        # Auto-pick the GPU with the most free VRAM
        torch.cuda.init()  # ensure CUDA context is ready
        best_gpu = 0
        best_free = -1

        print("  Selecting GPU with most free VRAM:")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_gb = total_gb - free_gb
            print(f"    GPU {i}: free={free_gb:.2f}GB used={used_gb:.2f}GB total={total_gb:.2f}GB")

            if free > best_free:
                best_free = free
                best_gpu = i

        if (best_free / (1024**3)) < min_free_gb:
            raise RuntimeError(
                f"No GPU meets min_free_gb={min_free_gb}. Best free={(best_free/(1024**3)):.2f}GB"
            )

        model_kwargs["device_map"] = {"": best_gpu}  # Single GPU (auto-picked)
        print(f"  Using single GPU (auto-picked GPU {best_gpu})")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.eval()

    return model, tokenizer


def generate_with_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    """
    Generate text using a local HuggingFace model.

    This function handles the local model inference pipeline:
    1. Apply chat template if available (for instruction-tuned models)
    2. Tokenize the prompt
    3. Run greedy decoding (temperature=0 for reproducibility)
    4. Decode only the newly generated tokens

    Args:
        model: HuggingFace model instance (already loaded to GPU)
        tokenizer: HuggingFace tokenizer instance
        prompt: User prompt / instruction text
        max_new_tokens: Maximum tokens to generate (default: 2048)

    Returns:
        Generated text string (decoded, stripped)

    Note:
        This is the legacy backend. For production use, prefer the SGLang backend
        which provides JSON schema constraints via xgrammar.
    """
    import torch  # Lazy import

    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    )
    return generated.strip()


# =============================================================================
# SGLang Backend Support (grammar-constrained decoding via xgrammar)
# =============================================================================

def generate_with_sglang(
    prompt: str,
    dataset: str,
    sglang_url: str = "http://127.0.0.1:30000",
    max_new_tokens: int = 8192,  # Increased to prevent truncation for verbose models
    grammar_mode: str = "strict_eval_schema",
    strict_parse: bool = True,
    sample_id: str = "",
    dry_run: bool = False,
    use_schema_constraint: bool = True,  # NEW: when False, skip JSON schema for comparison
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Generate text using SGLang server with optional xgrammar constraint.

    Uses JSON schema (Pydantic or eval_schemas) to enforce output format that
    directly matches what metrics.py expects.

    Args:
        prompt: Input prompt
        dataset: Dataset name (gaia, delta, taskbench, ultratool)
        sglang_url: SGLang server URL
        max_new_tokens: Maximum tokens to generate
        grammar_mode: Schema mode (currently only "strict_eval_schema")
        strict_parse: If True, raise error on parse failure
        sample_id: Sample ID for error reporting
        dry_run: If True, print request info and return empty
        use_schema_constraint: If True, enforce JSON schema via xgrammar.
                               If False, generate without constraint and use fallback parsing.
                               Useful for comparing constrained vs unconstrained outputs.

    Features:
    - Grammar-constrained decoding via xgrammar (when use_schema_constraint=True)
    - Fallback robust JSON parsing (when use_schema_constraint=False)
    - Pydantic schema validation when available
    - Strict sampling control (temperature=0, top_p=1.0)
    - Stop sequences for cleaner output

    Args:
        prompt: Input prompt
        dataset: Dataset name (gaia, delta, taskbench, ultratool)
    """
    from src.inference.sglang_client import SGLangClient, SGLangParseError

    # For SGLang with xgrammar, prefer the simpler eval_schemas over Pydantic
    # Pydantic schemas can be too complex and cause repetition issues with xgrammar
    json_schema = None
    if use_schema_constraint:
        from src.inference.eval_schemas import get_eval_schema
        json_schema = get_eval_schema(dataset)
        print(f"[INFO] Using eval_schema for SGLang xgrammar (dataset: {dataset})")

    if dry_run:
        import json as json_module
        print("=" * 60)
        print("[DRY RUN] SGLang Request Info")
        print("=" * 60)
        print(f"Server URL: {sglang_url}")
        print(f"Dataset: {dataset}")
        print(f"Grammar Mode: {grammar_mode}")
        print(f"Max Tokens: {max_new_tokens}")
        print(f"Strict Parse: {strict_parse}")
        print(f"Sample ID: {sample_id}")
        print("-" * 60)
        print("PROMPT (first 2000 chars):")
        print(prompt[:2000])
        print("-" * 60)
        print("JSON SCHEMA:")
        print(json_module.dumps(json_schema, indent=2))
        print("=" * 60)
        # Return empty for dry run
        return "", None

    # Create client and generate
    client = SGLangClient(base_url=sglang_url, timeout=180.0)

    try:
        parsed, raw_text = client.generate_and_validate(
            prompt=prompt,
            json_schema=json_schema,
            sample_id=sample_id,
            max_tokens=max_new_tokens,
            strict_parse=strict_parse,
        )

        # Additional validation using Pydantic if available
        if SCHEMAS_AVAILABLE:
            is_valid, errors = validate_output(parsed, dataset)
        else:
            from src.inference.eval_schemas import validate_schema_compliance
            is_valid, errors = validate_schema_compliance(parsed, dataset)
        if not is_valid:
            print(f"[WARN] Schema validation warnings for sample {sample_id}: {errors}")

        return raw_text, parsed

    except SGLangParseError as e:
        # Fail loudly as required, but return raw_text for debugging
        print(f"\n{'='*60}")
        print("[FATAL] SGLang Parse Error - Grammar-constrained output failed to parse!")
        print(f"{'='*60}")
        print(str(e))
        print(f"{'='*60}\n")
        # Re-raise with raw_response attached for debugging
        # The caller can extract e.raw_response to save for debugging
        raise


def _encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    import base64
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to encode image {image_path}: {e}")
        return ""


_IMAGE_ATTACHMENT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _image_media_type(image_path: str, file_name: str = "") -> str:
    """Return a conservative media type for a provider image attachment."""
    import mimetypes

    candidate = image_path or file_name or ""
    media_type, _ = mimetypes.guess_type(candidate)
    if media_type and media_type.startswith("image/"):
        return media_type
    lower = candidate.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _provider_supports_vision(api_provider: str, api_model: str) -> bool:
    """Whether this provider/model should receive image attachments natively."""
    provider = (api_provider or "").strip().lower()
    model = (api_model or "").strip().lower()
    if supports_native_vision_input(api_model):
        return True
    if provider == "anthropic" and model.startswith("claude-"):
        return True
    if provider == "gemini" and model.startswith("gemini-"):
        return True
    if provider == "openai" and (
        model.startswith("gpt-4")
        or model.startswith("gpt-5")
        or "vision" in model
        or "multimodal" in model
    ):
        return True
    return False


def _collect_image_attachments(attachments: Optional[List[Dict]]) -> List[Dict[str, str]]:
    """Resolve and encode image attachments once for all remote providers."""
    image_payloads: List[Dict[str, str]] = []
    for att in attachments or []:
        fpath = att.get("file_path", "")
        fname_raw = att.get("file_name", "")
        source_hint = (fname_raw or fpath or "").lower()
        if not source_hint.endswith(_IMAGE_ATTACHMENT_EXTENSIONS):
            continue
        resolved_fpath = resolve_attachment_path(fpath, fname_raw)
        image_path = resolved_fpath or fpath
        b64_image = _encode_image(image_path)
        if not b64_image:
            continue
        print(f"[INFO] Attaching image to VLM request: {image_path}")
        image_payloads.append({
            "path": image_path,
            "media_type": _image_media_type(image_path, fname_raw),
            "data": b64_image,
        })
    return image_payloads


def generate_with_api(
    api_model: str,
    prompt: str,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    dataset: str = "unknown",
    use_guided_json: bool = False,
    sglang_api_url: Optional[str] = None,
    attachments: Optional[List[Dict]] = None,
    allow_reasoning_content_fallback: bool = True,
    require_json_like_reasoning_fallback: bool = False,
) -> str:
    """
    Generate text using an OpenAI-compatible API or SGLang-backed API.

    Features:
    - Model-specific system prompts for better JSON output
    - Special handling for DeepSeek R1 models (disable thinking mode)
    - Retry logic with exponential backoff for transient errors
    - SGLang guided decoding via extra_body when available
    - Strict sampling: temperature=0, top_p=1.0
    - Stop sequences for cleaner output
    - Multimodal support for Llama-3.2-Vision and Phi-4-Multimodal

    Args:
        api_model: Model name (e.g., "qwen3:4b", "gpt-oss-20b-sglang")
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        retry_delay: Initial delay between retries
        dataset: Dataset name for schema selection
        use_guided_json: If True, use guided_json in extra_body (SGLang-backed)
        sglang_api_url: Optional URL for SGLang-backed API endpoint
        attachments: List of attachments (from query.attachments) for VLM
        allow_reasoning_content_fallback: Whether to use reasoning_content when
            message.content is empty
        require_json_like_reasoning_fallback: If True, only use reasoning_content
            when it appears to start with a JSON object

    Returns:
        Generated text string
    """
    import time
    import requests
    from openai import OpenAI

    # Detect SGLang-backed models by name pattern
    is_sglang_backed = (
        "-sglang" in api_model.lower() or
        use_guided_json or
        sglang_api_url is not None
    )

    # Use appropriate API endpoint
    api_provider = "openai_compatible" if sglang_api_url else get_api_provider()
    api_base = sglang_api_url if sglang_api_url else get_api_base()
    api_key = get_api_key() if not sglang_api_url else "EMPTY"  # SGLang doesn't require key

    # Dynamic timeout: allow override via env var (for audio tasks)
    api_timeout = float(os.getenv("LLM_API_TIMEOUT", "180.0"))
    client = None
    if api_provider in {"openai_compatible", "openai"}:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            default_headers={"User-Agent": "curl/7.68.0"},
            timeout=api_timeout,  # Default 3min, can be overridden for audio tasks
        )

    # Unified system prompt for ALL models (fair comparison)
    # Using the same prompt ensures consistent evaluation across models
    system_prompt = """[JSON OUTPUT REQUIRED]

You are a JSON-only assistant. Your ENTIRE response must be a valid JSON object.

CRITICAL RULES:
1. Response STARTS with { (the very first character)
2. Response ENDS with } (the very last character)
3. NO text before the opening brace
4. NO text after the closing brace
5. NO markdown formatting (```)
6. NO explanations or commentary
7. NO thinking tags like <think> or <analysis>
8. Valid JSON syntax only

Output a single JSON object now:"""
    system_prompt_suffix = get_system_prompt_suffix(dataset)
    if system_prompt_suffix:
        system_prompt = f"{system_prompt}\n\n{system_prompt_suffix}"

    image_payloads: List[Dict[str, str]] = []
    if attachments and _provider_supports_vision(api_provider, api_model):
        image_payloads = _collect_image_attachments(attachments)

    # Build messages with system prompt for better JSON compliance
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Multimodal Logic
    if image_payloads:
        # Construct multimodal user message
        content_parts = [{"type": "text", "text": prompt}]
        for image in image_payloads:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['media_type']};base64,{image['data']}"
                }
            })
        
        messages.append({"role": "user", "content": content_parts})
    else:
        # Standard text-only
        messages.append({"role": "user", "content": prompt})

    # Build extra_body for SGLang-backed models with guided decoding
    extra_body = None
    if is_sglang_backed and SCHEMAS_AVAILABLE:
        try:
            json_schema = get_planning_schema(dataset)
            extra_body = {
                "guided_json": json_schema,
                "guided_decoding_backend": "xgrammar",
            }
            print(f"[INFO] Using guided_json for SGLang-backed model: {api_model}")
        except Exception as e:
            print(f"[WARN] Failed to get schema for guided decoding: {e}")
            extra_body = None

    def _clean_model_text(raw_text: str) -> str:
        raw_text = raw_text.strip() if raw_text else ""
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        raw_text = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL)
        return raw_text.strip()

    def _call_anthropic() -> str:
        url = f"{api_base.rstrip('/')}/v1/messages"
        user_content: Any
        if image_payloads:
            user_content = [{"type": "text", "text": prompt}]
            for image in image_payloads:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image["media_type"],
                        "data": image["data"],
                    },
                })
        else:
            user_content = prompt
        payload = {
            "model": api_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_new_tokens,
            "temperature": PLANNING_SAMPLING_PARAMS["temperature"],
            "top_p": PLANNING_SAMPLING_PARAMS["top_p"],
        }
        response = requests.post(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "user-agent": "curl/7.68.0",
            },
            json=payload,
            timeout=api_timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return _clean_model_text("".join(parts))

    def _call_gemini() -> str:
        url = (
            f"{api_base.rstrip('/')}/v1beta/models/{api_model}:generateContent"
            f"?key={api_key}"
        )
        user_parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image in image_payloads:
            user_parts.append({
                "inline_data": {
                    "mime_type": image["media_type"],
                    "data": image["data"],
                }
            })
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": PLANNING_SAMPLING_PARAMS["temperature"],
                "topP": PLANNING_SAMPLING_PARAMS["top_p"],
                "maxOutputTokens": max_new_tokens,
                "stopSequences": PLANNING_STOP_SEQUENCES[:3],
            },
        }
        response = requests.post(
            url,
            headers={
                "content-type": "application/json",
                "user-agent": "curl/7.68.0",
            },
            json=payload,
            timeout=api_timeout,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", []) or []
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return _clean_model_text("".join(text_parts))

    def _call_openai_responses() -> str:
        user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image in image_payloads:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:{image['media_type']};base64,{image['data']}",
            })
        response_kwargs = {
            "model": api_model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            "max_output_tokens": max_new_tokens,
        }
        response_kwargs.update(_openai_sampling_kwargs(api_model))
        response = client.responses.create(**response_kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return _clean_model_text(output_text)

        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text is None and isinstance(content, dict):
                    text = content.get("text")
                if text:
                    parts.append(text)
        return _clean_model_text("".join(parts))

    last_error = None
    for attempt in range(max_retries):
        try:
            if api_provider == "anthropic":
                return _call_anthropic()
            if api_provider == "gemini":
                return _call_gemini()
            if api_provider == "openai":
                return _call_openai_responses()

            # OpenAI-compatible / OpenAI path
            api_kwargs = {
                "model": api_model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": PLANNING_SAMPLING_PARAMS["temperature"],  # 0 for greedy
                "top_p": PLANNING_SAMPLING_PARAMS["top_p"],  # 1.0
                "stop": PLANNING_STOP_SEQUENCES[:3],  # Use first 3 stop sequences
            }

            if extra_body:
                api_kwargs["extra_body"] = extra_body

            completion = client.chat.completions.create(**api_kwargs)
            message = completion.choices[0].message
            raw_response = message.content or ""

            reasoning_content = getattr(message, "reasoning_content", None)
            if not raw_response.strip() and reasoning_content:
                reasoning_text = reasoning_content.strip()
                reasoning_json_block = extract_json_block(reasoning_text) if reasoning_text else ""
                if (
                    allow_reasoning_content_fallback and
                    (
                        not require_json_like_reasoning_fallback or
                        reasoning_text.startswith("{") or
                        bool(reasoning_json_block)
                    )
                ):
                    print(f"[DEBUG] Using reasoning_content for {api_model} (content was empty)")
                    raw_response = reasoning_json_block or reasoning_content
                else:
                    print(f"[DEBUG] Ignoring reasoning_content for {api_model} because it does not look like action JSON")

            return _clean_model_text(raw_response)

        except KeyboardInterrupt:
            # Allow manual interruption
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"[WARN] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[WARN] Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] API call failed after {max_retries} attempts for {api_model}: {last_error}")

    return ""


# ============================================================================
# Multi-Stage Pipeline Functions
# ============================================================================
# The following functions implement the 3-5 stage planning pipeline:
# Stage 1: Abstract Planning (tool-agnostic)
# Stage 2: Tool Creation (evaluate and propose new tools)
# Stage 3: Refinement (concrete planning with tools)
# Stage 4-5: Execution + Answer Generation (optional)


def validate_and_fix_tool_ids(final_pred: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
    """
    Validate tool_ids in plan and auto-fix common mistakes.

    Common fixes:
    - "wikipedia_search" -> "web_search" (user should then use web_browser for Wikipedia URL)
    - "google_search" -> "web_search"
    - Invalid tool_id -> suggest closest match or raise error

    Args:
        final_pred: The prediction dict with plan_dag and tool_calls
        available_tools: List of valid tool names

    Returns:
        Modified prediction dict with fixed tool_ids

    Raises:
        ValueError: If tool_id cannot be fixed and no close match found
    """
    from difflib import get_close_matches

    fixed_pred = final_pred.copy()
    modified = False

    # Fix tool_ids in plan_dag nodes
    plan_dag = fixed_pred.get("plan_dag", {})
    if "nodes" in plan_dag:
        for node in plan_dag["nodes"]:
            tool_id = node.get("tool_id")

            if not tool_id or tool_id in available_tools:
                continue

            # Try to auto-fix common mistakes
            original_tool_id = tool_id
            if "wikipedia" in tool_id.lower():
                tool_id = "web_search"
                print(f"[FIX] Replacing '{original_tool_id}' with 'web_search' in plan node {node.get('node_id')}")
                node["tool_id"] = tool_id
                modified = True
            elif "google" in tool_id.lower() and "search" in tool_id.lower():
                tool_id = "web_search"
                print(f"[FIX] Replacing '{original_tool_id}' with 'web_search' in plan node {node.get('node_id')}")
                node["tool_id"] = tool_id
                modified = True
            else:
                # Find closest match
                matches = get_close_matches(tool_id, available_tools, n=1, cutoff=0.6)
                if matches:
                    suggestion = matches[0]
                    print(f"[FIX] Invalid tool '{tool_id}' in node {node.get('node_id')}, replacing with '{suggestion}'")
                    node["tool_id"] = suggestion
                    modified = True
                else:
                    print(f"[ERROR] Invalid tool_id '{tool_id}' in node {node.get('node_id')}. Available: {available_tools}")
                    # Don't raise error, just log warning and keep original
                    # This allows execution to proceed and show proper error message

    # Fix tool_ids in tool_calls
    if "tool_calls" in fixed_pred:
        for tc in fixed_pred["tool_calls"]:
            tool_id = tc.get("tool_id")

            if not tool_id or tool_id in available_tools:
                continue

            original_tool_id = tool_id
            if "wikipedia" in tool_id.lower():
                tool_id = "web_search"
                print(f"[FIX] Replacing '{original_tool_id}' with 'web_search' in tool call")
                tc["tool_id"] = tool_id
                modified = True
            elif "google" in tool_id.lower() and "search" in tool_id.lower():
                tool_id = "web_search"
                print(f"[FIX] Replacing '{original_tool_id}' with 'web_search' in tool call")
                tc["tool_id"] = tool_id
                modified = True
            else:
                matches = get_close_matches(tool_id, available_tools, n=1, cutoff=0.6)
                if matches:
                    suggestion = matches[0]
                    print(f"[FIX] Invalid tool '{tool_id}' in tool call, replacing with '{suggestion}'")
                    tc["tool_id"] = suggestion
                    modified = True

    if modified:
        print(f"[INFO] Tool validation completed with {sum(1 for _ in [True])} fixes applied")

    return fixed_pred


def resolve_variables(value: Any, var_store: Dict[str, str]) -> Any:
    """
    Recursively resolve variable references like <n0>, <n1> in value.

    Handles:
    - String: "<n0>" -> var_store["<n0>"]
    - String with refs: "total = <n0>" -> "total = 1002"
    - Dict/List: recursively process

    Args:
        value: The value to process (can be str, dict, list, or other)
        var_store: Dictionary mapping variable names to their actual values

    Returns:
        Value with all variable references resolved
    """
    import re

    if isinstance(value, str):
        # Find all <nX> patterns
        pattern = r'<n\d+>'
        matches = re.findall(pattern, value)

        if not matches:
            return value

        # If entire string is just "<n0>", replace directly with actual value
        if len(matches) == 1 and value.strip() == matches[0]:
            var_name = matches[0]
            if var_name in var_store:
                return var_store[var_name]
            else:
                print(f"[WARN] Variable {var_name} not found in store, keeping as-is")
                return value

        # Otherwise, substitute all occurrences in string
        result = value
        for var_name in matches:
            if var_name in var_store:
                # Convert to string for substitution
                result = result.replace(var_name, str(var_store[var_name]))
            else:
                print(f"[WARN] Variable {var_name} not found in store, keeping as-is")
        return result

    elif isinstance(value, dict):
        return {k: resolve_variables(v, var_store) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_variables(item, var_store) for item in value]

    else:
        # For other types (int, float, bool, None), return as-is
        return value


# Tool signature mapping for argument validation and guidance
TOOL_SIGNATURES = {
    "calculator": {"expression": "str - mathematical expression to evaluate"},
    "python_executor": {"code": "str - Python code to execute", "output_dir": "str (optional) - output directory"},
    "web_search": {"query": "str - search query", "engine": "str (optional) - search engine", "max_results": "int (optional) - max results"},
    "web_browser": {"url": "str - URL to visit", "action": "str (optional) - browser action"},
    "excel_reader": {"file_path": "str - path to Excel/CSV file", "sheet": "str (optional) - sheet name", "query": "str (optional) - query"},
    "file_reader": {"file_path": "str - path to file"},
    "pdf_reader": {"file_path": "str - path to PDF file", "page": "int (optional) - specific page number"},
    "audio_transcription": {"file_path": "str - path to audio file", "language": "str (optional) - short language code like en or zh"},
    "pptx_reader": {"file_path": "str - path to PowerPoint file", "slide_number": "int (optional) - specific slide number"},
    "zip_extractor": {"file_path": "str - path to ZIP file", "extract_to": "str (optional) - extraction directory", "output_dir": "str (optional) - extraction directory"},
    "download_file": {"url": "str - URL to download", "save_path": "str (optional) - explicit save path", "output_dir": "str (optional) - download directory"},
    "submit_final_answer": {"answer": "str - final answer", "answer_type": "str (optional) - answer type"},
    "image_recognition": {"file_path": "str - path to image", "task": "str (optional) - describe/extract_text/extract_numbers/extract_by_color/chess/music_sheet/geometry/fractions", "custom_prompt": "str (optional) - custom analysis prompt for specialized tasks"},
    "video_analysis": {"video_path": "str - path to video", "task": "str (optional) - analysis task"},
    "reasoning": {"problem": "str - problem statement for internal reasoning"},
    "code_interpreter": {"code": "str - code to interpret", "language": "str (optional) - language name", "action": "str (optional) - execute or analyze"},
}


def get_tool_signature_string(tool_id: str) -> str:
    """Get human-readable tool signature for prompting."""
    if tool_id not in TOOL_SIGNATURES:
        return f"{tool_id}(...) - signature unknown"

    params = TOOL_SIGNATURES[tool_id]
    param_strings = [f"{name}: {desc}" for name, desc in params.items()]
    return f"{tool_id}({', '.join(param_strings)})"


def _first_present_arg(model_args: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[str], Any]:
    """Return the first non-empty argument value for any alias in ``keys``."""
    for key in keys:
        if key in model_args:
            value = model_args.get(key)
            if value is not None and not (isinstance(value, str) and not value.strip()):
                return key, value
    return None, None


def _looks_like_url(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.match(r"^(https?://|ftp://|www\.)", text, flags=re.IGNORECASE))


def normalize_tool_call_for_signature(
    tool_id: str,
    model_args: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Conservatively normalize common tool-call argument aliases before execution.

    This rescues semantically clear calls such as ``python_executor(script=...)``
    or ``reasoning(query=...)`` while leaving ambiguous calls for the existing
    model-refinement path.
    """
    if not isinstance(model_args, dict):
        model_args = {}

    original_tool_id = tool_id
    args = dict(model_args)
    notes: List[str] = []

    if tool_id == "python_executor":
        _, code = _first_present_arg(args, ("code", "script", "code_text", "python", "python_code", "program", "source"))
        if code is not None:
            normalized = {"code": str(code)}
            timeout = args.get("timeout_seconds") or args.get("timeout")
            if timeout is not None:
                normalized["timeout_seconds"] = timeout
            args = normalized
            notes.append("mapped_python_code_alias")

    elif tool_id == "code_interpreter":
        _, code = _first_present_arg(args, ("code", "script", "code_text", "program", "source"))
        if code is not None:
            normalized = {"code": str(code)}
            for optional_key in ("language", "action"):
                if optional_key in args:
                    normalized[optional_key] = args[optional_key]
            args = normalized
            notes.append("mapped_code_interpreter_code_alias")

    elif tool_id == "reasoning":
        _, problem = _first_present_arg(args, ("problem", "question", "query", "prompt", "task", "input", "text"))
        if problem is not None:
            args = {"problem": str(problem)}
            notes.append("mapped_reasoning_problem_alias")

    elif tool_id == "submit_final_answer":
        _, answer = _first_present_arg(args, ("answer", "final_answer", "response", "result", "output", "value"))
        if answer is not None:
            normalized = {"answer": answer}
            if "answer_type" in args:
                normalized["answer_type"] = args["answer_type"]
            args = normalized
            notes.append("mapped_submit_answer_alias")

    elif tool_id == "web_search":
        _, query = _first_present_arg(args, ("query", "search_query", "q", "question", "prompt", "keywords"))
        if query is not None:
            normalized = {"query": str(query)}
            for optional_key in ("engine", "max_results"):
                if optional_key in args:
                    normalized[optional_key] = args[optional_key]
            args = normalized
            notes.append("mapped_web_search_query_alias")

    elif tool_id == "web_browser":
        query_key, query = _first_present_arg(args, ("query", "search_query", "q", "keywords"))
        _, url = _first_present_arg(args, ("url", "link", "href", "website", "page"))
        if url is None and query is not None:
            if _looks_like_url(query):
                url = query
                notes.append("mapped_web_browser_query_url")
            else:
                tool_id = "web_search"
                normalized = {"query": str(query)}
                if "max_results" in args:
                    normalized["max_results"] = args["max_results"]
                args = normalized
                notes.append("rerouted_web_browser_query_to_web_search")
        if tool_id == "web_browser" and url is not None:
            normalized = {"url": str(url)}
            action = args.get("action") or args.get("find_text") or args.get("operation")
            if action is not None:
                normalized["action"] = str(action)
                if query_key == "find_text":
                    notes.append("mapped_find_text_to_action")
            args = normalized

    elif tool_id in {"file_reader", "pdf_reader", "excel_reader", "pptx_reader", "audio_transcription"}:
        _, file_path = _first_present_arg(args, ("file_path", "path", "filename", "file", "audio_path", "document_path", "input_file"))
        if file_path is not None:
            normalized = {"file_path": str(file_path)}
            for optional_key in ("sheet", "query", "page", "slide_number", "language"):
                if optional_key in args:
                    normalized[optional_key] = args[optional_key]
            args = normalized
            notes.append("mapped_file_path_alias")

    elif tool_id == "download_file":
        _, url = _first_present_arg(args, ("url", "link", "href", "download_url"))
        if url is not None:
            normalized = {"url": str(url)}
            for optional_key in ("save_path", "output_dir"):
                if optional_key in args:
                    normalized[optional_key] = args[optional_key]
            args = normalized
            notes.append("mapped_download_url_alias")

    elif tool_id == "image_recognition":
        _, file_path = _first_present_arg(args, ("file_path", "path", "filename", "file", "image_path", "image", "input_file"))
        if file_path is not None:
            normalized = {"file_path": str(file_path)}
            task = args.get("task") or args.get("mode") or args.get("analysis_type")
            if task is not None:
                normalized["task"] = str(task)
            custom_prompt = args.get("custom_prompt") or args.get("prompt") or args.get("question") or args.get("query")
            if custom_prompt is not None:
                normalized["custom_prompt"] = str(custom_prompt)
            args = normalized
            notes.append("mapped_image_argument_alias")

    elif tool_id == "calculator":
        _, expression = _first_present_arg(args, ("expression", "expr", "formula", "calculation", "query"))
        if expression is not None:
            args = {"expression": str(expression)}
            notes.append("mapped_calculator_expression_alias")

    if tool_id in TOOL_SIGNATURES:
        args = map_arguments_to_tool_signature(tool_id, args)

    if notes or tool_id != original_tool_id:
        note = "; ".join(notes + ([f"tool_id:{original_tool_id}->{tool_id}"] if tool_id != original_tool_id else []))
        return tool_id, args, note
    return tool_id, args, None


def _extract_tool_environment(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize the per-record tool environment into a flat list of tool specs."""
    return normalize_tool_environment(record)


def _normalize_created_tools(new_tools: Any) -> List[Dict[str, Any]]:
    """Keep only well-formed Stage-2 tool definitions."""
    if not isinstance(new_tools, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for tool in new_tools:
        if not isinstance(tool, dict):
            continue
        tool_id = str(tool.get("tool_id") or tool.get("name") or "").strip()
        if not tool_id or tool_id in seen:
            continue
        arguments = tool.get("arguments") or tool.get("arguments_schema") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        normalized.append({
            **tool,
            "tool_id": tool_id,
            "description": str(tool.get("description") or ""),
            "arguments": arguments,
        })
        seen.add(tool_id)
    return normalized


def _format_schema_param(param: Dict[str, Any]) -> str:
    """Render one dataset tool-schema parameter for prompt display."""
    name = str(param.get("name", "arg"))
    typ = str(param.get("type", "string"))
    required = bool(param.get("required", False))
    suffix = "" if required else " (optional)"
    return f"{name}: {typ}{suffix}"


def _get_stage4_visible_tool_specs(
    record: Dict[str, Any],
    new_tools: Optional[List[Dict[str, Any]]] = None,
    tool_scope: str = "record",
) -> List[Dict[str, Any]]:
    """
    Return the actual tool specs exposed during Stage 4.

    We intentionally anchor this to the record's tool_environment and then
    intersect with executable runtime tools so that the benchmark annotation,
    prompt exposure, and execution-time validation all share the same tool
    universe.
    """
    context = get_answer_mode_tool_context(record, new_tools, tool_scope=tool_scope)
    return context.get("tools", [])


def _build_stage4_tool_prompt_lines(
    record: Dict[str, Any],
    new_tools: Optional[List[Dict[str, Any]]] = None,
    tool_scope: str = "record",
) -> List[str]:
    """Format the Stage 4 visible tool list directly from tool_environment."""
    lines: List[str] = []
    for tool in _get_stage4_visible_tool_specs(record, new_tools, tool_scope=tool_scope):
        tool_id = tool.get("tool_id", "")
        schema = tool.get("arguments_schema") or []
        if isinstance(schema, list) and schema:
            signature = ", ".join(_format_schema_param(param) for param in schema)
            line = f"- {tool_id}({signature})"
        else:
            line = f"- {get_tool_signature_string(tool_id)}"

        description = (tool.get("description") or "").strip().rstrip(".")
        if description:
            line += f" - {description}"
        lines.append(line)

        if tool_id == "image_recognition":
            lines.extend([
                "  * task=\"describe\" - General image description",
                "  * task=\"extract_text\" - OCR text extraction",
                "  * task=\"extract_numbers\" - Extract numbers with their colors",
                "  * task=\"extract_by_color\" - Organize content by color",
                "  * task=\"chess\" - Analyze chess board positions",
                "  * task=\"music_sheet\" - Analyze music notation (bass/treble clef)",
                "  * task=\"geometry\" - Analyze geometric shapes and measurements",
                "  * task=\"fractions\" - Analyze math worksheets",
                "  * custom_prompt=\"...\" - Use custom prompt for specialized analysis",
            ])
        elif tool_id == "audio_transcription":
            lines.append(
                "  * Use short language codes like \"en\" or \"zh\" if you pass the optional language argument."
            )
        elif tool_id == "submit_final_answer":
            lines.append("  * Use this only when you have a clear final answer candidate.")

    return lines


def map_arguments_to_tool_signature(tool_id: str, model_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map model's argument names to the tool's expected parameter names.

    This handles cases where the model provides semantically correct values
    but with different parameter names.

    Args:
        tool_id: The tool being called
        model_args: Arguments provided by the model (may have wrong names)

    Returns:
        Dictionary with corrected parameter names matching tool signature
    """
    if tool_id not in TOOL_SIGNATURES:
        return model_args

    expected_params = list(TOOL_SIGNATURES[tool_id].keys())

    # If model args already match, return as-is
    if all(k in expected_params for k in model_args.keys()):
        return model_args

    # Special handling for single-parameter tools
    if len(expected_params) == 1 or (len(expected_params) == 2 and "output_dir" in expected_params):
        main_param = expected_params[0]

        # For calculator: if we have numeric values, construct an expression
        if tool_id == "calculator" and main_param == "expression":
            values = list(model_args.values())
            if len(values) == 1:
                return {"expression": str(values[0])}
            elif len(values) >= 2:
                # Try to infer operation - for now, just use division as common case
                # User can improve this heuristic based on their needs
                expr = f"{values[0]} / {values[1]}"
                print(f"  [INFO] Constructed calculator expression: {expr}")
                return {"expression": expr}

        # For python_executor: if we have variable assignments, construct code
        elif tool_id == "python_executor" and main_param == "code":
            # Build simple Python code from key-value pairs
            code_lines = [f"{k} = {repr(v)}" for k, v in model_args.items()]
            code = "\n".join(code_lines)
            print(f"  [INFO] Constructed python code: {code}")
            return {"code": code}

        # For other single-param tools, use first value
        else:
            first_value = next(iter(model_args.values()))
            return {main_param: first_value}

    # For multi-parameter tools, try fuzzy matching
    mapped = {}
    for expected_param in expected_params:
        # Try exact match first
        if expected_param in model_args:
            mapped[expected_param] = model_args[expected_param]
            continue

        # Try fuzzy match (e.g., "search_query" -> "query")
        for model_key, model_value in model_args.items():
            if expected_param in model_key.lower() or model_key.lower() in expected_param:
                mapped[expected_param] = model_value
                break

    return mapped if mapped else model_args


def _stringify_answer_payload(answer: Any) -> Tuple[str, Optional[str]]:
    """Normalize nested answer payloads into a stable string representation."""
    normalized_answer, note = normalize_submitted_answer(answer)
    if normalized_answer is None:
        return "", note
    if isinstance(normalized_answer, (dict, list)):
        try:
            return json.dumps(normalized_answer, ensure_ascii=False), note
        except Exception:
            return str(normalized_answer), note
    return str(normalized_answer).strip(), note


def _extract_submit_answer_from_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> Tuple[str, bool, Optional[int], Optional[str]]:
    """
    Extract the latest non-empty submit_final_answer payload from executed tool calls.

    Returns:
        tuple:
            - submit_answer: normalized answer string (may be empty)
            - submit_tool_present: whether submit_final_answer was called at all
            - submit_call_index: call_index / turn of the latest submit call if present
            - normalization_note: note returned by normalize_submitted_answer
    """
    submit_tool_present = False
    for call in reversed(tool_calls or []):
        if not isinstance(call, dict) or call.get("tool_id") != "submit_final_answer":
            continue
        submit_tool_present = True
        for arg in call.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("name") == "answer":
                answer, note = _stringify_answer_payload(arg.get("value"))
                return answer, True, call.get("call_index"), note
        return "", True, call.get("call_index"), None
    return "", submit_tool_present, None, None


def _build_answer_handoff_record(
    actual_tool_calls: List[Dict[str, Any]],
    final_answer_obj: Dict[str, Any],
    stage5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize how the run handed off its final answer for downstream evaluation."""
    native_submit_answer, submit_tool_present, submit_call_index, submit_note = _extract_submit_answer_from_tool_calls(
        actual_tool_calls
    )
    stage5_answer, stage5_note = _stringify_answer_payload(final_answer_obj)

    native_submit_present = bool(native_submit_answer)
    stage5_answer_present = bool(stage5_answer)
    final_submit_same = (
        native_submit_present and stage5_answer_present and native_submit_answer == stage5_answer
    )

    if native_submit_present:
        status = "native_submit_consistent" if (not stage5_answer_present or final_submit_same) else "native_submit_stage5_disagree"
        source = "native_submit_tool"
        effective_submit_answer = native_submit_answer
        if status == "native_submit_stage5_disagree":
            reason = (
                "Stage 4 submit_final_answer provided a non-empty answer, but it differs from the Stage 5 final_answer."
            )
        else:
            reason = "Stage 4 submit_final_answer provided the canonical final answer."
    elif submit_tool_present and stage5_answer_present:
        status = "submit_tool_empty_stage5_backfill"
        source = "stage5_final_answer"
        effective_submit_answer = stage5_answer
        reason = (
            "A submit_final_answer call was present but its answer argument was empty after normalization; "
            "using the Stage 5 final_answer as the effective final handoff."
        )
    elif stage5_answer_present:
        status = "missing_submit_tool_stage5_backfill"
        source = "stage5_final_answer"
        effective_submit_answer = stage5_answer
        reason = (
            "No Stage 4 submit_final_answer call was recorded; using the Stage 5 final_answer as the effective final handoff."
        )
    elif submit_tool_present:
        status = "submit_tool_empty_and_stage5_missing"
        source = "missing"
        effective_submit_answer = ""
        reason = (
            "A submit_final_answer call was present, but both the submit payload and the Stage 5 final_answer were empty after normalization."
        )
    else:
        status = "missing_submit_and_stage5_answer"
        source = "missing"
        effective_submit_answer = ""
        reason = "Neither Stage 4 submit_final_answer nor Stage 5 final_answer produced a non-empty final answer."

    return {
        "status": status,
        "source": source,
        "reason": reason,
        "effective_submit_answer": effective_submit_answer,
        "native_submit_answer": native_submit_answer,
        "stage5_answer": stage5_answer,
        "submit_tool_present": submit_tool_present,
        "submit_call_index": submit_call_index,
        "submit_tool_answer_present": native_submit_present,
        "stage5_answer_present": stage5_answer_present,
        "submit_stage5_same": final_submit_same if (native_submit_present and stage5_answer_present) else None,
        "submit_normalization_note": submit_note,
        "stage5_normalization_note": stage5_note,
        "stage5_parse_status": stage5_result.get("parse_status"),
        "stage5_parse_error": stage5_result.get("parse_error"),
    }


def run_stage_4_iterative_execution(
    query_record: Dict[str, Any],
    final_pred: Dict[str, Any],
    new_tools: Optional[List[Dict[str, Any]]],
    args,
    model,
    tokenizer,
    output_dir: str = "",
    debug: bool = False,
    max_turns: int = 15
) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 4: Iterative Execution (ReAct-style for answer mode).

    Uses a true ReAct loop where the model decides each next step based on observations.
    Not bound by the initial plan length.

    Args:
        max_turns: Maximum number of tool execution turns (default: 15)
        debug: If True, adds ipdb breakpoint at each turn

    Returns:
        tuple: (observations, actual_tool_calls, execution_trace)
            - observations: List of tool outputs
            - actual_tool_calls: List of actually executed tool calls with concrete arguments
            - execution_trace: Raw Stage 4 decision / execution stream for debugging
    """
    observations = []
    actual_tool_calls = []

    # Determine attachments directory
    attachments = query_record.get("query", {}).get("attachments", [])
    attachments_dir = ""
    if attachments:
        first_path = resolve_attachment_path(
            attachments[0].get("file_path"),
            attachments[0].get("file_name"),
            query_record,
        )
        if first_path:
            attachments_dir = os.path.dirname(first_path)

    sample_id = query_record.get("meta", {}).get("id", "unknown")
    user_query = query_record.get("query", {}).get("user_query", "")
    text_only_task = not attachments
    known_stage4_tools = set(TOOL_SIGNATURES.keys()) | {"submit_final_answer"}
    visible_stage4_tools = _get_stage4_visible_tool_specs(
        query_record,
        new_tools,
        tool_scope=getattr(args, "tool_scope", "record"),
    )
    visible_stage4_tool_ids = [tool.get("tool_id", "") for tool in visible_stage4_tools if tool.get("tool_id")]
    visible_stage4_tool_map = {tool.get("tool_id", ""): tool for tool in visible_stage4_tools if tool.get("tool_id")}

    print(f"[EXEC-ITER] Starting iterative execution (max_turns={max_turns})")

    execution_trace = []

    low_signal_markers = (
        "[error]",
        "web browser failed",
        "vision model returned no usable visual analysis",
        "no text detected",
        "timed out",
        "timeout",
        "403",
        "404",
        "forbidden",
        "blocked",
        "failed to download",
        "execution failed",
        "insufficient data",
        "no records found",
    )
    vision_refusal_markers = (
        "can't analyze images",
        "cannot analyze images",
        "unable to analyze images",
        "unable to view images",
        "unable to inspect the image",
        "as an ai text-based model",
        "as a text-only model",
        "provide a description",
        "cannot directly view",
        "i'm sorry, but i can't analyze",
    )
    hard_failure_markers = (
        "this video isn't available anymore",
        "video unavailable",
        "not found",
        "no such file",
        "filenotfounderror",
        "gateway error",
        "internal server error",
        "unable to open",
        "failed to fetch",
        "bad request",
    )

    def _stringify_tool_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)

    def _tool_call_args_dict(call: Dict[str, Any]) -> Dict[str, Any]:
        arg_dict = {}
        for item in call.get("arguments", []):
            if isinstance(item, dict) and item.get("name"):
                arg_dict[item["name"]] = item.get("value")
        return arg_dict

    def _tool_call_signature(call: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        args_dict = _tool_call_args_dict(call)
        return (
            call.get("tool_id", ""),
            tuple(sorted((key, _stringify_tool_value(value)) for key, value in args_dict.items())),
        )

    def _tool_family_signature(tool_id: str, args_dict: Dict[str, Any]) -> Tuple[Any, ...]:
        if tool_id == "image_recognition":
            task = str(args_dict.get("task", "describe") or "describe").strip().lower() or "describe"
            file_path = str(args_dict.get("file_path", "") or "").strip()
            prompt_family = canonicalize_image_recognition_prompt_family(task, str(args_dict.get("custom_prompt", "") or ""))
            return (tool_id, file_path, task, prompt_family)
        return (
            tool_id,
            tuple(sorted((key, _stringify_tool_value(value)) for key, value in args_dict.items())),
        )

    def _tool_call_family_signature(call: Dict[str, Any]) -> Tuple[Any, ...]:
        return _tool_family_signature(call.get("tool_id", ""), _tool_call_args_dict(call))

    def _normalize_observation_for_loop(text: str) -> str:
        norm = str(text or "")
        norm = re.sub(r"Debug Record:\s+\S+", "Debug Record: <path>", norm)
        norm = re.sub(r"(?:/[A-Za-z0-9._-]+){2,}", "<path>", norm)
        norm = re.sub(r"\s+", " ", norm).strip().lower()
        return norm

    def _same_observation_repeat_count(family_signature: Tuple[Any, ...]) -> int:
        normalized_obs = [
            _normalize_observation_for_loop(obs)
            for call, obs in zip(actual_tool_calls, observations)
            if _tool_call_family_signature(call) == family_signature
        ]
        if not normalized_obs:
            return 0
        return sum(1 for obs in normalized_obs if obs == normalized_obs[-1])

    def _is_same_observation_loop_candidate(tool_id: str, args_dict: Dict[str, Any]) -> Tuple[bool, str]:
        family_signature = _tool_family_signature(tool_id, args_dict)
        matching_pairs = [
            (call, obs)
            for call, obs in zip(actual_tool_calls, observations)
            if _tool_call_family_signature(call) == family_signature
        ]
        if not matching_pairs:
            return False, ""

        repeat_count = _same_observation_repeat_count(family_signature)
        last_obs = matching_pairs[-1][1]
        if tool_id == "image_recognition":
            if repeat_count >= 2:
                return True, (
                    f"The same image_recognition family on the same image already produced the same observation "
                    f"{repeat_count} time(s). Do not call that family again."
                )
            if len(matching_pairs) >= 2 and _is_failed_observation(last_obs):
                return True, (
                    "Recent image_recognition attempts for this same image/task family stayed low-yield or refusal-like. "
                    "Do not repeat that same family again."
                )
        return False, ""

    def _contains_marker(text: str, markers: Tuple[str, ...]) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in markers)

    def _is_low_signal_observation(text: str) -> bool:
        return _contains_marker(text, low_signal_markers) or _contains_marker(text, vision_refusal_markers)

    def _is_failed_observation(text: str) -> bool:
        return _is_low_signal_observation(text) or _contains_marker(text, hard_failure_markers)

    def _primary_tool_target(call: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        args_dict = _tool_call_args_dict(call)
        for key in ("url", "query", "file_path", "sheet_name", "page", "task"):
            value = args_dict.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return key, value_str
        return None

    def _recent_non_low_signal_count(obs_list: List[str], window: int = 3) -> int:
        return sum(1 for obs in obs_list[-window:] if not _is_low_signal_observation(obs))

    def _looks_like_gpt_oss_model(model_name: str) -> bool:
        return (model_name or "").strip().lower().startswith("gpt-oss")

    def _raw_is_blank(raw_text: Optional[str]) -> bool:
        return not (isinstance(raw_text, str) and raw_text.strip())

    def _format_short_tool_signature(tool_id: str) -> str:
        tool = visible_stage4_tool_map.get(tool_id, {})
        schema = tool.get("arguments_schema") or []
        if isinstance(schema, list) and schema:
            signature = ", ".join(_format_schema_param(param) for param in schema)
            return f"{tool_id}({signature})"
        return get_tool_signature_string(tool_id)

    def _build_last_resort_tool_shortlist() -> List[str]:
        ordered_candidates = [
            "submit_final_answer",
            "reasoning",
            "python_executor",
            "calculator",
            "web_browser",
            "web_search",
            "file_reader",
            "pdf_reader",
            "excel_reader",
            "audio_transcription",
            "image_recognition",
        ]
        failing_tools: Set[str] = set()
        if actual_tool_calls and observations and _is_failed_observation(observations[-1]):
            failing_tools.add(actual_tool_calls[-1].get("tool_id", ""))
        shortlist: List[str] = []
        for tool_id in ordered_candidates:
            if tool_id not in visible_stage4_tool_ids:
                continue
            if tool_id in failing_tools and tool_id not in {"submit_final_answer", "reasoning"}:
                continue
            shortlist.append(tool_id)
            if len(shortlist) >= 4:
                break
        if "submit_final_answer" in visible_stage4_tool_ids and "submit_final_answer" not in shortlist:
            shortlist.insert(0, "submit_final_answer")
        if "reasoning" in visible_stage4_tool_ids and "reasoning" not in shortlist and len(shortlist) < 4:
            insert_at = 1 if shortlist and shortlist[0] == "submit_final_answer" else 0
            shortlist.insert(insert_at, "reasoning")
        return shortlist[:4]

    def _build_last_resort_tool_lines(tool_ids: List[str]) -> str:
        lines: List[str] = []
        for tool_id in tool_ids:
            line = f"- {_format_short_tool_signature(tool_id)}"
            description = (visible_stage4_tool_map.get(tool_id, {}).get("description") or "").strip().rstrip(".")
            if description:
                line += f" - {description}"
            lines.append(line)
        return "\n".join(lines)

    def _should_force_submit_last_resort(current_turn: int) -> bool:
        if not observations:
            return False
        last_obs = observations[-1]
        last_tool = actual_tool_calls[-1].get("tool_id", "") if actual_tool_calls else ""
        if actual_tool_calls and last_tool in {"calculator", "python_executor", "excel_reader", "file_reader", "pdf_reader", "audio_transcription", "reasoning"} and not _is_low_signal_observation(last_obs):
            return True
        if len(observations) >= 4 and _recent_non_low_signal_count(observations, window=3) >= 2:
            return True
        if text_only_task and current_turn >= max_turns - 4 and not _is_low_signal_observation(last_obs):
            return True
        return False

    def _canonicalize_stage4_decision(candidate: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(candidate, dict):
            return None, None

        current = dict(candidate)
        for wrapper_key in ("action", "next_action", "decision", "response", "result", "call"):
            inner = current.get(wrapper_key)
            if len(current) == 1 and isinstance(inner, dict):
                current = dict(inner)
                break

        reasoning = ""
        for reasoning_key in ("reasoning", "thought", "thoughts", "rationale", "why", "note"):
            value = current.get(reasoning_key)
            if isinstance(value, str) and value.strip():
                reasoning = value.strip()
                break

        arguments = current.get("arguments")
        if not isinstance(arguments, dict):
            for alt_key in ("args", "params", "parameters", "input", "kwargs"):
                alt_value = current.get(alt_key)
                if isinstance(alt_value, dict):
                    arguments = dict(alt_value)
                    break
        if not isinstance(arguments, dict):
            arguments = {}

        tool_id = current.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id.strip():
            for alt_key in ("tool", "tool_name", "name", "action_name", "action_type"):
                alt_value = current.get(alt_key)
                if isinstance(alt_value, str) and alt_value.strip():
                    tool_id = alt_value.strip()
                    break

        answer = None
        if isinstance(arguments.get("answer"), str) and arguments["answer"].strip():
            answer = arguments["answer"].strip()
        else:
            for answer_key in ("answer", "final_answer", "final", "response", "output"):
                answer_value = current.get(answer_key)
                if isinstance(answer_value, str) and answer_value.strip():
                    answer = answer_value.strip()
                    break

        if isinstance(tool_id, str) and tool_id.strip():
            normalized_tool = re.sub(r"[\s\-]+", "_", tool_id.strip().lower())
            tool_aliases = {
                "submit": "submit_final_answer",
                "submit_answer": "submit_final_answer",
                "submit_final": "submit_final_answer",
                "final_answer": "submit_final_answer",
                "answer": "submit_final_answer",
                "browser": "web_browser",
                "search": "web_search",
                "python": "python_executor",
                "code": "python_executor",
                "ocr": "image_recognition",
            }
            tool_id = tool_aliases.get(normalized_tool, normalized_tool)

        if (not tool_id or tool_id == "none") and answer:
            return {
                "tool_id": "submit_final_answer",
                "arguments": {"answer": answer},
                "reasoning": reasoning or "Converted answer-only response to submit_final_answer.",
            }, "answer_only_json"

        if tool_id == "submit_final_answer":
            if answer and not arguments.get("answer"):
                arguments = dict(arguments)
                arguments["answer"] = answer
            return {
                "tool_id": tool_id,
                "arguments": arguments,
                "reasoning": reasoning,
            }, "normalized_submit_action"

        if tool_id in known_stage4_tools:
            return {
                "tool_id": tool_id,
                "arguments": arguments,
                "reasoning": reasoning,
            }, "normalized_action"

        return None, None

    for turn in range(max_turns):
        # ipdb debug support
        if debug:
            import ipdb
            ipdb.set_trace()

        print(f"\n[EXEC-ITER] Turn {turn+1}/{max_turns}")

        # Build prompt for model to decide next action

        # --- [P1] Extract plan from Stage 3 for injection ---
        plan_text = ""
        plan_nodes = final_pred.get("plan_dag", {}).get("nodes", []) if isinstance(final_pred, dict) else []
        if plan_nodes:
            plan_lines = ["Your execution plan (from planning stage):"]
            for idx, node in enumerate(plan_nodes, 1):
                tool = node.get("tool_id", "unknown")
                desc = node.get("description", node.get("task", ""))
                plan_lines.append(f"  {idx}. [{tool}] {desc}")
            plan_text = "\n".join(plan_lines)

        # --- [P2] Recency-aware observation truncation ---
        obs_text = ""
        if observations:
            obs_parts = []
            n_obs = len(observations)
            for j, obs in enumerate(observations):
                is_recent = (j >= n_obs - 2)  # last 2 observations get full text
                if is_recent:
                    truncated = obs[:2000] + "..." if len(obs) > 2000 else obs
                else:
                    truncated = obs[:200] + "..." if len(obs) > 200 else obs
                obs_parts.append(f"Step {j+1}:\n{truncated}")
            obs_text = "\n\n".join(obs_parts)

        # --- [P3] Reflection prompt ---
        reflection_text = ""
        if observations:
            reflection_text = f"""
Reflect on your progress so far:
- You have completed {len(observations)} step(s) out of a maximum of {max_turns}.
- What information have you gathered? What is still missing?
- If previous attempts failed or returned insufficient data, try a different approach.
- Do NOT repeat the same tool call with the same arguments.
"""

            recent_hints = []
            if len(actual_tool_calls) >= 2 and len(observations) >= 2:
                last_call = actual_tool_calls[-1]
                prev_call = actual_tool_calls[-2]
                last_obs = observations[-1]
                prev_obs = observations[-2]
                last_tool = last_call.get("tool_id", "")
                prev_tool = prev_call.get("tool_id", "")
                last_args = _tool_call_args_dict(last_call)
                prev_args = _tool_call_args_dict(prev_call)
                same_signature = _tool_call_signature(last_call) == _tool_call_signature(prev_call)

                if last_tool == prev_tool == "image_recognition":
                    same_image = last_args.get("file_path") == prev_args.get("file_path")
                    refusal_like = _contains_marker(last_obs, vision_refusal_markers) or _contains_marker(prev_obs, vision_refusal_markers)
                    low_signal = _contains_marker(last_obs, low_signal_markers) or _contains_marker(prev_obs, low_signal_markers)
                    if same_image and (refusal_like or low_signal):
                        recent_hints.append(
                            "Recent image_recognition attempts on the same image did not add usable visual evidence. Change the task or information source instead of repeating the same visual query."
                        )
                    family_repeat_count = _same_observation_repeat_count(_tool_call_family_signature(last_call))
                    if same_image and family_repeat_count >= 2:
                        recent_hints.append(
                            f"The same image_recognition family on this image has already returned the same observation {family_repeat_count} time(s). Change strategy instead of asking the same visual family again."
                        )

                if same_signature and (
                    _contains_marker(last_obs, low_signal_markers) or _contains_marker(prev_obs, low_signal_markers)
                ):
                    recent_hints.append(
                        f"The last two {last_tool} calls used the same arguments and did not produce new usable information. Do not repeat that exact call again."
                    )
                elif last_tool == prev_tool and last_tool in {"web_search", "web_browser"} and (
                    _contains_marker(last_obs, low_signal_markers) and _contains_marker(prev_obs, low_signal_markers)
                ):
                    recent_hints.append(
                        "Recent web access attempts stayed low-yield. Change the query, change the source, or switch tools instead of continuing the same browsing pattern."
                    )

            if actual_tool_calls and observations:
                failed_signature_counts = Counter()
                failed_target_counts = Counter()
                target_counts = Counter()
                for call, obs in zip(actual_tool_calls, observations):
                    signature = _tool_call_signature(call)
                    target = _primary_tool_target(call)
                    if target:
                        target_counts[target] += 1
                    if _is_failed_observation(obs):
                        failed_signature_counts[signature] += 1
                        if target:
                            failed_target_counts[target] += 1

                last_call = actual_tool_calls[-1]
                last_tool = last_call.get("tool_id", "")
                last_obs = observations[-1]
                last_obs_low_signal = _is_low_signal_observation(last_obs)
                last_signature = _tool_call_signature(last_call)
                last_target = _primary_tool_target(last_call)

                if failed_signature_counts[last_signature] >= 2:
                    recent_hints.append(
                        f"You have already seen {failed_signature_counts[last_signature]} failed or low-yield attempt(s) for this exact {last_tool} call. Change the query, URL, or tool instead of repeating it."
                    )

                if last_tool == "image_recognition":
                    family_repeat_count = _same_observation_repeat_count(_tool_call_family_signature(last_call))
                    if family_repeat_count >= 2:
                        recent_hints.append(
                            f"The latest image_recognition family has already produced the same observation {family_repeat_count} time(s). Do not call that same visual family again unless you materially change the request."
                        )

                if last_target and failed_target_counts[last_target] >= 2:
                    target_text = last_target[1]
                    if len(target_text) > 140:
                        target_text = target_text[:137] + "..."
                    recent_hints.append(
                        f"The same {last_target[0]} has already failed or stayed low-yield {failed_target_counts[last_target]} time(s): {target_text}. Move on instead of retrying it."
                    )
                elif last_target and target_counts[last_target] >= 3:
                    target_text = last_target[1]
                    if len(target_text) > 140:
                        target_text = target_text[:137] + "..."
                    recent_hints.append(
                        f"You have already targeted the same {last_target[0]} {target_counts[last_target]} time(s): {target_text}. Only repeat it if you have a specific new reason."
                    )

                if text_only_task and _recent_non_low_signal_count(observations, window=3) >= 2 and len(observations) >= 4:
                    recent_hints.append(
                        "You already have multiple non-empty observations. Prefer synthesizing and submitting the answer instead of launching another exploratory search unless one specific missing fact is still identified."
                    )

                if text_only_task and turn >= max_turns - 5:
                    recent_hints.append(
                        "You are close to the turn limit. If you have one plausible answer candidate, call submit_final_answer now instead of continuing exploratory browsing."
                    )

                if last_tool in {"calculator", "python_executor", "excel_reader", "file_reader", "pdf_reader", "audio_transcription"} and not last_obs_low_signal:
                    recent_hints.append(
                        "Your latest observation may already contain the answer. If it yields a single clear candidate, call submit_final_answer now instead of exploring further."
                    )

            if recent_hints:
                deduped_hints = list(dict.fromkeys(recent_hints))
                reflection_text += "Recent execution notes:\n"
                reflection_text += "\n".join(f"- {hint}" for hint in deduped_hints)
                reflection_text += "\n"

        # --- [P1] Attachment info prompt ---
        attachment_text = ""
        has_image_attachments = False
        has_audio_attachments = False
        has_document_attachments = False
        document_guidance_enabled = args.dataset == "gaia_cat_B"
        if attachments:
            attachment_lines = ["ATTACHED FILES:"]
            for att in attachments:
                fname = att.get("file_name", "unknown")
                fpath = get_attachment_display_path(att, query_record)
                fname_lower = fname.lower()
                if fname_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                    has_image_attachments = True
                    attachment_lines.append(f"  - {fname} (image) at {fpath}")
                    if supports_native_vision_input(args.model_name):
                        attachment_lines.append(f"    → The image is attached to the model input. You may inspect it directly.")
                    else:
                        attachment_lines.append(f"    → This model does not receive the image directly; use image_recognition for visual evidence.")
                    attachment_lines.append(f"    → For specialized analysis, use: image_recognition(file_path='{fpath}', task='<task>', custom_prompt='<specific question>')")
                elif fname_lower.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac')):
                    has_audio_attachments = True
                    attachment_lines.append(f"  - {fname} (audio) at {fpath}")
                    attachment_lines.append(f"    → Start with audio_transcription(file_path='{fpath}', language='<short_code_optional>')")
                else:
                    has_document_attachments = True
                    attachment_lines.append(f"  - {fname} at {fpath}")
            attachment_text = "\n".join(attachment_lines) + "\n"

            # Add image/audio guidance based on actual runtime capabilities
            if has_image_attachments:
                if supports_native_vision_input(args.model_name):
                    attachment_text += """
VISION GUIDANCE:
- This run supports native image input, so you may inspect the attached image(s) directly.
- For specialized tasks (chess positions, music sheets, colored numbers), use image_recognition with task or custom_prompt when you want a more structured extraction.
- Be specific about positions, colors, and details in your analysis.
"""
                else:
                    attachment_text += """
VISION GUIDANCE:
- This run does NOT expose the image directly to the model.
- Use image_recognition to gather visual evidence before reasoning about the answer.
- Do not guess unseen visual details; if the answer depends on the image, call image_recognition first.
"""
            if has_audio_attachments:
                attachment_text += """
AUDIO GUIDANCE:
- For audio attachments (.mp3, .wav, .m4a, .ogg, .flac, .aac), start with audio_transcription.
- Do NOT use file_reader on audio attachments.
- After transcription, reason over the returned transcript directly instead of trying to import file_reader inside python_executor.
- If the transcript already identifies the answer clearly, call submit_final_answer instead of looping through extra reasoning steps.
- If you provide the optional language argument, use a short language code like "en" or "zh", not locale variants like "en-US".
"""
            if document_guidance_enabled and has_document_attachments:
                attachment_text += """
DOCUMENT GUIDANCE:
- Treat attached documents, spreadsheets, slides, archives, and local files as the primary evidence source unless the question explicitly requires outside information.
- Prefer attachment-first solving: read the attached file, inspect the relevant rows/pages/columns, compute locally, then submit.
- Do NOT start with web_search or web_browser when the attachment itself is likely sufficient.
- For spreadsheets or CSV-like files, prefer excel_reader first, then python_executor in short inspection/compute steps.
- For PDFs, prefer pdf_reader before any generic browsing.
- For archives, unzip or inspect the attachment contents before searching the web.
"""

        tool_list_lines = _build_stage4_tool_prompt_lines(
            query_record,
            new_tools,
            tool_scope=getattr(args, "tool_scope", "record"),
        )
        available_tools_str = "\n".join(tool_list_lines)
        image_policy_text = ""
        if has_image_attachments:
            if supports_native_vision_input(args.model_name):
                image_policy_text = """
Image decision policy:
- First inspect the attached image directly and use that native visual evidence when it is sufficient.
- Use image_recognition only when you need structured extraction such as OCR, color-organized lists, chess boards, music notation, or geometry measurements.
"""
            else:
                image_policy_text = """
Image decision policy:
- This model does not receive the image directly.
- If the answer depends on the attachment, call image_recognition before reasoning further.
"""
        document_policy_text = ""
        if document_guidance_enabled and has_document_attachments:
            document_policy_text = """
Document decision policy:
- Attachment-first: if an attached file can answer the question, inspect that file before using web_search or web_browser.
- Use outside search only when the attachment is incomplete, ambiguous, or the question explicitly asks for external background knowledge.
- Keep the document workflow short and local: attachment tool -> short python step if needed -> submit_final_answer.
- Avoid repeated re-reading of the same file section once the needed values are already extracted.
"""
        document_rules_text = ""
        if document_guidance_enabled and has_document_attachments:
            document_rules_text = """
- If an attached document or spreadsheet appears sufficient, do not use web_search or web_browser before trying the attachment-first route.
"""

        decision_prompt = f"""Task:
{user_query}

{attachment_text}
{plan_text}

{'Previous steps and observations:' if observations else 'This is your first step.'}
{obs_text}
{reflection_text}
{image_policy_text}
{document_policy_text}
Available tools:
{available_tools_str}

Return EXACTLY ONE JSON object for the next action:
{{
  "tool_id": "tool_name",
  "arguments": {{"param": "value"}},
  "reasoning": "brief reason for this one action"
}}

Rules:
- Output JSON only. No prose, no markdown, no code fences.
- Choose exactly one next tool action.
- Do NOT output plan_dag, tool_calls, final_answer, or any multi-step plan.
- If you already know the answer, or a recent observation gives one clear answer candidate, call submit_final_answer with {{"answer": "..."}}.
- Do not keep searching after the answer is already identified with reasonable confidence.
- For calculations, gather the required values first, then calculate.
- For files, tables, or spreadsheets, use python_executor in short steps: inspect columns/head/shape first, then compute.
{document_rules_text}
"""

        def _request_stage4_decision(prompt_text: str) -> Tuple[Optional[Dict[str, Any]], str, str]:
            if args.backend == "sglang":
                # WARN: SGLang backend doesn't support multimodal - vision tasks may perform poorly
                if has_image_attachments:
                    print(f"  [WARN] SGLang backend does not support images. Consider using 'api' backend with VLM for vision tasks.")
                from src.inference.sglang_client import SGLangClient
                client = SGLangClient(base_url=args.sglang_url, timeout=60.0)
                decision_output, _ = client.generate_and_validate(
                    prompt=prompt_text,
                    json_schema=None,
                    sample_id=sample_id,
                    max_tokens=2048,
                    strict_parse=False
                )
                if isinstance(decision_output, dict):
                    return decision_output, json.dumps(decision_output, ensure_ascii=False), ""
                decision_raw_local = str(decision_output)
                parsed_local, parse_err_local = parse_action_json(decision_raw_local)
                return parsed_local, decision_raw_local, parse_err_local

            if args.backend == "api":
                decision_text = generate_with_api(
                    api_model=args.model_name,
                    prompt=prompt_text,
                    max_new_tokens=2048,
                    attachments=attachments,  # Pass image attachments to VLM
                    allow_reasoning_content_fallback=True,
                    require_json_like_reasoning_fallback=True,
                )
            else:
                decision_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    max_new_tokens=2048
                )

            parsed_local, parse_err_local = parse_action_json(decision_text)
            return parsed_local, decision_text, parse_err_local

        try:
            decision_raw = None
            decision = None
            parse_failure_reason = None
            initial_decision_raw = None
            repair_raw = None
            fallback_raw = None
            last_resort_raw = None
            decision_salvage_note = None
            last_resort_mode = None

            decision, decision_raw, parse_failure_reason = _request_stage4_decision(decision_prompt)
            initial_decision_raw = decision_raw
            decision, decision_salvage_note = _canonicalize_stage4_decision(decision)

            if not decision or "tool_id" not in decision:
                if decision and "tool_id" not in decision:
                    parse_failure_reason = "Parsed decision missing required field: tool_id"

                should_repair = True
                if should_repair:
                    print(f"  [INFO] Attempting one repair reprompt for invalid Stage 4 decision...")
                    previous_response_block = (
                        decision_raw[:2500]
                        if decision_raw and decision_raw.strip()
                        else "[EMPTY OUTPUT]"
                    )
                    repair_prompt = f"""{decision_prompt}

Your previous response was invalid because it did not match the required Stage 4 action schema.

Previous response:
{previous_response_block}

Fix it now.
Output ONLY one JSON object with exactly these top-level keys:
- tool_id
- arguments
- reasoning
- If you already know the answer, do NOT return {{"answer": "..."}} alone. Wrap it as submit_final_answer with arguments.answer.
"""
                    repaired_decision, repaired_raw, repaired_error = _request_stage4_decision(repair_prompt)
                    repair_raw = repaired_raw
                    repaired_decision, repaired_salvage_note = _canonicalize_stage4_decision(repaired_decision)
                    if repaired_decision and "tool_id" in repaired_decision:
                        decision = repaired_decision
                        decision_raw = repaired_raw
                        decision_salvage_note = repaired_salvage_note or decision_salvage_note
                        parse_failure_reason = None
                    else:
                        if repaired_decision and "tool_id" not in repaired_decision:
                            repaired_error = "Parsed repair decision missing required field: tool_id"
                        if repaired_error:
                            parse_failure_reason = f"{parse_failure_reason or 'Invalid decision'}; repair failed: {repaired_error}"

            if (not decision or "tool_id" not in decision) and _looks_like_gpt_oss_model(args.model_name):
                recent_obs_for_fallback = obs_text[-3000:] if obs_text else "[NO OBSERVATIONS YET]"
                fallback_prompt = f"""Task:
{user_query}

Recent observations:
{recent_obs_for_fallback}

Available tools:
{available_tools_str}

Return EXACTLY ONE JSON object for the immediate next action.
If the answer is already identifiable, return:
{{"tool_id":"submit_final_answer","arguments":{{"answer":"..."}},"reasoning":"brief reason"}}

Do NOT return {{"answer":"..."}} alone.
Do NOT leave the response empty.
Output JSON only with top-level keys:
- tool_id
- arguments
- reasoning
"""
                print("  [INFO] Attempting gpt-oss fallback action repair...")
                fallback_decision, fallback_raw_candidate, fallback_error = _request_stage4_decision(fallback_prompt)
                fallback_raw = fallback_raw_candidate
                fallback_decision, fallback_salvage_note = _canonicalize_stage4_decision(fallback_decision)
                if fallback_decision and "tool_id" in fallback_decision:
                    decision = fallback_decision
                    decision_raw = fallback_raw_candidate
                    decision_salvage_note = fallback_salvage_note or decision_salvage_note or "gpt_oss_fallback_repair"
                    parse_failure_reason = None
                else:
                    if fallback_decision and "tool_id" not in fallback_decision:
                        fallback_error = "Parsed gpt-oss fallback decision missing required field: tool_id"
                    if fallback_error:
                        parse_failure_reason = f"{parse_failure_reason or 'Invalid decision'}; gpt-oss fallback failed: {fallback_error}"

            repeated_blank_case = (
                _looks_like_gpt_oss_model(args.model_name) and
                not decision and
                sum(
                    1
                    for raw in (initial_decision_raw, repair_raw, fallback_raw)
                    if raw is not None and _raw_is_blank(raw)
                ) >= 2
            )
            if repeated_blank_case:
                shortlist = _build_last_resort_tool_shortlist()
                shortlist_lines = _build_last_resort_tool_lines(shortlist)
                recent_obs_for_last_resort = obs_text[-3500:] if obs_text else "[NO OBSERVATIONS YET]"
                last_target = _primary_tool_target(actual_tool_calls[-1]) if actual_tool_calls else None
                anti_repeat_line = ""
                if last_target:
                    target_value = last_target[1]
                    if len(target_value) > 180:
                        target_value = target_value[:177] + "..."
                    anti_repeat_line = f"Do NOT repeat the same failed {last_target[0]} again: {target_value}\n"

                if _should_force_submit_last_resort(turn):
                    last_resort_mode = "forced_submit"
                    last_resort_prompt = f"""Task:
{user_query}

You returned empty output repeatedly. This is a recovery step.

Recent observations:
{recent_obs_for_last_resort}

Preferred action:
- If the observations already support one best answer candidate, return submit_final_answer now.
- Only if one tiny synthesis step is still needed, you may use reasoning instead.

Allowed tools in this recovery step:
{_build_last_resort_tool_lines([tool_id for tool_id in shortlist if tool_id in {'submit_final_answer', 'reasoning'}] or shortlist[:2])}

{anti_repeat_line}Output EXACTLY ONE JSON object with top-level keys:
- tool_id
- arguments
- reasoning

Do NOT return empty output.
Do NOT return {{\"answer\":\"...\"}} alone.
"""
                else:
                    last_resort_mode = "forced_single_action"
                    last_resort_prompt = f"""Task:
{user_query}

You returned empty output repeatedly. This is a recovery step.

Recent observations:
{recent_obs_for_last_resort}

Choose EXACTLY ONE next action from this shortlist only:
{shortlist_lines}

Rules:
- Prefer submit_final_answer if one answer candidate is already supported.
- Otherwise choose one concrete next action from the shortlist only.
- Do not repeat the same failed query or URL.
{anti_repeat_line}Output EXACTLY ONE JSON object with top-level keys:
- tool_id
- arguments
- reasoning

Do NOT return empty output.
Do NOT return {{\"answer\":\"...\"}} alone.
"""

                print(f"  [INFO] Attempting gpt-oss last-resort {last_resort_mode} repair...")
                last_resort_decision, last_resort_raw_candidate, last_resort_error = _request_stage4_decision(last_resort_prompt)
                last_resort_raw = last_resort_raw_candidate
                last_resort_decision, last_resort_salvage_note = _canonicalize_stage4_decision(last_resort_decision)
                if last_resort_decision and "tool_id" in last_resort_decision:
                    decision = last_resort_decision
                    decision_raw = last_resort_raw_candidate
                    decision_salvage_note = last_resort_salvage_note or last_resort_mode
                    parse_failure_reason = None
                else:
                    if last_resort_decision and "tool_id" not in last_resort_decision:
                        last_resort_error = "Parsed gpt-oss last-resort decision missing required field: tool_id"
                    if last_resort_error:
                        parse_failure_reason = f"{parse_failure_reason or 'Invalid decision'}; gpt-oss last-resort failed: {last_resort_error}"

            if not decision or "tool_id" not in decision:
                print(f"  [WARN] Model did not provide valid decision, stopping.")
                execution_trace.append({
                    "step": turn,
                    "status": "invalid_decision",
                    "tool_id": decision.get("tool_id") if isinstance(decision, dict) else None,
                    "args": decision.get("arguments", {}) if isinstance(decision, dict) else {},
                    "reasoning": decision.get("reasoning", "") if isinstance(decision, dict) else "",
                    "output": None,
                    "raw_output": decision_raw,
                    "initial_raw_output": initial_decision_raw,
                    "repair_raw_output": repair_raw,
                    "fallback_raw_output": fallback_raw,
                    "last_resort_raw_output": last_resort_raw,
                    "decision_salvage_note": decision_salvage_note,
                    "last_resort_mode": last_resort_mode,
                    "parsed_decision": decision,
                    "error": parse_failure_reason,
                })
                break

            tool_id = decision["tool_id"]
            model_args = decision.get("arguments", {})
            if not isinstance(model_args, dict):
                model_args = {}
            reasoning = decision.get("reasoning", "")

            answer_unwrap_note = None
            if tool_id == "submit_final_answer":
                submit_candidate = model_args.get("answer") if "answer" in model_args else model_args
                normalized_answer, unwrap_note = normalize_submitted_answer(submit_candidate)
                if isinstance(normalized_answer, str) and normalized_answer.strip():
                    normalized_args = {}
                    answer_type = model_args.get("answer_type")
                    if isinstance(answer_type, str) and answer_type.strip():
                        normalized_args["answer_type"] = answer_type.strip()
                    normalized_args["answer"] = normalized_answer.strip()
                    model_args = normalized_args
                    answer_unwrap_note = unwrap_note
                    if unwrap_note:
                        if decision_salvage_note:
                            decision_salvage_note = f"{decision_salvage_note}; {unwrap_note}"
                        else:
                            decision_salvage_note = unwrap_note

            tool_id, model_args, arg_normalize_note = normalize_tool_call_for_signature(tool_id, model_args)
            if arg_normalize_note:
                if decision_salvage_note:
                    decision_salvage_note = f"{decision_salvage_note}; {arg_normalize_note}"
                else:
                    decision_salvage_note = arg_normalize_note

            loop_guard_raw = None
            loop_guard_triggered, loop_guard_reason = _is_same_observation_loop_candidate(tool_id, model_args)
            if loop_guard_triggered:
                print(f"  [INFO] Same-observation loop guard triggered for {tool_id}: {loop_guard_reason}")
                guard_prompt = decision_prompt + f"""

Loop guard:
- {loop_guard_reason}
- Do NOT call that same image_recognition family again on this turn.
- If the image is directly visible to the model, inspect it directly instead of repeating the tool call.
- Otherwise choose a materially different tool, a materially different visual task family, or submit_final_answer.
"""
                guarded_decision, guarded_raw, _ = _request_stage4_decision(guard_prompt)
                loop_guard_raw = guarded_raw
                guarded_salvage_note = None
                if guarded_decision:
                    guarded_decision, guarded_salvage_note = _canonicalize_stage4_decision(guarded_decision)

                if guarded_decision and "tool_id" in guarded_decision:
                    guarded_tool_id = guarded_decision["tool_id"]
                    guarded_args = guarded_decision.get("arguments", {})
                    if not isinstance(guarded_args, dict):
                        guarded_args = {}
                    retriggered, _ = _is_same_observation_loop_candidate(guarded_tool_id, guarded_args)
                    if not retriggered:
                        decision = guarded_decision
                        tool_id = guarded_tool_id
                        model_args = guarded_args
                        reasoning = guarded_decision.get("reasoning", reasoning)
                        decision_raw = guarded_raw
                        if guarded_salvage_note:
                            if decision_salvage_note:
                                decision_salvage_note = f"{decision_salvage_note}; {guarded_salvage_note}"
                            else:
                                decision_salvage_note = guarded_salvage_note
                        if decision_salvage_note:
                            decision_salvage_note = f"{decision_salvage_note}; same_observation_guard_reprompt"
                        else:
                            decision_salvage_note = "same_observation_guard_reprompt"
                    else:
                        blocked_message = f"[LOOP GUARD] Blocked repeated image_recognition family. {loop_guard_reason}"
                        print(f"  [WARN] {blocked_message}")
                        observations.append(blocked_message)
                        execution_trace.append({
                            "step": turn,
                            "status": "blocked_same_observation_loop",
                            "tool_id": tool_id,
                            "args": model_args,
                            "output": blocked_message,
                            "reasoning": reasoning,
                            "raw_output": decision_raw,
                            "initial_raw_output": initial_decision_raw,
                            "repair_raw_output": repair_raw,
                            "fallback_raw_output": fallback_raw,
                            "last_resort_raw_output": last_resort_raw,
                            "loop_guard_raw_output": loop_guard_raw,
                            "decision_salvage_note": "same_observation_guard_blocked",
                            "last_resort_mode": last_resort_mode,
                            "answer_unwrap_note": answer_unwrap_note,
                            "parsed_decision": decision,
                        })
                        continue
                else:
                    blocked_message = f"[LOOP GUARD] Blocked repeated image_recognition family. {loop_guard_reason}"
                    print(f"  [WARN] {blocked_message}")
                    observations.append(blocked_message)
                    execution_trace.append({
                        "step": turn,
                        "status": "blocked_same_observation_loop",
                        "tool_id": tool_id,
                        "args": model_args,
                        "output": blocked_message,
                        "reasoning": reasoning,
                        "raw_output": decision_raw,
                        "initial_raw_output": initial_decision_raw,
                        "repair_raw_output": repair_raw,
                        "fallback_raw_output": fallback_raw,
                        "last_resort_raw_output": last_resort_raw,
                        "loop_guard_raw_output": loop_guard_raw,
                        "decision_salvage_note": "same_observation_guard_blocked",
                        "last_resort_mode": last_resort_mode,
                        "answer_unwrap_note": answer_unwrap_note,
                        "parsed_decision": decision,
                    })
                    continue

            if tool_id == "submit_final_answer":
                submit_candidate = model_args.get("answer") if "answer" in model_args else model_args
                normalized_answer, unwrap_note = normalize_submitted_answer(submit_candidate)
                if isinstance(normalized_answer, str) and normalized_answer.strip():
                    normalized_args = {}
                    answer_type = model_args.get("answer_type")
                    if isinstance(answer_type, str) and answer_type.strip():
                        normalized_args["answer_type"] = answer_type.strip()
                    normalized_args["answer"] = normalized_answer.strip()
                    model_args = normalized_args
                    answer_unwrap_note = unwrap_note
                    if unwrap_note:
                        if decision_salvage_note:
                            decision_salvage_note = f"{decision_salvage_note}; {unwrap_note}"
                        else:
                            decision_salvage_note = unwrap_note

            print(f"  Tool: {tool_id}")
            print(f"  Reasoning: {reasoning}")
            print(f"  Model arguments: {model_args}")

            # Get tool signature
            tool_sig = get_tool_signature_string(tool_id)

            # Check if arguments match tool signature
            if tool_id in TOOL_SIGNATURES:
                expected_params = set(TOOL_SIGNATURES[tool_id].keys())
                provided_params = set(model_args.keys())

                # If mismatch, ask model to refine with correct parameter names
                if not provided_params.issubset(expected_params):
                    print(f"  [INFO] Parameter names don't match tool signature, asking model to refine...")
                    print(f"  [INFO] Tool signature: {tool_sig}")

                    refine_prompt = f"""You selected tool: {tool_id}

Tool signature: {tool_sig}

Your provided arguments: {json.dumps(model_args, indent=2)}

The parameter names don't match the tool's expected parameters.
Please provide the arguments using the CORRECT parameter names from the signature above.

Output ONLY a JSON object:
{{"correct_param_name": "value", ...}}
"""

                    try:
                        if args.backend == "sglang":
                            client = SGLangClient(base_url=args.sglang_url, timeout=60.0)
                            refined_output, _ = client.generate_and_validate(
                                prompt=refine_prompt,
                                json_schema=None,
                                sample_id=sample_id,
                                max_tokens=2048,
                                strict_parse=False
                            )
                            refined_args = refined_output if isinstance(refined_output, dict) else json.loads(refined_output)
                        elif args.backend == "api":
                            refined_text = generate_with_api(
                                api_model=args.model_name,
                                prompt=refine_prompt,
                                max_new_tokens=2048
                            )
                            refined_args, _ = robust_json_parse(refined_text, fallback_empty=True)
                        else:
                            refined_text = generate_with_model(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=refine_prompt,
                                max_new_tokens=2048
                            )
                            refined_args, _ = robust_json_parse(refined_text, fallback_empty=True)

                        if refined_args:
                            print(f"  [INFO] Model refined to: {refined_args}")
                            model_args = refined_args
                        else:
                            # Fallback: try automatic mapping
                            print(f"  [WARN] Model refinement failed, using automatic mapping")
                            model_args = map_arguments_to_tool_signature(tool_id, model_args)
                            print(f"  [INFO] Mapped to: {model_args}")

                    except Exception as e:
                        print(f"  [ERROR] Refinement failed: {e}, using automatic mapping")
                        model_args = map_arguments_to_tool_signature(tool_id, model_args)
                        print(f"  [INFO] Mapped to: {model_args}")

                tool_id, model_args, arg_normalize_note = normalize_tool_call_for_signature(tool_id, model_args)
                if arg_normalize_note:
                    print(f"  [INFO] Argument normalizer applied after refinement: {arg_normalize_note}")
                    if decision_salvage_note:
                        decision_salvage_note = f"{decision_salvage_note}; {arg_normalize_note}"
                    else:
                        decision_salvage_note = arg_normalize_note

            # Execute the tool
            print(f"  > Executing: {tool_id}({model_args})")
            output = execute_tool(tool_id, model_args, attachments_dir=attachments_dir, output_dir=output_dir)
            print(f"    Result: {output[:150]}..." if len(output) > 150 else f"    Result: {output}")

            # Store observation
            observations.append(f"Tool: {tool_id}\nArgs: {model_args}\nOutput:\n{output}")

            # Record actual tool call
            actual_call = {
                "call_index": turn,
                "tool_id": tool_id,
                "arguments": [{"name": k, "value": v} for k, v in model_args.items()]
            }
            actual_tool_calls.append(actual_call)

            # Log trace
            execution_trace.append({
                "step": turn,
                "status": "executed",
                "tool_id": tool_id,
                "args": model_args,
                "output": output,
                "reasoning": reasoning,
                "raw_output": decision_raw,
                "initial_raw_output": initial_decision_raw,
                "repair_raw_output": repair_raw,
                "fallback_raw_output": fallback_raw,
                "last_resort_raw_output": last_resort_raw,
                "decision_salvage_note": decision_salvage_note,
                "last_resort_mode": last_resort_mode,
                "answer_unwrap_note": answer_unwrap_note,
                "parsed_decision": decision,
            })

            # Check if this was submit_final_answer
            if tool_id == "submit_final_answer":
                print(f"  [INFO] Model submitted final answer, stopping.")
                break

        except Exception as e:
            print(f"  [ERROR] Turn {turn+1} failed: {e}")
            import traceback
            traceback.print_exc()
            execution_trace.append({
                "step": turn,
                "status": "turn_error",
                "tool_id": decision.get("tool_id") if isinstance(decision, dict) else None,
                "args": decision.get("arguments", {}) if isinstance(decision, dict) else {},
                "output": None,
                "reasoning": decision.get("reasoning", "") if isinstance(decision, dict) else "",
                "raw_output": decision_raw,
                "initial_raw_output": initial_decision_raw,
                "repair_raw_output": repair_raw,
                "fallback_raw_output": fallback_raw,
                "last_resort_raw_output": last_resort_raw,
                "decision_salvage_note": decision_salvage_note,
                "last_resort_mode": last_resort_mode,
                "answer_unwrap_note": answer_unwrap_note if 'answer_unwrap_note' in locals() else None,
                "parsed_decision": decision,
                "error": str(e),
            })
            break

    # Save execution trace
    if output_dir:
        trace_path = os.path.join(output_dir, f"execution_trace_{sample_id}.json")
        try:
            with open(trace_path, "w") as f:
                json.dump(execution_trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to save execution trace: {e}")

    return observations, actual_tool_calls, execution_trace


def run_stage_4_execution(query_record: Dict[str, Any], final_pred: Dict[str, Any], output_dir: str = "", debug: bool = False) -> List[str]:
    """
    Stage 4: Execute the plan using real tools with variable resolution.

    Legacy execution mode (non-iterative) - used for non-answer modes.

    Iterates through tool_calls in the prediction and executes them.
    Resolves variable references (<n0>, <n1>, etc.) using outputs from previous steps.
    Returns a list of observations (tool outputs).
    """
    observations = []
    tool_calls = final_pred.get("tool_calls", [])

    # Determine attachments directory from query record if possible
    attachments = query_record.get("query", {}).get("attachments", [])
    attachments_dir = ""
    if attachments:
        first_path = resolve_attachment_path(
            attachments[0].get("file_path"),
            attachments[0].get("file_name"),
            query_record,
        )
        if first_path:
            attachments_dir = os.path.dirname(first_path)

    # Sample ID for logging
    sample_id = query_record.get("meta", {}).get("id", "unknown")

    print(f"[EXEC] Executing {len(tool_calls)} tools with attachments_dir: {attachments_dir}")

    # Initialize variable store for resolving <n0>, <n1>, etc.
    var_store = {}  # Maps "<n0>" -> actual tool output string

    # Build mapping from node_id to output_vars for storing results
    plan_nodes = final_pred.get("plan_dag", {}).get("nodes", [])
    node_output_map = {}  # Maps "n0" -> "<n0>"
    for node in plan_nodes:
        node_id = node.get("node_id")
        output_vars = node.get("output_vars", [])
        if node_id and output_vars:
            # Usually output_vars is ["<n0>"], we take the first one
            node_output_map[node_id] = output_vars[0]

    execution_trace = []

    for i, tc in enumerate(tool_calls):
        tool_id = tc.get("tool_id")
        args_list = tc.get("arguments", [])
        node_id = tc.get("node_id")  # Which plan node this call belongs to

        # Convert list of {name, value} to dict for execution
        tool_args = {}
        if isinstance(args_list, list):
            for arg in args_list:
                raw_value = arg["value"]
                # Resolve any variable references in the value
                resolved_value = resolve_variables(raw_value, var_store)
                tool_args[arg["name"]] = resolved_value
        elif isinstance(args_list, dict):
            # If already a dict, resolve all values
            tool_args = {k: resolve_variables(v, var_store) for k, v in args_list.items()}

        # Debug mode breakpoint - pause before EACH tool execution
        if debug:
            print(f"\n[DEBUG] About to execute step {i+1}/{len(tool_calls)}: {tool_id}")
            print(f"[DEBUG] Arguments: {tool_args}")
            print(f"[DEBUG] Sample: {sample_id}")
            try:
                import ipdb; ipdb.set_trace()
            except ImportError:
                print("[WARN] ipdb not installed, skipping breakpoint. (pip install ipdb)")

        # Execute
        print(f"  > Call: {tool_id}({tool_args})")
        output = execute_tool(tool_id, tool_args, attachments_dir=attachments_dir, output_dir=output_dir)
        print(f"    Result: {output[:100]}..." if output else "    Result: (empty)")

        observations.append(f"Tool: {tool_id}\nArgs: {tool_args}\nOutput:\n{output}")

        # Store output in variable store for use by subsequent steps
        if node_id and node_id in node_output_map:
            var_name = node_output_map[node_id]  # e.g., "<n0>"
            var_store[var_name] = output
            print(f"  > Stored {var_name} = {output[:80]}..." if len(output) > 80 else f"  > Stored {var_name} = {output}")

        # Log step
        execution_trace.append({
            "step": i,
            "tool_id": tool_id,
            "args": tool_args,
            "output": output
        })

    # Save detailed execution trace to JSON if output_dir is provided
    if output_dir:
        trace_path = os.path.join(output_dir, f"execution_trace_{sample_id}.json")
        try:
            with open(trace_path, "w") as f:
                json.dump(execution_trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to save execution trace: {e}")

    return observations


def _truncate_prompt_text(text: str, max_chars: int) -> str:
    """Keep prompt snippets bounded while preserving both the start and end."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 200:
        return text[:max_chars] + "\n...[truncated]"
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + f"\n\n...[truncated {len(text) - max_chars} chars]...\n\n"
        + text[-tail_chars:]
    )


def _format_observations_for_final_answer(observations: List[str]) -> str:
    """Build a bounded Stage 5 history; raw outputs stay in traces/results."""
    if not observations:
        return "[NO TOOL OBSERVATIONS]"

    parts = []
    n_obs = len(observations)
    for idx, obs in enumerate(observations):
        is_recent = idx >= n_obs - 3
        max_chars = 6000 if is_recent else 800
        parts.append(f"Step {idx + 1}:\n{_truncate_prompt_text(obs, max_chars)}")

    return _truncate_prompt_text("\n\n".join(parts), 28000)


def generate_final_answer(query_record: Dict[str, Any], observations: List[str], args, model, tokenizer) -> Dict[str, Any]:
    """
    Stage 5: Generate Final Answer from Observations.
    """
    user_query = query_record.get("query", {}).get("user_query", "")
    obs_text = _format_observations_for_final_answer(observations)

    prompt = f"""[ANSWER GENERATION]

## User Request
{user_query}

## Tool Execution History
{obs_text}

## Task
Synthesize the Tool Execution History to provide the Final Answer to the User Request.
Strictly output the answer in JSON format.

## MANDATORY OUTPUT FORMAT
{{
  "final_answer": {{
    "answer": "The final concise answer string",
    "reasoning": "Brief justification based on tool outputs"
  }}
}}

## OUTPUT JSON NOW:"""

    if args.backend == "sglang":
        raw_output, _ = generate_with_sglang(
            prompt=prompt,
            dataset="answer",
            sglang_url=args.sglang_url,
            max_new_tokens=2048,
            use_schema_constraint=False
        )
    elif args.backend == "api":
        attachments = query_record.get("query", {}).get("attachments", [])
        raw_output = generate_with_api(
            prompt=prompt,
            api_model=args.model_name,
            dataset="answer",
            max_new_tokens=1024,
            attachments=attachments
        )
    else:  # local backend
        raw_output = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=1024
        )

    parsed, parse_error = robust_json_parse(raw_output, fallback_empty=True)
    if parsed and "final_answer" in parsed:
        return {
            "final_answer": parsed["final_answer"],
            "raw_output": raw_output,
            "parsed_output": parsed,
            "parse_status": "parsed_final_answer",
            "parse_error": parse_error,
        }

    # Fallback structure
    return {
        "final_answer": {"answer": raw_output.strip(), "reasoning": "Failed to parse final answer JSON"},
        "raw_output": raw_output,
        "parsed_output": parsed or {},
        "parse_status": "fallback_raw_text",
        "parse_error": parse_error or "final_answer_json_missing",
    }


def _as_list(value: Any) -> List[Any]:
    """Normalize scalar-or-list JSON edge endpoints."""
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _chain_dag_from_abstract_steps(abstract_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback abstract DAG that preserves the Stage-1 step order as a chain."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    steps = abstract_plan.get("steps", []) if isinstance(abstract_plan, dict) else []

    for i, step in enumerate(steps or []):
        if isinstance(step, dict):
            step_index = step.get("step_index", i)
            label = str(step.get("description") or step.get("label") or step.get("task") or "").strip()
        else:
            step_index = i
            label = str(step).strip()
        if not isinstance(step_index, int):
            try:
                step_index = int(step_index)
            except (TypeError, ValueError):
                step_index = i
        if not label:
            continue
        node_id = f"a{step_index}"
        nodes.append({
            "node_id": node_id,
            "step_index": step_index,
            "label": label,
            "step_type": "thought",
            "needs_tool": False,
        })

    for source, target in zip(nodes, nodes[1:]):
        edges.append({
            "source": source["node_id"],
            "target": target["node_id"],
            "edge_type": "dependency",
        })

    return {"nodes": nodes, "edges": edges}


def normalize_abs_plan_dag(abstract_plan: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Normalize the Stage-1 planning-intent DAG without changing the downstream
    Stage-2/3 abstract-plan contract. The original `steps` field remains the
    only field consumed by later stages.
    """
    if not isinstance(abstract_plan, dict):
        return {"nodes": [], "edges": []}, "empty"

    raw_dag = None
    source = "empty"
    for candidate_key in ("abs_plan_dag", "abstract_plan_dag", "plan_dag", "dag"):
        candidate = abstract_plan.get(candidate_key)
        if isinstance(candidate, dict):
            raw_dag = candidate
            source = candidate_key
            break

    if not isinstance(raw_dag, dict) or not isinstance(raw_dag.get("nodes"), list):
        fallback = _chain_dag_from_abstract_steps(abstract_plan)
        if fallback.get("nodes"):
            return fallback, "steps_chain_fallback"
        return {"nodes": [], "edges": []}, "empty"

    nodes: List[Dict[str, Any]] = []
    id_map: Dict[str, str] = {}

    for i, node in enumerate(raw_dag.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        raw_node_id = str(node.get("node_id") or node.get("id") or f"a{i}")
        step_index = node.get("step_index", i)
        if not isinstance(step_index, int):
            try:
                step_index = int(step_index)
            except (TypeError, ValueError):
                step_index = i
        node_id = raw_node_id.strip() or f"a{step_index}"
        if node_id in {n["node_id"] for n in nodes}:
            node_id = f"a{step_index}"
        label = str(
            node.get("label")
            or node.get("description")
            or node.get("task")
            or node.get("name")
            or ""
        ).strip()
        if not label:
            continue
        id_map[raw_node_id] = node_id
        id_map[str(step_index)] = node_id
        nodes.append({
            "node_id": node_id,
            "step_index": step_index,
            "label": label,
            "step_type": "thought",
            "needs_tool": False,
        })

    valid_node_ids = {node["node_id"] for node in nodes}
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[Tuple[str, str]] = set()
    for edge in raw_dag.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for raw_source in _as_list(edge.get("source")):
            source_id = id_map.get(str(raw_source), str(raw_source))
            if source_id not in valid_node_ids:
                continue
            for raw_target in _as_list(edge.get("target")):
                target_id = id_map.get(str(raw_target), str(raw_target))
                if target_id not in valid_node_ids or target_id == source_id:
                    continue
                edge_key = (source_id, target_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "edge_type": str(edge.get("edge_type") or "dependency"),
                })

    if not nodes:
        fallback = _chain_dag_from_abstract_steps(abstract_plan)
        if fallback.get("nodes"):
            return fallback, "steps_chain_fallback"
        return {"nodes": [], "edges": []}, "empty"

    return {"nodes": nodes, "edges": edges}, source


def run_stage_1_abstract(query_record: Dict[str, Any], args, model, tokenizer) -> Dict[str, Any]:
    """
    Run Stage 1: Abstract Planning.

    Generates a high-level plan WITHOUT showing tools to the model.
    This ensures the plan is a pure logical decomposition of the user's intent,
    not constrained by available tools.
    """
    prompt = build_abstract_plan_prompt(query_record, dataset=args.dataset)

    if args.backend == "sglang":
        raw_output, _ = generate_with_sglang(
            prompt=prompt,
            dataset="abstract",  # Dummy dataset to avoid wrong schema
            sglang_url=args.sglang_url,
            max_new_tokens=1024,
            use_schema_constraint=False  # Disable constraint for intermediate stages
        )
    elif args.backend == "api":
        attachments = query_record.get("query", {}).get("attachments", [])
        raw_output = generate_with_api(
            prompt=prompt,
            api_model=args.model_name,
            dataset="abstract",
            max_new_tokens=1024,
            attachments=attachments
        )
    else:  # local backend
        raw_output = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=1024
        )

    # Use empty fallback for abstract plan to avoid noise. Some models emit a
    # valid JSON string instead of an object; keep the pipeline fail-closed.
    parsed, _ = robust_json_parse(raw_output, fallback_empty=True)
    if not isinstance(parsed, dict):
        return {"steps": [], "reasoning": "Failed to parse abstract plan"}
    return parsed


def run_stage_2_creation(query_record: Dict[str, Any], abstract_plan: Dict[str, Any], args, model, tokenizer) -> Dict[str, Any]:
    """
    Run Stage 2: Tool Creation.

    Evaluates if existing tools are sufficient for the abstract plan.
    If not, proposes new tools to fill the gaps.
    """
    prompt = build_tool_creation_prompt(
        query_record,
        abstract_plan,
        dataset=args.dataset,
        tool_scope=getattr(args, "tool_scope", "record"),
    )

    if args.backend == "sglang":
        raw_output, _ = generate_with_sglang(
            prompt=prompt,
            dataset="creation",
            sglang_url=args.sglang_url,
            max_new_tokens=1024,
            use_schema_constraint=False
        )
    elif args.backend == "api":
        attachments = query_record.get("query", {}).get("attachments", [])
        raw_output = generate_with_api(
            prompt=prompt,
            api_model=args.model_name,
            dataset="creation",
            max_new_tokens=1024,
            attachments=attachments
        )
    else:  # local backend
        raw_output = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=1024
        )

    parsed, _ = robust_json_parse(raw_output, fallback_empty=True)
    if not isinstance(parsed, dict):
        return {"new_tools": [], "reasoning": "Failed to parse tool creation"}
    new_tools = parsed.get("new_tools", [])
    if not isinstance(new_tools, list):
        parsed["new_tools"] = []
    return parsed


def run_stage_3_refinement(query_record: Dict[str, Any], abstract_plan: Dict[str, Any], new_tools: List[Dict[str, Any]], args, model, tokenizer) -> Optional[Dict[str, Any]]:
    """
    Run Stage 3: Refinement (Final Output).

    Takes the abstract plan and full tool set (existing + new) to generate
    a concrete execution plan with tool calls.

    In answer mode, uses actual executable tools from tools.py to ensure
    the model only selects tools that can be executed.
    """
    # Use actual tools in answer mode to prevent [ERROR] Unknown tool
    use_actual_tools = (args.mode == "answer")
    prompt = build_refinement_prompt(
        query_record,
        abstract_plan,
        new_tools,
        use_actual_tools=use_actual_tools,
        dataset=args.dataset,
        tool_scope=getattr(args, "tool_scope", "record"),
    )

    if args.backend == "sglang":
        # Use taskbench schema (generic tool_id as string)
        # Note: If new tools were created, strict enum constraint might reject them
        raw_output, _ = generate_with_sglang(
            prompt=prompt,
            dataset=args.dataset,
            sglang_url=args.sglang_url,
            max_new_tokens=2048,
            use_schema_constraint=not args.no_schema_constraint
        )
    elif args.backend == "api":
        attachments = query_record.get("query", {}).get("attachments", [])
        raw_output = generate_with_api(
            prompt=prompt,
            api_model=args.model_name,
            dataset=args.dataset,
            max_new_tokens=2048,
            attachments=attachments
        )
    else:  # local backend
        raw_output = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=2048
        )

    used_stage3_fallback = False

    # Standard robust parse
    parsed, _ = robust_json_parse(raw_output)
    if parsed:
        parsed = normalize_pred_structure(parsed)

    # Minimal fallback: if Stage 3 did not preserve a usable DAG but Stage 1 has
    # abstract steps, synthesize a simple chain so planning metrics still have a
    # concrete structure to compare against.
    abstract_steps = abstract_plan.get("steps", []) if isinstance(abstract_plan, dict) else []
    plan_nodes = ((parsed or {}).get("plan_dag", {}) or {}).get("nodes", []) if isinstance(parsed, dict) else []
    if abstract_steps and not plan_nodes:
        chain_nodes = []
        chain_edges = []
        for i, step in enumerate(abstract_steps):
            if isinstance(step, dict):
                step_index = step.get("step_index", i)
                label = str(step.get("description") or step.get("label") or step.get("task") or "").strip()
            else:
                step_index = i
                label = str(step).strip()
            if not isinstance(step_index, int):
                try:
                    step_index = int(step_index)
                except (TypeError, ValueError):
                    step_index = i
            if not label:
                continue
            chain_nodes.append({
                "node_id": f"n{step_index}",
                "step_index": step_index,
                "label": label,
                "step_type": "thought",
                "needs_tool": False,
            })

        for prev, curr in zip(chain_nodes, chain_nodes[1:]):
            chain_edges.append({
                "source": prev["node_id"],
                "target": curr["node_id"],
                "edge_type": "data_dep",
            })

        if chain_nodes:
            if not isinstance(parsed, dict):
                parsed = {}
            parsed["plan_dag"] = {"nodes": chain_nodes, "edges": chain_edges}
            parsed.setdefault("tool_calls", [])
            used_stage3_fallback = True
            print(f"[INFO] Stage 3 fallback: synthesized chain DAG from {len(chain_nodes)} abstract steps")

    if isinstance(parsed, dict):
        parsed["_stage3_used_fallback"] = used_stage3_fallback

    return parsed


def load_processed_ids(output_path: Path) -> Set[str]:
    """Load already processed sample IDs from existing output file."""
    processed_ids = set()
    if output_path.exists():
        try:
            with output_path.open("r") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        # Try multiple ID sources for compatibility
                        sample_id = rec.get("meta", {}).get("id")
                        if sample_id is None:
                            sample_id = rec.get("id")
                        if sample_id is not None:
                            processed_ids.add(str(sample_id))
        except Exception as e:
            print(f"[WARN] Could not read existing output file: {e}")
    return processed_ids


# ============================================================================
# Main Entry Point - Multi-Stage Pipeline
# ============================================================================

def main():
    """
    Main entry point for the multi-stage planning pipeline.
    Supports SGLang, Local, and API backends.
    """
    parser = argparse.ArgumentParser(description="Multi-Stage Pipeline Runner")
    parser.add_argument("--unified_path", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--model_name", type=str, default="default")
    parser.add_argument("--backend", type=str, default="sglang", choices=["local", "sglang", "api"])
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_schema_constraint", action="store_true",
                        help="Disable JSON schema constraint for Stage 3")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already processed samples)")
    parser.add_argument("--save_trace", action="store_true", default=True,
                        help="Save intermediate stage outputs in pred._trace (default: True)")
    parser.add_argument("--mode", type=str, default="order", choices=["order", "answer"],
                        help="Execution mode: 'order' (planning only) or 'answer' (planning + execution)")
    parser.add_argument("--dataset", type=str, default="taskbench",
                        help="Dataset name for schema selection (e.g., 'gaia', 'delta')")
    # Legacy arguments for compatibility
    parser.add_argument("--save_raw", action="store_true", help="(Ignored - for API backend compatibility)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (ipdb breakpoints and detailed step logging)")
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Maximum number of iterative tool-execution turns in answer mode (default: 15)")
    parser.add_argument("--tool_scope", type=str, default="record", choices=["record", "global"],
                        help="Prompt-visible tool scope. 'record' uses each task record's annotated tool inventory; 'global' exposes the full executable tools.py library for ablations.")
    parser.add_argument("--stage3_abs_dag_reference", "--stage3-abs-dag-reference",
                        action="store_true",
                        help="Deprecated no-op. Stage 3 no longer receives Stage-1 abs_plan_dag context; Stage-1 planning intent is stored and evaluated separately.")

    args = parser.parse_args()
    os.environ["LLM_PLANNING_CURRENT_MODEL"] = args.model_name
    os.environ["LLM_PLANNING_CURRENT_DATASET"] = args.dataset
    if not MULTISTAGE_AVAILABLE:
        raise RuntimeError(
            "Multi-stage inference imports are unavailable. "
            "Check that the llm_planning environment is active and dependencies "
            f"are installed. Original import error: {MULTISTAGE_IMPORT_ERROR}"
        )

    print("=" * 60)
    print("Multi-Stage Pipeline Runner")
    print("=" * 60)
    print(f"  Input:     {args.unified_path}")
    print(f"  Output:    {args.output}")
    print(f"  Model:     {args.model_name}")
    print(f"  Backend:   {args.backend}")
    print(f"  Mode:      {args.mode}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  ToolScope: {args.tool_scope}")
    print(f"  Limit:     {args.limit or 'all'}")
    print(f"  Resume:    {args.resume}")
    if args.stage3_abs_dag_reference:
        print("  Stage3 DAG Ref: deprecated flag requested; ignored")
    else:
        print("  Stage3 DAG Ref: disabled")
    # Schema constraint only works with SGLang backend (xgrammar)
    if args.backend == "sglang":
        print(f"  Schema:    {'disabled' if args.no_schema_constraint else 'enabled (xgrammar)'}")
    else:
        print(f"  Schema:    N/A ({args.backend} backend)")
    print("=" * 60)

    # Initialize Model/Client based on backend
    model, tokenizer = None, None

    if args.backend == "local":
        cfg = MODEL_CONFIGS.get(args.model_name)
        if not cfg:
            raise ValueError(f"Model config not found for {args.model_name}")
        print(f"[INFO] Loading local model: {cfg.get('model_id', args.model_name)}")
        model, tokenizer = load_model(args.model_name, cfg)

    elif args.backend == "sglang":
        print(f"[INFO] Using SGLang server at {args.sglang_url}")
        from src.inference.sglang_client import SGLangClient
        client = SGLangClient(args.sglang_url)
        if not client.health_check():
            print(f"[WARN] SGLang server at {args.sglang_url} not healthy")
            print(f"[WARN] Continuing anyway - server may come up later")

    elif args.backend == "api":
        api_base = get_api_base()
        api_key = get_api_key()
        if not api_base:
            print("[ERROR] LLM_API_BASE environment variable not set")
            print("[ERROR] Please set: export LLM_API_BASE='https://your-api-server' or use --provider-profile openai")
            sys.exit(1)
        if not api_key:
            print("[ERROR] LLM_API_KEY environment variable not set")
            print("[ERROR] Please set: export LLM_API_KEY='your-api-key'")
            sys.exit(1)
        print(f"[INFO] Using API backend with model: {args.model_name}")
        print(f"[INFO] API Base: {api_base}")

    # Resume support
    processed_ids: Set[str] = set()
    resume_mode = False

    if args.resume:
        processed_ids = load_processed_ids(args.output)
        if processed_ids:
            print(f"[RESUME] Found {len(processed_ids)} already processed samples")
            resume_mode = True
        else:
            print(f"[RESUME] No existing results found, starting fresh")

    # Prepare output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Open output file (append if resuming, write if not)
    out_mode = "a" if resume_mode else "w"
    if resume_mode:
        print(f"[RESUME] Appending to existing file (mode='a')")

    # Load and filter records
    records = list(load_json(args.unified_path))
    if args.limit:
        records = records[:args.limit]

    # Statistics
    total_processed = 0
    total_skipped = 0
    parse_errors = 0

    # Tool creation log file
    log_file = args.output.parent / f"tool_creation_log.{args.output.stem}.jsonl"
    log_mode = "a" if resume_mode else "w"

    with open(args.output, out_mode) as f, open(log_file, log_mode) as f_log:
        for i, rec in enumerate(tqdm(records, desc="Processing")):
            # Get sample ID
            # Get sample ID (support task_id for GAIA, id, or meta.id)
            sample_id = rec.get("task_id") or rec.get("id") or rec.get("meta", {}).get("id") or i
            sample_id_str = str(sample_id)

            # Skip already processed samples when resuming
            if resume_mode and sample_id_str in processed_ids:
                total_skipped += 1
                continue

            total_processed += 1

            # Log data structure
            creation_log_entry = {
                "id": sample_id_str,
                "original_tool_ids": [],
                "created_tools": [],
                "has_new_tools": False
            }

            try:
                # Stage 1: Abstract Planning (tool-agnostic)
                abstract = run_stage_1_abstract(rec, args, model, tokenizer)
                abs_plan_dag, abs_plan_dag_source = normalize_abs_plan_dag(abstract)

                # Stage 2: Tool Creation
                creation = run_stage_2_creation(rec, abstract, args, model, tokenizer)
                new_tools = _normalize_created_tools(creation.get("new_tools", []))

                # Log tool diff
                original_tools = _extract_tool_environment(rec)
                creation_log_entry["original_tool_ids"] = [
                    t.get("tool_id") for t in original_tools if isinstance(t, dict)
                ]
                creation_log_entry["created_tools"] = new_tools
                creation_log_entry["has_new_tools"] = len(new_tools) > 0

                # Stage 3: Refinement (concrete plan with tools)
                final_pred = run_stage_3_refinement(rec, abstract, new_tools, args, model, tokenizer)

                if final_pred is None:
                    final_pred = create_empty_pred(args.model_name)
                    parse_errors += 1

                # Preserve Stage-3 execution planning under pred.plan_dag, and
                # store Stage-1 planning intent separately for intent-level
                # evaluation. Downstream stages continue to consume only
                # abstract["steps"], so this does not alter Stage 2+ logic.
                if final_pred is not None:
                    final_pred["abs_plan_dag"] = abs_plan_dag
                    final_pred["_abs_plan_dag_source"] = abs_plan_dag_source

                # Validate tool IDs against the correct tool universe.
                # In answer mode we must use executable runtime tools because
                # Stage 4 will actually call them. We therefore validate against
                # the same record-level tool universe exposed in Stage 4 rather
                # than the full GAIA runtime registry. In order mode we should
                # stay within the dataset's tool environment (plus any created
                # tools) rather than the GAIA runtime tool set.
                if final_pred is not None and "_error" not in final_pred:
                    if args.mode == "answer":
                        available_tools = [
                            tool.get("tool_id")
                            for tool in _get_stage4_visible_tool_specs(
                                rec,
                                new_tools,
                                tool_scope=getattr(args, "tool_scope", "record"),
                            )
                            if tool.get("tool_id")
                        ]
                    else:
                        env_tools = _extract_tool_environment(rec)
                        available_tools = []
                        for t in env_tools:
                            tool_id = t.get("tool_id") if isinstance(t, dict) else t
                            if tool_id and tool_id not in available_tools:
                                available_tools.append(tool_id)
                        for nt in new_tools:
                            tool_id = nt.get("tool_id") if isinstance(nt, dict) else None
                            if tool_id and tool_id not in available_tools:
                                available_tools.append(tool_id)
                    final_pred = validate_and_fix_tool_ids(final_pred, available_tools)

                stage4_execution_trace: List[Dict[str, Any]] = []
                stage5_result: Dict[str, Any] = {}
                answer_handoff: Dict[str, Any] = {}

                # Stage 4 & 5: Execution and Answer Generation (only in answer mode)
                if args.mode == "answer" and final_pred is not None and "_error" not in final_pred:
                    try:
                        # Stage 4: Iterative Execution (ReAct-style)
                        observations, actual_tool_calls, stage4_execution_trace = run_stage_4_iterative_execution(
                            rec, final_pred, new_tools, args, model, tokenizer,
                            output_dir=str(args.output.parent),
                            debug=args.debug,
                            max_turns=args.max_turns,
                        )

                        # Update final_pred with ACTUAL tool calls (not planned ones)
                        # This ensures evaluation uses what was actually executed
                        final_pred["tool_calls"] = actual_tool_calls
                        print(f"[INFO] Updated prediction with {len(actual_tool_calls)} actual tool calls")

                        # Stage 5: Answer Generation
                        stage5_result = generate_final_answer(rec, observations, args, model, tokenizer)
                        final_answer_obj = stage5_result.get("final_answer", {}) or {}
                        answer_handoff = _build_answer_handoff_record(
                            actual_tool_calls=actual_tool_calls,
                            final_answer_obj=final_answer_obj,
                            stage5_result=stage5_result,
                        )

                        # Update prediction with execution results
                        final_pred["final_answer"] = final_answer_obj
                        final_pred["tool_outputs"] = observations
                        final_pred["submit_answer"] = answer_handoff.get("effective_submit_answer", "")
                        final_pred["_answer_handoff"] = answer_handoff

                    except Exception as e:
                        print(f"[ERROR] Execution stage failed for {sample_id_str}: {e}")
                        import traceback
                        traceback.print_exc()
                        final_pred["_execution_error"] = str(e)
                        stage4_execution_trace = []
                        stage5_result = {}
                        answer_handoff = {
                            "status": "execution_failed_before_handoff",
                            "source": "missing",
                            "reason": str(e),
                            "effective_submit_answer": "",
                            "native_submit_answer": "",
                            "stage5_answer": "",
                            "submit_tool_present": False,
                            "submit_call_index": None,
                            "submit_tool_answer_present": False,
                            "stage5_answer_present": False,
                            "submit_stage5_same": None,
                            "submit_normalization_note": None,
                            "stage5_normalization_note": None,
                            "stage5_parse_status": None,
                            "stage5_parse_error": None,
                        }
                        final_pred["submit_answer"] = ""
                        final_pred["_answer_handoff"] = answer_handoff

                # Add trace information for debugging
                if args.save_trace and final_pred:
                    used_stage3_fallback = bool(final_pred.pop("_stage3_used_fallback", False))
                    final_pred["_trace"] = {
                        "abstract_plan": abstract,
                        "abs_plan_dag": abs_plan_dag,
                        "abs_plan_dag_source": abs_plan_dag_source,
                        "created_tools": new_tools,
                        "creation_reasoning": creation.get("reasoning"),
                        "pipeline": "multi_stage",
                        "mode": args.mode,
                        "stage3_abs_dag_reference": False,
                        "stage3_abs_dag_reference_requested": bool(args.stage3_abs_dag_reference),
                        "used_stage3_fallback": used_stage3_fallback,
                        "stage4_decision_stream": stage4_execution_trace if args.mode == "answer" else [],
                        "stage5": stage5_result if args.mode == "answer" else {},
                        "answer_handoff": final_pred.get("_answer_handoff", {}),
                    }

            except Exception as e:
                print(f"[ERROR] Sample {sample_id_str}: {e}")
                final_pred = create_empty_pred(args.model_name)
                final_pred["_error"] = str(e)
                parse_errors += 1
                creation_log_entry["error"] = str(e)

            # Build output record
            out_rec = rec.copy()
            out_meta = dict(out_rec.get("meta", {}) or {})
            out_meta["model_name"] = args.model_name
            out_meta["supports_native_vision"] = supports_native_vision_input(args.model_name)
            out_rec["meta"] = out_meta
            out_rec["pred"] = final_pred

            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()

            # Write creation log
            f_log.write(json.dumps(creation_log_entry, ensure_ascii=False) + "\n")
            f_log.flush()

    # Summary
    print("")
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"  Total processed: {total_processed}")
    print(f"  Skipped (resume): {total_skipped}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Output: {args.output}")
    print(f"  Tool Log: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
