"""
Centralized Model Registry for LLM Planning Framework.

All model configurations in one place:
- HuggingFace model IDs
- Context lengths (critical for avoiding SGLang errors)
- Routing (local SGLang vs remote API)
- Default parameters

Usage:
    from src.config.model_registry import MODEL_REGISTRY, get_model_config
    config = get_model_config("qwen2.5-7b")
"""

from typing import Dict, Any, Optional, List

# =============================================================================
# Model Registry - Single Source of Truth
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Qwen 2.5 Series (Alibaba) - Long context, good JSON following
    # =========================================================================
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "qwen2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Vicuna Series (LMSYS) - LIMITED CONTEXT LENGTH (4096)
    # =========================================================================
    "vicuna-7b": {
        "hf_id": "lmsys/vicuna-7b-v1.5",
        "context_length": 4096,  # CRITICAL: Vicuna only supports 4096
        "max_new_tokens": 2048,
        "routing": "local",
        "supports_json_schema": True,
    },
    "vicuna-13b": {
        "hf_id": "lmsys/vicuna-13b-v1.5",
        "context_length": 4096,  # CRITICAL: Vicuna only supports 4096
        "max_new_tokens": 2048,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Mistral Series
    # =========================================================================
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Phi Series (Microsoft) - LIMITED CONTEXT LENGTH
    # =========================================================================
    "phi3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "context_length": 4096,  # 4k version
        "max_new_tokens": 2048,
        "routing": "local",
        "supports_json_schema": True,
    },
    "phi3.5-mini": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "context_length": 131072,  # 128k context
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Llama 3.2 Series (Meta)
    # =========================================================================
    "llama3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "context_length": 131072,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_length": 131072,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Gemma 2 Series (Google)
    # =========================================================================
    "gemma2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "context_length": 8192,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },
    "gemma2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "context_length": 8192,
        "max_new_tokens": 4096,
        "routing": "local",
        "supports_json_schema": True,
    },

    # =========================================================================
    # Remote API Models (NCCU Server)
    # =========================================================================
    "phi4-mini": {
        "api_id": "phi4-mini",
        "context_length": 16384,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,  # API may not support strict schema
    },
    "phi4": {
        "api_id": "phi4",
        "context_length": 16384,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,
    },
    "llama3-8b": {
        "api_id": "llama3:8b",
        "context_length": 8192,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,
    },
    "deepseek-r1-8b": {
        "api_id": "deepseek-r1:8b",
        "context_length": 32768,
        "max_new_tokens": 8192,
        "routing": "remote",
        "supports_json_schema": False,
    },
    "gemma3-27b": {
        "api_id": "gemma3:27b",
        "context_length": 8192,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,
    },
    "gpt-oss-20b": {
        "api_id": "gpt-oss:20b",
        "context_length": 8192,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,
    },
    "qwen3-4b": {
        "api_id": "qwen3:4b",
        "context_length": 32768,
        "max_new_tokens": 4096,
        "routing": "remote",
        "supports_json_schema": False,
    },
}

# =============================================================================
# Default Model Lists
# =============================================================================

LOCAL_MODELS: List[str] = [
    name for name, cfg in MODEL_REGISTRY.items()
    if cfg.get("routing") == "local"
]

REMOTE_MODELS: List[str] = [
    name for name, cfg in MODEL_REGISTRY.items()
    if cfg.get("routing") == "remote"
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a model.

    Args:
        model_name: Model name (e.g., "qwen2.5-7b", "vicuna-7b")

    Returns:
        Model configuration dict or None if not found
    """
    return MODEL_REGISTRY.get(model_name)


def get_context_length(model_name: str, default: int = 8192) -> int:
    """
    Get context length for a model.

    Args:
        model_name: Model name
        default: Default context length if model not found

    Returns:
        Context length in tokens
    """
    config = get_model_config(model_name)
    if config:
        return config.get("context_length", default)
    return default


def get_hf_id(model_name: str) -> Optional[str]:
    """Get HuggingFace ID for a local model."""
    config = get_model_config(model_name)
    if config:
        return config.get("hf_id")
    return None


def get_routing(model_name: str, default: str = "local") -> str:
    """Get routing type (local or remote) for a model."""
    config = get_model_config(model_name)
    if config:
        return config.get("routing", default)
    return default


def supports_json_schema(model_name: str) -> bool:
    """Check if model supports JSON schema constraint."""
    config = get_model_config(model_name)
    if config:
        return config.get("supports_json_schema", True)
    return True


def list_all_models() -> List[str]:
    """Get list of all available model names."""
    return list(MODEL_REGISTRY.keys())


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Registry Self-Test")
    print("=" * 60)

    print(f"\nTotal models: {len(MODEL_REGISTRY)}")
    print(f"Local models: {len(LOCAL_MODELS)}")
    print(f"Remote models: {len(REMOTE_MODELS)}")

    print("\n--- Context Lengths (Critical for SGLang) ---")
    for name in sorted(MODEL_REGISTRY.keys()):
        ctx = get_context_length(name)
        routing = get_routing(name)
        marker = "⚠️ " if ctx <= 4096 else "  "
        print(f"{marker}{name:20} context={ctx:6} routing={routing}")

    print("\n--- Low Context Models (require special handling) ---")
    low_ctx = [n for n, c in MODEL_REGISTRY.items() if c.get("context_length", 8192) <= 4096]
    for name in low_ctx:
        print(f"  - {name}: {get_context_length(name)} tokens")
