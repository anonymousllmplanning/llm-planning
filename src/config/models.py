"""Model configuration and API client setup."""

import os


def normalize_api_base(base_url: str) -> str:
    """Normalize user-provided API base URLs to the OpenAI-compatible endpoint."""
    base_url = (base_url or "").strip()
    if not base_url:
        return ""

    return base_url.rstrip("/")


def get_api_base() -> str:
    """Read and normalize the shared API base from environment variables."""
    raw = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_COMPATIBLE_API_BASE") or ""
    if raw:
        return normalize_api_base(raw)

    provider = get_api_provider()
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "anthropic":
        return "https://api.anthropic.com"
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com"
    return ""


def get_api_key() -> str:
    """Read the shared API key from environment variables."""
    raw = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "").strip()
    if raw:
        return raw

    provider = get_api_provider()
    if provider == "openai":
        return (os.getenv("OPENAI_API_KEY") or "").strip()
    if provider == "anthropic":
        return (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if provider == "gemini":
        return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return ""


def get_api_provider() -> str:
    """Read the active remote provider profile."""
    return (os.getenv("LLM_API_PROVIDER") or "openai_compatible").strip().lower()


def get_verifier_api_provider() -> str:
    """Read the verifier provider profile, defaulting to the shared provider."""
    return (os.getenv("ANSWER_VERIFIER_API_PROVIDER") or get_api_provider()).strip().lower()


def get_verifier_api_base() -> str:
    """Read and normalize the verifier API base, defaulting to the shared API base."""
    raw = (os.getenv("ANSWER_VERIFIER_API_BASE") or "").strip()
    if raw:
        return normalize_api_base(raw)
    return get_api_base()


def get_verifier_api_key() -> str:
    """Read the verifier API key, defaulting to the shared API key."""
    raw = (os.getenv("ANSWER_VERIFIER_API_KEY") or "").strip()
    if raw:
        return raw
    return get_api_key()


def get_api_client():
    """Get OpenAI-compatible API client for the configured LLM server."""
    api_base = get_api_base()
    api_key = get_api_key()
    if not api_base:
        raise ValueError("LLM_API_BASE environment variable not set")
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable not set")

    from openai import OpenAI

    return OpenAI(base_url=api_base, api_key=api_key)


# Representative API models on the current OpenAI-compatible servers.
AVAILABLE_API_MODELS = [
    "Mistral-Large-3-675B-Instruct-2512",
    "Llama-3.1-405B-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "gemma-4-31B-it",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "gemma-3-12b-it",
]


def _split_env_list(raw: str) -> set[str]:
    """Parse a comma-separated environment variable into a normalized set."""
    return {
        item.strip().lower()
        for item in (raw or "").split(",")
        if item.strip()
    }


def supports_native_vision_input(model_name: str) -> bool:
    """
    Return True if the current endpoint should receive image attachments natively.

    The default heuristic intentionally stays conservative, but it includes model
    families that are known to accept OpenAI-style ``image_url`` content.
    Environment variables can extend or
    disable the built-in behavior without code changes.
    """
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return False

    force_disable = _split_env_list(os.getenv("DISABLE_NATIVE_VISION_MODELS", ""))
    if normalized in force_disable:
        return False

    force_enable = _split_env_list(os.getenv("ENABLE_NATIVE_VISION_MODELS", ""))
    if normalized in force_enable:
        return True

    if any(marker in normalized for marker in ("vision", "multimodal", "phi-4")):
        return True

    if normalized.startswith(("gpt-4", "gpt-5", "claude-", "gemini-")):
        return True

    verified_prefixes = (
        "llama-4-",
        "google-gemma-3-27b",
        "gemma-3-12b-it",
        "gemma-4-",
        "mistral-large-3-",
        "mistral-small-3.2-",
    )
    return any(normalized.startswith(prefix) for prefix in verified_prefixes)
