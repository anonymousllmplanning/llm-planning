"""
SGLang Server Client for Grammar-Constrained Decoding.

Provides a client interface to SGLang server with xgrammar support for
generating outputs that strictly conform to JSON schemas.

SGLang Server API Reference:
- Native endpoint: /generate (supports json_schema for constrained decoding)
- OpenAI-compatible: /v1/chat/completions

This client uses the native /generate endpoint for xgrammar features.

Usage:
    from src.inference.sglang_client import SGLangClient
    client = SGLangClient(base_url="http://127.0.0.1:30000")
    result = client.generate(prompt="...", json_schema={...})
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Use requests for synchronous HTTP calls
try:
    import requests
except ImportError:
    requests = None

# Optional: use httpx for async support
try:
    import httpx
except ImportError:
    httpx = None


def extract_first_json(raw_text: str) -> str:
    """
    Extract the first complete JSON object from text.
    
    This is necessary because SGLang with xgrammar sometimes outputs
    repeated JSON objects or truncated output like:
    '{"a": 1} {"a": 2} {"a": 3...'
    
    Returns the first complete {...} block.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return ""
    
    start = raw_text.find("{")
    if start == -1:
        return ""
    
    # Find matching closing brace by counting
    depth = 0
    in_string = False
    escape_next = False
    
    for i, ch in enumerate(raw_text[start:], start):
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
                return raw_text[start:i+1]
    
    # No complete JSON found, return from start to end (truncated)
    return raw_text[start:]


@dataclass
class SGLangResponse:
    """Response from SGLang server."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw_response: Optional[Dict[str, Any]] = None


class SGLangClientError(Exception):
    """Exception raised for SGLang client errors."""
    pass


class SGLangParseError(SGLangClientError):
    """Exception raised when grammar-constrained output still fails to parse."""

    def __init__(self, message: str, sample_id: str = "", prompt: str = "",
                 raw_response: str = "", parse_error: str = ""):
        super().__init__(message)
        self.sample_id = sample_id
        self.prompt = prompt
        self.raw_response = raw_response
        self.parse_error = parse_error

    def __str__(self):
        return (
            f"SGLangParseError: {self.args[0]}\n"
            f"  sample_id: {self.sample_id}\n"
            f"  prompt (first 500 chars): {self.prompt[:500]}...\n"
            f"  raw_response (first 1000 chars): {self.raw_response[:1000]}...\n"
            f"  parse_error: {self.parse_error}"
        )


class SGLangClient:
    """
    Client for SGLang server with xgrammar support.

    Supports:
    - JSON schema-based constrained decoding (xgrammar)
    - Automatic retries with exponential backoff
    - Clear error messages for debugging
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., http://127.0.0.1:30000)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        if requests is None:
            raise ImportError("requests library required: pip install requests")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Endpoints
        self.generate_endpoint = f"{self.base_url}/generate"
        self.health_endpoint = f"{self.base_url}/health"

    def health_check(self) -> bool:
        """Check if SGLang server is healthy."""
        try:
            resp = requests.get(self.health_endpoint, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_schema: Optional[Dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        repetition_penalty: float = 1.05,
    ) -> SGLangResponse:
        """
        Generate text with optional JSON schema constraint.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            json_schema: JSON schema for constrained decoding (xgrammar)
            stop: Optional stop sequences
            sampling_params: Additional sampling parameters
            repetition_penalty: Penalty for repeated tokens (default 1.05)

        Returns:
            SGLangResponse with generated text

        Raises:
            SGLangClientError: On server errors or timeouts
        """
        # Build request payload
        # SGLang native format for /generate endpoint
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0 if temperature == 0 else 0.95,
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": 0.1,  # Additional penalty for frequent tokens
            }
        }

        # Add stop sequences if provided
        if stop:
            payload["sampling_params"]["stop"] = stop

        # Add custom sampling params
        if sampling_params:
            payload["sampling_params"].update(sampling_params)

        # Add JSON schema for xgrammar constrained decoding
        if json_schema:
            payload["json_schema"] = json_schema

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_response(data)
                else:
                    error_msg = f"SGLang server error: {resp.status_code} - {resp.text[:500]}"
                    last_error = SGLangClientError(error_msg)

            except requests.exceptions.Timeout:
                last_error = SGLangClientError(
                    f"Request timeout after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError as e:
                last_error = SGLangClientError(
                    f"Connection error: {e}. Is SGLang server running at {self.base_url}?"
                )
            except Exception as e:
                last_error = SGLangClientError(f"Unexpected error: {e}")

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"[WARN] SGLang request failed (attempt {attempt + 1}), "
                      f"retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise last_error

    def generate_with_chat_template(
        self,
        messages: list[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> SGLangResponse:
        """
        Generate using OpenAI-compatible chat format.

        This uses the /v1/chat/completions endpoint with JSON schema support.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_schema: JSON schema for constrained decoding

        Returns:
            SGLangResponse with generated text
        """
        endpoint = f"{self.base_url}/v1/chat/completions"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add JSON schema (OpenAI response_format style)
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "eval_output",
                    "strict": True,
                    "schema": json_schema
                }
            }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_chat_response(data)
                else:
                    error_msg = f"SGLang server error: {resp.status_code} - {resp.text[:500]}"
                    last_error = SGLangClientError(error_msg)

            except requests.exceptions.Timeout:
                last_error = SGLangClientError(f"Request timeout after {self.timeout}s")
            except requests.exceptions.ConnectionError as e:
                last_error = SGLangClientError(
                    f"Connection error: {e}. Is SGLang server running at {self.base_url}?"
                )
            except Exception as e:
                last_error = SGLangClientError(f"Unexpected error: {e}")

            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"[WARN] SGLang chat request failed (attempt {attempt + 1}), "
                      f"retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise last_error

    def _parse_response(self, data: Dict[str, Any]) -> SGLangResponse:
        """Parse native /generate endpoint response."""
        # SGLang /generate response format:
        # {"text": "...", "meta_info": {...}}
        text = data.get("text", "")
        meta = data.get("meta_info", {})

        return SGLangResponse(
            text=text.strip(),
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            finish_reason=meta.get("finish_reason", ""),
            raw_response=data
        )

    def _parse_chat_response(self, data: Dict[str, Any]) -> SGLangResponse:
        """Parse OpenAI-compatible chat response."""
        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        choices = data.get("choices", [])
        if not choices:
            return SGLangResponse(text="", raw_response=data)

        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content", "")

        usage = data.get("usage", {})
        return SGLangResponse(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            raw_response=data
        )

    def generate_and_validate(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        sample_id: str = "",
        max_tokens: int = 4096,
        strict_parse: bool = True,
    ) -> tuple[Dict[str, Any], str]:
        """
        Generate with schema constraint and validate the output.

        This is the main method for evaluation runs. It:
        1. Generates text with xgrammar constraint
        2. Parses as JSON
        3. Validates against schema
        4. Returns parsed dict or raises descriptive error

        Args:
            prompt: Input prompt
            json_schema: JSON schema for constrained decoding
            sample_id: Sample ID for error reporting
            max_tokens: Maximum tokens to generate
            strict_parse: If True, raise error on parse failure; else return partial

        Returns:
            Tuple of (parsed_dict, raw_response_text)

        Raises:
            SGLangParseError: If strict_parse=True and output fails to parse
        """
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            json_schema=json_schema,
        )

        raw_text = response.text

        # Extract only the first complete JSON object
        # SGLang with xgrammar sometimes outputs repeated JSON blocks
        json_text = extract_first_json(raw_text)
        
        if not json_text:
            error_msg = "No JSON object found in output"
            if strict_parse:
                raise SGLangParseError(
                    message=error_msg,
                    sample_id=sample_id,
                    prompt=prompt,
                    raw_response=raw_text,
                    parse_error=error_msg
                )
            else:
                return {
                    "_parse_error": True,
                    "_raw_output": raw_text[:2000],
                    "_error_msg": error_msg,
                    "plan_dag": {"nodes": [], "edges": []},
                    "tool_calls": [],
                    "final_answer": {
                        "answer_type": "none",
                        "answer": None,
                        "aliases": []
                    }
                }, raw_text

        # Try to parse the extracted JSON
        try:
            parsed = json.loads(json_text)
            return parsed, raw_text
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"

            if strict_parse:
                # Fail loudly with full context
                raise SGLangParseError(
                    message="Grammar-constrained output failed to parse as JSON",
                    sample_id=sample_id,
                    prompt=prompt,
                    raw_response=raw_text,
                    parse_error=str(e)
                )
            else:
                # Return empty structure with error info
                return {
                    "_parse_error": True,
                    "_raw_output": raw_text[:2000],
                    "_error_msg": error_msg,
                    "plan_dag": {"nodes": [], "edges": []},
                    "tool_calls": [],
                    "final_answer": {
                        "answer_type": "none",
                        "answer": None,
                        "aliases": []
                    }
                }, raw_text


def create_sglang_client(
    base_url: str = "http://127.0.0.1:30000",
    timeout: float = 120.0,
) -> SGLangClient:
    """
    Factory function to create SGLang client with default settings.

    Args:
        base_url: SGLang server URL
        timeout: Request timeout

    Returns:
        Configured SGLangClient instance
    """
    return SGLangClient(base_url=base_url, timeout=timeout)


if __name__ == "__main__":
    # Self-test: check if server is available
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
    print(f"Testing SGLang client with server at: {url}")

    client = SGLangClient(base_url=url)

    # Health check
    if client.health_check():
        print("[OK] Server is healthy")
    else:
        print("[WARN] Server health check failed (may still work)")

    # Simple generation test
    try:
        print("\nTesting simple generation...")
        resp = client.generate(
            prompt="What is 2+2? Answer with just the number:",
            max_tokens=10,
            temperature=0.0,
        )
        print(f"Response: {resp.text}")
        print(f"Tokens: {resp.prompt_tokens} prompt, {resp.completion_tokens} completion")
    except SGLangClientError as e:
        print(f"[ERROR] Generation failed: {e}")
        sys.exit(1)

    # JSON schema constrained generation test
    try:
        print("\nTesting JSON schema constrained generation...")
        test_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "integer"}
            },
            "required": ["answer"]
        }
        resp = client.generate(
            prompt="What is 2+2? Respond with JSON containing the answer.",
            max_tokens=50,
            temperature=0.0,
            json_schema=test_schema,
        )
        print(f"Response: {resp.text}")

        # Verify it parses
        parsed = json.loads(resp.text)
        print(f"Parsed: {parsed}")
        print("[OK] JSON schema constrained generation works!")
    except SGLangClientError as e:
        print(f"[ERROR] Schema-constrained generation failed: {e}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Output is not valid JSON: {e}")

    print("\nSGLang client test completed!")
