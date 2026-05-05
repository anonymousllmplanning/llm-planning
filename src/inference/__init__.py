# Inference module
"""
LLM Planning Inference Module.

All experiments now use the unified multi-stage pipeline:
  Stage 1: Abstract Planning (tool-agnostic)
  Stage 2: Tool Creation (evaluate and propose new tools)
  Stage 3: Refinement (concrete planning with tool calls)
  Stage 4-5: Execution + Answer Generation (optional, --mode answer)

Components:
- runner: Unified multi-stage inference engine (SGLang, Local, API backends)
- sglang_client: SGLang server client with xgrammar support
- schemas: Pydantic schema definitions
- eval_schemas: Evaluation-specific JSON schemas
- prompts: Multi-stage prompt builders
- tools: Tool execution engine
"""

# Main entry point - unified multistage pipeline
# from .runner import main

# Legacy single-stage pipeline has been deprecated
# All experiments now use the multi-stage approach

__all__ = [
    "main",
]
