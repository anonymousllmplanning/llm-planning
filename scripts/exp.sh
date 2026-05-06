#!/bin/bash
# =============================================================================
# Unified Experiment Runner for LLM Planning Framework
# =============================================================================
#
# Two evaluation modes:
#   - order:  Evaluates plan structure (Node F1, Edge F1, Tool F1) - all datasets
#   - answer: Evaluates final answer correctness (EM, Token F1) - GAIA only
#
# Three backends:
#   - sglang: SGLang server with JSON schema constraints (default)
#   - local:  HuggingFace transformers (no server needed)
#   - api:    Remote OpenAI-compatible API
#
# Usage:
#   ./scripts/exp.sh --dataset gaia_cat_A --mode answer --model qwen2.5-7b --limit 10
#   ./scripts/exp.sh --dataset gaia_cat_C --mode answer --model qwen2.5-7b --limit 10
#   ./scripts/exp.sh --dataset taskbench --mode order --all-models
#   ./scripts/exp.sh --help
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Derive BASE_DIR from script location (portable)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="${BASE_DIR}/logs"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$(which python 2>/dev/null || true)"
    if [[ -z "${PYTHON}" ]]; then
        PYTHON="$(which python3 2>/dev/null || true)"
    fi
    if [[ -z "${PYTHON}" ]]; then
        echo "[ERROR] Neither 'python' nor 'python3' found in PATH" >&2
        exit 1
    fi
fi
echo "[DEBUG] Using python interpreter: ${PYTHON}"

# Create per-run log directory
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${LOGS_DIR}/${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

# SGLang Server Settings (Hardened)
SGLANG_PORT=30000
SGLANG_DEFAULT_CONTEXT_LENGTH=8192  # Fallback, model-specific values below
SGLANG_MEM_FRACTION=0.85
SGLANG_CHUNKED_PREFILL=2048
SGLANG_STARTUP_TIMEOUT=1000
SGLANG_HEALTH_CHECK_INTERVAL=5

# Context Length Constraints (CRITICAL for GAIA long-context tasks)
# SAFETY_BASELINE = Prompt (3660) + Output (3000) + Buffer (540) = 7200
# This ensures sufficient context for GAIA prompts with embedded attachments
SAFETY_BASELINE=7200
# HARDWARE_LIMIT caps context to prevent 24GB VRAM OOM on RTX 3090/4090
# Reduced from 16384 to 8192 to significantly speed up KV cache allocation
HARDWARE_LIMIT=8192
# Max new tokens for generation (ensures enough space for GAIA long plans)
MAX_NEW_TOKENS=3000

# Remote API configuration. Set these via environment variables before running.
# OpenAI-compatible gateways should provide both LLM_API_BASE and LLM_API_KEY.
LLM_API_BASE="${LLM_API_BASE:-}"
LLM_API_KEY="${LLM_API_KEY:-}"
LLM_API_PROVIDER="${LLM_API_PROVIDER:-openai_compatible}"
# Disabled by default: final GAIA answer correctness is reviewed with the
# human-evaluation tables. Set ANSWER_VERIFIER_MODEL or --verifier_model
# explicitly to re-enable the optional LLM verifier for diagnostics.
ANSWER_VERIFIER_MODEL="${ANSWER_VERIFIER_MODEL:-}"
ANSWER_VERIFIER_API_BASE="${ANSWER_VERIFIER_API_BASE:-}"
ANSWER_VERIFIER_API_KEY="${ANSWER_VERIFIER_API_KEY:-}"
ANSWER_VERIFIER_API_PROVIDER="${ANSWER_VERIFIER_API_PROVIDER:-}"

normalize_api_base() {
    local base="${1:-}"
    base="${base%/}"
    if [[ -z "$base" ]]; then
        printf '%s' ""
        return
    fi
    printf '%s' "$base"
}

LLM_API_BASE="$(normalize_api_base "$LLM_API_BASE")"
TOOL_SCOPE="${TOOL_SCOPE:-record}"
DEFAULT_GAIA_DATA_ROOT="${BASE_DIR}/data/Augmented"
GAIA_DATA_ROOT="${GAIA_DATA_ROOT:-${DEFAULT_GAIA_DATA_ROOT}}"
GAIA_MODIFIED_GT_DIR="${GAIA_MODIFIED_GT_DIR:-${GAIA_DATA_ROOT}}"
GAIA_MODIFIED_GT_MODE="${GAIA_MODIFIED_GT_MODE:-auto}"

infer_provider_profile_from_model() {
    local model_name="${1:-}"
    case "${model_name,,}" in
        gpt-5*|gpt-4.1*|o3*|o4* )
            printf '%s' "openai"
            ;;
        claude-* )
            printf '%s' "anthropic"
            ;;
        gemini-* )
            printf '%s' "gemini"
            ;;
        * )
            printf '%s' "openai_compatible"
            ;;
    esac
}

configure_api_provider_profile() {
    local profile="${1:-auto}"
    local inferred="${2:-}"
    local explicit_profile="false"

    if [[ "$profile" == "auto" ]]; then
        profile="$inferred"
    else
        explicit_profile="true"
    fi

    case "$profile" in
        openai)
            LLM_API_PROVIDER="openai"
            if [[ "$explicit_profile" == "true" ]]; then
                LLM_API_BASE="https://api.openai.com/v1"
                if [[ -n "${OPENAI_API_KEY:-}" ]]; then
                    LLM_API_KEY="${OPENAI_API_KEY}"
                fi
            elif [[ -z "$LLM_API_BASE" ]]; then
                LLM_API_BASE="https://api.openai.com/v1"
            fi
            if [[ -z "$LLM_API_KEY" ]]; then
                LLM_API_KEY="${OPENAI_API_KEY:-}"
            fi
            ;;
        anthropic)
            LLM_API_PROVIDER="anthropic"
            if [[ "$explicit_profile" == "true" ]]; then
                LLM_API_BASE="https://api.anthropic.com"
                if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
                    LLM_API_KEY="${ANTHROPIC_API_KEY}"
                fi
            elif [[ -z "$LLM_API_BASE" ]]; then
                LLM_API_BASE="https://api.anthropic.com"
            fi
            if [[ -z "$LLM_API_KEY" ]]; then
                LLM_API_KEY="${ANTHROPIC_API_KEY:-}"
            fi
            ;;
        gemini)
            LLM_API_PROVIDER="gemini"
            if [[ "$explicit_profile" == "true" ]]; then
                LLM_API_BASE="https://generativelanguage.googleapis.com"
                if [[ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
                    LLM_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
                fi
            elif [[ -z "$LLM_API_BASE" ]]; then
                LLM_API_BASE="https://generativelanguage.googleapis.com"
            fi
            if [[ -z "$LLM_API_KEY" ]]; then
                LLM_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
            fi
            ;;
        openai_compatible|*)
            LLM_API_PROVIDER="openai_compatible"
            ;;
    esac

    LLM_API_BASE="$(normalize_api_base "$LLM_API_BASE")"
}

configure_verifier_provider_defaults() {
    if [[ -z "${ANSWER_VERIFIER_MODEL:-}" ]]; then
        return 0
    fi

    if [[ -z "$ANSWER_VERIFIER_API_BASE" && -n "$LLM_API_BASE" ]]; then
        ANSWER_VERIFIER_API_BASE="$(normalize_api_base "${LLM_API_BASE}")"
    fi
    if [[ -z "$ANSWER_VERIFIER_API_KEY" && -n "$LLM_API_KEY" ]]; then
        ANSWER_VERIFIER_API_KEY="${LLM_API_KEY}"
    fi
    if [[ -z "$ANSWER_VERIFIER_API_PROVIDER" && -n "$ANSWER_VERIFIER_API_BASE" ]]; then
        ANSWER_VERIFIER_API_PROVIDER="openai_compatible"
    fi
}

GAIA_CATEGORY_KEYS=("A" "B" "C" "D")
declare -A GAIA_CATEGORY_DIRS=(
    ["A"]="cat_A_text"
    ["B"]="cat_B_document"
    ["C"]="cat_C_vision"
    ["D"]="cat_D_audio"
)

GAIA_REMOTE_DEFAULT_MODELS=(
    "Mistral-Large-3-675B-Instruct-2512"
    "Llama-3.1-405B-Instruct-FP8"
    "Llama-3.3-70B-Instruct"
    "gemma-4-31B-it"
    "Mistral-Small-3.2-24B-Instruct-2506"
    "Llama-4-Maverick-17B-128E-Instruct-FP8"
    "gemma-3-12b-it"
)

OPENAI_REMOTE_MODELS=(
    "gpt-5.5"
    "gpt-5.4"
    "gpt-5.4-mini"
    "gpt-5.4-nano"
)

ANTHROPIC_REMOTE_MODELS=(
    "claude-opus-4-7"
    "claude-sonnet-4-6"
    "claude-haiku-4-5-20251001"
)

GEMINI_REMOTE_MODELS=(
    "gemini-3.1-pro-preview"
    "gemini-3-flash-preview"
    "gemini-3.1-flash-lite-preview"
    "gemini-2.5-pro"
    "gemini-2.5-flash"
    "gemini-2.5-flash-lite"
)

# Dataset paths (organized by category)
# Order-based evaluation: all datasets
# Answer-based evaluation: maintained GAIA Cat A/B/C/D family datasets only
declare -A DATASETS=()

# Datasets that support answer-based evaluation (have final_answer field)
declare -A ANSWER_MODE_DATASETS=()

# Attachments directories for GAIA answer-based mode
declare -A ATTACHMENTS_DIRS=()

# Output directories (unified structure)
declare -A OUTPUT_DIRS=()

# Async-ordering analysis source files
declare -A ORDERINGS_ASYNC_PATHS=(
    ["gaia_cat_A"]="${GAIA_DATA_ROOT}/DAGs/gaia_cat_A_async_plan.jsonl"
    ["gaia_cat_A_async"]="${GAIA_DATA_ROOT}/DAGs/gaia_cat_A_async_plan.jsonl"
)

for gaia_cat in "${GAIA_CATEGORY_KEYS[@]}"; do
    gaia_dir="${GAIA_CATEGORY_DIRS[$gaia_cat]}"
    DATASETS["gaia_cat_${gaia_cat}"]="${GAIA_DATA_ROOT}/${gaia_dir}/gaia.cat_${gaia_cat}.json"
    DATASETS["gaia_cat_${gaia_cat}_zh"]="${GAIA_DATA_ROOT}/${gaia_dir}/gaia.cat_${gaia_cat}_zh.json"

    ANSWER_MODE_DATASETS["gaia_cat_${gaia_cat}"]="true"
    ANSWER_MODE_DATASETS["gaia_cat_${gaia_cat}_zh"]="true"

    ATTACHMENTS_DIRS["gaia_cat_${gaia_cat}"]="${GAIA_DATA_ROOT}/${gaia_dir}/attachments"
    ATTACHMENTS_DIRS["gaia_cat_${gaia_cat}_zh"]="${GAIA_DATA_ROOT}/${gaia_dir}/attachments"

    OUTPUT_DIRS["gaia_cat_${gaia_cat}"]="${BASE_DIR}/organized_results/gaia/cat_${gaia_cat}"
    OUTPUT_DIRS["gaia_cat_${gaia_cat}_zh"]="${BASE_DIR}/organized_results/gaia/cat_${gaia_cat}_zh"
done

DATASETS["gaia_cat_A_async"]="${GAIA_DATA_ROOT}/DAGs/gaia_cat_A_async_plan.jsonl"
ANSWER_MODE_DATASETS["gaia_cat_A_async"]="true"
OUTPUT_DIRS["gaia_cat_A_async"]="${BASE_DIR}/organized_results/gaia/cat_A_async"

DATASETS["taskbench"]="${BASE_DIR}/data/Taskbench/unified_taskbench_order_chain500_dag500.jsonl"
DATASETS["taskbench_balanced_1000"]="${BASE_DIR}/data/Taskbench/unified_taskbench_order_chain500_dag500.jsonl"
DATASETS["ultratool_en"]="${BASE_DIR}/data/Ultratool/unified_ultratool_en_1000.jsonl"
DATASETS["ultratool_en_1000"]="${BASE_DIR}/data/Ultratool/unified_ultratool_en_1000.jsonl"

OUTPUT_DIRS["taskbench"]="${BASE_DIR}/organized_results/taskbench"
OUTPUT_DIRS["taskbench_balanced_1000"]="${BASE_DIR}/organized_results/taskbench_balanced_1000"
OUTPUT_DIRS["ultratool_en"]="${BASE_DIR}/organized_results/ultratool/en"
OUTPUT_DIRS["ultratool_en_1000"]="${BASE_DIR}/organized_results/ultratool/en_1000"

# =============================================================================
# Model Configuration
# =============================================================================


declare -A MODEL_ROUTING=(
    # ["mistral-7b"]="local"
    ["llama3.1-8b"]="local"
    ["gemma2-9b"]="local"
    ["qwen2.5-0.5b"]="local"
    ["qwen2.5-1.5b"]="local"
    ["qwen2.5-3b"]="local"
    ["qwen2.5-7b"]="local"
    ["qwen2.5-14b"]="local"
    ["qwen2.5-32b"]="local"
    # Remote models (API)
    ["Devstral-Small-2505"]="remote"
    ["Devstral-Small-2507"]="remote"
    ["Foundation-Sec-8B-Instruct"]="remote"
    ["Gemma-3-12b-it-TAIDE-distilled-v1"]="remote"
    ["Gemma-3-TAIDE-12b-Chat"]="remote"
    ["Google-Gemma-3-27B"]="remote"
    ["Granite-3.1-8B-Instruct"]="remote"
    ["Granite-Guardian-3.1-8B"]="remote"
    ["Llama-3.1-405B-Instruct-FP8"]="remote"
    ["Llama-3.1-70B"]="remote"
    ["Llama-3.1-8B-Instruct"]="remote"
    ["Llama-3.1-Nemotron-70B-Instruct"]="remote"
    ["Llama-3.1-Nemotron-70B-Instruct-Gaudi3"]="remote"
    ["Llama-3.1-TAIDE-LX-8B-Chat"]="remote"
    ["Llama-3.2-90B-Vision-Instruct"]="remote"
    ["Llama-3.3-70B-Instruct"]="remote"
    ["Llama-3.3-70B-Instruct-Gaudi3"]="remote"
    ["Llama-3.3-70B-Instruct-MI210"]="remote"
    ["Llama-3.3-Nemotron-Super-49B-v1"]="remote"
    ["Llama-4-Maverick-17B-128E-Instruct-FP8"]="remote"
    ["Llama-4-Scout-17B-16E-Instruct-FP8"]="remote"
    ["Llama3-TAIDE-LX-70B-Chat"]="remote"
    ["Llama3-TAIDE-LX-8B-Chat-Alpha1"]="remote"
    ["Llama3.1-8B-NSTC"]="remote"
    ["Magistral-Small-2506"]="remote"
    ["Microsoft-Phi-4"]="remote"
    ["Microsoft-Phi-4-multimodal-instruct"]="remote"
    ["Ministral-3-14B-Instruct-2512"]="remote"
    ["Ministral-3-8B-Instruct-2512"]="remote"
    ["Ministral-8B-Instruct-2410"]="remote"
    ["Mistral-Large-3-675B-Instruct-2512"]="remote"
    ["Mistral-Small-24B-Instruct-2501"]="remote"
    ["Mistral-Small-3.1-24B-Instruct-2503"]="remote"
    ["Mistral-Small-3.2-24B-Instruct-2506"]="remote"
    ["Mistral-Small-3.2-24B-Instruct-2506-CS-0625"]="remote"
    ["Mistral-Small-3.2-24B-Instruct-2506-CS-3in1-0915"]="remote"
    ["NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"]="remote"
    ["NVIDIA-Nemotron-3-Super-120B-A12B"]="remote"
    ["Phi-3.5-TAIDE-Instruct"]="remote"
    ["Phi-4-Reasoning-Plus"]="remote"
    ["Rainbell_v2"]="remote"
    ["TAIDE-LX-70B-Chat"]="remote"
    ["TAIDE-LX-7B-Chat"]="remote"
    ["TAIDE/Llava-Phi-3.5"]="remote"
    ["TAIDE/b.12.0.0"]="remote"
    ["gemma-3-12b-it"]="remote"
    ["gemma-3-12b_raft-v1.2.1"]="remote"
    ["gemma-3-taide-12b_alpha_v1.0"]="remote"
    ["gemma-4-31B-it"]="remote"
    ["gemma-4-E4B-it"]="remote"
    ["gpt-oss-120b"]="remote"
    ["gpt-oss-20b"]="remote"
    ["gpt-5.5"]="remote"
    ["gpt-5.4"]="remote"
    ["gpt-5.4-mini"]="remote"
    ["gpt-5.4-nano"]="remote"
    ["gpt-5"]="remote"
    ["claude-opus-4-7"]="remote"
    ["claude-sonnet-4-6"]="remote"
    ["claude-haiku-4-5-20251001"]="remote"
    ["claude-haiku-4-5"]="remote"
    ["claude-opus-4-1-20250805"]="remote"
    ["gemini-3.1-pro-preview"]="remote"
    ["gemini-3-flash-preview"]="remote"
    ["gemini-3.1-flash-lite-preview"]="remote"
    ["gemini-2.5-pro"]="remote"
    ["gemini-2.5-flash"]="remote"
    ["gemini-2.5-flash-lite"]="remote"
    ["llama-3.1-8B-m1-raft-v1-ep5"]="remote"
    ["llama-3.1-8B-m1-raft-v1.1.4-ep5"]="remote"
    ["llama-3.1-8B-m1-raft-v2-ep5"]="remote"
    ["medgemma-27b-it"]="remote"
    ["medgemma-27b-text-it"]="remote"
    ["phi-3.5-mini-instruct_zhtw_ld1_hq3.1_b8.3-p3_st-task-1-2-3-v3_e4_ORPO_1103-1019_fix3"]="remote"
)

# HuggingFace model IDs for local models (verified downloaded)
declare -A MODEL_HF_IDS=(
    ["qwen2.5-0.5b"]="Qwen/Qwen2.5-0.5B-Instruct"
    ["qwen2.5-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
    ["qwen2.5-3b"]="Qwen/Qwen2.5-3B-Instruct"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B-Instruct"
    ["qwen2.5-14b"]="Qwen/Qwen2.5-14B-Instruct"
    ["qwen2.5-32b"]="Qwen/Qwen2.5-32B-Instruct"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["llama3.1-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["gemma2-9b"]="google/gemma-2-9b-it"
)

# Model-specific DEFAULT context lengths (optimized for 24GB VRAM)
declare -A MODEL_CONTEXT_LENGTHS=(
    ["qwen2.5-0.5b"]=8192
    ["qwen2.5-1.5b"]=8192
    ["qwen2.5-3b"]=8192
    ["qwen2.5-7b"]=8192
    ["qwen2.5-14b"]=8192
    ["qwen2.5-32b"]=8192
    ["mistral-7b"]=8192
    ["llama3.1-8b"]=8192
    ["gemma2-9b"]=8192
)


# Default model sets (Editable Candidate Lists)
# You can add/remove models here to change the default sets run by --all-models, --local-only, etc.

LOCAL_MODELS=(
    # "qwen2.5-0.5b"
    # "qwen2.5-1.5b"
    # "qwen2.5-3b"
    # "qwen2.5-7b"
    # "qwen2.5-14b"
    # "qwen2.5-32b"
    # "llama3.1-8b"
    # "gemma2-9b"
    # "mistral-7b"
)

REMOTE_MODELS=(
    # Main GAIA seven-model benchmark cohort
    "Mistral-Large-3-675B-Instruct-2512"
    "Llama-3.1-405B-Instruct-FP8"
    "Llama-3.3-70B-Instruct"
    "gemma-4-31B-it"
    "Mistral-Small-3.2-24B-Instruct-2506"
    "Llama-4-Maverick-17B-128E-Instruct-FP8"
    "gemma-3-12b-it"
)

# =============================================================================
# State Variables
# =============================================================================

SERVER_PID=""
CURRENT_LOG_FILE=""
CLEANUP_DONE=false
SGLANG_CLEANUP_NEEDED=false

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo "$msg"
    if [[ -n "${CURRENT_LOG_FILE:-}" ]]; then
        echo "$msg" >> "$CURRENT_LOG_FILE"
    fi
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1"
    echo "$msg" >&2
    if [[ -n "${CURRENT_LOG_FILE:-}" ]]; then
        echo "$msg" >> "$CURRENT_LOG_FILE"
    fi
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo "$msg" >&2
    if [[ -n "${CURRENT_LOG_FILE:-}" ]]; then
        echo "$msg" >> "$CURRENT_LOG_FILE"
    fi
}

log_header() {
    echo ""
    echo "============================================================================"
    echo " $1"
    echo "============================================================================"
    if [[ -n "${CURRENT_LOG_FILE:-}" ]]; then
        {
            echo ""
            echo "============================================================================"
            echo " $1"
            echo "============================================================================"
        } >> "$CURRENT_LOG_FILE"
    fi
}

load_zh_profile_list() {
    "$PYTHON" -m src.config.zh_profile "$@"
}

get_port_pids() {
    local port="$1"
    if ! command -v lsof &>/dev/null; then
        return 0
    fi
    if command -v timeout &>/dev/null; then
        timeout 3s lsof -ti:"${port}" 2>/dev/null || true
    else
        lsof -ti:"${port}" 2>/dev/null || true
    fi
}

get_gaia_modified_gt_category() {
    local dataset="$1"
    case "$dataset" in
        gaia_cat_A*) printf '%s' "A" ;;
        gaia_cat_B*) printf '%s' "B" ;;
        gaia_cat_C*) printf '%s' "C" ;;
        gaia_cat_D*) printf '%s' "D" ;;
        *) return 1 ;;
    esac
}

get_gaia_modified_gt_dataset() {
    local dataset="$1"
    local cat
    cat="$(get_gaia_modified_gt_category "$dataset")" || return 1
    local gaia_dir="${GAIA_CATEGORY_DIRS[$cat]}"
    local suffix=""
    if [[ "$dataset" == *_zh ]]; then
        suffix="_zh"
    fi
    printf '%s' "${GAIA_MODIFIED_GT_DIR}/${gaia_dir}/gaia.cat_${cat}${suffix}.json"
}

get_gaia_data_async_dataset() {
    local dataset="$1"
    local cat
    cat="$(get_gaia_modified_gt_category "$dataset")" || return 1
    printf '%s' "${GAIA_DATA_ROOT}/DAGs/gaia_cat_${cat}_async_plan.jsonl"
}

get_gaia_modified_gt_async_dataset() {
    local dataset="$1"
    local cat
    cat="$(get_gaia_modified_gt_category "$dataset")" || return 1
    printf '%s' "${GAIA_MODIFIED_GT_DIR}/DAGs/gaia_cat_${cat}_async_plan.jsonl"
}

# =============================================================================
# Cleanup Function (Critical for tmux sessions)
# =============================================================================

cleanup() {
    # Prevent double cleanup
    if $CLEANUP_DONE; then
        return 0
    fi
    CLEANUP_DONE=true

    if [[ "$SGLANG_CLEANUP_NEEDED" != "true" ]] && { [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; }; then
        return 0
    fi

    echo ""
    log_warn "Cleanup triggered (signal received or script exit)"

    # Kill SGLang server if running
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping SGLang server (PID: $SERVER_PID)..."

        # Try graceful shutdown first
        kill -TERM "$SERVER_PID" 2>/dev/null || true

        # Wait up to 10 seconds for graceful shutdown
        local wait_count=0
        while kill -0 "$SERVER_PID" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # Force kill if still running
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log_warn "Force killing SGLang server..."
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi

        log_info "SGLang server stopped."
    fi

    # VRAM Cleanup: Kill ALL processes on port 30000 (critical for tmux sessions)
    log_info "Performing comprehensive VRAM cleanup on port ${SGLANG_PORT}..."

    # Method 1: Kill by port using lsof
    local orphan_pids
    orphan_pids=$(get_port_pids "${SGLANG_PORT}")
    if [[ -n "$orphan_pids" ]]; then
        log_warn "Killing orphaned processes on port ${SGLANG_PORT}: $orphan_pids"
        echo "$orphan_pids" | xargs -r kill -KILL 2>/dev/null || true
        sleep 1
    fi

    # Method 2: Kill any remaining sglang python processes (backup)
    local sglang_pids
    sglang_pids=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
    if [[ -n "$sglang_pids" ]]; then
        log_warn "Killing remaining sglang processes: $sglang_pids"
        echo "$sglang_pids" | xargs -r kill -KILL 2>/dev/null || true
        sleep 1
    fi

    # Method 3: Final check with fuser (most aggressive)
    if command -v fuser &>/dev/null; then
        if command -v timeout &>/dev/null; then
            timeout 3s fuser -k ${SGLANG_PORT}/tcp 2>/dev/null || true
        else
            fuser -k ${SGLANG_PORT}/tcp 2>/dev/null || true
        fi
    fi

    # Verify port is free
    if [[ -n "$(get_port_pids "${SGLANG_PORT}")" ]]; then
        log_error "WARNING: Port ${SGLANG_PORT} still in use after cleanup!"
    else
        log_info "Port ${SGLANG_PORT} is now free."
    fi

    log_info "Cleanup complete."
}

# Set up trap for all exit scenarios (critical for tmux)
trap cleanup EXIT INT TERM HUP QUIT

# =============================================================================
# SGLang Server Management
# =============================================================================

wait_for_server() {
    local url="$1"
    local timeout="$2"
    local elapsed=0

    log_info "Waiting for SGLang server at ${url}..."

    while [[ $elapsed -lt $timeout ]]; do
        # Poll /health endpoint - check HTTP status code (not response body)
        # SGLang may return 200 OK with empty body when healthy
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null || echo "000")

        if [[ "$http_code" == "200" ]]; then
            log_info "Server is ready! (HTTP 200, took ${elapsed}s)"
            return 0
        fi

        sleep "$SGLANG_HEALTH_CHECK_INTERVAL"
        elapsed=$((elapsed + SGLANG_HEALTH_CHECK_INTERVAL))

        # Check if server process died
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_error "Server process died during startup"
            return 1
        fi

        echo "  ... waiting (${elapsed}s / ${timeout}s)"
    done

    log_error "Server did not start within ${timeout} seconds"
    return 1
}

calculate_context_length() {
    # Calculate optimal context length for a model
    # ctx_len = max(MODEL_DEFAULT_LIMIT, SAFETY_BASELINE), capped at HARDWARE_LIMIT
    local model_name="$1"
    local model_default="${MODEL_CONTEXT_LENGTHS[$model_name]:-$SGLANG_DEFAULT_CONTEXT_LENGTH}"

    # Step 1: Ensure at least SAFETY_BASELINE for GAIA prompts
    local ctx_len=$model_default
    if [[ $ctx_len -lt $SAFETY_BASELINE ]]; then
        ctx_len=$SAFETY_BASELINE
        log_warn "Model $model_name default ($model_default) < SAFETY_BASELINE ($SAFETY_BASELINE), using $ctx_len"
    fi

    # Step 2: Cap at HARDWARE_LIMIT to prevent OOM on 24GB VRAM
    if [[ $ctx_len -gt $HARDWARE_LIMIT ]]; then
        ctx_len=$HARDWARE_LIMIT
        log_warn "Context length capped at HARDWARE_LIMIT ($HARDWARE_LIMIT) to prevent OOM"
    fi

    echo "$ctx_len"
}

start_sglang_server() {
    local model_name="$1"
    local hf_id="${MODEL_HF_IDS[$model_name]:-}"
    SGLANG_CLEANUP_NEEDED=true

    if [[ -z "$hf_id" ]]; then
        log_error "Unknown model: $model_name (no HuggingFace ID configured)"
        return 1
    fi

    # Dynamic context length calculation with GAIA safety constraints
    local model_default="${MODEL_CONTEXT_LENGTHS[$model_name]:-$SGLANG_DEFAULT_CONTEXT_LENGTH}"
    local context_length
    context_length=$(calculate_context_length "$model_name")

    log_header "Starting SGLang Server for $model_name"
    log_info "HuggingFace ID: $hf_id"
    log_info "Port: $SGLANG_PORT"
    log_info "Context Length: $context_length (model default: $model_default, safety: $SAFETY_BASELINE, hw_limit: $HARDWARE_LIMIT)"

    # Check if port is already in use
    local existing_pids
    existing_pids=$(get_port_pids "${SGLANG_PORT}")
    if [[ -n "$existing_pids" ]]; then
        log_warn "Port ${SGLANG_PORT} is in use. Attempting to free it..."
        echo "$existing_pids" | xargs -r kill -KILL 2>/dev/null || true
        sleep 2
    fi

    # Build server command with hardened settings (optimized for fast startup)
    local server_cmd="$PYTHON -m sglang.launch_server"
    server_cmd+=" --model-path $hf_id"
    server_cmd+=" --port $SGLANG_PORT"
    server_cmd+=" --host 0.0.0.0"
    server_cmd+=" --context-length $context_length"
    server_cmd+=" --mem-fraction-static $SGLANG_MEM_FRACTION"
    server_cmd+=" --chunked-prefill-size $SGLANG_CHUNKED_PREFILL"
    # Reduce CUDA graph batch sizes to speed up startup (default 256 -> 64)
    server_cmd+=" --cuda-graph-max-bs 64"
    # Note: NOT using --disable-radix-cache (keep enabled for speed)
    # Note: NOT using --kv-cache-dtype fp8 as it can cause precision issues

    # Create server log file
    local server_log="${RUN_LOG_DIR}/sglang_server_${model_name}.log"

    log_info "Server command: $server_cmd"
    log_info "Server log: $server_log"

    # Start server in background
    $server_cmd > "$server_log" 2>&1 &
    SERVER_PID=$!

    log_info "Server PID: $SERVER_PID"

    # Wait for server to be ready
    if ! wait_for_server "http://127.0.0.1:${SGLANG_PORT}" "$SGLANG_STARTUP_TIMEOUT"; then
        log_error "Failed to start SGLang server. Last 50 lines of log:"
        tail -50 "$server_log"
        SERVER_PID=""
        return 1
    fi

    return 0
}

stop_sglang_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping SGLang server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true

        local wait_count=0
        while kill -0 "$SERVER_PID" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log_warn "Force killing server..."
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi

        log_info "Server stopped."
        SERVER_PID=""
        SGLANG_CLEANUP_NEEDED=false
    fi
}

# =============================================================================
# Inference Functions
# =============================================================================

run_inference_local() {
    local model_name="$1"
    local dataset="$2"
    local input_file="$3"
    local output_file="$4"
    local limit="$5"
    local resume="$6"
    local no_schema="$7"
    local backend="${8:-local}"  # Default to sglang if not specified
    local mode="${9:-order}"  # New argument: order or answer
    local debug_mode="${10:-false}" # New argument: debug mode
    local max_turns="${11:-15}"

    log_info "Running LOCAL inference"
    log_info "  Model: $model_name"
    log_info "  Dataset: $dataset"
    log_info "  Backend: $backend"
    log_info "  Pipeline: multi_stage (default)"
    log_info "  Schema Constraint: $([ "$no_schema" == "true" ] && echo "DISABLED" || echo "enabled")"
    log_info "  Stage3 abs DAG reference: disabled (Stage 1 planning intent is evaluated separately)"

    # Start SGLang server ONLY if backend is sglang
    if [[ "$backend" == "sglang" ]]; then
        if ! start_sglang_server "$model_name"; then
            log_error "Failed to start SGLang server for $model_name, skipping..."
            return 1
        fi
    fi

    # Build inference command
    local runner_module="src.inference.runner"
    # All experiments now use the unified multi-stage pipeline in runner.py
    local inference_cmd="cd ${BASE_DIR} && $PYTHON -m $runner_module"
    inference_cmd+=" --unified_path \"$input_file\""
    inference_cmd+=" --output \"$output_file\""
    inference_cmd+=" --model_name \"$model_name\""
    inference_cmd+=" --mode \"$mode\""
    inference_cmd+=" --dataset \"$dataset\""

    if [[ "$backend" == "sglang" ]]; then
        inference_cmd+=" --backend sglang"
        inference_cmd+=" --sglang_url http://127.0.0.1:${SGLANG_PORT}"
    elif [[ "$backend" == "hf" || "$backend" == "local" ]]; then
        inference_cmd+=" --backend local"
    else
        log_error "Unknown backend: $backend"
        return 1
    fi

    # pipeline_runner doesn't need these args (they're single_stage specific)
    # if [[ "$pipeline" == "multi_stage" ]]; then
    #     inference_cmd+=" --grammar_mode strict_eval_schema"
    #     inference_cmd+=" --save_raw"
    # fi

    if [[ -n "$limit" && "$limit" != "0" ]]; then
        inference_cmd+=" --limit $limit"
    fi

    if [[ "$resume" == "true" ]]; then
        inference_cmd+=" --resume"
    fi

    # Pass schema constraint flag
    if [[ "$no_schema" == "true" ]]; then
        inference_cmd+=" --no_schema_constraint"
    fi

    # Pass debug mode flag
    if [[ "$debug_mode" == "true" ]]; then
        inference_cmd+=" --debug"
    fi

    if [[ "$mode" == "answer" && -n "$max_turns" ]]; then
        inference_cmd+=" --max_turns $max_turns"
    fi
    inference_cmd+=" --tool_scope \"$TOOL_SCOPE\""

    log_info "Inference command: $inference_cmd"

    # Run inference with output to both console and log
    eval "$inference_cmd" 2>&1 | tee -a "$CURRENT_LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    # Stop server after inference (only if sglang backend)
    if [[ "$backend" == "sglang" ]]; then
        stop_sglang_server
    fi

    return $exit_code
}

run_inference_remote() {
    local model_name="$1"
    local dataset="$2"
    local input_file="$3"
    local output_file="$4"
    local limit="$5"
    local resume="$6"
    local no_schema="$7" # Added for consistency
    local mode="${8:-order}" # (Wait, remote call logic below passes backend at 6?)
    # Let's check call site for remote...
    # remote call: run_inference_remote "$model_name" "$dataset" "$input_file" "$output_file" "$limit" "$resume" "$no_schema" "$mode" "$debug_mode"
    # Args: 1..6, 7=no_schema, 8=mode, 9=debug
    local debug_mode="${9:-false}"
    local max_turns="${10:-15}"

    log_info "Running REMOTE inference"
    log_info "  Model: $model_name"
    log_info "  Dataset: $dataset"
    log_info "  Provider: $LLM_API_PROVIDER"
    log_info "  API Base: $LLM_API_BASE"
    log_info "  API Key: $( [[ -n "$LLM_API_KEY" ]] && printf 'set (redacted)' || printf 'not set' )"
    log_info "  Debug Mode: $debug_mode"
    log_info "  Stage3 abs DAG reference: disabled (Stage 1 planning intent is evaluated separately)"

    # Export API credentials for the Python script
    export LLM_API_PROVIDER="$LLM_API_PROVIDER"
    export LLM_API_BASE="$LLM_API_BASE"
    export LLM_API_KEY="$LLM_API_KEY"

    # Build inference command
    local inference_cmd="cd ${BASE_DIR} && $PYTHON -m src.inference.runner"
    inference_cmd+=" --unified_path \"$input_file\""
    inference_cmd+=" --output \"$output_file\""
    inference_cmd+=" --model_name \"$model_name\""
    inference_cmd+=" --backend api"
    inference_cmd+=" --mode \"$mode\""
    inference_cmd+=" --dataset \"$dataset\""
    inference_cmd+=" --save_raw" # Remote API often saves raw output

    if [[ -n "$limit" && "$limit" != "0" ]]; then
        inference_cmd+=" --limit $limit"
    fi

    if [[ "$resume" == "true" ]]; then
        inference_cmd+=" --resume"
    fi

    if [[ "$no_schema" == "true" ]]; then
        inference_cmd+=" --no_schema_constraint"
    fi

    if [[ "$debug_mode" == "true" ]]; then
        inference_cmd+=" --debug"
    fi

    if [[ "$mode" == "answer" && -n "$max_turns" ]]; then
        inference_cmd+=" --max_turns $max_turns"
    fi
    inference_cmd+=" --tool_scope \"$TOOL_SCOPE\""

    log_info "Inference command: $inference_cmd"

    # Run inference with output to both console and log
    eval "$inference_cmd" 2>&1 | tee -a "$CURRENT_LOG_FILE"
    return ${PIPESTATUS[0]}
}

run_inference() {
    local model_name="$1"
    local dataset="$2"

    local input_file="${DATASETS[$dataset]:-}"
    if [[ -z "$input_file" || ! -f "$input_file" ]]; then
        log_error "Input file not found for dataset $dataset: $input_file"
        return 1
    fi

    # Determine Output File
    local output_dir="$3"
    local limit="$4"
    local resume="$5"
    local no_schema="$6"
    local backend_arg="${7:-sglang}"
    local mode="${8:-order}"
    local mode_suffix="${9:-}"
    local debug_mode="${10:-false}"
    local max_turns="${11:-15}"

    if [ -z "$output_dir" ]; then
        log_error "Error: Output directory not provided for dataset '$dataset'"
        return 1
    fi

    if [[ "$mode" == "answer" ]]; then
        output_dir="${output_dir}/${model_name}"
    fi

    mkdir -p "$output_dir"
    local output_file="${output_dir}/unified.${model_name}${mode_suffix}.jsonl"

    # IDEMPOTENCY CHECK: Skip if output already exists (unless resume mode)
    if [[ -f "$output_file" && "$resume" != "true" ]]; then
        local line_count
        line_count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        if [[ "$line_count" -gt 0 ]]; then
            log_info "SKIPPING: Output already exists with $line_count samples: $output_file"
            log_info "  Use --resume to continue or delete the file to re-run"
            return 0
        fi
    fi

    # Determine routing
    local routing="${MODEL_ROUTING[$model_name]:-local}"
    if [[ "$backend_arg" == "api" ]]; then
        routing="remote"
    fi

    if [[ "$routing" == "local" ]]; then
        run_inference_local "$model_name" "$dataset" "$input_file" "$output_file" "$limit" "$resume" "$no_schema" "$backend_arg" "$mode" "$debug_mode" "$max_turns"
    else
        # Remote call signature: model dataset input output limit resume no_schema mode debug
        run_inference_remote "$model_name" "$dataset" "$input_file" "$output_file" "$limit" "$resume" "$no_schema" "$mode" "$debug_mode" "$max_turns"
    fi
}

# =============================================================================
# Evaluation and Figures
# =============================================================================

run_evaluation() {
    local dataset="$1"
    local model_name="$2"
    local results_dir="$3"

    if [[ -z "$results_dir" ]]; then
         results_dir="${OUTPUT_DIRS[$dataset]}"
    fi
    # Logic to find prediction file (handle model subdirectory for answer mode)
    local pred_file="${results_dir}/unified.${model_name}.jsonl"
    
    # If not found in root, check model subdirectory (new structure)
    if [[ ! -f "$pred_file" ]]; then
        local subdir_pred_file="${results_dir}/${model_name}/unified.${model_name}.jsonl"
        if [[ -f "$subdir_pred_file" ]]; then
             pred_file="$subdir_pred_file"
        fi
    fi

    if [[ ! -f "$pred_file" ]]; then
        log_warn "Prediction file not found: $pred_file, skipping evaluation"
        return 1
    fi

    local csv_file="${results_dir}/per_sample.${model_name}.csv"
    local summary_file="${results_dir}/summary.${model_name}.json"
    local gold_dataset="${DATASETS[$dataset]:-}"
    local gaia_async_dataset=""
    if [[ "$dataset" == gaia_cat_* ]]; then
        gaia_async_dataset="$(get_gaia_data_async_dataset "$dataset" 2>/dev/null || true)"
    fi
    if [[ "$dataset" == gaia_cat_* && "${GAIA_MODIFIED_GT_MODE}" != "off" ]]; then
        local modified_gt_dataset
        modified_gt_dataset="$(get_gaia_modified_gt_dataset "$dataset" 2>/dev/null || true)"
        if [[ -n "$modified_gt_dataset" && -f "$modified_gt_dataset" ]]; then
            gold_dataset="$modified_gt_dataset"
            log_info "Using GAIA modified-GT scoring dataset: $gold_dataset"
            gaia_async_dataset="$(get_gaia_modified_gt_async_dataset "$dataset" 2>/dev/null || true)"
        elif [[ "${GAIA_MODIFIED_GT_MODE}" == "require" ]]; then
            log_error "Required GAIA modified-GT dataset not found for $dataset under $GAIA_MODIFIED_GT_DIR"
            return 1
        fi
    fi

    log_info "Running evaluation for $model_name on $dataset"

    local eval_cmd="cd ${BASE_DIR} && $PYTHON -m src.evaluation.runner"
    eval_cmd+=" --input $pred_file"
    eval_cmd+=" --output_csv $csv_file"
    eval_cmd+=" --output_summary $summary_file"
    if [[ -n "$gold_dataset" && -f "$gold_dataset" ]]; then
        eval_cmd+=" --gold_dataset $gold_dataset"
    fi
    if [[ "${mode:-order}" == "answer" && -n "${ANSWER_VERIFIER_MODEL:-}" ]]; then
        eval_cmd+=" --verifier_model $ANSWER_VERIFIER_MODEL"
    fi
    local plan_source_arg="${PLAN_SOURCE:-stage1}"
    eval_cmd+=" --plan_source $plan_source_arg"
    if [[ "$dataset" == gaia_cat_* ]]; then
        local gaia_reference_mode="${GAIA_REFERENCE_MODE:-auto}"
        eval_cmd+=" --gaia_reference_mode $gaia_reference_mode"
        if [[ -n "$gaia_async_dataset" && -f "$gaia_async_dataset" ]]; then
            eval_cmd+=" --gaia_async_dataset $gaia_async_dataset"
            log_info "Using GAIA async reference: $gaia_async_dataset"
        fi
    fi

    eval "$eval_cmd" 2>&1 | tee -a "$CURRENT_LOG_FILE"
    return ${PIPESTATUS[0]}
}

run_figures() {
    local dataset="$1"
    local results_dir="$2"

    if [[ -z "$results_dir" ]]; then
         results_dir="${OUTPUT_DIRS[$dataset]}"
    fi
    local figures_dir="${results_dir}/figures"
    mkdir -p "$figures_dir"

    # Check if there are summary files
    if ! ls "${results_dir}"/summary.*.json > /dev/null 2>&1; then
        log_warn "No summary files found in $results_dir, skipping figures"
        return 1
    fi

    log_info "Generating figures for $dataset"

    local fig_cmd="cd ${BASE_DIR} && $PYTHON -m src.visualization.figures"
    fig_cmd+=" --results_dir $results_dir"
    fig_cmd+=" --output_dir $figures_dir"

    eval "$fig_cmd" 2>&1 | tee -a "$CURRENT_LOG_FILE"
    return ${PIPESTATUS[0]}
}

run_orderings_analysis() {
    local dataset="$1"
    local model_name="$2"
    local results_dir="$3"

    local async_plan_path="${ORDERINGS_ASYNC_PATHS[$dataset]:-}"
    if [[ -z "$async_plan_path" || ! -f "$async_plan_path" ]]; then
        log_warn "Ordering analysis not configured for dataset: $dataset"
        return 1
    fi

    local pred_file="${results_dir}/unified.${model_name}.jsonl"
    if [[ ! -f "$pred_file" ]]; then
        local subdir_pred_file="${results_dir}/${model_name}/unified.${model_name}.jsonl"
        if [[ -f "$subdir_pred_file" ]]; then
            pred_file="$subdir_pred_file"
        fi
    fi

    if [[ ! -f "$pred_file" ]]; then
        log_warn "Prediction file not found for ordering analysis: $pred_file"
        return 1
    fi

    local pred_dir
    pred_dir="$(dirname "$pred_file")"
    local output_dir="${pred_dir}/orderings_analysis.${model_name}"
    mkdir -p "$output_dir"

    log_info "Running ordering-sensitivity analysis for $model_name on $dataset"

    local orderings_cmd="cd ${BASE_DIR} && $PYTHON -m src.evaluation.dags_analysis"
    orderings_cmd+=" --async_plan_path \"$async_plan_path\""
    orderings_cmd+=" --prediction_path \"$pred_file\""
    orderings_cmd+=" --output_dir \"$output_dir\""

    eval "$orderings_cmd" 2>&1 | tee -a "$CURRENT_LOG_FILE"
    return ${PIPESTATUS[0]}
}

# =============================================================================
# Usage and Help
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified Experiment Runner for LLM Planning Framework

Evaluation Modes:
  --mode MODE            Evaluation mode: order (default) or answer
                         - order: Evaluates plan structure (Node F1, Edge F1, Tool F1)
                         - answer: Evaluates final answer (EM, Token F1) - GAIA only
  --plan-source SOURCE   Planning field to evaluate: stage1 (default) or stage3
                         - stage1: evaluate pred.abs_plan_dag
                         - stage3: evaluate pred.plan_dag
  --stage3-abs-dag-reference
                         Deprecated no-op kept for old scripts. Stage 3 no
                         longer receives pred.abs_plan_dag; planning is
                         evaluated from Stage 1 pred.abs_plan_dag.

Model Selection:
  --model MODEL          Run specific model (can be repeated)
  --all-models           Run all configured models (local + remote)
  --openai-models        Use the curated OpenAI remote model list
  --claude-models        Use the curated Claude remote model list
  --gemini-models        Use the curated Gemini remote model list
  --local-only           Run only local models (SGLang)
  --remote-only, --api-only  Run only remote models (OpenAI-compatible API)

Dataset Selection:
  --dataset DATASET      Run on specific dataset (default: gaia_cat_A)
  --all-datasets         Run on all configured datasets

Execution Control:
  --limit N              Limit to first N samples
  --resume               Resume from existing output
  --run N                Run experiment N times for std calculation (default: 1)
  --aggregate            Aggregate results across multiple runs
  --max-turns N          Maximum iterative tool-execution turns in answer mode
                         (default: 15)
  --tool-scope SCOPE     Prompt-visible tools: record (default) or global.
                         global exposes the full executable tools.py library
                         for ablation runs without changing the default setup.
  --verifier_model MODEL Optional answer-mode LLM-as-a-judge verifier
                         (disabled by default; human evaluation is the main
                         GAIA answer audit path)
  --orderings            Run async ordering-sensitivity + efficiency analysis when available
  --inference-only       Run only inference (skip eval/figures)
  --eval-only            Run only evaluation (skip inference/figures)
  --figures-only         Run only figure generation
  --debug                Enable debug mode for inference (more verbose logging)
  --analysis DIR         Run correlation analysis on results directory

Backend Options:
  --backend BACKEND      Backend type: sglang (default), local/hf, api
  --provider-profile P   Remote provider profile: auto (default), openai,
                         anthropic, gemini, openai_compatible
  --pipeline PIPELINE    Pipeline type: multi_stage (default), multi_stage
  --port PORT            SGLang server port (default: $SGLANG_PORT)
  --context-length N     Context length (default: $SGLANG_DEFAULT_CONTEXT_LENGTH)

General:
  --verbose              Show detailed output
  --dry-run              Print commands without executing
  -h, --help             Show this help

Environment Variables:
  LLM_API_BASE          API server URL (required for remote models)
  LLM_API_KEY           API key for authentication (required for remote models)
  LLM_API_PROVIDER      Remote provider profile (openai_compatible by default)
  OPENAI_API_KEY        OpenAI key used when --provider-profile openai
  ANTHROPIC_API_KEY     Anthropic key used when --provider-profile anthropic
  GEMINI_API_KEY        Gemini key used when --provider-profile gemini
  GOOGLE_API_KEY        Alias of GEMINI_API_KEY
  ANSWER_VERIFIER_MODEL   Optional LLM-as-a-judge verifier model. Leave unset
                        for the default human-evaluation workflow.
  ANSWER_VERIFIER_API_BASE / ANSWER_VERIFIER_API_KEY / ANSWER_VERIFIER_API_PROVIDER
                        Optional dedicated verifier endpoint used only when
                        ANSWER_VERIFIER_MODEL or --verifier_model is set.
  PLAN_SOURCE           Default planning field for evaluation: stage1 or stage3
  GAIA_DATA_ROOT        Canonical GAIA data root used for inference input
                        (default: data/Augmented generated locally by
                        scripts/prepare_gaia_from_official.py).
  GAIA_REFERENCE_MODE   GAIA planning GT mode: auto (default), chain, or augmented
                        auto/augmented score against original chain and dependency
                        DAG candidates and keep chain-only comparison columns.
  GAIA_MODIFIED_GT_DIR  Optional GAIA modified-GT scoring layer
                        (default: GAIA_DATA_ROOT)
  GAIA_MODIFIED_GT_MODE auto (default), off, or require. auto uses modified-GT
                        for evaluation when present without changing inference input.
  RUN_OUTPUT_TAG        Optional tag that writes this run under a fresh dataset
                        subdirectory, useful for reruns without deleting old outputs.
  STAGE3_ABS_DAG_REFERENCE
                        Deprecated no-op. Stage 3 never receives Stage-1 DAG context.
  TOOL_SCOPE             Prompt-visible tool scope: record (default) or global.

Available Datasets:
  ${!DATASETS[*]}

Available Local Models:
  ${LOCAL_MODELS[*]}

Available Remote Models:
  ${REMOTE_MODELS[*]}

Examples:
  # Order-based evaluation on GAIA Cat A
  $0 --dataset gaia_cat_A --mode order --model qwen2.5-7b --limit 10

  # Answer-based evaluation with tool execution (GAIA only)
  $0 --dataset gaia_cat_A --mode answer --model qwen2.5-7b

  # Async ordering analysis on GAIA Cat A
  $0 --dataset gaia_cat_A_async --mode order --model qwen2.5-32b --orderings

  # Full experiment with all models
  $0 --dataset taskbench --mode order --all-models

  # Run with an OpenAI-compatible API endpoint
  export LLM_API_KEY='your-key'
  export LLM_API_BASE='https://your-openai-compatible-endpoint/v1'
  $0 --dataset gaia_cat_A --mode answer --backend api --provider-profile openai_compatible --remote-only

  # Run GPT-5.5 on GAIA Cat A
  export OPENAI_API_KEY='your-openai-key'
  $0 --dataset gaia_cat_A --mode answer --model gpt-5.5 --provider-profile openai

  # Run Claude Sonnet 4.6 on GAIA Cat A
  export ANTHROPIC_API_KEY='your-anthropic-key'
  $0 --dataset gaia_cat_A --mode answer --model claude-sonnet-4-6 --provider-profile anthropic

  # Run Gemini 3.1 Pro Preview on GAIA Cat A
  export GEMINI_API_KEY='your-gemini-key'
  $0 --dataset gaia_cat_A --mode answer --model gemini-3.1-pro-preview --provider-profile gemini

EOF
    exit 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Default values
    local datasets=()
    local models=()
    local limit=""
    local resume="false"
    local run_inference="true"
    local run_eval="true"
    local run_figures="true"
    local verbose="false"
    local dry_run="false"
    local local_only="false"
    local remote_only="false"
    local compare_constraint="false"
    local no_schema_constraint="false"
    local debug_mode="false" # New: debug mode
    local analysis_dir=""    # New: correlation analysis
    local run_orderings_analysis="false"
    local num_runs=1         # NEW: number of experiment runs (for std calculation)
    local aggregate_runs="false"  # NEW: aggregate results across runs
    # pipeline variable removed
    local backend="local"
    local provider_profile="auto"
    local mode="order"  # NEW: order or answer
    local max_turns=15
    local tool_scope="${TOOL_SCOPE:-record}"
    local plan_source="${PLAN_SOURCE:-stage1}"
    local gaia_reference_mode="${GAIA_REFERENCE_MODE:-auto}"
    local stage3_abs_dag_reference="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                models+=("$2")
                shift 2
                ;;
            --dataset)
                datasets+=("$2")
                shift 2
                ;;
            --limit)
                limit="$2"
                shift 2
                ;;
            --resume)
                resume="true"
                shift
                ;;
            --all-models)
                models=("${LOCAL_MODELS[@]}" "${REMOTE_MODELS[@]}")
                shift
                ;;
            --openai-models)
                models=("${OPENAI_REMOTE_MODELS[@]}")
                provider_profile="openai"
                shift
                ;;
            --claude-models)
                models=("${ANTHROPIC_REMOTE_MODELS[@]}")
                provider_profile="anthropic"
                shift
                ;;
            --gemini-models)
                models=("${GEMINI_REMOTE_MODELS[@]}")
                provider_profile="gemini"
                shift
                ;;
            --all-datasets)
                datasets=("${!DATASETS[@]}")
                shift
                ;;
            --local-only)
                local_only="true"
                shift
                ;;
            --remote-only|--api-only)
                remote_only="true"
                shift
                ;;
            --run-cat-abd)
                datasets=("gaia_cat_A" "gaia_cat_B" "gaia_cat_D")
                models=("${GAIA_REMOTE_DEFAULT_MODELS[@]}")
                shift
                ;;
            --run-cat-d)
                datasets=("gaia_cat_D")
                models=("${GAIA_REMOTE_DEFAULT_MODELS[@]}")
                shift
                ;;
            --run-cat-c)
                datasets=("gaia_cat_C")
                models=("${GAIA_REMOTE_DEFAULT_MODELS[@]}")
                shift
                ;;
            # Chinese cross-lingual evaluation shortcuts
            --run-cat-zh)
                mapfile -t datasets < <(load_zh_profile_list --datasets)
                mapfile -t models < <(load_zh_profile_list --models)
                shift
                ;;
            --run-cat-a-zh)
                mapfile -t datasets < <(load_zh_profile_list --dataset gaia_cat_A_zh)
                mapfile -t models < <(load_zh_profile_list --models)
                shift
                ;;
            --run-cat-b-zh)
                mapfile -t datasets < <(load_zh_profile_list --dataset gaia_cat_B_zh)
                mapfile -t models < <(load_zh_profile_list --models)
                shift
                ;;
            --run-cat-c-zh)
                mapfile -t datasets < <(load_zh_profile_list --dataset gaia_cat_C_zh)
                mapfile -t models < <(load_zh_profile_list --models)
                shift
                ;;
            --run-cat-d-zh)
                mapfile -t datasets < <(load_zh_profile_list --dataset gaia_cat_D_zh)
                mapfile -t models < <(load_zh_profile_list --models)
                shift
                ;;
            --run-all-remote-gaia)
                log_info "Running all remote models on their respective GAIA categories natively..."
                "$0" --run-cat-abd "${@:2}"
                "$0" --run-cat-c "${@:2}"
                exit $?
                ;;
            --mode)
                mode="$2"
                if [[ "$mode" != "order" && "$mode" != "answer" ]]; then
                    log_error "Invalid mode: $mode. Use 'order' or 'answer'"
                    exit 1
                fi
                shift 2
                ;;
            --plan-source|--plan_source)
                plan_source="$2"
                if [[ "$plan_source" != "stage1" && "$plan_source" != "stage3" && "$plan_source" != "abs" && "$plan_source" != "abstract" && "$plan_source" != "abs_plan_dag" ]]; then
                    log_error "Invalid plan source: $plan_source. Use 'stage1' or 'stage3'"
                    exit 1
                fi
                PLAN_SOURCE="$plan_source"
                shift 2
                ;;
            --stage3-abs-dag-reference|--stage3_abs_dag_reference)
                log_warn "--stage3-abs-dag-reference is deprecated and ignored; Stage 3 will not receive Stage-1 DAG context."
                stage3_abs_dag_reference="false"
                STAGE3_ABS_DAG_REFERENCE="false"
                shift
                ;;
            --gaia-reference-mode|--gaia_reference_mode)
                gaia_reference_mode="$2"
                if [[ "$gaia_reference_mode" != "auto" && "$gaia_reference_mode" != "chain" && "$gaia_reference_mode" != "augmented" && "$gaia_reference_mode" != "both" ]]; then
                    log_error "Invalid GAIA reference mode: $gaia_reference_mode. Use auto, chain, augmented, or both"
                    exit 1
                fi
                GAIA_REFERENCE_MODE="$gaia_reference_mode"
                shift 2
                ;;
            --inference-only)
                run_inference="true"
                run_eval="false"
                run_figures="false"
                shift
                ;;
            --eval-only)
                run_inference="false"
                run_eval="true"
                run_figures="false"
                shift
                ;;
            --figures-only)
                run_inference="false"
                run_eval="false"
                run_figures="true"
                shift
                ;;
            --port)
                SGLANG_PORT="$2"
                shift 2
                ;;
            --context-length)
                SGLANG_CONTEXT_LENGTH="$2"
                shift 2
                ;;
            --mem-fraction)
                SGLANG_MEM_FRACTION="$2"
                shift 2
                ;;
            --verbose)
                verbose="true"
                shift
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --compare-constraint)
                compare_constraint="true"
                shift
                ;;
            # pipeline argument removed
            --no-schema-constraint)
                no_schema_constraint="true"
                shift
                ;;
            --backend)
                backend="$2"
                shift 2
                ;;
            --provider-profile)
                provider_profile="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            --debug)
                debug_mode="true"
                shift
                ;;
            --analysis)
                analysis_dir="$2"
                shift 2
                ;;
            --run)
                num_runs="$2"
                if [[ ! "$num_runs" =~ ^[0-9]+$ ]] || [[ "$num_runs" -lt 1 ]]; then
                    log_error "Invalid --run value: $num_runs. Must be a positive integer."
                    exit 1
                fi
                shift 2
                ;;
            --max-turns)
                max_turns="$2"
                if [[ ! "$max_turns" =~ ^[0-9]+$ ]] || [[ "$max_turns" -lt 1 ]]; then
                    log_error "Invalid --max-turns value: $max_turns. Must be a positive integer."
                    exit 1
                fi
                shift 2
                ;;
            --tool-scope|--tool_scope)
                tool_scope="$2"
                if [[ "$tool_scope" != "record" && "$tool_scope" != "global" ]]; then
                    log_error "Invalid --tool-scope value: $tool_scope. Use record or global."
                    exit 1
                fi
                TOOL_SCOPE="$tool_scope"
                export TOOL_SCOPE
                shift 2
                ;;
            --aggregate)
                aggregate_runs="true"
                shift
                ;;
            --verifier_model)
                ANSWER_VERIFIER_MODEL="$2"
                shift 2
                ;;
            --orderings)
                run_orderings_analysis="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Apply local/remote filters
    if [[ "$local_only" == "true" ]]; then
        if [[ ${#models[@]} -eq 0 ]]; then
            models=("${LOCAL_MODELS[@]}")
        else
            local filtered_models=()
            for m in "${models[@]}"; do
                if [[ "$backend" != "api" && "${MODEL_ROUTING[$m]:-local}" == "local" ]]; then
                    filtered_models+=("$m")
                fi
            done
            models=("${filtered_models[@]}")
        fi
    elif [[ "$remote_only" == "true" ]]; then
        if [[ ${#models[@]} -eq 0 ]]; then
            models=("${REMOTE_MODELS[@]}")
        else
            local filtered_models=()
            for m in "${models[@]}"; do
                if [[ "$backend" == "api" || "${MODEL_ROUTING[$m]:-local}" == "remote" ]]; then
                    filtered_models+=("$m")
                fi
            done
            models=("${filtered_models[@]}")
        fi
    fi

    # Set defaults if not specified
    if [[ ${#datasets[@]} -eq 0 ]]; then
        datasets=("gaia_cat_A")
    fi
    if [[ ${#models[@]} -eq 0 ]]; then
        # Default to all local models (not just one)
        models=("${LOCAL_MODELS[@]}")
    fi

    # Configure remote provider profile after model selection.
    local inferred_provider="openai_compatible"
    for m in "${models[@]}"; do
        if [[ "$backend" == "api" || "${MODEL_ROUTING[$m]:-local}" == "remote" ]]; then
            inferred_provider="$(infer_provider_profile_from_model "$m")"
            break
        fi
    done
    configure_api_provider_profile "$provider_profile" "$inferred_provider"
    configure_verifier_provider_defaults
    export LLM_API_PROVIDER LLM_API_BASE LLM_API_KEY
    export ANSWER_VERIFIER_API_BASE ANSWER_VERIFIER_API_KEY ANSWER_VERIFIER_API_PROVIDER

    # Check remote API credentials for remote models
    local has_remote_models=false
    for m in "${models[@]}"; do
        if [[ "$backend" == "api" || "${MODEL_ROUTING[$m]:-local}" == "remote" ]]; then
            has_remote_models=true
            break
        fi
    done

    if [[ "$has_remote_models" == "true" && -z "$LLM_API_KEY" ]]; then
        log_warn "LLM_API_KEY not set! Remote API models may fail."
        case "$LLM_API_PROVIDER" in
            openai)
                log_warn "Set it with: export OPENAI_API_KEY='your-openai-key'"
                ;;
            anthropic)
                log_warn "Set it with: export ANTHROPIC_API_KEY='your-anthropic-key'"
                ;;
            gemini)
                log_warn "Set it with: export GEMINI_API_KEY='your-gemini-key'"
                ;;
            *)
                log_warn "Set it with: export LLM_API_KEY='your-key-here'"
                ;;
        esac
    fi

    # Create logs directory
    mkdir -p "$LOGS_DIR"

    # SPECIAL MODE: Correlation Analysis
    if [[ -n "$analysis_dir" ]]; then
        log_header "Running Correlation Analysis"
        log_info "Results directory: $analysis_dir"

        if [[ ! -d "$analysis_dir" ]]; then
            log_error "Analysis directory not found: $analysis_dir"
            exit 1
        fi

        # Run correlation analysis
        log_info "Executing correlation analysis..."
        $PYTHON -m src.analysis.correlation_analysis \
            --results_dir "$analysis_dir"

        if [[ $? -eq 0 ]]; then
            log_info "Correlation analysis completed successfully!"
            log_info "Results saved to: ${analysis_dir}/correlation_analysis/"
        else
            log_error "Correlation analysis failed"
            exit 1
        fi

        exit 0
    fi

    # Display configuration
    log_header "Unified Experiment Runner"
    log_info "Datasets: ${datasets[*]}"
    log_info "Models: ${models[*]}"
    log_info "Limit: ${limit:-all}"
    log_info "Resume: $resume"
    log_info "Number of runs: $num_runs"
    log_info "Run inference: $run_inference"
    log_info "Run evaluation: $run_eval"
    log_info "Run figures: $run_figures"
    log_info "Run orderings analysis: $run_orderings_analysis"
    log_info "Plan source: ${PLAN_SOURCE:-$plan_source}"
    log_info "GAIA reference mode: ${GAIA_REFERENCE_MODE:-$gaia_reference_mode}"
    log_info "Stage3 abs DAG reference: disabled"
    log_info "Tool scope: ${TOOL_SCOPE:-$tool_scope}"
    log_info "GAIA data root: ${GAIA_DATA_ROOT}"
    log_info "GAIA modified-GT scoring: ${GAIA_MODIFIED_GT_MODE} (${GAIA_MODIFIED_GT_DIR})"
    log_info "Max turns (answer mode): $max_turns"
    log_info "Remote provider profile: $LLM_API_PROVIDER"
    if [[ "$has_remote_models" == "true" ]]; then
        log_info "Remote API base: ${LLM_API_BASE:-<provider-default>}"
    fi
    if [[ "$mode" == "answer" ]]; then
        log_info "Verifier model: ${ANSWER_VERIFIER_MODEL:-disabled}"
    fi

    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN MODE - commands will be printed but not executed"
    fi

    # Calculate total runs (accounting for multi-run)
    local total_experiment_runs=$((${#datasets[@]} * ${#models[@]} * num_runs))
    local current_experiment_run=0

    # Track all output directories for aggregation
    declare -A run_output_dirs

    # MULTI-RUN EXPERIMENT LOOP
    for ((run_idx=1; run_idx<=num_runs; run_idx++)); do
        if [[ "$num_runs" -gt 1 ]]; then
            log_header "==================== RUN $run_idx / $num_runs ===================="
        fi

        # Main experiment loop
        for dataset in "${datasets[@]}"; do
            # Validate mode compatibility
            if [[ "$mode" == "answer" && -z "${ANSWER_MODE_DATASETS[$dataset]:-}" ]]; then
                 log_error "Dataset '$dataset' does not support answer-based evaluation (--mode answer)."
                 log_error "Reason: This dataset does not contain 'final_answer' fields required for correctness assessment."
                 log_error "Please use '--mode order' for this dataset."
                 continue
            fi
            # Determine effective output directory for this dataset run
            local base_output_dir="${OUTPUT_DIRS[$dataset]:-}"
            local current_output_dir="$base_output_dir"

            if [[ -n "${RUN_OUTPUT_TAG:-}" ]]; then
                local safe_output_tag
                safe_output_tag="$(printf '%s' "$RUN_OUTPUT_TAG" | tr -c 'A-Za-z0-9._-' '_')"
                current_output_dir="${base_output_dir}/${safe_output_tag}.${dataset}.${mode}"
            elif [[ "$mode" == "answer" ]]; then
                local run_timestamp=$(date +"%Y%m%d_%H%M%S")
                local answer_suffix="answer"
                if [[ "${TOOL_SCOPE:-record}" == "global" ]]; then
                    answer_suffix="answer.global_tools"
                fi
                if [[ "$num_runs" -gt 1 ]]; then
                    # Multi-run mode: include run index in directory name
                    current_output_dir="${base_output_dir}/${run_timestamp}.${dataset}.${answer_suffix}.run${run_idx}"
                else
                    current_output_dir="${base_output_dir}/${run_timestamp}.${dataset}.${answer_suffix}"
                fi
            fi

            # Ensure directory exists
            mkdir -p "$current_output_dir"
            log_info "Target Output Directory: $current_output_dir"

            # Track output directory for aggregation
            run_output_dirs["${dataset}_run${run_idx}"]="$current_output_dir"

            for model in "${models[@]}"; do
                current_experiment_run=$((current_experiment_run + 1))
                local did_evaluate="false"

                # Set up log file for this run
                if [[ "$num_runs" -gt 1 ]]; then
                    CURRENT_LOG_FILE="${RUN_LOG_DIR}/${model}_${dataset}_${mode}_run${run_idx}.log"
                else
                    CURRENT_LOG_FILE="${RUN_LOG_DIR}/${model}_${dataset}_${mode}.log"
                fi

                log_header "[$current_experiment_run/$total_experiment_runs] Model: $model | Dataset: $dataset | Run: $run_idx/$num_runs"
                log_info "Log file: $CURRENT_LOG_FILE"

                local routing="${MODEL_ROUTING[$model]:-local}"
                if [[ "$backend" == "api" ]]; then
                    routing="remote"
                fi
                log_info "Routing: $routing"

                # Check if this specific run already completed (for resume across runs)
                local run_marker_file="${current_output_dir}/.run_${model}_complete"
                if [[ "$resume" == "true" && -f "$run_marker_file" ]]; then
                    log_info "SKIPPING: Run $run_idx for $model already completed (marker exists)"
                    continue
                fi

                # Inference
                if [[ "$run_inference" == "true" ]]; then
                    if [[ "$dry_run" == "true" ]]; then
                        log_info "[DRY RUN] Would run inference for $model on $dataset (run $run_idx)"
                    elif [[ "$compare_constraint" == "true" ]]; then
                        # Compare mode: run both constrained and unconstrained
                        log_info "Running COMPARISON mode: constrained vs raw"

                        # Run 1: With schema constraint (default)
                        log_info "=== Run 1: WITH schema constraint (xgrammar) ==="
                        if ! run_inference "$model" "$dataset" "$current_output_dir" "$limit" "$resume" "false" "$backend" "$mode" ".constrained"; then
                            log_warn "Constrained inference failed for $model"
                        fi

                        # VRAM GUARD: Clean up GPU memory between runs
                        log_info "VRAM Guard: Cleaning up between comparison runs..."
                        pkill -f "sglang.launch_server" 2>/dev/null || true
                        sleep 5

                        # Run 2: Without schema constraint (raw LLM output)
                        log_info "=== Run 2: WITHOUT schema constraint (raw) ==="
                        if ! run_inference "$model" "$dataset" "$current_output_dir" "$limit" "$resume" "true" "$backend" "$mode" ".raw"; then
                            log_warn "Raw inference failed for $model"
                        fi
                    else
                        # Normal mode: single run
                        if ! run_inference "$model" "$dataset" "$current_output_dir" "$limit" "$resume" "$no_schema_constraint" "$backend" "$mode" "" "$debug_mode" "$max_turns"; then
                            log_error "Inference failed for $model on $dataset (run $run_idx)"
                            # Continue to next model instead of exiting
                            continue
                        fi

                        if [[ "$run_eval" == "true" ]]; then
                            run_evaluation "$dataset" "$model" "$current_output_dir" || log_warn "Evaluation failed for $model"
                            did_evaluate="true"
                        fi

                        # Mark run as complete for resume support
                        touch "$run_marker_file"
                    fi
                fi

                if [[ "$run_inference" != "true" && "$run_eval" == "true" ]]; then
                    if [[ "$dry_run" == "true" ]]; then
                        log_info "[DRY RUN] Would run evaluation for $model on $dataset"
                    else
                        run_evaluation "$dataset" "$model" "$current_output_dir" || log_warn "Evaluation failed for $model"
                        did_evaluate="true"
                    fi
                fi

                if [[ "$run_orderings_analysis" == "true" ]]; then
                    if [[ "$dry_run" == "true" ]]; then
                        log_info "[DRY RUN] Would run ordering analysis for $model on $dataset"
                    else
                        run_orderings_analysis "$dataset" "$model" "$current_output_dir" || log_warn "Ordering analysis failed for $model"
                    fi
                fi

                # VRAM GUARD: Clean up between models (for local routing)
                if [[ "$routing" == "local" ]]; then
                    log_info "VRAM Guard: Ensuring cleanup before next model..."
                    pkill -f "sglang.launch_server" 2>/dev/null || true
                    sleep 3
                fi
            done

            # Figures (per dataset, after all models) - only on last run or single run
            if [[ "$run_figures" == "true" && ("$run_idx" -eq "$num_runs" || "$num_runs" -eq 1) ]]; then
                if [[ "$dry_run" == "true" ]]; then
                    log_info "[DRY RUN] Would generate figures for $dataset"
                else
                    run_figures "$dataset" "$current_output_dir" || log_warn "Figure generation failed for $dataset"
                fi
            fi
        done
    done

    # MULTI-RUN AGGREGATION: Aggregate results across runs if num_runs > 1
    if [[ "$num_runs" -gt 1 ]]; then
        log_header "Aggregating Results Across $num_runs Runs"

        for dataset in "${datasets[@]}"; do
            local base_output_dir="${OUTPUT_DIRS[$dataset]:-}"
            local aggregate_dir="${base_output_dir}/aggregated_${num_runs}runs_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$aggregate_dir"

            # Collect all run directories for this dataset
            local run_dirs=()
            for ((i=1; i<=num_runs; i++)); do
                local key="${dataset}_run${i}"
                if [[ -n "${run_output_dirs[$key]:-}" ]]; then
                    run_dirs+=("${run_output_dirs[$key]}")
                fi
            done

            if [[ ${#run_dirs[@]} -gt 0 ]]; then
                log_info "Aggregating ${#run_dirs[@]} runs for $dataset"
                log_info "Output: $aggregate_dir"

                # Run aggregation script
                if [[ "$dry_run" == "true" ]]; then
                    log_info "[DRY RUN] Would aggregate: ${run_dirs[*]}"
                else
                    $PYTHON -m src.evaluation.aggregate_runs \
                        --run_dirs "${run_dirs[@]}" \
                        --output_dir "$aggregate_dir" \
                        2>&1 | tee -a "$CURRENT_LOG_FILE" || log_warn "Aggregation failed for $dataset"
                fi
            else
                log_warn "No run directories found for $dataset"
            fi
        done
    fi

    # Summary
    log_header "Experiment Complete!"
    log_info "Total experiment runs: $current_experiment_run"
    log_info "Number of repetitions: $num_runs"
    log_info "Datasets: ${datasets[*]}"
    log_info "Models: ${models[*]}"
    log_info ""
    log_info "Results saved to:"
    for dataset in "${datasets[@]}"; do
        log_info "  $dataset: ${OUTPUT_DIRS[$dataset]}"
    done
    log_info ""
    log_info "Logs saved to: $LOGS_DIR"
}

# Run main
main "$@"
