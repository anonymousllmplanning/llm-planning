"""Versioned, model-visible GAIA tool interface; no API or runtime imports."""
from copy import deepcopy

TOOL_PROFILE_ID = "gaia-executable-tools-v1"


def _arg(name, kind, description, *, required=False, default=None, **constraints):
    value = {"name": name, "type": kind, "required": required, "description": description}
    if not required and default is not None:
        value["default"] = default
    value.update(constraints)
    return value


def _tool(name, description, *arguments):
    return {"tool_id": name, "name": name, "description": description,
            "arguments_schema": list(arguments)}


_TOOLS = [
    _tool("web_search", "Search the web using DDGS (Bing first, DuckDuckGo fallback). Return result titles, snippets and URLs; may fetch text from leading results.",
          _arg("query", "string", "Search query.", required=True),
          _arg("max_results", "integer", "Maximum number of search results, from 1 to 8.", default=5, minimum=1, maximum=8)),
    _tool("web_browser", "Fetch a concrete URL and extract readable text. Try Playwright, then requests; PDF URLs use PDF text extraction. This interface does not click, type, or submit forms.",
          _arg("url", "string", "HTTP(S) content URL, not a search-results or CAPTCHA page.", required=True)),
    _tool("download_file", "Download a URL to a local file and return its saved path for subsequent reader tools.",
          _arg("url", "string", "HTTP(S) file URL.", required=True),
          _arg("save_path", "string", "Optional destination file path; otherwise use the runtime artifact directory.")),
    _tool("file_reader", "Read local DOCX paragraphs/tables, JSON, JSONLD, XML, or text/code files. Runtime dispatch routes PDF, table, PPTX, ZIP and image files to their specialized readers. Does not transcribe audio or analyze video.",
          _arg("file_path", "string", "Local file path.", required=True)),
    _tool("excel_reader", "Inspect Excel or CSV tables: return dimensions, column names/types, sample values and row previews. The optional query prioritizes relevant columns; it is not a general query engine. Use Python for full-table calculations.",
          _arg("file_path", "string", "Local .xlsx, .xls or .csv path.", required=True),
          _arg("sheet", "string", "Optional Excel sheet name; otherwise inspect available sheets."),
          _arg("query", "string", "Optional terms for selecting relevant preview columns.")),
    _tool("pdf_reader", "Extract text from a local PDF or a PDF URL, with page markers. Does not perform OCR on scanned pages.",
          _arg("file_path", "string", "Local PDF path or HTTP(S) PDF URL.", required=True),
          _arg("page", "integer", "Optional page number starting at 1; otherwise read all pages.", minimum=1)),
    _tool("pptx_reader", "Extract text from PowerPoint shapes with slide markers. Does not interpret images or embedded charts visually.",
          _arg("file_path", "string", "Local .pptx path.", required=True),
          _arg("slide_number", "integer", "Optional slide number starting at 1; otherwise read all slides.", minimum=1)),
    _tool("zip_extractor", "Extract a local ZIP archive and return extracted file paths. Read those files with appropriate tools afterward.",
          _arg("file_path", "string", "Local .zip path.", required=True),
          _arg("extract_to", "string", "Optional output directory; otherwise use the runtime artifact directory.")),
    _tool("image_recognition", "Analyze a local image using the configured vision backend. Results may use a fallback model or local OCR/color extraction; backend provenance must be recorded. This tool does not analyze video.",
          _arg("file_path", "string", "Local image path.", required=True),
          _arg("task", "string", "Analysis task, e.g. describe, extract_text, extract_numbers, extract_by_color, chess, music_sheet, geometry or fractions.", default="describe"),
          _arg("custom_prompt", "string", "Optional specific visual question; overrides the task prompt template.", default="")),
    _tool("audio_transcription", "Transcribe a local audio file through the configured ASR API, with local Whisper fallback. Returns transcript text; backend provenance must be recorded.",
          _arg("file_path", "string", "Local audio file path.", required=True),
          _arg("language", "string", "Optional short language code, e.g. en or zh.")),
    _tool("python_executor", "Execute Python code and return printed output or a result variable. Runtime provides ATTACHMENTS_DIR and INPUT_FILES. Use copies and the artifact directory for generated files. This implementation executes Python in-process and is not a security sandbox.",
          _arg("code", "string", "Executable Python code. Print the needed result or assign result/answer/output/value.", required=True)),
    _tool("calculator", "Evaluate a Python arithmetic expression through python_executor, with math/statistics helpers. Use concrete observed values.",
          _arg("expression", "string", "Arithmetic expression.", required=True)),
    _tool("reasoning", "Call the configured reasoning helper on supplied facts and observations. Include all required evidence in the problem; the helper does not directly browse or inspect files. Its model and token cost are part of the execution configuration.",
          _arg("problem", "string", "Subproblem including the relevant facts and tool observations.", required=True)),
    _tool("submit_final_answer", "Submit the final answer and end the task. This is a terminal control action, not an evidence-gathering tool.",
          _arg("answer", "string", "Final answer in the format requested by the task.", required=True),
          _arg("answer_type", "string", "Answer type label.", default="string")),
]
TOOL_IDS = tuple(tool["tool_id"] for tool in _TOOLS)


def get_tool_environment():
    """Return an independent copy in the same stable order for every category."""
    return deepcopy(_TOOLS)


def validate_record_profile(record):
    """Return False for legacy data; reject unknown or altered versioned profiles."""
    metadata = record.get("meta") or {}
    profile_id = metadata.get("tool_profile_id")
    if profile_id is None:
        return False
    if profile_id != TOOL_PROFILE_ID:
        raise ValueError(f"Unsupported GAIA tool profile: {profile_id}")
    if record.get("tool_environment") != _TOOLS:
        raise ValueError(f"Record tool_environment does not match {TOOL_PROFILE_ID}")
    return True


def validate_tool_call(record, tool_id, arguments):
    """Validate versioned calls before dispatch; leave unmarked data unchanged."""
    if not validate_record_profile(record):
        return
    specs = {tool["tool_id"]: tool for tool in _TOOLS}
    if not isinstance(tool_id, str) or tool_id not in specs:
        raise ValueError(f"Tool {tool_id!r} is not in {TOOL_PROFILE_ID}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    parameters = {param["name"]: param for param in specs[tool_id]["arguments_schema"]}
    unknown = sorted(set(arguments) - parameters.keys())
    if unknown:
        raise ValueError(f"Unexposed arguments for {tool_id}: {unknown}")
    for name, parameter in parameters.items():
        if name not in arguments:
            if parameter["required"]:
                raise ValueError(f"Missing required argument: {tool_id}.{name}")
            continue
        value = arguments[name]
        kind = parameter["type"]
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{tool_id}.{name} must be a string")
        if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{tool_id}.{name} must be an integer")
        if "minimum" in parameter and value < parameter["minimum"]:
            raise ValueError(f"{tool_id}.{name} is below its minimum")
        if "maximum" in parameter and value > parameter["maximum"]:
            raise ValueError(f"{tool_id}.{name} is above its maximum")


def profile_document():
    return {
        "profile_id": TOOL_PROFILE_ID,
        "categories": ["A", "B", "C", "D"],
        "languages": ["en", "zh"],
        "tool_environment": get_tool_environment(),
        "tool_count": 14, "execution_tool_count": 13, "terminal_action_count": 1,
        "excluded_tools": {
            "video_analysis": "Returns mock output; no real video execution backend.",
            "code_interpreter": "Python duplicates python_executor; other languages return mock output.",
        },
        "omitted_runtime_parameters": ["output_dir", "attachments_dir", "input_files", "timeout_seconds"],
        "omitted_ineffective_parameters": {"web_search": ["engine"], "web_browser": ["action"]},
        "interface_frozen": True,
        "backend_qualified": False,
        "limitations": [
            "Freezes advertised interfaces, not helper models, network contents, installed dependencies or caches.",
            "Vision/ASR/reasoning backends and all fallbacks require a separate run manifest and quality checks.",
            "No video_analysis tool: video-dependent steps remain capability gaps; do not mark them verified.",
            "Historical tool scores are not results under this new profile; regenerate predictions to compare it.",
            "The legacy dependency annotator consumes question and fixed steps, not tool_environment.",
        ],
    }
