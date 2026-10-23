"""Opt-in dynamic skill loading support for answer-mode ReAct experiments.

This module deliberately contains no model or tool-execution logic. It loads
versioned procedural instructions and supplies the small amount of metadata
needed by the runner to expose a ``Skill`` pseudo-tool and log injections.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    description: str
    content: str
    content_hash: str
    extensions: frozenset[str]
    source_path: Path
    base_dir: Path
    files: Tuple[str, ...]
    paths: Tuple[str, ...]
    disable_model_invocation: bool
    injection_format: str

    def catalog_line(self) -> str:
        return f"- {self.skill_id}: {self.description}"


class SkillRegistry:
    """A validated, file-backed collection of procedural skills."""

    def __init__(
        self,
        version: str,
        definitions: Iterable[SkillDefinition],
        source_path: Path,
        injection_format: str = "classic",
    ):
        self.version = version
        self.source_path = source_path
        self.injection_format = injection_format
        self._by_id: Dict[str, SkillDefinition] = {}
        for definition in definitions:
            if definition.skill_id in self._by_id:
                raise ValueError(f"Duplicate skill_id in registry: {definition.skill_id}")
            self._by_id[definition.skill_id] = definition
        if not self._by_id:
            raise ValueError("Skill registry must contain at least one skill")

    @classmethod
    def load(cls, registry_path: Path) -> "SkillRegistry":
        registry_path = Path(registry_path)
        with registry_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        version = str(payload.get("version") or "unversioned")
        registry_injection_format = str(payload.get("injection_format") or "classic").strip() or "classic"
        definitions: List[SkillDefinition] = []
        for raw in payload.get("skills") or []:
            skill_id = str(raw.get("skill_id") or raw.get("name") or "").strip()
            rel_path = str(raw.get("path") or "").strip()
            if not skill_id or not rel_path:
                raise ValueError("Each skill requires skill_id/name and path")
            source_path = (registry_path.parent / rel_path).resolve()
            content = source_path.read_text(encoding="utf-8")
            frontmatter, body = _split_frontmatter(content)
            fm_name = str(frontmatter.get("name") or "").strip()
            if fm_name and fm_name != skill_id:
                raise ValueError(
                    f"Skill registry id {skill_id!r} does not match SKILL.md frontmatter name {fm_name!r}"
                )
            description = str(raw.get("description") or frontmatter.get("description") or "").strip()
            if not description:
                raise ValueError(f"Skill {skill_id!r} requires description in registry or SKILL.md frontmatter")
            disable_model_invocation = (
                _truthy(raw.get("disable_model_invocation"))
                or _truthy(raw.get("disable-model-invocation"))
                or _truthy(frontmatter.get("disable-model-invocation"))
                or _truthy(frontmatter.get("disable_model_invocation"))
            )
            if not body.strip():
                raise ValueError(f"Skill {skill_id!r} has empty body")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            extensions = frozenset(_normalize_extension(ext) for ext in (raw.get("extensions") or []))
            base_dir = source_path.parent
            paths = tuple(str(path) for path in _as_list(raw.get("paths") or frontmatter.get("paths") or []))
            injection_format = str(raw.get("injection_format") or registry_injection_format).strip() or "classic"
            definitions.append(
                SkillDefinition(
                    skill_id=skill_id,
                    description=description,
                    content=content,
                    content_hash=digest,
                    extensions=extensions,
                    source_path=source_path,
                    base_dir=base_dir,
                    files=tuple(_list_skill_files(base_dir)),
                    paths=paths,
                    disable_model_invocation=disable_model_invocation,
                    injection_format=injection_format,
                )
            )
        return cls(
            version=version,
            definitions=definitions,
            source_path=registry_path.resolve(),
            injection_format=registry_injection_format,
        )

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        return self._by_id.get(str(skill_id or "").strip())

    def skill_ids(self) -> List[str]:
        return list(self._by_id)

    def catalog_prompt(self) -> str:
        return "\n".join(self._by_id[skill_id].catalog_line() for skill_id in self.skill_ids())

    def catalog_id_prompt(self) -> str:
        return "\n".join(f"- {skill_id}" for skill_id in self.skill_ids())

    def expected_skill_ids(self, record: Dict[str, Any]) -> Set[str]:
        """Return extension-routed skills for oracle and routing evaluation.

        This routing uses only attachment types, never questions or answers.
        It is therefore a defensible oracle control rather than leaked task
        supervision.
        """
        extensions = _attachment_extensions(record)
        return {
            definition.skill_id
            for definition in self._by_id.values()
            if definition.extensions & extensions
        }

    def select_oracle(self, record: Dict[str, Any]) -> Optional[SkillDefinition]:
        expected = sorted(self.expected_skill_ids(record))
        if len(expected) > 1:
            raise ValueError(
                f"Oracle routing is ambiguous for record attachments: matched skills {expected}"
            )
        return self.get(expected[0]) if expected else None

    def tool_spec(self) -> Dict[str, Any]:
        return {
            "tool_id": "Skill",
            "description": "Load exactly one listed procedural skill before using evidence tools when it is relevant.",
            "arguments_schema": [
                {"name": "skill_id", "type": "string", "required": False},
                {"name": "skill", "type": "string", "required": False},
                {"name": "args", "type": "string", "required": False},
            ],
        }


def render_injection(definition: SkillDefinition, registry_version: str) -> str:
    """Render the context delivered to the model after a skill activates."""
    if definition.injection_format in {"delta", "delta_compatible", "delta-compatible"}:
        return _render_delta_injection(definition, registry_version)
    return (
        f"ACTIVE SKILL INJECTION\n"
        f"skill_id: {definition.skill_id}\n"
        f"registry_version: {registry_version}\n"
        f"content_sha256: {definition.content_hash}\n\n"
        f"{definition.content}"
    )


def render_control_injection(definition: SkillDefinition, registry_version: str) -> str:
    """Render length-matched control context without labeling it as a real skill."""
    return (
        f"ADDITIONAL PROCEDURAL CONTEXT\n"
        f"registry_version: {registry_version}\n"
        f"content_sha256: {definition.content_hash}\n\n"
        f"{definition.content}"
    )


def _normalize_extension(ext: str) -> str:
    text = str(ext or "").strip().lower()
    return text if text.startswith(".") else f".{text}"


def _split_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.DOTALL)
    if not match:
        return {}, content
    raw, body = match.groups()
    frontmatter: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_key:
            text = line.strip()
            if text.startswith("-"):
                frontmatter.setdefault(current_key, []).append(text[1:].strip().strip('"\''))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        if value in {"", "[]"}:
            frontmatter[key] = [] if value == "[]" else ""
        elif value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip('"\'') for item in value[1:-1].split(",") if item.strip()]
            frontmatter[key] = items
        else:
            frontmatter[key] = value.strip('"\'')
    return frontmatter, body


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _list_skill_files(base_dir: Path, max_files: int = 30) -> List[str]:
    files: List[str] = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base_dir).as_posix()
        if rel == "SKILL.md" or "__pycache__" in path.parts:
            continue
        files.append(rel)
        if len(files) >= max_files:
            break
    return files


def _render_delta_injection(definition: SkillDefinition, registry_version: str) -> str:
    files_xml = "\n".join(f"<file>{path}</file>" for path in definition.files)
    if not files_xml:
        files_xml = "(no additional files)"
    _, body = _split_frontmatter(definition.content)
    return "\n".join(
        [
            f'<skill_content name="{definition.skill_id}">',
            f"# Skill: {definition.skill_id}",
            "",
            body.strip(),
            "",
            f"Base directory for this skill: {definition.base_dir}",
            "Relative paths in this skill are relative to this base directory.",
            f"registry_version: {registry_version}",
            f"content_sha256: {definition.content_hash}",
            "",
            "<skill_files>",
            files_xml,
            "</skill_files>",
            "</skill_content>",
        ]
    )


def _attachment_extensions(record: Dict[str, Any]) -> Set[str]:
    attachments = (record.get("query") or {}).get("attachments") or []
    extensions: Set[str] = set()
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        path = str(attachment.get("file_name") or attachment.get("file_path") or "")
        suffix = Path(path).suffix.lower()
        if suffix:
            extensions.add(suffix)
    return extensions
