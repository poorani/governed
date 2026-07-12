"""Skills: reusable, versioned SOPs the agent loads on demand.

Progressive disclosure is the whole point. Injecting every skill body into the
system prompt would blow the context budget and bury the goal, so the system
prompt carries only ``index_markdown()`` -- one line per skill -- and the model
calls the ``load_skill`` tool to pull a body into context when it recognises the
situation. Ten skills cost ~200 tokens until one is actually needed.

YAML frontmatter is parsed with PyYAML if installed (``pip install
'governed[yaml]'``); a minimal built-in parser handles the common case
(flat ``key: value`` and ``key: [a, b, c]``) so the framework has no hard
dependency on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Skill",
    "SkillConfig",
    "SkillLibrary",
    "SkillSource",
    "register_skill_source",
    "registered_skill_sources",
    "resolve_skills",
]


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str = ""
    version: str = "0.0.0"
    tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None

    def index_line(self) -> str:
        suffix = f" -- use when: {self.when_to_use}" if self.when_to_use else ""
        return f"- `{self.name}` (v{self.version}): {self.description}{suffix}"


@dataclass
class SkillConfig:
    """Declarative skill-library selection, for config-driven bootstrapping.

    The data-only counterpart to passing a live ``SkillLibrary`` (or
    ``skills_dirs``) directly. ``resolve_skills`` turns this into a
    ``SkillLibrary`` the same way ``resolve_tools``/``resolve_llm`` turn
    their config counterparts into live objects.
    """

    dirs: list[str] = field(default_factory=lambda: ["./skills"])
    #: Off switch: skip scanning entirely and return an empty library, e.g.
    #: for a deployment that wants to explicitly disable skills via config
    #: rather than by emptying ``dirs``.
    enabled: bool = True
    #: Which ``SkillSource`` builds the library -- ``"directory"`` (the
    #: default) scans ``dirs`` for ``*/SKILL.md``, matching
    #: ``SkillLibrary.from_dirs`` exactly. Set this to a name registered via
    #: ``register_skill_source`` to load skills from anywhere else (a
    #: database, an S3 prefix, a Git repo) while everything downstream still
    #: just sees a ``SkillLibrary``.
    source: str = "directory"


class SkillLibrary:
    def __init__(self, skills: dict[str, Skill] | None = None) -> None:
        self._skills = dict(skills or {})

    @classmethod
    def from_dirs(cls, *dirs: str | Path) -> SkillLibrary:
        skills: dict[str, Skill] = {}
        for d in dirs:
            base = Path(d)
            if not base.is_dir():
                continue
            for skill_md in sorted(base.glob("*/SKILL.md")):
                skill = _parse_skill_file(skill_md)
                if skill.name in skills:
                    raise ValueError(
                        f"Duplicate skill name {skill.name!r}: {skill.path} and "
                        f"{skills[skill.name].path}"
                    )
                skills[skill.name] = skill
        return cls(skills)

    @property
    def names(self) -> set[str]:
        return set(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def index_markdown(self) -> str:
        if not self._skills:
            return ""
        lines = [s.index_line() for s in sorted(self._skills.values(), key=lambda s: s.name)]
        return "\n".join(lines)

    def validate_tool_references(self, available_tools: set[str]) -> list[str]:
        """Skills that name a tool the current tool set doesn't have.

        ``Agent.__init__`` raises on any problem returned here -- better to fail
        at construction than to have the model load a skill at iteration 7 that
        tells it to call a tool that doesn't exist.
        """
        problems: list[str] = []
        for skill in self._skills.values():
            for tool in skill.tools:
                if tool not in available_tools:
                    problems.append(
                        f"skill '{skill.name}' references unregistered tool '{tool}'"
                    )
        return problems


#: ``(config) -> a ready-to-use SkillLibrary``. See ``register_skill_source``.
SkillSource = Callable[[SkillConfig], SkillLibrary]


def _directory_source(config: SkillConfig) -> SkillLibrary:
    return SkillLibrary.from_dirs(*config.dirs)


_SKILL_SOURCE_REGISTRY: dict[str, SkillSource] = {"directory": _directory_source}


def register_skill_source(name: str, source: SkillSource) -> None:
    """Add or replace the loader ``SkillConfig(source=name)`` resolves to --
    the same pattern ``register_provider``/``register_tool`` use.

    ``source`` takes the ``SkillConfig`` (so it can read ``dirs`` for
    whatever ``dirs`` means in its own context -- an S3 prefix, a table name,
    a Git ref) and returns a ready-to-use ``SkillLibrary``. Names are matched
    case-insensitively; registering ``"directory"`` replaces the built-in
    local-filesystem loader.
    """
    _SKILL_SOURCE_REGISTRY[name.lower()] = source


def registered_skill_sources() -> list[str]:
    return sorted(_SKILL_SOURCE_REGISTRY)


def resolve_skills(config: SkillConfig | None) -> SkillLibrary:
    """Turn a ``SkillConfig`` into a ``SkillLibrary``. ``config=None`` and
    ``config.enabled=False`` both yield an empty library -- the former because
    there's nothing to resolve, the latter because that's what the toggle is
    for. Otherwise, dispatches to ``config.source`` in ``_SKILL_SOURCE_REGISTRY``
    -- ``"directory"`` (the default) reproduces ``SkillLibrary.from_dirs``
    exactly; anything else must have been added via ``register_skill_source``.
    """
    if config is None or not config.enabled:
        return SkillLibrary()
    source = _SKILL_SOURCE_REGISTRY.get(config.source.lower())
    if source is None:
        raise ValueError(
            f"Unknown skill source {config.source!r}. Registered: "
            f"{registered_skill_sources()}. Call register_skill_source() to add your own."
        )
    return source(config)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _parse_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(text)
    if "name" not in meta or "description" not in meta:
        raise ValueError(f"{path}: SKILL.md frontmatter requires 'name' and 'description'")
    return Skill(
        name=str(meta["name"]),
        description=str(meta["description"]),
        when_to_use=str(meta.get("when_to_use", "")),
        version=str(meta.get("version", "0.0.0")),
        tools=_as_list(meta.get("tools", [])),
        tags=_as_list(meta.get("tags", [])),
        body=body.strip(),
        path=path,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    block, body = parts[1], parts[2]
    try:
        import yaml

        meta = yaml.safe_load(block) or {}
    except ImportError:
        meta = _mini_yaml(block)
    return meta, body


def _mini_yaml(block: str) -> dict[str, Any]:
    """Flat `key: value` and `key: [a, b, c]` only. No nesting, no multiline."""
    out: dict[str, Any] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            out[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        else:
            out[key] = val.strip("'\"")
    return out


def _as_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and v:
        return [s.strip() for s in v.split(",") if s.strip()]
    return []
