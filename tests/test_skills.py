from __future__ import annotations

from pathlib import Path

from governed.skills import SkillLibrary

REPO_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def test_repo_skills_load_and_have_required_fields() -> None:
    lib = SkillLibrary.from_dirs(REPO_SKILLS_DIR)
    assert "csv_profiling" in lib.names
    skill = lib.get("csv_profiling")
    assert skill is not None
    assert skill.description
    assert skill.body.strip()


def test_index_markdown_is_one_line_per_skill() -> None:
    lib = SkillLibrary.from_dirs(REPO_SKILLS_DIR)
    lines = [line for line in lib.index_markdown().splitlines() if line.strip()]
    assert len(lines) == len(lib.names)


def test_validate_tool_references_flags_unregistered_tools() -> None:
    lib = SkillLibrary.from_dirs(REPO_SKILLS_DIR)
    problems = lib.validate_tool_references({"file_system"})  # missing analyze_data etc.
    assert problems  # csv_profiling references analyze_data


def test_validate_tool_references_clean_when_all_tools_present() -> None:
    lib = SkillLibrary.from_dirs(REPO_SKILLS_DIR)
    all_tools = {"file_system", "execute_code", "analyze_data", "scratchpad", "submit"}
    assert lib.validate_tool_references(all_tools) == []


def test_missing_dir_yields_empty_library(tmp_path: Path) -> None:
    lib = SkillLibrary.from_dirs(tmp_path / "does_not_exist")
    assert lib.names == set()
