"""The prompts. Plain strings, not a templating framework -- fork them.

Every phase pushes one of these as a user turn. The instructions here are the
other half of the contract enforced by ``contracts.py``: the schema described
here is exactly what ``parse_plan``/``parse_evaluation`` will accept, so a
prompt change and a schema change have to move together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.base import ToolSpec

__all__ = [
    "ACT_PROMPT",
    "AGENT_PREAMBLE",
    "ANALYZE_PROMPT",
    "BLOCKED_HINT",
    "OBSERVE_PROMPT",
    "VIOLATION_PROMPT",
    "build_system_prompt",
]

AGENT_PREAMBLE = """\
You are an autonomous agent working toward a goal through a disciplined loop:

  ANALYZE -> write a plan and commit to exactly one next tool call.
  ACT     -> call that tool. Nothing else.
  OBSERVE -> grade the result against the success criteria you set, with evidence.
  ITERATE -> repeat until the goal is met, then call `submit`.

Tools are withheld from you during ANALYZE and OBSERVE -- you cannot act while \
planning or grading, even if you want to. This is intentional: it keeps your \
plan and your self-assessment honest, because neither can be rationalised in \
the same breath as the action they cover.

A run only ends when you call `submit`. Call it as soon as the goal is met -- \
not one iteration later, and not before it actually is met with unmet \
requirements swept under the rug.\
"""

ANALYZE_PROMPT = """\
Iteration {iteration} (of at most {max_iterations}). Tools are not available in \
this message -- you may not call one. Write your plan.

Respond with exactly one <plan>...</plan> block containing a single JSON object:

{{
  "goal_restatement": "the goal in your own words, so drift becomes visible",
  "steps": [{{"id": "s1", "description": "...", "done": false}}],
  "next_action": {{
    "step_id": "the step this action advances",
    "tool": "the exact name of the tool you will call next",
    "rationale": "why this tool, with these arguments, right now",
    "success_criteria": "the observable condition that will prove it worked"
  }}
}}

Carry forward steps from earlier iterations instead of restating the goal from \
scratch; mark finished ones "done": true. success_criteria must be checkable \
after the tool runs -- "it worked" is not checkable, "the file exists and \
contains X" is.
{blocked_hint}\
"""

BLOCKED_HINT = """
You have produced {n} failing evaluations in a row. Before planning the same \
approach again, state specifically what you will do differently this time, or \
switch to a different tool or strategy.\
"""

ACT_PROMPT = """\
Call `{tool}` now, for step `{step_id}`. Your stated rationale was: {rationale}

Call exactly this tool. If you have changed your mind since planning, that's \
fine -- but say so in a new <plan> next iteration, not by calling a different \
tool here.\
"""

OBSERVE_PROMPT = """\
Iteration {iteration}. Tools are not available in this message. The result of \
your action appears above. Grade it against the success criteria you set: \
{success_criteria}

Respond with exactly one <evaluation>...</evaluation> block containing a single \
JSON object:

{{
  "outcome": "success | partial | failure",
  "evidence": "quote or cite the specific tool output that justifies this outcome",
  "completed_step_ids": ["ids of steps this iteration actually finished"],
  "goal_status": "complete | in_progress | blocked",
  "next_step": "what you'll do next, in one sentence"
}}

evidence under 10 characters is rejected -- "it worked" is not evidence. If \
goal_status is "complete", plan to call `submit` next iteration rather than \
continuing to act.\
"""

VIOLATION_PROMPT = """\
Your {phase}-phase output did not satisfy its contract (attempt {attempt} of \
{max_attempts}):

{feedback}

Try again, addressing this specifically.\
"""


def build_system_prompt(
    *,
    goal: str,
    tool_specs: list[ToolSpec],
    skill_index: str = "",
    extra_instructions: str = "",
) -> str:
    sections = [
        AGENT_PREAMBLE,
        f"# Goal\n\n{goal}",
        f"# Available tools\n\n{_render_tools(tool_specs)}",
    ]
    if skill_index:
        sections.append(
            "# Skill index\n\n"
            "These are written procedures for recurring tasks. Call "
            "`load_skill(name=...)` to pull one's full body into context when its "
            "description matches what you're about to do -- don't load one that "
            "doesn't apply.\n\n" + skill_index
        )
    if extra_instructions:
        sections.append(f"# Additional instructions\n\n{extra_instructions}")
    return "\n\n".join(sections)


def _render_tools(specs: list[ToolSpec]) -> str:
    return "\n\n".join(_render_tool(s) for s in sorted(specs, key=lambda s: s.name))


def _render_tool(spec: ToolSpec) -> str:
    props = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required", []))
    args = ", ".join(
        f"{name}{'' if name in required else '?'}: {schema.get('type', 'any')}"
        for name, schema in props.items()
    )
    return (
        f"## `{spec.name}` [{spec.safety.value}]\n"
        f"{spec.description}\n"
        f"Arguments: {args or '(none)'}\n"
        f"Returns: {spec.returns or '(unspecified)'}"
    )
