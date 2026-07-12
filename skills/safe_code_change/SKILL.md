---
name: safe_code_change
description: Judgement for editing an existing file in the workspace without a human reviewing every line first.
when_to_use: Before writing to any file that already exists and contains code, config, or another agent's prior output.
version: 1.0.0
tools: [file_system, execute_code, scratchpad]
tags: [code, safety]
---

## Procedure

1. **Read the whole file before touching it.** Not a `head`. A partial read of
   a config file means you don't know what your write is about to clobber.
   `file_system(operation="read")` is cheap; a destroyed file is not.

2. **State what you are preserving, not just what you're changing.** Before
   the write, name in your plan's rationale the parts of the file that must
   survive unchanged. This is what makes an accidental full-file overwrite
   visible in the trace instead of silent.

3. **Prefer the smallest edit that satisfies the goal.** A one-line fix
   rewritten as a full-file replace is indistinguishable, in the trace, from a
   bug that nuked the rest of the file. If the tool set only offers whole-file
   write (as `file_system` does), read-modify-write: read the current content,
   change only what's needed in memory, write the result back.

4. **Verify after writing, not before submitting.** Read the file back, or run
   it, immediately after the write -- in the *same* plan, as the next step, not
   three iterations later after you've moved on. A write that silently failed
   or truncated is a bug you want to catch this iteration, not in the final
   evaluation.

5. **Never edit a test to make it pass without saying so.** If the fix is
   "the test was wrong," that's a legitimate outcome -- but it belongs in your
   evidence, explicitly, not folded silently into "tests pass now." A human
   reading the trace later needs to be able to tell the difference between
   "I fixed the bug" and "I fixed the test."

6. **A fix for a bug you couldn't reproduce is a guess, and must be labelled
   one.** State the confidence accordingly in `submit` -- don't report 0.9
   confidence on a change you couldn't verify actually fixes anything.

## Anti-patterns

- Writing a whole new file over an existing one when only a section changed.
- Running `execute_code` to "clean up" or delete files as a side effect of an
  unrelated task -- that belongs in its own planned step with its own
  rationale, not bundled into something else.
- Treating "the code runs without an exception" as equivalent to "the code is
  correct." Exit code 0 is necessary evidence, not sufficient evidence.
