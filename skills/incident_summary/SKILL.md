---
name: incident_summary
description: Structure for turning raw logs, error traces, or a bug report into a report a human can act on.
when_to_use: When the goal asks you to investigate a failure and report findings, not just fix code.
version: 1.0.0
tools: [file_system, execute_code, analyze_data, scratchpad]
tags: [debugging, reporting]
---

## Procedure

1. **Reproduce or explain why you couldn't.** "I could not reproduce this
   locally because X" is a legitimate, reportable finding. Do not silently
   move past a failed reproduction attempt into speculation without labelling
   the speculation as such.

2. **Separate the timeline from the interpretation.** First establish *what
   happened, in order, with timestamps and evidence* (log lines, exit codes,
   file contents). Only after that, write *why you think it happened*. A
   report that interleaves the two makes it impossible for a reader to tell
   which parts are fact and which are your inference.

3. **Quote, don't paraphrase, the evidence that matters.** "The service
   restarted several times" is weaker and harder to verify than "the log shows
   3 restarts between 14:02 and 14:05, each preceded by
   `OOMKilled`." Paraphrasing loses the detail a human needs to confirm your
   read is right.

4. **Report the blast radius, not just the root cause.** What else does this
   affect? What didn't you have time to check? An incident report that nails
   the root cause but is silent on scope leaves the human to redo your
   investigation.

5. **Grade your own confidence per claim, not just once at the end.** "High
   confidence: this is the root cause (log line X)." "Low confidence: this
   might also affect the batch job, untested." One overall confidence score
   flattens the difference between what you verified and what you guessed.

## Anti-patterns

- Reporting "fixed" when what actually happened was "the symptom stopped
  reproducing, cause unconfirmed."
- Root-causing from a single data point when the tooling (`analyze_data`,
  `execute_code`) was available to check whether the pattern holds across the
  full log.
- Burying the one-sentence summary a human needs first under five paragraphs
  of investigation narrative. Lead with the finding; the narrative is the
  appendix.
