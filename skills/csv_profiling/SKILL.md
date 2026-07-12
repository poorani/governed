---
name: csv_profiling
description: Systematic first-pass profiling of an unfamiliar tabular dataset before answering any question about it.
when_to_use: Before answering any analytical question about a CSV, TSV, JSON, or Parquet file you haven't inspected yet.
version: 1.0.0
tools: [analyze_data, file_system, scratchpad]
tags: [data, analysis]
---

## Procedure

1. **Shape and schema first.** `analyze_data(operation="profile")`. Never `head`
   before `profile` -- you will anchor on the first five rows and miss that
   column `date` is 40% null or that `region` has 400 distinct values instead
   of the 5 you assumed.

2. **Write the schema to the scratchpad.** Column names, dtypes, and anything
   surprising (high null rate, suspicious cardinality, an ID column that isn't
   unique). The transcript compacts; the scratchpad doesn't. If the run is
   long, you will need this again after it's gone from context.

3. **Sample, don't dump.** `head` is for spot-checking format (is the date
   `2026-07-08` or `07/08/2026`?), not for understanding the data. For
   anything about distributions or totals, use `aggregate` or `value_counts` --
   they answer the question directly instead of making you eyeball 50 rows.

4. **Check cardinality before filtering or grouping.** A `groupby` on a
   near-unique column produces a useless one-row-per-group table and burns an
   iteration. `profile` already told you the unique count -- read it.

5. **State assumptions about nulls and duplicates explicitly** in your
   evaluation evidence, even if the goal didn't ask about data quality. A
   number computed while silently dropping 15% of rows is not the number the
   user asked for, and it will look identical to one that isn't.

## Anti-patterns

- Loading the whole file into `execute_code` and printing it. That's what
  `analyze_data` exists to prevent -- it bounds output at 50 rows and pushes
  you toward aggregation instead of enumeration.
- Reporting a total or an average without having checked null handling first.
- Re-running `profile` every iteration "just to check." Write it to the
  scratchpad once and read from there.
