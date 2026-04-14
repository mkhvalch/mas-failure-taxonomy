# MAD Dataset Analysis Notes

## Paper Claims vs. Available Data

The paper ("Why Do Multi-Agent LLM Systems Fail?", arXiv:2503.13657) claims
**1600+ annotated traces** across 7 MAS frameworks.

## HuggingFace Dataset (mcemri/MAD)

### `MAD_full_dataset.json` (LLM-annotated)

- **1,242 records** total
- Fields: `mas_name`, `llm_name`, `benchmark_name`, `trace_id`, `trace`, `mast_annotation`
- `mast_annotation` contains binary 0/1 for all 14 MAST failure modes (1.1–3.3)

| MAS Framework | Traces |
|---------------|--------|
| AG2           | 597    |
| MetaGPT       | 230    |
| Magentic      | 195    |
| ChatDev       | 130    |
| OpenManus     | 30     |
| AppWorld      | 30     |
| HyperAgent    | 30     |

| LLM    | Traces |
|--------|--------|
| GPT-4o | 919    |
| Claude | 323    |

| Benchmark       | Traces |
|-----------------|--------|
| ProgramDev      | 390    |
| GSM             | 223    |
| Olympiad        | 206    |
| GAIA            | 195    |
| MMLU            | 168    |
| SWE-Bench-Lite  | 30     |
| Test-C          | 30     |

### `MAD_human_labelled_dataset.json` (human-annotated)

- **19 records** total
- Fields: `round`, `mas_name`, `benchmark_name`, `trace_id`, `trace`, `annotations`
- `annotations` is a list of structs: `{annotator_1: bool, annotator_2: bool, annotator_3: bool, failure mode: str}`
- Note: **no `llm_name` field** — schema differs from full dataset (causes HF viewer error)

| MAS Framework | Traces |
|---------------|--------|
| ChatDev       | 4      |
| MetaGPT       | 4      |
| AG2           | 3      |
| AppWorld      | 3      |
| HyperAgent    | 3      |
| GAIA          | 2      |

### Overlap

- 12 traces overlap between full and human datasets (by mas_name + benchmark + trace_id)
- 7 human-labelled traces are not in the full dataset
- Combined unique: ~1,029 distinct (mas_name, benchmark, trace_id) tuples
- Simple sum: 1,242 + 19 = 1,261 records

## Repository `traces/` Directory

Raw trace files in their original formats (not annotated):

| Source                          | File Count | Format           |
|---------------------------------|-----------|------------------|
| AG2                             | 38        | `*_human.json`   |
| HyperAgent                      | 223       | `.json`          |
| AppWorld                        | 16        | `.txt`           |
| MagenticOne (GAIA, 3 levels)    | 165       | directories      |
| OpenManus (GAIA)                | 30        | `.log`           |
| math_interventions/org_traces   | 200       | trace files      |
| math_interventions/prompt_traces| 200       | trace files      |
| math_interventions/topology_traces| 200     | trace files      |
| mmlu/chatdev_mmlu               | 61        | log files        |
| mmlu/metagpt_mmlu               | 30        | log files        |
| programdev/chatdev              | 30        | trace files      |
| programdev/metagpt              | 30        | trace files      |
| **Total**                       | **~1,223**|                  |

## Repo vs. HF: Side-by-Side Comparison

**Not a single (MAS, Benchmark) pair matches between the repo and HF.**

| MAS        | Benchmark (HF)  | HF Count | Repo Location                     | Repo Count | Notes                                           |
|------------|-----------------|----------|-----------------------------------|------------|------------------------------------------------|
| AG2        | GSM             | 223      | `traces/AG2/` (38 json total)     | 38 total   | HF has 597 AG2 traces; repo has 38 across all  |
| AG2        | MMLU            | 168      | (same 38 files)                   | —          | No per-benchmark split in repo                  |
| AG2        | Olympiad        | 206      | (same 38 files)                   | —          | No per-benchmark split in repo                  |
| AppWorld   | Test-C          | 30       | `traces/AppWorld/`                | 16         | 14 traces missing from repo                     |
| ChatDev    | ProgramDev      | 130      | `traces/programdev/chatdev/`      | 30 dirs    | HF has 100 more (likely multi-LLM runs)         |
| HyperAgent | SWE-Bench-Lite  | 30       | `traces/HyperAgent/`              | 223        | Opposite: repo has 193 MORE than HF             |
| Magentic   | GAIA            | 195      | `traces/MagenticOne_GAIA/`        | 165        | 30 traces missing from repo                     |
| MetaGPT    | ProgramDev      | 230      | `traces/programdev/metagpt/`      | 30 files   | HF has 200 more                                 |
| OpenManus  | ProgramDev      | 30       | `traces/OpenManus_GAIA/`          | 30 logs    | Count matches but HF says ProgramDev, repo says GAIA |

Additional repo traces **not represented in HF**:

| Repo Location                          | Count | Status in HF                          |
|----------------------------------------|-------|---------------------------------------|
| `traces/math_interventions/org_traces` | 200   | Likely folded into AG2/GSM+Olympiad   |
| `traces/math_interventions/prompt_traces` | 200 | Likely folded into AG2                |
| `traces/math_interventions/topology_traces` | 200 | Likely folded into AG2              |
| `traces/mmlu/chatdev_mmlu`             | 61    | Possibly subset of AG2/MMLU or ChatDev |
| `traces/mmlu/metagpt_mmlu`             | 30    | Not clear                              |

### Key Observations

1. **The repo and HF dataset are not the same data.** The HF dataset contains traces
   that were re-collected or expanded beyond what's in the repo (e.g., AG2 has 597 in HF
   vs. 38 in repo; ChatDev has 130 vs. 30; MetaGPT has 230 vs. 30).

2. **The repo is an earlier/partial snapshot.** The `traces/` folder appears to be from
   an earlier stage of the project, while the HF dataset is the final published artifact.

3. **Benchmark labels differ.** OpenManus traces in the repo are from GAIA, but HF labels
   them as ProgramDev. The repo's `math_interventions/` are separate, but HF merges them
   under AG2 with specific benchmark names (GSM, Olympiad).

4. **HyperAgent is the exception.** The repo has 223 traces but HF only published 30 —
   the rest were collected but not included in the annotated release.

## Other Discrepancies

1. **Paper says 1600+, HF has 1,242 annotated + 19 human-labelled.**
   Possible explanations:
   - The 1600+ figure may count all annotation rounds (human + LLM, including duplicates)
   - Some traces may not have been included in the public release
   - The math_interventions (600 traces in 3 variants) may inflate the count differently

2. **Schema mismatch between the two HF files.**
   `MAD_human_labelled_dataset.json` uses `round` + `annotations` instead of `llm_name` +
   `mast_annotation`. This breaks the HF Dataset Viewer and requires loading them separately.

## Working Dataset for Experiments

Use `MAD_full_dataset.json` (1,242 traces) as the primary experimental dataset.
The human-labelled file (19 traces) is useful for validation / agreement analysis only.
The 61 `*_human.json` files in the repo provide additional human annotations but use
an earlier label scheme that doesn't directly map to the final 14 MAST codes.
