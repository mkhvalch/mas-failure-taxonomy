# Human Annotations Analysis

## What the Paper Says About the 150 Traces

From Section 3.1 ("Data Collection with Grounded Theory Analysis"):

> We first collect **150 traces from five MAS frameworks**, which are closely examined
> by **six human experts**. [...] We adopt the Grounded Theory (GT) approach. [...]
> This initial process requires significant human effort, **over 20 hours of annotation
> per expert** for these 150 traces.

The 150 traces were used for **taxonomy development** via Grounded Theory, not for
validation. The five frameworks used at this stage were: **HyperAgent, AppWorld, AG2,
ChatDev, and MetaGPT** (no Magentic-One or OpenManus yet).

The 150 traces were analyzed using open coding, constant comparative analysis, memoing,
and theorizing until **theoretical saturation** was reached.

## What Exists in the Repo

### Source 1: `traces/AG2/*_human.json` — 31 traces

Single-annotator notes using a **pre-taxonomy label scheme with 22 failure categories**.

Structure per file:
- `instance_id`, `problem_statement`, `other_data` (includes `correct: true/false`), `trajectory`
- `note.text` — freeform annotator observations
- `note.options` — 22-item yes/no checklist

Task outcome: 5 correct, 26 incorrect (83.9% failure rate).

**Label scheme (22 categories — pre-taxonomy):**

| Label | Flagged in N/31 | Rate |
|-------|----------------|------|
| Evaluator agent fails to be critical | 19 | 61.3% |
| No attempt to verify outcome | 7 | 22.6% |
| Proceed with incorrect assumptions | 7 | 22.6% |
| Fail to detect ambiguities/contradictions | 5 | 16.1% |
| Fail to elicit clarification | 4 | 12.9% |
| Invented content | 3 | 9.7% |
| Derailing from task objectives | 2 | 6.5% |
| Unaware of stopping conditions | 1 | 3.2% |
| Ignoring good suggestions from other agent | 1 | 3.2% |
| Poor adherence to specified constraints | 1 | 3.2% |
| (12 more labels never flagged) | 0 | 0% |

Labels unique to AG2 (not in HyperAgent): "Proceed with incorrect assumptions",
"Fail to elicit clarification", "Tendency to overachieve", "Underperform by waiting
on instructions", "Evaluator agent fails to be critical", "Poor adherence to specified
constraints", "Redundant conversation turns", "Difficulty in agreeing with agents",
"Unaware of stopping conditions", "Waiting on agents to discover known insights",
"Blurring roles" (plural).

### Source 2: `traces/HyperAgent/*_human.json` — 30 traces

Single-annotator notes using a **different pre-taxonomy label scheme with 12-13 categories**.

Structure per file:
- `instance_id`, `problem_statement`, `other_data`, `trajectory`
- `note.text` — freeform annotator observations
- `note.options` — 12-13 item yes/no checklist (varies; some files include "Blurring role")

**Label scheme (13 categories — pre-taxonomy):**

| Label | Flagged in N/30 | Rate |
|-------|----------------|------|
| Trajectory restart | 13 | 43.3% |
| No attempt to verify outcome | 10 | 33.3% |
| Step repetition | 7 | 23.3% |
| Derailing from task objectives | 7 | 23.3% |
| Invented content | 6 | 20.0% |
| Withholding relevant information | 4 | 13.3% |
| Blurring role | 4 | 13.3% |
| Fail to detect ambiguities/contradictions | 4 | 13.3% |
| Misalignment between internal thoughts and response | 2 | 6.7% |
| Ignoring good suggestions from other agent | 2 | 6.7% |
| Discontinued reasoning | 2 | 6.7% |
| Claiming that a task is done while it is not true | 1 | 3.3% |

Labels unique to HyperAgent (not in AG2): "Not exploring every option proposed by
other agents", "Blurring role" (singular).

### Total: 61 Traces with Human Annotations

Combined: 31 (AG2) + 30 (HyperAgent) = **61 traces** in the repo with structured
human annotations.

## The Gap: Where Are the Other ~89 Traces?

The paper says 150 traces from **five** frameworks. The repo only has human-annotated
files for two of them:

| Framework | Human-annotated in repo | Expected from paper |
|-----------|------------------------|---------------------|
| AG2       | 31                     | ~30                 |
| HyperAgent| 30                     | ~30                 |
| AppWorld  | 0                      | ~30                 |
| ChatDev   | 0                      | ~30                 |
| MetaGPT   | 0                      | ~30                 |
| **Total** | **61**                 | **~150**            |

The raw trace files for AppWorld (16), ChatDev (30+61), and MetaGPT (30) are present
in the repo, but **without the `_human.json` annotation wrappers**. The human expert
annotations for those ~89 traces were either:
- Never committed to the repo (kept in private spreadsheets/docs)
- Lost during the iterative GT coding process (memos rather than structured files)
- Only recorded in the annotators' working notes during the 120+ person-hours of analysis

## Overlap Between Repo and HF Human Annotations

6 of the 19 HF IAA traces use the **same underlying execution traces** as files in the
repo. These traces were annotated **twice**: once during the GT phase (repo, single
annotator, pre-taxonomy labels) and again during formal IAA validation (HF, 3
annotators, evolving MAST labels).

| HF # | Round   | MAS        | Repo file                                         |
|-------|---------|------------|----------------------------------------------------|
| 1     | Round 1 | HyperAgent | `traces/HyperAgent/astropy__astropy-12907_human.json` |
| 2     | Round 1 | AG2        | `traces/AG2/02da9c1f-7c36-5739-b723-33a7d4f8e7e7_human.json` |
| 6     | Round 2 | HyperAgent | `traces/HyperAgent/scikit-learn__scikit-learn-25570_human.json` |
| 7     | Round 2 | AG2        | `traces/AG2/6891537b-54d7-5f49-b9e4-790b5e6a5de6_human.json` |
| 10    | Round 3 | HyperAgent | `traces/HyperAgent/astropy__astropy-14365_human.json` |
| 12    | Round 3 | AG2        | `traces/AG2/8ebbc1ed-c0c2-522e-8949-60cb5dd24938_human.json` |

The other 13 HF traces (AppWorld, ChatDev, MetaGPT, Magentic, OpenManus) have **no
corresponding `_human.json`** in the repo. And 55 repo `_human.json` files (28 AG2 +
27 HyperAgent) have **no corresponding entry** in the HF dataset.

### Unique trace counts

| Source | Unique traces | Overlap |
|--------|--------------|---------|
| Repo `_human.json` only | 55 | — |
| HF human-labelled only | 13 | — |
| Both (same trace, different annotations) | 6 | 6 |
| **Total unique traces with any human annotation** | **74** | |

## Key Observations

1. **Two different label schemes were used.** AG2 files have 22 categories; HyperAgent
   files have 12-13. Neither matches the final 14-mode MAST taxonomy. This confirms
   these annotations are from the **Grounded Theory development phase**, before the
   taxonomy was finalized.

2. **The labels evolved significantly.** Some early categories were merged, renamed, or
   dropped during taxonomy refinement:
   - "Evaluator agent fails to be critical" (AG2's #1 failure) → split into 3.2/3.3
   - "Trajectory restart" → became 2.1 "Conversation Reset"
   - "Invented content" → not a final MAST mode (considered model capability issue)
   - "Proceed with incorrect assumptions" → folded into 2.2/2.3
   - "Blurring role(s)" → became 1.2 "Disobey Role Specification"

3. **Single annotator per trace.** These files each have one annotator's judgment, unlike
   the IAA study which used 3 independent annotators per trace. These are **working notes
   from the GT analysis**, not formal validation data.

4. **Rich freeform text.** The `note.text` fields contain detailed qualitative observations
   that motivated the taxonomy categories — these are the "memos" from the GT process.

5. **6 traces have dual annotations.** These are valuable for comparing how the same trace
   was labeled before vs. after taxonomy formalization, and by 1 vs. 3 annotators.

## Consolidated Data

All human annotations (repo + HF) are compiled in `all_human_annotations.json`.
Each record includes a `source` field indicating provenance (`repo`, `hf`, or `both`).
