# Taxonomy Validation Analysis

## What the Paper Says

From Section 3.2 ("Standardizing Failure Labels via Inter-Annotator Agreement"):

> In each round of IAA, **three expert annotators** independently label a subset of
> **five randomly selected traces** from our initial 150+ trace collection using MAST.
> We then facilitate discussions to collectively resolve any disagreements. [...]
> We conduct **three such rounds** of IAA, requiring about 10 hours in total solely
> for resolving disagreements. We measure agreement using Cohen's Kappa score,
> achieving a strong average of **κ = 0.88** in the final rounds.

From Section 3.4 ("Constructing the Multi-Agent Dataset"):

> We confirm the generalizability of our finalized MAST and the LLM annotator. We
> evaluate their performance on **two new MAS (OpenManus and Magentic-One)** with
> **two new benchmarks (MMLU and GAIA)** not part of the initial MAST development.
> An additional human IAA round on these out-of-domain traces using the finalized
> MAST yields a strong Cohen's Kappa score of **0.79**.

So the validation data consists of:
- **Rounds 1-3**: 5 traces each = 15 traces (taxonomy refinement, κ = 0.88)
- **Generalizability round**: 4 traces (out-of-domain validation, κ = 0.79)
- **Total**: 19 traces with 3-annotator labels

The paper also mentions **MAST-Data-human** (21 traces) in Table 1, which is slightly
more than the 19 in the HF file. The 2-trace discrepancy is unexplained.

### Original 5 Frameworks and 2 Task Types

The taxonomy was developed from traces of these 5 frameworks covering 2 task domains:

| Framework  | Task type              | Benchmark      |
|------------|------------------------|----------------|
| HyperAgent | Programming            | SWE-Bench-Lite |
| ChatDev    | Programming            | ProgramDev     |
| MetaGPT    | Programming            | ProgramDev     |
| AppWorld   | General agent          | Test-C         |
| AG2        | Math problem-solving   | GSM-Plus       |

The generalizability round then introduced **new frameworks** (Magentic-One, OpenManus)
and **new benchmarks/task types** (GAIA for general agent, MMLU for knowledge QA).

## Available Data

### HF: `MAD_human_labelled_dataset.json` — 19 traces

Each record has 3 independent annotator labels per failure mode.

#### Round 1 (5 traces) — Initial taxonomy calibration

| # | MAS        | Benchmark       | Failure modes per trace | Label numbering  |
|---|------------|-----------------|------------------------|------------------|
| 0 | AppWorld   | Test-C          | 18                     | 1.1–4.3 scheme   |
| 1 | HyperAgent | SWE-Bench-Lite  | 18                     | 1.1–4.3 scheme   |
| 2 | AG2        | GSM-Plus        | 18                     | 1.1–4.3 scheme   |
| 3 | ChatDev    | ProgramDev      | 18                     | 1.1–4.3 scheme   |
| 4 | MetaGPT    | ProgramDev      | 18                     | 1.1–4.3 scheme   |

Round 1 used an **earlier numbering** with 18 failure modes across 4 categories
(categories 1–4, not the final 1–3). Labels include modes not in the final MAST:
- "3.4 Waiting for known information"
- "4.2 Lack of result verification" / "4.3 Lack of critical verification"

#### Round 2 (5 traces) — Refinement

| # | MAS        | Benchmark       | Failure modes per trace | Label numbering  |
|---|------------|-----------------|------------------------|------------------|
| 5 | AppWorld   | Test-C          | 17                     | Intermediate     |
| 6 | HyperAgent | SWE-Bench-Lite  | 17                     | Intermediate     |
| 7 | AG2        | GSM-Plus        | 17                     | Intermediate     |
| 8 | ChatDev    | ProgramDev      | 17                     | Intermediate     |
| 9 | MetaGPT    | ProgramDev      | 17                     | Intermediate     |

Reduced from 18 to 17 modes — some modes were merged or dropped between rounds.

#### Round 3 (5 traces) — Final agreement (κ = 0.88)

| #  | MAS        | Benchmark       | Failure modes per trace | Label numbering |
|----|------------|-----------------|------------------------|-----------------|
| 10 | HyperAgent | SWE-Bench-Lite  | 17                     | Near-final      |
| 11 | AppWorld   | Test-C          | 17                     | Near-final      |
| 12 | AG2        | GSM-Plus        | 17                     | Near-final      |
| 13 | ChatDev    | ProgramDev      | 17                     | Near-final      |
| 14 | MetaGPT    | ProgramDev      | 17                     | Near-final      |

#### Generalizability Round (4 traces) — Out-of-domain (κ = 0.79)

| #  | MAS      | Benchmark | Failure modes per trace | Label numbering     |
|----|----------|-----------|------------------------|---------------------|
| 15 | ChatDev  | MMLU      | 14                     | Final MAST (1.1–3.3)|
| 16 | MetaGPT  | MMLU      | 14                     | Final MAST (1.1–3.3)|
| 17 | GAIA     | Magentic  | 14                     | Final MAST (1.1–3.3)|
| 18 | GAIA     | OpenManus | 14                     | Final MAST (1.1–3.3)|

Only the generalizability round uses the **final 14-mode MAST taxonomy** (1.1–3.3).
Note: MAS/benchmark fields appear swapped for traces 17-18 (GAIA listed as MAS name,
Magentic/OpenManus as benchmark — likely a data entry error).

## PDF Analysis: `inter_annotator_agreement_annotations/`

### What the PDFs Actually Contain

All four filenames end with `_trim`, indicating they are cropped exports from larger
spreadsheets. Examining the actual content:

| PDF file | Column headers | Label scheme | Trace groups | Pages |
|----------|---------------|-------------|-------------|-------|
| `round1_trim.pdf` | MMLU-ChatDev, MMLU-MetaGPT, GAIA-Magentic, GAIA-OpenManus | **Final 14 modes** (1.1–3.3) | 4 traces | 1 of 1 |
| `round_2_trim.pdf` | AppWorld Set 2, HyperAgent Set 2, AG2 Set 2, ChatDev Set 2, MetaGPT Set 2 | **Interim 17 modes** (1.1–3.3 with extras) | 5 traces | 1 of 2 (page 2 mostly empty) |
| `round3_trim.pdf` | AppWorld Set 3, HyperAgent Set 3, AG2 Set 3, ChatDev Set 3, MetaGPT Set 3 | **Interim 17 modes** (1.1–3.3 with extras) | 5 traces | 1 of 2 (page 2 mostly empty) |
| `new_benchmarks_trim.pdf` | MMLU-ChatDev, MMLU-MetaGPT, GAIA-Magentic, GAIA-OpenManus | **Final 14 modes** (1.1–3.3) | 4 traces | 1 of 1 |

### Problem: `round1_trim.pdf` Is Mislabeled

`round1_trim.pdf` and `new_benchmarks_trim.pdf` are **identical**. Both contain:
- The same 4 trace groups: MMLU-ChatDev, MMLU-MetaGPT, GAIA-Magentic, GAIA-OpenManus
- The same **final 14-mode** label scheme (1.1–3.3)
- The same TRUE/FALSE annotations and annotator notes

This cannot be Round 1 because:
- Round 1 per the HF data used the **original 5 frameworks** (AppWorld, HyperAgent, AG2,
  ChatDev, MetaGPT) — not MMLU/GAIA with Magentic/OpenManus
- Round 1 per the HF data had **18 modes** in a 4-category scheme — not 14 modes in 3
  categories
- MMLU and GAIA are the **out-of-domain benchmarks** introduced only in the
  generalizability round

**Conclusion**: The file named `round1_trim.pdf` is actually a **duplicate of the
generalizability round** (`new_benchmarks_trim.pdf`). The real Round 1 spreadsheet
was never released.

### What's Available vs. Missing

| Round | PDF available? | What we see |
|-------|---------------|-------------|
| Round 1 (18-mode, 5 original frameworks) | **No** — file named "round1" is mislabeled | Lost/unreleased |
| Round 2 (17-mode, 5 original frameworks) | **Yes** — `round_2_trim.pdf` | Trimmed, notes truncated |
| Round 3 (17-mode, 5 original frameworks) | **Yes** — `round3_trim.pdf` | Trimmed, notes truncated |
| Generalizability (14-mode, new frameworks) | **Yes** — `new_benchmarks_trim.pdf` | Trimmed, notes truncated |
| Generalizability (duplicate) | `round1_trim.pdf` | Identical to new_benchmarks |

### PDFs Are Trimmed

Every PDF shows evidence of truncation from the original spreadsheets:

- Annotator notes are cut mid-sentence:
  - `"strB has mixed case and strC is all lowercase, bu..."`
  - `"Line 302 it states that _add_prefix_for_feature_nam..."`
  - `"The simple_note agent should mark a task done, no..."`
- Discussion notes visible in Round 3 are fragments:
  - `"maybe add task ambiguity, or discuss it"`
  - `"This one is so interesting! we should make note of it that the agent 'lied'"`
  - `"did agents lie or were they simply insufficient or did not obey their own ideas?"`
- The original spreadsheets likely also contained trace text/links and
  kappa computation sheets, none of which appear in the trimmed exports.

### PDF vs. HF Data: Matching

The PDFs that are real (Round 2, Round 3, Generalizability) contain the **same
annotations** as the corresponding HF JSON records, confirming they are different
exports of the same data — not additional traces.

| PDF | Matches HF records | Frameworks shown |
|-----|-------------------|------------------|
| `round_2_trim.pdf` | HF #5-9 (Round 2) | AppWorld, HyperAgent, AG2, ChatDev, MetaGPT |
| `round3_trim.pdf` | HF #10-14 (Round 3) | AppWorld, HyperAgent, AG2, ChatDev, MetaGPT |
| `new_benchmarks_trim.pdf` | HF #15-18 (Generalizability) | ChatDev/MMLU, MetaGPT/MMLU, Magentic/GAIA, OpenManus/GAIA |

## Taxonomy Evolution Across Rounds

The failure mode labels changed significantly between rounds:

| Aspect | Round 1 (HF) | Round 2 (HF + PDF) | Round 3 (HF + PDF) | Generalizability (HF + PDF) |
|--------|-------------|--------------------|--------------------|---------------------------|
| # of modes | 18 | 17 | 17 | 14 (final) |
| # of categories | 4 | 3 | 3 | 3 |
| Numbering | 1.x, 2.x, 3.x, 4.x | 1.1–3.3 + extras | 1.1–3.3 + extras | 1.1–3.3 |

Notable label changes observed:
- Round 1 had "3.4 Waiting for known information" → dropped or merged
- Round 1 had "4.2 Lack of result verification" / "4.3 Lack of critical verification"
  → became 3.2 "No or Incorrect Verification" / 3.3 "Weak Verification"
- "1.2 Inconsistency between reasoning and action" → moved to 2.6
- "2.6 Disobey role specification" → moved to 1.2
- Category 4 was eliminated; its modes redistributed into categories 1-3
- Round 2→3: "Unbatched repetitive execution" and "Step repetition" still separate;
  merged into single 1.3 only for final
- Round 2→3: "Undetected ambiguities" and "Disagreement induced inaction" still
  separate; merged/dropped for final

## Key Observations

1. **The 19 HF traces are purely validation data**, not from the 150-trace GT analysis.
   They were selected to measure inter-annotator agreement and refine definitions.

2. **The taxonomy was still evolving during IAA.** Round 1 had 18 modes in 4 categories;
   the final taxonomy has 14 modes in 3 categories. Modes were merged, renumbered, and
   reorganized between rounds. This means Round 1-3 annotations **cannot be directly
   compared** using the final MAST codes.

3. **Only the generalizability round (4 traces) uses the final taxonomy.** If you need
   human annotations with the exact same label scheme as the LLM-annotated dataset,
   only traces 15-18 qualify.

4. **`round1_trim.pdf` is mislabeled.** It is identical to `new_benchmarks_trim.pdf`
   (generalizability round with final 14-mode MAST). It shows MMLU/GAIA traces, not the
   original 5 frameworks. The real Round 1 PDF (18-mode scheme, original frameworks)
   was never released.

5. **All PDFs are trimmed.** Filenames end with `_trim`. Annotator notes are truncated
   mid-sentence. The full spreadsheets with complete discussion notes, trace links, and
   kappa computations were not released.

6. **PDFs duplicate the HF data.** Round 2 PDF matches HF #5-9, Round 3 PDF matches
   HF #10-14, new_benchmarks PDF matches HF #15-18. They provide no additional traces
   or annotations beyond what's in the HF JSON.

7. **Data quality issues**:
   - MAS and benchmark names are swapped for HF traces 17-18 (GAIA as `mas_name`)
   - `round1_trim.pdf` mislabeled as Round 1
   - Paper mentions 21 human-annotated traces; HF has 19 (2 unexplained)

## Summary: Two Distinct Data Phases

| Phase | Purpose | Traces | Annotators | Label scheme | Available |
|-------|---------|--------|------------|--------------|-----------|
| GT Analysis | Build taxonomy | ~150 | 6 experts (single per trace) | Pre-taxonomy (22 or 13 labels) | 61 in repo `*_human.json` |
| IAA Rounds 1-3 | Refine & validate taxonomy | 15 (5 per round) | 3 per trace | Evolving (18→17→17 modes) | 15 in HF JSON; Round 2+3 PDFs |
| Generalizability | Out-of-domain validation | 4 | 3 per trace | Final 14-mode MAST | 4 in HF JSON + 2 identical PDFs |
| **Total accounted** | | **~80** | | | |
| **Paper claims** | | **150 (GT) + ~21 (IAA)** | | | |

### What Is Not Available

- Real Round 1 PDF (18-mode scheme on original 5 frameworks)
- Full untrimmed annotation spreadsheets with complete discussion notes
- Kappa computation sheets
- Human annotations for ~89 GT traces (AppWorld, ChatDev, MetaGPT)
- The 2 additional traces to reach the paper's 21-trace MAST-Data-human claim
