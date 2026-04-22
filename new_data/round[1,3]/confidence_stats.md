# Per-trace confidence stats (19 IAA traces)

Generated from `iaa_all19_confidence.json` (labels + confidence) and
`final_iaa_19_traces.json` (trace text / lengths).

## Per-trace stats

| id | round | MAS | benchmark | chars | n+ | mean conf (+) | mean conf (−) | overall | min cell |
|---:|-------|-----|-----------|------:|---:|------:|------:|------:|------:|
| 0  | Round 1 | AppWorld   | Test-C           |  26,714 | 2 | 0.30 | 0.22 | 0.23 | 0.09 |
| 1  | Round 1 | HyperAgent | SWE-Bench-Lite   | 265,138 | 2 | 0.30 | 0.22 | 0.23 | 0.09 |
| 2  | Round 1 | AG2        | GSM-Plus         |   8,221 | 2 | 0.30 | 0.22 | 0.23 | 0.09 |
| 3  | Round 1 | ChatDev    | ProgramDev       | 305,616 | 8 | 0.26 | 0.23 | 0.24 | 0.09 |
| 4  | Round 1 | MetaGPT    | ProgramDev       |   2,659 | 8 | 0.26 | 0.23 | 0.24 | 0.09 |
| 5  | Round 2 | AppWorld   | Test-C           |  33,933 | 1 | 0.30 | 0.39 | 0.38 | 0.15 |
| 6  | Round 2 | HyperAgent | SWE-Bench-Lite   |  46,376 | 1 | 0.30 | 0.39 | 0.38 | 0.15 |
| 7  | Round 2 | AG2        | GSM-Plus         |   6,660 | 1 | 0.30 | 0.39 | 0.38 | 0.15 |
| 8  | Round 2 | ChatDev    | ProgramDev       | 305,616 | 2 | 0.25 | 0.40 | 0.38 | 0.15 |
| 9  | Round 2 | MetaGPT    | ProgramDev       |   4,158 | 2 | 0.25 | 0.40 | 0.38 | 0.15 |
| 10 | Round 3 | HyperAgent | SWE-Bench-Lite   | 644,556 | 4 | 0.68 | 0.60 | 0.62 | 0.16 |
| 11 | Round 3 | AppWorld   | Test-C           |  34,565 | 4 | 0.68 | 0.60 | 0.62 | 0.16 |
| 12 | Round 3 | AG2        | GSM-Plus         |  11,568 | 4 | 0.68 | 0.60 | 0.62 | 0.16 |
| 13 | Round 3 | ChatDev    | ProgramDev       | 314,337 | 2 | 0.48 | 0.65 | 0.63 | 0.24 |
| 14 | Round 3 | MetaGPT    | ProgramDev       |   2,503 | 2 | 0.48 | 0.65 | 0.63 | 0.24 |
| 15 | Generalizability | ChatDev | MMLU        | 308,271 | 3 | 0.78 | 1.00 | 0.95 | 0.67 |
| 16 | Generalizability | MetaGPT | MMLU        |   6,569 | 3 | 0.78 | 1.00 | 0.95 | 0.67 |
| 17 | Generalizability | GAIA    | Magentic    |  55,842 | 3 | 0.78 | 1.00 | 0.95 | 0.67 |
| 18 | Generalizability | GAIA    | OpenManus   |  46,209 | 0 |  —   | 0.95 | 0.95 | 0.67 |

## Aggregates by round

| round            |  n | avg chars | avg n+ | mean conf |
|------------------|---:|---------:|------:|---------:|
| Round 1          |  5 | 121,670  |  4.4  | **0.24** |
| Round 2          |  5 |  79,349  |  1.4  | **0.38** |
| Round 3          |  5 | 201,506  |  3.2  | **0.62** |
| Generalizability |  4 | 104,223  |  2.2  | **0.95** |

## Column definitions

- **id** — Trace index (0–18) as assigned when we extracted the 19 IAA traces
  from HuggingFace. Matches the `index` field in `final_iaa_19_traces.json`.
  Not a paper-side ID; just our ordering.

- **round** — Which IAA annotation round the trace came from:
  - *Round 1*: earliest, used an 18-mode taxonomy (`early-mast-18`). Labels
    were produced before the authors converged on final mode definitions.
  - *Round 2*: same 5 MAS/benchmarks as Round 1, re-annotated with a revised
    17-mode interim taxonomy (`interim-17`).
  - *Round 3*: fresh traces, still using the 17-mode interim taxonomy, but the
    annotators had settled into their definitions.
  - *Generalizability*: a final post-hoc check on 4 extra MAS/benchmarks,
    annotated directly with the final 14 modes.

- **MAS** — The multi-agent system that produced the trace (AppWorld,
  HyperAgent, AG2, ChatDev, MetaGPT, GAIA-Magentic, GAIA-OpenManus).

- **benchmark** — The task benchmark the MAS was run against (GSM-Plus,
  SWE-Bench-Lite, ProgramDev, MMLU, Test-C, Magentic, OpenManus). Paired with
  MAS since each system was evaluated on its native benchmark.

- **chars** — Raw character length of the trace text. Proxy for how much
  content the LLM judge has to read. Anything over ~200K chars can't fit in
  our pipeline without truncation.

- **n+** — Number of failure modes labeled **positive** (mode present in the
  trace) by the humans, after majority-vote consensus and mapping to the
  final-14 taxonomy. Out of 14 possible modes. Higher n+ means humans flagged
  more distinct failures.

- **mean conf (+)** — Average confidence score across only the
  **positive-labeled** cells for this trace. Tells you how sure we are about
  the failures humans flagged. Computed as the mean of `combined` scores
  where `label == 1`.

- **mean conf (−)** — Same, but for the **negative-labeled** cells (modes
  humans said were absent). Often higher than the positive side in Rounds 1–2
  because "absent" is the default and takes less judgment.

- **overall** — Mean of `combined` confidence across **all 14 cells** for the
  trace (positive + negative). A single per-trace reliability number. The
  "conf score" for each trace.

- **min cell** — The lowest `combined` score among the 14 cells for that
  trace. Highlights the weakest label. If `min = 0.09`, there is at least one
  cell where we have almost no confidence (typically a forced mapping from an
  older taxonomy with one annotator agreeing).

## How `combined` is computed

Each of the 14 cells per trace gets three sub-scores, all in [0, 1]:

- **`annotator_consensus`** — Fraction of annotators who agreed on this label
  after resolving disagreements. A 3/3 unanimous label scores 1.0; a 2/3
  split scores 0.67; a post-discussion override scores lower.

- **`mapping_directness`** — How cleanly the original taxonomy label maps to
  the final 14 modes:
  - 1.0 = identity (Generalizability, already in final-14)
  - 0.9 = rename only (same meaning, new number)
  - 0.7 = near-synonym
  - 0.6 = ambiguous swap (the 3.2 ↔ 3.3 verification pair)
  - 0.3 = forced mapping (no clean analogue, best-guess)
  - 0 / dropped = modes that have no final-14 equivalent

- **`round_quality`** — Blanket reliability weight per round:
  - Round 1 = 0.3
  - Round 2 = 0.5
  - Round 3 = 0.7
  - Generalizability = 1.0

  Reflects how mature the taxonomy and annotator calibration were.

**`combined = annotator_consensus × mapping_directness × round_quality`** — a
multiplicative score, so any one weak component drags the whole cell down.
That's why Round 1 cells cap out around 0.3 no matter how unanimous the
annotators were.

### Aggregate column definitions (bottom table)

- **n** — Number of traces in the round (5 for Rounds 1–3, 4 for
  Generalizability).
- **avg chars** — Mean trace length within the round.
- **avg n+** — Mean number of positive-labeled modes per trace. Round 1 has
  the highest (4.4) because the older 18-mode taxonomy was coarser and more
  modes fired per trace.
- **mean conf** — Mean of the per-trace `overall` column. Dominated by
  `round_quality`, which is why the values cluster tightly by round
  (0.24 / 0.38 / 0.62 / 0.95).

## What jumps out

- **Confidence rises monotonically with round maturity**: 0.24 → 0.38 → 0.62
  → 0.95. This is by design (our `round_quality` weights), but the gap
  between Generalizability (0.95) and everything else is large enough that
  if you want a clean evaluation signal, **the 4 Generalizability traces
  carry most of the weight**.

- **Length has no relationship with confidence** — it's set independently.
  Traces of 2,503 chars (id 14) and 644,556 chars (id 10) both score 0.62
  because they're both Round 3.

- **Rounds 1 and 2 have per-MAS uniformity**: traces 0/1/2 and 5/6/7 have
  identical `n+` and identical confidence values because they also have
  identical labels. The lengths differ wildly (6K to 265K) but the labels
  don't — another signal these rounds were calibration, not independent
  trace-level annotation.

- **Weighted effective sample size**: if you weight by `mean conf`, the 19
  traces give you an effective n of Σ(mean_conf) ≈ **9.0** — which is
  roughly the "9 usable traces" number we had all along, now computed from
  first principles rather than hand-picked.
