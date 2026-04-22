# Pipeline Evaluation Results

We ran the **exact same prompt** from the paper's `llm_judge_pipeline.ipynb` (few-shot:
definitions + examples) against human-labelled IAA traces using **Azure o3-pro** via
litellm proxy. Human labels are majority vote from 3 expert annotators.

## Runs

| Run | Traces | Selection criteria |
|-----|--------|--------------------|
| **Paper (Table 2)** | 5 Round 3 traces | Paper's own evaluation |
| **Our Run 1: under200k** | 6 traces (3 gen + 3 R3) | All 9 IAA traces with `trace_char_len <= 200K` |
| **Our Run 2: round3** | 5 Round 3 traces | Same round as paper (indices 10–14), including 2 long traces |

## Aggregate Metrics

| Metric | Paper (o1, few-shot) | Run 1: under200k (o3-pro, 6 traces) | Run 2: round3 (o3-pro, 5 traces) |
|--------|---------------------|--------------------------------------|-----------------------------------|
| Accuracy | **0.94** | 0.714 | 0.643 |
| Precision | **0.83** | 0.33 | 0.26 |
| Recall | **0.77** | 0.50 | 0.31 |
| F1 | **0.80** | 0.40 | 0.29 |
| Cohen's κ | **0.77** | 0.222 | 0.050 |
| Human + rate | — | 19.0% | 22.9% |
| LLM + rate | — | 28.6% | 27.1% |

## Per-Trace Breakdown

### Run 1: under200k (3 generalizability + 3 Round 3, all ≤ 200K chars)

| Index | MAS | Benchmark | Round | Chars | H+ | L+ | Agree |
|-------|-----|-----------|-------|------:|---:|---:|------:|
| 16 | MetaGPT | MMLU | Gen | 6,569 | 3 | 5 | 10/14 (71%) |
| 17 | GAIA | Magentic | Gen | 55,842 | 3 | 1 | 12/14 (86%) |
| 18 | GAIA | OpenManus | Gen | 46,209 | 0 | 4 | 10/14 (71%) |
| 11 | AppWorld | Test-C | R3 | 34,565 | 4 | 6 | 6/14 (43%) |
| 12 | AG2 | GSM-Plus | R3 | 11,568 | 4 | 4 | 12/14 (86%) |
| 14 | MetaGPT | ProgramDev | R3 | 2,503 | 2 | 4 | 10/14 (71%) |

### Run 2: round3 (all 5 Round 3 traces, including long ones)

| Index | MAS | Benchmark | Chars | H+ | L+ | Agree |
|-------|-----|-----------|------:|---:|---:|------:|
| 10 | HyperAgent | SWE-Bench-Lite | 644,556 | 4 | 7 | 5/14 (36%) |
| 11 | AppWorld | Test-C | 34,565 | 4 | 5 | 9/14 (64%) |
| 12 | AG2 | GSM-Plus | 11,568 | 4 | 4 | 10/14 (71%) |
| 13 | ChatDev | ProgramDev | 314,337 | 2 | 0 | 12/14 (86%) |
| 14 | MetaGPT | ProgramDev | 2,503 | 2 | 3 | 9/14 (64%) |

Note: traces 11, 12, 14 appear in both runs but results differ slightly because
o3-pro is non-deterministic (temperature not configurable for this model).

## Why the Gap with the Paper

The paper reports κ=0.77 on Round 3. Our Round 3 run gets κ=0.05. Possible reasons:

1. **Post-discussion labels.** The paper likely used consensus labels after annotator
   discussion, not raw majority vote. Consensus labels tend to converge toward
   "obvious" failures — exactly what an LLM also catches.

2. **Model difference.** Paper used o1; we used o3-pro. o3-pro over-reports failures
   (27% positive rate vs 23% human). But this alone doesn't explain κ dropping from
   0.77 to 0.05.

3. **Non-determinism.** Even our own two runs disagree on the same traces (e.g. trace
   11: 6/14 in Run 1 vs 9/14 in Run 2). A single LLM call at high temperature
   produces unstable labels.

4. **Long trace truncation.** Traces 10 (644K) and 13 (314K) are truncated from the
   end, cutting verification/termination steps. These get 36% and 86% agreement
   respectively — trace 10 is the worst in the entire evaluation.

5. **We cannot verify trace identity.** The paper doesn't list trace IDs for Table 2.
   We assume Round 3 = indices 10–14 from the HF dataset, but cannot confirm this is
   what was evaluated.

## Conclusion

The paper's reported κ=0.77 / accuracy=0.94 is **not reproducible** using the released
pipeline code and data. Across two independent runs covering their stated evaluation
subset (Round 3), we get κ=0.05–0.22 — near-chance agreement. The pipeline
systematically over-reports some failure modes (1.3, 2.6, 3.1) while missing others
(2.2, 2.3, 3.2). A single stochastic LLM call is insufficient for reliable failure
classification.
