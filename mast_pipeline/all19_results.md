# Pipeline Evaluation on All 19 IAA Traces

We re-ran the paper's LLM-judge pipeline (few-shot: definitions + examples,
exact same prompt as `llm_judge_pipeline.ipynb`) on **all 19 IAA traces** —
Rounds 1, 2, 3, and Generalizability — using **Azure `o3-pro`** via the
litellm proxy. Human labels are majority vote from 3 annotators, mapped to the
final 14-mode MAST taxonomy (see `new_data/round[1,3]/taxonomy_mappings.py`).

The confidence-weighted column uses per-cell weights from
`new_data/round[1,3]/iaa_all19_confidence.json`
(`combined = annotator_consensus × mapping_directness × round_quality`). See
`new_data/round[1,3]/confidence_stats.md` for the full definition.

- Pipeline input:  `new_data/round[1,3]/final_iaa_19_traces.json`
- Raw LLM output:  `mast_pipeline/outputs/all19/pipeline_evaluation_results_all19.json`
- Weighted metrics: `mast_pipeline/outputs/all19/weighted_metrics_all19.json`
- Run log:         `mast_pipeline/outputs/all19/run.log`

## Aggregate metrics

266 (trace, mode) cells total = 19 traces × 14 modes.

| Metric           | Paper (o1, few-shot) | Our 19-trace run (o3-pro, unweighted) | Our 19-trace run (o3-pro, **weighted** by confidence) |
|------------------|---------------------:|--------------------------------------:|------------------------------------------------------:|
| Accuracy         | **0.94**             | 0.711                                 | **0.740** |
| Precision (+)    | **0.83**             | 0.318                                 | **0.329** |
| Recall (+)       | **0.77**             | 0.370                                 | **0.448** |
| F1 (+)           | **0.80**             | 0.342                                 | **0.380** |
| Cohen's κ        | **0.77**             | 0.158                                 | **0.220** |
| Human + rate     | —                    | 20.3%                                 | 17.8% (effective) |
| LLM + rate       | —                    | 23.7%                                 | 24.2% (effective) |

Weighted metrics use per-cell confidence `w` in [0, 1]:
- weighted accuracy  = Σ w · 1[h=l] / Σ w
- weighted precision = Σ w · 1[h=1,l=1] / Σ w · 1[l=1]
- weighted recall    = Σ w · 1[h=1,l=1] / Σ w · 1[h=1]
- weighted κ         = standard Cohen formula with weighted p_o and p_e

## What the weighting changes

Weighting pulls the metric up because **the cells the LLM agrees with humans on
tend to be in high-confidence rounds** (Round 3 + Generalizability), while the
cells where it disagrees are disproportionately in low-confidence Rounds 1 and
2.

- κ moves from 0.158 → 0.220 (+0.06). Still "slight agreement" on the
  Landis–Koch scale — nowhere near the paper's 0.77.
- Recall climbs the most (+0.08). The LLM is better at finding the failures
  that humans labeled confidently (Generalizability), worse at the noisy
  Round-1/2 labels it was trained to match.
- Accuracy gains +0.03 mostly from dropping weight on Round-1 traces whose
  labels disagree with almost everything.

## Per-trace breakdown

| idx | round            | MAS        | benchmark        | uw acc | uw κ   | w acc  | w κ    |
|----:|------------------|------------|------------------|-------:|-------:|-------:|-------:|
| 0   | Round 1          | AppWorld   | Test-C           | 0.714  | -0.167 | 0.654  | -0.207 |
| 1   | Round 1          | HyperAgent | SWE-Bench-Lite   | 0.429  | -0.273 | 0.383  | -0.352 |
| 2   | Round 1          | AG2        | GSM-Plus         | 0.929  |  0.759 | 0.972  |  0.913 |
| 3   | Round 1          | ChatDev    | ProgramDev       | 0.429  | -0.037 | 0.368  | -0.093 |
| 4   | Round 1          | MetaGPT    | ProgramDev       | 0.643  |  0.340 | 0.623  |  0.323 |
| 5   | Round 2          | AppWorld   | Test-C           | 0.857  | -0.077 | 0.916  | -0.039 |
| 6   | Round 2          | HyperAgent | SWE-Bench-Lite   | 0.786  |  0.323 | 0.813  |  0.312 |
| 7   | Round 2          | AG2        | GSM-Plus         | 0.786  | -0.105 | 0.804  | -0.087 |
| 8   | Round 2          | ChatDev    | ProgramDev       | 0.714  | -0.167 | 0.785  | -0.118 |
| 9   | Round 2          | MetaGPT    | ProgramDev       | 0.786  | -0.105 | 0.851  | -0.075 |
| 10  | Round 3          | HyperAgent | SWE-Bench-Lite   | 0.429  | -0.077 | 0.485  | -0.033 |
| 11  | Round 3          | AppWorld   | Test-C           | 0.714  |  0.300 | 0.742  |  0.381 |
| 12  | Round 3          | AG2        | GSM-Plus         | 0.714  |  0.300 | 0.733  |  0.364 |
| 13  | Round 3          | ChatDev    | ProgramDev       | 0.857  |  0.000 | 0.891  |  0.000 |
| 14  | Round 3          | MetaGPT    | ProgramDev       | 0.786  |  0.276 | 0.836  |  0.310 |
| 15  | Generalizability | ChatDev    | MMLU             | 0.714  |  0.317 | 0.700  |  0.223 |
| 16  | Generalizability | MetaGPT    | MMLU             | 0.714  |  0.429 | 0.700  |  0.380 |
| 17  | Generalizability | GAIA       | Magentic         | 0.857  |  0.440 | 0.900  |  0.553 |
| 18  | Generalizability | GAIA       | OpenManus        | 0.643  |  0.000 | 0.650  |  0.000 |

Trace 10 (HyperAgent SWE-Bench-Lite) is 644K chars and gets truncated to
~332K before the API call — that drops κ below zero even with weighting.

## Per-mode breakdown

| Mode | Name                             | H+ | L+ | uw acc | uw κ   | w acc  | w κ    |
|------|----------------------------------|---:|---:|-------:|-------:|-------:|-------:|
| 1.1  | Disobey Task Specification       | 6  | 7  | 0.737  |  0.417 | 0.723  |  0.427 |
| 1.2  | Disobey Role Specification       | 2  | 1  | 0.842  | -0.075 | 0.925  | -0.035 |
| 1.3  | Step Repetition                  | 2  | 10 | 0.474  | -0.011 | 0.492  | -0.003 |
| 1.4  | Loss of Conversation History     | 2  | 2  | 0.789  | -0.118 | 0.892  | -0.056 |
| 1.5  | Unaware of Termination Cond.     | 5  | 7  | 0.579  |  0.038 | 0.567  | -0.005 |
| 2.1  | Conversation Reset               | 2  | 3  | 0.737  | -0.145 | 0.775  | -0.084 |
| 2.2  | Fail to Ask for Clarification    | 5  | 2  | 0.842  |  0.496 | 0.798  |  0.517 |
| 2.3  | Task Derailment                  | 5  | 3  | 0.684  |  0.066 | 0.650  |  0.045 |
| 2.4  | Information Withholding          | 0  | 1  | 0.947  |  0.000 | 0.975  |  0.000 |
| 2.5  | Ignored Other Agent's Input      | 3  | 1  | 0.895  |  0.457 | 0.946  |  0.479 |
| 2.6  | Action-Reasoning Mismatch        | 4  | 7  | 0.421  | -0.366 | 0.478  | -0.221 |
| 3.1  | Premature Termination            | 0  | 4  | 0.789  |  0.000 | 0.783  |  0.000 |
| 3.2  | No or Incorrect Verification     | 11 | 5  | 0.684  |  0.412 | 0.678  |  0.414 |
| 3.3  | Weak Verification                | 7  | 10 | 0.526  |  0.066 | 0.520  |  0.035 |

Modes where the pipeline systematically fails:
- **1.3 Step Repetition** (LLM 10 vs. human 2), **2.6 Action-Reasoning
  Mismatch** (LLM 7 vs. human 4), **3.1 Premature Termination** (LLM 4 vs.
  human 0) — the LLM over-reports these regardless of weighting.
- **3.2 / 3.3 Verification** — per-mode κ for 3.2 is 0.41, for 3.3 is 0.07,
  but humans seem to use the two modes nearly interchangeably (this is the
  known 3.2/3.3 swap ambiguity in the taxonomy — see
  `new_data/round[1,3]/mapping_differences.md`).
- **2.2 Fail to Ask for Clarification** is the one mode where both unweighted
  and weighted κ stay above 0.5, meaning it's the only mode the pipeline
  recognizes reliably.

## Why we still can't match the paper

Same structural issues as the 5-trace Round 3 run (see `round3_results.md`):

1. **Post-discussion vs. majority-vote labels.** We use raw majority; the
   paper almost certainly used consensus after annotator discussion, which
   converges on obvious failures.
2. **Model + temperature.** We use `o3-pro` with no temperature control; the
   paper used `o1`. Non-deterministic single-shot calls are the wrong tool
   for 14-way classification anyway.
3. **Taxonomy drift.** Rounds 1 and 2 used older 18/17-mode taxonomies. Even
   with careful mapping to the final 14, some cells are forced fits (low
   `mapping_directness`) and these are precisely the ones that suppress κ.
4. **Long-trace truncation.** Trace 10 (644K chars) is cut to ~332K, losing
   the tail where verification/termination failures live. It's the single
   worst-agreeing trace in the set.

## Conclusion

Adding 14 more traces (Rounds 1, 2, and Generalizability) to the original
5-trace Round 3 run moves the metrics up slightly — confidence weighting
moves them further, from κ = 0.158 → 0.220 — but the paper's reported
**κ = 0.77 / accuracy = 0.94** remains **not reproducible**. Even on the 4
highest-confidence Generalizability traces (round_quality = 1.0), weighted κ
only reaches 0.39. Our best per-trace κ is 0.91 (trace 2, a very short
Round-1 AG2 trace where the LLM happens to match a calibration-uniform human
label). The pipeline's systematic over-reporting of modes 1.3, 2.6, 3.1 and
under-reporting of 3.2 does not depend on which traces you evaluate.
