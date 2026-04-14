# Pipeline Evaluation Findings

## Setup

We ran the **exact same prompt** from `llm_judge_pipeline.ipynb` against 6 IAA-labelled
traces using **Azure o3-pro** (via litellm proxy). Human labels are majority vote from
3 expert annotators in the HuggingFace `MAD_human_labelled_dataset.json`.

- 4 traces from the generalizability round (final 14-mode MAST taxonomy)
- 5 traces from Round 3 (near-final taxonomy, mapped to final 14 codes)
- 3 traces skipped (>200K chars: ChatDev/MMLU 308K, HyperAgent 644K, ChatDev/ProgramDev 314K)

The prompt, definitions, and examples are identical to the notebook. Only change:
`temperature` parameter removed (o3-pro does not support it).

## Per-Trace Results

| Trace | MAS | Benchmark | Round | Chars | Human yes | LLM yes | Agree | Acc |
|-------|-----|-----------|-------|-------|-----------|---------|-------|-----|
| 16 | MetaGPT | MMLU | Generalizability | 6,569 | 3 | 5 | 10/14 | 71% |
| 17 | GAIA | Magentic | Generalizability | 55,842 | 3 | 1 | 12/14 | 86% |
| 18 | GAIA | OpenManus | Generalizability | 46,209 | 0 | 4 | 10/14 | 71% |
| 11 | AppWorld | Test-C | Round 3 | 34,565 | 4 | 6 | 6/14 | 43% |
| 12 | AG2 | GSM-Plus | Round 3 | 11,568 | 4 | 4 | 12/14 | 86% |
| 14 | MetaGPT | ProgramDev | Round 3 | 2,503 | 2 | 4 | 10/14 | 71% |

Best agreement: **86%** on traces 17 (GAIA/Magentic) and 12 (AG2/GSM-Plus).
Worst agreement: **43%** on trace 11 (AppWorld/Test-C) — 8 disagreements out of 14.

## Aggregate Metrics (6 traces, 84 label pairs)

| Metric | Value |
|--------|-------|
| Raw agreement | 60/84 (71.4%) |
| **Cohen's Kappa** | **0.222** ("fair") |
| Human positive rate | 16/84 (19.0%) |
| LLM positive rate | 24/84 (28.6%) |

### Confusion Matrix

|                  | LLM = no | LLM = yes |
|------------------|----------|-----------|
| **Human = no**   | 52       | 16        |
| **Human = yes**  | 8        | 8         |

### Classification Report (positive = failure present)

| | Precision | Recall | F1 |
|--|-----------|--------|-----|
| No failure | 0.87 | 0.76 | 0.81 |
| **Failure** | **0.33** | **0.50** | **0.40** |

The LLM catches half the failures humans find (recall = 50%) but produces many
false positives (precision = 33%).

## Per-Mode Breakdown

| Mode | Name | Human yes | LLM yes | Agree | N | Accuracy |
|------|------|-----------|---------|-------|---|----------|
| 1.1 | Disobey Task Specification | 4 | 4 | 4 | 6 | 66.7% |
| 1.2 | Disobey Role Specification | 0 | 0 | 6 | 6 | 100.0% |
| **1.3** | **Step Repetition** | **0** | **3** | **3** | **6** | **50.0%** |
| 1.4 | Loss of Conversation History | 0 | 0 | 6 | 6 | 100.0% |
| **1.5** | **Unaware of Termination Conditions** | **0** | **3** | **3** | **6** | **50.0%** |
| 2.1 | Conversation Reset | 0 | 0 | 6 | 6 | 100.0% |
| 2.2 | Fail to Ask for Clarification | 3 | 1 | 4 | 6 | 66.7% |
| 2.3 | Task Derailment | 2 | 1 | 3 | 6 | 50.0% |
| 2.4 | Information Withholding | 0 | 0 | 6 | 6 | 100.0% |
| 2.5 | Ignored Other Agent's Input | 0 | 0 | 6 | 6 | 100.0% |
| **2.6** | **Action-Reasoning Mismatch** | **0** | **3** | **3** | **6** | **50.0%** |
| **3.1** | **Premature Termination** | **0** | **3** | **3** | **6** | **50.0%** |
| 3.2 | No or Incorrect Verification | 3 | 2 | 3 | 6 | 50.0% |
| 3.3 | Weak Verification | 4 | 4 | 4 | 6 | 66.7% |

### Problematic Modes

Four modes have 50% accuracy driven entirely by **LLM false positives** on modes
where humans found zero failures:

- **1.3 Step Repetition**: LLM flagged 3/6, humans 0/6
- **1.5 Unaware of Termination Conditions**: LLM flagged 3/6, humans 0/6
- **2.6 Action-Reasoning Mismatch**: LLM flagged 3/6, humans 0/6
- **3.1 Premature Termination**: LLM flagged 3/6, humans 0/6

These modes have broad, subjective definitions. The LLM interprets them more
aggressively than human experts.

### Under-detected by LLM

- **2.2 Fail to Ask for Clarification**: Humans flagged 3/6, LLM only 1/6
- **2.3 Task Derailment**: Humans flagged 2/6, LLM only 1/6

These are inter-agent communication failures that may require understanding the
full conversation flow — harder for a single-pass analysis.

## Specific Disagreements

### Trace 16 (MetaGPT/MMLU, Generalizability)
- **Human**: 1.1, 3.2, 3.3 (task spec violated, verification issues)
- **LLM**: 1.1, 1.5, 2.3, 3.1, 3.3 (agrees on 1.1/3.3, adds termination/derailment/premature)
- LLM missed 3.2 (No Verification) but caught 3.3 (Weak Verification)

### Trace 17 (GAIA/Magentic, Generalizability)
- **Human**: 1.1, 3.2, 3.3
- **LLM**: 3.3 only
- LLM missed 2 of 3 human-identified failures — very conservative here

### Trace 18 (GAIA/OpenManus, Generalizability)
- **Human**: no failures (0/14)
- **LLM**: 1.3, 1.5, 2.6, 3.3 (4 false positives)
- Worst false-positive case: humans say this trace is clean, LLM finds 4 problems

### Trace 11 (AppWorld/Test-C, Round 3) — worst agreement
- **Human**: 1.1, 2.2, 2.3, 3.3
- **LLM**: 1.1, 1.3, 1.5, 2.6, 3.1, 3.2
- Only agreed on 1.1. LLM missed all 3 inter-agent/verification human labels.
  LLM added 5 false positives in specification and termination categories.

### Trace 12 (AG2/GSM-Plus, Round 3) — best agreement (tied)
- **Human**: 1.1, 2.2, 2.3, 3.3
- **LLM**: 1.1, 2.2, 2.6, 3.3
- Near-perfect. Only disagreement: LLM saw Action-Reasoning Mismatch instead of
  Task Derailment — arguably a reasonable alternative interpretation.

## Comparison with Paper's Reported Numbers

| Metric | Paper (Table 2, o1 few-shot) | Our run (o3-pro) | Our stored-label comparison |
|--------|------------------------------|-------------------|-----------------------------|
| Accuracy | 0.94 | 0.714 | 0.709 |
| Recall | 0.77 | 0.50 | 0.15 |
| Precision | 0.83 | 0.33 | 0.32 |
| F1 | 0.80 | 0.40 | 0.21 |
| Cohen's κ | 0.77 | 0.222 | 0.054 |

### Why the gap?

1. **Paper used Round 3 only (5 traces)** where taxonomy was nearly finalized and
   the specific traces were carefully selected. We ran on a broader set including
   generalizability traces from different MAS types.

2. **Paper may have used post-discussion consensus labels**, not raw majority vote.
   After IAA discussions, annotators resolved disagreements — the resulting labels
   may be more lenient/aligned with what an LLM would produce.

3. **Model difference**: Paper used `o1`, we used `o3-pro`. The o3-pro model appears
   more aggressive at finding failures (over-reports rather than under-reports).

4. **Round 3 had 17 modes, not 14.** Our mapping from Round 3's 17-mode scheme to
   the final 14-mode scheme introduces noise. The paper's evaluation likely used
   the Round 3 taxonomy directly without remapping.

5. **Small sample sizes amplify disagreements.** With only 6 traces × 14 modes = 84
   label pairs, each disagreement moves kappa substantially.

## Key Takeaways

1. **The pipeline works but is noisy.** Best-case individual trace agreement is 86%,
   worst-case is 43%. The variance across traces is high.

2. **o3-pro over-reports failures** (28.6% positive rate vs 19% human). The original
   o1 labels in the HF dataset under-report (12% vs 24% human). Neither model
   calibrates well to human rates.

3. **Certain failure modes are unreliable.** Step Repetition (1.3), Termination
   Conditions (1.5), Action-Reasoning Mismatch (2.6), and Premature Termination (3.1)
   have high false-positive rates. These modes have broad definitions that the LLM
   interprets more aggressively than human experts.

4. **The paper's κ=0.77 is not reproducible** on a broader trace set with the released
   code and data. The actual agreement on diverse traces is substantially lower.

5. **Inter-agent communication failures are hardest.** Modes 2.2 (Clarification) and
   2.3 (Derailment) are under-detected by the LLM, possibly because they require
   reasoning about the full conversation arc rather than spotting local patterns.
