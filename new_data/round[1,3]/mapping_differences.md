# Taxonomy Mapping: Ours vs `load_mast.py`

The 19 IAA traces on HuggingFace use three different versions of the failure
taxonomy (one per round plus the final 14-mode version for Generalizability).
Any per-trace evaluation needs every label to be expressed in a single
taxonomy — the paper's final 14 modes.

There are two independent remap implementations in this repo:

| File | Purpose | Output |
|------|---------|--------|
| `new_data/taxonomy_mappings.py` | Our mapping, called by `extract_iaa_traces_all19.py` | `new_data/iaa_all19_traces_final14.json` |
| `new_data/load_mast.py` | Paper-authors' quick analysis script | Prints counts; does not save |

Both scripts read the same input (`MAD_human_labelled_dataset.json`), apply
the same 2-of-3 annotator majority vote, and then remap codes. They agree on
**248 / 266 (93.2%)** of label cells. This document explains the **18 cells
that disagree** and why.

Run `python new_data/compare_mappings.py` to reproduce the diff below.

---

## 1. Summary of disagreements

| Round | Disagreeing cells | Disagreeing modes |
|-------|------------------:|-------------------|
| Generalizability | 0 | — |
| Round 1 | 0 | — (modes cancel out, see §3) |
| Round 2 | 8 | `3.2`, `3.3`, `1.4` |
| Round 3 | 10 | `3.2`, `3.3` |
| **Total** | **18 / 266 = 6.8%** | |

Per-trace failure counts match for 17/19 traces. Traces 8 and 9 (Round 2
ChatDev and MetaGPT) each have one more positive label in our version (2 vs 1).

---

## 2. Disagreement pattern A: the `3.2` ↔ `3.3` swap (16 of 18 cells)

The final 14-mode taxonomy has two verification modes:

- **3.2 No or Incorrect Verification** — "Lack of verification or failure to
  perform correct verification" (verification is absent or plainly wrong).
- **3.3 Weak Verification** — "Agent performs verification but it is
  superficial" (verification happened but was cursory).

The interim-17 taxonomy (Rounds 2 & 3) has two similar-sounding modes:

- **3.2 Lack of critical verification**
- **3.3 Lack of result verification**

The early-18 taxonomy (Round 1) has:

- **4.2 Lack of result verification**
- **4.3 Lack of critical verification**

Both scripts map these to `3.2` / `3.3` — but in opposite directions.

| Source label | `taxonomy_mappings.py` (ours) | `load_mast.py` |
|--------------|-------------------------------|----------------|
| `Lack of result verification` | → `3.2` No or Incorrect Verification | → `3.3` Incorrect Verification |
| `Lack of critical verification` | → `3.3` Weak Verification | → `3.2` No or Incomplete Verification |

### Why we chose our direction (semantic)

- "Lack of **result** verification" = the agent didn't check the final result
  at all. That is verification **absence** → `3.2` (No Verification).
- "Lack of **critical** verification" = the agent did some verification but
  wasn't rigorous. That is **superficial** verification → `3.3` (Weak).

### Why `load_mast.py` chose the other direction (numerical)

- The interim-17 code `3.2` maps to final-14 code `3.2`, and `3.3` to `3.3`.
  This preserves code numbers across taxonomy versions but reverses the
  semantics: "Lack of critical verification" becomes "No or Incomplete
  Verification", and "Lack of result verification" becomes "Incorrect
  Verification".

### Consequence

Neither choice is "correct" — the paper does not document which interim
code became which final code. The difference is large enough to flip the
evaluation: every R2/R3 trace that has either verification failure will be
labeled on the opposite verification axis depending on which mapping you use.

For Round 1, the swap exists in the mapping tables too (`4.2` and `4.3`), but
it happens to cancel out in the output: both R1 traces with any verification
failure (traces 3 and 4) have **both** verification labels set positive, so
the OR of the two interpretations ends up identical.

---

## 3. Disagreement pattern B: modes we keep that `load_mast.py` drops (2 of 18 cells)

`load_mast.py` is more conservative: it drops three early/interim modes that
have no direct final-14 analogue, arguing a safe mapping is not obvious.
Our mapping preserves them by using the closest available final-14 category.

| Source label | `taxonomy_mappings.py` (ours) | `load_mast.py` |
|--------------|-------------------------------|----------------|
| `Undetected conversation ambiguities and contradictions` (R1 `1.3`, R2/R3 `2.4`) | → `2.2` Fail to Ask for Clarification | dropped |
| `Unbatched repetitive execution` (R1 `2.1`, R2/R3 `1.4`) | → `1.3` Step Repetition | dropped |
| `Backtracking interruption` (R1 `2.3`, R2/R3 `1.6`) | → `1.4` Loss of Conversation History | dropped |

Only `Backtracking interruption` surfaces in the cell-level diff because it
is the only one with positive labels that don't also collide with another
positive label already present in the trace:

- Round 2 trace 8 (ChatDev/ProgramDev): `Backtracking interruption` is
  positive, so `1.4` is `1` under our mapping and `0` under `load_mast.py`.
- Round 2 trace 9 (MetaGPT/ProgramDev): identical situation.

`Undetected ambiguities` never fires in the traces that reach this code path,
and `Unbatched repetitive execution` always co-occurs with `Step repetition`
so the OR is already 1 regardless.

### Why we chose to keep them

Dropping a positive label is lossy: a human annotator explicitly marked a
failure, and we would be hiding evidence of real (by human judgment)
problems. The closest mapping is defensible:

- "Undetected ambiguities" → `2.2` because ambiguity the agent fails to
  notice is exactly the situation where clarification is needed.
- "Unbatched repetitive execution" → `1.3` because redundantly repeating
  work is the heart of step repetition, even if the original label was
  narrower ("batchable" actions).
- "Backtracking interruption" → `1.4` because rolling back to an earlier
  state without reason resembles losing conversation history.

### Why `load_mast.py` drops them

Any approximate mapping introduces noise. If the paper's final-14 taxonomy
deliberately removed these modes as redundant or ill-defined, forcing them
into a similar-sounding bucket may misrepresent the original annotation.

---

## 4. When to use which mapping

- **`taxonomy_mappings.py` (ours)** — maximizes signal, use when evaluating
  an LLM judge. Dropping positive human labels would artificially raise
  precision but lower recall.
- **`load_mast.py`** — maximizes fidelity to the paper's taxonomy, use when
  reporting failure-mode prevalence counts for comparison with the paper's
  published numbers.

The actual scientific finding here is that both approaches are defensible
because the paper never publishes a mapping table. Anyone reproducing the
evaluation has to make these calls themselves, and differences of ±10% on
metrics can come purely from mapping choices.

---

## 5. Appendix — full disagreement list

Run `python new_data/compare_mappings.py` to regenerate. Cells shown as
`ours/theirs` where they disagree.

```
 tid round             mas          ...  1.4  ...  3.2  3.3   diff
   5 Round 2           AppWorld      ...   0   ... [1/0][0/1]  3.2,3.3
   6 Round 2           HyperAgent    ...   0   ... [1/0][0/1]  3.2,3.3
   7 Round 2           AG2           ...   0   ... [1/0][0/1]  3.2,3.3
   8 Round 2           ChatDev       ...[1/0] ...    0    0    1.4
   9 Round 2           MetaGPT       ...[1/0] ...    0    0    1.4
  10 Round 3           HyperAgent    ...   0   ... [1/0][0/1]  3.2,3.3
  11 Round 3           AppWorld      ...   0   ... [1/0][0/1]  3.2,3.3
  12 Round 3           AG2           ...   0   ... [1/0][0/1]  3.2,3.3
  13 Round 3           ChatDev       ...   0   ... [0/1][1/0]  3.2,3.3
  14 Round 3           MetaGPT       ...   0   ... [0/1][1/0]  3.2,3.3
```

Eight modes (`3.2` and `3.3` across eight traces) plus two `1.4` cells
= 18 total disagreements.
