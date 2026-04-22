"""
Taxonomy mappings used to produce iaa_all19_traces_final14.json.

Every IAA round used a different version of the failure taxonomy. To produce a
single dataset labeled in the final 14-mode MAST taxonomy, each round's labels
must be remapped.

This module contains:
  1. The three mapping tables (Round 1, Rounds 2/3, Generalizability).
  2. A `map_label(raw_code, round_name)` helper.
  3. A standalone `main()` that downloads the HF dataset and applies our
     mappings, producing `iaa_all19_traces_final14.json` (identical to the
     extract_iaa_traces_all19.py output).

See mapping_differences.md for a comparison against load_mast.py's choices.
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Final 14-mode MAST taxonomy
# --------------------------------------------------------------------------- #

FINAL_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

FINAL_MODE_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No or Incorrect Verification",
    "3.3": "Weak Verification",
}


# --------------------------------------------------------------------------- #
# Round 1 (early-mast-18) -> final-14
#
# HF label names that each code corresponds to (for reference):
#   1.1 Poor task constraint compliance
#   1.2 Inconsistency between reasoning and action
#   1.3 Undetected conversation ambiguities and contradictions
#   1.4 Fail to elicit clarification
#   1.5 Unaware of stopping conditions
#   2.1 Unbatched repetitive execution
#   2.2 Step repetition
#   2.3 Backtracking interruption
#   2.4 Conversation reset
#   2.5 Derailment from task
#   2.6 Disobey role specification
#   3.1 Disagreement induced inaction
#   3.2 Withholding relevant information
#   3.3 Ignoring suggestions from agents
#   3.4 Waiting for known information
#   4.1 Ill specified termination condition leading to premature termination
#   4.2 Lack of result verification
#   4.3 Lack of critical verification
# --------------------------------------------------------------------------- #

ROUND1_CODE_MAP: dict[str, str | None] = {
    "1.1": "1.1",  # Poor task constraint compliance    -> Disobey Task Specification
    "1.2": "2.6",  # Reasoning/action inconsistency     -> Action-Reasoning Mismatch
    "1.3": "2.2",  # Undetected ambiguities             -> Fail to Ask for Clarification
    "1.4": "2.2",  # Fail to elicit clarification       -> Fail to Ask for Clarification
    "1.5": "1.5",  # Unaware of stopping conditions     -> Unaware of Termination Conditions
    "2.1": "1.3",  # Unbatched repetitive execution     -> Step Repetition
    "2.2": "1.3",  # Step repetition                    -> Step Repetition
    "2.3": "1.4",  # Backtracking interruption          -> Loss of Conversation History
    "2.4": "2.1",  # Conversation reset                 -> Conversation Reset
    "2.5": "2.3",  # Derailment from task               -> Task Derailment
    "2.6": "1.2",  # Disobey role specification         -> Disobey Role Specification
    "3.1": None,   # Disagreement induced inaction      -> dropped (no final-14 analogue)
    "3.2": "2.4",  # Withholding relevant information   -> Information Withholding
    "3.3": "2.5",  # Ignoring suggestions                -> Ignored Other Agent's Input
    "3.4": None,   # Waiting for known information      -> dropped
    "4.1": "3.1",  # Ill-specified termination          -> Premature Termination
    "4.2": "3.2",  # Lack of result verification        -> No or Incorrect Verification
    "4.3": "3.3",  # Lack of critical verification      -> Weak Verification
}


# --------------------------------------------------------------------------- #
# Rounds 2 & 3 (interim-mast-17) -> final-14
#
# HF label names:
#   1.1 Poor task constraint compliance
#   1.2 Inconsistency between reasoning and action
#   1.3 Unaware of stopping conditions
#   1.4 Unbatched repetitive execution
#   1.5 Step repetition
#   1.6 Backtracking interruption
#   1.7 Disobey role specification
#   2.1 Conversation reset
#   2.2 Fail to elicit clarification
#   2.3 Derailment from task
#   2.4 Undetected conversation ambiguities and contradictions
#   2.5 Disagreement induced inaction
#   2.6 Withholding relevant information
#   2.7 Ignoring suggestions from agents
#   3.1 Ill specified termination condition leading to premature termination
#   3.2 Lack of critical verification
#   3.3 Lack of result verification
# --------------------------------------------------------------------------- #

INTERIM17_CODE_MAP: dict[str, str | None] = {
    "1.1": "1.1",  # Poor task constraint compliance    -> Disobey Task Specification
    "1.2": "2.6",  # Reasoning/action inconsistency     -> Action-Reasoning Mismatch
    "1.3": "1.5",  # Unaware of stopping conditions     -> Unaware of Termination Conditions
    "1.4": "1.3",  # Unbatched repetitive execution     -> Step Repetition
    "1.5": "1.3",  # Step repetition                    -> Step Repetition
    "1.6": "1.4",  # Backtracking interruption          -> Loss of Conversation History
    "1.7": "1.2",  # Disobey role specification         -> Disobey Role Specification
    "2.1": "2.1",  # Conversation reset                 -> Conversation Reset
    "2.2": "2.2",  # Fail to elicit clarification       -> Fail to Ask for Clarification
    "2.3": "2.3",  # Derailment from task               -> Task Derailment
    "2.4": "2.2",  # Undetected ambiguities             -> Fail to Ask for Clarification
    "2.5": None,   # Disagreement induced inaction      -> dropped
    "2.6": "2.4",  # Withholding relevant information   -> Information Withholding
    "2.7": "2.5",  # Ignoring suggestions                -> Ignored Other Agent's Input
    "3.1": "3.1",  # Ill-specified termination          -> Premature Termination
    "3.2": "3.3",  # Lack of CRITICAL verification      -> Weak Verification
    "3.3": "3.2",  # Lack of RESULT verification        -> No or Incorrect Verification
}


# Generalizability round already uses final-14 codes natively.
GENERALIZABILITY_CODE_MAP = None  # sentinel: identity mapping


ROUND_TO_MAP = {
    "Round 1": ROUND1_CODE_MAP,
    "Round 2": INTERIM17_CODE_MAP,
    "Round 3": INTERIM17_CODE_MAP,
    "Generlazability": GENERALIZABILITY_CODE_MAP,
}


def map_label(raw_code: str, round_name: str) -> str | None:
    """Return the final-14 code for a raw code from the given round.

    Returns None if the source mode has no final-14 analogue (dropped).
    Raises KeyError if the round is unknown or the code is not in the round's table.
    """
    code_map = ROUND_TO_MAP[round_name]
    if code_map is None:
        if raw_code in FINAL_MODE_NAMES:
            return raw_code
        raise KeyError(f"{raw_code!r} is not a final-14 code")
    return code_map[raw_code]


# --------------------------------------------------------------------------- #
# Application: apply mappings to the HF dataset and save iaa_all19_traces_final14.json
# --------------------------------------------------------------------------- #

_CODE_RE = re.compile(r"(\d+\.\d+)")


def _extract_code(failure_mode: str) -> str | None:
    m = _CODE_RE.match(failure_mode.strip())
    return m.group(1) if m else None


def _majority_vote(annotation: dict) -> int:
    votes = []
    for k in ("annotator_1", "annotator_2", "annotator_3"):
        v = annotation.get(k, False)
        if isinstance(v, str):
            votes.append(v.strip().upper() == "TRUE")
        else:
            votes.append(bool(v))
    return int(sum(votes) >= 2)


def process_trace(record: dict) -> dict:
    """Apply majority vote + our round-specific mapping to a single HF record."""
    round_name = str(record.get("round", ""))
    code_map = ROUND_TO_MAP.get(round_name)

    labels = {m: 0 for m in FINAL_MODES}

    annotations = record.get("annotations", [])
    if isinstance(annotations, str):
        annotations = json.loads(annotations)

    for ann in annotations:
        raw_code = _extract_code(ann.get("failure mode", ""))
        if raw_code is None:
            continue

        if code_map is None:
            final_code = raw_code if raw_code in FINAL_MODE_NAMES else None
        else:
            final_code = code_map.get(raw_code)

        if final_code is None or final_code not in FINAL_MODE_NAMES:
            continue

        # If multiple source codes collapse onto one final code, OR the votes.
        labels[final_code] = max(labels[final_code], _majority_vote(ann))

    return {
        "index": record.get("trace_id"),
        "round": round_name,
        "mas_name": record.get("mas_name"),
        "benchmark_name": record.get("benchmark_name"),
        "trace_text": record.get("trace", ""),
        "trace_char_len": len(str(record.get("trace", ""))),
        "human_labels": labels,
        "human_label_summary": {
            "total_failures": sum(labels.values()),
            "failures": [
                f"{c} {FINAL_MODE_NAMES[c]}"
                for c in FINAL_MODES if labels[c]
            ],
        },
    }


def main():
    from huggingface_hub import hf_hub_download

    print("Downloading MAD_human_labelled_dataset.json from HuggingFace...")
    path = hf_hub_download(
        repo_id="mcemri/MAD",
        filename="MAD_human_labelled_dataset.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    results = []
    for rec in data:
        rnd = str(rec.get("round", ""))
        if rnd not in ROUND_TO_MAP:
            print(f"  WARN: unknown round {rnd!r}, skipping trace_id={rec.get('trace_id')}")
            continue
        results.append(process_trace(rec))

    results.sort(key=lambda t: (t["round"], str(t["index"])))

    out = os.path.join(os.path.dirname(__file__), "iaa_all19_traces_final14.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    total_pos = sum(sum(t["human_labels"].values()) for t in results)
    print(f"\nWrote {len(results)} traces -> {out}")
    print(f"Total positive labels: {total_pos} / {len(results) * 14}")


if __name__ == "__main__":
    main()
