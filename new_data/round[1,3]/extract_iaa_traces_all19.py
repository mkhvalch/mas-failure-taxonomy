"""
Extract all 19 IAA traces (Rounds 1, 2, 3, + Generalizability) from the
HuggingFace human-labelled dataset, map every round's labels to the final
14-mode MAST taxonomy, and save as JSON for pipeline evaluation.

Mapping notes:
- Generalizability (final-mast-14): native, no mapping needed.
- Round 3 (interim-mast-17): near-final; map 1.x/2.x/3.x codes back to final
  categories (1.x: execution, 2.x: coordination, 3.x: verification).
- Round 2 (interim-mast-17): same 17-mode scheme as Round 3.
- Round 1 (early-mast-18): earlier scheme with different names and numbering;
  mapped best-effort. Modes with no clean final-14 analogue are dropped
  (noted in UNMAPPED comments).

Usage:
    python extract_iaa_traces_all19.py
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

from huggingface_hub import hf_hub_download

REPO_ID = "mcemri/MAD"

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


# Round 1 (early-mast-18) → final-14 mapping.
# Source: inspected label names in HF data.
#   1.1 Poor task constraint compliance           → 1.1 Disobey Task Specification
#   1.2 Inconsistency between reasoning and action → 2.6 Action-Reasoning Mismatch
#   1.3 Undetected conversation ambiguities        → 2.2 Fail to Ask for Clarification (closest)
#   1.4 Fail to elicit clarification               → 2.2 Fail to Ask for Clarification
#   1.5 Unaware of stopping conditions             → 1.5 Unaware of Termination Conditions
#   2.1 Unbatched repetitive execution             → 1.3 Step Repetition
#   2.2 Step repetition                            → 1.3 Step Repetition
#   2.3 Backtracking interruption                  → 1.4 Loss of Conversation History
#   2.4 Conversation reset                         → 2.1 Conversation Reset
#   2.5 Derailment from task                       → 2.3 Task Derailment
#   2.6 Disobey role specification                 → 1.2 Disobey Role Specification
#   3.1 Disagreement induced inaction              → dropped (no clean match in final-14)
#   3.2 Withholding relevant information           → 2.4 Information Withholding
#   3.3 Ignoring suggestions from agents           → 2.5 Ignored Other Agent's Input
#   3.4 Waiting for known information              → dropped
#   4.1 Ill specified termination → premature      → 3.1 Premature Termination
#   4.2 Lack of result verification                → 3.2 No or Incorrect Verification
#   4.3 Lack of critical verification              → 3.3 Weak Verification
ROUND1_CODE_MAP = {
    "1.1": "1.1",
    "1.2": "2.6",
    "1.3": "2.2",
    "1.4": "2.2",
    "1.5": "1.5",
    "2.1": "1.3",
    "2.2": "1.3",
    "2.3": "1.4",
    "2.4": "2.1",
    "2.5": "2.3",
    "2.6": "1.2",
    "3.1": None,
    "3.2": "2.4",
    "3.3": "2.5",
    "3.4": None,
    "4.1": "3.1",
    "4.2": "3.2",
    "4.3": "3.3",
}


# Round 2 / Round 3 (interim-mast-17) → final-14 mapping.
# Source: inspected label names in HF data.
#   1.1 Poor task constraint compliance           → 1.1 Disobey Task Specification
#   1.2 Inconsistency between reasoning and action → 2.6 Action-Reasoning Mismatch
#   1.3 Unaware of stopping conditions             → 1.5 Unaware of Termination Conditions
#   1.4 Unbatched repetitive execution             → 1.3 Step Repetition
#   1.5 Step repetition                            → 1.3 Step Repetition
#   1.6 Backtracking interruption                  → 1.4 Loss of Conversation History
#   1.7 Disobey role specification                 → 1.2 Disobey Role Specification
#   2.1 Conversation reset                         → 2.1 Conversation Reset
#   2.2 Fail to elicit clarification               → 2.2 Fail to Ask for Clarification
#   2.3 Derailment from task                       → 2.3 Task Derailment
#   2.4 Undetected conversation ambiguities        → 2.2 Fail to Ask for Clarification
#   2.5 Disagreement induced inaction              → dropped
#   2.6 Withholding relevant information           → 2.4 Information Withholding
#   2.7 Ignoring suggestions from agents           → 2.5 Ignored Other Agent's Input
#   3.1 Ill specified termination → premature      → 3.1 Premature Termination
#   3.2 Lack of critical verification              → 3.3 Weak Verification
#   3.3 Lack of result verification                → 3.2 No or Incorrect Verification
INTERIM17_CODE_MAP = {
    "1.1": "1.1",
    "1.2": "2.6",
    "1.3": "1.5",
    "1.4": "1.3",
    "1.5": "1.3",
    "1.6": "1.4",
    "1.7": "1.2",
    "2.1": "2.1",
    "2.2": "2.2",
    "2.3": "2.3",
    "2.4": "2.2",
    "2.5": None,
    "2.6": "2.4",
    "2.7": "2.5",
    "3.1": "3.1",
    "3.2": "3.3",
    "3.3": "3.2",
}


def extract_code(mode_name: str) -> str | None:
    m = re.match(r"(\d+\.\d+)", mode_name.strip())
    return m.group(1) if m else None


def majority_vote(item: dict) -> bool:
    votes = []
    for k in ["annotator_1", "annotator_2", "annotator_3"]:
        v = item.get(k, False)
        if isinstance(v, str):
            votes.append(v.strip().upper() == "TRUE")
        else:
            votes.append(bool(v))
    return sum(votes) >= 2


def process_trace(rec: dict, code_map: dict | None = None) -> dict:
    annotations = rec.get("annotations", [])
    if isinstance(annotations, str):
        annotations = json.loads(annotations)

    labels = {}
    for item in annotations:
        mode_name = item.get("failure mode", "")
        code = extract_code(mode_name)
        if code is None:
            continue

        if code_map is not None:
            final_code = code_map.get(code)
            if final_code is None:
                continue
        else:
            final_code = code

        if final_code not in FINAL_MODE_NAMES:
            continue

        vote = int(majority_vote(item))

        # If multiple source codes collapse into one final code, take OR
        if final_code in labels:
            labels[final_code] = max(labels[final_code], vote)
        else:
            labels[final_code] = vote

    for mode in FINAL_MODES:
        labels.setdefault(mode, 0)

    return {
        "index": rec.get("trace_id"),
        "round": rec.get("round"),
        "mas_name": rec.get("mas_name"),
        "benchmark_name": rec.get("benchmark_name"),
        "trace_text": rec.get("trace", ""),
        "trace_char_len": len(str(rec.get("trace", ""))),
        "human_labels": labels,
        "human_label_summary": {
            "total_failures": sum(labels.values()),
            "failures": [f"{code} {FINAL_MODE_NAMES[code]}"
                         for code in FINAL_MODES if labels.get(code)],
        },
    }


def main():
    print("Downloading human-labelled dataset...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="MAD_human_labelled_dataset.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    round_to_map = {
        "Round 1": ROUND1_CODE_MAP,
        "Round 2": INTERIM17_CODE_MAP,
        "Round 3": INTERIM17_CODE_MAP,
    }

    results = []
    for rec in data:
        rnd = str(rec.get("round", ""))
        if "generl" in rnd.lower() or "new" in rnd.lower():
            code_map = None
        else:
            code_map = round_to_map.get(rnd)
            if code_map is None:
                print(f"  WARN: unknown round '{rnd}', skipping trace_id={rec.get('trace_id')}")
                continue
        results.append(process_trace(rec, code_map=code_map))

    results.sort(key=lambda t: (t["round"], str(t["index"])))

    out_path = os.path.join(os.path.dirname(__file__), "iaa_all19_traces_final14.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nExtracted {len(results)} traces → {out_path}")
    print()
    for t in results:
        fails = t["human_label_summary"]["failures"]
        print(f"  [{t['index']}] round={t['round']:<18s} "
              f"mas={t['mas_name']:<12s} bench={t['benchmark_name']:<18s} "
              f"chars={t['trace_char_len']:>7,d}  "
              f"failures={t['human_label_summary']['total_failures']:2d}/14")
        for fm in fails:
            print(f"       [x] {fm}")
        print()


if __name__ == "__main__":
    main()
