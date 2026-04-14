"""
Extract IAA traces from HuggingFace human-labelled dataset, map all labels
to the final 14-mode MAST taxonomy, and save as JSON for pipeline evaluation.

Picks:
- All 4 generalizability-round traces (already use final 14-mode codes)
- All 5 Round 3 traces (near-final codes, mapped to final)

Usage:
    python extract_iaa_traces.py
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

# Round 3 uses 17 modes with slightly different names/numbering.
# Map by inspecting the actual label names in the HF data.
# Round 3 codes are close to final but have some differences.
ROUND3_CODE_MAP = {
    "1.1": "1.1",
    "1.2": "1.2",
    "1.3": "1.3",
    "1.4": "1.4",
    "1.5": "1.5",
    "2.1": "2.1",
    "2.2": "2.2",
    "2.3": "2.3",
    "2.4": "2.4",
    "2.5": "2.5",
    "2.6": "2.6",
    "3.1": "3.1",
    "3.2": "3.2",
    "3.3": "3.3",
    # Extra modes in Round 3 that don't exist in final 14 — best-effort map
    "3.4": None,   # dropped
    "3.5": None,   # dropped
    "4.1": "3.1",  # maps to Premature Termination if present
    "4.2": "3.3",  # maps to Weak Verification if present
    "4.3": "3.2",  # maps to No or Incorrect Verification if present
}


def extract_code(mode_name: str) -> str | None:
    m = re.match(r"(\d+\.\d+)", mode_name.strip())
    return m.group(1) if m else None


def majority_vote(item: dict) -> bool:
    """Return True if >=2 of 3 annotators say True."""
    votes = []
    for k in ["annotator_1", "annotator_2", "annotator_3"]:
        v = item.get(k, False)
        if isinstance(v, str):
            votes.append(v.strip().upper() == "TRUE")
        else:
            votes.append(bool(v))
    return sum(votes) >= 2


def process_trace(rec: dict, code_map: dict | None = None) -> dict:
    """Process a single human-labelled trace into evaluation format."""
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

        # If two old codes map to the same final code, take OR
        if final_code in labels:
            labels[final_code] = max(labels[final_code], vote)
        else:
            labels[final_code] = vote

    # Ensure all 14 modes are present (default 0 for unmapped)
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

    results = []

    # Generalizability round: indices 15-18, final 14-mode taxonomy (no mapping needed)
    for rec in data:
        rnd = str(rec.get("round", ""))
        if "generl" in rnd.lower() or "new" in rnd.lower():
            trace = process_trace(rec, code_map=None)
            results.append(trace)

    # Round 3: indices 10-14, near-final taxonomy (need mapping for any extras)
    for rec in data:
        rnd = str(rec.get("round", ""))
        if rnd == "Round 3":
            trace = process_trace(rec, code_map=ROUND3_CODE_MAP)
            results.append(trace)

    out_path = "iaa_traces_with_labels.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nExtracted {len(results)} traces → {out_path}")
    print()
    for t in results:
        fails = t["human_label_summary"]["failures"]
        print(f"  [{t['index']:2}] round={t['round']:<16s} "
              f"mas={t['mas_name']:<15s} bench={t['benchmark_name']:<18s} "
              f"chars={t['trace_char_len']:>7,d}  "
              f"failures={t['human_label_summary']['total_failures']:2d}/14")
        for fm in fails:
            print(f"       ✓ {fm}")
        print()


if __name__ == "__main__":
    main()
