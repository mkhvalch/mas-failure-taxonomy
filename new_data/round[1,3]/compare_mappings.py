"""Compare our iaa_all19_traces_final14.json vs load_mast.py's remap output."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from huggingface_hub import hf_hub_download

# load_mast.py's remap (verbatim, exactly as in that script)
REMAP_LOAD_MAST = {
    # Round 1
    "1.1 Poor task constraint compliance": "1.1 Disobey Task Specification",
    "1.2 Inconsistency between reasoning and action": "2.6 Reasoning-Action Mismatch",
    "1.3 Undetected conversation ambiguities and contradictions": None,
    "1.4 Fail to elicit clarification": "2.2 Fail to ask for clarification",
    "1.5 Unaware of stopping conditions": "1.5 Unaware of Termination Conditions",
    "2.1 Unbatched repetitive execution": None,
    "2.2 Step repetition": "1.3 Step Repetition",
    "2.3 Backtracking interruption": None,
    "2.4 Conversation reset": "2.1 Conversation reset",
    "2.5 Derailment from task": "2.3 Task derailment",
    "2.6 Disobey role specification": "1.2 Disobey Role Specification",
    "3.1 Disagreement induced inaction": None,
    "3.2 Withholding relevant information": "2.4 Information Witholding",
    "3.3 Ignoring suggestions from agents": "2.5 Ignored Other Agents' Input",
    "3.4 Waiting for known information": None,
    "4.1 Ill specified termination condition leading to premature termination": "3.1 Premature Termination",
    "4.2 Lack of result verification": "3.3 Incorrect Verification",
    "4.3 Lack of critical verification": "3.2 No or Incomplete Verification",
    # Round 2/3
    "1.3 Unaware of stopping conditions": "1.5 Unaware of Termination Conditions",
    "1.4 Unbatched repetitive execution": None,
    "1.5 Step repetition": "1.3 Step Repetition",
    "1.6 Backtracking interruption": None,
    "1.7 Disobey role specification": "1.2 Disobey Role Specification",
    "2.1 Conversation reset": "2.1 Conversation reset",
    "2.2 Fail to elicit clarification": "2.2 Fail to ask for clarification",
    "2.3 Derailment from task": "2.3 Task derailment",
    "2.4 Undetected conversation ambiguities and contradictions": None,
    "2.5 Disagreement induced inaction": None,
    "2.6 Withholding relevant information": "2.4 Information Witholding",
    "2.7 Ignoring suggestions from agents": "2.5 Ignored Other Agents' Input",
    "3.1 Ill specified termination condition leading to premature termination": "3.1 Premature Termination",
    "3.2 Lack of critical verification": "3.2 No or Incomplete Verification",
    "3.3 Lack of result verification": "3.3 Incorrect Verification",
    # Generalizability (identity)
    "1.1 Disobey Task Specification": "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification": "1.2 Disobey Role Specification",
    "1.3 Step Repetition": "1.3 Step Repetition",
    "1.4 Loss of Conversation History": "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions": "1.5 Unaware of Termination Conditions",
    "2.1 Conversation reset": "2.1 Conversation reset",
    "2.2 Fail to ask for clarification": "2.2 Fail to ask for clarification",
    "2.3 Task derailment": "2.3 Task derailment",
    "2.4 Information Witholding": "2.4 Information Witholding",
    "2.5 Ignored Other Agents' Input": "2.5 Ignored Other Agents' Input",
    "2.6 Reasoning-Action Mismatch": "2.6 Reasoning-Action Mismatch",
    "3.1 Premature Termination": "3.1 Premature Termination",
    "3.2 No or Incomplete Verification": "3.2 No or Incomplete Verification",
    "3.3 Incorrect Verification": "3.3 Incorrect Verification",
}


def code(name: str) -> str | None:
    m = re.match(r"(\d+\.\d+)", name.strip())
    return m.group(1) if m else None


def load_mast_labels(record: dict) -> dict[str, int]:
    """Apply load_mast.py logic: majority vote, then remap to canonical MAST."""
    labels = {f"{a}.{b}": 0 for a in (1, 2, 3) for b in range(1, 7) if (a, b) != (1, 6)}
    for code_str in ["1.1", "1.2", "1.3", "1.4", "1.5",
                      "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
                      "3.1", "3.2", "3.3"]:
        labels.setdefault(code_str, 0)
    labels = {c: 0 for c in ["1.1", "1.2", "1.3", "1.4", "1.5",
                              "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
                              "3.1", "3.2", "3.3"]}

    for ann in record.get("annotations", []):
        votes = [ann.get("annotator_1"), ann.get("annotator_2"), ann.get("annotator_3")]
        votes = [bool(v) if not isinstance(v, str) else v.strip().upper() == "TRUE" for v in votes]
        if sum(votes) < 2:
            continue
        raw = ann["failure mode"].split("\n")[0].strip()
        canonical = REMAP_LOAD_MAST.get(raw)
        if canonical is None:
            continue
        c = code(canonical)
        if c in labels:
            labels[c] = 1
    return labels


def main():
    path = hf_hub_download(repo_id="mcemri/MAD",
                           filename="MAD_human_labelled_dataset.json",
                           repo_type="dataset")
    with open(path) as f:
        raw = json.load(f)
    raw_by_tid = {str(r["trace_id"]): r for r in raw}

    with open("new_data/iaa_all19_traces_final14.json") as f:
        ours = json.load(f)

    modes = ["1.1", "1.2", "1.3", "1.4", "1.5",
             "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
             "3.1", "3.2", "3.3"]

    print(f"{'tid':>4} {'round':<17} {'mas':<11} "
          + " ".join(f"{m:>3}" for m in modes) + "  diff")
    print("-" * 110)

    total_diffs = 0
    diffs_by_mode = Counter()
    diffs_by_round = Counter()

    for t in ours:
        tid = str(t["index"])
        rnd = t["round"]
        mas = t["mas_name"]
        ours_lbl = t["human_labels"]
        theirs_lbl = load_mast_labels(raw_by_tid[tid])

        cells = []
        diffs = []
        for m in modes:
            o = ours_lbl.get(m, 0)
            th = theirs_lbl.get(m, 0)
            if o == th:
                cells.append(f"{o:>3}")
            else:
                cells.append(f"\033[91m{o}/{th}\033[0m")
                diffs.append(m)
                diffs_by_mode[m] += 1
                diffs_by_round[rnd] += 1
                total_diffs += 1
        diff_str = ",".join(diffs) if diffs else "-"
        print(f"{tid:>4} {rnd:<17} {mas:<11} "
              + " ".join(cells) + f"  {diff_str}")

    print()
    print(f"Total cell-level disagreements: {total_diffs} / {19*14} = {total_diffs/(19*14):.1%}")
    print(f"By round: {dict(diffs_by_round)}")
    print(f"By mode:  {dict(diffs_by_mode.most_common())}")

    # Per-trace totals
    print("\nPer-trace failure counts:")
    print(f"{'tid':>4} {'round':<17} ours theirs diff")
    for t in ours:
        tid = str(t["index"])
        ours_n = sum(t["human_labels"].values())
        theirs_n = sum(load_mast_labels(raw_by_tid[tid]).values())
        marker = "" if ours_n == theirs_n else "  <-- DIFF"
        print(f"{tid:>4} {t['round']:<17} {ours_n:4d} {theirs_n:6d} {ours_n - theirs_n:+d}{marker}")


if __name__ == "__main__":
    main()
