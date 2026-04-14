"""
Compare LLM judge labels (from MAD_full_dataset) against human IAA majority
labels (from iaa_test_traces<200kchar.json) on overlapping traces.

The human-labeled file contains pre-computed majority-vote labels in
`human_labels` (dict of mode code -> 0/1), already using the final 14-mode
MAST taxonomy.

Usage:
    python compare_llm_vs_human.py
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

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

REPO_ID = "mcemri/MAD"

FINAL_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

# Mapping from Round 1's 18-mode scheme to the final 14-mode scheme.
# Based on observing the label names across rounds (see taxonomy_validation_analysis.md).
OLD_TO_FINAL = {
    "1.1": "1.1",  # Poor task constraint compliance → Disobey Task Specification
    "1.2": "2.6",  # Inconsistency between reasoning and action → Action-Reasoning Mismatch
    "1.3": None,    # Undetected conversation ambiguities → dropped/merged
    "1.4": "2.2",  # Fail to elicit clarification → Fail to Ask for Clarification
    "1.5": "1.5",  # Unaware of stopping conditions → Unaware of Termination Conditions
    "2.1": "1.3",  # Unbatched repetitive execution → Step Repetition (merged with 2.2)
    "2.2": "1.3",  # Step repetition → Step Repetition
    "2.3": "1.4",  # Backtracking interruption → Loss of Conversation History
    "2.4": "2.1",  # Conversation reset → Conversation Reset
    "2.5": "2.3",  # Derailment from task → Task Derailment
    "2.6": "1.2",  # Disobey role specification → Disobey Role Specification
    "3.1": None,    # Disagreement induced inaction → dropped/merged
    "3.2": "2.4",  # Withholding relevant information → Information Withholding
    "3.3": "2.5",  # Ignoring suggestions from agents → Ignored Other Agent's Input
    "3.4": None,    # Waiting for known information → dropped
    "4.1": "3.1",  # Ill specified termination → Premature Termination
    "4.2": "3.3",  # Lack of result verification → No or Incorrect Verification
    "4.3": "3.2",  # Lack of critical verification → Weak Verification
}


def load_full_raw() -> list[dict]:
    path = hf_hub_download(repo_id=REPO_ID, filename="MAD_full_dataset.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def load_human_raw() -> list[dict]:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(project_root, "new_data", "iaa_test_traces<200kchar.json")
    with open(path) as f:
        return json.load(f)


def extract_mode_code(mode_name: str) -> str | None:
    """Extract the numeric prefix (e.g. '1.1') from a full mode description."""
    m = re.match(r"(\d+\.\d+)", mode_name.strip())
    return m.group(1) if m else None


def normalize_bench(name: str) -> str:
    """Normalize benchmark names for matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def human_majority(annotations: list[dict], use_mapping: bool = False) -> dict[str, int]:
    """Compute majority vote (>=2 of 3 say True) per mode.
    If use_mapping, maps old codes to final codes."""
    result = {}
    for item in annotations:
        mode_name = item.get("failure mode", "")
        code = extract_mode_code(mode_name)
        if code is None:
            continue

        if use_mapping:
            final_code = OLD_TO_FINAL.get(code)
            if final_code is None:
                continue
            # If two old modes map to same final code, take OR (any positive)
            was_set = result.get(final_code, 0)
        else:
            final_code = code

        votes = []
        for k in ["annotator_1", "annotator_2", "annotator_3"]:
            v = item.get(k, False)
            if isinstance(v, str):
                votes.append(v.strip().upper() == "TRUE")
            else:
                votes.append(bool(v))

        majority = int(sum(votes) >= 2)

        if use_mapping and final_code in result:
            result[final_code] = max(result[final_code], majority)
        else:
            result[final_code] = majority
    return result


def llm_labels(record: dict) -> dict[str, int]:
    ann = record.get("mast_annotation", {})
    if isinstance(ann, str):
        ann = json.loads(ann)
    return {code: int(ann.get(code, 0)) for code in FINAL_MODES}


def main():
    print("Loading datasets...")
    full_data = load_full_raw()
    human_data = load_human_raw()

    # Build lookup by normalized (mas, bench, trace_id)
    full_lookup = {}
    for rec in full_data:
        mas = str(rec.get("mas_name", "")).lower()
        bench = normalize_bench(str(rec.get("benchmark_name", "")))
        tid = str(rec.get("trace_id", ""))
        full_lookup[(mas, bench, tid)] = rec

    # Also by (mas, bench) for broader matching
    full_by_mb = {}
    for rec in full_data:
        mas = str(rec.get("mas_name", "")).lower()
        bench = normalize_bench(str(rec.get("benchmark_name", "")))
        full_by_mb.setdefault((mas, bench), []).append(rec)

    print(f"Full dataset: {len(full_data)} traces")
    print(f"Human dataset: {len(human_data)} traces")
    print(f"\nFull dataset MAS+Benchmark combos:")
    for (m, b), recs in sorted(full_by_mb.items()):
        print(f"  {m:<18s} + {b:<18s} = {len(recs)} traces")

    # Match human traces to full dataset
    print("\n" + "=" * 70)
    print("MATCHING HUMAN TRACES TO FULL DATASET")
    print("=" * 70)

    matched = []
    for i, hrec in enumerate(human_data):
        mas = str(hrec.get("mas_name", "")).lower()
        bench = normalize_bench(str(hrec.get("benchmark_name", "")))
        tid = str(hrec.get("trace_id", hrec.get("index", "")))
        rnd = hrec.get("round", "?")

        key = (mas, bench, str(tid))
        frec = full_lookup.get(key)

        # Try swapped MAS/bench
        if frec is None:
            swapped = (bench, mas, str(tid))
            frec = full_lookup.get(swapped)

        # Try matching just by MAS+bench combo (take first with same trace_id range)
        if frec is None:
            mb_list = full_by_mb.get((mas, bench), [])
            if not mb_list:
                mb_list = full_by_mb.get((bench, mas), [])
            try:
                idx = int(tid)
                if mb_list and idx < len(mb_list):
                    frec = mb_list[idx]
            except (ValueError, TypeError):
                pass

        status = "MATCHED" if frec else "NO MATCH"
        hlabels = hrec.get("human_labels", {})
        n_modes = len(hlabels)

        print(f"  [{i:2d}] {status:<10s} round={rnd:<16s} "
              f"mas={hrec.get('mas_name','?'):<15s} bench={hrec.get('benchmark_name','?'):<18s} "
              f"modes={n_modes}")

        if frec is not None:
            matched.append((hrec, frec))

    print(f"\n>>> Total matched: {len(matched)} of {len(human_data)}")

    # Compare using pre-computed human_labels (already final 14-mode codes)
    all_h, all_l = [], []
    per_mode = {m: {"h": [], "l": []} for m in FINAL_MODES}
    detail_rows = []

    for hrec, frec in matched:
        rnd = hrec.get("round", "")
        hmaj = hrec.get("human_labels", {})
        llm = llm_labels(frec)

        trace_h, trace_l = 0, 0
        n_compared = 0
        for mode in FINAL_MODES:
            if mode in hmaj:
                h_val = int(hmaj[mode])
                l_val = llm.get(mode, 0)
                all_h.append(h_val)
                all_l.append(l_val)
                per_mode[mode]["h"].append(h_val)
                per_mode[mode]["l"].append(l_val)
                trace_h += h_val
                trace_l += l_val
                n_compared += 1

        detail_rows.append({
            "round": rnd, "mas": hrec.get("mas_name"),
            "bench": hrec.get("benchmark_name"),
            "modes_compared": n_compared,
            "human_yes": trace_h, "llm_yes": trace_l,
            "agree": sum(1 for m in FINAL_MODES if m in hmaj and int(hmaj[m]) == llm.get(m, 0)),
        })

    # Print per-trace detail
    print("\n" + "=" * 70)
    print("PER-TRACE COMPARISON")
    print("=" * 70)
    for d in detail_rows:
        n = d["modes_compared"]
        ag = d["agree"]
        pct = ag / n * 100 if n else 0
        print(f"  round={d['round']:<16s} mas={d['mas']:<15s} bench={d['bench']:<18s} "
              f"modes={n:2d}  human_yes={d['human_yes']:2d}  llm_yes={d['llm_yes']:2d}  "
              f"agree={ag:2d}/{n:2d} ({pct:.0f}%)")

    if not all_h:
        print("\nNo label pairs to compare!")
        return

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE COMPARISON")
    print("=" * 70)
    n_total = len(all_h)
    agree_total = sum(h == l for h, l in zip(all_h, all_l))
    print(f"Total label pairs: {n_total}")
    print(f"Human positives:   {sum(all_h)} ({sum(all_h)/n_total*100:.1f}%)")
    print(f"LLM positives:     {sum(all_l)} ({sum(all_l)/n_total*100:.1f}%)")
    print(f"Raw agreement:     {agree_total}/{n_total} ({agree_total/n_total*100:.1f}%)")

    kappa = cohen_kappa_score(all_h, all_l)
    print(f"Cohen's Kappa:     {kappa:.3f}")

    print("\nConfusion matrix (rows=human majority, cols=LLM):")
    cm = confusion_matrix(all_h, all_l, labels=[0, 1])
    print(f"                  LLM=no  LLM=yes")
    print(f"  Human=no        {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Human=yes       {cm[1][0]:6d}  {cm[1][1]:6d}")

    print("\nClassification report:")
    print(classification_report(all_h, all_l,
                                target_names=["no failure", "failure"],
                                zero_division=0))

    # Per-mode breakdown
    print("=" * 70)
    print("PER-MODE BREAKDOWN")
    print("=" * 70)
    print(f"{'Mode':<6s} {'H_yes':>5s} {'L_yes':>5s} {'Agree':>5s} {'N':>3s}  {'Acc%':>5s}  {'Kappa':>6s}")
    for mode in FINAL_MODES:
        h_list = per_mode[mode]["h"]
        l_list = per_mode[mode]["l"]
        if h_list:
            n = len(h_list)
            h_y = sum(h_list)
            l_y = sum(l_list)
            ag = sum(a == b for a, b in zip(h_list, l_list))
            acc = ag / n * 100
            try:
                k = cohen_kappa_score(h_list, l_list)
            except Exception:
                k = float("nan")
            print(f"{mode:<6s} {h_y:5d} {l_y:5d} {ag:5d} {n:3d}  {acc:5.1f}  {k:6.3f}")
        else:
            print(f"{mode:<6s}   — no data —")



if __name__ == "__main__":
    main()
