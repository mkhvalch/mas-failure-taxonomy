"""
Load and explore the MAD (Multi-Agent System Traces) dataset from HuggingFace.

Usage:
    python load_mad.py            # runs a quick smoke-test / summary
    python load_mad.py --human    # also load and summarise the human-labelled split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "mcemri/MAD"

FAILURE_MODES = {
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

CATEGORIES = {
    "System Design": ["1.1", "1.2", "1.3", "1.4", "1.5"],
    "Inter-Agent Misalignment": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6"],
    "Task Verification": ["3.1", "3.2", "3.3"],
}

def load_full_dataset() -> pd.DataFrame:
    """Download and parse the LLM-annotated MAD dataset into a DataFrame."""
    path = hf_hub_download(repo_id=REPO_ID, filename="MAD_full_dataset.json", repo_type="dataset")
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        ann = rec.get("mast_annotation", {})
        if isinstance(ann, str):
            ann = json.loads(ann)

        trace_field = rec.get("trace", {})
        if isinstance(trace_field, str):
            trace_field = json.loads(trace_field)

        row = {
            "mas_name": rec.get("mas_name"),
            "llm_name": rec.get("llm_name"),
            "benchmark_name": rec.get("benchmark_name"),
            "trace_id": rec.get("trace_id"),
            "trajectory": trace_field.get("trajectory", ""),
            "trace_key": trace_field.get("key", ""),
        }
        for code in FAILURE_MODES:
            row[code] = int(ann.get(code, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def load_human_dataset() -> pd.DataFrame:
    """Download and parse the human-labelled MAD subset."""
    path = hf_hub_download(repo_id=REPO_ID, filename="MAD_human_labelled_dataset.json", repo_type="dataset")
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        annotations = rec.get("annotations", [])
        if isinstance(annotations, str):
            annotations = json.loads(annotations)

        row = {
            "round": rec.get("round"),
            "mas_name": rec.get("mas_name"),
            "benchmark_name": rec.get("benchmark_name"),
            "trace_id": rec.get("trace_id"),
            "trace": rec.get("trace", ""),
        }
        for ann_item in annotations:
            mode = ann_item.get("failure mode", "")
            row[f"{mode}_a1"] = ann_item.get("annotator_1", False)
            row[f"{mode}_a2"] = ann_item.get("annotator_2", False)
            row[f"{mode}_a3"] = ann_item.get("annotator_3", False)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def failure_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-failure-mode occurrence rate across the full dataset."""
    mode_cols = [c for c in FAILURE_MODES if c in df.columns]
    rates = df[mode_cols].mean().rename("rate")
    counts = df[mode_cols].sum().rename("count")
    total = len(df)
    summary = pd.concat([rates, counts], axis=1)
    summary["description"] = summary.index.map(FAILURE_MODES)
    summary["total_traces"] = total
    summary["rate_pct"] = (summary["rate"] * 100).round(1)
    return summary[["description", "count", "total_traces", "rate_pct"]]


def failure_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute failure-mode rates grouped by a column (e.g. mas_name, llm_name)."""
    mode_cols = [c for c in FAILURE_MODES if c in df.columns]
    grouped = df.groupby(group_col)[mode_cols].mean() * 100
    grouped = grouped.round(1)
    grouped.columns = [f"{c} ({FAILURE_MODES[c]})" for c in grouped.columns]
    return grouped


def category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-category hit rate (trace has >= 1 failure in that category)."""
    rows = []
    for cat_name, codes in CATEGORIES.items():
        present = [c for c in codes if c in df.columns]
        if present:
            has_any = (df[present].sum(axis=1) > 0).mean() * 100
            rows.append({"category": cat_name, "traces_with_failure_pct": round(has_any, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load and explore the MAD dataset")
    parser.add_argument("--human", action="store_true", help="Also load the human-labelled subset")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading MAD full dataset from HuggingFace...")
    print("=" * 60)
    df = load_full_dataset()
    print(f"\nLoaded {len(df)} traces.\n")

    print(f"MAS frameworks: {sorted(df['mas_name'].unique())}")
    print(f"LLMs:           {sorted(df['llm_name'].dropna().unique())}")
    print(f"Benchmarks:     {sorted(df['benchmark_name'].unique())}")

    print(f"\nTraces per framework:")
    print(df["mas_name"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("Overall failure-mode rates")
    print("=" * 60)
    summary = failure_rate_summary(df)
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("Category-level failure rates")
    print("=" * 60)
    print(category_rates(df).to_string(index=False))

    print("\n" + "=" * 60)
    print("Failure rates by MAS framework (%)")
    print("=" * 60)
    by_mas = failure_rate_by_group(df, "mas_name")
    print(by_mas.to_string())

    print("\n" + "=" * 60)
    print("Failure rates by LLM (%)")
    print("=" * 60)
    by_llm = failure_rate_by_group(df, "llm_name")
    print(by_llm.to_string())

    if args.human:
        print("\n" + "=" * 60)
        print("Loading human-labelled dataset...")
        print("=" * 60)
        hdf = load_human_dataset()
        print(f"Loaded {len(hdf)} human-annotated traces.")
        print(f"Columns: {list(hdf.columns)}")
        print(hdf.head())

    print("\n✓ Done. Import `load_mad` in your own scripts for further experiments.")


if __name__ == "__main__":
    main()
