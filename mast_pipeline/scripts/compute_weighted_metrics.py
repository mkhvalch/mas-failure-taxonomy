"""Compute unweighted and confidence-weighted metrics for the 19-trace run.

Joins:
  - LLM outputs:     mast_pipeline/outputs/all19/pipeline_evaluation_results_all19.json
  - Human labels +
    confidence:      new_data/round[1,3]/iaa_all19_confidence.json

For each (trace, mode) cell we have:
  - h = human label (0/1)
  - l = LLM label   (0/1)
  - w = combined confidence (in [0, 1])

Unweighted metrics treat every cell with weight 1.
Weighted metrics weight every cell by w.

Kappa is computed the standard way but with weighted counts substituted for raw
counts.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root

LLM_PATH  = os.path.join(ROOT, "mast_pipeline", "outputs", "all19",
                         "pipeline_evaluation_results_all19.json")
CONF_PATH = os.path.join(ROOT, "new_data", "round[1,3]", "iaa_all19_confidence.json")
OUT_PATH  = os.path.join(ROOT, "mast_pipeline", "outputs", "all19",
                         "weighted_metrics_all19.json")

FINAL_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]


def weighted_metrics(cells: list[dict]) -> dict:
    """cells: list of {h, l, w}. Returns dict with accuracy/P/R/F1/kappa/rates."""
    sw = sum(c["w"] for c in cells)
    if sw == 0:
        return {"accuracy": None, "precision": None, "recall": None,
                "f1": None, "kappa": None, "n_cells": 0, "weight_sum": 0}

    # confusion matrix, weighted
    tp = sum(c["w"] for c in cells if c["h"] == 1 and c["l"] == 1)
    fp = sum(c["w"] for c in cells if c["h"] == 0 and c["l"] == 1)
    fn = sum(c["w"] for c in cells if c["h"] == 1 and c["l"] == 0)
    tn = sum(c["w"] for c in cells if c["h"] == 0 and c["l"] == 0)

    acc = (tp + tn) / sw
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Cohen's kappa with weighted counts
    p_o = acc
    p_h_yes = (tp + fn) / sw
    p_l_yes = (tp + fp) / sw
    p_e = p_h_yes * p_l_yes + (1 - p_h_yes) * (1 - p_l_yes)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "kappa":     round(kappa, 4),
        "human_pos_rate": round(p_h_yes, 4),
        "llm_pos_rate":   round(p_l_yes, 4),
        "n_cells":   len(cells),
        "weight_sum": round(sw, 2),
    }


def main() -> None:
    with open(LLM_PATH) as f:
        llm_results = json.load(f)
    with open(CONF_PATH) as f:
        conf = json.load(f)

    conf_by_idx = {c["index"]: c for c in conf}

    unweighted_cells: list[dict] = []
    weighted_cells: list[dict]   = []

    per_trace_rows = []
    per_mode = {m: {"u": [], "w": []} for m in FINAL_MODES}

    skipped_traces = 0
    for rec in llm_results:
        if "error" in rec:
            skipped_traces += 1
            continue
        idx = rec["index"]
        c = conf_by_idx.get(idx)
        if c is None:
            skipped_traces += 1
            continue
        lc = c["label_confidence"]

        tu_cells, tw_cells = [], []
        for m in FINAL_MODES:
            h = int(rec["human_labels"].get(m, 0))
            l = int(rec["llm_labels"].get(m, 0))
            w = float(lc[m]["combined"]) if lc.get(m) else 0.0

            unweighted_cells.append({"h": h, "l": l, "w": 1.0})
            weighted_cells.append(  {"h": h, "l": l, "w": w})

            tu_cells.append({"h": h, "l": l, "w": 1.0})
            tw_cells.append({"h": h, "l": l, "w": w})

            per_mode[m]["u"].append({"h": h, "l": l, "w": 1.0})
            per_mode[m]["w"].append({"h": h, "l": l, "w": w})

        per_trace_rows.append({
            "index":    idx,
            "round":    rec["round"],
            "mas":      rec["mas"],
            "bench":    rec["bench"],
            "unweighted": weighted_metrics(tu_cells),
            "weighted":   weighted_metrics(tw_cells),
        })

    overall = {
        "unweighted": weighted_metrics(unweighted_cells),
        "weighted":   weighted_metrics(weighted_cells),
    }

    per_mode_out = {}
    for m in FINAL_MODES:
        per_mode_out[m] = {
            "unweighted": weighted_metrics(per_mode[m]["u"]),
            "weighted":   weighted_metrics(per_mode[m]["w"]),
        }

    out = {
        "n_traces_evaluated": len(per_trace_rows),
        "n_traces_skipped": skipped_traces,
        "overall": overall,
        "per_mode": per_mode_out,
        "per_trace": per_trace_rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # Print compact summary
    def row(label, m):
        print(f"  {label:<12} {m['accuracy']:>7.3f}  {m['precision']:>7.3f}  "
              f"{m['recall']:>7.3f}  {m['f1']:>7.3f}  {m['kappa']:>7.3f}  "
              f"{m['human_pos_rate']*100:>5.1f}%  {m['llm_pos_rate']*100:>5.1f}%")

    print(f"\nTraces: {len(per_trace_rows)} evaluated, {skipped_traces} skipped")
    print("=" * 78)
    print(f"  {'variant':<12} {'acc':>7}  {'prec':>7}  {'rec':>7}  {'f1':>7}  "
          f"{'kappa':>7}  {'H+%':>5}  {'L+%':>5}")
    print("-" * 78)
    row("unweighted", overall["unweighted"])
    row("weighted",   overall["weighted"])

    print("\nPER-MODE (unweighted / weighted kappa side-by-side)")
    print("=" * 78)
    print(f"  {'mode':<6} {'H':>3} {'L':>3} {'uw_acc':>8} {'uw_kappa':>9} "
          f"{'w_acc':>8} {'w_kappa':>9}")
    for m in FINAL_MODES:
        u, w = per_mode_out[m]["unweighted"], per_mode_out[m]["weighted"]
        h_y = sum(c["h"] for c in per_mode[m]["u"])
        l_y = sum(c["l"] for c in per_mode[m]["u"])
        print(f"  {m:<6} {h_y:>3} {l_y:>3} {u['accuracy']:>8.3f} "
              f"{u['kappa']:>9.3f} {w['accuracy']:>8.3f} {w['kappa']:>9.3f}")

    print(f"\nSaved detailed metrics to {OUT_PATH}")


if __name__ == "__main__":
    main()
