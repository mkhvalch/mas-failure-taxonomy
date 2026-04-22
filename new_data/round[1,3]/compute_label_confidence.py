"""
Compute per-cell confidence scores for iaa_all19_traces_final14.json.

For each of the 19 traces x 14 modes = 266 cells, we produce three orthogonal
confidence components and a combined score. Use any of them (or the combined)
as weights when computing evaluation metrics.

Components
----------

1. annotator_consensus [0, 1]
   How strongly the annotators agreed on the cell's value.
   - If the cell is positive (label=1): max over source labels mapping to this
     final mode of (positive_votes / 3). 1.0 = unanimous, 0.67 = 2-of-3.
   - If the cell is negative (label=0) and at least one source label maps to
     this final mode: min over those source labels of (negative_votes / 3).
     Strong negatives (3/3 say no) score higher than weak negatives (2/3 say no).
   - If NO source label maps to this final mode, the cell is 0 by taxonomy
     absence rather than by annotator judgment. We mark this with a separate
     `source_present=false` flag and set annotator_consensus = None.

2. mapping_directness [0, 1]
   How clean the taxonomy mapping was from the source round to final-14.
   - 1.0: Generalizability (native final-14, identity map)
   - 0.9: Straight rename, same semantics (e.g., "Conversation reset" -> 2.1)
   - 0.7: Clear semantic correspondence but different phrasing
         (e.g., "Inconsistency between reasoning and action" -> 2.6)
   - 0.5: Two source modes collapsed onto one final mode (ambiguity about
         which source actually fired)
   - 0.3: No clean analogue; mapped to nearest bin
         (e.g., "Backtracking interruption" -> 1.4)
   If multiple source labels map to the same final cell, we take the MIN
   (most uncertain source).

3. round_quality [0, 1]
   How trustworthy the round-level annotation process was.
   - 1.0: Generalizability (final taxonomy, independent annotation)
   - 0.8: Round 3 (near-final taxonomy)
   - 0.5: Round 2 (calibration; showed uniform labels per MAS across traces)
   - 0.3: Round 1 (early taxonomy; uniform labels per MAS; largest mapping
         loss)

combined = annotator_consensus * mapping_directness * round_quality
(when annotator_consensus is None, combined is None)

Run:
    python new_data/compute_label_confidence.py
Writes new_data/iaa_all19_confidence.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Ensure local taxonomy_mappings.py is importable regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from huggingface_hub import hf_hub_download

from taxonomy_mappings import (  # noqa: E402
    FINAL_MODES,
    FINAL_MODE_NAMES,
    ROUND_TO_MAP,
)

# --------------------------------------------------------------------------- #
# Per-source-label mapping-directness scores (manual, see docstring).
# Key = (round_name, source_code). Value = float in [0, 1].
# --------------------------------------------------------------------------- #

MAPPING_DIRECTNESS: dict[tuple[str, str], float] = {
    # Round 1 (early-mast-18)
    ("Round 1", "1.1"): 0.9,   # Poor task constraint compliance ~= 1.1
    ("Round 1", "1.2"): 0.7,   # Inconsistency reasoning/action -> 2.6 (rename)
    ("Round 1", "1.3"): 0.3,   # Undetected ambiguities -> 2.2 (forced)
    ("Round 1", "1.4"): 0.9,   # Fail to elicit clarification ~= 2.2
    ("Round 1", "1.5"): 1.0,   # Unaware of stopping conditions == 1.5
    ("Round 1", "2.1"): 0.3,   # Unbatched repetitive execution -> 1.3 (forced)
    ("Round 1", "2.2"): 1.0,   # Step repetition == 1.3
    ("Round 1", "2.3"): 0.3,   # Backtracking interruption -> 1.4 (forced)
    ("Round 1", "2.4"): 1.0,   # Conversation reset == 2.1
    ("Round 1", "2.5"): 1.0,   # Derailment from task == 2.3
    ("Round 1", "2.6"): 1.0,   # Disobey role specification == 1.2
    ("Round 1", "3.2"): 1.0,   # Withholding relevant information == 2.4
    ("Round 1", "3.3"): 1.0,   # Ignoring suggestions from agents == 2.5
    ("Round 1", "4.1"): 1.0,   # Ill-specified termination == 3.1
    ("Round 1", "4.2"): 0.6,   # Result verification -> 3.2 (ambiguous swap)
    ("Round 1", "4.3"): 0.6,   # Critical verification -> 3.3 (ambiguous swap)
    # Rounds 2 & 3 (interim-mast-17)
    ("interim17", "1.1"): 0.9,  # Poor task constraint compliance
    ("interim17", "1.2"): 0.7,  # Inconsistency reasoning/action -> 2.6
    ("interim17", "1.3"): 1.0,  # Unaware of stopping conditions -> 1.5
    ("interim17", "1.4"): 0.3,  # Unbatched repetitive execution -> 1.3 (forced)
    ("interim17", "1.5"): 1.0,  # Step repetition -> 1.3
    ("interim17", "1.6"): 0.3,  # Backtracking interruption -> 1.4 (forced)
    ("interim17", "1.7"): 1.0,  # Disobey role specification -> 1.2
    ("interim17", "2.1"): 1.0,  # Conversation reset
    ("interim17", "2.2"): 0.9,  # Fail to elicit clarification
    ("interim17", "2.3"): 1.0,  # Derailment from task
    ("interim17", "2.4"): 0.3,  # Undetected ambiguities -> 2.2 (forced)
    ("interim17", "2.6"): 1.0,  # Withholding relevant information
    ("interim17", "2.7"): 1.0,  # Ignoring suggestions
    ("interim17", "3.1"): 1.0,  # Ill-specified termination
    ("interim17", "3.2"): 0.6,  # Critical verification -> 3.3 (ambiguous swap)
    ("interim17", "3.3"): 0.6,  # Result verification -> 3.2 (ambiguous swap)
}


ROUND_QUALITY: dict[str, float] = {
    "Generlazability": 1.0,
    "Round 3": 0.8,
    "Round 2": 0.5,
    "Round 1": 0.3,
}


# --------------------------------------------------------------------------- #

_CODE_RE = re.compile(r"(\d+\.\d+)")


def extract_code(name: str) -> str | None:
    m = _CODE_RE.match(name.strip())
    return m.group(1) if m else None


def annotator_fractions(ann: dict) -> tuple[float, float]:
    """Return (positive_fraction, negative_fraction) across the 3 annotators."""
    votes = []
    for k in ("annotator_1", "annotator_2", "annotator_3"):
        v = ann.get(k, False)
        votes.append(bool(v) if not isinstance(v, str) else v.strip().upper() == "TRUE")
    pos = sum(votes)
    return pos / 3.0, (3 - pos) / 3.0


def source_scheme(round_name: str) -> str:
    return "interim17" if round_name in ("Round 2", "Round 3") else round_name


def process(record: dict) -> dict:
    round_name = str(record.get("round", ""))
    code_map = ROUND_TO_MAP.get(round_name)
    scheme_key = source_scheme(round_name) if code_map is not None else None
    round_q = ROUND_QUALITY.get(round_name, 0.5)

    # Collect per-source info: {final_code: [(source_code, pos_frac, neg_frac)]}
    per_final: dict[str, list[tuple[str, float, float]]] = {m: [] for m in FINAL_MODES}

    for ann in record.get("annotations", []):
        raw = extract_code(ann.get("failure mode", ""))
        if raw is None:
            continue
        if code_map is None:
            final = raw if raw in FINAL_MODE_NAMES else None
        else:
            final = code_map.get(raw)
        if final is None or final not in FINAL_MODE_NAMES:
            continue
        pos_f, neg_f = annotator_fractions(ann)
        per_final[final].append((raw, pos_f, neg_f))

    # Build per-cell confidence
    confidence = {}
    labels = {}
    for m in FINAL_MODES:
        sources = per_final[m]
        if not sources:
            labels[m] = 0
            confidence[m] = {
                "label": 0,
                "source_present": False,
                "annotator_consensus": None,
                "mapping_directness": None,
                "round_quality": round_q,
                "combined": None,
            }
            continue

        # Final label: 1 if any source's majority vote is positive.
        any_positive = any(pos_f >= 2 / 3 for _, pos_f, _ in sources)
        label = 1 if any_positive else 0
        labels[m] = label

        if label == 1:
            # Use the strongest positive source.
            positives = [(s, p) for s, p, _ in sources if p >= 2 / 3]
            ann_conf = max(p for _, p in positives)
            # Which source(s) drove it? Use their mapping scores; take MIN.
            if scheme_key is None:
                map_scores = [1.0 for _ in positives]  # Generalizability
            else:
                map_scores = [MAPPING_DIRECTNESS.get((scheme_key, s), 0.5)
                              for s, _ in positives]
            map_conf = min(map_scores)
        else:
            # All sources say negative. Weakest negative = min(neg_frac).
            negatives = [(s, n) for s, _, n in sources]
            ann_conf = min(n for _, n in negatives)
            if scheme_key is None:
                map_scores = [1.0 for _ in negatives]
            else:
                map_scores = [MAPPING_DIRECTNESS.get((scheme_key, s), 0.5)
                              for s, _ in negatives]
            map_conf = min(map_scores)

        combined = ann_conf * map_conf * round_q
        confidence[m] = {
            "label": label,
            "source_present": True,
            "annotator_consensus": round(ann_conf, 3),
            "mapping_directness": round(map_conf, 3),
            "round_quality": round_q,
            "combined": round(combined, 3),
            "source_codes": [s for s, _, _ in sources],
        }

    n_sources_absent = sum(1 for m in FINAL_MODES if not confidence[m]["source_present"])
    n_positive = sum(labels.values())

    return {
        "index": record.get("trace_id"),
        "round": round_name,
        "mas_name": record.get("mas_name"),
        "benchmark_name": record.get("benchmark_name"),
        "human_labels": labels,
        "label_confidence": confidence,
        "trace_summary": {
            "n_positive_labels": n_positive,
            "n_source_absent_modes": n_sources_absent,
            "round_quality": round_q,
        },
    }


def main():
    print("Downloading MAD_human_labelled_dataset.json ...")
    path = hf_hub_download(
        repo_id="mcemri/MAD",
        filename="MAD_human_labelled_dataset.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    results = [process(r) for r in data]
    results.sort(key=lambda t: (t["round"], str(t["index"])))

    out = os.path.join(_THIS_DIR, "iaa_all19_confidence.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {len(results)} traces -> {out}\n")

    # Print a compact summary
    print(f"{'tid':>4} {'round':<18} {'mas':<12} "
          f"{'n+':>3} {'absent':>6} {'mean_conf(+)':>13} {'mean_conf(-)':>13}")
    print("-" * 80)
    for t in results:
        pos_combs = [c["combined"] for m, c in t["label_confidence"].items()
                     if c["label"] == 1 and c["combined"] is not None]
        neg_combs = [c["combined"] for m, c in t["label_confidence"].items()
                     if c["label"] == 0 and c["combined"] is not None]
        mp = f"{sum(pos_combs)/len(pos_combs):.2f}" if pos_combs else "  -  "
        mn = f"{sum(neg_combs)/len(neg_combs):.2f}" if neg_combs else "  -  "
        print(f"{t['index']:>4} {t['round']:<18} {t['mas_name']:<12} "
              f"{t['trace_summary']['n_positive_labels']:>3} "
              f"{t['trace_summary']['n_source_absent_modes']:>6} "
              f"{mp:>13} {mn:>13}")


if __name__ == "__main__":
    main()
