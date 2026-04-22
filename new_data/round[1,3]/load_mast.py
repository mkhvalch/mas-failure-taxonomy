from huggingface_hub import hf_hub_download
from collections import Counter, defaultdict
import json

REPO_ID = "mcemri/MAD"
FILENAME = "MAD_human_labelled_dataset.json"

file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset"
)

with open(file_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records (human labelled).")

# Quick peek at structure
print("\nFirst record keys:", list(data[0].keys()) if data else "empty")

# Distribution across frameworks, benchmarks, and rounds
print("\n--- Distributions ---")
print("MAS frameworks:", Counter(r["mas_name"] for r in data))
print("Benchmarks:", Counter(r["benchmark_name"] for r in data))
print("Rounds:", Counter(r["round"] for r in data))

# Inspect annotations structure — the critical field for MAST taxonomy
print("\n--- Annotation structure (record 0) ---")
print(json.dumps(data[0]["annotations"], indent=2)[:2000])

# Trace length stats
trace_lens = [len(r["trace"]) for r in data]
print(f"\n--- Trace lengths (chars) ---")
print(f"min={min(trace_lens)}, max={max(trace_lens)}, "
      f"mean={sum(trace_lens)//len(trace_lens)}")

# --- Failure mode analysis ---

# 1. Failure mode prevalence across the 19 traces (majority vote of 3 annotators)
print("\n--- Failure mode prevalence (majority vote, 2-of-3) ---")
mode_counts = defaultdict(int)
for record in data:
    for ann in record["annotations"]:
        votes = [ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]
        if sum(votes) >= 2:
            mode = ann["failure mode"].split("\n")[0]  # just the label
            mode_counts[mode] += 1

for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:2d}  {mode}")

# 2. How many failure modes per trace (multi-label density)
labels_per_trace = []
for record in data:
    n = sum(1 for ann in record["annotations"]
            if sum([ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]) >= 2)
    labels_per_trace.append(n)
print(f"\n--- Failure modes per trace ---")
print(f"min={min(labels_per_trace)}, max={max(labels_per_trace)}, "
      f"mean={sum(labels_per_trace)/len(labels_per_trace):.1f}")

# 3. Inter-annotator agreement (unanimous rate)
agreements = []
for record in data:
    for ann in record["annotations"]:
        votes = [ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]
        agreements.append(len(set(votes)) == 1)  # all three agree
print(f"\n--- Inter-annotator agreement ---")
print(f"Unanimous agreement rate: {sum(agreements)/len(agreements):.1%}")

from collections import defaultdict

# Enumerate unique labels used in each round
labels_by_round = defaultdict(set)
for record in data:
    for ann in record["annotations"]:
        label = ann["failure mode"].split("\n")[0].strip()
        labels_by_round[record["round"]].add(label)

print("\n--- Labels used per round ---")
for rnd in ["Round 1", "Round 2", "Round 3", "Generlazability"]:
    labels = labels_by_round.get(rnd, set())
    print(f"\n{rnd} ({len(labels)} unique labels):")
    for lbl in sorted(labels):
        print(f"  {lbl}")

# Also useful: which labels ONLY appear in certain rounds?
all_labels = set().union(*labels_by_round.values())
print(f"\n--- Cross-round label analysis ---")
print(f"Total unique labels across all rounds: {len(all_labels)}")

for lbl in sorted(all_labels):
    rounds_with_label = [r for r, labs in labels_by_round.items() if lbl in labs]
    if len(rounds_with_label) < 4:  # appears in some but not all rounds
        print(f"  {lbl}")
        print(f"    → in: {rounds_with_label}")


# Canonical MAST = Generalizability labels (the final 14-mode taxonomy from the paper)

# Maps any round's label → canonical MAST label
remap_to_mast = {
    # --- Round 1 → MAST ---
    "1.1 Poor task constraint compliance":                               "1.1 Disobey Task Specification",
    "1.2 Inconsistency between reasoning and action":                    "2.6 Reasoning-Action Mismatch",
    "1.3 Undetected conversation ambiguities and contradictions":        None,  # dropped in final taxonomy
    "1.4 Fail to elicit clarification":                                  "2.2 Fail to ask for clarification",
    "1.5 Unaware of stopping conditions":                                "1.5 Unaware of Termination Conditions",
    "2.1 Unbatched repetitive execution":                                None,  # dropped
    "2.2 Step repetition":                                               "1.3 Step Repetition",
    "2.3 Backtracking interruption":                                     None,  # dropped
    "2.4 Conversation reset":                                            "2.1 Conversation reset",
    "2.5 Derailment from task":                                          "2.3 Task derailment",
    "2.6 Disobey role specification":                                    "1.2 Disobey Role Specification",
    "3.1 Disagreement induced inaction":                                 None,  # dropped
    "3.2 Withholding relevant information":                              "2.4 Information Witholding",
    "3.3 Ignoring suggestions from agents":                              "2.5 Ignored Other Agents' Input",
    "3.4 Waiting for known information":                                 None,  # dropped
    "4.1 Ill specified termination condition leading to premature termination": "3.1 Premature Termination",
    "4.2 Lack of result verification":                                   "3.3 Incorrect Verification",
    "4.3 Lack of critical verification":                                 "3.2 No or Incomplete Verification",

    # --- Round 2 & Round 3 → MAST (same taxonomy for both rounds) ---
    "1.1 Poor task constraint compliance":                               "1.1 Disobey Task Specification",
    "1.2 Inconsistency between reasoning and action":                    "2.6 Reasoning-Action Mismatch",
    "1.3 Unaware of stopping conditions":                                "1.5 Unaware of Termination Conditions",
    "1.4 Unbatched repetitive execution":                                None,  # dropped
    "1.5 Step repetition":                                               "1.3 Step Repetition",
    "1.6 Backtracking interruption":                                     None,  # dropped
    "1.7 Disobey role specification":                                    "1.2 Disobey Role Specification",
    "2.1 Conversation reset":                                            "2.1 Conversation reset",
    "2.2 Fail to elicit clarification":                                  "2.2 Fail to ask for clarification",
    "2.3 Derailment from task":                                          "2.3 Task derailment",
    "2.4 Undetected conversation ambiguities and contradictions":        None,  # dropped
    "2.5 Disagreement induced inaction":                                 None,  # dropped
    "2.6 Withholding relevant information":                              "2.4 Information Witholding",
    "2.7 Ignoring suggestions from agents":                              "2.5 Ignored Other Agents' Input",
    "3.1 Ill specified termination condition leading to premature termination": "3.1 Premature Termination",
    "3.2 Lack of critical verification":                                 "3.2 No or Incomplete Verification",
    "3.3 Lack of result verification":                                   "3.3 Incorrect Verification",

    # --- Generalizability → MAST (identity) ---
    "1.1 Disobey Task Specification":           "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification":           "1.2 Disobey Role Specification",
    "1.3 Step Repetition":                      "1.3 Step Repetition",
    "1.4 Loss of Conversation History":         "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions":    "1.5 Unaware of Termination Conditions",
    "2.1 Conversation reset":                   "2.1 Conversation reset",
    "2.2 Fail to ask for clarification":        "2.2 Fail to ask for clarification",
    "2.3 Task derailment":                      "2.3 Task derailment",
    "2.4 Information Witholding":               "2.4 Information Witholding",
    "2.5 Ignored Other Agents' Input":          "2.5 Ignored Other Agents' Input",
    "2.6 Reasoning-Action Mismatch":            "2.6 Reasoning-Action Mismatch",
    "3.1 Premature Termination":                "3.1 Premature Termination",
    "3.2 No or Incomplete Verification":        "3.2 No or Incomplete Verification",
    "3.3 Incorrect Verification":               "3.3 Incorrect Verification",
}

# Recompute failure mode prevalence on the normalized taxonomy
from collections import defaultdict

mode_counts_canonical = defaultdict(int)
dropped_count = 0
for record in data:
    for ann in record["annotations"]:
        votes = [ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]
        if sum(votes) >= 2:
            raw_label = ann["failure mode"].split("\n")[0].strip()
            canonical = remap_to_mast.get(raw_label, "UNMAPPED: " + raw_label)
            if canonical is None:
                dropped_count += 1  # mode was removed in final taxonomy
            else:
                mode_counts_canonical[canonical] += 1

print("\n--- Failure mode prevalence on CANONICAL MAST (all 19 traces, majority vote) ---")
for mode, count in sorted(mode_counts_canonical.items(), key=lambda x: -x[1]):
    print(f"  {count:2d}  {mode}")
print(f"\n  Dropped (modes removed in final taxonomy): {dropped_count}")


from collections import defaultdict

# --- Helper: roll a canonical MAST mode up to its category ---
def mode_to_category(canonical_mode):
    """Maps a 14-mode MAST label to its high-level category."""
    if canonical_mode.startswith("1."):
        return "Specification"
    elif canonical_mode.startswith("2."):
        return "Inter-agent Misalignment"
    elif canonical_mode.startswith("3."):
        return "Verification"
    return "Unknown"


def compute_prevalence(records, remap, label="ALL"):
    """Compute mode-level and category-level counts for a set of records."""
    mode_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    traces_with_mode = defaultdict(set)
    traces_with_cat = defaultdict(set)

    for record in records:
        trace_key = (record["mas_name"], record["trace_id"], record["round"])
        for ann in record["annotations"]:
            votes = [ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]
            if sum(votes) >= 2:
                raw = ann["failure mode"].split("\n")[0].strip()
                canonical = remap.get(raw)
                if canonical is None:
                    continue  # mode dropped in final taxonomy
                mode_counts[canonical] += 1
                traces_with_mode[canonical].add(trace_key)
                cat = mode_to_category(canonical)
                cat_counts[cat] += 1
                traces_with_cat[cat].add(trace_key)

    print(f"\n=== {label} (n={len(records)} traces) ===")

    print("\n  Mode-level (count | n_traces):")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:2d} | {len(traces_with_mode[mode]):2d}  {mode}")

    print("\n  Category-level (count | n_traces):")
    for cat in ["Specification", "Inter-agent Misalignment", "Verification"]:
        print(f"    {cat_counts[cat]:2d} | {len(traces_with_cat[cat]):2d}  {cat}")


# --- Run on all 19 traces (with remap) ---
compute_prevalence(data, remap_to_mast, label="ALL 19 traces (remapped)")

# --- Sensitivity check: Round 3 + Generalizability only (no remap ambiguity) ---
stable_subset = [r for r in data if r["round"] in ("Round 3", "Generlazability")]
compute_prevalence(stable_subset, remap_to_mast, label="Round 3 + Generalizability (n=9)")