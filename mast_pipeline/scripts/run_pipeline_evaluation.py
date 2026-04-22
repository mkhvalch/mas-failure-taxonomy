"""
Run the LLM judge pipeline (same prompt as llm_judge_pipeline.ipynb) on
IAA-labelled traces and compare against human majority labels.

Loads API config from .env:
    OPENAI_API_BASE, OPENAI_API_KEY, MODEL

Usage:
    python run_pipeline_evaluation.py                   # all 9 traces
    python run_pipeline_evaluation.py --max-traces 4    # first 4 (generalizability only)
    python run_pipeline_evaluation.py --skip-long       # skip traces > 200K chars
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

FINAL_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

MODE_NAMES = {
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


def build_prompt(trace: str, definitions: str, examples: str) -> str:
    """Exact same prompt as the notebook's openai_evaluator."""
    return (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here is the trace: \n"
        f"{trace}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{examples}"
    )


def parse_response(response: str) -> dict[str, int]:
    """Parse LLM response to extract yes/no for each failure mode.
    Same logic as the notebook's parse_responses but for a single response."""
    result = {}
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]

    for mode in FINAL_MODES:
        patterns = [
            rf"{mode}\s*[:]\s*(yes|no)",
            rf"{mode}\s+(yes|no)",
            rf"{mode}\s*\n\s*(yes|no)",
            rf"(?:C\.)?{mode}.*?(yes|no)",
        ]
        found = False
        for pat in patterns:
            match = re.search(pat, cleaned, re.IGNORECASE | re.DOTALL)
            if match:
                result[mode] = 1 if match.group(1).lower() == "yes" else 0
                found = True
                break
        if not found:
            print(f"  WARNING: Could not parse mode {mode}, defaulting to 0")
            result[mode] = 0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input traces JSON (default: iaa_traces_with_labels.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: tests/pipeline_evaluation_results.json)")
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--skip-long", action="store_true",
                        help="Skip traces > 200K chars")
    parser.add_argument("--max-trace-chars", type=int, default=800000,
                        help="Truncate trace text to this many chars")
    args = parser.parse_args()

    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("MODEL", "o1")

    if not api_key:
        print("ERROR: No OPENAI_API_KEY found in .env")
        return

    print(f"API base: {api_base}")
    print(f"Model:    {model}")
    print(f"Key:      {api_key[:8]}...")

    client = OpenAI(api_key=api_key, base_url=api_base)

    definitions = open("taxonomy_definitions_examples/definitions.txt").read()
    examples = open("taxonomy_definitions_examples/examples.txt").read()

    input_path = args.input or "iaa_traces_with_labels.json"
    with open(input_path) as f:
        traces = json.load(f)
    print(f"Input:    {input_path}")

    if args.skip_long:
        before = len(traces)
        traces = [t for t in traces if t["trace_char_len"] <= 200_000]
        print(f"Filtered {before} → {len(traces)} traces (skipped > 200K chars)")

    if args.max_traces:
        traces = traces[: args.max_traces]

    print(f"\nWill evaluate {len(traces)} traces\n")

    results = []
    for i, trace_rec in enumerate(traces):
        idx = trace_rec["index"]
        mas = trace_rec["mas_name"]
        bench = trace_rec["benchmark_name"]
        rnd = trace_rec["round"]
        human = trace_rec["human_labels"]
        trace_text = trace_rec["trace_text"]

        # Truncate if needed (same as notebook: cut from end)
        max_chars = args.max_trace_chars
        if len(trace_text) + len(examples) > max_chars:
            trace_text = trace_text[: max_chars - len(examples)]
            print(f"  (trace truncated to {len(trace_text):,} chars)")

        print(f"[{i+1}/{len(traces)}] idx={idx} round={rnd} "
              f"mas={mas} bench={bench} chars={len(trace_text):,}")

        prompt = build_prompt(trace_text, definitions, examples)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_response = resp.choices[0].message.content
            elapsed = time.time() - t0
            print(f"  Response received ({elapsed:.1f}s, "
                  f"{len(raw_response):,} chars)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "index": idx, "round": rnd, "mas": mas, "bench": bench,
                "error": str(e),
            })
            continue

        llm_labels = parse_response(raw_response)

        agree = sum(1 for m in FINAL_MODES if human.get(m, 0) == llm_labels.get(m, 0))
        h_yes = sum(human.get(m, 0) for m in FINAL_MODES)
        l_yes = sum(llm_labels.get(m, 0) for m in FINAL_MODES)

        print(f"  Human: {h_yes} failures | LLM: {l_yes} failures | "
              f"Agree: {agree}/14 ({agree/14*100:.0f}%)")

        # Show disagreements
        for m in FINAL_MODES:
            h = human.get(m, 0)
            l = llm_labels.get(m, 0)
            if h != l:
                direction = "MISSED" if h == 1 else "FALSE POS"
                print(f"    {direction}: {m} {MODE_NAMES[m]} "
                      f"(human={h}, llm={l})")

        results.append({
            "index": idx, "round": rnd, "mas": mas, "bench": bench,
            "human_labels": human,
            "llm_labels": llm_labels,
            "raw_response": raw_response,
            "agree": agree,
        })

    save_path = args.output or "tests/pipeline_evaluation_results.json"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {save_path}")

    # Aggregate metrics
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No successful evaluations!")
        return

    all_h, all_l = [], []
    for r in valid:
        for m in FINAL_MODES:
            all_h.append(r["human_labels"].get(m, 0))
            all_l.append(r["llm_labels"].get(m, 0))

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(valid)} traces, {len(all_h)} label pairs)")
    print("=" * 70)
    agree_total = sum(h == l for h, l in zip(all_h, all_l))
    print(f"Human positives:  {sum(all_h)} ({sum(all_h)/len(all_h)*100:.1f}%)")
    print(f"LLM positives:    {sum(all_l)} ({sum(all_l)/len(all_l)*100:.1f}%)")
    print(f"Raw agreement:    {agree_total}/{len(all_h)} ({agree_total/len(all_h)*100:.1f}%)")

    kappa = cohen_kappa_score(all_h, all_l)
    print(f"Cohen's Kappa:    {kappa:.3f}")

    print("\nConfusion matrix (rows=human, cols=LLM):")
    cm = confusion_matrix(all_h, all_l, labels=[0, 1])
    print(f"                  LLM=no  LLM=yes")
    print(f"  Human=no        {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Human=yes       {cm[1][0]:6d}  {cm[1][1]:6d}")

    print("\nClassification report:")
    print(classification_report(all_h, all_l,
                                target_names=["no failure", "failure"],
                                zero_division=0))

    # Per-mode
    print("=" * 70)
    print("PER-MODE BREAKDOWN")
    print("=" * 70)
    print(f"{'Mode':<6s} {'Name':<35s} {'H':>3s} {'L':>3s} {'Agr':>3s} {'N':>3s} {'Acc':>5s}")
    for mode in FINAL_MODES:
        mh = [r["human_labels"].get(mode, 0) for r in valid]
        ml = [r["llm_labels"].get(mode, 0) for r in valid]
        n = len(mh)
        ag = sum(a == b for a, b in zip(mh, ml))
        print(f"{mode:<6s} {MODE_NAMES[mode]:<35s} {sum(mh):3d} {sum(ml):3d} "
              f"{ag:3d} {n:3d} {ag/n*100:5.1f}%")

    # Per-trace summary
    print("\n" + "=" * 70)
    print("PER-TRACE SUMMARY")
    print("=" * 70)
    for r in valid:
        print(f"  idx={r['index']:2d}  round={r['round']:<16s} "
              f"mas={r['mas']:<15s} agree={r['agree']:2d}/14 "
              f"({r['agree']/14*100:.0f}%)")


if __name__ == "__main__":
    main()
