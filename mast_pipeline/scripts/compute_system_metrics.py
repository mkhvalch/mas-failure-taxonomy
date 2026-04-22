"""System-level MAS metrics on the 19 IAA traces.

Goes beyond MAST's classification metrics (accuracy, P/R/F1, kappa) and adds
system-level signals inspired by:

  Kim et al., "Towards a Science of Scaling Agent Systems" (arXiv:2512.08296)
  github.com/ybkim95/agent-scaling/tree/main/scripts

We cannot compute the paper's architecture-level error amplification
A_e = E_MAS / E_SAS because we have no single-agent baseline per task. Instead
we compute trace-level proxies that are recoverable from the raw MAST trace
text itself:

  1. EFFICIENCY (performance / cost)
     task_success / approx_tokens  (higher = cheaper solve)
     approx_tokens := trace_char_len / 4  (GPT tokenizer ~ 4 chars/tok)

  2. COORDINATION OVERHEAD
     turn_count     : number of distinguishable agent turns (heuristic)
     unique_agents  : distinct speaker roles observed
     chars_per_turn : trace_char_len / max(turn_count, 1)   (lower = more turns
                                                             per unit of work)

  3. ERROR PROPAGATION
     error_density  : error-keyword hits per 1K chars
     error_late_ratio : hits in last third / hits in first third
                        (> 1 means errors compound late, paper's Table 4 idea)
     mast_late_modes: how many MAST-late modes (3.1, 3.2, 3.3) humans flagged
                      (these are late-trace failures by definition)

  4. REDUNDANCY
     redundancy_score: fraction of 8-gram tokens that appear more than once
                       (simple, format-agnostic proxy for repeated actions)
     mast_step_repetition: whether humans flagged mode 1.3

All metrics are documented with a one-line reasoning in the output JSON.

Output:
  mast_pipeline/outputs/all19/system_metrics_all19.json
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRACES_PATH = os.path.join(ROOT, "new_data", "round[1,3]", "final_iaa_19_traces.json")
CONF_PATH   = os.path.join(ROOT, "new_data", "round[1,3]", "iaa_all19_confidence.json")
LLM_PATH    = os.path.join(ROOT, "mast_pipeline", "outputs", "all19",
                           "pipeline_evaluation_results_all19.json")
OUT_PATH    = os.path.join(ROOT, "mast_pipeline", "outputs", "all19",
                           "system_metrics_all19.json")

# Keywords that almost always indicate execution failure in an agent trace
# across ChatDev/AG2/HyperAgent/MetaGPT/GAIA formats. Kept deliberately narrow
# so false positives stay rare on non-error English sentences.
ERROR_PATTERNS = [
    r"\btraceback\b",
    r"\berror\b",
    r"\bexception\b",
    r"\bfailed?\b",
    r"\bretrying\b",
    r"\brollback\b",
    r"\bstack trace\b",
    r"\bassertionerror\b",
    r"\btypeerror\b",
    r"\bvalueerror\b",
    r"\btimeout\b",
    r"cannot\s+find",
    r"not\s+found",
]
ERROR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# Turn / role heuristics. The MAD traces use heterogeneous formats; we detect
# what we can and note format in the output for transparency.
TURN_PATTERNS = [
    (r'"role"\s*:\s*"(\w+)"',            "json-role"),       # AG2
    (r'role=\'(\w+)\'',                  "repr-role"),
    (r"\|\s*(INFO|ERROR|WARNING)\s*\|",  "metagpt-log"),     # MetaGPT
    (r"\n(?:Agent|User|Assistant|System|"
     r"Alice|Bob|Charlie|Dave|"
     r"Planner|Coder|Tester|Reviewer|"
     r"Manager|ProductManager|Architect|Engineer|"
     r"CEO|CTO|CPO|Programmer)"
     r"\s*[:(]",                         "named-agent"),
    (r"\n#+\s*(?:Step|Turn|Round|Phase)\s+\d+", "step-header"),
]
ROLE_RE = re.compile(r'"role"\s*:\s*"(\w+)"', re.IGNORECASE)
NAMED_ROLE_RE = re.compile(
    r"\n(Alice|Bob|Charlie|Dave|Planner|Coder|Tester|Reviewer|"
    r"Manager|ProductManager|Architect|Engineer|CEO|CTO|CPO|Programmer|"
    r"User|Assistant|System|Agent)[\s(:]",
    re.IGNORECASE,
)

# Task-success heuristics, in priority order. Conservative: returns None if
# we can't find a clear signal, rather than guessing.
SUCCESS_PATTERNS = [
    (re.compile(r'"correct"\s*:\s*(true|false)', re.I),           lambda m: m.group(1).lower() == "true"),
    (re.compile(r'"success"\s*:\s*(true|false)', re.I),           lambda m: m.group(1).lower() == "true"),
    (re.compile(r'"is_correct"\s*:\s*(true|false)', re.I),        lambda m: m.group(1).lower() == "true"),
    (re.compile(r'"final_result"\s*:\s*"?(pass|fail|passed|failed)"?', re.I),
                                                                   lambda m: m.group(1).lower().startswith("pass")),
    (re.compile(r'\bALL TESTS PASSED\b', re.I),                   lambda m: True),
    (re.compile(r'\b\d+ tests? passed\b, 0 failed', re.I),        lambda m: True),
    (re.compile(r'\btask (?:completed|solved) successfully\b', re.I),
                                                                   lambda m: True),
]

MAST_LATE_MODES = {"3.1", "3.2", "3.3"}   # termination / verification failures


def approx_tokens(text: str) -> int:
    """~4 chars/token for GPT-family tokenizers."""
    return max(1, len(text) // 4)


def count_turns(text: str) -> tuple[int, str]:
    """Return (turn_count, format_used)."""
    # Priority 1: JSON "role" fields (AG2 trajectories)
    roles = ROLE_RE.findall(text)
    if len(roles) >= 3:
        return len(roles), "json-role"
    # Priority 2: named agents (MetaGPT, ChatDev)
    named = NAMED_ROLE_RE.findall(text)
    if len(named) >= 3:
        return len(named), "named-agent"
    # Priority 3: MetaGPT log lines
    log_lines = re.findall(r"\|\s*(INFO|ERROR|WARNING)\s*\|", text)
    if len(log_lines) >= 3:
        return len(log_lines), "metagpt-log"
    # Fallback: number of newlines / 5 (every ~5 lines ~= 1 turn)
    nlines = text.count("\n")
    return max(1, nlines // 5), "line-fallback"


def unique_agents(text: str) -> int:
    named = set(m.lower() for m in NAMED_ROLE_RE.findall(text))
    roles = set(m.lower() for m in ROLE_RE.findall(text))
    return max(1, len(named | roles))


def error_density_and_late(text: str) -> tuple[float, float, int]:
    """Return (errors_per_1K_chars, late_ratio, total_errors).

    late_ratio = errors in last third / errors in first third, capped to 99.
    """
    total_hits = len(ERROR_RE.findall(text))
    density = total_hits / (len(text) / 1000)

    third = len(text) // 3
    if third == 0 or total_hits == 0:
        return density, 0.0, total_hits
    first = len(ERROR_RE.findall(text[:third]))
    last  = len(ERROR_RE.findall(text[-third:]))
    if first == 0:
        # avoid inf: treat as "99x" when we have late errors but no early ones
        late_ratio = 99.0 if last > 0 else 0.0
    else:
        late_ratio = min(99.0, last / first)
    return density, late_ratio, total_hits


def redundancy_score(text: str, n: int = 8) -> float:
    """Fraction of word-n-grams that repeat.

    A value in [0, 1]. 0 = every n-gram unique, 1 = fully repetitive.
    Using words (not chars) keeps this robust across trace formats.
    """
    words = re.findall(r"\S+", text)
    if len(words) < n + 1:
        return 0.0
    from collections import Counter
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    n_repeat = sum(c for c in counts.values() if c > 1)
    return round(n_repeat / len(grams), 4)


def detect_task_success(text: str) -> bool | None:
    for pat, extractor in SUCCESS_PATTERNS:
        m = pat.search(text)
        if m:
            return extractor(m)
    return None


def compute_for_trace(trace_rec: dict,
                      llm_labels: dict[str, int] | None) -> dict:
    text = trace_rec["trace_text"]
    chars = len(text)
    tokens = approx_tokens(text)
    turns, turn_fmt = count_turns(text)
    agents = unique_agents(text)
    err_density, err_late_ratio, err_total = error_density_and_late(text)
    red = redundancy_score(text)
    success = detect_task_success(text)

    # Efficiency: solve per 1K tokens. If success unknown, return None.
    if success is None:
        efficiency = None
    else:
        efficiency = round((1 if success else 0) / (tokens / 1000), 4)

    # Human MAST flags → link back to system metrics
    human = trace_rec["human_labels"]
    mast_step_rep   = bool(human.get("1.3", 0))                         # redundancy
    mast_late_modes = sum(human.get(m, 0) for m in MAST_LATE_MODES)     # error propagation
    mast_coord_fails = sum(human.get(m, 0) for m in                     # coordination
                           ("2.1", "2.4", "2.5"))
    mast_disobey    = sum(human.get(m, 0) for m in ("1.1", "1.2"))      # spec compliance

    llm_step_rep = bool((llm_labels or {}).get("1.3", 0))

    return {
        "index":           trace_rec["index"],
        "round":           trace_rec["round"],
        "mas":             trace_rec["mas_name"],
        "bench":           trace_rec["benchmark_name"],
        "trace_chars":     chars,
        "approx_tokens":   tokens,

        # 1. Efficiency
        "task_success":    success,
        "efficiency_per_1Ktok": efficiency,

        # 2. Coordination
        "turn_count":      turns,
        "turn_format":     turn_fmt,
        "unique_agents":   agents,
        "chars_per_turn":  round(chars / max(turns, 1), 1),

        # 3. Error propagation
        "error_density_per_1Kchar": round(err_density, 3),
        "error_late_ratio":         round(err_late_ratio, 2),
        "error_hits_total":         err_total,
        "mast_late_modes_human":    mast_late_modes,

        # 4. Redundancy
        "redundancy_8gram":         red,
        "mast_step_repetition_human": mast_step_rep,
        "mast_step_repetition_llm":   llm_step_rep,

        # Bonus context from MAST labels
        "mast_coord_failures_human": mast_coord_fails,
        "mast_disobey_human":        mast_disobey,
    }


def main() -> None:
    with open(TRACES_PATH) as f:
        traces = json.load(f)
    with open(LLM_PATH) as f:
        llm_results = json.load(f)

    llm_by_idx = {r["index"]: r.get("llm_labels", {}) for r in llm_results}

    rows = [compute_for_trace(t, llm_by_idx.get(t["index"])) for t in traces]

    def rk(r):
        order = {"Round 1": 1, "Round 2": 2, "Round 3": 3, "Generlazability": 4}
        return (order.get(r["round"], 9), r["index"])
    rows.sort(key=rk)

    # Print summary table
    print(f"{'idx':>3} {'round':<16} {'mas':<11} {'bench':<17} "
          f"{'turns':>6} {'ag':>3} {'c/turn':>8} {'err/K':>7} "
          f"{'latex':>6} {'red':>6} {'ok':>3} {'eff':>7}")
    print("-" * 112)
    for r in rows:
        ok = "?" if r["task_success"] is None else ("yes" if r["task_success"] else "no")
        eff = r["efficiency_per_1Ktok"]
        eff_s = f"{eff:.4f}" if eff is not None else "   —  "
        print(f"{r['index']:>3} {r['round']:<16} {r['mas']:<11} {r['bench']:<17} "
              f"{r['turn_count']:>6d} {r['unique_agents']:>3d} "
              f"{r['chars_per_turn']:>8.0f} {r['error_density_per_1Kchar']:>7.2f} "
              f"{r['error_late_ratio']:>6.2f} {r['redundancy_8gram']:>6.3f} "
              f"{ok:>3s} {eff_s:>7}")

    # Aggregates
    print("\nAggregates across 19 traces:")
    print(f"  mean turns:              {sum(r['turn_count'] for r in rows)/len(rows):.1f}")
    print(f"  mean chars/turn:         {sum(r['chars_per_turn'] for r in rows)/len(rows):.0f}")
    print(f"  mean error density:      {sum(r['error_density_per_1Kchar'] for r in rows)/len(rows):.2f}")
    print(f"  mean error late ratio:   {sum(r['error_late_ratio'] for r in rows)/len(rows):.2f}")
    print(f"  mean 8-gram redundancy:  {sum(r['redundancy_8gram'] for r in rows)/len(rows):.3f}")
    n_success = sum(1 for r in rows if r["task_success"] is True)
    n_fail    = sum(1 for r in rows if r["task_success"] is False)
    n_unk     = sum(1 for r in rows if r["task_success"] is None)
    print(f"  task outcomes:           {n_success} success, {n_fail} fail, {n_unk} unknown")

    # Link system metrics to MAST labels (correlation sanity)
    print("\nSanity: MAST labels ↔ system signals")
    # 1.3 step repetition vs our 8-gram redundancy
    red_yes = [r["redundancy_8gram"] for r in rows if r["mast_step_repetition_human"]]
    red_no  = [r["redundancy_8gram"] for r in rows if not r["mast_step_repetition_human"]]
    print(f"  8-gram redundancy when human flagged 1.3 (Step Repetition): "
          f"{sum(red_yes)/max(len(red_yes),1):.3f} (n={len(red_yes)}) "
          f"vs. no-flag: {sum(red_no)/max(len(red_no),1):.3f} (n={len(red_no)})")
    # late modes vs late error ratio
    late_lbl = [r["error_late_ratio"] for r in rows if r["mast_late_modes_human"] >= 1]
    late_no  = [r["error_late_ratio"] for r in rows if r["mast_late_modes_human"] == 0]
    print(f"  error_late_ratio when human flagged 3.1/3.2/3.3: "
          f"{sum(late_lbl)/max(len(late_lbl),1):.2f} (n={len(late_lbl)}) "
          f"vs. no-flag: {sum(late_no)/max(len(late_no),1):.2f} (n={len(late_no)})")

    out = {
        "n_traces": len(rows),
        "metric_definitions": {
            "efficiency_per_1Ktok": "task_success (0/1) / (approx_tokens / 1000)",
            "approx_tokens": "trace_char_len / 4 (GPT-family tokenizer ratio)",
            "turn_count": "agent turns detected (format-aware heuristic)",
            "chars_per_turn": "trace_chars / turn_count",
            "error_density_per_1Kchar": "hits of ERROR_PATTERNS per 1K chars",
            "error_late_ratio": "err_hits(last third) / err_hits(first third), capped 99",
            "redundancy_8gram": "fraction of 8-word n-grams that repeat",
            "mast_late_modes_human": "sum of human labels for 3.1, 3.2, 3.3",
        },
        "error_patterns": ERROR_PATTERNS,
        "per_trace": rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
