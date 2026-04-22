# LLM Judge Pipeline Critique

## Pipeline Overview

The LLM judge pipeline (`llm_judge_pipeline.ipynb`) is a **zero-shot, single-call
classification** system. It is not trained — it sends one prompt to OpenAI's `o1` model
per trace and parses the response with regex.

### How It Works

1. **Input**: Raw multi-agent execution trace (full conversation text)
2. **Prompt construction** (`openai_evaluator`): Concatenates instructions + trace +
   taxonomy definitions + examples into a single user message
3. **LLM call**: One call to `o1` with `temperature=1.0`
4. **Parsing** (`parse_responses`): Regex extracts binary yes/no for each of the 14
   failure modes from the free-text response
5. **Output**: Per-trace binary vector of 14 failure mode labels

Single judge, single pass, no ensemble, no structured output.

## Empirical Comparison: LLM vs Human Labels

We compared LLM labels (from `MAD_full_dataset.json`) against human IAA majority
labels (from `MAD_human_labelled_dataset.json`) on 13 overlapping traces.

### Aggregate Results

| Metric | Value |
|--------|-------|
| Label pairs compared | 158 |
| Human positive rate | 24.7% |
| LLM positive rate | 12.0% |
| Raw agreement | 70.9% |
| **Cohen's Kappa** | **0.054** |

Kappa of 0.054 = essentially no agreement beyond chance. The 70.9% raw agreement is
misleading because most labels are "no" — always predicting "no" yields ~75% accuracy.

### Confusion Matrix

|                  | LLM = no | LLM = yes |
|------------------|----------|-----------|
| **Human = no**   | 106      | 13        |
| **Human = yes**  | 33       | 6         |

- LLM missed **33 of 39** human-identified failures (recall = 15%)
- When LLM said "failure," it was wrong 13/19 times (precision = 32%)
- The LLM **under-reports failures by 2×** compared to humans

### Generalizability Round (cleanest comparison, same taxonomy)

Only 1 trace matched (Magentic/GAIA):
- Agreement: 11/14 modes (78.6%)
- Kappa: 0.276 ("fair")

### Per-Mode Breakdown

| Mode | Human yes | LLM yes | Agree | N  | Accuracy | Kappa  |
|------|-----------|---------|-------|----|----------|--------|
| 1.1  | 3         | 4       | 8     | 13 | 61.5%    | 0.030  |
| 1.2  | 2         | 0       | 11    | 13 | 84.6%    | 0.000  |
| 1.3  | 6         | 3       | 6     | 13 | 46.2%    | -0.123 |
| 1.4  | 2         | 1       | 12    | 13 | 92.3%    | 0.629  |
| 1.5  | 4         | 2       | 7     | 13 | 53.8%    | -0.258 |
| 2.1  | 2         | 1       | 10    | 13 | 76.9%    | -0.114 |
| 2.2  | 0         | 2       | 11    | 13 | 84.6%    | 0.000  |
| 2.3  | 2         | 2       | 11    | 13 | 84.6%    | 0.409  |
| 2.4  | 2         | 1       | 10    | 13 | 76.9%    | -0.114 |
| 2.5  | 6         | 0       | 7     | 13 | 53.8%    | 0.000  |
| 2.6  | 4         | 2       | 9     | 13 | 69.2%    | 0.161  |
| 3.1  | 0         | 0       | 5     | 5  | 100.0%   | NaN    |
| 3.2  | 3         | 1       | 3     | 5  | 60.0%    | 0.286  |
| 3.3  | 3         | 0       | 2     | 5  | 40.0%    | 0.000  |

Worst modes: 1.3 (Step Repetition), 1.5 (Termination Conditions), 2.5 (Ignored Input),
3.3 (No/Incorrect Verification) — all with negative or zero kappa.

### Caveats

1. **Mode mapping noise**: Rounds 1-3 used different taxonomy codes than the final 14.
   We built a best-effort mapping, but some old modes merged into one final code and
   3 old modes had no clear equivalent. This adds noise to the comparison.
2. **Possibly different trace text**: The full dataset was produced in bulk; human traces
   were hand-selected. The trace text fed to the LLM may differ from what humans read.
3. **Small sample**: Only 13 of 19 human traces matched the full dataset (AG2/GSM-Plus
   and most generalizability traces had no match).

## What They Did Right

1. **Using `o1` (reasoning model)** — appropriate for a task requiring deep trace
   comprehension and multi-step reasoning about agent behavior
2. **Providing both definitions and examples** — the prompt is well-grounded in the
   taxonomy with detailed definitions and real trace excerpts
3. **Binary per-mode labels** — clean, unambiguous output format that scales
4. **Checkpointing with pickle** — practical for long-running batch annotation over
   1000+ traces

## Design Problems

### 1. Single Judge, No Ensemble

The paper achieves κ=0.88 between humans using 3 annotators with majority vote. But
the LLM pipeline uses 1 call at `temperature=1.0`. A single stochastic sample is
unreliable. The human protocol already demonstrated that voting improves agreement —
the same principle applies to LLM calls.

**Fix**: Run 3-5 calls per trace, take majority vote. This directly mirrors the human
protocol and is straightforward to implement.

### 2. Prompt Is a Wall of Text

The entire prompt is one continuous concatenated string with no line breaks between
conceptual sections. The LLM receives instructions, trace, definitions, and examples
as a single run-on paragraph. This hurts comprehension, especially for long traces.

**Fix**: Use clear section delimiters (markdown headers, XML tags, or triple dashes).
Use a system message for instructions + definitions, and a user message for the trace.

### 3. Phantom Modes in the Example Answer

The example answer in the prompt includes modes `1.6` and `2.7` which don't exist in
the final 14-mode taxonomy. This creates confusion about the expected output schema:

```
"1.6 yes \n"
...
"2.7 no \n"
```

The model may try to output 17+ modes (matching the example) or get confused about
which numbering to use.

**Fix**: Update the example to use exactly the 14 final modes.

### 4. Trace Before Definitions

The prompt places the trace text *before* the taxonomy definitions:

```
"Here is the trace: \n"
f"{trace}"
"Also, here are the explanations (definitions)..."
```

For long traces (which can be hundreds of thousands of tokens), the model may form
opinions or lose attention before ever seeing what the failure modes are defined as.

**Fix**: Place definitions and examples first, then the trace. This gives the model
the analytical framework before the data.

### 5. Truncation Discards the Most Important Part

When traces exceed ~1M characters, the *end* is cut:

```python
full_trace_list[i] = full_trace_list[i][:1048570 - len(examples)]
```

In multi-agent traces, the termination, final output, and verification steps happen
at the end. These are exactly the parts needed to judge:
- 3.1 Premature Termination
- 3.2 Weak Verification
- 3.3 No or Incorrect Verification

Cutting the end makes it impossible to judge these modes correctly.

**Fix options**:
- Truncate from the middle (keep beginning and end)
- Summarize long traces in a first pass, then classify the summary
- Use a sliding-window approach that processes the trace in chunks
- Prioritize the first and last N tokens

### 6. Silent Failures in Parsing

The regex parser defaults to "no" when it can't find a mode in the response:

```python
if not found:
    print(f"Warning: Could not find mode {mode} in response {i}")
    failure_modes[mode].append(0)
```

Any formatting deviation from the expected output (which `o1` at temperature=1.0
can produce) becomes a silent false negative. This systematically biases the
dataset toward under-reporting failures.

**Fix**: Log and flag any response that doesn't fully parse. Retry those traces.
Or better, use structured output so parsing can't fail.

### 7. No Structured Output

The pipeline relies on regex to parse free-text responses. Modern OpenAI APIs
support JSON mode (`response_format={"type": "json_object"}`), function calling,
and structured outputs — all of which eliminate parsing failures.

**Fix**: Use JSON mode or function calling with a schema that defines all 14 modes
as boolean fields. This guarantees parseable output and removes the regex complexity.

### 8. Temperature=1.0 with Single Sample

`temperature=1.0` maximizes output diversity — useful for creative tasks, but
counterproductive for deterministic classification with a single sample. High
temperature means the same trace can get different labels on different runs.

**Fix**: If using single-call, set `temperature=0` for deterministic output.
If using ensemble, `temperature=0.7-1.0` is appropriate to get diverse samples
for voting.

### 9. No Per-Mode Chain of Thought

The prompt asks for a single free-text summary and then 14 binary answers. The
model must reason about all 14 modes simultaneously. For nuanced modes
(e.g., distinguishing 3.2 Weak Verification from 3.3 No Verification), dedicated
reasoning per mode would improve accuracy.

**Fix**: Ask the model to provide a brief justification for each yes/no answer.
This forces it to reason about each mode individually and provides an audit trail.
Alternatively, use a two-stage approach: first generate a structured analysis, then
classify.

### 10. No Calibration

The prompt was not iteratively refined against traces with known labels. The 19
human-annotated traces (or even just the 4 generalizability traces) could serve as
a calibration set to tune the prompt until the LLM's output better matches human
consensus.

**Fix**: Use the IAA traces as a development set. Iterate on the prompt, examples,
and instructions until agreement with human majority labels improves. Report the
final kappa on a held-out set.

## Recommended Improved Pipeline

```
1. LOAD definitions + examples + trace

2. CONSTRUCT PROMPT:
   - System message: role definition, taxonomy definitions, examples, output schema
   - User message: trace text (truncated from middle if needed, keeping start + end)

3. CALL LLM × 3-5 times:
   - Model: o1 or GPT-4o
   - temperature: 0.7 (for ensemble diversity)
   - response_format: JSON with per-mode {label: bool, reasoning: string}

4. AGGREGATE:
   - Majority vote across calls per mode
   - Flag traces where calls disagree strongly (split votes)

5. VALIDATE:
   - Parse check: 100% of responses must contain all 14 modes (guaranteed by JSON mode)
   - Calibration: compare against IAA traces; report kappa before full deployment
   - Flagging: traces with many split votes go to human review
```

### Estimated Cost Impact

Running 3 calls instead of 1 triples cost, but:
- `o1` pricing at ~$15/1M input tokens for 1242 traces is already substantial
- Using GPT-4o (cheaper, also capable) with 3 calls may cost less than 1 o1 call
- Structured output eliminates retry costs from parse failures
- Better accuracy means less manual correction downstream

## Summary

The pipeline is the simplest possible LLM-as-a-judge: one call, one parse, done.
It works as a first prototype but has systematic issues that cause it to
**under-report failures by 2× relative to humans** with near-zero kappa agreement.
The most impactful improvements would be: (1) ensemble voting with 3+ calls,
(2) structured JSON output, (3) fixing the prompt structure and phantom modes,
and (4) calibrating against the existing human-labelled data before deploying at
scale.
