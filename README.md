# MAS Failure Taxonomy — Pipeline & Taxonomy Evaluation

Based on the [MAST paper](https://arxiv.org/abs/2503.09561) (Multi-Agent System failure Taxonomy). Dataset: [mcemri/MAD on HuggingFace](https://huggingface.co/datasets/mcemri/MAD).

## Goal

Evaluate MAST's LLM-judge pipeline and 14-mode failure taxonomy for multi-agent systems. Investigate whether parts of the taxonomy can be detected on-the-fly during agent execution rather than post-hoc.

## Findings so far

- **Data quality issues**: human-labeled subset is small (19 traces) with limited MAS/benchmark coverage; inter-annotator agreement varies significantly across failure modes.
- **Pipeline quality issues**: the original LLM-judge pipeline shows low agreement with human labels on several failure modes (see `old_pipeline/`).
- **Taxonomy observations**: some modes are rarely observed or hard to distinguish; others (e.g. step repetition, premature termination) are strong candidates for real-time detection.

## Structure

```
old_pipeline/          # Original MAST pipeline reproduction & critique
  tests/               # LLM-vs-human comparison, evaluation results
  taxonomy_definitions_examples/
data_exploration/      # Dataset analysis scripts & notes
new_data/              # Extracted human-labeled traces, annotations
```
