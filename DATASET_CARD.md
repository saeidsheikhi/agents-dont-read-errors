# Dataset Card: Agents-Don't-Read-Errors Trajectories

## Dataset summary

Trial trajectories from a counterfactual evaluation of error-message
reading in tool-using LLM agents. 2,600 trials across four
open-weight models on the MBPP-sanitized programming benchmark, under
four error-content conditions.

Use this dataset if you are studying:

- Agent error recovery and how it depends on error content.
- Engagement vs. reading trade-offs in reasoning-tuned models.
- Behavior of open-weight LLM agents on a controlled coding benchmark.
- Methods for counterfactual evaluation of language-model agents.

## Quick stats

| | |
|---|---|
| Total trials | 2,600 |
| Trials used in paper's primary analysis | 1,600 (MBPP only, 4 models × 4 conditions × 100 tasks) |
| Models | Mistral-7B, Llama-3.1-8B, Qwen2.5-14B, Qwen3-14B |
| Conditions | Real / Scrambled / Misleading / Empty |
| Benchmark | MBPP-sanitized (and GAIA-validation, excluded from primary analysis; see paper Appendix B) |
| Wall-clock time | ~12 hours total |
| Format | Line-delimited JSON (UTF-8) |

## How the dataset was collected

1. For each (model, condition, task) cell, a single trial was run via
   the agent harness (a minimal ReAct-style scaffold using each
   model's native function-calling API exposed by Ollama).
2. The agent received a coding task description and a single tool
   `submit_code` for evaluating candidate solutions.
3. When the tool returned an error, an error-substitution layer
   replaced the message body according to the cell's condition before
   the agent saw it.
4. After up to 10 turns or 3 consecutive errors, the trial ended; the
   agent's final answer was scored against the canonical MBPP unit
   tests.

All inference was deterministic at `temperature=0`. Trajectories
should reproduce bit-identically given the same model checkpoint and
prompt template.

## Recommended uses

- **Reproducing or extending the paper**: see `README.md` for the
  analysis pipeline.
- **Evaluating new error-handling scaffolds**: a proposed scaffold
  can be evaluated by running it on MBPP-100 and computing the
  Content Sensitivity Index (CSI) defined in the paper. This dataset
  provides four baseline points to compare against.
- **Studying disengagement in reasoning-tuned models**: the Qwen3-14B
  trajectories include 64% of trials where the model emitted extended
  reasoning without calling any tool. Researchers studying
  "thinking without acting" failures may find this corpus useful.

## Out-of-distribution uses

This dataset is **not appropriate** for:

- Evaluating proprietary frontier models (we tested only open-weight
  models up to 14B).
- Drawing conclusions about agent behavior on long-context or
  domain-specific code (MBPP problems are short, ~10-15 lines).
- Training (the dataset is too small and specialized; it is intended
  for evaluation, not training).

## Limitations and biases

- **Model coverage**: only four open-weight models, all ≤14B.
  Conclusions may not transfer to larger or proprietary models.
- **Benchmark coverage**: only MBPP-sanitized. GAIA was collected
  but excluded due to confounding tool-name hallucination (see paper
  Appendix B).
- **Sample size on Qwen3**: due to disengagement, only ~36 paired
  tasks per cell contribute to the headline Qwen3 statistic.
- **Scrambling protocol**: the POS-class scrambler preserves token
  length and structural cues. It is possible that some semantic
  signal survives scrambling (e.g., from individual tokens like
  `IndexError` appearing within scrambled output). The paper's
  scrambler audit reports a 0% identity rate but cannot quantify
  residual semantic information.

## Personal data

The dataset contains no personal data. It contains only model
outputs (Python code, error messages, chain-of-thought) on public
programming benchmark inputs.

## Maintenance

This dataset is a fixed snapshot corresponding to the paper's
experimental run. We do not plan ongoing updates. Errata, if any,
will be posted as a tagged release on the repository.

## License

CC-BY-4.0 with the model-output-redistribution caveats noted in the
top-level README.
