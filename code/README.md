# `code/`

Analysis pipeline for the released trial trajectories.

## Quick start

```sh
python3 -m pip install -r requirements.txt
python3 analyze.py        # reads ../data/*.jsonl, writes results_summary.json
python3 build_figures.py  # writes ../figures/*.{pdf,png} and ../paper/tables/*.tex
```

The output of `analyze.py` should match the per-cell numbers in
Table 1 of the paper. The output of `build_figures.py` reproduces
Figures 1-4.

## What's here

- `analyze.py` — loads the released trajectories, filters to MBPP-only,
  computes per-cell ERR / CSI, BCa bootstrap CIs on real-vs-scrambled,
  Wilcoxon signed-rank tests with Holm correction, and engagement
  rates. Writes `results_summary.json`.
- `build_figures.py` — uses the analysis output to generate the four
  paper figures (PDF + PNG) and the three LaTeX tables.
- `requirements.txt` — minimal Python dependencies.

## What's not here

The agent harness that produced the trajectories (the scaffold,
prompt templates, condition transformers, and Ollama client) is
**not** in this repo. We are confident the released trajectories
together with the analysis pipeline are sufficient to verify all
quantitative claims in the paper without re-running inference. If
you wish to reproduce the trajectories themselves, contact the
authors after de-anonymization for access to the harness.

## Determinism

Both scripts are fully deterministic given the released JSONLs:

- `analyze.py` uses fixed seeds for the BCa bootstrap (seed=42, 2,000
  resamples). Repeated runs produce identical CIs.
- `build_figures.py` produces deterministic PDFs (we set `pdf.fonttype=42`
  to embed editable fonts and disable any random visual jitter).

## License

MIT. See repository root.
