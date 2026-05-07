# `data/`

This directory holds the released trial trajectories.

**These files are not bundled in the source ZIP** — they total ~22 MB
and live in the GitHub repo, which is the canonical source. Each file
is line-delimited JSON (UTF-8); each line is one agent trial.

| File | Trials | Description |
|---|---|---|
| `20260428T081951Z-06f3bf77_results.jsonl` | 1,800 | Original sweep: Mistral-7B + Qwen2.5-14B + Qwen3-14B at 1,024-token budget. MBPP and GAIA. |
| `20260428T174000Z-6e3d244c_results.jsonl` | 400 | Qwen3-14B replication at 8,192-token budget. MBPP only. |
| `20260428T214430Z-759dec7b_results.jsonl` | 400 | Llama-3.1-8B at 1,024-token budget. MBPP only. |

Run `python3 ../code/analyze.py` from this directory after downloading
the files to reproduce the paper's per-cell statistics.

See the top-level README for the full record schema. See
`../croissant.json` for machine-readable metadata.
