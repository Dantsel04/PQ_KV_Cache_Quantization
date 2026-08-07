# AGENTS.md

Stable instructions for Codex agents working in this repository.

## Project Overview

This project develops a Google Colab workflow for Qwen3-8B KV-cache product quantization (PQ). The notebook downloads/loads Qwen3-8B, extracts key/value calibration vectors, trains grouped PQ codebooks, and evaluates baseline vs dynamic/static outlier-preserving PQ using perplexity and LongBench.

## Repository Structure

- `KV_Cache.py`: primary editable source. This is a Jupytext percent-format export of the notebook.
- `KV_Cache.ipynb`: Colab notebook artifact synced from `KV_Cache.py`.
- `NOTEBOOK_MAP.md`: map of notebook sections, functions/classes, inputs/outputs, and dependencies.
- `AGENTS.md`: durable Codex rules.
- `STATUS.md`: current verified project state.
- `PLAN.md`: active milestone and checklist.

There are no package source directories, test directories, or requirements files currently present.

## Required Workflow

1. Read `STATUS.md` and `PLAN.md` before starting work.
2. Edit `KV_Cache.py`, not notebook JSON directly.
3. After changes, run `py -m jupytext --sync KV_Cache.ipynb`.
4. Commit and push changes to GitHub when asked or when a completed milestone should be shared.
5. Colab is the remote GPU execution environment. When a run is requested, Colab pulls the latest GitHub version and executes `KV_Cache.ipynb`.
6. Read the shared Colab bridge result file before deciding what to change next.
7. Never assume a GPU run succeeded unless the Colab result explicitly confirms it.
8. Update `STATUS.md` and `PLAN.md` after meaningful project changes.

## Colab Bridge

- Local Google Drive bridge: `G:\My Drive\PQ_agent`.
- Do not use browser automation for Colab.
- Write commands atomically to `command.json` with schema:
  `{"id": "<unique-id>", "command": ["program", "arg1", "..."]}`.
- The controller writes `result.json` with `id`, `returncode`, `stdout`, and
  `stderr`.
- Ignore every result whose `id` does not exactly match the active command.
- Use a hidden local polling process to watch `result.json`; polling must not consume
  agent turns. A normal watcher cannot wake a non-goal Codex chat, so use `/goal` for
  autonomous multi-stage runs.
- Run the pipeline behind explicit gates: model setup, calibration extraction,
  codebook training, then LongBench-E evaluation. Never submit a later stage until
  the matching prior result returns code 0 and its expected artifacts are verified.
- The synced notebook currently has no reliable cell IDs. Select pipeline cells by a
  unique source marker, not by Jupytext/Colab cell ID.

Do not create per-chat markdown logs. Use Git history for detailed change tracking, `STATUS.md` for current state, and `PLAN.md` for next steps.

## Build, Run, and Test Commands

- Sync notebook: `py -m jupytext --sync KV_Cache.ipynb`
- Python syntax check: `py -m py_compile KV_Cache.py`
- Local full execution: not recommended; most cells assume Colab paths, GPU, Hugging Face downloads, and Google Drive.
- GPU validation: run through the Colab controller notebook and inspect the bridge result file.

No formal unit test command is currently defined.

## Coding Conventions

- Keep notebook cells coherent and runnable independently where practical.
- Preserve Jupytext percent cell markers.
- Prefer small, reviewable edits to one notebook section at a time.
- Keep paths and model names explicit; avoid silently switching model families.
- Mark experimental modes clearly, especially contaminated calibration modes.
- Prefer deterministic seeds where data sampling or clustering is involved.
- Keep generated data, model weights, codebooks, and run outputs out of Git unless explicitly requested.

## Architecture Rules

- Treat `KV_Cache.py` as the source of truth and `KV_Cache.ipynb` as the synced execution form.
- Keep Qwen3-8B, PPL, LongBench, calibration extraction, and codebook training/evaluation paths aligned.
- Do not mix codebook maps across codebook directories; each trained codebook set must use its own `head_to_codebook_map.json`.
- Dynamic outlier evaluation must run before static-mask evaluation because static masks are derived from dynamic observations.
- LongBench calibration provenance matters. Do not report contaminated calibration results as final benchmark results.

## Do-Not-Change Rules

- Do not edit `KV_Cache.ipynb` directly.
- Do not reorganize the repository unless explicitly asked.
- Do not remove `NOTEBOOK_MAP.md`, `STATUS.md`, `PLAN.md`, or `AGENTS.md`.
- Do not commit downloaded models, calibration vectors, codebooks, result CSVs, bridge outputs, or other generated artifacts unless explicitly requested.
- Do not assume local CPU execution is equivalent to Colab GPU execution.

## Dependency Rules

- Notebook dependencies are currently installed ad hoc inside cells.
- If dependencies become stable, prefer documenting or adding a requirements file in a dedicated task.
- Do not add new dependencies without noting why they are needed in `STATUS.md` or the relevant plan.

## Verification Expectations

For source-only edits:
- Run `py -m py_compile KV_Cache.py`.
- Run `py -m jupytext --sync KV_Cache.ipynb` if notebook sync is required.

For workflow or GPU behavior changes:
- Sync the notebook.
- Run the relevant Colab command through the controller.
- Read and summarize the Colab bridge result before making further changes.

If a verification command cannot be run, record that clearly in the final response and update `STATUS.md` if it affects current state.
