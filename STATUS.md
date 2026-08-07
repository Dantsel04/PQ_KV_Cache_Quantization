# STATUS.md

Current verified project state.

## Working State

The repository contains a Jupytext-managed Colab notebook workflow for Qwen3-8B KV-cache PQ experiments. `KV_Cache.py` is the editable source and `KV_Cache.ipynb` is the synced notebook artifact used by Colab.

Current local verification:
- `py -m py_compile KV_Cache.py`: passed for the LongBench-E optimization revision.
- `py -m jupytext --sync KV_Cache.ipynb`: passed; repeat after subsequent source edits.
- Static AST/config validation: passed for held-out calibration, 4096-token alignment, versioned artifacts, full LongBench-E mode, and matching K/V 64-bank × 128-codeword settings.
- Colab GPU run: pending. Browser control is unavailable in the current Codex tool runtime and no Colab bridge/controller files exist in the repository.

## Repository Inventory

- `KV_Cache.py`: main notebook source, Python/Jupytext percent format.
- `KV_Cache.ipynb`: notebook artifact for Colab execution.
- `NOTEBOOK_MAP.md`: current map of notebook sections and dependencies.
- `AGENTS.md`: Codex workflow rules.
- `STATUS.md`: this current-state file.
- `PLAN.md`: active milestone plan.

No package directories, test directories, requirements files, setup files, or CI configs are present.

## Architecture Notes

- Main language: Python.
- Execution environment: Google Colab with GPU for full runs.
- Local environment: useful for text edits, syntax checks, Jupytext sync, and Git operations.
- The notebook includes multiple runnable pipeline sections:
  - Qwen3-8B setup and calibration text download.
  - Simple WikiText/PPL-oriented PQ vector extraction.
  - LongBench calibration vector extraction.
  - First-generation codebook training.
  - Improved LongBench-aware codebook training with held-out reconstruction MSE checks.
  - Local calibration PPL evaluation with dynamic/static outlier modes.
  - Standalone LongBench evaluation harness.

## Implemented Features

- Qwen3-8B download/loading workflow in notebook cells.
- Pure-Python safetensors loading helpers.
- Custom Qwen model definitions for extraction and evaluation.
- K/V activation extraction for PQ training data.
- LongBench calibration extraction with document-level train/test split and manifest.
- Grouped PQ codebook training for keys and values.
- Improved codebook training with balanced head groups and MSE reporting.
- PPL evaluation comparing baseline, dynamic PQ, and static-mask PQ.
- LongBench/LongBench-E evaluation comparing baseline, dynamic PQ, and static PQ.
- Reportable LongBench-E flow with task-held-out calibration, 4096-token train/eval alignment, versioned codebook selection, and calibration-derived static masks.
- Per-subvector best-restart selection for fine PQ k-means.
- Notebook section map in `NOTEBOOK_MAP.md`.

## Incomplete Features

- No formal test suite.
- No requirements or environment file.
- No automated Colab bridge scripts/configs are present in this repo.
- No documented GitHub remote/branch workflow beyond the user-provided Codex-Colab workflow.
- Full L4 GPU validation and benchmark result reporting are pending.

## Known Issues

- Full notebook execution depends on Colab paths such as `/content/qwen3_8B`, GPU availability, Hugging Face network access, and possibly Google Drive.
- Some notebook sections redefine the same helper/model names; execution order matters.
- LongBench calibration defaults to held-out mode; contaminated mode remains diagnostic only and should not be treated as reportable.
- The full L4 extraction/training/evaluation path has not yet been GPU-verified for the current revision.
- Generated artifacts such as model weights, `.npy` vectors, codebooks, CSVs, and bridge results are not represented in the repo.
- The top of `KV_Cache.py` still contains a commented Qwen3-0.6B download cell; it is inert but unrelated to the main Qwen3-8B flow.

## Build/Test Status

- Build system: none.
- Local syntax check: available with `py -m py_compile KV_Cache.py`.
- Notebook sync: available with `py -m jupytext --sync KV_Cache.ipynb`.
- Unit tests: none found.
- GPU/integration tests: unknown until run through Colab controller.

## Last Completed Milestone

LongBench-E source-flow optimization:
- Unified calibration, training, and evaluation around versioned held-out 4096-token artifacts.
- Fixed evaluator selection of the improved codebooks.
- Pinned both keys and values to 64 banks, 128 codewords, and 64 head groups.
- Removed evaluation-set leakage from static-mask construction.
- Added per-bank k-means restart selection and increased unique calibration/training supply.

## Current Task

Run the optimized codebook extraction/training and full LongBench-E test suite on the open Colab L4 runtime, then record verified results.

## Next Steps

1. Publish the synced notebook revision so Colab can pull it.
2. Run extraction, improved codebook training, and the full LongBench-E evaluator on the L4 GPU.
3. Inspect the calibration manifest, per-side NMSE reports, aggregate score CSV, and per-dataset score CSV.
4. Write `LONGBENCH_E_RESULTS.md` with verified results and next-step recommendations.
