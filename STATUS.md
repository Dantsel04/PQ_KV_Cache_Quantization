# STATUS.md

Current verified project state. Read this file before continuing the Colab goal.

## Working State

This repository contains a Jupytext-managed Google Colab workflow for Qwen3-8B
KV-cache product quantization. `KV_Cache.py` is the editable source and
`KV_Cache.ipynb` is the synced Colab artifact.

- Published revision: `da0056f46319280f8d555b26b24277fde80225b5` on `main`.
- `py -m py_compile KV_Cache.py`: passed.
- `py -m jupytext --sync KV_Cache.ipynb`: passed.
- Static configuration checks passed for held-out calibration, 4096-token
  train/eval alignment, versioned artifacts, full LongBench-E, and matching key/value
  64-bank x 128-codeword settings.
- Local Drive bridge: `G:\My Drive\PQ_agent`.
- Colab preflight confirmed CUDA and an NVIDIA L4.
- Qwen3-8B setup returned code 0 and confirmed
  `/content/qwen3_8B/config.json`.
- Calibration extraction retry is pending under command
  `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7`.
- Codebook training and LongBench-E evaluation have not started.

## Colab Bridge Contract

- Do not use browser automation for Colab.
- `command.json` schema:
  `{"id": "<unique-id>", "command": ["program", "arg1", "..."]}`.
- `result.json` contains `id`, `returncode`, `stdout`, and `stderr`.
- Ignore stale results whose ID does not exactly match the active command.
- A hidden Windows watcher may poll without consuming model turns and copy the
  matching result locally. It cannot wake an ordinary chat; use `/goal` for
  autonomous continuation.
- The synced notebook does not currently retain reliable cell IDs. Select the large
  pipeline cells by unique source markers.

## Colab Run Checkpoints

| Command ID | Stage | Return code | Verified result |
|---|---|---:|---|
| `pq-preflight-d86c6cf21fe8485fa393f581412601bd` | Environment preflight | 1 | Git revision, CUDA, and L4 passed; model was missing. |
| `pq-model-setup-d0ade7303b954045962f75ea2bc9c64f` | Qwen3-8B download | 0 | `/content/qwen3_8B/config.json` exists. |
| `pq-extract-220dc7449fa640858e5d00da29d44106` | Extraction attempt 1 | 1 | No extraction ran; the notebook cell-ID selector found no ID. |
| `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7` | Extraction attempt 2 | Pending | Uses the unique extraction source marker. |

## Implemented LongBench-E Configuration

- Task-held-out LongBench calibration with document-level train/test splitting.
- 800 calibration documents and 40,000 vectors per layer/head before the split.
- 4096-token extraction and evaluation context.
- Keys: 64 banks, 128 codewords, 64 shared head groups, 3 outlier dimensions.
- Values: 64 banks, 128 codewords, 64 shared head groups, 3 outlier dimensions.
- Balanced head grouping and per-bank best-of-four fine k-means restarts.
- Calibration-derived static masks, avoiding evaluation-set mask leakage.
- Held-out reconstruction NMSE checks and versioned codebook directories/maps.

## Required Stage Gates

1. Model setup must return code 0 and model files must exist. Completed.
2. Extraction must return code 0. Verify
   `/content/qwen3_8B/pq_training_data_longbench_e_held_out_4096/calibration_manifest.json`,
   held-out mode, 4096 tokens, no task overlap, and key/value train/test arrays.
3. Codebook training must return code 0. Verify both head maps, runtime
   compatibility, key/value NMSE reports, and calibration-derived static masks.
4. LongBench-E must return code 0. Read aggregate and per-dataset results before
   updating any reported score.

## Known Issues

- Full execution depends on Colab paths, GPU availability, Hugging Face downloads,
  and Google Drive.
- Several notebook sections redefine helper/model names; only the intended large
  pipeline cell should be selected for each bridge stage.
- Contaminated calibration remains diagnostic only and must not be reported.
- Generated calibration arrays, weights, codebooks, results, and bridge outputs stay
  out of Git.

## Next Steps for the Goal Chat

1. Inspect only the result matching
   `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7`.
2. On extraction success, verify the manifest and arrays before submitting training.
3. On training success, verify maps, NMSE reports, and masks before LongBench-E.
4. Update `PLAN.md`, this file, and `LONGBENCH_E_RESULTS.md` after each verified
   checkpoint; commit and push meaningful source/documentation changes.
