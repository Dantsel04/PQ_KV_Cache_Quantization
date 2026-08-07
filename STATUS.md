# STATUS.md

Current verified project state. Read this file before continuing the Colab goal.

## Working State

This repository contains a Jupytext-managed Google Colab workflow for Qwen3-8B
KV-cache product quantization. `KV_Cache.py` is the editable source and
`KV_Cache.ipynb` is the synced Colab artifact.

- Published active-training revision: `12cef19` on `main`.
- `py -m py_compile KV_Cache.py`: passed.
- `py -m jupytext --sync KV_Cache.ipynb`: passed.
- Static configuration checks passed for held-out calibration, 4096-token
  train/eval alignment, versioned artifacts, full LongBench-E, and matching key/value
  64-bank x 128-codeword settings.
- Local Drive bridge: `G:\My Drive\PQ_agent`.
- Colab preflight confirmed CUDA and an NVIDIA L4.
- Qwen3-8B setup returned code 0 and confirmed
  `/content/qwen3_8B/config.json`.
- Held-out calibration extraction returned code 0 under command
  `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7`.
- Artifact audit `pq-extract-verify-63a78449724942d9ba3d822e7b2976d8`
  returned code 0: held-out provenance, 4096-token alignment, zero task overlap,
  and all 1,152 key/value train/test arrays were verified.
- Codebook training command `pq-train-e80750afac6c45ac8a30322536a84957`
  returned code 0. Independent artifact audit
  `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a` also returned code 0.
- The audit verified separate key/value maps for all 288 heads, 64 balanced groups
  per side, and per side 4,096 fine-centroid files, 4,096 coarse-centroid files,
  and 4,096 LUT files. Every fine table is 128 x 2, every coarse table is 11 x 2,
  and every LUT has 128 valid entries.
- Held-out reconstruction results: keys NMSE `0.0036450881517603095`
  (`0.0033515924495468403` with 3 outliers); values NMSE
  `0.020132016182836856` (`0.0195675781971871` with 3 outliers).
- Runtime compatibility passed during training, and the independent audit verified
  all 576 calibration-derived static masks with 3 unique dimensions per head.
- LongBench-E evaluation has not started.

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
| `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7` | Extraction attempt 2 | 0 | Saved and backed up 36,000 train plus 4,000 document-disjoint test vectors per layer/head. |
| `pq-extract-verify-63a78449724942d9ba3d822e7b2976d8` | Extraction artifact audit | 0 | Verified held-out mode, 800 documents (720 train/80 test), 4096 tokens, no eval-task overlap, and 1,152 float16 arrays across 36 layers and 8 KV heads. |
| `pq-train-e80750afac6c45ac8a30322536a84957` | K/V codebook training | 0 | Trained both 64-bank x 128-codeword sides, passed runtime compatibility, wrote held-out NMSE reports, and produced 576 calibration-static masks. |
| `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a` | Codebook artifact audit | 0 | Parsed every fine/coarse/LUT artifact, verified both 288-head maps and 64 groups, validated finite NMSE, and checked all static masks. |

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
   Completed and independently audited.
3. Codebook training must return code 0. Verify both head maps, runtime
   compatibility, key/value NMSE reports, and calibration-derived static masks.
   Completed and independently audited.
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

1. Run the full baseline/dynamic/static LongBench-E evaluation now that training and
   its artifact audit have both returned matching code 0.
2. Independently audit aggregate and per-dataset CSV outputs after evaluation.
3. Update `PLAN.md`, this file, and `LONGBENCH_E_RESULTS.md` after each verified
   checkpoint; commit and push meaningful source/documentation changes.
