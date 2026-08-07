# STATUS.md

Current verified project state. Read this file before continuing the Colab goal.

## Working State

This repository contains a Jupytext-managed Google Colab workflow for Qwen3-8B
KV-cache product quantization. `KV_Cache.py` is the editable source and
`KV_Cache.ipynb` is the synced Colab artifact.

- A HotpotQA-focused calibration-data audit is documented in `training_plan.md`.
  It identifies the current held-out pool's 45.7% Chinese-document share, limited
  English multi-hop coverage, and uniform 50-position sampling as priority data
  mismatches. No notebook or GPU artifact was changed by this analysis.
- `PLAN.md` now defines the active milestone: implement clean calibration Variants A
  and B with role-aware extraction and an adaptive group budget of
  `max(50,000, 10,000 * heads_in_group)`. The local source implementation for
  Variant A/B configuration, Variant A source loading, decontamination, role-aware
  extraction sidecars, adaptive group training budgets, and Variant A evaluation
  path selection is implemented in `KV_Cache.py` and synced to `KV_Cache.ipynb`.
  No Variant A GPU extraction/training/evaluation result exists yet.

- Local source checkpoint on 2026-08-07:
  - Commit pushed to `origin/main`: `5891d0f`
    (`Implement clean calibration variants and adaptive PQ budgets`).
  - `py -m py_compile KV_Cache.py`: passed.
  - `py -m jupytext --sync KV_Cache.ipynb`: passed.
  - `git diff --check`: passed with only Git CRLF conversion warnings.
  - Bridge source markers verified unique:
    `longbench_calibration_extraction_role_aware`,
    `longbench_adaptive_codebook_training`, and
    `longbench_eval_dynamic_static_variant`.
- Variant A extraction command
  `pq-variant-a-extract-retry-8da35f1df1ea4fa1a90676ddcdffae43` returned code 0.
  It checked out commit `85f79b1`, selected the clean Hotpot-specialized 800-document
  mixture, wrote 800 shards with zero failures, assembled 36,000 train and 4,000
  document-disjoint test vectors per layer/head under
  `/content/qwen3_8B/pq_training_data_longbench_e_clean_hotpot_a_4096`, and started
  a Drive backup.
- Variant A extraction audit
  `pq-variant-a-extract-audit-a3b514de5c564010936cf89a01885d67` returned code 0.
  It verified 800 documents, exact Variant A source counts, 36,000 train and 4,000
  test sidecar rows, all five position roles in train/test, all 1,152 key/value
  Train/Test arrays, finite sampled vectors, matching Drive backup files, and zero
  exact source-ID overlap with the pinned LongBench-E evaluation revision.

- Published larger-diagnostic-controls checkpoint: `4c94213` on `main`.
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
- Full evaluation attempt `pq-longbench-e-9bd412b7687c4b57ac97be1498e56daf`
  returned code 1 before inference: the current LongBench repository branch no
  longer exposed the expected `qasper_e` Parquet path. The loader is being fixed to
  pin an immutable complete Parquet revision; no benchmark scores were produced.
- Data preflight `pq-longbench-data-preflight-d46390595570438da3429edfe529a40b`
  showed that the first candidate revision predated `trec_e`. Revision probe
  `pq-longbench-revision-probe-abace52f0e6f4ee3937bd2f695632114` returned code 0
  and verified all 13 required `_e` configs at immutable revision
  `36914d6211386125c6fc4ce7db4a6a777fadd34c`.
- Full download/schema preflight
  `pq-longbench-data-preflight-57f3129627314f75b7791cc2bb5437a4` returned code 0:
  all 13 shards opened with the required standardized columns and 3,668 total
  LongBench-E samples. Evaluation retry
  `pq-longbench-e-107ac5e62ab54a7c94c2c975c0838832` was submitted and remains
  without a matching terminal bridge result. The user requested that this run be
  manually interrupted in favor of a small diagnostic run.
- LongBench smoke mode can now be selected per bridge command with
  `PQ_LONGBENCH_TEST_MODE=1`; full evaluation remains the default. Smoke mode uses
  three datasets, two samples per dataset, and a 64-token generation cap.
- Smoke execution `pq-longbench-smoke-afb2f5e22e5746cc89f04a7d4f92fe34`
  returned code 0. Independent artifact audit
  `pq-longbench-smoke-audit-1dd591c21e7e4c1b94acdc47c7849964` returned code 0
  and reconciled all 18 prediction records, 9 summary rows, 3 comparison rows,
  and the aggregate result. Diagnostic dataset-average scores were baseline
  `77.77777777777777`, dynamic `66.36363636363636`, and static
  `64.44444444444444`. These two-sample smoke scores are not final benchmark
  results.
- Full retry `pq-longbench-e-full-13d26f49dd214c84bf8f6b46d6f46d20`
  has an exact matching terminal result with return code 130. It was manually
  interrupted after 64 of 224 baseline qasper samples, before any complete dataset
  result was saved. It produced no valid full-suite benchmark score.
- The next requested experiment is explicitly diagnostic contaminated calibration.
  Extraction, training, evaluation codebook selection, and result paths are now
  selectable by environment variable and use separate mode-qualified directories.
  Contaminated extraction reads the exact pinned LongBench-E Parquet revision rather
  than the legacy task-name-matched `data.zip` corpus.
- Contaminated extraction
  `pq-contaminated-extract-27437bd820e64c808c864b18b013a61c` returned code 0.
  Independent audit
  `pq-contaminated-extract-audit-4cd266491e374f89b102ecfd0541dca5`
  returned code 0 and verified exact pinned LongBench-E provenance, 800 unique
  documents across all 13 tasks, a document-disjoint 720/80 split, 36,000/4,000
  train/test vectors per head, all 1,152 arrays, and a matching Drive backup.
- Contaminated codebook training
  `pq-contaminated-train-cf73fe9e2f344b23a89d67df2d17db1c` returned code 0
  after about 213 minutes. Independent audit
  `pq-contaminated-codebook-audit-2252173ff42240e7bc261206bc3e6c6d`
  returned code 0 and exhaustively verified both 288-head maps, 64 groups per side,
  4,096 fine/coarse/LUT artifacts per side, and 576 static masks.
- Contaminated reconstruction metrics: keys NMSE
  `0.0036763673336488215` (`0.003374926364974923` with 3 outliers); values NMSE
  `0.020654950321943002` (`0.020074927513791755` with 3 outliers). These are
  diagnostic contaminated-calibration metrics, not reportable held-out results.
- Contaminated smoke execution
  `pq-contaminated-smoke-37bbd549507d41049768e41d6bb4be22` returned code 0.
  Independent audit
  `pq-contaminated-smoke-audit-f3624950a53b4b33bfa0d94ed59580fd`
  returned code 0 and reconciled all 18 predictions, 9 summary rows, 3 comparison
  rows, and the aggregate. Diagnostic dataset-average scores were baseline
  `77.77777777777777`, dynamic `83.33333333333333`, and static `80.0`.
- Against the prior held-out-codebook smoke, contaminated dynamic improved by
  `16.96969696969697` points and static improved by `15.55555555555556` points;
  baseline was unchanged. Per the requested branch rule, a larger roughly one-hour
  contaminated diagnostic is next. It was not launched before the Colab runtime
  disconnected.
- Post-disconnect preflight
  `pq-post-disconnect-artifact-preflight-7607f43198df41a58cc45bd2de614a8c`
  returned code 0 and confirmed the model, both calibration manifests, both complete
  held-out/contaminated codebook trees, and both 576-mask files survived.
- The user ended the goal at this verified diagnostic checkpoint. No larger
  diagnostic or full 13-dataset evaluation was launched afterward. Full LongBench-E
  aggregate scores remain unavailable; only the audited smoke scores are recorded.

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
| `pq-longbench-e-9bd412b7687c4b57ac97be1498e56daf` | Full LongBench-E attempt 1 | 1 | Model and both codebook sets loaded; failed before inference because current repository `main` lacked direct `qasper_e` Parquet shards. |
| `pq-longbench-data-preflight-d46390595570438da3429edfe529a40b` | Data preflight candidate 1 | 1 | Candidate revision contained early converted configs but predated `trec_e`. |
| `pq-longbench-revision-probe-abace52f0e6f4ee3937bd2f695632114` | Complete revision probe | 0 | Verified a test Parquet shard for each of all 13 LongBench-E configs at revision `36914d6211386125c6fc4ce7db4a6a777fadd34c`. |
| `pq-longbench-data-preflight-57f3129627314f75b7791cc2bb5437a4` | Full data/schema preflight | 0 | Downloaded and opened all 13 shards, verified required columns, and counted 3,668 samples. |
| `pq-longbench-smoke-afb2f5e22e5746cc89f04a7d4f92fe34` | Three-dataset LongBench smoke execution | 0 | Executed baseline, dynamic, then static evaluation for two samples on each of qasper, hotpotqa, and passage_retrieval_en. |
| `pq-longbench-smoke-audit-1dd591c21e7e4c1b94acdc47c7849964` | Smoke artifact audit | 0 | Verified and reconciled 18 predictions, 9 summaries, 3 comparisons, and one aggregate row; scores are diagnostic only. |
| `pq-longbench-e-full-13d26f49dd214c84bf8f6b46d6f46d20` | Full LongBench-E retry 3 | 130 | User interruption after 64/224 baseline qasper samples; no complete dataset or valid benchmark result. |
| `pq-contaminated-extract-27437bd820e64c808c864b18b013a61c` | Contaminated calibration extraction | 0 | Extracted exact pinned LongBench-E documents: 800 documents, 36,000 train and 4,000 test vectors per layer/head. |
| `pq-contaminated-extract-audit-4cd266491e374f89b102ecfd0541dca5` | Contaminated extraction audit | 0 | Verified provenance, all 13 tasks, document separation, 1,152 arrays, finite samples, and matching Drive backup. |
| `pq-contaminated-train-cf73fe9e2f344b23a89d67df2d17db1c` | Contaminated K/V codebook training | 0 | Trained both 64-bank x 128-codeword sides, produced NMSE reports and 576 static masks, and passed runtime compatibility. |
| `pq-contaminated-codebook-audit-2252173ff42240e7bc261206bc3e6c6d` | Contaminated codebook audit | 0 | Exhaustively validated both maps, 64 groups per side, all codebook files/LUTs, exact NMSE, and static masks. |
| `pq-contaminated-smoke-37bbd549507d41049768e41d6bb4be22` | Contaminated-codebook smoke execution | 0 | Ran baseline/dynamic/static on two samples from each of qasper, hotpotqa, and passage_retrieval_en. |
| `pq-contaminated-smoke-audit-f3624950a53b4b33bfa0d94ed59580fd` | Contaminated smoke audit | 0 | Reconciled 18 predictions and all summaries; baseline 77.78, dynamic 83.33, static 80.00. |
| `pq-post-disconnect-artifact-preflight-7607f43198df41a58cc45bd2de614a8c` | Post-disconnect persistence preflight | 0 | Confirmed model and complete held-out/contaminated calibration/codebook/mask artifacts survived. |

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
- Contaminated calibration remains diagnostic only and must not be reported as a
  final or generalizing benchmark result.
- Generated calibration arrays, weights, codebooks, results, and bridge outputs stay
  out of Git.

## Next Steps for the Goal Chat

The user ended the goal at the audited smoke checkpoint. See
`LONGBENCH_E_RESULTS.md` for the verified results, limitations, learnings, and
recommended future work. A future full LongBench-E run must still obtain its own
matching code-0 result and independent output audit.
