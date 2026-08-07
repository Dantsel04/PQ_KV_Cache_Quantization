# PLAN.md

## Active Milestone

Complete the verified Qwen3-8B LongBench-E PQ pipeline through the Google Drive
Colab bridge.

## Goal Objective

Continue without manual stage-by-stage prompting until held-out calibration
extraction, 64-bank x 128-codeword key/value codebook training, and the full
baseline/dynamic/static LongBench-E evaluation all have matching successful bridge
results. Diagnose failures, make minimal source fixes, sync, commit, push, and retry.

Never claim a GPU stage succeeded without the matching `result.json`.

## Task Checklist

- [x] Align extraction and evaluation prompt construction at 4096 tokens.
- [x] Use task-held-out LongBench-E calibration with versioned artifacts.
- [x] Pin keys and values to 64 banks, 128 codewords, and 64 groups.
- [x] Select fine k-means restarts independently per PQ bank.
- [x] Learn static outlier masks from calibration rather than evaluation samples.
- [x] Sync the notebook and pass local syntax/configuration checks.
- [x] Publish revision `da0056f` to GitHub.
- [x] Verify Colab CUDA and NVIDIA L4 availability.
- [x] Download and verify Qwen3-8B under `/content/qwen3_8B`.
- [x] Complete held-out calibration-vector extraction.
- [x] Verify the calibration manifest and key/value train/test arrays.
- [x] Train key and value 64 x 128 codebooks.
- [x] Verify head maps, runtime compatibility, NMSE reports, and static masks.
- [x] Run and inspect the requested small diagnostic LongBench smoke test.
- [ ] Run the full LongBench-E baseline/dynamic/static suite.
- [ ] Record verified aggregate and per-dataset results.
- [ ] Update and publish final project documentation.

## Active Command

- Stage: prepare a fresh full LongBench-E baseline/dynamic/static evaluation retry
- Active command ID: none; the next full run must use a new unique ID
- Verified smoke execution ID: `pq-longbench-smoke-afb2f5e22e5746cc89f04a7d4f92fe34`
- Verified smoke audit ID: `pq-longbench-smoke-audit-1dd591c21e7e4c1b94acdc47c7849964`
- Failed evaluation ID: `pq-longbench-e-9bd412b7687c4b57ac97be1498e56daf`
- Prior training command ID: `pq-train-e80750afac6c45ac8a30322536a84957`
- Verification command ID: `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a`
- Bridge: `G:\My Drive\PQ_agent`
- State: the command-selectable smoke test and its independent artifact audit both
  returned matching code 0. All 18 predictions and their CSV summaries reconciled.
  The full 13-dataset evaluation is next and its outputs must replace the smoke
  artifacts before final reporting.
- Prior gates: extraction, training, and their artifact audits returned matching
  code 0.
- Selection method: use the unique standalone LongBench-cell source marker because
  notebook cell IDs are absent after Jupytext sync.

## Verified Training Metrics

- Keys: NMSE `0.0036450881517603095`; with 3 preserved outliers
  `0.0033515924495468403`.
- Values: NMSE `0.020132016182836856`; with 3 preserved outliers
  `0.0195675781971871`.
- Each side: 288 heads, 64 groups, 4,096 fine files, 4,096 coarse files, and
  4,096 LUT files.
- Static masks: 576 heads with 3 calibration-derived dimensions per head.

## Verified Smoke Metrics (Diagnostic Only)

- Dataset-average baseline: `77.77777777777777`.
- Dataset-average dynamic PQ: `66.36363636363636`.
- Dataset-average static top-3 PQ: `64.44444444444444`.
- Scope: qasper, hotpotqa, and passage_retrieval_en; two samples per dataset and
  mode, 18 predictions total. These scores must not be reported as final
  LongBench-E results.

## Bridge Loop

1. Read `STATUS.md` and this plan.
2. Check the locally copied watcher result or `result.json` for the exact active ID.
3. If pending, use a background local watcher rather than spending agent turns.
4. If failed, diagnose `stdout`/`stderr`, edit `KV_Cache.py` if needed, run
   `py -m jupytext --sync KV_Cache.ipynb`, compile, commit, push, and retry with a
   new unique ID.
5. If successful, verify the stage artifacts before submitting the next stage.
6. Update `STATUS.md` and this plan after every verified checkpoint.

## Expected Artifacts

- Calibration manifest:
  `/content/qwen3_8B/pq_training_data_longbench_e_held_out_4096/calibration_manifest.json`
- Codebook root:
  `/content/qwen3_8B/codebooks_64_128_64_longbench_e_held_out_4096_balanced_kpp_noclip`
- Key/value `head_to_codebook_map.json` and `codebook_mse_report.json` files.
- `static_outlier_masks.json` under the codebook root.
- `longbench_pq_outputs/aggregate_result.csv`.
- `longbench_pq_dynamic_static_result.csv`.
- `longbench_pq_summary_by_dataset.csv`.

## Verification Commands

- `py -m py_compile KV_Cache.py`
- `py -m jupytext --sync KV_Cache.ipynb`
- `git diff --check`
- Matching Colab bridge result plus stage-specific artifact validation.

## Done Criteria

- Extraction, training, and evaluation each have a matching return code of 0.
- Manifest provenance and task separation are valid.
- Both K/V codebook maps are complete and use their own directories.
- Held-out key/value NMSE reports exist and are summarized.
- Full LongBench-E baseline, dynamic, and calibration-static scores exist.
- `LONGBENCH_E_RESULTS.md`, `STATUS.md`, and this plan contain only verified results.

## Suggested New-Chat Goal

```text
/goal Complete PLAN.md without stopping until all extraction, codebook-training,
and LongBench-E done criteria are proven by matching Colab bridge results. Read
STATUS.md first, ignore stale bridge IDs, update the Markdown checkpoints after each
verified stage, and do not report contaminated or unverified results.
```
