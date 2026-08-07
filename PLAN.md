# PLAN.md

## Active Milestone

Complete the requested contaminated-calibration diagnostic through retraining and
an audited LongBench-E smoke test, while preserving the verified held-out artifacts.

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
- [x] Extract and audit exact LongBench-E contaminated calibration vectors.
- [x] Train and audit separate contaminated 64 x 128 key/value codebooks.
- [ ] Run and audit a smoke test using only the contaminated codebook directory.
- [ ] Run the full LongBench-E baseline/dynamic/static suite.
- [ ] Record verified aggregate and per-dataset results.
- [ ] Update and publish final project documentation.

## Active Command

- Stage: contaminated-codebook LongBench-E smoke test
- Active command ID: `pq-contaminated-smoke-37bbd549507d41049768e41d6bb4be22`
- Verified contaminated extraction ID: `pq-contaminated-extract-27437bd820e64c808c864b18b013a61c`
- Verified contaminated extraction audit ID: `pq-contaminated-extract-audit-4cd266491e374f89b102ecfd0541dca5`
- Verified contaminated training ID: `pq-contaminated-train-cf73fe9e2f344b23a89d67df2d17db1c`
- Verified contaminated codebook audit ID: `pq-contaminated-codebook-audit-2252173ff42240e7bc261206bc3e6c6d`
- Verified smoke execution ID: `pq-longbench-smoke-afb2f5e22e5746cc89f04a7d4f92fe34`
- Verified smoke audit ID: `pq-longbench-smoke-audit-1dd591c21e7e4c1b94acdc47c7849964`
- Failed evaluation ID: `pq-longbench-e-9bd412b7687c4b57ac97be1498e56daf`
- Prior training command ID: `pq-train-e80750afac6c45ac8a30322536a84957`
- Verification command ID: `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a`
- Bridge: `G:\My Drive\PQ_agent`
- State: contaminated training and its exhaustive codebook audit returned matching
  code 0. Both sides have 288 heads, 64 groups, all 12,288 codebook files, exact
  NMSE reports, and 576 valid static masks. The same three-dataset/two-sample smoke
  test is next for a controlled comparison with held-out codebooks.
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

## Verified Contaminated Training Metrics (Diagnostic Only)

- Keys: NMSE `0.0036763673336488215`; with 3 preserved outliers
  `0.003374926364974923`.
- Values: NMSE `0.020654950321943002`; with 3 preserved outliers
  `0.020074927513791755`.

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
- Contaminated calibration root:
  `/content/qwen3_8B/pq_training_data_longbench_e_contaminated_4096`.
- Contaminated codebook root:
  `/content/qwen3_8B/codebooks_64_128_64_longbench_e_contaminated_4096_balanced_kpp_noclip`.
- Contaminated smoke outputs use `_contaminated`-suffixed directories/CSV names and
  must never be substituted for final held-out results.

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
