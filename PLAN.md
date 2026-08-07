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
- [ ] Train key and value 64 x 128 codebooks.
- [ ] Verify head maps, runtime compatibility, NMSE reports, and static masks.
- [ ] Run the full LongBench-E baseline/dynamic/static suite.
- [ ] Record verified aggregate and per-dataset results.
- [ ] Update and publish final project documentation.

## Active Command

- Stage: codebook training preparation
- Command ID: not yet submitted
- Bridge: `G:\My Drive\PQ_agent`
- Prior gate: extraction and its artifact audit both returned matching code 0.
- Selection method: use the unique training-cell source marker because notebook cell
  IDs are absent after Jupytext sync.

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
