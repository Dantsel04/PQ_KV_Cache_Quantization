# PLAN.md

## Active Milestone

Optimized LongBench-E codebook training and evaluation.

## Milestone Goal

Train reportable held-out LongBench-E codebooks for Qwen3-8B with keys and values both fixed at 64 banks × 128 codewords, then run and document the full evaluation suite.

## Task Checklist

- [x] Align extraction and evaluation prompt construction at 4096 tokens.
- [x] Use task-held-out LongBench-E calibration with versioned artifacts.
- [x] Pin key and value configurations to 64 banks, 128 codewords, and 64 groups.
- [x] Select fine k-means restarts independently per PQ bank.
- [x] Learn static outlier masks from calibration rather than evaluation samples.
- [x] Sync `KV_Cache.ipynb` and pass local syntax/config checks.
- [x] Publish the runnable revision to GitHub for Colab (`73b6b54`).
- [ ] Run calibration extraction and codebook training on the L4 GPU.
- [ ] Run the full LongBench-E baseline/dynamic/static suite.
- [ ] Inspect GPU artifacts and write `LONGBENCH_E_RESULTS.md`.

## Expected Files to Modify

- `KV_Cache.py`
- `KV_Cache.ipynb`
- `STATUS.md`
- `PLAN.md`
- `NOTEBOOK_MAP.md`
- `LONGBENCH_E_RESULTS.md` after verified GPU results exist

Do not modify generated model weights, calibration arrays, codebooks, or result files.

## Test / Verification Commands

- `py -m py_compile KV_Cache.py`
- `py -m jupytext --sync KV_Cache.ipynb`
- Static AST/config consistency check.
- Colab L4: extraction cell, improved trainer cell, then LongBench-E evaluator cell.

## Done Criteria

- `KV_Cache.py` and `KV_Cache.ipynb` are synced and published.
- Extraction manifest confirms held-out mode, 4096 tokens, and no overlap with LongBench-E tasks.
- Both key and value 64×128 codebooks pass runtime compatibility and held-out NMSE checks.
- Full LongBench-E baseline, dynamic, and calibration-static scores are saved.
- `LONGBENCH_E_RESULTS.md` records verified results and next steps.

## Likely Next Milestone

Run score/capacity ablations using the verified 64×128 result as the reference configuration.
