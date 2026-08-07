# LongBench-E PQ Training and Evaluation Results

## Run Status

GPU execution is pending. The optimized source was committed and pushed to `main` as commit `73b6b54`, but the current Codex session did not expose the browser-control runtime required to operate the open Colab tab. No repository Colab bridge/controller was available as a fallback.

No calibration, NMSE, or LongBench-E score below is being inferred or reported as successful without explicit Colab output.

## Published Configuration

- Model: Qwen3-8B
- Evaluation: full LongBench-E, 13 datasets
- Context: 4096 input tokens for both extraction and evaluation
- Calibration: task-held-out LongBench v1 tasks, 800 documents, 40,000 vectors per layer/head before the document-level split
- Keys: 64 PQ banks, 128 codewords, 64 shared head groups, 3 preserved outlier dimensions
- Values: 64 PQ banks, 128 codewords, 64 shared head groups, 3 preserved outlier dimensions
- Group training supply: up to 50,000 unique vectors, minimum 5,000 vectors per head
- Fine k-means: k-means++, four restarts, best restart selected independently per PQ bank
- Static masks: learned from calibration training vectors, not from LongBench-E evaluation samples

## Expected Artifacts

- Calibration manifest: `/content/qwen3_8B/pq_training_data_longbench_e_held_out_4096/calibration_manifest.json`
- Codebooks: `/content/qwen3_8B/codebooks_64_128_64_longbench_e_held_out_4096_balanced_kpp_noclip`
- Key NMSE report: `keys/codebook_mse_report.json`
- Value NMSE report: `values/codebook_mse_report.json`
- Static masks: `static_outlier_masks.json`
- Aggregate scores: `longbench_pq_outputs/aggregate_result.csv`
- Per-dataset scores: `longbench_pq_dynamic_static_result.csv`

## Verification Completed

- `py -m py_compile KV_Cache.py`: passed
- `py -m jupytext --sync KV_Cache.ipynb`: passed
- `git diff --check`: passed before commit
- Static configuration assertions: passed for held-out mode, 4096-token alignment, versioned paths, full LongBench-E mode, and matching key/value 64×128 settings
- GitHub publication: passed at commit `73b6b54`

## Required Colab Run Order

1. Pull commit `73b6b54` and open the synced `KV_Cache.ipynb`.
2. Run the Qwen3-8B setup/download prerequisites if `/content/qwen3_8B` is not already populated.
3. Run the LongBench calibration extraction cell (`Ij6JVcn0L5Hg`).
4. Run the improved LongBench-aware codebook trainer cell (`B-FHtYwfM1rv`).
5. Confirm both NMSE reports and runtime compatibility pass.
6. Run the standalone LongBench-E evaluator cell (`Ql7Z9OaFee2q`).
7. Copy the manifest, NMSE reports, aggregate CSV, and per-dataset CSV into the shared Colab bridge or otherwise return them for verification.

## Results Table

| Metric | Result |
|---|---:|
| Key held-out NMSE | Pending GPU run |
| Key NMSE with 3 outliers | Pending GPU run |
| Value held-out NMSE | Pending GPU run |
| Value NMSE with 3 outliers | Pending GPU run |
| Baseline LongBench-E dataset average | Pending GPU run |
| Dynamic PQ LongBench-E dataset average | Pending GPU run |
| Calibration-static PQ LongBench-E dataset average | Pending GPU run |

## Potential Next Steps After Results

1. Compare per-dataset score loss against per-head NMSE to identify whether keys or values dominate degradation.
2. If NMSE improves but LongBench-E does not, replace raw reconstruction selection with an attention-output or next-token KL validation proxy.
3. Test 96 or 128 head groups while keeping both sides at 64 banks × 128 codewords; this changes shared-codebook specialization without changing per-vector index width.
4. Add quantization-in-the-loop calibration for later layers to address activation drift caused by earlier-layer PQ.
5. Test learned dimension permutations or OPQ before changing bank or codeword capacity.
6. Repeat the full suite with at least two calibration seeds before treating a small aggregate gain as robust.
