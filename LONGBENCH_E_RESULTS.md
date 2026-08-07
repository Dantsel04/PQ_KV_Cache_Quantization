# LongBench-E PQ Training and Evaluation Results

## Run Status

The GPU pipeline is in progress. Environment and model setup are verified; calibration
extraction is pending. Codebook training and LongBench-E evaluation have not started.

No NMSE or LongBench-E score is reported without a matching successful Colab bridge
result.

## Verified Preparation

- Git revision: `da0056f46319280f8d555b26b24277fde80225b5`.
- Colab GPU: NVIDIA L4 with CUDA available.
- Qwen3-8B: downloaded successfully to `/content/qwen3_8B`.
- Model evidence: setup command returned code 0 and confirmed `config.json`.
- Active extraction command: `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7`.

The first extraction command failed before GPU work because it selected a nonexistent
notebook cell ID. The active retry selects the extraction cell by a unique source
marker.

## Published Configuration

- Evaluation: full LongBench-E, 13 datasets.
- Context: 4096 input tokens for extraction and evaluation.
- Calibration: task-held-out LongBench v1 tasks, 800 documents, 40,000 vectors per
  layer/head before the document split.
- Keys: 64 PQ banks, 128 codewords, 64 shared groups, 3 preserved dimensions.
- Values: 64 PQ banks, 128 codewords, 64 shared groups, 3 preserved dimensions.
- Training supply: up to 50,000 unique vectors/group, at least 5,000/head.
- Fine k-means: k-means++, four restarts, per-bank best-restart selection.
- Static masks: derived from calibration, not LongBench-E evaluation samples.

## Expected Artifacts

- Calibration manifest:
  `/content/qwen3_8B/pq_training_data_longbench_e_held_out_4096/calibration_manifest.json`
- Codebook root:
  `/content/qwen3_8B/codebooks_64_128_64_longbench_e_held_out_4096_balanced_kpp_noclip`
- Key and value `codebook_mse_report.json` files.
- Calibration-derived `static_outlier_masks.json`.
- `longbench_pq_outputs/aggregate_result.csv`.
- `longbench_pq_dynamic_static_result.csv`.
- `longbench_pq_summary_by_dataset.csv`.

## Results

| Metric | Verified result |
|---|---:|
| Calibration extraction | Pending matching result |
| Key held-out NMSE | Pending training |
| Key NMSE with 3 outliers | Pending training |
| Value held-out NMSE | Pending training |
| Value NMSE with 3 outliers | Pending training |
| Baseline LongBench-E dataset average | Pending evaluation |
| Dynamic PQ LongBench-E dataset average | Pending evaluation |
| Calibration-static PQ LongBench-E dataset average | Pending evaluation |

## Required Run Order

1. Verify the active extraction result and calibration manifest.
2. Run the improved LongBench-aware trainer and verify both K/V artifacts and NMSE.
3. Run the full LongBench-E evaluator only after training passes.
4. Replace the pending table entries with values read from verified GPU artifacts.

## Potential Next Experiments

1. Compare per-dataset loss against per-head NMSE to identify whether keys or values
   dominate degradation.
2. If NMSE improves without score improvement, add an attention-output or next-token
   KL validation proxy.
3. Test 96 or 128 shared head groups while retaining 64 banks x 128 codewords.
4. Add quantization-in-the-loop calibration for later-layer activation drift.
5. Test learned dimension permutations or OPQ.
6. Repeat the best configuration with at least two calibration seeds.
