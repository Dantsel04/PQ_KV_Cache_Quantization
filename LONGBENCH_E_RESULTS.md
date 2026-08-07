# LongBench-E PQ Training and Evaluation Results

## Run Status

The GPU pipeline is in progress. Environment/model setup, held-out calibration
extraction, and key/value codebook training are verified. Full LongBench-E attempt
`pq-longbench-e-9bd412b7687c4b57ac97be1498e56daf` returned code 1 before inference
because the repository's current branch lacked the direct Parquet layout. No
benchmark score was produced. The loader is now pinned to immutable revision
`36914d6211386125c6fc4ce7db4a6a777fadd34c`, whose matching probe verified all 13
required LongBench-E Parquet configs before retry.

Full data/schema preflight `pq-longbench-data-preflight-57f3129627314f75b7791cc2bb5437a4`
also returned code 0, opened every shard with the required columns, and counted
3,668 evaluation samples. Full retry
`pq-longbench-e-107ac5e62ab54a7c94c2c975c0838832` is the active command.

No NMSE or LongBench-E score is reported without a matching successful Colab bridge
result.

## Verified Preparation

- Published verified-training checkpoint: `22cb8d5`.
- Colab GPU: NVIDIA L4 with CUDA available.
- Qwen3-8B: downloaded successfully to `/content/qwen3_8B`.
- Model evidence: setup command returned code 0 and confirmed `config.json`.
- Extraction command: `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7`, return code 0.
- Extraction audit: `pq-extract-verify-63a78449724942d9ba3d822e7b2976d8`,
  return code 0.
- Codebook training: `pq-train-e80750afac6c45ac8a30322536a84957`, return code 0.
- Codebook audit: `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a`,
  return code 0.

The independent codebook audit parsed every saved centroid and LUT file, verified
separate 288-head maps and all 64 groups for both sides, and checked all 576
calibration-derived static masks.

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
| Calibration extraction | Verified: 36,000 train + 4,000 held-out vectors/head |
| Key held-out NMSE | 0.0036450881517603095 |
| Key NMSE with 3 outliers | 0.0033515924495468403 |
| Value held-out NMSE | 0.020132016182836856 |
| Value NMSE with 3 outliers | 0.0195675781971871 |
| Baseline LongBench-E dataset average | Pending evaluation |
| Dynamic PQ LongBench-E dataset average | Pending evaluation |
| Calibration-static PQ LongBench-E dataset average | Pending evaluation |

## Required Run Order

1. Run the full LongBench-E evaluator now that training and its audit passed.
2. Independently audit aggregate and per-dataset CSV outputs.
3. Replace the pending benchmark entries only with values from matching successful
   bridge results.

## Potential Next Experiments

1. Compare per-dataset loss against per-head NMSE to identify whether keys or values
   dominate degradation.
2. If NMSE improves without score improvement, add an attention-output or next-token
   KL validation proxy.
3. Test 96 or 128 shared head groups while retaining 64 banks x 128 codewords.
4. Add quantization-in-the-loop calibration for later-layer activation drift.
5. Test learned dimension permutations or OPQ.
6. Repeat the best configuration with at least two calibration seeds.
