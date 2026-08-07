# LongBench-E PQ Training and Evaluation Results

## Final Status of This Test Run

This run ended at the user's requested diagnostic checkpoint. Held-out extraction,
held-out 64-bank x 128-codeword key/value training, contaminated extraction,
contaminated retraining, and both three-dataset smoke evaluations have matching
successful Colab bridge results plus independent artifact audits.

The full 13-dataset LongBench-E evaluation was **not completed**. It has no valid
aggregate benchmark score and must not be inferred from the smoke results. The last
full attempt, `pq-longbench-e-full-13d26f49dd214c84bf8f6b46d6f46d20`, returned
code 130 after a user interruption at baseline qasper 64/224.

## Verified Configuration

- Model: Qwen3-8B on an NVIDIA L4.
- Context: 4096 tokens for calibration extraction and evaluation.
- Each K/V side: 64 PQ banks, 128 fine codewords, 64 balanced head groups, and 3
  preserved outlier dimensions.
- Compression: 496 bits/vector, 3.875 average bits/scalar, 4.129032x ratio.
- Calibration: 800 documents, split at document granularity into 720 train and 80
  reconstruction-test documents.
- Vectors per layer/head: 36,000 train and 4,000 test.
- Fine k-means: k-means++, four restarts, best restart selected per bank.
- Static masks: three dimensions per head derived from calibration vectors.

Both held-out and contaminated codebook audits verified, per side, 288 mapped
heads, 64 groups, 4,096 fine tables, 4,096 coarse tables, and 4,096 LUTs. Each
variant also contains 576 valid static masks.

## Verified Reconstruction Results

| Calibration | Side | PQ NMSE | NMSE with 3 preserved outliers |
|---|---|---:|---:|
| Held-out | Keys | 0.0036450881517603095 | 0.0033515924495468403 |
| Held-out | Values | 0.020132016182836856 | 0.0195675781971871 |
| Contaminated | Keys | 0.0036763673336488215 | 0.003374926364974923 |
| Contaminated | Values | 0.020654950321943002 | 0.020074927513791755 |

The contaminated codebooks were slightly worse on their own document-disjoint
reconstruction split for both keys and values. Preserving three outlier dimensions
improved every NMSE result, but did not reverse that ordering.

## Verified Smoke Results

Each smoke run evaluated the first two LongBench-E samples from qasper, hotpotqa,
and passage_retrieval_en in baseline, dynamic-PQ, and calibration-static-PQ modes:
18 predictions per run. Scores are diagnostic only.

### Aggregate comparison

| Codebook calibration | Baseline | Dynamic PQ | Static top-3 PQ |
|---|---:|---:|---:|
| Held-out | 77.77777777777777 | 66.36363636363636 | 64.44444444444444 |
| Contaminated | 77.77777777777777 | 83.33333333333333 | 80.0 |
| Contaminated minus held-out | 0.0 | +16.96969696969697 | +15.55555555555556 |

### Per-dataset comparison

| Dataset | Baseline | Held-out dynamic | Contaminated dynamic | Held-out static | Contaminated static |
|---|---:|---:|---:|---:|---:|
| qasper | 50.0 | 56.666666666666664 | 100.0 | 50.0 | 90.0 |
| hotpotqa | 83.33333333333333 | 42.42424242424242 | 50.0 | 43.33333333333333 | 50.0 |
| passage_retrieval_en | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

The aggregate improvement is dominated by two qasper samples. Hotpotqa improved
modestly but remained far below baseline, while passage retrieval was unchanged.

## Evidence

| Stage | Command ID | Return code |
|---|---|---:|
| Held-out extraction | `pq-extract-0a5600d2d5cb45e29d889cd9af035fb7` | 0 |
| Held-out extraction audit | `pq-extract-verify-63a78449724942d9ba3d822e7b2976d8` | 0 |
| Held-out K/V training | `pq-train-e80750afac6c45ac8a30322536a84957` | 0 |
| Held-out codebook audit | `pq-codebook-audit-b9dc71e5b20e484f8431c60d72daf61a` | 0 |
| Held-out smoke | `pq-longbench-smoke-afb2f5e22e5746cc89f04a7d4f92fe34` | 0 |
| Held-out smoke audit | `pq-longbench-smoke-audit-1dd591c21e7e4c1b94acdc47c7849964` | 0 |
| Contaminated extraction | `pq-contaminated-extract-27437bd820e64c808c864b18b013a61c` | 0 |
| Contaminated extraction audit | `pq-contaminated-extract-audit-4cd266491e374f89b102ecfd0541dca5` | 0 |
| Contaminated K/V training | `pq-contaminated-train-cf73fe9e2f344b23a89d67df2d17db1c` | 0 |
| Contaminated codebook audit | `pq-contaminated-codebook-audit-2252173ff42240e7bc261206bc3e6c6d` | 0 |
| Contaminated smoke | `pq-contaminated-smoke-37bbd549507d41049768e41d6bb4be22` | 0 |
| Contaminated smoke audit | `pq-contaminated-smoke-audit-f3624950a53b4b33bfa0d94ed59580fd` | 0 |
| Post-disconnect artifact preflight | `pq-post-disconnect-artifact-preflight-7607f43198df41a58cc45bd2de614a8c` | 0 |

The post-disconnect preflight confirmed that the model, both calibration manifests,
both complete held-out/contaminated codebook trees, and both 576-mask files remained
present in the same Colab VM.

## Learnings

1. **Reconstruction NMSE did not predict this smoke score.** Contaminated NMSE was
   slightly worse, while contaminated dynamic/static smoke scores were much higher.
   NMSE remains a useful artifact-quality gate, but it is not sufficient as a model
   quality proxy.
2. **The smoke sample is too small for a benchmark conclusion.** With two samples
   per dataset, a few changed generations cause very large score swings. The qasper
   pair accounts for most of the apparent gain.
3. **Domain/evaluation-distribution matching matters.** Training from the pinned
   LongBench-E document pool changed generation behavior materially even though
   low-level reconstruction error did not improve. This is intentionally
   contaminated evidence and cannot establish generalization.
4. **Dynamic outliers remained better than static masks.** Dynamic exceeded static
   by 1.92 points with held-out codebooks and 3.33 points with contaminated
   codebooks. During the contaminated smoke, 280/288 key heads and 288/288 value
   heads changed their dynamic head-level outlier sets, showing that one fixed mask
   is a coarse approximation.
5. **Preserved dimensions consistently help reconstruction.** Three preserved
   dimensions reduced key and value NMSE in both calibration variants.
6. **The full suite is a long-running job.** Contaminated codebook training took
   about 213 minutes. A full dynamic/static LongBench-E run is much longer than the
   smoke path and needs durable progress/checkpointing before another attempt.
7. **Artifact isolation and provenance gates prevented accidental mixing.** Mode-
   qualified calibration, codebook, and result directories plus independent audits
   kept held-out and contaminated maps/masks separate.

## Recommended Next Work

If this investigation resumes, the highest-value next steps are:

1. Run a statistically larger held-out-versus-contaminated comparison using the
   exact same sample IDs, not just a contaminated-only expansion.
2. Add a task-aligned validation proxy such as attention-output error or next-token
   KL divergence; use it alongside NMSE when choosing codebooks and head groups.
3. Add durable per-dataset/per-mode checkpoints before retrying the full 3,668-
   sample suite.
4. Investigate input-conditioned or small mask-bank static outliers, since dynamic
   top dimensions were unstable across nearly every value head.
5. Repeat promising settings across multiple calibration seeds and more samples
   before attributing improvements to contamination or training methodology.
