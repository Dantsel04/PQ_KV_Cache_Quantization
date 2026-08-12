# LongBench-E PQ Training and Evaluation Results

## Variant D Key-Side Counting Calibration Result

Variant D preserved Variant C's 800-document clean mixture and all model,
compression, codebook-capacity, and evaluation settings. It changed only the
deterministic synthetic count construction and side-specific calibration-vector
selection, realizing an audited 44% count-critical key share. Extraction, codebook
training, and their exhaustive audits returned code 0.

Reconstruction results:

| Side | PQ NMSE | NMSE with 3 preserved outliers |
|---|---:|---:|
| Keys | 0.0035743668248075927 | 0.0032615700334262384 |
| Values | 0.01985270412498231 | 0.0192994723730238 |

At the user's request, evaluation was shortened. A paired 20-example deterministic
subset of the prior Scenario A passage-count sample scored baseline `5.0`, Variant A
dynamic `0.0`, Variant C dynamic `0.0`, and Variant D dynamic `5.0`. This is 100%
retention and +5 points over Variant C, but represents only one correct example and
is directional rather than a reliable count-quality estimate.

The audited targeted 100-example-per-dataset baseline/dynamic result was:

| Dataset | Baseline | Variant D dynamic | Delta | Variant A dynamic | D minus A |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 55.343050808840275 | 49.1991822991823 | -6.143868509657975 | 47.81751535501536 | +1.3816669441669376 |
| RepoBench-P | 56.52 | 54.47 | -2.05 | 55.169999999999995 | -0.6999999999999957 |
| GovReport | 19.82597819300822 | 19.928210311601248 | +0.102232118593028 | 19.87307478923544 | +0.05513552236580921 |
| Qasper | 35.14161725131099 | 31.293124262967982 | -3.848492988343008 | 30.508659825156776 | +0.7844644378112058 |
| **Dataset average** | **41.707661563289875** | **38.72262921843788** | **-2.9850323448519944** | **38.342312492351894** | **+0.38031672608598655** |

Variant D modestly improves over Variant A on this paired four-dataset average and
does not collapse the small count subset, but it still loses 7.16% of baseline on
average and regresses HotpotQA materially. It is not strong enough to justify the
cancelled 13-dataset suite. A deterministic 4K needle-in-a-haystack diagnostic is a
possible inexpensive next step; it would diagnose positional retrieval without new
codebook training.

Evidence:

| Stage | Command ID | Return code |
|---|---|---:|
| Clean extraction | `pq-count-key-d-extract-r3-20260811-095202` | 0 |
| Extraction audit | `pq-count-key-d-extract-audit-r3-20260811-101547` | 0 |
| Codebook training | `pq-count-key-d-train-r3-20260811-201105` | 0 |
| Codebook audit | `pq-count-key-d-codebook-audit-r1-20260812-050952` | 0 |
| Passage setup attempt 1 | `pq-count-key-d-passage20-r1-20260812-051119` | 1 |
| Passage setup attempt 2 | `pq-count-key-d-passage20-r2-20260812-051307` | 1 |
| Passage setup attempt 3 | `pq-count-key-d-passage20-r3-20260812-051525` | 1 |
| Paired 20-example passage check | `pq-count-key-d-passage20-r4-20260812-052118` | 0 |
| Passage audit | `pq-count-key-d-passage20-audit-r1-20260812-053657` | 0 |
| Four-dataset targeted evaluation | `pq-count-key-d-targeted4-r1-20260812-053849` | 0 |
| Targeted evaluation audit | `pq-count-key-d-targeted4-audit-r1-20260812-090018` | 0 |

## Variant C QA/Counting Calibration Result

Variant C used 800 deterministic clean 4K prompts: 240 HotpotQA, 160 MuSiQue,
160 2WikiMultihopQA, 80 NarrativeQA rendered in Qasper form, and 160 synthetic
passage-count prompts. Extraction, exhaustive calibration audit, adaptive key/value
training, and exhaustive codebook audit all returned code 0 on the A100.

Reconstruction results:

| Side | PQ NMSE | NMSE with 3 preserved outliers |
|---|---:|---:|
| Keys | 0.0035588322915693724 | 0.0032526608594300468 |
| Values | 0.01935353161805151 | 0.018816424291494507 |

The audited five-dataset smoke used ten deterministic random samples per dataset and
four modes. Scores were:

| Dataset | Baseline | Key only | Value only | Combined dynamic | Combined delta |
|---|---:|---:|---:|---:|---:|
| qasper | 27.292313332966696 | 20.8 | 26.08655596209776 | 21.200318979266346 | -6.09199435370035 |
| multifieldqa_en | 37.69059011164275 | 37.56368563685638 | 41.54041353383459 | 41.96703296703297 | +4.27644285539022 |
| hotpotqa | 64.66666666666667 | 65.38095238095238 | 64.66666666666667 | 68.71428571428572 | +4.04761904761905 |
| 2wikimqa | 10.0 | 0.0 | 10.0 | 10.0 | 0.0 |
| passage_count | 0.0 | 3.3333333333333335 | 0.0 | 0.0 | 0.0 |
| **Dataset average** | **27.929914022255225** | **25.415594270228418** | **28.458727232519806** | **28.376327532117006** | **+0.446413509861781** |

Because the ten passage-count examples had a zero baseline, a paired 100-example
confirmation reused the exact Scenario A source indices. Baseline reproduced `18.0`;
Variant A dynamic was `2.0`, and Variant C dynamic was `2.6666666666666665`.
Variant C therefore improved only `+0.6666666666666665` over Variant A and retained
`14.814814814814814%` of baseline. It failed the predeclared five-point-improvement
and 50%-retention gate.

The conditional 13-dataset rerun was not launched. The QA smoke was directionally
mixed and the stronger counting check showed that adding synthetic counting documents
to a globally stratified calibration mixture did not repair the counting collapse.
The next data-only attempt should change *which key vectors are sampled*: assign an
explicit high key-side quota to duplicate-passage identity tokens and answer-decode
transitions, selected by deterministic token/paragraph metadata. Merely increasing
the number of synthetic counting documents is not supported by this result.

Evidence:

| Stage | Command ID | Return code |
|---|---|---:|
| Variant C extraction | `pq-qa-count-c-extract-r5-20260811-001100` | 0 |
| Extraction audit | `pq-qa-count-c-extract-audit-r5-20260811-003120` | 0 |
| Codebook training | `pq-qa-count-c-train-r1-20260811-003240` | 0 |
| Corrected codebook audit | `pq-qa-count-c-codebook-audit-r2-20260811-065330` | 0 |
| Five-dataset smoke | `pq-qa-count-c-smoke-r1-20260811-065530` | 0 |
| Smoke audit | `pq-qa-count-c-smoke-audit-r1-20260811-072100` | 0 |
| 100-example passage-count confirmation | `pq-qa-count-c-passage-confirm-r1-20260811-072300` | 0 |
| Passage-count confirmation audit | `pq-qa-count-c-passage-confirm-audit-r1-20260811-075520` | 0 |

## Final Status of This Test Run

This run ended at the user's requested diagnostic checkpoint. Held-out extraction,
held-out 64-bank x 128-codeword key/value training, contaminated extraction,
contaminated retraining, and both three-dataset smoke evaluations have matching
successful Colab bridge results plus independent artifact audits.

The full 13-dataset LongBench-E evaluation was **not completed**. It has no valid
aggregate benchmark score and must not be inferred from the smoke results. The last
full attempt, `pq-longbench-e-full-13d26f49dd214c84bf8f6b46d6f46d20`, returned
code 130 after a user interruption at baseline qasper 64/224.

A later Scenario A / Variant A A100 mini run used the clean `clean_hotpot_a`
held-out codebooks, all 13 LongBench-E datasets, 100 deterministic random samples
per dataset, seed `20260809`, a 128-token generation cap, and baseline plus
dynamic modes only. Eval command
`pq-scenario-a-a100-mini-100x128-bd-eval-20260809-163944` returned code 0 on
`NVIDIA A100-SXM4-80GB` after `39530.790924315` seconds. Independent artifact
audit reconciled 1,300 baseline and 1,300 dynamic prediction records with the
sample manifest, summary CSV, comparison CSV, and aggregate CSV; no static mode
outputs were produced.

### Scenario A A100 mini result

| Metric | Baseline | Dynamic PQ | Dynamic minus baseline |
|---|---:|---:|---:|
| Dataset average | 47.65967494764164 | 44.349905555086025 | -3.3097693925556158 |
| Sample weighted | 47.65967494764164 | 44.349905555086025 | -3.3097693925556158 |

Per-dataset results:

| Dataset | Baseline | Dynamic PQ | Delta |
|---|---:|---:|---:|
| qasper | 35.14161725131099 | 30.50865982515678 | -4.63295742615421 |
| multifieldqa_en | 46.54651948400876 | 42.95391510560119 | -3.5926043784075716 |
| hotpotqa | 55.34305080884028 | 47.817515355015345 | -7.525535453824936 |
| 2wikimqa | 34.331428571428575 | 30.82907268170426 | -3.5023558897243134 |
| gov_report | 19.82597819300822 | 19.87307478923543 | +0.04709659622721318 |
| multi_news | 18.580842268952967 | 18.645302373440863 | +0.06446010448789607 |
| trec | 71.0 | 70.0 | -1.0 |
| triviaqa | 90.9 | 88.9 | -2.0 |
| samsum | 41.056337741791566 | 41.11123208596446 | +0.05489434417289374 |
| passage_count | 18.0 | 2.0 | -16.0 |
| passage_retrieval_en | 62.0 | 60.0 | -2.0 |
| lcc | 70.33 | 68.74 | -1.5900000000000034 |
| repobench-p | 56.52 | 55.17 | -1.3500000000000014 |

Under the expected roughly +/-1.5 to 4 point mini-benchmark aggregate error, the
`-3.3097693925556158` aggregate delta is not a clear pass. It is ambiguous on
aggregate uncertainty alone, but the direction is negative and the largest
regressions on HotpotQA and passage counting make the diagnostic lean bad for
advancing this exact Variant A key/value PQ setting.

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
