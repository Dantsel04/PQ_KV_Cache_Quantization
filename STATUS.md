# STATUS.md

Current verified project state. Read this file before continuing the Colab goal.

## Working State

This repository contains a Jupytext-managed Google Colab workflow for Qwen3-8B
KV-cache product quantization. `KV_Cache.py` is the editable source and
`KV_Cache.ipynb` is the synced Colab artifact.

- Variant D (`clean_count_key_d`) is the active data-only experiment. Its local
  implementation adds a fixed 2--30 synthetic counting schedule, deterministic
  multiplicity/order/length/paragraph-count variation, six explicit token roles,
  and separate key/value position policies. The key policy targets exactly 44%
  count-critical vectors when role supply permits; the value policy stays
  context-heavy across the unchanged 800-document Variant C QA/count mixture.
- Variant D performs no hard-example selection, manual document selection, or
  model-generated ranking. Its manifest records qualified source IDs, the extraction
  seed, generated prompt hashes, answer distribution, side-specific role counts,
  and pinned LongBench-E decontamination evidence. GPU extraction, training, and
  evaluation have not yet been claimed; each remains behind an exact-ID/code-0
  bridge result and independent artifact audit.
- Variant D extraction `pq-count-key-d-extract-r1-20260811-085510` returned code 0
  on an A100 at commit `166e977` in `1177.9181083500007` seconds, with 800/800
  shards, 36,000 train and 4,000 test vectors per head, zero failed documents, and
  an exact 44% key count-critical share. Independent audit
  `pq-count-key-d-extract-audit-r1-20260811-091831` returned code 1 because the
  shortest prompt was 3,828 tokens, below the preserved 3,900-token floor. No
  training was launched. The source now gives low-count schedules more clean donor
  capacity and fails generation explicitly below 3,900 tokens; fresh extraction and
  audit are required.
- Corrected extraction `pq-count-key-d-extract-r2-20260811-092215` returned code 0
  on an A100 at commit `fba71cb8d9544589836b245d45bc26efeb944fa3` in
  `1242.9397719459957` seconds. It regenerated all 800 documents at 3,901--4,096
  tokens, wrote 36,000 train and 4,000 test vectors per head, covered every answer
  from 2 through 30, and reported an exact 44% key count-critical share.
  Independent audit `pq-count-key-d-extract-audit-r2-20260811-094730` returned code
  1 because the key index had no `repeated_span` rows. The cause was a deterministic
  sampler bug: the 22-position critical budget was filled in alphabetical role order
  before that role was reached. No training was launched. The source now scales the
  critical-role weights within the 22-position key budget (6 boundaries, 5 anchors,
  8 repeated spans, 2 suffix tokens, and 1 decode transition) and refuses to finish
  extraction unless all six explicit roles occur in both splits and tensor sides.
- Clean extraction retry `pq-count-key-d-extract-r3-20260811-095202` returned code 0
  on an A100 at commit `6e86f53b202c213d0b802c93236066b07b3efea3` in
  `1244.0775129879985` seconds. It produced all 800 prompts at 3,901--4,096
  tokens, 36,000 train and 4,000 test vectors per head, all six roles on both
  tensor sides and splits, and the exact 44% key count-critical share.
  Independent audit `pq-count-key-d-extract-audit-r3-20260811-101547` returned
  code 0. It verified the exact 240/160/160/80/160 task mixture, 32,992 clean donor
  references, all 160 deterministic count examples and answers 2--30, all 1,152
  finite arrays, position-index provenance, prompt bounds, and matching Drive
  artifacts. Training command `pq-count-key-d-train-r1-20260811-102031` was then
  launched, but the Colab runtime disconnected and never published a matching
  terminal result. After more than nine hours, `result.json` still contained the
  prior extraction-audit ID and no Variant D codebook tree existed on Drive, so r1
  is recorded as interrupted with no return code—not as success or failure. The
  audited calibration tree remained intact. Clean retry
  `pq-count-key-d-train-r2-20260811-200442` was then submitted.
- The restarted controller wrote `controller_ready.json` with
  `ignored_existing_id=pq-count-key-d-train-r2-20260811-200442`, proving r2 was
  intentionally ignored at startup and never became a GPU attempt. After additional
  compute units were made available, fresh command
  `pq-count-key-d-train-r3-20260811-201105` was submitted. It is the active training
  attempt; no later stage is authorized until its matching result and audit pass.
- On 2026-08-12 the user shortened the post-training evaluation. After a successful
  codebook audit, run a paired 20-example `passage_count` check using a deterministic
  subset of the prior Scenario A/Variant C IDs, then baseline/dynamic 100-example
  runs on `hotpotqa`, `repobench-p`, `gov_report`, and `qasper`. The former
  five-dataset four-mode smoke and 13-dataset mini are cancelled for this candidate.
  Local exact-result alarm watchers raise Windows master volume, keep the machine
  awake, and sound for two minutes when codebook training and the final targeted
  evaluation finish.

- Deterministic long-context QA/count calibration Variant C was implemented and
  pushed beginning at commit `cf6cf9e24d5db27da4a0b6aa7456ad0ce6bcd408`. The new notebook
  cell marker is `deterministic_long_context_calibration_generation`; it uses stable
  lexical donor ranking with source-ID tie breaks and exact synthetic duplicate
  construction, with no manual or model-based document selection.
- Variant C uses 800 pinned, clean training examples: 240 HotpotQA, 160 MuSiQue,
  160 2WikiMultihopQA, 80 NarrativeQA/Qasper-shaped, and 160 synthetic passage-count
  examples. It targets 4K prompts while retaining the existing 40K vectors/head,
  role-aware sampling, PQ configuration, and adaptive trainer budget.
- Local verification passed: `py -m py_compile KV_Cache.py`, Jupytext sync,
  `git diff --check`, unique bridge markers, and a deterministic helper harness that
  verified repeat generation, exact count labels, and approximately 4K prompts.
- Follow-up commits `18ae890`, `f839664`, and `ab610b6` added deterministic prompt
  hashes, replaced all-pairs near-duplicate matching with an exact inverted-shingle
  candidate index, and fixed title-wrapped donor validation. Local regression checks
  verified indexed/exhaustive decontamination equivalence and deterministic 4K QA/count
  generation.
- Extraction `r4` reached Variant C generation and returned code 1 because donor
  paragraphs rendered as `Title: body` were compared against body-only provenance
  hashes, yielding an empty donor pool. Commit `ab610b6` fixes that mismatch without
  weakening provenance validation.
- Extraction retry `pq-qa-count-c-extract-r5-20260811-001100` returned code 0 on an
  NVIDIA A100 at commit `ab610b6` in `1094.5512245959999` seconds. It generated all
  800 prompts at 3,901--4,096 tokens (mean `4017.91875`), including 160 exact
  synthetic passage-count examples, extracted 36,000 train and 4,000 test vectors
  per head, and copied the calibration tree to Drive.
- Independent extraction audit
  `pq-qa-count-c-extract-audit-r5-20260811-003120` returned code 0. It verified the
  exact 240/160/160/80/160 source mix, all 8,568 donor references, all 160 exact
  counting constructions, document-disjoint train/test metadata, all five position
  roles in both splits, all 1,152 finite K/V arrays, and matching Drive artifacts.
- Codebook training command `pq-qa-count-c-train-r1-20260811-003240` returned code 0
  on the A100 in `19570.492242776` seconds. It trained separate 64-group key/value
  trees with the adaptive budget and copied them to Drive. Reconstruction NMSE was
  keys `0.0035588322915693724` (`0.0032526608594300468` with three outliers) and
  values `0.01935353161805151` (`0.018816424291494507` with three outliers).
- Codebook audit `r1` returned code 1 because the independent verifier treated the
  Python-list LUT text as a whitespace matrix; no trained artifact failed. After
  switching only the verifier to `ast.literal_eval`, audit
  `pq-qa-count-c-codebook-audit-r2-20260811-065330` returned code 0. It verified, per
  side, 64 groups, all 288 heads, all 12,288 fine/coarse/LUT files and their shapes,
  exact adaptive budgets with no duplication or shortfall, all source/role strata,
  finite reconstruction metrics, 576 static masks, and 24,583 matching local/Drive
  files.
- Five-dataset smoke command `pq-qa-count-c-smoke-r1-20260811-065530` returned code
  0 in `1434.0522092989995` seconds. Independent audit
  `pq-qa-count-c-smoke-audit-r1-20260811-072100` returned code 0 and reconciled all
  200 predictions across baseline, key-only, value-only, and combined dynamic modes.
  Dataset-average scores were baseline `27.929914022255225`, key-only
  `25.415594270228418`, value-only `28.458727232519806`, and combined dynamic
  `28.376327532117006`. Combined dynamic improved HotpotQA by `+4.04761904761905`
  and MultiFieldQA-English by `+4.27644285539022`, held 2WikiMQA flat, regressed
  Qasper by `-6.09199435370035`, and tied a zero baseline on the ten passage-count
  examples. The tiny passage-count slice was therefore inconclusive.
- Paired 100-example passage-count confirmation
  `pq-qa-count-c-passage-confirm-r1-20260811-072300` returned code 0, and audit
  `pq-qa-count-c-passage-confirm-audit-r1-20260811-075520` returned code 0. It used
  the exact prior Scenario A source indices and reproduced baseline `18.0`. Variant C
  dynamic scored `2.6666666666666665`, only `+0.6666666666666665` over Variant A's
  `2.0` and just `0.14814814814814814` retention versus baseline. It failed both the
  five-point-improvement and 50%-retention gates.
- The conditional 13-dataset 100-sample rerun was not launched. Variant C gave
  encouraging but noisy QA smoke changes while failing its central passage-count
  objective on the statistically stronger paired check. Spending another roughly
  11 A100-hours is not justified for this codebook under the agreed gate.

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
  Variant A extraction, training, artifact audits, clean locked Hotpot validation,
  and held-out-codebook comparison are complete for the first calibration seed.
  Variant B has not been extracted, trained, or evaluated.

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
- Variant A training attempt
  `pq-variant-a-train-fef2a8857b8d4c72ae2a97c554a526ec` returned code 1 before
  training codebooks. The bridge marker selected the legacy WikiText trainer, which
  looked for `/content/qwen3_8B/pq_training_data/keys`. The marker has been moved to
  the LongBench-aware adaptive trainer and local `py_compile`, Jupytext sync, and
  `git diff --check` passed after the fix.
- Variant A training retry
  `pq-variant-a-train-retry-c48a885533d048b2b88127e274bab585` reached the correct
  LongBench-aware trainer, validated the adaptive group budget rule, loaded Variant A
  provenance, clustered key heads, and failed before k-means with
  `NameError: Counter is not defined` in the new stratified group sampler. The
  missing local import has been added to the trainer cell and local `py_compile`,
  Jupytext sync, and `git diff --check` passed after the fix.
- Variant A training retry 2
  `pq-variant-a-train-retry2-9643da62ccd74903b9dae061a0e08b13` returned code 0.
  It checked out commit `fa03b06`, validated the clean Variant A manifest and the
  adaptive group budget rule, trained key and value codebooks under
  `/content/qwen3_8B/codebooks_64_128_64_longbench_e_clean_hotpot_a_4096_adaptive10k`,
  produced calibration-static masks, and copied the codebook tree to Drive.
- Variant A codebook audit
  `pq-variant-a-codebook-audit-f3fe3e90c0364cd998a23d4039efe422` returned code 0.
  It verified both key/value sides: 64 groups, 288 mapped heads, 4,096 fine,
  4,096 coarse, and 4,096 LUT files per side, complete Drive backup, 576 static
  masks, adaptive group budgets with zero shortfalls/duplication, and role/source
  counts preserved into trainer sampling. Reconstruction NMSE was keys
  `0.0035121105743023766` (`0.003212381993457284` with 3 outliers) and values
  `0.019508355385729165` (`0.018970648296492462` with 3 outliers).
- Clean Hotpot validation attempt
  `pq-variant-a-hotpot-val-a607914f1e3a43818c33374334b143dc` returned code `-9`
  before any prediction records were produced. It selected 100 clean locked HotpotQA
  training-split examples, then was killed after loading key/value PQ processors,
  likely due to memory pressure from materializing too much HotpotQA source data
  before model loading. No validation score was produced; retry with memory-efficient
  selection is required.
- Clean Hotpot validation retry
  `pq-variant-a-hotpot-val-retry-d3d8e6e030804907a4fa71c6e6ea9550` selected 100
  clean locked HotpotQA training-split examples but returned code 1 before
  prediction because the custom validation script constructed the model in fp32 and
  hit CUDA OOM while moving it to the L4. No validation score was produced; retry
  must set the default dtype to `torch.bfloat16` before model construction, matching
  the notebook's evaluation main.
- Clean Hotpot validation retry 2
  `pq-variant-a-hotpot-val-retry2-14942209f11045108b3a43c706602589` returned code 0.
  Independent audit `pq-variant-a-hotpot-val-audit-5195c46a1b5d41abbd45a31b61c4e36b`
  also returned code 0 and verified 100 locked clean decontaminated HotpotQA
  training-split examples, matching prediction IDs across modes, summary consistency,
  and Drive backup. Verified F1-style scores were baseline `76.59870129870131`,
  key-only dynamic `71.02727272727273`, value-only dynamic `77.77468169085817`,
  and combined dynamic `70.86735466116272`. Combined dynamic retained
  `0.9251769737558754` of baseline but was `5.731346637538593` points below
  baseline, missing the five-point Variant A gate. Value-only improved by
  `1.175980392156859`, while key-only dropped by `5.571428571428584`, identifying
  key quantization as the immediate limiting factor on this locked set.
- Held-out Hotpot comparison
  `pq-heldout-hotpot-compare-1f4f69c34a844267bd2e346facb2c60e` returned code 0
  on the same 100 locked HotpotQA IDs and prompts using the original held-out
  codebooks. Independent audit
  `pq-heldout-hotpot-audit2-b7dd988a54984d298d607990a5004c17` returned code 0
  after an initial audit-script schema correction. It verified matching locked IDs,
  100 prediction records for each mode, summary recomputation, per-answer-type
  consistency, and Drive backup. Verified held-out-codebook scores were baseline
  `76.59870129870131`, key-only dynamic `66.8650569872538`,
  value-only dynamic `75.36536796536798`, and combined dynamic
  `64.14951714951715`. Variant A minus held-out on identical prompts was `0.0`
  baseline, `+4.162215740018937` key-only, `+2.4093137254901933` value-only,
  and `+6.71783751164557` combined dynamic. Variant A is therefore materially
  better than the original held-out codebooks on this locked Hotpot diagnostic,
  but it still does not advance because combined PQ remains `5.731346637538593`
  points below the uncompressed baseline on seed 0.
- Scenario A A100 mini LongBench-E local/source controls were verified on
  2026-08-09/2026-08-10 at commit `e2992b0`: `py -m py_compile KV_Cache.py`,
  `py -m jupytext --sync KV_Cache.ipynb`, `git diff --check`, and bridge marker
  uniqueness all passed. The notebook already supports all-13 LongBench-E mini
  evaluation with `PQ_LONGBENCH_TEST_MODE=0`,
  `PQ_LONGBENCH_MAX_SAMPLES_PER_DATASET=100`, deterministic random sampling with
  seed `20260809`, `PQ_LONGBENCH_MAX_NEW_TOKENS=128`,
  `PQ_LONGBENCH_EVAL_MODES=baseline,dynamic`,
  `PQ_LONGBENCH_CALIBRATION_MODE=held_out`,
  `PQ_LONGBENCH_CALIBRATION_VARIANT=clean_hotpot_a`, and run tag
  `scenario_a_a100_100x128_bd`.
- A stale Scenario A eval command
  `pq-scenario-a-a100-mini-100x128-bd-20260809-e2992b0` was ignored by the Colab
  controller after restart. Retry
  `pq-scenario-a-a100-mini-100x128-bd-rerun-20260809-163056` returned code 1 only
  after confirming `NVIDIA A100-SXM4-80GB`, stopping before evaluation because the
  fresh runtime lacked `/content/qwen3_8B/config.json` and the Drive model backup.
  Model setup command `pq-a100-model-setup-20260809-163246` then returned code 0,
  confirmed the same A100, and downloaded `Qwen/Qwen3-8B` with `config.json`,
  tokenizer files, and five safetensor shards under `/content/qwen3_8B`.
- Scenario A eval command
  `pq-scenario-a-a100-mini-100x128-bd-eval-20260809-163944` returned code 0.
  It confirmed `NVIDIA A100-SXM4-80GB` and completed in
  `39530.790924315` seconds (`10.980775256754166` hours). Independent artifact
  audit verified 13 LongBench-E datasets, 100 deterministic random examples per
  dataset, seed `20260809`, 1,300 baseline and 1,300 dynamic records, matching
  sample IDs across modes, no short datasets, no static-mode outputs, Variant A
  `clean_hotpot_a` held-out codebook provenance, and aggregate/per-dataset CSV
  reconciliation against the prediction JSONL files. Dataset-average scores were
  baseline `47.65967494764164`, dynamic `44.349905555086025`, dynamic minus
  baseline `-3.3097693925556158`.

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
| `pq-scenario-a-a100-mini-100x128-bd-rerun-20260809-163056` | Scenario A A100 mini attempt pre-model | 1 | Confirmed `NVIDIA A100-SXM4-80GB`, then stopped before evaluation because `/content/qwen3_8B/config.json` and Drive model backup were missing. |
| `pq-a100-model-setup-20260809-163246` | A100 model setup | 0 | Confirmed `NVIDIA A100-SXM4-80GB` and downloaded Qwen3-8B config/tokenizer plus five safetensor shards under `/content/qwen3_8B`. |
| `pq-scenario-a-a100-mini-100x128-bd-eval-20260809-163944` | Scenario A A100 100x128 baseline/dynamic mini eval | 0 | Confirmed A100 and reconciled all 2,600 prediction records. Baseline dataset avg `47.65967494764164`; dynamic `44.349905555086025`; delta `-3.3097693925556158`; no static mode was run. |

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
