# PLAN.md

## Active Milestone

Extract, train, and evaluate deterministic long-context QA/count Variant C. The
immediate gate is a paired smoke run on Qasper, MultiFieldQA-English, HotpotQA,
2WikiMQA, and passage counting; a good smoke result advances to the same 13-dataset
100-sample A100 mini evaluation used for Variant A.

Current command gate: extraction command
`pq-qa-count-c-extract-20260810-233059` is present in the Drive bridge but has not
yet been acknowledged by the Colab controller. Do not start training until its exact
matching code-0 result and an independent artifact audit are available.

The detailed data rationale and proposed mixtures are in `training_plan.md`. The
prior held-out/contaminated smoke milestone is archived in `STATUS.md` and
`LONGBENCH_E_RESULTS.md`.

## Goal Objective

Build, train, and compare:

- **Variant A — Hotpot-specialized ceiling:** a clean in-domain calibration corpus
  dominated by official HotpotQA training data and other English multi-hop sources.
- **Variant B — suite-safe mixture:** a broader English LongBench-oriented corpus
  that retains substantial multi-hop coverage while limiting regressions on Qasper,
  retrieval, summarization, and code tasks.

Both variants must use role-aware position sampling and this adaptive group budget:

```text
group_target = max(50_000, 10_000 * number_of_heads_in_group)
```

This yields 25K vectors/head for a two-head group, 10K/head for a five-head group,
and 10K/head with a 100K total for a ten-head group. Sampling must remain balanced
across heads and must not duplicate vectors when sufficient unique vectors exist.

The project succeeds only if a clean candidate improves downstream LongBench quality;
lower reconstruction NMSE alone is not sufficient.

## Scope and Guardrails

- Edit `KV_Cache.py`, then sync `KV_Cache.ipynb` with Jupytext.
- Keep Qwen3-8B, 4,096-token extraction/evaluation, 64 banks, 128 codewords, 64
  head groups, and three preserved outlier dimensions fixed for the first comparison.
- Keep calibration source variant separate from contamination mode. Do not label a
  clean official-training-split corpus as `contaminated`.
- Pin every external dataset revision and record source split, source ID, license,
  prompt hash, and deduplication result in the manifest.
- Never use a LongBench-E evaluation example to train a reportable codebook.
- Use separate calibration, codebook, cluster-cache, mask, and result directories for
  Variants A and B. Never mix their head maps or static masks.
- Preserve the existing held-out and contaminated artifacts unchanged.
- Run Variant A before Variant B so the Hotpot-focused ceiling is known before
  trading specialization for suite breadth.
- Run dynamic evaluation before static-mask evaluation.
- Never claim a GPU stage succeeded without a matching code-0 Colab bridge result and
  an independent artifact audit.

## Dataset Definitions

### Variant A: Hotpot-specialized ceiling

Start with 800 calibration documents and the existing 40,000-vector extraction budget
per layer/head so corpus quality and sampling policy are isolated from raw volume.

| Source | Documents | Share |
|---|---:|---:|
| Official HotpotQA training split | 400 | 50% |
| MuSiQue training split | 160 | 20% |
| 2WikiMultihopQA training split | 120 | 15% |
| English Wikipedia retrieval with hard distractors | 80 | 10% |
| General English long-form QA | 40 | 5% |

### Variant B: suite-safe mixture

Start with the same document and vector totals for a controlled comparison.

| Source family | Share |
|---|---:|
| Clean HotpotQA training split | 30% |
| MuSiQue and clean 2WikiMultihopQA training splits | 20% |
| English single-document/scientific QA | 20% |
| English retrieval and counting | 15% |
| English summarization | 10% |
| Code or Chinese tasks according to desired suite weighting | 5% |

Exact document counts must sum to 800 and be recorded in the manifest before
extraction. If a source cannot supply enough decontaminated examples, redistribute
only within the same functional family and record the change.

## Role-Aware Position Budget

Each source document initially contributes 50 unique positions:

| Position role | Positions/document |
|---|---:|
| Supporting facts and local neighborhoods | 15 |
| Bridge entities, titles, and paragraph boundaries | 8 |
| Question and final instruction suffix | 8 |
| Teacher-forced answer/decode states | 4 |
| Uniform context and distractors | 15 |

For sources without gold support annotations, use baseline teacher attention from
answer tokens or a documented deterministic fallback. Preserve a uniform quota so
ordinary context and distractors remain represented.

Every shard must retain `variant`, `task`, `source_dataset`, `source_split`,
`source_id`, `position_role`, token offset, paragraph/title identity where available,
and train/test assignment. Assembly must not destroy the role metadata before
stratified trainer sampling occurs.

## Decontamination Requirements

Reject source examples matching pinned LongBench-E by any of:

1. source ID;
2. normalized question;
3. normalized question plus answer;
4. supporting-title set;
5. supporting-sentence or paragraph hash;
6. documented near-duplicate context threshold.

Write accepted and rejected counts plus rejection reasons to the manifest. Calibration
train/test and task-validation splits must be separated by question and supporting-page
cluster, not just by prompt row.

## Implementation Checklist

### Phase 1 — configuration and provenance

- [x] Add an explicit calibration variant selector independent of
  `PQ_CALIBRATION_MODE`.
- [x] Add immutable source revisions and official training-split loaders for every
  Variant A/B source.
- [x] Implement the LongBench-E decontamination index and auditable rejection report.
- [x] Add variant-qualified calibration, cluster-cache, codebook, static-mask, and
  evaluation result paths.
- [x] Validate that source mixtures sum to the configured document total.
- [x] Extend the calibration manifest schema with source, role, deduplication, and
  realized-mixture fields.

### Phase 2 — role-aware extraction

- [x] Map question, supporting-fact, title/entity, distractor, and answer spans to
  final post-template token offsets.
- [x] Guarantee unique per-role quotas with deterministic fallback when a role has
  too few tokens.
- [x] Capture teacher-forced answer/decode K/V states without exposing any evaluation
  answer to reportable training.
- [x] Save role/source sidecars with every resumable shard.
- [x] Assemble train/test arrays while retaining role-aware indices.
- [ ] Audit document separation, role totals, source totals, prompt lengths, and zero
  LongBench-E overlap before training.

### Phase 3 — adaptive group training budget

- [x] Replace the fixed `TARGET_TRAIN_SIZE` use with
  `max(50_000, 10_000 * len(file_paths))` for each group.
- [x] Keep head contributions equal within a group, subject to unique-vector supply.
- [x] Do not duplicate vectors; report any requested-versus-realized shortfall.
- [x] Stratify the group sample by source family and position role so the intended
  mixture survives the group cap.
- [x] Record group head count, requested total, realized total, vectors per head,
  task counts, role counts, and duplication count in the training summary.
- [x] Add static validation for the expected allocation at group sizes 2 through 10.

### Phase 4 — local verification

- [x] Run `py -m py_compile KV_Cache.py`.
- [x] Run `py -m jupytext --sync KV_Cache.ipynb`.
- [x] Run `git diff --check`.
- [x] Verify notebook source markers used by the Colab bridge remain unique.
- [ ] Review generated configuration and manifests without running the local CPU
  pipeline as a substitute for Colab GPU validation.

### Phase 5 — Variant A extraction and training

- [x] Extract Variant A calibration vectors through the Colab bridge.
- [x] Audit all expected arrays, source/role quotas, deduplication report, and Drive
  backup before training.
- [x] Train Variant A key/value codebooks using adaptive group budgets.
- [x] Audit every head map, group budget, codebook/LUT artifact, static mask, and
  role-specific reconstruction report.
- [x] Run key-only, value-only, and combined-PQ validation on a locked set of unused,
  decontaminated HotpotQA-training examples.
- [x] Compare against the original held-out codebooks using identical sample IDs and
  prompts.

### Phase 6 — Variant A decision gate

- [ ] Confirm combined PQ retains at least 90% of the uncompressed Hotpot validation
  F1 and is no more than 5 absolute points below baseline.
- [ ] Confirm improvement is not confined to one answer type; report bridge,
  comparison, yes/no, and short-entity subsets separately.
- [ ] Confirm support/question-vector error or next-token KL improves even if global
  NMSE is unchanged.
- [ ] Repeat the promising configuration across at least three calibration seeds.
- [ ] If Variant A does not improve materially, stop expanding data and investigate
  selective precision or lower compression instead of proceeding blindly to B.

Seed-0 gate status: Variant A combined PQ scored `70.86735466116272` versus
baseline `76.59870129870131`, retaining `0.9251769737558754` of baseline but
missing the five-point-drop gate by `0.731346637538593` points. It materially
improved over the original held-out codebooks on identical prompts by
`+6.71783751164557` combined dynamic, `+4.162215740018937` key-only, and
`+2.4093137254901933` value-only. The limiting factor is key-side PQ: value-only
Variant A beat baseline, while key-only remained `5.571428571428584` points below
baseline. Do not advance to Variant B or a full LongBench-E run from this seed-0
result alone; next investigate key-selective precision, lower key compression, or
role-specific key handling before spending more GPU time on broader corpus
expansion.

### Phase 7 — Variant B extraction, training, and suite validation

- [ ] Extract and independently audit Variant B.
- [ ] Train and independently audit Variant B using the same adaptive budget rule.
- [ ] Evaluate Variant B on the same locked Hotpot validation set.
- [ ] Run a paired multi-task diagnostic with fixed sample IDs covering at least
  HotpotQA, Qasper, passage retrieval, summarization, and code.
- [ ] Compare Variant B against Variant A and the original held-out codebooks.
- [ ] Advance Variant B only if it preserves most of Variant A's Hotpot gain and
  reduces broader-task regressions.

### Phase 8 — final LongBench-E evaluation

- [ ] Lock the finalist and all hyperparameters before evaluating LongBench-E.
- [ ] Run the complete baseline/dynamic/static suite with durable per-dataset and
  per-mode checkpoints.
- [ ] Audit predictions, per-dataset scores, aggregate scores, and calibration
  provenance.
- [ ] Report only clean Variant A/B results as candidates for final claims; retain
  contaminated results as diagnostics only.
- [ ] Update `STATUS.md`, this plan, and `LONGBENCH_E_RESULTS.md` with exact command
  IDs, return codes, artifacts, and limitations.

## Evaluation Metrics and Advancement Gates

- Primary: HotpotQA token-level F1 retention versus the same model's uncompressed
  baseline.
- Secondary: per-task LongBench-E scores, next-token KL, attention-output error,
  role-specific K/V NMSE, and generation stability.
- Diagnostic: key-only versus value-only PQ to localize sensitivity.
- Variant A advances when it meets the 90%-of-baseline and five-point-drop gates on
  the locked clean validation set across three seeds.
- Variant B advances when it retains the material Hotpot improvement and gives a
  better suite-wide tradeoff than Variant A.
- A full benchmark claim requires all 13 LongBench-E datasets and a matching code-0
  bridge result; smoke scores are never final results.

## Expected Artifact Names

Use names that include both contamination mode and data variant, for example:

- `pq_training_data_longbench_e_clean_hotpot_a_4096`
- `pq_training_data_longbench_e_clean_suite_b_4096`
- `codebooks_64_128_64_longbench_e_clean_hotpot_a_4096_adaptive10k`
- `codebooks_64_128_64_longbench_e_clean_suite_b_4096_adaptive10k`
- variant-qualified prediction directories and aggregate/result CSV files.

Each codebook directory must use only its own `head_to_codebook_map.json`, cluster
cache, reconstruction report, and static masks.

## Active Command

- Stage: Variant C deterministic long-context extraction awaiting controller pickup
- Active command ID: `pq-qa-count-c-extract-20260810-233059`
- Colab bridge: `G:\My Drive\PQ_agent`
- Last pushed implementation commit: `cf6cf9e24d5db27da4a0b6aa7456ad0ce6bcd408`
- Current checkpoint: command
  `pq-scenario-a-a100-mini-100x128-bd-eval-20260809-163944` returned code 0 on
  `NVIDIA A100-SXM4-80GB` after `39530.790924315` seconds. The audited run used
  all 13 LongBench-E datasets, 100 deterministic random samples per dataset,
  seed `20260809`, max new tokens 128, baseline plus dynamic only, held-out
  `clean_hotpot_a` Variant A codebooks, and run tag
  `scenario_a_a100_100x128_bd`. Verified dataset-average scores were baseline
  `47.65967494764164`, dynamic `44.349905555086025`, delta
  `-3.3097693925556158`.
- Next action: restart or confirm the Colab bridge controller so it acknowledges the
  active extraction ID. After an exact code-0 result, independently audit Variant C
  before submitting codebook training.

## Completion Criteria

- Variants A and B are reproducible from pinned, decontaminated sources.
- Role-aware vector quotas survive extraction, assembly, and group sampling.
- Every group obeys `max(50K, 10K * heads)` or reports a verified unique-data
  shortfall without duplication.
- Variant A establishes the achievable clean Hotpot-specialized ceiling.
- Variant B is compared on identical samples and demonstrates the preferred
  LongBench-wide tradeoff.
- The final result is supported by audited GPU artifacts and complete benchmark
  outputs, not reconstruction NMSE or two-sample smoke results alone.
