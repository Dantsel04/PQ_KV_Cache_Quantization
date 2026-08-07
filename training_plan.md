# HotpotQA-Focused PQ Calibration Training Plan

## Objective

Increase HotpotQA answer F1 under KV-cache PQ without using any LongBench-E
evaluation example for reportable codebook training. The immediate quality target is
to retain at least 90% of the uncompressed HotpotQA baseline score on a sufficiently
large, locked evaluation set, rather than optimizing reconstruction NMSE alone.

This plan concerns PQ calibration vectors and codebook training data. It does not
fine-tune Qwen3-8B.

## What the Current Data Pipeline Is Actually Training On

The held-out calibration manifest contains 720 training documents and 36,000 vectors
per layer/head. Each document contributes exactly 50 token positions.

| Held-out task family | Training documents | Share |
|---|---:|---:|
| English multi-hop QA: MuSiQue | 133 | 18.5% |
| English single-document QA: NarrativeQA | 134 | 18.6% |
| English query-focused summarization: QMSum | 90 | 12.5% |
| Chinese QA/retrieval/summarization/classification | 329 | 45.7% |
| Other | 34 | 4.7% |

Consequently, 16,450 of the 36,000 vectors per head come from Chinese prompts, while
only 6,650 come from the closest English multi-hop proxy, MuSiQue. There are no clean
HotpotQA training-split examples in the held-out pool.

The contaminated run is not a strong Hotpot-focused calibration experiment. It used
only 52 HotpotQA training documents, or 2,600 of 36,000 pre-subsampling vectors per
head. It mixed HotpotQA equally with twelve unrelated evaluation tasks. Its manifest
also shows that one of the two HotpotQA smoke examples entered the calibration-train
split, so its score cannot be treated as held-out evidence.

Other important properties of the current extraction/training flow are:

- 92.5% of held-out training prompts reach the 4,096-token cap.
- Only 50/4,096, or about 1.2%, of positions are sampled from a capped prompt.
- Positions are sampled uniformly. The extractor does not distinguish questions,
  supporting facts, bridge entities, titles, distractors, or prompt boilerplate.
- A 50-token region has roughly a 54% probability of contributing no vector at all
  under this sampling policy.
- Document/task/position-role identity is discarded during assembly. The trainer
  sees one shuffled array per layer/head and cannot rebalance important roles.
- The trainer randomly subsamples again to a 50,000-vector cap per head group. Merely
  adding more source documents will not guarantee that their vectors reach k-means.
- Calibration contains prompt-prefill states but no teacher-forced answer/decode
  states, even though generated-token K/V states are also quantized at evaluation.

## Why MuSiQue Alone Is an Incomplete HotpotQA Proxy

Inspection of the exact LongBench-E HotpotQA data and the selected held-out MuSiQue
documents found:

| Property | HotpotQA-E | Selected MuSiQue |
|---|---:|---:|
| Median passages | 10 | 9 |
| One-token answers | 34.7% | 13.3% |
| Yes/no answers | 11.0% | 0% |
| `or`-comparison questions, approximate | 11.3% | 1.3% |
| Literal answer present in context | 85.7% | 85.3% |

MuSiQue supplies useful multi-hop structure, but it substantially underrepresents
HotpotQA's comparison, yes/no, and very-short entity-answer regimes. HotpotQA's
official training data includes sentence-level supporting facts, which makes it much
better suited to importance-aware activation sampling.

## Recommended Dataset Variants

### Variant A: clean Hotpot-specialized ceiling

Train this first to determine how much of the failure is caused by calibration data.
Use 800 source documents and keep the initial total at 40,000 vectors per head so the
result is directly comparable with the existing codebooks.

| Source | Documents | Share | Purpose |
|---|---:|---:|---|
| Official HotpotQA training split | 400 | 50% | Exact multi-hop, distractor, comparison, and answer-type match |
| MuSiQue training split | 160 | 20% | Diverse two-to-four-hop reasoning |
| 2WikiMultihopQA training split | 120 | 15% | Additional bridge/comparison structure |
| English Wikipedia retrieval with hard distractors | 80 | 10% | Attention competition across similar passages |
| General English long-form QA | 40 | 5% | Prevent an excessively narrow activation distribution |

This is the recommended candidate for a dramatic HotpotQA improvement. It should use
a separate mode-qualified directory and must never reuse the current contaminated
label.

### Variant B: suite-safe mixed codebook

If one codebook set must serve all LongBench-E tasks, reduce the specialized share but
remove the present 45.7% Chinese skew unless Chinese performance is an explicit goal:

- 30% clean HotpotQA training split;
- 20% MuSiQue and clean 2Wiki training splits;
- 20% English single-document QA and scientific QA;
- 15% English retrieval/counting;
- 10% summarization;
- 5% code or Chinese tasks, selected according to the desired benchmark weighting.

First establish Variant A's ceiling. Only then trade some HotpotQA quality for suite
breadth. If runtime permits dataset-specific codebook routing, retain Variant A for
multi-hop QA instead of forcing one compromise distribution on every task.

### Variant C: strict task-held-out control

For an experiment that forbids all HotpotQA-source calibration, use MuSiQue training
data plus synthetic Hotpot-shaped contexts:

- convert each MuSiQue example to ten passages;
- preserve its two-to-four supporting passages;
- add topically similar Wikipedia distractors;
- randomize supporting-passage order across the retained first/last 2,048-token
  regions;
- construct balanced entity, comparison, and yes/no question subsets;
- use the exact HotpotQA evaluation prompt wrapper.

This control separates gains from Hotpot-specific source data versus gains from
context structure and position sampling.

## Decontamination Contract

LongBench documents are based on validation/test examples from their source datasets.
Use only official training splits for the clean in-domain corpus and reject an example
if any of the following matches the pinned LongBench-E evaluation set:

1. source ID or normalized question;
2. normalized question plus answer;
3. pair/set of supporting Wikipedia titles;
4. exact supporting-sentence or supporting-paragraph hash;
5. near-duplicate prompt/context according to a documented MinHash threshold.

Write all rejected IDs and reasons into the calibration manifest. Split remaining
examples by question and supporting-page cluster, not merely by generated prompt, so
near-identical Wikipedia evidence cannot cross calibration-train and validation.

## Replace Uniform Token Sampling With Role-Aware Sampling

Keep the first experiment at 50 unique positions per document, allocated as follows:

| Position role | Positions/document | Selection rule |
|---|---:|---|
| Supporting facts and local neighborhoods | 15 | Gold support sentences plus nearby tokens |
| Bridge entities, article titles, paragraph boundaries | 8 | Entity/title spans joining the reasoning chain |
| Question and final instruction suffix | 8 | Guarantee question representation in every document |
| Teacher-forced answer/decode states | 4 | Cover generated-token K/V distribution |
| Uniform context and distractors | 15 | Preserve the background activation distribution |

For examples without support annotations, use a baseline teacher pass and sample
prompt positions according to attention mass from teacher-forced answer tokens, mixed
with the uniform quota. Do not replace all uniform samples: the quantizer must still
represent distractors and ordinary context.

Save `task`, `source_id`, `position_role`, token offset, paragraph/title ID, and split
beside every shard. Either write separate arrays per role or a sidecar index that the
trainer uses for stratified sampling. Do not shuffle away this metadata before the
trainer has enforced its quotas.

## Context Construction

- Use the exact evaluation prompt and Qwen chat-template order.
- Construct examples at the actual 4,096-token inference length rather than relying
  on arbitrary truncation of much longer documents.
- Use approximately ten passages, matching HotpotQA's distractor setting.
- Place supporting passages across early, middle, and late locations before
  truncation, but verify that required evidence survives the evaluator's first/last
  truncation policy.
- Prefer hard, topically related distractors. Random unrelated text does not recreate
  the close attention competition that makes HotpotQA sensitive to K/V error.
- Balance bridge and comparison questions and include about 10% yes/no examples and
  at least 30% one-token-answer examples to approximate HotpotQA-E.

## Trainer Changes Required for the New Data to Matter

These are data-delivery changes rather than a new PQ algorithm:

1. Stratify `load_raw_data_for_group` by task and position role. The current random
   group cap can erase the intended mixture.
2. Record the realized task/role counts used by every head group.
3. Raise `TARGET_TRAIN_SIZE` only after the fixed-40K experiment. If extraction grows
   to 75K-100K vectors per head while the group cap remains 50K, much of the extra
   data will never reach k-means.
4. Report NMSE separately for support, question, distractor, and decode vectors.
5. Continue using document-disjoint reconstruction data, but add a task-level
   validation set from unused official training examples.

## Experiment Sequence

1. **Lock evaluation before training.** Create a 100-200-example validation set from
   unused HotpotQA training examples, decontaminated from LongBench-E. Preserve a
   stable 4K prompt and sample-ID list for every experiment.
2. **Run localization baselines.** Measure baseline, key-only PQ, value-only PQ, and
   key+value PQ. This determines whether data changes should prioritize key or value
   roles; current value NMSE is the larger suspect but is not causal proof.
3. **Train Variant A with the current 40K-vector budget.** This isolates corpus mix
   and position selection from raw data volume.
4. **Ablate position sampling.** Compare uniform versus role-aware sampling using the
   same documents and seeds.
5. **Add decode states.** Compare role-aware prefill-only data against the full
   role-aware mixture.
6. **Scale only if needed.** Increase to 75K-100K vectors per head and increase the
   per-group training cap proportionally if the fixed-budget candidate improves but
   remains below target.
7. **Build the suite-safe mixture.** Test Variant B only after the Hotpot-specialized
   ceiling is known.
8. **Run LongBench-E once per finalist.** Do not repeatedly tune against its 300
   HotpotQA examples.

## Metrics and Gates

A candidate advances only if it passes all applicable gates:

- HotpotQA validation PQ F1 is at least 90% of its own uncompressed baseline and no
  more than 5 absolute points below baseline.
- Improvement appears across bridge, comparison, and yes/no subsets, not just one
  answer type.
- At least three calibration seeds show the same direction of improvement.
- Support-vector and question-vector NMSE improve or remain stable even if global
  NMSE changes little.
- Next-token KL divergence or attention-output error on held-out Hotpot-style prompts
  improves; global vector NMSE alone is insufficient.
- The suite-safe candidate does not cause unacceptable regressions on Qasper and
  passage retrieval.
- All final claims use a complete, matching code-0 Colab result and audited prediction
  records.

## Decision Rule

If Variant A plus role-aware sampling does not produce a large improvement, stop
expanding the corpus. That outcome would indicate that the 64-bank/128-codeword,
three-outlier compression setting is the limiting factor rather than calibration
coverage. The next work should then be selective precision: more preserved outliers,
less compression for sensitive layers/heads, or separate key/value budgets.

## Final Recommendation

The first new training run should use a decontaminated, Hotpot-specialized 800-document
corpus with 50% official HotpotQA training data and role-aware 50-position sampling.
Keep the total vector budget fixed at 40K per head for a clean comparison. The present
held-out corpus is too Chinese-heavy and contains too little English multi-hop data;
the present uniform sampler also spends almost all of its budget on generic context
and distractor tokens. Correcting both issues is much more likely to help than merely
increasing the number of randomly sampled LongBench documents.

## Evidence Sources

- Repository artifacts: `KV_Cache.py`, `LONGBENCH_E_RESULTS.md`, and the held-out
  and contaminated `calibration_manifest.json` files under the Drive artifact roots.
- [LongBench task definitions and dataset construction](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md).
- [Official HotpotQA dataset and downloads](https://hotpotqa.github.io/).
- [Official HotpotQA paper](https://aclanthology.org/D18-1259/).
- [Official MuSiQue repository](https://github.com/stonybrooknlp/musique).
