# KV_Cache.py Notebook Map

`KV_Cache.py` is a Jupytext percent-format export of `KV_Cache.ipynb`. It is organized as a Colab workflow for downloading Qwen-family models, extracting K/V activation vectors, training product-quantization (PQ) codebooks, and evaluating dynamic vs static outlier-preserving KV-cache PQ.

## End-to-End Dependency Flow

1. Model download/setup cells create local model folders such as `qwen3_8B` or `/content/qwen3_8B`.
2. Calibration text or LongBench extraction cells generate per-layer/per-head `*_Train.npy` and `*_Test.npy` files under `pq_training_data` or `pq_training_data_longbench`.
3. Codebook training cells consume those `.npy` files and write grouped PQ codebooks plus `head_to_codebook_map.json`.
4. Evaluation cells load model weights and codebooks, inject a `DualPQManager` into custom Qwen attention, and compare baseline, dynamic-outlier PQ, and static-mask PQ.
5. Final helper/chat cells are exploratory and depend on earlier model/PQ classes or on a local Hugging Face model directory.

## Section 1: Initial Model Downloads and Basic Data Setup

Cells: lines 15-112

What it does:
- Contains a commented shell download for `minpeter/Qwen3-0.6B-Instruct`.
- Downloads `Qwen/Qwen2.5-1.5B-Instruct` into `qwen2.5_1.5B`.
- Reads `HF_TOKEN` from Colab userdata.
- Clears CUDA memory after previous attempts.
- Downloads `Qwen/Qwen3-8B` into `qwen3_8B`.
- Downloads `calibration.txt` from WikiText-2 test text, with a fallback dummy text.

Important functions/classes:
- No reusable functions/classes in this section.
- Uses `snapshot_download` from `huggingface_hub`.

Inputs:
- Hugging Face token from `os.environ["HF_TOKEN"]` / Colab userdata.
- Hugging Face model IDs.
- WikiText-2 URL for `calibration.txt`.

Outputs:
- Local model directories: `qwen2.5_1.5B`, `qwen3_8B`.
- Local `calibration.txt`.

Dependencies:
- Later pure-PyTorch model loading expects `config.json`, tokenizer files, and `.safetensors` under the model directory.
- Perplexity evaluation sections depend on `calibration.txt`.

## Section 2: Simple Pure-PyTorch Qwen Model and WikiText PQ Vector Extraction

Cells: lines 113-402

What it does:
- Defines a minimal Qwen causal LM in PyTorch.
- Loads safetensors without the `safetensors` package.
- Runs text through the model and captures key/value vectors from every layer/head.
- Saves sampled K/V activation vectors for PQ training.

Important functions/classes:
- `load_safetensors_pure(filepath)`: memory maps `.safetensors` and returns a tensor dictionary.
- `QwenRMSNorm`, `QwenRotaryEmbedding`, `QwenAttention`, `QwenMLP`, `QwenDecoderLayer`, `QwenModel`, `QwenForCausalLM`: custom Qwen implementation.
- `rotate_half(x)`: RoPE helper.
- `generate_pq_text_vectors(model, tokenizer, full_text, output_dir, num_chunks=100, chunk_size=512)`: collects and writes K/V vectors.

Inputs:
- Model files from `qwen3_8B`.
- Tokenizer from the model directory.
- `calibration.txt`.
- Config values such as number of chunks and chunk size.

Outputs:
- Per-head NumPy files, typically under `qwen3_8B/pq_training_data/keys` and `.../values`.
- Files are named like `L{layer}_H{head}_Train.npy` / `Test.npy`.

Dependencies:
- Requires the model download/setup section.
- Its output is consumed by the first PQ codebook trainer.

## Section 3: Deterministic Long-Context Data and LongBench Calibration Extraction

Cells: lines 403-1156

What it does:
- Replaces simple WikiText calibration with a LongBench-based extraction pipeline.
- Defaults to reportable task-held-out calibration against the full LongBench-E suite.
- Supports clean calibration variants selected with `PQ_CALIBRATION_VARIANT`,
  including Variant A `clean_hotpot_a`, with pinned training-source revisions,
  LongBench-E decontamination, role-aware token sampling, teacher-forced answer
  positions, and `position_index_{train,test}.jsonl` sidecars.
- Adds a separate deterministic generation cell for `clean_qa_count_c`. It expands
  clean QA prompts toward 4K tokens with lexically ranked training-only donor
  paragraphs and constructs exact synthetic passage-count examples without manual
  or model-based selection.
- Builds prompts with the same template, chat-template, and 4096-token middle-truncation order as evaluation.
- Captures post-`k_norm`, pre-RoPE key vectors and raw value vectors at sampled positions.
- Writes one resumable shard per document, then assembles document-level train/test splits.
- Writes a provenance manifest and optionally backs up outputs to Google Drive.

Important functions/classes:
- LongBench data/prompt helpers: `ensure_longbench_data`, `load_longbench_task`, `load_dataset2prompt`, `build_prompt`, `allocate_per_task`, `select_calibration_samples`.
- Deterministic generation helpers: `build_deterministic_donor_pool`,
  `rank_deterministic_donors`, `fit_qa_sample_to_long_context`,
  `make_deterministic_passage_count_sample`, and
  `generate_deterministic_long_context_calibration`.
- Custom model stack: `load_safetensors_pure`, `QwenRMSNorm`, `QwenRotaryEmbedding`, `QwenAttention`, `QwenMLP`, `QwenDecoderLayer`, `QwenModel`, `QwenForCausalLM`.
- Sharding/output helpers: `shard_path`, `extract_shards`, `assemble_and_save`.

Inputs:
- `/content/qwen3_8B` model directory.
- LongBench data from `zai-org/LongBench` or `THUDM/LongBench`.
- Prompt template JSON from the LongBench GitHub repo.
- Calibration config: `PQ_CALIBRATION_MODE`, `PQ_CALIBRATION_VARIANT`, `PQ_CALIBRATION_SEED`, `EVAL_TASKS`, `CALIB_TASK_WEIGHTS`, `NUM_CALIB_SAMPLES`, `MAX_INPUT_LENGTH`, `VECTORS_PER_HEAD`, `TEST_FRACTION`.

Outputs:
- Shards: `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/_shards/*.npz`.
- Training/test tensors:
  - `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/keys/L{layer}_H{head}_Train.npy`
  - `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/keys/L{layer}_H{head}_Test.npy`
  - matching `values` files.
- `calibration_manifest.json`.
- `position_index_train.jsonl` and `position_index_test.jsonl` role/source sidecars.
- Optional Drive backup at `/content/drive/MyDrive/qwen3_8B/pq_training_data_longbench_e_{mode}_4096`.

Dependencies:
- Requires downloaded Qwen3 8B model files and tokenizer.
- Produces the preferred inputs for the improved LongBench-aware PQ codebook trainer.
- The trainer validates `calibration_manifest.json` to avoid accidentally using contaminated or mismatched calibration data.

Command controls:
- `PQ_CALIBRATION_MODE` selects held-out, matched, or contaminated extraction.
- `PQ_CALIBRATION_VARIANT=clean_hotpot_a` selects the Hotpot-specialized clean Variant A output path.
- `PQ_CALIBRATION_VARIANT=clean_qa_count_c` selects the deterministic long-context
  QA/counting calibration path.
- `PQ_LONGBENCH_DATASETS` accepts a comma-separated LongBench-E dataset subset for
  paired smoke evaluation.
- `PQ_LONGBENCH_TEST_SAMPLES_PER_DATASET` sizes diagnostic evaluation runs.
- `PQ_LONGBENCH_RUN_TAG` isolates diagnostic output directories and CSV names.

## Section 4: Quantized Hugging Face Model Loading Experiments

Cells: lines 1157-1255

What it does:
- Notes installing `bitsandbytes`.
- Loads Qwen3 8B or Qwen2.5 7B via `AutoModelForCausalLM` using 4-bit NF4 quantization.
- Contains cleanup cells to delete large variables and clear CUDA memory.

Important functions/classes:
- `BitsAndBytesConfig`.
- Hugging Face `AutoModelForCausalLM`.

Inputs:
- `qwen3_8B` or `Qwen/Qwen2.5-7B-Instruct`.
- CUDA-capable Colab runtime.

Outputs:
- In-memory quantized Hugging Face model and tokenizer.
- `qwen3_8B_quantized` local directory in one experiment.

Dependencies:
- Mostly standalone exploration.
- Memory cleanup supports subsequent sections.

## Section 5: First PQ Codebook Training Pipeline

Cells: lines 1256-2109

What it does:
- Trains grouped PQ codebooks for keys and values from `pq_training_data`.
- Clusters heads using diagonal or full Wasserstein distances over activation moments.
- Samples raw activation vectors per group.
- Runs batched CUDA k-means for all subvectors.
- Saves fine codebooks, coarse codebooks, lookup tables, group summaries, and head maps.

Important functions/classes:
- Setup/filesystem: `set_seed`, `setup_torch`, `get_device`, `cleanup`, `ensure_dir`, `parse_head_name`, `load_train_files`, `remove_old_codebook_files`.
- Clustering: `load_diag_moments_from_files`, `diag_wasserstein_distance`, `load_full_moments_from_files`, `safe_sym_sqrt`, `full_wasserstein_distance`, `make_dense_cluster_groups`, `get_or_create_groups`.
- Data/k-means: `load_raw_data_for_group`, `init_centroids_batched`, `assign_chunk_gemm`, `batched_kmeans_cuda`, `train_all_subvector_codebooks`.
- Output/validation: `save_group_codebooks`, `write_head_map`, `write_cluster_summary`, `validate_generated_codebooks`, `discover_complete_groups`, `validate_runtime_compatibility`, `run_pipeline_for_tensor_type`.

Inputs:
- `/content/qwen3_8B/pq_training_data/{keys,values}/*_Train.npy`.
- Training config: `CLUSTERS`, `DIMS`, `NUM_SUBVECTORS`, `CODEWORDS`, `TARGET_TRAIN_SIZE`, `MIN_VECTORS_PER_HEAD`.

Outputs:
- `/content/qwen3_8B/codebooks_{NUM_SUBVECTORS}_{CODEWORDS}_{CLUSTERS}/{keys,values}/`.
- `group_*_sub_*_fine.txt`, `group_*_sub_*_coarse.txt`, `group_*_sub_*_lut.txt`.
- `head_to_codebook_map.json`.
- `cluster_summary.json`.
- Optional Drive copy.

Dependencies:
- Consumes Section 2 extraction outputs.
- Its codebooks are consumed by the local perplexity evaluation section.

## Section 6: Copy Model Files from Google Drive

Cells: lines 2110-2139

What it does:
- Copies core model files from `/content/drive/MyDrive/qwen3` to `/content/qwen3`.
- Copies `*.json`, `*.safetensors`, and `*.txt`.
- Prints local file sizes.

Important functions/classes:
- No reusable functions/classes.

Inputs:
- Google Drive directory `/content/drive/MyDrive/qwen3`.

Outputs:
- Local model directory `/content/qwen3`.

Dependencies:
- Supports later local chat and model loading cells that expect `/content/qwen3`.

## Section 7: Improved LongBench-Aware PQ Codebook Training Pipeline

Cells: lines 2140-3549

What it does:
- Trains codebooks from the LongBench extraction output instead of WikiText calibration.
- Validates calibration provenance before training.
- Uses balanced group-size constraints for head clusters.
- Uses adaptive per-group training budgets:
  `max(50,000, 10,000 * heads_in_group)`, with no duplication when unique vectors
  are sufficient and stratified sampling from role/source sidecars when available.
- Improves k-means with k-means++ initialization, multiple restarts, per-subvector best-restart selection, relative tolerance, and inertia reporting.
- Adds a held-out reconstruction MSE check against `*_Test.npy` vectors, with optional baseline comparison.
- Learns static outlier masks from calibration vectors rather than evaluation samples.

Important functions/classes:
- Manifest validation: `load_and_validate_calibration_manifest`.
- Clustering and data loading: same families as Section 5, with balanced `make_dense_cluster_groups` and duplication controls.
- K-means: `init_centroids_batched`, `compute_inertia`, `_batched_kmeans_cuda_single`, `batched_kmeans_cuda`, `train_all_subvector_codebooks`.
- MSE check: `load_codebook_stack`, `pq_reconstruct_torch`, `run_mse_check`.
- Output/validation: `save_group_codebooks`, `write_head_map`, `write_cluster_summary`, `validate_generated_codebooks`, `validate_runtime_compatibility`, `run_pipeline_for_tensor_type`.

Inputs:
- `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/{keys,values}/*_Train.npy`.
- `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/{keys,values}/*_Test.npy`.
- `/content/qwen3_8B/pq_training_data_longbench_e_{mode}_4096/calibration_manifest.json`.
- `/content/qwen3_8B/pq_training_data_longbench_e_{variant}_4096/position_index_train.jsonl` for clean Variant A/B stratification.
- Optional baseline codebooks at `/content/qwen3_8B/codebooks_64_128_64`.

Outputs:
- Versioned LongBench-E codebooks under:
  - `/content/qwen3_8B/codebooks_64_128_64_longbench_e_{mode}_4096_balanced_kpp_noclip/{keys,values}/`
  - `/content/qwen3_8B/codebooks_64_128_64_longbench_e_clean_hotpot_a_4096_adaptive10k/{keys,values}/`
- Codebook text files, head maps, cluster summaries, and calibration-derived `static_outlier_masks.json`.
- Per-side `codebook_mse_report.json`.
- Optional Drive copy.

Dependencies:
- Consumes Section 3 outputs.
- Produces the strongest codebook artifacts for the LongBench evaluation section.
- The MSE check depends on document-disjoint `*_Test.npy` files from Section 3.

## Section 8: Local Calibration Perplexity Evaluation with Dynamic vs Static Outliers

Cells: lines 3550-5021

What it does:
- Loads generated codebooks and injects quantization into a custom Qwen attention implementation.
- Compares baseline perplexity against dynamic-outlier PQ and static-mask PQ on `calibration.txt`.
- Tracks whether selected outlier dimensions are stable across chunks and heads.
- Learns static top-k masks from a dynamic run, then re-evaluates with those masks.
- Saves results and tracker reports.

Important functions/classes:
- Console/helpers: `banner`, `section`, `print_kv_table`, `print_df`, `overall_compression_ratio`, `side_dir_from_cfg`.
- Codebook loading/validation: `list_available_codebook_groups`, `validate_group_has_all_fine_files`, `load_head_to_codebook_map`, `resolve_head_map_to_available_groups`.
- PQ runtime: `TorchGroupProcessor`, `OutlierReuseTracker`, `DualPQManager`.
- Model runtime: `load_safetensors_pure`, `QwenRMSNorm`, `QwenRotaryEmbedding`, `repeat_kv`, `QwenAttention`, `QwenMLP`, `QwenDecoderLayer`, `QwenModel`, `QwenForCausalLM`, `set_model_pq_manager`.
- Eval/output: `calculate_perplexity`, `load_model_weights`, `validate_single_config`, `save_static_masks`, `save_tracker_csvs`.

Inputs:
- `/content/qwen3_8B/config.json`, tokenizer, and `.safetensors`.
- Codebooks from configured `KEY_CONFIG` and `VALUE_CONFIG`.
- `calibration.txt`.

Outputs:
- `pq_dynamic_static_result.csv`.
- `outlier_static_top3_masks.json` and `.csv`.
- Dynamic/static outlier reuse CSVs.
- Console tables for baseline, dynamic PQ, and static PQ perplexity.

Dependencies:
- Requires trained codebooks and matching `head_to_codebook_map.json`.
- The static-mask run depends on dynamic-run tracker results.

## Section 9: Standalone LongBench Evaluation Harness

Cells: lines 5022-7018

What it does:
- Installs missing evaluation dependencies.
- Defines a standalone full LongBench-E evaluation flow for baseline, dynamic-outlier PQ, and calibration-derived static-mask PQ.
- Loads LongBench data directly from Parquet shards to avoid legacy dataset-script issues.
- Generates answers with cache-aware greedy decoding.
- Scores predictions with task-specific LongBench metrics.
- Saves per-sample predictions, per-dataset summaries, aggregate results, and outlier reports.

Important functions/classes:
- Dependency install block: `_REQUIRED_PACKAGES`, `_missing_specs`.
- Reuses PQ runtime families: `TorchGroupProcessor`, `OutlierReuseTracker`, `DualPQManager`.
- Model stack with cache support: `QwenRMSNorm`, `QwenRotaryEmbedding`, `apply_rotary_pos_emb`, `repeat_kv`, `QwenAttention`, `QwenMLP`, `QwenDecoderLayer`, `QwenModel`, `QwenForCausalLM`.
- LongBench prompts/metrics: `DATASET2PROMPT`, `DATASET2MAXLEN`, `DATASET2METRIC`, `normalize_answer`, `qa_f1_score`, `rouge_score`, `classification_score`, `retrieval_score`, `count_score`, `code_sim_score`, `score_longbench_prediction`.
- Data/generation/eval: `middle_truncate_ids`, `build_longbench_prompt`, `eos_token_ids`, `greedy_generate`, `_download_longbench_parquet_paths`, `load_longbench_examples`, `preload_longbench_data`, `save_jsonl`, `evaluate_longbench_mode`, `comparison_rows`.
- Model/output helpers: `load_model_weights`, `validate_single_config`, `save_static_masks`, `save_tracker_csvs`.

Inputs:
- `/content/qwen3_8B` model files.
- Codebooks from `KEY_CONFIG` and `VALUE_CONFIG`.
- LongBench or LongBench-E Parquet test data from `zai-org/LongBench` or `THUDM/LongBench`.
- Runtime controls such as `TEST_MODE`, `USE_LONG_BENCH_E`, `MAX_INPUT_TOKENS`, and `MAX_NEW_TOKENS_CAP`.

Outputs:
- Prediction JSONL files under `longbench_pq_outputs/{baseline,dynamic,static}/`.
- `longbench_pq_dynamic_static_result.csv`.
- `longbench_pq_summary_by_dataset.csv`.
- `longbench_pq_outputs/aggregate_result.csv`.
- Static mask JSON/CSV and dynamic/static outlier reuse CSVs.

Dependencies:
- Requires trained codebooks and model files.
- Dynamic and static modes use matching 64-bank, 128-codeword key/value codebooks; static masks come from held-out calibration.
- Generation uses model KV-cache support, so this section supersedes the simpler fixed-window perplexity evaluation for LongBench-style tasks.

## Section 10: Exploratory Outlier Tracking With Hugging Face Wrapper

Cells: lines 7019-7175

What it does:
- Attempts a lighter outlier-dimension consistency analysis using a Hugging Face `AutoModelForCausalLM` wrapper.
- Defines simplified helper functions and a wrapper `QwenForCausalLM`.
- Runs a short perplexity pass and prints observed outlier-dimension frequencies.

Important functions/classes:
- `load_model_weights`, `set_model_pq_manager`, `calculate_perplexity`.
- Simplified `QwenForCausalLM` wrapper around Hugging Face `AutoModelForCausalLM`.
- Local helper `analyze_side`.

Inputs:
- Earlier globals: `MODEL_DIR`, `KEY_CONFIG`, `VALUE_CONFIG`, `AUTO_REMAP_MISSING_GROUPS`, `CALIBRATION_FILE`, `WINDOW_SIZE`, `DualPQManager`, `get_device`.
- Hugging Face model files.

Outputs:
- Console-only consistency analysis.

Dependencies:
- Depends on the PQ runtime definitions from earlier evaluation sections.
- The comments note that tracking may be empty unless the model forward path actually calls `pq_manager.quantize_tensor`.

## Section 11: Local Qwen3 Chat UI

Cells: lines 7176-7264

What it does:
- Loads `/content/qwen3` with Hugging Face `AutoModelForCausalLM`.
- Defines a simple Colab form-style chat loop.
- Maintains `messages` across reruns and renders conversation with IPython Markdown.

Important functions/classes:
- No reusable custom functions/classes.
- Uses `AutoTokenizer`, `AutoModelForCausalLM`, `IPython.display`.

Inputs:
- Local model path `/content/qwen3`.
- User form variables `User_Input` and `Clear_History`.

Outputs:
- In-notebook chat transcript display.
- In-memory `messages` conversation history.

Dependencies:
- Can use files copied by Section 6.
- Independent of the PQ codebook/evaluation pipeline.

## Major Artifacts

- Model folders:
  - `qwen2.5_1.5B`
  - `qwen3_8B`
  - `/content/qwen3_8B`
  - `/content/qwen3`
- Calibration data:
  - `calibration.txt`
  - `/content/qwen3_8B/pq_training_data`
  - `/content/qwen3_8B/pq_training_data_longbench`
- Codebooks:
  - `/content/qwen3_8B/codebooks_64_128_64`
  - `/content/qwen3_8B/codebooks_64_128_64_longbench_held_out_balanced_kpp_noclip`
- Evaluation outputs:
  - `pq_dynamic_static_result.csv`
  - `longbench_pq_dynamic_static_result.csv`
  - `longbench_pq_summary_by_dataset.csv`
  - `longbench_pq_outputs/`
  - `outlier_static_top3_masks*.json`
  - `outlier_*_reuse_*.csv`

## Repeated Definitions

Several functions/classes are redefined in later cells with improved behavior. In notebook execution order, the latest definition wins.

- `load_safetensors_pure` appears in extraction and evaluation sections.
- Qwen model classes appear in extraction, local perplexity evaluation, and LongBench evaluation sections.
- PQ trainer utilities appear first for WikiText-style data, then again for LongBench calibration with stronger validation and MSE checks.
- `calculate_perplexity`, `load_model_weights`, and `set_model_pq_manager` appear in multiple evaluation experiments.

When running the notebook top-to-bottom, later cells may shadow earlier versions. When running individual cells out of order, make sure the matching helper/class definitions from the same section have already been executed.
