# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="luVgrjwCoD8o" outputId="2dc2c096-0fb4-45a9-97a1-73e6a950c21c"
# !mkdir -p qwen3
# !wget https://huggingface.co/minpeter/Qwen3-0.6B-Instruct/resolve/main/config.json -O qwen3/config.json
# !wget https://huggingface.co/minpeter/Qwen3-0.6B-Instruct/resolve/main/model.safetensors -O qwen3/model.safetensors
# !wget https://huggingface.co/minpeter/Qwen3-0.6B-Instruct/resolve/main/vocab.json -O qwen3/vocab.json
# !wget https://huggingface.co/minpeter/Qwen3-0.6B-Instruct/resolve/main/merges.txt -O qwen3/merges.txt

# %% [markdown] id="6593b389"
# ### Download Qwen3-8B

# %% id="6S17_Bk7wslm"
import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

# %% id="d16bb8c6"
import torch
import gc

# 1. Clear memory from the previous failed run
if 'model' in locals():
    del model
if 'state_dict' in locals():
    del state_dict

gc.collect()
torch.cuda.empty_cache()
print(f"GPU Memory after clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# %% colab={"base_uri": "https://localhost:8080/", "height": 182, "referenced_widgets": ["938b21ea424343cdb29f56f0af053b5f", "66a01eb7f3fb474da0420632a8e188b1", "785d02cb94e34b469edfae3467c349cd", "3cdfcd2dadb84a1fb12ecc414f22dec8", "7e6cb1fdc9524697af45ed925d900157", "9a9312b4a79143c7ac2dc84f3a990606", "624ebd19105a42bea4fc4d75efca120a", "0e0d4cf975634da99ae4888ce556527b", "a0395bbed05d49f5bf796caf5d61dc0c", "540102e549274c84b9c12cdc40dbb889", "c2bf305511a84b92ab9dea7ba8aa9681", "d61601e3f4554a47910206ad12e3ff19", "217b2356a15448febc7641338c08d551", "7873b3f54bca4f32b4b766513948d425", "359cea1e5ea04cf7a68861bb13ac2e87", "cda0a2d5eb2b42adb93c7634d7a03423", "05ff9186012f4a76877c3b090a786ddc", "5efe508548814a62b7807153323a743a", "3dcd2dd4ca2143e89179eb8a53cb5e1d", "43eba541394142939ab331530defa503", "0c869caadbd34a039576ce888a116eaa", "98ca1e8460894cc9b1918f0f9eec3cee", "8470fc9452fa4e2c92cd8df07fef5cb5", "83681a3fb0a84769b2b0e8b2c9bfb86a", "303ec031856e448b964256139d7ad1cf", "e7ec014eb46447a3b6cf88b931fa91c5", "4e04afd6316e4877935f3f5424d690f8", "abd908a7de9d4fca93b6e39c521e0cad", "9d71ca6d83b74cdcb42a2cd0930cd7ce", "4a4b78d943e6424d90357ed93728865b", "2aeb5a7919dd40b98a4e248b7d901522", "3e1a439a8f7042f68c617f307785fc6d", "57b19ce0c55b4bae9d82a78db5039b0a"]} id="9b24c364" outputId="0844ef8a-ff88-4401-a5ec-e6bd15ba1091"
from huggingface_hub import snapshot_download
import os

# Download only the official Qwen3 8B model
model_id = "Qwen/Qwen3-8B"
local_dir = "qwen3_8B"

print(f"Downloading {model_id}...")
try:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=os.environ.get('HF_TOKEN')
    )
    print(f"Qwen3 8B downloaded successfully to: {path}")
    # List the contents to verify
    print("\nContents of qwen3_8B:")
    os.system(f"ls -lh {local_dir}")
except Exception as e:
    print(f"Error during download: {e}")

# %% colab={"base_uri": "https://localhost:8080/"} id="bda0acd5" outputId="ecfa749e-6c4f-4619-fb52-6984b539cbc2"
import os
import urllib.request

test_file = "calibration.txt"
if not os.path.exists(test_file) or os.path.getsize(test_file) == 0:
    print(f"Downloading {test_file}...")
    try:
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
        urllib.request.urlretrieve(url, test_file)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating dummy fallback data...")
        with open(test_file, "w") as f:
            f.write("This is a fallback calibration text since the download failed. " * 1000)
else:
    print(f"{test_file} already exists.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Ea7VJdMIwtZt" outputId="a907d047-3a4f-4b9d-99c3-10bb00086e4a"
import os
import glob
import json
import math
import struct
import mmap
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

def load_safetensors_pure(filepath):
    with open(filepath, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
        offset = 8 + header_size

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tensors = {}
        for k, v in header.items():
            if k == '__metadata__': continue
            dtype = v['dtype']
            shape = v['shape']
            data_offsets = v['data_offsets']
            start = offset + data_offsets[0]
            end = offset + data_offsets[1]
            raw_data = mm[start:end]

            if dtype == 'F32': np_dtype = np.float32
            elif dtype == 'F16': np_dtype = np.float16
            elif dtype == 'BF16':
                t = torch.frombuffer(bytearray(raw_data), dtype=torch.int16).view(torch.bfloat16).clone()
                tensors[k] = t.reshape(shape)
                continue
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            tensors[k] = torch.from_numpy(np.frombuffer(raw_data, dtype=np_dtype).copy()).reshape(shape)
        mm.close()
    return tensors

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class QwenRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.get('num_key_value_heads', self.num_heads)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.get('attention_bias', False))
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.get('attention_bias', False))
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.get('attention_bias', False))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.get('attention_bias', False))

        self.q_norm = QwenRMSNorm(self.head_dim, eps=config.get('rms_norm_eps', 1e-6))
        self.k_norm = QwenRMSNorm(self.head_dim, eps=config.get('rms_norm_eps', 1e-6))
        self.rotary_emb = QwenRotaryEmbedding(self.head_dim, base=config.get('rope_theta', 1000000.0))

    def forward(self, hidden_states, position_ids):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        present_key_value = (key_states.detach().cpu(), value_states.detach().cpu())

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        if self.num_key_value_heads != self.num_heads:
            num_key_value_groups = self.num_heads // self.num_key_value_heads
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = QwenRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, hidden_states, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv

class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([QwenDecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = QwenRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        all_kvs = []
        for layer in self.layers:
            hidden_states, present_kv = layer(hidden_states, position_ids)
            all_kvs.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_kvs

class QwenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, input_ids):
        hidden_states, all_kvs = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits, all_kvs

def generate_pq_text_vectors(model, tokenizer, full_text, output_dir, num_chunks=100, chunk_size=512):
    os.makedirs(f"{output_dir}/keys", exist_ok=True)
    os.makedirs(f"{output_dir}/values", exist_ok=True)

    print(f"Encoding text into tokens...")
    all_tokens = tokenizer.encode(full_text)
    print(f"Encoding finished. Total tokens: {len(all_tokens)}")

    num_layers = len(model.model.layers)
    num_kv_heads = model.model.layers[0].self_attn.num_key_value_heads

    layer_head_k = {l: {h: [] for h in range(num_kv_heads)} for l in range(num_layers)}
    layer_head_v = {l: {h: [] for h in range(num_kv_heads)} for l in range(num_layers)}

    device = next(model.parameters()).device
    model.eval()

    print(f"Starting inference for {num_chunks} chunks...")
    chunks_processed = 0
    with torch.no_grad():
        for i in range(0, len(all_tokens), chunk_size):
            if chunks_processed >= num_chunks: break
            chunk = all_tokens[i : i + chunk_size]
            if len(chunk) < chunk_size: continue

            input_ids = torch.tensor([chunk], device=device)
            _, all_kvs = model(input_ids)

            for l, (k, v) in enumerate(all_kvs):
                for h in range(num_kv_heads):
                    layer_head_k[l][h].append(k[0, h].view(-1, k.shape[-1]).float().cpu().numpy())
                    layer_head_v[l][h].append(v[0, h].view(-1, v.shape[-1]).float().cpu().numpy())

            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f"Chunk progress: {chunks_processed}/{num_chunks}")

    print("Saving extracted vectors to disk...")
    for l in range(num_layers):
        for h in range(num_kv_heads):
            k_data = np.concatenate(layer_head_k[l][h], axis=0)
            v_data = np.concatenate(layer_head_v[l][h], axis=0)
            split_idx_k = int(len(k_data) * 0.9)
            split_idx_v = int(len(v_data) * 0.9)
            np.save(f"{output_dir}/keys/L{l}_H{h}_Train.npy", k_data[:split_idx_k])
            np.save(f"{output_dir}/keys/L{l}_H{h}_Test.npy", k_data[split_idx_k:])
            np.save(f"{output_dir}/values/L{l}_H{h}_Train.npy", v_data[:split_idx_v])
            np.save(f"{output_dir}/values/L{l}_H{h}_Test.npy", v_data[split_idx_v:])
        print(f"Finished Layer {l}")

if __name__ == "__main__":
    import gc

    model_dir = "qwen3_8B"
    pq_output_dir = f"{model_dir}/pq_training_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.bfloat16)

    print("Initializing script...")
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config = json.load(f)

    model = QwenForCausalLM(config)

    state_dict = {}
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        print(f"Reading state dict from {f}...")
        state_dict.update(load_safetensors_pure(f))

    if 'lm_head.weight' not in state_dict and 'model.embed_tokens.weight' in state_dict:
        print("Tying lm_head.weight to model.embed_tokens.weight to save memory...")
        model.lm_head.weight = model.model.embed_tokens.weight

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Moving model to {device}...")
    model.to(device)

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    with open("calibration.txt", 'r', encoding='utf-8') as f:
        full_text = f.read()

    generate_pq_text_vectors(model, tokenizer, full_text, pq_output_dir, num_chunks=100, chunk_size=512)

    drive_pq_path = f"/content/drive/MyDrive/{model_dir}/pq_training_data"
    os.makedirs(drive_pq_path, exist_ok=True)
    print(f"Backing up to {drive_pq_path}...")
    os.system(f"rsync -ah --progress {pq_output_dir}/ {drive_pq_path}/")
    print("Process complete.")

# %% id="pqDeterministicLongContext"
# =====================================================================================
# Deterministic long-context calibration-data generation
# PQ_BRIDGE_CELL_MARKER: deterministic_long_context_calibration_generation
#
# This cell deliberately contains no model-based ranking or manual selection. It
# turns clean, decontaminated source examples into task-shaped long prompts using a
# stable lexical score and source-ID tie break. Synthetic passage-count examples are
# assembled from the same clean paragraph pool with exact, recorded duplicate counts.
# =====================================================================================

import hashlib
import random
import re

LONG_CONTEXT_MIN_PROMPT_TOKENS = 3900
LONG_CONTEXT_TARGET_PROMPT_TOKENS = 4000
LONG_CONTEXT_MAX_PROMPT_TOKENS = 4096


def deterministic_text_terms(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def deterministic_paragraphs(text):
    blocks = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", str(text))]
    return [part for part in blocks if len(part) >= 80]


def stable_digest_int(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def build_deterministic_donor_pool(chosen):
    donors = []
    seen_hashes = set()
    for _, _, sample in chosen:
        source_id = str(sample.get("_source_id", ""))
        source_dataset = str(sample.get("_source_dataset", ""))
        declared_hashes = set(sample.get("_paragraph_hashes", []) or [])
        for paragraph_index, text in enumerate(deterministic_paragraphs(sample.get("context", ""))):
            paragraph_hash = hashlib.sha1(
                re.sub(
                    r"[^\w\s]",
                    "",
                    re.sub(r"\s+", " ", text.lower()),
                ).strip().encode("utf-8")
            ).hexdigest()
            if paragraph_hash not in declared_hashes:
                continue
            if paragraph_hash in seen_hashes:
                continue
            seen_hashes.add(paragraph_hash)
            donors.append({
                "source_id": source_id,
                "source_dataset": source_dataset,
                "paragraph_index": int(paragraph_index),
                "paragraph_hash": paragraph_hash,
                "text": text,
                "terms": deterministic_text_terms(text),
            })
    return donors


def rank_deterministic_donors(sample, donor_pool):
    query_terms = deterministic_text_terms(
        " ".join([
            str(sample.get("input", "")),
            " ".join(str(x) for x in sample.get("_supporting_titles", []) or []),
        ])
    )
    own_source_id = str(sample.get("_source_id", ""))
    own_hashes = set(sample.get("_paragraph_hashes", []) or [])
    ranked = []
    for donor in donor_pool:
        if donor["source_id"] == own_source_id or donor["paragraph_hash"] in own_hashes:
            continue
        overlap = len(query_terms & donor["terms"])
        score = overlap / max(1, len(query_terms))
        ranked.append((
            -score,
            donor["source_dataset"],
            donor["source_id"],
            donor["paragraph_index"],
            donor,
        ))
    ranked.sort(key=lambda row: row[:4])
    return [(float(-row[0]), row[4]) for row in ranked]


def render_calibration_prompt(sample, dataset, tokenizer, prompts):
    prompt_task = sample.get("_prompt_task", dataset)
    prompt = prompts[prompt_task].format(**sample)
    used_chat_template = False
    if prompt_task not in NO_CHAT_TEMPLATE:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        used_chat_template = True
    ids = tokenizer.encode(prompt, add_special_tokens=not used_chat_template)
    return prompt, ids


def fit_qa_sample_to_long_context(sample, dataset, tokenizer, prompts, donor_pool):
    sample = dict(sample)
    original_context = str(sample.get("context", ""))
    donors_used = []
    existing_hashes = set(sample.get("_paragraph_hashes", []) or [])

    _, prompt_ids = render_calibration_prompt(sample, dataset, tokenizer, prompts)
    for score, donor in rank_deterministic_donors(sample, donor_pool):
        if len(prompt_ids) >= LONG_CONTEXT_MIN_PROMPT_TOKENS:
            break
        candidate_label = (
            f"Donor passage [{donor['source_dataset']}:{donor['source_id']}:"
            f"{donor['paragraph_index']}]: {donor['text']}"
        )
        candidate = dict(sample)
        candidate["context"] = sample["context"] + "\n\n" + candidate_label
        _, candidate_ids = render_calibration_prompt(candidate, dataset, tokenizer, prompts)
        if len(candidate_ids) > LONG_CONTEXT_MAX_PROMPT_TOKENS:
            available = max(0, LONG_CONTEXT_TARGET_PROMPT_TOKENS - len(prompt_ids) - 8)
            donor_ids = tokenizer.encode(donor["text"], add_special_tokens=False)
            if available < 16:
                continue
            candidate_label = (
                f"Donor passage [{donor['source_dataset']}:{donor['source_id']}:"
                f"{donor['paragraph_index']}]: "
                + tokenizer.decode(donor_ids[:available], skip_special_tokens=True)
            )
            candidate["context"] = sample["context"] + "\n\n" + candidate_label
            _, candidate_ids = render_calibration_prompt(candidate, dataset, tokenizer, prompts)
            if len(candidate_ids) > LONG_CONTEXT_MAX_PROMPT_TOKENS:
                continue
        sample = candidate
        prompt_ids = candidate_ids
        existing_hashes.add(donor["paragraph_hash"])
        donors_used.append({
            "source_dataset": donor["source_dataset"],
            "source_id": donor["source_id"],
            "paragraph_index": donor["paragraph_index"],
            "paragraph_hash": donor["paragraph_hash"],
            "lexical_score": score,
        })

    sample["_paragraph_hashes"] = sorted(existing_hashes)
    sample["_long_context_generation"] = {
        "method": "stable_lexical_overlap_then_source_id",
        "original_context_sha1": hashlib.sha1(original_context.encode("utf-8")).hexdigest(),
        "donors": donors_used,
        "prompt_tokens_before_answer": int(len(prompt_ids)),
        "target_prompt_tokens": LONG_CONTEXT_TARGET_PROMPT_TOKENS,
    }
    return sample


def make_deterministic_passage_count_sample(sample, dataset, tokenizer, prompts, donor_pool):
    ranked = rank_deterministic_donors(sample, donor_pool)
    source_id = str(sample.get("_source_id", ""))
    unique_count = 6 + stable_digest_int("count", source_id) % 5
    components_per_unique = 4
    selected = [
        donor for _, donor in ranked[:unique_count * components_per_unique]
    ]
    if len(selected) != unique_count * components_per_unique:
        raise RuntimeError(
            f"Only {len(selected)} clean donors available for passage-count sample"
        )
    bundles = [
        " ".join(
            donor["text"]
            for donor in selected[
                index * components_per_unique:(index + 1) * components_per_unique
            ]
        )
        for index in range(unique_count)
    ]

    order = [index % unique_count for index in range(30)]
    random.Random(stable_digest_int("order", source_id)).shuffle(order)
    transformed = dict(sample)
    transformed.update({
        "input": "",
        "answers": [str(unique_count)],
        "answer": str(unique_count),
        "_prompt_task": "passage_count",
        "_answer_type": "exact_count",
        "_supporting_titles": [],
        "_all_titles": [],
        "_paragraph_hashes": sorted(donor["paragraph_hash"] for donor in selected),
        "_long_context_generation": {
            "method": "deterministic_synthetic_passage_count",
            "unique_count": int(unique_count),
            "paragraph_instances": 30,
            "duplicate_order": order,
            "donors": [
                {
                    "source_dataset": donor["source_dataset"],
                    "source_id": donor["source_id"],
                    "paragraph_index": donor["paragraph_index"],
                    "paragraph_hash": donor["paragraph_hash"],
                }
                for donor in selected
            ],
            "target_prompt_tokens": LONG_CONTEXT_TARGET_PROMPT_TOKENS,
        },
    })

    bundle_token_ids = [
        tokenizer.encode(bundle, add_special_tokens=False) for bundle in bundles
    ]

    def context_at_cap(token_cap):
        visible_bundles = [
            tokenizer.decode(ids[:token_cap], skip_special_tokens=True)
            for ids in bundle_token_ids
        ]
        return "\n\n".join(
            f"Paragraph {position + 1}: {visible_bundles[bundle_index]}"
            for position, bundle_index in enumerate(order)
        ), visible_bundles

    low, high = 16, max(len(ids) for ids in bundle_token_ids)
    best_context, best_bundles = context_at_cap(low)
    while low <= high:
        mid = (low + high) // 2
        candidate_context, candidate_bundles = context_at_cap(mid)
        transformed["context"] = candidate_context
        _, candidate_ids = render_calibration_prompt(
            transformed, dataset, tokenizer, prompts
        )
        if len(candidate_ids) <= LONG_CONTEXT_TARGET_PROMPT_TOKENS:
            best_context, best_bundles = candidate_context, candidate_bundles
            low = mid + 1
        else:
            high = mid - 1

    transformed["context"] = best_context
    transformed["_supporting_texts"] = best_bundles
    _, prompt_ids = render_calibration_prompt(transformed, dataset, tokenizer, prompts)
    transformed["_long_context_generation"]["prompt_tokens_before_answer"] = int(len(prompt_ids))
    return transformed


def generate_deterministic_long_context_calibration(chosen, tokenizer, prompts):
    donor_pool = build_deterministic_donor_pool(chosen)
    if len(donor_pool) < 100:
        raise RuntimeError(f"Deterministic donor pool is unexpectedly small: {len(donor_pool)}")

    generated = []
    for task, sample_idx, sample in chosen:
        if sample.get("_training_transform") == "passage_count":
            sample = make_deterministic_passage_count_sample(
                sample, task, tokenizer, prompts, donor_pool
            )
            task = "synthetic_passage_count_train"
        else:
            sample = fit_qa_sample_to_long_context(
                sample, task, tokenizer, prompts, donor_pool
            )
        generated.append((task, sample_idx, sample))
    return generated


# %% colab={"base_uri": "https://localhost:8080/", "height": 381} id="Ij6JVcn0L5Hg" outputId="f2ac9135-04e5-40bc-a413-e67b6af7f97b"
# =====================================================================================
# PQ calibration-vector extraction from LongBench  --  single Colab cell
#
# Captures post-k_norm, PRE-RoPE K/V activations -- the same point the eval harness
# quantizes -- and writes per-(layer, head) .npy files for the codebook trainer.
#
# CHANGES VS THE PREVIOUS VERSION
#   1. DOCUMENT DIVERSITY. The old run drew 20000 vectors/head from only 56 documents
#      (357 correlated positions each). Effective sample size was ~56, and the
#      resulting codebooks overfit those documents. NUM_CALIB_SAMPLES now defaults to
#      400 -> 50 positions from each of 400 documents. Same RAM, same vector count,
#      ~7x the diversity, since per_sample is derived by division.
#   2. RESUME. Each document is written as its own shard, so a Colab disconnect at
#      document 380 costs one document, not the whole run.
#   3. DOCUMENT-LEVEL TRAIN/TEST SPLIT. The split is still at document granularity
#      (so the Test set is genuinely held out and usable as a reconstruction-MSE gate)
#      but WHICH documents are held out is now randomized, and there are ~40 of them
#      instead of ~6. Do NOT change this to a position-level shuffle: that would put
#      positions from the same document on both sides and bias the gate.
#   4. TASK MIX. Calibration uses LongBench v1 tasks excluded from LongBench-E,
#      spanning QA, summarization, retrieval, classification, and Chinese text.
#   5. fp16 STORAGE. The trainer does .astype(np.float32) on load anyway, so this is
#      lossless in effect and halves disk + Drive sync time.
#
# CONTAMINATION CONTROL -- read this before choosing a mode.
#   "held_out"     calibrate on tasks NOT in EVAL_TASKS.        <- default, defensible
#   "matched"      same tasks, sample indices disjoint from eval.  Needs the eval to
#                  use a subset; hotpotqa has exactly 200 samples and the pinned eval
#                  uses all of them, so this mode is unavailable for that task.
#   "contaminated" calibrate on the exact eval samples. NOT a reportable result. Useful
#                  only as a diagnostic ceiling: the gap between this and "held_out"
#                  tells you how much of any gain is real generalization.
# =====================================================================================

import os
import gc
import sys
import json
import glob
import math
import mmap
import random
import re
import shutil
import struct
import zipfile
import subprocess
import hashlib
from collections import Counter, defaultdict

# Set to False once the session is warm; this re-runs on every execution otherwise.
INSTALL_DEPS = True
if INSTALL_DEPS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub", "datasets>=4.0.0"],
                   check=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer


# =====================================================================================
# Config
# PQ_BRIDGE_CELL_MARKER: longbench_calibration_extraction_role_aware
# =====================================================================================

MODEL_DIR = "/content/qwen3_8B"
_calibration_mode_env = os.environ.get("PQ_CALIBRATION_MODE", "held_out").strip().lower()
if _calibration_mode_env not in {"held_out", "matched", "contaminated"}:
    raise ValueError(
        "PQ_CALIBRATION_MODE must be held_out, matched, or contaminated"
    )
CALIBRATION_MODE = _calibration_mode_env
_calibration_variant_env = os.environ.get(
    "PQ_CALIBRATION_VARIANT", "prior_held_out"
).strip().lower()
CALIBRATION_VARIANT_ALIASES = {
    "": "prior_held_out",
    "none": "prior_held_out",
    "held_out": "prior_held_out",
    "prior": "prior_held_out",
    "prior_held_out": "prior_held_out",
    "clean_hotpot_a": "clean_hotpot_a",
    "hotpot_a": "clean_hotpot_a",
    "variant_a": "clean_hotpot_a",
    "clean_suite_b": "clean_suite_b",
    "suite_b": "clean_suite_b",
    "variant_b": "clean_suite_b",
    "clean_qa_count_c": "clean_qa_count_c",
    "qa_count_c": "clean_qa_count_c",
    "variant_c": "clean_qa_count_c",
}
if _calibration_variant_env not in CALIBRATION_VARIANT_ALIASES:
    raise ValueError(
        "PQ_CALIBRATION_VARIANT must be prior_held_out, clean_hotpot_a, "
        "clean_suite_b, or clean_qa_count_c"
    )
CALIBRATION_VARIANT = CALIBRATION_VARIANT_ALIASES[_calibration_variant_env]
if CALIBRATION_VARIANT != "prior_held_out" and CALIBRATION_MODE != "held_out":
    raise ValueError(
        "Clean calibration variants require PQ_CALIBRATION_MODE=held_out. "
        "Use the legacy contaminated mode only for diagnostic non-reportable runs."
    )
CALIBRATION_OUTPUT_TAG = (
    CALIBRATION_MODE if CALIBRATION_VARIANT == "prior_held_out"
    else CALIBRATION_VARIANT
)
PQ_OUTPUT_DIR = (
    f"{MODEL_DIR}/pq_training_data_longbench_e_{CALIBRATION_OUTPUT_TAG}_4096"
)
SHARD_DIR = f"{PQ_OUTPUT_DIR}/_shards"          # per-document, enables resume
LONGBENCH_ROOT = "/content/longbench_data"
LONGBENCH_E_REVISION = "36914d6211386125c6fc4ce7db4a6a777fadd34c"

# What you evaluate on. Excluded from calibration in held_out mode.
EVAL_TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
EVAL_SAMPLES_PER_TASK = 200          # only used by "matched" / "contaminated" modes

# Calibration pool, weighted. hotpotqa is English multi-doc QA, so the multi-doc QA
# tasks get the most weight; the rest are there so the codebooks are not tuned to a
# single text domain. Weights are relative, not counts.
#
# `lcc` was removed deliberately: code activations are a different regime and it was
# 1/7 of the previous pool. Add it back only if you start evaluating on code tasks.
CALIB_TASK_WEIGHTS = {
    # LongBench v1 tasks excluded from LongBench-E. This gives broad task coverage
    # without using any task that contributes to the reportable LongBench-E score.
    "narrativeqa":          3,  # long single-document QA
    "musique":              3,  # multi-document reasoning
    "qmsum":                2,  # long meeting summarization
    "multifieldqa_zh":      2,  # Chinese single-document QA
    "dureader":             2,  # Chinese QA
    "passage_retrieval_zh": 2,  # retrieval activation regime
    "vcsum":                1,  # Chinese dialogue summarization
    "lsht":                 1,  # classification
}

ROLE_POSITION_QUOTAS = {
    "supporting_fact": 15,
    "bridge_title_entity": 8,
    "question_instruction": 8,
    "answer_decode": 4,
    "uniform_context_distractor": 15,
}

SOURCE_REVISIONS = {
    "hotpotqa/hotpot_qa": "1908d6afbbead072334abe2965f91bd2709910ab",
    "dgslibisey/MuSiQue": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
    "framolfese/2WikiMultihopQA": "fe713bfbd1afbca1a65246741a75890405d56a3a",
    "deepmind/narrativeqa": "2e643e7363944af1c33a652d1c87320d0871c4e4",
}

CALIBRATION_VARIANT_SOURCES = {
    "clean_hotpot_a": [
        {
            "name": "official_hotpotqa_train",
            "family": "english_multihop_hotpot",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 400,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot",
            "license": "HotpotQA dataset terms",
        },
        {
            "name": "musique_train",
            "family": "english_multihop_musique",
            "repo_id": "dgslibisey/MuSiQue",
            "config": None,
            "split": "train",
            "documents": 160,
            "prompt_task": "hotpotqa",
            "normalizer": "musique",
            "license": "MuSiQue dataset terms",
        },
        {
            "name": "2wikimultihopqa_train",
            "family": "english_multihop_2wiki",
            "repo_id": "framolfese/2WikiMultihopQA",
            "config": None,
            "split": "train",
            "documents": 120,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot",
            "license": "2WikiMultihopQA dataset terms",
        },
        {
            "name": "hotpotqa_hard_distractors_train",
            "family": "english_wikipedia_hard_distractors",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 80,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot_distractor",
            "license": "HotpotQA dataset terms",
        },
        {
            "name": "narrativeqa_train",
            "family": "english_long_form_qa",
            "repo_id": "deepmind/narrativeqa",
            "config": None,
            "split": "train",
            "documents": 40,
            "prompt_task": "hotpotqa",
            "normalizer": "narrativeqa",
            "license": "NarrativeQA dataset terms",
        },
    ],
    "clean_suite_b": [
        {
            "name": "official_hotpotqa_train",
            "family": "english_multihop_hotpot",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 240,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot",
            "license": "HotpotQA dataset terms",
        },
        {
            "name": "musique_train",
            "family": "english_multihop_musique",
            "repo_id": "dgslibisey/MuSiQue",
            "config": None,
            "split": "train",
            "documents": 100,
            "prompt_task": "hotpotqa",
            "normalizer": "musique",
            "license": "MuSiQue dataset terms",
        },
        {
            "name": "2wikimultihopqa_train",
            "family": "english_multihop_2wiki",
            "repo_id": "framolfese/2WikiMultihopQA",
            "config": None,
            "split": "train",
            "documents": 60,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot",
            "license": "2WikiMultihopQA dataset terms",
        },
        {
            "name": "narrativeqa_train",
            "family": "english_single_doc_scientific_qa",
            "repo_id": "deepmind/narrativeqa",
            "config": None,
            "split": "train",
            "documents": 160,
            "prompt_task": "hotpotqa",
            "normalizer": "narrativeqa",
            "license": "NarrativeQA dataset terms",
        },
        {
            "name": "hotpotqa_hard_distractors_train",
            "family": "english_retrieval_counting",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 120,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot_distractor",
            "license": "HotpotQA dataset terms",
        },
        {
            "name": "longbench_qmsum_train_proxy",
            "family": "english_summarization",
            "legacy_longbench_task": "qmsum",
            "documents": 80,
            "prompt_task": "hotpotqa",
            "normalizer": "longbench_qa_proxy",
            "license": "LongBench dataset terms",
        },
        {
            "name": "longbench_lcc_train_proxy",
            "family": "code_or_chinese",
            "legacy_longbench_task": "lcc",
            "documents": 40,
            "prompt_task": "hotpotqa",
            "normalizer": "longbench_qa_proxy",
            "license": "LongBench dataset terms",
        },
    ],
    "clean_qa_count_c": [
        {
            "name": "official_hotpotqa_long_train",
            "family": "english_multihop_hotpot_long_4k",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 240,
            "prompt_task": "hotpotqa",
            "normalizer": "hotpot",
            "license": "HotpotQA dataset terms",
        },
        {
            "name": "musique_long_train",
            "family": "english_multihop_musique_long_4k",
            "repo_id": "dgslibisey/MuSiQue",
            "config": None,
            "split": "train",
            "documents": 160,
            "prompt_task": "hotpotqa",
            "normalizer": "musique",
            "license": "MuSiQue dataset terms",
        },
        {
            "name": "2wikimultihopqa_long_train",
            "family": "english_multihop_2wiki_long_4k",
            "repo_id": "framolfese/2WikiMultihopQA",
            "config": None,
            "split": "train",
            "documents": 160,
            "prompt_task": "2wikimqa",
            "normalizer": "hotpot",
            "license": "2WikiMultihopQA dataset terms",
        },
        {
            "name": "narrativeqa_qasper_long_train",
            "family": "english_single_document_qa_long_4k",
            "repo_id": "deepmind/narrativeqa",
            "config": None,
            "split": "train",
            "documents": 80,
            "prompt_task": "qasper",
            "normalizer": "narrativeqa",
            "license": "NarrativeQA dataset terms",
        },
        {
            "name": "synthetic_passage_count_train",
            "family": "english_passage_count_long_4k",
            "repo_id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "documents": 160,
            "prompt_task": "passage_count",
            "normalizer": "hotpot",
            "training_transform": "passage_count",
            "license": "HotpotQA dataset terms",
        },
    ],
}

# Number of DISTINCT DOCUMENTS. This is the number that was too low before (56).
# Memory does not scale with it -- VECTORS_PER_HEAD is fixed and positions per
# document are derived by division. Ceiling is the pool size: most LongBench tasks
# have 200 samples, so with the weights above the practical max is ~1200.
NUM_CALIB_SAMPLES = 800

MAX_INPUT_LENGTH = 4096              # exactly matches the eval harness
TEACHER_FORCE_ANSWER_TOKENS = 16
VECTORS_PER_HEAD = 40000             # per layer/head, before the train/test split
TEST_FRACTION = 0.10                 # fraction of DOCUMENTS held out for the MSE gate
SAVE_DTYPE = np.float16              # trainer upcasts to f32 on load
SEED = int(os.environ.get("PQ_CALIBRATION_SEED", "0"))

CHECKPOINT_RESUME = True             # skip documents whose shard already exists
DELETE_SHARDS_AFTER = False          # set True to reclaim ~3GB once assembly succeeds

BACKUP_TO_DRIVE = True
DRIVE_PQ_PATH = (
    "/content/drive/MyDrive/qwen3_8B/"
    f"pq_training_data_longbench_e_{CALIBRATION_OUTPUT_TAG}_4096"
)


# =====================================================================================
# LongBench data + prompts (same logic as the eval harness)
# =====================================================================================

LONGBENCH_REPO_CANDIDATES = ["zai-org/LongBench", "THUDM/LongBench"]
DATASET2PROMPT_URL = ("https://raw.githubusercontent.com/THUDM/LongBench/main/"
                      "LongBench/config/dataset2prompt.json")

NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
_LONGBENCH_E_REPO_FILES = None


def ensure_longbench_data(root=LONGBENCH_ROOT):
    os.makedirs(root, exist_ok=True)
    if glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        return root

    from huggingface_hub import hf_hub_download
    last_exc = None
    for repo_id in LONGBENCH_REPO_CANDIDATES:
        try:
            zip_path = hf_hub_download(repo_id=repo_id, filename="data.zip",
                                       repo_type="dataset")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)
            print(f"  LongBench data unpacked from {repo_id}")
            return root
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not download LongBench data.zip: {last_exc}")


def load_longbench_task(name, root=LONGBENCH_ROOT):
    if CALIBRATION_MODE in {"matched", "contaminated"}:
        # Use the exact immutable LongBench-E examples consumed by the evaluator.
        # Loading the legacy data.zip task here would only match task names, not
        # evaluation documents, and therefore would not be genuinely contaminated.
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download

        config_name = f"{name}_e"
        global _LONGBENCH_E_REPO_FILES
        if _LONGBENCH_E_REPO_FILES is None:
            api = HfApi()
            _LONGBENCH_E_REPO_FILES = api.list_repo_files(
                "zai-org/LongBench",
                repo_type="dataset",
                revision=LONGBENCH_E_REVISION,
            )
        files = _LONGBENCH_E_REPO_FILES
        shard_names = sorted(
            filename for filename in files
            if filename.startswith(f"{config_name}/") and filename.endswith(".parquet")
        )
        if not shard_names:
            raise FileNotFoundError(
                f"No Parquet shards for {config_name} at {LONGBENCH_E_REVISION}"
            )
        paths = [
            hf_hub_download(
                "zai-org/LongBench",
                filename=filename,
                repo_type="dataset",
                revision=LONGBENCH_E_REVISION,
                token=False,
            )
            for filename in shard_names
        ]
        dataset = load_dataset(
            "parquet", data_files={"test": paths}, split="test"
        )
        return [dict(row) for row in dataset]

    ensure_longbench_data(root)
    matches = glob.glob(os.path.join(root, "**", f"{name}.jsonl"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Task '{name}' not found under {root}")
    with open(matches[0], "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_longbench_e_task(name):
    """Load the immutable LongBench-E evaluation shard used for decontamination."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    config_name = f"{name}_e"
    global _LONGBENCH_E_REPO_FILES
    if _LONGBENCH_E_REPO_FILES is None:
        api = HfApi()
        _LONGBENCH_E_REPO_FILES = api.list_repo_files(
            "zai-org/LongBench",
            repo_type="dataset",
            revision=LONGBENCH_E_REVISION,
        )
    shard_names = sorted(
        filename for filename in _LONGBENCH_E_REPO_FILES
        if filename.startswith(f"{config_name}/") and filename.endswith(".parquet")
    )
    if not shard_names:
        raise FileNotFoundError(
            f"No Parquet shards for {config_name} at {LONGBENCH_E_REVISION}"
        )
    paths = [
        hf_hub_download(
            "zai-org/LongBench",
            filename=filename,
            repo_type="dataset",
            revision=LONGBENCH_E_REVISION,
            token=False,
        )
        for filename in shard_names
    ]
    dataset = load_dataset("parquet", data_files={"test": paths}, split="test")
    return [dict(row) for row in dataset]


def load_pinned_training_rows(spec):
    if "legacy_longbench_task" in spec:
        return load_longbench_task(spec["legacy_longbench_task"])

    from datasets import load_dataset

    repo_id = spec["repo_id"]
    revision = SOURCE_REVISIONS.get(repo_id)
    if revision is None:
        raise ValueError(f"No pinned revision configured for {repo_id}")
    args = [repo_id]
    if spec.get("config"):
        args.append(spec["config"])
    ds = load_dataset(*args, split=spec["split"], revision=revision)
    return [dict(row) for row in ds]


def load_dataset2prompt():
    import urllib.request
    with urllib.request.urlopen(DATASET2PROMPT_URL, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_for_match(text):
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def short_hash(text):
    return hashlib.sha1(normalize_for_match(text).encode("utf-8")).hexdigest()


def first_present(row, names, default=""):
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0:
            continue
        return value
    return default


def coerce_answers(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("text", "answer", "answers"):
            if key in value:
                return coerce_answers(value[key])
        return [json.dumps(value, sort_keys=True)]
    if isinstance(value, (list, tuple, np.ndarray)):
        answers = []
        for item in value:
            answers.extend(coerce_answers(item))
        return [str(x) for x in answers if str(x).strip()]
    return [str(value)]


def hotpot_context_parts(row):
    context = row.get("context", {})
    titles = []
    paragraphs = []
    if isinstance(context, dict):
        titles = list(context.get("title", []) or [])
        sentences = list(context.get("sentences", []) or [])
        for title, sent_list in zip(titles, sentences):
            if isinstance(sent_list, str):
                sent_list = [sent_list]
            text = " ".join(str(s) for s in (sent_list or []) if str(s).strip())
            paragraphs.append({"title": str(title), "sentences": list(sent_list or []), "text": text})
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                title = str(first_present(item, ["title", "name"], ""))
                text = str(first_present(item, ["paragraph_text", "text", "context"], ""))
                paragraphs.append({"title": title, "sentences": [text], "text": text})
                titles.append(title)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sent_list = item[1] if isinstance(item[1], (list, tuple)) else [item[1]]
                text = " ".join(str(s) for s in sent_list if str(s).strip())
                paragraphs.append({"title": title, "sentences": list(sent_list), "text": text})
                titles.append(title)
    context_text = "\n\n".join(
        f"{p['title']}: {p['text']}" if p["title"] else p["text"]
        for p in paragraphs
        if p["text"]
    )
    return titles, paragraphs, context_text


def extract_support_texts(row, paragraphs):
    support = row.get("supporting_facts") or row.get("supports") or {}
    titles = []
    sent_ids = []
    if isinstance(support, dict):
        titles = list(support.get("title", []) or [])
        sent_ids = list(support.get("sent_id", []) or support.get("sent_ids", []) or [])
    elif isinstance(support, list):
        for item in support:
            if isinstance(item, dict):
                titles.append(str(first_present(item, ["title"], "")))
                sent_ids.append(first_present(item, ["sent_id", "sentence_id"], None))
            elif isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
                sent_ids.append(item[1] if len(item) > 1 else None)

    by_title = {normalize_for_match(p["title"]): p for p in paragraphs}
    support_texts = []
    for title, sent_id in zip(titles, sent_ids or [None] * len(titles)):
        para = by_title.get(normalize_for_match(title))
        if not para:
            continue
        sentences = para.get("sentences", [])
        try:
            sid = int(sent_id)
        except (TypeError, ValueError):
            sid = None
        if sid is not None and 0 <= sid < len(sentences):
            support_texts.append(str(sentences[sid]))
        elif para.get("text"):
            support_texts.append(str(para["text"]))
    return support_texts, [str(t) for t in titles if str(t).strip()]


def normalize_hotpot_like(row, spec, source_position):
    titles, paragraphs, context_text = hotpot_context_parts(row)
    support_texts, support_titles = extract_support_texts(row, paragraphs)
    question = str(first_present(row, ["question", "input", "query"], ""))
    answers = coerce_answers(first_present(row, ["answer", "answers"], ""))
    source_id = str(first_present(row, ["id", "_id", "qid"], source_position))
    if spec.get("normalizer") == "hotpot_distractor":
        support_title_norm = {normalize_for_match(t) for t in support_titles}
        distractors = [
            p for p in paragraphs
            if normalize_for_match(p.get("title", "")) not in support_title_norm
        ]
        if distractors:
            context_text = "\n\n".join(
                f"{p['title']}: {p['text']}" if p["title"] else p["text"]
                for p in distractors
                if p["text"]
            )
    return {
        "context": context_text,
        "input": question,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "_prompt_task": spec["prompt_task"],
        "_source_dataset": spec["name"],
        "_source_family": spec["family"],
        "_source_split": spec["split"],
        "_source_repo": spec.get("repo_id", "longbench_v1"),
        "_source_revision": SOURCE_REVISIONS.get(spec.get("repo_id"), "longbench_v1_data_zip"),
        "_source_id": source_id,
        "_source_license": spec.get("license", ""),
        "_answer_type": str(first_present(row, ["type", "answer_type"], "")),
        "_supporting_titles": sorted({t for t in support_titles if t}),
        "_supporting_texts": support_texts,
        "_all_titles": sorted({t for t in titles if t}),
        "_paragraph_hashes": [short_hash(p["text"]) for p in paragraphs if p.get("text")],
        "_raw_index": int(source_position),
    }


def normalize_musique(row, spec, source_position):
    paragraphs = []
    for p in row.get("paragraphs", []) or row.get("contexts", []) or []:
        if not isinstance(p, dict):
            continue
        title = str(first_present(p, ["title", "paragraph_title"], ""))
        text = str(first_present(p, ["paragraph_text", "text", "context"], ""))
        paragraphs.append({
            "title": title,
            "sentences": [text],
            "text": text,
            "is_supporting": bool(first_present(p, ["is_supporting", "supporting"], False)),
        })
    context_text = "\n\n".join(
        f"{p['title']}: {p['text']}" if p["title"] else p["text"]
        for p in paragraphs
        if p["text"]
    )
    question = str(first_present(row, ["question", "input"], ""))
    answers = coerce_answers(first_present(row, ["answer", "answers", "answer_aliases"], ""))
    support_texts = [p["text"] for p in paragraphs if p.get("is_supporting") and p.get("text")]
    support_titles = [p["title"] for p in paragraphs if p.get("is_supporting") and p.get("title")]
    source_id = str(first_present(row, ["id", "_id", "qid"], source_position))
    return {
        "context": context_text,
        "input": question,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "_prompt_task": spec["prompt_task"],
        "_source_dataset": spec["name"],
        "_source_family": spec["family"],
        "_source_split": spec["split"],
        "_source_repo": spec.get("repo_id", "longbench_v1"),
        "_source_revision": SOURCE_REVISIONS.get(spec.get("repo_id"), "longbench_v1_data_zip"),
        "_source_id": source_id,
        "_source_license": spec.get("license", ""),
        "_answer_type": "multihop",
        "_supporting_titles": sorted({t for t in support_titles if t}),
        "_supporting_texts": support_texts,
        "_all_titles": sorted({p["title"] for p in paragraphs if p.get("title")}),
        "_paragraph_hashes": [short_hash(p["text"]) for p in paragraphs if p.get("text")],
        "_raw_index": int(source_position),
    }


def normalize_narrativeqa(row, spec, source_position):
    document = row.get("document", {})
    if isinstance(document, dict):
        context_text = str(first_present(
            document,
            ["text", "summary", "story", "document", "kind"],
            "",
        ))
        title = str(first_present(document, ["title", "id"], ""))
    else:
        context_text = str(document)
        title = ""
    question_obj = row.get("question", "")
    question = (
        str(first_present(question_obj, ["text", "question"], ""))
        if isinstance(question_obj, dict) else str(question_obj)
    )
    answers = coerce_answers(first_present(row, ["answers", "answer"], ""))
    source_id = str(first_present(row, ["id", "_id", "qid"], source_position))
    return {
        "context": context_text,
        "input": question,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "_prompt_task": spec["prompt_task"],
        "_source_dataset": spec["name"],
        "_source_family": spec["family"],
        "_source_split": spec["split"],
        "_source_repo": spec.get("repo_id", "longbench_v1"),
        "_source_revision": SOURCE_REVISIONS.get(spec.get("repo_id"), "longbench_v1_data_zip"),
        "_source_id": source_id,
        "_source_license": spec.get("license", ""),
        "_answer_type": "long_form",
        "_supporting_titles": [title] if title else [],
        "_supporting_texts": [],
        "_all_titles": [title] if title else [],
        "_paragraph_hashes": [short_hash(context_text)] if context_text else [],
        "_raw_index": int(source_position),
    }


def normalize_variant_row(row, spec, source_position):
    normalizer = spec.get("normalizer", "hotpot")
    if normalizer == "musique":
        sample = normalize_musique(row, spec, source_position)
    elif normalizer == "narrativeqa":
        sample = normalize_narrativeqa(row, spec, source_position)
    elif normalizer == "longbench_qa_proxy":
        proxy = {
            "context": first_present(row, ["context", "document"], ""),
            "question": first_present(row, ["input", "question"], ""),
            "answers": first_present(row, ["answers", "answer"], ""),
            "id": first_present(row, ["_id", "id"], source_position),
        }
        sample = normalize_hotpot_like(proxy, spec, source_position)
    else:
        sample = normalize_hotpot_like(row, spec, source_position)
    if spec.get("training_transform"):
        sample["_training_transform"] = spec["training_transform"]
    return sample


def decontamination_signatures(sample):
    question = normalize_for_match(first_present(sample, ["input", "question"], ""))
    answers = sorted({normalize_for_match(a) for a in coerce_answers(sample.get("answers") or sample.get("answer")) if a})
    source_id = normalize_for_match(first_present(sample, ["_source_id", "_id", "id"], ""))
    titles = tuple(sorted({
        normalize_for_match(t)
        for t in sample.get("_supporting_titles", []) or []
        if normalize_for_match(t)
    }))
    paragraph_hashes = set(sample.get("_paragraph_hashes", []) or [])
    context_hash = short_hash(first_present(sample, ["context"], ""))
    qa_pairs = {(question, answer) for answer in answers if question and answer}
    return {
        "source_id": source_id,
        "question": question,
        "answers": answers,
        "qa_pairs": qa_pairs,
        "supporting_titles": titles,
        "paragraph_hashes": paragraph_hashes,
        "context_hash": context_hash,
    }


def build_longbench_e_decontamination_index():
    index = {
        "source_ids": set(),
        "questions": set(),
        "qa_pairs": set(),
        "supporting_title_sets": set(),
        "paragraph_hashes": set(),
        "context_hashes": set(),
        "context_shingles": [],
        "documents": 0,
    }
    for task in EVAL_TASKS:
        for row in load_longbench_e_task(task):
            sample = {
                "_source_id": first_present(row, ["_id", "id"], ""),
                "input": first_present(row, ["input", "question"], ""),
                "answers": coerce_answers(row.get("answers") or row.get("answer")),
                "context": first_present(row, ["context"], ""),
                "_supporting_titles": row.get("supporting_facts", {}).get("title", [])
                if isinstance(row.get("supporting_facts"), dict) else [],
                "_paragraph_hashes": [short_hash(first_present(row, ["context"], ""))],
            }
            sig = decontamination_signatures(sample)
            if sig["source_id"]:
                index["source_ids"].add(sig["source_id"])
            if sig["question"]:
                index["questions"].add(sig["question"])
            index["qa_pairs"].update(sig["qa_pairs"])
            if sig["supporting_titles"]:
                index["supporting_title_sets"].add(sig["supporting_titles"])
            index["paragraph_hashes"].update(sig["paragraph_hashes"])
            index["context_hashes"].add(sig["context_hash"])
            context_norm = normalize_for_match(sample["context"])
            shingles = {
                context_norm[i:i + 80]
                for i in range(0, max(len(context_norm) - 79, 0), 40)
            }
            if shingles:
                index["context_shingles"].append((task, shingles))
            index["documents"] += 1
    return index


def near_duplicate_reason(sample, decon_index, threshold=0.82):
    context_norm = normalize_for_match(sample.get("context", ""))
    shingles = {
        context_norm[i:i + 80]
        for i in range(0, max(len(context_norm) - 79, 0), 40)
    }
    if not shingles:
        return None
    for task, eval_shingles in decon_index["context_shingles"]:
        union = len(shingles | eval_shingles)
        if union == 0:
            continue
        score = len(shingles & eval_shingles) / union
        if score >= threshold:
            return f"near_duplicate_context:{task}:{score:.3f}"
    return None


def decontamination_reasons(sample, decon_index):
    sig = decontamination_signatures(sample)
    reasons = []
    if sig["source_id"] and sig["source_id"] in decon_index["source_ids"]:
        reasons.append("source_id")
    if sig["question"] and sig["question"] in decon_index["questions"]:
        reasons.append("normalized_question")
    if sig["qa_pairs"] & decon_index["qa_pairs"]:
        reasons.append("question_answer_pair")
    if sig["supporting_titles"] and sig["supporting_titles"] in decon_index["supporting_title_sets"]:
        reasons.append("supporting_title_set")
    if sig["paragraph_hashes"] & decon_index["paragraph_hashes"]:
        reasons.append("paragraph_hash")
    if sig["context_hash"] in decon_index["context_hashes"]:
        reasons.append("context_hash")
    near = near_duplicate_reason(sample, decon_index)
    if near:
        reasons.append(near)
    return reasons


def build_prompt(sample, dataset, tokenizer, prompts, max_length=MAX_INPUT_LENGTH):
    """Return token IDs using the evaluator's template/chat/truncation order."""
    prompt_task = sample.get("_prompt_task", dataset)
    prompt = prompts[prompt_task].format(**sample)
    used_chat_template = False
    if prompt_task not in NO_CHAT_TEMPLATE:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        used_chat_template = True

    token_ids = tokenizer.encode(prompt, add_special_tokens=not used_chat_template)
    if len(token_ids) > max_length:
        first = max_length // 2
        token_ids = token_ids[:first] + token_ids[-(max_length - first):]
    return token_ids


def allocate_per_task(pool_tasks, weights, total, availability):
    """Split `total` documents across tasks by weight, capped by what each task has.
    Anything a capped task cannot absorb is redistributed to the others."""
    alloc = {t: 0 for t in pool_tasks}
    remaining = total
    active = list(pool_tasks)

    # Iterate: tasks that hit their cap are frozen and their surplus reallocated.
    for _ in range(len(pool_tasks) + 1):
        if remaining <= 0 or not active:
            break
        wsum = sum(weights.get(t, 1) for t in active)
        if wsum <= 0:
            break

        added_any = False
        for t in list(active):
            want = int(math.floor(remaining * weights.get(t, 1) / wsum))
            room = availability[t] - alloc[t]
            take = max(0, min(want, room))
            if take > 0:
                alloc[t] += take
                added_any = True
            if alloc[t] >= availability[t]:
                active.remove(t)

        remaining = total - sum(alloc.values())
        if not added_any:
            break

    # Distribute any leftover from flooring, one at a time.
    idx = 0
    order = [t for t in pool_tasks if alloc[t] < availability[t]]
    while remaining > 0 and order:
        t = order[idx % len(order)]
        if alloc[t] < availability[t]:
            alloc[t] += 1
            remaining -= 1
        else:
            order.remove(t)
            continue
        idx += 1

    return alloc


def select_calibration_samples(tokenizer=None, prompts=None):
    """Returns [(task, sample_index, sample)] honouring the contamination mode."""
    rng = random.Random(SEED)

    if CALIBRATION_VARIANT != "prior_held_out":
        return select_variant_calibration_samples(rng, tokenizer=tokenizer, prompts=prompts)

    if CALIBRATION_MODE == "held_out":
        pool_tasks = [t for t in CALIB_TASK_WEIGHTS if t not in EVAL_TASKS]
        dropped = [t for t in CALIB_TASK_WEIGHTS if t in EVAL_TASKS]
        if dropped:
            print(f"  held_out: dropping {dropped} from the calibration pool "
                  f"(they are in EVAL_TASKS)")
        if not pool_tasks:
            raise ValueError("held_out mode left no calibration tasks.")
    elif CALIBRATION_MODE in ("matched", "contaminated"):
        pool_tasks = list(EVAL_TASKS)
    else:
        raise ValueError(f"Unknown CALIBRATION_MODE: {CALIBRATION_MODE}")

    # Index pools first, so allocation can respect what each task actually has.
    task_indices = {}
    for task in pool_tasks:
        samples = load_longbench_task(task)
        indices = list(range(len(samples)))

        if CALIBRATION_MODE == "matched":
            indices = indices[EVAL_SAMPLES_PER_TASK:]
            if not indices:
                raise ValueError(
                    f"matched mode: '{task}' has {len(samples)} samples and the eval "
                    f"uses the first {EVAL_SAMPLES_PER_TASK}, leaving nothing held out. "
                    f"Use held_out mode, or lower EVAL_SAMPLES_PER_TASK."
                )
        elif CALIBRATION_MODE == "contaminated":
            indices = indices[:EVAL_SAMPLES_PER_TASK]

        rng.shuffle(indices)
        task_indices[task] = (samples, indices)

    availability = {t: len(task_indices[t][1]) for t in pool_tasks}
    alloc = allocate_per_task(pool_tasks, CALIB_TASK_WEIGHTS, NUM_CALIB_SAMPLES,
                              availability)

    total_available = sum(availability.values())
    if sum(alloc.values()) < NUM_CALIB_SAMPLES:
        print(f"  NOTE: pool holds {total_available} documents, "
              f"{NUM_CALIB_SAMPLES} requested -> using {sum(alloc.values())}.")

    chosen = []
    for task in pool_tasks:
        samples, indices = task_indices[task]
        for idx in indices[:alloc[task]]:
            chosen.append((task, idx, samples[idx]))

    # Shuffle so the document-level train/test split below holds out a random,
    # task-balanced set rather than whichever task happens to be last.
    rng.shuffle(chosen)
    return chosen


def validate_variant_source_mix(variant):
    specs = CALIBRATION_VARIANT_SOURCES[variant]
    total = sum(int(spec["documents"]) for spec in specs)
    if total != NUM_CALIB_SAMPLES:
        raise ValueError(
            f"{variant} source mix sums to {total}, expected {NUM_CALIB_SAMPLES}"
        )
    return specs


def select_variant_calibration_samples(rng, tokenizer=None, prompts=None):
    """Select clean official-training-split calibration rows for Variant A/B."""
    specs = validate_variant_source_mix(CALIBRATION_VARIANT)
    decon_index = build_longbench_e_decontamination_index()

    chosen = []
    rejected_rows = []
    accepted_signatures = set()
    source_reports = []

    for spec in specs:
        raw_rows = load_pinned_training_rows(spec)
        order = list(range(len(raw_rows)))
        rng.shuffle(order)
        target = int(spec["documents"])
        accepted = 0
        rejected_by_reason = Counter()

        for source_position in order:
            if accepted >= target:
                break
            try:
                sample = normalize_variant_row(raw_rows[source_position], spec, source_position)
            except Exception as exc:
                rejected_by_reason["normalization_error"] += 1
                rejected_rows.append({
                    "variant": CALIBRATION_VARIANT,
                    "source_dataset": spec["name"],
                    "source_position": int(source_position),
                    "reasons": [f"normalization_error:{type(exc).__name__}"],
                })
                continue

            sig = decontamination_signatures(sample)
            local_key = (
                sig["question"],
                tuple(sig["answers"]),
                tuple(sorted(sample.get("_supporting_titles", []))),
            )
            reasons = decontamination_reasons(sample, decon_index)
            if local_key in accepted_signatures:
                reasons.append("duplicate_within_calibration_selection")
            if reasons:
                for reason in reasons:
                    rejected_by_reason[reason.split(":")[0]] += 1
                rejected_rows.append({
                    "variant": CALIBRATION_VARIANT,
                    "source_dataset": spec["name"],
                    "source_split": spec.get("split"),
                    "source_id": sample.get("_source_id"),
                    "source_position": int(source_position),
                    "reasons": reasons,
                })
                continue

            sample["_decontamination"] = {
                "status": "accepted",
                "checked_against": f"zai-org/LongBench@{LONGBENCH_E_REVISION}",
                "rules": [
                    "source_id",
                    "normalized_question",
                    "question_answer_pair",
                    "supporting_title_set",
                    "paragraph_hash",
                    "context_hash",
                    "near_duplicate_context_shingles_jaccard_0.82",
                ],
            }
            sample["_variant"] = CALIBRATION_VARIANT
            task = sample.get("_source_dataset", spec["name"])
            sample_idx = f"{spec['name']}__{source_position}"
            chosen.append((task, sample_idx, sample))
            accepted_signatures.add(local_key)
            accepted += 1

        if accepted < target:
            raise RuntimeError(
                f"{spec['name']} supplied {accepted}/{target} decontaminated rows. "
                "Redistribute only within the same functional family and record the change."
            )
        source_reports.append({
            "source_dataset": spec["name"],
            "source_family": spec["family"],
            "repo_id": spec.get("repo_id", "longbench_v1"),
            "source_revision": SOURCE_REVISIONS.get(spec.get("repo_id"), "longbench_v1_data_zip"),
            "source_split": spec.get("split"),
            "requested_documents": target,
            "accepted_documents": accepted,
            "rejected_counts": dict(rejected_by_reason),
        })

    rng.shuffle(chosen)
    if CALIBRATION_VARIANT == "clean_qa_count_c":
        if tokenizer is None or prompts is None:
            raise ValueError(
                "clean_qa_count_c requires tokenizer and prompt templates for "
                "deterministic 4K context construction"
            )
        chosen = generate_deterministic_long_context_calibration(
            chosen, tokenizer, prompts
        )
    select_variant_calibration_samples.last_report = {
        "variant": CALIBRATION_VARIANT,
        "longbench_e_revision": LONGBENCH_E_REVISION,
        "longbench_e_documents_indexed": decon_index["documents"],
        "source_reports": source_reports,
        "rejected_examples": rejected_rows,
        "rejected_counts": dict(Counter(
            reason.split(":")[0]
            for row in rejected_rows
            for reason in row.get("reasons", [])
        )),
        "near_duplicate_threshold": 0.82,
        "long_context_generation": (
            {
                "method": "deterministic lexical donors and synthetic passage counting",
                "min_prompt_tokens": LONG_CONTEXT_MIN_PROMPT_TOKENS,
                "target_prompt_tokens": LONG_CONTEXT_TARGET_PROMPT_TOKENS,
                "max_prompt_tokens": LONG_CONTEXT_MAX_PROMPT_TOKENS,
                "manual_selection": False,
                "model_based_ranking": False,
            }
            if CALIBRATION_VARIANT == "clean_qa_count_c" else None
        ),
    }
    return chosen


select_variant_calibration_samples.last_report = None


# =====================================================================================
# Model (K/V captured pre-RoPE, after k_norm)
# =====================================================================================

def load_safetensors_pure(filepath):
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        offset = 8 + header_size
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        tensors = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dtype, shape = v["dtype"], v["shape"]
            raw = mm[offset + v["data_offsets"][0]: offset + v["data_offsets"][1]]

            if dtype == "BF16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).clone()
                tensors[k] = t.reshape(shape)
            elif dtype == "F32":
                tensors[k] = torch.from_numpy(
                    np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)
            elif dtype == "F16":
                tensors[k] = torch.from_numpy(
                    np.frombuffer(raw, dtype=np.float16).copy()).reshape(shape)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        mm.close()
    return tensors


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    """Shared across layers and sized to the pinned evaluation context."""

    def __init__(self, dim, max_position_embeddings, base=1000000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

        t = torch.arange(max_position_embeddings, device=inv_freq.device,
                         dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len, dtype):
        if seq_len > self.max_seq_len_cached:
            raise ValueError(f"seq_len {seq_len} exceeds rotary cache "
                             f"{self.max_seq_len_cached}")
        return (self.cos_cached[:, :, :seq_len, :].to(dtype=dtype),
                self.sin_cached[:, :, :seq_len, :].to(dtype=dtype))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)

        bias = config.get("attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        eps = config.get("rms_norm_eps", 1e-6)
        self.q_norm = QwenRMSNorm(self.head_dim, eps=eps)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=eps)

    def forward(self, hidden_states, cos, sin, keep_idx):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Capture here: post-k_norm, PRE-RoPE. Same point the eval quantizes.
        # Only the sampled positions are pulled to CPU, which is what keeps this
        # bounded at long context. Do not move this below the cos/sin multiply --
        # RoPE mixes the (c, c+64) channel pairs and destroys the channel-magnitude
        # structure the codebooks and the outlier dims both depend on.
        captured = (
            key_states[0, :, keep_idx, :].detach().to(torch.float16).cpu().numpy(),
            value_states[0, :, keep_idx, :].detach().to(torch.float16).cpu().numpy(),
        )

        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        if self.num_key_value_heads != self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), captured


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, hidden_states, cos, sin, keep_idx):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, captured = self.self_attn(hidden_states, cos, sin, keep_idx)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, captured


class QwenModel(nn.Module):
    def __init__(self, config, max_position_embeddings):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([QwenDecoderLayer(config)
                                     for _ in range(config["num_hidden_layers"])])
        self.norm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.rotary_emb = QwenRotaryEmbedding(head_dim, max_position_embeddings,
                                              base=config.get("rope_theta", 1000000.0))

    def forward(self, input_ids, keep_idx):
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(input_ids.shape[1], hidden_states.dtype)

        all_kvs = []
        for layer in self.layers:
            hidden_states, captured = layer(hidden_states, cos, sin, keep_idx)
            all_kvs.append(captured)
        return self.norm(hidden_states), all_kvs


class QwenForCausalLM(nn.Module):
    def __init__(self, config, max_position_embeddings):
        super().__init__()
        self.model = QwenModel(config, max_position_embeddings)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, keep_idx):
        # lm_head is never needed here; we only want the K/V activations.
        _, all_kvs = self.model(input_ids, keep_idx)
        return all_kvs


# =====================================================================================
# Sharded extraction (resumable)
# =====================================================================================

def shard_path(task, sample_idx):
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task))
    safe_idx = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_idx))
    if str(sample_idx).isdigit():
        safe_idx = f"{int(sample_idx):06d}"
    return os.path.join(SHARD_DIR, f"{safe_task}__{safe_idx}.npz")


def middle_truncate_tokenized(token_ids, offsets, max_length):
    if len(token_ids) <= max_length:
        return list(token_ids), list(offsets)
    first = max_length // 2
    keep = list(range(first)) + list(range(len(token_ids) - (max_length - first), len(token_ids)))
    return [token_ids[i] for i in keep], [offsets[i] for i in keep]


def find_spans(haystack, needles):
    spans = []
    hay_lower = haystack.lower()
    for needle in needles:
        if not needle:
            continue
        needle = str(needle).strip()
        if len(needle) < 2:
            continue
        start = 0
        needle_lower = needle.lower()
        while True:
            idx = hay_lower.find(needle_lower, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(needle)))
            start = idx + max(1, len(needle))
    return spans


def role_for_offset(start, end, role_spans):
    if start == end:
        return None
    for role in ["supporting_fact", "bridge_title_entity", "question_instruction"]:
        for span_start, span_end in role_spans.get(role, []):
            if start < span_end and end > span_start:
                return role
    return "uniform_context_distractor"


def build_capture_inputs_and_roles(sample, dataset, tokenizer, prompts):
    prompt_task = sample.get("_prompt_task", dataset)
    prompt = prompts[prompt_task].format(**sample)
    used_chat_template = False
    if prompt_task not in NO_CHAT_TEMPLATE:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        used_chat_template = True

    enc = tokenizer(
        prompt,
        add_special_tokens=not used_chat_template,
        return_offsets_mapping=True,
    )
    token_ids, offsets = middle_truncate_tokenized(
        enc["input_ids"],
        enc.get("offset_mapping", [(0, 0)] * len(enc["input_ids"])),
        MAX_INPUT_LENGTH,
    )

    support_texts = sample.get("_supporting_texts", []) or []
    titles = sample.get("_supporting_titles", []) or sample.get("_all_titles", []) or []
    role_spans = {
        "supporting_fact": find_spans(prompt, support_texts),
        "bridge_title_entity": find_spans(prompt, titles),
        "question_instruction": find_spans(
            prompt,
            [
                sample.get("input", ""),
                "Question:",
                "Answer the question",
                "Only give me the answer",
            ],
        ),
    }

    roles = []
    titles_for_token = []
    for start, end in offsets:
        role = role_for_offset(int(start), int(end), role_spans)
        roles.append(role or "uniform_context_distractor")
        title = ""
        if role == "bridge_title_entity":
            fragment = prompt[int(start):int(end)].lower()
            for candidate in titles:
                if fragment and fragment in str(candidate).lower():
                    title = str(candidate)
                    break
        titles_for_token.append(title)

    answer_text = " " + str(sample.get("answer", "") or "")
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if answer_ids:
        answer_ids = answer_ids[:TEACHER_FORCE_ANSWER_TOKENS]
        token_ids = list(token_ids) + answer_ids
        roles.extend(["answer_decode"] * len(answer_ids))
        titles_for_token.extend([""] * len(answer_ids))

    return list(token_ids), roles, titles_for_token, len(offsets)


def scaled_role_quotas(per_sample):
    base_total = sum(ROLE_POSITION_QUOTAS.values())
    quotas = {
        role: int(math.floor(per_sample * count / base_total))
        for role, count in ROLE_POSITION_QUOTAS.items()
    }
    remaining = per_sample - sum(quotas.values())
    order = sorted(
        ROLE_POSITION_QUOTAS,
        key=lambda role: ROLE_POSITION_QUOTAS[role],
        reverse=True,
    )
    idx = 0
    while remaining > 0:
        quotas[order[idx % len(order)]] += 1
        remaining -= 1
        idx += 1
    return quotas


def choose_role_aware_positions(roles, per_sample, rng):
    role_to_positions = defaultdict(list)
    for pos, role in enumerate(roles):
        role_to_positions[role].append(pos)

    keep = []
    shortfalls = {}
    quotas = scaled_role_quotas(per_sample)
    for role, quota in quotas.items():
        candidates = [p for p in role_to_positions.get(role, []) if p not in keep]
        if len(candidates) >= quota:
            take = rng.choice(candidates, size=quota, replace=False).tolist()
        else:
            take = list(candidates)
            shortfalls[role] = int(quota - len(take))
        keep.extend(int(x) for x in take)

    if len(keep) < per_sample:
        used = set(keep)
        fallback = [i for i in range(len(roles)) if i not in used]
        need = per_sample - len(keep)
        if len(fallback) >= need:
            keep.extend(int(x) for x in rng.choice(fallback, size=need, replace=False).tolist())
        elif fallback:
            keep.extend(int(x) for x in fallback)

    keep = sorted(set(int(x) for x in keep))
    if len(keep) > per_sample:
        keep = sorted(rng.choice(keep, size=per_sample, replace=False).tolist())
    if len(keep) < min(per_sample, len(roles)):
        raise RuntimeError(
            f"Role-aware sampler selected only {len(keep)} unique positions "
            f"from {len(roles)} available tokens"
        )
    return np.array(keep, dtype=np.int64), shortfalls, quotas


@torch.no_grad()
def extract_shards(model, tokenizer, prompts, chosen):
    """One shard per document. Safe to interrupt and re-run."""
    os.makedirs(SHARD_DIR, exist_ok=True)

    num_layers = len(model.model.layers)
    num_kv_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    device = next(model.parameters()).device

    per_sample = max(1, VECTORS_PER_HEAD // max(1, len(chosen)))

    print(f"  layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim}")
    print(f"  {len(chosen)} documents x {per_sample} sampled positions "
          f"= {per_sample * len(chosen)} vectors per layer/head")
    print(f"  (the previous run used 56 documents x 357 positions -- same total, "
          f"far less diversity)")

    shard_mb = num_layers * num_kv_heads * per_sample * head_dim * 2 * 2 / 1e6
    print(f"  ~{shard_mb:.1f} MB per shard, ~{shard_mb * len(chosen) / 1000:.1f} GB "
          f"of shards on disk")

    rng = np.random.default_rng(SEED)
    written, skipped, failed = 0, 0, []

    model.eval()
    for task, sample_idx, sample in tqdm(chosen, desc="Extracting", leave=True):
        out_path = shard_path(task, sample_idx)
        if CHECKPOINT_RESUME and os.path.exists(out_path):
            skipped += 1
            continue

        prompt_ids, roles, token_titles, prompt_token_count = build_capture_inputs_and_roles(
            sample, task, tokenizer, prompts
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        seq_len = input_ids.shape[1]

        keep, role_shortfalls, role_quotas = choose_role_aware_positions(
            roles, min(per_sample, seq_len), rng
        )
        keep_idx = torch.tensor(keep, device=device, dtype=torch.long)

        try:
            all_kvs = model(input_ids.to(device), keep_idx)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM on {task}[{sample_idx}] at {seq_len} tokens; skipping")
            failed.append((task, str(sample_idx), "oom"))
            gc.collect(); torch.cuda.empty_cache()
            continue

        k_stack = np.stack([k for k, _ in all_kvs], axis=0)   # [L, H, per_sample, D]
        v_stack = np.stack([v for _, v in all_kvs], axis=0)

        # Write to a temp name first so an interrupt cannot leave a half-written shard
        # that resume would then trust.
        tmp_path = out_path + ".tmp"
        np.savez(tmp_path, k=k_stack, v=v_stack,
                 task=np.array(task), sample_index=np.array(str(sample_idx)),
                 sample_key=np.array(str(sample_idx)),
                 prompt_tokens=np.array(int(prompt_token_count)),
                 capture_tokens=np.array(int(seq_len)),
                 role=np.array([roles[int(pos)] for pos in keep], dtype=object),
                 token_offset=np.array(keep, dtype=np.int32),
                 paragraph_title=np.array([token_titles[int(pos)] for pos in keep], dtype=object),
                 variant=np.array(CALIBRATION_VARIANT),
                 calibration_mode=np.array(CALIBRATION_MODE),
                 source_dataset=np.array(sample.get("_source_dataset", task)),
                 source_family=np.array(sample.get("_source_family", task)),
                 source_split=np.array(sample.get("_source_split", "")),
                 source_id=np.array(sample.get("_source_id", str(sample_idx))),
                 source_revision=np.array(sample.get("_source_revision", "")),
                 source_license=np.array(sample.get("_source_license", "")),
                 answer_type=np.array(sample.get("_answer_type", "")),
                 long_context_generation=np.array(json.dumps(
                     sample.get("_long_context_generation", {}), sort_keys=True
                 )),
                 decontamination_status=np.array(
                     (sample.get("_decontamination") or {}).get("status", "legacy")
                 ),
                 role_shortfalls=np.array(json.dumps(role_shortfalls)),
                 role_quotas=np.array(json.dumps(role_quotas)))
        os.replace(tmp_path + ".npz" if os.path.exists(tmp_path + ".npz") else tmp_path,
                   out_path)

        written += 1
        del all_kvs, k_stack, v_stack
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n  shards: {written} written, {skipped} already present, "
          f"{len(failed)} failed")
    return per_sample, failed


def assemble_and_save(chosen, per_sample, output_dir, failed):
    """Load shards, split at DOCUMENT granularity, write per-(layer, head) .npy."""
    os.makedirs(f"{output_dir}/keys", exist_ok=True)
    os.makedirs(f"{output_dir}/values", exist_ok=True)

    present = [(t, i, s) for (t, i, s) in chosen if os.path.exists(shard_path(t, i))]
    if not present:
        raise RuntimeError("No shards found. Did extraction run?")

    probe = np.load(shard_path(present[0][0], present[0][1]), allow_pickle=True)
    num_layers, num_kv_heads, _, head_dim = probe["k"].shape
    probe.close()

    n_docs = len(present)
    shard_lengths = []
    for task, sample_idx, _ in present:
        with np.load(shard_path(task, sample_idx), allow_pickle=True) as z:
            shard_lengths.append(int(z["k"].shape[2]))
    total_slots = int(sum(shard_lengths))
    est_gb = num_layers * num_kv_heads * total_slots * head_dim * 2 * 2 / 1e9
    print(f"  assembling {n_docs} documents -> {total_slots} vectors per layer/head")
    print(f"  ~{est_gb:.1f} GB held in RAM as float16 during assembly")

    key_buf = np.zeros((num_layers, num_kv_heads, total_slots, head_dim), dtype=np.float16)
    val_buf = np.zeros_like(key_buf)

    doc_rows = []
    slot_metadata = []
    cursor = 0

    def z_scalar(z, key, default=""):
        if key not in z.files:
            return default
        value = z[key]
        try:
            return value.item()
        except ValueError:
            return value.tolist()

    for d, (task, sample_idx, sample) in enumerate(tqdm(present, desc="Loading shards")):
        with np.load(shard_path(task, sample_idx), allow_pickle=True) as z:
            actual_positions = int(z["k"].shape[2])
            start, end = cursor, cursor + actual_positions
            key_buf[:, :, start:end, :] = z["k"]
            val_buf[:, :, start:end, :] = z["v"]
            roles = [str(x) for x in z["role"].tolist()]
            token_offsets = [int(x) for x in z["token_offset"].tolist()]
            titles = [str(x) for x in z["paragraph_title"].tolist()]
            for local_idx in range(actual_positions):
                slot_metadata.append({
                    "global_slot": int(start + local_idx),
                    "task": task,
                    "variant": str(z_scalar(z, "variant", CALIBRATION_VARIANT)),
                    "calibration_mode": str(z_scalar(z, "calibration_mode", CALIBRATION_MODE)),
                    "source_dataset": str(z_scalar(z, "source_dataset", task)),
                    "source_family": str(z_scalar(z, "source_family", task)),
                    "source_split": str(z_scalar(z, "source_split", "")),
                    "source_id": str(z_scalar(z, "source_id", sample_idx)),
                    "source_revision": str(z_scalar(z, "source_revision", "")),
                    "source_license": str(z_scalar(z, "source_license", "")),
                    "position_role": roles[local_idx],
                    "token_offset": token_offsets[local_idx],
                    "paragraph_title": titles[local_idx],
                "answer_type": str(z_scalar(z, "answer_type", "")),
                "long_context_generation": json.loads(str(z_scalar(
                    z, "long_context_generation", "{}"
                ))),
                "decontamination_status": str(z_scalar(z, "decontamination_status", "legacy")),
                    "document_ordinal": int(d),
                })
            doc_rows.append({
                "task": task,
                "sample_index": str(sample_idx),
                "source_dataset": str(z_scalar(z, "source_dataset", task)),
                "source_family": str(z_scalar(z, "source_family", task)),
                "source_split": str(z_scalar(z, "source_split", "")),
                "source_id": str(z_scalar(z, "source_id", sample_idx)),
                "source_revision": str(z_scalar(z, "source_revision", "")),
                "source_license": str(z_scalar(z, "source_license", "")),
                "prompt_tokens": int(z["prompt_tokens"]),
                "capture_tokens": int(z_scalar(z, "capture_tokens", int(z["prompt_tokens"]))),
                "positions_kept": int(actual_positions),
                "slot_start": int(start),
                "slot_end": int(end),
                "role_counts": dict(Counter(roles)),
                "role_shortfalls": json.loads(str(z_scalar(z, "role_shortfalls", "{}"))),
                "long_context_generation": json.loads(str(z_scalar(
                    z, "long_context_generation", "{}"
                ))),
            })
            cursor = end

    # ---- DOCUMENT-level split -------------------------------------------------------
    # `present` follows the shuffled `chosen` order, so the tail is already a random,
    # task-mixed set of documents. Keeping the split at document granularity is what
    # makes the Test set a valid reconstruction-MSE gate: a position-level shuffle
    # would put positions from the same document on both sides and bias it optimistic.
    n_test_docs = max(1, int(round(n_docs * TEST_FRACTION)))
    n_train_docs = n_docs - n_test_docs
    if n_train_docs < 1:
        raise ValueError("TEST_FRACTION leaves no training documents.")

    train_end = int(sum(shard_lengths[:n_train_docs]))
    train_slots = np.arange(0, train_end)
    test_slots = np.arange(train_end, total_slots)

    for d, row in enumerate(doc_rows):
        row["split"] = "train" if d < n_train_docs else "test"

    train_tasks, test_tasks = {}, {}
    for row in doc_rows:
        bucket = train_tasks if row["split"] == "train" else test_tasks
        bucket[row["task"]] = bucket.get(row["task"], 0) + 1

    print(f"  split: {n_train_docs} train documents ({len(train_slots)} vectors/head), "
          f"{n_test_docs} test documents ({len(test_slots)} vectors/head)")
    print(f"    train tasks: {train_tasks}")
    print(f"    test  tasks: {test_tasks}")

    # Shuffle positions WITHIN each split. Harmless for the trainer (it samples
    # randomly anyway) but keeps any future head-of-file truncation unbiased.
    rng = np.random.default_rng(SEED + 1)
    rng.shuffle(train_slots)
    rng.shuffle(test_slots)

    slot_lookup = {row["global_slot"]: row for row in slot_metadata}

    def write_position_index(slots, split_name):
        path = os.path.join(output_dir, f"position_index_{split_name}.jsonl")
        role_counts = Counter()
        task_counts = Counter()
        source_counts = Counter()
        with open(path, "w", encoding="utf-8") as f:
            for row_index, global_slot in enumerate(slots.tolist()):
                meta = dict(slot_lookup[int(global_slot)])
                meta["row_index"] = int(row_index)
                meta["split"] = split_name
                role_counts[meta["position_role"]] += 1
                task_counts[meta["task"]] += 1
                source_counts[meta["source_dataset"]] += 1
                f.write(json.dumps(meta, sort_keys=True) + "\n")
        return {
            "path": path,
            "rows": int(len(slots)),
            "role_counts": dict(role_counts),
            "task_counts": dict(task_counts),
            "source_counts": dict(source_counts),
        }

    train_index_report = write_position_index(train_slots, "train")
    test_index_report = write_position_index(test_slots, "test")

    for layer in tqdm(range(num_layers), desc="Saving", leave=True):
        for head in range(num_kv_heads):
            k_tr = key_buf[layer, head][train_slots].astype(SAVE_DTYPE)
            k_te = key_buf[layer, head][test_slots].astype(SAVE_DTYPE)
            v_tr = val_buf[layer, head][train_slots].astype(SAVE_DTYPE)
            v_te = val_buf[layer, head][test_slots].astype(SAVE_DTYPE)
            np.save(f"{output_dir}/keys/L{layer}_H{head}_Train.npy", k_tr)
            np.save(f"{output_dir}/keys/L{layer}_H{head}_Test.npy", k_te)
            np.save(f"{output_dir}/values/L{layer}_H{head}_Train.npy", v_tr)
            np.save(f"{output_dir}/values/L{layer}_H{head}_Test.npy", v_te)

    # Provenance: exactly which documents produced these codebooks, and which side of
    # the split each landed on. Keep this next to any result you report.
    # NOTE: the trainer validates calibration_mode / max_input_length / samples[].task
    # against this file -- do not rename those keys.
    manifest = {
        "calibration_mode": CALIBRATION_MODE,
        "calibration_variant": CALIBRATION_VARIANT,
        "calibration_output_tag": CALIBRATION_OUTPUT_TAG,
        "source_dataset_variant": (
            CALIBRATION_VARIANT
            if CALIBRATION_VARIANT != "prior_held_out" else
            "longbench_e" if CALIBRATION_MODE in {"matched", "contaminated"}
            else "longbench_v1_excluded_tasks"
        ),
        "longbench_e_revision": (
            LONGBENCH_E_REVISION
            if CALIBRATION_MODE in {"matched", "contaminated"}
            else None
        ),
        "eval_tasks": EVAL_TASKS,
        "eval_samples_per_task": EVAL_SAMPLES_PER_TASK,
        "calib_tasks": sorted({task for task, _, _ in chosen}),
        "calib_task_weights": (
            {
                spec["name"]: spec["documents"]
                for spec in CALIBRATION_VARIANT_SOURCES.get(CALIBRATION_VARIANT, [])
            }
            if CALIBRATION_VARIANT != "prior_held_out" else
            CALIB_TASK_WEIGHTS if CALIBRATION_MODE == "held_out"
            else {task: 1 for task in EVAL_TASKS}
        ),
        "source_revisions": SOURCE_REVISIONS,
        "max_input_length": MAX_INPUT_LENGTH,
        "teacher_force_answer_tokens": TEACHER_FORCE_ANSWER_TOKENS,
        "vectors_per_head": int(len(train_slots)),      # train side, what the trainer reads
        "test_vectors_per_head": int(len(test_slots)),
        "num_documents": int(n_docs),
        "num_documents_requested": int(NUM_CALIB_SAMPLES),
        "positions_per_document": int(per_sample),
        "train_documents": int(n_train_docs),
        "test_documents": int(n_test_docs),
        "split_granularity": "document",
        "documents_per_task_train": train_tasks,
        "documents_per_task_test": test_tasks,
        "position_index_train": train_index_report,
        "position_index_test": test_index_report,
        "role_position_quotas": ROLE_POSITION_QUOTAS,
        "prompt_length_summary": {
            "minimum": int(min(row["prompt_tokens"] for row in doc_rows)),
            "maximum": int(max(row["prompt_tokens"] for row in doc_rows)),
            "mean": float(np.mean([row["prompt_tokens"] for row in doc_rows])),
            "at_least_3900": int(sum(
                row["prompt_tokens"] >= 3900 for row in doc_rows
            )),
        },
        "realized_role_counts_train": train_index_report["role_counts"],
        "realized_role_counts_test": test_index_report["role_counts"],
        "realized_source_counts_train": train_index_report["source_counts"],
        "realized_source_counts_test": test_index_report["source_counts"],
        "decontamination_report": (
            select_variant_calibration_samples.last_report
            if CALIBRATION_VARIANT != "prior_held_out" else None
        ),
        "source_mixture": (
            CALIBRATION_VARIANT_SOURCES.get(CALIBRATION_VARIANT)
            if CALIBRATION_VARIANT != "prior_held_out" else None
        ),
        "failed_documents": failed,
        "save_dtype": np.dtype(SAVE_DTYPE).name,
        "seed": SEED,
        "samples": doc_rows,
    }
    with open(f"{output_dir}/calibration_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    del key_buf, val_buf
    gc.collect()
    return len(train_slots), len(test_slots)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 90)
    print(f"PQ calibration extraction from LongBench  [mode: {CALIBRATION_MODE}]".center(90))
    print("=" * 90)

    if CALIBRATION_MODE == "contaminated":
        print("\n  *** CONTAMINATED MODE: codebooks will see the eval samples. ***")
        print("  *** Diagnostic ceiling only. Do not report scores from this. ***\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.bfloat16)

    print("\nSelecting calibration documents")
    prompts = load_dataset2prompt()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    chosen = select_calibration_samples(tokenizer=tokenizer, prompts=prompts)
    by_task = {}
    for task, _, _ in chosen:
        by_task[task] = by_task.get(task, 0) + 1
    print(f"  {len(chosen)} documents: {by_task}")

    overlap = [t for t in by_task if t in EVAL_TASKS]
    if overlap and CALIBRATION_MODE == "held_out":
        raise RuntimeError(f"held_out mode still selected eval tasks: {overlap}")

    print("\nLoading model")
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)

    model = QwenForCausalLM(config, max_position_embeddings=MAX_INPUT_LENGTH + 64)

    state_dict = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        state_dict.update(load_safetensors_pure(path))
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect(); torch.cuda.empty_cache()

    model.to(device)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  model on {device}, rotary cache = {MAX_INPUT_LENGTH + 64} positions")

    print("\nExtracting (resumable -- re-run this cell after a disconnect)")
    per_sample, failed = extract_shards(model, tokenizer, prompts, chosen)

    # Free the model before assembly so the 3GB buffers have room.
    del model
    gc.collect(); torch.cuda.empty_cache()

    print("\nAssembling")
    n_train, n_test = assemble_and_save(chosen, per_sample, PQ_OUTPUT_DIR, failed)

    print(f"\nSaved to {PQ_OUTPUT_DIR}")
    print(f"  train: {n_train} vectors per layer/head")
    print(f"  test:  {n_test} vectors per layer/head  (document-disjoint -- use this "
          f"for the reconstruction-MSE gate)")
    print(f"  manifest: {PQ_OUTPUT_DIR}/calibration_manifest.json")

    if DELETE_SHARDS_AFTER and os.path.isdir(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)
        print(f"  removed shards at {SHARD_DIR}")

    if BACKUP_TO_DRIVE:
        os.makedirs(DRIVE_PQ_PATH, exist_ok=True)
        print(f"Backing up to {DRIVE_PQ_PATH}")
        # Shards excluded: they are a resume artifact, and Drive throttles badly on
        # hundreds of small files.
        os.system(f"rsync -ah --info=progress2 --exclude '_shards' "
                  f"{PQ_OUTPUT_DIR}/ {DRIVE_PQ_PATH}/")
    print("Done.")

# %% id="X_JHUHWKnd8H"
import torch
import gc
import sys

#sys.modules[__name__].__dict__.clear()

vars_to_delete = ['model', 'state_dict', 'orig_ppl', 'quant_ppl', 'results', 'tokenizer', 'pq_manager']
for var in vars_to_delete:
    if var in globals():
        del globals()[var]
    if var in locals():
        del locals()[var]


for _ in range(3):
     gc.collect()



# 3. Force PyTorch to release cached GPU memory back to the system
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()

print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved GPU Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# %% colab={"base_uri": "https://localhost:8080/"} id="eabc79b6" outputId="3787b72b-f8dc-48bf-db7f-248acb23d6d3"
import os
import glob
import json
import math
import shutil
import gc
import warnings
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import torch
from tqdm import tqdm
from collections import Counter, defaultdict


# ============================================================
# Configurable Parameters
# ============================================================

MODEL_NAME = "qwen3_8B"
DRIVE_BASE = f"/content/{MODEL_NAME}"

CLUSTERS = 64
USE_CACHED_CLUSTERS = False

DIMS = 128
NUM_SUBVECTORS = 64
CODEWORDS = 128

TARGET_TRAIN_SIZE = 50000
MIN_VECTORS_PER_HEAD = 5000

SEED = 1234

SAVE_TO_DRIVE = False
DRIVE_OUTPUT_BASE = f"/content/drive/MyDrive/{MODEL_NAME}"

CODEBOOK_NAME = f"codebooks_{NUM_SUBVECTORS}_{CODEWORDS}_{CLUSTERS}"

# Optimized default.
# "diag_wasserstein" is much faster than full covariance Wasserstein.
# "full_wasserstein" is included but can be very slow for many heads.
CLUSTER_METRIC = "diag_wasserstein"

# CUDA k-means controls.
KMEANS_MAX_ITERS_FINE = 60
KMEANS_MAX_ITERS_COARSE = 40
KMEANS_TOL = 1e-4
KMEANS_ASSIGN_CHUNK = 1024

# If memory is tight, lower to 512.
# If L4 has plenty of memory free, 2048 may be faster.
BATCHED_KMEANS_CHUNK = KMEANS_ASSIGN_CHUNK


# ============================================================
# Setup Utilities
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_head_name(path):
    return os.path.basename(path).replace("_Train.npy", "")


def load_train_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*_Train.npy")))

    if len(files) == 0:
        raise FileNotFoundError(f"No _Train.npy files found in {data_dir}")

    return files


def remove_old_codebook_files(output_dir):
    ensure_dir(output_dir)

    patterns = [
        os.path.join(output_dir, "group_*_sub_*_fine.txt"),
        os.path.join(output_dir, "group_*_sub_*_coarse.txt"),
        os.path.join(output_dir, "group_*_sub_*_lut.txt"),
        os.path.join(output_dir, "head_to_codebook_map.json"),
        os.path.join(output_dir, "cluster_summary.json"),
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            os.remove(path)


# ============================================================
# Moment Loading / Clustering
# ============================================================

def load_diag_moments_from_files(file_paths):
    means = []
    vars_ = []

    for f in tqdm(file_paths, desc="Loading diagonal moments"):
        data = np.load(f).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {data.shape}")

        means.append(np.mean(data, axis=0))
        vars_.append(np.var(data, axis=0) + 1e-6)

        del data

    return np.asarray(means, dtype=np.float32), np.asarray(vars_, dtype=np.float32)


def diag_wasserstein_distance(means, vars_):
    device = get_device()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    sqrt_vars_t = torch.sqrt(torch.tensor(vars_, dtype=torch.float32, device=device))

    n = means_t.shape[0]
    dist = torch.empty((n, n), dtype=torch.float32, device=device)

    for start in tqdm(range(0, n, 256), desc="Diagonal Wasserstein distance"):
        end = min(start + 256, n)

        mean_diff = means_t[start:end, None, :] - means_t[None, :, :]
        var_diff = sqrt_vars_t[start:end, None, :] - sqrt_vars_t[None, :, :]

        dist_sq = torch.sum(mean_diff * mean_diff, dim=-1) + torch.sum(var_diff * var_diff, dim=-1)
        dist[start:end] = torch.sqrt(torch.clamp(dist_sq, min=0.0))

        del mean_diff, var_diff, dist_sq

    dist = 0.5 * (dist + dist.T)
    dist_np = dist.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(dist_np, 0.0)

    del means_t, sqrt_vars_t, dist
    cleanup()

    return dist_np


def load_full_moments_from_files(file_paths):
    means = []
    covs = []

    for f in tqdm(file_paths, desc="Loading full moments"):
        data = np.load(f).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {data.shape}")

        means.append(np.mean(data, axis=0))

        if data.shape[0] < 2:
            cov = np.eye(DIMS, dtype=np.float32) * 1e-6
        else:
            cov = np.cov(data, rowvar=False).astype(np.float32)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = 0.5 * (cov + cov.T)
            cov += np.eye(DIMS, dtype=np.float32) * 1e-6

        covs.append(cov)

        del data

    return np.asarray(means, dtype=np.float32), np.asarray(covs, dtype=np.float32)


def safe_sym_sqrt(C):
    C = 0.5 * (C + C.mT)
    vals, vecs = torch.linalg.eigh(C)
    vals = torch.clamp(vals, min=0.0)
    return vecs @ torch.diag(torch.sqrt(vals)) @ vecs.mT


def full_wasserstein_distance(means, covs):
    device = get_device()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    covs_t = torch.tensor(covs, dtype=torch.float32, device=device)

    n = len(means)

    sqrt_covs_t = torch.stack(
        [safe_sym_sqrt(c) for c in tqdm(covs_t, desc="Precomputing covariance sqrts")]
    )

    trace_covs_t = torch.stack([torch.trace(c) for c in covs_t])
    dist_matrix = torch.zeros((n, n), device=device, dtype=torch.float32)

    for i in tqdm(range(n), desc="Full Wasserstein rows"):
        m1 = means_t[i]
        sqrt_c1 = sqrt_covs_t[i]
        trace_c1 = trace_covs_t[i]

        diff = m1 - means_t
        mean_dist = torch.sum(diff ** 2, dim=-1)

        cross_cov = torch.matmul(torch.matmul(sqrt_c1, covs_t), sqrt_c1)
        cross_cov = 0.5 * (cross_cov + cross_cov.transpose(-1, -2))

        vals = torch.linalg.eigvalsh(cross_cov)
        vals = torch.clamp(vals, min=0.0)

        trace_cross = torch.sum(torch.sqrt(vals), dim=-1)
        cov_dist = trace_c1 + trace_covs_t - 2.0 * trace_cross

        dist_sq = mean_dist + cov_dist
        dist_matrix[i] = torch.sqrt(torch.clamp(dist_sq, min=0.0))

    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

    dist = dist_matrix.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(dist, 0.0)

    del means_t, covs_t, sqrt_covs_t, trace_covs_t, dist_matrix
    cleanup()

    return dist


def make_dense_cluster_groups(train_files, dist_matrix, requested_clusters):
    if len(train_files) < requested_clusters:
        raise ValueError(
            f"Cannot create {requested_clusters} clusters from only {len(train_files)} heads."
        )

    condensed_dist = squareform(dist_matrix, checks=False)
    Z = sch.linkage(condensed_dist, method="ward")

    raw_labels = sch.fcluster(Z, t=requested_clusters, criterion="maxclust")
    unique_raw = sorted(set(int(x) for x in raw_labels))

    if len(unique_raw) != requested_clusters:
        warnings.warn(
            f"Requested {requested_clusters} clusters but got "
            f"{len(unique_raw)} non-empty clusters: {unique_raw}"
        )

    label_remap = {
        old_label: new_label
        for new_label, old_label in enumerate(unique_raw, start=1)
    }

    dense_labels = np.array([label_remap[int(x)] for x in raw_labels], dtype=np.int32)

    groups = {}

    for idx, label in enumerate(dense_labels):
        groups.setdefault(int(label), []).append(train_files[idx])

    return groups, dense_labels, raw_labels


def get_or_create_groups(tensor_type, train_files):
    cluster_cache_dir = os.path.join(DRIVE_BASE, f"wasserstein_clusters_{CLUSTERS}_{CLUSTER_METRIC}")
    ensure_dir(cluster_cache_dir)

    cache_path = os.path.join(cluster_cache_dir, f"{tensor_type}_dist.npy")
    labels_cache_path = os.path.join(cluster_cache_dir, f"{tensor_type}_labels.json")

    if USE_CACHED_CLUSTERS and os.path.exists(cache_path) and os.path.exists(labels_cache_path):
        print(f"Loading cached clusters for {tensor_type.upper()}")

        with open(labels_cache_path, "r") as f:
            payload = json.load(f)

        head_to_group = payload["head_to_group"]

        groups = {}

        for path in train_files:
            head = parse_head_name(path)

            if head not in head_to_group:
                raise KeyError(f"Cached cluster labels missing head {head}")

            group_id = int(head_to_group[head].replace("group_", ""))
            groups.setdefault(group_id, []).append(path)

        return groups

    print(f"Generating clusters for {tensor_type.upper()} using {CLUSTER_METRIC}")

    if CLUSTER_METRIC == "diag_wasserstein":
        means, vars_ = load_diag_moments_from_files(train_files)
        dist_matrix = diag_wasserstein_distance(means, vars_)
        del means, vars_

    elif CLUSTER_METRIC == "full_wasserstein":
        means, covs = load_full_moments_from_files(train_files)
        dist_matrix = full_wasserstein_distance(means, covs)
        del means, covs

    else:
        raise ValueError(f"Unknown CLUSTER_METRIC: {CLUSTER_METRIC}")

    np.save(cache_path, dist_matrix)

    groups, dense_labels, raw_labels = make_dense_cluster_groups(
        train_files,
        dist_matrix,
        CLUSTERS,
    )

    head_to_group = {}

    for idx, path in enumerate(train_files):
        head = parse_head_name(path)
        head_to_group[head] = f"group_{int(dense_labels[idx])}"

    payload = {
        "tensor_type": tensor_type,
        "cluster_metric": CLUSTER_METRIC,
        "clusters_requested": CLUSTERS,
        "groups_produced": sorted(groups.keys()),
        "head_to_group": head_to_group,
        "raw_labels": [int(x) for x in raw_labels],
        "dense_labels": [int(x) for x in dense_labels],
    }

    with open(labels_cache_path, "w") as f:
        json.dump(payload, f, indent=2)

    del dist_matrix
    cleanup()

    return groups


# ============================================================
# Raw Data Loading
# ============================================================

def load_raw_data_for_group(file_paths, target_total, min_per_head, tensor_type="keys"):
    num_heads = len(file_paths)

    if num_heads == 0:
        raise ValueError("Empty group")

    base_count = num_heads * min_per_head

    extra_per_head = 0
    if base_count < target_total:
        shortage = target_total - base_count
        extra_per_head = math.ceil(shortage / num_heads)

    per_head_target = min_per_head + extra_per_head
    data = []

    for f in tqdm(file_paths, desc="Loading raw data for group"):
        head_data = np.load(f).astype(np.float32)

        if head_data.ndim != 2 or head_data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {head_data.shape}")

        if len(head_data) >= per_head_target:
            indices = np.random.choice(len(head_data), per_head_target, replace=False)
        else:
            indices = np.random.choice(len(head_data), per_head_target, replace=True)

        sampled = head_data[indices]
        data.append(sampled)

        del head_data, sampled

    data = np.vstack(data).astype(np.float32)

    if tensor_type == "values":
        low, high = np.percentile(data, [0.1, 99.9], axis=0)
        data = np.clip(data, low, high)

    np.random.shuffle(data)

    final_target = max(target_total, base_count)
    data = data[:final_target]

    return data.astype(np.float32)


# ============================================================
# Batched CUDA K-Means
# ============================================================

def init_centroids_batched(x, k):
    s_count, n, dim = x.shape
    device = x.device

    all_idx = []

    for _ in range(s_count):
        all_idx.append(torch.randperm(n, device=device)[:k])

    idx = torch.stack(all_idx, dim=0)

    s_idx = torch.arange(s_count, device=device).view(s_count, 1).expand(s_count, k)

    centroids = x[s_idx, idx, :].contiguous()

    return centroids


def assign_chunk_gemm(x_chunk, centroids):
    x_sq = torch.sum(x_chunk * x_chunk, dim=-1, keepdim=True)
    c_sq = torch.sum(centroids * centroids, dim=-1).unsqueeze(1)
    prod = torch.bmm(x_chunk, centroids.transpose(1, 2))
    dist = x_sq + c_sq - 2.0 * prod
    labels = torch.argmin(dist, dim=-1)
    return labels


def batched_kmeans_cuda(
    x,
    k,
    max_iters,
    chunk_size,
    tol,
    return_labels=False,
    desc="kmeans",
):
    device = get_device()

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device=device, dtype=torch.float32)

    x = x.contiguous()

    if x.ndim != 3:
        raise ValueError(f"Expected x shape [S, N, D], got {tuple(x.shape)}")

    s_count, n, dim = x.shape

    if n < k:
        raise ValueError(f"kmeans needs N >= K, got N={n}, K={k}")

    centroids = init_centroids_batched(x, k)

    prev_shift = None
    final_labels_cpu = None

    iterator = tqdm(range(max_iters), desc=desc)

    for _ in iterator:
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((s_count, k), dtype=torch.float32, device=device)

        all_labels = [] if return_labels else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            x_chunk = x[:, start:end, :].contiguous()
            labels = assign_chunk_gemm(x_chunk, centroids)

            if return_labels:
                all_labels.append(labels.detach().cpu())

            for s in range(s_count):
                lab_s = labels[s]
                x_s = x_chunk[s]

                new_centroids[s].scatter_add_(
                    0,
                    lab_s.view(-1, 1).expand(-1, dim),
                    x_s,
                )

                ones = torch.ones((end - start,), dtype=torch.float32, device=device)
                counts[s].scatter_add_(0, lab_s, ones)

            del x_chunk, labels

        empty = counts == 0

        if empty.any():
            for s in range(s_count):
                empty_s = torch.where(empty[s])[0]
                if empty_s.numel() > 0:
                    replacement_idx = torch.randperm(n, device=device)[:empty_s.numel()]
                    new_centroids[s, empty_s, :] = x[s, replacement_idx, :]
                    counts[s, empty_s] = 1.0

        new_centroids = new_centroids / torch.clamp(counts.unsqueeze(-1), min=1.0)

        shift = torch.mean(torch.norm(new_centroids - centroids, dim=-1)).item()
        iterator.set_postfix({"shift": f"{shift:.6f}"})

        centroids = new_centroids

        if prev_shift is not None and abs(prev_shift - shift) < tol:
            break

        if shift < tol:
            break

        prev_shift = shift

        if return_labels:
            final_labels_cpu = torch.cat(all_labels, dim=1).numpy().astype(np.int32)

        del counts, empty
        cleanup()

    if return_labels:
        final_labels = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x[:, start:end, :].contiguous()
            labels = assign_chunk_gemm(x_chunk, centroids)
            final_labels.append(labels.detach().cpu())
            del x_chunk, labels

        final_labels_cpu = torch.cat(final_labels, dim=1).numpy().astype(np.int32)

    centroids_np = centroids.detach().cpu().numpy().astype(np.float32)

    del x, centroids
    cleanup()

    return centroids_np, final_labels_cpu


def train_all_subvector_codebooks(raw_data):
    sub_dim = DIMS // NUM_SUBVECTORS

    if DIMS % NUM_SUBVECTORS != 0:
        raise ValueError(f"DIMS={DIMS} must be divisible by NUM_SUBVECTORS={NUM_SUBVECTORS}")

    if raw_data.ndim != 2 or raw_data.shape[1] != DIMS:
        raise ValueError(f"Expected raw_data shape [N, {DIMS}], got {raw_data.shape}")

    n = raw_data.shape[0]

    sub_data = raw_data.reshape(n, NUM_SUBVECTORS, sub_dim)
    sub_data = np.transpose(sub_data, (1, 0, 2)).copy()

    fine_centroids, _ = batched_kmeans_cuda(
        sub_data,
        k=CODEWORDS,
        max_iters=KMEANS_MAX_ITERS_FINE,
        chunk_size=BATCHED_KMEANS_CHUNK,
        tol=KMEANS_TOL,
        return_labels=False,
        desc="Fine k-means all subvectors",
    )

    num_coarse = int(math.sqrt(CODEWORDS))

    coarse_centroids, lut = batched_kmeans_cuda(
        fine_centroids,
        k=num_coarse,
        max_iters=KMEANS_MAX_ITERS_COARSE,
        chunk_size=BATCHED_KMEANS_CHUNK,
        tol=KMEANS_TOL,
        return_labels=True,
        desc="Coarse k-means all subvectors",
    )

    return fine_centroids, coarse_centroids, lut


# ============================================================
# Saving / Validation
# ============================================================

def save_group_codebooks(output_dir, group_id, fine_centroids, coarse_centroids, lut):
    ensure_dir(output_dir)

    if fine_centroids.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"fine_centroids expected first dim {NUM_SUBVECTORS}, got {fine_centroids.shape}")

    if coarse_centroids.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"coarse_centroids expected first dim {NUM_SUBVECTORS}, got {coarse_centroids.shape}")

    if lut.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"lut expected first dim {NUM_SUBVECTORS}, got {lut.shape}")

    for s in tqdm(range(NUM_SUBVECTORS), desc=f"Saving group_{group_id}"):
        prefix = f"group_{group_id}_sub_{s}"

        np.savetxt(
            os.path.join(output_dir, f"{prefix}_fine.txt"),
            fine_centroids[s],
            fmt="%.6f",
        )

        np.savetxt(
            os.path.join(output_dir, f"{prefix}_coarse.txt"),
            coarse_centroids[s],
            fmt="%.6f",
        )

        with open(os.path.join(output_dir, f"{prefix}_lut.txt"), "w") as f:
            json.dump(lut[s].astype(int).tolist(), f)


def write_head_map(output_dir, groups):
    head_to_codebook_map = {}

    for group_id, paths in sorted(groups.items()):
        for p in paths:
            head_to_codebook_map[os.path.basename(p)] = f"group_{group_id}"

    with open(os.path.join(output_dir, "head_to_codebook_map.json"), "w") as f:
        json.dump(head_to_codebook_map, f, indent=2)

    return head_to_codebook_map


def write_cluster_summary(output_dir, tensor_type, groups, head_to_codebook_map):
    summary = {
        "tensor_type": tensor_type,
        "model_name": MODEL_NAME,
        "cluster_metric": CLUSTER_METRIC,
        "clusters_requested": CLUSTERS,
        "groups_produced": sorted(int(g) for g in groups.keys()),
        "num_groups_produced": len(groups),
        "dims": DIMS,
        "num_subvectors": NUM_SUBVECTORS,
        "sub_dim": DIMS // NUM_SUBVECTORS,
        "codewords": CODEWORDS,
        "target_train_size": TARGET_TRAIN_SIZE,
        "min_vectors_per_head": MIN_VECTORS_PER_HEAD,
        "kmeans_max_iters_fine": KMEANS_MAX_ITERS_FINE,
        "kmeans_max_iters_coarse": KMEANS_MAX_ITERS_COARSE,
        "group_sizes": {
            f"group_{int(g)}": len(paths)
            for g, paths in sorted(groups.items())
        },
        "heads": head_to_codebook_map,
    }

    with open(os.path.join(output_dir, "cluster_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def validate_generated_codebooks(output_dir, groups):
    missing = []

    for group_id in sorted(groups.keys()):
        for s in range(NUM_SUBVECTORS):
            prefix = f"group_{group_id}_sub_{s}"

            fine_path = os.path.join(output_dir, f"{prefix}_fine.txt")
            coarse_path = os.path.join(output_dir, f"{prefix}_coarse.txt")
            lut_path = os.path.join(output_dir, f"{prefix}_lut.txt")

            if not os.path.exists(fine_path):
                missing.append(fine_path)

            if not os.path.exists(coarse_path):
                missing.append(coarse_path)

            if not os.path.exists(lut_path):
                missing.append(lut_path)

    if missing:
        raise FileNotFoundError(
            "Generated codebook validation failed. Missing files:\n"
            + "\n".join(missing[:100])
        )


def discover_complete_groups(base_dir):
    group_files = {}

    for path in glob.glob(os.path.join(base_dir, "group_*_sub_*_fine.txt")):
        filename = os.path.basename(path)
        group_name = filename.split("_sub_")[0]
        sub_idx = int(filename.split("_sub_")[1].split("_")[0])
        group_files.setdefault(group_name, set()).add(sub_idx)

    complete = sorted(
        group
        for group, subs in group_files.items()
        if len(subs) == NUM_SUBVECTORS and set(subs) == set(range(NUM_SUBVECTORS))
    )

    return complete


def validate_runtime_compatibility():
    print("\n============================================================")
    print("Runtime compatibility validation")
    print("============================================================")

    for tensor_type in ["keys", "values"]:
        output_dir = os.path.join(DRIVE_BASE, CODEBOOK_NAME, tensor_type)
        map_path = os.path.join(output_dir, "head_to_codebook_map.json")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Missing map: {map_path}")

        with open(map_path, "r") as f:
            head_map = json.load(f)

        requested = sorted(set(head_map.values()))
        complete = discover_complete_groups(output_dir)

        missing = sorted(set(requested) - set(complete))

        print(f"\n{tensor_type}:")
        print(f"  map groups:      {requested}")
        print(f"  complete groups: {complete}")

        if missing:
            raise FileNotFoundError(
                f"{tensor_type} map references missing/incomplete groups: {missing}"
            )

    print("\nRuntime compatibility validation passed.")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline_for_tensor_type(tensor_type="keys"):
    data_dir = os.path.join(DRIVE_BASE, "pq_training_data", tensor_type)
    output_dir = os.path.join(DRIVE_BASE, CODEBOOK_NAME, tensor_type)

    ensure_dir(output_dir)

    if USE_CACHED_CLUSTERS and os.path.exists(os.path.join(output_dir, "head_to_codebook_map.json")):
        print(f"Using existing codebooks for {tensor_type.upper()} ({CODEBOOK_NAME}).")
        return

    print("\n============================================================")
    print(f"Processing {tensor_type.upper()}")
    print(f"Output: {output_dir}")
    print("============================================================")

    remove_old_codebook_files(output_dir)

    train_files = load_train_files(data_dir)
    print(f"Found {len(train_files)} training files.")

    groups = get_or_create_groups(tensor_type, train_files)

    print(f"{tensor_type.upper()} produced dense groups: {sorted(groups.keys())}")

    if len(groups) != CLUSTERS:
        warnings.warn(
            f"{tensor_type.upper()}: requested {CLUSTERS} clusters, "
            f"but got {len(groups)} non-empty groups."
        )

    for group_id, paths in sorted(groups.items()):
        print("\n------------------------------------------------------------")
        print(f"{tensor_type.upper()} group_{group_id}: {len(paths)} heads")
        print("------------------------------------------------------------")

        raw_data = load_raw_data_for_group(
            paths,
            TARGET_TRAIN_SIZE,
            MIN_VECTORS_PER_HEAD,
            tensor_type,
        )

        print(f"Training data shape for group_{group_id}: {raw_data.shape}")

        fine_centroids, coarse_centroids, lut = train_all_subvector_codebooks(raw_data)

        save_group_codebooks(
            output_dir,
            group_id,
            fine_centroids,
            coarse_centroids,
            lut,
        )

        del raw_data, fine_centroids, coarse_centroids, lut
        cleanup()

    head_to_codebook_map = write_head_map(output_dir, groups)

    write_cluster_summary(
        output_dir,
        tensor_type,
        groups,
        head_to_codebook_map,
    )

    validate_generated_codebooks(output_dir, groups)

    print(f"\nFinished {tensor_type.upper()}.")
    print(f"Generated groups: {sorted(groups.keys())}")
    print(f"Map path: {os.path.join(output_dir, 'head_to_codebook_map.json')}")


if __name__ == "__main__":
    set_seed(SEED)
    setup_torch()

    if DIMS % NUM_SUBVECTORS != 0:
        raise ValueError(
            f"DIMS={DIMS} must be divisible by NUM_SUBVECTORS={NUM_SUBVECTORS}"
        )

    print(f"Using device: {get_device()}")

    run_pipeline_for_tensor_type("keys")
    run_pipeline_for_tensor_type("values")

    validate_runtime_compatibility()

    local_codebooks_versioned = os.path.join(DRIVE_BASE, CODEBOOK_NAME)
    drive_codebooks_versioned = os.path.join(DRIVE_OUTPUT_BASE, CODEBOOK_NAME)

    if SAVE_TO_DRIVE and os.path.exists(local_codebooks_versioned):
        print("\nSaving results to Google Drive:")
        print(f"  from: {local_codebooks_versioned}")
        print(f"  to:   {drive_codebooks_versioned}")

        ensure_dir(os.path.dirname(drive_codebooks_versioned))

        shutil.copytree(
            local_codebooks_versioned,
            drive_codebooks_versioned,
            dirs_exist_ok=True,
        )

    print("\nDone.")

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="B-FHtYwfM1rv" outputId="14e58a9c-6acc-417d-f6bd-e03c539aaf26"
import os
import glob
import json
import math
import shutil
import gc
import time
import warnings
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import torch
from tqdm import tqdm
from collections import Counter, defaultdict


# ============================================================
# Configurable Parameters
# PQ_BRIDGE_CELL_MARKER: longbench_adaptive_codebook_training
# ============================================================

MODEL_NAME = "qwen3_8B"
DRIVE_BASE = f"/content/{MODEL_NAME}"

# Output produced by the LongBench calibration-vector extraction script.
_training_mode_env = os.environ.get(
    "PQ_TRAINING_CALIBRATION_MODE", "held_out"
).strip().lower()
if _training_mode_env not in {"held_out", "matched", "contaminated"}:
    raise ValueError(
        "PQ_TRAINING_CALIBRATION_MODE must be held_out, matched, or contaminated"
    )
EXPECTED_CALIBRATION_MODE = _training_mode_env
_training_variant_env = os.environ.get(
    "PQ_TRAINING_CALIBRATION_VARIANT", "prior_held_out"
).strip().lower()
TRAINING_VARIANT_ALIASES = {
    "": "prior_held_out",
    "none": "prior_held_out",
    "held_out": "prior_held_out",
    "prior": "prior_held_out",
    "prior_held_out": "prior_held_out",
    "clean_hotpot_a": "clean_hotpot_a",
    "hotpot_a": "clean_hotpot_a",
    "variant_a": "clean_hotpot_a",
    "clean_suite_b": "clean_suite_b",
    "suite_b": "clean_suite_b",
    "variant_b": "clean_suite_b",
    "clean_qa_count_c": "clean_qa_count_c",
    "qa_count_c": "clean_qa_count_c",
    "variant_c": "clean_qa_count_c",
}
if _training_variant_env not in TRAINING_VARIANT_ALIASES:
    raise ValueError(
        "PQ_TRAINING_CALIBRATION_VARIANT must be prior_held_out, "
        "clean_hotpot_a, clean_suite_b, or clean_qa_count_c"
    )
EXPECTED_CALIBRATION_VARIANT = TRAINING_VARIANT_ALIASES[_training_variant_env]
if EXPECTED_CALIBRATION_VARIANT != "prior_held_out" and EXPECTED_CALIBRATION_MODE != "held_out":
    raise ValueError("Clean training variants require PQ_TRAINING_CALIBRATION_MODE=held_out")
TRAINING_OUTPUT_TAG = (
    EXPECTED_CALIBRATION_MODE
    if EXPECTED_CALIBRATION_VARIANT == "prior_held_out"
    else EXPECTED_CALIBRATION_VARIANT
)
TRAINING_DATA_NAME = (
    f"pq_training_data_longbench_e_{TRAINING_OUTPUT_TAG}_4096"
)
TRAINING_DATA_ROOT = os.path.join(DRIVE_BASE, TRAINING_DATA_NAME)
CALIBRATION_MANIFEST = os.path.join(TRAINING_DATA_ROOT, "calibration_manifest.json")

CLUSTERS = 64
USE_CACHED_CLUSTERS = False

# Balanced head-group constraints. These bounds apply to the number of heads
# sharing each codebook group.
MIN_HEADS_PER_GROUP = 2
MAX_HEADS_PER_GROUP = 10
CLUSTER_LINKAGE = "average"   # valid for a precomputed distance matrix

DIMS = 128
NUM_SUBVECTORS = 64
CODEWORDS = 128

# Adaptive group budget used by clean Variant A/B:
# group_target = max(50,000, 10,000 * number_of_heads_in_group).
TARGET_TRAIN_SIZE = 50000
ADAPTIVE_GROUP_MIN_TOTAL = 50000
ADAPTIVE_GROUP_PER_HEAD = 10000
MIN_VECTORS_PER_HEAD = 5000

# When a group cannot supply per_head_target vectors, cap at what exists rather than
# duplicating points. 18k points for 128 centroids in 2D is ~140 points/centroid,
# which is plenty; duplication just reweights the same data.
ALLOW_DUPLICATION = False

# Do not clip value activations during codebook training. Evaluation does not clip,
# so leaving the training distribution untouched avoids a train/eval mismatch.
CLIP_VALUE_TRAINING_DATA = False

SEED = 1234

SAVE_TO_DRIVE = False
DRIVE_OUTPUT_BASE = f"/content/drive/MyDrive/{MODEL_NAME}"

# Keep these codebooks separate from the original WikiText-trained codebooks.
EXPECTED_EVAL_TASKS = {
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
}
EXPECTED_LONGBENCH_E_REVISION = "36914d6211386125c6fc4ce7db4a6a777fadd34c"
CODEBOOK_TAG = (
    f"longbench_e_{TRAINING_OUTPUT_TAG}_4096_adaptive10k"
    if EXPECTED_CALIBRATION_VARIANT != "prior_held_out"
    else f"longbench_e_{EXPECTED_CALIBRATION_MODE}_4096_balanced_kpp_noclip"
)
CODEBOOK_NAME = (
    f"codebooks_{NUM_SUBVECTORS}_{CODEWORDS}_{CLUSTERS}_{CODEBOOK_TAG}"
)

# ---- built-in reconstruction MSE check -----------------------------------------------
# Runs after each side finishes training, on the held-out *_Test.npy vectors that no
# training run reads. Nothing else in this pipeline measures whether the codebooks are
# any good -- validate_generated_codebooks() only checks that files exist.
RUN_MSE_CHECK = True
MSE_MAX_VECTORS_PER_HEAD = 2000     # cap for speed; the check should take seconds
MSE_OUTLIER_DIMS = 3                # mirrors what the eval keeps in fp16
MSE_REPORT_NAME = "codebook_mse_report.json"
STATIC_OUTLIER_DIMS = 3
STATIC_MASK_REPORT_NAME = "static_outlier_masks.json"

# Optional: an existing codebook root to compare against (must contain keys/ and
# values/ with their own head_to_codebook_map.json). Set to None to skip.
MSE_BASELINE_CODEBOOK_ROOT = os.path.join(DRIVE_BASE, "codebooks_64_128_64")

CLUSTER_METRIC = "diag_wasserstein"

# CUDA k-means controls.
KMEANS_MAX_ITERS_FINE = 150
KMEANS_MAX_ITERS_COARSE = 40
KMEANS_RESTARTS_FINE = 4
KMEANS_RESTARTS_COARSE = 1
KMEANS_INIT = "kmeans++"

# Relative tolerance: stop when mean centroid movement falls below this fraction of the
# mean centroid norm.
KMEANS_TOL = 1e-4
KMEANS_ASSIGN_CHUNK = 1024

BATCHED_KMEANS_CHUNK = KMEANS_ASSIGN_CHUNK


# ============================================================
# Setup Utilities
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_and_validate_calibration_manifest():
    """Verify that the trainer is consuming the intended extraction output."""
    if not os.path.exists(CALIBRATION_MANIFEST):
        raise FileNotFoundError(
            f"Missing calibration manifest: {CALIBRATION_MANIFEST}\n"
            "Run the LongBench calibration-vector extraction script first."
        )

    with open(CALIBRATION_MANIFEST, "r") as f:
        manifest = json.load(f)

    mode = manifest.get("calibration_mode")
    if mode != EXPECTED_CALIBRATION_MODE:
        raise ValueError(
            f"Expected calibration_mode={EXPECTED_CALIBRATION_MODE!r}, "
            f"but manifest reports {mode!r}. "
            "Change EXPECTED_CALIBRATION_MODE intentionally before training "
            "a different calibration variant."
        )
    variant = manifest.get("calibration_variant", "prior_held_out")
    if variant != EXPECTED_CALIBRATION_VARIANT:
        raise ValueError(
            f"Expected calibration_variant={EXPECTED_CALIBRATION_VARIANT!r}, "
            f"but manifest reports {variant!r}."
        )

    if int(manifest.get("max_input_length", -1)) != 4096:
        warnings.warn(
            "Calibration max_input_length is not 4096; results may not be "
            "directly comparable to the pinned evaluation."
        )

    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Calibration manifest contains no source samples.")

    eval_tasks = set(manifest.get("eval_tasks", []))
    if eval_tasks != EXPECTED_EVAL_TASKS:
        raise ValueError(
            "Calibration manifest eval_tasks do not match the pinned LongBench-E "
            f"suite. Expected {sorted(EXPECTED_EVAL_TASKS)}, got {sorted(eval_tasks)}."
        )
    calib_tasks_used = {row.get("task") for row in samples}
    overlap = sorted(eval_tasks & calib_tasks_used)
    if mode == "held_out" and variant == "prior_held_out" and overlap:
        raise ValueError(
            f"Held-out calibration is contaminated by eval tasks: {overlap}"
        )
    if variant != "prior_held_out":
        if manifest.get("source_dataset_variant") != variant:
            raise ValueError(
                f"Clean variant manifest source_dataset_variant must be {variant!r}"
            )
        decon = manifest.get("decontamination_report")
        if not decon or decon.get("longbench_e_revision") != EXPECTED_LONGBENCH_E_REVISION:
            raise ValueError(
                "Clean variant manifest is missing the pinned LongBench-E "
                "decontamination report."
            )
        rejected = decon.get("rejected_counts", {})
        print(f"  decontamination rejected counts: {rejected}")
    if mode in {"matched", "contaminated"}:
        if manifest.get("source_dataset_variant") != "longbench_e":
            raise ValueError(
                "Matched/contaminated calibration must use exact LongBench-E data"
            )
        if manifest.get("longbench_e_revision") != EXPECTED_LONGBENCH_E_REVISION:
            raise ValueError(
                "Calibration LongBench-E revision does not match the evaluator"
            )
        unexpected = sorted(calib_tasks_used - eval_tasks)
        if unexpected:
            raise ValueError(
                f"Matched/contaminated calibration contains non-eval tasks: {unexpected}"
            )
    if mode == "contaminated" and not overlap:
        raise ValueError(
            "Contaminated calibration contains no LongBench-E evaluation documents"
        )

    n_docs = manifest.get("num_documents", len(samples))

    print("\nCalibration provenance")
    print(f"  mode:              {mode}")
    print(f"  variant:           {variant}")
    print(f"  source directory:  {TRAINING_DATA_ROOT}")
    print(f"  documents:         {n_docs}")
    print(f"  tasks used:        {sorted(calib_tasks_used)}")
    print(f"  vectors/head:      {manifest.get('vectors_per_head')}")
    print(f"  test vectors/head: {manifest.get('test_vectors_per_head', 'n/a')}")
    print(f"  split granularity: {manifest.get('split_granularity', 'n/a')}")
    if manifest.get("position_index_train"):
        print(f"  train sidecar:     {manifest['position_index_train'].get('path')}")

    # Document count is the number that determines effective sample size; positions
    # within one document are heavily correlated.
    if isinstance(n_docs, int) and n_docs < 150:
        warnings.warn(
            f"Only {n_docs} calibration documents. Positions within a document are "
            "highly correlated, so effective sample size is roughly the document "
            "count, not the vector count. Raise NUM_CALIB_SAMPLES in the extractor."
        )

    return manifest


def parse_head_name(path):
    return os.path.basename(path).replace("_Train.npy", "")


def load_train_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*_Train.npy")))

    if len(files) == 0:
        raise FileNotFoundError(f"No _Train.npy files found in {data_dir}")

    return files


def remove_old_codebook_files(output_dir):
    ensure_dir(output_dir)

    patterns = [
        os.path.join(output_dir, "group_*_sub_*_fine.txt"),
        os.path.join(output_dir, "group_*_sub_*_coarse.txt"),
        os.path.join(output_dir, "group_*_sub_*_lut.txt"),
        os.path.join(output_dir, "head_to_codebook_map.json"),
        os.path.join(output_dir, "cluster_summary.json"),
        os.path.join(output_dir, MSE_REPORT_NAME),
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            os.remove(path)


# ============================================================
# Moment Loading / Clustering
# ============================================================

def load_diag_moments_from_files(file_paths):
    means = []
    vars_ = []

    for f in tqdm(file_paths, desc="Loading diagonal moments"):
        data = np.load(f).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {data.shape}")

        means.append(np.mean(data, axis=0))
        vars_.append(np.var(data, axis=0) + 1e-6)

        del data

    return np.asarray(means, dtype=np.float32), np.asarray(vars_, dtype=np.float32)


def diag_wasserstein_distance(means, vars_):
    device = get_device()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    sqrt_vars_t = torch.sqrt(torch.tensor(vars_, dtype=torch.float32, device=device))

    n = means_t.shape[0]
    dist = torch.empty((n, n), dtype=torch.float32, device=device)

    for start in tqdm(range(0, n, 256), desc="Diagonal Wasserstein distance"):
        end = min(start + 256, n)

        mean_diff = means_t[start:end, None, :] - means_t[None, :, :]
        var_diff = sqrt_vars_t[start:end, None, :] - sqrt_vars_t[None, :, :]

        dist_sq = torch.sum(mean_diff * mean_diff, dim=-1) + torch.sum(var_diff * var_diff, dim=-1)
        dist[start:end] = torch.sqrt(torch.clamp(dist_sq, min=0.0))

        del mean_diff, var_diff, dist_sq

    dist = 0.5 * (dist + dist.T)
    dist_np = dist.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(dist_np, 0.0)

    del means_t, sqrt_vars_t, dist
    cleanup()

    return dist_np


def load_full_moments_from_files(file_paths):
    means = []
    covs = []

    for f in tqdm(file_paths, desc="Loading full moments"):
        data = np.load(f).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {data.shape}")

        means.append(np.mean(data, axis=0))

        if data.shape[0] < 2:
            cov = np.eye(DIMS, dtype=np.float32) * 1e-6
        else:
            cov = np.cov(data, rowvar=False).astype(np.float32)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = 0.5 * (cov + cov.T)
            cov += np.eye(DIMS, dtype=np.float32) * 1e-6

        covs.append(cov)

        del data

    return np.asarray(means, dtype=np.float32), np.asarray(covs, dtype=np.float32)


def safe_sym_sqrt(C):
    C = 0.5 * (C + C.mT)
    vals, vecs = torch.linalg.eigh(C)
    vals = torch.clamp(vals, min=0.0)
    return vecs @ torch.diag(torch.sqrt(vals)) @ vecs.mT


def full_wasserstein_distance(means, covs):
    device = get_device()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    covs_t = torch.tensor(covs, dtype=torch.float32, device=device)

    n = len(means)

    sqrt_covs_t = torch.stack(
        [safe_sym_sqrt(c) for c in tqdm(covs_t, desc="Precomputing covariance sqrts")]
    )

    trace_covs_t = torch.stack([torch.trace(c) for c in covs_t])
    dist_matrix = torch.zeros((n, n), device=device, dtype=torch.float32)

    for i in tqdm(range(n), desc="Full Wasserstein rows"):
        m1 = means_t[i]
        sqrt_c1 = sqrt_covs_t[i]
        trace_c1 = trace_covs_t[i]

        diff = m1 - means_t
        mean_dist = torch.sum(diff ** 2, dim=-1)

        cross_cov = torch.matmul(torch.matmul(sqrt_c1, covs_t), sqrt_c1)
        cross_cov = 0.5 * (cross_cov + cross_cov.transpose(-1, -2))

        vals = torch.linalg.eigvalsh(cross_cov)
        vals = torch.clamp(vals, min=0.0)

        trace_cross = torch.sum(torch.sqrt(vals), dim=-1)
        cov_dist = trace_c1 + trace_covs_t - 2.0 * trace_cross

        dist_sq = mean_dist + cov_dist
        dist_matrix[i] = torch.sqrt(torch.clamp(dist_sq, min=0.0))

    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

    dist = dist_matrix.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(dist, 0.0)

    del means_t, covs_t, sqrt_covs_t, trace_covs_t, dist_matrix
    cleanup()

    return dist


def make_dense_cluster_groups(train_files, dist_matrix, requested_clusters):
    """Create exactly requested_clusters balanced groups using agglomerative merging.

    The original Ward linkage was not appropriate for a precomputed arbitrary
    distance matrix and could create highly pathological singleton/giant groups.
    This routine starts with one head per group and greedily merges the nearest
    pair whose combined size does not exceed MAX_HEADS_PER_GROUP. It then repairs
    undersized groups until every group has at least MIN_HEADS_PER_GROUP.
    """
    n = len(train_files)

    if n < requested_clusters * MIN_HEADS_PER_GROUP:
        raise ValueError(
            f"Cannot create {requested_clusters} groups with minimum size "
            f"{MIN_HEADS_PER_GROUP} from only {n} heads."
        )

    if n > requested_clusters * MAX_HEADS_PER_GROUP:
        raise ValueError(
            f"Cannot create {requested_clusters} groups with maximum size "
            f"{MAX_HEADS_PER_GROUP} from {n} heads."
        )

    clusters = [[i] for i in range(n)]

    def cluster_distance(a, b):
        block = dist_matrix[np.ix_(a, b)]
        if CLUSTER_LINKAGE == "average":
            return float(block.mean())
        if CLUSTER_LINKAGE == "complete":
            return float(block.max())
        if CLUSTER_LINKAGE == "single":
            return float(block.min())
        raise ValueError(f"Unknown CLUSTER_LINKAGE: {CLUSTER_LINKAGE}")

    # Greedy constrained agglomeration until exactly requested_clusters remain.
    while len(clusters) > requested_clusters:
        best = None
        best_dist = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if len(clusters[i]) + len(clusters[j]) > MAX_HEADS_PER_GROUP:
                    continue
                d = cluster_distance(clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best = (i, j)

        if best is None:
            raise RuntimeError(
                "Balanced clustering became infeasible before reaching the requested "
                "number of groups. Relax MIN/MAX_HEADS_PER_GROUP."
            )

        i, j = best
        merged = clusters[i] + clusters[j]
        clusters[i] = merged
        del clusters[j]

    # Repair any group below the minimum size by moving heads from donor groups.
    while True:
        small_idx = next(
            (i for i, c in enumerate(clusters) if len(c) < MIN_HEADS_PER_GROUP),
            None,
        )
        if small_idx is None:
            break

        small = clusters[small_idx]
        best_move = None
        best_cost = float("inf")

        for donor_idx, donor in enumerate(clusters):
            if donor_idx == small_idx or len(donor) <= MIN_HEADS_PER_GROUP:
                continue

            for head in donor:
                remaining = [x for x in donor if x != head]
                attach_cost = cluster_distance([head], small)
                donor_penalty = (
                    cluster_distance([head], remaining) if remaining else 0.0
                )
                cost = attach_cost - donor_penalty

                if cost < best_cost and len(small) + 1 <= MAX_HEADS_PER_GROUP:
                    best_cost = cost
                    best_move = (donor_idx, head)

        if best_move is None:
            raise RuntimeError(
                "Could not repair undersized groups within the configured bounds."
            )

        donor_idx, head = best_move
        clusters[donor_idx].remove(head)
        clusters[small_idx].append(head)

    sizes = [len(c) for c in clusters]
    if len(clusters) != requested_clusters:
        raise RuntimeError("Balanced clustering produced the wrong number of groups.")
    if min(sizes) < MIN_HEADS_PER_GROUP or max(sizes) > MAX_HEADS_PER_GROUP:
        raise RuntimeError(f"Balanced group-size validation failed: {sizes}")

    groups = {}
    dense_labels = np.zeros(n, dtype=np.int32)

    for group_id, members in enumerate(clusters, start=1):
        groups[group_id] = [train_files[i] for i in members]
        for i in members:
            dense_labels[i] = group_id

    # Preserve the old return signature; labels are already dense.
    raw_labels = dense_labels.copy()

    print(
        f"Balanced groups: count={len(groups)}, min={min(sizes)}, "
        f"max={max(sizes)}, mean={np.mean(sizes):.2f}"
    )

    return groups, dense_labels, raw_labels


def get_or_create_groups(tensor_type, train_files):
    cluster_cache_dir = os.path.join(
        DRIVE_BASE,
        f"wasserstein_clusters_{CLUSTERS}_{CLUSTER_METRIC}_{CODEBOOK_TAG}",
    )
    ensure_dir(cluster_cache_dir)

    cache_path = os.path.join(cluster_cache_dir, f"{tensor_type}_dist.npy")
    labels_cache_path = os.path.join(cluster_cache_dir, f"{tensor_type}_labels.json")

    if USE_CACHED_CLUSTERS and os.path.exists(cache_path) and os.path.exists(labels_cache_path):
        print(f"Loading cached clusters for {tensor_type.upper()}")

        with open(labels_cache_path, "r") as f:
            payload = json.load(f)

        head_to_group = payload["head_to_group"]

        groups = {}

        for path in train_files:
            head = parse_head_name(path)

            if head not in head_to_group:
                raise KeyError(f"Cached cluster labels missing head {head}")

            group_id = int(head_to_group[head].replace("group_", ""))
            groups.setdefault(group_id, []).append(path)

        return groups

    print(f"Generating clusters for {tensor_type.upper()} using {CLUSTER_METRIC}")

    if CLUSTER_METRIC == "diag_wasserstein":
        means, vars_ = load_diag_moments_from_files(train_files)
        dist_matrix = diag_wasserstein_distance(means, vars_)
        del means, vars_

    elif CLUSTER_METRIC == "full_wasserstein":
        means, covs = load_full_moments_from_files(train_files)
        dist_matrix = full_wasserstein_distance(means, covs)
        del means, covs

    else:
        raise ValueError(f"Unknown CLUSTER_METRIC: {CLUSTER_METRIC}")

    np.save(cache_path, dist_matrix)

    groups, dense_labels, raw_labels = make_dense_cluster_groups(
        train_files,
        dist_matrix,
        CLUSTERS,
    )

    head_to_group = {}

    for idx, path in enumerate(train_files):
        head = parse_head_name(path)
        head_to_group[head] = f"group_{int(dense_labels[idx])}"

    payload = {
        "tensor_type": tensor_type,
        "cluster_metric": CLUSTER_METRIC,
        "cluster_linkage": CLUSTER_LINKAGE,
        "min_heads_per_group": MIN_HEADS_PER_GROUP,
        "max_heads_per_group": MAX_HEADS_PER_GROUP,
        "clusters_requested": CLUSTERS,
        "groups_produced": sorted(groups.keys()),
        "head_to_group": head_to_group,
        "raw_labels": [int(x) for x in raw_labels],
        "dense_labels": [int(x) for x in dense_labels],
    }

    with open(labels_cache_path, "w") as f:
        json.dump(payload, f, indent=2)

    del dist_matrix
    cleanup()

    return groups


# ============================================================
# Raw Data Loading
# ============================================================

_POSITION_INDEX_CACHE = None


def adaptive_group_target(num_heads):
    return max(ADAPTIVE_GROUP_MIN_TOTAL, ADAPTIVE_GROUP_PER_HEAD * int(num_heads))


def validate_adaptive_group_budget_rule():
    expected = {
        2: 50000,
        3: 50000,
        4: 50000,
        5: 50000,
        6: 60000,
        7: 70000,
        8: 80000,
        9: 90000,
        10: 100000,
    }
    observed = {heads: adaptive_group_target(heads) for heads in range(2, 11)}
    if observed != expected:
        raise AssertionError(
            f"Adaptive group budget validation failed: {observed} != {expected}"
        )
    print(f"Adaptive group budget validation passed: {observed}")


def load_position_index():
    global _POSITION_INDEX_CACHE
    if _POSITION_INDEX_CACHE is not None:
        return _POSITION_INDEX_CACHE

    path = os.path.join(TRAINING_DATA_ROOT, "position_index_train.jsonl")
    if not os.path.exists(path):
        _POSITION_INDEX_CACHE = None
        return None

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["row_index"]))
    _POSITION_INDEX_CACHE = rows
    return rows


def stratified_unique_indices(available, requested, metadata_rows, seed_offset):
    if requested <= 0 or available <= 0:
        return np.array([], dtype=np.int64), {}

    rng = np.random.default_rng(SEED + seed_offset)
    realized = min(int(requested), int(available))
    if not metadata_rows or len(metadata_rows) < available:
        return rng.choice(available, size=realized, replace=False), {
            "stratified": False,
            "reason": "missing_or_short_position_index",
        }

    buckets = defaultdict(list)
    for row in metadata_rows[:available]:
        key = (
            row.get("source_family") or row.get("source_dataset") or row.get("task") or "unknown",
            row.get("position_role") or "unknown",
        )
        buckets[key].append(int(row["row_index"]))

    total_available = sum(len(v) for v in buckets.values())
    allocations = {}
    fractional = []
    for key, indices in buckets.items():
        raw = realized * len(indices) / max(total_available, 1)
        take = min(len(indices), int(math.floor(raw)))
        allocations[key] = take
        fractional.append((raw - take, key))

    remaining = realized - sum(allocations.values())
    for _, key in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        room = len(buckets[key]) - allocations[key]
        if room <= 0:
            continue
        add = min(room, remaining)
        allocations[key] += add
        remaining -= add

    selected = []
    for key, take in allocations.items():
        if take <= 0:
            continue
        selected.extend(rng.choice(buckets[key], size=take, replace=False).tolist())

    if len(selected) < realized:
        used = set(selected)
        fallback = [i for i in range(available) if i not in used]
        need = min(realized - len(selected), len(fallback))
        if need > 0:
            selected.extend(rng.choice(fallback, size=need, replace=False).tolist())

    rng.shuffle(selected)
    role_counts = Counter(
        metadata_rows[i].get("position_role", "unknown")
        for i in selected
        if i < len(metadata_rows)
    )
    task_counts = Counter(
        metadata_rows[i].get("task", "unknown")
        for i in selected
        if i < len(metadata_rows)
    )
    source_counts = Counter(
        metadata_rows[i].get("source_family", metadata_rows[i].get("source_dataset", "unknown"))
        for i in selected
        if i < len(metadata_rows)
    )
    return np.array(selected, dtype=np.int64), {
        "stratified": True,
        "bucket_count": len(buckets),
        "role_counts": dict(role_counts),
        "task_counts": dict(task_counts),
        "source_family_counts": dict(source_counts),
    }


def load_raw_data_for_group(file_paths, target_total, min_per_head, tensor_type="keys"):
    """Sample an adaptive group budget while balancing heads and preserving strata."""
    num_heads = len(file_paths)

    if num_heads == 0:
        raise ValueError("Empty group")

    if target_total < num_heads * min_per_head:
        raise ValueError(
            f"TARGET_TRAIN_SIZE={target_total} cannot provide MIN_VECTORS_PER_HEAD="
            f"{min_per_head} to {num_heads} heads."
        )

    # The total is capped exactly. Each head gets an approximately equal allocation.
    per_head_base = target_total // num_heads
    remainder = target_total % num_heads

    data = []
    vectors_per_head = {}
    shortfalls = {}
    task_counts = Counter()
    role_counts = Counter()
    source_family_counts = Counter()
    duplication_count_total = 0
    metadata_rows = load_position_index()

    for head_idx, f in enumerate(tqdm(file_paths, desc="Loading raw data for group")):
        head_data = np.load(f).astype(np.float32)

        if head_data.ndim != 2 or head_data.shape[1] != DIMS:
            raise ValueError(f"Bad shape in {f}: expected [N, {DIMS}], got {head_data.shape}")

        requested = per_head_base + (1 if head_idx < remainder else 0)
        available = len(head_data)

        if requested > available and not ALLOW_DUPLICATION:
            shortfalls[parse_head_name(f)] = int(requested - available)
            print(
                f"  NOTE: {os.path.basename(f)} supplies {available} unique vectors "
                f"for {requested} requested; no duplication"
            )

        if ALLOW_DUPLICATION and requested > available:
            indices = np.random.choice(available, requested, replace=True)
            strata = {"stratified": False, "reason": "duplication_enabled"}
            duplication_count = int(requested - available)
        else:
            indices, strata = stratified_unique_indices(
                available,
                min(requested, available),
                metadata_rows,
                seed_offset=head_idx + 1009 * num_heads,
            )
            duplication_count = 0

        data.append(head_data[indices])
        duplication_count_total += duplication_count
        head_name = parse_head_name(f)
        vectors_per_head[head_name] = int(len(indices))
        role_counts.update(strata.get("role_counts", {}))
        task_counts.update(strata.get("task_counts", {}))
        source_family_counts.update(strata.get("source_family_counts", {}))
        del head_data

    data = np.vstack(data).astype(np.float32)

    if tensor_type == "values" and CLIP_VALUE_TRAINING_DATA:
        low, high = np.percentile(data, [0.1, 99.9], axis=0)
        data = np.clip(data, low, high)

    rng = np.random.default_rng(SEED + 17 * num_heads)
    rng.shuffle(data)

    stats = {
        "head_count": int(num_heads),
        "requested_total": int(target_total),
        "realized_total": int(len(data)),
        "vectors_per_head": vectors_per_head,
        "task_counts": dict(task_counts),
        "role_counts": dict(role_counts),
        "source_family_counts": dict(source_family_counts),
        "shortfalls": shortfalls,
        "duplication_count": int(duplication_count_total),
        "allow_duplication": bool(ALLOW_DUPLICATION),
        "stratified_by": ["source_family", "position_role"],
    }

    return data.astype(np.float32), stats


# ============================================================
# Batched CUDA K-Means
# ============================================================

def init_centroids_batched(x, k, method=KMEANS_INIT):
    """Initialize [S, K, D] centroids independently for every subspace."""
    s_count, n, dim = x.shape
    device = x.device

    if method == "random":
        all_idx = [torch.randperm(n, device=device)[:k] for _ in range(s_count)]
        idx = torch.stack(all_idx, dim=0)
        s_idx = torch.arange(s_count, device=device).view(s_count, 1).expand(s_count, k)
        return x[s_idx, idx, :].contiguous()

    if method != "kmeans++":
        raise ValueError(f"Unknown KMEANS_INIT: {method}")

    centroids = torch.empty((s_count, k, dim), dtype=x.dtype, device=device)

    first_idx = torch.randint(0, n, (s_count,), device=device)
    s_idx = torch.arange(s_count, device=device)
    centroids[:, 0, :] = x[s_idx, first_idx, :]

    closest_dist_sq = torch.sum(
        (x - centroids[:, 0:1, :]) ** 2,
        dim=-1,
    )

    for c in range(1, k):
        probs = closest_dist_sq / torch.clamp(
            closest_dist_sq.sum(dim=1, keepdim=True),
            min=1e-12,
        )
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        centroids[:, c, :] = x[s_idx, next_idx, :]

        new_dist_sq = torch.sum(
            (x - centroids[:, c:c + 1, :]) ** 2,
            dim=-1,
        )
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

    return centroids.contiguous()


def assign_chunk_gemm(x_chunk, centroids):
    x_sq = torch.sum(x_chunk * x_chunk, dim=-1, keepdim=True)
    c_sq = torch.sum(centroids * centroids, dim=-1).unsqueeze(1)
    prod = torch.bmm(x_chunk, centroids.transpose(1, 2))
    dist = x_sq + c_sq - 2.0 * prod
    labels = torch.argmin(dist, dim=-1)
    return labels


def compute_inertia(x, centroids, chunk_size, reduce=True):
    """Mean within-cluster squared distance, averaged over subspaces. Cheap sanity
    number: if this barely improves over a random-init baseline, k-means stopped early."""
    s_count, n, dim = x.shape
    total = torch.zeros((s_count,), dtype=torch.float32, device=x.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[:, start:end, :].contiguous()

        x_sq = torch.sum(x_chunk * x_chunk, dim=-1, keepdim=True)
        c_sq = torch.sum(centroids * centroids, dim=-1).unsqueeze(1)
        prod = torch.bmm(x_chunk, centroids.transpose(1, 2))
        dist = torch.clamp(x_sq + c_sq - 2.0 * prod, min=0.0)
        total += dist.min(dim=-1).values.sum(dim=-1)

        del x_chunk, x_sq, c_sq, prod, dist

    per_subspace = total / max(n, 1)
    if reduce:
        return per_subspace.mean().item()
    return per_subspace.detach().cpu().numpy().astype(np.float64)


def _batched_kmeans_cuda_single(
    x,
    k,
    max_iters,
    chunk_size,
    tol,
    return_labels=False,
    desc="kmeans",
    stats=None,
):
    device = get_device()

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device=device, dtype=torch.float32)

    x = x.contiguous()

    if x.ndim != 3:
        raise ValueError(f"Expected x shape [S, N, D], got {tuple(x.shape)}")

    s_count, n, dim = x.shape

    if n < k:
        raise ValueError(f"kmeans needs N >= K, got N={n}, K={k}")

    centroids = init_centroids_batched(x, k, method=KMEANS_INIT)

    final_labels_cpu = None
    iters_used = 0
    last_rel_shift = float("nan")

    iterator = tqdm(range(max_iters), desc=desc)

    for it in iterator:
        iters_used = it + 1
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((s_count, k), dtype=torch.float32, device=device)

        all_labels = [] if return_labels else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            x_chunk = x[:, start:end, :].contiguous()
            labels = assign_chunk_gemm(x_chunk, centroids)

            if return_labels:
                all_labels.append(labels.detach().cpu())

            for s in range(s_count):
                lab_s = labels[s]
                x_s = x_chunk[s]

                new_centroids[s].scatter_add_(
                    0,
                    lab_s.view(-1, 1).expand(-1, dim),
                    x_s,
                )

                ones = torch.ones((end - start,), dtype=torch.float32, device=device)
                counts[s].scatter_add_(0, lab_s, ones)

            del x_chunk, labels

        empty = counts == 0
        empty_count = int(empty.sum().item())

        if empty.any():
            for s in range(s_count):
                empty_s = torch.where(empty[s])[0]
                if empty_s.numel() > 0:
                    replacement_idx = torch.randperm(n, device=device)[:empty_s.numel()]
                    new_centroids[s, empty_s, :] = x[s, replacement_idx, :]
                    counts[s, empty_s] = 1.0

        new_centroids = new_centroids / torch.clamp(counts.unsqueeze(-1), min=1.0)

        shift = torch.mean(torch.norm(new_centroids - centroids, dim=-1)).item()
        scale = torch.mean(torch.norm(new_centroids, dim=-1)).item()
        rel_shift = shift / (scale + 1e-12)
        last_rel_shift = rel_shift

        iterator.set_postfix({"rel_shift": f"{rel_shift:.2e}", "empty": empty_count})

        centroids = new_centroids

        # Relative convergence only. The old rule also stopped when two consecutive
        # shifts were close, which fires on transient plateaus well before convergence.
        if rel_shift < tol:
            break

        del counts, empty
        cleanup()

    if return_labels:
        final_labels = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x[:, start:end, :].contiguous()
            labels = assign_chunk_gemm(x_chunk, centroids)
            final_labels.append(labels.detach().cpu())
            del x_chunk, labels

        final_labels_cpu = torch.cat(final_labels, dim=1).numpy().astype(np.int32)

    inertia_per_subspace = compute_inertia(x, centroids, chunk_size, reduce=False)
    inertia = float(np.mean(inertia_per_subspace))

    if stats is not None:
        stats.update({
            "iters_used": int(iters_used),
            "max_iters": int(max_iters),
            "converged": bool(last_rel_shift < tol),
            "final_rel_shift": float(last_rel_shift),
            "inertia": float(inertia),
            "inertia_per_subspace": inertia_per_subspace.tolist(),
        })

    centroids_np = centroids.detach().cpu().numpy().astype(np.float32)

    del x, centroids
    cleanup()

    return centroids_np, final_labels_cpu


def batched_kmeans_cuda(
    x,
    k,
    max_iters,
    chunk_size,
    tol,
    return_labels=False,
    desc="kmeans",
    stats=None,
    restarts=1,
):
    """Run independent initializations and retain the best result per subspace.

    Fine PQ banks are independent objectives. Selecting one whole batched restart by
    aggregate inertia can discard a better solution for many individual banks.
    """
    best_centroids = None
    best_labels = None
    best_stats = None
    best_inertia = float("inf")

    for restart in range(restarts):
        run_stats = {}
        centroids, labels = _batched_kmeans_cuda_single(
            x,
            k,
            max_iters=max_iters,
            chunk_size=chunk_size,
            tol=tol,
            return_labels=return_labels,
            desc=f"{desc} restart {restart + 1}/{restarts}",
            stats=run_stats,
        )

        inertia_by_subspace = np.asarray(
            run_stats["inertia_per_subspace"], dtype=np.float64)
        inertia = float(inertia_by_subspace.mean())

        if best_centroids is None:
            best_centroids = centroids.copy()
            best_labels = labels
            best_stats = dict(run_stats)
            best_per_subspace = inertia_by_subspace.copy()
        elif return_labels:
            # The coarse stage needs a single internally consistent LUT. It currently
            # uses one restart, but retain aggregate selection if that is changed.
            if inertia < best_inertia:
                best_centroids = centroids.copy()
                best_labels = labels
                best_stats = dict(run_stats)
                best_per_subspace = inertia_by_subspace.copy()
        else:
            improved = inertia_by_subspace < best_per_subspace
            best_centroids[improved] = centroids[improved]
            best_per_subspace[improved] = inertia_by_subspace[improved]

        best_inertia = float(best_per_subspace.mean())

    if stats is not None:
        stats.update(best_stats)
        stats["restarts"] = int(restarts)
        stats["best_inertia"] = float(best_inertia)
        stats["inertia"] = float(best_inertia)
        stats["inertia_per_subspace"] = best_per_subspace.tolist()
        stats["restart_selection"] = (
            "aggregate" if return_labels else "per_subspace"
        )

    return best_centroids, best_labels


def train_all_subvector_codebooks(raw_data, stats=None):
    sub_dim = DIMS // NUM_SUBVECTORS

    if DIMS % NUM_SUBVECTORS != 0:
        raise ValueError(f"DIMS={DIMS} must be divisible by NUM_SUBVECTORS={NUM_SUBVECTORS}")

    if raw_data.ndim != 2 or raw_data.shape[1] != DIMS:
        raise ValueError(f"Expected raw_data shape [N, {DIMS}], got {raw_data.shape}")

    n = raw_data.shape[0]

    sub_data = raw_data.reshape(n, NUM_SUBVECTORS, sub_dim)
    sub_data = np.transpose(sub_data, (1, 0, 2)).copy()

    fine_stats = {}
    fine_centroids, _ = batched_kmeans_cuda(
        sub_data,
        k=CODEWORDS,
        max_iters=KMEANS_MAX_ITERS_FINE,
        chunk_size=BATCHED_KMEANS_CHUNK,
        tol=KMEANS_TOL,
        return_labels=False,
        desc="Fine k-means all subvectors",
        stats=fine_stats,
        restarts=KMEANS_RESTARTS_FINE,
    )

    num_coarse = int(math.sqrt(CODEWORDS))

    coarse_stats = {}
    coarse_centroids, lut = batched_kmeans_cuda(
        fine_centroids,
        k=num_coarse,
        max_iters=KMEANS_MAX_ITERS_COARSE,
        chunk_size=BATCHED_KMEANS_CHUNK,
        tol=KMEANS_TOL,
        return_labels=True,
        desc="Coarse k-means all subvectors",
        stats=coarse_stats,
        restarts=KMEANS_RESTARTS_COARSE,
    )

    if stats is not None:
        stats["fine_kmeans"] = fine_stats
        stats["coarse_kmeans"] = coarse_stats

    print(f"  fine k-means: {fine_stats['iters_used']}/{fine_stats['max_iters']} iters, "
          f"converged={fine_stats['converged']}, inertia={fine_stats['inertia']:.6f}")

    if not fine_stats["converged"]:
        warnings.warn(
            f"Fine k-means hit the iteration cap without converging "
            f"(rel_shift={fine_stats['final_rel_shift']:.2e}). "
            "Raise KMEANS_MAX_ITERS_FINE."
        )
    elif fine_stats["iters_used"] <= 5:
        warnings.warn(
            f"Fine k-means stopped after only {fine_stats['iters_used']} iterations. "
            "Centroids may be barely refined from their random init; check KMEANS_TOL."
        )

    return fine_centroids, coarse_centroids, lut


# ============================================================
# Reconstruction MSE check
# ============================================================

def load_codebook_stack(side_dir, group_id, num_sub, codewords, dim_per_sub, device):
    """[M, C, Dsub] float32 tensor for one group, read from the saved .txt files --
    the same files the eval harness reads, so this exercises the real artifact."""
    book = np.empty((num_sub, codewords, dim_per_sub), dtype=np.float32)
    for s in range(num_sub):
        path = os.path.join(side_dir, f"group_{group_id}_sub_{s}_fine.txt")
        if not os.path.exists(path):
            path_alt = os.path.join(side_dir, f"{group_id}_sub_{s}_fine.txt")
            path = path_alt if os.path.exists(path_alt) else path
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.size != codewords * dim_per_sub:
            raise ValueError(f"{path}: expected {codewords * dim_per_sub} values, "
                             f"got {arr.size}")
        book[s] = arr.reshape(codewords, dim_per_sub)
    return torch.tensor(book, device=device)


def pq_reconstruct_torch(x, book):
    """x: [N, DIMS] on device.  book: [M, C, Dsub].  Returns [N, DIMS]."""
    n = x.shape[0]
    m, c, dsub = book.shape
    xr = x.reshape(n, m, dsub)

    x_sq = (xr * xr).sum(dim=-1, keepdim=True)                  # [N, M, 1]
    cross = torch.einsum("nmd,mcd->nmc", xr, book)              # [N, M, C]
    book_sq = (book * book).sum(dim=-1).unsqueeze(0)            # [1, M, C]
    dist = x_sq - 2.0 * cross + book_sq
    labels = dist.argmin(dim=-1)                                # [N, M]

    sub_idx = torch.arange(m, device=x.device).unsqueeze(0).expand(n, -1)
    return book[sub_idx, labels].reshape(n, -1)


def run_mse_check(tensor_type, output_dir, head_to_group, baseline_root=None):
    """Reconstruction error on the held-out *_Test.npy vectors, which no training
    step reads. NMSE = sum((x-x_hat)^2)/sum(x^2), summed across heads before dividing
    so that heads carrying more signal count for more."""
    device = get_device()
    test_dir = os.path.join(TRAINING_DATA_ROOT, tensor_type)
    dim_per_sub = DIMS // NUM_SUBVECTORS

    test_files = sorted(glob.glob(os.path.join(test_dir, "*_Test.npy")))
    if not test_files:
        print(f"  no *_Test.npy files in {test_dir}; skipping MSE check")
        return None

    variants = {"new": (output_dir, head_to_group)}

    if baseline_root:
        base_side_dir = os.path.join(baseline_root, tensor_type)
        base_map_path = os.path.join(base_side_dir, "head_to_codebook_map.json")
        if os.path.exists(base_map_path):
            with open(base_map_path, "r") as f:
                raw = json.load(f)
            # Each codebook set must use its OWN map: clustering is rerun per training
            # run, so group_17 does not denote the same heads across directories.
            base_map = {k.replace("_Train.npy", ""): v for k, v in raw.items()}
            variants["baseline"] = (base_side_dir, base_map)
        else:
            print(f"  baseline map not found at {base_map_path}; scoring new only")

    results = {}

    for label, (book_dir, head_map) in variants.items():
        book_cache = {}
        sq_err = 0.0
        sq_err_outlier = 0.0
        signal = 0.0
        heads_scored = 0
        vectors = 0
        per_head = []

        t0 = time.time()
        for path in tqdm(test_files, desc=f"MSE [{label}/{tensor_type}]", leave=False):
            head_name = os.path.basename(path).replace("_Test.npy", "")
            group_name = head_map.get(head_name)
            if group_name is None:
                continue

            x_np = np.load(path).astype(np.float32)
            if x_np.ndim != 2 or x_np.shape[1] != DIMS or x_np.shape[0] == 0:
                continue
            if x_np.shape[0] > MSE_MAX_VECTORS_PER_HEAD:
                x_np = x_np[:MSE_MAX_VECTORS_PER_HEAD]

            group_id = group_name.replace("group_", "")
            if group_name not in book_cache:
                book_cache[group_name] = load_codebook_stack(
                    book_dir, group_id, NUM_SUBVECTORS, CODEWORDS, dim_per_sub, device)
            book = book_cache[group_name]

            x = torch.tensor(x_np, device=device)
            recon = pq_reconstruct_torch(x, book)
            err = x - recon

            head_sq = float((err * err).sum().item())
            head_sig = float((x * x).sum().item())

            # Outlier dims: top-k by mean |activation|, matching the eval's rule.
            if MSE_OUTLIER_DIMS > 0:
                rank = x.abs().mean(dim=0)
                keep = torch.topk(rank, MSE_OUTLIER_DIMS).indices
                err_ol = err.clone()
                err_ol[:, keep] = 0.0
                head_sq_ol = float((err_ol * err_ol).sum().item())
            else:
                head_sq_ol = head_sq

            sq_err += head_sq
            sq_err_outlier += head_sq_ol
            signal += head_sig
            vectors += x_np.shape[0]
            heads_scored += 1
            per_head.append({
                "head": head_name,
                "group": group_name,
                "nmse": head_sq / head_sig if head_sig > 0 else float("nan"),
            })

            del x, recon, err

        elapsed = time.time() - t0
        nmse = sq_err / signal if signal > 0 else float("nan")
        nmse_ol = sq_err_outlier / signal if signal > 0 else float("nan")

        per_head.sort(key=lambda r: -r["nmse"])
        results[label] = {
            "nmse_pq": nmse,
            "nmse_with_outliers": nmse_ol,
            "mse_pq": sq_err / max(vectors * DIMS, 1),
            "heads_scored": heads_scored,
            "vectors_per_head_cap": MSE_MAX_VECTORS_PER_HEAD,
            "vectors_total": vectors,
            "seconds": round(elapsed, 1),
            "worst_heads": per_head[:10],
        }

        print(f"  [{label}] NMSE(PQ)={nmse:.5f}  NMSE(+{MSE_OUTLIER_DIMS} outliers)="
              f"{nmse_ol:.5f}  over {heads_scored} heads, {vectors} vectors "
              f"({elapsed:.1f}s)")

        del book_cache
        cleanup()

    if "baseline" in results and "new" in results:
        ratio = results["new"]["nmse_pq"] / results["baseline"]["nmse_pq"]
        results["ratio_new_over_baseline"] = ratio
        verdict = "BETTER" if ratio < 1.0 else "WORSE"
        print(f"  [{tensor_type}] new / baseline = {ratio:.3f}x  -> new codebooks "
              f"are {verdict} on held-out data")
        if ratio >= 1.0:
            warnings.warn(
                f"{tensor_type}: new codebooks reconstruct held-out data worse than "
                f"the baseline set. Do not spend an eval run on these without "
                f"understanding why."
            )

    return results


# ============================================================
# Saving / Validation
# ============================================================

def save_group_codebooks(output_dir, group_id, fine_centroids, coarse_centroids, lut):
    ensure_dir(output_dir)

    if fine_centroids.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"fine_centroids expected first dim {NUM_SUBVECTORS}, got {fine_centroids.shape}")

    if coarse_centroids.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"coarse_centroids expected first dim {NUM_SUBVECTORS}, got {coarse_centroids.shape}")

    if lut.shape[0] != NUM_SUBVECTORS:
        raise ValueError(f"lut expected first dim {NUM_SUBVECTORS}, got {lut.shape}")

    for s in tqdm(range(NUM_SUBVECTORS), desc=f"Saving group_{group_id}", leave=False):
        prefix = f"group_{group_id}_sub_{s}"

        np.savetxt(
            os.path.join(output_dir, f"{prefix}_fine.txt"),
            fine_centroids[s],
            fmt="%.6f",
        )

        np.savetxt(
            os.path.join(output_dir, f"{prefix}_coarse.txt"),
            coarse_centroids[s],
            fmt="%.6f",
        )

        with open(os.path.join(output_dir, f"{prefix}_lut.txt"), "w") as f:
            json.dump(lut[s].astype(int).tolist(), f)


def write_head_map(output_dir, groups):
    head_to_codebook_map = {}

    for group_id, paths in sorted(groups.items()):
        for p in paths:
            head_to_codebook_map[os.path.basename(p)] = f"group_{group_id}"

    with open(os.path.join(output_dir, "head_to_codebook_map.json"), "w") as f:
        json.dump(head_to_codebook_map, f, indent=2)

    return head_to_codebook_map


def write_cluster_summary(output_dir, tensor_type, groups, head_to_codebook_map,
                          group_stats):
    summary = {
        "tensor_type": tensor_type,
        "model_name": MODEL_NAME,
        "training_data_root": TRAINING_DATA_ROOT,
        "calibration_manifest": CALIBRATION_MANIFEST,
        "expected_calibration_mode": EXPECTED_CALIBRATION_MODE,
        "expected_calibration_variant": EXPECTED_CALIBRATION_VARIANT,
        "codebook_tag": CODEBOOK_TAG,
        "cluster_metric": CLUSTER_METRIC,
        "clusters_requested": CLUSTERS,
        "groups_produced": sorted(int(g) for g in groups.keys()),
        "num_groups_produced": len(groups),
        "dims": DIMS,
        "num_subvectors": NUM_SUBVECTORS,
        "sub_dim": DIMS // NUM_SUBVECTORS,
        "codewords": CODEWORDS,
        "target_train_size": TARGET_TRAIN_SIZE,
        "adaptive_group_min_total": ADAPTIVE_GROUP_MIN_TOTAL,
        "adaptive_group_per_head": ADAPTIVE_GROUP_PER_HEAD,
        "adaptive_group_budget_rule": "max(50000, 10000 * number_of_heads_in_group)",
        "min_vectors_per_head": MIN_VECTORS_PER_HEAD,
        "allow_duplication": ALLOW_DUPLICATION,
        "kmeans_max_iters_fine": KMEANS_MAX_ITERS_FINE,
        "kmeans_max_iters_coarse": KMEANS_MAX_ITERS_COARSE,
        "kmeans_init": KMEANS_INIT,
        "kmeans_restarts_fine": KMEANS_RESTARTS_FINE,
        "kmeans_restarts_coarse": KMEANS_RESTARTS_COARSE,
        "clip_value_training_data": CLIP_VALUE_TRAINING_DATA,
        "kmeans_tol": KMEANS_TOL,
        "group_sizes": {
            f"group_{int(g)}": len(paths)
            for g, paths in sorted(groups.items())
        },
        "group_training_stats": group_stats,
        "heads": head_to_codebook_map,
    }

    with open(os.path.join(output_dir, "cluster_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def validate_generated_codebooks(output_dir, groups):
    missing = []

    for group_id in sorted(groups.keys()):
        for s in range(NUM_SUBVECTORS):
            prefix = f"group_{group_id}_sub_{s}"

            for suffix in ["fine", "coarse", "lut"]:
                path = os.path.join(output_dir, f"{prefix}_{suffix}.txt")
                if not os.path.exists(path):
                    missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Generated codebook validation failed. Missing files:\n"
            + "\n".join(missing[:100])
        )


def discover_complete_groups(base_dir):
    group_files = {}

    for path in glob.glob(os.path.join(base_dir, "group_*_sub_*_fine.txt")):
        filename = os.path.basename(path)
        group_name = filename.split("_sub_")[0]
        sub_idx = int(filename.split("_sub_")[1].split("_")[0])
        group_files.setdefault(group_name, set()).add(sub_idx)

    complete = sorted(
        group
        for group, subs in group_files.items()
        if len(subs) == NUM_SUBVECTORS and set(subs) == set(range(NUM_SUBVECTORS))
    )

    return complete


def validate_runtime_compatibility():
    print("\n============================================================")
    print("Runtime compatibility validation")
    print("============================================================")

    for tensor_type in ["keys", "values"]:
        output_dir = os.path.join(DRIVE_BASE, CODEBOOK_NAME, tensor_type)
        map_path = os.path.join(output_dir, "head_to_codebook_map.json")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Missing map: {map_path}")

        with open(map_path, "r") as f:
            head_map = json.load(f)

        requested = sorted(set(head_map.values()))
        complete = discover_complete_groups(output_dir)

        missing = sorted(set(requested) - set(complete))

        print(f"\n{tensor_type}:")
        print(f"  map groups:      {len(requested)}")
        print(f"  complete groups: {len(complete)}")

        if missing:
            raise FileNotFoundError(
                f"{tensor_type} map references missing/incomplete groups: {missing}"
            )

    print("\nRuntime compatibility validation passed.")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline_for_tensor_type(tensor_type="keys"):
    data_dir = os.path.join(TRAINING_DATA_ROOT, tensor_type)
    output_dir = os.path.join(DRIVE_BASE, CODEBOOK_NAME, tensor_type)

    ensure_dir(output_dir)

    if USE_CACHED_CLUSTERS and os.path.exists(os.path.join(output_dir, "head_to_codebook_map.json")):
        print(f"Using existing codebooks for {tensor_type.upper()} ({CODEBOOK_NAME}).")
        return None

    print("\n============================================================")
    print(f"Processing {tensor_type.upper()}")
    print(f"Output: {output_dir}")
    print("============================================================")

    remove_old_codebook_files(output_dir)

    train_files = load_train_files(data_dir)
    print(f"Found {len(train_files)} training files.")

    probe = np.load(train_files[0], mmap_mode="r")
    print(f"Supply per head: {probe.shape[0]} vectors  "
          f"(adaptive group rule=max(50K, 10K * heads), "
          f"MIN_VECTORS_PER_HEAD={MIN_VECTORS_PER_HEAD})")
    del probe

    groups = get_or_create_groups(tensor_type, train_files)

    print(f"{tensor_type.upper()} produced dense groups: {len(groups)}")

    singletons = [g for g, paths in groups.items() if len(paths) == 1]
    if singletons:
        print(f"  {len(singletons)} single-head groups (these have the least data)")

    if len(groups) != CLUSTERS:
        warnings.warn(
            f"{tensor_type.upper()}: requested {CLUSTERS} clusters, "
            f"but got {len(groups)} non-empty groups."
        )

    group_stats = {}

    for group_id, paths in sorted(groups.items()):
        print("\n------------------------------------------------------------")
        print(f"{tensor_type.upper()} group_{group_id}: {len(paths)} heads")
        print("------------------------------------------------------------")

        group_target = adaptive_group_target(len(paths))
        raw_data, stats = load_raw_data_for_group(
            paths,
            group_target,
            MIN_VECTORS_PER_HEAD,
            tensor_type,
        )

        print(f"Training data shape for group_{group_id}: {raw_data.shape}")

        stats["heads"] = len(paths)
        stats["train_vectors"] = int(raw_data.shape[0])
        stats["adaptive_group_target"] = int(group_target)
        fine_centroids, coarse_centroids, lut = train_all_subvector_codebooks(
            raw_data, stats=stats)
        group_stats[f"group_{group_id}"] = stats

        save_group_codebooks(
            output_dir,
            group_id,
            fine_centroids,
            coarse_centroids,
            lut,
        )

        del raw_data, fine_centroids, coarse_centroids, lut
        cleanup()

    head_to_codebook_map = write_head_map(output_dir, groups)

    write_cluster_summary(
        output_dir,
        tensor_type,
        groups,
        head_to_codebook_map,
        group_stats,
    )

    validate_generated_codebooks(output_dir, groups)

    print(f"\nFinished {tensor_type.upper()}. Groups: {len(groups)}")

    # ---- built-in MSE check ----------------------------------------------------------
    mse_results = None
    if RUN_MSE_CHECK:
        print(f"\nReconstruction MSE check ({tensor_type}, held-out *_Test.npy)")
        head_map_stripped = {
            k.replace("_Train.npy", ""): v for k, v in head_to_codebook_map.items()
        }
        mse_results = run_mse_check(
            tensor_type,
            output_dir,
            head_map_stripped,
            baseline_root=MSE_BASELINE_CODEBOOK_ROOT,
        )
        if mse_results is not None:
            with open(os.path.join(output_dir, MSE_REPORT_NAME), "w") as f:
                json.dump(mse_results, f, indent=2)

    return mse_results


def build_calibration_static_masks():
    """Learn static masks from the selected calibration training vectors."""
    payload = {}
    for tensor_type in ["keys", "values"]:
        data_dir = os.path.join(TRAINING_DATA_ROOT, tensor_type)
        for path in tqdm(
            sorted(glob.glob(os.path.join(data_dir, "*_Train.npy"))),
            desc=f"Static masks [{tensor_type}]",
        ):
            head = os.path.basename(path).replace("_Train.npy", "")
            data = np.load(path, mmap_mode="r")
            mean_abs = np.mean(np.abs(data.astype(np.float32)), axis=0)
            dims = np.argsort(mean_abs)[-STATIC_OUTLIER_DIMS:][::-1]
            payload[f"{tensor_type}/{head}"] = [int(x) for x in dims]

    report_path = os.path.join(DRIVE_BASE, CODEBOOK_NAME, STATIC_MASK_REPORT_NAME)
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Calibration-derived static masks: {report_path} ({len(payload)} heads)")
    return report_path


if __name__ == "__main__":
    set_seed(SEED)
    setup_torch()
    validate_adaptive_group_budget_rule()
    calibration_manifest = load_and_validate_calibration_manifest()

    if DIMS % NUM_SUBVECTORS != 0:
        raise ValueError(
            f"DIMS={DIMS} must be divisible by NUM_SUBVECTORS={NUM_SUBVECTORS}"
        )

    print(f"Using device: {get_device()}")

    mse_all = {}
    mse_all["keys"] = run_pipeline_for_tensor_type("keys")
    mse_all["values"] = run_pipeline_for_tensor_type("values")

    validate_runtime_compatibility()
    build_calibration_static_masks()

    # ---- summary ---------------------------------------------------------------------
    print("\n============================================================")
    print("Reconstruction MSE summary (document-disjoint test vectors)")
    print("============================================================")
    for side, res in mse_all.items():
        if not res:
            print(f"  {side}: not run")
            continue
        new = res.get("new", {})
        base = res.get("baseline")
        line = (f"  {side:<7} new NMSE={new.get('nmse_pq', float('nan')):.5f}  "
                f"(+outliers {new.get('nmse_with_outliers', float('nan')):.5f})")
        if base:
            line += (f"   baseline NMSE={base['nmse_pq']:.5f}   "
                     f"ratio={res.get('ratio_new_over_baseline', float('nan')):.3f}x")
        print(line)
    print("\n  Ratio < 1.0 means the new codebooks reconstruct held-out data better.")
    print(f"  Full per-side reports: {CODEBOOK_NAME}/<side>/{MSE_REPORT_NAME}")

    local_codebooks_versioned = os.path.join(DRIVE_BASE, CODEBOOK_NAME)
    drive_codebooks_versioned = os.path.join(DRIVE_OUTPUT_BASE, CODEBOOK_NAME)

    if SAVE_TO_DRIVE and os.path.exists(local_codebooks_versioned):
        print("\nSaving results to Google Drive:")
        print(f"  from: {local_codebooks_versioned}")
        print(f"  to:   {drive_codebooks_versioned}")

        ensure_dir(os.path.dirname(drive_codebooks_versioned))

        shutil.copytree(
            local_codebooks_versioned,
            drive_codebooks_versioned,
            dirs_exist_ok=True,
        )

    print("\nDone.")

# %% id="LVza8-hA7MQM" colab={"base_uri": "https://localhost:8080/", "height": 423} outputId="2311fded-621a-4095-e4f7-136766bf24ef"
import os
import glob
import json
import math
import struct
import mmap
import gc
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Global Model / Eval Config
# ============================================================

MODEL_NAME = "qwen3_8B"
MODEL_DIR = f"/content/{MODEL_NAME}"

DIMS = 128
WINDOW_SIZE = 512
TEST_MODE = False
CALIBRATION_FILE = "calibration.txt"

MAX_CHUNKS = 10 if TEST_MODE else 50
PQ_CHUNK_SIZE = 64

AUTO_REMAP_MISSING_GROUPS = False
ALLOW_SINGLE_GROUP_FALLBACK = False

RESULTS_CSV = "pq_dynamic_static_result.csv"
STATIC_MASK_JSON = "outlier_static_top3_masks.json"
STATIC_MASK_CSV = "outlier_static_top3_masks.csv"
DYNAMIC_OUTLIER_OVERALL_CSV = "outlier_dynamic_reuse_overall.csv"
DYNAMIC_OUTLIER_BY_HEAD_CSV = "outlier_dynamic_reuse_by_head.csv"
STATIC_OUTLIER_OVERALL_CSV = "outlier_static_reuse_overall.csv"
STATIC_OUTLIER_BY_HEAD_CSV = "outlier_static_reuse_by_head.csv"


# ============================================================
# Priority 1 Mixed Config
# Keys:   32 banks x 2048 words, C4,  out2
# Values: 32 banks x 1024 words, C16, out1
# Average bits/scalar = 2.8125
# ============================================================

KEY_CONFIG = {
    "num_sub_vectors": 64,
    "num_codewords": 128,
    "clusters": 64,
    "outlier_dims": 3,
}

VALUE_CONFIG = {
    "num_sub_vectors": 64,
    "num_codewords": 32,
    "clusters": 128,
    "outlier_dims": 3,
}

EXPERIMENT_NAME = "K64x128_C64_out2__V64x128_C64_out1"


# ============================================================
# Utility
# ============================================================


# ============================================================
# Pretty Console Output
# ============================================================

PRINT_WIDTH = 96
MAX_LIST_PREVIEW = 12
MAX_OUTLIER_TABLE_ROWS = 30
MAX_TOKEN_MISMATCH_EXAMPLES = 12


def hr(char="=", width=PRINT_WIDTH):
    print(char * width)


def banner(title):
    print("\n" + "=" * PRINT_WIDTH)
    print(title.center(PRINT_WIDTH))
    print("=" * PRINT_WIDTH)


def section(title):
    print("\n" + title)
    print("-" * min(len(title), PRINT_WIDTH))


def fmt_float(x, digits=4):
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return str(x)
        return f"{x:.{digits}f}"
    return str(x)


def print_kv_table(title, rows):
    section(title)
    if not rows:
        print("  <empty>")
        return

    key_width = max(len(str(k)) for k, _ in rows)
    for key, value in rows:
        if isinstance(value, float):
            value = fmt_float(value)
        print(f"  {str(key):<{key_width}} : {value}")


def preview_list(items, max_items=MAX_LIST_PREVIEW):
    items = list(items)
    if len(items) <= max_items:
        return str(items)
    shown = ", ".join(repr(x) for x in items[:max_items])
    return f"[{shown}, ...]  ({len(items)} total)"


def print_df(title, df, index=False, max_rows=None, sort_by=None, ascending=True):
    section(title)
    if df is None or len(df) == 0:
        print("  <empty>")
        return

    shown = df.copy()
    total_rows = len(shown)

    if sort_by is not None and sort_by in shown.columns:
        shown = shown.sort_values(sort_by, ascending=ascending)

    if max_rows is not None and total_rows > max_rows:
        shown = shown.head(max_rows)
        print(f"  showing {len(shown)} of {total_rows} rows; full table is saved to CSV")

    with pd.option_context(
        "display.max_rows", max_rows if max_rows is not None else 50,
        "display.max_columns", 50,
        "display.width", PRINT_WIDTH,
        "display.max_colwidth", 60,
    ):
        print(shown.to_string(index=index))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def codebook_dir_from_cfg(cfg):
    if "codebook_dir" in cfg and cfg["codebook_dir"] is not None:
        return cfg["codebook_dir"]

    return os.path.join(
        MODEL_DIR,
        f"codebooks_{cfg['num_sub_vectors']}_{cfg['num_codewords']}_{cfg['clusters']}",
    )


def side_dir_from_cfg(cfg, side):
    return os.path.join(codebook_dir_from_cfg(cfg), side)


def bits_per_vector(cfg):
    index_bits = cfg["num_sub_vectors"] * math.log2(cfg["num_codewords"])
    outlier_bits = cfg.get("outlier_dims", 0) * 16
    return index_bits + outlier_bits


def overall_compression_ratio(key_cfg, value_cfg):
    key_bits = bits_per_vector(key_cfg)
    value_bits = bits_per_vector(value_cfg)

    avg_bits_per_scalar = (key_bits + value_bits) / (2 * DIMS)
    compression_ratio = 16.0 / avg_bits_per_scalar

    return key_bits, value_bits, avg_bits_per_scalar, compression_ratio


def list_available_codebook_groups(base_dir, num_sub_vectors):
    groups = {}

    if not os.path.isdir(base_dir):
        return groups

    for path in glob.glob(os.path.join(base_dir, "*.txt")):
        filename = os.path.basename(path)

        if "_sub_" not in filename:
            continue

        group_prefix, rest = filename.split("_sub_", 1)
        sub_str = rest.split("_", 1)[0]

        try:
            sub_idx = int(sub_str)
        except ValueError:
            continue

        if filename.endswith("_fine.txt"):
            kind = "fine"
        elif filename.endswith("_lut.txt"):
            kind = "lut"
        elif filename.endswith("_coarse.txt"):
            kind = "coarse"
        else:
            continue

        groups.setdefault(group_prefix, {"fine": {}, "lut": {}, "coarse": {}})
        groups[group_prefix][kind][sub_idx] = path

    return {
        g: data
        for g, data in groups.items()
        if len(data["fine"]) > 0
    }


def validate_group_has_all_fine_files(base_dir, group_id, num_sub_vectors):
    missing = []

    for s in range(num_sub_vectors):
        fine_path = os.path.join(base_dir, f"{group_id}_sub_{s}_fine.txt")
        if not os.path.exists(fine_path):
            missing.append(fine_path)

    return missing

def load_head_to_codebook_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing head_to_codebook_map.json: {path}")

    with open(path, "r") as f:
        raw_map = json.load(f)

    return {
        k.replace("_Train.npy", "") : v
        for k, v in raw_map.items()
    }


def resolve_head_map_to_available_groups(
    base_dir,
    head_map,
    num_sub_vectors,
    auto_remap=False,
    allow_single_group_fallback=False,
    map_name="keys",
):
    available = list_available_codebook_groups(base_dir, num_sub_vectors)

    complete_groups = []
    incomplete_groups = {}

    for group_id in sorted(available.keys()):
        missing = validate_group_has_all_fine_files(
            base_dir,
            group_id,
            num_sub_vectors,
        )

        if len(missing) == 0:
            complete_groups.append(group_id)
        else:
            incomplete_groups[group_id] = missing

    requested_groups = sorted(set(head_map.values()))
    missing_requested = []

    for group_id in requested_groups:
        if group_id not in complete_groups:
            missing_requested.append(group_id)

    print_kv_table(
        f"{map_name.upper()} codebook group validation",
        [
            ("directory", base_dir),
            ("requested groups", preview_list(requested_groups)),
            ("complete groups", preview_list(complete_groups)),
            ("incomplete groups", len(incomplete_groups)),
        ],
    )

    if incomplete_groups:
        rows = [
            {"group": g, "missing_fine_files": len(missing)}
            for g, missing in sorted(incomplete_groups.items())
        ]
        print_df(f"{map_name.upper()} incomplete group details", pd.DataFrame(rows), index=False)

    if not missing_requested:
        return head_map, {}

    print_kv_table(
        f"{map_name.upper()} missing requested groups",
        [("missing requested groups", preview_list(missing_requested))],
    )

    if not auto_remap:
        raise FileNotFoundError(
            f"{map_name}: map references groups that are not complete on disk: "
            f"{missing_requested}. Complete groups: {complete_groups}"
        )

    if len(complete_groups) == 0:
        raise FileNotFoundError(
            f"{map_name}: no complete codebook groups found in {base_dir}."
        )

    if len(complete_groups) == 1 and allow_single_group_fallback:
        fallback_group = complete_groups[0]
    else:
        fallback_group = complete_groups[0]

    remap = {
        missing_group: fallback_group
        for missing_group in missing_requested
    }

    warnings.warn(
        f"{map_name}: remapping missing groups {remap}. "
        f"PQ accuracy may be invalid."
    )

    fixed_map = {
        head: remap.get(group, group)
        for head, group in head_map.items()
    }

    return fixed_map, remap


# ============================================================
# PQ Classes
# ============================================================

class TorchGroupProcessor:
    def __init__(
        self,
        base_dir,
        group_id,
        num_sub_vectors,
        num_codewords,
        dim_per_sub,
        device=None
    ):
        self.device = device if device is not None else get_device()
        self.group_id = group_id
        self.num_sub_vectors = num_sub_vectors
        self.num_codewords = num_codewords
        self.dim_per_sub = dim_per_sub

        self._load_codebooks(base_dir, group_id)

    def _load_codebooks(self, base_dir, group_id):
        fines = []

        for s in range(self.num_sub_vectors):
            prefix = f"{group_id}_sub_{s}"
            fine_path = os.path.join(base_dir, f"{prefix}_fine.txt")

            if not os.path.exists(fine_path):
                raise FileNotFoundError(f"Missing fine codebook file: {fine_path}")

            fine_data = np.loadtxt(fine_path, dtype=np.float32)

            expected_elems = self.num_codewords * self.dim_per_sub
            actual_elems = fine_data.size

            if actual_elems != expected_elems:
                raise ValueError(
                    f"Bad shape in {fine_path}. "
                    f"Expected {expected_elems}, got {actual_elems}."
                )

            fine_data = fine_data.reshape(self.num_codewords, self.dim_per_sub)

            fines.append(
                torch.tensor(
                    fine_data,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            )

        self.fine = torch.stack(fines, dim=0)

    def quantize(self, x, chunk_size=64):
        x_dtype = x.dtype
        x_f32 = x.float()
        fine_f32 = self.fine.float()

        N, M, D = x_f32.shape

        if M != self.num_sub_vectors or D != self.dim_per_sub:
            raise ValueError(
                f"Expected x shape [N, {self.num_sub_vectors}, {self.dim_per_sub}], "
                f"got {tuple(x.shape)}"
            )

        out_chunks = []
        fine_sq = torch.sum(fine_f32 ** 2, dim=-1).unsqueeze(0)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_f32[start:end]

            x_sq = torch.sum(x_chunk ** 2, dim=-1, keepdim=True)

            interaction = torch.bmm(
                x_chunk.transpose(0, 1),
                fine_f32.transpose(1, 2),
            ).transpose(0, 1)

            dists = x_sq + fine_sq - 2.0 * interaction
            labels = torch.argmin(dists, dim=-1)

            chunk_N = end - start
            M_idx = torch.arange(M, device=self.device).unsqueeze(0).expand(chunk_N, -1)

            quantized = self.fine[M_idx, labels]
            out_chunks.append(quantized.to(x_dtype))

            del x_chunk, x_sq, interaction, dists, labels, quantized

        out = torch.cat(out_chunks, dim=0)

        del x_f32, fine_f32, fine_sq, out_chunks

        return out


class OutlierReuseTracker:
    """Tracks whether outlier dimensions are reused for each side/layer/head.

    The quantizer currently *applies* one outlier set per layer/head per forward chunk,
    computed from mean absolute activation across batch and sequence. This tracker also
    checks token-level top-k outliers to answer whether individual tokens would have
    chosen different dimensions from the applied head-level set.
    """

    def __init__(self, max_examples=20):
        self.max_examples = max_examples
        self.head_stats = {}
        self.examples = []

    @staticmethod
    def _canon(indices):
        return tuple(sorted(int(x) for x in indices))

    def update(self, side, layer_idx, head_idx, applied_indices, token_indices=None):
        key = (side, int(layer_idx), int(head_idx))
        applied = self._canon(applied_indices)

        if key not in self.head_stats:
            self.head_stats[key] = {
                "side": side,
                "layer": int(layer_idx),
                "head": int(head_idx),
                "calls": 0,
                "first": applied,
                "previous": None,
                "same_as_first_calls": 0,
                "same_as_previous_transitions": 0,
                "changed_calls": 0,
                "unique_sets": set(),
                "token_total": 0,
                "token_matches_applied": 0,
                "token_matches_first": 0,
                "token_differs_from_applied": 0,
                "dimension_counts": {},
            }

        st = self.head_stats[key]
        st["calls"] += 1
        st["unique_sets"].add(applied)
        for dim in applied:
            st["dimension_counts"][dim] = st["dimension_counts"].get(dim, 0) + 1

        if applied == st["first"]:
            st["same_as_first_calls"] += 1
        else:
            st["changed_calls"] += 1

        if st["previous"] is not None and applied == st["previous"]:
            st["same_as_previous_transitions"] += 1
        st["previous"] = applied

        if token_indices is None:
            return

        # token_indices shape: [batch, seq, k] for one layer/head.
        token_cpu = token_indices.detach().cpu().reshape(-1, token_indices.shape[-1]).tolist()
        for token_pos, inds in enumerate(token_cpu):
            token_set = self._canon(inds)
            st["token_total"] += 1

            if token_set == applied:
                st["token_matches_applied"] += 1
            else:
                st["token_differs_from_applied"] += 1
                if len(self.examples) < self.max_examples:
                    self.examples.append({
                        "side": side,
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "call": st["calls"],
                        "flat_token_index_in_chunk": int(token_pos),
                        "applied_head_outliers": applied,
                        "token_outliers": token_set,
                    })

            if token_set == st["first"]:
                st["token_matches_first"] += 1

    def overall_rows(self):
        rows = []
        for side in ["keys", "values"]:
            stats = [st for st in self.head_stats.values() if st["side"] == side]
            if not stats:
                continue

            calls = sum(st["calls"] for st in stats)
            repeat_obs = sum(max(st["calls"] - 1, 0) for st in stats)
            same_as_first_repeat = sum(
                max(st["same_as_first_calls"] - 1, 0)
                for st in stats
            )
            same_prev = sum(st["same_as_previous_transitions"] for st in stats)
            token_total = sum(st["token_total"] for st in stats)
            token_match_applied = sum(st["token_matches_applied"] for st in stats)
            token_match_first = sum(st["token_matches_first"] for st in stats)
            unstable_heads = sum(1 for st in stats if len(st["unique_sets"]) > 1)

            rows.append({
                "side": side,
                "heads_seen": len(stats),
                "head_chunk_calls": calls,
                "unstable_heads": unstable_heads,
                "same_as_first_%": 100.0 * same_as_first_repeat / repeat_obs if repeat_obs else 100.0,
                "same_as_previous_%": 100.0 * same_prev / repeat_obs if repeat_obs else 100.0,
                "token_same_as_applied_%": 100.0 * token_match_applied / token_total if token_total else 100.0,
                "token_same_as_first_%": 100.0 * token_match_first / token_total if token_total else 100.0,
                "token_instances": token_total,
            })
        return rows

    def head_rows(self, only_unstable_or_token_mismatch=True):
        rows = []
        for key in sorted(self.head_stats.keys()):
            st = self.head_stats[key]
            calls = st["calls"]
            repeat_obs = max(calls - 1, 0)
            token_total = st["token_total"]
            token_same_applied_pct = (
                100.0 * st["token_matches_applied"] / token_total
                if token_total else 100.0
            )
            same_first_pct = (
                100.0 * max(st["same_as_first_calls"] - 1, 0) / repeat_obs
                if repeat_obs else 100.0
            )

            include = True
            if only_unstable_or_token_mismatch:
                include = len(st["unique_sets"]) > 1 or st["token_differs_from_applied"] > 0
            if not include:
                continue

            rows.append({
                "side": st["side"],
                "layer": st["layer"],
                "head": st["head"],
                "calls": calls,
                "unique_head_sets": len(st["unique_sets"]),
                "same_as_first_%": same_first_pct,
                "token_same_as_applied_%": token_same_applied_pct,
                "token_mismatch_count": st["token_differs_from_applied"],
                "first_outliers": st["first"],
            })
        return rows

    def static_mask_rows(self):
        """Return one row per side/layer/head with the most frequently selected dims.

        Popularity is counted over the head-level outlier set that was actually
        applied during the dynamic pass. For top-k=3, every dynamic call casts
        one vote for each of its three selected dimensions.
        """
        rows = []
        for key in sorted(self.head_stats.keys()):
            st = self.head_stats[key]
            counts = st.get("dimension_counts", {})
            if not counts:
                continue

            k = len(st["first"])
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_dims = tuple(int(dim) for dim, _ in ranked[:k])
            top_counts = tuple(int(cnt) for _, cnt in ranked[:k])

            rows.append({
                "side": st["side"],
                "layer": st["layer"],
                "head": st["head"],
                "calls": st["calls"],
                "static_outliers": top_dims,
                "static_outlier_counts": top_counts,
                "unique_head_sets": len(st["unique_sets"]),
                "first_outliers": st["first"],
            })
        return rows

    def build_static_masks(self):
        """Build masks keyed by (side, layer, head) from dynamic popularity."""
        masks = {}
        for row in self.static_mask_rows():
            masks[(row["side"], int(row["layer"]), int(row["head"]))] = tuple(row["static_outliers"])
        return masks

    def print_summary(self):
        banner("Outlier Reuse Summary")
        overall_raw = pd.DataFrame(self.overall_rows())
        if len(overall_raw) == 0:
            print("No outlier tracking data was collected. Check that outlier_dims > 0 and PQ was enabled.")
            return

        overall = overall_raw.copy()
        for col in [
            "same_as_first_%",
            "same_as_previous_%",
            "token_same_as_applied_%",
            "token_same_as_first_%",
        ]:
            overall[col] = overall[col].map(lambda x: f"{x:.2f}")
        print_df("Overall reuse", overall, index=False)

        head_rows_raw = pd.DataFrame(self.head_rows(only_unstable_or_token_mismatch=True))
        if len(head_rows_raw) == 0:
            print("\nEvery tracked layer/head reused the same head-level outlier set, and token-level top-k matched the applied set.")
            return

        unstable = head_rows_raw[head_rows_raw["unique_head_sets"] > 1].copy()
        if len(unstable) > 0:
            unstable_display = unstable.copy()
            for col in ["same_as_first_%", "token_same_as_applied_%"]:
                unstable_display[col] = unstable_display[col].map(lambda x: f"{x:.2f}")
            print_df(
                f"Most unstable head-level outlier sets, top {MAX_OUTLIER_TABLE_ROWS}",
                unstable_display,
                index=False,
                max_rows=MAX_OUTLIER_TABLE_ROWS,
                sort_by="same_as_first_%",
                ascending=True,
            )
        else:
            section("Head-level stability")
            print("  Every tracked layer/head reused the same head-level outlier set.")

        token_worst = head_rows_raw[head_rows_raw["token_mismatch_count"] > 0].copy()
        if len(token_worst) > 0:
            token_worst_display = token_worst.copy()
            for col in ["same_as_first_%", "token_same_as_applied_%"]:
                token_worst_display[col] = token_worst_display[col].map(lambda x: f"{x:.2f}")
            print_df(
                f"Worst token-level mismatch cases, top {MAX_OUTLIER_TABLE_ROWS}",
                token_worst_display,
                index=False,
                max_rows=MAX_OUTLIER_TABLE_ROWS,
                sort_by="token_same_as_applied_%",
                ascending=True,
            )

        if self.examples:
            examples = pd.DataFrame(self.examples[:MAX_TOKEN_MISMATCH_EXAMPLES])
            print_df("Example token-level mismatches", examples, index=False)

        section("Outlier table note")
        print("  The notebook printout is intentionally capped. Full per-layer/head data is saved to outlier_reuse_by_head.csv.")


class DualPQManager:
    def __init__(
        self,
        key_cfg,
        value_cfg,
        device=None,
        auto_remap_missing_groups=False,
        track_outliers=True,
        outlier_mode="dynamic",
        static_outlier_masks=None,
    ):
        self.device = device if device is not None else get_device()
        self.key_cfg = dict(key_cfg)
        self.value_cfg = dict(value_cfg)
        self.enabled = False
        self.track_outliers = track_outliers
        self.outlier_tracker = OutlierReuseTracker() if track_outliers else None
        self.outlier_mode = outlier_mode
        self.static_outlier_masks = static_outlier_masks or {}

        self.key_cfg["dim_per_sub"] = DIMS // self.key_cfg["num_sub_vectors"]
        self.value_cfg["dim_per_sub"] = DIMS // self.value_cfg["num_sub_vectors"]

        if self.key_cfg["num_sub_vectors"] != self.value_cfg["num_sub_vectors"]:
            raise ValueError(
                "Key/value configs must use the same number of banks/subvectors "
                f"for shared hardware. Got key={self.key_cfg['num_sub_vectors']}, "
                f"value={self.value_cfg['num_sub_vectors']}."
            )

        if DIMS % self.key_cfg["num_sub_vectors"] != 0:
            raise ValueError(
                f"DIMS={DIMS} not divisible by key num_sub_vectors={self.key_cfg['num_sub_vectors']}"
            )

        if DIMS % self.value_cfg["num_sub_vectors"] != 0:
            raise ValueError(
                f"DIMS={DIMS} not divisible by value num_sub_vectors={self.value_cfg['num_sub_vectors']}"
            )

        self.key_dir = side_dir_from_cfg(self.key_cfg, "keys")
        self.value_dir = side_dir_from_cfg(self.value_cfg, "values")

        self.key_map, self.key_group_remap, self.key_processors = self._init_side(
            cfg=self.key_cfg,
            base_dir=self.key_dir,
            side_name="keys",
            auto_remap_missing_groups=auto_remap_missing_groups,
        )

        self.val_map, self.val_group_remap, self.val_processors = self._init_side(
            cfg=self.value_cfg,
            base_dir=self.value_dir,
            side_name="values",
            auto_remap_missing_groups=auto_remap_missing_groups,
        )

    def reset_outlier_tracker(self):
        if self.track_outliers:
            self.outlier_tracker = OutlierReuseTracker()

    def set_outlier_mode(self, mode, static_outlier_masks=None):
        if mode not in {"dynamic", "static"}:
            raise ValueError(f"Unknown outlier mode: {mode}. Expected 'dynamic' or 'static'.")
        self.outlier_mode = mode
        if static_outlier_masks is not None:
            self.static_outlier_masks = static_outlier_masks

    def _init_side(self, cfg, base_dir, side_name, auto_remap_missing_groups):
        map_path = os.path.join(base_dir, "head_to_codebook_map.json")
        raw_map = load_head_to_codebook_map(map_path)

        fixed_map, remap = resolve_head_map_to_available_groups(
            base_dir=base_dir,
            head_map=raw_map,
            num_sub_vectors=cfg["num_sub_vectors"],
            auto_remap=auto_remap_missing_groups,
            allow_single_group_fallback=ALLOW_SINGLE_GROUP_FALLBACK,
            map_name=side_name,
        )

        processors = {}
        unique_groups = sorted(set(fixed_map.values()))

        print_kv_table(
            f"Initializing {side_name} PQ processors",
            [("unique groups", preview_list(unique_groups))],
        )
        for g in tqdm(unique_groups, desc=f"Loading {side_name} processors", leave=False):
            processors[g] = TorchGroupProcessor(
                base_dir=base_dir,
                group_id=g,
                num_sub_vectors=cfg["num_sub_vectors"],
                num_codewords=cfg["num_codewords"],
                dim_per_sub=cfg["dim_per_sub"],
                device=self.device,
            )

        return fixed_map, remap, processors

    def quantize_tensor(self, tensor, layer_idx, is_key=True):
        if not self.enabled:
            return tensor

        bsz, num_heads, seq_len, head_dim = tensor.shape

        if head_dim != DIMS:
            raise ValueError(f"Expected head_dim={DIMS}, got {head_dim}")

        if is_key:
            cfg, head_map, processors = self.key_cfg, self.key_map, self.key_processors
        else:
            cfg, head_map, processors = self.value_cfg, self.val_map, self.val_processors

        out = torch.empty_like(tensor)

        groups_to_heads = {}

        for h in range(num_heads):
            head_key = f"L{layer_idx}_H{h}"

            if head_key not in head_map:
                out[:, h, :, :] = tensor[:, h, :, :]
            else:
                group_id = head_map[head_key]
                groups_to_heads.setdefault(group_id, []).append(h)

        for group_id, heads in groups_to_heads.items():
            processor = processors[group_id]

            group_tensor = tensor[:, heads, :, :].contiguous()
            orig_shape = group_tensor.shape

            outlier_dims = int(cfg.get("outlier_dims", 0))
            outlier_indices = None
            token_outlier_indices = None

            if outlier_dims > 0:
                side_name = "keys" if is_key else "values"

                if self.outlier_mode == "static":
                    # Static pass: use the top-k most popular dims learned from the prior
                    # dynamic pass for each specific side/layer/head.
                    static_rows = []
                    missing_static = []
                    for h in heads:
                        mask_key = (side_name, int(layer_idx), int(h))
                        if mask_key in self.static_outlier_masks:
                            static_rows.append(list(self.static_outlier_masks[mask_key]))
                        else:
                            missing_static.append(h)
                            static_rows.append(None)

                    if missing_static:
                        # Fallback should almost never happen if dynamic and static use the
                        # same eval path. It keeps the run from crashing on unmapped heads.
                        mean_abs = group_tensor.abs().mean(dim=(0, 2))
                        _, dynamic_fallback = torch.topk(mean_abs, outlier_dims, dim=-1)
                        for local_i, row in enumerate(static_rows):
                            if row is None:
                                static_rows[local_i] = [int(x) for x in dynamic_fallback[local_i].detach().cpu().tolist()]

                    outlier_indices = torch.tensor(
                        static_rows,
                        dtype=torch.long,
                        device=group_tensor.device,
                    )
                else:
                    # Dynamic pass: one top-k set per selected layer/head for this forward chunk.
                    # Shape: [selected_heads, outlier_dims].
                    mean_abs = group_tensor.abs().mean(dim=(0, 2))
                    _, outlier_indices = torch.topk(mean_abs, outlier_dims, dim=-1)

                # Diagnostic only: token-level top-k outliers. This lets us detect whether
                # individual tokens would choose different outlier dimensions than the
                # head-level set that the quantizer actually applies.
                if self.track_outliers and self.outlier_tracker is not None:
                    _, token_outlier_indices = torch.topk(
                        group_tensor.abs(),
                        outlier_dims,
                        dim=-1,
                    )

                    for local_i, h in enumerate(heads):
                        self.outlier_tracker.update(
                            side=side_name,
                            layer_idx=layer_idx,
                            head_idx=h,
                            applied_indices=outlier_indices[local_i],
                            token_indices=token_outlier_indices[:, local_i, :, :],
                        )

            group_flat = group_tensor.reshape(
                -1,
                cfg["num_sub_vectors"],
                cfg["dim_per_sub"],
            )

            quant_flat = processor.quantize(group_flat, chunk_size=PQ_CHUNK_SIZE)
            quantized_group = quant_flat.reshape(orig_shape)

            if outlier_indices is not None:
                for i in range(len(heads)):
                    head_outliers = outlier_indices[i]
                    quantized_group[:, i, :, head_outliers] = group_tensor[:, i, :, head_outliers]

            out[:, heads, :, :] = quantized_group

            del group_tensor, group_flat, quant_flat, quantized_group

        return out


# ============================================================
# Safetensors Loader
# ============================================================

def load_safetensors_pure(filepath):
    with open(filepath, "rb") as f:
        header_size_bytes = f.read(8)

        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {filepath}")

        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        offset = 8 + header_size
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        tensors = {}

        for k, v in header.items():
            if k == "__metadata__":
                continue

            dtype = v["dtype"]
            shape = v["shape"]
            data_offsets = v["data_offsets"]

            start = offset + data_offsets[0]
            end = offset + data_offsets[1]
            raw_data = mm[start:end]

            if dtype == "BF16":
                t = torch.frombuffer(
                    bytearray(raw_data),
                    dtype=torch.int16,
                ).view(torch.bfloat16).clone()
                tensors[k] = t.reshape(shape)

            elif dtype == "F32":
                t = torch.from_numpy(
                    np.frombuffer(raw_data, dtype=np.float32).copy()
                ).reshape(shape)
                tensors[k] = t

            elif dtype == "F16":
                t = torch.from_numpy(
                    np.frombuffer(raw_data, dtype=np.float16).copy()
                ).reshape(shape)
                tensors[k] = t

            else:
                raise NotImplementedError(
                    f"Unsupported dtype {dtype} in {filepath} for tensor {k}"
                )

        mm.close()

    return tensors


# ============================================================
# Qwen Model
# ============================================================

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=4096,
        base=1000000.0,
        device=None
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float().to(device) / dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            emb.cos()[None, None, :, :],
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            emb.sin()[None, None, :, :],
            persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            raise ValueError(
                f"seq_len={seq_len} exceeds rotary cache size {self.max_seq_len_cached}"
            )

        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim
    )

    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim
    )


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx, pq_manager=None):
        super().__init__()

        self.layer_idx = layer_idx
        self.pq_manager = pq_manager

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.get("attention_bias", False)
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.get("attention_bias", False)
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.get("attention_bias", False)
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.get("attention_bias", False)
        )

        self.q_norm = QwenRMSNorm(
            self.head_dim,
            eps=config.get("rms_norm_eps", 1e-6)
        )

        self.k_norm = QwenRMSNorm(
            self.head_dim,
            eps=config.get("rms_norm_eps", 1e-6)
        )

        max_pos = config.get(
            "max_position_embeddings",
            max(4096, WINDOW_SIZE + 16)
        )

        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_pos,
            base=config.get("rope_theta", 1000000.0)
        )

    def forward(self, hidden_states, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.pq_manager is not None and self.pq_manager.enabled:
            key_states = self.pq_manager.quantize_tensor(key_states, self.layer_idx, is_key=True)
            value_states = self.pq_manager.quantize_tensor(value_states, self.layer_idx, is_key=False)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # ROBUST ADDITIVE CAUSAL MASK
        mask = torch.full((q_len, kv_seq_len), -10000.0, device=query_states.device)
        mask = torch.triu(mask, diagonal=1)
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(1)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, pq_manager=None):
        super().__init__()
        self.self_attn = QwenAttention(config, layer_idx, pq_manager)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, hidden_states, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, None


class QwenModel(nn.Module):
    def __init__(self, config, pq_manager=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([QwenDecoderLayer(config, i, pq_manager) for i in range(config["num_hidden_layers"])])
        self.norm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return hidden_states, None


class QwenForCausalLM(nn.Module):
    def __init__(self, config, pq_manager=None):
        super().__init__()
        self.model = QwenModel(config, pq_manager)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        hidden_states, _ = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits, None


def set_model_pq_manager(model, pq_manager):
    for layer in model.model.layers:
        layer.self_attn.pq_manager = pq_manager


# ============================================================
# Eval / Model Loading
# ============================================================

def calculate_perplexity(eval_model, tokenizer, test_file, chunk_size=WINDOW_SIZE, max_chunks=None, desc="Evaluating PPL"):
    if not os.path.exists(test_file):
        print(f"Missing test file: {test_file}")
        return float("inf")
    with open(test_file, "r", encoding="utf-8") as f: text = f.read()
    tokens = tokenizer.encode(text)
    if len(tokens) < chunk_size + 1:
        return float("inf")
    eval_model.eval()
    device = next(eval_model.parameters()).device
    total_loss, num_batches = 0.0, 0
    chunks = (len(tokens) - 1) // chunk_size if max_chunks is None else min((len(tokens) - 1) // chunk_size, max_chunks)
    with torch.no_grad():
        for i in tqdm(range(int(chunks)), desc=desc, leave=True):
            chunk = tokens[i * chunk_size : (i + 1) * chunk_size + 1]
            input_ids = torch.tensor([chunk[:-1]], device=device, dtype=torch.long)
            target_ids = torch.tensor([chunk[1:]], device=device, dtype=torch.long)
            logits, _ = eval_model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), target_ids.view(-1))
            total_loss += loss.item()
            num_batches += 1
            if i % 5 == 0: cleanup_cuda()
    return math.exp(total_loss / num_batches) if num_batches > 0 else float("inf")


def load_model_weights(model, model_dir):
    state_dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for path in safetensor_files:
        tensors = load_safetensors_pure(path)
        state_dict.update(tensors)
        cleanup_cuda()
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    model.load_state_dict(state_dict, strict=False)
    cleanup_cuda()


def validate_single_config():
    for side_name, cfg, folder_side in [("key", KEY_CONFIG, "keys"), ("value", VALUE_CONFIG, "values")]:
        side_folder = side_dir_from_cfg(cfg, folder_side)
        map_path = os.path.join(side_folder, "head_to_codebook_map.json")
        if not os.path.exists(map_path): raise FileNotFoundError(f"Missing map: {map_path}")




def save_static_masks(static_masks, json_path=STATIC_MASK_JSON, csv_path=STATIC_MASK_CSV):
    """Save static masks in both machine-readable JSON and spreadsheet-friendly CSV."""
    json_obj = {}
    rows = []
    for (side, layer, head), dims in sorted(static_masks.items()):
        key = f"{side}/L{int(layer)}_H{int(head)}"
        dims = tuple(int(x) for x in dims)
        json_obj[key] = list(dims)
        rows.append({
            "side": side,
            "layer": int(layer),
            "head": int(head),
            "static_outliers": dims,
        })

    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_tracker_csvs(tracker, overall_csv, by_head_csv):
    pd.DataFrame(tracker.overall_rows()).to_csv(overall_csv, index=False)
    pd.DataFrame(tracker.head_rows(only_unstable_or_token_mismatch=False)).to_csv(by_head_csv, index=False)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    banner("PQ Dynamic vs Static Outlier Evaluation")
    device = get_device()
    cleanup_cuda()
    torch.set_default_dtype(torch.bfloat16)
    validate_single_config()

    key_bits, value_bits, avg_bps, comp_ratio = overall_compression_ratio(KEY_CONFIG, VALUE_CONFIG)
    print_kv_table(
        "Run configuration",
        [
            ("experiment", EXPERIMENT_NAME),
            ("model_dir", MODEL_DIR),
            ("device", device),
            ("window_size", WINDOW_SIZE),
            ("max_chunks", MAX_CHUNKS),
            ("pq_chunk_size", PQ_CHUNK_SIZE),
            ("key config", KEY_CONFIG),
            ("value config", VALUE_CONFIG),
            ("key bits/vector", key_bits),
            ("value bits/vector", value_bits),
            ("avg bits/scalar", avg_bps),
            ("compression ratio", comp_ratio),
        ],
    )

    section("Loading model configuration")
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)
    print_kv_table(
        "Model summary",
        [
            ("hidden_size", config.get("hidden_size")),
            ("layers", config.get("num_hidden_layers")),
            ("attention heads", config.get("num_attention_heads")),
            ("kv heads", config.get("num_key_value_heads", config.get("num_attention_heads"))),
            ("vocab_size", config.get("vocab_size")),
        ],
    )

    pq_manager = DualPQManager(
        KEY_CONFIG,
        VALUE_CONFIG,
        device,
        AUTO_REMAP_MISSING_GROUPS,
        track_outliers=True,
    )

    section("Loading model weights")
    model = QwenForCausalLM(config, pq_manager=None)
    load_model_weights(model, MODEL_DIR)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("  Model and tokenizer loaded.")

    section("Baseline perplexity")
    set_model_pq_manager(model, None)
    baseline_ppl = calculate_perplexity(
        model,
        tokenizer,
        CALIBRATION_FILE,
        max_chunks=MAX_CHUNKS,
        desc="Baseline PPL",
    )
    print(f"  Baseline PPL: {baseline_ppl:.6f}")

    section("Dynamic PQ perplexity with outlier tracking")
    pq_manager.enabled = True
    pq_manager.set_outlier_mode("dynamic")
    pq_manager.reset_outlier_tracker()
    set_model_pq_manager(model, pq_manager)
    dynamic_ppl = calculate_perplexity(
        model,
        tokenizer,
        CALIBRATION_FILE,
        max_chunks=MAX_CHUNKS,
        desc="Dynamic PQ PPL",
    )
    print(f"  Dynamic PQ PPL: {dynamic_ppl:.6f}")

    dynamic_tracker = pq_manager.outlier_tracker
    dynamic_tracker.print_summary()

    section("Building static top-3 outlier masks from dynamic run")
    static_masks = dynamic_tracker.build_static_masks()
    static_mask_rows = pd.DataFrame(dynamic_tracker.static_mask_rows())
    save_static_masks(static_masks, json_path=STATIC_MASK_JSON, csv_path=STATIC_MASK_CSV)
    print_kv_table(
        "Static mask summary",
        [
            ("masks learned", len(static_masks)),
            ("json", STATIC_MASK_JSON),
            ("csv", STATIC_MASK_CSV),
        ],
    )
    if len(static_mask_rows) > 0:
        display_rows = static_mask_rows.copy()
        print_df(
            "Example learned static masks",
            display_rows[["side", "layer", "head", "calls", "static_outliers", "static_outlier_counts", "unique_head_sets"]],
            index=False,
            max_rows=20,
        )

    section("Static-mask PQ perplexity")
    pq_manager.set_outlier_mode("static", static_outlier_masks=static_masks)
    pq_manager.reset_outlier_tracker()
    static_ppl = calculate_perplexity(
        model,
        tokenizer,
        CALIBRATION_FILE,
        max_chunks=MAX_CHUNKS,
        desc="Static-mask PQ PPL",
    )
    print(f"  Static-mask PQ PPL: {static_ppl:.6f}")

    static_tracker = pq_manager.outlier_tracker
    static_tracker.print_summary()

    result = {
        "experiment": EXPERIMENT_NAME,
        "key_bits_per_vector": key_bits,
        "value_bits_per_vector": value_bits,
        "avg_bits_per_scalar": avg_bps,
        "compression_ratio": comp_ratio,
        "baseline_ppl": baseline_ppl,
        "dynamic_pq_ppl": dynamic_ppl,
        "dynamic_ppl_error_%": 100.0 * (dynamic_ppl - baseline_ppl) / baseline_ppl if baseline_ppl else float("nan"),
        "static_top3_pq_ppl": static_ppl,
        "static_top3_ppl_error_%": 100.0 * (static_ppl - baseline_ppl) / baseline_ppl if baseline_ppl else float("nan"),
        "static_minus_dynamic_ppl": static_ppl - dynamic_ppl,
    }

    result_df = pd.DataFrame([result])
    print_df("Final dynamic vs static result", result_df, index=False)

    # Save outputs so Colab keeps a clean record.
    result_df.to_csv(RESULTS_CSV, index=False)
    save_tracker_csvs(dynamic_tracker, DYNAMIC_OUTLIER_OVERALL_CSV, DYNAMIC_OUTLIER_BY_HEAD_CSV)
    save_tracker_csvs(static_tracker, STATIC_OUTLIER_OVERALL_CSV, STATIC_OUTLIER_BY_HEAD_CSV)

    print_kv_table(
        "Saved files",
        [
            ("result csv", RESULTS_CSV),
            ("static mask json", STATIC_MASK_JSON),
            ("static mask csv", STATIC_MASK_CSV),
            ("dynamic outlier overall csv", DYNAMIC_OUTLIER_OVERALL_CSV),
            ("dynamic outlier by-head csv", DYNAMIC_OUTLIER_BY_HEAD_CSV),
            ("static outlier overall csv", STATIC_OUTLIER_OVERALL_CSV),
            ("static outlier by-head csv", STATIC_OUTLIER_BY_HEAD_CSV),
        ],
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="Ql7Z9OaFee2q" outputId="9a5080b2-33bd-46c0-bfd6-fff9a8efbe8d"
# ============================================================
# Standalone Google Colab cell: Qwen3 KV-cache PQ on LongBench
# Requires only the existing model/codebook tree at /content/qwen3_8B
# ============================================================
import sys
import subprocess
import importlib.util

_REQUIRED_PACKAGES = {
    "datasets": "datasets>=4.0.0",
    "huggingface_hub": "huggingface_hub>=0.34.0",
    "rouge": "rouge>=1.0.1",
    "jieba": "jieba>=0.42.1",
    "fuzzywuzzy": "fuzzywuzzy>=0.18.0",
    "Levenshtein": "python-Levenshtein>=0.27.0",
}
_missing_specs = [
    pip_spec
    for import_name, pip_spec in _REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]
if _missing_specs:
    print("Installing missing dependencies:", _missing_specs)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", *_missing_specs
    ])

import os
import glob
import json
import math
import struct
import mmap
import gc
import warnings
import time
import re
import string
import random
import hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Global Model / Eval Config
# ============================================================

MODEL_NAME = "qwen3_8B"
MODEL_DIR = f"/content/{MODEL_NAME}"

DIMS = 128
PQ_CHUNK_SIZE = 64

# LongBench v1 / LongBench-E evaluation.
# PQ_BRIDGE_CELL_MARKER: longbench_eval_dynamic_static_variant
# Keep full evaluation as the default, while allowing a bridge command to request
# the small end-to-end smoke test without editing and republishing the notebook.
_test_mode_env = os.environ.get("PQ_LONGBENCH_TEST_MODE", "0").strip().lower()
if _test_mode_env not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
    raise ValueError(
        "PQ_LONGBENCH_TEST_MODE must be one of "
        "0/1, false/true, no/yes, or off/on"
    )
TEST_MODE = _test_mode_env in {"1", "true", "yes", "on"}
USE_LONG_BENCH_E = True
_test_samples_env = os.environ.get(
    "PQ_LONGBENCH_TEST_SAMPLES_PER_DATASET", "2"
).strip()
try:
    _test_samples_per_dataset = int(_test_samples_env)
except ValueError as exc:
    raise ValueError(
        "PQ_LONGBENCH_TEST_SAMPLES_PER_DATASET must be a positive integer"
    ) from exc
if _test_samples_per_dataset < 1:
    raise ValueError(
        "PQ_LONGBENCH_TEST_SAMPLES_PER_DATASET must be a positive integer"
    )
_max_samples_env = os.environ.get(
    "PQ_LONGBENCH_MAX_SAMPLES_PER_DATASET", ""
).strip()
if _max_samples_env:
    try:
        MAX_SAMPLES_PER_DATASET = int(_max_samples_env)
    except ValueError as exc:
        raise ValueError(
            "PQ_LONGBENCH_MAX_SAMPLES_PER_DATASET must be a positive integer"
        ) from exc
    if MAX_SAMPLES_PER_DATASET < 1:
        raise ValueError(
            "PQ_LONGBENCH_MAX_SAMPLES_PER_DATASET must be a positive integer"
        )
else:
    MAX_SAMPLES_PER_DATASET = _test_samples_per_dataset if TEST_MODE else None

_sample_selection_default = (
    "first" if TEST_MODE and not _max_samples_env else
    "random" if MAX_SAMPLES_PER_DATASET is not None else
    "all"
)
SAMPLE_SELECTION = os.environ.get(
    "PQ_LONGBENCH_SAMPLE_SELECTION", _sample_selection_default
).strip().lower()
if SAMPLE_SELECTION not in {"all", "first", "random"}:
    raise ValueError(
        "PQ_LONGBENCH_SAMPLE_SELECTION must be all, first, or random"
    )
if SAMPLE_SELECTION == "all" and MAX_SAMPLES_PER_DATASET is not None:
    raise ValueError(
        "PQ_LONGBENCH_SAMPLE_SELECTION=all cannot be combined with a sample cap"
    )
if SAMPLE_SELECTION in {"first", "random"} and MAX_SAMPLES_PER_DATASET is None:
    raise ValueError(
        f"PQ_LONGBENCH_SAMPLE_SELECTION={SAMPLE_SELECTION} requires "
        "PQ_LONGBENCH_MAX_SAMPLES_PER_DATASET or test mode"
    )
_sample_seed_env = os.environ.get("PQ_LONGBENCH_SAMPLE_SEED", "0").strip()
try:
    SAMPLE_SELECTION_SEED = int(_sample_seed_env)
except ValueError as exc:
    raise ValueError("PQ_LONGBENCH_SAMPLE_SEED must be an integer") from exc

MAX_INPUT_TOKENS = 4096
_max_new_tokens_env = os.environ.get("PQ_LONGBENCH_MAX_NEW_TOKENS", "").strip()
if _max_new_tokens_env:
    try:
        MAX_NEW_TOKENS_CAP = int(_max_new_tokens_env)
    except ValueError as exc:
        raise ValueError(
            "PQ_LONGBENCH_MAX_NEW_TOKENS must be a positive integer"
        ) from exc
    if MAX_NEW_TOKENS_CAP < 1:
        raise ValueError("PQ_LONGBENCH_MAX_NEW_TOKENS must be a positive integer")
else:
    MAX_NEW_TOKENS_CAP = 64 if TEST_MODE else None
USE_CHAT_TEMPLATE = True
DISABLE_QWEN_THINKING = True
TRACK_TOKEN_LEVEL_OUTLIERS = False
RUN_KEY_VALUE_SIDE_DIAGNOSTICS = os.environ.get(
    "PQ_LONGBENCH_SIDE_DIAGNOSTICS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_eval_modes_env = os.environ.get(
    "PQ_LONGBENCH_EVAL_MODES", "baseline,dynamic,static"
).strip()
EVAL_MODES = tuple(
    mode.strip().lower()
    for mode in _eval_modes_env.split(",")
    if mode.strip()
)
_allowed_eval_modes = {"baseline", "dynamic", "static"}
if not EVAL_MODES or any(mode not in _allowed_eval_modes for mode in EVAL_MODES):
    raise ValueError(
        "PQ_LONGBENCH_EVAL_MODES must be a comma-separated subset of "
        "baseline,dynamic,static"
    )
if len(set(EVAL_MODES)) != len(EVAL_MODES):
    raise ValueError("PQ_LONGBENCH_EVAL_MODES contains duplicate modes")
if "dynamic" in EVAL_MODES and "baseline" not in EVAL_MODES:
    raise ValueError("Dynamic LongBench-E evaluation requires baseline mode")
if "static" in EVAL_MODES and "dynamic" not in EVAL_MODES:
    raise ValueError("Static LongBench-E evaluation requires dynamic mode first")
EVAL_MODES = tuple(mode for mode in ("baseline", "dynamic", "static") if mode in EVAL_MODES)
REQUIRED_GPU_NAME = os.environ.get("PQ_REQUIRED_GPU_NAME", "").strip()

LONG_BENCH_REPOS = ["zai-org/LongBench", "THUDM/LongBench"]
# Pin the converted Parquet tree. The repository's current main branch can point
# back to the legacy data.zip layout, which makes a main-branch file listing
# nondeterministic for direct-Parquet loading.
LONG_BENCH_REVISION = "36914d6211386125c6fc4ce7db4a6a777fadd34c"
LONG_BENCH_LOADER_VERSION = "parquet-local-v4-pinned-36914d6"
LONG_BENCH_E_DATASETS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
LONG_BENCH_FULL_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "lcc", "repobench-p",
]

_dataset_override_env = os.environ.get("PQ_LONGBENCH_DATASETS", "").strip()
if _dataset_override_env:
    LONG_BENCH_DATASETS = [
        dataset.strip() for dataset in _dataset_override_env.split(",")
        if dataset.strip()
    ]
    unknown_datasets = sorted(set(LONG_BENCH_DATASETS) - set(LONG_BENCH_E_DATASETS))
    if unknown_datasets:
        raise ValueError(
            "PQ_LONGBENCH_DATASETS contains datasets outside LongBench-E: "
            f"{unknown_datasets}"
        )
    if len(set(LONG_BENCH_DATASETS)) != len(LONG_BENCH_DATASETS):
        raise ValueError("PQ_LONGBENCH_DATASETS contains duplicates")
elif TEST_MODE:
    LONG_BENCH_DATASETS = ["qasper", "hotpotqa", "passage_retrieval_en"]
else:
    LONG_BENCH_DATASETS = LONG_BENCH_E_DATASETS if USE_LONG_BENCH_E else LONG_BENCH_FULL_DATASETS

AUTO_REMAP_MISSING_GROUPS = False
ALLOW_SINGLE_GROUP_FALLBACK = False

_eval_calibration_mode_env = os.environ.get(
    "PQ_LONGBENCH_CALIBRATION_MODE", "held_out"
).strip().lower()
if _eval_calibration_mode_env not in {"held_out", "matched", "contaminated"}:
    raise ValueError(
        "PQ_LONGBENCH_CALIBRATION_MODE must be held_out, matched, or contaminated"
    )
EVAL_CALIBRATION_MODE = _eval_calibration_mode_env
_eval_calibration_variant_env = os.environ.get(
    "PQ_LONGBENCH_CALIBRATION_VARIANT", "prior_held_out"
).strip().lower()
EVAL_VARIANT_ALIASES = {
    "": "prior_held_out",
    "none": "prior_held_out",
    "held_out": "prior_held_out",
    "prior": "prior_held_out",
    "prior_held_out": "prior_held_out",
    "clean_hotpot_a": "clean_hotpot_a",
    "hotpot_a": "clean_hotpot_a",
    "variant_a": "clean_hotpot_a",
    "clean_suite_b": "clean_suite_b",
    "suite_b": "clean_suite_b",
    "variant_b": "clean_suite_b",
    "clean_qa_count_c": "clean_qa_count_c",
    "qa_count_c": "clean_qa_count_c",
    "variant_c": "clean_qa_count_c",
}
if _eval_calibration_variant_env not in EVAL_VARIANT_ALIASES:
    raise ValueError(
        "PQ_LONGBENCH_CALIBRATION_VARIANT must be prior_held_out, "
        "clean_hotpot_a, clean_suite_b, or clean_qa_count_c"
    )
EVAL_CALIBRATION_VARIANT = EVAL_VARIANT_ALIASES[_eval_calibration_variant_env]
if EVAL_CALIBRATION_VARIANT != "prior_held_out" and EVAL_CALIBRATION_MODE != "held_out":
    raise ValueError("Clean LongBench eval variants require held_out calibration mode")
EVAL_OUTPUT_TAG = (
    EVAL_CALIBRATION_MODE
    if EVAL_CALIBRATION_VARIANT == "prior_held_out"
    else EVAL_CALIBRATION_VARIANT
)
_result_suffix = (
    "" if EVAL_OUTPUT_TAG == "held_out" else f"_{EVAL_OUTPUT_TAG}"
)
_run_tag = os.environ.get("PQ_LONGBENCH_RUN_TAG", "").strip()
if _run_tag and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", _run_tag) is None:
    raise ValueError(
        "PQ_LONGBENCH_RUN_TAG must be 1-64 alphanumeric, underscore, or hyphen "
        "characters and start with an alphanumeric character"
    )
if _run_tag:
    _result_suffix += f"_{_run_tag}"

OUTPUT_DIR = f"longbench_pq_outputs{_result_suffix}"
RESULTS_CSV = f"longbench_pq_dynamic_static_result{_result_suffix}.csv"
SUMMARY_CSV = f"longbench_pq_summary_by_dataset{_result_suffix}.csv"
STATIC_MASK_JSON = f"outlier_static_top3_masks_longbench{_result_suffix}.json"
STATIC_MASK_CSV = f"outlier_static_top3_masks_longbench{_result_suffix}.csv"
DYNAMIC_OUTLIER_OVERALL_CSV = f"outlier_dynamic_reuse_overall_longbench{_result_suffix}.csv"
DYNAMIC_OUTLIER_BY_HEAD_CSV = f"outlier_dynamic_reuse_by_head_longbench{_result_suffix}.csv"
STATIC_OUTLIER_OVERALL_CSV = f"outlier_static_reuse_overall_longbench{_result_suffix}.csv"
STATIC_OUTLIER_BY_HEAD_CSV = f"outlier_static_reuse_by_head_longbench{_result_suffix}.csv"

# ============================================================
# Reportable LongBench-E configuration. Keys and values use the same 64-bank,
# 128-codeword capacity and the exact versioned output of the trainer above.
# ============================================================

LONGBENCH_E_CODEBOOK_DIR = os.path.join(
    MODEL_DIR,
    (
        f"codebooks_64_128_64_longbench_e_{EVAL_OUTPUT_TAG}_4096_adaptive10k"
        if EVAL_CALIBRATION_VARIANT != "prior_held_out"
        else f"codebooks_64_128_64_longbench_e_{EVAL_CALIBRATION_MODE}_4096_balanced_kpp_noclip"
    ),
)

KEY_CONFIG = {
    "num_sub_vectors": 64,
    "num_codewords": 128,
    "clusters": 64,
    "outlier_dims": 3,
    "codebook_dir": LONGBENCH_E_CODEBOOK_DIR,
}

VALUE_CONFIG = {
    "num_sub_vectors": 64,
    "num_codewords": 128,
    "clusters": 64,
    "outlier_dims": 3,
    "codebook_dir": LONGBENCH_E_CODEBOOK_DIR,
}

EXPERIMENT_NAME = "LongBenchE_K64x128_C64_out3__V64x128_C64_out3"
if EVAL_CALIBRATION_MODE != "held_out":
    EXPERIMENT_NAME += f"__{EVAL_CALIBRATION_MODE}_calibration"
if EVAL_CALIBRATION_VARIANT != "prior_held_out":
    EXPERIMENT_NAME += f"__{EVAL_CALIBRATION_VARIANT}"
if _run_tag:
    EXPERIMENT_NAME += f"__{_run_tag}"


def validate_required_gpu():
    if not REQUIRED_GPU_NAME:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Required GPU {REQUIRED_GPU_NAME!r}, but CUDA is not available"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if REQUIRED_GPU_NAME.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"Required GPU containing {REQUIRED_GPU_NAME!r}, but runtime has "
            f"{gpu_name!r}. Stop before LongBench-E evaluation."
        )
    print(f"Required GPU confirmed: {gpu_name}")
    return gpu_name


# ============================================================
# Utility
# ============================================================


# ============================================================
# Pretty Console Output
# ============================================================

PRINT_WIDTH = 96
MAX_LIST_PREVIEW = 12
MAX_OUTLIER_TABLE_ROWS = 30
MAX_TOKEN_MISMATCH_EXAMPLES = 12


def hr(char="=", width=PRINT_WIDTH):
    print(char * width)


def banner(title):
    print("\n" + "=" * PRINT_WIDTH)
    print(title.center(PRINT_WIDTH))
    print("=" * PRINT_WIDTH)


def section(title):
    print("\n" + title)
    print("-" * min(len(title), PRINT_WIDTH))


def fmt_float(x, digits=4):
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return str(x)
        return f"{x:.{digits}f}"
    return str(x)


def print_kv_table(title, rows):
    section(title)
    if not rows:
        print("  <empty>")
        return

    key_width = max(len(str(k)) for k, _ in rows)
    for key, value in rows:
        if isinstance(value, float):
            value = fmt_float(value)
        print(f"  {str(key):<{key_width}} : {value}")


def preview_list(items, max_items=MAX_LIST_PREVIEW):
    items = list(items)
    if len(items) <= max_items:
        return str(items)
    shown = ", ".join(repr(x) for x in items[:max_items])
    return f"[{shown}, ...]  ({len(items)} total)"


def print_df(title, df, index=False, max_rows=None, sort_by=None, ascending=True):
    section(title)
    if df is None or len(df) == 0:
        print("  <empty>")
        return

    shown = df.copy()
    total_rows = len(shown)

    if sort_by is not None and sort_by in shown.columns:
        shown = shown.sort_values(sort_by, ascending=ascending)

    if max_rows is not None and total_rows > max_rows:
        shown = shown.head(max_rows)
        print(f"  showing {len(shown)} of {total_rows} rows; full table is saved to CSV")

    with pd.option_context(
        "display.max_rows", max_rows if max_rows is not None else 50,
        "display.max_columns", 50,
        "display.width", PRINT_WIDTH,
        "display.max_colwidth", 60,
    ):
        print(shown.to_string(index=index))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def codebook_dir_from_cfg(cfg):
    if "codebook_dir" in cfg and cfg["codebook_dir"] is not None:
        return cfg["codebook_dir"]

    return os.path.join(
        MODEL_DIR,
        f"codebooks_{cfg['num_sub_vectors']}_{cfg['num_codewords']}_{cfg['clusters']}",
    )


def side_dir_from_cfg(cfg, side):
    return os.path.join(codebook_dir_from_cfg(cfg), side)


def bits_per_vector(cfg):
    index_bits = cfg["num_sub_vectors"] * math.log2(cfg["num_codewords"])
    outlier_bits = cfg.get("outlier_dims", 0) * 16
    return index_bits + outlier_bits


def overall_compression_ratio(key_cfg, value_cfg):
    key_bits = bits_per_vector(key_cfg)
    value_bits = bits_per_vector(value_cfg)

    avg_bits_per_scalar = (key_bits + value_bits) / (2 * DIMS)
    compression_ratio = 16.0 / avg_bits_per_scalar

    return key_bits, value_bits, avg_bits_per_scalar, compression_ratio


def list_available_codebook_groups(base_dir, num_sub_vectors):
    groups = {}

    if not os.path.isdir(base_dir):
        return groups

    for path in glob.glob(os.path.join(base_dir, "*.txt")):
        filename = os.path.basename(path)

        if "_sub_" not in filename:
            continue

        group_prefix, rest = filename.split("_sub_", 1)
        sub_str = rest.split("_", 1)[0]

        try:
            sub_idx = int(sub_str)
        except ValueError:
            continue

        if filename.endswith("_fine.txt"):
            kind = "fine"
        elif filename.endswith("_lut.txt"):
            kind = "lut"
        elif filename.endswith("_coarse.txt"):
            kind = "coarse"
        else:
            continue

        groups.setdefault(group_prefix, {"fine": {}, "lut": {}, "coarse": {}})
        groups[group_prefix][kind][sub_idx] = path

    return {
        g: data
        for g, data in groups.items()
        if len(data["fine"]) > 0
    }


def validate_group_has_all_fine_files(base_dir, group_id, num_sub_vectors):
    missing = []

    for s in range(num_sub_vectors):
        fine_path = os.path.join(base_dir, f"{group_id}_sub_{s}_fine.txt")
        if not os.path.exists(fine_path):
            missing.append(fine_path)

    return missing

def load_head_to_codebook_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing head_to_codebook_map.json: {path}")

    with open(path, "r") as f:
        raw_map = json.load(f)

    return {
        k.replace("_Train.npy", "") : v
        for k, v in raw_map.items()
    }


def resolve_head_map_to_available_groups(
    base_dir,
    head_map,
    num_sub_vectors,
    auto_remap=False,
    allow_single_group_fallback=False,
    map_name="keys",
):
    available = list_available_codebook_groups(base_dir, num_sub_vectors)

    complete_groups = []
    incomplete_groups = {}

    for group_id in sorted(available.keys()):
        missing = validate_group_has_all_fine_files(
            base_dir,
            group_id,
            num_sub_vectors,
        )

        if len(missing) == 0:
            complete_groups.append(group_id)
        else:
            incomplete_groups[group_id] = missing

    requested_groups = sorted(set(head_map.values()))
    missing_requested = []

    for group_id in requested_groups:
        if group_id not in complete_groups:
            missing_requested.append(group_id)

    print_kv_table(
        f"{map_name.upper()} codebook group validation",
        [
            ("directory", base_dir),
            ("requested groups", preview_list(requested_groups)),
            ("complete groups", preview_list(complete_groups)),
            ("incomplete groups", len(incomplete_groups)),
        ],
    )

    if incomplete_groups:
        rows = [
            {"group": g, "missing_fine_files": len(missing)}
            for g, missing in sorted(incomplete_groups.items())
        ]
        print_df(f"{map_name.upper()} incomplete group details", pd.DataFrame(rows), index=False)

    if not missing_requested:
        return head_map, {}

    print_kv_table(
        f"{map_name.upper()} missing requested groups",
        [("missing requested groups", preview_list(missing_requested))],
    )

    if not auto_remap:
        raise FileNotFoundError(
            f"{map_name}: map references groups that are not complete on disk: "
            f"{missing_requested}. Complete groups: {complete_groups}"
        )

    if len(complete_groups) == 0:
        raise FileNotFoundError(
            f"{map_name}: no complete codebook groups found in {base_dir}."
        )

    if len(complete_groups) == 1 and allow_single_group_fallback:
        fallback_group = complete_groups[0]
    else:
        fallback_group = complete_groups[0]

    remap = {
        missing_group: fallback_group
        for missing_group in missing_requested
    }

    warnings.warn(
        f"{map_name}: remapping missing groups {remap}. "
        f"PQ accuracy may be invalid."
    )

    fixed_map = {
        head: remap.get(group, group)
        for head, group in head_map.items()
    }

    return fixed_map, remap


# ============================================================
# PQ Classes
# ============================================================

class TorchGroupProcessor:
    def __init__(
        self,
        base_dir,
        group_id,
        num_sub_vectors,
        num_codewords,
        dim_per_sub,
        device=None
    ):
        self.device = device if device is not None else get_device()
        self.group_id = group_id
        self.num_sub_vectors = num_sub_vectors
        self.num_codewords = num_codewords
        self.dim_per_sub = dim_per_sub

        self._load_codebooks(base_dir, group_id)

    def _load_codebooks(self, base_dir, group_id):
        fines = []

        for s in range(self.num_sub_vectors):
            prefix = f"{group_id}_sub_{s}"
            fine_path = os.path.join(base_dir, f"{prefix}_fine.txt")

            if not os.path.exists(fine_path):
                raise FileNotFoundError(f"Missing fine codebook file: {fine_path}")

            fine_data = np.loadtxt(fine_path, dtype=np.float32)

            expected_elems = self.num_codewords * self.dim_per_sub
            actual_elems = fine_data.size

            if actual_elems != expected_elems:
                raise ValueError(
                    f"Bad shape in {fine_path}. "
                    f"Expected {expected_elems}, got {actual_elems}."
                )

            fine_data = fine_data.reshape(self.num_codewords, self.dim_per_sub)

            fines.append(
                torch.tensor(
                    fine_data,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            )

        self.fine = torch.stack(fines, dim=0)

    def quantize(self, x, chunk_size=64):
        x_dtype = x.dtype
        x_f32 = x.float()
        fine_f32 = self.fine.float()

        N, M, D = x_f32.shape

        if M != self.num_sub_vectors or D != self.dim_per_sub:
            raise ValueError(
                f"Expected x shape [N, {self.num_sub_vectors}, {self.dim_per_sub}], "
                f"got {tuple(x.shape)}"
            )

        out_chunks = []
        fine_sq = torch.sum(fine_f32 ** 2, dim=-1).unsqueeze(0)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_f32[start:end]

            x_sq = torch.sum(x_chunk ** 2, dim=-1, keepdim=True)

            interaction = torch.bmm(
                x_chunk.transpose(0, 1),
                fine_f32.transpose(1, 2),
            ).transpose(0, 1)

            dists = x_sq + fine_sq - 2.0 * interaction
            labels = torch.argmin(dists, dim=-1)

            chunk_N = end - start
            M_idx = torch.arange(M, device=self.device).unsqueeze(0).expand(chunk_N, -1)

            quantized = self.fine[M_idx, labels]
            out_chunks.append(quantized.to(x_dtype))

            del x_chunk, x_sq, interaction, dists, labels, quantized

        out = torch.cat(out_chunks, dim=0)

        del x_f32, fine_f32, fine_sq, out_chunks

        return out


class OutlierReuseTracker:
    """Tracks whether outlier dimensions are reused for each side/layer/head.

    The quantizer currently *applies* one outlier set per layer/head per forward chunk,
    computed from mean absolute activation across batch and sequence. This tracker also
    checks token-level top-k outliers to answer whether individual tokens would have
    chosen different dimensions from the applied head-level set.
    """

    def __init__(self, max_examples=20):
        self.max_examples = max_examples
        self.head_stats = {}
        self.examples = []

    @staticmethod
    def _canon(indices):
        return tuple(sorted(int(x) for x in indices))

    def update(self, side, layer_idx, head_idx, applied_indices, token_indices=None):
        key = (side, int(layer_idx), int(head_idx))
        applied = self._canon(applied_indices)

        if key not in self.head_stats:
            self.head_stats[key] = {
                "side": side,
                "layer": int(layer_idx),
                "head": int(head_idx),
                "calls": 0,
                "first": applied,
                "previous": None,
                "same_as_first_calls": 0,
                "same_as_previous_transitions": 0,
                "changed_calls": 0,
                "unique_sets": set(),
                "token_total": 0,
                "token_matches_applied": 0,
                "token_matches_first": 0,
                "token_differs_from_applied": 0,
                "dimension_counts": {},
            }

        st = self.head_stats[key]
        st["calls"] += 1
        st["unique_sets"].add(applied)
        for dim in applied:
            st["dimension_counts"][dim] = st["dimension_counts"].get(dim, 0) + 1

        if applied == st["first"]:
            st["same_as_first_calls"] += 1
        else:
            st["changed_calls"] += 1

        if st["previous"] is not None and applied == st["previous"]:
            st["same_as_previous_transitions"] += 1
        st["previous"] = applied

        if token_indices is None:
            return

        # token_indices shape: [batch, seq, k] for one layer/head.
        token_cpu = token_indices.detach().cpu().reshape(-1, token_indices.shape[-1]).tolist()
        for token_pos, inds in enumerate(token_cpu):
            token_set = self._canon(inds)
            st["token_total"] += 1

            if token_set == applied:
                st["token_matches_applied"] += 1
            else:
                st["token_differs_from_applied"] += 1
                if len(self.examples) < self.max_examples:
                    self.examples.append({
                        "side": side,
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "call": st["calls"],
                        "flat_token_index_in_chunk": int(token_pos),
                        "applied_head_outliers": applied,
                        "token_outliers": token_set,
                    })

            if token_set == st["first"]:
                st["token_matches_first"] += 1

    def overall_rows(self):
        rows = []
        for side in ["keys", "values"]:
            stats = [st for st in self.head_stats.values() if st["side"] == side]
            if not stats:
                continue

            calls = sum(st["calls"] for st in stats)
            repeat_obs = sum(max(st["calls"] - 1, 0) for st in stats)
            same_as_first_repeat = sum(
                max(st["same_as_first_calls"] - 1, 0)
                for st in stats
            )
            same_prev = sum(st["same_as_previous_transitions"] for st in stats)
            token_total = sum(st["token_total"] for st in stats)
            token_match_applied = sum(st["token_matches_applied"] for st in stats)
            token_match_first = sum(st["token_matches_first"] for st in stats)
            unstable_heads = sum(1 for st in stats if len(st["unique_sets"]) > 1)

            rows.append({
                "side": side,
                "heads_seen": len(stats),
                "head_chunk_calls": calls,
                "unstable_heads": unstable_heads,
                "same_as_first_%": 100.0 * same_as_first_repeat / repeat_obs if repeat_obs else 100.0,
                "same_as_previous_%": 100.0 * same_prev / repeat_obs if repeat_obs else 100.0,
                "token_same_as_applied_%": 100.0 * token_match_applied / token_total if token_total else 100.0,
                "token_same_as_first_%": 100.0 * token_match_first / token_total if token_total else 100.0,
                "token_instances": token_total,
            })
        return rows

    def head_rows(self, only_unstable_or_token_mismatch=True):
        rows = []
        for key in sorted(self.head_stats.keys()):
            st = self.head_stats[key]
            calls = st["calls"]
            repeat_obs = max(calls - 1, 0)
            token_total = st["token_total"]
            token_same_applied_pct = (
                100.0 * st["token_matches_applied"] / token_total
                if token_total else 100.0
            )
            same_first_pct = (
                100.0 * max(st["same_as_first_calls"] - 1, 0) / repeat_obs
                if repeat_obs else 100.0
            )

            include = True
            if only_unstable_or_token_mismatch:
                include = len(st["unique_sets"]) > 1 or st["token_differs_from_applied"] > 0
            if not include:
                continue

            rows.append({
                "side": st["side"],
                "layer": st["layer"],
                "head": st["head"],
                "calls": calls,
                "unique_head_sets": len(st["unique_sets"]),
                "same_as_first_%": same_first_pct,
                "token_same_as_applied_%": token_same_applied_pct,
                "token_mismatch_count": st["token_differs_from_applied"],
                "first_outliers": st["first"],
            })
        return rows

    def static_mask_rows(self):
        """Return one row per side/layer/head with the most frequently selected dims.

        Popularity is counted over the head-level outlier set that was actually
        applied during the dynamic pass. For top-k=3, every dynamic call casts
        one vote for each of its three selected dimensions.
        """
        rows = []
        for key in sorted(self.head_stats.keys()):
            st = self.head_stats[key]
            counts = st.get("dimension_counts", {})
            if not counts:
                continue

            k = len(st["first"])
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_dims = tuple(int(dim) for dim, _ in ranked[:k])
            top_counts = tuple(int(cnt) for _, cnt in ranked[:k])

            rows.append({
                "side": st["side"],
                "layer": st["layer"],
                "head": st["head"],
                "calls": st["calls"],
                "static_outliers": top_dims,
                "static_outlier_counts": top_counts,
                "unique_head_sets": len(st["unique_sets"]),
                "first_outliers": st["first"],
            })
        return rows

    def build_static_masks(self):
        """Build masks keyed by (side, layer, head) from dynamic popularity."""
        masks = {}
        for row in self.static_mask_rows():
            masks[(row["side"], int(row["layer"]), int(row["head"]))] = tuple(row["static_outliers"])
        return masks

    def print_summary(self):
        banner("Outlier Reuse Summary")
        overall_raw = pd.DataFrame(self.overall_rows())
        if len(overall_raw) == 0:
            print("No outlier tracking data was collected. Check that outlier_dims > 0 and PQ was enabled.")
            return

        overall = overall_raw.copy()
        for col in [
            "same_as_first_%",
            "same_as_previous_%",
            "token_same_as_applied_%",
            "token_same_as_first_%",
        ]:
            overall[col] = overall[col].map(lambda x: f"{x:.2f}")
        print_df("Overall reuse", overall, index=False)

        head_rows_raw = pd.DataFrame(self.head_rows(only_unstable_or_token_mismatch=True))
        if len(head_rows_raw) == 0:
            print("\nEvery tracked layer/head reused the same head-level outlier set, and token-level top-k matched the applied set.")
            return

        unstable = head_rows_raw[head_rows_raw["unique_head_sets"] > 1].copy()
        if len(unstable) > 0:
            unstable_display = unstable.copy()
            for col in ["same_as_first_%", "token_same_as_applied_%"]:
                unstable_display[col] = unstable_display[col].map(lambda x: f"{x:.2f}")
            print_df(
                f"Most unstable head-level outlier sets, top {MAX_OUTLIER_TABLE_ROWS}",
                unstable_display,
                index=False,
                max_rows=MAX_OUTLIER_TABLE_ROWS,
                sort_by="same_as_first_%",
                ascending=True,
            )
        else:
            section("Head-level stability")
            print("  Every tracked layer/head reused the same head-level outlier set.")

        token_worst = head_rows_raw[head_rows_raw["token_mismatch_count"] > 0].copy()
        if len(token_worst) > 0:
            token_worst_display = token_worst.copy()
            for col in ["same_as_first_%", "token_same_as_applied_%"]:
                token_worst_display[col] = token_worst_display[col].map(lambda x: f"{x:.2f}")
            print_df(
                f"Worst token-level mismatch cases, top {MAX_OUTLIER_TABLE_ROWS}",
                token_worst_display,
                index=False,
                max_rows=MAX_OUTLIER_TABLE_ROWS,
                sort_by="token_same_as_applied_%",
                ascending=True,
            )

        if self.examples:
            examples = pd.DataFrame(self.examples[:MAX_TOKEN_MISMATCH_EXAMPLES])
            print_df("Example token-level mismatches", examples, index=False)

        section("Outlier table note")
        print("  The notebook printout is intentionally capped. Full per-layer/head data is saved to outlier_reuse_by_head.csv.")


class DualPQManager:
    def __init__(
        self,
        key_cfg,
        value_cfg,
        device=None,
        auto_remap_missing_groups=False,
        track_outliers=True,
        outlier_mode="dynamic",
        static_outlier_masks=None,
        track_token_outliers=False,
        quantize_keys=True,
        quantize_values=True,
    ):
        self.device = device if device is not None else get_device()
        self.key_cfg = dict(key_cfg)
        self.value_cfg = dict(value_cfg)
        self.enabled = False
        self.track_outliers = track_outliers
        self.outlier_tracker = OutlierReuseTracker() if track_outliers else None
        self.outlier_mode = outlier_mode
        self.static_outlier_masks = static_outlier_masks or {}
        self.track_token_outliers = track_token_outliers
        self.quantize_keys = quantize_keys
        self.quantize_values = quantize_values

        self.key_cfg["dim_per_sub"] = DIMS // self.key_cfg["num_sub_vectors"]
        self.value_cfg["dim_per_sub"] = DIMS // self.value_cfg["num_sub_vectors"]

        if self.key_cfg["num_sub_vectors"] != self.value_cfg["num_sub_vectors"]:
            raise ValueError(
                "Key/value configs must use the same number of banks/subvectors "
                f"for shared hardware. Got key={self.key_cfg['num_sub_vectors']}, "
                f"value={self.value_cfg['num_sub_vectors']}."
            )

        if DIMS % self.key_cfg["num_sub_vectors"] != 0:
            raise ValueError(
                f"DIMS={DIMS} not divisible by key num_sub_vectors={self.key_cfg['num_sub_vectors']}"
            )

        if DIMS % self.value_cfg["num_sub_vectors"] != 0:
            raise ValueError(
                f"DIMS={DIMS} not divisible by value num_sub_vectors={self.value_cfg['num_sub_vectors']}"
            )

        self.key_dir = side_dir_from_cfg(self.key_cfg, "keys")
        self.value_dir = side_dir_from_cfg(self.value_cfg, "values")

        self.key_map, self.key_group_remap, self.key_processors = self._init_side(
            cfg=self.key_cfg,
            base_dir=self.key_dir,
            side_name="keys",
            auto_remap_missing_groups=auto_remap_missing_groups,
        )

        self.val_map, self.val_group_remap, self.val_processors = self._init_side(
            cfg=self.value_cfg,
            base_dir=self.value_dir,
            side_name="values",
            auto_remap_missing_groups=auto_remap_missing_groups,
        )

    def reset_outlier_tracker(self):
        if self.track_outliers:
            self.outlier_tracker = OutlierReuseTracker()

    def set_outlier_mode(self, mode, static_outlier_masks=None):
        if mode not in {"dynamic", "static"}:
            raise ValueError(f"Unknown outlier mode: {mode}. Expected 'dynamic' or 'static'.")
        self.outlier_mode = mode
        if static_outlier_masks is not None:
            self.static_outlier_masks = static_outlier_masks

    def _init_side(self, cfg, base_dir, side_name, auto_remap_missing_groups):
        map_path = os.path.join(base_dir, "head_to_codebook_map.json")
        raw_map = load_head_to_codebook_map(map_path)

        fixed_map, remap = resolve_head_map_to_available_groups(
            base_dir=base_dir,
            head_map=raw_map,
            num_sub_vectors=cfg["num_sub_vectors"],
            auto_remap=auto_remap_missing_groups,
            allow_single_group_fallback=ALLOW_SINGLE_GROUP_FALLBACK,
            map_name=side_name,
        )

        processors = {}
        unique_groups = sorted(set(fixed_map.values()))

        print_kv_table(
            f"Initializing {side_name} PQ processors",
            [("unique groups", preview_list(unique_groups))],
        )
        for g in tqdm(unique_groups, desc=f"Loading {side_name} processors", leave=False):
            processors[g] = TorchGroupProcessor(
                base_dir=base_dir,
                group_id=g,
                num_sub_vectors=cfg["num_sub_vectors"],
                num_codewords=cfg["num_codewords"],
                dim_per_sub=cfg["dim_per_sub"],
                device=self.device,
            )

        return fixed_map, remap, processors

    def quantize_tensor(self, tensor, layer_idx, is_key=True):
        if not self.enabled:
            return tensor
        if is_key and not self.quantize_keys:
            return tensor
        if (not is_key) and not self.quantize_values:
            return tensor

        bsz, num_heads, seq_len, head_dim = tensor.shape

        if head_dim != DIMS:
            raise ValueError(f"Expected head_dim={DIMS}, got {head_dim}")

        if is_key:
            cfg, head_map, processors = self.key_cfg, self.key_map, self.key_processors
        else:
            cfg, head_map, processors = self.value_cfg, self.val_map, self.val_processors

        out = torch.empty_like(tensor)

        groups_to_heads = {}

        for h in range(num_heads):
            head_key = f"L{layer_idx}_H{h}"

            if head_key not in head_map:
                out[:, h, :, :] = tensor[:, h, :, :]
            else:
                group_id = head_map[head_key]
                groups_to_heads.setdefault(group_id, []).append(h)

        for group_id, heads in groups_to_heads.items():
            processor = processors[group_id]

            group_tensor = tensor[:, heads, :, :].contiguous()
            orig_shape = group_tensor.shape

            outlier_dims = int(cfg.get("outlier_dims", 0))
            outlier_indices = None
            token_outlier_indices = None

            if outlier_dims > 0:
                side_name = "keys" if is_key else "values"

                if self.outlier_mode == "static":
                    # Static pass: use the top-k most popular dims learned from the prior
                    # dynamic pass for each specific side/layer/head.
                    static_rows = []
                    missing_static = []
                    for h in heads:
                        mask_key = (side_name, int(layer_idx), int(h))
                        if mask_key in self.static_outlier_masks:
                            static_rows.append(list(self.static_outlier_masks[mask_key]))
                        else:
                            missing_static.append(h)
                            static_rows.append(None)

                    if missing_static:
                        # Fallback should almost never happen if dynamic and static use the
                        # same eval path. It keeps the run from crashing on unmapped heads.
                        mean_abs = group_tensor.abs().mean(dim=(0, 2))
                        _, dynamic_fallback = torch.topk(mean_abs, outlier_dims, dim=-1)
                        for local_i, row in enumerate(static_rows):
                            if row is None:
                                static_rows[local_i] = [int(x) for x in dynamic_fallback[local_i].detach().cpu().tolist()]

                    outlier_indices = torch.tensor(
                        static_rows,
                        dtype=torch.long,
                        device=group_tensor.device,
                    )
                else:
                    # Dynamic pass: one top-k set per selected layer/head for this forward chunk.
                    # Shape: [selected_heads, outlier_dims].
                    mean_abs = group_tensor.abs().mean(dim=(0, 2))
                    _, outlier_indices = torch.topk(mean_abs, outlier_dims, dim=-1)

                if self.track_outliers and self.outlier_tracker is not None:
                    if self.track_token_outliers:
                        _, token_outlier_indices = torch.topk(
                            group_tensor.abs(),
                            outlier_dims,
                            dim=-1,
                        )

                    for local_i, h in enumerate(heads):
                        token_indices_for_head = (
                            token_outlier_indices[:, local_i, :, :]
                            if token_outlier_indices is not None
                            else None
                        )
                        self.outlier_tracker.update(
                            side=side_name,
                            layer_idx=layer_idx,
                            head_idx=h,
                            applied_indices=outlier_indices[local_i],
                            token_indices=token_indices_for_head,
                        )

            group_flat = group_tensor.reshape(
                -1,
                cfg["num_sub_vectors"],
                cfg["dim_per_sub"],
            )

            quant_flat = processor.quantize(group_flat, chunk_size=PQ_CHUNK_SIZE)
            quantized_group = quant_flat.reshape(orig_shape)

            if outlier_indices is not None:
                for i in range(len(heads)):
                    head_outliers = outlier_indices[i]
                    quantized_group[:, i, :, head_outliers] = group_tensor[:, i, :, head_outliers]

            out[:, heads, :, :] = quantized_group

            del group_tensor, group_flat, quant_flat, quantized_group

        return out


# ============================================================
# Safetensors Loader
# ============================================================

def load_safetensors_pure(filepath):
    with open(filepath, "rb") as f:
        header_size_bytes = f.read(8)

        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {filepath}")

        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        offset = 8 + header_size
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        tensors = {}

        for k, v in header.items():
            if k == "__metadata__":
                continue

            dtype = v["dtype"]
            shape = v["shape"]
            data_offsets = v["data_offsets"]

            start = offset + data_offsets[0]
            end = offset + data_offsets[1]
            raw_data = mm[start:end]

            if dtype == "BF16":
                t = torch.frombuffer(
                    bytearray(raw_data),
                    dtype=torch.int16,
                ).view(torch.bfloat16).clone()
                tensors[k] = t.reshape(shape)

            elif dtype == "F32":
                t = torch.from_numpy(
                    np.frombuffer(raw_data, dtype=np.float32).copy()
                ).reshape(shape)
                tensors[k] = t

            elif dtype == "F16":
                t = torch.from_numpy(
                    np.frombuffer(raw_data, dtype=np.float16).copy()
                ).reshape(shape)
                tensors[k] = t

            else:
                raise NotImplementedError(
                    f"Unsupported dtype {dtype} in {filepath} for tensor {k}"
                )

        mm.close()

    return tensors


# ============================================================
# Qwen Model
# ============================================================

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=4096,
        base=1000000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device=None):
        device = device if device is not None else self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.max_seq_len_cached = seq_len

    def forward(self, x, position_ids):
        needed = int(position_ids.max().item()) + 1
        if needed > self.max_seq_len_cached or self.cos_cached.device != x.device:
            new_len = max(needed, max(2 * self.max_seq_len_cached, 16))
            self._set_cos_sin_cache(new_len, device=x.device)

        flat_pos = position_ids.reshape(-1)
        cos = self.cos_cached.index_select(0, flat_pos).view(*position_ids.shape, self.dim)
        sin = self.sin_cached.index_select(0, flat_pos).view(*position_ids.shape, self.dim)
        return cos.unsqueeze(1).to(dtype=x.dtype), sin.unsqueeze(1).to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx, pq_manager=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.pq_manager = pq_manager
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=config.get("attention_bias", False),
        )
        self.q_norm = QwenRMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))
        self.k_norm = QwenRMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))
        max_pos = max(
            int(config.get("max_position_embeddings", 4096)),
            MAX_INPUT_TOKENS + max(DATASET2MAXLEN.values()) + 16,
        )
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_pos,
            base=config.get("rope_theta", 1000000.0),
        )

    def forward(self, hidden_states, position_ids, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Preserve the original experiment's quantization point: K/V are quantized
        # before RoPE, then the quantized key is rotated and stored in the cache.
        if self.pq_manager is not None and self.pq_manager.enabled:
            key_states = self.pq_manager.quantize_tensor(
                key_states, self.layer_idx, is_key=True
            )
            value_states = self.pq_manager.quantize_tensor(
                value_states, self.layer_idx, is_key=False
            )

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        past_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_len = past_key.shape[-2]
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        present_key_value = (key_states, value_states) if use_cache else None
        attn_key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_mask = None
        is_causal = past_len == 0 and q_len > 1
        if past_len > 0 and q_len > 1:
            kv_len = attn_key_states.shape[-2]
            q_positions = torch.arange(q_len, device=hidden_states.device) + past_len
            k_positions = torch.arange(kv_len, device=hidden_states.device)
            attn_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            attn_key_states,
            attn_value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output), present_key_value


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, pq_manager=None):
        super().__init__()
        self.self_attn = QwenAttention(config, layer_idx, pq_manager)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.post_attention_layernorm = QwenRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )

    def forward(self, hidden_states, position_ids, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class QwenModel(nn.Module):
    def __init__(self, config, pq_manager=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, i, pq_manager) for i in range(config["num_hidden_layers"])]
        )
        self.norm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError("past_key_values length does not match number of layers")
            first = past_key_values[0]
            past_len = 0 if first is None else first[0].shape[-2]

        position_ids = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        next_cache = [] if use_cache else None
        for layer, layer_past in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                next_cache.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class QwenForCausalLM(nn.Module):
    def __init__(self, config, pq_manager=None):
        super().__init__()
        self.model = QwenModel(config, pq_manager)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, logits_to_keep=0):
        hidden_states, next_cache = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if logits_to_keep and hidden_states.shape[1] > logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        return logits, next_cache


def set_model_pq_manager(model, pq_manager):
    for layer in model.model.layers:
        layer.self_attn.pq_manager = pq_manager


# ============================================================
# LongBench prompts, metrics, generation, and evaluation
# ============================================================

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}

DATASET2MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "multifieldqa_zh": 64, "hotpotqa": 32, "2wikimqa": 32,
    "musique": 32, "dureader": 128, "gov_report": 512, "qmsum": 512,
    "multi_news": 512, "vcsum": 512, "trec": 64, "triviaqa": 32,
    "samsum": 128, "lsht": 64, "passage_count": 32,
    "passage_retrieval_en": 32, "passage_retrieval_zh": 32,
    "lcc": 64, "repobench-p": 64,
}

NO_CHAT_TEMPLATE_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def normalize_zh_answer(s):
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in str(s).lower() if ch not in all_punctuation).replace(" ", "")


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Install jieba for Chinese LongBench metrics: pip install jieba") from exc
    p = [normalize_zh_answer(x) for x in jieba.cut(prediction, cut_all=False)]
    g = [normalize_zh_answer(x) for x in jieba.cut(ground_truth, cut_all=False)]
    return f1_score([x for x in p if x], [x for x in g if x])


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise ImportError("Install rouge for summarization metrics: pip install rouge") from exc
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction, ground_truth, **kwargs):
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Install jieba for Chinese LongBench metrics: pip install jieba") from exc
    return rouge_score(
        " ".join(jieba.cut(prediction, cut_all=False)),
        " ".join(jieba.cut(ground_truth, cut_all=False)),
    )


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs.get("all_classes") or []
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [x for x in matches if not (x in ground_truth and x != ground_truth)]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", str(ground_truth))
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    right = sum(str(x) == matches[0] for x in numbers)
    return 0.0 if not numbers else right / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"段落(\d+)", str(ground_truth))
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    right = sum(str(x) == matches[0] for x in numbers)
    return 0.0 if not numbers else right / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right = sum(str(x) == str(ground_truth) for x in numbers)
    return 0.0 if not numbers else right / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    try:
        from fuzzywuzzy import fuzz
    except ImportError as exc:
        raise ImportError("Install fuzzywuzzy for code metrics: pip install fuzzywuzzy") from exc
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return fuzz.ratio(candidate, ground_truth) / 100.0


DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score,
    "gov_report": rouge_score, "qmsum": rouge_score,
    "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score,
    "samsum": rouge_score, "lsht": classification_score,
    "passage_retrieval_en": retrieval_score, "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


def score_longbench_prediction(dataset, prediction, answers, all_classes):
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    best = 0.0
    for answer in answers:
        best = max(
            best,
            DATASET2METRIC[dataset](prediction, answer, all_classes=all_classes),
        )
    return best


def middle_truncate_ids(input_ids, max_tokens):
    if len(input_ids) <= max_tokens:
        return input_ids
    first = max_tokens // 2
    second = max_tokens - first
    return input_ids[:first] + input_ids[-second:]


def build_longbench_prompt(tokenizer, dataset, example):
    prompt = DATASET2PROMPT[dataset].format(**example)
    used_chat_template = False
    if USE_CHAT_TEMPLATE and dataset not in NO_CHAT_TEMPLATE_TASKS and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if DISABLE_QWEN_THINKING:
            kwargs["enable_thinking"] = False
        try:
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        used_chat_template = True
    token_ids = tokenizer.encode(prompt, add_special_tokens=not used_chat_template)
    return middle_truncate_ids(token_ids, MAX_INPUT_TOKENS)


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos}
    return {int(eos)}


def greedy_generate(eval_model, tokenizer, input_ids, max_new_tokens):
    eval_model.eval()
    device = next(eval_model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []
    eos_ids = eos_token_ids(tokenizer)
    past_key_values = None

    with torch.inference_mode():
        logits, past_key_values = eval_model(
            input_tensor,
            use_cache=True,
            logits_to_keep=1,
        )
        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if next_token in eos_ids:
                break
            generated.append(next_token)
            step_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, past_key_values = eval_model(
                step_input,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )

    return tokenizer.decode(generated, skip_special_tokens=True), generated


def _download_longbench_parquet_paths(repo, config_name):
    """Download and return local paths for one LongBench test configuration.

    This deliberately uses ``hf_hub_download`` and the generic Parquet loader.
    It never asks ``datasets`` to load the LongBench repository itself, so the
    legacy LongBench.py dataset script cannot be selected.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo_files = HfApi(token=False).list_repo_files(
        repo_id=repo,
        repo_type="dataset",
        revision=LONG_BENCH_REVISION,
    )
    prefix = f"{config_name}/"
    shard_names = sorted(
        path
        for path in repo_files
        if path.startswith(prefix)
        and path.endswith(".parquet")
        and os.path.basename(path).startswith("test-")
    )

    if not shard_names:
        raise FileNotFoundError(
            f"No test Parquet shards found under {repo}/{config_name}/"
        )

    return [
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=LONG_BENCH_REVISION,
            token=False,
        )
        for filename in shard_names
    ]


def _stable_dataset_seed(config_name):
    material = f"{SAMPLE_SELECTION_SEED}:{LONG_BENCH_REVISION}:{config_name}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_longbench_indices(config_name, total_rows):
    if SAMPLE_SELECTION == "all":
        return list(range(total_rows))

    sample_count = min(MAX_SAMPLES_PER_DATASET, total_rows)
    if SAMPLE_SELECTION == "first":
        return list(range(sample_count))

    rng = random.Random(_stable_dataset_seed(config_name))
    return sorted(rng.sample(range(total_rows), sample_count))


def load_longbench_examples(dataset):
    from datasets import load_dataset

    config_name = f"{dataset}_e" if USE_LONG_BENCH_E else dataset
    errors = []

    for repo in LONG_BENCH_REPOS:
        try:
            parquet_paths = _download_longbench_parquet_paths(repo, config_name)

            # IMPORTANT: the first argument must remain exactly "parquet".
            # Do not replace it with repo or LongBench; doing so invokes the
            # unsupported legacy LongBench.py dataset script.
            ds = load_dataset(
                "parquet",
                data_files={"test": parquet_paths},
                split="test",
            )

            total_rows = len(ds)
            selected_indices = select_longbench_indices(config_name, total_rows)
            ds = ds.select(selected_indices)

            source = (
                f"{repo}@{LONG_BENCH_REVISION}/{config_name} "
                f"({len(parquet_paths)} locally cached parquet shard(s))"
            )
            rows = []
            for selected_position, (source_index, row) in enumerate(
                zip(selected_indices, ds)
            ):
                item = dict(row)
                item["_longbench_source_index"] = int(source_index)
                item["_longbench_selected_position"] = int(selected_position)
                item["_longbench_source_rows"] = int(total_rows)
                item["_longbench_selection_method"] = SAMPLE_SELECTION
                item["_longbench_selection_seed"] = int(SAMPLE_SELECTION_SEED)
                rows.append(item)
            return rows, source, config_name

        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")

    error_text = "\n  - ".join(errors)
    raise RuntimeError(
        f"Could not load LongBench config {config_name} from direct Parquet files."
        f"\n  - {error_text}"
    )


def preload_longbench_data():
    loaded = {}
    rows = []
    for dataset in LONG_BENCH_DATASETS:
        examples, repo, config_name = load_longbench_examples(dataset)
        selected_indices = [
            int(example["_longbench_source_index"])
            for example in examples
        ]
        source_rows = (
            int(examples[0]["_longbench_source_rows"])
            if examples else 0
        )
        loaded[dataset] = examples
        rows.append({
            "dataset": dataset,
            "config": config_name,
            "samples": len(examples),
            "source_samples": source_rows,
            "selection": SAMPLE_SELECTION,
            "selection_seed": SAMPLE_SELECTION_SEED,
            "selected_indices_preview": selected_indices[:10],
            "repository": repo,
        })
    print_df("Loaded LongBench datasets", pd.DataFrame(rows), index=False)
    return loaded


def write_sample_selection_manifest(data_by_dataset):
    manifest_path = os.path.join(OUTPUT_DIR, "sample_selection_manifest.json")
    datasets = {}
    for dataset, examples in data_by_dataset.items():
        source_rows = (
            int(examples[0]["_longbench_source_rows"])
            if examples else 0
        )
        datasets[dataset] = {
            "samples": len(examples),
            "source_samples": source_rows,
            "selected_source_indices": [
                int(example["_longbench_source_index"])
                for example in examples
            ],
            "selected_ids": [
                example.get("_id")
                for example in examples
            ],
        }

    payload = {
        "experiment": EXPERIMENT_NAME,
        "run_tag": _run_tag,
        "output_dir": OUTPUT_DIR,
        "use_longbench_e": USE_LONG_BENCH_E,
        "longbench_revision": LONG_BENCH_REVISION,
        "longbench_loader": LONG_BENCH_LOADER_VERSION,
        "datasets": list(LONG_BENCH_DATASETS),
        "max_samples_per_dataset": MAX_SAMPLES_PER_DATASET,
        "sample_selection": SAMPLE_SELECTION,
        "sample_selection_seed": SAMPLE_SELECTION_SEED,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens_cap": MAX_NEW_TOKENS_CAP,
        "eval_modes": list(EVAL_MODES),
        "calibration_mode": EVAL_CALIBRATION_MODE,
        "calibration_variant": EVAL_CALIBRATION_VARIANT,
        "codebook_dir": LONGBENCH_E_CODEBOOK_DIR,
        "datasets_loaded": datasets,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def evaluate_longbench_mode(eval_model, tokenizer, data_by_dataset, mode_name):
    banner(f"LongBench mode: {mode_name}")
    all_records = []
    summary_rows = []

    for dataset, examples in data_by_dataset.items():
        section(f"{mode_name}: {dataset}")
        dataset_records = []
        max_new = DATASET2MAXLEN[dataset]
        if MAX_NEW_TOKENS_CAP is not None:
            max_new = min(max_new, MAX_NEW_TOKENS_CAP)

        for sample_idx, example in enumerate(tqdm(examples, desc=f"{mode_name}/{dataset}")):
            input_ids = build_longbench_prompt(tokenizer, dataset, example)
            started = time.perf_counter()
            prediction, generated_ids = greedy_generate(
                eval_model,
                tokenizer,
                input_ids,
                max_new_tokens=max_new,
            )
            elapsed = time.perf_counter() - started
            answers = list(example.get("answers") or [])
            all_classes = example.get("all_classes")
            score = score_longbench_prediction(
                dataset,
                prediction,
                answers,
                all_classes,
            )
            record = {
                "mode": mode_name,
                "dataset": dataset,
                "sample_index": sample_idx,
                "_id": example.get("_id"),
                "source_index": example.get("_longbench_source_index"),
                "selected_position": example.get("_longbench_selected_position"),
                "source_rows": example.get("_longbench_source_rows"),
                "sample_selection": example.get("_longbench_selection_method"),
                "sample_selection_seed": example.get("_longbench_selection_seed"),
                "run_tag": _run_tag,
                "calibration_mode": EVAL_CALIBRATION_MODE,
                "calibration_variant": EVAL_CALIBRATION_VARIANT,
                "codebook_dir": LONGBENCH_E_CODEBOOK_DIR,
                "max_new_tokens": max_new,
                "prediction": prediction,
                "answers": answers,
                "all_classes": all_classes,
                "dataset_length": example.get("length"),
                "input_tokens": len(input_ids),
                "generated_tokens": len(generated_ids),
                "score": score,
                "elapsed_seconds": elapsed,
            }
            dataset_records.append(record)
            all_records.append(record)
            cleanup_cuda()

        score_pct = 100.0 * np.mean([x["score"] for x in dataset_records])
        summary_rows.append({
            "mode": mode_name,
            "dataset": dataset,
            "samples": len(dataset_records),
            "score": score_pct,
            "avg_input_tokens": np.mean([x["input_tokens"] for x in dataset_records]),
            "avg_generated_tokens": np.mean([x["generated_tokens"] for x in dataset_records]),
            "total_seconds": sum(x["elapsed_seconds"] for x in dataset_records),
        })
        print(f"  {dataset} score: {score_pct:.2f}")
        save_jsonl(
            os.path.join(OUTPUT_DIR, mode_name, f"{dataset}.jsonl"),
            dataset_records,
        )

    dataset_scores = [x["score"] for x in summary_rows]
    overall_unweighted = float(np.mean(dataset_scores)) if dataset_scores else float("nan")
    overall_weighted = (
        100.0 * np.mean([x["score"] for x in all_records])
        if all_records else float("nan")
    )
    print_kv_table(
        f"{mode_name} aggregate",
        [
            ("dataset-average score", overall_unweighted),
            ("sample-weighted score", overall_weighted),
            ("samples", len(all_records)),
        ],
    )
    return all_records, summary_rows, overall_unweighted, overall_weighted


def comparison_rows(summary_by_mode):
    rows = []
    datasets = sorted({r["dataset"] for rows_ in summary_by_mode.values() for r in rows_})
    lookup = {
        (mode, row["dataset"]): row
        for mode, rows_ in summary_by_mode.items()
        for row in rows_
    }
    for dataset in datasets:
        row = {
            "experiment": EXPERIMENT_NAME,
            "dataset": dataset,
        }
        baseline_row = lookup.get(("baseline", dataset))
        dynamic_row = lookup.get(("dynamic", dataset))
        static_row = lookup.get(("static", dataset))

        if baseline_row is not None:
            row["baseline_score"] = baseline_row["score"]
        if dynamic_row is not None:
            row["dynamic_pq_score"] = dynamic_row["score"]
            if baseline_row is not None:
                row["dynamic_minus_baseline"] = (
                    dynamic_row["score"] - baseline_row["score"]
                )
        if static_row is not None:
            row["static_top3_pq_score"] = static_row["score"]
            if baseline_row is not None:
                row["static_minus_baseline"] = (
                    static_row["score"] - baseline_row["score"]
                )
            if dynamic_row is not None:
                row["static_minus_dynamic"] = (
                    static_row["score"] - dynamic_row["score"]
                )
        rows.append(row)
    return rows


def load_model_weights(model, model_dir):
    state_dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for path in safetensor_files:
        tensors = load_safetensors_pure(path)
        state_dict.update(tensors)
        cleanup_cuda()
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    model.load_state_dict(state_dict, strict=False)
    cleanup_cuda()


def validate_single_config():
    for side_name, cfg, folder_side in [("key", KEY_CONFIG, "keys"), ("value", VALUE_CONFIG, "values")]:
        side_folder = side_dir_from_cfg(cfg, folder_side)
        map_path = os.path.join(side_folder, "head_to_codebook_map.json")
        if not os.path.exists(map_path): raise FileNotFoundError(f"Missing map: {map_path}")




def save_static_masks(static_masks, json_path=STATIC_MASK_JSON, csv_path=STATIC_MASK_CSV):
    """Save static masks in both machine-readable JSON and spreadsheet-friendly CSV."""
    json_obj = {}
    rows = []
    for (side, layer, head), dims in sorted(static_masks.items()):
        key = f"{side}/L{int(layer)}_H{int(head)}"
        dims = tuple(int(x) for x in dims)
        json_obj[key] = list(dims)
        rows.append({
            "side": side,
            "layer": int(layer),
            "head": int(head),
            "static_outliers": dims,
        })

    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_calibration_static_masks(codebook_dir):
    path = os.path.join(codebook_dir, "static_outlier_masks.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing calibration-derived static masks: {path}. "
            "Run the improved codebook trainer before LongBench-E evaluation."
        )
    with open(path, "r") as f:
        raw = json.load(f)

    masks = {}
    for key, dims in raw.items():
        side, head_name = key.split("/", 1)
        match = re.fullmatch(r"L(\d+)_H(\d+)", head_name)
        if match is None:
            raise ValueError(f"Malformed static-mask key: {key}")
        masks[(side, int(match.group(1)), int(match.group(2)))] = tuple(
            int(x) for x in dims
        )
    return masks, path


def save_tracker_csvs(tracker, overall_csv, by_head_csv):
    pd.DataFrame(tracker.overall_rows()).to_csv(overall_csv, index=False)
    pd.DataFrame(tracker.head_rows(only_unstable_or_token_mismatch=False)).to_csv(by_head_csv, index=False)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    banner("LongBench Baseline vs Dynamic/Static KV-Cache PQ")
    device = get_device()
    gpu_name = validate_required_gpu()
    cleanup_cuda()
    torch.set_default_dtype(torch.bfloat16)
    validate_single_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    key_bits, value_bits, avg_bps, comp_ratio = overall_compression_ratio(
        KEY_CONFIG, VALUE_CONFIG
    )
    print_kv_table(
        "Run configuration",
        [
            ("experiment", EXPERIMENT_NAME),
            ("model_dir", MODEL_DIR),
            ("device", device),
            ("gpu", gpu_name),
            ("required gpu", REQUIRED_GPU_NAME or "<none>"),
            ("test_mode", TEST_MODE),
            ("eval modes", EVAL_MODES),
            ("calibration mode", EVAL_CALIBRATION_MODE),
            ("longbench_e", USE_LONG_BENCH_E),
            ("LongBench loader", LONG_BENCH_LOADER_VERSION),
            ("datasets", preview_list(LONG_BENCH_DATASETS)),
            ("calibration variant", EVAL_CALIBRATION_VARIANT),
            ("max samples/dataset", MAX_SAMPLES_PER_DATASET),
            ("sample selection", SAMPLE_SELECTION),
            ("sample seed", SAMPLE_SELECTION_SEED),
            ("max input tokens", MAX_INPUT_TOKENS),
            ("max new token cap", MAX_NEW_TOKENS_CAP),
            ("chat template", USE_CHAT_TEMPLATE),
            ("Qwen thinking disabled", DISABLE_QWEN_THINKING),
            ("track token-level outliers", TRACK_TOKEN_LEVEL_OUTLIERS),
            ("key config", KEY_CONFIG),
            ("value config", VALUE_CONFIG),
            ("key bits/vector", key_bits),
            ("value bits/vector", value_bits),
            ("avg bits/scalar", avg_bps),
            ("compression ratio", comp_ratio),
        ],
    )

    section("Loading model configuration")
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)
    print_kv_table(
        "Model summary",
        [
            ("hidden_size", config.get("hidden_size")),
            ("layers", config.get("num_hidden_layers")),
            ("attention heads", config.get("num_attention_heads")),
            ("kv heads", config.get("num_key_value_heads", config.get("num_attention_heads"))),
            ("head_dim", config.get("head_dim", config.get("hidden_size") // config.get("num_attention_heads"))),
            ("vocab_size", config.get("vocab_size")),
            ("max positions", config.get("max_position_embeddings")),
        ],
    )

    if int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])) != DIMS:
        raise ValueError(
            f"DIMS={DIMS} does not match model head_dim="
            f"{config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])}"
        )

    pq_manager = DualPQManager(
        KEY_CONFIG,
        VALUE_CONFIG,
        device,
        AUTO_REMAP_MISSING_GROUPS,
        track_outliers=True,
        track_token_outliers=TRACK_TOKEN_LEVEL_OUTLIERS,
    )

    section("Loading model weights")
    model = QwenForCausalLM(config, pq_manager=None)
    load_model_weights(model, MODEL_DIR)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("  Model and tokenizer loaded.")

    data_by_dataset = preload_longbench_data()
    sample_manifest_path = write_sample_selection_manifest(data_by_dataset)
    print(f"  Sample manifest saved: {sample_manifest_path}")
    summary_by_mode = {}
    mode_avgs = {}
    mode_weighted = {}
    dynamic_tracker = None
    static_tracker = None

    if "baseline" in EVAL_MODES:
        section("Baseline LongBench")
        pq_manager.enabled = False
        set_model_pq_manager(model, None)
        _, baseline_summary, baseline_avg, baseline_weighted = evaluate_longbench_mode(
            model, tokenizer, data_by_dataset, "baseline"
        )
        summary_by_mode["baseline"] = baseline_summary
        mode_avgs["baseline"] = baseline_avg
        mode_weighted["baseline"] = baseline_weighted

    if RUN_KEY_VALUE_SIDE_DIAGNOSTICS:
        section("Key-only dynamic PQ LongBench")
        pq_manager.enabled = True
        pq_manager.quantize_keys = True
        pq_manager.quantize_values = False
        pq_manager.set_outlier_mode("dynamic")
        pq_manager.reset_outlier_tracker()
        set_model_pq_manager(model, pq_manager)
        _, key_only_summary, key_only_avg, key_only_weighted = evaluate_longbench_mode(
            model, tokenizer, data_by_dataset, "key_only_dynamic"
        )
        summary_by_mode["key_only_dynamic"] = key_only_summary
        mode_avgs["key_only_dynamic"] = key_only_avg
        mode_weighted["key_only_dynamic"] = key_only_weighted

        section("Value-only dynamic PQ LongBench")
        pq_manager.enabled = True
        pq_manager.quantize_keys = False
        pq_manager.quantize_values = True
        pq_manager.set_outlier_mode("dynamic")
        pq_manager.reset_outlier_tracker()
        set_model_pq_manager(model, pq_manager)
        _, value_only_summary, value_only_avg, value_only_weighted = evaluate_longbench_mode(
            model, tokenizer, data_by_dataset, "value_only_dynamic"
        )
        summary_by_mode["value_only_dynamic"] = value_only_summary
        mode_avgs["value_only_dynamic"] = value_only_avg
        mode_weighted["value_only_dynamic"] = value_only_weighted

    if "dynamic" in EVAL_MODES:
        section("Dynamic-outlier PQ LongBench")
        pq_manager.enabled = True
        pq_manager.quantize_keys = True
        pq_manager.quantize_values = True
        pq_manager.set_outlier_mode("dynamic")
        pq_manager.reset_outlier_tracker()
        set_model_pq_manager(model, pq_manager)
        _, dynamic_summary, dynamic_avg, dynamic_weighted = evaluate_longbench_mode(
            model, tokenizer, data_by_dataset, "dynamic"
        )
        summary_by_mode["dynamic"] = dynamic_summary
        mode_avgs["dynamic"] = dynamic_avg
        mode_weighted["dynamic"] = dynamic_weighted
        dynamic_tracker = pq_manager.outlier_tracker
        dynamic_tracker.print_summary()

    if "static" in EVAL_MODES:
        section(f"Loading {EVAL_CALIBRATION_MODE} calibration static masks")
        static_masks, static_mask_source = load_calibration_static_masks(
            LONGBENCH_E_CODEBOOK_DIR
        )
        save_static_masks(static_masks, json_path=STATIC_MASK_JSON, csv_path=STATIC_MASK_CSV)
        print_kv_table(
            "Static mask summary",
            [
                ("masks learned", len(static_masks)),
                ("source", static_mask_source),
                ("json", STATIC_MASK_JSON),
                ("csv", STATIC_MASK_CSV),
            ],
        )

        section("Static-mask PQ LongBench")
        pq_manager.quantize_keys = True
        pq_manager.quantize_values = True
        pq_manager.set_outlier_mode("static", static_outlier_masks=static_masks)
        pq_manager.reset_outlier_tracker()
        set_model_pq_manager(model, pq_manager)
        _, static_summary, static_avg, static_weighted = evaluate_longbench_mode(
            model, tokenizer, data_by_dataset, "static"
        )
        summary_by_mode["static"] = static_summary
        mode_avgs["static"] = static_avg
        mode_weighted["static"] = static_weighted
        static_tracker = pq_manager.outlier_tracker
        static_tracker.print_summary()

    comparison = pd.DataFrame(comparison_rows(summary_by_mode))
    print_df("Per-dataset LongBench comparison", comparison, index=False)
    comparison.to_csv(RESULTS_CSV, index=False)

    summary_df = pd.DataFrame(
        [row for rows_ in summary_by_mode.values() for row in rows_]
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    aggregate_row = {
        "experiment": EXPERIMENT_NAME,
        "gpu": gpu_name,
        "required_gpu": REQUIRED_GPU_NAME,
        "eval_modes": ",".join(EVAL_MODES),
        "test_mode": TEST_MODE,
        "calibration_mode": EVAL_CALIBRATION_MODE,
        "calibration_variant": EVAL_CALIBRATION_VARIANT,
        "codebook_dir": LONGBENCH_E_CODEBOOK_DIR,
        "output_dir": OUTPUT_DIR,
        "sample_manifest": sample_manifest_path,
        "max_samples_per_dataset": MAX_SAMPLES_PER_DATASET,
        "sample_selection": SAMPLE_SELECTION,
        "sample_selection_seed": SAMPLE_SELECTION_SEED,
        "max_new_tokens_cap": MAX_NEW_TOKENS_CAP,
        "key_bits_per_vector": key_bits,
        "value_bits_per_vector": value_bits,
        "avg_bits_per_scalar": avg_bps,
        "compression_ratio": comp_ratio,
    }
    for mode in EVAL_MODES:
        aggregate_row[f"{mode}_dataset_avg"] = mode_avgs.get(mode, float("nan"))
        aggregate_row[f"{mode}_sample_weighted"] = mode_weighted.get(mode, float("nan"))
    if "baseline" in mode_avgs and "dynamic" in mode_avgs:
        aggregate_row["dynamic_minus_baseline"] = (
            mode_avgs["dynamic"] - mode_avgs["baseline"]
        )
    if "baseline" in mode_avgs and "static" in mode_avgs:
        aggregate_row["static_minus_baseline"] = (
            mode_avgs["static"] - mode_avgs["baseline"]
        )
    if "dynamic" in mode_avgs and "static" in mode_avgs:
        aggregate_row["static_minus_dynamic"] = (
            mode_avgs["static"] - mode_avgs["dynamic"]
        )
    aggregate_df = pd.DataFrame([aggregate_row])
    aggregate_path = os.path.join(OUTPUT_DIR, "aggregate_result.csv")
    aggregate_df.to_csv(aggregate_path, index=False)
    print_df("Aggregate LongBench result", aggregate_df, index=False)

    saved_files = [
        ("per-dataset comparison", RESULTS_CSV),
        ("mode summary", SUMMARY_CSV),
        ("aggregate result", aggregate_path),
        ("sample manifest", sample_manifest_path),
        ("predictions root", OUTPUT_DIR),
    ]
    if dynamic_tracker is not None:
        save_tracker_csvs(
            dynamic_tracker,
            DYNAMIC_OUTLIER_OVERALL_CSV,
            DYNAMIC_OUTLIER_BY_HEAD_CSV,
        )
        saved_files.extend([
            ("dynamic outlier overall csv", DYNAMIC_OUTLIER_OVERALL_CSV),
            ("dynamic outlier by-head csv", DYNAMIC_OUTLIER_BY_HEAD_CSV),
        ])
    if static_tracker is not None:
        save_tracker_csvs(
            static_tracker,
            STATIC_OUTLIER_OVERALL_CSV,
            STATIC_OUTLIER_BY_HEAD_CSV,
        )
        saved_files.extend([
            ("static mask json", STATIC_MASK_JSON),
            ("static mask csv", STATIC_MASK_CSV),
            ("static outlier overall csv", STATIC_OUTLIER_OVERALL_CSV),
            ("static outlier by-head csv", STATIC_OUTLIER_BY_HEAD_CSV),
        ])

    print_kv_table(
        "Saved files",
        saved_files,
    )
