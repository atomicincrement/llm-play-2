# Transformer LLM from Scratch in Rust — Implementation Plan

**Stack:** `ndarray` (tensor math via BLAS), `serde_json` (JSON parsing), `memmap2` (zero-copy file mapping)  
**Target model:** `Qwen/Qwen2.5-0.5B`

---

## Where to get the weights

Using **Qwen2.5-0.5B** (0.5 billion parameters) from ModelScope.

```
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='./models/Qwen2.5-0.5B')
"
```

Key files:
- `model.safetensors` — raw weight tensors (all BF16)
- `config.json` — model hyperparameters
- `tokenizer.json` — BPE vocab and merge rules

---

## Steps

Each step below is self-contained and can be turned into its own prompt.

---

### Step 1 — Project setup and dependencies

Add `ndarray`, `memmap2`, and `serde_json` to `Cargo.toml`.  
Confirm the project builds with a `Hello, world!` smoke test.

---

### Step 2 — Safetensors loader using serde_json ✅

Safetensors files begin with an 8-byte little-endian length prefix followed by
a UTF-8 JSON header, then a binary data blob.

Implement `load_safetensors(path: &Path) -> HashMap<String, ArrayD<f32>>`:
- `mmap` the file with the `memmap2` crate.
- Read the 8-byte header length.
- Use `serde_json::from_str` to parse the JSON header.
- For each tensor entry, read `dtype`, `shape`, and `data_offsets`.
- Slice the binary region at `data_offsets` and copy/cast into an
  `ndarray::ArrayD<f32>`. BF16 and F16 are promoted to F32 on load.

---

### Step 3 — Config loader using serde_json ✅

Implement `ModelConfig` (plain struct with public fields) and
`ModelConfig::from_file(path: &Path) -> ModelConfig`:
- Use `serde_json::from_str` to parse `config.json`.
- Extract: `hidden_size`, `intermediate_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`,
  `rope_theta`, `vocab_size`, `max_position_embeddings`,
  `bos_token_id`, `eos_token_id`, `sliding_window`, `use_sliding_window`.

---

### Step 4 — BPE tokenizer (encode + decode) ✅

Implement a minimal byte-pair encoding tokenizer from `tokenizer.json`:
- Parse the file with `serde_json` to extract `vocab` (token→id map) and
  `merges` (ordered list of byte-pair rules).
- Implement `encode(text: &str) -> Vec<u32>` using the standard BPE greedy merge loop.
- Implement `decode(ids: &[u32]) -> String` by reverse-lookup and byte-to-UTF-8 healing.
- Add the Qwen2.5 special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, etc.

---

### Step 5 — RMS Norm ✅

Implement `fn rms_norm(x: ArrayView1<f32>, weight: ArrayView1<f32>, eps: f32) -> Array1<f32>`
in `src/norm.rs`:

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \varepsilon}} \cdot w_i$$

Use `ndarray` element-wise operations (`dot`, scalar multiply, broadcast) with `f32::sqrt`; no loops.

---

### Step 6 — Rotary Position Encoding (RoPE) ✅

Implemented in `src/rope.rs`.

`fn apply_rope(xq: ArrayViewMut2<f32>, xk: ArrayViewMut2<f32>, pos: usize, head_dim: usize, base: f32)`:
- Precompute the frequency table:
  $\theta_i = \text{theta}^{-2i/d}$ for $i \in [0, d/2)$.
- At each position $p$, split each head vector into pairs and apply the 2D rotation:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

---

### Step 7 — Grouped-Query Attention (GQA) ✅

Implemented in `src/attention.rs`.

`fn attention(x: ArrayView1<f32>, w: &AttentionWeights, pos: usize, kv_cache: &mut KvCache, cfg: &ModelConfig) -> Array1<f32>`:
- Project to Q, K, V with `wq`, `wk`, `wv` weight matrices (ndarray `dot`).
- Apply RoPE to Q and K.
- Append K, V slices to the KV cache (a `Vec<Array2<f32>>` per layer).
- Repeat K and V heads to match Q head count (`num_attention_heads / num_key_value_heads`).
- Compute scaled dot-product scores:

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$

- Apply causal mask (set scores for future positions to $-\infty$).
- Softmax over the sequence dimension.
- Weighted sum over V, flatten heads, project with `wo`.

---

### Step 8 — SwiGLU Feed-Forward Network ✅

Qwen2.5 uses the SwiGLU activation:

$$\text{FFN}(x) = (\text{SiLU}(xW_\text{gate}) \odot xW_\text{up}) \cdot W_\text{down}$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$.

Implement `fn feed_forward(x: ArrayView1<f32>, layer: &FfnWeights) -> Array1<f32>` using `ndarray` dot products and element-wise multiply.

---

### Step 9 — Transformer block and full model ✅

Assemble one `TransformerBlock::forward`:
1. `residual = x`
2. `x = rms_norm(x, attn_norm_weight)`
3. `x = attention(x, ...) + residual`
4. `residual = x`
5. `x = rms_norm(x, ffn_norm_weight)`
6. `x = feed_forward(x, ...) + residual`

Then implement `QwenModel::forward(token_ids: &[u32]) -> Array2<f32>`:
- Embed tokens via the embedding table (row-index lookup).
- Run through all `num_hidden_layers` blocks with the shared KV cache.
- Apply final RMS norm.
- Project to logits with the `lm_head` weight matrix.

---

### Step 10 — Sampling and inference loop ✅

Implement `fn sample_top_p(logits: ArrayView1<f32>, top_p: f32, temperature: f32) -> u32`:
- Divide logits by temperature, softmax, sort descending.
- Accumulate probabilities until the cumulative sum exceeds `top_p`.
- Sample from the truncated distribution.

Implement the generation loop:
```
tokens = encode(prompt)
while last_token != EOS && tokens.len() < max_len:
    logits = model.forward(&tokens)
    next = sample_top_p(logits.last_row(), top_p, temperature)
    tokens.push(next)
    print(decode(&[next]))
```

---

### Step 11 — End-to-end test ✅

Load `Qwen2.5-0.5B` weights, prompt with `"The capital of France is"`,
and verify the next token is `" Paris"`.  
Measure tokens/sec on a single CPU core as a baseline.
