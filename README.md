# llm-play-2

A learning project: implementing a transformer language model from scratch in
Rust using only [`ndarray`](https://docs.rs/ndarray) as a BLAS wrapper and
[`asmjson`](https://crates.io/crates/asmjson) — a hand-written
AVX-512/SWAR JSON parser — for all configuration and weight file parsing.

No `candle`, no `tch`, no `ort`.  Every matmul, every attention head, every
RoPE rotation is written explicitly so you can read it and understand it.

The model it runs is **Qwen2.5-0.5B**: 24 transformer layers, 896-dimensional
hidden state, 14 query heads / 2 key-value heads (grouped-query attention),
~500 million BF16 parameters, vocabulary of 151 936 tokens.

```
Prompt: "The capital of France is"
Output: " Paris. It is the largest city in France and ..."
~17 tokens/sec on a single CPU core
```

---

## From prompt to token: the full journey

This section walks through every step of the inference pipeline using the
prompt `"The capital of France is"`.

### 1 — Tokenisation (`src/tokenizer.rs`)

The raw text is split into sub-word tokens by a byte-level BPE tokenizer
loaded from `tokenizer.json`.

```
"The capital of France is"
    ↓  pre-tokenize (GPT-2 regex: words, contractions, punctuation)
["The", " capital", " of", " France", " is"]
    ↓  each chunk: bytes → byte-tokens → BPE merge loop
[791, 6864, 315, 9625, 374]      ← token ids
```

Special tokens such as `<|im_start|>` and `<|endoftext|>` are recognised
before BPE and emitted directly as their assigned ids.

`asmjson` is used to parse `tokenizer.json` (7 MB) during loading.
The CPUID-dispatched entry point selects AVX-512BW assembly or the portable
SWAR path automatically.

### 2 — Embedding lookup (`src/model.rs`)

Each token id is a row index into the **embedding matrix**
`[vocab=151936, hidden=896]`.  Five ids → five 896-dimensional vectors
stacked into an array `[5, 896]`.

```
[791, 6864, 315, 9625, 374]
    ↓  embed_tokens.weight[id, :]
[[...896 floats for "The"...],
 [...896 floats for " capital"...],
 ...
 [...896 floats for " is"...]]     shape [5, 896]
```

### 3 — 24 transformer blocks (`src/model.rs`, `src/attention.rs`, `src/ffn.rs`)

Each of the 24 layers applies the same "look then update" pattern:

```
x  (shape [5, 896])
│
├─ RMS Norm ──────────────────────────────────────────── src/norm.rs
│      rsqrt = 1 / sqrt(mean(x²) + ε)
│      x_norm = x · rsqrt · weight
│
├─ Grouped-Query Attention ──────────────────────────── src/attention.rs
│
│   For each of the 5 positions (and reusing cached positions on decode):
│
│   Q = x_norm · Wq  +  bq     [5, 14*64]  — 14 query heads
│   K = x_norm · Wk  +  bk     [5,  2*64]  — 2 key/value heads
│   V = x_norm · Wv  +  bv     [5,  2*64]
│
│   ── RoPE (Rotary Position Embedding) ──────────────── src/rope.rs
│   Rotating head[i] and head[i+half] by the angle for its position
│   encodes position directly into Q and K without a separate embedding.
│   Qwen2 uses the half-split convention (not LLaMA-1's interleaved pairs).
│
│   ── Grouped-query attention (GQA) ─────────────────
│   Each K/V head is shared by 7 Q heads (ratio = 14/2).
│   For head h:
│     scores = Q_h · K_hkv^T / sqrt(64)        [5, seq_len]
│     weights = softmax(scores)
│     out_h  = weights · V_hkv                  [5, 64]
│   Heads concatenated → [5, 896], projected by Wo → [5, 896]
│
│   residual add: x = x + attention_out
│
├─ RMS Norm
│
├─ SwiGLU Feed-Forward Network ──────────────────────── src/ffn.rs
│
│   gate  = x_norm · W_gate   [5, 4864]
│   up    = x_norm · W_up     [5, 4864]
│   h     = silu(gate) ⊙ up   element-wise: silu(z) = z / (1 + exp(−z))
│   out   = h · W_down         [5, 896]
│
│   residual add: x = x + ffn_out
│
└─ repeat ×24
```

### 4 — Final norm + language-model head

After all 24 layers a final RMS Norm is applied, then the last hidden state
(the row for the final token `" is"`) is dot-producted with the **tied**
embedding matrix (lm_head shares weights with embed_tokens):

```
x[-1]  (shape [896])
    ↓  · embed_tokens.weight^T
logits  (shape [151936])   ← one score per vocabulary token
```

### 5 — Sampling (`src/sample.rs`)

The logits for the next token are converted to a probability distribution and
sampled with **top-p (nucleus) sampling**:

```
logits [151936]
    ↓  divide by temperature (0.1) — sharpens the distribution
    ↓  softmax
    ↓  sort descending, keep tokens until cumulative prob ≥ top_p (0.9)
    ↓  sample from the nucleus with a simple Xorshift32 RNG
→  token id 12366  ("Paris")
    ↓  decode
→  " Paris"
```

The sampled token is appended to the running sequence, its key/value vectors
are saved in the **KV cache**, and `forward_next()` is called with just the
single new token — reusing all previous K/V pairs instead of recomputing them.
This continues until `<|endoftext|>` (id 151643) is produced or
`max_new_tokens` is reached.

---

## Project layout

| File | Responsibility |
|---|---|
| `src/tokenizer.rs` | BPE tokenizer; loads `tokenizer.json` with `asmjson` |
| `src/safetensors.rs` | Memory-maps `.safetensors`; decodes BF16/F16 → f32 |
| `src/config.rs` | Reads `config.json`; exposes `head_dim()`, `gqa_ratio()` |
| `src/norm.rs` | RMS normalisation |
| `src/rope.rs` | Rotary position embeddings (half-split convention) |
| `src/attention.rs` | Grouped-query attention + KV cache |
| `src/ffn.rs` | SwiGLU feed-forward network |
| `src/model.rs` | `QwenModel`: loads tensors, wires blocks, `forward()` / `forward_next()` |
| `src/sample.rs` | Top-p sampling, generation loop, `Xorshift32` RNG |
| `src/main.rs` | End-to-end pipeline |

---

## Dependencies

| Crate | Why |
|---|---|
| [`ndarray`](https://docs.rs/ndarray) `0.16` | All tensor storage and matrix multiplication |
| [`asmjson`](https://crates.io/crates/asmjson) `0.2` | JSON parsing for tokenizer, safetensors header, config |
| [`memmap2`](https://docs.rs/memmap2) `0.9` | Zero-copy memory mapping of the weight file |
| [`fancy-regex`](https://docs.rs/fancy-regex) `0.14` | GPT-2 pre-tokenize regex (requires lookahead) |

No deep-learning framework, no Python bindings, no GPU runtime.

---

## Running

Requires the model weights in `models/Qwen2.5-0.5B/` (download with
ModelScope or Hugging Face):

```sh
cargo run --release
```

Expected output (single CPU core, ~17 tok/sec):

```
Loading tokenizer … ok  (151643 vocab, 151387 merges, 293 special tokens)
Loading weights  … ok  (290 tensors, 0.3s)
Config: hidden=896 layers=24 heads=14q/2kv head_dim=64 vocab=151936
Building model  … ok  (0.4s)

Prompt: "The capital of France is"
Generating up to 64 tokens  (top_p=0.9, temp=0.1)…

 Paris. It is the largest city in France and ...
```

## Tests

```sh
cargo test          # 34 unit tests (1 slow real-weights test ignored)
cargo test -- --include-ignored   # includes the full forward-pass test
```

---

## Key lessons learned

- **RoPE pairing convention matters.** Qwen2 pairs dimensions `(i, i+half)`
  (half-split); LLaMA-1 uses adjacent pairs `(2i, 2i+1)`.  Getting this wrong
  produces fluent but incoherent output — everything compiles and runs, the
  model just generates word salad.

- **BF16 is just the top 16 bits of an f32.** No lookup table needed:
  `f32::from_bits((bits as u32) << 16)`.

- **GQA saves memory without hurting quality much.** Sharing each K/V head
  across 7 Q heads cuts the KV cache to 2/14 of what full MHA would need.

- **Tied embeddings halve one parameter block.** `lm_head` and `embed_tokens`
  share the same `[vocab, hidden]` matrix; the safetensors file has only one
  copy.
