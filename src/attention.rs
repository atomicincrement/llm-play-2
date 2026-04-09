//! Grouped-Query Attention (GQA) with KV cache.
//!
//! Operates in **decode mode**: one new token per call, attending to all
//! previous positions stored in the KV cache.
//!
//! # Layout (Qwen2.5 / Qwen2)
//! * `q_proj`: `[num_q_heads * head_dim,  hidden]` + bias `[num_q_heads * head_dim]`
//! * `k_proj`: `[num_kv_heads * head_dim, hidden]` + bias `[num_kv_heads * head_dim]`
//! * `v_proj`: `[num_kv_heads * head_dim, hidden]` + bias `[num_kv_heads * head_dim]`
//! * `o_proj`: `[hidden, num_q_heads * head_dim]`  (no bias)

use ndarray::{Array1, Array2, ArrayView1};

use crate::{config::ModelConfig, rope::apply_rope};

// ─── Weight bundle ────────────────────────────────────────────────────────────

/// The six learned matrices (and three bias vectors) that power one
/// self-attention layer.
///
/// ## How they are used
///
/// Given the incoming hidden state `x` (a single vector of length `hidden`):
///
/// * `wq` + `bq` project `x` into *query* space  — "what am I looking for?"
/// * `wk` + `bk` project `x` into *key*   space  — "what kind of thing am I?"
/// * `wv` + `bv` project `x` into *value* space  — "what information do I hold?"
/// * After attention is computed, `wo` projects the concatenated head outputs
///   back to the original `hidden` dimension.
///
/// In Qwen2.5-0.5B the shapes are:
/// * `wq`: `[896, 896]`  (`n_heads × head_dim` = 14 × 64 = 896 rows)
/// * `wk`: `[128, 896]`  (`n_kv × head_dim`    =  2 × 64 = 128 rows)
/// * `wv`: `[128, 896]`  (same as wk)
/// * `wo`: `[896, 896]`
pub struct AttentionWeights {
    /// `[num_q_heads * head_dim, hidden]`
    pub wq: Array2<f32>,
    /// `[num_q_heads * head_dim]` (optional: zero if model has no Q bias)
    pub bq: Array1<f32>,
    /// `[num_kv_heads * head_dim, hidden]`
    pub wk: Array2<f32>,
    /// `[num_kv_heads * head_dim]`
    pub bk: Array1<f32>,
    /// `[num_kv_heads * head_dim, hidden]`
    pub wv: Array2<f32>,
    /// `[num_kv_heads * head_dim]`
    pub bv: Array1<f32>,
    /// `[hidden, num_q_heads * head_dim]`
    pub wo: Array2<f32>,
}

// ─── KV cache ─────────────────────────────────────────────────────────────────

/// A memory of every token the model has already processed.
///
/// ## Why this exists
///
/// When generating text the model works one token at a time: it reads
/// everything so far and predicts the next word.  Naively, that would mean
/// re-processing the entire prompt from scratch for every new token —
/// extremely wasteful.
///
/// The KV cache is the solution.  During the attention step each token
/// produces two compact summaries of itself:
///
/// * **Key** — "what kind of information am I?"  Used by later tokens to
///   decide how much attention to pay to this one.
/// * **Value** — "what information do I carry?"  The actual content that
///   gets mixed in when another token attends to this one.
///
/// These two summaries are computed once and stored here.  When a new token
/// arrives it can simply *look up* the stored keys and values for every
/// previous position without recomputing them.  This makes decoding roughly
/// O(n) instead of O(n²) in wall-clock time.
///
/// ## Structure
///
/// There is one `KvCache` per transformer layer (the model has 24).  Each
/// cache grows by one entry every time a token is processed:
/// * `keys[i]`   — the key   summary for the token at position `i`, shape `[num_kv_heads, head_dim]`
/// * `values[i]` — the value summary for the token at position `i`, shape `[num_kv_heads, head_dim]`
///
/// When starting a new conversation the cache is cleared so stale context
/// from a previous prompt cannot bleed through.
#[derive(Default)]
pub struct KvCache {
    /// Past keys,   one `[num_kv_heads, head_dim]` array per position.
    pub keys: Vec<Array2<f32>>,
    /// Past values, one `[num_kv_heads, head_dim]` array per position.
    pub values: Vec<Array2<f32>>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

/// Compute single-token GQA, updating `kv_cache` in place.
///
/// # Arguments
/// * `x`        — input hidden state for the current token, shape `[hidden]`
/// * `w`        — weight bundle for this layer
/// * `pos`      — current token position (= `kv_cache.len()` before this call)
/// * `kv_cache` — mutable KV cache; this call appends one entry
/// * `cfg`      — model configuration
///
/// # Returns
/// Output hidden state, shape `[hidden]`.
pub fn attention(
    x: ArrayView1<f32>,
    w: &AttentionWeights,
    pos: usize,
    kv_cache: &mut KvCache,
    cfg: &ModelConfig,
) -> Array1<f32> {
    let head_dim = cfg.head_dim();
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let gqa = cfg.gqa_ratio(); // n_heads / n_kv
    let scale = (head_dim as f32).sqrt().recip();

    // ── 1. Linear projections ─────────────────────────────────────────────────
    // q: [n_heads * head_dim]
    let q_flat = w.wq.dot(&x) + &w.bq;
    // k, v: [n_kv * head_dim]
    let k_flat = w.wk.dot(&x) + &w.bk;
    let v_flat = w.wv.dot(&x) + &w.bv;

    // ── 2. Reshape into heads ─────────────────────────────────────────────────
    let mut q = q_flat
        .into_shape_with_order((n_heads, head_dim))
        .expect("q reshape");
    let mut k = k_flat
        .into_shape_with_order((n_kv, head_dim))
        .expect("k reshape");
    let v = v_flat
        .into_shape_with_order((n_kv, head_dim))
        .expect("v reshape");

    // ── 3. RoPE ───────────────────────────────────────────────────────────────
    apply_rope(q.view_mut(), k.view_mut(), pos, head_dim, cfg.rope_theta);

    // ── 4. Append to KV cache ─────────────────────────────────────────────────
    kv_cache.keys.push(k);
    kv_cache.values.push(v);

    let seq_len = kv_cache.len(); // includes the token we just added

    // ── 5. Per-head scaled dot-product attention ───────────────────────────────
    // Output accumulator: [n_heads, head_dim]
    let mut attn_out = Array2::<f32>::zeros((n_heads, head_dim));

    for h in 0..n_heads {
        let h_kv = h / gqa;
        let q_head = q.row(h); // [head_dim]

        // Gather K and V for this KV head across all cached positions.
        // k_mat: [seq_len, head_dim], v_mat: [seq_len, head_dim]
        let mut k_mat = Array2::<f32>::zeros((seq_len, head_dim));
        let mut v_mat = Array2::<f32>::zeros((seq_len, head_dim));
        for t in 0..seq_len {
            k_mat.row_mut(t).assign(&kv_cache.keys[t].row(h_kv));
            v_mat.row_mut(t).assign(&kv_cache.values[t].row(h_kv));
        }

        // scores = q · K^T / sqrt(head_dim)   [seq_len]
        let scores_raw = k_mat.dot(&q_head) * scale;

        // Causal mask: in decode mode every cached token is in the past,
        // so no masking needed — the cache only ever contains pos ≤ current.

        // Softmax over seq_len.
        let scores = softmax(scores_raw.view());

        // Weighted sum over V.
        // out_head = scores^T · v_mat   [head_dim]
        let out_head = v_mat.t().dot(&scores);
        attn_out.row_mut(h).assign(&out_head);
    }

    // ── 6. Flatten heads and project ──────────────────────────────────────────
    let flat = attn_out
        .into_shape_with_order(n_heads * head_dim)
        .expect("flatten");
    w.wo.dot(&flat)
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Convert a vector of raw scores into probabilities that sum to 1.
///
/// Each element is replaced by `exp(x_i - max) / Σ exp(x_j - max)`.
/// Subtracting `max` first is a numerical trick: it prevents `exp` from
/// overflowing to infinity while producing the same probabilities, because
/// the subtracted constant cancels in the fraction.
///
/// In attention, `softmax` turns the raw similarity scores into weights
/// ("how much attention should I pay to each past token?") that sum to 1.
fn softmax(x: ndarray::ArrayView1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Array1<f32> = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    fn test_cfg(n_heads: usize, n_kv: usize, hidden: usize) -> ModelConfig {
        let _head_dim = hidden / n_heads;
        ModelConfig {
            hidden_size: hidden,
            intermediate_size: hidden * 4,
            num_hidden_layers: 1,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            vocab_size: 256,
            max_position_embeddings: 64,
            bos_token_id: 0,
            eos_token_id: 1,
            sliding_window: 0,
            use_sliding_window: false,
        }
    }

    /// Build identity-like attention weights (no-op projection).
    fn identity_weights(cfg: &ModelConfig) -> AttentionWeights {
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
        AttentionWeights {
            wq: Array2::eye(h).slice(s![..q_dim, ..]).to_owned(),
            bq: Array1::zeros(q_dim),
            wk: Array2::eye(h).slice(s![..kv_dim, ..]).to_owned(),
            bk: Array1::zeros(kv_dim),
            wv: Array2::eye(h).slice(s![..kv_dim, ..]).to_owned(),
            bv: Array1::zeros(kv_dim),
            wo: Array2::eye(h).slice(s![..h, ..q_dim]).to_owned(),
        }
    }

    #[test]
    fn output_shape_mha() {
        // standard MHA: n_kv == n_heads
        let cfg = test_cfg(4, 4, 16);
        let w = identity_weights(&cfg);
        let mut cache = KvCache::new();
        let x = Array1::<f32>::ones(16);
        let out = attention(x.view(), &w, 0, &mut cache, &cfg);
        assert_eq!(out.len(), 16);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn output_shape_gqa() {
        // GQA: n_heads=4, n_kv=2, ratio=2
        let cfg = test_cfg(4, 2, 16);
        let w = identity_weights(&cfg);
        let mut cache = KvCache::new();
        let x = Array1::<f32>::ones(16);
        let out = attention(x.view(), &w, 0, &mut cache, &cfg);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn kv_cache_grows() {
        let cfg = test_cfg(2, 2, 8);
        let w = identity_weights(&cfg);
        let mut cache = KvCache::new();
        for pos in 0..5 {
            let x = Array1::<f32>::ones(8);
            attention(x.view(), &w, pos, &mut cache, &cfg);
        }
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = ndarray::array![1.0_f32, 2.0, 3.0, 4.0];
        let s = super::softmax(x.view());
        assert!((s.sum() - 1.0).abs() < 1e-6);
        for v in s.iter() {
            assert!(*v > 0.0);
        }
    }

    #[test]
    fn softmax_numerically_stable() {
        // Large values should not produce NaN.
        let x = ndarray::array![1e38_f32, 1e38, 0.0];
        let s = super::softmax(x.view());
        assert!(
            s.iter().all(|v| v.is_finite()),
            "softmax produced non-finite: {s:?}"
        );
    }
}
