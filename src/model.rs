//! Transformer block and full Qwen2.5 model.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};

use crate::{
    attention::{AttentionWeights, KvCache, attention},
    config::ModelConfig,
    ffn::{FfnWeights, feed_forward},
    norm::rms_norm,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Pull a named tensor from the weight dictionary and return it as a 2-D matrix.
///
/// Every weight in the model is stored in a flat `HashMap<name, ArrayD<f32>>`.
/// This helper looks up `key`, returns an error if it is missing, and then
/// reshapes the dynamic-rank array into a concrete `Array2<f32>`.
fn get2(t: &HashMap<String, ArrayD<f32>>, key: &str) -> Result<Array2<f32>, String> {
    t.get(key)
        .ok_or_else(|| format!("missing tensor: {key}"))?
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|e| format!("tensor {key} is not 2D: {e}"))
}

/// Pull a named tensor from the weight dictionary and return it as a 1-D vector.
///
/// Same as `get2` but for bias and norm-weight tensors that are rank-1.
/// Typical use: `get1(tensors, "model.norm.weight")` → shape `[896]`.
fn get1(t: &HashMap<String, ArrayD<f32>>, key: &str) -> Result<Array1<f32>, String> {
    t.get(key)
        .ok_or_else(|| format!("missing tensor: {key}"))?
        .clone()
        .into_dimensionality::<Ix1>()
        .map_err(|e| format!("tensor {key} is not 1D: {e}"))
}

// ─── Transformer block ────────────────────────────────────────────────────────

/// Everything needed to run one transformer layer.
///
/// A transformer layer is the repeating unit stacked 24 times in Qwen2.5-0.5B.
/// Each block holds four groups of learned parameters:
/// * `attn_norm`  — RMS-norm weights applied *before* attention.
/// * `attn`       — the seven attention weight matrices (Q, K, V, O projections).
/// * `ffn_norm`   — RMS-norm weights applied *before* the feed-forward network.
/// * `ffn`        — the three FFN matrices (gate, up, down).
pub struct TransformerBlock {
    /// RMS norm weights applied before the attention sub-layer.
    attn_norm: Array1<f32>,
    /// All seven attention weight tensors for this layer.
    attn: AttentionWeights,
    /// RMS norm weights applied before the FFN sub-layer.
    ffn_norm: Array1<f32>,
    /// The three FFN weight matrices for this layer.
    ffn: FfnWeights,
}

impl TransformerBlock {
    /// Load one transformer layer from the weight dictionary.
    fn from_tensors(t: &HashMap<String, ArrayD<f32>>, layer: usize) -> Result<Self, String> {
        let p = |s: &str| format!("model.layers.{layer}.{s}");

        // Derive bias sizes from the corresponding weight shapes rather than
        // loading a non-existent tensor — Qwen2 has Q/K/V biases but no O bias.
        let q_dim = get2(t, &p("self_attn.q_proj.weight"))?.nrows();
        let k_dim = get2(t, &p("self_attn.k_proj.weight"))?.nrows();
        let v_dim = get2(t, &p("self_attn.v_proj.weight"))?.nrows();

        Ok(Self {
            attn_norm: get1(t, &p("input_layernorm.weight"))?,
            attn: AttentionWeights {
                wq: get2(t, &p("self_attn.q_proj.weight"))?,
                bq: get1(t, &p("self_attn.q_proj.bias")).unwrap_or_else(|_| Array1::zeros(q_dim)),
                wk: get2(t, &p("self_attn.k_proj.weight"))?,
                bk: get1(t, &p("self_attn.k_proj.bias")).unwrap_or_else(|_| Array1::zeros(k_dim)),
                wv: get2(t, &p("self_attn.v_proj.weight"))?,
                bv: get1(t, &p("self_attn.v_proj.bias")).unwrap_or_else(|_| Array1::zeros(v_dim)),
                wo: get2(t, &p("self_attn.o_proj.weight"))?,
            },
            ffn_norm: get1(t, &p("post_attention_layernorm.weight"))?,
            ffn: FfnWeights {
                w_gate: get2(t, &p("mlp.gate_proj.weight"))?,
                w_up: get2(t, &p("mlp.up_proj.weight"))?,
                w_down: get2(t, &p("mlp.down_proj.weight"))?,
            },
        })
    }

    /// Run one token through this transformer layer.
    ///
    /// ## What this does, step by step
    ///
    /// Think of the hidden state `x` as a running summary — a compact vector
    /// of numbers that captures everything the model needs to know about the
    /// current token *in the context of all preceding tokens*.
    ///
    /// A transformer layer refines that summary in two sequential passes, each
    /// following the same **"look, then update"** recipe:
    ///
    /// **Pass 1 — Token attends to context (self-attention)**
    /// 1. *Normalise* `x` with RMS norm so the numbers are well-scaled.
    /// 2. *Attention*: the token "looks" at every token that came before it
    ///    (via the KV cache) and mixes in relevant information — e.g. a pronoun
    ///    weighs heavily on its antecedent noun.
    /// 3. *Residual add*: add the original `x` back in.  This skip-connection
    ///    lets the layer make a small *correction* rather than having to
    ///    reconstruct the entire signal from scratch, which is much easier to
    ///    learn.
    ///
    /// **Pass 2 — Token processes what it learnt (feed-forward)**
    /// 4. *Normalise* the updated `x` again.
    /// 5. *FFN (SwiGLU)*: two learned linear projections with a non-linear gate
    ///    in between.  This is where the model can "think through" the attended
    ///    context with more expressive, position-wise computation.
    /// 6. *Residual add*: same skip-connection as before.
    ///
    /// After 24 such layers the hidden state is rich enough to predict the
    /// next token.
    fn forward(
        &self,
        x: Array1<f32>,
        pos: usize,
        kv_cache: &mut KvCache,
        cfg: &ModelConfig,
    ) -> Array1<f32> {
        // ── Attention sub-layer ───────────────────────────────────────────────
        let residual = x.clone();
        let normed = rms_norm(x.view(), self.attn_norm.view(), cfg.rms_norm_eps);
        let attn_out = attention(normed.view(), &self.attn, pos, kv_cache, cfg);
        let x = attn_out + residual;

        // ── FFN sub-layer ─────────────────────────────────────────────────────
        let residual = x.clone();
        let normed = rms_norm(x.view(), self.ffn_norm.view(), cfg.rms_norm_eps);
        let ffn_out = feed_forward(normed.view(), &self.ffn);
        ffn_out + residual
    }
}

// ─── Full model ───────────────────────────────────────────────────────────────

/// The complete Qwen2.5 transformer model.
pub struct QwenModel {
    /// Token embedding table, shape `[vocab_size, hidden_size]`.
    /// Each row is the initial hidden state for that token id.
    embed: Array2<f32>,
    /// The 24 transformer layers.
    blocks: Vec<TransformerBlock>,
    /// Final RMS norm applied after the last layer.
    final_norm: Array1<f32>,
    /// Output projection to vocabulary logits, shape `[vocab_size, hidden_size]`.
    /// Tied to the embedding table in Qwen2.5 (same weights, no extra memory).
    lm_head: Array2<f32>,
    /// One KV cache per layer, grown position by position during a forward pass.
    kv_caches: Vec<KvCache>,
    /// Model hyperparameters.
    pub cfg: ModelConfig,
}

impl QwenModel {
    /// Build the model from a flat tensor dictionary (from `load_safetensors`)
    /// and a parsed config.
    pub fn from_tensors(
        tensors: &HashMap<String, ArrayD<f32>>,
        cfg: ModelConfig,
    ) -> Result<Self, String> {
        let embed = get2(tensors, "model.embed_tokens.weight")?;
        // Qwen2.5 uses tied embeddings: the lm_head shares weights with the
        // embedding table, so no separate "lm_head.weight" tensor exists.
        let lm_head = embed.clone();
        let final_norm = get1(tensors, "model.norm.weight")?;

        let n_layers = cfg.num_hidden_layers;
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(TransformerBlock::from_tensors(tensors, i)?);
        }
        let kv_caches = (0..n_layers).map(|_| KvCache::new()).collect();

        Ok(Self {
            embed,
            blocks,
            final_norm,
            lm_head,
            kv_caches,
            cfg,
        })
    }

    /// Run the model on a sequence of token ids and return logits for every
    /// position.
    ///
    /// ## What the full model does, explained simply
    ///
    /// Imagine reading a sentence one word at a time and continuously updating
    /// a mental notepad.  That is exactly what the model does:
    ///
    /// **Embedding** — Convert each integer token id into its starting hidden
    /// state by looking it up in the embedding table.  Think of this as
    /// "what does this word mean in isolation?".
    ///
    /// **24 transformer layers** — Each layer takes the hidden state and makes
    /// it richer by (a) letting the token attend to everything said so far and
    /// (b) running it through a learned non-linear transformation.  With each
    /// layer the representation becomes more contextualised: early layers notice
    /// syntax, later layers reason about meaning.
    ///
    /// **Final norm** — Scale the hidden state one last time so the numbers are
    /// well-behaved before the final projection.
    ///
    /// **Logits** — Multiply the final hidden state by the vocabulary matrix to
    /// get one score per token in the vocabulary.  Higher score = the model
    /// thinks that token is more likely to come next.  The scores are *not* yet
    /// probabilities; apply softmax (or sample with temperature/top-p) to
    /// convert them.
    ///
    /// The KV cache is cleared and rebuilt from scratch on each call, so this
    /// function is suitable for full-sequence prefill.  For single-token
    /// decoding after a prefill use `forward_next`.
    ///
    /// ## Returns
    /// `Array2<f32>` of shape `[seq_len, vocab_size]` — one row of logits per
    /// input token.
    pub fn forward(&mut self, token_ids: &[u32]) -> Array2<f32> {
        // Start fresh — clear every layer's KV cache.
        for cache in &mut self.kv_caches {
            *cache = KvCache::new();
        }

        let vocab = self.lm_head.nrows();
        let mut all_logits = Array2::zeros((token_ids.len(), vocab));

        for (pos, &tok_id) in token_ids.iter().enumerate() {
            // 1. Embedding lookup: grab the row for this token.
            let mut x = self.embed.row(tok_id as usize).to_owned();

            // 2. Run through all transformer layers sequentially.
            for (block, kv) in self.blocks.iter().zip(self.kv_caches.iter_mut()) {
                x = block.forward(x, pos, kv, &self.cfg);
            }

            // 3. Final RMS norm.
            let x = rms_norm(x.view(), self.final_norm.view(), self.cfg.rms_norm_eps);

            // 4. Project to logits: lm_head [vocab, hidden] · x [hidden] = [vocab].
            all_logits.row_mut(pos).assign(&self.lm_head.dot(&x));
        }

        all_logits
    }

    /// Decode one additional token, reusing the existing KV cache.
    ///
    /// Call `forward` first to prefill the cache, then call this for each
    /// subsequent generated token.  Returns logits of shape `[vocab_size]`.
    pub fn forward_next(&mut self, token_id: u32) -> Array1<f32> {
        let pos = self.kv_caches[0].len(); // tokens already in the cache

        let mut x = self.embed.row(token_id as usize).to_owned();

        for (block, kv) in self.blocks.iter().zip(self.kv_caches.iter_mut()) {
            x = block.forward(x, pos, kv, &self.cfg);
        }

        let x = rms_norm(x.view(), self.final_norm.view(), self.cfg.rms_norm_eps);
        self.lm_head.dot(&x)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::load_safetensors;
    use std::path::Path;

    /// Smoke-test: load real weights and run two tokens through the model.
    /// Checks that the output shape is correct and logits are finite.
    #[test]
    #[ignore] // slow — requires the model files on disk
    fn forward_shape_and_finite() {
        let base = Path::new("models/Qwen2.5-0.5B");
        let tensors = load_safetensors(&base.join("model.safetensors")).unwrap();
        let cfg = ModelConfig::from_file(&base.join("config.json")).unwrap();

        let vocab = cfg.vocab_size;
        let mut model = QwenModel::from_tensors(&tensors, cfg).unwrap();
        let logits = model.forward(&[1u32, 2u32]); // two arbitrary token ids

        assert_eq!(logits.shape(), &[2, vocab]);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits contain NaN or Inf"
        );
    }

    /// Unit-test: build a tiny 1-layer model with random weights and verify
    /// the output shape is `[seq_len, vocab_size]`.
    #[test]
    fn tiny_model_output_shape() {
        use ndarray::Array;

        let hidden = 8usize;
        let intermediate = 16usize;
        let vocab = 32usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = hidden / n_heads; // 4

        let cfg = ModelConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_hidden_layers: 1,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv_heads,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            vocab_size: vocab,
            max_position_embeddings: 512,
            bos_token_id: 0,
            eos_token_id: 1,
            sliding_window: 0,
            use_sliding_window: false,
        };

        let q_dim = n_heads * head_dim; // 8
        let kv_dim = n_kv_heads * head_dim; // 4

        let block = TransformerBlock {
            attn_norm: Array1::ones(hidden),
            attn: AttentionWeights {
                wq: Array2::eye(q_dim),
                bq: Array1::zeros(q_dim),
                wk: Array::from_elem((kv_dim, hidden), 0.0),
                bk: Array1::zeros(kv_dim),
                wv: Array::from_elem((kv_dim, hidden), 0.0),
                bv: Array1::zeros(kv_dim),
                wo: Array2::eye(hidden),
            },
            ffn_norm: Array1::ones(hidden),
            ffn: FfnWeights {
                w_gate: Array2::zeros((intermediate, hidden)),
                w_up: Array2::zeros((intermediate, hidden)),
                w_down: Array2::zeros((hidden, intermediate)),
            },
        };

        let mut model = QwenModel {
            embed: Array2::zeros((vocab, hidden)),
            blocks: vec![block],
            final_norm: Array1::ones(hidden),
            lm_head: Array2::zeros((vocab, hidden)),
            kv_caches: vec![KvCache::new()],
            cfg,
        };

        let logits = model.forward(&[0u32, 1u32, 2u32]);
        assert_eq!(logits.shape(), &[3, vocab]);
    }
}
