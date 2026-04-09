//! Model configuration — loads `config.json` with `asmjson`.
//!
//! The struct is model-agnostic; field names follow the HuggingFace convention
//! shared by Qwen2, LLaMA, Mistral, etc.

use std::{fs, path::Path};

use asmjson::{parse_to_dom, JsonRef};

// ─── Public struct ────────────────────────────────────────────────────────────

/// All the numbers that describe how a transformer model is built.
///
/// Every model ships a `config.json` file that says things like "I have 24
/// layers", "each layer has 14 attention heads", "my vocabulary has 151936
/// tokens", etc.  `ModelConfig` reads those numbers so the rest of the code
/// can size its arrays correctly at runtime.
///
/// ## Example
///
/// ```text
/// let cfg = ModelConfig::from_file(Path::new("models/Qwen2.5-0.5B/config.json")).unwrap();
/// assert_eq!(cfg.hidden_size, 896);   // 896-dimensional hidden state
/// assert_eq!(cfg.head_dim(), 64);     // each of the 14 heads handles 64 dims
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelConfig {
    /// Embedding / hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (MLP) dimension.
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (< `num_attention_heads` for GQA).
    pub num_key_value_heads: usize,
    /// RMS-norm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base theta.
    pub rope_theta: f32,
    /// Vocabulary size (number of token ids).
    pub vocab_size: usize,
    /// Maximum sequence length the model was trained with.
    pub max_position_embeddings: usize,
    /// Beginning-of-sequence token id.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
    /// Sliding window size (0 = disabled / full attention).
    pub sliding_window: usize,
    /// Whether to use sliding-window attention.
    pub use_sliding_window: bool,
}

#[allow(dead_code)]
impl ModelConfig {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Load from a `config.json` file on disk.
    ///
    /// Reads the whole file into memory, parses it as JSON, then extracts
    /// the fields that the model needs.  Returns an error string if the file
    /// cannot be read or a required field is missing.
    ///
    /// ## Example
    ///
    /// ```text
    /// let cfg = ModelConfig::from_file(Path::new("models/Qwen2.5-0.5B/config.json"))?;
    /// println!("layers: {}", cfg.num_hidden_layers); // 24
    /// ```
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let src = fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
        Self::from_json(&src)
    }

    /// Parse directly from a JSON string.
    ///
    /// Useful in tests when you do not want to write a file to disk.
    ///
    /// ## Example
    ///
    /// ```text
    /// let cfg = ModelConfig::from_json(r#"{"hidden_size":128,...}"#)?;
    /// ```
    pub fn from_json(src: &str) -> Result<Self, String> {
        let tape = parse_to_dom(src, None).ok_or("JSON parse failed")?;
        let root = tape.root().ok_or("JSON: empty document")?;

        let get_usize = |key: &str| -> Result<usize, String> {
            root.get(key)
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("missing or non-integer field '{key}'"))
        };
        let get_u32 = |key: &str| -> Result<u32, String> {
            root.get(key)
                .as_u64()
                .map(|n| n as u32)
                .ok_or_else(|| format!("missing or non-integer field '{key}'"))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            root.get(key)
                .as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| format!("missing or non-numeric field '{key}'"))
        };
        let get_bool =
            |key: &str, default: bool| -> bool { root.get(key).as_bool().unwrap_or(default) };

        // `sliding_window` is optional in some configs.
        let sliding_window = root.get("sliding_window").as_u64().unwrap_or(0) as usize;

        Ok(ModelConfig {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            num_key_value_heads: get_usize("num_key_value_heads")?,
            rms_norm_eps: get_f32("rms_norm_eps")?,
            rope_theta: get_f32("rope_theta")?,
            vocab_size: get_usize("vocab_size")?,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            bos_token_id: get_u32("bos_token_id")?,
            eos_token_id: get_u32("eos_token_id")?,
            sliding_window,
            use_sliding_window: get_bool("use_sliding_window", false),
        })
    }

    // ── Derived helpers ───────────────────────────────────────────────────────

    /// Dimension of each attention head.
    ///
    /// The hidden state is split evenly across all query heads, so each head
    /// operates on `hidden_size / num_attention_heads` values.
    ///
    /// For Qwen2.5-0.5B: `896 / 14 = 64`.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// How many query heads share each key/value head (GQA ratio).
    ///
    /// In standard multi-head attention every query head has its own K and V.
    /// Grouped-query attention (GQA) saves memory by sharing one K/V head
    /// across several query heads.  This ratio tells you how many query heads
    /// each K/V head serves.
    ///
    /// For Qwen2.5-0.5B: `14 / 2 = 7` — each of the 2 K/V heads serves 7 Q heads.
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": 1000,
        "max_position_embeddings": 256,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;

    #[test]
    fn parse_minimal() {
        let cfg = ModelConfig::from_json(MINIMAL).unwrap();
        assert_eq!(cfg.hidden_size, 128);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim(), 32); // 128 / 4
        assert_eq!(cfg.gqa_ratio(), 2); // 4 / 2
        assert!(!cfg.use_sliding_window);
        assert_eq!(cfg.sliding_window, 0);
    }

    #[test]
    fn load_qwen_config() {
        let path = Path::new("models/Qwen2.5-0.5B/config.json");
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        let cfg = ModelConfig::from_file(path).unwrap();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.intermediate_size, 4864);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.head_dim(), 64); // 896 / 14
        assert_eq!(cfg.gqa_ratio(), 7); // 14 / 2
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        println!("{cfg:#?}");
    }
}
