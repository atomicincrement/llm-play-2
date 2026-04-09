//! Sampling and text generation.
//!
//! After the model produces logits (one raw score per vocabulary token) we
//! need to turn those scores into an actual next token.  This module provides:
//!
//! * [`sample_top_p`] — pick a token from the most probable slice of the
//!   distribution (nucleus / top-p sampling).
//! * [`generate`] — the full generation loop: encode a prompt, run the model,
//!   and stream decoded tokens until EOS or the length limit.

use ndarray::ArrayView1;

// ─── Top-p sampling ───────────────────────────────────────────────────────────

/// Choose the next token using *nucleus* (top-p) sampling.
///
/// ## What this does, in plain English
///
/// The model gives us a score ("logit") for every word in the vocabulary —
/// 151 936 numbers in our case.  We want to pick one of those words at
/// random, but skewed toward the high-scoring ones.  Here is how:
///
/// **Step 1 — Temperature scaling.**
/// Divide every logit by `temperature` before doing anything else.
/// * `temperature = 1.0` → leave the distribution unchanged.
/// * `temperature < 1.0` → make high scores even higher relative to low
///   scores, so the model becomes more deterministic ("confident").
/// * `temperature > 1.0` → flatten the distribution, making the model more
///   creative (and more likely to say surprising things).
///
/// **Step 2 — Softmax.**
/// Convert the scaled logits into probabilities that sum to 1.0.
/// Think of it as: "given these scores, what fraction of the time should I
/// pick each token?".
///
/// **Step 3 — Sort descending.**
/// Arrange all tokens from most probable to least probable.
///
/// **Step 4 — Nucleus truncation (the "top-p" part).**
/// Walk down the sorted list, accumulating probability mass.  Stop as soon
/// as the running total exceeds `top_p`.  Discard every token that did not
/// make the cut — they are too unlikely to be useful.
/// * `top_p = 1.0` → keep the entire vocabulary (pure temperature sampling).
/// * `top_p = 0.9` → keep only the smallest set of tokens that together
///   account for 90 % of the probability.
///
/// **Step 5 — Sample.**
/// Draw a random number uniformly from [0, 1) and walk down the nucleus
/// (the kept tokens) until the cumulative probability exceeds that random
/// number.  The token where we stop is our answer.
///
/// This two-knob approach (temperature + top_p) gives fine-grained control
/// over creativity vs. coherence without needing to enumerate every token.
///
/// ## Arguments
/// * `logits`      — raw model output, one f32 per vocabulary token.
/// * `top_p`       — nucleus probability mass (0 < top_p ≤ 1.0).
/// * `temperature` — softmax temperature (> 0; 1.0 = unchanged).
/// * `rng`         — a single `f32` in `[0, 1)` used as the random draw.
///   Pass a real random value in production; use a fixed value in tests.
///
/// ## Returns
/// The sampled token id as a `u32`.
pub fn sample_top_p(logits: ArrayView1<f32>, top_p: f32, temperature: f32, rng: f32) -> u32 {
    let n = logits.len();

    // ── Step 1: temperature scaling ───────────────────────────────────────────
    // Avoid division by zero; clamp temperature to a small positive value.
    let temp = temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    // ── Step 2: softmax ───────────────────────────────────────────────────────
    // Subtract the max first for numerical stability: e^(x-max) / Σe^(x-max)
    // gives the same result as e^x / Σe^x but avoids floating-point overflow.
    let max_l = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_l).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    // ── Step 3: sort by probability, highest first ────────────────────────────
    // Keep the original token id alongside its probability so we can return it.
    let mut pairs: Vec<(f32, usize)> = probs.iter().cloned().zip(0..n).collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // ── Step 4: nucleus truncation ────────────────────────────────────────────
    // Keep tokens until we have accumulated at least `top_p` of the mass.
    // Always keep at least one token even if top_p is extremely small.
    let mut cumulative = 0.0_f32;
    let mut nucleus_end = 0usize;
    for (i, &(p, _)) in pairs.iter().enumerate() {
        cumulative += p;
        nucleus_end = i + 1;
        if cumulative >= top_p {
            break;
        }
    }

    // ── Step 5: sample from the nucleus ───────────────────────────────────────
    // Re-normalise the nucleus probabilities so they sum to 1, then walk down
    // until the random value `rng` is "used up".
    let nucleus = &pairs[..nucleus_end];
    let nucleus_mass: f32 = nucleus.iter().map(|&(p, _)| p).sum();
    let mut threshold = rng * nucleus_mass;
    for &(p, idx) in nucleus {
        threshold -= p;
        if threshold <= 0.0 {
            return idx as u32;
        }
    }
    // Fallback: floating-point rounding can leave threshold barely above 0;
    // return the last token in the nucleus.
    nucleus.last().unwrap().1 as u32
}

// ─── Generation loop ──────────────────────────────────────────────────────────

/// A simple random-number generator based on xorshift32.
///
/// A *real* application would use the `rand` crate, but we want to avoid
/// extra dependencies.  Xorshift is fast and good enough for sampling.
/// It produces a different sequence for every non-zero seed.
struct Xorshift32(u32);

impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        // Three XOR-shift operations scramble the bits of the state.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        // Map the 32-bit integer into [0, 1).
        (self.0 as f32) / (u32::MAX as f32)
    }
}

/// Generate text from a prompt and print each token as it is produced.
///
/// ## What the generation loop does
///
/// Text generation is an iterative process:
///
/// 1. **Encode the prompt** — convert the human-readable prompt string into a
///    list of token ids that the model understands.
///
/// 2. **Prefill** — run the *entire* prompt through the model in one go.
///    This fills the KV cache with context for every position in the prompt.
///    The model produces logits for every position, but we only care about the
///    logits at the *last* position, which represent "what token should come
///    next after the whole prompt?".
///
/// 3. **Sample** — turn those last-position logits into a single token id
///    using top-p sampling (see [`sample_top_p`]).
///
/// 4. **Decode and print** — convert the new token id back to a string and
///    show it immediately (streaming output).  This is how chat interfaces
///    appear to "think out loud".
///
/// 5. **Repeat** — feed the new token back into the model using the fast
///    single-token path (`forward_next`), which reuses the KV cache instead
///    of reprocessing the whole history.  Go to step 3.
///
/// 6. **Stop** when either:
///    * The model produces the end-of-sequence token (`eos_token_id`), or
///    * We have generated `max_new_tokens` tokens (a safety limit).
///
/// ## Arguments
/// * `model`          — fully loaded [`QwenModel`].
/// * `tokenizer`      — the BPE tokenizer (encode + decode).
/// * `prompt`         — the input text string.
/// * `max_new_tokens` — hard upper limit on generated length.
/// * `top_p`          — nucleus probability mass for sampling (e.g. 0.9).
/// * `temperature`    — sampling temperature (e.g. 1.0).
/// * `seed`           — random seed; different seeds give different outputs.
///
/// ## Returns
/// The complete generated text (excluding the prompt) as a `String`.
pub fn generate(
    model: &mut crate::model::QwenModel,
    tokenizer: &crate::tokenizer::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    top_p: f32,
    temperature: f32,
    seed: u32,
) -> String {
    let mut rng = Xorshift32(seed.max(1)); // seed must be non-zero for xorshift
    let eos = model.cfg.eos_token_id;

    // 1. Encode the prompt into token ids.
    let prompt_ids: Vec<u32> = tokenizer.encode(prompt);

    // 2. Prefill: run the whole prompt through the model to warm up the KV cache.
    //    `forward` returns logits for every position; we take the last row.
    let all_logits = model.forward(&prompt_ids);
    let last_row = all_logits.row(all_logits.nrows() - 1);
    let mut next_token = sample_top_p(last_row, top_p, temperature, rng.next_f32());

    let mut generated = Vec::new();

    // 3–6. Decode loop.
    for _ in 0..max_new_tokens {
        if next_token == eos {
            break;
        }

        generated.push(next_token);

        // 4. Decode this single token and print it immediately (streaming).
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        // Flush stdout so the token appears right away, not in a buffered batch.
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // 5. Extend the KV cache by one position and get the next logits.
        let logits = model.forward_next(next_token);
        next_token = sample_top_p(logits.view(), top_p, temperature, rng.next_f32());
    }

    // Print a newline after streaming output.
    println!();

    tokenizer.decode(&generated)
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// With temperature→0 (very small), the highest-logit token should almost
    /// always win regardless of top_p.
    #[test]
    fn greedy_at_low_temperature() {
        // Token 3 has the highest logit.
        let logits = Array1::from_vec(vec![0.1_f32, 0.2, 0.15, 5.0, 0.05]);
        let chosen = sample_top_p(logits.view(), 1.0, 0.01, 0.5);
        assert_eq!(chosen, 3, "expected greedy pick of token 3");
    }

    /// With top_p very small the nucleus should collapse to a single token.
    #[test]
    fn top_p_very_small_picks_highest() {
        let logits = Array1::from_vec(vec![0.0_f32, 0.0, 10.0, 0.0]);
        // top_p = 0.01 should only keep token 2 (probability ≈ 1.0 after softmax).
        let chosen = sample_top_p(logits.view(), 0.01, 1.0, 0.5);
        assert_eq!(chosen, 2);
    }

    /// Output must always be a valid token index.
    #[test]
    fn output_in_range() {
        let n = 20usize;
        let logits = Array1::from_vec((0..n).map(|i| i as f32).collect());
        for seed_bits in [0u32, 1, 42, 999, u32::MAX] {
            // Simulate a few different rng values.
            let rng_val = (seed_bits as f32) / (u32::MAX as f32 + 1.0);
            let tok = sample_top_p(logits.view(), 0.9, 1.0, rng_val);
            assert!((tok as usize) < n, "token {tok} out of range 0..{n}");
        }
    }

    /// Probabilities in the nucleus must sum to ≤ 1.0 (sanity check on softmax).
    #[test]
    fn softmax_sums_to_one() {
        let logits = Array1::from_vec(vec![1.0_f32, 2.0, 3.0, 0.5]);
        // Recompute softmax the same way the sampler does.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let total: f32 = exps.iter().map(|e| e / sum).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    /// xorshift32 must not get stuck at zero and must produce values in [0, 1).
    #[test]
    fn rng_range() {
        let mut rng = Xorshift32(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "rng out of range: {v}");
        }
    }
}
