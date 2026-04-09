//! Rotary Position Encoding (RoPE).
//!
//! Each head vector of dimension `d` is split into `d/2` adjacent pairs.
//! For token at position `pos` and frequency index `i ∈ [0, d/2)`:
//!
//! $$\theta_i = \text{base}^{-2i/d}$$
//! $$\phi_i = \text{pos} \cdot \theta_i$$
//!
//! The pair $(x_{2i},\, x_{2i+1})$ is rotated in-place:
//!
//! $$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
//! \begin{pmatrix} \cos\phi_i & -\sin\phi_i \\ \sin\phi_i & \cos\phi_i \end{pmatrix}
//! \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

use ndarray::ArrayViewMut2;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Stamp each attention head vector with the position of its token.
///
/// ## Why position encoding?
///
/// Attention by itself is "position-blind" — swapping two tokens produces
/// identical outputs.  RoPE fixes that by rotating each head vector by an
/// angle that depends on the token's position.  Two tokens at *similar*
/// positions end up with similar rotations, making the model aware of
/// relative distance.
///
/// ## How it works (the half-split convention)
///
/// Each head of dimension `d` is treated as `d/2` two-dimensional pairs:
/// element `i` is paired with element `i + d/2`.  Each pair is rotated by
/// the angle `φ_i = pos × base^(−2·i/d)`.  Pairs earlier in the head spin
/// fast (large angle per position); later pairs spin slowly — encoding both
/// coarse and fine-grained position simultaneously.
///
/// At `pos = 0` all angles are zero, so the rotation is the identity and
/// the vectors are unchanged.
///
/// # Arguments
/// * `xq` — query tensor, shape `[num_q_heads, head_dim]`; modified in place
/// * `xk` — key tensor,   shape `[num_kv_heads, head_dim]`; modified in place
/// * `pos`      — 0-based token position in the sequence
/// * `head_dim` — size of each head vector; must be even
/// * `base`     — RoPE base frequency (10 000 for LLaMA; 1 000 000 for Qwen2)
pub fn apply_rope(
    mut xq: ArrayViewMut2<f32>,
    mut xk: ArrayViewMut2<f32>,
    pos: usize,
    head_dim: usize,
    base: f32,
) {
    assert_eq!(head_dim % 2, 0, "head_dim must be even");

    let half = head_dim / 2;
    let pos_f = pos as f32;

    // Precompute cos/sin for every frequency index in this position.
    // θ_i = base^{-2i/d}  →  φ_i = pos · θ_i
    let freqs: Vec<(f32, f32)> = (0..half)
        .map(|i| {
            let theta_i = base.powf(-2.0 * i as f32 / head_dim as f32);
            let phi = pos_f * theta_i;
            (phi.cos(), phi.sin())
        })
        .collect();

    rotate_heads(xq.view_mut(), &freqs, half);
    rotate_heads(xk.view_mut(), &freqs, half);
}

/// Pre-calculate all the (cos, sin) rotation pairs for a whole sequence.
///
/// Rather than recomputing `cos(pos × θ_i)` and `sin(pos × θ_i)` every time
/// a token is processed, this function builds the complete table upfront for
/// positions `0..seq_len`.  The result is indexed as `table[pos][i]` where
/// `i` ranges over `0..head_dim/2`.
///
/// Useful for prefill (processing many tokens at once) when the same position
/// values would otherwise be recomputed many times across layers.
///
/// ## Example
///
/// ```text
/// let table = build_freq_table(128, 64, 1_000_000.0);
/// // table[3][0] == (cos(3 · base^0), sin(3 · base^0))
/// // table[3][1] == (cos(3 · base^(−2/64)), sin(...))
/// ```
#[allow(dead_code)]
pub fn build_freq_table(seq_len: usize, head_dim: usize, base: f32) -> Vec<Vec<(f32, f32)>> {
    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    let half = head_dim / 2;
    (0..seq_len)
        .map(|pos| {
            (0..half)
                .map(|i| {
                    let theta_i = base.powf(-2.0 * i as f32 / head_dim as f32);
                    let phi = pos as f32 * theta_i;
                    (phi.cos(), phi.sin())
                })
                .collect()
        })
        .collect()
}

/// Apply RoPE using an already-computed `(cos, sin)` row.
///
/// This is the low-level counterpart to [`apply_rope`]: instead of computing
/// the angles from scratch, you pass in a slice of `(cos(φ_i), sin(φ_i))`
/// pairs for a specific position (one row from `build_freq_table`).
///
/// Useful when you want to cache the frequency table and re-use it across
/// the 24 transformer layers without recomputing trig functions each time.
///
/// `freqs` must have length `head_dim / 2`.
#[allow(dead_code)]
pub fn apply_rope_precomputed(
    mut xq: ArrayViewMut2<f32>,
    mut xk: ArrayViewMut2<f32>,
    freqs: &[(f32, f32)],
) {
    let half = freqs.len();
    rotate_heads(xq.view_mut(), freqs, half);
    rotate_heads(xk.view_mut(), freqs, half);
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Rotate every row (head) of a 2-D tensor using the given `(cos, sin)` table.
///
/// Uses the **half-split** convention (matching HuggingFace / Qwen2 / LLaMA 2+):
/// frequency index `i` pairs element `head[i]` with element `head[i + half]`,
/// NOT adjacent elements `head[2i]` / `head[2i+1]` (that would be the older
/// "interleaved" LLaMA-1 convention).
///
/// The rotation for pair `(a, b) = (head[i], head[i+half])` is:
/// ```
///   head[i]        = a * cos - b * sin
///   head[i + half] = a * sin + b * cos
/// ```
fn rotate_heads(mut x: ArrayViewMut2<f32>, freqs: &[(f32, f32)], half: usize) {
    for mut head in x.rows_mut() {
        for i in 0..half {
            let (cos, sin) = freqs[i];
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f32::consts::PI;

    #[test]
    fn rotation_at_pos_zero_is_identity() {
        // At pos=0 all angles are 0: cos=1, sin=0 → no change.
        let orig = array![[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let mut xq = orig.clone();
        let mut xk = orig.clone();
        apply_rope(xq.view_mut(), xk.view_mut(), 0, 4, 10_000.0);
        for (a, b) in xq.iter().zip(orig.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "pos=0 should be identity, got {a} vs {b}"
            );
        }
        for (a, b) in xk.iter().zip(orig.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "pos=0 should be identity, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        // 2-D rotation is an isometry: ‖x'‖ = ‖x‖.
        let mut xq = array![[1.0_f32, 2.0, -1.0, 3.0]];
        let mut xk = array![[0.5_f32, -0.5, 2.0, 1.0]];
        let norm_q_before = xq.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_k_before = xk.iter().map(|v| v * v).sum::<f32>().sqrt();
        apply_rope(xq.view_mut(), xk.view_mut(), 7, 4, 10_000.0);
        let norm_q_after = xq.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_k_after = xk.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_q_before - norm_q_after).abs() < 1e-5,
            "Q norm changed"
        );
        assert!(
            (norm_k_before - norm_k_after).abs() < 1e-5,
            "K norm changed"
        );
    }

    #[test]
    fn rotation_by_known_angle() {
        // Use base=1.0 so θ_i = 1^(-0) = 1 for all i.
        // At pos=1, head_dim=2: φ_0 = 1·1 = 1 rad.
        // Rotate (1, 0) by 1 rad → (cos 1, sin 1).
        let mut xq = array![[1.0_f32, 0.0]];
        let mut xk = array![[1.0_f32, 0.0]];
        apply_rope(xq.view_mut(), xk.view_mut(), 1, 2, 1.0);
        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        assert!((xq[[0, 0]] - cos1).abs() < 1e-6, "x' = {}", xq[[0, 0]]);
        assert!((xq[[0, 1]] - sin1).abs() < 1e-6, "y' = {}", xq[[0, 1]]);
    }

    #[test]
    fn rotation_by_quarter_turn() {
        // base=1, head_dim=2, pos=π/2 → φ = π/2
        // (1, 0) rotated by 90° → (0, 1)
        let pos = (PI / 2.0).round() as usize; // 2, close enough for illustration
        // Instead, use a direct freq table with exact π/2.
        let freqs = vec![(0.0_f32, 1.0_f32)]; // cos(π/2)=0, sin(π/2)=1
        let mut xq = array![[1.0_f32, 0.0]];
        let mut xk = array![[1.0_f32, 0.0]];
        apply_rope_precomputed(xq.view_mut(), xk.view_mut(), &freqs);
        assert!(
            (xq[[0, 0]] - 0.0).abs() < 1e-6,
            "x' should be 0, got {}",
            xq[[0, 0]]
        );
        assert!(
            (xq[[0, 1]] - 1.0).abs() < 1e-6,
            "y' should be 1, got {}",
            xq[[0, 1]]
        );
        let _ = pos; // suppress unused warning
    }

    #[test]
    fn half_split_not_interleaved() {
        // Verifies the correct (half-split) pairing vs the wrong (adjacent-pair)
        // convention.  With head_dim=4, half=2:
        //   half-split:  pairs (0,2) and (1,3) — what Qwen2/LLaMA-2+ expect
        //   interleaved: pairs (0,1) and (2,3) — wrong
        //
        // Use base=1 so θ_i=1 for all i, and pos=1 → φ=1 rad for both pairs.
        // Start from (a, b) = (0, 1) in each pair:
        //   half-split rot:  head = [0, 1, ?, ?] → pairs (head[0],head[2]) and (head[1],head[3])
        //   With a=head[0]=1.0, b=head[2]=3.0 and a=head[1]=2.0, b=head[3]=4.0:
        //     head[0] = 1*cos1 - 3*sin1
        //     head[2] = 1*sin1 + 3*cos1
        //     head[1] = 2*cos1 - 4*sin1
        //     head[3] = 2*sin1 + 4*cos1
        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        let mut xq = array![[1.0_f32, 2.0, 3.0, 4.0]];
        let mut xk = xq.clone();
        apply_rope(xq.view_mut(), xk.view_mut(), 1, 4, 1.0);
        assert!(
            (xq[[0, 0]] - (1.0 * cos1 - 3.0 * sin1)).abs() < 1e-5,
            "head[0] wrong: {}",
            xq[[0, 0]]
        );
        assert!(
            (xq[[0, 1]] - (2.0 * cos1 - 4.0 * sin1)).abs() < 1e-5,
            "head[1] wrong: {}",
            xq[[0, 1]]
        );
        assert!(
            (xq[[0, 2]] - (1.0 * sin1 + 3.0 * cos1)).abs() < 1e-5,
            "head[2] wrong: {}",
            xq[[0, 2]]
        );
        assert!(
            (xq[[0, 3]] - (2.0 * sin1 + 4.0 * cos1)).abs() < 1e-5,
            "head[3] wrong: {}",
            xq[[0, 3]]
        );
    }

    #[test]
    fn freq_table_shape() {
        let table = build_freq_table(8, 64, 10_000.0);
        assert_eq!(table.len(), 8);
        assert_eq!(table[0].len(), 32); // head_dim / 2
    }

    #[test]
    fn freq_table_matches_apply_rope() {
        // A result from the table at pos=3 must equal apply_rope at pos=3.
        let head_dim = 8;
        let base = 10_000.0_f32;
        let table = build_freq_table(16, head_dim, base);

        let orig = array![[1.0_f32, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]];
        let mut xq_a = orig.clone();
        let mut xk_a = orig.clone();
        apply_rope(xq_a.view_mut(), xk_a.view_mut(), 3, head_dim, base);

        let mut xq_b = orig.clone();
        let mut xk_b = orig.clone();
        apply_rope_precomputed(xq_b.view_mut(), xk_b.view_mut(), &table[3]);

        for (a, b) in xq_a.iter().zip(xq_b.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }
}
