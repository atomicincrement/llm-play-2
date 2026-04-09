use ndarray::{Array1, Array2, ArrayView1};

/// The three learned matrices for one SwiGLU feed-forward block.
///
/// Each transformer layer contains one FFN that runs *after* self-attention.
/// It is purely position-wise: every token's hidden state is processed
/// independently by the same three matrices.
///
/// * `w_gate` and `w_up` both expand `hidden → intermediate` (896 → 4864).
///   They work in tandem: `w_gate` produces a learned filter and `w_up`
///   produces content; the SiLU gate decides how much of the content passes.
/// * `w_down` compresses back `intermediate → hidden` (4864 → 896).
pub struct FfnWeights {
    /// Gate projection: shape [intermediate_size, hidden_size]
    pub w_gate: Array2<f32>,
    /// Up projection:   shape [intermediate_size, hidden_size]
    pub w_up: Array2<f32>,
    /// Down projection: shape [hidden_size, intermediate_size]
    pub w_down: Array2<f32>,
}

/// SwiGLU feed-forward network.
///
/// ## What this does, step by step
///
/// Think of the input `x` as a list of numbers (a "hidden state") that
/// summarises what the model has understood so far about the current token.
///
/// **Step 1 — Two parallel projections.**
/// We multiply `x` by two separate weight matrices to get two intermediate
/// vectors of the same (larger) size:
///   * `gate = W_gate · x`  — a "gate" vector, which will act as a filter.
///   * `up   = W_up   · x`  — an "up" vector, which carries the actual content.
///
/// **Step 2 — Squash the gate with SiLU.**
/// We apply the *SiLU activation* (also called *Swish*) to the gate:
///     `silu(z) = z * sigmoid(z) = z / (1 + e^{-z})`
/// SiLU is a smooth, non-linear "squashing" function.  Positive values come
/// through roughly unchanged; very negative values are pushed toward zero.
/// The key insight is that it lets values *pass through* proportionally rather
/// than clipping them hard like ReLU does.
///
/// **Step 3 — Gate the content.**
/// We multiply the squashed gate by the up vector element-wise:
///     `hidden = silu(gate) ⊙ up`
/// This is the *gating* step.  Each position in `up` is either amplified or
/// suppressed by the corresponding gate value.  The network learns which
/// "dimensions" of information are actually useful at this point and which
/// should be discarded.
///
/// **Step 4 — Project back down.**
/// Finally we multiply by a third weight matrix to compress back to the
/// original hidden size:
///     `output = W_down · hidden`
///
/// Together, steps 1–4 are called **SwiGLU** (Swish-Gated Linear Unit).
/// It is a non-linear "memory update" that lets each layer transform the
/// hidden state in a richer way than a plain single linear projection would.
pub fn feed_forward(x: ArrayView1<f32>, w: &FfnWeights) -> Array1<f32> {
    // Step 1: project up into the intermediate space.
    let gate: Array1<f32> = w.w_gate.dot(&x);
    let up: Array1<f32> = w.w_up.dot(&x);

    // Step 2 & 3: silu(gate) ⊙ up
    // silu(z) = z * σ(z) = z / (1 + exp(-z))
    let gated: Array1<f32> = gate.mapv(silu) * up;

    // Step 4: project back down to hidden size.
    w.w_down.dot(&gated)
}

/// SiLU (Swish) activation: `z * sigmoid(z) = z / (1 + e^{−z})`.
///
/// A smooth, almost-linear activation for positive values that tapers
/// gently to zero (not hard-clipped like ReLU) for negative values.
/// `silu(0) = 0`, `silu(large positive) ≈ identity`, `silu(large negative) ≈ 0`.
#[inline]
fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Output has the right shape.
    #[test]
    fn output_shape() {
        let hidden = 4;
        let intermediate = 8;
        let w = FfnWeights {
            w_gate: Array2::zeros((intermediate, hidden)),
            w_up: Array2::zeros((intermediate, hidden)),
            w_down: Array2::zeros((hidden, intermediate)),
        };
        let x = Array1::ones(hidden);
        let out = feed_forward(x.view(), &w);
        assert_eq!(out.len(), hidden);
    }

    /// With zero input the output must be zero (all-zero gate kills everything).
    #[test]
    fn zero_input_gives_zero_output() {
        let hidden = 4;
        let intermediate = 8;
        let w = FfnWeights {
            w_gate: Array2::ones((intermediate, hidden)),
            w_up: Array2::ones((intermediate, hidden)),
            w_down: Array2::ones((hidden, intermediate)),
        };
        let x = Array1::zeros(hidden);
        let out = feed_forward(x.view(), &w);
        for v in out.iter() {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    /// silu(0) == 0, silu(large positive) ≈ identity, silu(large negative) ≈ 0.
    #[test]
    fn silu_properties() {
        assert!((silu(0.0)).abs() < 1e-7);
        // For large z, silu(z) ≈ z
        let big: f32 = 20.0;
        assert!((silu(big) - big).abs() < 1e-3);
        // For very negative z, silu(z) ≈ 0
        assert!(silu(-20.0).abs() < 1e-3);
    }

    /// silu(1.0) == 1 / (1 + e^{-1})  (known value).
    #[test]
    fn silu_known_value() {
        let expected = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!((silu(1.0) - expected).abs() < 1e-6);
    }
}
