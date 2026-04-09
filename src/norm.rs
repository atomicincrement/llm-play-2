//! RMS Normalisation.
//!
//! $$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \varepsilon}} \cdot w_i$$

use ndarray::{Array1, ArrayView1};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Scale a vector so that its root-mean-square magnitude becomes 1, then
/// multiply element-wise by a learned `weight` vector.
///
/// ## Why normalisation matters
///
/// After many matrix multiplications the numbers inside the model can grow or
/// shrink by large factors, which makes learning unstable.  RMS norm is a
/// cheap "volume knob" that brings values back to a consistent scale before
/// each attention or FFN block.
///
/// ## The formula
///
/// For a vector x of length d:
///
/// ```text
///   rms  = sqrt( mean(x²) + eps )        ← "how loud is the signal?"
///   x̂_i  = (x_i / rms) * weight_i        ← rescale and then re-weight
/// ```
///
/// `eps` (typically 1e-6) prevents division by zero when x is all-zeros.
///
/// ## Example
///
/// ```rust
/// # use ndarray::array;
/// # use llm_play_2::norm::rms_norm; // illustrative
/// let x = array![3.0_f32, 4.0];           // RMS = sqrt((9+16)/2) = sqrt(12.5)
/// let w = array![1.0_f32, 1.0];           // unit weights → no extra scaling
/// // After norm each element is divided by √12.5 ≈ 3.536
/// // x̂ ≈ [0.849, 1.131]
/// ```
///
/// # Panics
/// Panics if `x` and `weight` differ in length.
pub fn rms_norm(x: ArrayView1<f32>, weight: ArrayView1<f32>, eps: f32) -> Array1<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm: x and weight must have the same length"
    );

    let mean_sq = x.dot(&x) / x.len() as f32 + eps;
    let rsqrt = 1.0 / mean_sq.sqrt();
    &x * rsqrt * weight
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn rms_norm_unit_weights() {
        // If all weights are 1 and eps≈0, output should have RMS ≈ 1.
        let x = ndarray::array![1.0_f32, 2.0, 3.0, 4.0];
        let w = ndarray::array![1.0_f32, 1.0, 1.0, 1.0];
        let out = rms_norm(x.view(), w.view(), 1e-6);
        let rms_out = (out.dot(&out) / out.len() as f32).sqrt();
        assert!((rms_out - 1.0).abs() < 1e-5, "output RMS = {rms_out}");
    }

    #[test]
    fn rms_norm_matches_reference() {
        // Compare against the direct sqrt formula for a small vector.
        let x = ndarray::array![0.5_f32, -1.5, 2.0, -0.25];
        let w = ndarray::array![1.0_f32, 2.0, 0.5, 1.5];
        let eps = 1e-6_f32;

        // Reference: standard sqrt.
        let mean_sq = x.dot(&x) / x.len() as f32 + eps;
        let ref_out: Array1<f32> = &x * (1.0 / mean_sq.sqrt()) * &w;

        // Function under test.
        let out = rms_norm(x.view(), w.view(), eps);

        for (a, b) in out.iter().zip(ref_out.iter()) {
            assert!((a - b).abs() < EPS, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn rms_norm_zero_vector() {
        // All-zeros input should produce all-zeros output (eps prevents div-by-zero).
        let x = ndarray::array![0.0_f32, 0.0, 0.0];
        let w = ndarray::array![1.0_f32, 1.0, 1.0];
        let out = rms_norm(x.view(), w.view(), 1e-6);
        for v in out.iter() {
            assert!(v.abs() < 1e-4, "expected ~0, got {v}");
        }
    }
}
