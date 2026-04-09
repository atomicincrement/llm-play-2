//! Safetensors loader — parses the JSON header with `asmjson`,
//! memory-maps the file with `memmap2`, and returns every tensor
//! as an `ndarray::ArrayD<f32>` (BF16 and F16 weights are widened on load).
//!
//! # File format
//! ```text
//! [ 8 bytes LE u64 : header_len ][ header_len bytes UTF-8 JSON ][ binary blob ]
//! ```
//! Each JSON entry (except `__metadata__`) has the shape:
//! ```json
//! { "dtype": "BF16", "shape": [1024, 512], "data_offsets": [0, 1048576] }
//! ```
//! `data_offsets` are byte ranges **into the binary blob** (i.e. relative to
//! `8 + header_len`, not to the start of the file).

use std::{collections::HashMap, fs::File, path::Path};

use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use asmjson::{parse_to_dom, JsonRef};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Load every tensor from a `.safetensors` file into memory as `f32` arrays.
///
/// ## What this does
///
/// A `.safetensors` file is the standard way HuggingFace stores model weights.
/// Think of it as a zip archive where each entry is a named multi-dimensional
/// array (tensor).  The file starts with a small JSON header that lists every
/// tensor's name, data type, shape, and byte location; the raw numbers follow.
///
/// This function:
/// 1. **Memory-maps** the file — the OS maps the file bytes into the process's
///    address space without copying.  Reading a 500 MB file takes milliseconds.
/// 2. **Parses the JSON header** with `asmjson`.
/// 3. **Decodes each tensor**: BF16 and F16 weights are widened to `f32` on
///    the fly.  Integer tensors (token ids stored as metadata) are skipped.
/// 4. Returns a `HashMap<name, ArrayD<f32>>` — you can look up any tensor by
///    its HuggingFace name, e.g. `"model.embed_tokens.weight"`.
///
/// ## Example
///
/// ```text
/// let tensors = load_safetensors(Path::new("models/Qwen2.5-0.5B/model.safetensors"))?;
/// println!("{} tensors loaded", tensors.len());          // 290
/// let embed = &tensors["model.embed_tokens.weight"];
/// println!("{:?}", embed.shape());                       // [151936, 896]
/// ```
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, ArrayD<f32>>, String> {
    // ── mmap ──────────────────────────────────────────────────────────────────
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    // SAFETY: we treat the mapping as read-only and do not mutate the backing file.
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };

    // ── header length ─────────────────────────────────────────────────────────
    if mmap.len() < 8 {
        return Err("file is shorter than 8 bytes".into());
    }
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;

    // ── JSON header ───────────────────────────────────────────────────────────
    let header_end = 8 + header_len;
    let header_bytes = mmap.get(8..header_end).ok_or_else(|| {
        format!(
            "header length {header_len} exceeds file size {}",
            mmap.len()
        )
    })?;
    let header_str =
        std::str::from_utf8(header_bytes).map_err(|e| format!("header not UTF-8: {e}"))?;
    let tape = parse_to_dom(header_str, None)
        .ok_or_else(|| "header JSON: parse failed".to_string())?;
    let root = tape.root().ok_or_else(|| "header JSON: empty document".to_string())?;

    // ── data blob starts right after the header ────────────────────────────────
    let blob = mmap.get(header_end..).ok_or("no data blob after header")?;

    // ── iterate tensors ───────────────────────────────────────────────────────
    let mut tensors: HashMap<String, ArrayD<f32>> = HashMap::new();

    for (name, meta) in root
        .object_iter()
        .ok_or("safetensors header must be a JSON object")?
    {
        if name == "__metadata__" {
            continue;
        }

        // Extract dtype, shape, and byte offsets from the JSON metadata entry.
        let dtype = meta
            .get("dtype")
            .as_str()
            .ok_or_else(|| format!("missing dtype for {name}"))?;
        let shape: Vec<usize> = meta
            .get("shape")
            .and_then(|v| v.array_iter())
            .ok_or("missing shape")?
            .enumerate()
            .map(|(i, v)| {
                v.as_u64()
                    .ok_or_else(|| format!("non-integer shape dim {i} in {name}"))
                    .map(|n| n as usize)
            })
            .collect::<Result<_, _>>()?;
        let start = meta
            .get("data_offsets")
            .index_at(0)
            .as_u64()
            .ok_or("data_offsets[0] not u64")? as usize;
        let end = meta
            .get("data_offsets")
            .index_at(1)
            .as_u64()
            .ok_or("data_offsets[1] not u64")? as usize;

        match decode_entry(name, dtype, &shape, start, end, blob) {
            Ok(Some(arr)) => {
                tensors.insert(name.to_owned(), arr);
            }
            Ok(None) => {
                // non-float dtype — silently skip
            }
            Err(e) => {
                return Err(format!("tensor {name:?}: {e}"));
            }
        }
    }

    Ok(tensors)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Decode one tensor from the binary blob.
///
/// Called with pre-extracted `dtype` (e.g. `"BF16"`), `shape` (e.g. `&[4864, 896]`),
/// and the `start`/`end` byte offsets into `blob`.
///
/// Returns `Ok(Some(array))` for float tensors, `Ok(None)` for integer
/// tensors we intentionally skip, or `Err` if the data is malformed.
fn decode_entry(
    name: &str,
    dtype: &str,
    shape: &[usize],
    start: usize,
    end: usize,
    blob: &[u8],
) -> Result<Option<ArrayD<f32>>, String> {
    // Skip integer / boolean tensors (not needed for inference).
    match dtype {
        "F32" | "BF16" | "F16" => {}
        other => {
            eprintln!("safetensors: skipping tensor {name:?} with unsupported dtype {other}");
            return Ok(None);
        }
    }

    let raw = blob.get(start..end).ok_or_else(|| {
        format!(
            "data_offsets [{start}, {end}) out of range (blob len {})",
            blob.len()
        )
    })?;

    let numel: usize = shape.iter().copied().product::<usize>().max(1); // treat scalar as 1-element

    let data: Vec<f32> = match dtype {
        "F32" => decode_f32(raw, numel)?,
        "BF16" => decode_bf16(raw, numel)?,
        "F16" => decode_f16(raw, numel)?,
        _ => unreachable!(),
    };

    let arr =
        ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| format!("shape error: {e}"))?;

    Ok(Some(arr))
}

// ── Dtype converters ─────────────────────────────────────────────────────────

fn decode_f32(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 4 {
        return Err(format!(
            "F32 expected {} bytes, got {}",
            numel * 4,
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn decode_bf16(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 2 {
        return Err(format!(
            "BF16 expected {} bytes, got {}",
            numel * 2,
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|b| {
            // BF16 occupies the high 16 bits of an f32 mantissa.
            let bits = (u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16;
            f32::from_bits(bits)
        })
        .collect())
}

fn decode_f16(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 2 {
        return Err(format!(
            "F16 expected {} bytes, got {}",
            numel * 2,
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
        .collect())
}

/// Convert an IEEE 754 binary16 half-precision value to f32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    // Decompose the half-precision value.
    let sign = ((bits as u32) & 0x8000) << 16; // bit 31
    let exp = ((bits as u32) & 0x7C00) >> 10; // bits 14-10
    let mant = ((bits as u32) & 0x03FF) << 13; // bits 9-0 → 22-13

    let val = if exp == 0 {
        // Subnormal — denormalize to nearest f32 subnormal.
        let m = mant;
        if m == 0 {
            0 // ±zero
        } else {
            // Normalise the subnormal mantissa.
            let mut e = 127 - 14u32; // f32 bias minus f16 bias
            let mut m2 = m << 1;
            while m2 & 0x00800000 == 0 {
                m2 <<= 1;
                e -= 1;
            }
            (e << 23) | (m2 & 0x007FFFFF)
        }
    } else if exp == 31 {
        0x7F800000 | mant // ±Inf or NaN
    } else {
        ((exp + 112) << 23) | mant // normalised
    };
    f32::from_bits(sign | val)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        // 1.0 in f16 = 0x3C00
        let v = f16_to_f32(0x3C00);
        assert!((v - 1.0).abs() < 1e-6, "f16 1.0 → {v}");
        // -2.0 in f16 = 0xC000
        let v = f16_to_f32(0xC000);
        assert!((v - (-2.0)).abs() < 1e-6, "f16 -2.0 → {v}");
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // +Inf
        assert!(f16_to_f32(0x7C00).is_infinite());
        // NaN
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn bf16_known_values() {
        // 1.0 in BF16 = 0x3F80  (same as top 16 bits of f32 1.0 = 0x3F80_0000)
        let bits: u32 = (0x3F80_u32) << 16;
        let f = f32::from_bits(bits);
        assert!((f - 1.0).abs() < 1e-6, "bf16 1.0 → {f}");
    }

    #[test]
    fn load_qwen_safetensors() {
        let path = Path::new("models/Qwen2.5-0.5B/model.safetensors");
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        let tensors = load_safetensors(path).expect("load failed");
        println!("loaded {} tensors", tensors.len());
        // embed_tokens must be present and have the right shape.
        let embed = tensors
            .get("model.embed_tokens.weight")
            .expect("embed_tokens.weight missing");
        assert_eq!(embed.ndim(), 2);
        assert_eq!(
            embed.shape()[1],
            896,
            "hidden_size should be 896 for Qwen2.5-0.5B"
        );
        println!("embed_tokens shape: {:?}", embed.shape());
    }
}
