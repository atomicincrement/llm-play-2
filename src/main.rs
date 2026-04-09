mod attention;
mod config;
mod ffn;
mod model;
mod norm;
mod rope;
mod safetensors;
mod sample;
mod tokenizer;

use std::path::Path;
use std::time::Instant;

use config::ModelConfig;
use model::QwenModel;
use tokenizer::Tokenizer;

fn main() {
    let model_dir = "models/Qwen2.5-0.5B";

    // ── Load tokenizer ────────────────────────────────────────────────────────
    // The tokenizer turns a human-readable string like "Hello" into a list of
    // integer token ids, and turns ids back into text after generation.
    print!("Loading tokenizer … ");
    let tok = Tokenizer::from_file(format!("{model_dir}/tokenizer.json"))
        .expect("failed to load tokenizer");
    println!(
        "ok  ({} vocab, {} merges, {} special tokens)",
        tok.vocab.len(),
        tok.merges.len(),
        tok.special_tokens.len(),
    );

    // ── Load weights ──────────────────────────────────────────────────────────
    // All 290 tensors are memory-mapped from disk and promoted from BF16 to
    // f32.  This is the biggest pause — ~0.5 GB of floats to decode.
    print!("Loading weights … ");
    let t0 = Instant::now();
    let tensors =
        safetensors::load_safetensors(Path::new(&format!("{model_dir}/model.safetensors")))
            .expect("failed to load safetensors");
    println!(
        "ok  ({} tensors, {:.1}s)",
        tensors.len(),
        t0.elapsed().as_secs_f32()
    );

    // ── Load config ───────────────────────────────────────────────────────────
    let cfg = ModelConfig::from_file(Path::new(&format!("{model_dir}/config.json")))
        .expect("failed to load config");
    println!(
        "Config: hidden={} layers={} heads={}q/{}kv head_dim={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.vocab_size,
    );

    // ── Build the model ───────────────────────────────────────────────────────
    // Wire all the loaded tensors into the QwenModel struct.
    print!("Building model … ");
    let t0 = Instant::now();
    let mut model = QwenModel::from_tensors(&tensors, cfg).expect("failed to build model");
    println!("ok  ({:.1}s)", t0.elapsed().as_secs_f32());

    // ── Generate ──────────────────────────────────────────────────────────────
    // Use the Qwen2.5 BOS token id (151643) so the model knows it is at the
    // start of a fresh sequence.  We prepend it to the prompt manually;
    // some models expect it, others do not — Qwen2.5 does.
    let prompt = "The capital of France is";
    let max_new_tokens = 64;
    let top_p = 0.9;
    let temperature = 0.1;
    let seed = 42;

    println!("\nPrompt: {prompt:?}");
    println!("Generating up to {max_new_tokens} tokens  (top_p={top_p}, temp={temperature})…\n");

    let t0 = Instant::now();
    let generated = sample::generate(
        &mut model,
        &tok,
        prompt,
        max_new_tokens,
        top_p,
        temperature,
        seed,
    );
    let elapsed = t0.elapsed().as_secs_f32();

    // ── Report speed ──────────────────────────────────────────────────────────
    // Count tokens the same way the sampler does: encode the output.
    let n_tokens = tok.encode(&generated).len();
    let tps = n_tokens as f32 / elapsed;
    println!("\n{n_tokens} tokens in {elapsed:.1}s  →  {tps:.1} tokens/sec (single CPU core)");
}
