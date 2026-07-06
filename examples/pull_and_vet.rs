//! The controlled-model workflow, programmatically: preflight a
//! checkpoint's metadata (no weight download, no GPU), then pull it.
//!
//! This is the same admission path the registry uses for large family
//! members: the preflight proves the architecture, tokenizer family, and
//! the complete tensor table (dtypes and block grain) match what the
//! engine executes.
//!
//! Run: `cargo run --example pull_and_vet -- Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0`

use cima::{hub, vet};

fn main() -> cima::traits::Res<()> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0".into());
    let (repo, tag) = match model.split_once(':') {
        Some((r, t)) => (r.to_string(), Some(t.to_string())),
        None => (model.clone(), None),
    };

    println!("preflight {} (metadata only)…", model);
    let tensors = vet::preflight_deep(&repo, tag.as_deref())?;
    println!(
        "preflight PASS — {} tensors validated without downloading weights",
        tensors
    );

    println!("pulling {}…", model);
    hub::pull(&model, false, tag.as_deref())?;
    println!("ready in {}", hub::local_dir(&model).display());
    println!("full on-GPU certification: cima vet {}", model);
    Ok(())
}
