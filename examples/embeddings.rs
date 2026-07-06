//! Batch embeddings and cosine similarity between the results.
//!
//! Prerequisites: `cima serve` running and a model pulled (a generative
//! model answers via its pooled-embedding fallback). Honors
//! `CIMA_HOST`/`CIMA_PORT`; override the model with `CIMA_EXAMPLE_MODEL`.
//!
//!     cargo run --example embeddings

use cima::api::client::Client;
use cima::json::Json;
use cima::traits::Res;

fn main() -> Res<()> {
    let model = std::env::var("CIMA_EXAMPLE_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0".into());
    let texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Quarterly revenue grew eight percent.",
    ];

    let client = Client::local();
    let body = Json::obj().set("model", Json::s(&model)).set(
        "input",
        Json::Arr(texts.iter().map(|t| Json::s(t)).collect()),
    );
    let resp = client.post_json("/api/embed", &body)?;

    let embs = resp
        .arr_of("embeddings")
        .ok_or_else(|| cima::err!("example", "response carries no 'embeddings' array"))?;

    let vecs: Vec<Vec<f64>> = embs
        .iter()
        .map(|e| match e {
            Json::Arr(a) => a.iter().filter_map(Json::as_f64).collect(),
            _ => Vec::new(),
        })
        .collect();

    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            println!(
                "cos({:?}, {:?}) = {:.3}",
                texts[i],
                texts[j],
                cosine(&vecs[i], &vecs[j])
            );
        }
    }
    Ok(())
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}
