//! One-shot completion against a running cima server.
//!
//! Prerequisites: `cima serve` running (honors `CIMA_HOST`/`CIMA_PORT`)
//! and the model pulled, e.g.
//! `cima pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0`.
//!
//! Run: `cargo run --example generate -- "Why is the sky blue?"`

use cima::api::client::Client;
use cima::json::Json;

fn main() -> cima::traits::Res<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Why is the sky blue? Answer in two sentences.".into());
    let model = std::env::var("CIMA_EXAMPLE_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0".into());

    let client = Client::local();
    let body = Json::obj()
        .set("model", Json::s(&model))
        .set("prompt", Json::s(&prompt))
        .set("stream", Json::b(false))
        .set(
            "options",
            Json::obj()
                .set("num_predict", Json::n(128.0))
                .set("temperature", Json::n(0.7)),
        );

    let resp = client.post_json("/api/generate", &body)?;
    println!("{}", resp.str_of("response").unwrap_or(""));

    // The response carries the engine's own telemetry, in nanoseconds.
    if let (Some(n), Some(dur)) = (resp.u64_of("eval_count"), resp.u64_of("eval_duration")) {
        if dur > 0 {
            eprintln!("[{:.1} tok/s]", n as f64 / (dur as f64 / 1e9));
        }
    }
    Ok(())
}
