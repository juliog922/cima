//! Streaming multi-turn chat: tokens print as they are sampled, and the
//! model stays resident between turns via `keep_alive`.
//!
//! Prerequisites: `cima serve` running and the model pulled.
//!
//! Run: `cargo run --example chat_stream`

use cima::api::client::Client;
use cima::json::Json;
use std::io::Write;

fn main() -> cima::traits::Res<()> {
    let model = std::env::var("CIMA_EXAMPLE_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0".into());
    let client = Client::local();

    let turns = [
        "Name three uses for a brick, one sentence each.",
        "Which of those would survive rain the longest, and why?",
    ];
    let mut messages: Vec<Json> = Vec::new();

    for user in turns {
        println!("\n>>> {}\n", user);
        messages.push(
            Json::obj()
                .set("role", Json::s("user"))
                .set("content", Json::s(user)),
        );
        let body = Json::obj()
            .set("model", Json::s(&model))
            .set("messages", Json::Arr(messages.clone()))
            .set("stream", Json::b(true))
            .set("keep_alive", Json::s("10m"))
            .set("options", Json::obj().set("num_predict", Json::n(160.0)));

        let mut reply = String::new();
        client.generate_stream(&body, true, |chunk| {
            if let Some(piece) = chunk.get("message").and_then(|m| m.str_of("content")) {
                print!("{}", piece);
                let _ = std::io::stdout().flush();
                reply.push_str(piece);
            }
        })?;
        println!();
        messages.push(
            Json::obj()
                .set("role", Json::s("assistant"))
                .set("content", Json::s(&reply)),
        );
    }
    Ok(())
}
