//! # api — the Ollama-compatible HTTP surface
//!
//! One file, four inline modules, split by responsibility (the client half
//! is destined for its own crate):
//!
//! * [`protocol`] — wire types shared by both sides: request parsing,
//!   response shaping, NDJSON chunk layouts. No I/O, no engine types
//!   beyond `GenOptions`. **This is the contract**; anything the server
//!   emits or the client reads is built here.
//! * [`queue`] — the strict-FIFO single-user GPU ticket queue the server
//!   admits requests through.
//! * [`server`] — owns the listener, the request queue, the
//!   [`crate::models::ModelManager`] (GPU residency: load on demand,
//!   keep-alive eviction), and the streaming writers. Everything that
//!   touches the engine lives here.
//! * [`client`] — a dependency-free typed client over the same protocol
//!   (plain HTTP/1.1 over `TcpStream`). The CLI's server-facing commands
//!   (`cima ps`, `cima stop`) use it; exporting it as `cima-client` means
//!   lifting the `client` + `protocol` modules + the tiny JSON module.

pub use server::{serve, Server, OLLAMA_COMPAT_VERSION, VERSION};

pub mod protocol {
    //! Wire types of the Ollama-compatible API: pure JSON shaping, no I/O.
    //! The server parses requests and builds responses through these; a client
    //! crate reads/writes the very same shapes, so the two cannot drift.

    use crate::json::Json;

    /// Default bind/connect port. 11435 — NOT ollama's 11434 — so both
    /// daemons coexist on one machine (`CIMA_PORT` overrides on both sides).
    pub const DEFAULT_PORT: u16 = 11435;

    use crate::traits::GenOptions;

    /// `options` sub-object of generate/chat requests — the full Ollama set we
    /// honor. Unknown keys are ignored (Ollama-compatible behavior); honored:
    /// `temperature`, `top_k`, `top_p`, `seed`, `num_predict`, `stop`,
    /// `repeat_penalty`. (`num_ctx` is fixed at load time by the KV
    /// allocation; requests beyond it are rejected with a clear error.)
    pub fn options_from_json(opts: &mut GenOptions, body: &Json) {
        let Some(o) = body.get("options") else { return };
        if let Some(v) = o.f64_of("temperature") {
            opts.temperature = v as f32;
        }
        if let Some(v) = o.f64_of("top_k") {
            opts.top_k = v as usize;
        }
        if let Some(v) = o.f64_of("top_p") {
            opts.top_p = v as f32;
        }
        if let Some(v) = o.f64_of("seed") {
            opts.seed = v as u64;
        }
        if let Some(v) = o.f64_of("num_predict") {
            if v >= 0.0 {
                opts.max_tokens = v as usize;
            }
        }
        if let Some(v) = o.f64_of("repeat_penalty") {
            opts.repeat_penalty = v as f32;
        }
        if let Some(stops) = o.arr_of("stop") {
            opts.stop = stops
                .iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect();
        } else if let Some(s) = o.str_of("stop") {
            opts.stop = vec![s.to_owned()];
        }
        // Benchmark/parity flag: generate the full num_predict budget without
        // stopping at EOS. Accepts a JSON bool (or a "true"/"false" string).
        if let Some(v) = o.bool_of("ignore_eos") {
            opts.ignore_eos = v;
        } else if let Some(s) = o.str_of("ignore_eos") {
            opts.ignore_eos = matches!(s.trim(), "true" | "1" | "yes");
        }
    }

    /// One NDJSON chunk of a streaming generate/chat response.
    pub fn stream_chunk(model: &str, chat: bool, piece: &str, done: bool) -> Json {
        let mut j = Json::obj()
            .set("model", Json::s(model))
            .set("created_at", Json::s(&now_rfc3339()))
            .set("done", Json::b(done));
        if chat {
            j = j.set(
                "message",
                Json::obj()
                    .set("role", Json::s("assistant"))
                    .set("content", Json::s(piece)),
            );
        } else {
            j = j.set("response", Json::s(piece));
        }
        j
    }

    /// The final chunk (or the whole body when `stream: false`), with timing
    /// and token-count metadata in Ollama's field names (durations are
    /// nanoseconds). Built here — the one place — for both the in-tree server
    /// and the client crate. `full_text: None` means a streaming close (empty
    /// message/response); `Some` carries the whole reply for `stream: false`.
    #[allow(clippy::too_many_arguments)]
    pub fn final_chunk(
        model: &str,
        chat: bool,
        full_text: Option<&str>,
        prompt_tokens: usize,
        gen_tokens: usize,
        total_ms: f64,
        ttft_ms: f64,
        load_ms: f64,
        done_reason: &str,
    ) -> Json {
        stream_chunk(model, chat, full_text.unwrap_or(""), true)
            .set("done_reason", Json::s(done_reason))
            // Ollama specifies these as integer nanoseconds. ms * 1e6
            // leaves float residue (670802022.0000001), which the
            // serializer then prints with a fraction and typed clients
            // reject when deserializing into u64. Round at the source.
            .set("total_duration", Json::u((total_ms * 1e6).round() as u64))
            .set("load_duration", Json::u((load_ms * 1e6).round() as u64))
            .set("prompt_eval_count", Json::u(prompt_tokens as u64))
            .set(
                "prompt_eval_duration",
                Json::u((ttft_ms * 1e6).round() as u64),
            )
            .set("eval_count", Json::u(gen_tokens as u64))
            .set(
                "eval_duration",
                Json::u(((total_ms - ttft_ms).max(0.0) * 1e6).round() as u64),
            )
    }

    /// `/api/ps` entry for a resident model.
    pub fn ps_entry(name: &str, vram: usize, expires_in_secs: Option<u64>) -> Json {
        let mut j = Json::obj()
            .set("name", Json::s(name))
            .set("model", Json::s(name))
            .set("size_vram", Json::n(vram as f64));
        j = match expires_in_secs {
            Some(s) => j.set("expires_in", Json::n(s as f64)),
            None => j.set("expires_in", Json::s("forever")),
        };
        j
    }

    pub fn now_rfc3339() -> String {
        // Wall-clock RFC3339 (UTC, seconds precision). The civil-time math
        // lives in log::stamp_at — a standalone client-crate lift takes that
        // 20-line function along with this module.
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        crate::log::stamp_at(secs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::json;

        #[test]
        fn options_full_set_and_stop_forms() {
            let body = json::parse(
                r#"{"options":{"temperature":0.5,"top_k":7,"top_p":0.8,"seed":42,
                "num_predict":99,"repeat_penalty":1.2,"stop":["a","b"]}}"#,
            )
            .unwrap();
            let mut o = GenOptions::default();
            options_from_json(&mut o, &body);
            assert_eq!((o.top_k, o.max_tokens, o.seed), (7, 99, 42));
            assert_eq!(o.stop, vec!["a", "b"]);
            let body2 = json::parse(r#"{"options":{"stop":"END"}}"#).unwrap();
            options_from_json(&mut o, &body2);
            assert_eq!(o.stop, vec!["END"]);
        }
    }
}

pub mod queue {
    //! # queue — single-user GPU execution gate
    //!
    //! On-premise constraint: exactly one inference process may own the GPU at a
    //! time. Concurrent API connections each run on their own thread; before
    //! touching the [`crate::models::ModelManager`] they pass through
    //! [`GpuQueue::acquire`], a strict FIFO ticket lock built on
    //! `Mutex` + `Condvar` (no spinning, no fairness lottery — arrival order is
    //! service order, and the wait time is measured for telemetry).
    //!
    //! ```text
    //! request thread ──► acquire() ──► [FIFO ticket wait] ──► GpuPermit ──► GPU
    //!                                                          │
    //!                                  drop(GpuPermit) ◄───────┘  (next ticket wakes)
    //! ```

    use std::sync::{Condvar, Mutex};
    use std::time::Instant;

    /// Internal ticket-lock state.
    struct State {
        /// Next ticket number to hand out.
        next: u64,
        /// Ticket currently allowed to run.
        serving: u64,
    }

    /// Strict-FIFO GPU admission queue.
    pub struct GpuQueue {
        state: Mutex<State>,
        cv: Condvar,
    }

    /// RAII permit: holding it means owning the GPU. Dropping it admits the next
    /// ticket in arrival order.
    pub struct GpuPermit<'a> {
        queue: &'a GpuQueue,
        /// Time spent waiting in the queue (for the `queue_wait_ms` metric).
        pub wait_ms: f64,
        /// Queue depth observed at enqueue time (logged for capacity planning).
        pub depth_at_enqueue: u64,
    }

    impl Default for GpuQueue {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GpuQueue {
        pub fn new() -> GpuQueue {
            GpuQueue {
                state: Mutex::new(State {
                    next: 0,
                    serving: 0,
                }),
                cv: Condvar::new(),
            }
        }

        /// Block until this caller reaches the head of the queue.
        pub fn acquire(&self) -> GpuPermit<'_> {
            let t0 = Instant::now();
            let mut st = self.state.lock().unwrap();
            let ticket = st.next;
            let depth = ticket - st.serving;
            st.next += 1;
            while st.serving != ticket {
                st = self.cv.wait(st).unwrap();
            }
            GpuPermit {
                queue: self,
                wait_ms: t0.elapsed().as_secs_f64() * 1e3,
                depth_at_enqueue: depth,
            }
        }

        /// Bounded admission: refuse instead of queueing when the depth has
        /// reached `CIMA_MAX_QUEUE` (0 or unset = unbounded). An unbounded
        /// queue turns a traffic spike into minutes of invisible latency;
        /// operators who prefer fast 429s over slow 200s set the cap and
        /// let their client retry.
        pub fn try_acquire(&self) -> Result<GpuPermit<'_>, u64> {
            let cap = max_queue();
            if cap > 0 {
                let st = self.state.lock().unwrap();
                let depth = st.next - st.serving;
                if depth >= cap {
                    return Err(depth);
                }
            }
            Ok(self.acquire())
        }

        /// Current number of waiting + running requests (for `/api/ps`-style info).
        pub fn depth(&self) -> u64 {
            let st = self.state.lock().unwrap();
            st.next - st.serving
        }
    }

    /// Admission cap from `CIMA_MAX_QUEUE` (0 or unset/unparseable = unbounded).
    fn max_queue() -> u64 {
        static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("CIMA_MAX_QUEUE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        })
    }

    impl Drop for GpuPermit<'_> {
        fn drop(&mut self) {
            let mut st = self.queue.state.lock().unwrap();
            st.serving += 1;
            drop(st);
            // All waiters wake; only the matching ticket proceeds (FIFO holds).
            self.queue.cv.notify_all();
        }
    }
}

pub mod client {
    //! A dependency-free client for the cima server: plain HTTP/1.1 over
    //! `TcpStream`, NDJSON streaming, the same `protocol` shapes the server
    //! emits. The CLI's server-facing commands use it; it is written to be
    //! liftable into a standalone `cima-client` crate together with
    //! `protocol.rs` and the JSON module (no engine types cross this line).

    use std::io::{Read, Write};
    use std::net::TcpStream;

    use crate::json::{self, Json};
    use crate::{err, traits::Res};

    // `version`/`tags`/`generate_stream` are part of the exported client
    // surface even though the in-tree CLI doesn't call them all yet.
    #[allow(dead_code)]
    pub struct Client {
        host: String,
        port: u16,
    }

    #[allow(dead_code)]
    impl Client {
        pub fn new(host: &str, port: u16) -> Client {
            Client {
                host: host.into(),
                port,
            }
        }

        /// Default endpoint, honoring the same env the server reads.
        pub fn local() -> Client {
            let host = std::env::var("CIMA_HOST").unwrap_or_else(|_| "127.0.0.1".into());
            let port = std::env::var("CIMA_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(super::protocol::DEFAULT_PORT);
            Client::new(&host, port)
        }

        fn request(&self, method: &str, path: &str, body: Option<&Json>) -> Res<TcpStream> {
            let mut s = TcpStream::connect((self.host.as_str(), self.port)).map_err(|e| {
                err!(
                    "client",
                    "cannot reach cima server at {}:{} ({}) — is `cima serve` running?",
                    self.host,
                    self.port,
                    e
                )
            })?;
            let payload = body.map(|b| b.dump()).unwrap_or_default();
            // One write_all: a request split across syscalls races servers
            // that close after responding.
            let req = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            method, path, self.host, payload.len(), payload
        );
            s.write_all(req.as_bytes())
                .map_err(|e| err!("client", "write: {}", e))?;
            Ok(s)
        }

        /// Read a full (non-streaming) response body as JSON. Non-2xx
        /// statuses — and bodies carrying an "error" field — surface as
        /// `Err`: a typed client must never report an API failure as
        /// success (a deleted-nothing "deleted" is the bug class this
        /// prevents).
        fn read_json(mut s: TcpStream) -> Res<Json> {
            let (header, early) = read_header(&mut s)?;
            let status: u16 = header
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut dec = ChunkDecoder::new(is_chunked(&header));
            let mut body = Vec::new();
            dec.feed(&early, &mut body);
            let mut tmp = [0u8; 8192];
            loop {
                let n = s
                    .read(&mut tmp)
                    .map_err(|e| err!("client", "read: {}", e))?;
                if n == 0 {
                    break;
                }
                dec.feed(&tmp[..n], &mut body);
            }
            let text = String::from_utf8_lossy(&body);
            let j =
                json::parse(text.trim()).map_err(|e| err!("client", "bad response JSON: {}", e))?;
            if !(200..300).contains(&status) {
                let msg = j.str_of("error").unwrap_or("(no error body)");
                return Err(err!("client", "server returned HTTP {}: {}", status, msg));
            }
            if let Some(e) = j.str_of("error") {
                return Err(err!("client", "server error: {}", e));
            }
            Ok(j)
        }

        /// Generic JSON POST — the escape hatch for endpoints without a
        /// dedicated wrapper (`/api/embed`, `/api/show`, ...).
        pub fn post_json(&self, path: &str, body: &Json) -> Res<Json> {
            Self::read_json(self.request("POST", path, Some(body))?)
        }

        pub fn version(&self) -> Res<Json> {
            Self::read_json(self.request("GET", "/api/version", None)?)
        }

        pub fn ps(&self) -> Res<Json> {
            Self::read_json(self.request("GET", "/api/ps", None)?)
        }

        pub fn tags(&self) -> Res<Json> {
            Self::read_json(self.request("GET", "/api/tags", None)?)
        }

        /// Readiness probe. With `models` empty, a liveness check; otherwise
        /// per-model presence plus an overall `ready` bool. Mirrors
        /// `GET`/`POST /api/ready`.
        pub fn ready(&self, models: &[String]) -> Res<Json> {
            let body = Json::obj().set(
                "models",
                Json::Arr(models.iter().map(|m| Json::s(m)).collect()),
            );
            Self::read_json(self.request("POST", "/api/ready", Some(&body))?)
        }

        /// Ask the server to release a model immediately (`keep_alive: 0` with
        /// an empty prompt — Ollama's unload idiom).
        pub fn stop(&self, model: &str) -> Res<()> {
            let body = Json::obj()
                .set("model", Json::s(model))
                .set("prompt", Json::s(""))
                .set("stream", Json::b(false))
                .set("keep_alive", Json::n(0.0));
            Self::read_json(self.request("POST", "/api/generate", Some(&body))?)?;
            Ok(())
        }

        /// Server-routed delete that NEVER logs: callers with a disk
        /// fallback (cima rm) probe the server as a courtesy — a 404 or a
        /// refused connection there is the expected path, not an ERROR for
        /// the terminal. Returns true only on a confirmed 2xx delete.
        pub fn delete_quiet(&self, model: &str) -> bool {
            // Truly quiet: `err!` logs at construction, so probing through
            // `request` would print a scary ERROR for the perfectly normal
            // no-server case. Check reachability silently first.
            if std::net::TcpStream::connect((self.host.as_str(), self.port)).is_err() {
                return false;
            }
            (|| -> Option<bool> {
                let body = Json::obj().set("model", Json::s(model));
                let mut s = self.request("DELETE", "/api/delete", Some(&body)).ok()?;
                let mut raw = Vec::new();
                s.read_to_end(&mut raw).ok()?;
                let head = String::from_utf8_lossy(&raw);
                Some(
                    head.split_whitespace()
                        .nth(1)
                        .map(|c| c.starts_with('2'))
                        .unwrap_or(false),
                )
            })()
            .unwrap_or(false)
        }

        pub fn delete(&self, model: &str) -> Res<Json> {
            let body = Json::obj().set("model", Json::s(model));
            Self::read_json(self.request("DELETE", "/api/delete", Some(&body))?)
        }

        /// Streaming generate/chat: `on_chunk` receives each NDJSON object;
        /// returns the final (done) chunk. Chunked transfer is decoded
        /// incrementally, so NDJSON lines split across TCP reads or HTTP
        /// chunk boundaries reassemble correctly.
        pub fn generate_stream(
            &self,
            body: &Json,
            chat: bool,
            mut on_chunk: impl FnMut(&Json),
        ) -> Res<Json> {
            let path = if chat { "/api/chat" } else { "/api/generate" };
            let mut s = self.request("POST", path, Some(body))?;
            let (header, early) = read_header(&mut s)?;
            let mut dec = ChunkDecoder::new(is_chunked(&header));
            let mut buf = Vec::new();
            dec.feed(&early, &mut buf);
            let mut tmp = [0u8; 8192];
            let mut last = Json::Null;
            loop {
                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..nl + 1).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(j) = json::parse(line) {
                        let done = j.bool_of("done").unwrap_or(false);
                        on_chunk(&j);
                        if done {
                            last = j;
                        }
                    }
                }
                let n = s
                    .read(&mut tmp)
                    .map_err(|e| err!("client", "read: {}", e))?;
                if n == 0 {
                    // Flush a final unterminated line — non-streaming bodies
                    // (the load/unload idiom, stream:false) end without '\n'.
                    let tail = String::from_utf8_lossy(&buf).trim().to_string();
                    if !tail.is_empty() {
                        if let Ok(j) = json::parse(&tail) {
                            if j.bool_of("done").unwrap_or(false) {
                                on_chunk(&j);
                                last = j;
                            }
                        }
                    }
                    break;
                }
                dec.feed(&tmp[..n], &mut buf);
            }
            if matches!(last, Json::Null) {
                return Err(err!("client", "stream ended without a done chunk"));
            }
            Ok(last)
        }
    }

    #[allow(dead_code)]
    fn find2(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// Incremental HTTP/1.1 `Transfer-Encoding: chunked` decoder. Fed raw
    /// socket bytes, it appends only PAYLOAD bytes to `out` — correct even
    /// when the peer flushes one byte at a time (the adversarial case the
    /// integration tests exercise). Identity transfer passes bytes through.
    struct ChunkDecoder {
        chunked: bool,
        /// Bytes remaining in the current chunk's data section; when 0 we are
        /// parsing a size line (or trailing CRLF) accumulated in `line`.
        remaining: usize,
        line: Vec<u8>,
        done: bool,
    }

    impl ChunkDecoder {
        fn new(chunked: bool) -> ChunkDecoder {
            ChunkDecoder {
                chunked,
                remaining: 0,
                line: Vec::new(),
                done: false,
            }
        }

        fn feed(&mut self, mut input: &[u8], out: &mut Vec<u8>) {
            if !self.chunked {
                out.extend_from_slice(input);
                return;
            }
            while !input.is_empty() && !self.done {
                if self.remaining > 0 {
                    let take = self.remaining.min(input.len());
                    out.extend_from_slice(&input[..take]);
                    self.remaining -= take;
                    input = &input[take..];
                    continue;
                }
                // Accumulate a framing line: "<hex>\r\n" or the bare "\r\n"
                // that terminates the previous chunk's data.
                self.line.push(input[0]);
                input = &input[1..];
                if self.line.ends_with(b"\r\n") {
                    let t: Vec<u8> = self.line[..self.line.len() - 2].to_vec();
                    self.line.clear();
                    if t.is_empty() {
                        continue; // CRLF after chunk data
                    }
                    let hex = std::str::from_utf8(&t).unwrap_or("");
                    match usize::from_str_radix(hex.trim(), 16) {
                        Ok(0) => self.done = true,
                        Ok(n) => self.remaining = n,
                        Err(_) => {} // tolerate chunk extensions/garbage lines
                    }
                }
            }
        }
    }

    /// Read the status line + headers off `s`, returning (header text, body
    /// bytes already consumed past the header).
    fn read_header(s: &mut TcpStream) -> Res<(String, Vec<u8>)> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            if let Some(i) = find2(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..i]).into_owned();
                return Ok((head, buf.split_off(i + 4)));
            }
            let n = s
                .read(&mut tmp)
                .map_err(|e| err!("client", "read: {}", e))?;
            if n == 0 {
                return Err(err!("client", "connection closed before headers completed"));
            }
            buf.extend_from_slice(&tmp[..n]);
        }
    }

    fn is_chunked(header: &str) -> bool {
        header.lines().any(|l| {
            l.to_ascii_lowercase().starts_with("transfer-encoding:")
                && l.to_ascii_lowercase().contains("chunked")
        })
    }
}

pub mod server {
    //! # api — Ollama-compatible REST server
    //!
    //! A from-scratch HTTP/1.1 server on `std::net::TcpListener` — no framework,
    //! no async runtime. One OS thread per connection; the GPU itself is
    //! serialized behind [`crate::queue::GpuQueue`], so connection threads cost
    //! only a stack while they wait.
    //!
    //! Implemented endpoints (Ollama wire format):
    //! * `POST /api/generate`    — completion, NDJSON streaming by default
    //! * `POST /api/chat`        — chat, NDJSON streaming by default
    //! * `POST /api/embeddings`  (and `/api/embed`) — pooled embedding vector
    //! * `GET  /api/tags`        — locally available models
    //! * `POST /api/show`        — config + details of one model
    //! * `POST /api/pull`        — Hub download with streamed status lines
    //! * `GET  /api/ps`          — resident model + queue depth
    //! * `GET  /api/version`, `GET|HEAD /` — liveness
    //!
    //! Streaming uses `Transfer-Encoding: chunked`, one JSON object per line,
    //! exactly as Ollama clients expect.

    use super::protocol::now_rfc3339;
    use crate::json::{self, Json};
    use crate::models::{ModelConfig, ModelManager};
    use crate::queue::GpuQueue;
    use crate::tokenizer::ChatTurn;

    /// Parsed request media: optional chat turns, plus decoded image and
    /// audio byte-blobs in wire order.
    type PreparedInput = (Option<Vec<ChatTurn>>, Vec<Vec<u8>>, Vec<Vec<u8>>);
    use crate::traits::{Capability, GenOptions, Res};
    use crate::{err, hub, log};
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    /// The Ollama version cima advertises as wire-compatible. `/api/version`
    /// returns THIS, not cima's own version — Ollama clients parse it as a
    /// plain semver to feature-gate, and a prerelease suffix (`-cima`) or an
    /// unfamiliar number makes some clients treat the server as too old and
    /// disable endpoints. cima implements the generate/chat/embed/pull/tags/
    /// show/ps surface, stable in Ollama since well before this baseline, so
    /// a recent stable Ollama semver is the safe thing to report.
    ///
    /// Override at runtime with `CIMA_OLLAMA_VERSION` if a specific client
    /// demands a different minimum; keep it a bare `X.Y.Z`.
    pub const OLLAMA_COMPAT_VERSION: &str = "0.6.0";

    /// cima's own version (build/support identity). Surfaced in logs, the
    /// `cima --version` CLI, and the `engine` field of `/api/version`, so a
    /// human can tell which engine answered without confusing Ollama's
    /// semver parser. Not what a stock Ollama client keys on.
    pub const VERSION: &str = concat!("cima ", env!("CARGO_PKG_VERSION"));

    /// The Ollama-compat semver actually served, honoring the env override.
    fn ollama_version() -> String {
        std::env::var("CIMA_OLLAMA_VERSION")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| OLLAMA_COMPAT_VERSION.to_string())
    }

    /// Maximum accepted request body (64 MiB — base64 media payloads are large).
    const MAX_BODY: usize = 64 << 20;

    /// Shared server state handed to every connection thread.
    pub struct Server {
        pub manager: Mutex<ModelManager>,
        pub queue: GpuQueue,
        /// Startup-pull progress. When `CIMA_PULL_AT_STARTUP` lists models,
        /// the server binds immediately (so `/api/ready` is pollable) but
        /// reports `ready: false` until every listed model is on disk.
        pub startup: Arc<Startup>,
    }

    /// Shared, lock-free view of the background startup pull. An orchestrator
    /// polls `/api/ready`; these fields drive the `ready`/`healthy` bits and
    /// give a human-readable phase for logs and the JSON body.
    #[derive(Default)]
    pub struct Startup {
        /// Models the server was asked to pre-pull (empty ⇒ nothing gated).
        pub required: Vec<String>,
        /// Set once every required model is present (or none were required).
        pub done: std::sync::atomic::AtomicBool,
        /// Set if the pull loop gave up after exhausting retries. The server
        /// stays live and serving whatever IS present, but `/api/ready`
        /// reports the failure so a dependent fails fast instead of hanging.
        pub failed: std::sync::atomic::AtomicBool,
        /// Human-readable last phase, e.g. "pulling 2/3: org/repo:tag".
        pub phase: Mutex<String>,
    }

    impl Startup {
        pub fn ready(&self) -> bool {
            self.done.load(std::sync::atomic::Ordering::Relaxed)
        }
        pub fn failed(&self) -> bool {
            self.failed.load(std::sync::atomic::Ordering::Relaxed)
        }
        pub fn phase(&self) -> String {
            self.phase.lock().map(|p| p.clone()).unwrap_or_default()
        }
        fn set_phase(&self, s: impl Into<String>) {
            if let Ok(mut p) = self.phase.lock() {
                *p = s.into();
            }
        }
    }

    // (RFC3339 UTC timestamps — Ollama's `created_at` — are produced by
    // now_rfc3339 in the protocol module.)

    // ===========================================================================
    // HTTP plumbing
    // ===========================================================================

    /// A parsed inbound request.
    struct Request {
        method: String,
        path: String,
        body: Vec<u8>,
    }

    /// Read and parse one HTTP/1.1 request from the socket.
    fn read_request(stream: &mut TcpStream) -> Res<Request> {
        let mut buf = Vec::with_capacity(4096);
        let mut tmp = [0u8; 4096];
        let header_end;
        loop {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| err!("http", "read: {}", e))?;
            if n == 0 {
                return Err(err!("http", "connection closed mid-request"));
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = p + 4;
                break;
            }
            if buf.len() > 64 << 10 {
                return Err(err!("http", "header block exceeds 64 KiB"));
            }
        }
        let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        let mut lines = head.split("\r\n");
        let req_line = lines.next().unwrap_or("");
        let mut it = req_line.split_whitespace();
        let method = it.next().unwrap_or("").to_uppercase();
        let path = it.next().unwrap_or("/").to_string();

        let mut content_length = 0usize;
        for l in lines {
            let lower = l.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
        if content_length > MAX_BODY {
            return Err(err!(
                "http",
                "body of {} bytes exceeds the {} byte limit",
                content_length,
                MAX_BODY
            ));
        }
        let mut body = buf[header_end..].to_vec();
        while body.len() < content_length {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| err!("http", "read body: {}", e))?;
            if n == 0 {
                return Err(err!("http", "connection closed mid-body"));
            }
            body.extend_from_slice(&tmp[..n]);
        }
        body.truncate(content_length);
        Ok(Request { method, path, body })
    }

    /// Plain (non-streaming) JSON response.
    fn respond_json(stream: &mut TcpStream, status: u16, body: &Json) {
        let payload = body.dump();
        let reason = if status < 400 { "OK" } else { "Error" };
        let _ = write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        reason,
        payload.len(),
        payload
    );
    }

    /// Map an engine error to an Ollama-style `{"error": "..."}` payload.
    fn respond_error(stream: &mut TcpStream, status: u16, msg: &str) {
        respond_json(stream, status, &Json::obj().set("error", Json::s(msg)));
    }

    /// Chunked NDJSON stream writer.
    struct ChunkedWriter<'a> {
        stream: &'a mut TcpStream,
        started: bool,
        dead: bool,
    }

    impl<'a> ChunkedWriter<'a> {
        fn new(stream: &'a mut TcpStream) -> Self {
            ChunkedWriter {
                stream,
                started: false,
                dead: false,
            }
        }
        /// Emit one JSON object as an NDJSON line (lazily sending headers first).
        fn line(&mut self, obj: &Json) {
            if self.dead {
                return;
            }
            if !self.started {
                self.started = true;
                if write!(
                self.stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            .is_err()
            {
                self.dead = true;
                return;
            }
            }
            let mut data = obj.dump();
            data.push('\n');
            if write!(self.stream, "{:x}\r\n{}\r\n", data.len(), data).is_err()
                || self.stream.flush().is_err()
            {
                // Client went away: keep generating silently is wasteful, but the
                // generation loop checks nothing — mark dead so writes stop.
                self.dead = true;
            }
        }
        fn finish(&mut self) {
            if self.started && !self.dead {
                let _ = write!(self.stream, "0\r\n\r\n");
            }
        }
        fn client_gone(&self) -> bool {
            self.dead
        }
    }

    // ===========================================================================
    // Request-body helpers
    // ===========================================================================

    /// Parse the JSON body, with a 400-grade error on malformed input.
    fn parse_body(req: &Request) -> Res<Json> {
        let text = std::str::from_utf8(&req.body)
            .map_err(|_| err!("http", "request body is not UTF-8"))?;
        if text.trim().is_empty() {
            return Ok(Json::obj());
        }
        json::parse(text).map_err(|e| err!("http", "request body is not valid JSON: {}", e))
    }

    /// Extract Ollama `options` into [`GenOptions`] through the same option
    /// table the CLI uses ([`GenOptions::set`]) — one parameter list, every
    /// surface. Unknown keys are logged and skipped (Ollama compatibility:
    /// clients send options we don't implement).
    fn parse_options(body: &Json) -> GenOptions {
        let mut opts = GenOptions::default();
        super::protocol::options_from_json(&mut opts, body);
        opts
    }

    /// Decode the base64 `images` (or `audio`) array of an Ollama request.
    fn parse_media(body: &Json, key: &str) -> Res<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        if let Some(items) = body.arr_of(key) {
            for (i, item) in items.iter().enumerate() {
                let s = item
                    .as_str()
                    .ok_or_else(|| err!("http", "'{}[{}]' must be a base64 string", key, i))?;
                out.push(
                    base64_decode(s)
                        .map_err(|e| err!("http", "'{}[{}]' is not valid base64: {}", key, i, e))?,
                );
            }
        }
        Ok(out)
    }

    /// Minimal strict base64 decoder (standard alphabet, `=` padding).
    fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
        fn val(c: u8) -> Result<u32, String> {
            match c {
                b'A'..=b'Z' => Ok((c - b'A') as u32),
                b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
                b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
                b'+' => Ok(62),
                b'/' => Ok(63),
                _ => Err(format!("invalid character 0x{:02x}", c)),
            }
        }
        let raw: Vec<u8> = s.bytes().filter(|b| !b" \t\r\n".contains(b)).collect();
        let mut out = Vec::with_capacity(raw.len() / 4 * 3);
        let mut i = 0;
        while i < raw.len() {
            let chunk = &raw[i..(i + 4).min(raw.len())];
            if chunk.len() < 2 {
                return Err("truncated base64 quantum".into());
            }
            let pad = chunk.iter().filter(|&&c| c == b'=').count();
            let mut acc = 0u32;
            let n = chunk.len() - pad;
            for (j, &c) in chunk.iter().enumerate().take(n) {
                acc |= val(c)? << (18 - 6 * j);
            }
            out.push((acc >> 16) as u8);
            if n > 2 {
                out.push((acc >> 8) as u8);
            }
            if n > 3 {
                out.push(acc as u8);
            }
            i += 4;
            if pad > 0 {
                break;
            }
        }
        Ok(out)
    }

    // ===========================================================================
    // Endpoint handlers
    // ===========================================================================

    /// Render an Ollama `tools` array into the Hermes-style prompt block that
    /// tool-tuned checkpoints (Qwen 2.5's native dialect; understood broadly)
    /// were trained on: signatures inside <tools>, calls requested inside
    /// <tool_call> tags. Appended to the system turn (created if absent).
    fn tools_prompt_block(tools: &[Json]) -> String {
        let mut b = String::from(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for t in tools {
            b.push('\n');
            b.push_str(&t.dump());
        }
        b.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
        b
    }

    /// Pull `<tool_call>{...}</tool_call>` blocks out of a completion. Returns
    /// the surrounding text (the assistant's prose, if any) and the parsed
    /// calls in Ollama's wire shape: `{"function": {"name", "arguments"}}`.
    /// Blocks that fail to parse as JSON stay in the text untouched.
    /// Byte offset one past the first complete brace-balanced JSON object in
    /// `s` (starting at the first `{`), or None if it never balances. Brace
    /// counting ignores `{`/`}` inside double-quoted strings and honours `\`
    /// escapes — enough to delimit a `<tool_call>` object that a model emitted
    /// without a closing tag, without pulling in trailing prose.
    fn balanced_json_end(s: &str) -> Option<usize> {
        let b = s.as_bytes();
        let start = b.iter().position(|&c| c == b'{')?;
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        for (i, &c) in b.iter().enumerate().skip(start) {
            if in_str {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn extract_tool_calls(full: &str) -> (String, Vec<Json>) {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        let mut text = String::new();
        let mut calls = Vec::new();
        let mut rest = full;
        // Push one parsed call; returns false if the JSON wasn't a tool call.
        let push_call = |calls: &mut Vec<Json>, inner: &str| -> bool {
            match crate::json::parse(inner.trim()) {
                Ok(j) if j.str_of("name").is_some() => {
                    let name = j.str_of("name").unwrap_or_default().to_string();
                    let args = j.get("arguments").cloned().unwrap_or(Json::obj());
                    calls.push(
                        Json::obj().set(
                            "function",
                            Json::obj()
                                .set("name", Json::s(&name))
                                .set("arguments", args),
                        ),
                    );
                    true
                }
                _ => false,
            }
        };
        while let Some(o) = rest.find(OPEN) {
            let after = &rest[o + OPEN.len()..];
            if let Some(c) = after.find(CLOSE) {
                // Well-formed: open ... close.
                if push_call(&mut calls, &after[..c]) {
                    text.push_str(&rest[..o]);
                } else {
                    text.push_str(&rest[..o + OPEN.len() + c + CLOSE.len()]);
                }
                rest = &after[c + CLOSE.len()..];
            } else {
                // No close tag — a very common model behaviour (emits the
                // open tag + JSON, then keeps talking). Parse the first
                // balanced-brace JSON object after the tag; anything past it
                // (prose, extra text) stays in content.
                match balanced_json_end(after) {
                    Some(end) if push_call(&mut calls, &after[..end]) => {
                        text.push_str(&rest[..o]);
                        rest = &after[end..];
                    }
                    _ => {
                        // Not parseable as a call: leave the open tag in text
                        // and move past it to avoid an infinite loop.
                        text.push_str(&rest[..o + OPEN.len()]);
                        rest = after;
                    }
                }
            }
        }
        text.push_str(rest);
        (text.trim().to_string(), calls)
    }

    /// `format` field: `"json"` → syntax-constrained decoding; a JSON-schema
    /// object → the same constraint plus the schema injected into the prompt
    /// (the decoder guarantees valid JSON; key/type conformance is steered by
    /// the schema text — the honest extent of schema support today).
    fn parse_format(body: &Json) -> (bool, Option<String>) {
        match body.get("format") {
            Some(Json::Str(s)) if s == "json" => (true, None),
            Some(j @ Json::Obj(_)) => (true, Some(j.dump())),
            _ => (false, None),
        }
    }

    /// `POST /api/generate` and `POST /api/chat` share one execution path.
    fn handle_generate(server: &Arc<Server>, stream: &mut TcpStream, body: Json, chat: bool) {
        let model = match body.str_of("model") {
            Some(m) => m.to_string(),
            None => return respond_error(stream, 400, "missing required field 'model'"),
        };
        let streaming = body.bool_of("stream").unwrap_or(true);
        let keep_alive = crate::models::KeepAlive::parse(body.get("keep_alive"));
        let mut opts = parse_options(&body);
        let (json_mode, schema) = parse_format(&body);
        opts.json_mode = json_mode;
        opts.json_schema = schema.clone();
        let fmt_instr = if json_mode {
            Some(match &schema {
                Some(sch) => format!(
                    "\n\nRespond using JSON. The JSON object must conform to this JSON Schema:\n{}",
                    sch
                ),
                None => "\n\nRespond using JSON.".to_string(),
            })
        } else {
            None
        };
        let tools: Vec<Json> = if chat {
            body.arr_of("tools").map(|t| t.to_vec()).unwrap_or_default()
        } else {
            Vec::new()
        };
        let tools_active = !tools.is_empty();

        // Ollama's load/unload idiom: an empty request body (no prompt, no
        // messages) loads the model into VRAM — or releases it when
        // `keep_alive` is 0 — and returns a single done response.
        let empty_req = body.str_of("prompt").map(|p| p.is_empty()).unwrap_or(!chat)
            && body
                .arr_of("messages")
                .map(|m| m.is_empty())
                .unwrap_or(!chat);
        if empty_req {
            let mut mgr = server.manager.lock().unwrap();
            if keep_alive == crate::models::KeepAlive::Now {
                if mgr
                    .current
                    .as_ref()
                    .map(|m| m.name == model)
                    .unwrap_or(false)
                {
                    mgr.evict();
                }
                let j = super::protocol::stream_chunk(&model, chat, "", true)
                    .set("done_reason", Json::s("unload"));
                return respond_json(stream, 200, &j);
            }
            // Time the load here too: this path is the load idiom clients
            // (and benchmarks) hit on purpose — Ollama reports durations on it.
            let t_load = std::time::Instant::now();
            if let Err(e) = mgr.ensure(&model) {
                return respond_error(stream, 500, &e.to_string());
            }
            let load_ns = t_load.elapsed().as_secs_f64() * 1e9;
            mgr.touch(keep_alive);
            let j = super::protocol::stream_chunk(&model, chat, "", true)
                .set("done_reason", Json::s("load"))
                .set("load_duration", Json::n(load_ns))
                .set("total_duration", Json::n(load_ns));
            return respond_json(stream, 200, &j);
        }

        // Build the prompt text + media payloads from either request shape.
        let (prompt_is_chat, images, audio): PreparedInput = if chat {
            let mut turns: Vec<ChatTurn> = match body.arr_of("messages") {
                Some(msgs) => msgs
                    .iter()
                    .map(|m| {
                        let mut role = m.str_of("role").unwrap_or("user").to_string();
                        let mut content = m.str_of("content").unwrap_or("").to_string();
                        // Tool results ride back as user-visible observations in
                        // the dialect the tools block establishes; assistant
                        // turns that carried tool_calls are re-rendered the same
                        // way, so the transcript round-trips faithfully.
                        if role == "tool" {
                            role = "user".into();
                            content = format!("<tool_response>\n{}\n</tool_response>", content);
                        }
                        if let Some(tcs) = m.arr_of("tool_calls") {
                            for tc in tcs {
                                if let Some(f) = tc.get("function") {
                                    content.push_str(&format!(
                                        "\n<tool_call>\n{}\n</tool_call>",
                                        f.dump()
                                    ));
                                }
                            }
                        }
                        ChatTurn {
                            role,
                            content,
                            n_images: m.arr_of("images").map(|a| a.len()).unwrap_or(0),
                            n_audio: m.arr_of("audio").map(|a| a.len()).unwrap_or(0),
                        }
                    })
                    .collect(),
                None => return respond_error(stream, 400, "missing required field 'messages'"),
            };
            if tools_active {
                let block = tools_prompt_block(&tools);
                match turns.iter_mut().find(|t| t.role == "system") {
                    Some(sys) => sys.content.push_str(&block),
                    None => turns.insert(
                        0,
                        ChatTurn {
                            role: "system".into(),
                            content: format!("You are a helpful assistant.{}", block),
                            n_images: 0,
                            n_audio: 0,
                        },
                    ),
                }
            }
            if let Some(instr) = &fmt_instr {
                if let Some(last_user) = turns.iter_mut().rev().find(|t| t.role == "user") {
                    last_user.content.push_str(instr);
                }
            }
            let mut imgs = Vec::new();
            let mut auds = Vec::new();
            for m in body.arr_of("messages").unwrap_or(&[]) {
                match parse_media(m, "images") {
                    Ok(v) => imgs.extend(v),
                    Err(e) => return respond_error(stream, 400, &e.to_string()),
                }
                match parse_media(m, "audio") {
                    Ok(v) => auds.extend(v),
                    Err(e) => return respond_error(stream, 400, &e.to_string()),
                }
            }
            (Some(turns), imgs, auds)
        } else {
            let imgs = match parse_media(&body, "images") {
                Ok(v) => v,
                Err(e) => return respond_error(stream, 400, &e.to_string()),
            };
            let auds = match parse_media(&body, "audio") {
                Ok(v) => v,
                Err(e) => return respond_error(stream, 400, &e.to_string()),
            };
            (None, imgs, auds)
        };

        // ---- GPU admission: strict FIFO, wait time measured; bounded
        // when CIMA_MAX_QUEUE is set (fast 429 over invisible latency) ----
        let permit = match server.queue.try_acquire() {
            Ok(p) => p,
            Err(depth) => {
                log::warn(&format!(
                    "request refused: model={} queue depth {} at CIMA_MAX_QUEUE cap",
                    model, depth
                ));
                return respond_error(
                    stream,
                    429,
                    "server busy: request queue is full (CIMA_MAX_QUEUE)",
                );
            }
        };
        log::info(&format!(
            "request admitted: model={} queue_wait={:.1}ms depth_at_enqueue={}",
            model, permit.wait_ms, permit.depth_at_enqueue
        ));
        let mut mgr = server.manager.lock().unwrap();
        // Ollama semantics: load_duration = time this request spent making the
        // model resident. ensure() is exactly that span — cold hit = full
        // weight load, warm hit = a residency check (small but nonzero).
        let t_load = std::time::Instant::now();
        let lm = match mgr.ensure(&model) {
            Ok(m) => m,
            Err(e) => return respond_error(stream, 404, &e.to_string()),
        };
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
        if !lm.capabilities().contains(&Capability::Generate) {
            return respond_error(
                stream,
                400,
                &format!(
                    "model '{}' cannot generate (capabilities: {}) — use /api/embeddings",
                    model,
                    lm.capabilities()
                        .iter()
                        .map(|c| c.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }

        // Ollama semantics: /api/generate renders the model's chat template
        // around the prompt (a single user turn) unless `raw: true`. Instruct
        // models degenerate on untemplated input — the symptom is
        // prompt_eval_count barely above the raw token count.
        let raw = body.bool_of("raw").unwrap_or(false);
        let prompt_text = match &prompt_is_chat {
            Some(turns) => lm.render_chat(turns),
            None => match body.str_of("prompt") {
                Some(p) if raw => format!("{}{}", p, fmt_instr.as_deref().unwrap_or("")),
                Some(p) => {
                    let turn = ChatTurn {
                        role: "user".into(),
                        content: format!("{}{}", p, fmt_instr.as_deref().unwrap_or("")),
                        n_images: images.len(),
                        n_audio: audio.len(),
                    };
                    lm.render_chat(std::slice::from_ref(&turn))
                }
                None => return respond_error(stream, 400, "missing required field 'prompt'"),
            },
        };

        let prepared = match lm.prepare(&prompt_text, &images, &audio) {
            Ok(p) => p,
            Err(e) => return respond_error(stream, 400, &e.to_string()),
        };

        // ---- token loop with NDJSON streaming ----
        let mut w = ChunkedWriter::new(stream);
        let mut full = String::new();
        let model_name = model.clone();
        let stats = lm.generate(&prepared, &opts, permit.wait_ms, |piece| {
            full.push_str(piece);
            if streaming && !tools_active {
                let obj = if chat {
                    Json::obj()
                        .set("model", Json::s(&model_name))
                        .set("created_at", Json::s(&now_rfc3339()))
                        .set(
                            "message",
                            Json::obj()
                                .set("role", Json::s("assistant"))
                                .set("content", Json::s(piece)),
                        )
                        .set("done", Json::b(false))
                } else {
                    Json::obj()
                        .set("model", Json::s(&model_name))
                        .set("created_at", Json::s(&now_rfc3339()))
                        .set("response", Json::s(piece))
                        .set("done", Json::b(false))
                };
                w.line(&obj);
            }
        });
        let stats = match stats {
            Ok(s) => s,
            Err(e) => {
                if streaming && w.client_gone() {
                    return;
                }
                return respond_error(stream, 500, &e.to_string());
            }
        };

        // Tool-calling chats: lift <tool_call> blocks out of the completion
        // into Ollama's structured `message.tool_calls`; the remaining prose
        // (if any) stays in `content`. Tool parsing works on the raw stream
        // (`full`); everything else uses the stop-trimmed `stats.text`.
        let (content_out, tool_calls) = if chat && tools_active {
            extract_tool_calls(&full)
        } else {
            (String::new(), Vec::new())
        };
        let final_text = if chat && tools_active {
            content_out.as_str()
        } else {
            // Authoritative, stop-sequence-trimmed text (not the raw callback
            // accumulation, which may still contain a matched stop string).
            stats.text.as_str()
        };

        // Final frame (Ollama duration fields are nanoseconds).
        let mut fin = super::protocol::final_chunk(
            &model,
            chat,
            if streaming && !tools_active {
                None
            } else {
                Some(final_text)
            },
            stats.prompt_tokens,
            stats.gen_tokens,
            stats.total_ms,
            stats.ttft_ms,
            load_ms,
            stats.stop_reason,
        );
        if !tool_calls.is_empty() {
            if let Json::Obj(ref mut top) = fin {
                for (k, v) in top.iter_mut() {
                    if k == "message" {
                        if let Json::Obj(ref mut msg) = v {
                            msg.push(("tool_calls".to_string(), Json::Arr(tool_calls.clone())));
                        }
                    }
                }
            }
        }
        mgr.touch(keep_alive);
        if streaming {
            w.line(&fin);
            w.finish();
        } else {
            respond_json(stream, 200, &fin);
        }
    }

    /// `POST /api/embeddings` (`prompt`) and `POST /api/embed` (`input`).
    fn handle_embeddings(server: &Arc<Server>, stream: &mut TcpStream, body: Json, legacy: bool) {
        let keep_alive = crate::models::KeepAlive::parse(body.get("keep_alive"));
        let model = match body.str_of("model") {
            Some(m) => m.to_string(),
            None => return respond_error(stream, 400, "missing required field 'model'"),
        };
        // legacy: {"prompt": str}. current: {"input": str | [str, ...]}.
        let mut texts: Vec<String> = Vec::new();
        if let Some(p) = body.str_of("prompt") {
            texts.push(p.to_string());
        } else if let Some(i) = body.get("input") {
            if let Some(one) = i.as_str() {
                texts.push(one.to_string());
            } else if let Some(arr) = i.as_arr() {
                for v in arr {
                    match v.as_str() {
                        Some(t) => texts.push(t.to_string()),
                        None => {
                            return respond_error(
                                stream,
                                400,
                                "'input' array must contain only strings",
                            )
                        }
                    }
                }
            }
        }
        if texts.is_empty() {
            return respond_error(stream, 400, "missing 'prompt' (or 'input') field");
        }
        let permit = match server.queue.try_acquire() {
            Ok(p) => p,
            Err(_) => {
                return respond_error(
                    stream,
                    429,
                    "server busy: request queue is full (CIMA_MAX_QUEUE)",
                )
            }
        };
        let mut mgr = server.manager.lock().unwrap();
        let lm = match mgr.ensure(&model) {
            Ok(m) => m,
            Err(e) => return respond_error(stream, 404, &e.to_string()),
        };
        let t0 = std::time::Instant::now();
        let mut vecs = Vec::with_capacity(texts.len());
        for text in &texts {
            match lm.embed(text) {
                Ok(v) => vecs.push(v),
                Err(e) => return respond_error(stream, 500, &e.to_string()),
            }
        }
        log::metric(
            "embedding",
            &[
                ("model", model.clone()),
                ("queue_wait_ms", format!("{:.2}", permit.wait_ms)),
                (
                    "total_ms",
                    format!("{:.2}", t0.elapsed().as_secs_f64() * 1e3),
                ),
                ("inputs", vecs.len().to_string()),
                (
                    "dim",
                    vecs.first().map(|v| v.len()).unwrap_or(0).to_string(),
                ),
            ],
        );
        mgr.touch(keep_alive);
        let to_arr = |v: &Vec<f32>| Json::Arr(v.iter().map(|&x| Json::n(x as f64)).collect());
        if legacy {
            // Legacy endpoint stays single-vector: first input.
            respond_json(stream, 200, &Json::obj().set("embedding", to_arr(&vecs[0])));
        } else {
            let embs = Json::Arr(vecs.iter().map(to_arr).collect());
            respond_json(
                stream,
                200,
                &Json::obj()
                    .set("model", Json::s(&model))
                    .set("embeddings", embs)
                    .set(
                        "total_duration",
                        Json::n((t0.elapsed().as_nanos() as f64).round()),
                    ),
            );
        }
    }

    /// `GET /api/tags` — list local models with sizes and timestamps.
    /// `GET /api/available` — the curated registry of vetted, pull-ready
    /// models (the HTTP twin of the `cima available` CLI). Unlike `/api/tags`
    /// (what is on local disk), this lists what can be pulled, with the same
    /// id/family/size/capabilities/status/notes the CLI table shows. A
    /// `local` boolean flags rows already present so a UI can show a
    /// pull-vs-run affordance.
    fn handle_available(stream: &mut TcpStream) {
        let present: std::collections::HashSet<String> = hub::list_local_caps()
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        let models: Vec<Json> = hub::registry::REGISTRY
            .iter()
            .map(|e| {
                Json::obj()
                    .set("id", Json::s(e.id))
                    .set("model", Json::s(e.id))
                    .set("family", Json::s(e.family))
                    .set("size", Json::s(e.size))
                    .set("capabilities", Json::s(e.capabilities))
                    .set("status", Json::s(e.status))
                    .set("notes", Json::s(e.notes))
                    .set("local", Json::b(present.contains(e.id)))
            })
            .collect();
        respond_json(stream, 200, &Json::obj().set("models", Json::Arr(models)));
    }

    fn handle_tags(stream: &mut TcpStream) {
        let mut models = Vec::new();
        for (name, size, mtime, caps) in hub::list_local_caps() {
            let secs = mtime
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // A quant tag (`:q8_0`) marks a GGUF checkpoint; a bare name is
            // safetensors. `caps` (text / text+vision+audio) is disk-truth
            // from list_local_caps, so surface it as the family rather than
            // hardcoding "transformer".
            let format = if name.contains(':') {
                "gguf"
            } else {
                "safetensors"
            };
            models.push(
                Json::obj()
                    .set("name", Json::s(&name))
                    .set("model", Json::s(&name))
                    .set("size", Json::u(size))
                    .set("digest", Json::s(""))
                    .set("modified_at", Json::s(&format!("{}", secs)))
                    .set(
                        "details",
                        Json::obj()
                            .set("format", Json::s(format))
                            .set("family", Json::s(caps)),
                    ),
            );
        }
        respond_json(stream, 200, &Json::obj().set("models", Json::Arr(models)));
    }

    /// `POST /api/show` — config.json passthrough plus engine details.
    fn handle_show(stream: &mut TcpStream, body: Json) {
        let model = match body.str_of("model").or_else(|| body.str_of("name")) {
            Some(m) => m.to_string(),
            None => return respond_error(stream, 400, "missing required field 'model'"),
        };
        // Presence is format-aware: only a genuinely absent model 404s.
        if !hub::is_local(&model) {
            return respond_error(
                stream,
                404,
                &format!("model '{}' not found — run `cima pull {}`", model, model),
            );
        }
        let dir = hub::local_dir(&model);
        // GGUF checkpoints carry no config.json; their architecture is read
        // from the file header at load. Report what disk truth allows rather
        // than 404ing a model that is in fact present.
        let cfg_text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_default();
        let cfg = json::parse(&cfg_text).unwrap_or(Json::Null);
        let arch = cfg.str_of("model_type").unwrap_or("gguf").to_string();
        // The chat template source, as shipped by the checkpoint (the renderer
        // detects its family from this same string). Families share templates
        // across sizes, so this is also how you verify two repos prompt alike.
        let template = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| json::parse(&t).ok())
            .and_then(|j| {
                j.get("chat_template").map(|c| match c {
                    Json::Str(s) => s.clone(),
                    other => other.dump(),
                })
            })
            .unwrap_or_default();
        respond_json(
            stream,
            200,
            &Json::obj()
                .set(
                    "modelfile",
                    Json::s(&format!("# pulled from https://huggingface.co/{}", model)),
                )
                .set("parameters", Json::s(""))
                .set("template", Json::s(&template))
                .set(
                    "details",
                    Json::obj()
                        .set("format", Json::s("safetensors"))
                        .set("family", Json::s(&arch)),
                )
                .set("model_info", cfg),
        );
    }

    /// `POST /api/pull` — streamed download status (Ollama clients poll lines).
    /// Extensions beyond Ollama: optional `"include"` (substring filter selecting
    /// one quantization of a multi-quant repo) and `"force"` (skip the
    /// architecture preflight gate).
    fn handle_pull(stream: &mut TcpStream, body: Json) {
        let model = match body.str_of("model").or_else(|| body.str_of("name")) {
            Some(m) => m.to_string(),
            None => return respond_error(stream, 400, "missing required field 'model'"),
        };
        // `ORG/REPO:TAG` selects a GGUF quantization. Keep the tag on `model`
        // for the download itself — hub::pull maps the FULL selector to the
        // storage dir (`ORG__REPO@TAG`), which is exactly where is_local and
        // the load path look. Pre-stripping it here (passing the bare repo)
        // would download into the untagged dir and then read as "not
        // present". `repo` is used only for the preflight file listing.
        let (repo, tag) = match model.split_once(':') {
            Some((r, t)) => (r.to_string(), Some(t.to_string())),
            None => (model.clone(), None),
        };
        let include = body.str_of("include").map(str::to_string).or(tag);
        let force = body.bool_of("force").unwrap_or(false);
        let mut w = ChunkedWriter::new(stream);
        w.line(&Json::obj().set("status", Json::s("pulling manifest")));

        // Preflight, format-aware (mirrors the CLI): GGUF repos have no
        // config.json — their gate is the file list + tag match; the
        // architecture inside the file is validated at load.
        if !force {
            let gguf_gate = hub::list_repo(&repo, Some(".gguf")).map(|files| {
                let ggufs: Vec<String> = files
                    .into_iter()
                    .map(|(n, _)| n)
                    .filter(|n| n.ends_with(".gguf"))
                    .collect();
                (!ggufs.is_empty(), ggufs)
            });
            let gate = match gguf_gate {
                Ok((true, ggufs)) => match include.as_deref() {
                    Some(t)
                        if ggufs
                            .iter()
                            .any(|n| n.to_ascii_lowercase().contains(&t.to_ascii_lowercase())) =>
                    {
                        Ok(())
                    }
                    Some(t) => Err(crate::err!(
                        "hub",
                        "no .gguf matches '{}'. Available: {}",
                        t,
                        ggufs.join(", ")
                    )),
                    None if ggufs.len() == 1 => Ok(()),
                    None => Err(crate::err!(
                        "hub",
                        "{} quantizations — pick one with a :TAG suffix: {}",
                        ggufs.len(),
                        ggufs.join(", ")
                    )),
                },
                _ => hub::pull_config(&repo).and_then(|dir| ModelConfig::load(&dir).map(|_| ())),
            };
            if let Err(e) = gate {
                w.line(&Json::obj().set(
                    "error",
                    Json::s(&format!(
                        "preflight rejected (set \"force\":true to override): {}",
                        e
                    )),
                ));
                w.finish();
                return;
            }
            w.line(&Json::obj().set("status", Json::s("preflight ok")));
        }
        match hub::pull(&model, false, include.as_deref()) {
            Ok(()) => {
                w.line(&Json::obj().set("status", Json::s("success")));
                w.finish();
            }
            Err(e) => {
                w.line(&Json::obj().set("error", Json::s(&e.to_string())));
                w.finish();
            }
        }
    }

    /// `GET /api/ready` — liveness + a snapshot of what's on disk.
    /// `POST /api/ready` with `{"models": ["ORG/REPO:TAG", ...]}` — the
    /// orchestration check: reports per-model presence and an overall
    /// `ready` bool that is true only when every requested model is local.
    ///
    /// Designed for a start-up gate: a dependent service pulls its models
    /// (CLI `cima pull` or `POST /api/pull`), then polls this until
    /// `ready == true`. Presence is disk-truth and format-aware (safetensors
    /// or GGUF), so a pulled GGUF reports ready — unlike `/api/show`, which
    /// historically only recognized safetensors. No network, no GPU load.
    fn handle_ready(server: &Arc<Server>, stream: &mut TcpStream, body: Json) {
        // `healthy` means the server is serving and its model manager lock is
        // reachable (not poisoned). A momentary contention (WouldBlock) still
        // counts as healthy — someone is actively using it.
        let healthy = match server.manager.try_lock() {
            Ok(_) => true,
            Err(std::sync::TryLockError::WouldBlock) => true,
            Err(std::sync::TryLockError::Poisoned(_)) => false,
        };

        // Startup-pull gate: if models were required at boot, `ready` is
        // false until they are all present. A hard failure (retries
        // exhausted) is surfaced explicitly so a dependent fails fast.
        let startup_done = server.startup.ready();
        let startup_failed = server.startup.failed();
        let startup_phase = server.startup.phase();
        let gated = !server.startup.required.is_empty();

        let requested: Vec<String> = body
            .arr_of("models")
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();

        if requested.is_empty() {
            // Liveness form: also surface what's already local, so a caller
            // can see progress without diffing /api/tags themselves.
            let local: Vec<Json> = hub::list_local()
                .into_iter()
                .map(|(n, _, _)| Json::s(&n))
                .collect();
            let ready = healthy && startup_done && !startup_failed;
            let mut j = Json::obj()
                .set("healthy", Json::b(healthy))
                .set("ready", Json::b(ready))
                .set("models_present", Json::Arr(local));
            if gated {
                j = j
                    .set("startup_pull", Json::b(true))
                    .set("startup_complete", Json::b(startup_done))
                    .set("startup_phase", Json::s(&startup_phase));
                if startup_failed {
                    j = j.set("error", Json::s(&startup_phase));
                }
            }
            return respond_json(stream, if ready { 200 } else { 503 }, &j);
        }

        // Readiness form: per-model presence + overall gate.
        let mut per_model = Vec::with_capacity(requested.len());
        let mut all_present = true;
        for m in &requested {
            let present = hub::is_local(m);
            all_present &= present;
            per_model.push(
                Json::obj()
                    .set("model", Json::s(m))
                    .set("present", Json::b(present)),
            );
        }
        // A caller asking about specific models is gated on THOSE models,
        // but a hard startup-pull failure still counts against readiness so
        // an operator error is not masked by a coincidentally-present model.
        let ready = healthy && all_present && !startup_failed;
        let mut j = Json::obj()
            .set("healthy", Json::b(healthy))
            .set("ready", Json::b(ready))
            .set("models", Json::Arr(per_model));
        if startup_failed {
            j = j.set("error", Json::s(&startup_phase));
        }
        // 200 when ready, 503 while a dependent should keep waiting — so a
        // plain HTTP health probe works without parsing the body.
        respond_json(stream, if ready { 200 } else { 503 }, &j);
    }

    /// `GET /api/ps` — resident model and queue depth.
    fn handle_ps(server: &Arc<Server>, stream: &mut TcpStream) {
        let mut models = Vec::new();
        if let Ok(mgr) = server.manager.try_lock() {
            if let Some(m) = &mgr.current {
                let entry =
                    super::protocol::ps_entry(&m.name, m.arch.vram_bytes(), mgr.expires_in_secs())
                        .set(
                            "modality",
                            Json::s(&format!("{:?}", m.modality()).to_lowercase()),
                        );
                models.push(entry);
            }
        }
        let j = Json::obj()
            .set("models", Json::Arr(models))
            .set("queue_depth", Json::n(server.queue.depth() as f64));
        respond_json(stream, 200, &j);
    }

    /// `DELETE /api/delete` — remove a model from local disk (evicting it
    /// first when resident). This frees `./models/ORG__REPO`; the registry
    /// entry (what `cima available` shows) is unaffected.
    fn handle_delete(server: &Arc<Server>, stream: &mut TcpStream, body: Json) {
        let Some(model) = body.str_of("model").map(str::to_owned) else {
            return respond_error(stream, 400, "missing required field 'model'");
        };
        {
            let mut mgr = server.manager.lock().unwrap();
            if mgr
                .current
                .as_ref()
                .map(|m| m.name == model)
                .unwrap_or(false)
            {
                mgr.evict();
            }
        }
        // Presence is format-aware: safetensors snapshots carry config.json,
        // GGUF snapshots carry .gguf files and nothing else.
        let repo = model.split(':').next().unwrap_or(&model);
        let dir = crate::hub::local_dir(repo);
        let has_gguf = std::fs::read_dir(&dir)
            .map(|d| {
                d.flatten()
                    .any(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
            })
            .unwrap_or(false);
        if !dir.join("config.json").exists() && !has_gguf {
            return respond_error(
                stream,
                404,
                &format!("model '{}' is not on local disk", model),
            );
        }
        match std::fs::remove_dir_all(&dir) {
            Ok(_) => respond_json(stream, 200, &Json::obj().set("status", Json::s("deleted"))),
            Err(e) => respond_error(stream, 500, &format!("delete failed: {}", e)),
        }
    }

    // ===========================================================================
    // Server loop
    // ===========================================================================

    /// Route one connection.
    fn handle_connection(server: Arc<Server>, mut stream: TcpStream) {
        let req = match read_request(&mut stream) {
            Ok(r) => r,
            Err(e) => return respond_error(&mut stream, 400, &e.to_string()),
        };
        let body = match parse_body(&req) {
            Ok(b) => b,
            Err(e) => return respond_error(&mut stream, 400, &e.to_string()),
        };
        log::info(&format!("{} {}", req.method, req.path));
        match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/api/generate") => handle_generate(&server, &mut stream, body, false),
        ("POST", "/api/chat") => handle_generate(&server, &mut stream, body, true),
        // Two wire contracts: legacy /api/embeddings ({"prompt"} →
        // {"embedding": [...]}) and current /api/embed ({"input": str |
        // [str]} → {"embeddings": [[...], ...]}).
        ("POST", "/api/embeddings") => handle_embeddings(&server, &mut stream, body, true),
        ("POST", "/api/embed") => handle_embeddings(&server, &mut stream, body, false),
        ("GET", "/api/tags") => handle_tags(&mut stream),
        ("GET", "/api/available") => handle_available(&mut stream),
        ("POST", "/api/show") => handle_show(&mut stream, body),
        ("POST", "/api/pull") => handle_pull(&mut stream, body),
        ("GET", "/api/ps") => handle_ps(&server, &mut stream),
        ("GET", "/api/ready") | ("POST", "/api/ready") => handle_ready(&server, &mut stream, body),
        ("DELETE", "/api/delete") | ("POST", "/api/delete") => handle_delete(&server, &mut stream, body),
        // Ollama endpoints that don't map onto Hugging Face-native models:
        // answer 501 with the reason instead of a mute 404, so clients
        // built for ollama fail informatively.
        ("POST", "/api/create") | ("POST", "/api/push") | ("POST", "/api/copy") => respond_error(
            &mut stream,
            501,
            "not implemented: cima serves Hugging Face repositories directly (no Modelfile layer); use `cima pull ORG/REPO`",
        ),
        (_, p) if p.starts_with("/api/blobs") => respond_error(
            &mut stream,
            501,
            "not implemented: cima has no blob store; weights live as Hugging Face snapshots under ./models",
        ),
        ("GET", "/api/version") => respond_json(
            &mut stream,
            200,
            // `version` is the Ollama-compatible semver clients gate on;
            // `engine` names the real server for humans and tooling that
            // care. Ollama clients ignore the extra field.
            &Json::obj()
                .set("version", Json::s(&ollama_version()))
                .set("engine", Json::s(VERSION)),
        ),
        ("GET", "/") | ("HEAD", "/") => {
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 18\r\nConnection: close\r\n\r\nInferno is running"
            );
        }
        _ => respond_error(&mut stream, 404, &format!("no route for {} {}", req.method, req.path)),
    }
    }

    /// Parse `CIMA_PULL_AT_STARTUP` (comma-separated `ORG/REPO[:TAG]`) into a
    /// clean, de-duplicated list. Blank entries and surrounding whitespace
    /// are ignored so a trailing comma or `" a , b "` is harmless.
    pub fn startup_models_from_env() -> Vec<String> {
        std::env::var("CIMA_PULL_AT_STARTUP")
            .ok()
            .map(|s| {
                let mut seen = std::collections::HashSet::new();
                s.split(',')
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                    .filter(|m| seen.insert(m.to_string()))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Launch the background pull loop. The `Startup` state was populated
    /// with the required list at construction; here we drive it to `done`
    /// (or, after exhausting retries, `failed`). Idempotent per model:
    /// anything already on disk is skipped, so a restart with a warm volume
    /// reaches ready almost immediately.
    fn spawn_startup_pull(server: Arc<Server>) {
        let startup = server.startup.clone();
        if startup.required.is_empty() {
            // Nothing gated — ready is simply liveness.
            startup
                .done
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        std::thread::spawn(move || {
            let models = startup.required.clone();
            let total = models.len();
            log::info(&format!(
                "startup pull: {} model(s) required before ready: {}",
                total,
                models.join(", ")
            ));
            for (i, model) in models.iter().enumerate() {
                if hub::is_local(model) {
                    log::info(&format!(
                        "startup pull [{}/{}]: {} already present",
                        i + 1,
                        total,
                        model
                    ));
                    continue;
                }
                // Bounded ret/ry with capped exponential backoff. A transient
                // Hub hiccup shouldn't strand the whole deployment, and an
                // unbounded loop would hide a genuine misconfiguration (typo'd
                // repo, missing HF_TOKEN) behind an eternal "not ready".
                const MAX_ATTEMPTS: u32 = 5;
                let mut attempt = 0;
                let mut ok = false;
                while attempt < MAX_ATTEMPTS {
                    attempt += 1;
                    startup.set_phase(format!(
                        "pulling {}/{}: {} (attempt {}/{})",
                        i + 1,
                        total,
                        model,
                        attempt,
                        MAX_ATTEMPTS
                    ));
                    log::info(&format!(
                        "startup pull [{}/{}]: fetching {} (attempt {}/{})",
                        i + 1,
                        total,
                        model,
                        attempt,
                        MAX_ATTEMPTS
                    ));
                    // Catch panics so a bug in the pull path degrades to a
                    // logged failure, never taking down the serving thread.
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        hub::pull(model, false, None)
                    }));
                    match res {
                        Ok(Ok(())) if hub::is_local(model) => {
                            ok = true;
                            break;
                        }
                        Ok(Ok(())) => {
                            // pull returned success but disk-check disagrees;
                            // treat as a failed attempt and retry.
                            log::warn(&format!(
                                "startup pull [{}/{}]: {} reported complete but is not on disk",
                                i + 1,
                                total,
                                model
                            ));
                        }
                        Ok(Err(e)) => {
                            log::error(&format!(
                                "startup pull [{}/{}]: {} failed: {}",
                                i + 1,
                                total,
                                model,
                                e
                            ));
                        }
                        Err(_) => {
                            log::error(&format!(
                                "startup pull [{}/{}]: {} panicked during fetch",
                                i + 1,
                                total,
                                model
                            ));
                        }
                    }
                    if attempt < MAX_ATTEMPTS {
                        let backoff = 2u64.saturating_pow(attempt).min(60);
                        std::thread::sleep(std::time::Duration::from_secs(backoff));
                    }
                }
                if !ok {
                    startup
                        .failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    startup.set_phase(format!(
                        "failed on {} after {} attempts — see logs",
                        model, MAX_ATTEMPTS
                    ));
                    log::error(&format!(
                        "startup pull aborted: {} could not be fetched after {} attempts; \
                         server stays live but /api/ready will report not-ready",
                        model, MAX_ATTEMPTS
                    ));
                    return;
                }
            }
            startup.set_phase("complete".to_string());
            startup
                .done
                .store(true, std::sync::atomic::Ordering::Relaxed);
            log::info(&format!(
                "startup pull: all {} model(s) present — ready",
                total
            ));
        });
    }

    /// Bind and serve forever (one thread per connection; GPU FIFO inside).
    pub fn serve(server: Server, host: &str, port: u16) -> Res<()> {
        let addr = format!("{}:{}", host, port);
        let listener =
            TcpListener::bind(&addr).map_err(|e| err!("http", "bind {}: {}", addr, e))?;
        println!("cima listening on http://{}", addr);
        log::info(&format!("cima listening on http://{}", addr));
        let shared = Arc::new(server);
        // Keep-alive sweeper: releases the resident model once its window
        // elapses (default 5 minutes; per-request `keep_alive` overrides).
        {
            let server = shared.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                if let Ok(mut mgr) = server.manager.try_lock() {
                    mgr.sweep();
                }
            });
        }
        // Startup pull: fetch every model in `CIMA_PULL_AT_STARTUP` before
        // reporting ready. Runs in the background so the server is already
        // bound and `/api/ready` is pollable throughout. Never panics — a
        // pull failure is retried with backoff and, if it ultimately fails,
        // surfaced as `ready:false, error:...` rather than a crash or hang.
        spawn_startup_pull(shared.clone());
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let s = shared.clone();
                    std::thread::spawn(move || handle_connection(s, stream));
                }
                Err(e) => log::warn(&format!("accept failed: {}", e)),
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod startup_tests {
        use super::{ollama_version, startup_models_from_env, Startup, OLLAMA_COMPAT_VERSION};
        use std::sync::atomic::Ordering;

        // Env access is process-global; keep these serialized and scoped.
        fn with_env(val: Option<&str>, f: impl FnOnce()) {
            match val {
                Some(v) => std::env::set_var("CIMA_PULL_AT_STARTUP", v),
                None => std::env::remove_var("CIMA_PULL_AT_STARTUP"),
            }
            f();
            std::env::remove_var("CIMA_PULL_AT_STARTUP");
        }

        #[test]
        fn unset_env_means_no_required_models() {
            with_env(None, || assert!(startup_models_from_env().is_empty()));
        }

        #[test]
        fn parses_comma_list_trimming_and_dedup() {
            with_env(Some(" org/a:q8_0 , org/b ,, org/a:q8_0 ,org/c "), || {
                assert_eq!(
                    startup_models_from_env(),
                    vec!["org/a:q8_0", "org/b", "org/c"]
                );
            });
        }

        #[test]
        fn empty_string_yields_nothing() {
            with_env(Some("   ,  , "), || {
                assert!(startup_models_from_env().is_empty());
            });
        }

        #[test]
        fn served_ollama_version_is_bare_semver() {
            // Must not carry a prerelease/build suffix — Ollama clients parse
            // it as semver to feature-gate, and `-cima` would read as "older".
            std::env::remove_var("CIMA_OLLAMA_VERSION");
            let v = ollama_version();
            assert_eq!(v, OLLAMA_COMPAT_VERSION);
            assert!(!v.contains('-') && !v.contains('+') && !v.contains("cima"));
            assert_eq!(v.split('.').count(), 3, "expected X.Y.Z, got {v}");
            assert!(v.split('.').all(|p| p.parse::<u32>().is_ok()));
        }

        #[test]
        fn ollama_version_env_override_wins_and_is_trimmed() {
            std::env::set_var("CIMA_OLLAMA_VERSION", "  0.11.4 ");
            assert_eq!(ollama_version(), "0.11.4");
            std::env::set_var("CIMA_OLLAMA_VERSION", "");
            assert_eq!(ollama_version(), OLLAMA_COMPAT_VERSION); // blank ignored
            std::env::remove_var("CIMA_OLLAMA_VERSION");
        }

        #[test]
        fn startup_state_defaults_and_transitions() {
            let s = Startup {
                required: vec!["org/x".into()],
                ..Default::default()
            };
            assert!(!s.ready());
            assert!(!s.failed());
            s.done.store(true, Ordering::Relaxed);
            assert!(s.ready());
            let f = Startup {
                required: vec!["org/y".into()],
                ..Default::default()
            };
            f.failed.store(true, Ordering::Relaxed);
            assert!(f.failed());
            assert!(!f.ready());
        }
    }

    #[cfg(test)]
    mod tool_call_tests {
        use super::{balanced_json_end, extract_tool_calls};

        #[test]
        fn extracts_tool_call_without_close_tag() {
            // The real-world case: the model emits the open tag + JSON, no
            // closing </tool_call>, then keeps talking. The call must still be
            // lifted out and the trailing prose kept as content.
            let full = "<tool_call> {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n\nThe weather in Paris is sunny.";
            let (text, calls) = extract_tool_calls(full);
            assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
            let f = calls[0].get("function").unwrap();
            assert_eq!(f.str_of("name"), Some("get_weather"));
            assert_eq!(f.get("arguments").unwrap().str_of("city"), Some("Paris"));
            assert!(
                text.contains("The weather in Paris is sunny."),
                "prose lost: {text:?}"
            );
            assert!(
                !text.contains("get_weather"),
                "call leaked into content: {text:?}"
            );
        }

        #[test]
        fn extracts_tool_call_with_close_tag() {
            let full = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
            let (text, calls) = extract_tool_calls(full);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].get("function").unwrap().str_of("name"), Some("f"));
            assert_eq!(text, "");
        }

        #[test]
        fn plain_text_is_untouched() {
            let (text, calls) = extract_tool_calls("just a normal answer");
            assert!(calls.is_empty());
            assert_eq!(text, "just a normal answer");
        }

        #[test]
        fn balanced_end_respects_strings_and_nesting() {
            // Braces inside a string must not close the object early.
            let s = "{\"a\":{\"b\":\"}}\"}} trailing";
            let end = balanced_json_end(s).unwrap();
            assert_eq!(&s[..end], "{\"a\":{\"b\":\"}}\"}}");
            assert_eq!(balanced_json_end("no object here"), None);
            assert_eq!(balanced_json_end("{ unterminated"), None);
        }
    }
}
