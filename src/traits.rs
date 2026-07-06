//! # traits — the engine's extensibility surface
//!
//! Every swappable behavior in the engine is expressed as a trait in this one
//! file. Static dispatch is used wherever a call sits on the token hot path
//! (zero-cost, monomorphized); dynamic dispatch (`Box<dyn …>`) is used only at
//! load-time boundaries where the cost is irrelevant (model loading, media
//! decoding, format selection).
//!
//! Trait map:
//! * [`ModelLoader`]   — weight container formats (safetensors today; gguf/pt tomorrow).
//! * [`WeightCodec`]   — quantization schemes (FP16/BF16 today; AWQ/GPTQ slots defined).
//! * [`Tokenizer`]     — text <-> token ids.
//! * [`ImageDecoder`]  — raw image bytes -> normalized pixel tensor.
//! * [`AudioDecoder`]  — raw audio bytes -> mono PCM -> features.
//! * [`Architecture`]  — a runnable model graph bound to the GPU.
//! * [`LogitsSampler`] — token selection strategies.

use crate::cuda::{CudaCtx, DeviceBuf};
use std::collections::HashMap;
use std::fmt;

// ===========================================================================
// Error type
// ===========================================================================

/// Engine-wide error. Every failure path carries a *granular*, human-readable
/// explanation: which tensor, which dimension, which config key, which CUDA
/// call. This is the backbone of the "broken model fail-safe".
#[derive(Debug)]
pub struct EngineError {
    /// Subsystem that raised the error (`"safetensors"`, `"cuda"`, `"config"`, …).
    pub scope: &'static str,
    /// Full human-readable diagnosis.
    pub msg: String,
}

impl EngineError {
    pub fn new(scope: &'static str, msg: impl Into<String>) -> Self {
        let e = EngineError {
            scope,
            msg: msg.into(),
        };
        crate::log::error(&format!("[{}] {}", e.scope, e.msg));
        e
    }

    /// Construct an error WITHOUT logging it. For expected, recoverable
    /// conditions where the immediate caller decides whether the situation
    /// is actually a failure — e.g. a bounds check that a retry loop is
    /// prepared to handle. The caller logs (via [`EngineError::new`]) only
    /// once it concludes the error is terminal.
    pub fn quiet(scope: &'static str, msg: impl Into<String>) -> Self {
        EngineError {
            scope,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.scope, self.msg)
    }
}
impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    /// Allow `?` on `std::io` results anywhere in the engine; the scope is
    /// tagged `io` so the originating syscall family is still identifiable.
    fn from(e: std::io::Error) -> Self {
        EngineError::new("io", e.to_string())
    }
}

/// Engine-wide result alias.
pub type Res<T> = Result<T, EngineError>;

/// Shorthand constructor: `err!("cuda", "cuMemAlloc failed: {}", code)`.
#[macro_export]
macro_rules! err {
    ($scope:expr, $($arg:tt)*) => {
        $crate::traits::EngineError::new($scope, format!($($arg)*))
    };
}

// ===========================================================================
// Data types & tensor metadata
// ===========================================================================

/// On-disk / on-device element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 4-bit AWQ packed weights (recognized, not yet executable — see [`WeightCodec`]).
    #[allow(dead_code)] // reserved: constructed once a 4-bit codec lands
    AwqInt4,
    /// 4-bit GPTQ packed weights (recognized, not yet executable).
    #[allow(dead_code)] // reserved: constructed once a 4-bit codec lands
    GptqInt4,
    /// GGUF/ggml block formats — storage layout in `quant::gguf`.
    GgufQ8_0,
    /// Legacy 32-grain ggml formats. llama.cpp's quantizer falls back to
    /// these for tensors whose row length is not a multiple of 256 (e.g.
    /// Qwen2.5-0.5B, hidden 896): a "Q4_K_M" file of such a model contains
    /// Q5_0/Q5_1 tensors — without them the checkpoint is unloadable.
    GgufQ4_0,
    GgufQ4_1,
    GgufQ5_0,
    GgufQ5_1,
    GgufQ4K,
    GgufQ5K,
    GgufQ6K,
    GgufIQ4XS,
    I64,
    U8,
}

impl DType {
    /// Bytes per element (packed formats report their *storage* width).
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
            // Block formats have no per-element width; container code
            // computes nbytes via `formats::gguf::storage_bytes`.
            DType::U8 | DType::AwqInt4 | DType::GptqInt4 => 1,
            DType::GgufQ8_0
            | DType::GgufQ4_0
            | DType::GgufQ4_1
            | DType::GgufQ5_0
            | DType::GgufQ5_1
            | DType::GgufQ4K
            | DType::GgufQ5K
            | DType::GgufQ6K
            | DType::GgufIQ4XS => 1,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::AwqInt4 => "AWQ-INT4",
            DType::GptqInt4 => "GPTQ-INT4",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::GgufQ8_0 => "Q8_0",
            DType::GgufQ4_0 => "Q4_0",
            DType::GgufQ4_1 => "Q4_1",
            DType::GgufQ5_0 => "Q5_0",
            DType::GgufQ5_1 => "Q5_1",
            DType::GgufQ4K => "Q4_K",
            DType::GgufQ5K => "Q5_K",
            DType::GgufQ6K => "Q6_K",
            DType::GgufIQ4XS => "IQ4_XS",
        }
    }
}

/// Metadata of one tensor inside a weight container (no data, just shape/locality).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte range inside the container's data section.
    pub offset: usize,
    pub nbytes: usize,
    /// Which shard file the tensor lives in (multi-file checkpoints).
    pub file: String,
}

impl TensorMeta {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

// ===========================================================================
// ModelLoader — weight container formats
// ===========================================================================

/// A parsed-and-validated weight container, ready for zero-copy upload.
///
/// The loader owns the host mapping (e.g. `mmap` of the safetensors files,
/// page-locked via `cuMemHostRegister`) so the GPU can DMA directly from the
/// page cache without an intermediate staging copy.
pub trait LoadedWeights: Send {
    /// All tensors in the container, keyed by canonical name.
    fn tensors(&self) -> &HashMap<String, TensorMeta>;
    /// Borrow the raw bytes of one tensor from the pinned host mapping.
    fn bytes(&self, meta: &TensorMeta) -> Res<&[u8]>;
    /// Advise the OS to prefetch the backing pages of one tensor (async,
    /// best-effort). Used for host-resident tables that are read with
    /// per-token latency sensitivity (the Gemma 4 PLE table): without it,
    /// every decode step pays a synchronous page fault chain into a cold
    /// pageable mmap. Default: no-op.
    fn prefetch(&self, _meta: &TensorMeta) {}
}

/// A weight container *format* (safetensors, gguf, pytorch …).
///
/// Implementations must perform exhaustive structural validation in `load`
/// and return granular [`EngineError`]s referencing the exact tensor /
/// dimension / header field that is malformed ("broken model fail-safe").
pub trait ModelLoader: Send + Sync {
    /// `true` if this loader can handle the files present in `dir`.
    fn detect(&self, dir: &std::path::Path) -> bool;
    /// Parse, validate and pin all weight shards found in `dir`.
    fn load(&self, dir: &std::path::Path, ctx: &CudaCtx) -> Res<Box<dyn LoadedWeights>>;
}

// ===========================================================================
// WeightCodec — quantization schemes
// ===========================================================================

/// Decodes a stored tensor into its on-device execution layout.
///
/// FP16/BF16 are identity codecs (direct DMA). Integer quantizations
/// (AWQ/GPTQ) plug in here: `upload` would dequantize-on-GPU or install the
/// packed weights plus scales for fused dequant-GEMM kernels.
pub trait WeightCodec: Send + Sync {
    /// Codec name for logs/errors.
    fn name(&self) -> &'static str;
    /// `true` if this codec handles `dtype`.
    fn accepts(&self, dtype: DType) -> bool;
    /// Predicted VRAM footprint of `meta` once resident (bytes).
    fn device_bytes(&self, meta: &TensorMeta) -> usize;
    /// Upload one tensor into a fresh device buffer (async on `ctx`'s stream).
    fn upload(&self, ctx: &CudaCtx, meta: &TensorMeta, host: &[u8]) -> Res<DeviceBuf>;
    /// `true` if quantized block tensors stay PACKED on device (the
    /// transformer must then route them through the fused gguf GEMV /
    /// dequant-scratch paths instead of treating buffers as f16).
    fn resident_quant(&self) -> bool {
        false
    }
}

/// `true` for the GGUF block formats that have resident kernels.
pub fn is_gguf_block(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::GgufQ8_0
            | DType::GgufQ4_0
            | DType::GgufQ4_1
            | DType::GgufQ5_0
            | DType::GgufQ5_1
            | DType::GgufQ4K
            | DType::GgufQ5K
            | DType::GgufQ6K
            | DType::GgufIQ4XS
    )
}

/// Elements per block of a gguf block format — the row-length grain every
/// kernel and validation must use. 32 for Q8_0 and the legacy formats
/// (Q4_0/Q4_1/Q5_0/Q5_1 — llama.cpp's fallback when a row is not
/// 256-divisible), 256 for the K-quants and IQ4_XS. 1 for non-block dtypes
/// so callers can divide unconditionally.
pub fn block_elems(dtype: DType) -> usize {
    match dtype {
        DType::GgufQ8_0 | DType::GgufQ4_0 | DType::GgufQ4_1 | DType::GgufQ5_0 | DType::GgufQ5_1 => {
            32
        }
        DType::GgufQ4K | DType::GgufQ5K | DType::GgufQ6K | DType::GgufIQ4XS => 256,
        _ => 1,
    }
}

// ===========================================================================
// Tokenizer
// ===========================================================================

/// Text <-> token-id codec. The shipped implementation is a from-scratch
/// byte-level BPE driven by Hugging Face `tokenizer.json`.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool) -> Vec<u32>;
    /// Decode a *single* token id to its byte string (streaming-safe).
    ///
    /// Returns a borrow into the tokenizer's decode table: this is called
    /// once per generated token on the decode hot path, so it must not
    /// allocate. Out-of-range ids yield an empty slice.
    fn decode_token(&self, id: u32) -> &[u8];
    fn eos_ids(&self) -> &[u32];
    /// Lookup a literal special token (e.g. `"<|im_start|>"`).
    fn special(&self, literal: &str) -> Option<u32>;
}

// ===========================================================================
// Media decoders — multi-modal input frontends
// ===========================================================================

/// Normalized image: planar CHW f32 in `[0,1]`-then-normalized space, ready
/// to feed a vision tower's patch embedding.
pub struct ImageTensor {
    pub data: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

/// Decodes raw image bytes (as received in the API `images` field) into an
/// [`ImageTensor`]. Implementations must never panic on hostile input: a
/// malformed file yields a descriptive [`EngineError`].
pub trait ImageDecoder: Send + Sync {
    /// Header-only dimension probe `(width, height)`, when cheaply available
    /// (drives the aspect-preserving resize target). Default: unknown.
    fn dims(&self, _b: &[u8]) -> Option<(usize, usize)> {
        None
    }
    /// Format name (`"ppm"`, `"bmp"`, …).
    fn name(&self) -> &'static str;
    /// Cheap magic-byte sniff.
    fn detect(&self, bytes: &[u8]) -> bool;
    /// Decode and resize to `(target_h, target_w)` with bilinear sampling,
    /// applying `mean`/`std` channel normalization.
    fn decode(
        &self,
        bytes: &[u8],
        target_h: usize,
        target_w: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor>;
}

/// Mono PCM audio at a fixed sample rate.
pub struct AudioPcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Decodes raw audio container bytes into mono PCM. The engine then computes
/// log-mel features on the CPU and hands them to the audio encoder tower.
pub trait AudioDecoder: Send + Sync {
    fn name(&self) -> &'static str;
    fn detect(&self, bytes: &[u8]) -> bool;
    /// Decode to mono f32 PCM resampled to `target_rate`.
    fn decode(&self, bytes: &[u8], target_rate: u32) -> Res<AudioPcm>;
}

// ===========================================================================
// Architecture — a runnable model bound to the GPU
// ===========================================================================

/// What the model can do — drives API routing and capability errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// Plain causal LM (text -> text).
    TextToText,
    /// Vision-language model (text+image -> text).
    VisionText,
    /// Speech transcription (audio -> text).
    AudioText,
    /// Embedding model (text -> vector).
    Embedding,
}

impl Modality {
    pub fn name(self) -> &'static str {
        match self {
            Modality::TextToText => "text",
            Modality::VisionText => "vision-text",
            Modality::AudioText => "audio-text",
            Modality::Embedding => "embedding",
        }
    }
}

/// One fully-prepared multimodal prompt: interleaved token ids and media
/// embeddings produced by the encoder towers.
pub struct PreparedPrompt {
    /// Token ids with media placeholder positions already spliced in.
    pub tokens: Vec<u32>,
    /// (position-in-`tokens`, embedding rows `[n, hidden]`) for each media chunk.
    pub media_embeds: Vec<(usize, DeviceBuf, usize)>,
    /// Per-token media-block ids (`-1` = text). Tokens sharing a block id
    /// attend bidirectionally during prefill (Gemma4 image spans). Empty =
    /// no bidirectional blocks (the standard llama/qwen path).
    pub block_ids: Vec<i32>,
}

/// Decoding hyper-parameters (subset of Ollama `options`).
#[derive(Debug, Clone)]
pub struct GenOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub stop: Vec<String>,
    /// Benchmark fairness: generate exactly `max_tokens` ignoring EOS, so
    /// throughput comparisons measure the same work on both engines.
    pub ignore_eos: bool,
    /// `format: "json"` / schema: constrain decoding to valid JSON via
    /// [`crate::json::JsonGuard`]. Forces the host full-logits
    /// sampling path (masking needs the whole row).
    pub json_mode: bool,
    /// `format: {schema}` — the schema, serialized. Compiled into a
    /// [`crate::json::SchemaGuard`] plan (forced scaffold + typed holes);
    /// uncompilable schemas fall back to plain JSON-mode.
    pub json_schema: Option<String>,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1024,
            seed: 0xC0FFEE,
            repeat_penalty: 1.1,
            stop: Vec::new(),
            ignore_eos: false,
            json_mode: false,
            json_schema: None,
        }
    }
}

impl GenOptions {
    /// Single point of truth for generation parameters: every surface (CLI
    /// flags, API JSON `options`, future config files) routes through this
    /// name → field mapping, so **adding a parameter is one match arm here**
    /// plus the field — exactly like Ollama's option table.
    ///
    /// Accepts both CLI (`top-p`) and JSON (`top_p`) spellings plus common
    /// aliases; values are validated, not clamped silently.
    /// The option table — one row per generation parameter. [`set`] parses
    /// against it and [`render_help`] prints it, so the CLI flag, the REPL
    /// `/set`, the API `options` object, and the help text cannot drift.
    /// Columns: canonical name, aliases, value range, default, description.
    pub const TABLE: &'static [(
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    )] = &[
        (
            "temperature",
            "temp",
            "0..=10 (0 = greedy)",
            "0.8",
            "softmax temperature; 0 selects the device-greedy path",
        ),
        (
            "top_k",
            "",
            "0 = unlimited; 1..=64 stays fully on-device",
            "40",
            "keep only the k most likely tokens before top_p",
        ),
        (
            "top_p",
            "",
            "0..=1",
            "0.9",
            "nucleus truncation over the top_k survivors",
        ),
        (
            "repeat_penalty",
            "",
            "> 0 (1.0 = off)",
            "1.1",
            "penalize tokens seen in the last 64 (rp^count, sign-aware)",
        ),
        (
            "seed",
            "",
            "u64",
            "random",
            "rng seed; same seed + options reproduces a sampled run",
        ),
        (
            "max_tokens",
            "num_predict, n",
            "u32",
            "512",
            "generation cap (EOS may stop earlier)",
        ),
        (
            "stop",
            "",
            "string, repeatable",
            "none",
            "stop sequence; generation halts when the text ends with it",
        ),
        (
            "ignore_eos",
            "",
            "true|false",
            "false",
            "keep generating past EOS (benchmarks)",
        ),
    ];

    /// Render [`TABLE`] for `--help` output.
    pub fn render_help() -> String {
        let mut out = String::from("generation parameters (--PARAM VALUE; also /set in the REPL and the API options object):\n");
        for (name, aliases, range, default, desc) in Self::TABLE {
            let alias = if aliases.is_empty() {
                String::new()
            } else {
                format!(" (alias: {})", aliases)
            };
            out.push_str(&format!(
                "  --{:<16}{}\n      {}  [range: {}; default: {}]\n",
                name.replace('_', "-"),
                alias,
                desc,
                range,
                default
            ));
        }
        out
    }

    pub fn set(&mut self, key: &str, value: &str) -> Res<()> {
        fn num<T: std::str::FromStr>(key: &str, v: &str) -> Res<T> {
            v.parse()
                .map_err(|_| crate::err!("options", "'{}' expects a number, got '{}'", key, v))
        }
        let k = key.trim_start_matches("--").replace('-', "_");
        match k.as_str() {
            "temp" | "temperature" => {
                let v: f32 = num(&k, value)?;
                if !(0.0..=10.0).contains(&v) {
                    return Err(crate::err!(
                        "options",
                        "temperature out of range [0,10]: {}",
                        v
                    ));
                }
                self.temperature = v;
            }
            "top_p" => {
                let v: f32 = num(&k, value)?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(crate::err!("options", "top_p out of range [0,1]: {}", v));
                }
                self.top_p = v;
            }
            "top_k" => self.top_k = num(&k, value)?,
            "max_tokens" | "num_predict" | "n" => self.max_tokens = num(&k, value)?,
            "seed" => self.seed = num(&k, value)?,
            "repeat_penalty" => {
                let v: f32 = num(&k, value)?;
                if v <= 0.0 {
                    return Err(crate::err!("options", "repeat_penalty must be > 0: {}", v));
                }
                self.repeat_penalty = v;
            }
            "stop" => self.stop.push(value.to_string()),
            "ignore_eos" => {
                self.ignore_eos = value
                    .parse()
                    .map_err(|_| crate::err!("options", "ignore_eos expects true/false"))?;
            }
            _ => return Err(crate::err!("options", "unknown option '{}'", key)),
        }
        Ok(())
    }
}

/// A model graph resident on the GPU, exposing the two primitive passes the
/// generation loop is built from. Specialized architectures (vision towers,
/// audio encoders, any-to-any graphs) implement this same trait and are bound
/// to the orchestration pipeline through it — the scheduler never knows the
/// concrete type.
/// What a resident model can do — drives surface-level dispatch (CLI/API
/// reject unsupported requests with a capability message instead of a deep
/// engine error) and is reported in `cima list`/`/api/tags`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    /// Autoregressive text generation (chat / completion).
    Generate,
    /// Pooled embedding extraction (text → vector).
    Embed,
    /// Image inputs accepted (vision tower present).
    Vision,
    /// Audio inputs accepted (audio encoder present).
    Audio,
}

impl Capability {
    pub fn name(self) -> &'static str {
        match self {
            Capability::Generate => "generate",
            Capability::Embed => "embed",
            Capability::Vision => "vision",
            Capability::Audio => "audio",
        }
    }
}

/// The decode-performance levers a family has actually pulled. Returned by
/// [`Architecture::perf_levers`], printed by `cima profile`, and commented
/// on by `cima vet` — a family that skips a lever ships with its scorecard
/// visible, not with silent slowness.
///
/// The levers, in the order they should be implemented (each preserves
/// greedy output exactly, so the vet battery doubles as the regression
/// net; see README § Extending for the symptoms each
/// one fixes in `cima profile`):
///
/// 1. [`device_greedy`](Self::device_greedy) — argmax on device
///    ([`CudaCtx::argmax_softcap_enqueue`]); never ship a logits row over
///    PCIe to pick a token.
/// 2. [`cuda_graph`](Self::cuda_graph) — capture the decode step
///    ([`CudaCtx::capture_begin`] / [`CudaCtx::capture_end`]) with
///    positions read from a device counter (`pos_dev` kernel parameters,
///    [`CudaCtx::pos_bump`]). Host-side work (e.g. gemma-4's PLE mmap
///    gather) stays outside a *partial* graph — still one launch for
///    everything else.
/// 3. [`fused_weights`](Self::fused_weights) — Q|K|V and gate|up
///    row-concatenated at load so decode runs one large GEMV instead of
///    several small ones; row sub-ranges still feed the prefill GEMMs.
/// 4. [`seq_parallel_attention`](Self::seq_parallel_attention) —
///    flash-decode ([`CudaCtx::attn_decode_split`] + reduce) so decode
///    latency stops growing with context length.
/// 5. [`device_pipeline`](Self::device_pipeline) — the next token id never
///    visits the host between steps ([`CudaCtx::fetch_token_async`]);
///    requires that nothing in the step needs the id host-side.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerfLevers {
    pub device_greedy: bool,
    pub cuda_graph: bool,
    pub fused_weights: bool,
    pub seq_parallel_attention: bool,
    pub device_pipeline: bool,
}

impl PerfLevers {
    /// Human-readable scorecard, e.g.
    /// `device_greedy+cuda_graph+fused_weights (missing: seq_parallel_attention, device_pipeline)`.
    pub fn summary(&self) -> String {
        let mut on = Vec::new();
        let mut off = Vec::new();
        for (name, v) in [
            ("device_greedy", self.device_greedy),
            ("cuda_graph", self.cuda_graph),
            ("fused_weights", self.fused_weights),
            ("seq_parallel_attention", self.seq_parallel_attention),
            ("device_pipeline", self.device_pipeline),
        ] {
            if v {
                on.push(name)
            } else {
                off.push(name)
            }
        }
        match (on.is_empty(), off.is_empty()) {
            (_, true) => format!("{} (all levers)", on.join("+")),
            (true, _) => format!("NONE (missing: {})", off.join(", ")),
            _ => format!("{} (missing: {})", on.join("+"), off.join(", ")),
        }
    }
}

/// A model family executable by the engine.
///
/// # Performance contract
///
/// Implementing the four methods below makes a family *correct*. Making it
/// *fast* means pulling the levers described on [`PerfLevers`], which this
/// trait deliberately surfaces as required methods rather than optional
/// extras: [`perf_levers`](Self::perf_levers) is your public scorecard
/// (`cima profile` prints it; `cima vet` notes missing levers), and
/// [`weight_bytes_resident`](Self::weight_bytes_resident) defines the
/// bandwidth floor your decode step is judged against
/// (`weight_bytes_resident / measured_bandwidth` per token). A new family
/// is expected to reach ≤ 1.5× of that floor before claiming a
/// `registry.rs` row as `verified`; the generic decoder sits at ~1.3×.
///
/// Greedy decode (`temperature == 0`, `repeat_penalty == 1.0`) is the hot
/// path every lever applies to; the sampling path may stay on full logits.
pub trait Architecture: Send {
    /// Supported modality of this instance.
    fn modality(&self) -> Modality;
    /// Maximum sequence length the KV cache was allocated for.
    fn max_seq(&self) -> usize;

    /// Run the prompt (prefill) pass; returns logits of the final position
    /// (host f32, `vocab_size` long). `pos0` is the absolute position offset.
    fn prefill(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<Vec<f32>>;

    /// Run one incremental decode step for `token` at absolute position `pos`.
    fn decode_step(&mut self, token: u32, pos: usize) -> Res<Vec<f32>>;

    /// Produce a pooled embedding for `tokens` (Embedding modality only).
    fn embed(&mut self, tokens: &[u32]) -> Res<Vec<f32>>;

    /// Reset the KV cache between requests (deterministic eviction).
    fn reset(&mut self) -> Res<()>;

    /// Exact measured VRAM held by this instance (weights + KV + workspace).
    fn vram_bytes(&self) -> usize;

    /// Resident weight bytes streamed per decode token: the numerator of
    /// the bandwidth floor (`cima profile` divides it by the measured GEMV
    /// bandwidth and reports your headroom multiple). For quantized
    /// codecs, count PHYSICAL bytes (packed + scales), not logical f16.
    fn weight_bytes_resident(&self) -> usize;

    /// The performance scorecard for this family — see [`PerfLevers`].
    /// Be honest: report what the DECODE path actually does at runtime
    /// (e.g. `cuda_graph` only when a graph is armed or armable), because
    /// `cima profile` will print this next to the measured step time and
    /// reviewers will compare.
    fn perf_levers(&self) -> PerfLevers;
}

// ===========================================================================
// LogitsSampler
// ===========================================================================

/// Token selection strategy. The default implementation chain is
/// repeat-penalty -> temperature -> top-k -> top-p -> categorical sample,
/// with greedy argmax when `temperature == 0`.
pub trait LogitsSampler: Send {
    fn sample(&mut self, logits: &mut [f32], history: &[u32], opts: &GenOptions) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aliases and both spellings hit the same fields.
    #[test]
    fn option_aliases() {
        let mut o = GenOptions::default();
        o.set("--temp", "0.5").unwrap();
        assert_eq!(o.temperature, 0.5);
        o.set("temperature", "0.7").unwrap();
        assert_eq!(o.temperature, 0.7);
        o.set("--top-p", "0.95").unwrap();
        assert_eq!(o.top_p, 0.95);
        o.set("top_p", "0.5").unwrap();
        assert_eq!(o.top_p, 0.5);
        o.set("num_predict", "32").unwrap();
        assert_eq!(o.max_tokens, 32);
        o.set("max-tokens", "64").unwrap();
        assert_eq!(o.max_tokens, 64);
    }

    /// Invalid values are rejected with errors, not clamped silently.
    #[test]
    fn option_validation() {
        let mut o = GenOptions::default();
        assert!(o.set("temp", "abc").is_err());
        assert!(o.set("temp", "-1").is_err());
        assert!(o.set("temp", "99").is_err());
        assert!(o.set("top_p", "1.5").is_err());
        assert!(o.set("repeat_penalty", "0").is_err());
        assert!(o.set("seed", "-3").is_err());
        assert!(o.set("definitely_unknown", "1").is_err());
        // failed sets must not have mutated state
        assert_eq!(o.temperature, GenOptions::default().temperature);
    }

    /// ignore_eos parses strict booleans only.
    #[test]
    fn ignore_eos_strict() {
        let mut o = GenOptions::default();
        o.set("ignore_eos", "true").unwrap();
        assert!(o.ignore_eos);
        o.set("ignore_eos", "false").unwrap();
        assert!(!o.ignore_eos);
        assert!(o.set("ignore_eos", "yes").is_err());
    }

    /// stop accumulates (multiple --stop flags).
    #[test]
    fn stop_accumulates() {
        let mut o = GenOptions::default();
        o.set("stop", "</s>").unwrap();
        o.set("stop", "\n\n").unwrap();
        assert_eq!(o.stop, vec!["</s>".to_string(), "\n\n".to_string()]);
    }

    #[test]
    fn option_table_matches_set() {
        // Every TABLE row (and alias) must be accepted by set(); the help
        // text and the parser share one source of truth by construction,
        // but this guards against rows whose canonical name drifts.
        let mut o = GenOptions::default();
        for (name, aliases, _, _, _) in GenOptions::TABLE {
            let probe = match *name {
                "stop" => "X",
                "ignore_eos" => "true",
                "temperature" | "top_p" => "0.5",
                "repeat_penalty" => "1.2",
                _ => "3",
            };
            assert!(
                o.set(name, probe).is_ok(),
                "TABLE row '{}' rejected by set()",
                name
            );
            for alias in aliases.split(',').map(str::trim).filter(|a| !a.is_empty()) {
                assert!(
                    o.set(alias, probe).is_ok(),
                    "alias '{}' rejected by set()",
                    alias
                );
            }
        }
        assert!(GenOptions::render_help().contains("--repeat-penalty"));
    }
}

pub mod num {
    //! # num — half-precision conversion primitives
    //!
    //! The single source of truth for f16 (IEEE-754 binary16) and bf16
    //! conversions on the host. Every module that touches raw checkpoint
    //! bytes (formats, quant codecs, model builders, self-tests) uses these —
    //! two implementations that round differently would silently break the
    //! host/device bit-equality checks in `cima selftest`.

    /// IEEE-754 binary16 → f32 — exact for every input, subnormals and
    /// NaN payloads included (validated exhaustively over all 65536 values
    /// against the hardware semantics the device kernels use).
    #[inline]
    pub fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) as u32) << 31;
        let exp = ((h >> 10) & 0x1f) as u32;
        let man = (h & 0x3ff) as u32;
        let bits = match (exp, man) {
            (0, 0) => sign,
            (0, m) => {
                // subnormal: value = m × 2^-24; normalize on the msb
                let p = 31 - m.leading_zeros(); // msb position 0..9
                let e = 103 + p; // 127 + (p - 24)
                let frac = (m << (23 - p)) & 0x7f_ffff;
                sign | (e << 23) | frac
            }
            (0x1f, 0) => sign | 0x7f80_0000,
            (0x1f, m) => sign | 0x7f80_0000 | (m << 13), // NaN, payload preserved
            (e, m) => sign | ((e + 112) << 23) | (m << 13),
        };
        f32::from_bits(bits)
    }

    /// f32 → IEEE-754 binary16, round-to-nearest-even on every path
    /// (normal, subnormal, overflow), NaN → canonical quiet NaN — matching
    /// the GPU's __float2half so host/device decode comparisons can demand
    /// bit equality.
    #[inline]
    pub fn f32_to_f16(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp_f = (x >> 23) & 0xff;
        let mut man = x & 0x7f_ffff;
        if exp_f == 0xff {
            // Inf stays Inf; NaN canonicalizes to the quiet NaN the GPU's
            // __float2half produces, so host and device decoders agree bit
            // for bit on poisoned blocks too.
            return if man != 0 {
                sign | 0x7e00
            } else {
                sign | 0x7c00
            };
        }
        let exp = exp_f as i32 - 127 + 15;
        if exp >= 31 {
            return sign | 0x7c00; // finite overflow → inf (RTNE)
        }
        if exp <= 0 {
            if exp < -10 {
                return sign; // underflow → signed zero
            }
            man |= 0x80_0000;
            let shift = (14 - exp) as u32;
            let mut half = (man >> shift) as u16;
            let rem = man & ((1u32 << shift) - 1);
            let tie = 1u32 << (shift - 1);
            if rem > tie || (rem == tie && half & 1 == 1) {
                half += 1; // round-to-nearest-even, subnormal range included
            }
            return sign | half;
        }
        let mut half = ((exp as u32) << 10 | man >> 13) as u16;
        let rem = man & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
            half += 1;
        }
        sign | half
    }

    /// bfloat16 → f32 (exact: bf16 is the top 16 bits of an f32).
    #[inline]
    pub fn bf16_to_f32(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }
}
