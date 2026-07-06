//! Model registry, architecture dispatch, and lifecycle management.
//!
//! `Arch` is the closed enum of resident graphs (enum dispatch = zero-cost:
//! no vtable, every call inlines). `ModelManager` owns detection
//! (config.json `model_type` / `architectures`), construction, and eviction.
//!
//! # Adding a new model family
//! 1. If the forward pass is standard (attention+MLP+norms), add a
//!    `family()` mapping in [`detect`] and parametrize [`Transformer`] —
//!    no new module needed.
//! 2. If it deviates structurally (multimodal splice, per-layer embeddings,
//!    shared KV…), create `models/yourfamily.rs` exposing
//!    `build(weights, ctx, config) -> Res<YourModel>` plus
//!    `prefill/decode_step/embed/reset/max_seq`, then add an `Arch` variant
//!    here and wire the match arms. The compiler enforces exhaustiveness —
//!    a missing arm is a build error, not a runtime surprise.

use crate::cuda::{fmt_bytes, CudaCtx, DeviceBuf};
use crate::json::{self};
use crate::log;
use crate::media::MediaRegistry;
use crate::models::gemma4::{G4Config, Gemma4};
use crate::models::transformer::Transformer;
use crate::formats::safetensors::{HalfCodec, SafetensorsLoader};
use crate::tokenizer::{render_chat, BpeTokenizer, ChatTurn};
use crate::traits::*;
use crate::err;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub mod gemma4;
pub mod towers;
pub mod transformer;

pub mod sampler {
//! # sampler — the host-side sampling chain
//!
//! Deterministic, dependency-free token selection: repeat-penalty →
//! temperature → top-k → top-p → categorical draw (xorshift64*), greedy
//! argmax at `temperature == 0`. Split out of `transformer.rs` because it
//! is pure host math shared by every architecture — the device top-k path
//! (see `SAMPLE_TOPK`) feeds the same `finish` tail so the two cannot
//! drift.

use crate::traits::{GenOptions, LogitsSampler};


/// Default sampling chain: repeat-penalty → temperature → top-k → top-p →
/// categorical draw (xorshift64*). Greedy argmax when `temperature == 0`.
pub struct DefaultSampler {
    rng: u64,
    /// Reusable per-token scratch. The full-logits path ranks the whole
    /// vocabulary (100k+ ids) every token; without persistent buffers that
    /// is a vocab-sized allocation per token on the CPU-sampling path.
    idx: Vec<u32>,
    vals: Vec<f32>,
    probs: Vec<f32>,
}

impl DefaultSampler {
    pub fn new(seed: u64) -> Self {
        DefaultSampler {
            rng: seed.max(1),
            idx: Vec::new(),
            vals: Vec::new(),
            probs: Vec::new(),
        }
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // xorshift64* — deterministic, dependency-free.
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

impl DefaultSampler {
    /// Steps 3–7 of the chain over candidates ALREADY sorted descending
    /// (and already repeat-penalized): temperature → softmax → top-p →
    /// categorical draw. Shared verbatim by the full-logits path and the
    /// device top-k path so the two cannot drift.
    pub fn finish(&mut self, vals: &[f32], ids: &[u32], opts: &GenOptions) -> u32 {
        debug_assert!(!vals.is_empty());
        // GREEDY GUARD — candidates arrive sorted descending, so argmax is
        // ids[0]. Without this, temperature 0 flowed into 1/T = ∞ below:
        // the top candidate's softmax term became 0·∞ = NaN, every NaN
        // comparison failed, and the fallthrough returned ids[cut-1] — the
        // WORST candidate of the top-k, deterministically, every token
        // (word-salad generations on all backends). The full-logits path
        // had this guard; this shared tail did not.
        if opts.temperature <= 0.0 {
            return ids[0];
        }
        let inv_t = 1.0 / opts.temperature.max(1e-4);
        let maxl = vals[0];
        // Build the (temperature-scaled, softmaxed, top-p-truncated)
        // distribution in the persistent buffer.
        {
            let probs = &mut self.probs;
            probs.clear();
            probs.extend(vals.iter().map(|v| ((v - maxl) * inv_t).exp()));
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let mut cut = probs.len();
            let mut cum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= opts.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
        }
        let renorm: f32 = self.probs.iter().sum();
        let cut = self.probs.len();
        let mut r = self.next_f32() * renorm;
        for (i, &p) in self.probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return ids[i];
            }
        }
        ids[cut - 1]
    }

    /// The device top-k twin of [`LogitsSampler::sample`]: candidates come
    /// from `CudaCtx::topk_enqueue` (descending, penalty applied on
    /// device), so only top-k truncation + the shared tail remain.
    pub fn sample_candidates(&mut self, vals: &[f32], ids: &[u32], opts: &GenOptions) -> u32 {
        let k = if opts.top_k == 0 { vals.len() } else { opts.top_k.min(vals.len()) };
        self.finish(&vals[..k], &ids[..k], opts)
    }
}

impl LogitsSampler for DefaultSampler {
    fn sample(&mut self, logits: &mut [f32], history: &[u32], opts: &GenOptions) -> u32 {
        // 1. Repeat penalty over the recent window (last 64 tokens).
        if opts.repeat_penalty != 1.0 {
            let window = &history[history.len().saturating_sub(64)..];
            for &t in window {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / opts.repeat_penalty } else { *l * opts.repeat_penalty };
                }
            }
        }
        // 2. Greedy path.
        if opts.temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        // 3..7 shared with the device-candidate path.
        let k = if opts.top_k == 0 { logits.len() } else { opts.top_k.min(logits.len()) };
        // Tie-break canon: equal logits rank by DESCENDING index — the
        // same order the device masked-argmax produces (the packed u64
        // compares index bits when values tie). Without a canon, ties at
        // the top-k boundary make the host and device paths diverge.
        let by_canon = |&a: &u32, &b: &u32| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.cmp(&a))
        };
        // Take the scratch buffers so finish() can hold &mut self without
        // aliasing them; return them afterward for reuse next token.
        let mut idx = std::mem::take(&mut self.idx);
        let mut vals = std::mem::take(&mut self.vals);
        idx.clear();
        idx.extend(0..logits.len() as u32);
        idx.select_nth_unstable_by(k - 1, by_canon);
        idx.truncate(k);
        idx.sort_unstable_by(by_canon);
        vals.clear();
        vals.extend(idx.iter().map(|&i| logits[i as usize]));
        let chosen = self.finish(&vals, &idx, opts);
        self.idx = idx;
        self.vals = vals;
        chosen
    }
}
}

pub use sampler::DefaultSampler;
pub use transformer::{GenStats, ModelConfig, VramForecast, CHUNK};

/// The resident computation graph. Architectures with shared math run on the
/// generic [`Transformer`]; families with their own pipeline (Gemma 4) bind
/// here as additional variants without touching the API layer.
// One Arch exists per loaded model (never in a hot array), so the size gap
// between variants costs nothing; boxing would add a pointer-chase to every
// dispatched call on the decode path for no memory benefit that matters.
#[allow(clippy::large_enum_variant)]
pub enum Arch {
    Std(Transformer),
    Gemma4(Gemma4),
}

/// Uniform two-arm dispatch, macro-generated: no vtable, every call
/// inlines into a `match` (the zero-cost property the enum exists for).
/// Only methods with the SAME name and signature on both families belong
/// here; anything asymmetric (capability sets, device-pipeline gating,
/// family-specific probes) stays hand-written below, where the asymmetry
/// is visible. Adding an `Arch` variant makes every expansion a
/// non-exhaustive-match build error — the compiler walks you through the
/// whole surface.
macro_rules! dispatch {
    () => {};
    ($(#[$m:meta])* pub fn $name:ident(&mut self $(, $arg:ident: $ty:ty)*) -> $ret:ty; $($rest:tt)*) => {
        $(#[$m])*
        pub fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            match self {
                Arch::Std(t) => t.$name($($arg),*),
                Arch::Gemma4(g) => g.$name($($arg),*),
            }
        }
        dispatch! { $($rest)* }
    };
    ($(#[$m:meta])* pub fn $name:ident(&self $(, $arg:ident: $ty:ty)*) -> $ret:ty; $($rest:tt)*) => {
        $(#[$m])*
        pub fn $name(&self $(, $arg: $ty)*) -> $ret {
            match self {
                Arch::Std(t) => t.$name($($arg),*),
                Arch::Gemma4(g) => g.$name($($arg),*),
            }
        }
        dispatch! { $($rest)* }
    };
}

impl Arch {
    dispatch! {
        pub fn modality(&self) -> Modality;
        pub fn prefill(&mut self, p: &PreparedPrompt, pos0: usize) -> Res<Vec<f32>>;
        pub fn decode_step(&mut self, tok: u32, pos: usize) -> Res<Vec<f32>>;
        /// Greedy fast path: token id straight from the device (see
        /// project_argmax). Both families support it.
        pub fn prefill_argmax(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<u32>;
        pub fn decode_step_argmax(&mut self, tok: u32, pos: usize) -> Res<u32>;
        /// Initialize the device position counter and (once) capture the decode
        /// graph. `pos` = first decode position (prompt length).
        pub fn arm_decode_graph(&mut self, pos: usize) -> Res<()>;
        /// The family's performance scorecard (see [`crate::traits::PerfLevers`]).
        pub fn perf_levers(&self) -> crate::traits::PerfLevers;
        pub fn decode_graph_active(&self) -> bool;
        pub fn arm_sample_graph(&mut self, pos: usize, rp: f32) -> Res<()>;
        pub fn sample_graph_active(&self) -> bool;
        pub fn decode_step_sample(&mut self, token: u32, pos: usize) -> Res<Vec<u64>>;
        pub fn hist_reset_and_seed(&mut self, history: &[u32], next_pos: usize) -> Res<()>;
        pub fn argmax_slot(&self) -> &crate::cuda::DeviceBuf;
        pub fn reset(&mut self) -> Res<()>;
        pub fn max_seq(&self) -> usize;
        pub fn vram_bytes(&self) -> usize;
    }

    /// Capability set of the resident graph — drives request validation at
    /// the API/CLI surface.
    pub fn capabilities(&self) -> Vec<Capability> {
        let mut caps = vec![Capability::Generate];
        match self {
            Arch::Std(t) => {
                if t.cfg.is_embedding {
                    // Dedicated embedding checkpoints don't generate.
                    caps = vec![Capability::Embed];
                } else {
                    // Every decoder serves pooled embeddings; the API
                    // validates announced capabilities, so omitting Embed
                    // here would reject /api/embeddings for capable models.
                    caps.push(Capability::Embed);
                }
                if t.vision.is_some() {
                    caps.push(Capability::Vision);
                }
                if t.audio.is_some() {
                    caps.push(Capability::Audio);
                }
            }
            Arch::Gemma4(g) => {
                caps.push(Capability::Embed);
                if g.has_vision() {
                    caps.push(Capability::Vision);
                }
                if g.has_audio() {
                    caps.push(Capability::Audio);
                }
            }
        }
        caps
    }

    pub fn ctx(&self) -> &Arc<CudaCtx> {
        match self {
            Arch::Std(t) => &t.ctx,
            Arch::Gemma4(g) => &g.ctx,
        }
    }
    pub fn embed(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        match self {
            Arch::Std(t) => t.embed(tokens),
            Arch::Gemma4(g) => g.embed_pool(tokens),
        }
    }
    /// Whether the family supports the device-resident greedy pipeline
    /// (no per-step host transfers). Gemma-4 does not: its PLE gather
    /// reads a host-resident table by token id.
    pub fn supports_device_pipeline(&self) -> bool {
        matches!(self, Arch::Std(_))
    }
    pub fn seed_device_token(&self) -> Res<()> {
        match self {
            Arch::Std(t) => t.seed_device_token(),
            Arch::Gemma4(_) => Err(err!("generate", "device pipeline unsupported for this family")),
        }
    }
    pub fn decode_step_device(&mut self, pos: usize) -> Res<()> {
        match self {
            Arch::Std(t) => t.decode_step_device(pos),
            Arch::Gemma4(_) => Err(err!("generate", "device pipeline unsupported for this family")),
        }
    }
    /// Total resident weight bytes (the [`Architecture`] contract; the
    /// bandwidth floor is `weight_bytes / memory_bandwidth` per token).
    pub fn weight_bytes_resident(&self) -> usize {
        match self {
            Arch::Std(t) => Architecture::weight_bytes_resident(t),
            Arch::Gemma4(g) => Architecture::weight_bytes_resident(g),
        }
    }
    /// Vision GPU-vs-CPU self-test (gemma4 only).
    pub fn vision_selftest(&mut self) -> Res<()> {
        match self {
            Arch::Std(_) => Err(err!("selftest", "vision self-test is implemented for the gemma4 pipeline only")),
            Arch::Gemma4(g) => g.vision_selftest(),
        }
    }

    /// Architecture-specific host-side timing breakdown accumulated since
    /// the last call (e.g. Gemma 4's per-token PLE row gathers and the
    /// logits dtoh+softcap). `None` when the architecture has none.
    pub fn perf_take(&mut self) -> Option<(f64, f64)> {
        match self {
            Arch::Std(_) => None,
            Arch::Gemma4(g) => Some(g.perf_take()),
        }
    }

    /// Architecture-specific stop ids beyond the tokenizer's EOS (Gemma 4
    /// stops on both `<eos>` and `<end_of_turn>`).
    pub fn extra_eos(&self) -> &[u32] {
        match self {
            Arch::Std(_) => &[],
            Arch::Gemma4(g) => g.extra_eos(),
        }
    }
}

/// A model fully bound to the GPU: graph + tokenizer + media frontends.
pub struct LoadedModel {
    pub name: String,
    pub dir: PathBuf,
    pub arch: Arch,
    pub tokenizer: BpeTokenizer,
    pub media: MediaRegistry,
    /// Placeholder literal spliced into the prompt per media item.
    pub media_token: Option<String>,
    /// Chat template source carried by the container itself (GGUF
    /// metadata `tokenizer.chat_template`) — overrides file discovery.
    pub chat_template: Option<String>,
    /// Token ids whose KV rows are valid from the previous `generate` call
    /// (prompt + every token whose forward ran). Incremental prefill skips
    /// the longest common prefix with the next prompt — O(delta) ttft for
    /// chat instead of O(history). Cleared on entry, set on clean exit, so
    /// an aborted generation falls back to a full prefill.
    pub session_ids: Vec<u32>,
    /// Exact text whose encoding-as-fed equals `session_ids` (fed prompt
    /// text + streamed reply). Lets chat turns extend the session at the
    /// id level instead of re-encoding history — the model's own generated
    /// segmentation is preserved verbatim, so the KV prefix stays valid.
    pub session_text: String,
}

/// Known media-placeholder literals, probed against the tokenizer's specials.
const MEDIA_LITERALS: &[&str] = &[
    "<image>", "<|image_pad|>", "<|IMAGE|>", "<|vision_pad|>",
    "<|AUDIO|>", "<|audio_pad|>", "<audio>",
];

/// Byte offset of the earliest stop-sequence occurrence in `text`, or None.
/// Stop matching is EXCLUSIVE and position-based (not `ends_with`): a decode
/// token can append the stop plus trailing characters ("Banana\n"), and a
/// stop can straddle two tokens, so scanning the whole accumulated text and
/// truncating at the earliest match is the only robust rule. Empty stops are
/// ignored (an empty stop would otherwise match at 0 and abort immediately).
fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    let mut hit: Option<usize> = None;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop.as_str()) {
            hit = Some(hit.map_or(pos, |h| h.min(pos)));
        }
    }
    hit
}

impl LoadedModel {
    /// Capability set of the underlying graph (see [`Capability`]).
    pub fn capabilities(&self) -> Vec<Capability> {
        self.arch.capabilities()
    }

    /// Run the vision self-test: phase 1 = tower GPU-vs-CPU divergence;
    /// phase 2 = LM-side consumption (template render, token frame, splice
    /// bit-exactness, multimodal PLE rows).
    pub fn vision_selftest(&mut self) -> Res<()> {
        self.arch.vision_selftest()?;

        println!("--- phase 2: LM-side consumption ---");
        // synthetic 768×768 quadrant P6 PPM in memory: red | green over
        // blue | yellow. Maximally diagnostic — each soft-token quadrant has
        // a distinct dominant color, so spatial mixing anywhere downstream
        // shows up directly in the consumption probe's cosines.
        let (w, h) = (768usize, 768usize);
        let mut ppm = format!("P6\n{} {}\n255\n", w, h).into_bytes();
        let colors: [[u8; 3]; 4] = [[220, 40, 40], [40, 180, 60], [50, 80, 220], [240, 200, 40]];
        for y in 0..h {
            for x in 0..w {
                let c = colors[usize::from(y >= h / 2) * 2 + usize::from(x >= w / 2)];
                ppm.extend_from_slice(&c);
            }
        }
        println!("tokenizer specials: {:?}", self.tokenizer.specials());
        let turns = [ChatTurn { role: "user".into(), content: "Describe this image".into(), n_images: 1, n_audio: 0 }];
        let rendered = self.render_chat(&turns);
        println!("rendered template ({} chars): {:?}", rendered.len(), rendered);
        let prepared = self.prepare(&rendered, &[ppm], &[])?;
        println!("prepared: {} tokens, {} media spans", prepared.tokens.len(), prepared.media_embeds.len());
        let media: Vec<(usize, u64, usize)> = prepared
            .media_embeds
            .iter()
            .map(|(at, buf, rows)| (*at, buf.ptr, *rows))
            .collect();
        match &mut self.arch {
            Arch::Gemma4(g) => g.splice_check(&prepared.tokens, &media),
            Arch::Std(_) => unreachable!(),
        }
    }

    /// Modality of the resident graph.
    pub fn modality(&self) -> Modality {
        self.arch.modality()
    }

    /// Build a [`PreparedPrompt`]: tokenize, run encoder towers over each
    /// media item, and splice the embedding spans over placeholder tokens.
    pub fn prepare(&mut self, prompt: &str, images: &[Vec<u8>], audio: &[Vec<u8>]) -> Res<PreparedPrompt> {
        if let Arch::Gemma4(_) = &self.arch {
            return self.prepare_gemma4(prompt, images, audio);
        }
        // 1. Encode media through the towers.
        let mut chunks: Vec<(DeviceBuf, usize)> = Vec::new();
        for (i, bytes) in images.iter().enumerate() {
            let std_arch = match &self.arch {
                Arch::Std(t) => t,
                Arch::Gemma4(_) => unreachable!(),
            };
            let tower = std_arch.vision.as_ref().ok_or_else(|| {
                err!("media", "model '{}' has no vision tower but image #{} was supplied", self.name, i)
            })?;
            let edge = tower.ec.input_size;
            let img = self
                .media
                .decode_image(bytes, edge, edge, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])?;
            chunks.push(tower.encode_image(&std_arch.ctx, &img)?);
        }
        for (i, bytes) in audio.iter().enumerate() {
            let std_arch = match &self.arch {
                Arch::Std(t) => t,
                Arch::Gemma4(_) => unreachable!(),
            };
            let tower = std_arch.audio.as_ref().ok_or_else(|| {
                err!("media", "model '{}' has no audio tower but audio clip #{} was supplied", self.name, i)
            })?;
            let pcm = self.media.decode_audio(bytes, 16_000)?;
            chunks.push(tower.encode_audio(&std_arch.ctx, &pcm)?);
        }

        // 2. Tokenize. Placeholders may or may not be present in the text.
        let mut text = prompt.to_string();
        if !chunks.is_empty() && self.media_token.as_deref().map(|t| !text.contains(t)).unwrap_or(true) {
            // No placeholder in the prompt: prepend one per media item.
            if let Some(tok) = &self.media_token {
                let mut pre = String::new();
                for _ in 0..chunks.len() {
                    pre.push_str(tok);
                    pre.push('\n');
                }
                text = format!("{}{}", pre, text);
            }
        }
        let tokens = self.tokenizer.encode(&text, true);

        // 3. Splice: each placeholder id expands to `rows` ids, and the span
        //    is recorded so prefill overwrites those embedding rows.
        let ph_id = self.media_token.as_deref().and_then(|t| self.tokenizer.special(t));
        let mut out_tokens = Vec::with_capacity(tokens.len());
        let mut media_embeds = Vec::new();
        let mut next = chunks.into_iter();
        for &t in &tokens {
            if Some(t) == ph_id {
                match next.next() {
                    Some((buf, rows)) => {
                        media_embeds.push((out_tokens.len(), buf, rows));
                        for _ in 0..rows {
                            out_tokens.push(t);
                        }
                    }
                    None => {
                        return Err(err!(
                            "media",
                            "prompt contains more {} placeholders than supplied media items",
                            self.media_token.as_deref().unwrap_or("<media>")
                        ))
                    }
                }
            } else {
                out_tokens.push(t);
            }
        }
        if next.next().is_some() {
            return Err(err!("media", "more media items supplied than placeholders consumed — encoder output would be dropped"));
        }
        Ok(PreparedPrompt { tokens: out_tokens, media_embeds, block_ids: Vec::new() })
    }

    /// Gemma 4 prompt preparation. Media items are encoded through the
    /// gemma4 towers and framed exactly like the reference processor:
    /// images  -> `<start_of_image>` + image_token × rows + `<end_of_image>`
    /// audio   -> `<start_of_audio>` + audio_token × rows + `<end_of_audio>`
    /// (token *ids* come straight from config.json, no literal lookup).
    /// Image soft-token spans receive consecutive block ids so prefill
    /// attends bidirectionally inside each image, matching the reference
    /// mask; audio spans stay causal.
    fn prepare_gemma4(&mut self, prompt: &str, images: &[Vec<u8>], audio: &[Vec<u8>]) -> Res<PreparedPrompt> {
        let g = match &self.arch {
            Arch::Gemma4(g) => g,
            Arch::Std(_) => unreachable!(),
        };
        let cfg = g.config();
        let (img_id, boi, eoi) = (cfg.image_token_id, cfg.boi_token_id, cfg.eoi_token_id);
        let (aud_id, boa, eoa) = (cfg.audio_token_id, cfg.boa_token_id, cfg.eoa_token_id);
        if !audio.is_empty() && (aud_id == 0 || boa == 0 || eoa == 0) {
            return Err(err!(
                "media",
                "audio marker ids unresolved (audio_token_id={}, boa={}, eoa={}): config.json omits them and the tokenizer \
                 registers no audio-marker specials — audio spans cannot be framed for this checkpoint",
                aud_id, boa, eoa
            ));
        }

        // ---- encode media ----
        let mut img_chunks: Vec<(DeviceBuf, usize)> = Vec::new();
        for (i, bytes) in images.iter().enumerate() {
            if !g.has_vision() {
                return Err(err!("media", "model '{}' has no vision tower but image #{} was supplied", self.name, i));
            }
            // Aspect-preserving resize per the reference processor; the
            // soft-token count varies with the image's aspect ratio.
            let (sw, sh) = self.media.image_dims(bytes).ok_or_else(|| {
                err!("media", "image #{}: cannot determine dimensions from the header (unrecognized or truncated file)", i)
            })?;
            let (th, tw) = Gemma4::image_target_size(sw, sh)?;
            let (mean, std) = Gemma4::image_norm();
            let img = self.media.decode_image(bytes, th, tw, mean, std)?;
            let chunk = g.encode_image(&img)?;
            // CIMA_DUMP_SOFT=/path — write the projected soft tokens as
            // little-endian f32 [rows, lm_hidden] for A/B against the
            // reference implementation's `get_image_features` output.
            if let Ok(base) = std::env::var("CIMA_DUMP_SOFT") {
                let hs = g.config().text.hidden;
                gemma4::dump_f16_matrix(g.ctx(), chunk.0.ptr, chunk.1, hs, &format!("{}.{}.soft", base, i), "soft tokens")?;
            }
            img_chunks.push(chunk);
        }
        let mut aud_chunks: Vec<(DeviceBuf, usize)> = Vec::new();
        for (i, bytes) in audio.iter().enumerate() {
            if !g.has_audio() {
                return Err(err!("media", "model '{}' has no audio tower but audio clip #{} was supplied", self.name, i));
            }
            let pcm = self.media.decode_audio(bytes, 16_000)?;
            let chunk = g.encode_audio(&pcm)?;
            // Same A/B dump as image soft tokens — `.aud.{i}.soft` files.
            if let Ok(base) = std::env::var("CIMA_DUMP_SOFT") {
                let hs = g.config().text.hidden;
                gemma4::dump_f16_matrix(g.ctx(), chunk.0.ptr, chunk.1, hs, &format!("{}.aud.{}.soft", base, i), "audio soft tokens")?;
            }
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                // Degenerate-output triage: healthy soft tokens carry RMS
                // comparable to the image tokens (~0.5-2 pre-LM); near-zero
                // (or NaN) here means the tower is producing silence and
                // the model will rightly claim there is no audio.
                let hs = g.config().text.hidden;
                let mut b = vec![0u8; chunk.1 * hs * 2];
                g.ctx().dtoh_at(&mut b, chunk.0.ptr)?;
                let (mut sq, mut n_nan) = (0f64, 0usize);
                for c in b.chunks_exact(2) {
                    let v = crate::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                    if v.is_nan() {
                        n_nan += 1;
                    } else {
                        sq += (v as f64) * (v as f64);
                    }
                }
                let rms = (sq / (chunk.1 * hs) as f64).sqrt();
                eprintln!(
                    "g4 audio clip #{}: {:.2}s pcm → {} soft tokens, rms {:.4}{}",
                    i, pcm.samples.len() as f64 / pcm.sample_rate as f64, chunk.1, rms,
                    if n_nan > 0 { format!(", {} NaN!", n_nan) } else { String::new() }
                );
            }
            aud_chunks.push(chunk);
        }

        // ---- split the text on placeholders, preserving order ----
        // Recognized placeholders: "<image>" and "<audio>" (the engine's
        // generic literals). Without placeholders, media is prepended.
        #[derive(Clone, Copy, PartialEq)]
        enum Seg { Img, Aud }
        let mut segments: Vec<(String, Option<Seg>)> = Vec::new();
        {
            let mut rest = prompt;
            loop {
                let pi = rest.find("<image>");
                let pa = rest.find("<audio>");
                match (pi, pa) {
                    (None, None) => {
                        segments.push((rest.to_string(), None));
                        break;
                    }
                    (i, a) => {
                        let (at, seg, len) = match (i, a) {
                            (Some(i), Some(a)) if i < a => (i, Seg::Img, 7),
                            (Some(i), None) => (i, Seg::Img, 7),
                            (_, Some(a)) => (a, Seg::Aud, 7),
                            _ => unreachable!(),
                        };
                        segments.push((rest[..at].to_string(), Some(seg)));
                        rest = &rest[at + len..];
                    }
                }
            }
        }
        let ph_imgs = segments.iter().filter(|(_, s)| *s == Some(Seg::Img)).count();
        let ph_auds = segments.iter().filter(|(_, s)| *s == Some(Seg::Aud)).count();
        if ph_imgs > img_chunks.len() || ph_auds > aud_chunks.len() {
            return Err(err!(
                "media",
                "prompt references {} image / {} audio placeholders but {} / {} items were supplied",
                ph_imgs, ph_auds, img_chunks.len(), aud_chunks.len()
            ));
        }

        // ---- assemble tokens + block ids ----
        let mut tokens: Vec<u32> = Vec::new();
        let mut block_ids: Vec<i32> = Vec::new();
        let mut media_embeds: Vec<(usize, DeviceBuf, usize)> = Vec::new();
        let mut next_img = img_chunks.into_iter();
        let mut next_aud = aud_chunks.into_iter();
        let mut img_block = 0i32;

        let push_text = |s: &str, tokens: &mut Vec<u32>, block_ids: &mut Vec<i32>, tk: &BpeTokenizer, bos: bool| {
            if s.is_empty() && !bos {
                return;
            }
            for t in tk.encode(s, bos) {
                tokens.push(t);
                block_ids.push(-1);
            }
        };
        // Only the larger Gemma 4 variants attend bidirectionally inside
        // image spans; the smaller ones (E2B/E4B) keep a conventional causal
        // mask over the soft tokens. The reference gates this on
        // `text_config.use_bidirectional_attention == "vision"`.
        let bidir = cfg.text.bidir_vision;
        let mut splice_img = |buf: DeviceBuf, rows: usize, tokens: &mut Vec<u32>, block_ids: &mut Vec<i32>, media: &mut Vec<(usize, DeviceBuf, usize)>| {
            tokens.push(boi);
            block_ids.push(-1);
            media.push((tokens.len(), buf, rows));
            for _ in 0..rows {
                tokens.push(img_id);
                block_ids.push(if bidir { img_block } else { -1 });
            }
            img_block += 1;
            tokens.push(eoi);
            block_ids.push(-1);
        };
        let splice_aud = |buf: DeviceBuf, rows: usize, tokens: &mut Vec<u32>, block_ids: &mut Vec<i32>, media: &mut Vec<(usize, DeviceBuf, usize)>| {
            tokens.push(boa);
            block_ids.push(-1);
            media.push((tokens.len(), buf, rows));
            for _ in 0..rows {
                tokens.push(aud_id);
                block_ids.push(-1); // audio is causal in the reference mask
            }
            tokens.push(eoa);
            block_ids.push(-1);
        };

        // BOS only on the very first text segment. Media without a matching
        // placeholder is prepended (framed) ahead of the text, mirroring the
        // standard path's behaviour.
        let mut first = true;
        let n_img_total = next_img.len();
        let n_aud_total = next_aud.len();
        for _ in 0..n_img_total.saturating_sub(ph_imgs) {
            let (buf, rows) = next_img.next().unwrap();
            if first {
                push_text("", &mut tokens, &mut block_ids, &self.tokenizer, true);
                first = false;
            }
            splice_img(buf, rows, &mut tokens, &mut block_ids, &mut media_embeds);
        }
        for _ in 0..n_aud_total.saturating_sub(ph_auds) {
            let (buf, rows) = next_aud.next().unwrap();
            if first {
                push_text("", &mut tokens, &mut block_ids, &self.tokenizer, true);
                first = false;
            }
            splice_aud(buf, rows, &mut tokens, &mut block_ids, &mut media_embeds);
        }
        for (text, seg) in segments {
            push_text(&text, &mut tokens, &mut block_ids, &self.tokenizer, first);
            first = false;
            match seg {
                Some(Seg::Img) => {
                    let (buf, rows) = next_img.next().expect("placeholder count validated above");
                    splice_img(buf, rows, &mut tokens, &mut block_ids, &mut media_embeds);
                }
                Some(Seg::Aud) => {
                    let (buf, rows) = next_aud.next().expect("placeholder count validated above");
                    splice_aud(buf, rows, &mut tokens, &mut block_ids, &mut media_embeds);
                }
                None => {}
            }
        }
        if std::env::var("CIMA_G4_DEBUG").is_ok() {
            // eprintln, not log::info: this must reach the terminal
            // regardless of log level/sink configuration. Print ids AND
            // the round-tripped pieces — wrong ids with a right-looking
            // count is the failure mode this exists to expose.
            let head: Vec<u32> = tokens.iter().take(24).cloned().collect();
            let pieces: Vec<String> = head
                .iter()
                .map(|&id| String::from_utf8_lossy(&self.tokenizer.decode_bytes(id)).into_owned())
                .collect();
            eprintln!("g4 prompt ids (first {} of {}): {:?}", head.len(), tokens.len(), head);
            eprintln!("g4 prompt pieces: {:?}", pieces);
        }
        Ok(PreparedPrompt { tokens, media_embeds, block_ids })
    }

    /// Chat-aware prepare: when the freshly rendered prompt extends the
    /// previous session text, reuse the session ids verbatim and encode
    /// only the suffix (its boundary is the <end_of_turn> special, which
    /// hard-splits segments — standalone encoding equals in-context
    /// encoding). `generate`'s common-prefix scan then lands exactly on
    /// the session length, and prefill touches only the delta.
    pub fn prepare_chat(&mut self, rendered: &str) -> Res<PreparedPrompt> {
        if !self.session_ids.is_empty()
            && !self.session_text.is_empty()
            && rendered.starts_with(&self.session_text)
            && std::env::var("CIMA_NO_INCR").is_err()
        {
            let suffix = &rendered[self.session_text.len()..];
            let suffix_ids = {
                use crate::traits::Tokenizer as _;
                self.tokenizer.encode(suffix, false)
            };
            let mut tokens = self.session_ids.clone();
            tokens.extend_from_slice(&suffix_ids);
            // CIMA_INCR_CHECK=1: arbiter for the id chain. The session ids
            // legitimately differ from the canonical re-encode inside
            // PREVIOUS REPLY spans (the model's own segmentation is the
            // valid one for the KV) — but a divergence in the SUFFIX span,
            // or a length drift, would mean the chain is feeding the model
            // different text than the template renders. Print and fall
            // back to the canonical path so quality is never hostage.
            if std::env::var("CIMA_INCR_CHECK").is_ok() {
                let canonical = self.prepare(rendered, &[], &[])?;
                let div = tokens.iter().zip(canonical.tokens.iter()).position(|(a, b)| a != b);
                eprintln!(
                    "incr-check: chained={} canonical={} first_divergence={:?} suffix_starts_at={}",
                    tokens.len(), canonical.tokens.len(), div, self.session_ids.len()
                );
                if tokens.len() != canonical.tokens.len() {
                    eprintln!("incr-check: LENGTH DRIFT — using canonical (full prefill) this turn");
                    return Ok(canonical);
                }
            }
            return Ok(PreparedPrompt { tokens, media_embeds: Vec::new(), block_ids: Vec::new() });
        }
        self.prepare(rendered, &[], &[])
    }

    /// Record the text twin of `session_ids` after a chat turn completes.
    pub fn note_session_text(&mut self, t: String) {
        self.session_text = t;
    }

    /// The generation loop: prefill → sample → decode steps, streaming each
    /// token's text through `on_token`, with strict telemetry on completion.
    pub fn generate(
        &mut self,
        prepared: &PreparedPrompt,
        opts: &GenOptions,
        queue_wait_ms: f64,
        mut on_token: impl FnMut(&str),
    ) -> Res<GenStats> {
        let snap0 = self.arch.ctx().snapshot();
        let t0 = Instant::now();
        self.arch.reset()?;
        // Ollama-parity: a prompt longer than the context window is
        // TRUNCATED (first token kept — BOS for the families that use
        // one — plus the most recent tail), never a 500. Sixteen slots
        // stay reserved so the model can still answer. Media prompts are
        // exempt: their block ids address absolute positions.
        let truncated;
        let prepared: &PreparedPrompt = if prepared.media_embeds.is_empty()
            && prepared.block_ids.is_empty()
            && prepared.tokens.len() + 16 > self.arch.max_seq()
        {
            let keep = self.arch.max_seq().saturating_sub(16).max(2);
            let mut toks = Vec::with_capacity(keep);
            toks.push(prepared.tokens[0]);
            toks.extend_from_slice(&prepared.tokens[prepared.tokens.len() - (keep - 1)..]);
            log::warn(&format!(
                "prompt truncated to fit context window: {} -> {} tokens (max_seq {})",
                prepared.tokens.len(),
                toks.len(),
                self.arch.max_seq()
            ));
            truncated = PreparedPrompt { tokens: toks, media_embeds: Vec::new(), block_ids: Vec::new() };
            &truncated
        } else {
            prepared
        };
        let mut sampler = DefaultSampler::new(opts.seed);
        // Turn-closer stop set: chat generations must stop at the closing
        // marker of WHICHEVER dialect the renderer emitted. config.json's
        // eos list only covers the checkpoint's native dialect, so both
        // families' closers (when registered as single specials) join the
        // stop set — a no-op for non-gemma tokenizers.
        let turn_stops: Vec<u32> = ["<end_of_turn>", "<turn|>"]
            .iter()
            .filter_map(|m| self.tokenizer.special(m))
            .collect();

        // Incremental prefill: KV rows for the longest common prefix with
        // the previous session are already on device — only the delta needs
        // computing. Media prompts keep the full path (block ids are
        // absolute), as does CIMA_NO_INCR=1. At least one token is always
        // prefilled so the head has a fresh row to project.
        // Std-arch ONLY for now: Gemma4::reset() clears the absolute
        // block-id mirror (blk_host), so a delta prefill at pos0>0 rebuilds
        // it without rows 0..pos0 and the attention tables misalign — the
        // field symptom is a repeated or extended prompt generating exactly
        // one token (the turn closer). Full prefill is always correct.
        let mut cp = 0usize;
        if matches!(self.arch, Arch::Std(_))
            && prepared.media_embeds.is_empty()
            && prepared.block_ids.is_empty()
            && std::env::var("CIMA_NO_INCR").is_err()
        {
            let cap = self.session_ids.len().min(prepared.tokens.len().saturating_sub(1));
            while cp < cap && self.session_ids[cp] == prepared.tokens[cp] {
                cp += 1;
            }
        }
        self.session_ids.clear();
        let suffix_prepared;
        let prefill_input: &PreparedPrompt = if cp > 0 {
            suffix_prepared = PreparedPrompt {
                tokens: prepared.tokens[cp..].to_vec(),
                media_embeds: Vec::new(),
                block_ids: Vec::new(),
            };
            &suffix_prepared
        } else {
            prepared
        };

        // Greedy fast path: when sampling reduces to argmax and nothing
        // needs the logits row on the host (no repeat penalty), the token id
        // comes straight from the device (8 bytes) instead of the full
        // vocab·f32 row (1 MB for gemma-4) + host softcap + host scan —
        // which dominated short generations (METRIC logits= ~36-45 ms/tok).
        // JSON-constrained decoding needs the full logits row on the host
        // (rejected tokens are masked and redrawn), so every device
        // shortcut — argmax fast path, token pipeline, top-k sample graph —
        // stands down while the guard is active.
        // Schema-constrained: compile `format:{schema}` into a plan of
        // forced scaffold text + typed value holes. An uncompilable schema
        // (or plain `format:"json"`) uses the syntax-only guard instead.
        let mut schema: Option<crate::json::SchemaGuard> = opts
            .json_schema
            .as_deref()
            .and_then(|s| crate::json::parse(s).ok())
            .and_then(|j| crate::json::compile_schema(&j))
            .map(crate::json::SchemaGuard::new);
        let mut guard = if opts.json_mode && schema.is_none() {
            Some(crate::json::JsonGuard::new())
        } else {
            None
        };
        let constrained = guard.is_some() || schema.is_some();
        let greedy =
            (opts.temperature <= 0.0 || opts.top_k == 1) && opts.repeat_penalty == 1.0 && !constrained;
        // Device-resident pipeline: each step reads its input token from
        // device memory (written by the previous argmax) and the 8-byte id
        // fetch overlaps the next step's compute. Identical output to the
        // sync path; CIMA_NO_PIPELINE=1 forces the sync path.
        let pipeline = greedy
            && self.arch.supports_device_pipeline()
            && std::env::var("CIMA_NO_PIPELINE").map(|v| v != "1").unwrap_or(true);

        // Device top-k sampling: the sampler truncates to top-k before
        // top-p, so extracting the top-SAMPLE_TOPK candidates on device is
        // EXACT for top_k in 1..=64 — 512 bytes/token cross PCIe instead
        // of the logits row, and the whole tail replays as a CUDA graph.
        let sample_graph_ok = !greedy
            && !constrained
            && opts.top_k >= 1
            && opts.top_k <= crate::models::transformer::SAMPLE_TOPK
            && std::env::var("CIMA_NO_GRAPH").map(|v| v != "1").unwrap_or(true);

        let mut logits = Vec::new();
        let mut candidates: Option<(Vec<f32>, Vec<u32>)> = None;
        let mut next_greedy = 0u32;
        let fetch = if pipeline { Some(self.arch.ctx().token_fetch()?) } else { None };
        if greedy {
            next_greedy = self.arch.prefill_argmax(prefill_input, cp)?;
            if pipeline {
                self.arch.seed_device_token()?;
            }
            // Both families capture their decode step; gemma-4's graph is
            // partial (the host PLE gather stays outside).
            self.arch.arm_decode_graph(prepared.tokens.len())?;
        } else {
            logits = self.arch.prefill(prefill_input, cp)?;
            if sample_graph_ok {
                self.arch.arm_sample_graph(prepared.tokens.len(), opts.repeat_penalty)?;
                self.arch.hist_reset_and_seed(&prepared.tokens, prepared.tokens.len())?;
            }
        }
        let mut history = prepared.tokens.clone();
        // Schema scaffold tokens awaiting force-feed (encoded from the
        // plan's Fixed segments; never sampled, never eligible as EOS).
        let mut forced: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let mut text_acc = String::new();
        let mut pending: Vec<u8> = Vec::new();
        let mut ttft_ms = 0.0;
        let mut n_gen = 0usize;
        // Ollama done_reason: assume budget exhaustion ("length") until an
        // EOS or a stop string proves otherwise.
        let mut stop_reason: &'static str = "length";

        let budget = opts.max_tokens.min(self.arch.max_seq().saturating_sub(prepared.tokens.len()));
        'gen: for step in 0..budget {
            if let Some(sg) = schema.as_mut() {
                if sg.finished() {
                    break 'gen;
                }
                if forced.is_empty() {
                    if let Some(txt) = sg.forced_text() {
                        // Force-feed only if encode∘decode reproduces the
                        // scaffold bytes EXACTLY — SentencePiece dialects
                        // can prepend a phantom space, which would desync
                        // the guard. On mismatch the sampled arm emits the
                        // scaffold instead (the guard accepts those bytes
                        // like any others; slower, never wrong).
                        let txt = txt.to_string();
                        let toks = self.tokenizer.encode(&txt, false);
                        let mut rt: Vec<u8> = Vec::with_capacity(txt.len());
                        for &t in &toks {
                            rt.extend_from_slice(self.tokenizer.decode_token(t));
                        }
                        if rt == txt.as_bytes() {
                            forced.extend(toks);
                        }
                    }
                }
            }
            let tok = if let Some(t) = forced.pop_front() {
                // Scaffold token: advance the guard by its bytes (accepted
                // by construction — the text came from the plan itself).
                if let Some(sg) = schema.as_mut() {
                    let bytes = self.tokenizer.decode_token(t);
                    let ok = sg.push_bytes(bytes);
                    debug_assert!(ok, "verified scaffold token rejected by guard");
                }
                t
            } else if let Some(sg) = schema.as_mut() {
                // Typed hole: mask-and-redraw under the schema acceptor.
                // A token that CLOSES the hole and runs into scaffold text
                // (e.g. `36}` or `", "`) is accepted wholesale — the guard
                // re-dispatches the boundary bytes internally.
                let mut opts_now = opts.clone();
                let mut chosen: Option<u32> = None;
                for _ in 0..4096 {
                    let t = sampler.sample(&mut logits, &history, &opts_now);
                    opts_now.repeat_penalty = 1.0;
                    let is_end = self.tokenizer.eos_ids().contains(&t)
                        || self.arch.extra_eos().contains(&t)
                        || turn_stops.contains(&t);
                    if is_end {
                        if sg.complete() {
                            chosen = Some(t);
                            break;
                        }
                        logits[t as usize] = f32::NEG_INFINITY;
                        continue;
                    }
                    let bytes = self.tokenizer.decode_token(t);
                    let mut probe = sg.clone();
                    if probe.push_bytes(bytes) {
                        *sg = probe;
                        chosen = Some(t);
                        break;
                    }
                    logits[t as usize] = f32::NEG_INFINITY;
                }
                match chosen {
                    Some(t) => t,
                    None => break 'gen,
                }
            } else if let Some(g) = guard.as_mut() {
                // Constrained draw: mask-and-redraw until a token's bytes
                // keep the JSON prefix valid. EOS is masked while the value
                // is open and accepted once it closes. `sample` applies the
                // repeat penalty in-place, so redraws run penalty-free to
                // avoid compounding it.
                let mut opts_now = opts.clone();
                let mut chosen: Option<u32> = None;
                for _ in 0..4096 {
                    let t = sampler.sample(&mut logits, &history, &opts_now);
                    opts_now.repeat_penalty = 1.0;
                    let is_end = self.tokenizer.eos_ids().contains(&t)
                        || self.arch.extra_eos().contains(&t)
                        || turn_stops.contains(&t);
                    if is_end {
                        if g.complete() {
                            chosen = Some(t);
                            break;
                        }
                        logits[t as usize] = f32::NEG_INFINITY;
                        continue;
                    }
                    let bytes = self.tokenizer.decode_token(t);
                    let mut probe = g.clone();
                    if probe.push_bytes(bytes) {
                        *g = probe;
                        chosen = Some(t);
                        break;
                    }
                    logits[t as usize] = f32::NEG_INFINITY;
                }
                match chosen {
                    Some(t) => t,
                    // No token can extend a valid JSON prefix (pathological
                    // vocab exhaustion): end the generation cleanly.
                    None => break 'gen,
                }
            } else if greedy {
                next_greedy
            } else if let Some((vals, ids)) = &candidates {
                // Penalty already applied on device; candidates arrive in
                // descending order, so only the shared tail remains.
                sampler.sample_candidates(vals, ids, opts)
            } else {
                sampler.sample(&mut logits, &history, opts)
            };
            if !opts.ignore_eos
                && (self.tokenizer.eos_ids().contains(&tok)
                    || self.arch.extra_eos().contains(&tok)
                    || turn_stops.contains(&tok))
            {
                stop_reason = "stop";
                break;
            }
            if step == 0 {
                ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
            }
            // Pipeline: launch step+1 (device-resident input token) and
            // enqueue its 8-byte result fetch BEFORE the host-side emit
            // work below, so streaming/UTF-8/socket time overlaps the GPU.
            // Slot reuse is stream-ordered: the fetch sits between its
            // producing argmax and the next step's memset.
            if let Some(f) = &fetch {
                self.arch.decode_step_device(prepared.tokens.len() + step)?;
                self.arch.ctx().fetch_token_async(self.arch.argmax_slot(), f)?;
            }
            n_gen += 1;
            history.push(tok);
            // Stream with UTF-8 fusing: byte-fallback tokenizers (Gemma /
            // Llama-2 lineage) split multi-byte characters across tokens, so
            // bytes are buffered until a complete code point is available
            // (the reference decoder's `Fuse` step).
            pending.extend_from_slice(self.tokenizer.decode_token(tok));
            let valid = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(e) => e.valid_up_to(),
            };
            // An invalid prefix can't be the start of a longer sequence —
            // flush it lossily rather than stalling the stream (max UTF-8
            // continuation is 3 trailing bytes).
            let flush = if pending.len() - valid > 3 { pending.len() } else { valid };
            if flush > 0 {
                let piece = String::from_utf8_lossy(&pending[..flush]).into_owned();
                pending.drain(..flush);
                text_acc.push_str(&piece);
                on_token(&piece);
            }
            // Stop sequences are EXCLUSIVE: generation halts and the stop
            // text is removed from the output. See `earliest_stop` — matches
            // by position (not `ends_with`) so a token carrying the stop plus
            // trailing characters, or a stop straddling a token boundary, is
            // still caught and trimmed.
            if let Some(pos) = earliest_stop(&text_acc, &opts.stop) {
                // Non-stream body and returned stats reflect the trim; a
                // streamed tail may already have emitted the stop bytes —
                // clients needing byte-exact exclusion should use
                // stream:false.
                text_acc.truncate(pos);
                stop_reason = "stop";
                break 'gen;
            }
            // Constrained mode: the moment the top-level JSON value closes,
            // the generation is over — trailing prose is exactly what the
            // guard exists to prevent.
            if guard.as_ref().map(|g| g.complete()).unwrap_or(false)
                || schema.as_ref().map(|s| s.finished()).unwrap_or(false)
            {
                stop_reason = "stop";
                break 'gen;
            }
            if let Some(f) = &fetch {
                // Wait blocks on the copy event only; a stop-sequence break
                // above merely wastes one speculative launch.
                next_greedy = f.wait()?;
            } else if greedy {
                next_greedy = self.arch.decode_step_argmax(tok, prepared.tokens.len() + step)?;
            } else if sample_graph_ok && self.arch.sample_graph_active() {
                let packed = self.arch.decode_step_sample(tok, prepared.tokens.len() + step)?;
                let mut vals = Vec::with_capacity(packed.len());
                let mut ids = Vec::with_capacity(packed.len());
                for p in packed {
                    let (v, i) = crate::cuda::unpack_candidate(p);
                    vals.push(v);
                    ids.push(i);
                }
                candidates = Some((vals, ids));
            } else {
                logits = self.arch.decode_step(tok, prepared.tokens.len() + step)?;
            }
        }
        if !pending.is_empty() {
            let piece = String::from_utf8_lossy(&pending).into_owned();
            text_acc.push_str(&piece);
            on_token(&piece);
        }
        self.arch.ctx().sync()?;

        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        let decode_ms = (total_ms - ttft_ms).max(1e-3);
        let tok_per_s = if n_gen > 1 { (n_gen - 1) as f64 / (decode_ms / 1e3) } else { 0.0 };
        let snap1 = self.arch.ctx().snapshot();
        log::metric(
            "inference",
            &[
                ("model", self.name.clone()),
                ("prompt_tokens", prepared.tokens.len().to_string()),
                ("gen_tokens", n_gen.to_string()),
                ("queue_wait_ms", format!("{:.2}", queue_wait_ms)),
                ("ttft_ms", format!("{:.2}", ttft_ms)),
                ("tok_per_s", format!("{:.2}", tok_per_s)),
                ("total_ms", format!("{:.2}", total_ms)),
                ("vram_used", snap1.vram_used.to_string()),
                ("vram_delta", (snap1.vram_used as i64 - snap0.vram_used as i64).to_string()),
                (
                    "host_ms",
                    match self.arch.perf_take() {
                        Some((ple, logits)) => format!("ple={:.1} logits={:.1}", ple, logits),
                        None => "-".to_string(),
                    },
                ),
            ],
        );
        self.session_ids = history.clone();
        Ok(GenStats {
            prompt_tokens: prepared.tokens.len(),
            gen_tokens: n_gen,
            ttft_ms,
            tok_per_s,
            total_ms,
            stop_reason,
            text: text_acc,
        })
    }

    /// Embedding entry point (Embedding modality, or pooled fallback).
    pub fn embed(&mut self, text: &str) -> Res<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true);
        self.arch.embed(&tokens)
    }

    /// Render an Ollama chat request into a single prompt string.
    /// Decode a token sequence to text through this model's tokenizer
    /// (lossy on broken UTF-8 boundaries — comparison/debug use).
    pub fn detokenize(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in ids {
            bytes.extend_from_slice(self.tokenizer.decode_token(t));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn render_chat(&self, turns: &[ChatTurn]) -> String {
        let family = match &self.arch {
            Arch::Gemma4(_) => Some("gemma"),
            Arch::Std(_) => None,
        };
        render_chat(&self.dir, &self.tokenizer, family, turns, self.media_token.as_deref(), self.chat_template.as_deref())
    }
}

// ===========================================================================
// ModelManager — deterministic single-slot load / evict
// ===========================================================================

/// Owns the single GPU residency slot. Loading a different model first drops
/// the resident one (RAII frees every buffer deterministically), verifies the
/// VRAM forecast against live telemetry, then builds the new graph.
/// Requested GPU residency for a model after a request completes —
/// ollama's `keep_alive` semantics: a duration string ("5m", "30s", "1h"),
/// a number of seconds, `0` (unload immediately), or any negative value
/// (resident forever).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeepAlive {
    Default,
    Seconds(u64),
    Forever,
    Now,
}

impl KeepAlive {
    /// Parse the `keep_alive` field of an API request (absent → Default).
    pub fn parse(v: Option<&crate::json::Json>) -> KeepAlive {
        let Some(v) = v else { return KeepAlive::Default };
        if let Some(n) = v.as_f64() {
            return if n < 0.0 {
                KeepAlive::Forever
            } else if n == 0.0 {
                KeepAlive::Now
            } else {
                KeepAlive::Seconds(n as u64)
            };
        }
        let Some(s) = v.as_str() else { return KeepAlive::Default };
        let s = s.trim();
        if s == "0" {
            return KeepAlive::Now;
        }
        if s.starts_with('-') {
            return KeepAlive::Forever;
        }
        let (num, unit) = s.split_at(s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len()));
        let Ok(n) = num.trim().parse::<f64>() else { return KeepAlive::Default };
        let secs = match unit.trim() {
            "" | "s" => n,
            "m" => n * 60.0,
            "h" => n * 3600.0,
            _ => return KeepAlive::Default,
        };
        if secs <= 0.0 { KeepAlive::Now } else { KeepAlive::Seconds(secs as u64) }
    }
}

pub struct ModelManager {
    pub ctx: std::sync::Arc<CudaCtx>,
    pub current: Option<LoadedModel>,
    /// When the resident model becomes evictable (None = forever).
    expires_at: Option<std::time::Instant>,
    /// Residency granted when a request doesn't specify `keep_alive`
    /// (None = forever). Default 5 minutes; every served request resets the
    /// clock via [`ModelManager::touch`]. Overridable at startup with the
    /// `CIMA_KEEP_ALIVE` environment variable, same grammar as the wire
    /// field: plain seconds (`600`), suffixed durations (`90s`, `10m`,
    /// `2h`), `0` (evict right after each request) or `-1`/`forever`
    /// (never evict). Per-request `keep_alive` still overrides per call.
    pub keep_alive_default: Option<std::time::Duration>,
}

impl ModelManager {
    pub fn new(ctx: std::sync::Arc<CudaCtx>) -> ModelManager {
        let keep_alive_default = match std::env::var("CIMA_KEEP_ALIVE") {
            Err(_) => Some(std::time::Duration::from_secs(300)),
            Ok(raw) => {
                let parsed = if raw.trim().eq_ignore_ascii_case("forever") {
                    KeepAlive::Forever
                } else {
                    KeepAlive::parse(Some(&crate::json::Json::s(raw.trim())))
                };
                match parsed {
                KeepAlive::Forever => {
                    log::info("CIMA_KEEP_ALIVE=forever — models stay resident until evicted explicitly");
                    None
                }
                KeepAlive::Seconds(s) => {
                    log::info(&format!("CIMA_KEEP_ALIVE — default residency window set to {}s", s));
                    Some(std::time::Duration::from_secs(s))
                }
                KeepAlive::Now => {
                    log::info("CIMA_KEEP_ALIVE=0 — models are released right after each request");
                    Some(std::time::Duration::ZERO)
                }
                KeepAlive::Default => {
                    log::warn(&format!(
                        "CIMA_KEEP_ALIVE='{}' is not a duration (expected e.g. 600, 90s, 10m, 2h, -1, forever) — keeping the 5-minute default",
                        raw
                    ));
                    Some(std::time::Duration::from_secs(300))
                }
                }
            }
        };
        ModelManager {
            ctx,
            current: None,
            expires_at: None,
            keep_alive_default,
        }
    }

    /// Refresh the residency clock after serving a request.
    pub fn touch(&mut self, ka: KeepAlive) {
        let dur = match ka {
            KeepAlive::Default => self.keep_alive_default,
            KeepAlive::Seconds(s) => Some(std::time::Duration::from_secs(s)),
            KeepAlive::Forever => None,
            KeepAlive::Now => {
                self.evict();
                return;
            }
        };
        self.expires_at = dur.map(|d| std::time::Instant::now() + d);
    }

    /// Evict the resident model if its keep-alive window has elapsed.
    /// Called by the server's sweeper between requests.
    pub fn sweep(&mut self) {
        if self.current.is_some() && self.expires_at.map(|t| std::time::Instant::now() >= t).unwrap_or(false) {
            log::info("keep-alive expired; releasing the resident model");
            self.evict();
        }
    }

    /// Seconds until eviction (None = forever / nothing resident).
    pub fn expires_in_secs(&self) -> Option<u64> {
        self.expires_at.map(|t| t.saturating_duration_since(std::time::Instant::now()).as_secs())
    }

    /// After a fatal (sticky) CUDA error the primary context is dead
    /// device-wide: every later call fails and the server zombifies —
    /// HTTP-alive, inference-dead. Recovery: drop the resident model
    /// (best-effort frees on the corpse), reset the primary context, and
    /// rebuild a fresh CudaCtx (NVRTC recompile included). The request
    /// that hit the fault still fails — honestly — but the NEXT one runs.
    fn recover(&mut self) -> Res<()> {
        log::warn("CUDA context poisoned — evicting, resetting the device, rebuilding");
        self.current = None;
        self.expires_at = None;
        let gpu = self.ctx.gpu_index();
        let rebuilt = crate::cuda::reset_primary(gpu).and_then(|_| {
            // A freshly-reset device can need a beat before Retain works;
            // field runs showed the sticky error echoing on the first try.
            let mut last = None;
            for _ in 0..5 {
                match CudaCtx::init(gpu) {
                    Ok(ctx) => return Ok(ctx),
                    Err(e) => {
                        log::warn(&format!("CudaCtx rebuild attempt failed: {}", e));
                        last = Some(e);
                        std::thread::sleep(std::time::Duration::from_millis(300));
                    }
                }
            }
            Err(last.unwrap())
        });
        match rebuilt {
            Ok(ctx) => {
                self.ctx = std::sync::Arc::new(ctx);
                log::warn("CUDA context rebuilt; models will reload on demand");
                Ok(())
            }
            Err(e) => {
                // In-process recovery exhausted: the only remaining clean
                // slate is a fresh process (Ollama gets this for free by
                // respawning its runner subprocess). Under Docker/systemd
                // with a restart policy this IS the recovery path; exit
                // loudly with a recognizable code.
                log::error(&format!(
                    "GPU unrecoverable in-process ({}); exiting 86 for the supervisor to restart",
                    e
                ));
                std::process::exit(86);
            }
        }
    }

    /// Ensure `name` is resident, evicting any other model.
    pub fn ensure(&mut self, name: &str) -> Res<&mut LoadedModel> {
        if crate::cuda::context_poisoned() {
            self.recover()?;
        }
        // Every API connection runs on its own OS thread; make the primary
        // context current here, the choke point all GPU work flows through.
        self.ctx.bind();
        if self.current.as_ref().map(|m| m.name == name).unwrap_or(false) {
            return Ok(self.current.as_mut().unwrap());
        }
        self.evict();
        let model = self.load(name)?;
        self.current = Some(model);
        Ok(self.current.as_mut().unwrap())
    }

    /// Drop the resident model and log the reclaimed VRAM.
    pub fn evict(&mut self) {
        if let Some(m) = self.current.take() {
            let name = m.name.clone();
            let before = self.ctx.tracked_bytes();
            let t0 = Instant::now();
            drop(m);
            let _ = self.ctx.sync();
            let after = self.ctx.tracked_bytes();
            log::info(&format!(
                "evicted '{}' in {:?}: {} VRAM reclaimed ({} still tracked)",
                name,
                t0.elapsed(),
                fmt_bytes(before.saturating_sub(after)),
                fmt_bytes(after)
            ));
        }
    }

    /// Full load path: locate → parse config → quant gate → parse weights →
    /// forecast & fit-check → build graph → load tokenizer.
    fn load(&self, name: &str) -> Res<LoadedModel> {
        // `ORG/REPO:TAG` selects a quantization inside a GGUF repo
        // (HF repo ids never contain ':').
        let (repo, tag) = match name.split_once(':') {
            Some((r, t)) => (r, Some(t)),
            None => (name, None),
        };
        // The pull path stores under the FULL selector
        // (`ORG/REPO:TAG` -> `ORG__REPO@TAG`), so resolve the directory from
        // `name`, not the bare `repo`. Fall back to the bare-repo directory
        // for checkpoints pulled by older versions (which stored untagged),
        // and for a bare `ORG/REPO` with no tag the two are identical.
        let dir = {
            let tagged = crate::hub::local_dir(name);
            if tagged.exists() {
                tagged
            } else {
                crate::hub::local_dir(repo)
            }
        };
        let ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .collect();
        if tag.is_some() || (!ggufs.is_empty() && !dir.join("config.json").exists()) {
            return self.load_gguf(name, repo, tag, dir, ggufs);
        }
        if !dir.join("config.json").exists() {
            // A directory without config.json is an interrupted pull (or a
            // hand-made husk) — say so, instead of "not present" about a
            // path that visibly exists.
            if dir.exists() {
                return Err(err!(
                    "models",
                    "snapshot at {} is incomplete (no config.json — interrupted pull?). \
                     Run `cima rm {}` then pull the full ORG/REPO id.",
                    dir.display(),
                    name
                ));
            }
            return Err(err!(
                "models",
                "model '{}' is not present locally (expected {}). Run `cima pull {}` first.",
                name,
                dir.display(),
                name
            ));
        }

        let t0 = Instant::now();
        let snap0 = self.ctx.snapshot();
        let cfg = ModelConfig::load(&dir)?;

        // ---- quantization gate: precise rejection before any work ----
        // bitsandbytes 4-bit (NF4/FP4, including unsloth "dynamic" mixed
        // checkpoints) executes natively on the gemma4 pipeline: packed
        // weights stay resident in 4-bit, decode uses a fused dequant-GEMV
        // and prefill dequantizes per-GEMM into a scratch buffer.
        // Every other architecture runs bnb through the host-dequant bridge
        // below: weights expand to f16 at load (no VRAM saving vs the fp16
        // revision, but the 4-bit-rounded numbers execute faithfully).
        if let Some(q) = &cfg.quant_method {
            if q != "bitsandbytes" {
                return Err(err!(
                    "quant",
                    "model '{}' is quantized with method '{}' (config.json: quantization_config.quant_method). \
                     Packed INT4 execution kernels for this method are not registered in this build — \
                     pull the FP16/BF16 revision, or register a WeightCodec implementing fused dequant-GEMM.",
                    name, q
                ));
            }
        }

        // ---- container format selection via the ModelLoader trait ----
        let loader = SafetensorsLoader;
        if !loader.detect(&dir) {
            return Err(err!(
                "models",
                "no *.safetensors files in {} — only the safetensors container is registered \
                 (gguf / pytorch loaders plug in via the ModelLoader trait)",
                dir.display()
            ));
        }
        let mut weights = loader.load(&dir, &self.ctx)?;
        // bnb on a non-gemma4 pipeline: fold + dequantize the packed 4-bit
        // families to f16 on the host so the standard Transformer sees an
        // ordinary f16 checkpoint (gemma4 keeps its native packed residency).
        if cfg.quant_method.as_deref() == Some("bitsandbytes") && cfg.model_type != "gemma4" {
            weights = Box::new(crate::quant::bnb::DequantizedWeights::wrap(weights)?);
        }

        // ---- forecast BEFORE any device allocation ----
        let codec = HalfCodec;
        let arch = if cfg.model_type == "gemma4" {
            // Gemma 4 runs its own exhaustive config parse: the generic
            // ModelConfig cannot express per-type RoPE, KV sharing, PLE or
            // the tower geometries.
            let raw = std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| err!("config", "cannot re-read {}/config.json: {}", dir.display(), e))?;
            let j = json::parse(&raw)
                .map_err(|e| err!("config", "{}/config.json is not valid JSON: {}", dir.display(), e))?;
            let g4 = G4Config::parse(&j)?;
            // Forecast mirrors Gemma4::build exactly: the PLE token table and
            // the audio subsampling convs stay host-resident, so the largest
            // conversion transient also excludes them.
            let (w, kv, ws) = Gemma4::forecast_bytes(&g4, weights.as_ref(), &codec);
            let transient = weights
                .tensors()
                .values()
                .filter(|t| !matches!(t.dtype, DType::F16 | DType::U8))
                .filter(|t| {
                    !t.name.ends_with("embed_tokens_per_layer.weight")
                        && !t.name.contains("subsample_conv_projection.")
                        && !crate::quant::bnb::is_aux(&t.name)
                })
                .map(|t| t.numel() * t.dtype.size())
                .max()
                .unwrap_or(0);
            let forecast = VramForecast { weights: w, kv_cache: kv, workspace: ws, load_transient: transient };
            forecast.check(&self.ctx, name)?;
            VramForecast::host_guard(weights.as_ref(), name)?;
            Arch::Gemma4(Gemma4::build(self.ctx.clone(), g4, weights, &codec)?)
        } else {
            let forecast = VramForecast::compute(&cfg, weights.as_ref(), &codec);
            forecast.check(&self.ctx, name)?;
            VramForecast::host_guard(weights.as_ref(), name)?;
            Arch::Std(Transformer::build(self.ctx.clone(), cfg, weights.as_ref(), &codec)?)
        };
        let tokenizer = BpeTokenizer::load(&dir)?;
        let media_token = MEDIA_LITERALS
            .iter()
            .find(|l| tokenizer.special(l).is_some())
            .map(|s| s.to_string());

        let snap1 = self.ctx.snapshot();
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        log::metric(
            "model_load",
            &[
                ("model", name.to_string()),
                ("load_ms", format!("{:.1}", load_ms)),
                ("vram_pre", snap0.vram_used.to_string()),
                ("vram_post", snap1.vram_used.to_string()),
                ("vram_delta", (snap1.vram_used as i64 - snap0.vram_used as i64).to_string()),
                ("modality", arch.modality().name().to_string()),
                (
                    "capabilities",
                    arch.capabilities().iter().map(|c| c.name()).collect::<Vec<_>>().join("+"),
                ),
            ],
        );
        Ok(LoadedModel {
            name: name.to_string(),
            dir,
            arch,
            tokenizer,
            media: MediaRegistry::standard(),
            media_token,
            chat_template: None,
            session_ids: Vec::new(),
            session_text: String::new(),
        })
    }

    /// GGUF loading: single-file checkpoint, quantization chosen by tag
    /// (`ORG/REPO:Q4_K_XL`). Config, tokenizer and chat template are
    /// synthesized from GGUF metadata; tensors dequantize to f16 on
    /// upload (see `quant::gguf` for the resident-quantized roadmap).
    /// gemma-4 GGUF: llama.cpp's reduced-graph export served by the native
    /// gemma4 pipeline. Hyper-parameters come from the HF config.json that
    /// quantizers ship next to the gguf (the gguf metadata lacks several
    /// gemma4 text fields); weights flow through WTensor::Gguf — dp4a
    /// decode, slab-dequant prefill, PLE rows host-dequantized per token.
    fn load_gguf_gemma4(&self, name: &str, dir: &std::path::Path, weights: crate::formats::gguf::GgufWeights) -> Res<LoadedModel> {
        let cfg_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&cfg_path).map_err(|_| {
            err!(
                "gguf",
                "gemma-4 gguf needs the HF config.json beside it ({} not found) — re-pull; quantizer repos ship it",
                cfg_path.display()
            )
        })?;
        let j = json::parse(&raw)
            .map_err(|e| err!("config", "{}/config.json is not valid JSON: {}", dir.display(), e))?;
        let mut g4 = G4Config::parse(&j)?;
        // The rope recipe comes from the GGUF's own metadata, not the
        // config.json sidecar: the export's weights are consistent with the
        // graph llama.cpp builds FROM that metadata. Empirically settled on
        // the E4B export — config.json declares proportional 0.25 partial
        // rotary, the gguf declares full-width rotation + rope_freqs
        // factors, and only the latter generates correctly.
        // CIMA_G4_FULL_ROTARY=0 disables this override entirely (config.json
        // recipe wins) — the A/B lever for exports whose metadata recipe may
        // not match their weights the way the E4B export's did.
        if crate::models::gemma4::support::env_flag("CIMA_G4_FULL_ROTARY") != Some(false) {
            let t = &mut g4.text;
            if let Some(dims) = weights.meta_usize("gemma4.rope.dimension_count") {
                let nf = (dims / 2).clamp(1, t.global_head_dim / 2);
                if nf != t.full_nfreqs {
                    crate::log::info(&format!(
                        "gemma4-gguf rope: full-attention nfreqs {} (metadata rope.dimension_count {}) overrides config.json's {}",
                        nf, dims, t.full_nfreqs
                    ));
                    t.full_nfreqs = nf;
                }
            }
            if let Some(fb) = weights.meta_f32("gemma4.rope.freq_base") {
                t.theta_full = fb;
            }
            if let Some(fb) = weights.meta_f32("gemma4.rope.freq_base_swa") {
                t.theta_sliding = fb;
            }
            t.use_rope_factors = true; // guarded by tensor presence at build
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                eprintln!(
                    "g4 rope recipe (gguf): full layers nfreqs {} of {} pairs, theta {} | sliding theta {} | rope_freqs factors: on when shipped",
                    t.full_nfreqs, t.global_head_dim / 2, t.theta_full, t.theta_sliding
                );
            }
        }
        let codec = crate::quant::gguf::GgufCodec { resident: true };
        let (wb, kvb, wsb) = Gemma4::forecast_bytes(&g4, &weights, &codec);
        crate::log::info(&format!(
            "gemma4-gguf forecast: {} weights + {} KV + {} workspace (PLE table stays in host RAM)",
            crate::cuda::fmt_bytes(wb), crate::cuda::fmt_bytes(kvb), crate::cuda::fmt_bytes(wsb)
        ));
        // PRODUCTION SAFETY: this path used to LOG the forecast and then
        // allocate regardless — announcing "does not fit" and trying anyway,
        // which OOMs mid-load and can swap-thrash the whole machine. The
        // forecast must gate the load here exactly as on every other path.
        {
            let transient = weights
                .tensors()
                .values()
                .filter(|t| !matches!(t.dtype, crate::traits::DType::F16 | crate::traits::DType::U8))
                .filter(|t| {
                    !t.name.ends_with("embed_tokens_per_layer.weight")
                        && !t.name.contains("subsample_conv_projection.")
                        && !crate::quant::bnb::is_aux(&t.name)
                })
                .map(|t| t.numel() * t.dtype.size())
                .max()
                .unwrap_or(0);
            let forecast = VramForecast { weights: wb, kv_cache: kvb, workspace: wsb, load_transient: transient };
            forecast.check(&self.ctx, name)?;
            VramForecast::host_guard(&weights, name)?;
        }
        // Tokenizer: prefer the repo's tokenizer.json (SPM-aware loader);
        // otherwise synthesize from the gguf metadata — from_gguf detects
        // the SentencePiece lineage via the <0xXX> byte-fallback tokens.
        let tokenizer = if dir.join("tokenizer.json").exists() {
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                eprintln!("g4 tokenizer source: repo tokenizer.json (gguf-metadata synth as fallback)");
            }
            BpeTokenizer::load(dir).or_else(|_| tokenizer_from_gguf_meta(&weights, true))?
        } else {
            // expected for gguf-only repos: synthesize from metadata
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                eprintln!("g4 tokenizer source: synthesized from gguf metadata (no repo tokenizer.json)");
            }
            tokenizer_from_gguf_meta(&weights, true)?
        };
        // Media-marker ids: quantizer repos ship trimmed config.json files
        // that omit some of the six marker ids (this export carries the
        // image trio but not the audio one), and the parser's silent 0
        // default frames media spans with <pad> — the model then correctly
        // reports "no audio". The tokenizer registers the markers as
        // specials, so resolve any missing id from the literals (gemma-4
        // dialect first, classic second). Done at load time so the PLE
        // placeholder logic sees the same ids as the splice.
        {
            let resolve = |cur: u32, lits: &[&str], what: &str| -> u32 {
                if cur != 0 {
                    return cur;
                }
                for l in lits {
                    if let Some(id) = tokenizer.special(l) {
                        if std::env::var("CIMA_G4_DEBUG").is_ok() {
                            eprintln!("g4 media marker {} missing from config.json — resolved from tokenizer special {:?} = {}", what, l, id);
                        }
                        return id;
                    }
                }
                cur
            };
            g4.audio_token_id = resolve(g4.audio_token_id, &["<|audio|>", "<audio_soft_token>"], "audio_token_id");
            g4.boa_token_id = resolve(g4.boa_token_id, &["<|audio>", "<start_of_audio>"], "boa_token_id");
            g4.eoa_token_id = resolve(g4.eoa_token_id, &["<audio|>", "<end_of_audio>"], "eoa_token_id");
            g4.image_token_id = resolve(g4.image_token_id, &["<|image|>", "<image_soft_token>"], "image_token_id");
            g4.boi_token_id = resolve(g4.boi_token_id, &["<|image>", "<start_of_image>"], "boi_token_id");
            g4.eoi_token_id = resolve(g4.eoi_token_id, &["<image|>", "<end_of_image>"], "eoi_token_id");
        }
        let chat_template = weights.meta_str("tokenizer.chat_template").map(str::to_owned);
        if std::env::var("CIMA_G4_DEBUG").is_ok() {
            // Full metadata dump: the export's hyperparameters are the graph
            // llama.cpp runs; any disagreement with the config.json-derived
            // G4Config is a candidate for the quality gap. Long values are
            // truncated; the token/merge arrays are elided.
            let mut keys: Vec<&String> = weights.meta.keys().collect();
            keys.sort();
            for k in keys {
                use crate::formats::gguf::Value as V;
                let v = &weights.meta[k.as_str()];
                let shown = match v {
                    V::Arr(a) if a.len() > 8 => format!("[{} items]", a.len()),
                    V::Str(s) if s.chars().count() > 100 => {
                        format!("{:?}… ({} chars)", s.chars().take(100).collect::<String>(), s.chars().count())
                    }
                    other => format!("{:?}", other),
                };
                eprintln!("gguf meta {} = {}", k, shown);
            }
        }
        let arch = Arch::Gemma4(Gemma4::build(self.ctx.clone(), g4, Box::new(weights), &codec)?);
        let media_token = MEDIA_LITERALS
            .iter()
            .find(|l| tokenizer.special(l).is_some())
            .map(|s| s.to_string());
        Ok(LoadedModel {
            name: name.to_string(),
            dir: dir.to_path_buf(),
            arch,
            tokenizer,
            media: MediaRegistry::standard(),
            media_token,
            chat_template,
            session_ids: Vec::new(),
            session_text: String::new(),
        })
    }

    fn load_gguf(&self, name: &str, repo: &str, tag: Option<&str>, dir: PathBuf, ggufs: Vec<PathBuf>) -> Res<LoadedModel> {
        let t0 = Instant::now();
        let snap0 = self.ctx.snapshot();
        if ggufs.is_empty() {
            return Err(err!(
                "models",
                "model '{}' is not present locally (no .gguf under {}). Run `cima pull {}` first.",
                name,
                dir.display(),
                name
            ));
        }
        let quants: Vec<String> = ggufs
            .iter()
            .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
            .collect();
        // mmproj companions are tower files, never runnable quants: they
        // must not participate in tag resolution (mmproj-F16 would match
        // tag F16) — they merge into whichever quant is chosen, below.
        let is_mmproj = |p: &&std::path::PathBuf| {
            p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase().contains("mmproj")
        };
        let chosen: Vec<&std::path::PathBuf> = match tag {
            Some(t) => {
                let tl = t.to_ascii_lowercase();
                ggufs
                    .iter()
                    .filter(|p| !is_mmproj(p))
                    .filter(|p| p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase().contains(&tl))
                    .collect()
            }
            None => ggufs.iter().filter(|p| !is_mmproj(p)).collect(),
        };
        let path = match chosen.as_slice() {
            [one] => (*one).clone(),
            [] => {
                return Err(err!(
                    "gguf",
                    "no .gguf in {} matches tag '{}'. Available: {}",
                    dir.display(), tag.unwrap_or(""), quants.join(", ")
                ))
            }
            many => {
                // Prefer the shortest matching filename (Q4_K vs Q4_K_XL
                // style overlaps resolve to the exact-most tag).
                let exact = many.iter().min_by_key(|p| p.file_name().unwrap_or_default().len()).unwrap();
                if tag.is_none() {
                    return Err(err!(
                        "gguf",
                        "{} holds {} quantizations — pick one: {}",
                        repo,
                        many.len(),
                        quants.iter().map(|q| format!("{}:{}", repo, crate::hub::quant_tag(q))).collect::<Vec<_>>().join(", ")
                    ));
                }
                (**exact).clone()
            }
        };

        let mut weights = crate::formats::gguf::GgufWeights::open(&path)?;
        // Multimodal towers ship as companion mmproj files (llama.cpp keeps
        // the LM gguf text-only). Merge every mmproj in the repo dir so the
        // tower builders can find their tensors; a repo without one loads
        // text-only exactly as before.
        {
            // The mmproj is a CAPABILITY of the model, not a variant: merge
            // exactly one. Repos ship several precisions of the same towers
            // (F16/BF16/F32) — pick the best fit and ignore the rest, or
            // every tensor of the second file "collides" with the first.
            let mmprojs: Vec<&std::path::PathBuf> = ggufs
                .iter()
                .filter(|p| {
                    p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase().contains("mmproj")
                })
                .collect();
            let rank = |p: &std::path::PathBuf| {
                let n = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
                if n.contains("bf16") {
                    1 // bf16 → f16 conversion loses mantissa; prefer true f16
                } else if n.contains("f16") {
                    0
                } else if n.contains("f32") {
                    2
                } else {
                    3
                }
            };
            if let Some(best) = mmprojs.iter().min_by_key(|p| rank(p)) {
                if **best != path {
                    weights.merge_extra(best)?;
                    if mmprojs.len() > 1 {
                        crate::log::info(&format!(
                            "gguf: {} mmproj precisions present; using '{}'",
                            mmprojs.len(),
                            best.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    }
                }
            }
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                let mut vt = 0usize;
                let mut at = 0usize;
                let mut emb = 0usize;
                let mut other: Vec<&str> = Vec::new();
                for k in weights.tensors().keys() {
                    if k.starts_with("vision_tower.") {
                        vt += 1;
                    } else if k.starts_with("audio_tower.") {
                        at += 1;
                    } else if k.starts_with("embed_vision.") || k.starts_with("embed_audio.") {
                        emb += 1;
                    } else if !k.starts_with("language_model.") && !k.starts_with("lm_head") {
                        other.push(k);
                    }
                }
                other.sort();
                eprintln!(
                    "g4 tower census: {} vision_tower.* | {} audio_tower.* | {} embed_*.* | {} unrecognized",
                    vt, at, emb, other.len()
                );
                // LM-critical tensor shapes: the E-series PLE table and
                // embeddings are where a silently-accepted reshape (same
                // leading dim, same numel, different axis ORDER) would
                // lobotomize comprehension while leaving speech fluent.
                for name in [
                    "embed_tokens.weight",
                    "embed_tokens_per_layer.weight",
                    "model.embed_tokens.weight",
                    "model.embed_tokens_per_layer.weight",
                ] {
                    if let Some(m) = weights.tensors().get(name) {
                        eprintln!("g4 lm tensor: {} shape {:?} dtype {}", name, m.shape, m.dtype.name());
                    }
                }
                // Calibration fingerprints: the mmproj ships per-weight
                // activation ranges (input/output min/max). They identify
                // each slot semantically — GELU-fed inputs are one-sided
                // (min ≈ −0.17), post-LayerNorm inputs symmetric, later
                // macaron halves see larger residuals — which pins the
                // same-shape mapping ambiguities the shape checks cannot.
                let scalar = |name: &str| -> Option<f32> {
                    let t = weights.tensors();
                    let m = t.get(name)?;
                    let b = weights.bytes(m).ok()?;
                    if b.len() >= 4 {
                        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    } else {
                        None
                    }
                };
                for layer in [0usize, 6] {
                    for stem in [
                        "ffn_up", "ffn_down", "ffn_up_1", "ffn_down_1",
                        "attn_q", "attn_k", "attn_v", "attn_out",
                        "conv_pw1", "conv_pw2",
                    ] {
                        let base = format!("audio_tower.layers.{}.{}", layer, stem);
                        let g = |suf: &str| scalar(&format!("{}.{}", base, suf)).map(|v| format!("{:+.3}", v)).unwrap_or_else(|| "—".into());
                        eprintln!(
                            "g4 calib L{} {:<10} in [{} .. {}]  out [{} .. {}]",
                            layer, stem, g("input_min"), g("input_max"), g("output_min"), g("output_max")
                        );
                    }
                }
                for k in other.iter().take(48) {
                    eprintln!("g4 unmapped tensor: {}", k);
                }
            }
        }
        if weights.architecture.starts_with("gemma4") || weights.architecture.starts_with("gemma3n") {
            return self.load_gguf_gemma4(name, &dir, weights);
        }
        let std_archs = ["qwen2", "llama", "mistral", "qwen3"];
        if !std_archs.contains(&weights.architecture.as_str()) {
            // Name the evidence, not just the verdict: hybrid SSM
            // (Mamba-class) checkpoints carry `ARCH.ssm.*` keys — those
            // need a state-space execution engine, and an alias onto the
            // attention graph would degrade silently instead of failing
            // loudly here.
            let hybrid = weights.meta.keys().any(|k| k.contains(".ssm."));
            let why = if hybrid {
                "this is a hybrid attention+SSM (Mamba-class) architecture (`.ssm.*` metadata keys) — it needs a state-space execution engine, not a graph alias"
            } else {
                "inspect its hyper-parameters with `cima check REPO:TAG --meta`"
            };
            return Err(err!(
                "gguf",
                "GGUF architecture '{}' is not registered for execution in this build (registered: {}); {}",
                weights.architecture,
                std_archs.join(", "),
                why
            ));
        }
        let cfg = crate::formats::gguf::model_config(&weights)?;
        let codec = crate::quant::gguf::GgufCodec { resident: true };
        let forecast = VramForecast::compute(&cfg, &weights, &codec);
        forecast.check(&self.ctx, name)?;
        VramForecast::host_guard(&weights, name)?;
        let arch = Arch::Std(Transformer::build(self.ctx.clone(), cfg, &weights, &codec)?);

        // Tokenizer + chat template from metadata.
        let tk_model = weights.meta_str("tokenizer.ggml.model").unwrap_or("gpt2");
        if tk_model != "gpt2" {
            return Err(err!(
                "gguf",
                "tokenizer.ggml.model '{}' is not wired (gpt2/byte-level only for now)",
                tk_model
            ));
        }
        let tokenizer = tokenizer_from_gguf_meta(&weights, false)?;
        let chat_template = weights.meta_str("tokenizer.chat_template").map(str::to_owned);
        if std::env::var("CIMA_G4_DEBUG").is_ok() {
            // Full metadata dump: the export's hyperparameters are the graph
            // llama.cpp runs; any disagreement with the config.json-derived
            // G4Config is a candidate for the quality gap. Long values are
            // truncated; the token/merge arrays are elided.
            let mut keys: Vec<&String> = weights.meta.keys().collect();
            keys.sort();
            for k in keys {
                use crate::formats::gguf::Value as V;
                let v = &weights.meta[k.as_str()];
                let shown = match v {
                    V::Arr(a) if a.len() > 8 => format!("[{} items]", a.len()),
                    V::Str(s) if s.chars().count() > 100 => {
                        format!("{:?}… ({} chars)", s.chars().take(100).collect::<String>(), s.chars().count())
                    }
                    other => format!("{:?}", other),
                };
                eprintln!("gguf meta {} = {}", k, shown);
            }
        }

        let snap1 = self.ctx.snapshot();
        log::metric(
            "model_load",
            &[
                ("model", name.to_string()),
                ("load_ms", format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3)),
                ("vram_pre", snap0.vram_used.to_string()),
                ("vram_post", snap1.vram_used.to_string()),
                ("vram_delta", (snap1.vram_used as i64 - snap0.vram_used as i64).to_string()),
                ("modality", arch.modality().name().to_string()),
                ("capabilities", arch.capabilities().iter().map(|c| c.name()).collect::<Vec<_>>().join("+")),
            ],
        );
        Ok(LoadedModel {
            name: name.to_string(),
            dir,
            arch,
            tokenizer,
            media: MediaRegistry::standard(),
            media_token: None,
            chat_template,
            session_ids: Vec::new(),
            session_text: String::new(),
        })
    }
}

/// Synthesize a tokenizer from `tokenizer.ggml.*` GGUF metadata (the one
/// decode of that key family — both the standard and the gemma4 gguf load
/// paths run through here). `add_bos_default` covers metadata that omits
/// `add_bos_token`: SentencePiece-lineage exports (gemma) default to true,
/// gpt2/byte-level exports to false.
fn tokenizer_from_gguf_meta(w: &crate::formats::gguf::GgufWeights, add_bos_default: bool) -> Res<BpeTokenizer> {
    let arr = |key: &str| w.meta.get(key).and_then(|v| v.as_arr());
    let tokens: Vec<String> = arr("tokenizer.ggml.tokens")
        .ok_or_else(|| err!("gguf", "metadata is missing tokenizer.ggml.tokens"))?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    let merges: Vec<String> = arr("tokenizer.ggml.merges")
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect())
        .unwrap_or_default();
    let types: Vec<i64> = arr("tokenizer.ggml.token_type")
        .map(|a| a.iter().map(|v| v.as_u64().map(|u| u as i64).unwrap_or(1)).collect())
        .unwrap_or_default();
    let scores: Vec<f32> = arr("tokenizer.ggml.scores")
        .map(|a| a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
        .unwrap_or_default();
    let bos = w.meta_usize("tokenizer.ggml.bos_token_id").map(|v| v as u32);
    let eos: Vec<u32> = w.meta_usize("tokenizer.ggml.eos_token_id").map(|e| e as u32).into_iter().collect();
    let add_bos = w
        .meta
        .get("tokenizer.ggml.add_bos_token")
        .and_then(|v| v.as_bool())
        .unwrap_or(add_bos_default);
    BpeTokenizer::from_gguf(&tokens, &merges, &types, bos, eos, add_bos, scores)
}
#[cfg(test)]
mod sampler_tests {
    use super::DefaultSampler;
    use crate::traits::{GenOptions, LogitsSampler};

    fn opts(temp: f32, top_p: f32, top_k: usize) -> GenOptions {
        GenOptions {
            temperature: temp,
            top_p,
            top_k,
            repeat_penalty: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn stop_is_exclusive_and_position_based() {
        use super::earliest_stop;
        // Trailing chars after the stop (the "Banana\n" case) must still
        // trigger, trimming from the stop's start.
        let t = "1. Apple\n2. Banana\n3. Cherry";
        let pos = earliest_stop(t, &["Banana".to_string()]).unwrap();
        assert_eq!(&t[..pos], "1. Apple\n2. ");
        // Case-sensitive: lowercase stop does not match "Banana".
        assert_eq!(earliest_stop(t, &["banana".to_string()]), None);
        // Earliest of several wins.
        let pos2 = earliest_stop(t, &["Cherry".to_string(), "Apple".to_string()]).unwrap();
        assert_eq!(&t[..pos2], "1. ");
        // Empty stops are ignored (never an instant abort at 0).
        assert_eq!(earliest_stop(t, &["".to_string()]), None);
        // No match → None.
        assert_eq!(earliest_stop(t, &["Durian".to_string()]), None);
        // The case that defeated an `ends_with` implementation: the stop
        // arrives fused with a trailing char (" green," in one piece), so the
        // accumulated text does NOT end with the stop — position-based find
        // still catches it and trims from the stop's start.
        let fused = "red, green, and blue.";
        let p = earliest_stop(fused, &["green".to_string()]).unwrap();
        assert_eq!(&fused[..p], "red, ");
    }

    #[test]
    fn greedy_picks_argmax_and_is_allocation_stable() {
        // Greedy must return the max-logit index regardless of buffer reuse
        // across repeated calls (the hoisted scratch must not leak state).
        let mut s = DefaultSampler::new(1);
        let mut logits = vec![0.1f32, 0.5, 0.2, 0.9, 0.3];
        let o = opts(0.0, 1.0, 0);
        for _ in 0..8 {
            assert_eq!(s.sample(&mut logits.clone(), &[], &o), 3);
        }
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        // Same seed + same logits => same token every run; reused buffers
        // must not perturb the draw.
        let logits = vec![2.0f32, 1.0, 0.5, 0.25, 0.1, 3.0, 0.7, 1.5];
        let o = opts(0.8, 0.95, 5);
        let mut a = DefaultSampler::new(0xABCDEF);
        let first = a.sample(&mut logits.clone(), &[], &o);
        let mut b = DefaultSampler::new(0xABCDEF);
        assert_eq!(first, b.sample(&mut logits.clone(), &[], &o));
        // And a run of draws is reproducible under buffer reuse.
        let mut c = DefaultSampler::new(7);
        let seq1: Vec<u32> = (0..16).map(|_| c.sample(&mut logits.clone(), &[], &o)).collect();
        let mut d = DefaultSampler::new(7);
        let seq2: Vec<u32> = (0..16).map(|_| d.sample(&mut logits.clone(), &[], &o)).collect();
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn top_p_one_can_reach_any_candidate() {
        // With top_p=1 and top_k=0 the whole vocab is eligible; every draw
        // must be a valid index.
        let logits = vec![1.0f32; 100];
        let o = opts(1.0, 1.0, 0);
        let mut s = DefaultSampler::new(99);
        for _ in 0..1000 {
            assert!(s.sample(&mut logits.clone(), &[], &o) < 100);
        }
    }

    #[test]
    fn candidate_path_matches_full_path_when_greedy() {
        // The device-candidate tail and the full-logits path share finish();
        // at temperature 0 both must return the top candidate.
        let mut s = DefaultSampler::new(3);
        let vals = [5.0f32, 3.0, 1.0];
        let ids = [42u32, 7, 9];
        assert_eq!(s.sample_candidates(&vals, &ids, &opts(0.0, 1.0, 0)), 42);
    }
}