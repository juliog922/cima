//! # omni — any-to-any architectures (text + vision + audio → text)
//!
//! Currently binds the `gemma4` model family; the module is named for the
//! capability it implements, not the vendor.
//!
//! Implemented 1:1 from the `transformers` reference (`modeling_gemma4.py`,
//! version 5.5.0.dev0). The family departs from the llama/qwen pipeline in
//! every block, so it lives in its own module behind the [`Architecture`]
//! seam:
//!
//! * **Text decoder** — sandwich RMSNorms (pre+post around attention *and*
//!   MLP), GeGLU activation, QK-RMSNorm plus scale-less V-RMSNorm, attention
//!   scale 1.0, alternating sliding(512)/full attention layers with per-type
//!   RoPE theta, "proportional" partial RoPE on full layers (`global_head_dim`
//!   512, only the first 25% of frequency pairs rotate), **KV-cache sharing**
//!   (the last `num_kv_shared_layers` layers carry no K/V projections and
//!   reuse the cache of the last non-shared layer of the same type), double-
//!   wide MLP on shared layers, `sqrt(hidden)` embedding scaling, per-layer
//!   embeddings (PLE), and `tanh` soft-capping of the final logits.
//! * **PLE** — the token-identity table (`embed_tokens_per_layer`,
//!   `vocab × layers·256` ≈ multiple GiB) **never enters VRAM**: rows are
//!   gathered on the CPU straight out of the zero-copy mmap per chunk/token,
//!   while the context projection runs on the GPU.
//! * **Vision tower** — linear patch embed over `2·(p−0.5)` pixels, learned
//!   2-D position table, 16 bidirectional layers with 2-D RoPE (θ=100) and
//!   the same sandwich/GeGLU/QK-norm block, 3×3 average pooling, `sqrt(h)`
//!   scaling, then a scale-less-RMSNorm + linear multimodal embedder. Image
//!   token spans attend **bidirectionally** inside the causal LM prefill.
//! * **Audio tower** — USM-style conformer: two 3×3 stride-2 convs (CPU im2col
//!   + LayerNorm/ReLU), 12 layers of {½-residual FFN, chunked local attention
//!     (chunk 12, left 13) with relative-position bias and tanh logit cap 50,
//!     causal depthwise light-conv (GLU, k=5), ½-residual FFN, RMSNorm}, output
//!     projection, then the multimodal embedder.

pub(crate) mod support;
mod towers;

pub use support::config::{G4Config, LayerType};

use towers::audio::G4Audio;

use crate::num::{bf16_to_f32, f16_to_f32, f32_to_f16};
use support::weights::G4Index;
use towers::vision::{G4Vision, G4_MAX_PATCHES, G4_SIDE_MULT};

use crate::cuda::{CudaCtx, DeviceBuf};
use crate::err;
use std::sync::Arc;

use crate::log;
use crate::models::CHUNK;
use crate::quant::bnb::{self, WTensor};
use crate::traits::{
    AudioPcm, DType, ImageTensor, LoadedWeights, Modality, PreparedPrompt, Res, TensorMeta,
    WeightCodec,
};

// ===========================================================================
// Text decoder
// ===========================================================================

/// Per-layer KV source map, reference semantics: a shared layer reads the
/// cache written by the **last computing layer of its own attention type**
/// (sliding → last computing sliding layer, full → last computing full
/// layer); computing layers read their own. Pure function — locked by unit
/// tests because a silently wrong source map degrades text *gracefully*
/// while destroying precise multimodal retrieval (hard-won lesson).
pub(crate) fn kv_share_sources(layer_types: &[LayerType], first_shared: usize) -> Res<Vec<usize>> {
    let mut last_of_type: [usize; 2] = [usize::MAX, usize::MAX];
    for (i, ty) in layer_types.iter().enumerate().take(first_shared) {
        last_of_type[if *ty == LayerType::Sliding { 0 } else { 1 }] = i;
    }
    layer_types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            if i < first_shared {
                Ok(i)
            } else {
                let src = last_of_type[if *ty == LayerType::Sliding { 0 } else { 1 }];
                if src == usize::MAX {
                    return Err(err!("weights", "gemma4: layer {} ({:?}) shares KV but no earlier layer of that type computes it", i, ty));
                }
                Ok(src)
            }
        })
        .collect()
}

/// Debug: read a device f16 matrix back and write it as the A/B
/// interchange format `[u32 rows][u32 cols][f32 …]` — the serializer shared
/// by every `CIMA_DUMP_*` probe (soft tokens, LM embeds, audio frontend,
/// embedding rows).
/// Under CIMA_G4_DEBUG: list a failed tower's ACTUAL tensor names so
/// translation gaps can be closed against ground truth in one round trip
/// (the build error names the tensor the builder wanted; this names what
/// the export shipped).
fn dump_tower_names(weights: &dyn crate::traits::LoadedWeights, prefix: &str) {
    if std::env::var("CIMA_G4_DEBUG").is_err() {
        return;
    }
    // Per-layer block names repeat per layer and are already mapped; the
    // calibration stats are never requested. Filter both so the dump shows
    // the top-level singletons — exactly the names still unknown.
    let solved = |k: &str| {
        k.contains(".layers.")
            || k.contains(".blk.")
            || k.ends_with(".input_min")
            || k.ends_with(".input_max")
            || k.ends_with(".output_min")
            || k.ends_with(".output_max")
    };
    let all: Vec<&String> = weights
        .tensors()
        .keys()
        .filter(|k| k.starts_with(prefix))
        .collect();
    let mut names: Vec<&&String> = all.iter().filter(|k| !solved(k)).collect();
    names.sort();
    eprintln!(
        "g4 {}* holds {} tensors ({} outside layer blocks):",
        prefix,
        all.len(),
        names.len()
    );
    for n in names.iter().take(80) {
        eprintln!("g4 shipped: {}", n);
    }
}

pub(crate) fn dump_f16_matrix(
    ctx: &CudaCtx,
    ptr: u64,
    rows: usize,
    cols: usize,
    path: &str,
    what: &str,
) -> Res<()> {
    ctx.sync()?;
    let mut b = vec![0u8; rows * cols * 2];
    ctx.dtoh_at(&mut b, ptr)?;
    let mut out = Vec::with_capacity(8 + rows * cols * 4);
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(cols as u32).to_le_bytes());
    for c in b.chunks_exact(2) {
        out.extend_from_slice(&f16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| err!("debug", "cannot write '{}': {}", path, e))?;
    log::info(&format!("{} dumped to {} ({}×{})", what, path, rows, cols));
    Ok(())
}

/// of `kv_src` and carry no K/V projections or norms.
struct G4Layer {
    input_norm: DeviceBuf,
    post_attn_norm: DeviceBuf,
    pre_ffw_norm: DeviceBuf,
    post_ffw_norm: DeviceBuf,
    wq: WTensor,
    q_norm: DeviceBuf,
    wk: Option<WTensor>,
    wv: Option<WTensor>,
    k_norm: Option<DeviceBuf>,
    wo: WTensor,
    w_gate: WTensor,
    w_up: WTensor,
    w_down: WTensor,
    // Per-layer-embedding block
    ple_gate: Option<WTensor>,
    ple_proj: Option<WTensor>,
    ple_norm: Option<DeviceBuf>,
    /// `layer_scalar` checkpoint buffer (ones in this release; honored anyway).
    scalar: f32,
    /// Proportional-RoPE frequency factors (full-attention layers in the
    /// llama.cpp export); divides each pair frequency.
    rope_factors: Option<DeviceBuf>,
    /// KV cache, allocated only on non-shared layers.
    kv: Option<(DeviceBuf, DeviceBuf)>,
    /// Index of the layer whose cache this layer reads (== own index when not shared).
    kv_src: usize,
    head_dim: usize,
    inter: usize,
    theta: f32,
    nfreqs: usize,
    window: usize, // 0 = full attention
}

/// Device workspace sized for `CHUNK` rows.
struct G4Ws {
    x: DeviceBuf,  // residual stream      [CHUNK, hidden]
    h: DeviceBuf,  // normed activations   [CHUNK, hidden]
    h2: DeviceBuf, // block outputs        [CHUNK, hidden]
    q: DeviceBuf,  // [CHUNK, heads*max_head_dim]
    k: DeviceBuf,  // [CHUNK, kv_heads*max_head_dim]
    v: DeviceBuf,
    att: DeviceBuf,  // [CHUNK, heads*max_head_dim]
    gate: DeviceBuf, // [CHUNK, max_inter]
    up: DeviceBuf,
    ple: DeviceBuf,    // combined PLE slab    [CHUNK, layers*ple_dim]
    ple_id: DeviceBuf, // token-identity slab  [CHUNK, layers*ple_dim]
    ple_g: DeviceBuf,  // gated activations    [CHUNK, ple_dim]
    wsc: DeviceBuf,    // NF4 prefill dequant scratch [max quantized tensor, f16]
    blkid: DeviceBuf,  // absolute-position media-block ids [max_seq] i32
    ids: DeviceBuf,    // token ids            [CHUNK] u32
    logits_h: DeviceBuf,
    logits_f: DeviceBuf,
    /// 8-byte scratch for the device-side greedy argmax (packed val|idx).
    argmax_slot: DeviceBuf,
    /// Device position counter (partial-graph decode).
    pos_dev: DeviceBuf,
    /// Flash-decode partials `[n_heads, n_chunks, max_head_dim+2]` f32.
    att_part: DeviceBuf,
    /// Device-sampling state (see transformer.rs::SAMPLE_TOPK).
    cand: DeviceBuf,
    hist_ring: DeviceBuf,
    hist_counts: DeviceBuf,
}

/// The resident Gemma 4 graph (text decoder + optional towers).
pub struct Gemma4 {
    pub ctx: Arc<CudaCtx>,
    cfg: G4Config,
    /// Pinned host mapping retained for CPU-side PLE row gathers.
    weights: Box<dyn LoadedWeights>,
    ple_meta: Option<TensorMeta>,
    embed: DeviceBuf,
    /// Some(fmt) when the tied embedding lives packed (gguf checkpoints):
    /// lookups run gguf_gather, the head runs the dp4a gguf GEMV.
    embed_gguf: Option<DType>,
    /// Debug: tokens of the chunk currently in flight (NANHUNT labels).
    dbg_tokens: Vec<u32>,
    embed_bf16: bool,
    final_norm: DeviceBuf,
    ple_proj_w: Option<WTensor>, // per_layer_model_projection [layers*ple, hidden]
    ple_proj_norm: Option<DeviceBuf>, // per_layer_projection_norm [ple]
    layers: Vec<G4Layer>,
    ws: G4Ws,
    /// Host mirror of media-block ids for every absolute position so far.
    blk_host: Vec<i32>,
    /// Accumulated host-side milliseconds since the last `perf_take`:
    /// (PLE row gathers, logits dtoh+softcap).
    perf_ple_ms: f64,
    perf_logits_ms: f64,
    pub pos: usize,
    vram: usize,
    vision: Option<G4Vision>,
    audio: Option<G4Audio>,
    /// Resident weight bytes (the per-token streaming floor).
    pub(crate) weight_bytes: usize,
    /// Captured decode tail (everything after the host PLE upload).
    decode_graph: Option<crate::cuda::GraphExec>,
    /// Captured sampling tail (penalty + top-64); rp baked.
    sample_graph: Option<(crate::cuda::GraphExec, f32)>,
}

impl Gemma4 {
    /// Predicted VRAM for the forecast gauntlet, mirroring exactly what
    /// `build` allocates. The PLE token-identity table is host-resident and
    /// therefore **excluded** from device weight bytes; it is reported as
    /// `load_transient = 0` since it is never staged through VRAM.
    /// Bytes the prefill dequant scratch (`ws.wsc`) must hold: the largest
    /// single weight that flows through `WTensor::gemm` with `m > 1`,
    /// expanded to dense f16. Shared by [`Gemma4::forecast_bytes`] and the
    /// `build` allocation so the bill and the buffer can never disagree —
    /// they once did (the allocation kept only the NF4 term), which left a
    /// 2-byte scratch for GGUF checkpoints and every prefill GEMM sprayed
    /// megabytes of dequantized weights over the neighbouring device
    /// buffers. Correct-looking output (the GEMM reads back exactly what
    /// the dequant wrote) with corrupted-everything-else was the symptom.
    fn prefill_scratch_bytes(weights: &dyn LoadedWeights) -> usize {
        let mut need = 0usize;
        for (name, meta) in weights.tensors() {
            // Host-resident pieces never enter a device prefill GEMM.
            if name.ends_with("embed_tokens_per_layer.weight")
                || name.contains("subsample_conv_projection.")
                || bnb::is_aux(name)
            {
                continue;
            }
            if bnb::state_name(weights.tensors(), name).is_some() {
                // NF4: packed nibbles expand ×2 to elements, ×2 to f16 bytes.
                need = need.max(meta.nbytes * 4);
                continue;
            }
            // gguf block weights dequantize whole into the scratch
            // (WTensor::Gguf, m>1). The tied embedding is the one
            // exception: it never enters a prefill GEMM (device gather +
            // dp4a head), and sizing scratch for a 256k-row table would
            // burn 1.3 GiB for nothing.
            if crate::traits::is_gguf_block(meta.dtype) && !name.ends_with("embed_tokens.weight") {
                let numel: usize = meta.shape.iter().product();
                need = need.max(numel * 2);
            }
        }
        need
    }

    pub fn forecast_bytes(
        cfg: &G4Config,
        weights: &dyn LoadedWeights,
        codec: &dyn WeightCodec,
    ) -> (usize, usize, usize) {
        let t = &cfg.text;
        let mut w_bytes = 0usize;
        let wsc_bytes = Self::prefill_scratch_bytes(weights);
        for (name, meta) in weights.tensors() {
            // host-resident pieces: PLE table + audio subsample convs
            if name.ends_with("embed_tokens_per_layer.weight")
                || name.contains("subsample_conv_projection.")
            {
                continue;
            }
            // bitsandbytes families: aux tensors are consumed on the host;
            // packed weights stay resident as nibbles + folded f32 absmax.
            if bnb::is_aux(name) {
                continue;
            }
            if bnb::state_name(weights.tensors(), name).is_some() {
                let bs = bnb::parse_state(weights, name)
                    .map(|s| s.blocksize)
                    .unwrap_or(64);
                w_bytes += bnb::nf4_device_bytes(meta.nbytes * 2, bs);
                continue;
            }
            w_bytes += codec.device_bytes(meta);
        }
        let first_shared = t.n_layers - t.n_kv_shared;
        let mut kv_bytes = 0usize;
        for (i, ty) in t.layer_types.iter().enumerate() {
            if i >= first_shared {
                continue;
            }
            let d = match ty {
                LayerType::Sliding => t.head_dim,
                LayerType::Full => t.global_head_dim,
            };
            kv_bytes += 2 * t.n_kv_heads * t.max_seq * d * 2;
        }
        let dmax = t.head_dim.max(t.global_head_dim);
        let imax = if t.double_wide_mlp {
            t.inter * 2
        } else {
            t.inter
        };
        let mut ws = CHUNK * t.hidden * 2 * 3                     // x, h, h2
            + CHUNK * t.n_heads * dmax * 2 * 2                    // q, att
            + CHUNK * t.n_kv_heads * dmax * 2 * 2                 // k, v
            + CHUNK * imax * 2 * 2                                // gate, up
            + t.max_seq * 4                                       // blkid
            + CHUNK * 4                                           // ids
            + t.vocab * 6; // logits f16+f32
        if t.ple_dim > 0 {
            ws += CHUNK * t.n_layers * t.ple_dim * 2 * 2 + CHUNK * t.ple_dim * 2;
        }
        if let Some(v) = &cfg.vision {
            // patch scratch for a 960×672 image: 2520 patches
            let n = 2520;
            ws += n * v.hidden * 2 * 6 + n * v.inter * 2 * 2 + n * v.n_heads * v.head_dim * 2 * 3;
        }
        if let Some(a) = &cfg.audio {
            let n = 3008; // max subsampled frames before the 750-token cap, padded
            ws += n * a.hidden * 2 * 8;
        }
        ws += wsc_bytes; // NF4 prefill dequant scratch
        (w_bytes, kv_bytes, ws)
    }

    /// Build the full resident graph. `weights` is moved in and retained for
    /// the lifetime of the model: PLE token-identity rows are gathered from
    /// its pinned mmap on every chunk.
    pub fn build(
        ctx: Arc<CudaCtx>,
        cfg: G4Config,
        weights: Box<dyn LoadedWeights>,
        codec: &dyn WeightCodec,
    ) -> Res<Gemma4> {
        let t0 = std::time::Instant::now();
        let weight_bytes = Gemma4::forecast_bytes(&cfg, weights.as_ref(), codec).0;
        let t = &cfg.text;
        let hs = t.hidden;
        let first_shared = t.n_layers - t.n_kv_shared;

        let (embed, embed_bf16, final_norm, ple_meta, ple_proj_w, ple_proj_norm, layers, ws);
        let mut embed_gguf: Option<DType> = None;
        {
            let ix = G4Index::new(&ctx, weights.as_ref(), codec);

            // CIMA_G4_NORM_SHIFT=<f32>: add a constant to EVERY LM RMSNorm
            // gamma at load. The rmsnorm kernel multiplies gammas directly;
            // gemma's HF convention is zero-centered (1+w), and converters
            // differ in whether they bake the +1 into the file. This knob
            // turns that convention question into a one-command experiment:
            // a gguf whose gammas were exported zero-centered runs correctly
            // with +1; one that was double-shifted runs correctly with -1.
            let norm_shift: f32 = std::env::var("CIMA_G4_NORM_SHIFT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            if norm_shift != 0.0 {
                crate::log::warn(&format!(
                    "gemma4: applying norm gamma shift {:+} (CIMA_G4_NORM_SHIFT)",
                    norm_shift
                ));
            }
            let norm_up = |name: &str, len: usize| -> Res<crate::cuda::DeviceBuf> {
                let buf = ix.upload(name, &[len])?;
                if norm_shift != 0.0 {
                    let mut b = vec![0u8; len * 2];
                    ctx.dtoh_at(&mut b, buf.ptr)?;
                    for c in b.chunks_exact_mut(2) {
                        let v =
                            crate::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]])) + norm_shift;
                        let h = crate::num::f32_to_f16(v);
                        c.copy_from_slice(&h.to_le_bytes());
                    }
                    ctx.htod(&buf, &b)?;
                }
                Ok(buf)
            };
            let lm = |s: &str| format!("language_model.{}", s);

            // ---- embeddings (tied head: table must land as F16 GEMM operand) ----
            let emb_meta = ix.meta(&lm("embed_tokens.weight"))?;
            if emb_meta.shape != vec![t.vocab, hs] {
                return Err(err!(
                    "weights",
                    "gemma4: embed_tokens has shape {:?}, expected [{}, {}]",
                    emb_meta.shape,
                    t.vocab,
                    hs
                ));
            }
            // The head is tied. gguf checkpoints stay PACKED: lookups run
            // the device gguf_gather and the head runs the dp4a GEMV — a
            // dense f16 copy would cost ~1 GiB of VRAM and ~1.3 GB/token of
            // head reads on a 256k vocab. Other storages route through the
            // codec so bf16 still yields a legal f16 GEMM operand.
            if crate::traits::is_gguf_block(emb_meta.dtype) {
                let host = weights.bytes(emb_meta)?;
                let buf = ctx.alloc(host.len())?;
                ctx.htod(&buf, host)?;
                embed = buf;
                embed_gguf = Some(emb_meta.dtype);
                ctx.ensure_q8_scratch(hs)?;
            } else {
                embed = ix.upload(&lm("embed_tokens.weight"), &[t.vocab, hs])?;
            }
            embed_bf16 = false;
            final_norm = norm_up(&lm("norm.weight"), hs)?;

            // ---- per-layer embeddings (token table stays on the host) ----
            if t.ple_dim > 0 {
                let pm = ix.meta(&lm("embed_tokens_per_layer.weight"))?;
                let expect = vec![t.ple_vocab, t.n_layers * t.ple_dim];
                if pm.shape != expect {
                    return Err(err!(
                        "weights",
                        "gemma4: embed_tokens_per_layer has shape {:?}, expected {:?}",
                        pm.shape,
                        expect
                    ));
                }
                if pm.dtype != DType::BF16
                    && pm.dtype != DType::F16
                    && !crate::traits::is_gguf_block(pm.dtype)
                {
                    return Err(err!(
                        "weights",
                        "gemma4: embed_tokens_per_layer stored as {} — host gather supports bf16/f16/gguf blocks",
                        pm.dtype.name()
                    ));
                }
                ple_meta = Some(pm.clone());
                ple_proj_w = Some(ix.upload_w(
                    &lm("per_layer_model_projection.weight"),
                    t.n_layers * t.ple_dim,
                    hs,
                )?);
                ple_proj_norm = Some(norm_up(&lm("per_layer_projection_norm.weight"), t.ple_dim)?);
            } else {
                ple_meta = None;
                ple_proj_w = None;
                ple_proj_norm = None;
            }

            // ---- decoder layers ----
            let mut ls = Vec::with_capacity(t.n_layers);
            let kv_sources = kv_share_sources(&t.layer_types, first_shared)?;
            // i indexes several parallel per-layer arrays and the output
            // kv_sources; an iterator over one would obscure the others.
            #[allow(clippy::needless_range_loop)]
            for i in 0..t.n_layers {
                let ty = t.layer_types[i];
                let d = if ty == LayerType::Sliding {
                    t.head_dim
                } else {
                    t.global_head_dim
                };
                let qd = t.n_heads * d;
                let kvd = t.n_kv_heads * d;
                let shared = i >= first_shared;
                let inter = if shared && t.double_wide_mlp {
                    t.inter * 2
                } else {
                    t.inter
                };
                let p = |s: &str| lm(&format!("layers.{}.{}", i, s));

                let kv_src = kv_sources[i];
                // Checkpoints legitimately ship k/v projections for shared
                // layers — the reference lists them in
                // `_keys_to_ignore_on_load_unexpected` and never loads them.
                // We mirror that: shared layers ignore those tensors. The
                // inverse (a computing layer *missing* k_proj) is real
                // corruption and still fails loudly.
                if !shared && !ix.exists(&p("self_attn.k_proj.weight")) {
                    return Err(err!(
                        "weights",
                        "gemma4: layer {} must compute K/V but the checkpoint has no k_proj — \
                         num_kv_shared_layers and the weights disagree",
                        i
                    ));
                }

                let scalar = match ix.roots.iter().find_map(|r| {
                    ix.weights
                        .tensors()
                        .get(&format!("{}{}", r, p("layer_scalar")))
                }) {
                    Some(m) => ix.scalar_f32(m)?,
                    // llama.cpp's reduced-graph export names it layer_output_scale.
                    None => match ix.roots.iter().find_map(|r| {
                        ix.weights.tensors().get(&format!(
                            "{}{}",
                            r,
                            p("layer_output_scale.weight")
                        ))
                    }) {
                        Some(m) => ix.scalar_f32(m)?,
                        None => 1.0,
                    },
                };

                ls.push(G4Layer {
                    input_norm: norm_up(&p("input_layernorm.weight"), hs)?,
                    post_attn_norm: norm_up(&p("post_attention_layernorm.weight"), hs)?,
                    pre_ffw_norm: norm_up(&p("pre_feedforward_layernorm.weight"), hs)?,
                    post_ffw_norm: norm_up(&p("post_feedforward_layernorm.weight"), hs)?,
                    wq: ix.upload_w(&p("self_attn.q_proj.weight"), qd, hs)?,
                    q_norm: norm_up(&p("self_attn.q_norm.weight"), d)?,
                    wk: if shared {
                        None
                    } else {
                        Some(ix.upload_w(&p("self_attn.k_proj.weight"), kvd, hs)?)
                    },
                    wv: if shared {
                        None
                    } else {
                        Some(ix.upload_w(&p("self_attn.v_proj.weight"), kvd, hs)?)
                    },
                    k_norm: if shared {
                        None
                    } else {
                        Some(norm_up(&p("self_attn.k_norm.weight"), d)?)
                    },
                    wo: ix.upload_w(&p("self_attn.o_proj.weight"), hs, qd)?,
                    w_gate: ix.upload_w(&p("mlp.gate_proj.weight"), inter, hs)?,
                    w_up: ix.upload_w(&p("mlp.up_proj.weight"), inter, hs)?,
                    w_down: ix.upload_w(&p("mlp.down_proj.weight"), hs, inter)?,
                    ple_gate: if t.ple_dim > 0 {
                        Some(ix.upload_w(&p("per_layer_input_gate.weight"), t.ple_dim, hs)?)
                    } else {
                        None
                    },
                    ple_proj: if t.ple_dim > 0 {
                        Some(ix.upload_w(&p("per_layer_projection.weight"), hs, t.ple_dim)?)
                    } else {
                        None
                    },
                    ple_norm: if t.ple_dim > 0 {
                        Some(norm_up(&p("post_per_layer_input_norm.weight"), hs)?)
                    } else {
                        None
                    },
                    scalar,
                    // rope_freqs frequency factors on full-attention layers.
                    // Settled empirically on the E4B gguf export: the trained
                    // recipe is FULL-width rotation (rope.dimension_count ==
                    // key_length in the metadata) WITH these factors — the
                    // llama.cpp graph. The earlier "factors wreck generation"
                    // finding was an artifact of applying 256-frequency
                    // factors over a 64-frequency partial rotation. The gguf
                    // loader sets `use_rope_factors`; CIMA_G4_ROPE_FACTORS=1
                    // remains as a manual override for experiments.
                    rope_factors: if ty != LayerType::Sliding
                        && (crate::models::gemma4::support::env_flag("CIMA_G4_ROPE_FACTORS")
                            .unwrap_or(t.use_rope_factors))
                        && ix.exists(&lm("rope_freqs.weight"))
                    {
                        let fm = ix.meta(&lm("rope_freqs.weight"))?;
                        let fb = ix.weights.bytes(fm)?;
                        let buf = ctx.alloc(fb.len())?;
                        ctx.htod(&buf, fb)?;
                        Some(buf)
                    } else {
                        None
                    },
                    kv: if shared {
                        None
                    } else {
                        Some((
                            ctx.alloc(t.n_kv_heads * t.max_seq * d * 2)?,
                            ctx.alloc(t.n_kv_heads * t.max_seq * d * 2)?,
                        ))
                    },
                    kv_src,
                    head_dim: d,
                    inter,
                    theta: if ty == LayerType::Sliding {
                        t.theta_sliding
                    } else {
                        t.theta_full
                    },
                    nfreqs: if ty == LayerType::Sliding {
                        t.head_dim / 2
                    } else {
                        t.full_nfreqs
                    },
                    window: if ty == LayerType::Sliding {
                        t.sliding_window
                    } else {
                        0
                    },
                });
            }
            layers = ls;

            // ---- workspace ----
            let dmax = t.head_dim.max(t.global_head_dim);
            let imax = if t.double_wide_mlp {
                t.inter * 2
            } else {
                t.inter
            };
            // Prefill dequant scratch. Sized by the SAME function as the
            // VRAM forecast (see prefill_scratch_bytes) — the largest weight
            // that dequantizes through it, GGUF included. The previous
            // NF4-only filter allocated 2 bytes for GGUF checkpoints and
            // every prefill GEMM wrote the full dequantized matrix out of
            // bounds: the GEMM itself stayed correct (it reads back exactly
            // what the dequant wrote), while the overflow corrupted
            // neighbouring device buffers — the multimodal
            // first-call-right / later-calls-wrong bug.
            let wsc_bytes = Self::prefill_scratch_bytes(weights.as_ref());
            ws = G4Ws {
                x: ctx.alloc(CHUNK * hs * 2)?,
                h: ctx.alloc(CHUNK * hs * 2)?,
                h2: ctx.alloc(CHUNK * hs * 2)?,
                q: ctx.alloc(CHUNK * t.n_heads * dmax * 2)?,
                k: ctx.alloc(CHUNK * t.n_kv_heads * dmax * 2)?,
                v: ctx.alloc(CHUNK * t.n_kv_heads * dmax * 2)?,
                att: ctx.alloc(CHUNK * t.n_heads * dmax * 2)?,
                gate: ctx.alloc(CHUNK * imax * 2)?,
                up: ctx.alloc(CHUNK * imax * 2)?,
                ple: ctx.alloc((CHUNK * t.n_layers * t.ple_dim * 2).max(2))?,
                ple_id: ctx.alloc((CHUNK * t.n_layers * t.ple_dim * 2).max(2))?,
                ple_g: ctx.alloc((CHUNK * t.ple_dim * 2).max(2))?,
                wsc: ctx.alloc(wsc_bytes.max(2))?,
                blkid: ctx.alloc(t.max_seq * 4)?,
                ids: ctx.alloc(CHUNK * 4)?,
                logits_h: ctx.alloc(t.vocab * 2)?,
                logits_f: ctx.alloc(t.vocab * 4)?,
                argmax_slot: ctx.alloc(8)?,
                pos_dev: ctx.alloc(4)?,
                att_part: ctx.alloc(
                    t.n_heads
                        * t.max_seq.div_ceil(crate::cuda::ATT_CSZ)
                        * (t.head_dim.max(t.global_head_dim) + 2)
                        * 4,
                )?,
                cand: ctx.alloc(crate::models::transformer::SAMPLE_TOPK * 8)?,
                hist_ring: ctx.alloc(64 * 4)?,
                hist_counts: ctx.alloc(t.vocab * 4)?,
            };
        }

        // The media-block table covers every absolute position but each
        // prefill uploads only rows 0..prompt_len — initialize the whole
        // buffer to -1 ("no block") once, so the never-rewritten tail can
        // never read as a valid block id (0 is a REAL id, so a zero memset
        // would be wrong; -1 is 0xFFFFFFFF in i32).
        {
            let none = vec![0xFFu8; t.max_seq * 4];
            ctx.htod(&ws.blkid, &none)?;
        }

        // Towers build only when the checkpoint actually carries them:
        // llama.cpp gguf exports are text-only (towers ship in a separate
        // mmproj file), while config.json still describes the full model.
        let has_tower = |prefix: &str| weights.tensors().keys().any(|k| k.contains(prefix));
        // Tower construction must NEVER take text generation down with it:
        // a partially-translated mmproj (name-mapping gaps for a new export
        // vintage) surfaces as a WARN + disabled capability, not a failed
        // load. Missing tensors are named in the warning so the mapping can
        // be extended against ground truth.
        let vision = match &cfg.vision {
            Some(vc) if has_tower("vision_tower.") => {
                let ix = G4Index::new(&ctx, weights.as_ref(), codec);
                match G4Vision::build(&ctx, vc, &ix, hs) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        crate::log::warn(&format!("gemma4: vision tower present but failed to build — vision disabled: {}", e));
                        dump_tower_names(weights.as_ref(), "vision_tower.");
                        None
                    }
                }
            }
            Some(_) => {
                crate::log::info("gemma4: config declares a vision tower but the checkpoint carries no vision tensors (text-only gguf) — vision disabled");
                None
            }
            None => None,
        };
        let audio = match &cfg.audio {
            Some(ac) if has_tower("audio_tower.") => {
                let ix = G4Index::new(&ctx, weights.as_ref(), codec);
                match G4Audio::build(&ctx, ac, &ix, hs) {
                    Ok(a) => Some(a),
                    Err(e) => {
                        crate::log::warn(&format!(
                            "gemma4: audio tower present but failed to build — audio disabled: {}",
                            e
                        ));
                        dump_tower_names(weights.as_ref(), "audio_tower.");
                        None
                    }
                }
            }
            Some(_) => {
                crate::log::info("gemma4: config declares an audio tower but the checkpoint carries no audio tensors (text-only gguf) — audio disabled");
                None
            }
            None => None,
        };

        // Kick async page-cache prefetch of the host-resident PLE table:
        // decode gathers one row per token and a cold pageable mmap turns
        // that into a synchronous disk fault per step.
        if let Some(pm) = &ple_meta {
            weights.prefetch(pm);
        }
        ctx.sync()?;
        let vram = ctx.tracked_bytes();
        // Quantization census: which subsystems did this checkpoint pack to
        // 4-bit? (unsloth "dynamic" repos mix freely; a 4-bit vision tower
        // costs real visual fidelity and this line makes that visible.)
        {
            let mut counts = [[0usize; 2]; 3]; // [vision, audio, text] x [nf4, 16-bit]
            for (name, meta) in weights.tensors() {
                if meta.shape.len() < 2 || bnb::is_aux(name) {
                    continue;
                }
                let sub = if name.contains("vision_tower") || name.contains("embed_vision") {
                    0
                } else if name.contains("audio_tower") || name.contains("embed_audio") {
                    1
                } else {
                    2
                };
                let q = bnb::state_name(weights.tensors(), name).is_some() as usize;
                counts[sub][1 - q] += 1; // index 0 = nf4, 1 = 16-bit
            }
            log::info(&format!(
                "quantization census: vision {} nf4 / {} 16-bit, audio {} nf4 / {} 16-bit, text {} nf4 / {} 16-bit weights",
                counts[0][0], counts[0][1], counts[1][0], counts[1][1], counts[2][0], counts[2][1]
            ));
        }
        log::info(&format!(
            "gemma4 LM image-span attention: {} (text_config.use_bidirectional_attention)",
            if cfg.text.bidir_vision {
                "bidirectional blocks"
            } else {
                "causal"
            }
        ));
        log::info(&format!(
            "gemma4 graph resident: {} layers ({} computing KV, {} shared), vision={}, audio={}, {} VRAM, built in {:?}",
            t.n_layers,
            first_shared,
            t.n_kv_shared,
            vision.is_some(),
            audio.is_some(),
            crate::cuda::fmt_bytes(vram),
            t0.elapsed()
        ));
        // Model card: everything the engine inferred from config + tensors —
        // the reference sheet for porting the next family (CIMA_LOG=debug).
        if log::debug_on() {
            let mut types = String::new();
            for l in &layers {
                types.push(if l.window > 0 { 'S' } else { 'F' });
            }
            let share: Vec<String> = layers
                .iter()
                .enumerate()
                .filter(|(i, l)| l.kv_src != *i)
                .map(|(i, l)| format!("{}→{}", i, l.kv_src))
                .collect();
            let card = format!(
                "model card [gemma4]\n  text: hidden={} layers={} heads={} kv_heads={} vocab={} max_seq={}\n  layer types ({}=sliding {}=full): {}\n  head_dim: sliding={} full={}  attn scale=1.0 (no 1/sqrt(d))\n  rope: sliding theta={} full theta={} nfreqs={} (proportional partial-rotary)\n  sliding window={}  kv share (layer→src): {}\n  PLE: dim={} table+context-projection, identity uses PAD at media rows, projection consumes spliced soft tokens, combine ×1/√2\n  norms: sandwich (pre/post attn + pre/post ffw), q/k per-head RMSNorm, weightless v-norm\n  softcap: final={}  specials: pad={} eos={:?} img={} aud={}",
                t.hidden, t.n_layers, t.n_heads, t.n_kv_heads, t.vocab, t.max_seq,
                'S', 'F', types,
                t.head_dim, t.global_head_dim,
                t.theta_sliding, t.theta_full, t.full_nfreqs,
                t.sliding_window,
                if share.is_empty() { "none".into() } else { share.join(" ") },
                t.ple_dim,
                t.softcap,
                cfg.pad_token_id, t.eos, cfg.image_token_id, cfg.audio_token_id,
            );
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                eprintln!("{}", card);
            }
            log::debug(&card);
        }
        // CIMA_G4_STATS=1: print the statistics that discriminate weight-
        // convention bugs, straight off the freshly uploaded tensors.
        // The rmsnorm kernel multiplies gammas DIRECTLY (the HF-validated
        // convention); a checkpoint whose gammas cluster near 0 is
        // zero-centered (needs +1), and one near 2 was double-shifted by
        // its converter. Healthy direct gammas sit roughly in [0.3, 3].
        if std::env::var("CIMA_G4_STATS").is_ok() {
            let stat =
                |ctx: &CudaCtx, buf: &crate::cuda::DeviceBuf, n: usize, name: &str| -> Res<()> {
                    let mut b = vec![0u8; n * 2];
                    ctx.dtoh_at(&mut b, buf.ptr)?;
                    let vals: Vec<f32> = b
                        .chunks_exact(2)
                        .map(|c| crate::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
                    for &v in &vals {
                        mn = mn.min(v);
                        mx = mx.max(v);
                        sum += v as f64;
                    }
                    crate::log::info(&format!(
                        "G4_STATS {:<28} n={:<6} mean={:+.4} min={:+.4} max={:+.4}",
                        name,
                        n,
                        sum / n as f64,
                        mn,
                        mx
                    ));
                    Ok(())
                };
            for li in [0usize, t.n_layers / 2, t.n_layers - 1] {
                let l = &layers[li];
                stat(&ctx, &l.input_norm, hs, &format!("L{}.input_layernorm", li))?;
                stat(
                    &ctx,
                    &l.post_attn_norm,
                    hs,
                    &format!("L{}.post_attention_norm", li),
                )?;
                stat(
                    &ctx,
                    &l.pre_ffw_norm,
                    hs,
                    &format!("L{}.pre_feedforward_norm", li),
                )?;
                stat(
                    &ctx,
                    &l.post_ffw_norm,
                    hs,
                    &format!("L{}.post_feedforward_norm", li),
                )?;
                let d = if t.layer_types[li] == LayerType::Sliding {
                    t.head_dim
                } else {
                    t.global_head_dim
                };
                stat(&ctx, &l.q_norm, d, &format!("L{}.q_norm", li))?;
                if let Some(pn) = &l.ple_norm {
                    stat(&ctx, pn, hs, &format!("L{}.post_per_layer_norm", li))?;
                }
                crate::log::info(&format!(
                    "G4_STATS L{}.layer_scalar             = {:+.6}",
                    li, l.scalar
                ));
            }
            stat(&ctx, &final_norm, hs, "final_norm")?;
            if let Some(m) = &ple_meta {
                crate::log::info(&format!(
                    "G4_STATS ple_table dtype = {} shape = {:?}",
                    m.dtype.name(),
                    m.shape
                ));
            }
        }
        Ok(Gemma4 {
            decode_graph: None,
            sample_graph: None,
            weight_bytes,
            ctx,
            cfg,
            weights,
            ple_meta,
            embed,
            embed_gguf,
            dbg_tokens: Vec::new(),
            embed_bf16,
            final_norm,
            ple_proj_w,
            ple_proj_norm,
            layers,
            ws,
            blk_host: Vec::new(),
            perf_ple_ms: 0.0,
            perf_logits_ms: 0.0,
            pos: 0,
            vram,
            vision,
            audio,
        })
    }

    pub fn modality(&self) -> Modality {
        // The richest input modality wins for the `ps` column; any-to-any
        // (text+image+audio) reports as VisionText since the engine's
        // Modality taxonomy predates tri-modal models.
        match (&self.vision, &self.audio) {
            (Some(_), _) => Modality::VisionText,
            (None, Some(_)) => Modality::AudioText,
            (None, None) => Modality::TextToText,
        }
    }

    pub fn max_seq(&self) -> usize {
        self.cfg.text.max_seq
    }

    pub fn vram_bytes(&self) -> usize {
        self.vram
    }

    pub fn extra_eos(&self) -> &[u32] {
        &self.cfg.text.eos
    }

    pub fn ctx(&self) -> &CudaCtx {
        &self.ctx
    }

    pub fn config(&self) -> &G4Config {
        &self.cfg
    }

    fn check_capacity(&self, want: usize) -> Res<()> {
        if want > self.cfg.text.max_seq {
            return Err(err!(
                "context",
                "sequence of {} tokens exceeds the KV capacity of {} — truncate the prompt or lower max_tokens",
                want, self.cfg.text.max_seq
            ));
        }
        Ok(())
    }

    /// Gather + scale token embeddings into `ws.x` (`x = embed[ids] * sqrt(hidden)`).
    fn embed_tokens(&mut self, tokens: &[u32]) -> Res<()> {
        if std::env::var("CIMA_G4_NANHUNT").is_ok() {
            self.dbg_tokens = tokens.to_vec();
        }
        debug_assert!(tokens.len() <= CHUNK);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
        self.ctx.htod(&self.ws.ids, bytes)?;
        match self.embed_gguf {
            Some(fmt) => self.ctx.gguf_gather(
                fmt,
                self.embed.ptr,
                self.ws.ids.ptr,
                self.ws.x.ptr,
                tokens.len(),
                self.cfg.text.hidden,
            )?,
            None => self.ctx.gather(
                self.embed.ptr,
                self.embed_bf16,
                self.ws.ids.ptr,
                self.ws.x.ptr,
                tokens.len(),
                self.cfg.text.hidden,
            )?,
        }
        self.ctx.scalemul(
            self.ws.x.ptr,
            (self.cfg.text.hidden as f32).sqrt(),
            tokens.len() * self.cfg.text.hidden,
        )
    }

    /// CPU gather of PLE token-identity rows (× sqrt(ple_dim)) straight out of
    /// the pinned mmap, uploaded into `ws.ple_id`. The multi-GiB table never
    /// touches VRAM.
    fn gather_ple_identity(&mut self, tokens: &[u32]) -> Res<()> {
        let t0 = std::time::Instant::now();
        let r = self.gather_ple_identity_inner(tokens);
        self.perf_ple_ms += t0.elapsed().as_secs_f64() * 1e3;
        r
    }

    fn gather_ple_identity_inner(&self, tokens: &[u32]) -> Res<()> {
        let t = &self.cfg.text;
        let meta = self
            .ple_meta
            .as_ref()
            .expect("ple_meta present when ple_dim > 0");
        let table = self.weights.bytes(meta)?;
        let row_elems = t.n_layers * t.ple_dim;
        let scale = (t.ple_dim as f32).sqrt();
        let bf16 = meta.dtype == DType::BF16;
        let gguf_blk = crate::traits::is_gguf_block(meta.dtype);
        let row_bytes_gguf = if gguf_blk {
            crate::formats::gguf::storage_bytes(meta.dtype, row_elems)
        } else {
            0
        };
        let mut rowbuf = vec![0u16; if gguf_blk { row_elems } else { 0 }];
        let mut host: Vec<u16> = Vec::with_capacity(tokens.len() * row_elems);
        for &id in tokens {
            // Reference semantics: multimodal placeholder positions look up
            // the PAD token's PLE row, not the placeholder id's.
            let id = if id == self.cfg.image_token_id || id == self.cfg.audio_token_id {
                self.cfg.pad_token_id as usize
            } else {
                id as usize
            };
            if id >= t.ple_vocab {
                return Err(err!(
                    "generate",
                    "gemma4: token id {} outside per-layer vocab {}",
                    id,
                    t.ple_vocab
                ));
            }
            if gguf_blk {
                let off = id * row_bytes_gguf;
                crate::quant::gguf::dequant_host(
                    meta.dtype,
                    &table[off..off + row_bytes_gguf],
                    row_elems,
                    &mut rowbuf,
                )?;
                for &raw in &rowbuf {
                    host.push(f32_to_f16(f16_to_f32(raw) * scale));
                }
            } else {
                let off = id * row_elems * 2;
                let row = &table[off..off + row_elems * 2];
                for c in row.chunks_exact(2) {
                    let raw = u16::from_le_bytes([c[0], c[1]]);
                    let v = if bf16 {
                        bf16_to_f32(raw)
                    } else {
                        f16_to_f32(raw)
                    };
                    host.push(f32_to_f16(v * scale));
                }
            }
        }
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
        self.ctx.htod(&self.ws.ple_id, bytes)
    }

    /// Build the combined PLE slab for `rows` tokens already embedded (and
    /// media-spliced) in `ws.x`:
    /// `ple = (RMSNorm(x @ W_proj^T · hidden^-0.5) + identity·sqrt(ple)) · 2^-0.5`.
    /// PLE context projection + identity combine, reference semantics:
    /// the *projection* is context-aware — it consumes the residual input
    /// `ws.x` **after** the media splice, so image positions project their
    /// actual soft tokens (`project_per_layer_inputs(inputs_embeds, ...)`
    /// runs on the post-`masked_scatter` embeddings). Only the *identity*
    /// channel substitutes PAD at placeholder positions (the reference's
    /// `llm_input_ids = where(multimodal, pad, ids)` feeds
    /// `embed_tokens_per_layer`); that mapping lives in
    /// `gather_ple_identity`. Substituting PAD in the projection too starves
    /// every layer of its visual control signal and erodes the image
    /// positions layer by layer.
    fn compute_ple(&self, rows: usize) -> Res<()> {
        let t = &self.cfg.text;
        if t.ple_dim == 0 {
            return Ok(());
        }
        let w = self.ple_proj_w.as_ref().unwrap();
        let norm = self.ple_proj_norm.as_ref().unwrap();
        let slab = t.n_layers * t.ple_dim;
        w.gemm(
            &self.ctx,
            self.ws.x.ptr,
            self.ws.ple.ptr,
            rows,
            slab,
            t.hidden,
            self.ws.wsc.ptr,
        )?;
        self.ctx
            .scalemul(self.ws.ple.ptr, 1.0 / (t.hidden as f32).sqrt(), rows * slab)?;
        self.ctx.rmsnorm(
            self.ws.ple.ptr,
            norm.ptr,
            self.ws.ple.ptr,
            rows * t.n_layers,
            t.ple_dim,
            t.rms_eps,
        )?;
        self.ctx
            .add(self.ws.ple.ptr, self.ws.ple_id.ptr, rows * slab)?;
        self.ctx.scalemul(
            self.ws.ple.ptr,
            std::f32::consts::FRAC_1_SQRT_2,
            rows * slab,
        )
    }
}

impl Gemma4 {
    /// Run `rows` hidden states (resident in `ws.x`, PLE slab in `ws.ple`)
    /// through every decoder layer. Implements the reference layer exactly:
    ///
    /// ```text
    /// x += post_attn_norm(attn(input_norm(x)))
    /// x += post_ffw_norm(geglu_mlp(pre_ffw_norm(x)))
    /// x += post_ple_norm(ple_proj(gelu(ple_gate(x)) * ple_i))   [if PLE]
    /// x *= layer_scalar
    /// ```
    fn forward_chunk(&self, rows: usize, pos0: usize, blkid: bool) -> Res<()> {
        self.forward_chunk_traced(rows, pos0, blkid, 0, None)
    }

    /// `trace`: when `Some`, snapshot the LAST row of the residual after
    /// every layer (f32) — the per-layer divergence probe for the LM A/B.
    fn forward_chunk_traced(
        &self,
        rows: usize,
        pos0: usize,
        blkid: bool,
        pos_dev: u64,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Res<()> {
        let t = &self.cfg.text;
        let ctx = &self.ctx;
        let hs = t.hidden;
        let kvh = t.n_kv_heads;
        let nh = t.n_heads;
        // CIMA_TRACE_LAYER=N (with CIMA_DUMP_LM): sub-layer snapshots of the
        // last row at layer N — the op-level bisection probe.
        let sub_layer: Option<usize> = if trace.is_some() {
            std::env::var("CIMA_TRACE_LAYER")
                .ok()
                .and_then(|v| v.parse().ok())
        } else {
            None
        };
        // CIMA_TRACE_POS: which row to snapshot (default: last). Image-span
        // positions and text positions can diverge independently.
        let trace_pos: usize = std::env::var("CIMA_TRACE_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(rows.saturating_sub(1))
            .min(rows.saturating_sub(1));
        let snap_last = |ptr: u64, width: usize| -> Res<Vec<f32>> {
            ctx.sync()?;
            let mut b = vec![0u8; width * 2];
            ctx.dtoh_at(&mut b, ptr + (trace_pos * width * 2) as u64)?;
            Ok(b.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        };
        let mut subs: Vec<(String, Vec<f32>)> = Vec::new();

        // Divergence bisection: CIMA_G4_FP=<layer> prints sum-of-squares of
        // the whole q buffer (post-rope) and the raw attention output for
        // that layer, every prefill. Two identical requests must match; the
        // first metric that drifts says whether it's the projection/rope
        // (q differs) or the KV read (q matches, attn differs).
        let fp_layer: Option<usize> = std::env::var("CIMA_G4_FP")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(if std::env::var("CIMA_G4_DEBUG").is_ok() {
                Some(5)
            } else {
                None
            });
        let fp_sumsq = |ptr: u64, n_vals: usize| -> Res<f64> {
            ctx.sync()?;
            let mut b = vec![0u8; n_vals * 2];
            ctx.dtoh_at(&mut b, ptr)?;
            Ok(b.chunks_exact(2)
                .map(|c| {
                    let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64;
                    v * v
                })
                .sum())
        };

        if sub_layer.is_some() && t.ple_dim > 0 {
            let slab = t.n_layers * t.ple_dim;
            ctx.sync()?;
            let mut b = vec![0u8; slab * 2];
            ctx.dtoh_at(&mut b, self.ws.ple.ptr + (trace_pos * slab * 2) as u64)?;
            subs.push((
                "ple_row".into(),
                b.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            ));
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let d = layer.head_dim;
            let (qd, kvd) = (nh * d, kvh * d);

            // ---- attention ----
            ctx.rmsnorm(
                self.ws.x.ptr,
                layer.input_norm.ptr,
                self.ws.h.ptr,
                rows,
                hs,
                t.rms_eps,
            )?;
            if sub_layer == Some(i) {
                subs.push(("norm".into(), snap_last(self.ws.h.ptr, hs)?));
            }
            layer.wq.gemm(
                ctx,
                self.ws.h.ptr,
                self.ws.q.ptr,
                rows,
                qd,
                hs,
                self.ws.wsc.ptr,
            )?;
            // QK-RMSNorm operates per head: [rows*heads] rows of `d`.
            ctx.rmsnorm(
                self.ws.q.ptr,
                layer.q_norm.ptr,
                self.ws.q.ptr,
                rows * nh,
                d,
                t.rms_eps,
            )?;
            if sub_layer == Some(i) {
                subs.push(("q_postnorm".into(), snap_last(self.ws.q.ptr, qd)?));
            }
            ctx.rope(
                self.ws.q.ptr,
                rows,
                nh,
                d,
                pos0,
                layer.theta,
                layer.nfreqs,
                pos_dev,
                layer.rope_factors.as_ref().map(|b| b.ptr).unwrap_or(0),
            )?;
            if sub_layer == Some(i) {
                subs.push(("q_postrope".into(), snap_last(self.ws.q.ptr, qd)?));
            }
            if fp_layer == Some(i) {
                let q_ss = fp_sumsq(self.ws.q.ptr, rows * qd)?;
                eprintln!("g4 fp L{}: q_postrope_sumsq={:.4}", i, q_ss);
            }

            if let (Some(wk), Some(wv), Some(k_norm), Some((kc, vc))) =
                (&layer.wk, &layer.wv, &layer.k_norm, &layer.kv)
            {
                wk.gemm(
                    ctx,
                    self.ws.h.ptr,
                    self.ws.k.ptr,
                    rows,
                    kvd,
                    hs,
                    self.ws.wsc.ptr,
                )?;
                ctx.rmsnorm(
                    self.ws.k.ptr,
                    k_norm.ptr,
                    self.ws.k.ptr,
                    rows * kvh,
                    d,
                    t.rms_eps,
                )?;
                ctx.rope(
                    self.ws.k.ptr,
                    rows,
                    kvh,
                    d,
                    pos0,
                    layer.theta,
                    layer.nfreqs,
                    pos_dev,
                    layer.rope_factors.as_ref().map(|b| b.ptr).unwrap_or(0),
                )?;
                wv.gemm(
                    ctx,
                    self.ws.h.ptr,
                    self.ws.v.ptr,
                    rows,
                    kvd,
                    hs,
                    self.ws.wsc.ptr,
                )?;
                // V-RMSNorm is scale-less (`with_scale=False`): w == NULL.
                ctx.rmsnorm(self.ws.v.ptr, 0, self.ws.v.ptr, rows * kvh, d, t.rms_eps)?;
                ctx.kv_append(
                    self.ws.k.ptr,
                    self.ws.v.ptr,
                    kc.ptr,
                    vc.ptr,
                    rows,
                    kvh,
                    d,
                    pos0,
                    t.max_seq,
                    pos_dev,
                )?;
                if sub_layer == Some(i) {
                    // cache rows (head 0) at three probe positions, post-rope/post-norm
                    ctx.sync()?;
                    for (tag, p) in [
                        ("p0", 0usize),
                        ("mid", (pos0 + rows) / 2),
                        ("last", pos0 + rows - 1),
                    ] {
                        for (kind, buf) in [("k", kc.ptr), ("v", vc.ptr)] {
                            let mut b = vec![0u8; d * 2];
                            ctx.dtoh_at(&mut b, buf + (p * d * 2) as u64)?;
                            subs.push((
                                format!("{}cache_{}", kind, tag),
                                b.chunks_exact(2)
                                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                                    .collect(),
                            ));
                        }
                    }
                }
            }

            let (kc, vc) = self.layers[layer.kv_src]
                .kv
                .as_ref()
                .map(|(k, v)| (k.ptr, v.ptr))
                .ok_or_else(|| {
                    err!(
                        "generate",
                        "gemma4: layer {} kv source {} has no cache (internal sharing-map bug)",
                        i,
                        layer.kv_src
                    )
                })?;

            let blk = if blkid { self.ws.blkid.ptr } else { 0 };
            if rows == 1 {
                if d % 32 == 0 && d <= 256 {
                    let csz = crate::cuda::ATT_CSZ;
                    let nc = t.max_seq.div_ceil(csz);
                    ctx.attn_decode_split(
                        self.ws.q.ptr,
                        kc,
                        vc,
                        self.ws.att_part.ptr,
                        nh,
                        kvh,
                        d,
                        pos0 + 1,
                        t.max_seq,
                        csz,
                        nc,
                        1.0,
                        layer.window,
                        pos_dev,
                    )?;
                    ctx.attn_reduce(
                        self.ws.att_part.ptr,
                        self.ws.att.ptr,
                        nh,
                        d,
                        csz,
                        nc,
                        pos0 + 1,
                        layer.window,
                        pos_dev,
                    )?;
                } else {
                    ctx.attn_decode(
                        self.ws.q.ptr,
                        kc,
                        vc,
                        self.ws.att.ptr,
                        nh,
                        kvh,
                        d,
                        pos0 + 1,
                        t.max_seq,
                        1.0,
                        layer.window,
                        pos_dev,
                    )?;
                }
            } else {
                ctx.attn_prefill(
                    self.ws.q.ptr,
                    kc,
                    vc,
                    self.ws.att.ptr,
                    rows,
                    nh,
                    kvh,
                    d,
                    pos0,
                    t.max_seq,
                    true,
                    1.0,
                    layer.window,
                    blk,
                )?;
            }
            if sub_layer == Some(i) {
                subs.push(("attn_raw".into(), snap_last(self.ws.att.ptr, qd)?));
            }
            if fp_layer == Some(i) {
                let a_ss = fp_sumsq(self.ws.att.ptr, rows * qd)?;
                eprintln!("g4 fp L{}: attn_raw_sumsq={:.4}", i, a_ss);
            }
            layer.wo.gemm(
                ctx,
                self.ws.att.ptr,
                self.ws.h2.ptr,
                rows,
                hs,
                qd,
                self.ws.wsc.ptr,
            )?;
            if sub_layer == Some(i) {
                subs.push(("attn_o".into(), snap_last(self.ws.h2.ptr, hs)?));
            }
            ctx.rmsnorm(
                self.ws.h2.ptr,
                layer.post_attn_norm.ptr,
                self.ws.h2.ptr,
                rows,
                hs,
                t.rms_eps,
            )?;
            ctx.add(self.ws.x.ptr, self.ws.h2.ptr, rows * hs)?;
            self.nan_hunt("attn", i, rows)?;

            // ---- GeGLU MLP (double-wide on shared layers) ----
            ctx.rmsnorm(
                self.ws.x.ptr,
                layer.pre_ffw_norm.ptr,
                self.ws.h.ptr,
                rows,
                hs,
                t.rms_eps,
            )?;
            layer.w_gate.gemm(
                ctx,
                self.ws.h.ptr,
                self.ws.gate.ptr,
                rows,
                layer.inter,
                hs,
                self.ws.wsc.ptr,
            )?;
            layer.w_up.gemm(
                ctx,
                self.ws.h.ptr,
                self.ws.up.ptr,
                rows,
                layer.inter,
                hs,
                self.ws.wsc.ptr,
            )?;
            ctx.geglu(self.ws.gate.ptr, self.ws.up.ptr, rows * layer.inter)?;
            layer.w_down.gemm(
                ctx,
                self.ws.gate.ptr,
                self.ws.h2.ptr,
                rows,
                hs,
                layer.inter,
                self.ws.wsc.ptr,
            )?;
            ctx.rmsnorm(
                self.ws.h2.ptr,
                layer.post_ffw_norm.ptr,
                self.ws.h2.ptr,
                rows,
                hs,
                t.rms_eps,
            )?;
            ctx.add(self.ws.x.ptr, self.ws.h2.ptr, rows * hs)?;
            self.nan_hunt("ffn", i, rows)?;
            if sub_layer == Some(i) {
                subs.push(("mlp_res".into(), snap_last(self.ws.x.ptr, hs)?));
            }

            // ---- per-layer-embedding residual ----
            let ple_off = std::env::var("CIMA_G4_NO_PLE").is_ok();
            if let (false, Some(g), Some(p), Some(n)) =
                (ple_off, &layer.ple_gate, &layer.ple_proj, &layer.ple_norm)
            {
                g.gemm(
                    ctx,
                    self.ws.x.ptr,
                    self.ws.ple_g.ptr,
                    rows,
                    t.ple_dim,
                    hs,
                    self.ws.wsc.ptr,
                )?;
                ctx.gelu(self.ws.ple_g.ptr, rows * t.ple_dim)?;
                ctx.mul_strided(
                    self.ws.ple_g.ptr,
                    self.ws.ple.ptr,
                    rows,
                    t.ple_dim,
                    t.n_layers * t.ple_dim,
                    i * t.ple_dim,
                )?;
                p.gemm(
                    ctx,
                    self.ws.ple_g.ptr,
                    self.ws.h2.ptr,
                    rows,
                    hs,
                    t.ple_dim,
                    self.ws.wsc.ptr,
                )?;
                ctx.rmsnorm(self.ws.h2.ptr, n.ptr, self.ws.h2.ptr, rows, hs, t.rms_eps)?;
                ctx.add(self.ws.x.ptr, self.ws.h2.ptr, rows * hs)?;
                self.nan_hunt("ple", i, rows)?;
            }
            if sub_layer == Some(i) {
                subs.push(("ple_res".into(), snap_last(self.ws.x.ptr, hs)?));
            }

            if (layer.scalar - 1.0).abs() > 1e-9 {
                ctx.scalemul(self.ws.x.ptr, layer.scalar, rows * hs)?;
            }

            if let Some(tr) = trace.as_deref_mut() {
                ctx.sync()?;
                let mut b = vec![0u8; hs * 2];
                ctx.dtoh_at(&mut b, self.ws.x.ptr + (trace_pos * hs * 2) as u64)?;
                tr.push(
                    b.chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                );
            }
        }
        if !subs.is_empty() {
            if let Ok(base) = std::env::var("CIMA_DUMP_LM") {
                let mut out = Vec::new();
                out.extend_from_slice(&(subs.len() as u32).to_le_bytes());
                for (name, row) in &subs {
                    let nb = name.as_bytes();
                    out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
                    out.extend_from_slice(nb);
                    out.extend_from_slice(&(row.len() as u32).to_le_bytes());
                    for v in row {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                let path = format!("{}.sub", base);
                std::fs::write(&path, out)
                    .map_err(|e| err!("debug", "cannot write '{}': {}", path, e))?;
                log::info(&format!(
                    "sub-layer snapshots dumped to {} ({} stages)",
                    path,
                    subs.len()
                ));
            }
        }
        Ok(())
    }

    /// Final-norm row `row` of `ws.x`, project through the tied head, apply
    /// the logit soft-cap on the host: `l = cap * tanh(l / cap)`.
    fn project_logits(&mut self, row: usize) -> Res<Vec<f32>> {
        let t0 = std::time::Instant::now();
        let r = self.project_logits_inner(row);
        self.perf_logits_ms += t0.elapsed().as_secs_f64() * 1e3;
        r
    }

    fn project_logits_inner(&self, row: usize) -> Res<Vec<f32>> {
        let t = &self.cfg.text;
        let x_row = self.ws.x.ptr + (row * t.hidden * 2) as u64;
        self.ctx.rmsnorm(
            x_row,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            t.hidden,
            t.rms_eps,
        )?;
        match self.embed_gguf {
            Some(fmt) => self.ctx.gguf_gemv(
                fmt,
                self.ws.h.ptr,
                self.embed.ptr,
                0,
                self.ws.logits_h.ptr,
                t.vocab,
                t.hidden,
                0,
            )?,
            None => self.ctx.gemm_f16(
                self.ws.h.ptr,
                self.embed.ptr,
                self.ws.logits_h.ptr,
                1,
                t.vocab,
                t.hidden,
            )?,
        }
        self.ctx
            .h2f(self.ws.logits_h.ptr, self.ws.logits_f.ptr, t.vocab)?;
        let mut out = vec![0f32; t.vocab];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, t.vocab * 4) };
        self.ctx.dtoh(bytes, &self.ws.logits_f)?;
        if t.softcap > 0.0 {
            for l in &mut out {
                *l = t.softcap * (*l / t.softcap).tanh();
            }
        }
        Ok(out)
    }

    /// Greedy fast path: cap+argmax on device, 8-byte copy back — the 1 MB
    /// logits row never crosses PCIe (which dominated short generations:
    /// METRIC `logits=` was ~36-45 ms/token of host+sync per step on WSL2).
    fn project_argmax(&mut self, row: usize) -> Res<u32> {
        let t0 = std::time::Instant::now();
        let t = &self.cfg.text;
        let x_row = self.ws.x.ptr + (row * t.hidden * 2) as u64;
        self.ctx.rmsnorm(
            x_row,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            t.hidden,
            t.rms_eps,
        )?;
        match self.embed_gguf {
            Some(fmt) => self.ctx.gguf_gemv(
                fmt,
                self.ws.h.ptr,
                self.embed.ptr,
                0,
                self.ws.logits_h.ptr,
                t.vocab,
                t.hidden,
                0,
            )?,
            None => self.ctx.gemm_f16(
                self.ws.h.ptr,
                self.embed.ptr,
                self.ws.logits_h.ptr,
                1,
                t.vocab,
                t.hidden,
            )?,
        }
        let tok = self.ctx.argmax_softcap(
            self.ws.logits_h.ptr,
            &self.ws.argmax_slot,
            t.vocab,
            t.softcap,
        )?;
        self.perf_logits_ms += t0.elapsed().as_secs_f64() * 1e3;
        Ok(tok)
    }

    pub fn prefill_argmax(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<u32> {
        self.prefill_inner(prompt, pos0)?;
        self.project_argmax((prompt.tokens.len() + pos0 - 1) % CHUNK.max(1))
    }

    pub fn argmax_slot(&self) -> &crate::cuda::DeviceBuf {
        &self.ws.argmax_slot
    }

    pub fn decode_step_argmax(&mut self, token: u32, pos: usize) -> Res<u32> {
        if self.decode_graph.is_some() {
            return self.decode_step_graph(token, pos);
        }
        self.decode_step_inner(token, pos)?;
        self.project_argmax(0)
    }

    fn prefill_inner(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<()> {
        let n = prompt.tokens.len();
        if n == 0 {
            return Err(err!("generate", "empty prompt after tokenization"));
        }
        self.check_capacity(pos0 + n)?;

        // Maintain the absolute-position media-block id mirror and upload it
        // once per prefill (image spans attend bidirectionally; everything
        // else is -1).
        self.blk_host.truncate(pos0);
        if prompt.block_ids.is_empty() {
            self.blk_host.extend(std::iter::repeat_n(-1, n));
        } else {
            debug_assert_eq!(prompt.block_ids.len(), n);
            self.blk_host.extend_from_slice(&prompt.block_ids);
        }
        let has_blocks = self.blk_host.iter().any(|&b| b >= 0);
        if has_blocks {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.blk_host.as_ptr() as *const u8,
                    self.blk_host.len() * 4,
                )
            };
            self.ctx.htod(&self.ws.blkid, bytes)?;
        }

        let hs = self.cfg.text.hidden;
        let mut done = 0;
        while done < n {
            let mut take = (n - done).min(CHUNK);
            // Never split a media block across chunks: bidirectional block
            // attention scans forward keys, which must already be in the KV
            // cache when the block's queries run. If the cut lands inside a
            // block, retreat it to the block start (blocks are ≤ 280 tokens,
            // far below CHUNK, so a non-empty chunk always remains).
            if take < n - done {
                let cut = pos0 + done + take;
                if cut < self.blk_host.len()
                    && self.blk_host[cut] >= 0
                    && self.blk_host[cut] == self.blk_host[cut - 1]
                {
                    let blk = self.blk_host[cut];
                    let mut back = take;
                    while back > 0 && self.blk_host[pos0 + done + back - 1] == blk {
                        back -= 1;
                    }
                    if back > 0 {
                        take = back;
                    } // else: block ≥ CHUNK — leave the split (better than an empty chunk)
                }
            }
            let chunk_tokens = &prompt.tokens[done..done + take];
            self.embed_tokens(chunk_tokens)?;
            // Splice media rows over their placeholder positions (the
            // embedder already projected them into LM space; they replace the
            // scaled token embeddings, exactly like masked_scatter).
            for (at, buf, rows) in &prompt.media_embeds {
                let (start, end) = (*at, at + rows);
                let (c0, c1) = (done, done + take);
                if end > c0 && start < c1 {
                    let s = start.max(c0);
                    let e = end.min(c1);
                    let src = buf.ptr + ((s - start) * hs * 2) as u64;
                    let dst = self.ws.x.ptr + ((s - c0) * hs * 2) as u64;
                    self.ctx.dtod(dst, src, (e - s) * hs * 2)?;
                    // Per-request fingerprint of the actual media embedding
                    // consumed: sum of squares over the first spliced row.
                    // Identical images MUST produce an identical fingerprint
                    // every request; a drift here means the media buffer (or
                    // the tower that filled it) is carrying state between
                    // requests — the multimodal-degrades-after-first-call bug.
                    if std::env::var("CIMA_G4_DEBUG").is_ok() && s == start {
                        self.ctx.sync()?;
                        let mut b = vec![0u8; hs * 2];
                        self.ctx.dtoh_at(&mut b, src)?;
                        let ss: f64 = b
                            .chunks_exact(2)
                            .map(|c| {
                                let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64;
                                v * v
                            })
                            .sum();
                        eprintln!(
                            "g4 media splice: at={} rows={} row0_sumsq={:.4}",
                            at, rows, ss
                        );
                    }
                }
            }
            // CIMA_DUMP_LM: snapshot the post-splice embeddings of the first
            // chunk (the analogue of the reference's scattered inputs_embeds)
            // for the LM-side A/B.
            if done == 0 {
                if let Ok(base) = std::env::var("CIMA_DUMP_LM") {
                    dump_f16_matrix(
                        &self.ctx,
                        self.ws.x.ptr,
                        take,
                        hs,
                        &format!("{}.emb", base),
                        "LM embeds",
                    )?;
                }
            }
            // PLE: token-identity rows from the host table (media placeholder
            // ids included, matching the reference), context projection from
            // the spliced embeddings.
            if self.cfg.text.ple_dim > 0 {
                self.gather_ple_identity(chunk_tokens)?;
                self.compute_ple(take)?;
            }
            // Pre-layer state fingerprint: sum of squares over the ENTIRE
            // residual chunk and the PLE slab, after embed + media splice +
            // PLE. Two identical requests MUST print identical numbers here;
            // a drift means the divergence is upstream of the decoder layers
            // (embed gather, splice, or PLE), while identical numbers push it
            // into the layers — where CIMA_G4_NORMS then bisects by index.
            if std::env::var("CIMA_G4_DEBUG").is_ok() {
                self.ctx.sync()?;
                let sumsq_f16 = |ptr: u64, n_vals: usize| -> Res<f64> {
                    let mut b = vec![0u8; n_vals * 2];
                    self.ctx.dtoh_at(&mut b, ptr)?;
                    Ok(b.chunks_exact(2)
                        .map(|c| {
                            let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64;
                            v * v
                        })
                        .sum())
                };
                let x_ss = sumsq_f16(self.ws.x.ptr, take * hs)?;
                let ple_ss = if self.cfg.text.ple_dim > 0 {
                    let slab = self.cfg.text.n_layers * self.cfg.text.ple_dim;
                    sumsq_f16(self.ws.ple.ptr, take * slab)?
                } else {
                    0.0
                };
                eprintln!(
                    "g4 pre-layer state: rows={} x_sumsq={:.4} ple_sumsq={:.4}",
                    take, x_ss, ple_ss
                );
            }
            if done + take == n {
                // CIMA_G4_NORMS=1: per-layer L2 of the last-position hidden,
                // printed live. A deterministic collapse explodes (inf/NaN
                // or orders-of-magnitude jump) at one layer index — and the
                // index names the subsystem (5,11,17,23,29,35,41 are the
                // global-attention d=512 layers).
                if std::env::var("CIMA_G4_NORMS").is_ok() || std::env::var("CIMA_G4_DEBUG").is_ok()
                {
                    let mut layer_trace: Vec<Vec<f32>> = Vec::new();
                    self.forward_chunk_traced(
                        take,
                        pos0 + done,
                        has_blocks,
                        0,
                        Some(&mut layer_trace),
                    )?;
                    let mut line = String::new();
                    for (i, row) in layer_trace.iter().enumerate() {
                        let l2 = row
                            .iter()
                            .map(|v| (*v as f64) * (*v as f64))
                            .sum::<f64>()
                            .sqrt();
                        let bad = row.iter().any(|v| !v.is_finite());
                        line.push_str(&format!(
                            "L{}={:.1}{} ",
                            i,
                            l2,
                            if bad { "!NaN" } else { "" }
                        ));
                    }
                    eprintln!("g4-norms n={} pos0={} {}", n, pos0 + done, line.trim_end());
                    done += take;
                    continue;
                }
                if let Ok(base) = std::env::var("CIMA_DUMP_LM") {
                    let mut layer_trace: Vec<Vec<f32>> = Vec::new();
                    self.forward_chunk_traced(
                        take,
                        pos0 + done,
                        has_blocks,
                        0,
                        Some(&mut layer_trace),
                    )?;
                    let hsz = self.cfg.text.hidden;
                    let mut out = Vec::with_capacity(8 + layer_trace.len() * hsz * 4);
                    out.extend_from_slice(&(layer_trace.len() as u32).to_le_bytes());
                    out.extend_from_slice(&(hsz as u32).to_le_bytes());
                    for row in &layer_trace {
                        for v in row {
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    let path = format!("{}.layers", base);
                    std::fs::write(&path, out)
                        .map_err(|e| err!("debug", "cannot write '{}': {}", path, e))?;
                    log::info(&format!(
                        "per-layer last-position hiddens dumped to {} ({} layers)",
                        path,
                        layer_trace.len()
                    ));
                } else {
                    self.forward_chunk(take, pos0 + done, has_blocks)?;
                }
            } else {
                self.forward_chunk(take, pos0 + done, has_blocks)?;
            }
            done += take;
            if done < n {
                self.ctx.sync()?;
            }
        }
        self.pos = pos0 + n;
        Ok(())
    }

    /// CIMA_G4_NANHUNT=1: scan ws.x after every stage of every layer.
    /// Prints a heartbeat when armed (instrument silence must be
    /// meaningful), reports the FIRST non-finite value with stage / layer /
    /// row / token id, and always tracks the running max |v| so even an
    /// all-finite poisoned prefill yields its magnitude profile.
    fn nan_hunt(&self, stage: &str, layer: usize, rows: usize) -> Res<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ARMED: AtomicBool = AtomicBool::new(false);
        static FIRED: AtomicBool = AtomicBool::new(false);
        static MAXV: std::sync::Mutex<(f32, String)> = std::sync::Mutex::new((0.0, String::new()));
        if std::env::var("CIMA_G4_NANHUNT").is_err() {
            return Ok(());
        }
        if !ARMED.swap(true, Ordering::Relaxed) {
            eprintln!("g4-nanhunt armed: scanning ws.x at attn/ffn/ple of all layers");
        }
        if stage == "summary" {
            let mut hm = MAXV.lock().unwrap();
            eprintln!(
                "g4-nanhunt summary n={} max|x|={:.1} at {} (f16 ceiling 65504){}",
                rows,
                hm.0,
                hm.1,
                if FIRED.load(Ordering::Relaxed) {
                    " [NaN FIRED earlier]"
                } else {
                    ""
                }
            );
            *hm = (0.0, String::new());
            FIRED.store(false, Ordering::Relaxed);
            return Ok(());
        }
        if FIRED.load(Ordering::Relaxed) {
            return Ok(());
        }
        let hs = self.cfg.text.hidden;
        let mut buf = vec![0u8; rows * hs * 2];
        self.ctx.sync()?;
        self.ctx.dtoh_at(&mut buf, self.ws.x.ptr)?;
        for r in 0..rows {
            for c in 0..hs {
                let v = f16_to_f32(u16::from_le_bytes([
                    buf[(r * hs + c) * 2],
                    buf[(r * hs + c) * 2 + 1],
                ]));
                if !v.is_finite() {
                    FIRED.store(true, Ordering::Relaxed);
                    let tok = self.dbg_tokens.get(r).copied().unwrap_or(0);
                    eprintln!(
                        "g4-nanhunt FIRST POISON stage={} layer={} row={} token_id={} col={} v={}",
                        stage, layer, r, tok, c, v
                    );
                    return Ok(());
                }
                let a = v.abs();
                let mut hm = MAXV.lock().unwrap();
                if a > hm.0 {
                    let tok = self.dbg_tokens.get(r).copied().unwrap_or(0);
                    *hm = (
                        a,
                        format!("stage={} layer={} row={} token_id={}", stage, layer, r, tok),
                    );
                }
            }
        }
        Ok(())
    }

    /// CIMA_G4_AUDIT=1: hash immutable device buffers at every prefill.
    /// Weights must NEVER change across turns — a mutated hash names the
    /// stomped allocation and brackets the turn that stomped it.
    fn integrity_probe(&self, tag: &str) -> Res<()> {
        if std::env::var("CIMA_G4_AUDIT").is_err() {
            return Ok(());
        }
        let fnv = |bytes: &[u8]| -> u64 {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        let grab = |name: &str, ptr: u64, len: usize| -> Res<(String, u64)> {
            let n = len.min(64 * 1024);
            let mut buf = vec![0u8; n];
            self.ctx.dtoh_at(&mut buf, ptr)?;
            Ok((name.to_string(), fnv(&buf)))
        };
        let mut parts = Vec::new();
        parts.push(grab("embed", self.embed.ptr, self.embed.bytes)?);
        parts.push(grab(
            "final_norm",
            self.final_norm.ptr,
            self.final_norm.bytes,
        )?);
        if let Some(l0) = self.layers.first() {
            if let crate::quant::bnb::WTensor::Gguf { buf, .. } = &l0.wq {
                parts.push(grab("l0.wq", buf.ptr, buf.bytes)?);
            }
            if let crate::quant::bnb::WTensor::Gguf { buf, .. } = &l0.w_down {
                parts.push(grab("l0.w_down", buf.ptr, buf.bytes)?);
            }
            if let Some((k, _)) = &l0.kv {
                parts.push(grab("l0.kv_head", k.ptr, 4096)?);
            }
        }
        if let Some(last) = self.layers.last() {
            if let crate::quant::bnb::WTensor::Gguf { buf, .. } = &last.wq {
                parts.push(grab("l41.wq", buf.ptr, buf.bytes)?);
            }
        }
        let line: Vec<String> = parts
            .iter()
            .map(|(n, h)| format!("{}={:016x}", n, h))
            .collect();
        // eprintln, not log::info: chat mode silences info-level lines and
        // an invisible instrument is worse than none.
        eprintln!("g4-audit[{}] {}", tag, line.join(" "));
        Ok(())
    }

    pub fn prefill(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<Vec<f32>> {
        let n = prompt.tokens.len();
        self.integrity_probe(&format!("pre-prefill n={} pos0={}", n, pos0))?;
        self.prefill_inner(prompt, pos0)?;
        self.integrity_probe("post-prefill")?;
        self.nan_hunt("summary", 0, n)?;
        // CIMA_G4_REPLAY=1: determinism oracle. Re-run the exact same
        // prefill and compare a deep hash of layer-0's K cache. Two
        // identical runs MUST produce identical bytes — a mismatch is a
        // race / uninitialized read caught in the act; a match on a turn
        // that then generates garbage convicts deterministic logic instead.
        if std::env::var("CIMA_G4_REPLAY").is_ok() {
            let deep = |this: &Self| -> Res<u64> {
                let mut h: u64 = 0xcbf29ce484222325;
                if let Some(l0) = this.layers.first() {
                    if let Some((k, v)) = &l0.kv {
                        let take = (n * l0.head_dim * 2).min(k.bytes);
                        let mut buf = vec![0u8; take];
                        this.ctx.dtoh_at(&mut buf, k.ptr)?;
                        for &b in &buf {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                        let mut bv = vec![0u8; take.min(v.bytes)];
                        this.ctx.dtoh_at(&mut bv, v.ptr)?;
                        for &b in &bv {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                    }
                }
                Ok(h)
            };
            let h1 = deep(self)?;
            self.prefill_inner(prompt, pos0)?;
            let h2 = deep(self)?;
            eprintln!(
                "g4-replay n={} kv0_run1={:016x} kv0_run2={:016x} → {}",
                n,
                h1,
                h2,
                if h1 == h2 {
                    "DETERMINISTIC (logic bug at this n)"
                } else {
                    "NONDETERMINISTIC (race/uninitialized read)"
                }
            );
        }
        let logits = self.project_logits((n - 1) % CHUNK.max(1))?;
        if let Ok(base) = std::env::var("CIMA_DUMP_LM") {
            let mut out = Vec::with_capacity(4 + logits.len() * 4);
            out.extend_from_slice(&(logits.len() as u32).to_le_bytes());
            for v in &logits {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let path = format!("{}.logits", base);
            std::fs::write(&path, out)
                .map_err(|e| err!("debug", "cannot write '{}': {}", path, e))?;
            log::info(&format!(
                "prefill logits dumped to {} ({} entries)",
                path,
                logits.len()
            ));
        }
        Ok(logits)
    }

    /// Enqueue the device-side decode tail: embedding gather (ids already
    /// uploaded), PLE combine (identity slab already uploaded), all layers
    /// (positions from the device counter), final norm, head, then either
    /// the greedy argmax (`sample_rp: None`) or the device repeat-penalty +
    /// top-64 extraction (`Some(rp)`; see transformer.rs::SAMPLE_TOPK), and
    /// the counter bump. Pure kernel launches — capturable as a CUDA graph.
    fn enqueue_tail(&mut self, sample_rp: Option<f32>) -> Res<()> {
        let t = &self.cfg.text;
        let (hidden, vocab, eps, cap) = (t.hidden, t.vocab, t.rms_eps, t.softcap);
        let ple = t.ple_dim > 0;
        match self.embed_gguf {
            Some(fmt) => self.ctx.gguf_gather(
                fmt,
                self.embed.ptr,
                self.ws.ids.ptr,
                self.ws.x.ptr,
                1,
                hidden,
            )?,
            None => self.ctx.gather(
                self.embed.ptr,
                self.embed_bf16,
                self.ws.ids.ptr,
                self.ws.x.ptr,
                1,
                hidden,
            )?,
        }
        self.ctx
            .scalemul(self.ws.x.ptr, (hidden as f32).sqrt(), hidden)?;
        if ple {
            self.compute_ple(1)?;
        }
        self.forward_chunk_traced(1, 0, false, self.ws.pos_dev.ptr, None)?;
        self.ctx.rmsnorm(
            self.ws.x.ptr,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            hidden,
            eps,
        )?;
        match self.embed_gguf {
            Some(fmt) => self.ctx.gguf_gemv(
                fmt,
                self.ws.h.ptr,
                self.embed.ptr,
                0,
                self.ws.logits_h.ptr,
                vocab,
                hidden,
                0,
            )?,
            None => self.ctx.gemm_f16(
                self.ws.h.ptr,
                self.embed.ptr,
                self.ws.logits_h.ptr,
                1,
                vocab,
                hidden,
            )?,
        }
        match sample_rp {
            None => self.ctx.argmax_softcap_enqueue(
                self.ws.logits_h.ptr,
                &self.ws.argmax_slot,
                vocab,
                cap,
            )?,
            Some(rp) => {
                if rp != 1.0 {
                    self.ctx.apply_penalty(
                        self.ws.logits_h.ptr,
                        self.ws.hist_counts.ptr,
                        rp,
                        vocab,
                    )?;
                }
                self.ctx.topk_enqueue(
                    self.ws.logits_h.ptr,
                    &self.ws.argmax_slot,
                    self.ws.cand.ptr,
                    vocab,
                    cap,
                    crate::models::transformer::SAMPLE_TOPK,
                )?;
            }
        }
        self.ctx.pos_bump(&self.ws.pos_dev)
    }

    pub fn decode_graph_active(&self) -> bool {
        self.decode_graph.is_some()
    }

    /// Initialize the device counter and capture the decode tail. The host
    /// PLE gather stays outside the graph (it reads the mmapped table), so
    /// each greedy token costs: small host gather + 2 uploads + 1 graph
    /// launch + an 8-byte readback — instead of ~12 launches × 35 layers.
    pub fn arm_decode_graph(&mut self, pos: usize) -> Res<()> {
        self.ctx
            .htod(&self.ws.pos_dev, &(pos as u32).to_le_bytes())?;
        if self.decode_graph.is_some()
            || std::env::var("CIMA_NO_GRAPH")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            return Ok(());
        }
        self.ctx.sync()?;
        self.ctx.capture_begin()?;
        let enq = self.enqueue_tail(None);
        match enq.and_then(|_| self.ctx.capture_end()) {
            Ok(gr) => {
                self.decode_graph = Some(gr);
                log::info("gemma4 decode tail captured as a CUDA graph");
            }
            Err(e) => {
                log::warn(&format!(
                    "gemma4 graph capture failed ({}); using per-launch decode",
                    e
                ));
                self.ctx.sync().ok();
            }
        }
        Ok(())
    }

    pub fn arm_sample_graph(&mut self, pos: usize, rp: f32) -> Res<()> {
        self.ctx
            .htod(&self.ws.pos_dev, &(pos as u32).to_le_bytes())?;
        if std::env::var("CIMA_NO_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            return Ok(());
        }
        if matches!(&self.sample_graph, Some((_, r)) if *r == rp) {
            return Ok(());
        }
        self.sample_graph = None;
        self.ctx.sync()?;
        self.ctx.capture_begin()?;
        let enq = self.enqueue_tail(Some(rp));
        match enq.and_then(|_| self.ctx.capture_end()) {
            Ok(gr) => {
                self.sample_graph = Some((gr, rp));
                log::info("gemma4 sampling tail captured as a CUDA graph (device top-k)");
            }
            Err(e) => {
                log::warn(&format!(
                    "gemma4 sampling graph capture failed ({}); using full-logits decode",
                    e
                ));
                self.ctx.sync().ok();
            }
        }
        Ok(())
    }

    pub fn sample_graph_active(&self) -> bool {
        self.sample_graph.is_some()
    }

    pub fn decode_step_sample(&mut self, token: u32, pos: usize) -> Res<Vec<u64>> {
        use crate::models::transformer::SAMPLE_TOPK;
        self.check_capacity(pos + 1)?;
        self.blk_host.truncate(pos);
        self.blk_host.push(-1);
        self.ctx.htod(&self.ws.ids, &token.to_le_bytes())?;
        self.ctx.hist_push(
            self.ws.hist_ring.ptr,
            self.ws.hist_counts.ptr,
            self.ws.ids.ptr,
            0,
            self.ws.pos_dev.ptr,
        )?;
        if self.cfg.text.ple_dim > 0 {
            self.gather_ple_identity(&[token])?;
        }
        let gr = &self.sample_graph.as_ref().expect("armed").0;
        self.ctx.graph_launch(gr)?;
        self.pos = pos + 1;
        let mut raw = vec![0u8; SAMPLE_TOPK * 8];
        self.ctx.dtoh(&mut raw, &self.ws.cand)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn hist_reset_and_seed(&mut self, history: &[u32], next_pos: usize) -> Res<()> {
        self.ctx.memset(&self.ws.hist_ring)?;
        self.ctx.memset(&self.ws.hist_counts)?;
        let lo = next_pos
            .saturating_sub(64)
            .max(next_pos - history.len().min(next_pos));
        for p in lo..next_pos {
            let tok = history[history.len() - (next_pos - p)];
            self.ctx.htod(&self.ws.ids, &tok.to_le_bytes())?;
            self.ctx.hist_push(
                self.ws.hist_ring.ptr,
                self.ws.hist_counts.ptr,
                self.ws.ids.ptr,
                p,
                0,
            )?;
        }
        Ok(())
    }

    /// 8-byte readback of the argmax slot populated inside the graph.
    fn read_argmax_slot(&self) -> Res<u32> {
        let mut out = [0u8; 8];
        self.ctx.dtoh_at(&mut out, self.ws.argmax_slot.ptr)?;
        Ok(u32::from_le_bytes([out[0], out[1], out[2], out[3]]))
    }

    fn decode_step_inner(&mut self, token: u32, pos: usize) -> Res<()> {
        self.check_capacity(pos + 1)?;
        self.blk_host.truncate(pos);
        self.blk_host.push(-1);
        self.embed_tokens(&[token])?;
        if self.cfg.text.ple_dim > 0 {
            self.gather_ple_identity(&[token])?;
            self.compute_ple(1)?;
        }
        self.forward_chunk(1, pos, false)?;
        self.pos = pos + 1;
        Ok(())
    }

    /// Graph decode: host PLE gather + uploads, then one graph launch that
    /// covers embed gather → layers → head → argmax (and bumps the device
    /// position counter). Returns the greedy token.
    fn decode_step_graph(&mut self, token: u32, pos: usize) -> Res<u32> {
        let t0 = std::time::Instant::now();
        self.check_capacity(pos + 1)?;
        self.blk_host.truncate(pos);
        self.blk_host.push(-1);
        self.ctx.htod(&self.ws.ids, &token.to_le_bytes())?;
        if self.cfg.text.ple_dim > 0 {
            self.gather_ple_identity(&[token])?;
        }
        let g = self.decode_graph.as_ref().expect("armed");
        self.ctx.graph_launch(g)?;
        self.pos = pos + 1;
        let tok = self.read_argmax_slot()?;
        let _ = t0;
        Ok(tok)
    }

    pub fn decode_step(&mut self, token: u32, pos: usize) -> Res<Vec<f32>> {
        self.decode_step_inner(token, pos)?;
        self.project_logits(0)
    }

    pub fn embed_pool(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        self.reset()?;
        let (max_seq, ple_dim, hidden, rms_eps) = (
            self.cfg.text.max_seq,
            self.cfg.text.ple_dim,
            self.cfg.text.hidden,
            self.cfg.text.rms_eps,
        );
        let n = tokens.len().min(max_seq).min(CHUNK);
        self.embed_tokens(&tokens[..n])?;
        if ple_dim > 0 {
            self.gather_ple_identity(&tokens[..n])?;
            self.compute_ple(n)?;
        }
        self.forward_chunk(n, 0, false)?;
        self.ctx.rmsnorm(
            self.ws.x.ptr,
            self.final_norm.ptr,
            self.ws.h.ptr,
            n,
            hidden,
            rms_eps,
        )?;
        // CIMA_DUMP_EMB=path: per-token post-norm hidden rows [u32 n][u32 h][f32...]
        // + the token ids — the embed-path bisection probe (tokenization vs
        // forward vs pooling in one dump).
        if let Ok(path) = std::env::var("CIMA_DUMP_EMB") {
            dump_f16_matrix(&self.ctx, self.ws.h.ptr, n, hidden, &path, "embed hidden")?;
            log::info(&format!("embed token ids: {:?}", &tokens[..n]));
        }
        self.ctx
            .meanpool(self.ws.h.ptr, self.ws.logits_f.ptr, n, hidden)?;
        let mut out = vec![0f32; hidden];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, hidden * 4) };
        // dtoh permits a destination smaller than the source buffer.
        self.ctx.dtoh(bytes, &self.ws.logits_f)?;
        Ok(out)
    }

    /// Drain the accumulated host-side timing counters (for the inference
    /// METRIC line). Returns `(ple_ms, logits_ms)`.
    pub fn perf_take(&mut self) -> (f64, f64) {
        let v = (self.perf_ple_ms, self.perf_logits_ms);
        self.perf_ple_ms = 0.0;
        self.perf_logits_ms = 0.0;
        v
    }

    /// Phase-2 self-test: verify the LM-side consumption of soft tokens for
    /// an assembled prompt — the splice must land the tower rows bit-exactly
    /// at the placeholder positions, and the PLE rows at those positions
    /// must equal the PAD token's.
    pub fn splice_check(&mut self, tokens: &[u32], media: &[(usize, u64, usize)]) -> Res<()> {
        let hs = self.cfg.text.hidden;
        let n = tokens.len().min(CHUNK);
        self.reset()?;
        self.embed_tokens(&tokens[..n])?;
        for (at, ptr, rows) in media {
            let (start, end) = (*at, at + rows);
            if end <= n {
                let dst = self.ws.x.ptr + (start * hs * 2) as u64;
                self.ctx.dtod(dst, *ptr, rows * hs * 2)?;
            }
        }
        self.ctx.sync()?;
        let (at, mptr, rows) = media
            .first()
            .map(|(a, p, r)| (*a, *p, *r))
            .ok_or_else(|| err!("selftest", "no media spans in the prepared prompt"))?;
        // compare 3 rows: first, middle, last of the span
        let mut ok = true;
        for probe in [0usize, rows / 2, rows - 1] {
            let mut xrow = vec![0u8; hs * 2];
            self.ctx
                .dtoh_at(&mut xrow, self.ws.x.ptr + ((at + probe) * hs * 2) as u64)?;
            let mut mrow = vec![0u8; hs * 2];
            self.ctx
                .dtoh_at(&mut mrow, mptr + (probe * hs * 2) as u64)?;
            let same = xrow == mrow;
            ok &= same;
            println!(
                "splice row {:>4} (abs pos {:>4}): {}",
                probe,
                at + probe,
                if same { "bit-exact" } else { "MISMATCH" }
            );
        }
        // token frame around the span
        let lo = at.saturating_sub(2);
        let hi = (at + rows + 2).min(tokens.len());
        println!(
            "token frame: positions {}..{} = {:?}",
            lo,
            hi,
            &tokens[lo..hi.min(lo + 6)]
        );
        println!(
            "...span tail = {:?}",
            &tokens[(at + rows).saturating_sub(2)..hi]
        );
        println!(
            "expected: boi={} img={} eoi={} pad={}",
            self.cfg.boi_token_id,
            self.cfg.image_token_id,
            self.cfg.eoi_token_id,
            self.cfg.pad_token_id
        );
        // PLE identity: img placeholder row must equal the PAD row
        if self.cfg.text.ple_dim > 0 {
            self.gather_ple_identity(&[self.cfg.image_token_id])?;
            let bytes = self.cfg.text.n_layers * self.cfg.text.ple_dim * 2;
            let mut a = vec![0u8; bytes];
            self.ctx.dtoh_at(&mut a, self.ws.ple_id.ptr)?;
            self.gather_ple_identity(&[self.cfg.pad_token_id])?;
            let mut b = vec![0u8; bytes];
            self.ctx.dtoh_at(&mut b, self.ws.ple_id.ptr)?;
            let same = a == b;
            ok &= same;
            println!(
                "PLE identity (img placeholder vs PAD): {}",
                if same { "identical" } else { "MISMATCH" }
            );
        }
        println!(
            "splice_check: {}",
            if ok { "ALL OK" } else { "FAILURES FOUND" }
        );

        // ---- LM consumption probe ----
        // The soft tokens reach the LM bit-exactly; what has never been
        // measured is what the 35 layers do to them. Three statistics tell
        // the story: input magnitude (soft tokens vs scaled text embeddings),
        // f16 headroom through the stack, and whether spatially distinct
        // tokens stay distinct. The probe assumes a 16×16 soft-token grid
        // over a quadrant image: rows q0=top-left, q1=top-right,
        // q2=bottom-left, q3=bottom-right.
        let hsz = hs;
        let rms_of = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let fetch_row = |ptr: u64, row: usize| -> Res<Vec<f32>> {
            let mut b = vec![0u8; hsz * 2];
            self.ctx.dtoh_at(&mut b, ptr + (row * hsz * 2) as u64)?;
            Ok(b.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        };
        let cos = |a: &[f32], b: &[f32]| {
            let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            d / (na * nb + 1e-9)
        };
        // grid side of the soft-token raster
        let side = (rows as f32).sqrt() as usize;
        let q = side / 4; // quadrant-center offset
        let centers = [
            (q * side + q),                           // top-left
            (q * side + (side - 1 - q)),              // top-right
            ((side - 1 - q) * side + q),              // bottom-left
            ((side - 1 - q) * side + (side - 1 - q)), // bottom-right
        ];
        let probe = |label: &str, me: &Self| -> Res<()> {
            let mut img_rows = Vec::new();
            for &c in &centers {
                img_rows.push(fetch_row(me.ws.x.ptr, at + c)?);
            }
            let text_row = fetch_row(me.ws.x.ptr, at + rows + 2)?; // a text row after eoi
            let mut maxabs = 0f32;
            for r in img_rows.iter().chain(std::iter::once(&text_row)) {
                for v in r {
                    maxabs = maxabs.max(v.abs());
                }
            }
            println!(
                "{}: rms img=[{:.2} {:.2} {:.2} {:.2}] text={:.2} max|x|={:.0}",
                label,
                rms_of(&img_rows[0]),
                rms_of(&img_rows[1]),
                rms_of(&img_rows[2]),
                rms_of(&img_rows[3]),
                rms_of(&text_row),
                maxabs
            );
            println!(
                "{}: quadrant cosines TL-TR={:.3} TL-BL={:.3} TL-BR={:.3} TR-BL={:.3}",
                label,
                cos(&img_rows[0], &img_rows[1]),
                cos(&img_rows[0], &img_rows[2]),
                cos(&img_rows[0], &img_rows[3]),
                cos(&img_rows[1], &img_rows[2]),
            );
            Ok(())
        };
        probe("pre-LM ", self)?;
        let n_chunk = n;
        self.forward_chunk(n_chunk, 0, false)?;
        self.ctx.sync()?;
        probe("post-LM", self)?;
        self.reset()?;
        Ok(())
    }

    pub fn reset(&mut self) -> Res<()> {
        self.pos = 0;
        self.blk_host.clear();
        // Stateless per request: zero the KV cache so a shorter prompt (or a
        // replayed decode graph) can never attend to rows left by a previous
        // generation. Only non-shared layers own a cache; shared layers read
        // a sibling's, so clearing the owners covers everything. This costs a
        // handful of memsets — negligible beside prefill, and the price of
        // correctness. Also clear the greedy repeat-frequency table (the
        // sampling path reseeds it itself, but the greedy path never did, so
        // counts leaked across requests).
        for layer in &self.layers {
            if let Some((k, v)) = &layer.kv {
                self.ctx.memset(k)?;
                self.ctx.memset(v)?;
            }
        }
        self.ctx.memset(&self.ws.hist_ring)?;
        self.ctx.memset(&self.ws.hist_counts)?;
        Ok(())
    }
}

// ===========================================================================
// Media entry points (called from LoadedModel::prepare)
// ===========================================================================

impl Gemma4 {
    /// Per-channel normalization for images: Gemma4 scales pixels as
    /// `2·(p−0.5)`, i.e. mean=0.5 / std=0.5 (folded into the decoder).
    pub fn image_norm() -> ([f32; 3], [f32; 3]) {
        ([0.5; 3], [0.5; 3])
    }

    /// Aspect-preserving resize target — the reference
    /// `get_aspect_ratio_preserving_size` verbatim: the largest dimensions
    /// that (1) fit the patch budget and (2) are multiples of
    /// `pooling_kernel_size · patch_size` (48 px), with the degenerate
    /// extreme-aspect cases handled identically. Returns `(height, width)`.
    pub fn image_target_size(src_w: usize, src_h: usize) -> Res<(usize, usize)> {
        if src_w == 0 || src_h == 0 {
            return Err(err!("media", "gemma4 vision: image has zero dimension"));
        }
        let total_px = (src_h * src_w) as f64;
        let target_px = (G4_MAX_PATCHES * 16 * 16) as f64;
        let factor = (target_px / total_px).sqrt();
        let sm = G4_SIDE_MULT as f64;
        let mut th = ((factor * src_h as f64) / sm).floor() as usize * G4_SIDE_MULT;
        let mut tw = ((factor * src_w as f64) / sm).floor() as usize * G4_SIDE_MULT;
        let max_side = (G4_MAX_PATCHES / 9) * G4_SIDE_MULT;
        if th == 0 && tw == 0 {
            return Err(err!(
                "media",
                "gemma4 vision: image too small to resize to a multiple of {} px",
                G4_SIDE_MULT
            ));
        }
        if th == 0 {
            th = G4_SIDE_MULT;
            tw = ((src_w / src_h) * G4_SIDE_MULT)
                .min(max_side)
                .max(G4_SIDE_MULT);
        } else if tw == 0 {
            tw = G4_SIDE_MULT;
            th = ((src_h / src_w) * G4_SIDE_MULT)
                .min(max_side)
                .max(G4_SIDE_MULT);
        }
        if (th / 16) * (tw / 16) > G4_MAX_PATCHES {
            return Err(err!(
                "media",
                "gemma4 vision: internal resize overflow ({}×{} patches)",
                tw / 16,
                th / 16
            ));
        }
        Ok((th, tw))
    }

    pub fn encode_image(&self, img: &ImageTensor) -> Res<(DeviceBuf, usize)> {
        let v = self
            .vision
            .as_ref()
            .ok_or_else(|| err!("media", "gemma4: this checkpoint has no vision tower"))?;
        v.encode(&self.ctx, img, self.ws.wsc.ptr)
    }

    pub fn encode_audio(&self, pcm: &AudioPcm) -> Res<(DeviceBuf, usize)> {
        let a = self
            .audio
            .as_ref()
            .ok_or_else(|| err!("media", "gemma4: this checkpoint has no audio tower"))?;
        a.encode(&self.ctx, pcm, self.ws.wsc.ptr)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
}
/// The performance contract (see [`crate::traits::Architecture`]). The
/// inherent methods carry the implementation; this impl is the public
/// scorecard and what `cima profile` / the engine dispatch judge.
impl crate::traits::Architecture for Gemma4 {
    fn modality(&self) -> Modality {
        Gemma4::modality(self)
    }
    fn max_seq(&self) -> usize {
        Gemma4::max_seq(self)
    }
    fn prefill(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<Vec<f32>> {
        Gemma4::prefill(self, prompt, pos0)
    }
    fn decode_step(&mut self, token: u32, pos: usize) -> Res<Vec<f32>> {
        Gemma4::decode_step(self, token, pos)
    }
    fn embed(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        self.embed_pool(tokens)
    }
    fn reset(&mut self) -> Res<()> {
        Gemma4::reset(self)
    }
    fn vram_bytes(&self) -> usize {
        Gemma4::vram_bytes(self)
    }
    fn weight_bytes_resident(&self) -> usize {
        self.weight_bytes
    }
    fn perf_levers(&self) -> crate::traits::PerfLevers {
        crate::traits::PerfLevers {
            device_greedy: true,
            cuda_graph: true, // partial: the host PLE gather stays outside
            // Remaining levers are open work — ClipLin mixes NF4/16-bit
            // codecs (fusion needs same-codec runs) and the sliding-window
            // layers still use the monolithic attention kernel.
            fused_weights: false,
            seq_parallel_attention: true, // both head_dim variants are 32-multiples <= 256
            device_pipeline: false,       // PLE gathers a host table by token id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{self, Json};

    fn full_config() -> Json {
        json::parse(r#"{
            "model_type": "gemma4",
            "image_token_id": 258880, "audio_token_id": 258881,
            "boi_token_id": 255999, "eoi_token_id": 258882,
            "boa_token_id": 256000, "eoa_token_id": 258883,
            "text_config": {
                "hidden_size": 1536, "intermediate_size": 6144,
                "num_hidden_layers": 35, "num_attention_heads": 8,
                "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512,
                "vocab_size": 262144, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768, "sliding_window": 512,
                "num_kv_shared_layers": 20, "pad_token_id": 0,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "final_logit_softcapping": 30.0,
                "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
                }
            },
            "vision_config": null, "audio_config": null
        }"#).unwrap()
    }

    /// Locks the proportional-RoPE discovery: full layers rotate only
    /// `partial_rotary_factor · global_head_dim / 2` frequency pairs with the
    /// *full* head_dim in the exponent denominator.
    #[test]
    fn proportional_rope_params() {
        let cfg = G4Config::parse(&full_config()).unwrap();
        assert_eq!(cfg.text.full_nfreqs, 64, "0.25 · 512 / 2");
        assert_eq!(cfg.text.theta_sliding, 10_000.0);
        assert_eq!(cfg.text.theta_full, 1_000_000.0);
    }

    /// Stripped configs (the unsloth trap): missing rope_parameters must fall
    /// back to the reference defaults, not to garbage.
    #[test]
    fn stripped_config_defaults() {
        let mut j = full_config();
        if let Json::Obj(o) = &mut j {
            if let Some((_, Json::Obj(t))) = o.iter_mut().find(|(k, _)| k == "text_config") {
                t.retain(|(k, _)| k != "rope_parameters" && k != "final_logit_softcapping");
            }
        }
        let cfg = G4Config::parse(&j).unwrap();
        assert_eq!(cfg.text.theta_sliding, 10_000.0);
        assert_eq!(cfg.text.theta_full, 1_000_000.0);
        assert_eq!(cfg.text.full_nfreqs, 64);
    }

    /// Locks the KV-share semantics that cost a debugging marathon: shared
    /// layers read the last computing layer **of their own type**.
    #[test]
    fn kv_share_by_type() {
        let cfg = G4Config::parse(&full_config()).unwrap();
        let map = kv_share_sources(&cfg.text.layer_types, 15).unwrap();
        for (i, &src) in map.iter().enumerate().take(15) {
            assert_eq!(src, i, "computing layer {} reads itself", i);
        }
        for (i, &src) in map.iter().enumerate().take(35).skip(15) {
            let expect = if cfg.text.layer_types[i] == LayerType::Sliding {
                13
            } else {
                14
            };
            assert_eq!(
                src, expect,
                "shared layer {} ({:?})",
                i, cfg.text.layer_types[i]
            );
        }
    }

    /// A shared layer with no earlier computing layer of its type is a
    /// corrupt topology and must fail loudly, not index usize::MAX.
    #[test]
    fn kv_share_missing_source_errors() {
        let types = vec![LayerType::Sliding, LayerType::Full, LayerType::Full];
        // first_shared = 1: layer 1 (Full) shares but no computing Full exists
        assert!(kv_share_sources(&types, 1).is_err());
    }

    /// Degenerate pattern guard: zero layers, all shared, etc. must not panic.
    #[test]
    fn kv_share_degenerate() {
        assert!(kv_share_sources(&[], 0).unwrap().is_empty());
        let all = vec![LayerType::Sliding; 4];
        assert_eq!(kv_share_sources(&all, 4).unwrap(), vec![0, 1, 2, 3]);
    }

    /// Regression guard for the dequant-scratch overflow: a GGUF checkpoint
    /// has NO bitsandbytes quant-states, so the old NF4-only sizing returned
    /// 0 and the prefill GEMM wrote megabytes out of a 2-byte buffer. The
    /// scratch must be sized for the largest GGUF block weight that flows
    /// through a prefill GEMM. Runs with `cargo test` — no GPU, no HTTP.
    #[test]
    fn prefill_scratch_sized_for_gguf_weights() {
        use crate::traits::{DType, LoadedWeights, TensorMeta};
        use std::collections::HashMap;

        struct MockWeights(HashMap<String, TensorMeta>);
        impl LoadedWeights for MockWeights {
            fn tensors(&self) -> &HashMap<String, TensorMeta> {
                &self.0
            }
            fn bytes(&self, _m: &TensorMeta) -> crate::traits::Res<&[u8]> {
                Ok(&[])
            }
        }
        let meta = |name: &str, shape: Vec<usize>, dtype| {
            (
                name.to_string(),
                TensorMeta {
                    name: name.to_string(),
                    dtype,
                    shape,
                    offset: 0,
                    nbytes: 0,
                    file: String::new(),
                },
            )
        };
        let mut t = HashMap::new();
        // A realistic Q4_K FFN-down weight [hidden, inter] = [1536, 6144].
        t.extend([
            meta(
                "language_model.layers.0.mlp.down_proj.weight",
                vec![1536, 6144],
                DType::GgufQ4K,
            ),
            // The tied embedding is huge but excluded (device gather, not GEMM).
            meta(
                "language_model.embed_tokens.weight",
                vec![262144, 1536],
                DType::GgufQ4K,
            ),
            // A norm weight (f16) must not drive scratch.
            meta("language_model.norm.weight", vec![1536], DType::F16),
        ]);
        let w = MockWeights(t);
        let bytes = Gemma4::prefill_scratch_bytes(&w);

        // Must be sized for the 1536×6144 weight dequantized to f16
        // (2 bytes/elem), NOT zero, and NOT the 262144-row embedding.
        let ffn = 1536 * 6144 * 2;
        let embed = 262144 * 1536 * 2;
        assert_eq!(
            bytes, ffn,
            "scratch must fit the largest prefill-GEMM weight ({} bytes), got {}",
            ffn, bytes
        );
        assert!(bytes > 0, "GGUF scratch sized 0 — the overflow bug");
        assert!(
            bytes < embed,
            "embedding must be excluded from scratch sizing"
        );
    }
}
