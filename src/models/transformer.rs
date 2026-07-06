//! Generic dense decoder-only transformer (LLaMA / Qwen / Mistral / Gemma
//! text lineages). Families whose forward pass deviates structurally get
//! their own sibling module (see `gemma4`); everything that is "standard
//! attention + MLP + norms" parametrizes this one graph instead.

use super::towers::{EncoderTower, TowerKind};
use crate::cuda::{fmt_bytes, CudaCtx, DeviceBuf, ATT_CSZ};
use crate::err;
use crate::json::{self, Json};
use crate::log;
use crate::traits::*;
use std::path::Path;
use std::time::Instant;

/// Prefill micro-batch size: bounds workspace VRAM while keeping GEMMs fat.
pub const CHUNK: usize = 512;

// ===========================================================================
// 1. ModelConfig — validated view over config.json
// ===========================================================================

/// Decoder hyper-parameters extracted from `config.json`, with every access
/// validated up-front. A missing or mistyped attribute produces an error that
/// names the exact JSON key — the first layer of the broken-model fail-safe.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// `model_type` (e.g. `"qwen2"`, `"llama"`, `"mistral"`).
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Engine-side context limit (min of config and a sane cap).
    pub max_seq: usize,
    pub tie_word_embeddings: bool,
    /// Qwen2-style attention bias on q/k/v projections.
    pub qkv_bias: bool,
    /// `quantization_config.quant_method` if the checkpoint is quantized.
    pub quant_method: Option<String>,
    /// Present iff the checkpoint carries a vision tower (`vision_config`).
    pub vision: Option<EncoderConfig>,
    /// Present iff the checkpoint carries an audio tower (`audio_config`).
    pub audio: Option<EncoderConfig>,
    /// Token-classification of the repo as an embedding model
    /// (no `lm_head`, sentence-transformers layout, or explicit architecture).
    pub is_embedding: bool,
}

/// Hyper-parameters of a bidirectional encoder tower (vision or audio).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub layer_norm_eps: f32,
    /// Vision: square input edge in pixels. Audio: max source frames.
    pub input_size: usize,
    /// Vision: patch edge in pixels. Audio: mel bins.
    pub patch_size: usize,
}

/// Read a required `usize` key, with a granular error naming the key.
fn req_usize(cfg: &Json, key: &str) -> Res<usize> {
    cfg.usize_of(key).ok_or_else(|| {
        err!(
            "config",
            "config.json: required integer attribute '{}' is missing or not a number",
            key
        )
    })
}

impl ModelConfig {
    /// Parse and validate `dir/config.json`.
    pub fn load(dir: &Path) -> Res<ModelConfig> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            err!(
                "config",
                "cannot read {}: {} — repository is incomplete (was the pull interrupted?)",
                path.display(),
                e
            )
        })?;
        let cfg = json::parse(&text)
            .map_err(|e| err!("config", "{} is not valid JSON: {}", path.display(), e))?;

        let model_type = cfg
            .str_of("model_type")
            .ok_or_else(|| {
                err!(
                    "config",
                    "config.json: 'model_type' missing — cannot select an architecture"
                )
            })?
            .to_string();

        // ---- semantic architecture gate (allowlist) ------------------------
        // The shared pipeline implements *exactly* the llama/qwen2/mistral
        // computation: pre-norm RMSNorm×2 per layer, SwiGLU MLP, single-theta
        // RoPE, global causal attention. A checkpoint whose tensor names
        // happen to resolve but whose math differs (Gemma, Qwen3, …) would
        // load and produce silent garbage — the worst failure mode — so the
        // gate is an allowlist of types whose math is verifiably implemented,
        // with a precise architectural reason for every known-but-unsupported
        // family. New families bind via the `Architecture` trait (traits.rs).
        const SUPPORTED: &[&str] = &[
            "llama",
            "mistral",
            "qwen2",
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen2_audio",
            "llava",
            "gemma4", // dedicated any-to-any pipeline (src/gemma4.rs)
        ];
        if !SUPPORTED.contains(&model_type.as_str()) {
            let reason = match model_type.as_str() {
                "whisper" | "t5" | "bart" | "marian" =>
                    "encoder-decoder architecture with cross-attention; the decoder-only pipeline cannot execute it",
                "gemma" | "gemma2" | "gemma3" | "gemma3_text" | "gemma3n" =>
                    "Gemma-family math differs from the implemented pipeline: GeGLU activation (not SwiGLU), \
                     sandwich norms (pre+post RMSNorm around both attention and MLP), zero-centered RMSNorm \
                     gamma (1+w), sqrt(hidden_size) embedding scaling, and (Gemma 2/3) alternating \
                     sliding-window/global attention with per-type RoPE theta and QK-norm",
                "qwen3" | "qwen3_moe" =>
                    "Qwen3 adds per-head QK-RMSNorm (q_norm/k_norm) absent from the implemented attention path",
                "mixtral" =>
                    "Mixture-of-Experts routing (gate + expert MLPs) is not implemented",
                "phi3" | "phi" | "phi4" =>
                    "Phi uses fused qkv_proj/gate_up_proj tensor layouts the weight indexer does not split",
                "deepseek_v2" | "deepseek_v3" =>
                    "Multi-head Latent Attention (compressed KV) is incompatible with the standard KV cache",
                "bert" | "roberta" | "xlm-roberta" =>
                    "bidirectional encoder; only decoder-based embedding models are executable",
                _ => "its computation graph has not been verified against the implemented pipeline",
            };
            return Err(err!(
                "config",
                "model_type '{}' is not executable by this engine: {}. \
                 Supported model_type values: {}. Other architectures bind through the \
                 `Architecture` trait without touching the API layer (see README § Extending).",
                model_type,
                reason,
                SUPPORTED.join(", ")
            ));
        }

        // Nested text_config (multimodal wrappers like qwen2_vl, llava).
        let text = cfg
            .get("text_config")
            .filter(|t| t.as_obj().is_some())
            .unwrap_or(&cfg);

        let hidden_size = req_usize(text, "hidden_size")?;
        let n_layers = req_usize(text, "num_hidden_layers")?;
        let n_heads = req_usize(text, "num_attention_heads")?;
        let n_kv_heads = text.usize_or("num_key_value_heads", n_heads);
        let head_dim = text.usize_or("head_dim", hidden_size / n_heads.max(1));
        let intermediate_size = req_usize(text, "intermediate_size")?;
        let vocab_size = req_usize(text, "vocab_size")?;

        // ---- structural sanity: dimensions must compose ----
        if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(err!("config", "config.json: zero attention geometry (num_attention_heads={}, num_key_value_heads={}, head_dim={})", n_heads, n_kv_heads, head_dim));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(err!("config", "config.json: num_attention_heads ({}) is not a multiple of num_key_value_heads ({}) — invalid GQA geometry", n_heads, n_kv_heads));
        }
        if head_dim % 2 != 0 {
            return Err(err!(
                "config",
                "config.json: head_dim ({}) must be even for rotary embeddings",
                head_dim
            ));
        }

        let cfg_max = text.usize_or("max_position_embeddings", 4096);
        // Cap the KV allocation; long-context models would otherwise demand
        // tens of GB of cache for contexts a single-user box never reaches.
        // 8192 is the engine's default KV ceiling; CIMA_MAX_SEQ caps it
        // lower — KV is layers×2×kv_heads×head_dim×seq×2B, so halving the
        // context halves the cache, which is often exactly the margin a
        // big quantized checkpoint needs on a small card.
        let env_cap = std::env::var("CIMA_MAX_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let max_seq = match env_cap {
            Some(cap) if cap >= 256 => {
                let m = cfg_max.min(cap);
                crate::log::info(&format!(
                    "CIMA_MAX_SEQ={} — KV cache sized for {} positions",
                    cap, m
                ));
                m
            }
            _ => cfg_max.min(8192),
        };

        let quant_method = cfg
            .path(&["quantization_config", "quant_method"])
            .and_then(Json::as_str)
            .map(str::to_string);

        let vision = cfg.get("vision_config").and_then(|v| {
            Some(EncoderConfig {
                hidden_size: v.usize_of("hidden_size")?,
                intermediate_size: v.usize_or("intermediate_size", 4 * v.usize_of("hidden_size")?),
                n_layers: v
                    .usize_of("num_hidden_layers")
                    .or_else(|| v.usize_of("depth"))?,
                n_heads: v
                    .usize_of("num_attention_heads")
                    .or_else(|| v.usize_of("num_heads"))?,
                layer_norm_eps: v.f32_or("layer_norm_eps", 1e-6),
                input_size: v.usize_or("image_size", 336),
                patch_size: v.usize_or("patch_size", 14),
            })
        });

        let audio = cfg.get("audio_config").and_then(|a| {
            Some(EncoderConfig {
                hidden_size: a
                    .usize_of("d_model")
                    .or_else(|| a.usize_of("hidden_size"))?,
                intermediate_size: a
                    .usize_of("encoder_ffn_dim")
                    .or_else(|| a.usize_of("intermediate_size"))?,
                n_layers: a
                    .usize_of("encoder_layers")
                    .or_else(|| a.usize_of("num_hidden_layers"))?,
                n_heads: a
                    .usize_of("encoder_attention_heads")
                    .or_else(|| a.usize_of("num_attention_heads"))?,
                layer_norm_eps: a.f32_or("layer_norm_eps", 1e-5),
                input_size: a.usize_or("max_source_positions", 1500),
                patch_size: a.usize_or("num_mel_bins", 80),
            })
        });

        // Embedding models: no causal LM head architecture string.
        let is_embedding = cfg
            .arr_of("architectures")
            .map(|a| {
                a.iter()
                    .filter_map(Json::as_str)
                    .all(|s| !s.contains("ForCausalLM") && !s.contains("ConditionalGeneration"))
            })
            .unwrap_or(false);

        let qkv_bias = model_type.starts_with("qwen2");
        Ok(ModelConfig {
            model_type,
            hidden_size,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            rms_eps: text.f32_or("rms_norm_eps", 1e-6),
            rope_theta: text.f32_or("rope_theta", 10_000.0),
            max_seq,
            tie_word_embeddings: text.bool_of("tie_word_embeddings").unwrap_or(false),
            qkv_bias,
            quant_method,
            vision,
            audio,
            is_embedding,
        })
    }

    /// Engine modality derived from the config.
    pub fn modality(&self) -> Modality {
        if self.is_embedding {
            Modality::Embedding
        } else if self.vision.is_some() {
            Modality::VisionText
        } else if self.audio.is_some() {
            Modality::AudioText
        } else {
            Modality::TextToText
        }
    }
}

// ===========================================================================
// 2. VramForecast — exact footprint prediction BEFORE any allocation
// ===========================================================================

/// Predicted VRAM footprint of a model, computed before the first byte is
/// uploaded. If the forecast exceeds free VRAM the load is rejected with a
/// detailed breakdown — the model never partially occupies the GPU.
#[derive(Debug, Clone, Copy)]
pub struct VramForecast {
    pub weights: usize,
    pub kv_cache: usize,
    pub workspace: usize,
    /// Peak *transient* allocation during loading: BF16/F32 tensors stage a
    /// raw copy on device while converting to F16, briefly holding storage
    /// bytes + f16 bytes for the largest such tensor (typically the
    /// embedding table of a tied-head BF16 checkpoint).
    pub load_transient: usize,
}

impl VramForecast {
    pub fn total(&self) -> usize {
        self.weights + self.kv_cache + self.workspace + self.load_transient
    }

    /// Compute the forecast from the parsed weights and config.
    pub fn compute(
        cfg: &ModelConfig,
        weights: &dyn LoadedWeights,
        codec: &dyn WeightCodec,
    ) -> VramForecast {
        let mut w: usize = weights
            .tensors()
            .values()
            .map(|t| codec.device_bytes(t))
            .sum();
        if codec.resident_quant() {
            // The prefill dequant scratch is part of the resident bill.
            let (hs, qd, kvd, inter) = (
                cfg.hidden_size,
                cfg.n_heads * cfg.head_dim,
                cfg.n_kv_heads * cfg.head_dim,
                cfg.intermediate_size,
            );
            let largest = (qd * hs)
                .max(kvd * hs)
                .max(inter * hs)
                .max(hs * inter)
                .max(hs * qd);
            w += (largest * 2).min(32 << 20);
        }
        // Largest conversion staging buffer (raw storage bytes of the biggest
        // non-F16 tensor); zero for pure-F16 checkpoints.
        let transient = weights
            .tensors()
            .values()
            .filter(|t| !matches!(t.dtype, DType::F16))
            .map(|t| t.numel() * t.dtype.size())
            .max()
            .unwrap_or(0);
        // KV: layers × {K,V} × kv_heads × max_seq × head_dim × sizeof(f16)
        let kv = cfg.n_layers * 2 * cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2;
        // Workspace: x, h, residual at [CHUNK, hidden]; q at [CHUNK, heads*dim];
        // k/v at [CHUNK, kv_heads*dim]; ffn gate/up at [CHUNK, inter]; logits row.
        let h = cfg.hidden_size.max(cfg.n_heads * cfg.head_dim);
        let ws = CHUNK * (3 * h + cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim + 2 * cfg.intermediate_size) * 2
            + (cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim) * 2 // fused decode q|k|v row
            + cfg.vocab_size * 4 * 2;
        VramForecast {
            weights: w,
            kv_cache: kv,
            workspace: ws,
            load_transient: transient,
        }
    }

    /// Fail-fast fit check against live NVML telemetry.
    pub fn check(&self, ctx: &CudaCtx, model: &str) -> Res<()> {
        let (free, total) = ctx.free_vram()?;
        let need = self.total();
        log::info(&format!(
            "VRAM forecast for {}: weights={} kv={} workspace={} load-transient={} total={} | device free={} / {}",
            model,
            fmt_bytes(self.weights),
            fmt_bytes(self.kv_cache),
            fmt_bytes(self.workspace),
            fmt_bytes(self.load_transient),
            fmt_bytes(need),
            fmt_bytes(free),
            fmt_bytes(total)
        ));
        // Refuse without headroom: allocator fragmentation, driver variance
        // and per-kernel module load can eat ~100-200 MiB beyond the exact
        // sum, and an OOM MID-LOAD is far worse than a clean refusal.
        const HEADROOM: usize = 192 * 1024 * 1024;
        if need + HEADROOM > free {
            return Err(err!(
                "vram",
                "model '{}' does not fit: requires {} + 192 MiB headroom ({} weights + {} KV cache + {} workspace) \
                 but only {} of {} VRAM is free. Nothing was allocated. \
                 Note: free is measured inside the live CUDA process — context + kernel modules + cuBLAS \
                 cost real VRAM that nvidia-smi (between processes) doesn't show; CIMA_VRAM_TRACE=1 attributes it. \
                 Free VRAM (evict the resident model, close other GPU processes, lower CIMA_MAX_SEQ) or use a smaller checkpoint.",
                model,
                fmt_bytes(need),
                fmt_bytes(self.weights),
                fmt_bytes(self.kv_cache),
                fmt_bytes(self.workspace),
                fmt_bytes(free),
                fmt_bytes(total)
            ));
        }
        Ok(())
    }

    /// HOST-memory guard: weight loading stages tensors through host
    /// buffers (decode-to-f32, permutes, dequant scratch) that can
    /// transiently need several times the largest tensor's file size. If
    /// that exceeds MemAvailable, Linux doesn't fail cleanly — it swap-
    /// thrashes and takes the whole machine down. Refusing up front is the
    /// production-safe behavior. No-op where /proc/meminfo is unavailable.
    pub fn host_guard(weights: &dyn LoadedWeights, model: &str) -> Res<()> {
        let Ok(mi) = std::fs::read_to_string("/proc/meminfo") else {
            return Ok(());
        };
        let Some(avail_kb) = mi
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<usize>().ok())
        else {
            return Ok(());
        };
        let avail = avail_kb * 1024;
        // Model the loader's ACTUAL allocation, not a paranoid multiple:
        // sources are mmap'd (page cache — reclaimable, ~free), f16/u8
        // tensors upload directly, and the only real staging buffer is the
        // f16 conversion output for the largest non-f16 tensor (bf16/f32
        // convert, gguf-block dequant): numel × 2 bytes, plus slack.
        let largest_staged = weights
            .tensors()
            .values()
            .filter(|m| !matches!(m.dtype, DType::F16 | DType::U8))
            .map(|m| m.numel() * 2)
            .max()
            .unwrap_or(0);
        let need = largest_staged + 256 * 1024 * 1024;
        if avail < need {
            return Err(err!(
                "mem",
                "loading '{}' needs ~{} of host RAM for staging (largest conversion buffer {} + slack) but only {} is available — \
                 refusing to avoid swap-thrash that can freeze the machine. Close applications or add swap deliberately.",
                model, fmt_bytes(need), fmt_bytes(largest_staged), fmt_bytes(avail)
            ));
        }
        Ok(())
    }
}

// ===========================================================================
// 3. WeightIndex — granular tensor resolution
// ===========================================================================

/// Resolves canonical tensor names across the prefix conventions found on the
/// Hub (`model.layers.*`, bare `layers.*`, `language_model.model.*`, …) and
/// uploads them through the active [`WeightCodec`]. Every miss reports the
/// full candidate list — the second layer of the broken-model fail-safe.
pub(super) struct WeightIndex<'a> {
    pub(crate) ctx: &'a CudaCtx,
    pub(super) weights: &'a dyn LoadedWeights,
    pub(super) codec: &'a dyn WeightCodec,
    prefixes: Vec<&'static str>,
}

impl<'a> WeightIndex<'a> {
    fn new(ctx: &'a CudaCtx, weights: &'a dyn LoadedWeights, codec: &'a dyn WeightCodec) -> Self {
        WeightIndex {
            ctx,
            weights,
            codec,
            prefixes: vec![
                "",
                "model.",
                "language_model.model.",
                "language_model.",
                "transformer.",
            ],
        }
    }

    /// Find a tensor by suffix under any known prefix.
    pub(super) fn meta(&self, name: &str) -> Res<&'a TensorMeta> {
        for p in &self.prefixes {
            let full = format!("{}{}", p, name);
            if let Some(m) = self.weights.tensors().get(&full) {
                return Ok(m);
            }
        }
        let tried: Vec<String> = self
            .prefixes
            .iter()
            .map(|p| format!("{}{}", p, name))
            .collect();
        Err(err!(
            "weights",
            "required tensor '{}' not found in checkpoint (tried: {}). \
             The repository is incomplete or uses an unrecognized layout; total tensors present: {}",
            name,
            tried.join(", "),
            self.weights.tensors().len()
        ))
    }

    fn exists(&self, name: &str) -> bool {
        self.prefixes.iter().any(|p| {
            self.weights
                .tensors()
                .contains_key(&format!("{}{}", p, name))
        })
    }

    /// Validate shape then upload through the codec.
    /// Upload a linear as `Lin`: packed when the codec keeps gguf blocks
    /// resident and the tensor is a block format, f16 otherwise. Enforces
    /// the ggml invariant the kernels rely on: row length divisible by the
    /// block size.
    fn upload_lin(&self, name: &str, expect: &[usize]) -> Res<Lin> {
        let meta = self.meta(name)?;
        let (n, k) = (expect[0], expect[1]);
        if self.codec.resident_quant() && crate::traits::is_gguf_block(meta.dtype) {
            if meta.shape != expect {
                return Err(err!(
                    "weights",
                    "tensor '{}' has shape {:?}, expected {:?}",
                    name,
                    meta.shape,
                    expect
                ));
            }
            let elems = crate::traits::block_elems(meta.dtype);
            if k % elems != 0 {
                return Err(err!(
                    "gguf",
                    "tensor '{}': row length {} is not a multiple of the {:?} block ({} elems)",
                    name,
                    k,
                    meta.dtype,
                    elems
                ));
            }
            let host = self.weights.bytes(meta)?;
            let buf = self.codec.upload(self.ctx, meta, host)?;
            // Pre-grow the q8 activation scratch at load time: lazy growth
            // inside a CUDA-graph capture is illegal (CUresult 900).
            self.ctx.ensure_q8_scratch(k)?;
            return Ok(Lin::Quant {
                buf,
                fmt: meta.dtype,
                n,
                k,
            });
        }
        Ok(Lin::F16 {
            buf: self.upload(name, expect)?,
            n,
            k,
        })
    }

    fn upload(&self, name: &str, expect: &[usize]) -> Res<DeviceBuf> {
        let meta = self.meta(name)?;
        if !expect.is_empty() && meta.shape != expect {
            return Err(err!(
                "weights",
                "tensor '{}' has shape {:?} but the architecture requires {:?} — \
                 config.json and the checkpoint disagree (corrupted or mismatched repository)",
                meta.name,
                meta.shape,
                expect
            ));
        }
        if !self.codec.accepts(meta.dtype) {
            return Err(err!(
                "weights",
                "tensor '{}' is stored as {} which codec '{}' cannot execute",
                meta.name,
                meta.dtype.name(),
                self.codec.name()
            ));
        }
        let host = self.weights.bytes(meta)?;
        self.codec.upload(self.ctx, meta, host)
    }
}

// ===========================================================================
// 4. Transformer — the shared causal-LM execution pipeline
// ===========================================================================

/// Device-resident weights of one decoder layer.
/// One linear weight, in whichever residency the codec chose.
pub(crate) enum Lin {
    /// Uniform f16 (cuBLAS / house GEMV operand), `[n, k]`.
    F16 { buf: DeviceBuf, n: usize, k: usize },
    /// Packed GGUF blocks resident as stored: the fused gemv decodes in
    /// registers (decode), the dequant scratch feeds cuBLAS (prefill).
    Quant {
        buf: DeviceBuf,
        fmt: DType,
        n: usize,
        k: usize,
    },
}

/// Decode-path GEMV through a `Lin`: `y[n] = x[k]·W^T (+bias)`;
/// `mode 1` accumulates into y (residual epilogue), matching `gemv_f16`.
fn lin_gemv(ctx: &CudaCtx, w: &Lin, x: u64, y: u64, bias: u64, mode: i32) -> Res<()> {
    match w {
        Lin::F16 { buf, n, k } => ctx.gemv_f16(buf.ptr, x, y, bias, *n, *k, mode),
        Lin::Quant { buf, fmt, n, k } => ctx.gguf_gemv(*fmt, x, buf.ptr, bias, y, *n, *k, mode),
    }
}

/// Prefill-path GEMM through a `Lin`: quantized weights dequantize into
/// the shared layer scratch first (one layer's largest linear at a time —
/// never the whole model), then run the same tensor-core GEMM as f16.
pub(super) fn lin_gemm(
    ctx: &CudaCtx,
    w: &Lin,
    dq: Option<&DeviceBuf>,
    act: u64,
    out: u64,
    rows: usize,
) -> Res<()> {
    match w {
        Lin::F16 { buf, n, k } => ctx.gemm_f16(act, buf.ptr, out, rows, *n, *k),
        Lin::Quant { buf, fmt, n, k } => {
            // Slabbed: dequantize W in row slices through a FIXED scratch
            // (DQ_SCRATCH bytes — a 7B's largest linear would otherwise
            // want 135 MiB), one tensor-core GEMM per slab writing its
            // column range of the output. cuBLAS row-major trick: out
            // columns offset by slab start.
            let scratch =
                dq.ok_or_else(|| err!("transformer", "quant prefill without dequant scratch"))?;
            let elems = crate::traits::block_elems(*fmt);
            let blk_bytes = crate::formats::gguf::storage_bytes(*fmt, elems);
            let row_bytes = k / elems * blk_bytes;
            let slab = (scratch.bytes / (k * 2)).max(1);
            let mut r0 = 0usize;
            while r0 < *n {
                let nr = slab.min(n - r0);
                ctx.gguf_dequant(
                    *fmt,
                    buf.ptr + (r0 * row_bytes) as u64,
                    scratch.ptr,
                    nr * k / elems,
                )?;
                ctx.gemm_strided_out(act, scratch.ptr, out + (r0 * 2) as u64, rows, nr, *k, *n)?;
                r0 += nr;
            }
            Ok(())
        }
    }
}

/// Layer weights, in one of two residencies:
/// * `Fused` (uniform f16): Q|K|V and gate|up row-concatenated — one large
///   decode GEMV instead of three (two) small ones.
/// * `Split` (resident-quantized models): per-projection `Lin`s. GGUF
///   files mix block formats per tensor (unsloth dynamic quants), so
///   row-concatenation is impossible; each projection dispatches to its
///   own fused-dequant GEMV. Extra launches, hidden by the decode graph.
// Allocated once per layer at load time (not per token), holding small
// device handles; boxing the Split arm to shrink the enum would add a
// pointer indirection on every weight access in the decode loop — the
// wrong trade for a few KB saved once.
#[allow(clippy::large_enum_variant)]
enum LayerW {
    Fused {
        wqkv: DeviceBuf,
        wo: DeviceBuf,
        w_gateup: DeviceBuf,
        w_down: DeviceBuf,
    },
    Split {
        wq: Lin,
        wk: Lin,
        wv: Lin,
        wo: Lin,
        w_gate: Lin,
        w_up: Lin,
        w_down: Lin,
    },
}

struct Layer {
    attn_norm: DeviceBuf,
    w: LayerW,
    /// q|k|v biases concatenated `[qd + 2*kvd]` (None when the checkpoint
    /// has no attention biases); both paths apply row sub-ranges.
    bqkv: Option<DeviceBuf>,
    ffn_norm: DeviceBuf,
    k_cache: DeviceBuf,
    v_cache: DeviceBuf,
}

/// Reusable activation workspace, sized once for `CHUNK` rows.
struct Workspace {
    /// Hidden state `[CHUNK, hidden]`.
    x: DeviceBuf,
    /// Normed hidden `[CHUNK, hidden]`.
    h: DeviceBuf,
    /// Attention output / scratch `[CHUNK, q_heads*dim]`.
    att: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    gate: DeviceBuf,
    up: DeviceBuf,
    /// Fused decode q|k|v row `[1, qd+2*kvd]` (single-GEMV output; the
    /// gate|up pair GEMV writes its silu·mul result straight into `gate`).
    qkv: DeviceBuf,
    /// Flash-decode partials `[q_heads, n_chunks, head_dim+2]` f32.
    att_part: DeviceBuf,
    /// Device-sampling state: top-64 candidates (packed u64), the 64-token
    /// penalty ring, and per-vocab occurrence counts.
    cand: DeviceBuf,
    hist_ring: DeviceBuf,
    hist_counts: DeviceBuf,
    /// Final-token logits `[vocab]` f16 + f32 mirror.
    logits_h: DeviceBuf,
    logits_f: DeviceBuf,
    /// 8-byte scratch for the device-side greedy argmax.
    argmax_slot: DeviceBuf,
    /// Device position counter for graph-replayable decode steps.
    pos_dev: DeviceBuf,
    /// Token-id staging `[CHUNK]` u32.
    ids: DeviceBuf,
    /// Prefill dequant scratch for resident-quantized models: the largest
    /// single layer linear as f16 (≈ inter×hidden×2 — tens of MB, never
    /// the full model). `None` for uniform-f16 checkpoints.
    dq: Option<DeviceBuf>,
}

/// The shared decoder-only transformer (Llama / Qwen2 / Mistral family),
/// fully resident in VRAM in uniform f16, executing through the JIT kernel
/// set and cuBLAS tensor-core GEMMs.
/// Candidates extracted per device-sampling step. The host sampler
/// truncates to top-k before top-p, so any `top_k <= SAMPLE_TOPK` makes
/// the device path exact; larger/unbounded k falls back to full logits.
pub const SAMPLE_TOPK: usize = 64;

/// Prefill dequant scratch ceiling (bytes): big enough that slab GEMMs
/// stay tensor-core efficient, small enough to never decide whether a
/// model fits.
const DQ_SCRATCH: usize = 32 << 20;

pub struct Transformer {
    /// Captured greedy decode step (see `arm_decode_graph`).
    decode_graph: Option<crate::cuda::GraphExec>,
    /// Captured sampling decode step (penalty + top-64 extraction); the
    /// repeat-penalty value is baked into the capture.
    sample_graph: Option<(crate::cuda::GraphExec, f32)>,
    pub(crate) cfg: ModelConfig,
    embed: Lin,
    /// True if the embedding table stayed bf16 on device (gather converts).
    embed_bf16: bool,
    final_norm: DeviceBuf,
    lm_head: Option<Lin>, // None => tied to `embed`
    layers: Vec<Layer>,
    ws: Workspace,
    /// Current absolute sequence position (KV fill level).
    pos: usize,
    vram: usize,
    pub(crate) ctx: std::sync::Arc<CudaCtx>,
    /// Optional encoder towers feeding `media_embeds`.
    pub(crate) vision: Option<EncoderTower>,
    pub(crate) audio: Option<EncoderTower>,
    /// Resident weight bytes (the per-token streaming floor).
    pub(crate) weight_bytes: usize,
}

impl Transformer {
    /// Build the model: validate every tensor against the config geometry,
    /// upload through `codec`, allocate KV + workspace. On any error the
    /// partially-built buffers drop and VRAM returns to its prior state.
    pub fn build(
        ctx: std::sync::Arc<CudaCtx>,
        cfg: ModelConfig,
        weights: &dyn LoadedWeights,
        codec: &dyn WeightCodec,
    ) -> Res<Transformer> {
        let t0 = Instant::now();
        let ix = WeightIndex::new(&ctx, weights, codec);
        let (hs, qd, kvd, inter, vocab) = (
            cfg.hidden_size,
            cfg.n_heads * cfg.head_dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.intermediate_size,
            cfg.vocab_size,
        );

        // ---- embeddings ----------------------------------------------------
        // Two residency strategies, chosen by whether the LM head is tied:
        //  * untied head → keep the table at storage dtype (F16 or BF16); the
        //    gather kernel converts per-row and the table is never a GEMM
        //    operand, so no conversion pass is needed.
        //  * tied head   → the table doubles as the lm_head GEMM operand,
        //    which must be F16. Route BF16 (and F32) through the codec, which
        //    stages via pinned memory and converts on-device (`bf2h`/`f2h`).
        let emb_meta = ix.meta("embed_tokens.weight")?;
        if emb_meta.shape != vec![vocab, hs] {
            return Err(err!(
                "weights",
                "embedding table '{}' has shape {:?}, expected [{}, {}] (vocab_size × hidden_size from config.json)",
                emb_meta.name, emb_meta.shape, vocab, hs
            ));
        }
        let tied_head = cfg.tie_word_embeddings || !ix.exists("lm_head.weight");
        let resident = codec.resident_quant();
        let embed_bf16 = emb_meta.dtype == DType::BF16 && !tied_head && !resident;
        let embed = if resident && crate::traits::is_gguf_block(emb_meta.dtype) {
            // Packed table: the gguf gather kernel dequantizes rows on the
            // fly, and the (possibly tied) head runs the fused gemv.
            ix.upload_lin("embed_tokens.weight", &[vocab, hs])?
        } else if emb_meta.dtype == DType::F16 || embed_bf16 {
            // Raw upload: gather reads f16 or bf16 directly.
            let host = weights.bytes(emb_meta)?;
            let buf = ctx.alloc(host.len())?;
            ctx.htod(&buf, host)?;
            Lin::F16 {
                buf,
                n: vocab,
                k: hs,
            }
        } else {
            // Codec path: lands on device as F16 regardless of storage dtype,
            // making the table a legal cublasGemmEx operand for the tied head.
            Lin::F16 {
                buf: ix.upload("embed_tokens.weight", &[vocab, hs])?,
                n: vocab,
                k: hs,
            }
        };

        let final_norm = ix.upload("norm.weight", &[hs])?;
        let lm_head = if tied_head {
            if !cfg.tie_word_embeddings && !cfg.is_embedding {
                log::warn("lm_head.weight absent — assuming tied embeddings");
            }
            None
        } else {
            // lm_head lives outside the `model.` prefix in HF checkpoints.
            Some(ix.upload_lin("lm_head.weight", &[vocab, hs])?)
        };

        // ---- per-layer weights, exhaustively shape-checked ----
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("layers.{}.{}", l, s);
            let bias = |name: &str, n: usize| -> Res<Option<DeviceBuf>> {
                if cfg.qkv_bias && ix.exists(&p(name)) {
                    Ok(Some(ix.upload(&p(name), &[n])?))
                } else {
                    Ok(None)
                }
            };
            // Q|K|V and gate|up are row-concatenated into single buffers:
            // larger decode GEMVs reach far closer to peak bandwidth than
            // three (two) small ones, and row sub-ranges still feed the
            // prefill GEMMs. Sources are uploaded normally (codec handles
            // dtype) and packed with device-to-device copies.
            let fuse = |parts: &[&DeviceBuf]| -> Res<DeviceBuf> {
                let total = parts.iter().map(|b| b.bytes).sum();
                let fused = ctx.alloc(total)?;
                let mut off = 0u64;
                for b in parts {
                    ctx.dtod(fused.ptr + off, b.ptr, b.bytes)?;
                    off += b.bytes as u64;
                }
                Ok(fused)
            };
            let w = if resident {
                // GGUF files mix block formats per tensor, so projections
                // can't row-concatenate; each stays a separate Lin behind
                // its own fused-dequant GEMV.
                LayerW::Split {
                    wq: ix.upload_lin(&p("self_attn.q_proj.weight"), &[qd, hs])?,
                    wk: ix.upload_lin(&p("self_attn.k_proj.weight"), &[kvd, hs])?,
                    wv: ix.upload_lin(&p("self_attn.v_proj.weight"), &[kvd, hs])?,
                    wo: ix.upload_lin(&p("self_attn.o_proj.weight"), &[hs, qd])?,
                    w_gate: ix.upload_lin(&p("mlp.gate_proj.weight"), &[inter, hs])?,
                    w_up: ix.upload_lin(&p("mlp.up_proj.weight"), &[inter, hs])?,
                    w_down: ix.upload_lin(&p("mlp.down_proj.weight"), &[hs, inter])?,
                }
            } else {
                let (q, k, v) = (
                    ix.upload(&p("self_attn.q_proj.weight"), &[qd, hs])?,
                    ix.upload(&p("self_attn.k_proj.weight"), &[kvd, hs])?,
                    ix.upload(&p("self_attn.v_proj.weight"), &[kvd, hs])?,
                );
                let (gate, up) = (
                    ix.upload(&p("mlp.gate_proj.weight"), &[inter, hs])?,
                    ix.upload(&p("mlp.up_proj.weight"), &[inter, hs])?,
                );
                LayerW::Fused {
                    wqkv: fuse(&[&q, &k, &v])?,
                    wo: ix.upload(&p("self_attn.o_proj.weight"), &[hs, qd])?,
                    w_gateup: fuse(&[&gate, &up])?,
                    w_down: ix.upload(&p("mlp.down_proj.weight"), &[hs, inter])?,
                }
            };
            layers.push(Layer {
                attn_norm: ix.upload(&p("input_layernorm.weight"), &[hs])?,
                w,
                bqkv: match (
                    bias("self_attn.q_proj.bias", qd)?,
                    bias("self_attn.k_proj.bias", kvd)?,
                    bias("self_attn.v_proj.bias", kvd)?,
                ) {
                    (Some(bq), Some(bk), Some(bv)) => Some(fuse(&[&bq, &bk, &bv])?),
                    _ => None,
                },
                ffn_norm: ix.upload(&p("post_attention_layernorm.weight"), &[hs])?,
                k_cache: ctx.alloc(cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2)?,
                v_cache: ctx.alloc(cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2)?,
            });
        }

        // ---- workspace ----
        let ws = Workspace {
            x: ctx.alloc(CHUNK * hs * 2)?,
            h: ctx.alloc(CHUNK * hs.max(qd) * 2)?,
            att: ctx.alloc(CHUNK * qd * 2)?,
            q: ctx.alloc(CHUNK * qd * 2)?,
            k: ctx.alloc(CHUNK * kvd * 2)?,
            v: ctx.alloc(CHUNK * kvd * 2)?,
            qkv: ctx.alloc((qd + 2 * kvd) * 2)?,
            att_part: ctx.alloc(
                cfg.n_heads * ((cfg.max_seq + ATT_CSZ - 1) / ATT_CSZ) * (cfg.head_dim + 2) * 4,
            )?,
            cand: ctx.alloc(SAMPLE_TOPK * 8)?,
            hist_ring: ctx.alloc(64 * 4)?,
            hist_counts: ctx.alloc(cfg.vocab_size * 4)?,
            gate: ctx.alloc(CHUNK * inter * 2)?,
            up: ctx.alloc(CHUNK * inter * 2)?,
            logits_h: ctx.alloc(vocab * 2)?,
            logits_f: ctx.alloc(vocab * 4)?,
            argmax_slot: ctx.alloc(8)?,
            pos_dev: ctx.alloc(4)?,
            ids: ctx.alloc(CHUNK * 4)?,
            dq: if resident {
                // Fixed-size: prefill dequantizes weight ROW SLABS through
                // it (see lin_gemm), so it never needs the whole linear.
                let largest = (qd * hs)
                    .max(kvd * hs)
                    .max(inter * hs)
                    .max(hs * inter)
                    .max(hs * qd);
                Some(ctx.alloc((largest * 2).min(DQ_SCRATCH))?)
            } else {
                None
            },
        };

        // ---- optional encoder towers ----
        let vision = match &cfg.vision {
            Some(ec) => Some(EncoderTower::build(&ctx, ec, &ix, TowerKind::Vision, hs)?),
            None => None,
        };
        let audio = match &cfg.audio {
            Some(ec) => Some(EncoderTower::build(&ctx, ec, &ix, TowerKind::Audio, hs)?),
            None => None,
        };

        ctx.sync()?;
        let vram = ctx.tracked_bytes();

        if log::debug_on() {
            log::debug(&format!(
                "model card [{}]\n  hidden={} inter={} layers={} heads={} kv_heads={} head_dim={} vocab={} max_seq={}\n  rope theta={} rms_eps={} tied_embeddings={} qkv_bias={}\n  quant={:?} vision={} audio={} embedding_model={}",
                cfg.model_type,
                cfg.hidden_size, cfg.intermediate_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
                cfg.head_dim, cfg.vocab_size, cfg.max_seq,
                cfg.rope_theta, cfg.rms_eps, cfg.tie_word_embeddings, cfg.qkv_bias,
                cfg.quant_method, cfg.vision.is_some(), cfg.audio.is_some(), cfg.is_embedding,
            ));
        }
        log::info(&format!(
            "model graph resident: {} layers, {} VRAM, built in {:?}",
            cfg.n_layers,
            fmt_bytes(vram),
            t0.elapsed()
        ));
        Ok(Transformer {
            decode_graph: None,
            sample_graph: None,
            weight_bytes: VramForecast::compute(&cfg, weights, codec).weights,
            cfg,
            embed,
            embed_bf16,
            final_norm,
            lm_head,
            layers,
            ws,
            pos: 0,
            vram,
            ctx,
            vision,
            audio,
        })
    }

    /// Run `rows` already-embedded hidden states (resident in `ws.x`) through
    /// every layer, appending to the KV cache at absolute position `pos0`.
    fn forward_chunk(&mut self, rows: usize, pos0: usize) -> Res<()> {
        self.forward_chunk_at(rows, pos0, 0)
    }

    /// `pos_dev != 0`: position-dependent kernels read the device counter
    /// instead of `pos0`, making the enqueued work graph-replayable.
    fn forward_chunk_at(&mut self, rows: usize, pos0: usize, pos_dev: u64) -> Res<()> {
        let c = &self.cfg;
        let ctx = &self.ctx;
        let (hs, qd, kvd) = (
            c.hidden_size,
            c.n_heads * c.head_dim,
            c.n_kv_heads * c.head_dim,
        );
        let inter = c.intermediate_size;
        // Row offsets into the fused weight buffers (bytes).
        let (wk_off, wv_off) = ((qd * hs * 2) as u64, ((qd + kvd) * hs * 2) as u64);
        let up_off = (inter * hs * 2) as u64;
        for layer in &self.layers {
            // --- attention block ---
            ctx.rmsnorm(
                self.ws.x.ptr,
                layer.attn_norm.ptr,
                self.ws.h.ptr,
                rows,
                hs,
                c.rms_eps,
            )?;
            // Fused decode runs Q|K|V as ONE GEMV into the contiguous qkv
            // row (small per-matrix GEMVs leave most of the bus idle);
            // Split (resident-quantized) issues one fused-dequant GEMV per
            // projection into the same contiguous row. Prefill keeps
            // separate outputs either way.
            let (q_p, k_p, v_p) = if rows == 1 {
                let base = self.ws.qkv.ptr;
                let b = layer.bqkv.as_ref().map(|b| b.ptr).unwrap_or(0);
                match &layer.w {
                    LayerW::Fused { wqkv, .. } => {
                        ctx.gemv_f16(wqkv.ptr, self.ws.h.ptr, base, b, qd + 2 * kvd, hs, 0)?;
                    }
                    LayerW::Split { wq, wk, wv, .. } => {
                        let (bk, bv) = if b == 0 {
                            (0, 0)
                        } else {
                            (b + (qd * 2) as u64, b + ((qd + kvd) * 2) as u64)
                        };
                        lin_gemv(ctx, wq, self.ws.h.ptr, base, b, 0)?;
                        lin_gemv(ctx, wk, self.ws.h.ptr, base + (qd * 2) as u64, bk, 0)?;
                        lin_gemv(
                            ctx,
                            wv,
                            self.ws.h.ptr,
                            base + ((qd + kvd) * 2) as u64,
                            bv,
                            0,
                        )?;
                    }
                }
                (base, base + (qd * 2) as u64, base + ((qd + kvd) * 2) as u64)
            } else {
                match &layer.w {
                    LayerW::Fused { wqkv, .. } => {
                        ctx.gemm_f16(self.ws.h.ptr, wqkv.ptr, self.ws.q.ptr, rows, qd, hs)?;
                        ctx.gemm_f16(
                            self.ws.h.ptr,
                            wqkv.ptr + wk_off,
                            self.ws.k.ptr,
                            rows,
                            kvd,
                            hs,
                        )?;
                        ctx.gemm_f16(
                            self.ws.h.ptr,
                            wqkv.ptr + wv_off,
                            self.ws.v.ptr,
                            rows,
                            kvd,
                            hs,
                        )?;
                    }
                    LayerW::Split { wq, wk, wv, .. } => {
                        let dq = self.ws.dq.as_ref();
                        lin_gemm(ctx, wq, dq, self.ws.h.ptr, self.ws.q.ptr, rows)?;
                        lin_gemm(ctx, wk, dq, self.ws.h.ptr, self.ws.k.ptr, rows)?;
                        lin_gemm(ctx, wv, dq, self.ws.h.ptr, self.ws.v.ptr, rows)?;
                    }
                }
                if let Some(b) = &layer.bqkv {
                    ctx.bias(self.ws.q.ptr, b.ptr, rows, qd)?;
                    ctx.bias(self.ws.k.ptr, b.ptr + (qd * 2) as u64, rows, kvd)?;
                    ctx.bias(self.ws.v.ptr, b.ptr + ((qd + kvd) * 2) as u64, rows, kvd)?;
                }
                (self.ws.q.ptr, self.ws.k.ptr, self.ws.v.ptr)
            };
            ctx.rope(
                q_p,
                rows,
                c.n_heads,
                c.head_dim,
                pos0,
                c.rope_theta,
                c.head_dim / 2,
                pos_dev,
                0,
            )?;
            ctx.rope(
                k_p,
                rows,
                c.n_kv_heads,
                c.head_dim,
                pos0,
                c.rope_theta,
                c.head_dim / 2,
                pos_dev,
                0,
            )?;
            ctx.kv_append(
                k_p,
                v_p,
                layer.k_cache.ptr,
                layer.v_cache.ptr,
                rows,
                c.n_kv_heads,
                c.head_dim,
                pos0,
                c.max_seq,
                pos_dev,
            )?;
            if rows == 1 {
                let scale = 1.0 / (c.head_dim as f32).sqrt();
                if c.head_dim % 32 == 0 && c.head_dim <= 128 {
                    let nc = (c.max_seq + ATT_CSZ - 1) / ATT_CSZ;
                    ctx.attn_decode_split(
                        q_p,
                        layer.k_cache.ptr,
                        layer.v_cache.ptr,
                        self.ws.att_part.ptr,
                        c.n_heads,
                        c.n_kv_heads,
                        c.head_dim,
                        pos0 + 1,
                        c.max_seq,
                        ATT_CSZ,
                        nc,
                        scale,
                        0,
                        pos_dev,
                    )?;
                    ctx.attn_reduce(
                        self.ws.att_part.ptr,
                        self.ws.att.ptr,
                        c.n_heads,
                        c.head_dim,
                        ATT_CSZ,
                        nc,
                        pos0 + 1,
                        0,
                        pos_dev,
                    )?;
                } else {
                    // Exotic head_dim: the warp-register flash-decode path
                    // doesn't apply; use the monolithic kernel.
                    ctx.attn_decode(
                        q_p,
                        layer.k_cache.ptr,
                        layer.v_cache.ptr,
                        self.ws.att.ptr,
                        c.n_heads,
                        c.n_kv_heads,
                        c.head_dim,
                        pos0 + 1,
                        c.max_seq,
                        scale,
                        0,
                        pos_dev,
                    )?;
                }
            } else {
                ctx.attn_prefill(
                    q_p,
                    layer.k_cache.ptr,
                    layer.v_cache.ptr,
                    self.ws.att.ptr,
                    rows,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    pos0,
                    c.max_seq,
                    true,
                    1.0 / (c.head_dim as f32).sqrt(),
                    0,
                    0,
                )?;
            }
            if rows == 1 {
                match &layer.w {
                    LayerW::Fused {
                        wo,
                        w_gateup,
                        w_down,
                        ..
                    } => {
                        // Residual epilogue: x += Wo·att (no separate add kernel).
                        ctx.gemv_f16(wo.ptr, self.ws.att.ptr, self.ws.x.ptr, 0, hs, qd, 1)?;
                        ctx.rmsnorm(
                            self.ws.x.ptr,
                            layer.ffn_norm.ptr,
                            self.ws.h.ptr,
                            1,
                            hs,
                            c.rms_eps,
                        )?;
                        // gate|up pair GEMV with silu·mul epilogue, then x += Wd·g.
                        ctx.gemv_f16(
                            w_gateup.ptr,
                            self.ws.h.ptr,
                            self.ws.gate.ptr,
                            0,
                            inter,
                            hs,
                            2,
                        )?;
                        ctx.gemv_f16(w_down.ptr, self.ws.gate.ptr, self.ws.x.ptr, 0, hs, inter, 1)?;
                    }
                    LayerW::Split {
                        wo,
                        w_gate,
                        w_up,
                        w_down,
                        ..
                    } => {
                        lin_gemv(ctx, wo, self.ws.att.ptr, self.ws.x.ptr, 0, 1)?;
                        ctx.rmsnorm(
                            self.ws.x.ptr,
                            layer.ffn_norm.ptr,
                            self.ws.h.ptr,
                            1,
                            hs,
                            c.rms_eps,
                        )?;
                        // Separate gate / up GEMVs (mixed formats forbid the
                        // pair trick), then the same silu·mul elementwise.
                        lin_gemv(ctx, w_gate, self.ws.h.ptr, self.ws.gate.ptr, 0, 0)?;
                        lin_gemv(ctx, w_up, self.ws.h.ptr, self.ws.up.ptr, 0, 0)?;
                        ctx.swiglu(self.ws.gate.ptr, self.ws.up.ptr, inter)?;
                        lin_gemv(ctx, w_down, self.ws.gate.ptr, self.ws.x.ptr, 0, 1)?;
                    }
                }
            } else {
                match &layer.w {
                    LayerW::Fused {
                        wo,
                        w_gateup,
                        w_down,
                        ..
                    } => {
                        ctx.gemm_f16(self.ws.att.ptr, wo.ptr, self.ws.h.ptr, rows, hs, qd)?;
                        ctx.add(self.ws.x.ptr, self.ws.h.ptr, rows * hs)?;
                        ctx.rmsnorm(
                            self.ws.x.ptr,
                            layer.ffn_norm.ptr,
                            self.ws.h.ptr,
                            rows,
                            hs,
                            c.rms_eps,
                        )?;
                        ctx.gemm_f16(
                            self.ws.h.ptr,
                            w_gateup.ptr,
                            self.ws.gate.ptr,
                            rows,
                            inter,
                            hs,
                        )?;
                        ctx.gemm_f16(
                            self.ws.h.ptr,
                            w_gateup.ptr + up_off,
                            self.ws.up.ptr,
                            rows,
                            inter,
                            hs,
                        )?;
                        ctx.swiglu(self.ws.gate.ptr, self.ws.up.ptr, rows * inter)?;
                        ctx.gemm_f16(self.ws.gate.ptr, w_down.ptr, self.ws.h.ptr, rows, hs, inter)?;
                        ctx.add(self.ws.x.ptr, self.ws.h.ptr, rows * hs)?;
                    }
                    LayerW::Split {
                        wo,
                        w_gate,
                        w_up,
                        w_down,
                        ..
                    } => {
                        let dq = self.ws.dq.as_ref();
                        lin_gemm(ctx, wo, dq, self.ws.att.ptr, self.ws.h.ptr, rows)?;
                        ctx.add(self.ws.x.ptr, self.ws.h.ptr, rows * hs)?;
                        ctx.rmsnorm(
                            self.ws.x.ptr,
                            layer.ffn_norm.ptr,
                            self.ws.h.ptr,
                            rows,
                            hs,
                            c.rms_eps,
                        )?;
                        lin_gemm(ctx, w_gate, dq, self.ws.h.ptr, self.ws.gate.ptr, rows)?;
                        lin_gemm(ctx, w_up, dq, self.ws.h.ptr, self.ws.up.ptr, rows)?;
                        ctx.swiglu(self.ws.gate.ptr, self.ws.up.ptr, rows * inter)?;
                        lin_gemm(ctx, w_down, dq, self.ws.gate.ptr, self.ws.h.ptr, rows)?;
                        ctx.add(self.ws.x.ptr, self.ws.h.ptr, rows * hs)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Gather token embeddings for `tokens` into `ws.x` (rows = tokens.len()).
    fn embed_tokens(&self, tokens: &[u32]) -> Res<()> {
        debug_assert!(tokens.len() <= CHUNK);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
        self.ctx.htod(&self.ws.ids, bytes)?;
        self.gather_ids(tokens.len())
    }

    /// Gather embeddings for the ids already resident in `ws.ids` (the
    /// device-token pipeline writes them there without a host hop).
    fn gather_ids(&self, n: usize) -> Res<()> {
        match &self.embed {
            Lin::F16 { buf, .. } => self.ctx.gather(
                buf.ptr,
                self.embed_bf16,
                self.ws.ids.ptr,
                self.ws.x.ptr,
                n,
                self.cfg.hidden_size,
            ),
            Lin::Quant { buf, fmt, .. } => {
                // Rows dequantize on the fly from the packed table — the
                // f16 expansion of a 250k-vocab table never exists.
                self.ctx.gguf_gather(
                    *fmt,
                    buf.ptr,
                    self.ws.ids.ptr,
                    self.ws.x.ptr,
                    n,
                    self.cfg.hidden_size,
                )
            }
        }
    }

    /// Project the normed hidden row in `ws.h` to vocab logits in
    /// `ws.logits_h` through the (possibly tied, possibly packed) head.
    fn head_project(&self) -> Res<()> {
        let c = &self.cfg;
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        match head {
            Lin::F16 { buf, .. } => self.ctx.gemm_f16(
                self.ws.h.ptr,
                buf.ptr,
                self.ws.logits_h.ptr,
                1,
                c.vocab_size,
                c.hidden_size,
            ),
            Lin::Quant { buf, fmt, n, k } => self.ctx.gguf_gemv(
                *fmt,
                self.ws.h.ptr,
                buf.ptr,
                0,
                self.ws.logits_h.ptr,
                *n,
                *k,
                0,
            ),
        }
    }

    /// Greedy fast path: argmax on device, 8-byte copy (no softcap in the
    /// generic family — pass 0.0). See gemma4::project_argmax for rationale.
    fn project_argmax(&self, row: usize) -> Res<u32> {
        let c = &self.cfg;
        let x_row = self.ws.x.ptr + (row * c.hidden_size * 2) as u64;
        self.ctx.rmsnorm(
            x_row,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            c.hidden_size,
            c.rms_eps,
        )?;
        self.head_project()?;
        self.ctx.argmax_softcap(
            self.ws.logits_h.ptr,
            &self.ws.argmax_slot,
            c.vocab_size,
            0.0,
        )
    }

    pub fn prefill_argmax(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<u32> {
        Architecture::prefill(self, prompt, pos0)?;
        // prefill already projected host logits once (cheap relative to the
        // whole prefill); re-project the same row on device for the token.
        self.project_argmax((prompt.tokens.len() + pos0 - 1) % CHUNK.max(1))
    }

    pub fn decode_step_argmax(&mut self, token: u32, pos: usize) -> Res<u32> {
        self.check_capacity(pos + 1)?;
        self.embed_tokens(&[token])?;
        self.forward_chunk(1, pos)?;
        self.pos = pos + 1;
        self.project_argmax(0)
    }

    /// Pipelined greedy step: the input token is already in `ws.ids[0]` on
    /// device (written by the previous step's argmax extract), so the whole
    /// step enqueues without any host transfer. The caller overlaps the
    /// previous token's 8-byte fetch with this step's compute.
    pub fn decode_step_device(&mut self, pos: usize) -> Res<()> {
        self.check_capacity(pos + 1)?;
        if let Some(g) = &self.decode_graph {
            self.pos = pos + 1;
            return self.ctx.graph_launch(g);
        }
        self.enqueue_decode_device()?;
        self.pos = pos + 1;
        Ok(())
    }

    /// One full device-token decode step: gather → forward (positions from
    /// the device counter) → head → argmax → ids feedback → counter bump.
    fn enqueue_decode_device(&mut self) -> Res<()> {
        self.gather_ids(1)?;
        self.forward_chunk_at(1, 0, self.ws.pos_dev.ptr)?;
        let c = &self.cfg;
        self.ctx.rmsnorm(
            self.ws.x.ptr,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            c.hidden_size,
            c.rms_eps,
        )?;
        self.head_project()?;
        self.ctx.argmax_softcap_enqueue(
            self.ws.logits_h.ptr,
            &self.ws.argmax_slot,
            c.vocab_size,
            0.0,
        )?;
        self.ctx.argmax_to_ids(&self.ws.argmax_slot, &self.ws.ids)?;
        self.ctx.pos_bump(&self.ws.pos_dev)
    }

    /// Initialize the device counter and capture the decode step into a
    /// replayable graph (one launch/token afterwards). Capture failures
    /// (e.g. cuBLAS versions that resist stream capture) degrade to the
    /// per-launch path with a warning.
    pub fn arm_decode_graph(&mut self, pos: usize) -> Res<()> {
        let bytes = (pos as u32).to_le_bytes();
        self.ctx.htod(&self.ws.pos_dev, &bytes)?;
        if self.decode_graph.is_some()
            || std::env::var("CIMA_NO_GRAPH")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            return Ok(());
        }
        self.ctx.sync()?;
        self.ctx.capture_begin()?;
        let enq = self.enqueue_decode_device();
        match enq.and_then(|_| self.ctx.capture_end()) {
            Ok(g) => {
                self.decode_graph = Some(g);
                log::info("decode step captured as a CUDA graph (1 launch/token)");
            }
            Err(e) => {
                log::warn(&format!(
                    "CUDA graph capture failed ({}); using per-launch decode",
                    e
                ));
                self.ctx.sync().ok();
            }
        }
        Ok(())
    }

    /// Decode tail for the SAMPLING path: forward + head as in the greedy
    /// graph, then repeat-penalty from the device window counts and the
    /// top-`SAMPLE_TOPK` extraction (descending, masked argmax rounds).
    fn enqueue_sample_tail(&mut self, rp: f32) -> Res<()> {
        self.gather_ids(1)?;
        self.forward_chunk_at(1, 0, self.ws.pos_dev.ptr)?;
        let c = &self.cfg;
        self.ctx.rmsnorm(
            self.ws.x.ptr,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            c.hidden_size,
            c.rms_eps,
        )?;
        self.head_project()?;
        if rp != 1.0 {
            self.ctx.apply_penalty(
                self.ws.logits_h.ptr,
                self.ws.hist_counts.ptr,
                rp,
                c.vocab_size,
            )?;
        }
        self.ctx.topk_enqueue(
            self.ws.logits_h.ptr,
            &self.ws.argmax_slot,
            self.ws.cand.ptr,
            c.vocab_size,
            0.0,
            SAMPLE_TOPK,
        )?;
        self.ctx.pos_bump(&self.ws.pos_dev)
    }

    /// Capture the sampling decode step (re-captures when `rp` changes,
    /// since the penalty value is a baked kernel argument).
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
        let enq = self.enqueue_sample_tail(rp);
        match enq.and_then(|_| self.ctx.capture_end()) {
            Ok(g) => {
                self.sample_graph = Some((g, rp));
                log::info("sampling decode step captured as a CUDA graph (device top-k)");
            }
            Err(e) => {
                log::warn(&format!(
                    "sampling graph capture failed ({}); using full-logits decode",
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

    /// One sampling decode step through the captured graph. The input
    /// token is pushed into the device penalty window BEFORE the launch
    /// (slot `pos & 63`), so the penalty sees exactly the host sampler's
    /// last-64 window. Returns the packed top candidates, descending.
    pub fn decode_step_sample(&mut self, token: u32, pos: usize) -> Res<Vec<u64>> {
        self.check_capacity(pos + 1)?;
        self.ctx.htod(&self.ws.ids, &token.to_le_bytes())?;
        self.ctx.hist_push(
            self.ws.hist_ring.ptr,
            self.ws.hist_counts.ptr,
            self.ws.ids.ptr,
            0,
            self.ws.pos_dev.ptr,
        )?;
        let g = &self.sample_graph.as_ref().expect("armed").0;
        self.ctx.graph_launch(g)?;
        self.pos = pos + 1;
        let mut raw = vec![0u8; SAMPLE_TOPK * 8];
        self.ctx.dtoh(&mut raw, &self.ws.cand)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Reset the device penalty window and seed it with the prompt tail
    /// (absolute positions, so eviction stays aligned with the ring).
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
                p as i32 as usize,
                0,
            )?;
        }
        Ok(())
    }

    /// Seed the device-resident pipeline: run argmax over the current
    /// logits buffer and leave the winner in `ws.ids[0]`.
    pub fn decode_graph_active(&self) -> bool {
        self.decode_graph.is_some()
    }

    pub fn argmax_slot(&self) -> &crate::cuda::DeviceBuf {
        &self.ws.argmax_slot
    }

    pub fn seed_device_token(&self) -> Res<()> {
        self.ctx.argmax_to_ids(&self.ws.argmax_slot, &self.ws.ids)
    }

    /// Project the hidden state at `row` to vocabulary logits (host f32).
    fn project_logits(&self, row: usize) -> Res<Vec<f32>> {
        let c = &self.cfg;
        let x_row = self.ws.x.ptr + (row * c.hidden_size * 2) as u64;
        // Final norm in place into ws.h row 0.
        self.ctx.rmsnorm(
            x_row,
            self.final_norm.ptr,
            self.ws.h.ptr,
            1,
            c.hidden_size,
            c.rms_eps,
        )?;
        self.head_project()?;
        self.ctx
            .h2f(self.ws.logits_h.ptr, self.ws.logits_f.ptr, c.vocab_size)?;
        let mut out = vec![0f32; c.vocab_size];
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, c.vocab_size * 4)
        };
        self.ctx.dtoh(bytes, &self.ws.logits_f)?;
        Ok(out)
    }

    /// Bounds check against the KV allocation.
    fn check_capacity(&self, want: usize) -> Res<()> {
        if want > self.cfg.max_seq {
            return Err(err!(
                "context",
                "sequence of {} tokens exceeds the model's KV capacity of {} — \
                 truncate the prompt or lower max_tokens",
                want,
                self.cfg.max_seq
            ));
        }
        Ok(())
    }
}

impl Architecture for Transformer {
    fn modality(&self) -> Modality {
        self.cfg.modality()
    }
    fn weight_bytes_resident(&self) -> usize {
        self.weight_bytes
    }
    fn perf_levers(&self) -> crate::traits::PerfLevers {
        crate::traits::PerfLevers {
            device_greedy: true,
            cuda_graph: true, // armed on the first greedy generation
            fused_weights: true,
            // Warp-register flash-decode applies to the head_dims it
            // supports; exotic shapes fall back to the monolithic kernel.
            seq_parallel_attention: self.cfg.head_dim % 32 == 0 && self.cfg.head_dim <= 128,
            device_pipeline: true,
        }
    }
    fn max_seq(&self) -> usize {
        self.cfg.max_seq
    }

    fn prefill(&mut self, prompt: &PreparedPrompt, pos0: usize) -> Res<Vec<f32>> {
        let n = prompt.tokens.len();
        if n == 0 {
            return Err(err!("generate", "empty prompt after tokenization"));
        }
        self.check_capacity(pos0 + n)?;
        let mut done = 0;
        while done < n {
            let take = (n - done).min(CHUNK);
            self.embed_tokens(&prompt.tokens[done..done + take])?;
            // Splice media embeddings overlapping this chunk (dtod copy onto
            // the placeholder rows — the encoder already projected to f16
            // rows of `hidden_size`).
            for (at, buf, rows) in &prompt.media_embeds {
                let (start, end) = (*at, at + rows);
                let (c0, c1) = (done, done + take);
                if end > c0 && start < c1 {
                    let s = start.max(c0);
                    let e = end.min(c1);
                    let src = buf.ptr + ((s - start) * self.cfg.hidden_size * 2) as u64;
                    let dst = self.ws.x.ptr + ((s - c0) * self.cfg.hidden_size * 2) as u64;
                    self.ctx
                        .dtod(dst, src, (e - s) * self.cfg.hidden_size * 2)?;
                }
            }
            self.forward_chunk(take, pos0 + done)?;
            done += take;
            if done < n {
                self.ctx.sync()?; // chunk boundary: keep the queue shallow
            }
        }
        self.pos = pos0 + n;
        let logits = self.project_logits((n - 1) % CHUNK.max(1))?;
        // ^ last row of the final chunk
        Ok(logits)
    }

    fn decode_step(&mut self, token: u32, pos: usize) -> Res<Vec<f32>> {
        self.check_capacity(pos + 1)?;
        self.embed_tokens(&[token])?;
        self.forward_chunk(1, pos)?;
        self.pos = pos + 1;
        self.project_logits(0)
    }

    fn embed(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        self.reset()?;
        let n = tokens.len().min(self.cfg.max_seq).min(CHUNK);
        self.embed_tokens(&tokens[..n])?;
        self.forward_chunk(n, 0)?;
        // Final norm over all rows, then mean-pool to a single f32 vector.
        self.ctx.rmsnorm(
            self.ws.x.ptr,
            self.final_norm.ptr,
            self.ws.h.ptr,
            n,
            self.cfg.hidden_size,
            self.cfg.rms_eps,
        )?;
        self.ctx
            .meanpool(self.ws.h.ptr, self.ws.logits_f.ptr, n, self.cfg.hidden_size)?;
        let mut out = vec![0f32; self.cfg.hidden_size];
        let view =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        // Reuse logits_f (vocab*4 >= hidden*4 in practice; guard anyway).
        if self.cfg.hidden_size * 4 > self.ws.logits_f.bytes {
            return Err(err!(
                "embed",
                "hidden_size {} exceeds pooling buffer",
                self.cfg.hidden_size
            ));
        }
        let tmp = DeviceBuf {
            ptr: self.ws.logits_f.ptr,
            bytes: self.cfg.hidden_size * 4,
        };
        self.ctx.dtoh(view, &tmp)?;
        std::mem::forget(tmp); // borrowed view of ws.logits_f — must not free
        self.pos = 0;
        Ok(out)
    }

    fn reset(&mut self) -> Res<()> {
        // Deterministic eviction of conversational state: the cache contents
        // beyond `pos` are never read, so resetting the cursor is sufficient
        // and O(1).
        self.pos = 0;
        Ok(())
    }

    fn vram_bytes(&self) -> usize {
        self.vram
    }
}

// ===========================================================================
// 5. GenStats — per-request generation telemetry
// ===========================================================================

/// Per-request generation statistics (also emitted as `METRIC` log lines).
#[derive(Debug, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub ttft_ms: f64,
    pub tok_per_s: f64,
    pub total_ms: f64,
    /// Ollama `done_reason`: "stop" (EOS or a stop string matched) or
    /// "length" (hit the token budget / context limit).
    pub stop_reason: &'static str,
    /// The full generated text AFTER stop-sequence trimming. The streaming
    /// callback emits pieces as they are produced (a stop string may already
    /// have gone over the wire), but this field is authoritative for the
    /// non-streaming response body — it reflects the exclusive-stop trim.
    pub text: String,
}