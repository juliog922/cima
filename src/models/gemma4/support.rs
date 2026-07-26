//! # support — gemma4 configuration + weight resolution
//!
//! * [`config`] — the exhaustive `G4Config` parse of gemma4's config.json
//!   (per-type RoPE, KV sharing, PLE, tower geometries).
//! * [`weights`] — `G4Index`: prefix-tolerant tensor lookup, small-tensor
//!   host reads, and the shared `decode_f32` byte decoder.

pub(crate) mod config {
    //! Checkpoint configuration: parsing, defaults, and topology validation
    //! for the three sub-configs (text / vision / audio).

    use crate::err;
    use crate::json::Json;

    use crate::traits::Res;

    // ===========================================================================
    // Configuration
    // ===========================================================================

    /// Attention flavour of a text layer.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum LayerType {
        Sliding,
        Full,
    }

    /// Parsed and validated `text_config`.
    pub struct G4TextCfg {
        pub hidden: usize,
        pub n_layers: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        /// `num_global_key_value_heads` — full-attention layers carry their own
        /// GQA ratio (31B: 16 kv heads sliding, 4 full). Defaults to
        /// `n_kv_heads` when the field is absent (E2B/E4B).
        pub n_global_kv_heads: usize,
        pub head_dim: usize,        // sliding layers
        pub global_head_dim: usize, // full layers
        /// `attention_k_eq_v`: on full-attention layers the checkpoint ships no
        /// `v_proj`. K and V share one projection, then diverge — K takes the
        /// learned per-head norm plus rotary, V takes the weightless norm and
        /// no rotary. Sliding layers are unaffected.
        pub k_eq_v: bool,
        pub inter: usize,
        pub vocab: usize,
        pub rms_eps: f32,
        pub max_seq: usize,
        /// `text_config.use_bidirectional_attention == "vision"`: only the larger
        /// Gemma 4 models attend bidirectionally inside image spans; the smaller
        /// ones (E2B/E4B) use a conventional causal mask over soft tokens — the
        /// reference gates the block mask on this exact field.
        pub bidir_vision: bool,
        pub sliding_window: usize,
        pub layer_types: Vec<LayerType>,
        pub theta_sliding: f32,
        pub theta_full: f32,
        /// Nonzero rotary frequency pairs on full layers (proportional RoPE).
        pub full_nfreqs: usize,
        /// Apply the checkpoint's `rope_freqs.weight` frequency factors on
        /// full-attention layers. False for config.json-driven (safetensors)
        /// loads; the gguf loader turns it on because the export's metadata
        /// declares the llama.cpp recipe (full-width rotation + factors) —
        /// empirically verified as the trained behavior on the E4B export.
        pub use_rope_factors: bool,
        pub n_kv_shared: usize,
        pub double_wide_mlp: bool,
        pub softcap: f32,
        pub ple_dim: usize,   // hidden_size_per_layer_input (0 = disabled)
        pub ple_vocab: usize, // vocab_size_per_layer_input
        pub eos: Vec<u32>,
    }

    /// Parsed `vision_config`.
    #[derive(Clone)]
    pub struct G4VisionCfg {
        pub hidden: usize,
        pub n_layers: usize,
        pub n_heads: usize,
        pub head_dim: usize,
        pub inter: usize,
        pub patch: usize,
        pub pos_table: usize, // position_embedding_size
        pub pool_k: usize,
        pub rms_eps: f32,
        pub theta: f32,
    }

    /// Parsed `audio_config`.
    pub struct G4AudioCfg {
        pub hidden: usize,
        pub n_layers: usize,
        pub n_heads: usize,
        pub chunk: usize, // attention_chunk_size
        pub past: usize,  // attention_context_left - 1
        pub logit_cap: f32,
        /// `attention_invalid_logits_value` — masked-slot logit (dilution).
        pub invalid_logit: f32,
        pub conv_k: usize,      // light-conv kernel
        pub sub_ch: [usize; 2], // subsampling_conv_channels
        pub out_dims: usize,    // output_proj_dims
        pub residual_w: f32,
        pub rms_eps: f32,
        pub mels: usize,
    }

    /// Top-level Gemma4 configuration (text + optional towers + media token ids).
    pub struct G4Config {
        pub text: G4TextCfg,
        pub vision: Option<G4VisionCfg>,
        pub audio: Option<G4AudioCfg>,
        pub image_token_id: u32,
        pub audio_token_id: u32,
        /// PLE lookups for multimodal placeholder positions use the PAD token,
        /// not the placeholder id — the reference rewrites
        /// `llm_input_ids = where(multimodal, pad_token_id, input_ids)` before
        /// `get_per_layer_inputs`.
        pub pad_token_id: u32,
        pub boi_token_id: u32,
        pub eoi_token_id: u32,
        pub boa_token_id: u32,
        pub eoa_token_id: u32,
    }

    pub(in crate::models::gemma4) fn ru(o: &Json, k: &str) -> Res<usize> {
        o.usize_of(k).ok_or_else(|| {
            err!(
                "config",
                "gemma4 config: '{}' missing or not a positive integer",
                k
            )
        })
    }

    impl G4Config {
        /// Parse + exhaustively validate the raw `config.json` of a `gemma4`
        /// checkpoint. Every assumption baked into this implementation is checked
        /// here so a future config revision fails loudly instead of silently.
        pub fn parse(cfg: &Json) -> Res<G4Config> {
            let t = cfg
                .get("text_config")
                .filter(|t| t.as_obj().is_some())
                .ok_or_else(|| {
                    err!("config", "gemma4 config.json: 'text_config' object missing")
                })?;

            // ---- features this implementation deliberately rejects -----------
            if t.bool_of("enable_moe_block").unwrap_or(false) {
                return Err(err!(
                    "config",
                    "gemma4: enable_moe_block=true (MoE) is not implemented in this engine"
                ));
            }
            // attention_k_eq_v is implemented (see the header of this patch and
            // the k_eq_v field docs); parsed below rather than rejected.
            if let Some(b) = t.str_of("use_bidirectional_attention") {
                if b == "all" {
                    return Err(err!(
                        "config",
                        "gemma4: use_bidirectional_attention='all' is not a causal LM"
                    ));
                }
            }

            let n_layers = ru(t, "num_hidden_layers")?;
            let lt_raw = t
                .arr_of("layer_types")
                .ok_or_else(|| err!("config", "gemma4 text_config: 'layer_types' array missing"))?;
            if lt_raw.len() != n_layers {
                return Err(err!(
                    "config",
                    "gemma4: layer_types has {} entries but num_hidden_layers={}",
                    lt_raw.len(),
                    n_layers
                ));
            }
            let mut layer_types = Vec::with_capacity(n_layers);
            for (i, v) in lt_raw.iter().enumerate() {
                match v.as_str() {
                    Some("sliding_attention") => layer_types.push(LayerType::Sliding),
                    Some("full_attention") => layer_types.push(LayerType::Full),
                    other => {
                        return Err(err!(
                            "config",
                            "gemma4: layer_types[{}] = {:?} is not a known attention type",
                            i,
                            other
                        ))
                    }
                }
            }

            // ---- per-type RoPE parameters -------------------------------------
            // `rope_parameters` is OPTIONAL: quantizer pipelines strip config
            // fields (the unsloth trap), so absence falls back to the reference
            // defaults — sliding: default rope, θ=10 000; full: proportional
            // partial-rotary 0.25, θ=1 000 000. Erroring here bricks otherwise
            // valid checkpoints.
            static EMPTY: Json = Json::Null;
            let rp = t.get("rope_parameters").unwrap_or(&EMPTY);
            let sl = rp.get("sliding_attention").unwrap_or(&EMPTY);
            let fu = rp.get("full_attention").unwrap_or(&EMPTY);
            // Reference defaults differ per type: sliding layers default to the
            // standard rope, full layers to PROPORTIONAL with partial factor
            // 0.25 — defaulting full to "default"/1.0 would silently rotate 4×
            // too many frequencies on stripped configs.
            let sl_type = sl.str_of("rope_type").unwrap_or("default");
            let fu_type = fu.str_of("rope_type").unwrap_or("proportional");
            if sl_type != "default" {
                return Err(err!(
                    "config",
                    "gemma4: sliding-layer rope_type '{}' is not implemented (expected 'default')",
                    sl_type
                ));
            }
            if fu_type != "default" && fu_type != "proportional" {
                return Err(err!("config", "gemma4: full-layer rope_type '{}' is not implemented (expected 'default' or 'proportional')", fu_type));
            }
            if (fu.f64_of("factor").unwrap_or(1.0) - 1.0).abs() > 1e-9 {
                return Err(err!(
                    "config",
                    "gemma4: rope_parameters.full_attention.factor != 1.0 is not implemented"
                ));
            }

            let head_dim = ru(t, "head_dim")?;
            let global_head_dim = t.usize_or("global_head_dim", head_dim);
            let partial = fu.f32_or(
                "partial_rotary_factor",
                if fu_type == "proportional" { 0.25 } else { 1.0 },
            );
            // proportional RoPE: nfreqs nonzero frequencies over the FULL head_dim
            // exponent, identity rotation on the remaining rotate_half pairs.
            //
            // CIMA_G4_FULL_ROTARY=1 overrides to full rotation on the
            // full-attention layers — the recipe the GGUF metadata declares
            // (`gemma4.rope.dimension_count == key_length`) and llama.cpp
            // executes. Discriminating experiment for exports whose rope
            // recipe disagrees with the config.json sidecar.
            let full_nfreqs = if super::env_flag("CIMA_G4_FULL_ROTARY") == Some(true) {
                eprintln!(
                    "g4 rope: FULL rotation forced on full-attention layers (nfreqs {} → {})",
                    ((partial as f64) * (global_head_dim as f64) / 2.0) as usize,
                    global_head_dim / 2
                );
                global_head_dim / 2
            } else {
                ((partial as f64) * (global_head_dim as f64) / 2.0) as usize
            };
            if full_nfreqs == 0 || full_nfreqs > global_head_dim / 2 {
                return Err(err!(
                    "config",
                    "gemma4: partial_rotary_factor {} yields invalid rotary width {}",
                    partial,
                    full_nfreqs
                ));
            }

            let n_heads = ru(t, "num_attention_heads")?;
            let n_kv_heads = t.usize_or("num_key_value_heads", n_heads);
            if n_heads % n_kv_heads != 0 {
                return Err(err!(
                    "config",
                    "gemma4: num_attention_heads {} not divisible by num_key_value_heads {}",
                    n_heads,
                    n_kv_heads
                ));
            }
            let n_global_kv_heads = t.usize_or("num_global_key_value_heads", n_kv_heads);
            if n_global_kv_heads == 0 || n_heads % n_global_kv_heads != 0 {
                return Err(err!(
                    "config",
                    "gemma4: num_attention_heads {} not divisible by \
                     num_global_key_value_heads {}",
                    n_heads,
                    n_global_kv_heads
                ));
            }
            let k_eq_v = t.bool_of("attention_k_eq_v").unwrap_or(false);
            let n_kv_shared = t.usize_or("num_kv_shared_layers", 0);
            if n_kv_shared >= n_layers {
                return Err(err!(
                    "config",
                    "gemma4: num_kv_shared_layers {} >= num_hidden_layers {}",
                    n_kv_shared,
                    n_layers
                ));
            }
            // The sharing boundary must leave at least one computing layer of each
            // type present in the shared region (each shared layer needs a source).
            let first_shared = n_layers - n_kv_shared;
            for ty in [LayerType::Sliding, LayerType::Full] {
                let needed = layer_types[first_shared..].contains(&ty);
                let present = layer_types[..first_shared].contains(&ty);
                if needed && !present {
                    return Err(err!("config", "gemma4: kv-shared {:?} layers exist but no earlier layer of that type computes K/V", ty));
                }
            }

            let mut eos: Vec<u32> = Vec::new();
            match t.get("eos_token_id") {
                Some(Json::Num(n)) => eos.push(*n as u32),
                Some(a) => {
                    if let Some(arr) = a.as_arr() {
                        for v in arr {
                            if let Some(id) = v.as_u64() {
                                eos.push(id as u32);
                            }
                        }
                    }
                }
                None => {}
            }
            if let Some(arr) = cfg.arr_of("eos_token_id") {
                for v in arr {
                    if let Some(id) = v.as_u64() {
                        if !eos.contains(&(id as u32)) {
                            eos.push(id as u32);
                        }
                    }
                }
            }

            let text = G4TextCfg {
                hidden: ru(t, "hidden_size")?,
                n_layers,
                n_heads,
                n_kv_heads,
                n_global_kv_heads,
                head_dim,
                global_head_dim,
                k_eq_v,
                inter: ru(t, "intermediate_size")?,
                vocab: ru(t, "vocab_size")?,
                rms_eps: t.f32_or("rms_norm_eps", 1e-6),
                max_seq: ru(t, "max_position_embeddings")?.min(8192),
                bidir_vision: t
                    .get("use_bidirectional_attention")
                    .map(|v| v.as_str() == Some("vision") || v.as_bool() == Some(true))
                    .unwrap_or(false),
                sliding_window: ru(t, "sliding_window")?,
                layer_types,
                theta_sliding: sl.f32_or("rope_theta", 10_000.0),
                theta_full: fu.f32_or("rope_theta", 1_000_000.0),
                full_nfreqs,
                use_rope_factors: false,
                n_kv_shared,
                double_wide_mlp: t.bool_of("use_double_wide_mlp").unwrap_or(false),
                softcap: t.f32_or("final_logit_softcapping", 0.0),
                ple_dim: t.usize_or("hidden_size_per_layer_input", 0),
                ple_vocab: t.usize_or("vocab_size_per_layer_input", 0),
                eos,
            };
            if text.ple_dim > 0 && text.ple_vocab == 0 {
                return Err(err!("config", "gemma4: hidden_size_per_layer_input set but vocab_size_per_layer_input missing"));
            }

            let vision = match cfg.get("vision_config").filter(|v| v.as_obj().is_some()) {
                Some(v) => {
                    let hidden = ru(v, "hidden_size")?;
                    let n_heads = ru(v, "num_attention_heads")?;
                    let kv = v.usize_or("num_key_value_heads", n_heads);
                    if kv != n_heads {
                        return Err(err!("config", "gemma4 vision: GQA (kv_heads {} != heads {}) is not implemented in the vision tower", kv, n_heads));
                    }
                    Some(G4VisionCfg {
                        hidden,
                        n_layers: ru(v, "num_hidden_layers")?,
                        n_heads,
                        head_dim: v.usize_or("head_dim", hidden / n_heads),
                        inter: ru(v, "intermediate_size")?,
                        patch: ru(v, "patch_size")?,
                        pos_table: ru(v, "position_embedding_size")?,
                        pool_k: ru(v, "pooling_kernel_size")?,
                        rms_eps: v.f32_or("rms_norm_eps", 1e-6),
                        theta: v
                            .get("rope_parameters")
                            .map(|r| r.f32_or("rope_theta", 100.0))
                            .unwrap_or(100.0),
                    })
                }
                None => None,
            };

            let audio = match cfg.get("audio_config").filter(|a| a.as_obj().is_some()) {
                Some(a) => {
                    let right = a.usize_or("attention_context_right", 0);
                    if right != 0 {
                        return Err(err!("config", "gemma4 audio: attention_context_right={} (future context) is not implemented", right));
                    }
                    let ch = a
                        .arr_of("subsampling_conv_channels")
                        .and_then(|arr| {
                            let v: Vec<usize> = arr.iter().filter_map(Json::as_usize).collect();
                            if v.len() == 2 {
                                Some([v[0], v[1]])
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| {
                            err!(
                                "config",
                                "gemma4 audio: subsampling_conv_channels must be a 2-array"
                            )
                        })?;
                    Some(G4AudioCfg {
                        hidden: ru(a, "hidden_size")?,
                        n_layers: ru(a, "num_hidden_layers")?,
                        n_heads: ru(a, "num_attention_heads")?,
                        chunk: ru(a, "attention_chunk_size")?,
                        past: ru(a, "attention_context_left")? - 1,
                        logit_cap: a.f32_or("attention_logit_cap", 50.0),
                        invalid_logit: a.f32_or("attention_invalid_logits_value", 1e-9),
                        conv_k: ru(a, "conv_kernel_size")?,
                        sub_ch: ch,
                        out_dims: a.usize_or("output_proj_dims", ru(a, "hidden_size")?),
                        residual_w: a.f32_or("residual_weight", 0.5),
                        rms_eps: a.f32_or("rms_norm_eps", 1e-6),
                        mels: 128,
                    })
                }
                None => None,
            };

            let id = |k: &str| cfg.u64_of(k).map(|v| v as u32).unwrap_or(0);
            Ok(G4Config {
                text,
                vision,
                audio,
                image_token_id: id("image_token_id"),
                audio_token_id: id("audio_token_id"),
                pad_token_id: cfg
                    .u64_of("pad_token_id")
                    .or_else(|| {
                        cfg.get("text_config")
                            .and_then(|tc| tc.u64_of("pad_token_id"))
                    })
                    .unwrap_or(0) as u32,
                boi_token_id: id("boi_token_id"),
                eoi_token_id: id("eoi_token_id"),
                boa_token_id: id("boa_token_id"),
                eoa_token_id: id("eoa_token_id"),
            })
        }
    }
}

/// Value-aware boolean env flag: `X=1/true/yes` → Some(true),
/// `X=0/false/no` → Some(false), unset → None. The old `is_ok()` pattern
/// treated `CIMA_G4_FULL_ROTARY=0` as ON — an experiment ordered off that
/// silently ran on.
pub(crate) fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" | "" => Some(false),
            _ => Some(true),
        },
        Err(_) => None,
    }
}

pub(crate) mod weights {
    //! Weight access: the shard index, clipped linears (NF4 or 16-bit), and
    //! half/bfloat16 conversion primitives.

    use crate::cuda::{CudaCtx, DeviceBuf};
    use crate::err;
    use crate::num::{bf16_to_f32, f16_to_f32};

    use crate::traits::{DType, LoadedWeights, Res, TensorMeta, WeightCodec};

    use crate::quant::bnb::{self, WTensor};

    // ===========================================================================
    // Weight resolution helpers
    // ===========================================================================

    /// Tensor-name resolver for the `Gemma4ForConditionalGeneration` layout:
    /// `model.language_model.*`, `model.vision_tower.*`, `model.audio_tower.*`,
    /// `model.embed_vision.*`, `model.embed_audio.*` (and the same without the
    /// leading `model.` for sub-model exports).
    pub(in crate::models::gemma4) struct G4Index<'a> {
        ctx: &'a CudaCtx,
        pub(in crate::models::gemma4) weights: &'a dyn LoadedWeights,
        codec: &'a dyn WeightCodec,
        pub(in crate::models::gemma4) roots: [&'static str; 2],
    }

    impl<'a> G4Index<'a> {
        pub(in crate::models::gemma4) fn new(
            ctx: &'a CudaCtx,
            weights: &'a dyn LoadedWeights,
            codec: &'a dyn WeightCodec,
        ) -> Self {
            G4Index {
                ctx,
                weights,
                codec,
                roots: ["model.", ""],
            }
        }

        pub(in crate::models::gemma4) fn meta(&self, name: &str) -> Res<&'a TensorMeta> {
            for r in &self.roots {
                if let Some(m) = self.weights.tensors().get(&format!("{}{}", r, name)) {
                    return Ok(m);
                }
            }
            Err(err!(
                "weights",
                "gemma4: required tensor '{}' not found (tried prefixes {:?}); checkpoint has {} tensors — \
                 incomplete repository or a layout revision this engine does not know",
                name, self.roots, self.weights.tensors().len()
            ))
        }

        pub(in crate::models::gemma4) fn exists(&self, name: &str) -> bool {
            self.roots.iter().any(|r| {
                self.weights
                    .tensors()
                    .contains_key(&format!("{}{}", r, name))
            })
        }

        pub(in crate::models::gemma4) fn upload(
            &self,
            name: &str,
            expect: &[usize],
        ) -> Res<DeviceBuf> {
            let meta = self.meta(name)?;
            if bnb::state_name(self.weights.tensors(), &meta.name).is_some() {
                return Err(err!(
                    "quant",
                    "tensor '{}' is bitsandbytes-quantized but this slot (embedding / norm / table) \
                     must be 16-bit — the checkpoint quantizes a module this engine keeps dense",
                    meta.name
                ));
            }
            if !expect.is_empty() && !shape_view_ok(&meta.shape, expect) {
                return Err(err!(
                    "weights",
                    "gemma4: tensor '{}' has shape {:?} but the architecture requires {:?} — \
                     config.json and the checkpoint disagree",
                    meta.name,
                    meta.shape,
                    expect
                ));
            }
            // Norm-gamma convention instrumentation. Gemma's HF checkpoints
            // historically store zero-centered RMSNorm gammas (forward is
            // `x_norm·(1+w)`), while llama.cpp's converter bakes the +1 into
            // the serialized gguf and multiplies plainly — a silent +1
            // disagreement between weight sources degrades quality without
            // breaking fluency. This engine multiplies plainly, so the loaded
            // gamma must already be the direct multiplier.
            //   CIMA_G4_DEBUG=1          → print gamma stats (mean/min/max/%neg):
            //                              zero-centered ⇒ mean≈0, ~half negative;
            //                              direct ⇒ mean≈1, almost none negative.
            //   CIMA_G4_NORM_SHIFT=±1.0  → add the shift to every LM norm gamma
            //                              at load (the A/B experiment: if −1
            //                              or +1 fixes generation, the source's
            //                              convention differs from the pipeline's
            //                              and the loader owns the translation).
            if name.ends_with("norm.weight") {
                let dbg = std::env::var("CIMA_G4_DEBUG").is_ok();
                let shift: f32 = std::env::var("CIMA_G4_NORM_SHIFT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                if dbg || shift != 0.0 {
                    let mut v = decode_f32(meta.dtype, self.weights.bytes(meta)?, &meta.name)?;
                    if dbg && (name.contains("layers.0.") || !name.contains("layers.")) {
                        let n = v.len() as f32;
                        let mean = v.iter().sum::<f32>() / n;
                        let (mut mn, mut mx, mut neg) = (f32::MAX, f32::MIN, 0usize);
                        for &x in &v {
                            mn = mn.min(x);
                            mx = mx.max(x);
                            neg += (x < 0.0) as usize;
                        }
                        eprintln!(
                            "norm gamma '{}': mean {:+.4} min {:+.4} max {:+.4} neg {:.1}%{}",
                            name,
                            mean,
                            mn,
                            mx,
                            100.0 * neg as f32 / n,
                            if shift != 0.0 {
                                format!("  (applying shift {:+})", shift)
                            } else {
                                String::new()
                            }
                        );
                    }
                    for x in &mut v {
                        *x += shift;
                    }
                    let out: Vec<u16> = v.iter().map(|&x| crate::num::f32_to_f16(x)).collect();
                    let buf = self.ctx.alloc(out.len() * 2)?;
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 2)
                    };
                    self.ctx.htod(&buf, bytes)?;
                    return Ok(buf);
                }
            }
            if crate::traits::is_gguf_block(meta.dtype) {
                // Dense slot fed by a gguf checkpoint (e.g. the Q4_K embedding
                // table): host-dequant to f16 and upload dense.
                let numel: usize = meta.shape.iter().product();
                let host = self.weights.bytes(meta)?;
                let mut out = vec![0u16; numel];
                crate::quant::gguf::dequant_host(meta.dtype, host, numel, &mut out)?;
                let buf = self.ctx.alloc(numel * 2)?;
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 2) };
                self.ctx.htod(&buf, bytes)?;
                return Ok(buf);
            }
            if !self.codec.accepts(meta.dtype) {
                return Err(err!(
                    "weights",
                    "gemma4: tensor '{}' stored as {} which codec '{}' cannot execute",
                    meta.name,
                    meta.dtype.name(),
                    self.codec.name()
                ));
            }
            let host = self.weights.bytes(meta)?;
            self.codec.upload(self.ctx, meta, host)
        }

        /// Upload a *linear* weight `[n_out, n_in]`, landing as f16 or as packed
        /// NF4/FP4 when the checkpoint carries a bitsandbytes quant-state family
        /// for it (unsloth "dynamic" checkpoints mix both freely).
        pub(in crate::models::gemma4) fn upload_w(
            &self,
            name: &str,
            n_out: usize,
            n_in: usize,
        ) -> Res<WTensor> {
            let meta = self.meta(name)?;
            if crate::traits::is_gguf_block(meta.dtype) {
                // GGUF checkpoint: the packed blocks ARE the device weight.
                if !shape_view_ok(&meta.shape, &[n_out, n_in]) {
                    return Err(err!(
                        "quant",
                        "gguf tensor '{}' is {:?} but the architecture requires [{} × {}]",
                        meta.name,
                        meta.shape,
                        n_out,
                        n_in
                    ));
                }
                let host = self.weights.bytes(meta)?;
                let buf = self.ctx.alloc(host.len())?;
                self.ctx.htod(&buf, host)?;
                // Pre-grow the q8 activation scratch NOW (load time): the first
                // decode GEMV may run inside a CUDA-graph capture, where the
                // lazy allocation would be illegal (CUresult 900).
                self.ctx.ensure_q8_scratch(n_in)?;
                return Ok(WTensor::Gguf {
                    buf,
                    fmt: meta.dtype,
                    n: n_out,
                    k: n_in,
                });
            }
            if bnb::state_name(self.weights.tensors(), &meta.name).is_some() {
                let q = bnb::upload_nf4(self.ctx, self.weights, meta, &meta.name)?;
                if q.rows != n_out || q.cols != n_in {
                    return Err(err!(
                        "quant",
                        "quantized tensor '{}' is [{} × {}] but the architecture requires [{} × {}]",
                        meta.name, q.rows, q.cols, n_out, n_in
                    ));
                }
                return Ok(WTensor::Nf4(q));
            }
            Ok(WTensor::F16(self.upload(name, &[n_out, n_in])?))
        }

        /// `Gemma4ClippableLinear` wraps the weight under `.linear.` and stores
        /// four scalar clip-bound buffers next to it. Resolves both layouts.
        pub(in crate::models::gemma4) fn upload_clip(
            &self,
            base: &str,
            n_out: usize,
            n_in: usize,
        ) -> Res<ClipLin> {
            let wrapped = format!("{}.linear.weight", base);
            let plain = format!("{}.weight", base);
            let w = if self.exists(&wrapped) {
                self.upload_w(&wrapped, n_out, n_in)?
            } else {
                self.upload_w(&plain, n_out, n_in)?
            };
            let bound = |suffix: &str| -> Option<f32> {
                let n = format!("{}.{}", base, suffix);
                let meta = self
                    .roots
                    .iter()
                    .find_map(|r| self.weights.tensors().get(&format!("{}{}", r, n)))?;
                let v = self.scalar_f32(meta).ok()?;
                if v.is_finite() && v.abs() < 1e30 {
                    Some(v)
                } else {
                    None
                }
            };
            let clip_in = match (bound("input_min"), bound("input_max")) {
                (Some(lo), Some(hi)) => Some((lo, hi)),
                (Some(lo), None) => Some((lo, f32::MAX)),
                (None, Some(hi)) => Some((f32::MIN, hi)),
                (None, None) => None,
            };
            let clip_out = match (bound("output_min"), bound("output_max")) {
                (Some(lo), Some(hi)) => Some((lo, hi)),
                (Some(lo), None) => Some((lo, f32::MAX)),
                (None, Some(hi)) => Some((f32::MIN, hi)),
                (None, None) => None,
            };
            Ok(ClipLin {
                w,
                clip_in,
                clip_out,
            })
        }

        /// Read a scalar tensor (shape `[]` or `[1]`) into f32 (bf16/f16/f32).
        pub(in crate::models::gemma4) fn scalar_f32(&self, meta: &TensorMeta) -> Res<f32> {
            let b = self.weights.bytes(meta)?;
            Ok(match meta.dtype {
                DType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                DType::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                DType::BF16 => bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                other => {
                    return Err(err!(
                        "weights",
                        "gemma4: scalar '{}' has unsupported dtype {}",
                        meta.name,
                        other.name()
                    ))
                }
            })
        }

        /// Read a small tensor fully into host f32 (for CPU-side math: audio
        /// subsampling convs, per_dim_scale, …).
        pub(in crate::models::gemma4) fn host_f32(
            &self,
            name: &str,
            expect: &[usize],
        ) -> Res<Vec<f32>> {
            let meta = self.meta(name)?;
            if !expect.is_empty() && !shape_view_ok(&meta.shape, expect) {
                return Err(err!(
                    "weights",
                    "gemma4: tensor '{}' has shape {:?} but the architecture requires {:?}",
                    meta.name,
                    meta.shape,
                    expect
                ));
            }
            decode_f32(meta.dtype, self.weights.bytes(meta)?, &meta.name)
        }
    }

    /// Shape compatibility: exact match, or a row-major-identical VIEW — same
    /// leading dimension and the same number of trailing elements (e.g. the
    /// gguf export keeps the vision patch projection as a conv kernel
    /// `[768, 3, 16, 16]` where the HF checkpoint flattens it to `[768, 768]`;
    /// the bytes are identical). Transposes ([a,b] vs [b,a]) stay rejected —
    /// same numel, different bytes.
    pub(in crate::models::gemma4) fn shape_view_ok(got: &[usize], expect: &[usize]) -> bool {
        if got == expect {
            return true;
        }
        if got.is_empty() || expect.is_empty() || got[0] != expect[0] {
            return false;
        }
        // Symmetric: [1024, 5] satisfies an expected [1024, 1, 5] just as
        // [768, 3, 16, 16] satisfies an expected [768, 768].
        got[1..].iter().product::<usize>() == expect[1..].iter().product::<usize>()
    }

    /// Decode a 16/32-bit tensor's raw little-endian bytes to host f32 — the
    /// one dtype switch shared by every CPU-side weight reader (G4Index
    /// small-tensor loads, the vision CPU reference, …).
    pub(in crate::models::gemma4) fn decode_f32(
        dtype: DType,
        b: &[u8],
        who: &str,
    ) -> Res<Vec<f32>> {
        Ok(match dtype {
            DType::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F16 => b
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            DType::BF16 => b
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(err!(
                    "weights",
                    "gemma4: tensor '{}' has unsupported dtype {} (16/32-bit only)",
                    who,
                    other.name()
                ))
            }
        })
    }

    /// A linear whose input/output may be clamped (`Gemma4ClippableLinear`).
    pub(in crate::models::gemma4) struct ClipLin {
        w: WTensor,
        clip_in: Option<(f32, f32)>,
        clip_out: Option<(f32, f32)>,
    }

    impl ClipLin {
        /// `y[rows, n_out] = clip_out(clip_in(x) @ w^T)`. When the input is
        /// clamped it is staged through `scratch` so `x` is never mutated
        /// (several projections read the same normed activations). `wsc` is the
        /// NF4 dequant scratch, untouched for f16 weights.
        #[allow(clippy::too_many_arguments)]
        pub(in crate::models::gemma4) fn run(
            &self,
            ctx: &CudaCtx,
            x: u64,
            scratch: u64,
            y: u64,
            rows: usize,
            n_in: usize,
            n_out: usize,
            wsc: u64,
        ) -> Res<()> {
            let src = if let Some((lo, hi)) = self.clip_in {
                ctx.dtod(scratch, x, rows * n_in * 2)?;
                ctx.clampk(scratch, lo, hi, rows * n_in)?;
                scratch
            } else {
                x
            };
            self.w.gemm(ctx, src, y, rows, n_out, n_in, wsc)?;
            if let Some((lo, hi)) = self.clip_out {
                ctx.clampk(y, lo, hi, rows * n_out)?;
            }
            Ok(())
        }
    }
}
