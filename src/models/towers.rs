//! # towers — bidirectional vision / audio encoders
//!
//! The shared CLIP/SigLIP-style ViT and Whisper-style mel encoder that
//! feed the generic [`super::transformer::Transformer`] pipeline. Bound to
//! orchestration exclusively through the [`PreparedPrompt`] media-embedding
//! contract; split out of `transformer.rs` because the decoder never sees
//! the concrete tower type.

use super::transformer::{EncoderConfig, WeightIndex};
use crate::cuda::{CudaCtx, DeviceBuf};
use crate::err;
use crate::media::log_mel;
use crate::num::f32_to_f16;
use crate::traits::*;

/// Which media frontend a tower implements. Both kinds share the encoder
/// block structure (LayerNorm → MHA → LayerNorm → MLP/GELU) and differ only
/// in the patchification frontend and weight naming.
#[derive(Clone, Copy, PartialEq)]
pub enum TowerKind {
    /// CLIP/SigLIP-style ViT (LLaVA / Qwen-VL family).
    Vision,
    /// Whisper-style mel encoder (Qwen2-Audio family).
    Audio,
}

/// One encoder layer (LayerNorm flavor, with biases everywhere).
struct EncLayer {
    ln1_w: DeviceBuf,
    ln1_b: DeviceBuf,
    wq: DeviceBuf,
    bq: DeviceBuf,
    wk: DeviceBuf,
    bk: Option<DeviceBuf>, // Whisper k_proj has no bias
    wv: DeviceBuf,
    bv: DeviceBuf,
    wo: DeviceBuf,
    bo: DeviceBuf,
    ln2_w: DeviceBuf,
    ln2_b: DeviceBuf,
    fc1: DeviceBuf,
    fb1: DeviceBuf,
    fc2: DeviceBuf,
    fb2: DeviceBuf,
}

/// A bidirectional encoder tower plus its projector into the decoder's
/// embedding space. Bound to the orchestration pipeline exclusively through
/// the [`PreparedPrompt`] media-embedding contract — the decoder never sees
/// the concrete tower type.
///
/// Numerical scope note: features are taken from the final encoder layer
/// (no post-LayerNorm, no feature-layer offset) and audio towers skip the
/// optional stride-2 average pooler; this matches the dominant LLaVA-style
/// projector contract dimensionally while trading exact per-model fidelity
/// for a single shared pipeline.
pub struct EncoderTower {
    /// Retained for telemetry / future tower-specific dispatch.
    #[allow(dead_code)]
    kind: TowerKind,
    pub(crate) ec: EncoderConfig,
    head_dim: usize,
    /// Frontend: patch-embedding / conv weights reshaped to GEMM form.
    patch_w: DeviceBuf,
    patch_b: Option<DeviceBuf>,
    /// Audio only: second conv (stride 2).
    conv2_w: Option<DeviceBuf>,
    conv2_b: Option<DeviceBuf>,
    pos_embed: Option<DeviceBuf>,
    class_embed: Option<DeviceBuf>,
    pre_ln: Option<(DeviceBuf, DeviceBuf)>,
    layers: Vec<EncLayer>,
    final_ln: Option<(DeviceBuf, DeviceBuf)>,
    /// Projector: one or two linears (+GELU between when two).
    proj1_w: DeviceBuf,
    proj1_b: Option<DeviceBuf>,
    proj2_w: Option<DeviceBuf>,
    proj2_b: Option<DeviceBuf>,
    /// Workspace sized for `max_tokens` rows.
    max_tokens: usize,
    x: DeviceBuf,
    h: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    ff: DeviceBuf,
    kc: DeviceBuf,
    vc: DeviceBuf,
    stage: DeviceBuf,
    lm_hidden: usize,
}

impl EncoderTower {
    /// Resolve, validate and upload the tower + projector weights.
    pub(super) fn build(
        ctx: &CudaCtx,
        ec: &EncoderConfig,
        ix: &WeightIndex,
        kind: TowerKind,
        lm_hidden: usize,
    ) -> Res<EncoderTower> {
        // Tower-specific name prefixes layered on top of the global ones.
        let prefixes: &[&str] = match kind {
            TowerKind::Vision => &[
                "vision_tower.vision_model.",
                "model.vision_tower.vision_model.",
                "visual.",
                "vision_model.",
            ],
            TowerKind::Audio => &["audio_tower.", "model.audio_tower.", "audio_model."],
        };
        // Local resolver: try tower prefixes through the global index.
        let find = |suffixes: &[&str]| -> Res<&TensorMeta> {
            for p in prefixes {
                for s in suffixes {
                    if let Ok(m) = ix.meta(&format!("{}{}", p, s)) {
                        return Ok(m);
                    }
                }
            }
            Err(err!(
                "weights",
                "encoder tensor not found under any of {:?} with suffixes {:?} — \
                 the {} tower layout is not recognized or the checkpoint is incomplete",
                prefixes,
                suffixes,
                if kind == TowerKind::Vision {
                    "vision"
                } else {
                    "audio"
                }
            ))
        };
        let up = |m: &TensorMeta| -> Res<DeviceBuf> {
            if !ix.codec.accepts(m.dtype) {
                return Err(err!(
                    "weights",
                    "encoder tensor '{}' has unsupported dtype {}",
                    m.name,
                    m.dtype.name()
                ));
            }
            ix.codec.upload(ix.ctx, m, ix.weights.bytes(m)?)
        };
        let opt = |suffixes: &[&str]| -> Res<Option<DeviceBuf>> {
            match find(suffixes) {
                Ok(m) => Ok(Some(up(m)?)),
                Err(_) => Ok(None),
            }
        };

        let head_dim = ec.hidden_size / ec.n_heads.max(1);
        if head_dim * ec.n_heads != ec.hidden_size {
            return Err(err!(
                "config",
                "encoder hidden_size {} not divisible by num_heads {}",
                ec.hidden_size,
                ec.n_heads
            ));
        }

        // ---- frontend ----
        let (patch_w, patch_b, conv2_w, conv2_b, max_tokens) = match kind {
            TowerKind::Vision => {
                let pw = find(&[
                    "embeddings.patch_embedding.weight",
                    "patch_embed.proj.weight",
                ])?;
                let expect = ec.hidden_size * 3 * ec.patch_size * ec.patch_size;
                if pw.numel() != expect {
                    return Err(err!(
                        "weights",
                        "patch embedding '{}' has {} elements, expected {} ([hidden={}, 3, {p}, {p}])",
                        pw.name, pw.numel(), expect, ec.hidden_size, p = ec.patch_size
                    ));
                }
                let n = (ec.input_size / ec.patch_size).pow(2) + 1;
                (
                    up(pw)?,
                    opt(&["embeddings.patch_embedding.bias", "patch_embed.proj.bias"])?,
                    None,
                    None,
                    n,
                )
            }
            TowerKind::Audio => {
                let c1 = find(&["conv1.weight"])?;
                if c1.numel() != ec.hidden_size * ec.patch_size * 3 {
                    return Err(err!(
                        "weights",
                        "audio conv1 '{}' has {} elements, expected {} ([d_model={}, mel={}, k=3])",
                        c1.name,
                        c1.numel(),
                        ec.hidden_size * ec.patch_size * 3,
                        ec.hidden_size,
                        ec.patch_size
                    ));
                }
                let c2 = find(&["conv2.weight"])?;
                (
                    up(c1)?,
                    opt(&["conv1.bias"])?,
                    Some(up(c2)?),
                    opt(&["conv2.bias"])?,
                    ec.input_size,
                )
            }
        };

        // ---- encoder blocks ----
        let mut layers = Vec::with_capacity(ec.n_layers);
        for l in 0..ec.n_layers {
            let n = |s: &str| -> Vec<String> {
                vec![
                    format!("encoder.layers.{}.{}", l, s), // CLIP & Whisper
                    format!("blocks.{}.{}", l, s),         // Qwen-VL visual
                    format!("layers.{}.{}", l, s),
                ]
            };
            let g = |names: &[&str]| -> Res<DeviceBuf> {
                let mut all = Vec::new();
                for nm in names {
                    all.extend(n(nm));
                }
                let refs: Vec<&str> = all.iter().map(String::as_str).collect();
                up(find(&refs)?)
            };
            let go = |names: &[&str]| -> Res<Option<DeviceBuf>> {
                let mut all = Vec::new();
                for nm in names {
                    all.extend(n(nm));
                }
                let refs: Vec<&str> = all.iter().map(String::as_str).collect();
                opt(&refs)
            };
            layers.push(EncLayer {
                ln1_w: g(&[
                    "layer_norm1.weight",
                    "self_attn_layer_norm.weight",
                    "norm1.weight",
                ])?,
                ln1_b: g(&[
                    "layer_norm1.bias",
                    "self_attn_layer_norm.bias",
                    "norm1.bias",
                ])?,
                wq: g(&["self_attn.q_proj.weight", "attn.q_proj.weight"])?,
                bq: g(&["self_attn.q_proj.bias", "attn.q_proj.bias"])?,
                wk: g(&["self_attn.k_proj.weight", "attn.k_proj.weight"])?,
                bk: go(&["self_attn.k_proj.bias", "attn.k_proj.bias"])?,
                wv: g(&["self_attn.v_proj.weight", "attn.v_proj.weight"])?,
                bv: g(&["self_attn.v_proj.bias", "attn.v_proj.bias"])?,
                wo: g(&[
                    "self_attn.out_proj.weight",
                    "attn.proj.weight",
                    "self_attn.o_proj.weight",
                ])?,
                bo: g(&[
                    "self_attn.out_proj.bias",
                    "attn.proj.bias",
                    "self_attn.o_proj.bias",
                ])?,
                ln2_w: g(&[
                    "layer_norm2.weight",
                    "final_layer_norm.weight",
                    "norm2.weight",
                ])?,
                ln2_b: g(&["layer_norm2.bias", "final_layer_norm.bias", "norm2.bias"])?,
                fc1: g(&["mlp.fc1.weight", "fc1.weight"])?,
                fb1: g(&["mlp.fc1.bias", "fc1.bias"])?,
                fc2: g(&["mlp.fc2.weight", "fc2.weight"])?,
                fb2: g(&["mlp.fc2.bias", "fc2.bias"])?,
            });
        }

        // ---- pos / class / norms ----
        let pos_embed = opt(&[
            "embeddings.position_embedding.weight",
            "embed_positions.weight",
            "pos_embed",
        ])?;
        let class_embed = opt(&["embeddings.class_embedding"])?;
        let pre_ln = match (
            opt(&["pre_layrnorm.weight", "pre_layernorm.weight"])?,
            opt(&["pre_layrnorm.bias", "pre_layernorm.bias"])?,
        ) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        };
        let final_ln = match (
            opt(&["post_layernorm.weight", "layer_norm.weight"])?,
            opt(&["post_layernorm.bias", "layer_norm.bias"])?,
        ) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        };

        // ---- projector (lives at the checkpoint root, not under the tower) ----
        let proj1 = ix
            .meta("multi_modal_projector.linear_1.weight")
            .or_else(|_| ix.meta("multi_modal_projector.linear.weight"))
            .or_else(|_| ix.meta("mm_projector.0.weight"))
            .or_else(|_| ix.meta("merger.mlp.0.weight"))
            .map_err(|_| {
                err!(
                    "weights",
                    "multimodal projector not found (tried multi_modal_projector.linear_1/linear, mm_projector.0, merger.mlp.0) — \
                     the checkpoint cannot bind the {} tower to the language model",
                    if kind == TowerKind::Vision { "vision" } else { "audio" }
                )
            })?;
        let proj1_w = up(proj1)?;
        let proj1_b = match ix.meta(&proj1.name.replace(".weight", ".bias")) {
            Ok(m) => Some(up(m)?),
            Err(_) => None,
        };
        let (proj2_w, proj2_b) = {
            let cand = [
                "multi_modal_projector.linear_2.weight",
                "mm_projector.2.weight",
                "merger.mlp.2.weight",
            ];
            let mut w = None;
            let mut b = None;
            for c in cand {
                if let Ok(m) = ix.meta(c) {
                    w = Some(up(m)?);
                    if let Ok(mb) = ix.meta(&c.replace(".weight", ".bias")) {
                        b = Some(up(mb)?);
                    }
                    break;
                }
            }
            (w, b)
        };

        // ---- workspace ----
        let hv = ec.hidden_size;
        let mt = max_tokens;
        let proj_mid = lm_hidden.max(hv).max(ec.intermediate_size);
        let tower = EncoderTower {
            kind,
            ec: ec.clone(),
            head_dim,
            patch_w,
            patch_b,
            conv2_w,
            conv2_b,
            pos_embed,
            class_embed,
            pre_ln,
            layers,
            final_ln,
            proj1_w,
            proj1_b,
            proj2_w,
            proj2_b,
            max_tokens: mt,
            x: ix.ctx.alloc(mt * hv.max(lm_hidden) * 2)?,
            h: ix.ctx.alloc(mt * proj_mid * 2)?,
            q: ix.ctx.alloc(mt * hv * 2)?,
            k: ix.ctx.alloc(mt * hv * 2)?,
            v: ix.ctx.alloc(mt * hv * 2)?,
            ff: ix.ctx.alloc(mt * ec.intermediate_size.max(proj_mid) * 2)?,
            kc: ix.ctx.alloc(mt * hv * 2)?,
            vc: ix.ctx.alloc(mt * hv * 2)?,
            stage: ix.ctx.alloc(
                mt * (3 * ec.patch_size * ec.patch_size)
                    .max(ec.patch_size * 3)
                    .max(hv * 3)
                    * 2,
            )?,
            lm_hidden,
        };
        let _ = ctx;
        Ok(tower)
    }

    /// Shared encoder body over `rows` tokens resident in `self.x`.
    fn run_blocks(&self, ctx: &CudaCtx, rows: usize) -> Res<()> {
        let hv = self.ec.hidden_size;
        let eps = self.ec.layer_norm_eps;
        if let Some((w, b)) = &self.pre_ln {
            ctx.layernorm(self.x.ptr, w.ptr, b.ptr, self.h.ptr, rows, hv, eps)?;
            ctx.dtod(self.x.ptr, self.h.ptr, rows * hv * 2)?;
        }
        for l in &self.layers {
            ctx.layernorm(
                self.x.ptr,
                l.ln1_w.ptr,
                l.ln1_b.ptr,
                self.h.ptr,
                rows,
                hv,
                eps,
            )?;
            ctx.gemm_f16(self.h.ptr, l.wq.ptr, self.q.ptr, rows, hv, hv)?;
            ctx.gemm_f16(self.h.ptr, l.wk.ptr, self.k.ptr, rows, hv, hv)?;
            ctx.gemm_f16(self.h.ptr, l.wv.ptr, self.v.ptr, rows, hv, hv)?;
            ctx.bias(self.q.ptr, l.bq.ptr, rows, hv)?;
            if let Some(b) = &l.bk {
                ctx.bias(self.k.ptr, b.ptr, rows, hv)?;
            }
            ctx.bias(self.v.ptr, l.bv.ptr, rows, hv)?;
            ctx.kv_append(
                self.k.ptr,
                self.v.ptr,
                self.kc.ptr,
                self.vc.ptr,
                rows,
                self.ec.n_heads,
                self.head_dim,
                0,
                self.max_tokens,
                0,
            )?;
            ctx.attn_prefill(
                self.q.ptr,
                self.kc.ptr,
                self.vc.ptr,
                self.h.ptr,
                rows,
                self.ec.n_heads,
                self.ec.n_heads,
                self.head_dim,
                0,
                self.max_tokens,
                false,
                1.0 / (self.head_dim as f32).sqrt(),
                0,
                0,
            )?;
            ctx.gemm_f16(self.h.ptr, l.wo.ptr, self.q.ptr, rows, hv, hv)?;
            ctx.bias(self.q.ptr, l.bo.ptr, rows, hv)?;
            ctx.add(self.x.ptr, self.q.ptr, rows * hv)?;
            ctx.layernorm(
                self.x.ptr,
                l.ln2_w.ptr,
                l.ln2_b.ptr,
                self.h.ptr,
                rows,
                hv,
                eps,
            )?;
            ctx.gemm_f16(
                self.h.ptr,
                l.fc1.ptr,
                self.ff.ptr,
                rows,
                self.ec.intermediate_size,
                hv,
            )?;
            ctx.bias(self.ff.ptr, l.fb1.ptr, rows, self.ec.intermediate_size)?;
            ctx.gelu(self.ff.ptr, rows * self.ec.intermediate_size)?;
            ctx.gemm_f16(
                self.ff.ptr,
                l.fc2.ptr,
                self.h.ptr,
                rows,
                hv,
                self.ec.intermediate_size,
            )?;
            ctx.bias(self.h.ptr, l.fb2.ptr, rows, hv)?;
            ctx.add(self.x.ptr, self.h.ptr, rows * hv)?;
        }
        if let Some((w, b)) = &self.final_ln {
            ctx.layernorm(self.x.ptr, w.ptr, b.ptr, self.h.ptr, rows, hv, eps)?;
            ctx.dtod(self.x.ptr, self.h.ptr, rows * hv * 2)?;
        }
        Ok(())
    }

    /// Project `rows` encoder features (in `self.x`) into the LM embedding
    /// space; returns a fresh `[rows, lm_hidden]` f16 buffer.
    fn project(&self, ctx: &CudaCtx, rows: usize) -> Res<DeviceBuf> {
        let hv = self.ec.hidden_size;
        if let Some(p2) = &self.proj2_w {
            // linear_1 -> GELU -> linear_2 (LLaVA projector).
            let mid = self.h.bytes / (rows.max(1) * 2);
            let _ = mid;
            // Mid dim derives from proj1 shape; recompute from buffer geometry:
            // proj1: [mid, hv] => rows×mid output into ff.
            let mid_dim = self.proj1_w.bytes / (hv * 2);
            ctx.gemm_f16(self.x.ptr, self.proj1_w.ptr, self.ff.ptr, rows, mid_dim, hv)?;
            if let Some(b) = &self.proj1_b {
                ctx.bias(self.ff.ptr, b.ptr, rows, mid_dim)?;
            }
            ctx.gelu(self.ff.ptr, rows * mid_dim)?;
            let out = ctx.alloc(rows * self.lm_hidden * 2)?;
            ctx.gemm_f16(self.ff.ptr, p2.ptr, out.ptr, rows, self.lm_hidden, mid_dim)?;
            if let Some(b) = &self.proj2_b {
                ctx.bias(out.ptr, b.ptr, rows, self.lm_hidden)?;
            }
            Ok(out)
        } else {
            let out = ctx.alloc(rows * self.lm_hidden * 2)?;
            ctx.gemm_f16(
                self.x.ptr,
                self.proj1_w.ptr,
                out.ptr,
                rows,
                self.lm_hidden,
                hv,
            )?;
            if let Some(b) = &self.proj1_b {
                ctx.bias(out.ptr, b.ptr, rows, self.lm_hidden)?;
            }
            Ok(out)
        }
    }

    /// Encode one image into LM-space embeddings: `(buffer, rows)`.
    pub fn encode_image(&self, ctx: &CudaCtx, img: &ImageTensor) -> Res<(DeviceBuf, usize)> {
        let p = self.ec.patch_size;
        let hv = self.ec.hidden_size;
        let (gh, gw) = (img.height / p, img.width / p);
        let n = gh * gw;
        let cls = self.class_embed.is_some() as usize;
        if n + cls > self.max_tokens {
            return Err(err!(
                "media",
                "image yields {} patches, tower supports {}",
                n,
                self.max_tokens - cls
            ));
        }
        // CPU unfold: [n, 3*p*p] f16, channel-major within a patch.
        let mut unf = vec![0u16; n * 3 * p * p];
        for gy in 0..gh {
            for gx in 0..gw {
                let row = gy * gw + gx;
                let mut o = row * 3 * p * p;
                for c in 0..3 {
                    for y in 0..p {
                        for x in 0..p {
                            let v =
                                img.data[(c * img.height + gy * p + y) * img.width + gx * p + x];
                            unf[o] = f32_to_f16(v);
                            o += 1;
                        }
                    }
                }
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(unf.as_ptr() as *const u8, unf.len() * 2) };
        ctx.htod(&self.stage, bytes)?;
        // Patch embedding GEMM into x rows [cls..cls+n].
        let xrows = self.x.ptr + (cls * hv * 2) as u64;
        ctx.gemm_f16(self.stage.ptr, self.patch_w.ptr, xrows, n, hv, 3 * p * p)?;
        if let Some(b) = &self.patch_b {
            ctx.bias(xrows, b.ptr, n, hv)?;
        }
        if let Some(ce) = &self.class_embed {
            ctx.dtod(self.x.ptr, ce.ptr, hv * 2)?;
        }
        let rows = n + cls;
        if let Some(pe) = &self.pos_embed {
            ctx.add(self.x.ptr, pe.ptr, rows * hv)?;
        }
        self.run_blocks(ctx, rows)?;
        if cls == 1 {
            // Drop the class token: shift features up one row (LLaVA "patch" select).
            ctx.dtod(self.h.ptr, self.x.ptr + (hv * 2) as u64, n * hv * 2)?;
            ctx.dtod(self.x.ptr, self.h.ptr, n * hv * 2)?;
        }
        let out = self.project(ctx, n)?;
        Ok((out, n))
    }

    /// Encode mono PCM (already decoded) into LM-space embeddings.
    pub fn encode_audio(&self, ctx: &CudaCtx, pcm: &AudioPcm) -> Res<(DeviceBuf, usize)> {
        let mel = log_mel(pcm, self.ec.patch_size, &crate::media::MelParams::default()); // [frames][mel]
        let t_in = mel.len();
        if t_in == 0 {
            return Err(err!("media", "audio produced zero mel frames"));
        }
        let nm = self.ec.patch_size;
        let hv = self.ec.hidden_size;
        // conv1: k=3, s=1, pad=1 → unfold [t_in, nm*3].
        let t1 = t_in.min(self.max_tokens * 2); // conv2 halves; bound input
        let mut unf = vec![0u16; t1 * nm * 3];
        for t in 0..t1 {
            let mut o = t * nm * 3;
            // c is a mel channel index feeding the output-offset arithmetic.
            #[allow(clippy::needless_range_loop)]
            for c in 0..nm {
                for k in 0..3usize {
                    let src = (t + k)
                        .checked_sub(1)
                        .filter(|&s| s < t_in)
                        .map(|s| mel[s][c])
                        .unwrap_or(0.0);
                    unf[o] = f32_to_f16(src);
                    o += 1;
                }
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(unf.as_ptr() as *const u8, unf.len() * 2) };
        ctx.htod(&self.stage, bytes)?;
        ctx.gemm_f16(self.stage.ptr, self.patch_w.ptr, self.q.ptr, t1, hv, nm * 3)?;
        if let Some(b) = &self.patch_b {
            ctx.bias(self.q.ptr, b.ptr, t1, hv)?;
        }
        ctx.gelu(self.q.ptr, t1 * hv)?;
        // conv2: k=3, s=2, pad=1 over the [t1, hv] activation (host roundtrip
        // for the unfold — frontend cost is negligible next to the blocks).
        let mut act = vec![0u8; t1 * hv * 2];
        let tmp = DeviceBuf {
            ptr: self.q.ptr,
            bytes: t1 * hv * 2,
        };
        ctx.sync()?;
        ctx.dtoh(&mut act, &tmp)?;
        std::mem::forget(tmp);
        let half = |b0: u8, b1: u8| u16::from_le_bytes([b0, b1]);
        let t2 = (t1 / 2).min(self.max_tokens).max(1);
        let mut unf2 = vec![0u16; t2 * hv * 3];
        for t in 0..t2 {
            let mut o = t * hv * 3;
            for c in 0..hv {
                for k in 0..3usize {
                    let s = (2 * t + k).checked_sub(1).filter(|&s| s < t1);
                    unf2[o] = match s {
                        Some(s) => half(act[(s * hv + c) * 2], act[(s * hv + c) * 2 + 1]),
                        None => 0,
                    };
                    o += 1;
                }
            }
        }
        let bytes2 =
            unsafe { std::slice::from_raw_parts(unf2.as_ptr() as *const u8, unf2.len() * 2) };
        ctx.htod(&self.stage, bytes2)?;
        let c2 = self
            .conv2_w
            .as_ref()
            .ok_or_else(|| err!("weights", "audio tower missing conv2.weight"))?;
        ctx.gemm_f16(self.stage.ptr, c2.ptr, self.x.ptr, t2, hv, hv * 3)?;
        if let Some(b) = &self.conv2_b {
            ctx.bias(self.x.ptr, b.ptr, t2, hv)?;
        }
        ctx.gelu(self.x.ptr, t2 * hv)?;
        if let Some(pe) = &self.pos_embed {
            ctx.add(self.x.ptr, pe.ptr, t2 * hv)?;
        }
        self.run_blocks(ctx, t2)?;
        let out = self.project(ctx, t2)?;
        Ok((out, t2))
    }
}
