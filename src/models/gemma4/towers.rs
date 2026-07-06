//! # towers — the gemma4 media encoders
//!
//! * [`vision`] — the SigLIP-derived tower with 2-D RoPE, average pooling
//!   and the multimodal embedder, plus its CPU reference for `selftest`.
//! * [`audio`] — the USM-style conformer: CPU log-mel + subsampling convs,
//!   chunked local attention, light-conv, and the audio embedder.

pub(crate) mod vision {
    //! Vision tower: SigLIP-style encoder with avg-pool projector.

    use crate::cuda::{CudaCtx, DeviceBuf};
    use crate::err;

    use crate::log;
    use crate::traits::{DType, ImageTensor, LoadedWeights, Res};

    use crate::quant::bnb::WTensor;

    use super::super::support::config::*;
    use super::super::support::weights::*;
    use super::super::Gemma4;
    use crate::num::*;

    // ===========================================================================
    // Vision tower
    // ===========================================================================

    /// Patch budget: `vision_soft_tokens_per_image` (280) × pool² (9) = 2520
    /// patches max. Images are resized aspect-preserving so both sides are
    /// multiples of `pool·patch` = 48 px and the patch count fits the budget —
    /// the reference `get_aspect_ratio_preserving_size`, reproduced in
    /// [`Gemma4::image_target_size`]. Soft-token count varies per image.
    pub(in crate::models::gemma4) const G4_MAX_PATCHES: usize = 2520;
    pub(in crate::models::gemma4) const G4_SIDE_MULT: usize = 48; // pooling_kernel_size · patch_size

    struct G4VisLayer {
        input_norm: DeviceBuf,
        post_attn_norm: DeviceBuf,
        pre_ffw_norm: DeviceBuf,
        post_ffw_norm: DeviceBuf,
        q: ClipLin,
        k: ClipLin,
        v: ClipLin,
        o: ClipLin,
        q_norm: DeviceBuf,
        k_norm: DeviceBuf,
        gate: ClipLin,
        up: ClipLin,
        down: ClipLin,
    }

    pub struct G4Vision {
        input_proj: WTensor,  // [hidden, 3*patch²]
        pos_table: DeviceBuf, // [2, table, hidden] (x rows then y rows)
        table_rows: usize,
        layers: Vec<G4VisLayer>,
        emb_proj: WTensor, // embed_vision.embedding_projection [lm_hidden, hidden]
        /// `vision_tower.std_bias` / `std_scale` [hidden] — the reference's
        /// post-pooler standardization `(x - bias) * scale`, applied in f32
        /// *before* the embedder's scale-less RMSNorm ("the std_bias subtraction
        /// cancels large values"). Loaded when the checkpoint ships the buffers,
        /// regardless of the config flag — the tensors are authoritative.
        std_affine: Option<(Vec<f32>, Vec<f32>)>,
        // scratch (sized for the largest grid, G4_MAX_PATCHES)
        sx: DeviceBuf, // residual [n, hidden]
        sh: DeviceBuf,
        sh2: DeviceBuf,
        sq: DeviceBuf,
        sk: DeviceBuf,
        sv: DeviceBuf,
        satt: DeviceBuf,
        sgate: DeviceBuf,
        sup: DeviceBuf,
        sclip: DeviceBuf, // ClipLin input staging
        posx: DeviceBuf,  // i32 patch x  (rope2d)
        posy: DeviceBuf,
        gidx: DeviceBuf, // u32 gather ids (pos-embed x)
        gidy: DeviceBuf,
        pose: DeviceBuf,   // gathered position embeddings [n, hidden]
        pooled: DeviceBuf, // [n_tokens, hidden] after CPU pool re-upload
        out: DeviceBuf,    // [n_tokens, lm_hidden] after the embedder
        cfg_hidden: usize,
        lm_hidden: usize,
        head_dim: usize,
        n_heads: usize,
        inter: usize,
        pool_k: usize,
        theta: f32,
        rms_eps: f32,
    }

    impl G4Vision {
        pub(in crate::models::gemma4) fn build(
            ctx: &CudaCtx,
            vc: &G4VisionCfg,
            ix: &G4Index,
            lm_hidden: usize,
        ) -> Res<G4Vision> {
            let h = vc.hidden;
            let n = G4_MAX_PATCHES; // scratch sized for the largest grid
            let qd = vc.n_heads * vc.head_dim;

            let vt = |s: &str| format!("vision_tower.{}", s);
            let mut layers = Vec::with_capacity(vc.n_layers);
            for i in 0..vc.n_layers {
                let p = |s: &str| vt(&format!("encoder.layers.{}.{}", i, s));
                layers.push(G4VisLayer {
                    input_norm: ix.upload(&p("input_layernorm.weight"), &[h])?,
                    post_attn_norm: ix.upload(&p("post_attention_layernorm.weight"), &[h])?,
                    pre_ffw_norm: ix.upload(&p("pre_feedforward_layernorm.weight"), &[h])?,
                    post_ffw_norm: ix.upload(&p("post_feedforward_layernorm.weight"), &[h])?,
                    q: ix.upload_clip(&p("self_attn.q_proj"), qd, h)?,
                    k: ix.upload_clip(&p("self_attn.k_proj"), qd, h)?,
                    v: ix.upload_clip(&p("self_attn.v_proj"), qd, h)?,
                    o: ix.upload_clip(&p("self_attn.o_proj"), h, qd)?,
                    q_norm: ix.upload(&p("self_attn.q_norm.weight"), &[vc.head_dim])?,
                    k_norm: ix.upload(&p("self_attn.k_norm.weight"), &[vc.head_dim])?,
                    gate: ix.upload_clip(&p("mlp.gate_proj"), vc.inter, h)?,
                    up: ix.upload_clip(&p("mlp.up_proj"), vc.inter, h)?,
                    down: ix.upload_clip(&p("mlp.down_proj"), h, vc.inter)?,
                });
            }

            let n_tokens_max = G4_MAX_PATCHES / (vc.pool_k * vc.pool_k);
            Ok(G4Vision {
                input_proj: {
                    // Conv-form kernels need the (c,y,x)→(y,x,c) row permute
                    // before upload — see load_patch_proj_f32.
                    let w = load_patch_proj_f32(
                        ix.weights,
                        &vt("patch_embedder.input_proj.weight"),
                        h,
                        vc.patch,
                    )?;
                    let wh: Vec<u16> = w.iter().map(|v| crate::num::f32_to_f16(*v)).collect();
                    let buf = ctx.alloc(wh.len() * 2)?;
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(wh.as_ptr() as *const u8, wh.len() * 2)
                    };
                    ctx.htod(&buf, bytes)?;
                    crate::quant::bnb::WTensor::F16(buf)
                },
                pos_table: ix.upload(
                    &vt("patch_embedder.position_embedding_table"),
                    &[2, vc.pos_table, h],
                )?,
                table_rows: vc.pos_table,
                layers,
                emb_proj: ix.upload_w("embed_vision.embedding_projection.weight", lm_hidden, h)?,
                std_affine: {
                    // Probe, don't error: most checkpoints ship without the
                    // standardize buffers and their absence is the normal case
                    // (constructing an EngineError would log spurious ERRORs).
                    let present =
                        ix.exists("vision_tower.std_bias") && ix.exists("vision_tower.std_scale");
                    let pair = if present {
                        Some((
                            ix.host_f32("vision_tower.std_bias", &[h])?,
                            ix.host_f32("vision_tower.std_scale", &[h])?,
                        ))
                    } else {
                        None
                    };
                    match pair {
                        Some((b, s)) => {
                            log::info("gemma4 vision standardize: active (std_bias/std_scale buffers found)");
                            Some((b, s))
                        }
                        _ => {
                            log::info("gemma4 vision standardize: inactive (no std_bias/std_scale in checkpoint)");
                            None
                        }
                    }
                },
                sx: ctx.alloc(n * h * 2)?,
                sh: ctx.alloc(n * h * 2)?,
                sh2: ctx.alloc(n * h * 2)?,
                sq: ctx.alloc(n * qd * 2)?,
                sk: ctx.alloc(n * qd * 2)?,
                sv: ctx.alloc(n * qd * 2)?,
                satt: ctx.alloc(n * qd * 2)?,
                sgate: ctx.alloc(n * vc.inter * 2)?,
                sup: ctx.alloc(n * vc.inter * 2)?,
                sclip: ctx.alloc(n * h.max(vc.inter) * 2)?,
                posx: ctx.alloc(n * 4)?,
                posy: ctx.alloc(n * 4)?,
                gidx: ctx.alloc(n * 4)?,
                gidy: ctx.alloc(n * 4)?,
                pose: ctx.alloc(n * h * 2)?,
                pooled: ctx.alloc(n_tokens_max * h * 2)?,
                out: ctx.alloc(n_tokens_max * lm_hidden * 2)?,
                cfg_hidden: h,
                lm_hidden,
                head_dim: vc.head_dim,
                n_heads: vc.n_heads,
                inter: vc.inter,
                pool_k: vc.pool_k,
                theta: vc.theta,
                rms_eps: vc.rms_eps,
            })
        }

        /// Encode one image (already decoded as `2·(p−0.5)` pixels) to `[280,
        /// lm_hidden]` soft tokens resident on the device. Returns a fresh buffer
        /// plus its row count.
        pub(in crate::models::gemma4) fn encode(
            &self,
            ctx: &CudaCtx,
            img: &ImageTensor,
            wsc: u64,
        ) -> Res<(DeviceBuf, usize)> {
            self.encode_traced(ctx, img, wsc, None)
        }

        /// `trace`: when `Some`, dtoh-snapshot named stages (f16 raw) for the
        /// GPU-vs-CPU self-test. Zero overhead when `None`.
        pub(in crate::models::gemma4) fn encode_traced(
            &self,
            ctx: &CudaCtx,
            img: &ImageTensor,
            wsc: u64,
            mut trace: Option<&mut Vec<(String, Vec<u16>)>>,
        ) -> Res<(DeviceBuf, usize)> {
            let patch = 16;
            if img.channels != 3 || img.height % G4_SIDE_MULT != 0 || img.width % G4_SIDE_MULT != 0
            {
                return Err(err!(
                    "media",
                    "gemma4 vision: decoded image is {}×{}×{}; sides must be multiples of {} (aspect-preserving resize bug?)",
                    img.channels, img.height, img.width, G4_SIDE_MULT
                ));
            }
            let grid_w = img.width / patch;
            let grid_h = img.height / patch;
            let n = grid_w * grid_h;
            if n > G4_MAX_PATCHES {
                return Err(err!(
                    "media",
                    "gemma4 vision: {}×{} patches exceed the {} budget",
                    grid_w,
                    grid_h,
                    G4_MAX_PATCHES
                ));
            }
            if grid_w.max(grid_h) > self.table_rows {
                return Err(err!(
                    "media",
                    "gemma4 vision: patch grid exceeds the {}-entry position table",
                    self.table_rows
                ));
            }
            let h = self.cfg_hidden;
            let pd = 3 * patch * patch;

            // Per-image patch position ids (patches are y-major; positions raw).
            let xs: Vec<i32> = (0..n).map(|p| (p % grid_w) as i32).collect();
            let ys: Vec<i32> = (0..n).map(|p| (p / grid_w) as i32).collect();
            let gx: Vec<u32> = xs.iter().map(|&v| v as u32).collect();
            let gy: Vec<u32> = ys.iter().map(|&v| v as u32).collect();
            let up = |buf: &DeviceBuf, p: *const u8, len: usize| -> Res<()> {
                ctx.htod(buf, unsafe { std::slice::from_raw_parts(p, len) })
            };
            up(&self.posx, xs.as_ptr() as *const u8, n * 4)?;
            up(&self.posy, ys.as_ptr() as *const u8, n * 4)?;
            up(&self.gidx, gx.as_ptr() as *const u8, n * 4)?;
            up(&self.gidy, gy.as_ptr() as *const u8, n * 4)?;

            // ---- CPU unfold: [3, H, W] -> [n_patches, 16·16·3]. The reference
            // processor's `convert_image_to_patches` permutes (C,gy,py,gx,px) to
            // (gy,gx,py,px,C): the patch interior is **pixel-major with the
            // channel last** — (dy, dx, c), not channel-major. ----
            let mut host: Vec<u16> = vec![0; n * pd];
            for py in 0..grid_h {
                for px in 0..grid_w {
                    let row = py * grid_w + px;
                    for dy in 0..patch {
                        for dx in 0..patch {
                            for c in 0..3 {
                                let v = img.data[(c * img.height + py * patch + dy) * img.width
                                    + px * patch
                                    + dx];
                                host[row * pd + (dy * patch + dx) * 3 + c] = f32_to_f16(v);
                            }
                        }
                    }
                }
            }
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
            ctx.htod(&self.sh, bytes)?; // reuse sh as the patch staging buffer
            self.input_proj
                .gemm(ctx, self.sh.ptr, self.sx.ptr, n, h, pd, wsc)?;

            // ---- learned 2-D position embeddings: table[0][x] + table[1][y] ----
            ctx.gather(
                self.pos_table.ptr,
                false,
                self.gidx.ptr,
                self.pose.ptr,
                n,
                h,
            )?;
            ctx.add(self.sx.ptr, self.pose.ptr, n * h)?;
            let ytab = self.pos_table.ptr + (self.table_rows * h * 2) as u64;
            ctx.gather(ytab, false, self.gidy.ptr, self.pose.ptr, n, h)?;
            ctx.add(self.sx.ptr, self.pose.ptr, n * h)?;

            let snap = |name: &str,
                        ctx: &CudaCtx,
                        buf: &DeviceBuf,
                        len: usize,
                        tr: &mut Option<&mut Vec<(String, Vec<u16>)>>|
             -> Res<()> {
                if let Some(tr) = tr.as_deref_mut() {
                    ctx.sync()?;
                    let mut bytes = vec![0u8; len * 2];
                    ctx.dtoh(&mut bytes, buf)?;
                    let halfs: Vec<u16> = bytes
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    tr.push((name.to_string(), halfs));
                }
                Ok(())
            };
            snap("embed", ctx, &self.sx, n * h, &mut trace)?;

            // ---- 16 bidirectional encoder layers ----
            let (nh, d) = (self.n_heads, self.head_dim);
            let qd = nh * d;
            for (li, layer) in self.layers.iter().enumerate() {
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.input_norm.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                if li == 0 {
                    snap("l0_norm", ctx, &self.sh, n * h, &mut trace)?;
                }
                layer
                    .q
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sq.ptr, n, h, qd, wsc)?;
                layer
                    .k
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sk.ptr, n, h, qd, wsc)?;
                layer
                    .v
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sv.ptr, n, h, qd, wsc)?;
                if li == 0 {
                    snap("l0_q", ctx, &self.sq, n * qd, &mut trace)?;
                    snap("l0_v", ctx, &self.sv, n * qd, &mut trace)?;
                }
                ctx.rmsnorm(
                    self.sq.ptr,
                    layer.q_norm.ptr,
                    self.sq.ptr,
                    n * nh,
                    d,
                    self.rms_eps,
                )?;
                ctx.rmsnorm(
                    self.sk.ptr,
                    layer.k_norm.ptr,
                    self.sk.ptr,
                    n * nh,
                    d,
                    self.rms_eps,
                )?;
                ctx.rmsnorm(self.sv.ptr, 0, self.sv.ptr, n * nh, d, self.rms_eps)?;
                if li == 0 {
                    snap("l0_qn", ctx, &self.sq, n * qd, &mut trace)?;
                }
                ctx.rope2d(
                    self.sq.ptr,
                    self.posx.ptr,
                    self.posy.ptr,
                    n,
                    nh,
                    d,
                    self.theta,
                )?;
                ctx.rope2d(
                    self.sk.ptr,
                    self.posx.ptr,
                    self.posy.ptr,
                    n,
                    nh,
                    d,
                    self.theta,
                )?;
                if li == 0 {
                    snap("l0_qr", ctx, &self.sq, n * qd, &mut trace)?;
                }
                // K/V live in [rows, heads*dim]; the prefill kernel expects the
                // cache layout [heads, seq, dim] — kv_append lays them out into
                // the MLP scratch buffers (sgate/sup, idle during attention and
                // sized n·inter ≥ n·qd).
                ctx.kv_append(
                    self.sk.ptr,
                    self.sv.ptr,
                    self.sgate.ptr,
                    self.sup.ptr,
                    n,
                    nh,
                    d,
                    0,
                    n,
                    0,
                )?;
                ctx.attn_prefill(
                    self.sq.ptr,
                    self.sgate.ptr,
                    self.sup.ptr,
                    self.satt.ptr,
                    n,
                    nh,
                    nh,
                    d,
                    0,
                    n,
                    false,
                    1.0,
                    0,
                    0,
                )?;
                if li == 0 {
                    snap("l0_attn", ctx, &self.satt, n * qd, &mut trace)?;
                }
                layer.o.run(
                    ctx,
                    self.satt.ptr,
                    self.sclip.ptr,
                    self.sh2.ptr,
                    n,
                    qd,
                    h,
                    wsc,
                )?;
                ctx.rmsnorm(
                    self.sh2.ptr,
                    layer.post_attn_norm.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.add(self.sx.ptr, self.sh2.ptr, n * h)?;
                if li == 0 {
                    snap("l0_attnres", ctx, &self.sx, n * h, &mut trace)?;
                }

                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.pre_ffw_norm.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                layer.gate.run(
                    ctx,
                    self.sh.ptr,
                    self.sclip.ptr,
                    self.sgate.ptr,
                    n,
                    h,
                    self.inter,
                    wsc,
                )?;
                layer.up.run(
                    ctx,
                    self.sh.ptr,
                    self.sclip.ptr,
                    self.sup.ptr,
                    n,
                    h,
                    self.inter,
                    wsc,
                )?;
                ctx.geglu(self.sgate.ptr, self.sup.ptr, n * self.inter)?;
                layer.down.run(
                    ctx,
                    self.sgate.ptr,
                    self.sclip.ptr,
                    self.sh2.ptr,
                    n,
                    self.inter,
                    h,
                    wsc,
                )?;
                ctx.rmsnorm(
                    self.sh2.ptr,
                    layer.post_ffw_norm.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.add(self.sx.ptr, self.sh2.ptr, n * h)?;
                snap(&format!("layer{}", li), ctx, &self.sx, n * h, &mut trace)?;
            }

            // ---- 3×3 average pool by positions + sqrt(hidden) scale + embedder
            // RMSNorm, all in f32 on the CPU. The reference keeps the pooled
            // features in float32 precisely because the ×sqrt(hidden) scale "can
            // exceed the float16 range" (its own docstring) and only casts to
            // **bf16** — which has float32 range. Casting these to f16 (max
            // 65504) saturates exactly the highest-energy tokens and corrupts
            // their features. The scale-less RMSNorm collapses the magnitude to
            // O(1), so the safe order is: pool (f32) → norm (f32) → cast to f16
            // for the projection GEMM. ----
            let k = self.pool_k;
            let (ow, oh) = (grid_w / k, grid_h / k);
            let n_tokens = ow * oh;
            let mut dev: Vec<u8> = vec![0; n * h * 2];
            ctx.sync()?;
            ctx.dtoh(&mut dev, &self.sx)?;
            let halfs: &[u16] =
                unsafe { std::slice::from_raw_parts(dev.as_ptr() as *const u16, n * h) };
            // f16 saturation telemetry: the reference tower runs bf16 (float32
            // range); ours runs f16 (max 65504). If activations clip or blow up
            // to ±inf here, downstream features are corrupt and the cause is
            // numeric range, not algorithm. Counted nearly for free since the
            // activations are already on the host for pooling.
            let mut n_inf = 0usize;
            let mut n_sat = 0usize;
            for &u in halfs {
                let m = u & 0x7FFF;
                if m == 0x7C00 {
                    n_inf += 1; // ±inf (NaN would be > 0x7C00)
                } else if m >= 0x7BFF {
                    n_sat += 1; // |v| == 65504 (max finite)
                }
            }
            if n_inf + n_sat > 0 {
                log::warn(&format!(
                    "vision tower: f16 range exhausted in {} of {} activation values ({} inf, {} at max-finite) — \
                     features are numerically corrupt; the reference runs this tower in bf16 (float32 range)",
                    n_inf + n_sat, n * h, n_inf, n_sat
                ));
            }
            let scale = (h as f32).sqrt() / (k * k) as f32;
            let mut normed: Vec<u16> = vec![0; n_tokens * h];
            let mut acc = vec![0f32; h];
            for ty in 0..oh {
                for tx in 0..ow {
                    acc.iter_mut().for_each(|a| *a = 0.0);
                    for dy in 0..k {
                        for dx in 0..k {
                            let p = (ty * k + dy) * grid_w + (tx * k + dx);
                            for c in 0..h {
                                acc[c] += f16_to_f32(halfs[p * h + c]);
                            }
                        }
                    }
                    let mut ssq = 0f64;
                    for c in 0..h {
                        let mut v = acc[c] * scale;
                        if let Some((bias, sc)) = &self.std_affine {
                            v = (v - bias[c]) * sc[c];
                        }
                        acc[c] = v;
                        ssq += (v as f64) * (v as f64);
                    }
                    let inv = 1.0 / ((ssq / h as f64) + self.rms_eps as f64).sqrt();
                    let t = ty * ow + tx;
                    for c in 0..h {
                        normed[t * h + c] = f32_to_f16((acc[c] as f64 * inv) as f32);
                    }
                }
            }
            let pb: &[u8] = unsafe {
                std::slice::from_raw_parts(normed.as_ptr() as *const u8, normed.len() * 2)
            };
            ctx.htod(&self.pooled, pb)?;
            self.emb_proj.gemm(
                ctx,
                self.pooled.ptr,
                self.out.ptr,
                n_tokens,
                self.lm_hidden,
                h,
                wsc,
            )?;

            if let Some(tr) = trace {
                let normed_named: Vec<u16> = normed.clone();
                tr.push(("pooled_normed".to_string(), normed_named));
                ctx.sync()?;
                let mut bytes = vec![0u8; n_tokens * self.lm_hidden * 2];
                ctx.dtoh(&mut bytes, &self.out)?;
                tr.push((
                    "out".to_string(),
                    bytes
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect(),
                ));
            }
            let out = ctx.alloc(n_tokens * self.lm_hidden * 2)?;
            ctx.dtod(out.ptr, self.out.ptr, n_tokens * self.lm_hidden * 2)?;
            Ok((out, n_tokens))
        }
    }

    // ===========================================================================
    // Vision GPU-vs-CPU self-test
    // ===========================================================================
    //
    // Runs a synthetic image through (a) the GPU tower with per-stage snapshots
    // and (b) a straightforward f32 CPU forward using the same checkpoint
    // weights, then reports per-stage divergence. The first stage whose relative
    // error explodes names the buggy operation. f16-vs-f32 drift alone stays
    // small (≲1e-2 by the last layer); an algorithm/layout bug shows as O(1).

    impl Gemma4 {
        pub fn vision_selftest(&mut self) -> Res<()> {
            let v = self
                .vision
                .as_ref()
                .ok_or_else(|| err!("selftest", "model has no vision tower"))?;
            let vc = self
                .cfg
                .vision
                .as_ref()
                .ok_or_else(|| err!("selftest", "no vision config"))?
                .clone();

            // Synthetic 96×96 image: gradient + bright circle (exercises DC,
            // smooth ramps and a sharp edge). 6×6 patch grid → 2×2 soft tokens.
            let (w, h) = (96usize, 96usize);
            let mut data = vec![0f32; 3 * h * w];
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - 48.0;
                    let dy = y as f32 - 48.0;
                    let inside = (dx * dx + dy * dy).sqrt() < 28.0;
                    let px = [
                        if inside { 0.9 } else { x as f32 / w as f32 },
                        if inside { 0.1 } else { y as f32 / h as f32 },
                        if inside { 0.2 } else { 0.5 },
                    ];
                    for c in 0..3 {
                        data[c * h * w + y * w + x] = 2.0 * (px[c] - 0.5);
                    }
                }
            }
            let img = ImageTensor {
                data,
                channels: 3,
                height: h,
                width: w,
            };

            // ---- GPU pass with stage snapshots ----
            let mut gpu_trace: Vec<(String, Vec<u16>)> = Vec::new();
            let wsc = self.ws.wsc.ptr;
            let (_out, n_tokens) = v.encode_traced(&self.ctx, &img, wsc, Some(&mut gpu_trace))?;
            log::info(&format!(
                "selftest: GPU encode produced {} soft tokens, {} stages traced",
                n_tokens,
                gpu_trace.len()
            ));

            // visibility: which clip bounds did the checkpoint provide?
            for suffix in ["input_min", "input_max", "output_min", "output_max"] {
                let b = host_clip_bound(
                    self.weights.as_ref(),
                    "vision_tower.encoder.layers.0.self_attn.q_proj",
                    suffix,
                );
                log::info(&format!("selftest: layers.0.q_proj.{} = {:?}", suffix, b));
            }

            // ---- CPU f32 reference forward ----
            let cpu_trace =
                vision_cpu_forward(self.weights.as_ref(), &vc, &img, self.cfg.text.hidden)?;

            // ---- compare ----
            println!("stage           max|Δ|     rel        verdict");
            let mut first_bad: Option<String> = None;
            for (name, gh) in &gpu_trace {
                let Some((_, cf)) = cpu_trace.iter().find(|(n, _)| n == name) else {
                    println!("{:<15} (no CPU counterpart)", name);
                    continue;
                };
                let n = gh.len().min(cf.len());
                let mut maxd = 0f32;
                let mut maxabs = 0f32;
                for &v in cf.iter().take(n) {
                    maxabs = maxabs.max(v.abs());
                }
                for i in 0..n {
                    let g = f16_to_f32(gh[i]);
                    let d = (g - cf[i]).abs();
                    maxd = maxd.max(d);
                }
                let maxr = maxd / (maxabs + 1e-9);
                let bad = maxr > 0.05;
                if bad && first_bad.is_none() {
                    first_bad = Some(name.clone());
                }
                println!(
                    "{:<15} {:<10.4e} {:<10.4e} {}",
                    name,
                    maxd,
                    maxr,
                    if bad { "DIVERGES" } else { "ok" }
                );
            }
            match first_bad {
                Some(s) => println!("FIRST DIVERGENT STAGE: {} — the bug lives in the ops between the previous stage and this one", s),
                None => println!("GPU tower matches the CPU f32 reference at every stage (f16 drift only)."),
            }
            Ok(())
        }
    }

    /// Read a ClippableLinear bound scalar (`{base}.input_min` etc.) if present.
    fn host_clip_bound(weights: &dyn LoadedWeights, base: &str, suffix: &str) -> Option<f32> {
        let tensors = weights.tensors();
        for root in ["model.", ""] {
            if let Some(meta) = tensors.get(&format!("{}{}.{}", root, base, suffix)) {
                let raw = weights.bytes(meta).ok()?;
                let v = match meta.dtype {
                    DType::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    DType::F16 => f16_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
                    DType::BF16 => bf16_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
                    _ => return None,
                };
                if v.is_finite() && v.abs() < 1e30 {
                    return Some(v);
                }
                return None;
            }
        }
        None
    }

    /// Mirror ClipLin: clamp(x) @ W^T then clamp, with the same checkpoint bounds.
    #[allow(clippy::too_many_arguments)]
    fn cpu_clip_mm(
        weights: &dyn LoadedWeights,
        base: &str,
        x: &[f32],
        w: &[f32],
        m: usize,
        n_out: usize,
        k: usize,
    ) -> Vec<f32> {
        let (imin, imax) = (
            host_clip_bound(weights, base, "input_min"),
            host_clip_bound(weights, base, "input_max"),
        );
        let xin: Vec<f32> = if imin.is_some() || imax.is_some() {
            let lo = imin.unwrap_or(f32::MIN);
            let hi = imax.unwrap_or(f32::MAX);
            x.iter().map(|v| v.clamp(lo, hi)).collect()
        } else {
            x.to_vec()
        };
        let mut y = cpu_mm(&xin, w, m, n_out, k);
        let (omin, omax) = (
            host_clip_bound(weights, base, "output_min"),
            host_clip_bound(weights, base, "output_max"),
        );
        if omin.is_some() || omax.is_some() {
            let lo = omin.unwrap_or(f32::MIN);
            let hi = omax.unwrap_or(f32::MAX);
            for v in y.iter_mut() {
                *v = v.clamp(lo, hi);
            }
        }
        y
    }

    /// Resolve a vision tensor to host f32, tolerating the `model.` prefix root
    /// and the ClippableLinear `.linear.` wrapping.
    /// Load the vision patch projection as `[h, 3·p·p]` f32 in the ENGINE's
    /// patch order — `(y, x, c)` pixel-major, the order the im2col emits and
    /// the order the (validated) HF flattened checkpoints ship. GGUF exports
    /// keep the raw conv kernel `[h, C, kH, kW]`, whose rows are channel-major
    /// `(c, y, x)`; uploading those bytes verbatim projects every patch
    /// through column-permuted weights (the "model describes static" bug), so
    /// conv-form rows are permuted here. Shared by the GPU tower build and
    /// the CPU reference so `vision-selftest` compares like against like.
    pub(in crate::models::gemma4) fn load_patch_proj_f32(
        weights: &dyn LoadedWeights,
        name: &str,
        h: usize,
        patch: usize,
    ) -> Res<Vec<f32>> {
        let tensors = weights.tensors();
        let key = ["model.", ""]
            .iter()
            .map(|r| format!("{}{}", r, name))
            .find(|c| tensors.contains_key(c))
            .ok_or_else(|| err!("weights", "tensor '{}' not found", name))?;
        let meta = &tensors[&key];
        let v = super::super::support::weights::decode_f32(
            meta.dtype,
            weights.bytes(meta)?,
            &meta.name,
        )?;
        let pd = 3 * patch * patch;
        if v.len() != h * pd {
            return Err(err!(
                "weights",
                "gemma4 vision: patch projection '{}' has {} elems, expected {}×{}",
                key,
                v.len(),
                h,
                pd
            ));
        }
        if meta.shape.len() != 4 {
            return Ok(v); // already flattened in engine order
        }
        let mut out = vec![0f32; v.len()];
        for o in 0..h {
            for c in 0..3 {
                for y in 0..patch {
                    for x in 0..patch {
                        out[o * pd + (y * patch + x) * 3 + c] =
                            v[o * pd + (c * patch + y) * patch + x];
                    }
                }
            }
        }
        crate::log::info(&format!(
            "gemma4 vision: conv-form patch projection '{}' permuted (c,y,x) → (y,x,c)",
            key
        ));
        Ok(out)
    }

    fn host_f32_tensor(weights: &dyn LoadedWeights, name: &str) -> Res<Vec<f32>> {
        let tensors = weights.tensors();
        let mut candidates = Vec::new();
        for root in ["model.", ""] {
            candidates.push(format!("{}{}", root, name));
            if let Some(stripped) = name.strip_suffix(".weight") {
                candidates.push(format!("{}{}.linear.weight", root, stripped));
            }
        }
        let key = candidates
            .iter()
            .find(|c| tensors.contains_key(*c))
            .ok_or_else(|| {
                err!(
                    "selftest",
                    "tensor '{}' not found (tried {:?})",
                    name,
                    candidates
                )
            })?;
        let meta = &tensors[key];
        super::super::support::weights::decode_f32(meta.dtype, weights.bytes(meta)?, key)
    }

    fn cpu_rms(x: &mut [f32], w: Option<&[f32]>, dim: usize, eps: f32) {
        for row in x.chunks_mut(dim) {
            let ms: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + eps as f64).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                let s = w.map(|w| w[i]).unwrap_or(1.0);
                *v = ((*v as f64) * inv) as f32 * s;
            }
        }
    }

    /// `y[m,n] = x[m,k] @ w[n,k]^T`
    fn cpu_mm(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for t in 0..k {
                    acc += (x[i * k + t] as f64) * (w[j * k + t] as f64);
                }
                y[i * n + j] = acc as f32;
            }
        }
        y
    }

    fn cpu_gelu_tanh(v: f32) -> f32 {
        0.5 * v * (1.0 + (0.797_884_6_f32 * (v + 0.044715 * v * v * v)).tanh())
    }

    /// CPU f32 forward of the vision tower, snapshotting the same stages as the
    /// GPU path. Mirrors the reference (numerically A/B-identical to the GPU
    /// algorithm by construction).
    fn vision_cpu_forward(
        weights: &dyn LoadedWeights,
        vc: &G4VisionCfg,
        img: &ImageTensor,
        lm_hidden: usize,
    ) -> Res<Vec<(String, Vec<f32>)>> {
        let mut trace = Vec::new();
        let h = vc.hidden;
        let patch = vc.patch;
        let (gw, gh) = (img.width / patch, img.height / patch);
        let n = gw * gh;
        let pd = 3 * patch * patch;

        // unfold (dy, dx, c)
        let mut pat = vec![0f32; n * pd];
        for py in 0..gh {
            for px in 0..gw {
                let row = py * gw + px;
                for dy in 0..patch {
                    for dx in 0..patch {
                        for c in 0..3 {
                            pat[row * pd + (dy * patch + dx) * 3 + c] = img.data
                                [(c * img.height + py * patch + dy) * img.width + px * patch + dx];
                        }
                    }
                }
            }
        }
        let w_inp = load_patch_proj_f32(
            weights,
            "vision_tower.patch_embedder.input_proj.weight",
            h,
            patch,
        )?;
        let mut x = cpu_mm(&pat, &w_inp, n, h, pd);
        let pos_tab = host_f32_tensor(
            weights,
            "vision_tower.patch_embedder.position_embedding_table",
        )?;
        for p in 0..n {
            let (xx, yy) = (p % gw, p / gw);
            for c in 0..h {
                x[p * h + c] += pos_tab[xx * h + c] + pos_tab[(vc.pos_table + yy) * h + c];
            }
        }
        trace.push(("embed".to_string(), x.clone()));

        let (nh, d) = (vc.n_heads, vc.head_dim);
        let qd = nh * d;
        for li in 0..vc.n_layers {
            let p = |s: &str| format!("vision_tower.encoder.layers.{}.{}", li, s);
            let ln = |s: &str| host_f32_tensor(weights, &p(s));
            let in_n = ln("input_layernorm.weight")?;
            let pa_n = ln("post_attention_layernorm.weight")?;
            let pf_n = ln("pre_feedforward_layernorm.weight")?;
            let po_n = ln("post_feedforward_layernorm.weight")?;
            let qn = ln("self_attn.q_norm.weight")?;
            let kn = ln("self_attn.k_norm.weight")?;
            let wq = ln("self_attn.q_proj.weight")?;
            let wk = ln("self_attn.k_proj.weight")?;
            let wv = ln("self_attn.v_proj.weight")?;
            let wo = ln("self_attn.o_proj.weight")?;
            let wg = ln("mlp.gate_proj.weight")?;
            let wu = ln("mlp.up_proj.weight")?;
            let wd = ln("mlp.down_proj.weight")?;

            let mut hn = x.clone();
            cpu_rms(&mut hn, Some(&in_n), h, vc.rms_eps);
            if li == 0 {
                trace.push(("l0_norm".to_string(), hn.clone()));
            }
            let mut q = cpu_clip_mm(weights, &p("self_attn.q_proj"), &hn, &wq, n, qd, h);
            let mut k = cpu_clip_mm(weights, &p("self_attn.k_proj"), &hn, &wk, n, qd, h);
            let mut v = cpu_clip_mm(weights, &p("self_attn.v_proj"), &hn, &wv, n, qd, h);
            if li == 0 {
                trace.push(("l0_q".to_string(), q.clone()));
                trace.push(("l0_v".to_string(), v.clone()));
            }
            cpu_rms(&mut q, Some(&qn), d, vc.rms_eps);
            cpu_rms(&mut k, Some(&kn), d, vc.rms_eps);
            cpu_rms(&mut v, None, d, vc.rms_eps);
            if li == 0 {
                trace.push(("l0_qn".to_string(), q.clone()));
            }
            // 2-D rope (mirrors k_rope2d)
            let rope = |t: &mut [f32]| {
                for p in 0..n {
                    let (px_, py_) = ((p % gw) as f32, (p / gw) as f32);
                    for hh in 0..nh {
                        let base = (p * nh + hh) * d;
                        for sd in 0..2 {
                            let pos = if sd == 0 { px_ } else { py_ };
                            for j in 0..(d / 4) {
                                let freq = vc.theta.powf(-2.0 * j as f32 / (d / 2) as f32);
                                let (c, s) = ((pos * freq).cos(), (pos * freq).sin());
                                let i0 = base + sd * (d / 2) + j;
                                let i1 = i0 + d / 4;
                                let (a, b) = (t[i0], t[i1]);
                                t[i0] = a * c - b * s;
                                t[i1] = a * s + b * c;
                            }
                        }
                    }
                }
            };
            rope(&mut q);
            rope(&mut k);
            if li == 0 {
                trace.push(("l0_qr".to_string(), q.clone()));
            }
            // bidirectional attention, scale 1.0
            let mut att = vec![0f32; n * qd];
            for hh in 0..nh {
                for qi in 0..n {
                    let qrow = &q[(qi * nh + hh) * d..(qi * nh + hh + 1) * d];
                    let mut logits = vec![0f64; n];
                    let mut m = f64::NEG_INFINITY;
                    for ki in 0..n {
                        let krow = &k[(ki * nh + hh) * d..(ki * nh + hh + 1) * d];
                        let s: f64 = qrow
                            .iter()
                            .zip(krow)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum();
                        logits[ki] = s;
                        m = m.max(s);
                    }
                    let mut l = 0f64;
                    for v_ in logits.iter_mut() {
                        *v_ = (*v_ - m).exp();
                        l += *v_;
                    }
                    for c in 0..d {
                        let mut acc = 0f64;
                        for ki in 0..n {
                            acc += logits[ki] * (v[(ki * nh + hh) * d + c] as f64);
                        }
                        att[(qi * nh + hh) * d + c] = (acc / l) as f32;
                    }
                }
            }
            if li == 0 {
                trace.push(("l0_attn".to_string(), att.clone()));
            }
            let mut ao = cpu_clip_mm(weights, &p("self_attn.o_proj"), &att, &wo, n, h, qd);
            cpu_rms(&mut ao, Some(&pa_n), h, vc.rms_eps);
            for i in 0..n * h {
                x[i] += ao[i];
            }
            if li == 0 {
                trace.push(("l0_attnres".to_string(), x.clone()));
            }
            let mut hn = x.clone();
            cpu_rms(&mut hn, Some(&pf_n), h, vc.rms_eps);
            let g = cpu_clip_mm(weights, &p("mlp.gate_proj"), &hn, &wg, n, vc.inter, h);
            let u = cpu_clip_mm(weights, &p("mlp.up_proj"), &hn, &wu, n, vc.inter, h);
            let gu: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(a, b)| cpu_gelu_tanh(*a) * b)
                .collect();
            let mut mo = cpu_clip_mm(weights, &p("mlp.down_proj"), &gu, &wd, n, h, vc.inter);
            cpu_rms(&mut mo, Some(&po_n), h, vc.rms_eps);
            for i in 0..n * h {
                x[i] += mo[i];
            }
            trace.push((format!("layer{}", li), x.clone()));
        }

        // pool + sqrt(h) + scale-less norm
        let k = vc.pool_k;
        let (ow, oh) = (gw / k, gh / k);
        let n_tokens = ow * oh;
        let scale = (h as f32).sqrt() / (k * k) as f32;
        let mut pooled = vec![0f32; n_tokens * h];
        for ty in 0..oh {
            for tx in 0..ow {
                for dy in 0..k {
                    for dx in 0..k {
                        let p = (ty * k + dy) * gw + (tx * k + dx);
                        for c in 0..h {
                            pooled[(ty * ow + tx) * h + c] += x[p * h + c];
                        }
                    }
                }
            }
        }
        for v in pooled.iter_mut() {
            *v *= scale;
        }
        // Probe by key first: constructing the not-found EngineError logs an
        // ERROR line, and these buffers being absent is the normal case.
        let has = |n: &str| {
            let t = weights.tensors();
            t.contains_key(n) || t.contains_key(&format!("model.{}", n))
        };
        if has("vision_tower.std_bias") && has("vision_tower.std_scale") {
            let (bias, sc) = (
                host_f32_tensor(weights, "vision_tower.std_bias")?,
                host_f32_tensor(weights, "vision_tower.std_scale")?,
            );
            for (i, v) in pooled.iter_mut().enumerate() {
                *v = (*v - bias[i % h]) * sc[i % h];
            }
        }
        cpu_rms(&mut pooled, None, h, vc.rms_eps);
        trace.push(("pooled_normed".to_string(), pooled.clone()));
        let w_emb = host_f32_tensor(weights, "embed_vision.embedding_projection.weight")?;
        let out = cpu_mm(&pooled, &w_emb, n_tokens, lm_hidden, h);
        trace.push(("out".to_string(), out));
        Ok(trace)
    }
}

pub(crate) mod audio {
    //! Audio tower: log-mel frontend, conv subsampling, and the chunked
    //! local-attention conformer.

    use crate::cuda::{CudaCtx, DeviceBuf};
    use crate::err;

    use crate::log;
    use crate::traits::{AudioPcm, Res};

    use crate::quant::bnb::WTensor;

    use super::super::support::config::*;
    use super::super::support::weights::*;
    use crate::num::*;

    // ===========================================================================
    // Audio tower (USM-style conformer)
    // ===========================================================================

    const MEL_FRAME: usize = 320; // 20 ms @ 16 kHz
    const MEL_HOP: usize = 160; // 10 ms
    const MEL_FFT: usize = 512;
    const AUDIO_MAX_TOKENS: usize = 750; // audio_seq_length cap

    struct G4AudLayer {
        // feed_forward1 / feed_forward2
        ff1_pre: DeviceBuf,
        ff1_w1: ClipLin,
        ff1_w2: ClipLin,
        ff1_post: DeviceBuf,
        ff2_pre: DeviceBuf,
        ff2_w1: ClipLin,
        ff2_w2: ClipLin,
        ff2_post: DeviceBuf,
        // attention
        norm_pre_attn: DeviceBuf,
        norm_post_attn: DeviceBuf,
        q: ClipLin,
        k: ClipLin,
        v: ClipLin,
        post: ClipLin,
        relk_w: WTensor,   // relative_k_proj [hidden, hidden]
        qscale: DeviceBuf, // tiled q_scale·softplus(per_dim_scale) [hidden]
        // light conv
        lc_pre: DeviceBuf,
        lc_start: ClipLin,
        lc_dw: DeviceBuf, // depthwise [hidden, K]
        lc_norm: DeviceBuf,
        lc_end: ClipLin,
        norm_out: DeviceBuf,
    }

    pub struct G4Audio {
        // CPU-side subsampling stack (host f32: tiny tensors, heavy data motion)
        conv0_w: Vec<f32>,   // [128, 1, 3, 3]
        conv0_ln: Vec<f32>,  // [128]
        conv1_w: Vec<f32>,   // [32, 128, 3, 3]
        conv1_ln: Vec<f32>,  // [32]
        input_proj: WTensor, // [hidden, (128/4)*32]
        relpe: DeviceBuf,    // sinusoidal rel-pos table [past+1, hidden]
        relk: DeviceBuf,     // per-layer scratch: relpe @ relative_k_proj^T
        layers: Vec<G4AudLayer>,
        out_proj: WTensor,   // [out_dims, hidden]
        out_bias: DeviceBuf, // [out_dims]
        emb_proj: WTensor,   // embed_audio.embedding_projection [lm_hidden, out_dims]
        // scratch sized for AUDIO_MAX_TOKENS (+chunk padding)
        sx: DeviceBuf,
        sh: DeviceBuf,
        sh2: DeviceBuf,
        sg: DeviceBuf, // [n, 4·hidden] FF / lconv doublewide scratch
        sq: DeviceBuf,
        sk: DeviceBuf,
        sv: DeviceBuf,
        sclip: DeviceBuf,
        sout: DeviceBuf, // [n, out_dims]
        hidden: usize,
        lm_hidden: usize,
        out_dims: usize,
        n_heads: usize,
        chunk: usize,
        past: usize,
        logit_cap: f32,
        conv_k: usize,
        k_scale: f32,
        /// `attention_invalid_logits_value` — padded context slots enter the
        /// softmax with this logit (reference dilution semantics), not -inf.
        invalid_logit: f32,
        residual_w: f32,
        rms_eps: f32,
        sub_ch: [usize; 2],
        mels: usize,
    }

    impl G4Audio {
        pub(in crate::models::gemma4) fn build(
            ctx: &CudaCtx,
            ac: &G4AudioCfg,
            ix: &G4Index,
            lm_hidden: usize,
        ) -> Res<G4Audio> {
            let h = ac.hidden;
            let d = h / ac.n_heads;
            let at = |s: &str| format!("audio_tower.{}", s);

            // ---- host-resident subsampling convs ----
            let conv0_w = ix.host_f32(
                &at("subsample_conv_projection.layer0.conv.weight"),
                &[ac.sub_ch[0], 1, 3, 3],
            )?;
            let conv0_ln = ix.host_f32(
                &at("subsample_conv_projection.layer0.norm.weight"),
                &[ac.sub_ch[0]],
            )?;
            let conv1_w = ix.host_f32(
                &at("subsample_conv_projection.layer1.conv.weight"),
                &[ac.sub_ch[1], ac.sub_ch[0], 3, 3],
            )?;
            let conv1_ln = ix.host_f32(
                &at("subsample_conv_projection.layer1.norm.weight"),
                &[ac.sub_ch[1]],
            )?;
            let proj_in = (ac.sub_ch[0] / 4) * ac.sub_ch[1];

            // ---- sinusoidal relative position table (positions past..0) ----
            let n_rel = ac.past + 1;
            let half = h / 2;
            let mut pe: Vec<u16> = vec![0; n_rel * h];
            let log_inc = (10_000f32.ln() - 0.0) / (half.max(2) - 1) as f32;
            for r in 0..n_rel {
                let p = (ac.past - r) as f32; // positions context/2 .. 0
                for i in 0..half {
                    let t = p * (-(i as f32) * log_inc).exp();
                    pe[r * h + i] = f32_to_f16(t.sin());
                    pe[r * h + half + i] = f32_to_f16(t.cos());
                }
            }
            let pe_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(pe.as_ptr() as *const u8, pe.len() * 2) };
            let relpe = ctx.alloc(n_rel * h * 2)?;
            ctx.htod(&relpe, pe_bytes)?;

            let q_scale = (d as f32).powf(-0.5) / std::f32::consts::LN_2;
            let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;

            let mut layers = Vec::with_capacity(ac.n_layers);
            for i in 0..ac.n_layers {
                let p = |s: &str| at(&format!("layers.{}.{}", i, s));
                // q_scale · softplus(per_dim_scale), tiled across heads.
                //
                // Exporter quirk (found by `cima audio-map` cosine matching:
                // gguf-vs-safetensors cos ≈ −0.8, not +1): the gguf mmproj
                // ships per_dim_scale with softplus ALREADY APPLIED, while HF
                // checkpoints ship the raw parameter. Applying softplus again
                // poisons every attention layer. Softplus output is strictly
                // positive and raw trained scales carry negatives, so the two
                // conventions are distinguished by value, not by format.
                let pds = ix.host_f32(&p("self_attn.per_dim_scale"), &[d])?;
                // Serialization convention, MEASURED by `cima audio-map` (the
                // gguf/softplus(raw) ratio printed 1.0000000 for every dim of
                // every layer): HF checkpoints ship the raw parameter (has
                // negatives) and gguf exports ship exactly softplus(raw) —
                // nothing else folded. Both therefore take the q_scale factor
                // here; only the softplus is conditional.
                let pre_activated = pds.iter().all(|v| *v > 0.0);
                if i == 0 && std::env::var("CIMA_G4_DEBUG").is_ok() {
                    eprintln!(
                        "g4 audio per_dim_scale: first vals {:?} → {} convention",
                        &pds[..4.min(pds.len())],
                        if pre_activated {
                            "softplus(raw) as shipped (gguf)"
                        } else {
                            "raw (softplus applied here)"
                        }
                    );
                }
                let mut tiled: Vec<u16> = Vec::with_capacity(h);
                for _ in 0..ac.n_heads {
                    for v in &pds {
                        let sp = if pre_activated || *v > 20.0 {
                            *v
                        } else {
                            (1.0 + v.exp()).ln()
                        };
                        tiled.push(f32_to_f16(q_scale * sp));
                    }
                }
                let tb: &[u8] = unsafe {
                    std::slice::from_raw_parts(tiled.as_ptr() as *const u8, tiled.len() * 2)
                };
                let qscale = ctx.alloc(h * 2)?;
                ctx.htod(&qscale, tb)?;

                // depthwise conv weight [h,1,K] -> [h,K]
                let dw = ix.meta(&p("lconv1d.depthwise_conv1d.weight"))?;
                if !super::super::support::weights::shape_view_ok(&dw.shape, &[h, 1, ac.conv_k]) {
                    return Err(err!(
                        "weights",
                        "gemma4 audio: depthwise conv has shape {:?}, expected [{}, 1, {}]",
                        dw.shape,
                        h,
                        ac.conv_k
                    ));
                }
                let dwf = ix.host_f32(&p("lconv1d.depthwise_conv1d.weight"), &[])?;
                let dwh: Vec<u16> = dwf.iter().map(|v| f32_to_f16(*v)).collect();
                let db: &[u8] =
                    unsafe { std::slice::from_raw_parts(dwh.as_ptr() as *const u8, dwh.len() * 2) };
                let lc_dw = ctx.alloc(h * ac.conv_k * 2)?;
                ctx.htod(&lc_dw, db)?;

                layers.push(G4AudLayer {
                    ff1_pre: ix.upload(&p("feed_forward1.pre_layer_norm.weight"), &[h])?,
                    ff1_w1: ix.upload_clip(&p("feed_forward1.ffw_layer_1"), h * 4, h)?,
                    ff1_w2: ix.upload_clip(&p("feed_forward1.ffw_layer_2"), h, h * 4)?,
                    ff1_post: ix.upload(&p("feed_forward1.post_layer_norm.weight"), &[h])?,
                    ff2_pre: ix.upload(&p("feed_forward2.pre_layer_norm.weight"), &[h])?,
                    ff2_w1: ix.upload_clip(&p("feed_forward2.ffw_layer_1"), h * 4, h)?,
                    ff2_w2: ix.upload_clip(&p("feed_forward2.ffw_layer_2"), h, h * 4)?,
                    ff2_post: ix.upload(&p("feed_forward2.post_layer_norm.weight"), &[h])?,
                    norm_pre_attn: ix.upload(&p("norm_pre_attn.weight"), &[h])?,
                    norm_post_attn: ix.upload(&p("norm_post_attn.weight"), &[h])?,
                    q: ix.upload_clip(&p("self_attn.q_proj"), h, h)?,
                    k: ix.upload_clip(&p("self_attn.k_proj"), h, h)?,
                    v: ix.upload_clip(&p("self_attn.v_proj"), h, h)?,
                    post: ix.upload_clip(&p("self_attn.post"), h, h)?,
                    relk_w: ix.upload_w(&p("self_attn.relative_k_proj.weight"), h, h)?,
                    qscale,
                    lc_pre: ix.upload(&p("lconv1d.pre_layer_norm.weight"), &[h])?,
                    lc_start: ix.upload_clip(&p("lconv1d.linear_start"), h * 2, h)?,
                    lc_dw,
                    lc_norm: ix.upload(&p("lconv1d.conv_norm.weight"), &[h])?,
                    lc_end: ix.upload_clip(&p("lconv1d.linear_end"), h, h)?,
                    norm_out: ix.upload(&p("norm_out.weight"), &[h])?,
                });
            }

            let n = AUDIO_MAX_TOKENS + ac.chunk; // chunk padding headroom
            Ok(G4Audio {
                conv0_w,
                conv0_ln,
                conv1_w,
                conv1_ln,
                input_proj: ix.upload_w(
                    &at("subsample_conv_projection.input_proj_linear.weight"),
                    h,
                    proj_in,
                )?,
                relpe,
                relk: ctx.alloc((ac.past + 1) * h * 2)?,
                layers,
                out_proj: ix.upload_w(&at("output_proj.weight"), ac.out_dims, h)?,
                out_bias: ix.upload(&at("output_proj.bias"), &[ac.out_dims])?,
                emb_proj: ix.upload_w(
                    "embed_audio.embedding_projection.weight",
                    lm_hidden,
                    ac.out_dims,
                )?,
                sx: ctx.alloc(n * h * 2)?,
                sh: ctx.alloc(n * h * 2)?,
                sh2: ctx.alloc(n * h * 2)?,
                sg: ctx.alloc(n * h * 4 * 2)?,
                sq: ctx.alloc(n * h * 2)?,
                sk: ctx.alloc(n * h * 2)?,
                sv: ctx.alloc(n * h * 2)?,
                sclip: ctx.alloc(n * h * 4 * 2)?,
                sout: ctx.alloc(n * ac.out_dims.max(lm_hidden) * 2)?,
                hidden: h,
                lm_hidden,
                out_dims: ac.out_dims,
                n_heads: ac.n_heads,
                chunk: ac.chunk,
                past: ac.past,
                logit_cap: ac.logit_cap,
                invalid_logit: ac.invalid_logit,
                conv_k: ac.conv_k,
                k_scale,
                residual_w: ac.residual_w,
                rms_eps: ac.rms_eps,
                sub_ch: ac.sub_ch,
                mels: ac.mels,
            })
        }
    }

    impl G4Audio {
        /// Log-mel frontend (CPU, zero-dependency radix-2 FFT): 128 HTK mels,
        /// 20 ms periodic-Hann frames at 10 ms hop, semicausal padding,
        /// `log(mel + 1e-3)`.
        fn log_mel(&self, pcm: &AudioPcm) -> Res<Vec<f32>> {
            if pcm.sample_rate != 16_000 {
                return Err(err!(
                    "media",
                    "gemma4 audio: PCM must be 16 kHz (got {})",
                    pcm.sample_rate
                ));
            }
            // Cap input so subsampled tokens stay within audio_seq_length.
            let max_frames = AUDIO_MAX_TOKENS * 4;
            let max_samples = max_frames * MEL_HOP + MEL_FRAME;
            let samples: &[f32] = if pcm.samples.len() > max_samples {
                log::warn(&format!(
                    "gemma4 audio: clip of {} samples truncated to {} ({} soft tokens cap)",
                    pcm.samples.len(),
                    max_samples,
                    AUDIO_MAX_TOKENS
                ));
                &pcm.samples[..max_samples]
            } else {
                &pcm.samples
            };

            // Semicausal pad: frame_length/2 zeros up front.
            let mut padded = vec![0f32; MEL_FRAME / 2];
            padded.extend_from_slice(samples);
            if padded.len() < MEL_FRAME {
                padded.resize(MEL_FRAME, 0.0);
            }
            let n_frames = (padded.len() - MEL_FRAME) / MEL_HOP + 1;

            // Periodic Hann window.
            let window: Vec<f32> = (0..MEL_FRAME)
                .map(|i| {
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / MEL_FRAME as f32).cos()
                })
                .collect();

            // HTK mel filter bank (norm=None) over 257 bins, 0..8000 Hz.
            let bins = MEL_FFT / 2 + 1;
            let mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
            let imel = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);
            let (mlo, mhi) = (mel(0.0), mel(8000.0));
            let centers: Vec<f32> = (0..self.mels + 2)
                .map(|i| imel(mlo + (mhi - mlo) * i as f32 / (self.mels + 1) as f32))
                .collect();
            let bin_hz = 16_000.0 / MEL_FFT as f32;

            let mut out = vec![0f32; n_frames * self.mels];
            let mut re = vec![0f32; MEL_FFT];
            let mut im = vec![0f32; MEL_FFT];
            let mut mag = vec![0f32; bins];
            for fr in 0..n_frames {
                let s0 = fr * MEL_HOP;
                for i in 0..MEL_FFT {
                    re[i] = if i < MEL_FRAME {
                        padded[s0 + i] * window[i]
                    } else {
                        0.0
                    };
                    im[i] = 0.0;
                }
                fft_radix2(&mut re, &mut im);
                for b in 0..bins {
                    mag[b] = (re[b] * re[b] + im[b] * im[b]).sqrt();
                }
                for m in 0..self.mels {
                    let (l, c, r) = (centers[m], centers[m + 1], centers[m + 2]);
                    let mut acc = 0f32;
                    let b0 = (l / bin_hz).floor().max(0.0) as usize;
                    let b1 = ((r / bin_hz).ceil() as usize).min(bins - 1);
                    // b is a frequency-bin index; f = b*bin_hz needs it.
                    #[allow(clippy::needless_range_loop)]
                    for b in b0..=b1 {
                        let f = b as f32 * bin_hz;
                        let w = if f < l || f > r {
                            0.0
                        } else if f <= c {
                            if c > l {
                                (f - l) / (c - l)
                            } else {
                                0.0
                            }
                        } else if r > c {
                            (r - f) / (r - c)
                        } else {
                            0.0
                        };
                        acc += w * mag[b];
                    }
                    out[fr * self.mels + m] = (acc + 1e-3).ln();
                }
            }
            Ok(out)
        }

        /// CPU subsampling stack: two 3×3 stride-2 bias-less convs, each followed
        /// by channel LayerNorm (no bias) and ReLU; then flatten `[T/4, F·C]`
        /// (frequency-major) for the GPU input projection.
        fn subsample(&self, mel: &[f32], n_frames: usize) -> (Vec<f32>, usize) {
            let conv = |inp: &[f32],
                        ic: usize,
                        t: usize,
                        f: usize,
                        w: &[f32],
                        oc: usize,
                        ln: &[f32],
                        eps: f32|
             -> (Vec<f32>, usize, usize) {
                let (ot, of) = (t.div_ceil(2), f.div_ceil(2));
                let mut out = vec![0f32; oc * ot * of];
                for o in 0..oc {
                    for ty in 0..ot {
                        for tx in 0..of {
                            let mut acc = 0f32;
                            for i in 0..ic {
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let sy = (ty * 2 + ky) as isize - 1;
                                        let sx = (tx * 2 + kx) as isize - 1;
                                        if sy >= 0
                                            && (sy as usize) < t
                                            && sx >= 0
                                            && (sx as usize) < f
                                        {
                                            acc += w[((o * ic + i) * 3 + ky) * 3 + kx]
                                                * inp[(i * t + sy as usize) * f + sx as usize];
                                        }
                                    }
                                }
                            }
                            out[(o * ot + ty) * of + tx] = acc;
                        }
                    }
                }
                // LayerNorm over channels at each (t, f), weight only.
                let mut normed = vec![0f32; oc * ot * of];
                for ty in 0..ot {
                    for tx in 0..of {
                        let mut mean = 0f32;
                        for o in 0..oc {
                            mean += out[(o * ot + ty) * of + tx];
                        }
                        mean /= oc as f32;
                        let mut var = 0f32;
                        for o in 0..oc {
                            let d = out[(o * ot + ty) * of + tx] - mean;
                            var += d * d;
                        }
                        var /= oc as f32;
                        let inv = 1.0 / (var + eps).sqrt();
                        for o in 0..oc {
                            let v = (out[(o * ot + ty) * of + tx] - mean) * inv * ln[o];
                            normed[(o * ot + ty) * of + tx] = v.max(0.0); // ReLU
                        }
                    }
                }
                (normed, ot, of)
            };

            let (h1, t1, f1) = conv(
                mel,
                1,
                n_frames,
                self.mels,
                &self.conv0_w,
                self.sub_ch[0],
                &self.conv0_ln,
                self.rms_eps,
            );
            let (h2, t2, f2) = conv(
                &h1,
                self.sub_ch[0],
                t1,
                f1,
                &self.conv1_w,
                self.sub_ch[1],
                &self.conv1_ln,
                self.rms_eps,
            );
            // [C, T, F] -> [T, F·C] with frequency major (reference permute(0,2,3,1)).
            let c = self.sub_ch[1];
            let mut flat = vec![0f32; t2 * f2 * c];
            for t in 0..t2 {
                for f in 0..f2 {
                    for ch in 0..c {
                        flat[t * (f2 * c) + f * c + ch] = h2[(ch * t2 + t) * f2 + f];
                    }
                }
            }
            (flat, t2)
        }

        /// Encode one audio clip to `[n, lm_hidden]` soft tokens on the device.
        pub(in crate::models::gemma4) fn encode(
            &self,
            ctx: &CudaCtx,
            pcm: &AudioPcm,
            wsc: u64,
        ) -> Res<(DeviceBuf, usize)> {
            let mel = self.log_mel(pcm)?;
            let n_frames = mel.len() / self.mels;
            if n_frames < 4 {
                return Err(err!(
                    "media",
                    "gemma4 audio: clip too short ({} mel frames; need >= 4)",
                    n_frames
                ));
            }
            let (flat, n) = self.subsample(&mel, n_frames);
            // CIMA_DUMP_MEL=base: frontend bisection dumps — `.mel` (the log-mel
            // matrix) and `.sub4` (post-subsample, pre-projection), each
            // `[u32 rows][u32 cols][f32 data]`, comparable against the reference
            // processor's input_features and a SubSampleConvProjection hook.
            if let Ok(base) = std::env::var("CIMA_DUMP_MEL") {
                for (tag, rows, cols, data) in [
                    ("mel", n_frames, self.mels, &mel),
                    ("sub4", n, (self.sub_ch[0] / 4) * self.sub_ch[1], &flat),
                ] {
                    let mut out = Vec::with_capacity(8 + rows * cols * 4);
                    out.extend_from_slice(&(rows as u32).to_le_bytes());
                    out.extend_from_slice(&(cols as u32).to_le_bytes());
                    for v in data.iter().take(rows * cols) {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    let path = format!("{}.{}", base, tag);
                    std::fs::write(&path, out)
                        .map_err(|e| err!("debug", "cannot write '{}': {}", path, e))?;
                    log::info(&format!(
                        "audio {} dumped to {} ({}×{})",
                        tag, path, rows, cols
                    ));
                }
            }
            let n = n.min(AUDIO_MAX_TOKENS);
            let h = self.hidden;
            let proj_in = (self.sub_ch[0] / 4) * self.sub_ch[1];

            // Upload [n, proj_in] f16 and project to the conformer width.
            let mut host: Vec<u16> = Vec::with_capacity(n * proj_in);
            for v in flat.iter().take(n * proj_in) {
                host.push(f32_to_f16(*v));
            }
            let hb: &[u8] =
                unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
            ctx.htod(&self.sh, hb)?;
            self.input_proj
                .gemm(ctx, self.sh.ptr, self.sx.ptr, n, h, proj_in, wsc)?;
            // `.sub4p`: POST-projection conformer input — the apples-to-apples
            // point against the reference's SubSampleConvProjection output
            // (its name includes the projection; the pre-projection flatten and
            // the conformer width are both 1024 by coincidence — different
            // spaces, cosine ~0, no permutation can relate them).
            if let Ok(base) = std::env::var("CIMA_DUMP_MEL") {
                super::super::dump_f16_matrix(
                    ctx,
                    self.sx.ptr,
                    n,
                    h,
                    &format!("{}.sub4p", base),
                    "audio sub4p (post-projection)",
                )?;
            }

            let d = h / self.n_heads;
            for layer in &self.layers {
                // relative keys for this layer: relpe @ relative_k_proj^T
                layer
                    .relk_w
                    .gemm(ctx, self.relpe.ptr, self.relk.ptr, self.past + 1, h, h, wsc)?;

                // ---- feed_forward1 (½ residual) ----
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.ff1_pre.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                layer.ff1_w1.run(
                    ctx,
                    self.sh.ptr,
                    self.sclip.ptr,
                    self.sg.ptr,
                    n,
                    h,
                    h * 4,
                    wsc,
                )?;
                ctx.silu(self.sg.ptr, n * h * 4)?;
                layer.ff1_w2.run(
                    ctx,
                    self.sg.ptr,
                    self.sclip.ptr,
                    self.sh2.ptr,
                    n,
                    h * 4,
                    h,
                    wsc,
                )?;
                ctx.rmsnorm(
                    self.sh2.ptr,
                    layer.ff1_post.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.scalemul(self.sh2.ptr, self.residual_w, n * h)?;
                ctx.add(self.sx.ptr, self.sh2.ptr, n * h)?;

                // ---- chunked local attention with relative bias ----
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.norm_pre_attn.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                layer
                    .q
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sq.ptr, n, h, h, wsc)?;
                layer
                    .k
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sk.ptr, n, h, h, wsc)?;
                layer
                    .v
                    .run(ctx, self.sh.ptr, self.sclip.ptr, self.sv.ptr, n, h, h, wsc)?;
                ctx.mulvec(self.sq.ptr, layer.qscale.ptr, n, h)?;
                ctx.scalemul(self.sk.ptr, self.k_scale, n * h)?;
                ctx.audio_attn(
                    self.sq.ptr,
                    self.sk.ptr,
                    self.sv.ptr,
                    self.relk.ptr,
                    self.sh2.ptr,
                    n,
                    self.n_heads,
                    d,
                    self.chunk,
                    self.past,
                    self.logit_cap,
                    self.invalid_logit,
                )?;
                layer
                    .post
                    .run(ctx, self.sh2.ptr, self.sclip.ptr, self.sh.ptr, n, h, h, wsc)?;
                ctx.rmsnorm(
                    self.sh.ptr,
                    layer.norm_post_attn.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.add(self.sx.ptr, self.sh.ptr, n * h)?;

                // ---- light conv (GLU + causal depthwise) ----
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.lc_pre.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                layer.lc_start.run(
                    ctx,
                    self.sh.ptr,
                    self.sclip.ptr,
                    self.sg.ptr,
                    n,
                    h,
                    h * 2,
                    wsc,
                )?;
                ctx.glu(self.sg.ptr, self.sh.ptr, n, h)?;
                ctx.dwconv1d(
                    self.sh.ptr,
                    layer.lc_dw.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.conv_k,
                )?;
                ctx.rmsnorm(
                    self.sh2.ptr,
                    layer.lc_norm.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.silu(self.sh2.ptr, n * h)?;
                layer
                    .lc_end
                    .run(ctx, self.sh2.ptr, self.sclip.ptr, self.sh.ptr, n, h, h, wsc)?;
                ctx.add(self.sx.ptr, self.sh.ptr, n * h)?;

                // ---- feed_forward2 (½ residual) ----
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.ff2_pre.ptr,
                    self.sh.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                layer.ff2_w1.run(
                    ctx,
                    self.sh.ptr,
                    self.sclip.ptr,
                    self.sg.ptr,
                    n,
                    h,
                    h * 4,
                    wsc,
                )?;
                ctx.silu(self.sg.ptr, n * h * 4)?;
                layer.ff2_w2.run(
                    ctx,
                    self.sg.ptr,
                    self.sclip.ptr,
                    self.sh2.ptr,
                    n,
                    h * 4,
                    h,
                    wsc,
                )?;
                ctx.rmsnorm(
                    self.sh2.ptr,
                    layer.ff2_post.ptr,
                    self.sh2.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
                ctx.scalemul(self.sh2.ptr, self.residual_w, n * h)?;
                ctx.add(self.sx.ptr, self.sh2.ptr, n * h)?;

                // ---- per-layer output norm ----
                ctx.rmsnorm(
                    self.sx.ptr,
                    layer.norm_out.ptr,
                    self.sx.ptr,
                    n,
                    h,
                    self.rms_eps,
                )?;
            }

            // ---- output projection (+bias) and the multimodal embedder ----
            self.out_proj
                .gemm(ctx, self.sx.ptr, self.sout.ptr, n, self.out_dims, h, wsc)?;
            ctx.bias(self.sout.ptr, self.out_bias.ptr, n, self.out_dims)?;
            ctx.rmsnorm(
                self.sout.ptr,
                0,
                self.sout.ptr,
                n,
                self.out_dims,
                self.rms_eps,
            )?;
            let out = ctx.alloc(n * self.lm_hidden * 2)?;
            self.emb_proj.gemm(
                ctx,
                self.sout.ptr,
                out.ptr,
                n,
                self.lm_hidden,
                self.out_dims,
                wsc,
            )?;
            Ok((out, n))
        }
    }

    /// In-place radix-2 complex FFT (`n` must be a power of two).
    fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        debug_assert!(n.is_power_of_two() && im.len() == n);
        // bit reversal
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f32::consts::PI / len as f32;
            let (wr, wi) = (ang.cos(), ang.sin());
            let mut i = 0;
            while i < n {
                let (mut cr, mut ci) = (1f32, 0f32);
                for k in 0..len / 2 {
                    let (ur, ui) = (re[i + k], im[i + k]);
                    let (vr, vi) = (
                        re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                        re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                    );
                    re[i + k] = ur + vr;
                    im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr;
                    im[i + k + len / 2] = ui - vi;
                    let ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
                i += len;
            }
            len <<= 1;
        }
    }
}
