//! # quant — weight quantization codecs
//!
//! A codec turns a stored tensor encoding into something the kernels can
//! execute: either dequantized f16 at load time or a custom GEMV at decode
//! time (see `k_nf4_gemv`). Container formats (how tensors are *stored on
//! disk*, e.g. safetensors) live in `crate::formats`; this module is about
//! how the *bytes inside a tensor* are encoded.
//!
//! Implemented:
//! * [`bnb`] — bitsandbytes NF4/FP4 block quantization (the format of
//!   `*-bnb-4bit` Hub checkpoints): packed nibbles + per-block absmax +
//!   a 16-entry code table.
//!
//! Adding a codec means implementing `traits::WeightCodec` and, for decode
//! performance, a warp-per-row GEMV in `kernels.cu` following the house
//! pattern (`k_gemv_f16` / `k_nf4_gemv`) — see README § Extending.

pub mod bnb {
    //! # quant — bitsandbytes 4-bit weight execution
    //!
    //! Implements the `bitsandbytes` NF4/FP4 serialization as saved by
    //! `transformers` (including unsloth's "dynamic" checkpoints where only a
    //! subset of the linears is quantized). A quantized `nn.Linear` weight `W`
    //! appears in the safetensors container as a *family* of tensors:
    //!
    //! | tensor                                   | dtype | content |
    //! |------------------------------------------|-------|---------|
    //! | `W`                                      | U8    | packed nibbles, shape `[numel/2, 1]` (high nibble = even index) |
    //! | `W.absmax`                               | U8/F32| per-`blocksize` scale (U8 when double-quantized) |
    //! | `W.quant_map`                            | F32   | the 16-entry NF4/FP4 codebook |
    //! | `W.nested_absmax`                        | F32   | per-`nested_blocksize` scale of the absmax |
    //! | `W.nested_quant_map`                     | F32   | 256-entry codebook for the absmax |
    //! | `W.quant_state.bitsandbytes__{nf4,fp4}`  | U8    | JSON: blocksize, shape, quant_type, nested_offset, … |
    //!
    //! Dequantization (`bitsandbytes.functional.dequantize_4bit`):
    //!
    //! ```text
    //! absmax[i] = nested_quant_map[absmax_u8[i]] · nested_absmax[i / nested_blocksize] + nested_offset
    //! w[idx]    = quant_map[nibble(idx)] · absmax[idx / blocksize]      (row-major [out, in])
    //! ```
    //!
    //! **VRAM strategy**: quantized weights stay resident as packed nibbles plus
    //! a small f32 absmax vector (the double-quant layer is folded on the host at
    //! load). GEMMs route through [`WTensor::gemm`]: decode steps (`m == 1`) use
    //! the fused `nf4_gemv` kernel (reads 4× less weight memory than f16 — the
    //! decode path is bandwidth-bound, so this is *faster* than dequantizing);
    //! prefill and the towers dequantize tile-free into a per-pipeline scratch
    //! buffer and run the regular f16 cuBLAS GEMM.

    use crate::cuda::{CudaCtx, DeviceBuf};
    use crate::err;
    use crate::json::{self, Json};
    use crate::traits::{DType, LoadedWeights, Res, TensorMeta};

    /// Parsed `W.quant_state.bitsandbytes__*` metadata.
    pub struct BnbState {
        pub blocksize: usize,
        pub rows: usize,
        pub cols: usize,
        pub nested_blocksize: usize,
        pub nested_offset: f32,
    }

    /// A 4-bit weight resident on the device: packed nibbles + folded f32 absmax
    /// + the 16-entry codebook. Logical layout is row-major `[rows, cols]`.
    pub struct BnbNf4 {
        pub packed: DeviceBuf, // [rows*cols/2] u8
        pub absmax: DeviceBuf, // [rows*cols/blocksize] f32 (double-quant folded)
        pub qmap: DeviceBuf,   // [16] f32
        pub rows: usize,
        pub cols: usize,
        pub blocksize: usize,
    }

    /// A linear weight, either plain f16 or packed 4-bit.
    pub enum WTensor {
        F16(DeviceBuf),
        Nf4(BnbNf4),
        /// ggml block-quantized weight, resident as stored (see quant::gguf):
        /// decode runs the fused dp4a GEMV, prefill dequantizes row slabs
        /// through the shared scratch into cuBLAS.
        Gguf {
            buf: DeviceBuf,
            fmt: crate::traits::DType,
            n: usize,
            k: usize,
        },
    }

    impl WTensor {
        /// `y[m, n] = x[m, k] @ W[n, k]^T`, dispatching on residency. `scratch`
        /// must hold `n·k` f16 elements when any quantized weight flows through
        /// the prefill (`m > 1`) path; it is untouched for f16 weights and for
        /// the fused decode path.
        pub fn gemm(
            &self,
            ctx: &CudaCtx,
            x: u64,
            y: u64,
            m: usize,
            n: usize,
            k: usize,
            scratch: u64,
        ) -> Res<()> {
            match self {
                WTensor::F16(w) => ctx.gemm_f16(x, w.ptr, y, m, n, k),
                WTensor::Gguf {
                    buf,
                    fmt,
                    n: wn,
                    k: wk,
                } => {
                    if *wn != n || *wk != k {
                        return Err(err!(
                        "quant",
                        "gguf weight is [{}, {}] but the GEMM expects [{}, {}] — graph/checkpoint mismatch",
                        wn, wk, n, k
                    ));
                    }
                    if m == 1 {
                        ctx.gguf_gemv(*fmt, x, buf.ptr, 0, y, n, k, 0)
                    } else {
                        if scratch == 0 {
                            return Err(err!("quant", "gguf prefill GEMM requires a dequant scratch buffer (internal wiring bug)"));
                        }
                        let elems = crate::traits::block_elems(*fmt);
                        ctx.gguf_dequant(*fmt, buf.ptr, scratch, n * k / elems)?;
                        ctx.gemm_f16(x, scratch, y, m, n, k)
                    }
                }
                WTensor::Nf4(q) => {
                    if q.rows != n || q.cols != k {
                        return Err(err!(
                        "quant",
                        "nf4 weight is [{}, {}] but the GEMM expects [{}, {}] — graph/checkpoint mismatch",
                        q.rows, q.cols, n, k
                    ));
                    }
                    if m == 1 {
                        ctx.nf4_gemv(
                            x,
                            q.packed.ptr,
                            q.absmax.ptr,
                            q.qmap.ptr,
                            y,
                            n,
                            k,
                            q.blocksize,
                        )
                    } else {
                        if scratch == 0 {
                            return Err(err!("quant", "nf4 prefill GEMM requires a dequant scratch buffer (internal wiring bug)"));
                        }
                        ctx.nf4_dequant(
                            q.packed.ptr,
                            q.absmax.ptr,
                            q.qmap.ptr,
                            scratch,
                            n * k,
                            q.blocksize,
                        )?;
                        ctx.gemm_f16(x, scratch, y, m, n, k)
                    }
                }
            }
        }
    }

    /// Suffix of the quant-state sibling, if `name` is a bnb-packed weight.
    pub fn state_name(
        tensors: &std::collections::HashMap<String, TensorMeta>,
        name: &str,
    ) -> Option<String> {
        for qt in ["nf4", "fp4"] {
            let n = format!("{}.quant_state.bitsandbytes__{}", name, qt);
            if tensors.contains_key(&n) {
                return Some(n);
            }
        }
        None
    }

    /// `true` if `name` is one of the bnb auxiliary tensors that are consumed by
    /// the dequant path and must never be uploaded or counted as device weights.
    pub fn is_aux(name: &str) -> bool {
        name.ends_with(".absmax")
            || name.ends_with(".quant_map")
            || name.ends_with(".nested_absmax")
            || name.ends_with(".nested_quant_map")
            || name.contains(".quant_state.bitsandbytes__")
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Parse the serialized quant state for packed weight `name`.
    pub fn parse_state(weights: &dyn LoadedWeights, name: &str) -> Res<BnbState> {
        let sname = state_name(weights.tensors(), name).ok_or_else(|| {
            err!(
                "quant",
                "tensor '{}' has no bitsandbytes quant_state sibling",
                name
            )
        })?;
        let meta: &TensorMeta = weights.tensors().get(&sname).ok_or_else(|| {
            err!(
                "quant",
                "quant_state tensor '{}' vanished from the index",
                sname
            )
        })?;
        let raw = weights.bytes(meta)?;
        let txt = std::str::from_utf8(raw)
            .map_err(|_| err!("quant", "quant_state of '{}' is not UTF-8 JSON", name))?;
        let j = json::parse(txt).map_err(|e| {
            err!(
                "quant",
                "quant_state of '{}' is not valid JSON: {}",
                name,
                e
            )
        })?;

        let shape: Vec<usize> = j
            .arr_of("shape")
            .map(|a| a.iter().filter_map(Json::as_usize).collect())
            .unwrap_or_default();
        if shape.len() != 2 {
            return Err(err!(
                "quant",
                "quant_state of '{}' has shape {:?} — only 2-D linear weights are executable",
                name,
                shape
            ));
        }
        let blocksize = j.usize_of("blocksize").unwrap_or(64);
        if (shape[0] * shape[1]) % blocksize != 0 {
            return Err(err!(
                "quant",
                "'{}': numel {} not divisible by blocksize {}",
                name,
                shape[0] * shape[1],
                blocksize
            ));
        }
        Ok(BnbState {
            blocksize,
            rows: shape[0],
            cols: shape[1],
            nested_blocksize: j.usize_of("nested_blocksize").unwrap_or(256),
            nested_offset: j.f64_of("nested_offset").unwrap_or(0.0) as f32,
        })
    }

    /// Fold the double quantization on the host: per-block f32 absmax.
    fn folded_absmax(
        weights: &dyn LoadedWeights,
        name: &str,
        st: &BnbState,
        nblocks: usize,
    ) -> Res<Vec<f32>> {
        let am_meta = weights
            .tensors()
            .get(&format!("{}.absmax", name))
            .ok_or_else(|| err!("quant", "'{}': missing .absmax sibling", name))?;
        let am_raw = weights.bytes(am_meta)?;
        match am_meta.dtype {
            DType::F32 => {
                let v = f32s(am_raw);
                if v.len() != nblocks {
                    return Err(err!(
                        "quant",
                        "'{}': absmax has {} entries, expected {}",
                        name,
                        v.len(),
                        nblocks
                    ));
                }
                Ok(v)
            }
            DType::U8 => {
                let nmap_meta = weights
                    .tensors()
                    .get(&format!("{}.nested_quant_map", name))
                    .ok_or_else(|| {
                        err!(
                            "quant",
                            "'{}': double-quantized but .nested_quant_map is missing",
                            name
                        )
                    })?;
                let nabs_meta = weights
                    .tensors()
                    .get(&format!("{}.nested_absmax", name))
                    .ok_or_else(|| {
                        err!(
                            "quant",
                            "'{}': double-quantized but .nested_absmax is missing",
                            name
                        )
                    })?;
                let nmap = f32s(weights.bytes(nmap_meta)?);
                let nabs = f32s(weights.bytes(nabs_meta)?);
                if am_raw.len() != nblocks {
                    return Err(err!(
                        "quant",
                        "'{}': absmax has {} entries, expected {}",
                        name,
                        am_raw.len(),
                        nblocks
                    ));
                }
                let mut out = Vec::with_capacity(nblocks);
                for (i, &code) in am_raw.iter().enumerate() {
                    let scale = nabs.get(i / st.nested_blocksize).copied().ok_or_else(|| {
                        err!(
                            "quant",
                            "'{}': nested_absmax too short ({} for block {})",
                            name,
                            nabs.len(),
                            i
                        )
                    })?;
                    out.push(nmap[code as usize] * scale + st.nested_offset);
                }
                Ok(out)
            }
            other => Err(err!(
                "quant",
                "'{}': absmax stored as {} — expected U8 (double-quant) or F32",
                name,
                other.name()
            )),
        }
    }

    /// Load a packed 4-bit weight family onto the device.
    pub fn upload_nf4(
        ctx: &CudaCtx,
        weights: &dyn LoadedWeights,
        meta: &TensorMeta,
        name: &str,
    ) -> Res<BnbNf4> {
        let st = parse_state(weights, name)?;
        let numel = st.rows * st.cols;
        if meta.dtype != DType::U8 || meta.nbytes != numel / 2 {
            return Err(err!(
            "quant",
            "'{}': packed payload is {} bytes of {}, expected {} u8 bytes for [{} × {}] nibbles",
            name, meta.nbytes, meta.dtype.name(), numel / 2, st.rows, st.cols
        ));
        }
        let nblocks = numel / st.blocksize;
        let absmax_host = folded_absmax(weights, name, &st, nblocks)?;
        let qmap_meta = weights
            .tensors()
            .get(&format!("{}.quant_map", name))
            .ok_or_else(|| err!("quant", "'{}': missing .quant_map sibling", name))?;
        let qmap_host = f32s(weights.bytes(qmap_meta)?);
        if qmap_host.len() != 16 {
            return Err(err!(
                "quant",
                "'{}': quant_map has {} entries, expected 16",
                name,
                qmap_host.len()
            ));
        }

        let packed = ctx.alloc(meta.nbytes)?;
        ctx.htod(&packed, weights.bytes(meta)?)?;
        let absmax = ctx.alloc(nblocks * 4)?;
        let ab: &[u8] =
            unsafe { std::slice::from_raw_parts(absmax_host.as_ptr() as *const u8, nblocks * 4) };
        ctx.htod(&absmax, ab)?;
        let qmap = ctx.alloc(64)?;
        let qb: &[u8] = unsafe { std::slice::from_raw_parts(qmap_host.as_ptr() as *const u8, 64) };
        ctx.htod(&qmap, qb)?;

        Ok(BnbNf4 {
            packed,
            absmax,
            qmap,
            rows: st.rows,
            cols: st.cols,
            blocksize: st.blocksize,
        })
    }

    /// Forecast helper: device bytes a packed weight will occupy once resident
    /// (nibbles + folded f32 absmax + codebook).
    pub fn nf4_device_bytes(numel: usize, blocksize: usize) -> usize {
        numel / 2 + (numel / blocksize) * 4 + 64
    }

    /// Host dequantization of one packed NF4/FP4 weight family to f16,
    /// row-major `[rows, cols]` — the bit-exact host mirror of
    /// `bitsandbytes.functional.dequantize_4bit` (high nibble = even index).
    pub fn dequant_host(
        weights: &dyn LoadedWeights,
        meta: &TensorMeta,
        name: &str,
    ) -> Res<Vec<u16>> {
        let st = parse_state(weights, name)?;
        let numel = st.rows * st.cols;
        if meta.dtype != DType::U8 || meta.nbytes != numel / 2 {
            return Err(err!(
                "quant",
                "'{}': packed payload is {} bytes of {}, expected {} u8 bytes for [{} × {}] nibbles",
                name, meta.nbytes, meta.dtype.name(), numel / 2, st.rows, st.cols
            ));
        }
        let nblocks = numel / st.blocksize;
        let absmax = folded_absmax(weights, name, &st, nblocks)?;
        let qmap_meta = weights
            .tensors()
            .get(&format!("{}.quant_map", name))
            .ok_or_else(|| err!("quant", "'{}': missing .quant_map sibling", name))?;
        let qmap = f32s(weights.bytes(qmap_meta)?);
        if qmap.len() != 16 {
            return Err(err!(
                "quant",
                "'{}': quant_map has {} entries, expected 16",
                name,
                qmap.len()
            ));
        }
        let packed = weights.bytes(meta)?;
        let mut out = vec![0u16; numel];
        for (i, o) in out.iter_mut().enumerate() {
            let byte = packed[i / 2];
            let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
            *o = crate::num::f32_to_f16(qmap[nib as usize] * absmax[i / st.blocksize]);
        }
        Ok(out)
    }

    /// `LoadedWeights` adapter that presents a bitsandbytes 4-bit checkpoint
    /// as a plain f16 one: every packed weight dequantizes on the host at
    /// wrap time (folded double-quant, codebook lookup) and re-appears in
    /// the index with its LOGICAL `[rows, cols]` shape and dtype F16; the
    /// bnb auxiliary tensors vanish. This is the execution bridge for
    /// architectures whose pipeline has no native packed-NF4 residency
    /// (everything except gemma4): weights land dense f16 on the device —
    /// no VRAM saving vs the fp16 revision, but the checkpoint runs, and
    /// the numbers are the checkpoint's own 4-bit-rounded values.
    pub struct DequantizedWeights {
        tensors: std::collections::HashMap<String, TensorMeta>,
        dense: std::collections::HashMap<String, Vec<u8>>,
        inner: Box<dyn LoadedWeights>,
    }

    impl DequantizedWeights {
        pub fn wrap(inner: Box<dyn LoadedWeights>) -> Res<Self> {
            let mut tensors = std::collections::HashMap::new();
            let mut dense: std::collections::HashMap<String, Vec<u8>> =
                std::collections::HashMap::new();
            let names: Vec<String> = inner.tensors().keys().cloned().collect();
            let mut converted = 0usize;
            for name in names {
                if is_aux(&name) {
                    continue; // consumed below; must never reach the codec
                }
                let meta = inner.tensors()[&name].clone();
                if state_name(inner.tensors(), &name).is_some() {
                    let st = parse_state(&*inner, &name)?;
                    let h = dequant_host(&*inner, &meta, &name)?;
                    let bytes: Vec<u8> =
                        unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2) }
                            .to_vec();
                    tensors.insert(
                        name.clone(),
                        TensorMeta {
                            name: name.clone(),
                            dtype: DType::F16,
                            shape: vec![st.rows, st.cols],
                            offset: 0,
                            nbytes: bytes.len(),
                            file: meta.file.clone(),
                        },
                    );
                    dense.insert(name, bytes);
                    converted += 1;
                } else {
                    tensors.insert(name, meta);
                }
            }
            if converted == 0 {
                return Err(err!(
                    "quant",
                    "checkpoint declares quant_method=bitsandbytes but contains no packed 4-bit weights (missing quant_state siblings?)"
                ));
            }
            crate::log::info(&format!(
                "bnb: dequantized {} packed 4-bit weights to f16 on the host (standard-pipeline bridge; packed NF4 stays resident only on gemma4)",
                converted
            ));
            Ok(DequantizedWeights {
                tensors,
                dense,
                inner,
            })
        }
    }

    impl LoadedWeights for DequantizedWeights {
        fn tensors(&self) -> &std::collections::HashMap<String, TensorMeta> {
            &self.tensors
        }
        fn bytes(&self, meta: &TensorMeta) -> Res<&[u8]> {
            if let Some(b) = self.dense.get(&meta.name) {
                return Ok(b);
            }
            self.inner.bytes(meta)
        }
        fn prefetch(&self, meta: &TensorMeta) {
            if !self.dense.contains_key(&meta.name) {
                self.inner.prefetch(meta);
            }
        }
    }
}

pub mod gguf {
    //! ggml block-quantization codecs (GGUF checkpoints): Q8_0, Q4_K, Q5_K,
    //! Q6_K, IQ4_XS — the formats behind Q4_K_M / UD-Q4_K_XL / Q6_K / Q8_0
    //! repos, unsloth-dynamic mixes included — plus the legacy 32-grain
    //! formats Q4_0/Q4_1/Q5_0/Q5_1. Those are not just historical: the
    //! K-quantizer's super-block grain is 256, so any tensor whose row
    //! length is not a multiple of 256 (Qwen2.5-0.5B, hidden 896) is
    //! quantized with a 32-grain fallback INSIDE a "Q4_K_M" file — without
    //! them such checkpoints are unloadable.
    //!
    //! Execution is quantized-RESIDENT: packed blocks upload as stored and
    //! stay packed in VRAM. Decode (m = 1) runs fused dequant-GEMVs that
    //! expand weights in registers while dotting — VRAM *and* per-token
    //! weight traffic shrink to the storage width (~×3.5 vs f16 for Q4_K;
    //! decode is bandwidth-bound, so that ratio is also the speed ceiling
    //! lift). Prefill dequantizes one layer linear at a time into a reusable
    //! scratch (tens of MB) and runs the ordinary tensor-core GEMM. The host
    //! decoders below are the reference the device kernels are verified
    //! against, bit for bit, by `cima selftest gguf`; `GgufCodec { resident:
    //! false }` keeps the legacy dequantize-at-load path for tests.
    //!
    //! Block layouts (little-endian, per ggml-quants.c):
    //! * `Q8_0`  — 32 elems: d f16, qs i8[32]; x[i] = d * qs[i]
    //! * `Q4_0`  — 32 elems: d f16, qs u8[16] (byte j: elem j low nibble,
    //!   elem j+16 high nibble); x = d·(q − 8)
    //! * `Q4_1`  — 32 elems: d f16, m f16, qs u8[16]; x = d·q + m
    //! * `Q5_0`  — 32 elems: d f16, qh u8[4] (5th bits: bit j → elem j,
    //!   bit j+16 → elem j+16), qs u8[16]; x = d·((q4 | bit·16) − 16)
    //! * `Q5_1`  — 32 elems: d f16, m f16, qh u8[4], qs u8[16];
    //!   x = d·(q4 | bit·16) + m
    //! * `IQ4_XS` — 256 elems: d, split 6-bit scales, nibbles into a
    //!   non-linear 16-level codebook (kvalues_iq4nl)
    //! * `Q5_K`  — 256 elems: d, dmin, scales u8[12], qh u8[32] (5th bit),
    //!   qs u8[128]; x = d*sc*(q4 + 16·bit) − dmin*m
    //! * `Q4_K`  — 256 elems: d f16, dmin f16, scales u8[12] (6-bit packed
    //!   per 8 sub-blocks of 32), qs u8[128] (4-bit pairs);
    //!   x = d*sc[sub]*q − dmin*m[sub]
    //! * `Q6_K`  — 256 elems: ql u8[128] (low 4 bits), qh u8[64] (high 2
    //!   bits), scales i8[16] (per 16 elems), d f16; x = d*sc*(q−32)

    use crate::cuda::{CudaCtx, DeviceBuf};
    use crate::num::{f16_to_f32, f32_to_f16};
    use crate::traits::{DType, Res, TensorMeta, WeightCodec};
    use crate::{err, formats::gguf::storage_bytes};

    fn d16(b: &[u8]) -> f32 {
        f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
    }

    /// Q8_0: 34-byte blocks of 32.
    /// Host-side dequant dispatch over every gguf block format this engine
    /// decodes (used for slots that must land dense: embeddings, PLE rows).
    pub fn dequant_host(
        dt: crate::traits::DType,
        src: &[u8],
        numel: usize,
        out: &mut [u16],
    ) -> Res<()> {
        use crate::traits::DType as D;
        match dt {
            D::GgufQ8_0 => dequant_q8_0(src, numel, out),
            D::GgufQ4_0 => dequant_q4_0(src, numel, out),
            D::GgufQ4_1 => dequant_q4_1(src, numel, out),
            D::GgufQ5_0 => dequant_q5_0(src, numel, out),
            D::GgufQ5_1 => dequant_q5_1(src, numel, out),
            D::GgufQ4K => dequant_q4_k(src, numel, out),
            D::GgufQ5K => dequant_q5_k(src, numel, out),
            D::GgufQ6K => dequant_q6_k(src, numel, out),
            D::GgufIQ4XS => dequant_iq4_xs(src, numel, out),
            other => Err(err!("quant", "no host dequant for {}", other.name())),
        }
    }

    pub fn dequant_q8_0(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 32;
        if src.len() < blocks * 34 || numel % 32 != 0 {
            return Err(err!(
                "gguf",
                "Q8_0 size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 34..b * 34 + 34];
            let d = d16(blk);
            for i in 0..32 {
                out[b * 32 + i] = f32_to_f16(d * blk[2 + i] as i8 as f32);
            }
        }
        Ok(())
    }

    /// Q4_0: 18-byte blocks of 32 — d f16, 16 nibble bytes; x = d·(q − 8).
    /// Element order per ggml: byte j holds elem j (low nibble) and
    /// elem j+16 (high nibble). This and the three formats below are the
    /// legacy 32-grain codecs llama.cpp falls back to when a tensor's row
    /// length is not a multiple of 256 (K-quant grain).
    pub fn dequant_q4_0(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 32;
        if src.len() < blocks * 18 || numel % 32 != 0 {
            return Err(err!(
                "gguf",
                "Q4_0 size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 18..b * 18 + 18];
            let d = d16(blk);
            for j in 0..16 {
                let q = blk[2 + j];
                out[b * 32 + j] = f32_to_f16(d * ((q & 0x0F) as i32 - 8) as f32);
                out[b * 32 + j + 16] = f32_to_f16(d * ((q >> 4) as i32 - 8) as f32);
            }
        }
        Ok(())
    }

    /// Q4_1: 20-byte blocks of 32 — d f16, m f16, 16 nibble bytes;
    /// x = d·q + m.
    pub fn dequant_q4_1(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 32;
        if src.len() < blocks * 20 || numel % 32 != 0 {
            return Err(err!(
                "gguf",
                "Q4_1 size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 20..b * 20 + 20];
            let d = d16(&blk[0..2]);
            let m = d16(&blk[2..4]);
            for j in 0..16 {
                let q = blk[4 + j];
                out[b * 32 + j] = f32_to_f16(d * (q & 0x0F) as f32 + m);
                out[b * 32 + j + 16] = f32_to_f16(d * (q >> 4) as f32 + m);
            }
        }
        Ok(())
    }

    /// Q5_0: 22-byte blocks of 32 — d f16, qh u8[4] (the 5th bits), 16
    /// nibble bytes; x = d·(((q4 | bit·16)) − 16). Bit j of qh belongs to
    /// elem j, bit j+16 to elem j+16 (ggml's `(qh >> (j+12)) & 0x10`).
    pub fn dequant_q5_0(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 32;
        if src.len() < blocks * 22 || numel % 32 != 0 {
            return Err(err!(
                "gguf",
                "Q5_0 size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 22..b * 22 + 22];
            let d = d16(&blk[0..2]);
            let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
            for j in 0..16 {
                let q = blk[6 + j];
                let hi0 = ((qh >> j) << 4) & 0x10;
                let hi1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((q & 0x0F) as u32 | hi0) as i32 - 16;
                let x1 = ((q >> 4) as u32 | hi1) as i32 - 16;
                out[b * 32 + j] = f32_to_f16(d * x0 as f32);
                out[b * 32 + j + 16] = f32_to_f16(d * x1 as f32);
            }
        }
        Ok(())
    }

    /// Q5_1: 24-byte blocks of 32 — Q5_0's layout with an added m f16;
    /// x = d·(q4 | bit·16) + m.
    pub fn dequant_q5_1(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 32;
        if src.len() < blocks * 24 || numel % 32 != 0 {
            return Err(err!(
                "gguf",
                "Q5_1 size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 24..b * 24 + 24];
            let d = d16(&blk[0..2]);
            let m = d16(&blk[2..4]);
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            for j in 0..16 {
                let q = blk[8 + j];
                let hi0 = ((qh >> j) << 4) & 0x10;
                let hi1 = (qh >> (j + 12)) & 0x10;
                out[b * 32 + j] = f32_to_f16(d * ((q & 0x0F) as u32 | hi0) as f32 + m);
                out[b * 32 + j + 16] = f32_to_f16(d * ((q >> 4) as u32 | hi1) as f32 + m);
            }
        }
        Ok(())
    }

    /// Unpack the 6-bit scale/min pair of sub-block `j` from Q4_K's 12-byte
    /// scales field (ggml's get_scale_min_k4).
    fn scale_min_k4(j: usize, s: &[u8]) -> (f32, f32) {
        if j < 4 {
            ((s[j] & 63) as f32, (s[j + 4] & 63) as f32)
        } else {
            (
                ((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)) as f32,
                ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as f32,
            )
        }
    }

    /// Q4_K: 144-byte super-blocks of 256 (8 sub-blocks of 32).
    pub fn dequant_q4_k(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 256;
        if src.len() < blocks * 144 || numel % 256 != 0 {
            return Err(err!(
                "gguf",
                "Q4_K size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 144..b * 144 + 144];
            let d = d16(&blk[0..2]);
            let dmin = d16(&blk[2..4]);
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            // Layout: 4 chunks of 64 elements; each chunk reads 32 qs bytes —
            // low nibbles are sub-block 2c, high nibbles sub-block 2c+1.
            for c in 0..4 {
                let (sc0, m0) = scale_min_k4(2 * c, scales);
                let (sc1, m1) = scale_min_k4(2 * c + 1, scales);
                let (d0, mm0) = (d * sc0, dmin * m0);
                let (d1, mm1) = (d * sc1, dmin * m1);
                for i in 0..32 {
                    let q = qs[c * 32 + i];
                    out[b * 256 + c * 64 + i] = f32_to_f16(d0 * (q & 0x0F) as f32 - mm0);
                    out[b * 256 + c * 64 + 32 + i] = f32_to_f16(d1 * (q >> 4) as f32 - mm1);
                }
            }
        }
        Ok(())
    }

    /// Q5_K: 176-byte super-blocks of 256 — Q4_K's nibble layout plus a 5th
    /// bit per element packed in `qh` (rotating mask pair per 64-chunk).
    pub fn dequant_q5_k(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 256;
        if src.len() < blocks * 176 || numel % 256 != 0 {
            return Err(err!(
                "gguf",
                "Q5_K size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 176..b * 176 + 176];
            let d = d16(&blk[0..2]);
            let dmin = d16(&blk[2..4]);
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let mut ql = &blk[48..176];
            let (mut u1, mut u2) = (1u8, 2u8);
            for c in 0..4 {
                let (sc0, m0) = scale_min_k4(2 * c, scales);
                let (sc1, m1) = scale_min_k4(2 * c + 1, scales);
                let (d1, mm1) = (d * sc0, dmin * m0);
                let (d2, mm2) = (d * sc1, dmin * m1);
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                    let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    out[b * 256 + c * 64 + l] =
                        f32_to_f16(d1 * ((ql[l] & 0x0F) as f32 + hi1) - mm1);
                    out[b * 256 + c * 64 + 32 + l] =
                        f32_to_f16(d2 * ((ql[l] >> 4) as f32 + hi2) - mm2);
                }
                ql = &ql[32..];
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        Ok(())
    }

    /// IQ4_XS's non-linear 4-bit codebook (ggml kvalues_iq4nl).
    const IQ4NL: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];

    /// IQ4_XS: 136-byte super-blocks of 256 — i-quant 4-bit: nibbles index a
    /// non-linear codebook; 6-bit sub-block scales split across scales_h
    /// (2 bits × 8) and scales_l (4 bits × 8).
    pub fn dequant_iq4_xs(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 256;
        if src.len() < blocks * 136 || numel % 256 != 0 {
            return Err(err!(
                "gguf",
                "IQ4_XS size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 136..b * 136 + 136];
            let d = d16(&blk[0..2]);
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8];
            let qs = &blk[8..136];
            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
                    | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                let q = &qs[ib * 16..ib * 16 + 16];
                for j in 0..16 {
                    out[b * 256 + ib * 32 + j] = f32_to_f16(dl * IQ4NL[(q[j] & 0x0F) as usize]);
                    out[b * 256 + ib * 32 + 16 + j] = f32_to_f16(dl * IQ4NL[(q[j] >> 4) as usize]);
                }
            }
        }
        Ok(())
    }

    /// Q6_K: 210-byte super-blocks of 256.
    pub fn dequant_q6_k(src: &[u8], numel: usize, out: &mut [u16]) -> Res<()> {
        let blocks = numel / 256;
        if src.len() < blocks * 210 || numel % 256 != 0 {
            return Err(err!(
                "gguf",
                "Q6_K size mismatch: {} elems, {} bytes",
                numel,
                src.len()
            ));
        }
        for b in 0..blocks {
            let blk = &src[b * 210..b * 210 + 210];
            let ql = &blk[0..128];
            let qh = &blk[128..192];
            let sc = &blk[192..208];
            let d = d16(&blk[208..210]);
            // Two halves of 128; within each: l in 0..32 yields 4 elements
            // (offsets 0/32/64/96) per ggml's dequantize_row_q6_K.
            for half in 0..2 {
                let (qlh, qhh, sch, base) = (
                    &ql[half * 64..],
                    &qh[half * 32..],
                    &sc[half * 8..],
                    b * 256 + half * 128,
                );
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((qlh[l] & 0x0F) | ((qhh[l] & 0x03) << 4)) as i8 - 32;
                    let q2 = ((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 0x03) << 4)) as i8 - 32;
                    let q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 0x03) << 4)) as i8 - 32;
                    let q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 0x03) << 4)) as i8 - 32;
                    out[base + l] = f32_to_f16(d * sch[is] as i8 as f32 * q1 as f32);
                    out[base + l + 32] = f32_to_f16(d * sch[is + 2] as i8 as f32 * q2 as f32);
                    out[base + l + 64] = f32_to_f16(d * sch[is + 4] as i8 as f32 * q3 as f32);
                    out[base + l + 96] = f32_to_f16(d * sch[is + 6] as i8 as f32 * q4 as f32);
                }
            }
        }
        Ok(())
    }

    /// WeightCodec over GGUF tensors: block formats dequantize to f16 on the
    /// way up; F16 passes through; F32/BF16 convert. Resident footprint is
    /// always `numel × 2`.
    pub struct GgufCodec {
        /// `true` (the load path): quantized block tensors stay PACKED on
        /// device — the transformer routes them through the fused gguf GEMVs
        /// (decode) and the dequant-scratch GEMM path (prefill). VRAM and
        /// per-token weight traffic both shrink to the storage width (~×3.5
        /// for Q4_K vs f16), which is the entire point of the format.
        /// `false`: legacy dequantize-to-f16-at-load (kept for tests and as
        /// a fallback lever).
        pub resident: bool,
    }

    impl WeightCodec for GgufCodec {
        fn name(&self) -> &'static str {
            if self.resident {
                "gguf-resident"
            } else {
                "gguf-dequant-f16"
            }
        }
        fn resident_quant(&self) -> bool {
            self.resident
        }
        fn accepts(&self, dtype: DType) -> bool {
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
                    | DType::F16
                    | DType::F32
                    | DType::BF16
            )
        }
        fn device_bytes(&self, meta: &TensorMeta) -> usize {
            if self.resident && crate::traits::is_gguf_block(meta.dtype) {
                storage_bytes(meta.dtype, meta.numel())
            } else {
                meta.numel() * 2
            }
        }
        fn upload(&self, ctx: &CudaCtx, meta: &TensorMeta, host: &[u8]) -> Res<DeviceBuf> {
            let numel = meta.numel();
            if storage_bytes(meta.dtype, numel) > host.len() {
                return Err(err!(
                    "gguf",
                    "tensor '{}' is shorter than its dtype predicts",
                    meta.name
                ));
            }
            if self.resident && crate::traits::is_gguf_block(meta.dtype) {
                // Resident path: the packed blocks ARE the device representation.
                let packed = &host[..storage_bytes(meta.dtype, numel)];
                let buf = ctx.alloc(packed.len())?;
                ctx.htod(&buf, packed)?;
                return Ok(buf);
            }
            let f16: Vec<u16> = match meta.dtype {
                DType::F16 => host
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                DType::F32 => host
                    .chunks_exact(4)
                    .map(|c| f32_to_f16(f32::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                DType::BF16 => host
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
                        f32_to_f16(f32::from_bits(bits))
                    })
                    .collect(),
                q => {
                    let mut out = vec![0u16; numel];
                    match q {
                        DType::GgufQ8_0 => dequant_q8_0(host, numel, &mut out)?,
                        DType::GgufQ4_0 => dequant_q4_0(host, numel, &mut out)?,
                        DType::GgufQ4_1 => dequant_q4_1(host, numel, &mut out)?,
                        DType::GgufQ5_0 => dequant_q5_0(host, numel, &mut out)?,
                        DType::GgufQ5_1 => dequant_q5_1(host, numel, &mut out)?,
                        DType::GgufQ4K => dequant_q4_k(host, numel, &mut out)?,
                        DType::GgufQ5K => dequant_q5_k(host, numel, &mut out)?,
                        DType::GgufIQ4XS => dequant_iq4_xs(host, numel, &mut out)?,
                        DType::GgufQ6K => dequant_q6_k(host, numel, &mut out)?,
                        other => {
                            return Err(err!("gguf", "codec does not execute dtype {:?}", other))
                        }
                    }
                    out
                }
            };
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(f16.as_ptr() as *const u8, f16.len() * 2) };
            let buf = ctx.alloc(bytes.len())?;
            ctx.htod(&buf, bytes)?;
            Ok(buf)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn q8_0_block_reference() {
            // d = 0.5, qs = [-128..] ramp: x[i] = 0.5 * qs[i]
            let mut blk = vec![0u8; 34];
            blk[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
            for i in 0..32 {
                blk[2 + i] = (i as i8 * 3 - 40) as u8;
            }
            let mut out = vec![0u16; 32];
            dequant_q8_0(&blk, 32, &mut out).unwrap();
            for i in 0..32 {
                let want = 0.5 * (i as i8 * 3 - 40) as f32;
                assert!((f16_to_f32(out[i]) - want).abs() < 1e-3, "i={}", i);
            }
        }

        #[test]
        fn q4_k_sub_block_scales_and_mins() {
            // One super-block: d=1, dmin=1, scale[j]=j+1, min[j]=j (6-bit
            // packing exercised across the j<4 / j>=4 boundary), qs = nibble
            // ramp. Verify against the formula d*sc*q − dmin*m.
            let mut blk = vec![0u8; 144];
            blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            // pack scales/mins per ggml: j<4 → s[j]=sc, s[j+4]=m (low 6 bits);
            // j>=4 → low nibbles in s[j+4], high 2 bits spread into s[j-4]/s[j].
            let sc: Vec<u8> = (0..8).map(|j| j + 1).collect();
            let mn: Vec<u8> = (0..8).collect();
            let s = &mut blk[4..16];
            for j in 0..4 {
                s[j] = sc[j] & 63;
                s[j + 4] = mn[j] & 63;
            }
            for j in 4..8 {
                s[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
                s[j - 4] |= (sc[j] >> 4) << 6;
                s[j] |= (mn[j] >> 4) << 6;
            }
            for (i, q) in blk[16..144].iter_mut().enumerate() {
                *q = ((i % 16) | ((15 - i % 16) << 4)) as u8;
            }
            let mut out = vec![0u16; 256];
            dequant_q4_k(&blk, 256, &mut out).unwrap();
            for c in 0..4 {
                for i in 0..32 {
                    let q = blk[16 + c * 32 + i];
                    let want_lo = (2 * c + 1) as f32 * (q & 0x0F) as f32 - (2 * c) as f32;
                    let want_hi = (2 * c + 2) as f32 * (q >> 4) as f32 - (2 * c + 1) as f32;
                    assert!(
                        (f16_to_f32(out[c * 64 + i]) - want_lo).abs() < 0.51,
                        "c={} i={}",
                        c,
                        i
                    );
                    assert!(
                        (f16_to_f32(out[c * 64 + 32 + i]) - want_hi).abs() < 0.51,
                        "c={} i={}",
                        c,
                        i
                    );
                }
            }
        }

        #[test]
        fn q6_k_reconstructs_signed_six_bit() {
            // Encode a known pattern: element value v in -32..32 at scale 2.
            // Build ql/qh from v+32 (6 bits) for the first interleave slot and
            // verify the decoder's bit-reassembly.
            let mut blk = vec![0u8; 210];
            let scales = &mut blk[192..208];
            for s in scales.iter_mut() {
                *s = 2; // i8
            }
            blk[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            // For half=0, l=0..32: q1 uses ql[l]&0xF | (qh[l]&3)<<4 at out[l].
            for l in 0..32usize {
                let v = (l as i32 * 2 - 31).clamp(-32, 31); // odd ramp
                let q = (v + 32) as u8; // 6 bits
                blk[l] = (blk[l] & 0xF0) | (q & 0x0F);
                blk[128 + l] = (blk[128 + l] & !0x03) | ((q >> 4) & 0x03);
            }
            let mut out = vec![0u16; 256];
            dequant_q6_k(&blk, 256, &mut out).unwrap();
            for l in 0..32usize {
                let v = (l as i32 * 2 - 31).clamp(-32, 31) as f32;
                assert!((f16_to_f32(out[l]) - 2.0 * v).abs() < 1e-2, "l={}", l);
            }
        }

        #[test]
        fn q5_k_fifth_bit_and_rotating_masks() {
            // d=1, dmin=0, scale[j]=1: value = (nibble) + (qh bit ? 16 : 0).
            // Set the high bit only for chunk 1's first half (mask u1=4 there)
            // and verify both the +16 lift and that chunk 0 stays unlifted.
            let mut blk = vec![0u8; 176];
            blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
            let s = &mut blk[4..16];
            for j in 0..4 {
                s[j] = 1; // sc=1 (j<4 path)
                s[j + 4] = 0;
            }
            for j in 4..8 {
                s[j + 4] = 1; // sc low nibble = 1
            }
            for (i, v) in blk[48..176].iter_mut().enumerate() {
                *v = ((i % 16) | ((i % 16) << 4)) as u8; // both nibbles = i%16
            }
            for h in blk[16..48].iter_mut() {
                *h = 0b0000_0100; // bit 2 set → chunk 1 (u1=4) first half lifted
            }
            let mut out = vec![0u16; 256];
            dequant_q5_k(&blk, 256, &mut out).unwrap();
            for l in 0..32 {
                let nib = (l % 16) as f32;
                assert!(
                    (f16_to_f32(out[l]) - nib).abs() < 0.51,
                    "chunk0 unlifted l={}",
                    l
                );
                assert!(
                    (f16_to_f32(out[64 + l]) - (nib + 16.0)).abs() < 0.51,
                    "chunk1 lifted l={}",
                    l
                );
            }
        }

        #[test]
        fn iq4_xs_codebook_and_split_scales() {
            // d=1; sub-block 0 scale = 33 (ls-32 = 1) via scales_l low nibble 1
            // + scales_h bits 0b10 (<<4 = 32): value = codebook[nibble].
            // Sub-block 1 scale stays 0-32 = -32 → sign flip check.
            let mut blk = vec![0u8; 136];
            blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            blk[2] = 0b0000_0010; // scales_h: ib0 bits = 10b → +32
            blk[4] = 0x01; // scales_l: ib0 low nibble = 1 → ls = 33
            for (i, v) in blk[8..24].iter_mut().enumerate() {
                *v = ((i % 16) | ((i % 16) << 4)) as u8;
            }
            let mut out = vec![0u16; 256];
            dequant_iq4_xs(&blk, 256, &mut out).unwrap();
            for j in 0..16 {
                let want = IQ4NL[j % 16]; // dl = 1*(33-32) = 1
                assert!(
                    (f16_to_f32(out[j]) - want).abs() < 0.6,
                    "codebook j={}: {} vs {}",
                    j,
                    f16_to_f32(out[j]),
                    want
                );
            }
            // ib1: ls=0 → dl=-32 → value = -32*codebook[0] for zeroed qs
            assert!(
                (f16_to_f32(out[32]) - (-32.0 * IQ4NL[0])).abs() < 4.0,
                "negative scale path"
            );
        }

        #[test]
        fn f16_converters_are_ieee_exact() {
            // Round-trip identity over EVERY non-NaN f16 (subnormals, zeros,
            // infinities) — the property that makes host-vs-device decoder
            // comparisons meaningful at bit level.
            for h in 0u16..=0xffff {
                let exp = (h >> 10) & 0x1f;
                let man = h & 0x3ff;
                if exp == 0x1f && man != 0 {
                    // NaN: class preserved, canonical quiet on the way back
                    let f = f16_to_f32(h);
                    assert!(f.is_nan());
                    assert_eq!(f32_to_f16(f) & 0x7c00, 0x7c00);
                    assert_ne!(f32_to_f16(f) & 0x3ff, 0);
                    continue;
                }
                assert_eq!(f32_to_f16(f16_to_f32(h)), h, "roundtrip h={:#06x}", h);
            }
            // Ties round to even, in the subnormal range too: midpoints
            // between consecutive f16s land on the even neighbor.
            for h in [0u16, 1, 2, 0x3ff, 0x400, 0x7bfe, 0x1234] {
                let a = f16_to_f32(h) as f64;
                let b = f16_to_f32(h + 1) as f64;
                let mid = ((a + b) / 2.0) as f32;
                let r = f32_to_f16(mid);
                assert!(r == h || r == h + 1);
                assert_eq!(
                    r & 1,
                    0,
                    "tie at h={:#06x} must round to even, got {:#06x}",
                    h,
                    r
                );
            }
        }

        #[test]
        fn storage_sizes_match_ggml() {
            assert_eq!(storage_bytes(DType::GgufQ8_0, 1024), 1024 / 32 * 34);
            assert_eq!(storage_bytes(DType::GgufQ4_0, 1024), 1024 / 32 * 18);
            assert_eq!(storage_bytes(DType::GgufQ4_1, 1024), 1024 / 32 * 20);
            assert_eq!(storage_bytes(DType::GgufQ5_0, 1024), 1024 / 32 * 22);
            assert_eq!(storage_bytes(DType::GgufQ5_1, 1024), 1024 / 32 * 24);
            assert_eq!(storage_bytes(DType::GgufQ4K, 1024), 1024 / 256 * 144);
            assert_eq!(storage_bytes(DType::GgufQ5K, 1024), 1024 / 256 * 176);
            assert_eq!(storage_bytes(DType::GgufIQ4XS, 1024), 1024 / 256 * 136);
            assert_eq!(storage_bytes(DType::GgufQ6K, 1024), 1024 / 256 * 210);
        }

        #[test]
        fn q4_0_nibble_order_and_offset() {
            // d = 0.25; byte j = j | ((15−j)<<4): elem j = 0.25·(j−8),
            // elem j+16 = 0.25·(15−j−8) — checks the split-halves order.
            let mut blk = vec![0u8; 18];
            blk[0..2].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
            for j in 0..16u8 {
                blk[2 + j as usize] = j | ((15 - j) << 4);
            }
            let mut out = vec![0u16; 32];
            dequant_q4_0(&blk, 32, &mut out).unwrap();
            for j in 0..16usize {
                assert!(
                    (f16_to_f32(out[j]) - 0.25 * (j as f32 - 8.0)).abs() < 1e-3,
                    "lo j={}",
                    j
                );
                assert!(
                    (f16_to_f32(out[j + 16]) - 0.25 * ((15 - j) as f32 - 8.0)).abs() < 1e-3,
                    "hi j={}",
                    j
                );
            }
        }

        #[test]
        fn q4_1_scale_plus_min() {
            let mut blk = vec![0u8; 20];
            blk[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16(-1.5).to_le_bytes());
            for j in 0..16u8 {
                blk[4 + j as usize] = j | ((15 - j) << 4);
            }
            let mut out = vec![0u16; 32];
            dequant_q4_1(&blk, 32, &mut out).unwrap();
            for j in 0..16usize {
                assert!((f16_to_f32(out[j]) - (0.5 * j as f32 - 1.5)).abs() < 1e-3);
                assert!((f16_to_f32(out[j + 16]) - (0.5 * (15 - j) as f32 - 1.5)).abs() < 1e-3);
            }
        }

        #[test]
        fn q5_0_fifth_bit_mapping() {
            // qh bit j → elem j, bit j+16 → elem j+16 (ggml's (qh>>(j+12))&0x10).
            let mut blk = vec![0u8; 22];
            blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            let qh: u32 = 0xA5F0_0F5A; // arbitrary bit soup
            blk[2..6].copy_from_slice(&qh.to_le_bytes());
            for j in 0..16u8 {
                blk[6 + j as usize] = (j % 16) | ((j % 16) << 4);
            }
            let mut out = vec![0u16; 32];
            dequant_q5_0(&blk, 32, &mut out).unwrap();
            for r in 0..32usize {
                let nib = (r % 16) as i32;
                let hi = ((qh >> r) & 1) as i32 * 16;
                let want = (nib + hi - 16) as f32;
                assert!((f16_to_f32(out[r]) - want).abs() < 1e-3, "r={}", r);
            }
        }

        #[test]
        fn q5_1_fifth_bit_plus_min() {
            let mut blk = vec![0u8; 24];
            blk[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
            let qh: u32 = 0x1234_ABCD;
            blk[4..8].copy_from_slice(&qh.to_le_bytes());
            for j in 0..16u8 {
                blk[8 + j as usize] = j | ((15 - j) << 4);
            }
            let mut out = vec![0u16; 32];
            dequant_q5_1(&blk, 32, &mut out).unwrap();
            for r in 0..32usize {
                let nib = if r < 16 {
                    r as i32
                } else {
                    15 - (r - 16) as i32
                };
                let hi = ((qh >> r) & 1) as i32 * 16;
                let want = 0.5 * (nib + hi) as f32 + 2.0;
                assert!((f16_to_f32(out[r]) - want).abs() < 1e-2, "r={}", r);
            }
        }
    }
}
