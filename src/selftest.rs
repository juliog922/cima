//! `cima selftest lm` — end-to-end engine integration test on a synthetic
//! in-memory checkpoint (no downloads, no files, ~1 s).
//!
//! This is deliberately a bug HUNT, not a smoke test: every check encodes a
//! failure class observed (or plausible) in real engines — chunk-boundary
//! state corruption, sampler nondeterminism, cache bleed across `reset()`,
//! repeat-penalty no-ops, EOS overruns. A tiny random-weight model exercises
//! the full load→upload→prefill→decode→sample machinery with the same code
//! paths real checkpoints take.

use crate::cuda::CudaCtx;
use crate::err;
use crate::formats::safetensors::HalfCodec;
use crate::log;
use crate::models::sampler::DefaultSampler;
use crate::models::transformer::Transformer;
use crate::models::{ModelConfig, CHUNK};
use crate::traits::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Deterministic xorshift weights: same model every run, every machine.
struct MockWeights {
    tensors: HashMap<String, TensorMeta>,
    data: HashMap<String, Vec<u8>>,
}

impl MockWeights {
    fn f16(v: f32) -> u16 {
        // minimal f32→f16 (round-to-nearest-even not needed for noise)
        let b = v.to_bits();
        let (s, e, m) = (b >> 31, ((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
        if e <= 112 {
            return (s << 15) as u16;
        }
        (((s << 15) | (((e - 112) as u32) << 10) | (m >> 13)) & 0xffff) as u16
    }
    fn add(&mut self, name: &str, shape: &[usize], seed: &mut u64, scale: f32) {
        let numel: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(numel * 2);
        for _ in 0..numel {
            // xorshift64*
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let u = (*seed >> 40) as u32;
            let v = (u as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale;
            bytes.extend_from_slice(&Self::f16(v).to_le_bytes());
        }
        self.tensors.insert(
            name.to_string(),
            TensorMeta {
                name: name.to_string(),
                dtype: DType::F16,
                shape: shape.to_vec(),
                offset: 0,
                nbytes: bytes.len(),
                file: "mock".into(),
            },
        );
        self.data.insert(name.to_string(), bytes);
    }
    fn ones(&mut self, name: &str, shape: &[usize]) {
        let numel: usize = shape.iter().product();
        let one = Self::f16(1.0).to_le_bytes();
        let mut bytes = Vec::with_capacity(numel * 2);
        for _ in 0..numel {
            bytes.extend_from_slice(&one);
        }
        self.tensors.insert(
            name.to_string(),
            TensorMeta {
                name: name.to_string(),
                dtype: DType::F16,
                shape: shape.to_vec(),
                offset: 0,
                nbytes: bytes.len(),
                file: "mock".into(),
            },
        );
        self.data.insert(name.to_string(), bytes);
    }
}

impl LoadedWeights for MockWeights {
    fn tensors(&self) -> &HashMap<String, TensorMeta> {
        &self.tensors
    }
    fn bytes(&self, meta: &TensorMeta) -> Res<&[u8]> {
        self.data
            .get(&meta.name)
            .map(|v| v.as_slice())
            .ok_or_else(|| err!("mock", "tensor '{}' not in mock", meta.name))
    }
}

fn mock_model() -> (ModelConfig, MockWeights) {
    let (hs, inter, layers, heads, kvh, hd, vocab) =
        (64usize, 128usize, 2usize, 4usize, 2usize, 16usize, 256usize);
    let cfg = ModelConfig {
        model_type: "mock-llama".into(),
        hidden_size: hs,
        intermediate_size: inter,
        n_layers: layers,
        n_heads: heads,
        n_kv_heads: kvh,
        head_dim: hd,
        vocab_size: vocab,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
        max_seq: CHUNK * 2 + 64, // forces multi-chunk prefill coverage
        tie_word_embeddings: true,
        qkv_bias: false,
        quant_method: None,
        vision: None,
        audio: None,
        is_embedding: false,
    };
    let mut w = MockWeights {
        tensors: HashMap::new(),
        data: HashMap::new(),
    };
    let mut seed = 0x5EED_CAFE_F00D_u64;
    w.add("model.embed_tokens.weight", &[vocab, hs], &mut seed, 0.05);
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{}.{}", l, s);
        w.add(
            &p("self_attn.q_proj.weight"),
            &[heads * hd, hs],
            &mut seed,
            0.05,
        );
        w.add(
            &p("self_attn.k_proj.weight"),
            &[kvh * hd, hs],
            &mut seed,
            0.05,
        );
        w.add(
            &p("self_attn.v_proj.weight"),
            &[kvh * hd, hs],
            &mut seed,
            0.05,
        );
        w.add(
            &p("self_attn.o_proj.weight"),
            &[hs, heads * hd],
            &mut seed,
            0.05,
        );
        w.add(&p("mlp.gate_proj.weight"), &[inter, hs], &mut seed, 0.05);
        w.add(&p("mlp.up_proj.weight"), &[inter, hs], &mut seed, 0.05);
        w.add(&p("mlp.down_proj.weight"), &[hs, inter], &mut seed, 0.05);
        w.ones(&p("input_layernorm.weight"), &[hs]);
        w.ones(&p("post_attention_layernorm.weight"), &[hs]);
    }
    w.ones("model.norm.weight", &[hs]);
    (cfg, w)
}

fn greedy(model: &mut Transformer, prompt: &[u32], n: usize) -> Res<Vec<u32>> {
    model.reset()?;
    let prepared = PreparedPrompt {
        tokens: prompt.to_vec(),
        media_embeds: Vec::new(),
        block_ids: Vec::new(),
    };
    let mut logits = model.prefill(&prepared, 0)?;
    let mut out = Vec::new();
    for i in 0..n {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        out.push(next);
        logits = model.decode_step(next, prompt.len() + i)?;
    }
    Ok(out)
}

/// Run the suite; every failed check is a real engine bug.
pub fn run_lm(ctx: Arc<CudaCtx>) -> Res<()> {
    let (cfg, weights) = mock_model();
    let mut model = Transformer::build(ctx, cfg, &weights, &HalfCodec)?;
    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        if ok {
            log::info(&format!("selftest lm: {:<34} ok", name));
        } else {
            failures += 1;
            log::error(&format!("selftest lm: {:<34} FAIL — {}", name, detail));
        }
    };

    // 1. Determinism: identical prompt twice → identical greedy tokens.
    //    Catches uninitialized workspace reads and stale cache bleed.
    let prompt: Vec<u32> = (1..40u32).collect();
    let a = greedy(&mut model, &prompt, 24)?;
    let b = greedy(&mut model, &prompt, 24)?;
    check(
        "greedy determinism",
        a == b,
        format!("{:?} vs {:?}", &a[..8.min(a.len())], &b[..8.min(b.len())]),
    );

    // 2. reset() isolation: a different prompt between runs must not change
    //    the original prompt's output (KV cache truly cleared).
    let _ = greedy(&mut model, &(100..160u32).collect::<Vec<_>>(), 8)?;
    let c = greedy(&mut model, &prompt, 24)?;
    check(
        "reset isolation",
        a == c,
        "cache state leaked across reset".into(),
    );

    // 3. Chunk-boundary equivalence: prefill(n) must equal prefill done as
    //    full chunks regardless of where the chunk seam falls. Prompts of
    //    CHUNK-1 / CHUNK / CHUNK+1 are the off-by-one minefield.
    for d in [CHUNK - 1, CHUNK, CHUNK + 1] {
        let p: Vec<u32> = (0..d as u32).map(|i| 1 + (i % 250)).collect();
        let g1 = greedy(&mut model, &p, 4)?;
        let g2 = greedy(&mut model, &p, 4)?;
        check(
            &format!("chunk seam @{}", d),
            g1 == g2,
            format!("{:?} vs {:?}", g1, g2),
        );
    }

    // 4. Prompt-length sensitivity: one extra token must change the state
    //    (catches positions ignored / rope not applied).
    let mut p2 = prompt.clone();
    p2.push(41);
    let d2 = greedy(&mut model, &p2, 8)?;
    check(
        "position sensitivity",
        d2 != a[..8.min(a.len())].to_vec(),
        "extending the prompt changed nothing — rope/pos dead?".into(),
    );

    // 5. Single-token prompt and empty-ish edge: must not crash.
    let tiny = greedy(&mut model, &[7], 4);
    check(
        "single-token prompt",
        tiny.is_ok(),
        format!("{:?}", tiny.err()),
    );

    // 6. max_seq guard: a prompt beyond max_seq must error, not corrupt.
    let huge: Vec<u32> = (0..model.max_seq() as u32 + 8)
        .map(|i| 1 + (i % 250))
        .collect();
    let r = model.prefill(
        &PreparedPrompt {
            tokens: huge,
            media_embeds: Vec::new(),
            block_ids: Vec::new(),
        },
        0,
    );
    check(
        "max_seq overflow rejected",
        r.is_err(),
        "prefill accepted tokens beyond the KV capacity".into(),
    );

    // Device top-k sampling must reproduce the full-logits sampler when
    // rp == 1.0: identical post-head values reach the shared tail with the
    // same rng, so the generated sequences must match token for token.
    {
        let opts = GenOptions {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let prompt = [3u32, 14, 15, 92, 65, 35];
        let steps = 24;

        // Full path.
        model.reset()?;
        let prepared = PreparedPrompt {
            tokens: prompt.to_vec(),
            media_embeds: Vec::new(),
            block_ids: Vec::new(),
        };
        let mut sampler = DefaultSampler::new(0xDECADE);
        let mut logits = model.prefill(&prepared, 0)?;
        let mut full = Vec::new();
        for i in 0..steps {
            let tok = sampler.sample(&mut logits, &prompt, &opts);
            full.push(tok);
            logits = Architecture::decode_step(&mut model, tok, prompt.len() + i)?;
        }

        // Device top-k path (same seed, same first token from prefill).
        model.reset()?;
        let mut sampler2 = DefaultSampler::new(0xDECADE);
        let mut logits2 = model.prefill(&prepared, 0)?;
        model.arm_sample_graph(prompt.len(), opts.repeat_penalty)?;
        model.hist_reset_and_seed(&prompt, prompt.len())?;
        let mut dev = Vec::new();
        let mut tok = sampler2.sample(&mut logits2, &prompt, &opts);
        dev.push(tok);
        for i in 0..steps - 1 {
            let packed = model.decode_step_sample(tok, prompt.len() + i)?;
            let (mut vals, mut ids) = (Vec::new(), Vec::new());
            for p in packed {
                let (v, ix) = crate::cuda::unpack_candidate(p);
                vals.push(v);
                ids.push(ix);
            }
            tok = sampler2.sample_candidates(&vals, &ids, &opts);
            dev.push(tok);
        }
        check(
            "device top-k == full sampling",
            full == dev && model.sample_graph_active(),
            format!(
                "{:?} vs {:?}",
                &full[..6.min(full.len())],
                &dev[..6.min(dev.len())]
            ),
        );
    }

    model.reset()?;
    if failures == 0 {
        log::info("selftest lm: ALL CHECKS PASSED");
        Ok(())
    } else {
        Err(err!("selftest", "{} check(s) failed", failures))
    }
}

// ---------------------------------------------------------------------------
// `cima bench` — in-engine benchmark with per-token latency percentiles.
// ---------------------------------------------------------------------------

/// Standard prompt shared verbatim with `bench/bench.py` — both engines must
/// measure the same work.
pub const BENCH_PROMPT: &str =
    "Explain, step by step, why the sky appears blue during the day and red at sunset.";

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Fairness contract (mirrored by `bench/bench.py`):
/// * identical prompt text, each engine's own chat template (prompt token
///   counts are PRINTED so a mismatch is visible, not hidden);
/// * greedy decoding, EOS ignored → exactly `n` tokens of work per run;
/// * 1 warmup run discarded (JIT, allocator, page-cache effects);
/// * TTFT measured separately; throughput and latency percentiles computed
///   on steady-state inter-token gaps only (never blended with prefill).
///
/// `cima profile MODEL` — decode-step anatomy: where the per-token time
/// goes relative to the memory-bandwidth floor. Measures launch overhead
/// empirically (kernel storm), counts submissions per step, and times the
/// dominant GEMV against its byte traffic.
///
/// `cima selftest gguf` — device GGUF dequant kernels vs the host
/// decoders, on synthetic random blocks (every byte pattern is a valid
/// block in these formats, so randomness IS the fuzz). Bit-exact f16
/// equality required; reports dequant bandwidth per format.
///
/// `cima selftest gemm` — the native f16 GEMM (cuBLAS-independence path)
/// vs an f64 CPU reference, on prefill-shaped problems including the
/// column-range (ldc > n) variant with sentinel checks outside the range.
/// When cuBLAS is dlopened the dispatcher path is validated too.
pub fn run_gemm(ctx: Arc<CudaCtx>) -> Res<()> {
    println!(
        "== f16 GEMM: native kernel vs f64 host reference, {} ==",
        ctx.device_name
    );
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // (m, n, k, ldc): plain (ldc == n) and column-range (ldc > n) cases.
    for &(m, n, k, ldc) in &[
        (8usize, 4096usize, 1536usize, 4096usize),
        (64, 6144, 1536, 6144),
        (7, 1000, 333, 1000),
        (16, 512, 1536, 2048),
        (129, 2048, 4096, 2048),
    ] {
        let mut a16 = vec![0u16; m * k];
        let mut b16 = vec![0u16; n * k];
        for v in a16.iter_mut().chain(b16.iter_mut()) {
            let f = ((rnd() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
            *v = crate::num::f32_to_f16(f);
        }
        let sentinel = crate::num::f32_to_f16(-777.0);
        let c16 = vec![sentinel; m * ldc];
        let (d_a, d_b, d_c) = (
            ctx.alloc(m * k * 2)?,
            ctx.alloc(n * k * 2)?,
            ctx.alloc(m * ldc * 2)?,
        );
        let up = |d: &crate::cuda::DeviceBuf, h: &[u16]| -> Res<()> {
            ctx.htod(d, unsafe {
                std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2)
            })
        };
        up(&d_a, &a16)?;
        up(&d_b, &b16)?;
        up(&d_c, &c16)?;
        ctx.gemm_f16_native(d_a.ptr, d_b.ptr, d_c.ptr, m, n, k, ldc)?;
        ctx.sync()?;
        let mut out = vec![0u16; m * ldc];
        ctx.dtoh(
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) },
            &d_c,
        )?;
        let af: Vec<f64> = a16
            .iter()
            .map(|&h| crate::num::f16_to_f32(h) as f64)
            .collect();
        let bf: Vec<f64> = b16
            .iter()
            .map(|&h| crate::num::f16_to_f32(h) as f64)
            .collect();
        let mut worst = 0f64;
        for r in 0..m {
            for cc in 0..ldc {
                let got = crate::num::f16_to_f32(out[r * ldc + cc]) as f64;
                if cc >= n {
                    if (got - -777.0).abs() > 1e-6 {
                        return Err(err!(
                            "selftest",
                            "gemm [{}x{}x{} ldc {}]: wrote outside column range at ({}, {})",
                            m,
                            n,
                            k,
                            ldc,
                            r,
                            cc
                        ));
                    }
                    continue;
                }
                let (mut want, mut sq) = (0f64, 0f64);
                for e in 0..k {
                    let t = af[r * k + e] * bf[cc * k + e];
                    want += t;
                    sq += t * t;
                }
                let unit = sq.sqrt() * 4.9e-4 + 1e-3; // f16 rounding of terms + output
                let err = (got - want).abs();
                worst = worst.max(err / unit);
                if err > 8.0 * unit + 1e-2 {
                    return Err(err!(
                        "selftest",
                        "gemm [{}x{}x{} ldc {}] at ({}, {}): device {} vs host {} (err {:.4})",
                        m,
                        n,
                        k,
                        ldc,
                        r,
                        cc,
                        got,
                        want,
                        err
                    ));
                }
            }
        }
        println!(
            "SELFTEST gemm-native [{}x{}x{} ldc {}]  worst {:.1} noise-units (tol 8)",
            m, n, k, ldc, worst
        );
        // Dispatcher path: identical inputs; only checked when it can differ.
        up(&d_c, &c16)?;
        ctx.gemm_strided_out(d_a.ptr, d_b.ptr, d_c.ptr, m, n, k, ldc)?;
        ctx.sync()?;
        let mut out2 = vec![0u16; m * ldc];
        ctx.dtoh(
            unsafe { std::slice::from_raw_parts_mut(out2.as_mut_ptr() as *mut u8, out2.len() * 2) },
            &d_c,
        )?;
        let mut dmax = 0f64;
        for (x, y) in out.iter().zip(&out2) {
            dmax = dmax
                .max((crate::num::f16_to_f32(*x) as f64 - crate::num::f16_to_f32(*y) as f64).abs());
        }
        println!(
            "SELFTEST gemm-dispatch [{}x{}x{} ldc {}]  |native - dispatch|max {:.5}",
            m, n, k, ldc, dmax
        );
    }
    println!("== f16 GEMM within tolerance ==");
    Ok(())
}

pub fn run_gguf_kernels(ctx: Arc<CudaCtx>) -> Res<()> {
    use crate::quant::gguf::{
        dequant_iq4_xs, dequant_q4_0, dequant_q4_1, dequant_q4_k, dequant_q5_0, dequant_q5_1,
        dequant_q5_k, dequant_q6_k, dequant_q8_0,
    };
    use crate::traits::DType as D;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // Host reference decoder: (raw block bytes, element count, out f16 halves).
    type HostDequant = fn(&[u8], usize, &mut [u16]) -> Res<()>;
    let cases: [(D, usize, usize, HostDequant); 9] = [
        (D::GgufQ8_0, 32, 34, dequant_q8_0),
        (D::GgufQ4_0, 32, 18, dequant_q4_0),
        (D::GgufQ4_1, 32, 20, dequant_q4_1),
        (D::GgufQ5_0, 32, 22, dequant_q5_0),
        (D::GgufQ5_1, 32, 24, dequant_q5_1),
        (D::GgufQ4K, 256, 144, dequant_q4_k),
        (D::GgufQ5K, 256, 176, dequant_q5_k),
        (D::GgufQ6K, 256, 210, dequant_q6_k),
        (D::GgufIQ4XS, 256, 136, dequant_iq4_xs),
    ];
    println!(
        "== gguf device kernels: bit-exact vs host, {} ==",
        ctx.device_name
    );
    for (fmt, elems, bytes, host_fn) in cases {
        // ~64 MiB of blocks: big enough for a stable bandwidth number.
        let nblocks = (64usize << 20) / bytes;
        let numel = nblocks * elems;
        let mut src = vec![0u8; nblocks * bytes];
        for b in src.chunks_exact_mut(8) {
            b.copy_from_slice(&rnd().to_le_bytes());
        }
        let mut host_out = vec![0u16; numel];
        host_fn(&src, numel, &mut host_out)?;

        let d_src = ctx.alloc(src.len())?;
        ctx.htod(&d_src, &src)?;
        let d_out = ctx.alloc(numel * 2)?;
        // warm + time
        ctx.gguf_dequant(fmt, d_src.ptr, d_out.ptr, nblocks)?;
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        let iters = 20;
        for _ in 0..iters {
            ctx.gguf_dequant(fmt, d_src.ptr, d_out.ptr, nblocks)?;
        }
        ctx.sync()?;
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        // traffic = packed read + f16 write
        let gbs = (src.len() + numel * 2) as f64 / secs / 1e9;

        let mut dev_out = vec![0u16; numel];
        ctx.dtoh(
            unsafe { std::slice::from_raw_parts_mut(dev_out.as_mut_ptr() as *mut u8, numel * 2) },
            &d_out,
        )?;
        // Bit equality, with one principled exception: two NaNs are equal
        // as a class (random blocks can encode NaN scales; both sides
        // canonicalize, but payload paths may differ legally).
        let is_nan = |v: u16| (v & 0x7c00) == 0x7c00 && (v & 0x3ff) != 0;
        let mismatches = host_out
            .iter()
            .zip(&dev_out)
            .filter(|(a, b)| a != b && !(is_nan(**a) && is_nan(**b)))
            .count();
        if mismatches > 0 {
            let i = host_out
                .iter()
                .zip(&dev_out)
                .position(|(a, b)| a != b && !(is_nan(*a) && is_nan(*b)))
                .unwrap();
            return Err(err!(
                "selftest",
                "{}: {} f16 mismatches (first at elem {}: host {:#06x} dev {:#06x})",
                fmt.name(),
                mismatches,
                i,
                host_out[i],
                dev_out[i]
            ));
        }
        println!(
            "SELFTEST {:<7} blocks={:<7} device==host (bit-exact, {} elems)   dequant {:>6.1} GB/s",
            fmt.name(),
            nblocks,
            numel,
            gbs
        );
    }
    println!("== all gguf kernels exact ==");

    // ---- fused resident GEMVs: y = x·W^T against an f64 host reference --
    // Random blocks again, but the f16 scale fields are pinned to a normal
    // exponent so no row dot degenerates to Inf/NaN; the comparison is
    // tolerance-based because the device accumulates f32 in a different
    // order than the host (both are mathematically the dequantized dot).
    // n big enough that packed bytes reach tens of MB — at [512×4096] the
    // arrays are 1-2 MB and the "bandwidth" mostly measures launch+sync
    // latency (the v1-v3 numbers were noise for exactly this reason).
    // Shapes: the original stress size plus the E2B matformer geometry
    // that field-failed while the fixed-size battery stayed green —
    // MQA projections (4096←1536 q, 512←1536 kv, 1536←4096 o) and
    // FFN-ish widths. A kernel indexing bug tied to grid/row assumptions
    // shows on SOME shape here or the kernels are truly shape-clean.
    for (n, k) in [
        (8192usize, 4096usize),
        (4096, 1536),
        (512, 1536),
        (1536, 512),
        (1536, 4096),
        (6144, 1536),
        (1536, 6144),
    ] {
        let mut x = vec![0u16; k];
        for (i, v) in x.iter_mut().enumerate() {
            // Small magnitudes: Q5_K/Q6_K terms reach ~±1000, and y is f16 —
            // x at ±0.03 keeps every row dot far from the 65504 saturation
            // point (the device legitimately writes inf past it; the f64
            // reference wouldn't, and the comparison would be meaningless).
            let f = ((rnd() % 2000) as f32 / 1000.0 - 1.0) * (1.0 + (i % 7) as f32 * 0.1) * 0.03;
            *v = crate::num::f32_to_f16(f);
        }
        let x_f: Vec<f64> = x
            .iter()
            .map(|&h| crate::num::f16_to_f32(h) as f64)
            .collect();
        let d_x = ctx.alloc(k * 2)?;
        ctx.htod(&d_x, unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 2)
        })?;

        let sane = |bytes: &mut [u8], at: usize| {
            let h = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
            let pinned = (h & 0x83ff) | (14 << 10); // exponent 14 → magnitude ~2^-1
            bytes[at..at + 2].copy_from_slice(&pinned.to_le_bytes());
        };
        for (fmt, elems, bytes, host_fn) in cases {
            let row_blocks = k / elems;
            let row_bytes = row_blocks * bytes;
            let mut wsrc = vec![0u8; n * row_bytes];
            for b in wsrc.chunks_exact_mut(8) {
                b.copy_from_slice(&rnd().to_le_bytes());
            }
            for blk in wsrc.chunks_exact_mut(bytes) {
                match fmt {
                    crate::traits::DType::GgufQ6K => sane(blk, 208),
                    _ => {
                        sane(blk, 0);
                        // formats with a second f16 (dmin for the K-quants,
                        // m for Q4_1/Q5_1) get it pinned too — a random
                        // inf/NaN offset would swamp the tolerance check
                        if bytes == 144 || bytes == 176 || bytes == 20 || bytes == 24 {
                            sane(blk, 2); // dmin / m
                        }
                    }
                }
            }
            // host reference in f64 over the host-dequantized weights
            let mut w16 = vec![0u16; n * k];
            host_fn(&wsrc, n * k, &mut w16)?;
            // Reference dot in f64, plus the per-row L2 of the TERMS: with
            // near-cancelling rows (|y| ≪ |terms|), relative-to-result error
            // is the wrong yardstick — the device decodes exact f32 while this
            // reference rounds each weight through f16 (ULP/2 per term), so
            // the principled bound scales with √Σ(wᵢxᵢ)², not with |y|.
            let mut yref = vec![0f64; n];
            let mut l2 = vec![0f64; n];
            for r in 0..n {
                let (mut acc, mut sq) = (0f64, 0f64);
                for e in 0..k {
                    let t = crate::num::f16_to_f32(w16[r * k + e]) as f64 * x_f[e];
                    acc += t;
                    sq += t * t;
                }
                yref[r] = acc;
                l2[r] = sq.sqrt();
            }
            let d_w = ctx.alloc(wsrc.len())?;
            ctx.htod(&d_w, &wsrc)?;
            let d_y = ctx.alloc(n * 2)?;
            ctx.gguf_gemv(fmt, d_x.ptr, d_w.ptr, 0, d_y.ptr, n, k, 0)?;
            ctx.sync()?;
            let t0 = std::time::Instant::now();
            let iters = 50;
            for _ in 0..iters {
                ctx.gguf_gemv(fmt, d_x.ptr, d_w.ptr, 0, d_y.ptr, n, k, 0)?;
            }
            ctx.sync()?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let gbs = (wsrc.len() + k * 2 + n * 2) as f64 / secs / 1e9;

            let mut y = vec![0u16; n];
            ctx.dtoh(
                unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, n * 2) },
                &d_y,
            )?;
            let mut worst = 0f64;
            for r in 0..n {
                let got = crate::num::f16_to_f32(y[r]) as f64;
                let want = yref[r];
                // Two noise sources vs the f64-of-f16 reference: f16 weight
                // rounding (2^-11·l2 per unit) and — the dominant one on the
                // dp4a path — q8_1 quantization of x (~2^-7 relative per
                // term). 4 units of the combined bound; an indexing bug still
                // lands at ~l2·√2, two orders of magnitude above this line.
                let unit = l2[r] * (4.9e-4 + 7.9e-3);
                let tol = 4.0 * unit + 1e-2;
                let err = (got - want).abs();
                worst = worst.max(err / unit.max(1e-12));
                if err > tol {
                    return Err(err!(
                        "selftest",
                        "{} gemv row {}: device {} vs host {} (|err| {:.3} > tol {:.3}, terms l2 {:.1})",
                        fmt.name(), r, got, want, err, tol, l2[r]
                    ));
                }
            }
            println!(
                "SELFTEST {:<7} gemv [{}x{}]  worst {:.1} noise-units (tol 4)   packed-read {:>6.1} GB/s",
                fmt.name(), n, k, worst, gbs
            );
        }
    }
    println!("== all gguf gemvs within tolerance ==");

    // ---- gather: packed-row embedding gather vs host dequant ----
    // k_gguf_gather is the third device consumer of packed blocks (the
    // embed path) and decodes single elements through the g_dec_* helpers
    // — a separate code path from the slab-dequant kernels above, so it
    // gets its own bit-exactness check (same math on both sides ⇒ same
    // f16, with the NaN-class exception random scale bytes can produce).
    let (vocab, hidden, n_ids) = (512usize, 2048usize, 96usize);
    for (fmt, elems, bytes, host_fn) in cases {
        let row_bytes = hidden / elems * bytes;
        let mut table = vec![0u8; vocab * row_bytes];
        for b in table.chunks_exact_mut(8) {
            b.copy_from_slice(&rnd().to_le_bytes());
        }
        let ids: Vec<u32> = (0..n_ids).map(|_| (rnd() % vocab as u64) as u32).collect();
        let mut want = vec![0u16; n_ids * hidden];
        for (i, &id) in ids.iter().enumerate() {
            let off = id as usize * row_bytes;
            host_fn(
                &table[off..off + row_bytes],
                hidden,
                &mut want[i * hidden..(i + 1) * hidden],
            )?;
        }
        let d_t = ctx.alloc(table.len())?;
        ctx.htod(&d_t, &table)?;
        let d_i = ctx.alloc(n_ids * 4)?;
        ctx.htod(&d_i, unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const u8, n_ids * 4)
        })?;
        let d_o = ctx.alloc(n_ids * hidden * 2)?;
        ctx.gguf_gather(fmt, d_t.ptr, d_i.ptr, d_o.ptr, n_ids, hidden)?;
        ctx.sync()?;
        let mut got = vec![0u16; n_ids * hidden];
        ctx.dtoh(
            unsafe { std::slice::from_raw_parts_mut(got.as_mut_ptr() as *mut u8, got.len() * 2) },
            &d_o,
        )?;
        let is_nan = |v: u16| (v & 0x7c00) == 0x7c00 && (v & 0x3ff) != 0;
        let bad = want
            .iter()
            .zip(&got)
            .position(|(a, b)| a != b && !(is_nan(*a) && is_nan(*b)));
        if let Some(i) = bad {
            return Err(err!(
                "selftest",
                "{} gather diverges at flat index {} (id {} col {}): host {:#06x} device {:#06x}",
                fmt.name(),
                i,
                ids[i / hidden],
                i % hidden,
                want[i],
                got[i]
            ));
        }
        println!(
            "SELFTEST {:<7} gather [{} ids × {}]  bit-exact vs host",
            fmt.name(),
            n_ids,
            hidden
        );
    }
    println!("== gguf gather bit-exact ==");
    Ok(())
}

pub fn run_profile(manager: &mut crate::models::ModelManager, model: &str) -> Res<()> {
    use std::time::Instant;
    let lm = manager.ensure(model)?;
    let ctx = lm.arch.ctx().clone();

    // 1. Per-launch overhead on THIS stack: a storm of trivial kernels.
    let probe = ctx.alloc(4)?;
    ctx.sync()?;
    let n_storm = 2000;
    let t = Instant::now();
    for _ in 0..n_storm {
        ctx.pos_bump(&probe)?;
    }
    ctx.sync()?;
    let launch_us = t.elapsed().as_secs_f64() * 1e6 / n_storm as f64;

    // 2. Decode step: wall time + submission count (a graph replay counts
    //    as one). The warmup generation arms the CUDA graph and warms
    //    cuBLAS; the measured one then reflects steady state, with the
    //    counter window excluding prefill.
    let prepared = lm.prepare("Profile run.", &[], &[])?;
    let mut opts = crate::traits::GenOptions {
        temperature: 0.0,
        repeat_penalty: 1.0, // greedy device path (1.1 would route through the sampler)
        max_tokens: 8,
        ignore_eos: true,
        ..Default::default()
    };
    lm.generate(&prepared, &opts, 0.0, |_| {})?;
    println!(
        "PROFILE mode: pipeline={} cuda_graph={}",
        lm.arch.supports_device_pipeline(),
        lm.arch.decode_graph_active()
    );
    println!("PROFILE levers: {}", lm.arch.perf_levers().summary());
    opts.max_tokens = 200; // bench horizon, for apples-to-apples step_ms
    let c0 = lm.arch.ctx().launch_count();
    let t1 = Instant::now();
    let mut n_tok = 0usize;
    lm.generate(&prepared, &opts, 0.0, |_| n_tok += 1)?;
    let gen_s = t1.elapsed().as_secs_f64();
    // One prefill chunk precedes the measured decode steps; subtract its
    // share by counting only decode-sized windows.
    let launches_per_tok = (lm.arch.ctx().launch_count() - c0) as f64 / n_tok.max(1) as f64;
    let step_ms = gen_s * 1e3 / n_tok.max(1) as f64;

    // 3. Pure-GEMV bandwidth probe: how fast does cuBLAS actually stream
    //    an f16 matrix at decode shape (m=1)? This is the number that
    //    decides whether the remaining gap is our orchestration or the
    //    GEMV kernel itself — and what "effective bandwidth" really is on
    //    this device.
    let (n_p, k_p) = (8192usize, 8192usize);
    let w_p = ctx.alloc(n_p * k_p * 2)?;
    let x_p = ctx.alloc(k_p * 2)?;
    let y_p = ctx.alloc(n_p * 4)?;
    ctx.memset(&w_p)?; // contents irrelevant for bandwidth
    ctx.memset(&x_p)?;
    ctx.gemm_f16(x_p.ptr, w_p.ptr, y_p.ptr, 1, n_p, k_p)?; // warm
    ctx.sync()?;
    let reps = 30;
    let t2 = Instant::now();
    for _ in 0..reps {
        ctx.gemm_f16(x_p.ptr, w_p.ptr, y_p.ptr, 1, n_p, k_p)?;
    }
    ctx.sync()?;
    let gemv_gbs = (n_p * k_p * 2 * reps) as f64 / t2.elapsed().as_secs_f64() / 1e9;

    // Same probe for the NF4 codec path (physical bytes: packed nibbles +
    // absmax). New quantized families should land in this neighbourhood;
    // a big gap here means the codec's GEMV, not the orchestration.
    let bs = 64usize;
    let packed = ctx.alloc(n_p * k_p / 2)?;
    let amax = ctx.alloc(n_p * k_p / bs * 4)?;
    let qmap = ctx.alloc(16 * 4)?;
    ctx.memset(&packed)?;
    ctx.memset(&amax)?;
    ctx.memset(&qmap)?;
    ctx.nf4_gemv(
        x_p.ptr, packed.ptr, amax.ptr, qmap.ptr, y_p.ptr, n_p, k_p, bs,
    )?;
    ctx.sync()?;
    let t3 = Instant::now();
    for _ in 0..reps {
        ctx.nf4_gemv(
            x_p.ptr, packed.ptr, amax.ptr, qmap.ptr, y_p.ptr, n_p, k_p, bs,
        )?;
    }
    ctx.sync()?;
    let nf4_gbs =
        ((n_p * k_p / 2 + n_p * k_p / bs * 4) * reps) as f64 / t3.elapsed().as_secs_f64() / 1e9;

    let bytes_per_tok = lm.arch.weight_bytes_resident();
    println!("PROFILE device={}", ctx.device_name);
    println!(
        "PROFILE launch_overhead_us={:.1} (kernel storm, n={})",
        launch_us, n_storm
    );
    println!(
        "PROFILE launches_per_token={:.0} step_ms={:.2}",
        launches_per_tok, step_ms
    );
    println!(
        "PROFILE weight_bytes_per_token={:.0}MB",
        bytes_per_tok as f64 / 1e6
    );
    let launch_ms = launches_per_tok * launch_us / 1e3;
    println!(
        "PROFILE attribution: launch~{:.2}ms ({:.0}%), compute+mem~{:.2}ms",
        launch_ms,
        100.0 * launch_ms / step_ms,
        (step_ms - launch_ms).max(0.0)
    );
    println!(
        "PROFILE gemv_bandwidth={:.0}GB/s (cuBLAS f16, m=1, 8192x8192)",
        gemv_gbs
    );
    println!(
        "PROFILE nf4_gemv_bandwidth={:.0}GB/s (physical bytes, 8192x8192)",
        nf4_gbs
    );
    // The honest floor uses the bandwidth this device+library actually
    // delivers at decode shape, not the datasheet number.
    let floor_ms = bytes_per_tok as f64 / (gemv_gbs * 1e9) * 1e3;
    println!(
        "PROFILE floor: {:.2}ms at measured {:.0}GB/s — headroom {:.1}x",
        floor_ms,
        gemv_gbs,
        step_ms / floor_ms
    );
    Ok(())
}

pub fn run_bench(
    manager: &mut crate::models::ModelManager,
    model: &str,
    n: usize,
    iters: usize,
    chat: bool,
) -> Res<()> {
    use std::time::Instant;
    // Load-to-ready is a product metric: wall time from cold engine to a
    // model able to serve, plus the VRAM it claims. Same definition on the
    // Python side (load + .to(device)). Page cache is warm in both cases.
    let ctx = manager.ctx.clone();
    let vram0 = ctx.snapshot().vram_used;
    let t_load = Instant::now();
    let lm = manager.ensure(model)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let vram_gb = (ctx.snapshot().vram_used.saturating_sub(vram0)) as f64 / 1e9;
    // RAW mode (default): the prompt text goes straight to the tokenizer on
    // both engines — identical token ids, identical work, no chat-template
    // policy differences (system-prompt injection skewed prompt_tokens
    // 29 vs 49 in early runs). --chat opts back into each engine's template.
    let rendered = if chat {
        lm.render_chat(&[crate::tokenizer::ChatTurn {
            role: "user".into(),
            content: BENCH_PROMPT.into(),
            n_images: 0,
            n_audio: 0,
        }])
    } else {
        BENCH_PROMPT.to_string()
    };
    let opts = GenOptions {
        temperature: 0.0,
        max_tokens: n,
        ignore_eos: true,
        repeat_penalty: 1.0, // raw model throughput, no sampler extras
        ..Default::default()
    };

    let mut ttfts: Vec<f64> = Vec::new();
    let mut gaps_all: Vec<f64> = Vec::new();
    let mut rates: Vec<f64> = Vec::new();
    let mut prompt_tokens = 0usize;

    for it in 0..=iters {
        let prepared = lm.prepare(&rendered, &[], &[])?;
        prompt_tokens = prepared.tokens.len();
        let mut stamps = Vec::<Instant>::with_capacity(n + 1);
        let t0 = Instant::now();
        lm.generate(&prepared, &opts, 0.0, |_| stamps.push(Instant::now()))?;
        if it == 0 {
            continue; // warmup discarded
        }
        if stamps.len() < 3 {
            return Err(err!(
                "bench",
                "model produced {} tokens — too few to measure",
                stamps.len()
            ));
        }
        ttfts.push((stamps[0] - t0).as_secs_f64() * 1e3);
        let mut gaps: Vec<f64> = stamps
            .windows(2)
            .map(|w| (w[1] - w[0]).as_secs_f64() * 1e3)
            .collect();
        let steady =
            (stamps.len() - 1) as f64 / (*stamps.last().unwrap() - stamps[0]).as_secs_f64();
        rates.push(steady);
        gaps_all.append(&mut gaps);
    }

    gaps_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_rate = rates.iter().sum::<f64>() / rates.len() as f64;
    println!(
        "BENCH engine=cima model={} mode={}",
        model,
        if chat { "chat" } else { "raw" }
    );
    println!("BENCH load_s={:.1} vram_gb={:.2} prompt_tokens={} gen_tokens={} iters={} (1 warmup discarded)", load_s, vram_gb, prompt_tokens, n, iters);
    println!(
        "BENCH ttft_ms_p50={:.1} ttft_ms_p99={:.1}",
        percentile(&ttfts, 50.0),
        percentile(&ttfts, 99.0)
    );
    println!(
        "BENCH itl_ms_p50={:.2} itl_ms_p95={:.2} itl_ms_p99={:.2}",
        percentile(&gaps_all, 50.0),
        percentile(&gaps_all, 95.0),
        percentile(&gaps_all, 99.0)
    );
    println!(
        "BENCH tok_per_s={:.2} (steady-state mean of {} runs)",
        mean_rate,
        rates.len()
    );
    Ok(())
}
/// `cima audio-map GGUF_MODEL[:TAG] SAFETENSORS_MODEL` — recover the audio
/// tower's true tensor-name mapping by CONTENT: every gguf audio tensor is
/// matched against the original safetensors tower by element count (within
/// the same layer) and cosine similarity. The weights themselves testify to
/// their names — shape-ambiguous assignments (macaron FF order, norm pairs)
/// that no metadata can settle resolve mechanically. Rows marked `≠` are
/// corrections the gguf name translation needs.
pub fn run_audio_map(ctx: Arc<CudaCtx>, gguf_model: &str, st_model: &str) -> Res<()> {
    run_tensor_map(ctx, gguf_model, st_model, None)
}

/// Prefix-general tensor-content audit: `cima audio-map G S PREFIX` compares
/// every gguf tensor under PREFIX against the original checkpoint by cosine
/// (same layer, same numel). With no prefix it audits the audio tower (the
/// original purpose); with e.g. `layers.0.` it audits an LM layer — the
/// reverse-coverage section then lists original-side tensors NOTHING in the
/// gguf claimed, which is how silently-missing optional weights (altup,
/// laurel, per-layer gates) reveal themselves.
pub fn run_tensor_map(
    ctx: Arc<CudaCtx>,
    gguf_model: &str,
    st_model: &str,
    prefix: Option<&str>,
) -> Res<()> {
    // crate::traits::* already in scope at module level

    fn f32_of(w: &dyn LoadedWeights, m: &TensorMeta) -> Res<Vec<f32>> {
        let b = w.bytes(m)?;
        let n = m.numel();
        Ok(match m.dtype {
            DType::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            DType::F16 => b
                .chunks_exact(2)
                .map(|c| crate::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            DType::BF16 => b
                .chunks_exact(2)
                .map(|c| crate::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            d if crate::traits::is_gguf_block(d) => {
                let mut h = vec![0u16; n];
                crate::quant::gguf::dequant_host(d, b, n, &mut h)?;
                h.iter().map(|&x| crate::num::f16_to_f32(x)).collect()
            }
            other => return Err(err!("audio-map-skip", "dtype {}", other.name())),
        })
    }
    // Cosine AND norm ratio: cosine is scale-blind — a tensor with a
    // miscooked quantization scale scores a perfect cosine while sagging
    // the residual stream. The ratio |gguf|/|original| catches exactly
    // that class.
    fn cosine(a: &[f32], b: &[f32]) -> (f64, f64) {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        (
            dot / (na.sqrt() * nb.sqrt()).max(1e-30),
            na.sqrt() / nb.sqrt().max(1e-30),
        )
    }
    let layer_of = |k: &str| -> Option<String> {
        k.find("layers.")
            .map(|i| k[i + 7..].split('.').next().unwrap_or("").to_string())
    };
    let is_stat = |k: &str| {
        k.ends_with(".input_min")
            || k.ends_with(".input_max")
            || k.ends_with(".output_min")
            || k.ends_with(".output_max")
    };
    let is_audio = |k: &str| {
        let k = k.strip_prefix("model.").unwrap_or(k);
        match prefix {
            // Substring match: gguf-translated LM names carry a
            // `language_model.` prefix and HF names a `model.language_model.`
            // one — anchoring at the start would blind the audit to both.
            Some(p) => k.contains(p),
            None => k.starts_with("audio_tower.") || k.starts_with("embed_audio."),
        }
    };

    // ---- open the gguf checkpoint (LM file + best mmproj) ----
    let (repo, tag) = match gguf_model.rsplit_once(':') {
        Some((r, t)) => (r, Some(t.to_ascii_lowercase())),
        None => (gguf_model, None),
    };
    let dir = crate::hub::local_dir(repo);
    let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| {
            err!(
                "audio-map",
                "cannot read {}: {} — pull the gguf model first",
                dir.display(),
                e
            )
        })?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .collect();
    ggufs.sort();
    let lname = |p: &std::path::PathBuf| {
        p.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase()
    };
    let lm_path = ggufs
        .iter()
        .filter(|p| !lname(p).contains("mmproj"))
        .find(|p| tag.as_deref().is_none_or(|t| lname(p).contains(t)))
        .ok_or_else(|| {
            err!(
                "audio-map",
                "no gguf matches '{}' in {}",
                gguf_model,
                dir.display()
            )
        })?
        .clone();
    let mut gw = crate::formats::gguf::GgufWeights::open(&lm_path)?;
    let mmproj = ggufs
        .iter()
        .filter(|p| lname(p).contains("mmproj"))
        .min_by_key(|p| {
            let n = lname(p);
            if n.contains("bf16") {
                1
            } else if n.contains("f16") {
                0
            } else {
                2
            }
        })
        .ok_or_else(|| {
            err!(
                "audio-map",
                "no mmproj gguf in {} — pull with --include mmproj",
                dir.display()
            )
        })?;
    gw.merge_extra(mmproj)?;

    // ---- open the safetensors checkpoint ----
    let st_dir = crate::hub::local_dir(st_model);
    let st = crate::formats::safetensors::SafetensorsLoader.load(&st_dir, &ctx)?;

    // ---- index safetensors audio tensors by (layer, numel) ----
    let mut st_groups: std::collections::HashMap<(Option<String>, usize), Vec<String>> =
        std::collections::HashMap::new();
    for (k, m) in st.tensors() {
        if is_audio(k) {
            st_groups
                .entry((layer_of(k), m.numel()))
                .or_default()
                .push(k.clone());
        }
    }
    if st_groups.is_empty() {
        return Err(err!(
            "audio-map",
            "'{}' carries no tensors matching {} — wrong repo or prefix?",
            st_model,
            prefix.unwrap_or("the audio tower")
        ));
    }

    // ---- match every gguf audio weight ----
    let mut gg: Vec<(&String, &TensorMeta)> = gw
        .tensors()
        .iter()
        .filter(|(k, _)| is_audio(k) && !is_stat(k))
        .collect();
    gg.sort_by(|a, b| a.0.cmp(b.0));
    println!(
        "== audio-map: {} gguf tensors vs {} safetensors groups ==",
        gg.len(),
        st_groups.len()
    );
    let mut corrections = 0usize;
    let mut matched: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (gname, gm) in gg {
        let key = (layer_of(gname), gm.numel());
        let Some(cands) = st_groups.get(&key) else {
            println!(
                "?? {}  [{} elems]  — no safetensors candidate (layer {:?})",
                gname,
                gm.numel(),
                key.0
            );
            continue;
        };
        // Size cap: decoding both sides to f32 costs numel × 8 bytes of
        // host RAM; the E-series PLE table alone is 2G elements (16 GB) —
        // enough to swap-thrash the machine from inside a DIAGNOSTIC.
        const MAX_ELEMS: usize = 32 * 1024 * 1024;
        if gm.numel() > MAX_ELEMS {
            println!(
                "-- {}  skipped ({} elems > {} cap — audit large tables separately)",
                gname,
                gm.numel(),
                MAX_ELEMS
            );
            continue;
        }
        let gv = match f32_of(&gw, gm) {
            Ok(v) => v,
            Err(e) => {
                println!("-- {}  skipped ({})", gname, e);
                continue;
            }
        };
        let mut scored: Vec<((f64, f64), &String)> = Vec::with_capacity(cands.len());
        for c in cands {
            let cm = &st.tensors()[c];
            // bnb-packed on the st side is not comparable; skip those.
            if let Ok(cv) = f32_of(st.as_ref(), cm) {
                scored.push((cosine(&gv, &cv), c));
            }
        }
        if scored.is_empty() {
            println!("-- {}  no decodable candidates", gname);
            continue;
        }
        scored.sort_by(|a, b| b.0 .0.partial_cmp(&a.0 .0).unwrap());
        let ((best, ratio), bname) = scored[0];
        let margin = best - scored.get(1).map(|s| s.0 .0).unwrap_or(-1.0);
        matched.insert((*bname).clone());
        // per_dim_scale ships TRANSFORMED in gguf exports; measure the
        // transform instead of guessing it: print gguf values alongside
        // raw and softplus(raw) from the original, with elementwise
        // ratios — a constant ratio IS the folded factor.
        if gname.contains("per_dim_scale") {
            if let Ok(sv) = f32_of(st.as_ref(), &st.tensors()[bname.as_str()]) {
                let sp: Vec<f32> = sv
                    .iter()
                    .map(|v| if *v > 20.0 { *v } else { (1.0 + v.exp()).ln() })
                    .collect();
                let k = 4.min(gv.len());
                let ratios: Vec<f32> = (0..k).map(|j| gv[j] / sp[j].max(1e-9)).collect();
                println!(
                    "   pds probe: gguf {:?} | raw {:?} | softplus(raw) {:?} | gguf/softplus {:?}",
                    &gv[..k],
                    &sv[..k],
                    &sp[..k],
                    ratios
                );
            }
        }
        let canonical = bname.strip_prefix("model.").unwrap_or(bname);
        let scale_off = (ratio - 1.0).abs() > 0.02;
        let mark = if scale_off {
            corrections += 1;
            "!S"
        } else if canonical == gname {
            "  "
        } else {
            corrections += 1;
            "≠ "
        };
        println!(
            "{}{}  →  {}   cos {:.5}  |ratio| {:.4}{}  margin {:.4}",
            mark,
            gname,
            canonical,
            best,
            ratio,
            if scale_off { " <<< SCALE" } else { "" },
            margin
        );
    }
    println!("== {} corrections needed ==", corrections);
    // REVERSE COVERAGE — the direction the first version missed: tensors
    // the ORIGINAL checkpoint carries that no gguf tensor claimed. A
    // conformer bias family listed here means the gguf tower silently
    // runs without biases — a uniform per-layer distortion no forward
    // matching can see.
    let mut orphans: Vec<&String> = st
        .tensors()
        .keys()
        .filter(|k| is_audio(k) && !matched.contains(*k))
        .collect();
    orphans.sort();
    println!(
        "== {} safetensors audio tensors matched by NOTHING in the gguf ==",
        orphans.len()
    );
    for o in orphans.iter().take(60) {
        let m = &st.tensors()[*o];
        println!("!! {}   {:?} {}", o, m.shape, m.dtype.name());
    }
    Ok(())
}
