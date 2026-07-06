//! # formats — weight container formats
//!
//! How tensors are *stored on disk*. [`gguf`] reads the llama.cpp
//! ecosystem's single-file format (metadata + quantized blocks; tensor
//! names translated to Hugging Face convention). [`safetensors`] memory-maps
//! `.safetensors` shards (zero-copy: tensors are `&[u8]` views into the
//! mapping until upload). Quantization codecs — how the bytes *inside* a
//! tensor are encoded — live in `crate::quant`.
//!
//! # Adding a format (e.g. pytorch .bin)
//! Implement [`crate::traits::LoadedWeights`] over your container (name →
//! [`crate::traits::TensorMeta`] + byte views) and register the file
//! extension in `ModelManager::open_weights`. Builders consume the traits —
//! no model code changes.

pub mod gguf {
    //! GGUF container format (v2/v3): the llama.cpp ecosystem's single-file
    //! checkpoint. One mmap, zero-copy tensor views, typed metadata access.
    //!
    //! A GGUF file is: header (magic "GGUF", version, tensor count, kv count),
    //! a metadata key-value section (typed values, including arrays — the
    //! tokenizer lives here), tensor descriptors (name, dims, ggml type, data
    //! offset), then an aligned data section. Quantized ggml types (Q8_0,
    //! Q4_K, Q6_K, …) are block formats; their decode lives in
    //! `crate::quant::gguf`.
    //!
    //! GGUF names tensors in ggml style (`blk.0.attn_q.weight`); the engine's
    //! builders speak Hugging Face names. [`GgufWeights`] translates at the
    //! map level, so builders stay container-agnostic. Model hyper-parameters
    //! and the tokenizer are synthesized from metadata — a GGUF repo needs no
    //! config.json or tokenizer.json.

    use std::collections::HashMap;
    use std::path::Path;

    use crate::traits::{DType, LoadedWeights, Res, TensorMeta};
    use crate::{err, log};

    /// One parsed metadata value.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        U64(u64),
        I64(i64),
        F64(f64),
        Bool(bool),
        Str(String),
        Arr(Vec<Value>),
    }

    impl Value {
        pub fn as_u64(&self) -> Option<u64> {
            match self {
                Value::U64(v) => Some(*v),
                Value::I64(v) if *v >= 0 => Some(*v as u64),
                _ => None,
            }
        }
        pub fn as_usize(&self) -> Option<usize> {
            self.as_u64().map(|v| v as usize)
        }
        pub fn as_f32(&self) -> Option<f32> {
            match self {
                Value::F64(v) => Some(*v as f32),
                Value::U64(v) => Some(*v as f32),
                Value::I64(v) => Some(*v as f32),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Value::Bool(b) => Some(*b),
                _ => None,
            }
        }
        pub fn as_arr(&self) -> Option<&[Value]> {
            match self {
                Value::Arr(a) => Some(a),
                _ => None,
            }
        }
    }

    /// ggml tensor type ids (ggml.h) — only the ones the engine executes.
    fn ggml_dtype(t: u32) -> Res<DType> {
        Ok(match t {
            0 => DType::F32,
            1 => DType::F16,
            // Legacy 32-grain formats: llama.cpp's K-quantizer falls back to
            // Q5_0/Q5_1/Q4_0 for tensors whose row length isn't a multiple
            // of 256 (Qwen2.5-0.5B, hidden 896, ships them inside Q4_K_M).
            2 => DType::GgufQ4_0,
            3 => DType::GgufQ4_1,
            6 => DType::GgufQ5_0,
            7 => DType::GgufQ5_1,
            8 => DType::GgufQ8_0,
            12 => DType::GgufQ4K,
            13 => DType::GgufQ5K,
            14 => DType::GgufQ6K,
            23 => DType::GgufIQ4XS,
            30 => DType::BF16,
            other => {
                return Err(err!(
                    "gguf",
                    "ggml tensor type {} is not registered in this build (supported: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, IQ4_XS). \
                     Pull a Q4_K_* / Q6_K / Q8_0 quantization of this repo.",
                    other
                ))
            }
        })
    }

    /// Bytes a tensor of `dtype` with `numel` elements occupies on disk.
    pub fn storage_bytes(dtype: DType, numel: usize) -> usize {
        match dtype {
            DType::GgufQ8_0 => (numel / 32) * 34,
            DType::GgufQ4_0 => (numel / 32) * 18,
            DType::GgufQ4_1 => (numel / 32) * 20,
            DType::GgufQ5_0 => (numel / 32) * 22,
            DType::GgufQ5_1 => (numel / 32) * 24,
            DType::GgufQ4K => (numel / 256) * 144,
            DType::GgufQ5K => (numel / 256) * 176,
            DType::GgufQ6K => (numel / 256) * 210,
            DType::GgufIQ4XS => (numel / 256) * 136,
            other => numel * other.size(),
        }
    }

    struct Cursor<'a> {
        b: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn take(&mut self, n: usize) -> Res<&'a [u8]> {
            if self.pos + n > self.b.len() {
                // Quiet: a truncated buffer is an expected, retryable state
                // for the preflight's growing-window reader. GgufWeights::open
                // turns a genuine truncation into a logged error at its level.
                return Err(crate::traits::EngineError::quiet(
                    "gguf",
                    format!(
                        "truncated file: need {} bytes at offset {}, have {}",
                        n,
                        self.pos,
                        self.b.len() - self.pos.min(self.b.len())
                    ),
                ));
            }
            let s = &self.b[self.pos..self.pos + n];
            self.pos += n;
            Ok(s)
        }
        fn u32(&mut self) -> Res<u32> {
            Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
        }
        fn u64(&mut self) -> Res<u64> {
            Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
        }
        fn string(&mut self) -> Res<String> {
            let n = self.u64()? as usize;
            if n > 1 << 24 {
                return Err(err!(
                    "gguf",
                    "implausible string length {} — corrupt header",
                    n
                ));
            }
            Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
        }
        /// One typed value; `ty` per the GGUF spec.
        fn value(&mut self, ty: u32, depth: usize) -> Res<Value> {
            if depth > 2 {
                return Err(err!(
                    "gguf",
                    "metadata arrays nested deeper than the spec allows"
                ));
            }
            Ok(match ty {
                0 => Value::U64(self.take(1)?[0] as u64),       // u8
                1 => Value::I64(self.take(1)?[0] as i8 as i64), // i8
                2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64), // u16
                3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64), // i16
                4 => Value::U64(self.u32()? as u64), // u32
                5 => Value::I64(self.u32()? as i32 as i64), // i32
                6 => Value::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64), // f32
                7 => Value::Bool(self.take(1)?[0] != 0), // bool
                8 => Value::Str(self.string()?),         // string
                9 => {
                    // array: elem type + count + values
                    let et = self.u32()?;
                    let n = self.u64()? as usize;
                    if n > 1 << 26 {
                        return Err(err!(
                            "gguf",
                            "implausible array length {} — corrupt header",
                            n
                        ));
                    }
                    let mut a = Vec::with_capacity(n.min(1 << 20));
                    for _ in 0..n {
                        a.push(self.value(et, depth + 1)?);
                    }
                    Value::Arr(a)
                }
                10 => Value::U64(self.u64()?),        // u64
                11 => Value::I64(self.u64()? as i64), // i64
                12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())), // f64
                other => return Err(err!("gguf", "unknown metadata value type {}", other)),
            })
        }
    }

    /// A parsed, mmapped GGUF file.
    pub struct GgufWeights {
        /// One mmap per file part — single-file models have exactly one;
        /// llama.cpp `gguf-split` checkpoints (…-00001-of-0000N.gguf) have N.
        /// `TensorMeta.file` names the part a tensor lives in.
        parts: Vec<(String, crate::formats::safetensors::Mmap)>,
        /// HF-named tensor table (translated from ggml names), merged across parts.
        tensors: HashMap<String, TensorMeta>,
        pub meta: HashMap<String, Value>,
        /// `general.architecture` (e.g. "qwen2").
        pub architecture: String,
        data_start: usize,
    }

    /// Raw parse of one GGUF file: metadata + ggml-named tensor records
    /// (absolute offsets, truncation-checked). Name translation happens after
    /// the architecture is known (split parts other than the first don't
    /// carry it).
    struct GgufPart {
        file: String,
        map: crate::formats::safetensors::Mmap,
        meta: HashMap<String, Value>,
        /// (ggml name, dtype, row-major shape, ABSOLUTE byte offset, nbytes)
        raw: Vec<(String, DType, Vec<usize>, usize, usize)>,
        data_start: usize,
    }

    impl GgufWeights {
        pub fn open(path: &Path) -> Res<GgufWeights> {
            // gguf-split detection: …-NNNNN-of-MMMMM.gguf siblings load as one
            // logical checkpoint (metadata from the part that carries it,
            // tensor tables merged, bytes resolved per part).
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let paths: Vec<std::path::PathBuf> =
                match split_set(&name) {
                    Some((prefix, count)) => {
                        let dir = path.parent().unwrap_or_else(|| Path::new("."));
                        let mut v = Vec::with_capacity(count);
                        for i in 1..=count {
                            let p = dir.join(format!("{}-{:05}-of-{:05}.gguf", prefix, i, count));
                            if !p.is_file() {
                                return Err(err!(
                                "gguf",
                                "split checkpoint is missing part {} of {} ({}) — re-pull the repo",
                                i, count, p.display()
                            ));
                            }
                            v.push(p);
                        }
                        v
                    }
                    None => vec![path.to_path_buf()],
                };

            let mut parts_raw = Vec::with_capacity(paths.len());
            for p in &paths {
                parts_raw.push(parse_part(p)?);
            }
            // Architecture + merged metadata: the carrying part (part 1 in
            // split sets) wins on conflicts; others contribute split.* keys.
            let architecture = parts_raw
                .iter()
                .find_map(|p| {
                    p.meta
                        .get("general.architecture")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .ok_or_else(|| err!("gguf", "metadata is missing general.architecture"))?;
            let mut meta = HashMap::new();
            for p in parts_raw.iter().rev() {
                for (k, v) in &p.meta {
                    meta.insert(k.clone(), v.clone());
                }
            }

            let mut tensors = HashMap::new();
            let mut parts = Vec::with_capacity(parts_raw.len());
            let data_start = parts_raw[0].data_start;
            for p in parts_raw {
                for (gname, dtype, shape, offset, nbytes) in &p.raw {
                    let hf = translate_name(&architecture, gname);
                    if tensors
                        .insert(
                            hf.clone(),
                            TensorMeta {
                                name: hf.clone(),
                                dtype: *dtype,
                                shape: shape.clone(),
                                offset: *offset,
                                nbytes: *nbytes,
                                file: p.file.clone(),
                            },
                        )
                        .is_some()
                    {
                        return Err(err!(
                            "gguf",
                            "tensor '{}' appears in more than one split part",
                            hf
                        ));
                    }
                }
                parts.push((p.file, p.map));
            }
            log::info(&format!(
                "gguf: '{}' — {} part(s), {} tensors, arch={}, {} metadata keys",
                name,
                parts.len(),
                tensors.len(),
                architecture,
                meta.len()
            ));
            Ok(GgufWeights {
                parts,
                tensors,
                meta,
                architecture,
                data_start,
            })
        }

        pub fn meta_usize(&self, key: &str) -> Option<usize> {
            self.meta.get(key).and_then(|v| v.as_usize())
        }
        pub fn meta_f32(&self, key: &str) -> Option<f32> {
            self.meta.get(key).and_then(|v| v.as_f32())
        }
        pub fn meta_str(&self, key: &str) -> Option<&str> {
            self.meta.get(key).and_then(|v| v.as_str())
        }
        /// Offset of the aligned data section (tests).
        pub fn data_start(&self) -> usize {
            self.data_start
        }

        /// Merge a companion GGUF (an `mmproj-*.gguf` tower file) into this
        /// checkpoint. llama.cpp exports multimodal towers separately, under
        /// their own architecture key and tensor-name dialect —
        /// [`translate_mmproj`] rewrites them into the HF names the tower
        /// builders speak. Main-file metadata wins on key conflicts; duplicate
        /// tensor names are skipped with a warning (the main LM file is
        /// authoritative).
        pub fn merge_extra(&mut self, path: &Path) -> Res<()> {
            let part = parse_part(path)?;
            let mut added = 0usize;
            for (gname, dtype, shape, offset, nbytes) in &part.raw {
                let hf = translate_mmproj(gname);
                if self.tensors.contains_key(&hf) {
                    log::warn(&format!("mmproj tensor '{}' (from {}) collides with the main checkpoint — keeping the main tensor", hf, part.file));
                    continue;
                }
                self.tensors.insert(
                    hf.clone(),
                    TensorMeta {
                        name: hf,
                        dtype: *dtype,
                        shape: shape.clone(),
                        offset: *offset,
                        nbytes: *nbytes,
                        file: part.file.clone(),
                    },
                );
                added += 1;
            }
            for (k, v) in &part.meta {
                self.meta.entry(k.clone()).or_insert_with(|| v.clone());
            }
            log::info(&format!(
                "gguf: merged mmproj '{}' — {} tensors added ({} total)",
                part.file,
                added,
                self.tensors.len()
            ));
            self.parts.push((part.file, part.map));
            Ok(())
        }
    }

    impl LoadedWeights for GgufWeights {
        fn tensors(&self) -> &HashMap<String, TensorMeta> {
            &self.tensors
        }
        fn bytes(&self, meta: &TensorMeta) -> Res<&[u8]> {
            let (_, map) = self
                .parts
                .iter()
                .find(|(file, _)| *file == meta.file)
                .ok_or_else(|| {
                    err!(
                        "gguf",
                        "tensor '{}' references unknown part '{}'",
                        meta.name,
                        meta.file
                    )
                })?;
            map.bytes()
                .get(meta.offset..meta.offset + meta.nbytes)
                .ok_or_else(|| err!("gguf", "tensor '{}' range out of bounds", meta.name))
        }
    }

    /// Parse a `…-NNNNN-of-MMMMM.gguf` filename into (prefix, total parts).
    fn split_set(name: &str) -> Option<(String, usize)> {
        let stem = name.strip_suffix(".gguf")?;
        // …{prefix}-NNNNN-of-MMMMM
        let (rest, total) = stem.rsplit_once("-of-")?;
        let (prefix, no) = rest.rsplit_once('-')?;
        if no.len() == 5 && total.len() == 5 && no.chars().all(|c| c.is_ascii_digit()) {
            let count: usize = total.parse().ok()?;
            if count >= 2 {
                return Some((prefix.to_string(), count));
            }
        }
        None
    }

    /// Parse one GGUF file: header, metadata, tensor records with absolute
    /// offsets, truncation guard. Shapes come out row-major.
    /// Parse only the header of a GGUF byte prefix: KV metadata plus the
    /// tensor table as `(name, dtype, shape)`. This is the byte-range
    /// preflight path — no file length is available, so the data-section
    /// bounds check of [`GgufWeights::open`] does not apply here. A prefix
    /// that ends inside the header fails with a `gguf` error; callers
    /// fetch a larger range and retry.
    #[allow(clippy::type_complexity)]
    pub fn parse_header_bytes(
        b: &[u8],
    ) -> Res<(HashMap<String, Value>, Vec<(String, DType, Vec<usize>)>)> {
        if b.len() < 24 || &b[0..4] != b"GGUF" {
            return Err(err!("gguf", "not a GGUF payload (bad magic)"));
        }
        let mut c = Cursor { b, pos: 4 };
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(err!(
                "gguf",
                "GGUF version {} unsupported (this build reads v2/v3)",
                version
            ));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err(err!(
                "gguf",
                "implausible header counts (tensors={}, kv={})",
                n_tensors,
                n_kv
            ));
        }
        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty, 0)?;
            meta.insert(key, val);
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            if n_dims > 4 {
                return Err(err!(
                    "gguf",
                    "tensor '{}' has {} dims (max 4)",
                    name,
                    n_dims
                ));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()? as usize);
            }
            dims.reverse();
            let dtype = ggml_dtype(c.u32()?)?;
            let _off = c.u64()?;
            tensors.push((name, dtype, dims));
        }
        Ok((meta, tensors))
    }

    fn parse_part(path: &Path) -> Res<GgufPart> {
        // A real on-disk file that fails to parse IS a fatal error worth
        // logging; the quiet cursor errors (used for the preflight's
        // in-memory retry) are promoted to logged ones here at the file
        // boundary so a genuinely corrupt checkpoint still reports clearly.
        parse_part_inner(path).map_err(|e| err!("gguf", "{}: {}", path.display(), e.msg))
    }

    fn parse_part_inner(path: &Path) -> Res<GgufPart> {
        let map = crate::formats::safetensors::Mmap::open(path)?;
        let b = map.bytes();
        if b.len() < 24 || &b[0..4] != b"GGUF" {
            return Err(err!(
                "gguf",
                "'{}' is not a GGUF file (bad magic)",
                path.display()
            ));
        }
        let mut c = Cursor { b, pos: 4 };
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(err!(
                "gguf",
                "GGUF version {} unsupported (this build reads v2/v3)",
                version
            ));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err(err!(
                "gguf",
                "implausible header counts (tensors={}, kv={})",
                n_tensors,
                n_kv
            ));
        }
        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty, 0)?;
            meta.insert(key, val);
        }
        let mut raw0: Vec<(String, DType, Vec<usize>, u64)> = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            if n_dims > 4 {
                return Err(err!(
                    "gguf",
                    "tensor '{}' has {} dims (max 4)",
                    name,
                    n_dims
                ));
            }
            // GGUF stores dims innermost-first; the engine speaks
            // outermost-first (row-major shapes), so reverse.
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()? as usize);
            }
            dims.reverse();
            let dtype = ggml_dtype(c.u32()?)?;
            let off = c.u64()?;
            raw0.push((name, dtype, dims, off));
        }
        let align = meta
            .get("general.alignment")
            .and_then(|v| v.as_usize())
            .unwrap_or(32);
        let data_start = c.pos.div_ceil(align) * align;
        let file = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let mut raw = Vec::with_capacity(raw0.len());
        for (gname, dtype, shape, off) in raw0 {
            let numel: usize = shape.iter().product();
            let nbytes = storage_bytes(dtype, numel);
            let end = data_start + off as usize + nbytes;
            if end > b.len() {
                return Err(err!(
                    "gguf",
                    "tensor '{}' data ends at {} but '{}' is {} bytes — truncated",
                    gname,
                    end,
                    file,
                    b.len()
                ));
            }
            raw.push((gname, dtype, shape, data_start + off as usize, nbytes));
        }
        Ok(GgufPart {
            file,
            map,
            meta,
            raw,
            data_start,
        })
    }

    /// ggml → Hugging Face tensor names, per architecture. The builders only
    /// ever see HF names; adding a family means extending this table.
    pub fn translate_name(arch: &str, g: &str) -> String {
        if arch.starts_with("gemma4") || arch.starts_with("gemma3n") {
            return translate_gemma4(g);
        }
        match g {
            "token_embd.weight" => return "model.embed_tokens.weight".into(),
            "output_norm.weight" => return "model.norm.weight".into(),
            "output.weight" => return "lm_head.weight".into(),
            _ => {}
        }
        if let Some(rest) = g.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                let layer = &rest[..dot];
                let tail = &rest[dot + 1..];
                let hf_tail = match tail {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => other, // unknown tails pass through verbatim
                };
                return format!("model.layers.{}.{}", layer, hf_tail);
            }
        }
        let _ = arch;
        g.to_string()
    }

    /// gemma-4 (llama.cpp's reduced graph): map ggml names onto the HF names the
    /// gemma4 module requests (`language_model.*`). The ggml norm names are
    /// treacherous: `ffn_norm` is HF's *pre_feedforward* norm and
    /// `attn_post_norm` is HF's *post_attention* norm.
    /// Translate an mmproj (tower-file) tensor name into the HF names the
    /// gemma4 tower builders request. Dialects handled, in order:
    ///   1. already-HF names (with or without a `model.` prefix) pass through;
    ///   2. llama.cpp clip-style prefixes: `v.` → `vision_tower.`,
    ///      `a.` → `audio_tower.` (with clip tail renames for `v.blk.N.*`);
    ///   3. `mm.*` projector names → the `embed_vision.`/`embed_audio.`
    ///      embedding projections.
    ///
    /// Unknown names pass through UNCHANGED — the tower builders probe by
    /// prefix, so an untranslated file degrades to "towers disabled" (and the
    /// loader's debug dump exposes the raw names for extending this table)
    /// rather than a crash.
    pub fn translate_mmproj(g: &str) -> String {
        let g = g.strip_prefix("model.").unwrap_or(g);
        if g.starts_with("vision_tower.")
            || g.starts_with("audio_tower.")
            || g.starts_with("embed_vision.")
            || g.starts_with("embed_audio.")
        {
            return g.to_string();
        }
        // projector names
        match g {
            "mm.input_projection.weight" | "mm.vision.input_projection.weight" => {
                return "embed_vision.embedding_projection.weight".into()
            }
            "mm.audio.input_projection.weight" | "mm.a.input_projection.weight" => {
                return "embed_audio.embedding_projection.weight".into()
            }
            _ => {}
        }
        // clip-style vision blocks: v.blk.N.<tail>
        if let Some(rest) = g.strip_prefix("v.blk.") {
            if let Some(dot) = rest.find('.') {
                let (layer, tail) = (&rest[..dot], &rest[dot + 1..]);
                // Same QAT clip-bound translation as the audio blocks.
                for stat in [".input_min", ".input_max", ".output_min", ".output_max"] {
                    if let Some(stem) = tail.strip_suffix(stat) {
                        let hf_stem = match stem {
                            "attn_q" => "self_attn.q_proj",
                            "attn_k" => "self_attn.k_proj",
                            "attn_v" => "self_attn.v_proj",
                            "attn_out" => "self_attn.o_proj",
                            "ffn_gate" => "mlp.gate_proj",
                            "ffn_up" => "mlp.up_proj",
                            "ffn_down" => "mlp.down_proj",
                            other => other,
                        };
                        return format!(
                            "vision_tower.encoder.layers.{}.{}{}",
                            layer, hf_stem, stat
                        );
                    }
                }
                let hf_tail = match tail {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_out.weight" => "self_attn.o_proj.weight",
                    "attn_out.bias" => "self_attn.o_proj.bias",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "ln1.weight" | "attn_norm.weight" => "input_layernorm.weight",
                    // sandwich norms as serialized on the E4B mmproj export:
                    // ln1 = pre-attn, attn_post_norm = post-attn,
                    // ln2 = pre-ffn, ffn_post_norm = post-ffn.
                    "attn_post_norm.weight" | "post_attention_norm.weight" => {
                        "post_attention_layernorm.weight"
                    }
                    "ln2.weight" | "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                    "ffn_post_norm.weight" | "post_ffw_norm.weight" => {
                        "post_feedforward_layernorm.weight"
                    }
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => other,
                };
                return format!("vision_tower.encoder.layers.{}.{}", layer, hf_tail);
            }
        }
        match g {
            "v.patch_embd.weight" => return "vision_tower.patch_embedder.input_proj.weight".into(),
            "v.position_embd.weight" => {
                return "vision_tower.patch_embedder.position_embedding_table".into()
            }
            _ => {}
        }
        // audio conformer blocks: a.blk.N.<tail> — tail table read off the
        // E4B mmproj export (this is the llama.cpp serialization; the
        // `*_1`-suffixed FF family is the SECOND macaron feed-forward).
        // The `.input_min/max` / `.output_min/max` calibration stats pass
        // through under the audio prefix and are never requested.
        if let Some(rest) = g.strip_prefix("a.blk.") {
            if let Some(dot) = rest.find('.') {
                let (layer, tail) = (&rest[..dot], &rest[dot + 1..]);
                // Calibration stats (QAT clip bounds — config.use_clipped_linears):
                // the engine resolves them as `{hf_stem}.input_min` etc., so the
                // llama.cpp stems must translate exactly like their weights or
                // the tower silently runs UNCLIPPED (uniform per-layer error).
                for stat in [".input_min", ".input_max", ".output_min", ".output_max"] {
                    if let Some(stem) = tail.strip_suffix(stat) {
                        let hf_stem = match stem {
                            "ffn_up" => "feed_forward1.ffw_layer_1",
                            "ffn_down" => "feed_forward1.ffw_layer_2",
                            "ffn_up_1" => "feed_forward2.ffw_layer_1",
                            "ffn_down_1" => "feed_forward2.ffw_layer_2",
                            "attn_q" => "self_attn.q_proj",
                            "attn_k" => "self_attn.k_proj",
                            "attn_v" => "self_attn.v_proj",
                            "attn_out" => "self_attn.post",
                            "attn_k_rel" => "self_attn.relative_k_proj",
                            "conv_pw1" => "lconv1d.linear_start",
                            "conv_pw2" => "lconv1d.linear_end",
                            "conv_dw" => "lconv1d.depthwise_conv1d",
                            other => other,
                        };
                        return format!("audio_tower.layers.{}.{}{}", layer, hf_stem, stat);
                    }
                }
                // Two same-shape mapping groups can't be pinned by shape checks;
                // both were settled mechanically by `cima audio-map` cosine
                // matching against the original safetensors (cos 1.00000 across
                // all 12 layers): the unsuffixed ffn family IS feed_forward1,
                // and the lconv norm pair is CROSSED relative to its names —
                // the export's `norm_conv` is the in-conv norm and its
                // `conv_norm` is the block's pre-LayerNorm.
                let (ff_a, ff_b) = ("feed_forward1", "feed_forward2");
                let (cn_a, cn_b) = ("lconv1d.conv_norm.weight", "lconv1d.pre_layer_norm.weight");
                let owned: String;
                let hf_tail: &str = match tail {
                    "ffn_norm.weight" => {
                        owned = format!("{}.pre_layer_norm.weight", ff_a);
                        &owned
                    }
                    "ffn_up.weight" => {
                        owned = format!("{}.ffw_layer_1.weight", ff_a);
                        &owned
                    }
                    "ffn_up.bias" => {
                        owned = format!("{}.ffw_layer_1.bias", ff_a);
                        &owned
                    }
                    "ffn_down.weight" => {
                        owned = format!("{}.ffw_layer_2.weight", ff_a);
                        &owned
                    }
                    "ffn_down.bias" => {
                        owned = format!("{}.ffw_layer_2.bias", ff_a);
                        &owned
                    }
                    "ffn_post_norm.weight" => {
                        owned = format!("{}.post_layer_norm.weight", ff_a);
                        &owned
                    }
                    "ffn_norm_1.weight" => {
                        owned = format!("{}.pre_layer_norm.weight", ff_b);
                        &owned
                    }
                    "ffn_up_1.weight" => {
                        owned = format!("{}.ffw_layer_1.weight", ff_b);
                        &owned
                    }
                    "ffn_up_1.bias" => {
                        owned = format!("{}.ffw_layer_1.bias", ff_b);
                        &owned
                    }
                    "ffn_down_1.weight" => {
                        owned = format!("{}.ffw_layer_2.weight", ff_b);
                        &owned
                    }
                    "ffn_down_1.bias" => {
                        owned = format!("{}.ffw_layer_2.bias", ff_b);
                        &owned
                    }
                    "ffn_post_norm_1.weight" => {
                        owned = format!("{}.post_layer_norm.weight", ff_b);
                        &owned
                    }
                    "norm_conv.weight" => cn_a,
                    "conv_norm.weight" => cn_b,
                    "attn_pre_norm.weight" => "norm_pre_attn.weight",
                    "attn_post_norm.weight" => "norm_post_attn.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_out.weight" => "self_attn.post.weight",
                    "attn_out.bias" => "self_attn.post.bias",
                    "attn_k_rel.weight" => "self_attn.relative_k_proj.weight",
                    "per_dim_scale.weight" => "self_attn.per_dim_scale",
                    "conv_pw1.weight" => "lconv1d.linear_start.weight",
                    "conv_pw1.bias" => "lconv1d.linear_start.bias",
                    "conv_dw.weight" => "lconv1d.depthwise_conv1d.weight",
                    "conv_pw2.weight" => "lconv1d.linear_end.weight",
                    "conv_pw2.bias" => "lconv1d.linear_end.bias",
                    "ln2.weight" => "norm_out.weight",
                    other => other,
                };
                return format!("audio_tower.layers.{}.{}", layer, hf_tail);
            }
        }
        // audio subsampler + projections: top-level names read off the E4B
        // export. `pre_encode.out` carries a bias and the builder's output
        // projection is the only slot that takes one — that pins the mapping.
        match g {
            "a.conv1d.0.weight" => {
                return "audio_tower.subsample_conv_projection.layer0.conv.weight".into()
            }
            "a.conv1d.0.norm.weight" => {
                return "audio_tower.subsample_conv_projection.layer0.norm.weight".into()
            }
            "a.conv1d.1.weight" => {
                return "audio_tower.subsample_conv_projection.layer1.conv.weight".into()
            }
            "a.conv1d.1.norm.weight" => {
                return "audio_tower.subsample_conv_projection.layer1.norm.weight".into()
            }
            "a.input_projection.weight" => {
                return "audio_tower.subsample_conv_projection.input_proj_linear.weight".into()
            }
            "a.pre_encode.out.weight" => return "audio_tower.output_proj.weight".into(),
            "a.pre_encode.out.bias" => return "audio_tower.output_proj.bias".into(),
            _ => {}
        }
        // generic prefix rewrites (tails already HF-shaped in gemma-4 exports)
        if let Some(rest) = g.strip_prefix("v.") {
            return format!("vision_tower.{}", rest);
        }
        if let Some(rest) = g.strip_prefix("a.") {
            return format!("audio_tower.{}", rest);
        }
        g.to_string()
    }

    fn translate_gemma4(g: &str) -> String {
        match g {
            "token_embd.weight" => return "language_model.embed_tokens.weight".into(),
            "output_norm.weight" => return "language_model.norm.weight".into(),
            "output.weight" => return "lm_head.weight".into(),
            "per_layer_token_embd.weight" => {
                return "language_model.embed_tokens_per_layer.weight".into()
            }
            "per_layer_model_proj.weight" => {
                return "language_model.per_layer_model_projection.weight".into()
            }
            "per_layer_proj_norm.weight" => {
                return "language_model.per_layer_projection_norm.weight".into()
            }
            "rope_freqs.weight" => return "language_model.rope_freqs.weight".into(),
            _ => {}
        }
        if let Some(rest) = g.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                let layer = &rest[..dot];
                let tail = &rest[dot + 1..];
                let hf_tail = match tail {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "attn_norm.weight" => "input_layernorm.weight",
                    // llama.cpp's *serialized* names (not its enum names):
                    // post_attention_norm IS the HF post-attention norm and
                    // ffn_norm IS the HF pre-feedforward norm.
                    "post_attention_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                    "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                    "post_norm.weight" => "post_per_layer_input_norm.weight",
                    "inp_gate.weight" => "per_layer_input_gate.weight",
                    "proj.weight" => "per_layer_projection.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => other, // layer_output_scale & friends pass verbatim
                };
                return format!("language_model.layers.{}.{}", layer, hf_tail);
            }
        }
        g.to_string()
    }

    /// Synthesize the generic [`crate::models::transformer::ModelConfig`] from
    /// GGUF metadata. Keys follow the `ARCH.attribute` convention.
    pub fn model_config(w: &GgufWeights) -> Res<crate::models::transformer::ModelConfig> {
        let a = w.architecture.clone();
        let k = |attr: &str| format!("{}.{}", a, attr);
        let need = |attr: &str| {
            w.meta_usize(&k(attr)).ok_or_else(|| {
                err!(
                    "gguf",
                    "metadata is missing {} — cannot synthesize the model config",
                    k(attr)
                )
            })
        };
        let n_heads = need("attention.head_count")?;
        let hidden = need("embedding_length")?;
        let vocab = w
            .meta_usize(&k("vocab_size"))
            .or_else(|| {
                w.meta
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_arr())
                    .map(|a| a.len())
            })
            .ok_or_else(|| err!("gguf", "cannot determine vocab size from metadata"))?;
        let tie = !w.tensors.contains_key("lm_head.weight");
        Ok(crate::models::transformer::ModelConfig {
            model_type: a.clone(),
            hidden_size: hidden,
            intermediate_size: need("feed_forward_length")?,
            n_layers: need("block_count")?,
            n_heads,
            n_kv_heads: w
                .meta_usize(&k("attention.head_count_kv"))
                .unwrap_or(n_heads),
            head_dim: w
                .meta_usize(&k("attention.key_length"))
                .unwrap_or(hidden / n_heads),
            vocab_size: vocab,
            rms_eps: w
                .meta_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            rope_theta: w.meta_f32(&k("rope.freq_base")).unwrap_or(10_000.0),
            max_seq: {
                let cfg_max = w.meta_usize(&k("context_length")).unwrap_or(8192);
                match std::env::var("CIMA_MAX_SEQ")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    Some(cap) if cap >= 256 => {
                        let m = cfg_max.min(cap);
                        crate::log::info(&format!(
                            "CIMA_MAX_SEQ={} — KV cache sized for {} positions",
                            cap, m
                        ));
                        m
                    }
                    _ => cfg_max.min(8192),
                }
            },
            tie_word_embeddings: tie,
            qkv_bias: w
                .tensors
                .contains_key("model.layers.0.self_attn.q_proj.bias"),
            quant_method: None,
            vision: None,
            audio: None,
            is_embedding: false,
        })
    }
}

pub mod safetensors {
    //! # safetensors — weight container format
    //!
    //! From-scratch safetensors implementation:
    //!
    //! ```text
    //! [u64 header_len][JSON header][raw tensor data...]
    //! ```
    //!
    //! * Files are `mmap`ed (raw `mmap(2)` FFI — no crates) and page-locked with
    //!   `cuMemHostRegister` so the GPU DMAs weights straight out of the page
    //!   cache: **zero intermediate copies** on the host.
    //! * `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` primes read-ahead so the cold
    //!   load streams at disk speed.
    //! * The **broken-model fail-safe** lives here: every header field, dtype,
    //!   shape, offset pair and shard reference is validated *before* a single
    //!   byte is uploaded, and every violation names the exact tensor and field.

    #![allow(non_camel_case_types)]

    use crate::cuda::{CudaCtx, DeviceBuf, HostRegistration};
    use crate::json::{self, Json};
    use crate::traits::{DType, LoadedWeights, ModelLoader, Res, TensorMeta, WeightCodec};
    use crate::{err, log};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::os::unix::io::AsRawFd;
    use std::path::{Path, PathBuf};

    // ===========================================================================
    // Raw mmap FFI (Linux)
    // ===========================================================================

    const PROT_READ: i32 = 0x1;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
    const MADV_SEQUENTIAL: i32 = 2;
    const MADV_WILLNEED: i32 = 3;

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    /// RAII read-only memory mapping of one shard file. Optionally carries a CUDA
    /// host registration which must drop *before* the mapping itself (field order).
    pub struct Mmap {
        /// CUDA page-lock over the mapping (drop order: registration first).
        registration: Option<HostRegistration>,
        ptr: *mut c_void,
        len: usize,
    }

    unsafe impl Send for Mmap {}

    impl Mmap {
        /// Map `path` read-only and advise the kernel for sequential streaming.
        pub fn open(path: &Path) -> Res<Mmap> {
            let file = std::fs::File::open(path)
                .map_err(|e| err!("safetensors", "cannot open '{}': {}", path.display(), e))?;
            let len = file
                .metadata()
                .map_err(|e| err!("safetensors", "stat '{}': {}", path.display(), e))?
                .len() as usize;
            if len < 8 {
                return Err(err!("safetensors", "'{}' is {} bytes — too small to be a safetensors file (need >= 8-byte header length)", path.display(), len));
            }
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == MAP_FAILED {
                return Err(err!(
                    "safetensors",
                    "mmap('{}', {} bytes) failed: {}",
                    path.display(),
                    len,
                    std::io::Error::last_os_error()
                ));
            }
            unsafe {
                // Sequential readahead for the pages we *actually* touch during
                // the upload sweep. Deliberately no MADV_WILLNEED: that would
                // prefetch the whole shard — including multi-GiB host-resident
                // regions (the Gemma 4 PLE table) that are only ever read lazily.
                madvise(ptr, len, MADV_SEQUENTIAL);
            }
            Ok(Mmap {
                registration: None,
                ptr,
                len,
            })
        }

        /// Page-lock the whole mapping for zero-copy GPU DMA — **opt-in** via
        /// `CIMA_PIN=1`, off by default.
        ///
        /// Measured reality on consumer machines (and especially WSL2):
        /// `cuMemHostRegister` must fault in *and* lock every page of the
        /// mapping before it can register it. On a cold page cache that is a
        /// full disk read of the shard up front — tens of seconds — and it
        /// drags multi-GiB host-resident regions (the Gemma 4 PLE table, which
        /// only the CPU reads lazily) into locked RAM. The upload sweep is
        /// disk-bound either way, so pageable `cuMemcpyHtoDAsync` (driver-staged)
        /// loses almost nothing while paying only for the pages it touches.
        /// Pinning only wins on native Linux with a warm page cache, ample RAM,
        /// and a raised `ulimit -l` — hence the env opt-in.
        pub fn pin(&mut self, ctx: &CudaCtx) -> Res<()> {
            let want = std::env::var("CIMA_PIN").map(|v| v == "1").unwrap_or(false);
            if !want || self.registration.is_some() {
                return Ok(());
            }
            // SAFETY: self.ptr/self.len are this mmap's own valid mapping,
            // owned by `self`, which outlives the returned registration
            // (both are dropped together in Mmap's Drop).
            match unsafe { ctx.register_host(self.ptr, self.len) } {
                Ok(r) => self.registration = Some(r),
                Err(e) => log::warn(&format!(
                    "CIMA_PIN=1: host pinning of {} MiB mapping failed ({}); continuing with \
                     pageable memory. Raise `ulimit -l` (or the WSL2 pinned-memory budget).",
                    self.len / (1024 * 1024),
                    e
                )),
            }
            Ok(())
        }

        pub fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            self.registration = None; // unregister before unmap
            unsafe { munmap(self.ptr, self.len) };
        }
    }

    // ===========================================================================
    // Header parsing & validation
    // ===========================================================================

    fn parse_dtype(s: &str, tensor: &str) -> Res<DType> {
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            "I64" => Ok(DType::I64),
            "U8" => Ok(DType::U8),
            other => Err(err!(
                "safetensors",
                "tensor '{}': unsupported dtype '{}' (supported: F32, F16, BF16, I64, U8; \
                 AWQ/GPTQ packed tensors are detected via config, not raw dtype)",
                tensor,
                other
            )),
        }
    }

    /// Parse one shard's header into tensor metadata, validating exhaustively.
    fn parse_shard(path: &Path, map: &Mmap) -> Res<Vec<TensorMeta>> {
        let bytes = map.bytes();
        if bytes.len() < 8 {
            return Err(err!(
                "safetensors",
                "'{}': file is {} bytes — too short for the 8-byte header-length field",
                path.display(),
                bytes.len()
            ));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if header_len == 0 || header_len > 256 * 1024 * 1024 || 8 + header_len > bytes.len() {
            return Err(err!(
                "safetensors",
                "'{}': header length field is {} but file is {} bytes — file truncated or not safetensors",
                path.display(), header_len, bytes.len()
            ));
        }
        let header_str = std::str::from_utf8(&bytes[8..8 + header_len]).map_err(|e| {
            err!(
                "safetensors",
                "'{}': header is not UTF-8 at byte {}",
                path.display(),
                e.valid_up_to()
            )
        })?;
        let header = json::parse(header_str).map_err(|e| {
            err!(
                "safetensors",
                "'{}': header JSON malformed: {}",
                path.display(),
                e
            )
        })?;
        let obj = header.as_obj().ok_or_else(|| {
            err!(
                "safetensors",
                "'{}': header root is not a JSON object",
                path.display()
            )
        })?;

        let data_len = bytes.len() - 8 - header_len;
        let fname = path.file_name().unwrap().to_string_lossy().into_owned();
        let mut out = Vec::with_capacity(obj.len());
        let mut spans: Vec<(usize, usize, &str)> = Vec::with_capacity(obj.len());

        for (name, spec) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = spec
                .path(&["dtype"])
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    err!(
                        "safetensors",
                        "tensor '{}': missing or non-string 'dtype' field",
                        name
                    )
                })?;
            let dtype = parse_dtype(dtype_s, name)?;

            let shape_j = spec
                .arr_of("shape")
                .ok_or_else(|| err!("safetensors", "tensor '{}': missing 'shape' array", name))?;
            let mut shape = Vec::with_capacity(shape_j.len());
            for (i, d) in shape_j.iter().enumerate() {
                let v = d.as_usize().ok_or_else(|| {
                    err!(
                        "safetensors",
                        "tensor '{}': shape[{}] is not a non-negative integer (got {:?})",
                        name,
                        i,
                        d
                    )
                })?;
                shape.push(v);
            }

            let offs = spec
                .arr_of("data_offsets")
                .ok_or_else(|| err!("safetensors", "tensor '{}': missing 'data_offsets'", name))?;
            if offs.len() != 2 {
                return Err(err!(
                    "safetensors",
                    "tensor '{}': data_offsets must have exactly 2 entries, got {}",
                    name,
                    offs.len()
                ));
            }
            let (b0, b1) = (
                offs[0].as_usize().ok_or_else(|| {
                    err!("safetensors", "tensor '{}': data_offsets[0] invalid", name)
                })?,
                offs[1].as_usize().ok_or_else(|| {
                    err!("safetensors", "tensor '{}': data_offsets[1] invalid", name)
                })?,
            );
            if b1 < b0 {
                return Err(err!(
                    "safetensors",
                    "tensor '{}': data_offsets end {} < begin {}",
                    name,
                    b1,
                    b0
                ));
            }
            if b1 > data_len {
                return Err(err!(
                    "safetensors",
                    "tensor '{}': data_offsets end {} exceeds data section size {} — shard '{}' is truncated",
                    name, b1, data_len, fname
                ));
            }
            let expected = shape.iter().product::<usize>().max(1) * dtype.size();
            let actual = b1 - b0;
            if expected != actual {
                return Err(err!(
                    "safetensors",
                    "tensor '{}': shape {:?} × dtype {} predicts {} bytes but data_offsets span {} bytes — corrupted tensor map",
                    name, shape, dtype.name(), expected, actual
                ));
            }
            spans.push((b0, b1, name.as_str()));
            out.push(TensorMeta {
                name: name.clone(),
                dtype,
                shape,
                offset: 8 + header_len + b0,
                nbytes: actual,
                file: fname.clone(),
            });
        }

        // Overlap check: corrupted headers sometimes alias tensor storage.
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(err!(
                    "safetensors",
                    "tensors '{}' and '{}' have overlapping data ranges [{},{}) and [{},{}) — corrupted header",
                    w[0].2, w[1].2, w[0].0, w[0].1, w[1].0, w[1].1
                ));
            }
        }
        Ok(out)
    }

    // ===========================================================================
    // LoadedWeights implementation
    // ===========================================================================

    /// All shards of a checkpoint, mmapped + pinned, with a unified tensor map.
    pub struct SafetensorsWeights {
        maps: HashMap<String, Mmap>,
        tensors: HashMap<String, TensorMeta>,
    }

    impl LoadedWeights for SafetensorsWeights {
        fn tensors(&self) -> &HashMap<String, TensorMeta> {
            &self.tensors
        }
        fn bytes(&self, meta: &TensorMeta) -> Res<&[u8]> {
            let map = self.maps.get(&meta.file).ok_or_else(|| {
                err!(
                    "safetensors",
                    "tensor '{}' references unknown shard '{}'",
                    meta.name,
                    meta.file
                )
            })?;
            let b = map.bytes();
            if meta.offset + meta.nbytes > b.len() {
                return Err(err!(
                    "safetensors",
                    "tensor '{}': byte range exceeds shard '{}' size",
                    meta.name,
                    meta.file
                ));
            }
            Ok(&b[meta.offset..meta.offset + meta.nbytes])
        }
        fn prefetch(&self, meta: &TensorMeta) {
            // madvise(WILLNEED) is asynchronous: the kernel starts reading the
            // region into the page cache in the background while the build
            // continues. Page-aligned over the tensor's byte range only.
            if let Ok(slice) = self.bytes(meta) {
                let page = 4096usize;
                let addr = slice.as_ptr() as usize;
                let start = addr & !(page - 1);
                let len = slice.len() + (addr - start);
                unsafe {
                    madvise(start as *mut c_void, len, MADV_WILLNEED);
                }
            }
        }
    }

    // ===========================================================================
    // ModelLoader implementation
    // ===========================================================================

    /// The shipped [`ModelLoader`] for `*.safetensors` checkpoints, including
    /// sharded checkpoints described by `model.safetensors.index.json`.
    pub struct SafetensorsLoader;

    impl ModelLoader for SafetensorsLoader {
        fn detect(&self, dir: &Path) -> bool {
            dir.join("model.safetensors").exists()
                || dir.join("model.safetensors.index.json").exists()
                || std::fs::read_dir(dir)
                    .map(|rd| {
                        rd.flatten()
                            .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                    })
                    .unwrap_or(false)
        }

        fn load(&self, dir: &Path, ctx: &CudaCtx) -> Res<Box<dyn LoadedWeights>> {
            let t0 = std::time::Instant::now();
            let shard_paths = discover_shards(dir)?;
            log::info(&format!(
                "loading {} safetensors shard(s) from {}",
                shard_paths.len(),
                dir.display()
            ));

            let mut maps = HashMap::new();
            let mut tensors: HashMap<String, TensorMeta> = HashMap::new();
            let mut total = 0usize;

            for path in &shard_paths {
                let mut map = Mmap::open(path)?;
                // Zero-copy: page-lock the mapping for direct DMA.
                map.pin(ctx)?;
                let metas = parse_shard(path, &map)?;
                for m in metas {
                    if let Some(prev) = tensors.get(&m.name) {
                        return Err(err!(
                            "safetensors",
                            "tensor '{}' is defined in both '{}' and '{}' — corrupted sharded checkpoint",
                            m.name, prev.file, m.file
                        ));
                    }
                    total += m.nbytes;
                    tensors.insert(m.name.clone(), m);
                }
                maps.insert(
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    map,
                );
            }

            if tensors.is_empty() {
                return Err(err!(
                    "safetensors",
                    "no tensors found across {} shard(s) in {}",
                    shard_paths.len(),
                    dir.display()
                ));
            }
            log::info(&format!(
                "safetensors validated: {} tensors, {} of weights, mapped in {:?} (pageable; CIMA_PIN=1 for pinned DMA)",
                tensors.len(), crate::cuda::fmt_bytes(total), t0.elapsed()
            ));
            Ok(Box::new(SafetensorsWeights { maps, tensors }))
        }
    }

    /// Resolve shard files: prefer the index JSON, fall back to directory scan.
    /// Validates that every shard the index references actually exists.
    fn discover_shards(dir: &Path) -> Res<Vec<PathBuf>> {
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            let txt = std::fs::read_to_string(&index)
                .map_err(|e| err!("safetensors", "read '{}': {}", index.display(), e))?;
            let j = json::parse(&txt)
                .map_err(|e| err!("safetensors", "'{}' malformed: {}", index.display(), e))?;
            let wm = j.get("weight_map").and_then(Json::as_obj).ok_or_else(|| {
                err!(
                    "safetensors",
                    "'{}': missing 'weight_map' object",
                    index.display()
                )
            })?;
            let mut files: Vec<String> = Vec::new();
            for (tensor, file) in wm {
                let f = file.as_str().ok_or_else(|| {
                    err!(
                        "safetensors",
                        "index weight_map['{}'] is not a string",
                        tensor
                    )
                })?;
                if !files.iter().any(|x| x == f) {
                    files.push(f.to_string());
                }
            }
            let mut out = Vec::with_capacity(files.len());
            for f in files {
                let p = dir.join(&f);
                if !p.exists() {
                    return Err(err!(
                        "safetensors",
                        "index references shard '{}' which is missing from {} — incomplete download or broken repo",
                        f, dir.display()
                    ));
                }
                out.push(p);
            }
            return Ok(out);
        }
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Ok(vec![single]);
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| err!("safetensors", "read dir '{}': {}", dir.display(), e))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        found.sort();
        if found.is_empty() {
            return Err(err!(
                "safetensors",
                "no *.safetensors files in {} — wrong format? (gguf loader not yet registered)",
                dir.display()
            ));
        }
        Ok(found)
    }

    // ===========================================================================
    // Weight codecs
    // ===========================================================================

    /// Identity codec for FP16/BF16/F32 weights. BF16 and F32 are normalized to
    /// F16 on-device at load time (single fused conversion kernel) so the entire
    /// execution graph runs in one uniform activation dtype.
    pub struct HalfCodec;

    impl WeightCodec for HalfCodec {
        fn name(&self) -> &'static str {
            "fp16/bf16"
        }
        fn accepts(&self, dtype: DType) -> bool {
            matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        }
        fn device_bytes(&self, meta: &TensorMeta) -> usize {
            meta.numel() * 2 // everything is f16 once resident
        }
        fn upload(&self, ctx: &CudaCtx, meta: &TensorMeta, host: &[u8]) -> Res<DeviceBuf> {
            let n = meta.numel();
            match meta.dtype {
                DType::F16 => {
                    let buf = ctx.alloc(host.len())?;
                    ctx.htod(&buf, host)?; // direct DMA from pinned mmap
                    Ok(buf)
                }
                DType::BF16 => {
                    // Stage raw bf16, convert to f16 on-device, free the staging buffer.
                    let staging = ctx.alloc(host.len())?;
                    ctx.htod(&staging, host)?;
                    let out = ctx.alloc(n * 2)?;
                    ctx.bf2h(staging.ptr, out.ptr, n)?;
                    ctx.sync()?; // staging must outlive the kernel
                    Ok(out)
                }
                DType::F32 => {
                    let staging = ctx.alloc(host.len())?;
                    ctx.htod(&staging, host)?;
                    let out = ctx.alloc(n * 2)?;
                    ctx.f2h(staging.ptr, out.ptr, n)?;
                    ctx.sync()?;
                    Ok(out)
                }
                other => Err(err!(
                    "quant",
                    "HalfCodec asked to upload tensor '{}' of dtype {} — codec selection bug",
                    meta.name,
                    other.name()
                )),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Write;

        fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
            let p = std::env::temp_dir().join(format!("cima_st_{}_{}", std::process::id(), name));
            std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
            p
        }
        fn parse(bytes: &[u8]) -> Res<Vec<TensorMeta>> {
            let p = tmp(
                &format!(
                    "{:x}",
                    bytes.len() as u64 ^ bytes.iter().map(|&b| b as u64).sum::<u64>()
                ),
                bytes,
            );
            let map = Mmap::open(&p)?;
            let r = parse_shard(&p, &map);
            let _ = std::fs::remove_file(&p);
            r
        }

        /// Files shorter than the 8-byte length field must error, not panic
        /// (this exact slice-index panic shipped once).
        #[test]
        fn header_too_short() {
            assert!(parse(b"").is_err());
            assert!(parse(b"1234567").is_err());
        }

        /// A header-length field larger than the file must be rejected.
        #[test]
        fn header_len_lies() {
            let mut b = (1_000_000u64).to_le_bytes().to_vec();
            b.extend_from_slice(b"{}");
            assert!(parse(&b).is_err());
        }

        /// Header bytes that are not JSON must be rejected with an error.
        #[test]
        fn header_not_json() {
            let payload = b"not json at all";
            let mut b = (payload.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(payload);
            assert!(parse(&b).is_err());
        }

        /// Tensor data offsets beyond the file are corruption, not UB.
        #[test]
        fn offsets_beyond_file() {
            let hdr = br#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,80000]}}"#;
            let mut b = (hdr.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(hdr);
            b.extend_from_slice(&[0u8; 8]); // 8 bytes of data, offsets claim 80000
            assert!(parse(&b).is_err());
        }

        /// Shape/byte-count mismatch (F16 4-elem tensor = 8 bytes, offsets say 6).
        #[test]
        fn shape_bytes_mismatch() {
            let hdr = br#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,6]}}"#;
            let mut b = (hdr.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(hdr);
            b.extend_from_slice(&[0u8; 6]);
            assert!(parse(&b).is_err());
        }

        /// The happy path still parses: one F16 tensor, metadata key ignored.
        #[test]
        fn minimal_valid_shard() {
            let hdr = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}"#;
            let mut b = (hdr.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(hdr);
            b.extend_from_slice(&[0u8; 8]);
            let metas = parse(&b).unwrap();
            assert_eq!(metas.len(), 1);
            assert_eq!(metas[0].shape, vec![2, 2]);
            assert_eq!(metas[0].nbytes, 8);
        }
    }
}
