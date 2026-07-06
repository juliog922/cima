//! # tokenizer — from-scratch byte-level BPE
//!
//! Implements the GPT-2 / Llama-3 / Qwen family of byte-level BPE tokenizers
//! directly from Hugging Face `tokenizer.json`:
//!
//! * byte-to-unicode alphabet remapping (the classic GPT-2 table),
//! * greedy merge loop ranked by `merges` order,
//! * `added_tokens` (special tokens) matched literally before BPE,
//! * streaming-safe single-token decode for the generation loop.
//!
//! SentencePiece-style tokenizers (`model.type == "Unigram"`) are detected and
//! rejected with a precise error rather than mis-tokenizing silently.

use crate::json::{self, Json};
use crate::traits::{Res, Tokenizer};
use crate::{err, log};
use std::collections::HashMap;
use std::path::Path;

/// Byte-level BPE tokenizer loaded from `tokenizer.json`.
pub struct BpeTokenizer {
    /// token string (in byte-unicode alphabet) -> id
    vocab: HashMap<String, u32>,
    /// id -> raw bytes (already un-remapped, streaming-safe)
    id_to_bytes: Vec<Vec<u8>>,
    /// (left, right) -> merge rank
    merges: HashMap<(String, String), u32>,
    /// literal special tokens, longest-first
    specials: Vec<(String, u32)>,
    bos: Option<u32>,
    /// Resolved BOS policy — see the discovery hierarchy in `load`.
    add_bos: bool,
    /// The bos token's special-string form (e.g. "<bos>"), when registered.
    bos_literal: Option<String>,
    eos: Vec<u32>,
    /// byte -> alphabet char
    byte_enc: [char; 256],
    /// SentencePiece-BPE mode (Gemma / Llama-2 lineage): spaces are the
    /// metaspace `▁` (U+2581), unknown characters byte-fallback to `<0xXX>`
    /// tokens, and decode replaces `▁` with a space. Detected from
    /// tokenizer.json (`model.byte_fallback` / metaspace vocab). When false,
    /// the GPT-2 byte-level alphabet applies (llama3 / qwen lineage).
    sp: bool,
    /// SP normalizer carries a `Prepend("▁")` step (Llama-2 dummy prefix;
    /// Gemma does not).
    sp_prepend: bool,
    /// `<0x00>`..`<0xFF>` byte-fallback token ids (SP mode).
    byte_ids: Vec<Option<u32>>,
    /// SentencePiece unigram scores (log-probs), indexed by token id.
    /// Non-empty ⇒ encode segments by Viterbi over these scores (the
    /// algorithm gemma was trained with) instead of greedy BPE merges.
    scores: Vec<f32>,
    /// Longest vocab piece in chars (bounds the Viterbi inner loop).
    max_piece_chars: usize,
    /// Byte-fallback penalty: min(scores) - 10, precomputed at build.
    byte_pen: f32,
}

/// GPT-2 byte<->unicode bijection.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::new();
    let mut printable: Vec<u8> = (b'!'..=b'~').collect();
    printable.extend(0xA1u8..=0xACu8);
    printable.extend(0xAEu8..=0xFFu8);
    let mut n = 0u32;
    for b in 0u32..256 {
        let c = if printable.contains(&(b as u8)) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        enc[b as usize] = c;
        dec.insert(c, b as u8);
    }
    (enc, dec)
}

impl BpeTokenizer {
    /// Does the tokenizer itself prepend BOS on encode? (Template renderers
    /// that own BOS, like gemma-4's, must prepend the literal when false.)
    pub fn adds_bos(&self) -> bool {
        self.add_bos
    }

    /// The bos special literal (e.g. "<bos>"), when the id is registered.
    pub fn bos_literal(&self) -> Option<&str> {
        self.bos_literal.as_deref()
    }

    /// The bos token id, when the model defines one.
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// Id→bytes round-trip (debug tooling: piece inspection).
    pub fn decode_bytes(&self, id: u32) -> Vec<u8> {
        self.id_to_bytes
            .get(id as usize)
            .cloned()
            .unwrap_or_default()
    }

    /// Registered special tokens `(literal, id)` — diagnostics surface.
    pub fn specials(&self) -> &[(String, u32)] {
        &self.specials
    }

    /// Load and validate `tokenizer.json` from a model directory.
    pub fn load(dir: &Path) -> Res<BpeTokenizer> {
        let path = dir.join("tokenizer.json");
        let txt = std::fs::read_to_string(&path).map_err(|e| {
            err!(
                "tokenizer",
                "cannot read '{}': {} — repo is missing its tokenizer",
                path.display(),
                e
            )
        })?;
        let j = json::parse(&txt)
            .map_err(|e| err!("tokenizer", "'{}' is malformed JSON: {}", path.display(), e))?;

        let model = j
            .get("model")
            .ok_or_else(|| err!("tokenizer", "tokenizer.json: missing 'model' object"))?;
        let mtype = model.str_of("type").unwrap_or("BPE");
        if mtype != "BPE" {
            return Err(err!("tokenizer", "tokenizer model type '{}' is not supported (only byte-level BPE); register a Tokenizer impl for it", mtype));
        }

        // vocab
        let vocab_j = model.get("vocab").and_then(Json::as_obj).ok_or_else(|| {
            err!(
                "tokenizer",
                "tokenizer.json: model.vocab missing or not an object"
            )
        })?;
        let mut vocab = HashMap::with_capacity(vocab_j.len());
        let mut max_id = 0u32;
        for (tok, id) in vocab_j {
            let id = id
                .as_u64()
                .ok_or_else(|| err!("tokenizer", "vocab['{}'] id is not an integer", tok))?
                as u32;
            max_id = max_id.max(id);
            vocab.insert(tok.clone(), id);
        }

        // merges — either ["a b", ...] or [["a","b"], ...]
        let merges_j = model
            .arr_of("merges")
            .ok_or_else(|| err!("tokenizer", "tokenizer.json: model.merges missing"))?;
        let mut merges = HashMap::with_capacity(merges_j.len());
        for (rank, m) in merges_j.iter().enumerate() {
            let (a, b) = match m {
                Json::Str(s) => {
                    let mut it = s.splitn(2, ' ');
                    match (it.next(), it.next()) {
                        (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                        _ => {
                            return Err(err!(
                                "tokenizer",
                                "merges[{}] '{}' is not 'left right'",
                                rank,
                                s
                            ))
                        }
                    }
                }
                Json::Arr(pair) if pair.len() == 2 => (
                    pair[0]
                        .as_str()
                        .ok_or_else(|| err!("tokenizer", "merges[{}][0] not a string", rank))?
                        .to_string(),
                    pair[1]
                        .as_str()
                        .ok_or_else(|| err!("tokenizer", "merges[{}][1] not a string", rank))?
                        .to_string(),
                ),
                _ => return Err(err!("tokenizer", "merges[{}] has unrecognized shape", rank)),
            };
            merges.insert((a, b), rank as u32);
        }

        // added/special tokens
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(added) = j.arr_of("added_tokens") {
            for t in added {
                let content = t.str_of("content");
                let id = t.u64_of("id");
                if let (Some(c), Some(i)) = (content, id) {
                    max_id = max_id.max(i as u32);
                    vocab.entry(c.to_string()).or_insert(i as u32);
                    specials.push((c.to_string(), i as u32));
                }
            }
        }
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len())); // longest-first

        // ---- tokenizer family detection ----
        // SentencePiece-BPE (Gemma, Llama-2): `model.byte_fallback: true`
        // and/or metaspace `▁` pieces in the vocab; the reference pipeline
        // (transformers GemmaConverter) is normalizer Replace(" ","▁") and
        // decoder [Replace("▁"," "), ByteFallback, Fuse]. Byte-level BPE
        // (GPT-2/llama3/qwen): neither marker present.
        let sp = model.bool_of("byte_fallback").unwrap_or(false)
            || vocab.contains_key("\u{2581}")
            || vocab.keys().any(|t| t.starts_with('\u{2581}'));
        // Llama-2-style normalizers add a dummy-prefix Prepend("▁") step.
        fn has_prepend(j: &Json) -> bool {
            match j {
                Json::Obj(_) => {
                    if j.str_of("type") == Some("Prepend") {
                        return true;
                    }
                    j.arr_of("normalizers")
                        .map(|a| a.iter().any(has_prepend))
                        .unwrap_or(false)
                }
                Json::Arr(a) => a.iter().any(has_prepend),
                _ => false,
            }
        }
        let sp_prepend = sp && j.get("normalizer").map(has_prepend).unwrap_or(false);

        // <0xXX> byte-fallback ids (SP mode encodes unknown chars as bytes).
        let mut byte_ids: Vec<Option<u32>> = vec![None; 256];
        if sp {
            for b in 0u32..256 {
                if let Some(&id) = vocab.get(&format!("<0x{:02X}>", b)) {
                    byte_ids[b as usize] = Some(id);
                }
            }
        }

        // id -> bytes table (streaming decode path)
        let (byte_enc, byte_dec) = byte_alphabet();
        let mut id_to_bytes = vec![Vec::new(); (max_id + 1) as usize];
        for (tok, &id) in &vocab {
            let bytes: Vec<u8> = if specials.iter().any(|(s, _)| s == tok) {
                tok.as_bytes().to_vec() // specials decode literally
            } else if sp {
                // ByteFallback: `<0xXX>` pieces decode to the raw byte;
                // everything else replaces the metaspace with a space.
                if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                    match u8::from_str_radix(&tok[3..5], 16) {
                        Ok(b) => vec![b],
                        Err(_) => tok.as_bytes().to_vec(),
                    }
                } else {
                    tok.replace('\u{2581}', " ").into_bytes()
                }
            } else {
                tok.chars()
                    .map(|c| *byte_dec.get(&c).unwrap_or(&b'?'))
                    .collect()
            };
            id_to_bytes[id as usize] = bytes;
        }

        // bos/eos discovery from config files
        let (bos, eos) = discover_bos_eos(dir, &vocab);
        // BOS policy discovery hierarchy:
        //   1. explicit `add_bos_token` in tokenizer_config.json (some
        //      models define bos_token_id yet set the flag to false);
        //   2. flag absent: the tokenizer.json `post_processor` decides —
        //      template-owned-BOS models (gemma-4) add nothing on raw
        //      encode and their post_processor omits the bos literal;
        //   3. neither file informative: legacy prepend (llama lineage).
        let bos_literal = bos.and_then(|id| {
            specials
                .iter()
                .find(|(_, i)| *i == id)
                .map(|(s, _)| s.clone())
        });
        let add_bos = resolve_add_bos(
            std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .ok()
                .as_deref(),
            std::fs::read_to_string(dir.join("tokenizer.json"))
                .ok()
                .as_deref(),
            bos_literal.as_deref(),
        );
        log::info(&format!(
            "tokenizer: {} vocab entries, {} merges, {} specials, bos={:?}, eos={:?}, family={}",
            vocab.len(),
            merges.len(),
            specials.len(),
            bos,
            eos,
            if sp {
                "sentencepiece-bpe (metaspace + byte fallback)"
            } else {
                "byte-level bpe"
            }
        ));
        Ok(BpeTokenizer {
            vocab,
            id_to_bytes,
            merges,
            specials,
            bos,
            add_bos,
            bos_literal,
            eos,
            byte_enc,
            sp,
            sp_prepend,
            byte_ids,
            scores: Vec::new(),
            max_piece_chars: 0,
            byte_pen: 0.0,
        })
    }

    /// Build a byte-level BPE tokenizer from GGUF metadata: the
    /// `tokenizer.ggml.*` arrays carry the same vocab/merges that
    /// tokenizer.json would. Token types mark specials (control tokens);
    /// bos/eos and the add-BOS policy come from their metadata keys.
    /// (SentencePiece GGUF tokenizers — `tokenizer.ggml.model = "llama"` —
    /// are not wired yet; this path serves gpt2-family checkpoints.)
    pub fn from_gguf(
        tokens: &[String],
        merge_lines: &[String],
        token_types: &[i64],
        bos: Option<u32>,
        eos: Vec<u32>,
        add_bos: bool,
        scores: Vec<f32>,
    ) -> Res<BpeTokenizer> {
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), id as u32);
        }
        let mut merges = HashMap::with_capacity(merge_lines.len());
        for (rank, m) in merge_lines.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                merges.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }
        // ggml token types: 1 normal, 2 unknown, 3 control (special),
        // 4 user-defined, 6 byte. Specials decode literally and are
        // matched longest-first during encode.
        let mut specials: Vec<(String, u32)> = Vec::new();
        for (id, &ty) in token_types.iter().enumerate() {
            if (ty == 3 || ty == 4) && id < tokens.len() {
                specials.push((tokens[id].clone(), id as u32));
            }
        }
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));

        // SentencePiece lineage (gemma/llama-2): byte-fallback tokens are
        // the unambiguous fingerprint; gpt2 byte-level vocabs never carry
        // them. Decode semantics differ completely: metaspace, not
        // byte-alphabet.
        let sp = vocab.contains_key("<0x00>");
        let mut byte_ids: Vec<Option<u32>> = vec![None; 256];
        if sp {
            // b is the byte value itself (used in the <0xNN> lookup key).
            #[allow(clippy::needless_range_loop)]
            for b in 0..256usize {
                byte_ids[b] = vocab.get(&format!("<0x{:02X}>", b)).copied();
            }
        }
        let (byte_enc, byte_dec) = byte_alphabet();
        let mut id_to_bytes = vec![Vec::new(); tokens.len()];
        for (tok, &id) in &vocab {
            let bytes: Vec<u8> = if specials.iter().any(|(s, _)| s == tok) {
                tok.as_bytes().to_vec()
            } else if sp {
                if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                    vec![u8::from_str_radix(&tok[3..5], 16).unwrap_or(b'?')]
                } else {
                    tok.replace('\u{2581}', " ").into_bytes()
                }
            } else {
                tok.chars()
                    .map(|c| *byte_dec.get(&c).unwrap_or(&b'?'))
                    .collect()
            };
            id_to_bytes[id as usize] = bytes;
        }
        let bos_literal = bos.and_then(|id| tokens.get(id as usize).cloned());
        log::info(&format!(
            "tokenizer (gguf): {} vocab entries, {} merges, {} specials, bos={:?}, eos={:?}, family={}",
            vocab.len(), merges.len(), specials.len(), bos, eos,
            if sp { "sentencepiece" } else { "byte-level bpe" }
        ));
        // unigram only makes sense for SP vocabs with a full score table
        let scores = if sp && scores.len() == tokens.len() {
            scores
        } else {
            Vec::new()
        };
        let max_piece_chars = if scores.is_empty() {
            0
        } else {
            vocab
                .keys()
                .map(|k| k.chars().count())
                .max()
                .unwrap_or(1)
                .min(64)
        };
        let byte_pen = scores.iter().cloned().fold(f32::INFINITY, f32::min) - 10.0;
        Ok(BpeTokenizer {
            vocab,
            id_to_bytes,
            merges,
            specials,
            bos,
            add_bos,
            bos_literal,
            eos,
            byte_enc,
            sp,
            sp_prepend: false,
            byte_ids,
            scores,
            max_piece_chars,
            byte_pen,
        })
    }

    /// SentencePiece **unigram** segmentation by Viterbi over the gguf
    /// score table: pick the token sequence maximizing the sum of
    /// log-probs. This is the algorithm gemma-family models were trained
    /// with; greedy BPE over (synthetic) merges produces subtly different
    /// segmentations that read to the model like typos — worst in accented
    /// languages. Unknown characters fall back to <0xXX> byte tokens at a
    /// strong penalty, so every input remains encodable.
    fn unigram(&self, piece: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = piece.chars().collect();
        let n = chars.len();
        if n == 0 {
            return;
        }
        let byte_pen = self.byte_pen;
        // best[i]: (score, prev_index, ids emitted for the edge prev→i)
        let mut best: Vec<Option<(f32, usize, Vec<u32>)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, Vec::new()));
        let mut buf = String::with_capacity(self.max_piece_chars * 4);
        for i in 0..n {
            let Some((base, _, _)) = best[i].as_ref().map(|(s, p, _)| (*s, *p, ())) else {
                continue;
            };
            buf.clear();
            for j in i..n.min(i + self.max_piece_chars) {
                buf.push(chars[j]);
                if let Some(&id) = self.vocab.get(buf.as_str()) {
                    if let Some(&sc) = self.scores.get(id as usize) {
                        let cand = base + sc;
                        if best[j + 1].as_ref().is_none_or(|(s, _, _)| cand > *s) {
                            best[j + 1] = Some((cand, i, vec![id]));
                        }
                    }
                }
            }
            // byte fallback for the single char at i (guarantees progress)
            if best[i + 1].is_none() {
                let mut ids = Vec::new();
                let mut b4 = [0u8; 4];
                for &b in chars[i].encode_utf8(&mut b4).as_bytes() {
                    if let Some(id) = self.byte_ids.get(b as usize).copied().flatten() {
                        ids.push(id);
                    }
                }
                if !ids.is_empty() {
                    let cand = base + byte_pen;
                    best[i + 1] = Some((cand, i, ids));
                }
            }
        }
        if best[n].is_none() {
            // pathological input (no byte table?) — keep the old behavior
            self.bpe(piece, out);
            return;
        }
        // backtrack
        let mut edges: Vec<Vec<u32>> = Vec::new();
        let mut at = n;
        while at > 0 {
            let (_, prev, ids) = best[at].take().unwrap();
            edges.push(ids);
            at = prev;
        }
        for ids in edges.iter().rev() {
            out.extend_from_slice(ids);
        }
    }

    /// BPE over one pre-token (already byte-remapped string).
    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        if parts.is_empty() {
            return;
        }
        loop {
            // find the lowest-rank adjacent pair
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&rank) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            match best {
                None => break,
                Some((i, _)) => {
                    let merged = format!("{}{}", parts[i], parts[i + 1]);
                    parts.splice(i..i + 2, [merged]);
                }
            }
            if parts.len() == 1 {
                break;
            }
        }
        for p in &parts {
            match self.vocab.get(p) {
                Some(&id) => out.push(id),
                None => {
                    for c in p.chars() {
                        if let Some(&id) = self.vocab.get(&c.to_string()) {
                            out.push(id);
                        } else if self.sp {
                            // SentencePiece byte fallback: emit the char's
                            // UTF-8 bytes as <0xXX> tokens (decoder fuses
                            // them back). Never drop input silently.
                            let mut buf = [0u8; 4];
                            for b in c.encode_utf8(&mut buf).as_bytes() {
                                if let Some(id) = self.byte_ids[*b as usize] {
                                    out.push(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// SentencePiece segmentation: apply the normalizer (optional dummy
    /// prefix, spaces → `▁`), then split into pieces that start at each
    /// metaspace run. `split_by_whitespace: true` in the SP training config
    /// guarantees no vocab piece spans a word boundary, so per-word BPE is
    /// exactly equivalent to whole-segment BPE. Runs of consecutive `▁`
    /// (indentation) stay attached to the following word, matching the
    /// trained `▁▁…` pieces.
    fn encode_sp(&self, seg: &str, first_segment: bool, out: &mut Vec<u32>) {
        let mut norm = String::with_capacity(seg.len() + 1);
        if self.sp_prepend && first_segment {
            norm.push('\u{2581}');
        }
        for c in seg.chars() {
            norm.push(if c == ' ' { '\u{2581}' } else { c });
        }
        let chars: Vec<char> = norm.chars().collect();
        let mut start = 0usize;
        for i in 1..chars.len() {
            if chars[i] == '\u{2581}' && chars[i - 1] != '\u{2581}' {
                let piece: String = chars[start..i].iter().collect();
                if self.scores.is_empty() {
                    self.bpe(&piece, out)
                } else {
                    self.unigram(&piece, out)
                }
                start = i;
            }
        }
        if start < chars.len() {
            let piece: String = chars[start..].iter().collect();
            if self.scores.is_empty() {
                self.bpe(&piece, out)
            } else {
                self.unigram(&piece, out)
            }
        }
    }

    /// GPT-2-style pretokenizer: split into runs that keep leading spaces
    /// attached to words, isolate numbers and punctuation.
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let start = i;
            let lead_space = b[i] == b' ';
            if lead_space {
                i += 1;
                if i >= b.len() {
                    out.push(&text[start..i]);
                    break;
                }
            }
            let c = b[i];
            if c.is_ascii_alphabetic() || c >= 0x80 {
                while i < b.len() && (b[i].is_ascii_alphabetic() || b[i] >= 0x80) {
                    i += 1;
                }
            } else if c.is_ascii_digit() {
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            } else if c == b' ' {
                while i < b.len() && b[i] == b' ' {
                    i += 1;
                }
            } else if c == b'\n' || c == b'\r' {
                while i < b.len() && (b[i] == b'\n' || b[i] == b'\r') {
                    i += 1;
                }
            } else {
                while i < b.len()
                    && !b[i].is_ascii_alphanumeric()
                    && b[i] < 0x80
                    && b[i] != b' '
                    && b[i] != b'\n'
                    && b[i] != b'\r'
                {
                    i += 1;
                }
            }
            out.push(&text[start..i]);
        }
        out
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 2);
        if add_bos && self.add_bos {
            if let Some(b) = self.bos {
                ids.push(b);
            }
        }
        // split on literal special tokens first
        let mut segments: Vec<(bool, String)> = vec![(false, text.to_string())];
        for (special, _) in &self.specials {
            let mut next = Vec::new();
            for (is_special, seg) in segments {
                if is_special {
                    next.push((true, seg));
                    continue;
                }
                let mut rest = seg.as_str();
                while let Some(pos) = rest.find(special.as_str()) {
                    if pos > 0 {
                        next.push((false, rest[..pos].to_string()));
                    }
                    next.push((true, special.clone()));
                    rest = &rest[pos + special.len()..];
                }
                if !rest.is_empty() {
                    next.push((false, rest.to_string()));
                }
            }
            segments = next;
        }
        let mut first_text = true;
        for (is_special, seg) in segments {
            if is_special {
                ids.push(self.vocab[&seg]);
                continue;
            }
            if self.sp {
                // SentencePiece pipeline: metaspace normalization + per-word
                // BPE + byte fallback. No byte-level alphabet remapping.
                self.encode_sp(&seg, first_text, &mut ids);
            } else {
                for pre in self.pretokenize(&seg) {
                    let remapped: String = pre.bytes().map(|b| self.byte_enc[b as usize]).collect();
                    self.bpe(&remapped, &mut ids);
                }
            }
            first_text = false;
        }
        ids
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        self.id_to_bytes
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn eos_ids(&self) -> &[u32] {
        &self.eos
    }
    fn special(&self, literal: &str) -> Option<u32> {
        self.vocab.get(literal).copied()
    }
}

/// Pull bos/eos ids from `config.json` / `generation_config.json` /
/// `tokenizer_config.json`, tolerating all the shapes seen in the wild.
fn discover_bos_eos(dir: &Path, vocab: &HashMap<String, u32>) -> (Option<u32>, Vec<u32>) {
    let mut bos = None;
    let mut eos = Vec::new();
    for f in ["generation_config.json", "config.json"] {
        if let Ok(txt) = std::fs::read_to_string(dir.join(f)) {
            if let Ok(j) = json::parse(&txt) {
                if bos.is_none() {
                    bos = j.u64_of("bos_token_id").map(|v| v as u32);
                }
                match j.get("eos_token_id") {
                    Some(Json::Num(_)) => {
                        if let Some(v) = j.u64_of("eos_token_id") {
                            if !eos.contains(&(v as u32)) {
                                eos.push(v as u32);
                            }
                        }
                    }
                    Some(Json::Arr(a)) => {
                        for v in a {
                            if let Some(v) = v.as_u64() {
                                if !eos.contains(&(v as u32)) {
                                    eos.push(v as u32);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // tokenizer_config.json may carry string eos ("<|im_end|>")
    if let Ok(txt) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
        if let Ok(j) = json::parse(&txt) {
            if let Some(s) = j.str_of("eos_token") {
                if let Some(&id) = vocab.get(s) {
                    if !eos.contains(&id) {
                        eos.push(id);
                    }
                }
            }
        }
    }
    (bos, eos)
}

// ===========================================================================
// Chat templating (ChatML / Llama-3 family)
// ===========================================================================

/// A single chat turn as received from `/api/chat`.
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    /// Attached images (placeholders to emit ahead of the content).
    pub n_images: usize,
    /// Attached audio clips.
    pub n_audio: usize,
}

/// Render a chat into a prompt string. Detects the template family from
/// `tokenizer_config.json`'s `chat_template` (substring heuristics over the
/// Jinja source — full Jinja evaluation is out of scope and unnecessary for
/// the ChatML/Llama-3 families that dominate the Hub).
pub fn render_chat(
    dir: &Path,
    tk: &BpeTokenizer,
    family_hint: Option<&str>,
    turns: &[ChatTurn],
    media_token: Option<&str>,
    template_override: Option<&str>,
) -> String {
    // The chat template may live in several places depending on the repo's
    // vintage: `tokenizer_config.json:chat_template` as a string, the same
    // key as a *list* of named templates, or a separate `chat_template.jinja`
    // file. As a last resort the template *family* is sniffed from the raw
    // tokenizer_config text, whose added_tokens_decoder lists the turn
    // markers — repos that ship `<start_of_turn>` as a special are Gemma
    // regardless of where the jinja went. (An unsloth Gemma repo with the
    // template only in jinja form previously fell through to the ChatML
    // branch: wrong turn markers *and* no media placeholder.)
    let raw_cfg = std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap_or_default();
    let mut template_src = json::parse(&raw_cfg)
        .ok()
        .and_then(|j| {
            let ct = j.get("chat_template")?.clone();
            if let Some(s) = ct.as_str() {
                return Some(s.to_string());
            }
            // list form: [{"name": "default", "template": "..."}, ...]
            ct.as_arr().and_then(|a| {
                a.iter()
                    .find(|e| e.str_of("name") == Some("default"))
                    .or_else(|| a.first())
                    .and_then(|e| e.str_of("template"))
                    .map(str::to_string)
            })
        })
        .unwrap_or_default();
    if template_src.is_empty() {
        template_src = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap_or_default();
    }
    // The container itself may carry the template (GGUF metadata) — it
    // outranks file discovery; the family sniffs below read it the same.
    if let Some(src) = template_override {
        template_src = src.to_string();
    }

    // The authoritative family signal is the tokenizer itself: a repo whose
    // vocab registers `<start_of_turn>` as a special token is Gemma no
    // matter where (or whether) the jinja template is shipped. File sniffs
    // remain as corroboration — some repos carry the template but register
    // the markers as plain vocab.
    let llama3 = tk.special("<|start_header_id|>").is_some()
        || template_src.contains("<|start_header_id|>")
        || raw_cfg.contains("<|start_header_id|>");
    // The engine-level architecture is the strongest signal of all: a model
    // running on the gemma4 pipeline uses the Gemma turn format regardless
    // of what its (possibly stripped) tokenizer metadata says.
    let gemma = family_hint == Some("gemma")
        || tk.special("<start_of_turn>").is_some()
        || template_src.contains("<start_of_turn>")
        || raw_cfg.contains("<start_of_turn>");
    let mut out = String::new();
    if gemma {
        // Gemma family: BOS is added by the tokenizer; roles are user/model,
        // with system content folded into the first user turn (reference
        // template behaviour). Media placeholders are typed so the gemma4
        // prepare pass can frame each kind.
        //
        // The turn-marker *literals* changed across generations: Gemma ≤3
        // registers `<start_of_turn>`/`<end_of_turn>`; Gemma 4 renamed its
        // special tokens to the `<|x>` (open) / `<x|>` (close) scheme —
        // `<|turn>`/`<turn|>` — consistent with its boi/eoi image markers
        // (`<|image>`/`<image|>`, designated by config.json) and with
        // `<turn|>` being a configured generation EOS. The pair is resolved
        // against the tokenizer so one binary renders both generations; a
        // marker emitted as a non-special literal would silently BPE into
        // plain text pieces the model treats as prose.
        // Marker dialect resolution, in order of authority:
        //   1. the chat template the checkpoint itself ships (what
        //      llama.cpp executes verbatim — the trained dialect),
        //   2. tokenizer specials,
        //   3. classic literals with a warning.
        // A vocab can carry BOTH families (gemma-4 ggufs register the
        // renamed <|turn> specials while the template still speaks
        // <start_of_turn>) — trusting specials alone renders markers the
        // model was never trained to chat with.
        if std::env::var("CIMA_DUMP_RENDER").is_ok() {
            let head: String = template_src.chars().take(220).collect();
            eprintln!("template_src ({} chars): {:?}", template_src.len(), head);
        }
        let in_vocab = |s: &str| tk.vocab.contains_key(s);
        // CIMA_G4_MARKERS=classic|turn — empirical override for the dialect
        // resolution below (the discriminating experiment when a checkpoint's
        // template/specials disagree about the trained chat markers).
        let forced = match std::env::var("CIMA_G4_MARKERS").as_deref() {
            Ok("classic") => Some(("<start_of_turn>", "<end_of_turn>")),
            Ok("turn") => Some(("<|turn>", "<turn|>")),
            _ => None,
        };
        let (sot, eot) = if let Some(pair) = forced {
            pair
        } else if template_src.contains("<start_of_turn>") && in_vocab("<start_of_turn>") {
            ("<start_of_turn>", "<end_of_turn>")
        } else if template_src.contains("<|turn>") && in_vocab("<|turn>") {
            ("<|turn>", "<turn|>")
        } else if tk.special("<start_of_turn>").is_some() {
            ("<start_of_turn>", "<end_of_turn>")
        } else if tk.special("<|turn>").is_some() {
            ("<|turn>", "<turn|>")
        } else {
            crate::log::warn(
                "gemma chat format requested but neither <start_of_turn> nor <|turn> is a tokenizer special; \
                 emitting classic markers (they will tokenize as plain text)",
            );
            ("<start_of_turn>", "<end_of_turn>")
        };
        if std::env::var("CIMA_G4_DEBUG").is_ok() {
            eprintln!(
                "g4 chat markers: {} … {}   (template {} chars{})",
                sot,
                eot,
                template_src.len(),
                if forced.is_some() {
                    ", forced via CIMA_G4_MARKERS"
                } else {
                    ""
                }
            );
        }
        // BOS ownership: gemma-4 tokenizers don't prepend on encode (the
        // official chat template carries the literal); older gemmas do.
        // Exactly one of {tokenizer, template} must supply it.
        if !tk.adds_bos() {
            if let Some(b) = tk.bos_literal() {
                out.push_str(b);
            } else {
                crate::log::warn("gemma template owns BOS but the bos id has no special literal; sequence starts without BOS");
            }
        }
        let mut sys = String::new();
        for t in turns {
            if t.role == "system" {
                if !sys.is_empty() {
                    sys.push('\n');
                }
                sys.push_str(&t.content);
                continue;
            }
            let role = if t.role == "assistant" {
                "model"
            } else {
                "user"
            };
            out.push_str(&format!("{}{}\n", sot, role));
            if role == "user" && !sys.is_empty() {
                out.push_str(&sys);
                out.push_str("\n\n");
                sys.clear();
            }
            // The official template concatenates media markers directly
            // ahead of the text within the turn, with no separator.
            for _ in 0..t.n_images {
                out.push_str("<image>");
            }
            for _ in 0..t.n_audio {
                out.push_str("<audio>");
            }
            out.push_str(&t.content);
            out.push_str(&format!("{}\n", eot));
        }
        out.push_str(&format!("{}model\n", sot));
        return out;
    }
    if llama3 {
        out.push_str("<|begin_of_text|>");
        for t in turns {
            out.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n",
                t.role
            ));
            push_media(&mut out, t.n_images + t.n_audio, media_token);
            out.push_str(&t.content);
            out.push_str("<|eot_id|>");
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    } else {
        // ChatML (Qwen, and the de-facto default)
        for t in turns {
            out.push_str(&format!("<|im_start|>{}\n", t.role));
            push_media(&mut out, t.n_images + t.n_audio, media_token);
            out.push_str(&t.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn push_media(out: &mut String, n: usize, token: Option<&str>) {
    if let Some(tok) = token {
        for _ in 0..n {
            out.push_str(tok);
            out.push('\n');
        }
    }
}

/// Recursively search a Json tree for a string value or object key equal
/// to `needle` (used to ask "does this post_processor inject the bos
/// literal?" without modeling its many shapes).
fn json_mentions(j: &Json, needle: &str) -> bool {
    match j {
        Json::Str(s) => s == needle,
        Json::Arr(items) => items.iter().any(|x| json_mentions(x, needle)),
        Json::Obj(map) => map
            .iter()
            .any(|(k, v)| k == needle || json_mentions(v, needle)),
        _ => false,
    }
}

/// Resolve the add-BOS policy from the two tokenizer files. Pure function,
/// unit-tested; see the hierarchy note at the call site.
fn resolve_add_bos(
    tok_cfg: Option<&str>,
    tok_json: Option<&str>,
    bos_literal: Option<&str>,
) -> bool {
    if let Some(cfg) = tok_cfg {
        if let Ok(j) = json::parse(cfg) {
            if let Some(b) = j.bool_of("add_bos_token") {
                return b;
            }
        }
    }
    if let (Some(tj), Some(lit)) = (tok_json, bos_literal) {
        // The post_processor is the structure that injects specials around
        // raw encodings. We avoid modeling its many shapes: if its textual
        // form mentions the bos literal, raw encodes get a BOS; if it exists
        // and doesn't, they don't. (tokenizer.json is machine-generated, so
        // a stray mention outside post_processor is unlikely; worst case is
        // the legacy default.)
        if let Ok(j) = json::parse(tj) {
            if let Some(pp) = j.get("post_processor") {
                if !matches!(pp, Json::Null) {
                    return json_mentions(pp, lit);
                }
            }
        }
    }
    true // legacy prepend
}

#[cfg(test)]
mod tests {
    /// Locks the BOS discovery hierarchy: explicit flag wins; absent flag
    /// defers to the post_processor; uninformative files fall back to the
    /// legacy prepend. Covers explicit-false (Qwen) and template-owned
    /// (gemma-4) layouts.
    #[test]
    fn add_bos_discovery_hierarchy() {
        use super::resolve_add_bos as r;
        // 1. explicit flag wins, regardless of tokenizer.json
        assert!(!r(
            Some(r#"{"add_bos_token": false}"#),
            Some(r#"{"post_processor": {"single": "<bos> $A"}}"#),
            Some("<bos>")
        ));
        assert!(r(
            Some(r#"{"add_bos_token": true}"#),
            Some(r#"{"post_processor": null}"#),
            Some("<bos>")
        ));
        // 2. absent flag: the post_processor decides
        assert!(r(
            Some("{}"),
            Some(r#"{"post_processor": {"type": "TemplateProcessing", "single": ["<bos>", "A"]}}"#),
            Some("<bos>")
        ));
        assert!(!r(
            Some("{}"),
            Some(r#"{"post_processor": {"type": "ByteLevel", "trim_offsets": true}}"#),
            Some("<bos>")
        ));
        // 3. fallbacks: null/missing post_processor, missing files, no literal -> legacy prepend
        assert!(r(
            Some("{}"),
            Some(r#"{"post_processor": null}"#),
            Some("<bos>")
        ));
        assert!(r(None, None, Some("<bos>")));
        assert!(r(
            Some("{}"),
            Some(r#"{"post_processor": {"single": "<bos> $A"}}"#),
            None
        ));
    }
}
