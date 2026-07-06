//! # json — dependency-free JSON
//!
//! A small, strict, allocation-conscious JSON parser and serializer built on
//! `std` only. It backs every JSON surface in the engine: `config.json`,
//! `tokenizer.json`, the safetensors header, the Hugging Face Hub API and the
//! Ollama-compatible REST API.
//!
//! Design notes:
//! * Objects preserve insertion order (Vec of pairs) — required for
//!   deterministic API responses and stable error messages.
//! * Numbers are kept as `f64`; integer accessors validate losslessness.
//! * The parser is recursive-descent with an explicit depth limit so a
//!   malicious model repo cannot stack-overflow the daemon.

use std::fmt::Write as _;

/// Maximum nesting depth accepted by the parser (anti-DoS for untrusted
/// `config.json` / Hub payloads).
const MAX_DEPTH: usize = 128;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered object.
    Obj(Vec<(String, Json)>),
}

impl Json {
    // ----------------------------------------------------------------- accessors

    /// Object field lookup (first match).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Nested lookup: `j.path(&["a","b"])`.
    pub fn path(&self, keys: &[&str]) -> Option<&Json> {
        let mut cur = self;
        for k in keys {
            cur = cur.get(k)?;
        }
        Some(cur)
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    /// Lossless unsigned integer accessor.
    pub fn as_u64(&self) -> Option<u64> {
        let n = self.as_f64()?;
        if n >= 0.0 && n.fract() == 0.0 && n <= 2f64.powi(53) {
            Some(n as u64)
        } else {
            None
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|v| v as usize)
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self {
            Some(o)
        } else {
            None
        }
    }
    /// `true` when the value is `Null` or absent semantics are needed.
    /// True when the value is JSON `null` (part of the public surface).
    #[allow(dead_code)]
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    // ------------------------------------------------------- key accessors
    // `get(key)` + typed conversion — the idiom of every config/metadata
    // parser in the engine, folded to one call. `*_of` returns Option,
    // `*_or` applies a default. Kept dependency-free (no engine error
    // type) so this module stays liftable into the client crate.

    pub fn str_of(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Json::as_str)
    }
    pub fn f64_of(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Json::as_f64)
    }
    pub fn u64_of(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Json::as_u64)
    }
    pub fn usize_of(&self, key: &str) -> Option<usize> {
        self.get(key).and_then(Json::as_usize)
    }
    pub fn bool_of(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Json::as_bool)
    }
    pub fn arr_of(&self, key: &str) -> Option<&[Json]> {
        self.get(key).and_then(Json::as_arr)
    }

    pub fn usize_or(&self, key: &str, d: usize) -> usize {
        self.usize_of(key).unwrap_or(d)
    }
    pub fn f32_or(&self, key: &str, d: f32) -> f32 {
        self.f64_of(key).map(|v| v as f32).unwrap_or(d)
    }
    pub fn bool_or(&self, key: &str, d: bool) -> bool {
        self.bool_of(key).unwrap_or(d)
    }

    // ----------------------------------------------------------------- builders

    pub fn obj() -> Json {
        Json::Obj(Vec::new())
    }
    /// Fluent insertion for building API responses.
    pub fn set(mut self, key: &str, val: Json) -> Json {
        if let Json::Obj(ref mut m) = self {
            m.push((key.to_string(), val));
        }
        self
    }
    pub fn s(v: &str) -> Json {
        Json::Str(v.to_string())
    }
    pub fn n(v: f64) -> Json {
        Json::Num(v)
    }
    pub fn u(v: u64) -> Json {
        Json::Num(v as f64)
    }
    pub fn b(v: bool) -> Json {
        Json::Bool(v)
    }

    // ----------------------------------------------------------------- serialize

    /// Compact serialization (no whitespace) — used for NDJSON streaming.
    pub fn dump(&self) -> String {
        let mut out = String::with_capacity(256);
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 2f64.powi(53) {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{}", n);
                }
            }
            Json::Str(s) => write_escaped(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// --------------------------------------------------------------------- parser

/// Parse a complete JSON document. Trailing garbage is an error.
pub fn parse(input: &str) -> Result<Json, String> {
    let bytes = input.as_bytes();
    let mut p = Parser { b: bytes, i: 0 };
    p.skip_ws();
    let v = p.value(0)?;
    p.skip_ws();
    if p.i != bytes.len() {
        return Err(format!("json: trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("json: expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err("json: nesting too deep".into());
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("json: unexpected byte at {}", self.i)),
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("json: bad literal at byte {}", self.i))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "json: utf8")?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("json: bad number at byte {}", start))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err("json: unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.peek() {
                        Some(b'"') => {
                            out.push('"');
                            self.i += 1;
                        }
                        Some(b'\\') => {
                            out.push('\\');
                            self.i += 1;
                        }
                        Some(b'/') => {
                            out.push('/');
                            self.i += 1;
                        }
                        Some(b'b') => {
                            out.push('\u{8}');
                            self.i += 1;
                        }
                        Some(b'f') => {
                            out.push('\u{c}');
                            self.i += 1;
                        }
                        Some(b'n') => {
                            out.push('\n');
                            self.i += 1;
                        }
                        Some(b'r') => {
                            out.push('\r');
                            self.i += 1;
                        }
                        Some(b't') => {
                            out.push('\t');
                            self.i += 1;
                        }
                        Some(b'u') => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // surrogate pair
                                if self.b.get(self.i) == Some(&b'\\')
                                    && self.b.get(self.i + 1) == Some(&b'u')
                                {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00))
                                } else {
                                    return Err("json: lone surrogate".into());
                                }
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(format!("json: bad escape at byte {}", self.i)),
                    }
                }
                Some(_) => {
                    // Copy a UTF-8 run verbatim.
                    let start = self.i;
                    while matches!(self.peek(), Some(c) if c != b'"' && c != b'\\') {
                        self.i += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|_| "json: invalid utf8 in string")?,
                    );
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("json: truncated \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| "json: utf8")?;
        let v = u32::from_str_radix(s, 16).map_err(|_| "json: bad \\u escape")?;
        self.i += 4;
        Ok(v)
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                _ => return Err(format!("json: expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.value(depth + 1)?;
            out.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                _ => return Err(format!("json: expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Malformed inputs must return Err — never panic, never loop.
    #[test]
    fn malformed_inputs_error() {
        for bad in [
            "",
            "{",
            "}",
            "[",
            "[1,",
            "{\"a\":}",
            "{\"a\" 1}",
            "{\"a\":1,}",
            "nul",
            "tru",
            "+1",
            "01x",
            "\"unterminated",
            "\"bad escape \\q\"",
            "{\"a\":\"\\u12\"}",
        ] {
            assert!(parse(bad).is_err(), "accepted malformed: {:?}", bad);
        }
    }

    /// Deep nesting must not blow the stack (recursive descent guard).
    #[test]
    fn deep_nesting_bounded() {
        let depth = 100_000;
        let s = "[".repeat(depth) + &"]".repeat(depth);
        // Either parses or errors — must not crash the process.
        let _ = parse(&s);
    }

    /// Number forms: integers, floats, exponents, negatives round-trip.
    #[test]
    fn numbers() {
        for (s, v) in [
            ("0", 0.0),
            ("-1", -1.0),
            ("3.5", 3.5),
            ("1e3", 1000.0),
            ("-2.5e-1", -0.25),
        ] {
            assert_eq!(parse(s).unwrap().as_f64(), Some(v), "{}", s);
        }
        // Out-of-range usize casts must not wrap silently.
        assert_eq!(parse("-1").unwrap().as_usize(), None);
    }

    /// Escapes + UTF-8 + surrogate pairs in strings.
    #[test]
    fn strings() {
        assert_eq!(parse(r#""a\nb""#).unwrap().as_str(), Some("a\nb"));
        assert_eq!(parse(r#""\u00e9""#).unwrap().as_str(), Some("é"));
        assert_eq!(parse(r#""\ud83d\ude00""#).unwrap().as_str(), Some("😀"));
        assert_eq!(parse("\"caña\"").unwrap().as_str(), Some("caña"));
    }

    /// Object key order is preserved (the engine relies on insertion order
    /// for deterministic serialization).
    #[test]
    fn object_order_preserved() {
        let v = parse(r#"{"z":1,"a":2,"m":3}"#).unwrap();
        if let Json::Obj(pairs) = &v {
            let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(keys, ["z", "a", "m"]);
        } else {
            panic!("not an object");
        }
    }

    /// Duplicate keys: get() must resolve deterministically (first wins or
    /// last wins — locked here so a refactor can't silently flip it).
    #[test]
    fn duplicate_keys_deterministic() {
        let v = parse(r#"{"a":1,"a":2}"#).unwrap();
        let got = v.f64_of("a").unwrap();
        assert_eq!(got, 1.0, "get() resolves the FIRST occurrence");
    }
}

// ---------------------------------------------------------------------------
// JsonGuard — byte-level JSON prefix acceptor for constrained decoding.
//
// `format: "json"` (and the schema variant) must GUARANTEE syntactically
// valid output — prompt engineering alone yields markdown fences and
// trailing prose. The guarantee lives in the sampler: before a token is
// committed, its raw bytes are fed to a clone of this guard; if any byte
// would take the stream outside the language of JSON-value prefixes, the
// token's logit is masked and the sampler redraws. EOS is likewise masked
// until the top-level value is complete, and generation stops the moment
// it is — no fences before, no prose after, by construction.
//
// The acceptor recognizes exactly one top-level JSON value (RFC 8259
// grammar: object / array / string / number / true / false / null), with
// insignificant whitespace between structural tokens. It works on BYTES,
// not characters: inside strings every byte ≥ 0x20 except `"` and `\` is
// legal, which admits multi-byte UTF-8 sequences split across tokens
// without decoding them. Control characters must be escaped, as the RFC
// demands.
//
// State is a few bytes plus the container stack, so the clone-probe-commit
// pattern in the sampler is effectively free.

/// Sub-states of a number literal. "Complete" sub-states (those where the
/// number could legally end) are `IntZero`, `Int`, `Frac`, `Exp`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum NumS {
    Sign,     // consumed '-', digit required
    IntZero,  // consumed leading '0' (no more int digits allowed)
    Int,      // inside 1-9 digit run
    DotFirst, // consumed '.', digit required
    Frac,     // inside fraction digits
    ExpSign,  // consumed 'e'/'E', sign or digit required
    ExpFirst, // consumed exponent sign, digit required
    Exp,      // inside exponent digits
}

impl NumS {
    fn can_end(self) -> bool {
        matches!(self, NumS::IntZero | NumS::Int | NumS::Frac | NumS::Exp)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum S {
    /// Expecting the start of a value (top level, after ':', after ',' in
    /// an array).
    Value,
    /// After '{': first key or immediate '}'.
    ObjKeyOrEnd,
    /// After ',' in an object: a key string is required.
    ObjKeyStart,
    /// After a key string: ':' required.
    ObjColon,
    /// After a member value: ',' or '}'.
    ObjNext,
    /// After '[': first element or immediate ']'.
    ArrValOrEnd,
    /// After an element: ',' or ']'.
    ArrNext,
    /// Inside a string. `key` routes the closing quote (key → ':',
    /// value → container). `uni` counts remaining \uXXXX hex digits.
    Str {
        key: bool,
        esc: bool,
        uni: u8,
    },
    Num(NumS),
    /// Inside `true` / `false` / `null`; `pat` is the remaining suffix.
    Lit {
        pat: &'static [u8],
        i: u8,
    },
    /// Top-level value complete: only whitespace may follow.
    End,
}

#[derive(Clone, Debug)]
pub struct JsonGuard {
    /// Open containers, innermost last: b'{' or b'['.
    stack: Vec<u8>,
    state: S,
}

impl Default for JsonGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGuard {
    pub fn new() -> Self {
        JsonGuard {
            stack: Vec::new(),
            state: S::Value,
        }
    }

    /// True once the top-level value is closed — the generation-stop
    /// condition. A bare top-level number also counts when it could
    /// legally end (`42` is complete even though `425` could follow).
    pub fn complete(&self) -> bool {
        match self.state {
            S::End => true,
            S::Num(n) => self.stack.is_empty() && n.can_end(),
            _ => false,
        }
    }

    /// Definitively closed (container/string/literal ended). A bare
    /// top-level number is soft-complete but NOT hard-complete — more
    /// digits could follow.
    pub fn hard_complete(&self) -> bool {
        matches!(self.state, S::End)
    }

    /// Feed a token's bytes. Returns false at the FIRST illegal byte, at
    /// which point this guard's state is unspecified — callers probe a
    /// clone and commit it only on success (the sampler's pattern).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.push_byte(b))
    }

    /// A value just closed: route to the enclosing container, or End.
    fn value_done(&mut self) {
        self.state = match self.stack.last() {
            None => S::End,
            Some(b'{') => S::ObjNext,
            Some(b'[') => S::ArrNext,
            _ => unreachable!(),
        };
    }

    /// Dispatch a byte that must begin a value.
    fn begin_value(&mut self, b: u8) -> bool {
        match b {
            b'{' => {
                self.stack.push(b'{');
                self.state = S::ObjKeyOrEnd;
            }
            b'[' => {
                self.stack.push(b'[');
                self.state = S::ArrValOrEnd;
            }
            b'"' => {
                self.state = S::Str {
                    key: false,
                    esc: false,
                    uni: 0,
                }
            }
            b'-' => self.state = S::Num(NumS::Sign),
            b'0' => self.state = S::Num(NumS::IntZero),
            b'1'..=b'9' => self.state = S::Num(NumS::Int),
            b't' => self.state = S::Lit { pat: b"rue", i: 0 },
            b'f' => self.state = S::Lit { pat: b"alse", i: 0 },
            b'n' => self.state = S::Lit { pat: b"ull", i: 0 },
            _ => return false,
        }
        true
    }

    fn push_byte(&mut self, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        match self.state {
            S::Value => ws || self.begin_value(b),
            S::ObjKeyOrEnd => {
                if ws {
                    return true;
                }
                match b {
                    b'"' => {
                        self.state = S::Str {
                            key: true,
                            esc: false,
                            uni: 0,
                        }
                    }
                    b'}' => {
                        self.stack.pop();
                        self.value_done();
                    }
                    _ => return false,
                }
                true
            }
            S::ObjKeyStart => {
                if ws {
                    return true;
                }
                if b != b'"' {
                    return false;
                }
                self.state = S::Str {
                    key: true,
                    esc: false,
                    uni: 0,
                };
                true
            }
            S::ObjColon => {
                if ws {
                    return true;
                }
                if b != b':' {
                    return false;
                }
                self.state = S::Value;
                true
            }
            S::ObjNext => {
                if ws {
                    return true;
                }
                match b {
                    b',' => self.state = S::ObjKeyStart,
                    b'}' => {
                        self.stack.pop();
                        self.value_done();
                    }
                    _ => return false,
                }
                true
            }
            S::ArrValOrEnd => {
                if ws {
                    return true;
                }
                if b == b']' {
                    self.stack.pop();
                    self.value_done();
                    return true;
                }
                self.begin_value(b)
            }
            S::ArrNext => {
                if ws {
                    return true;
                }
                match b {
                    b',' => self.state = S::Value,
                    b']' => {
                        self.stack.pop();
                        self.value_done();
                    }
                    _ => return false,
                }
                true
            }
            S::Str { key, esc, uni } => {
                if uni > 0 {
                    if !b.is_ascii_hexdigit() {
                        return false;
                    }
                    self.state = S::Str {
                        key,
                        esc: false,
                        uni: uni - 1,
                    };
                    return true;
                }
                if esc {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.state = S::Str {
                                key,
                                esc: false,
                                uni: 0,
                            }
                        }
                        b'u' => {
                            self.state = S::Str {
                                key,
                                esc: false,
                                uni: 4,
                            }
                        }
                        _ => return false,
                    }
                    return true;
                }
                match b {
                    b'"' => {
                        if key {
                            self.state = S::ObjColon;
                        } else {
                            self.value_done();
                        }
                    }
                    b'\\' => {
                        self.state = S::Str {
                            key,
                            esc: true,
                            uni: 0,
                        }
                    }
                    // RFC 8259: unescaped control characters are illegal;
                    // everything else (incl. UTF-8 continuation bytes) is a
                    // legal string byte.
                    0x00..=0x1F => return false,
                    _ => {}
                }
                true
            }
            S::Num(n) => {
                let next = match (n, b) {
                    (NumS::Sign, b'0') => Some(NumS::IntZero),
                    (NumS::Sign, b'1'..=b'9') => Some(NumS::Int),
                    (NumS::Int, b'0'..=b'9') => Some(NumS::Int),
                    (NumS::IntZero | NumS::Int, b'.') => Some(NumS::DotFirst),
                    (NumS::DotFirst | NumS::Frac, b'0'..=b'9') => Some(NumS::Frac),
                    (NumS::IntZero | NumS::Int | NumS::Frac, b'e' | b'E') => Some(NumS::ExpSign),
                    (NumS::ExpSign, b'+' | b'-') => Some(NumS::ExpFirst),
                    (NumS::ExpSign | NumS::ExpFirst | NumS::Exp, b'0'..=b'9') => Some(NumS::Exp),
                    _ => None,
                };
                if let Some(next) = next {
                    self.state = S::Num(next);
                    return true;
                }
                // Not a number byte: the literal ends HERE, and `b` must be
                // legal in the enclosing context — re-dispatch it.
                if !n.can_end() {
                    return false;
                }
                self.value_done();
                self.push_byte(b)
            }
            S::Lit { pat, i } => {
                if b != pat[i as usize] {
                    return false;
                }
                if i as usize + 1 == pat.len() {
                    self.value_done();
                } else {
                    self.state = S::Lit { pat, i: i + 1 };
                }
                true
            }
            S::End => ws,
        }
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    fn accepts(s: &str) -> JsonGuard {
        let mut g = JsonGuard::new();
        assert!(g.push_bytes(s.as_bytes()), "should accept prefix: {s:?}");
        g
    }
    fn rejects(s: &str) {
        let mut g = JsonGuard::new();
        assert!(!g.push_bytes(s.as_bytes()), "should reject: {s:?}");
    }

    #[test]
    fn complete_object_and_nesting() {
        let g = accepts(r#" {"a": [1, 2.5e-3, true, null, "x\n\u00e9"], "b": {}} "#);
        assert!(g.complete());
        assert!(accepts(r#"{"answer": 4, "note": "ok"}"#).complete());
    }

    #[test]
    fn prefixes_are_accepted_but_incomplete() {
        for p in [
            r#"{"#,
            r#"{"a""#,
            r#"{"a": "#,
            r#"{"a": [1,"#,
            r#"{"a": "unterminated"#,
        ] {
            assert!(!accepts(p).complete(), "prefix should be incomplete: {p:?}");
        }
    }

    #[test]
    fn rejects_non_json_and_trailing_junk() {
        rejects("```");
        rejects("Sure, here is");
        rejects(r#"{"a" 1}"#);
        rejects(r#"{"a": 01}"#);
        rejects("{} extra");
        rejects("[1 2]");
        rejects("{\"a\": \"literal\ncontrol\"}"); // unescaped control char
    }

    #[test]
    fn utf8_bytes_inside_strings_pass_split_or_not() {
        let mut g = JsonGuard::new();
        assert!(g.push_bytes(br#"{"k": ""#));
        let e_acute = "é".as_bytes(); // 0xC3 0xA9 — feed split across pushes
        assert!(g.push_bytes(&e_acute[..1]));
        assert!(g.push_bytes(&e_acute[1..]));
        assert!(g.push_bytes(br#""}"#));
        assert!(g.complete());
    }

    #[test]
    fn top_level_number_completes_softly() {
        let g = accepts("42");
        assert!(g.complete()); // could extend, but may legally end
        assert!(!accepts("-").complete());
        assert!(!accepts("4.").complete());
    }

    #[test]
    fn clone_probe_commit_pattern() {
        let g = accepts(r#"{"a""#);
        let mut probe = g.clone();
        assert!(!probe.push_bytes(b"}")); // ':' required — probe corrupt, discard
        let mut probe2 = g.clone();
        assert!(probe2.push_bytes(b": 1}"));
        assert!(probe2.complete());
    }
}

// ---------------------------------------------------------------------------
// SchemaGuard — schema-compiled constrained decoding for `format: {schema}`.
//
// A (common-subset) JSON Schema compiles into a PLAN: fixed scaffold text
// (braces, quoted keys, separators — force-fed as tokens, never sampled)
// alternating with typed value HOLES the model fills under byte
// constraints. Required keys and value types are then guaranteed by
// construction, the same class of guarantee JsonGuard gives syntax.
//
// Supported subset: type=object with properties (+required — when present,
// only required keys are emitted; otherwise all), integer, number, string,
// boolean, nested objects; anything else (arrays, enums, unions, missing
// type) becomes a free JSON-value hole policed by JsonGuard. Schemas with
// no compilable object shape return None and the caller falls back to
// plain JSON-mode.

#[derive(Clone, Debug)]
pub enum Seg {
    /// Literal scaffold bytes: emitted by force-feeding their tokens.
    Fixed(String),
    /// A model-filled hole with a byte-level type constraint.
    Val(ValKind),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValKind {
    Int,
    Num,
    Bool,
    /// String BODY: the opening quote lives in the preceding Fixed; the
    /// closing quote is consumed by the hole itself.
    Str,
    /// Any single JSON value (JsonGuard-policed).
    Any,
}

/// Compile a schema into a plan. `None` → not representable, fall back.
pub fn compile_schema(schema: &Json) -> Option<Vec<Seg>> {
    let mut segs = Vec::new();
    compile_value(schema, &mut segs, 0)?;
    // Merge adjacent Fixed runs so forced text spans are maximal.
    let mut merged: Vec<Seg> = Vec::with_capacity(segs.len());
    for seg in segs {
        match (merged.last_mut(), seg) {
            (Some(Seg::Fixed(a)), Seg::Fixed(b)) => a.push_str(&b),
            (_, seg) => merged.push(seg),
        }
    }
    Some(merged)
}

fn compile_value(sch: &Json, segs: &mut Vec<Seg>, depth: usize) -> Option<()> {
    if depth > 16 {
        return None;
    }
    match sch.str_of("type") {
        Some("object") => {
            let props = match sch.get("properties") {
                Some(Json::Obj(p)) if !p.is_empty() => p,
                _ => return None,
            };
            let req: Vec<&str> = sch
                .arr_of("required")
                .map(|r| r.iter().filter_map(Json::as_str).collect())
                .unwrap_or_default();
            let keys: Vec<&(String, Json)> = props
                .iter()
                .filter(|(k, _)| req.is_empty() || req.contains(&k.as_str()))
                .collect();
            if keys.is_empty() {
                return None;
            }
            segs.push(Seg::Fixed("{".into()));
            for (i, (k, v)) in keys.iter().enumerate() {
                let mut lead = if i == 0 { String::new() } else { ", ".into() };
                lead.push_str(&Json::s(k).dump()); // quoted + escaped key
                lead.push_str(": ");
                segs.push(Seg::Fixed(lead));
                compile_value(v, segs, depth + 1)?;
            }
            segs.push(Seg::Fixed("}".into()));
        }
        Some("integer") => segs.push(Seg::Val(ValKind::Int)),
        Some("number") => segs.push(Seg::Val(ValKind::Num)),
        Some("boolean") => segs.push(Seg::Val(ValKind::Bool)),
        Some("string") => {
            segs.push(Seg::Fixed("\"".into()));
            segs.push(Seg::Val(ValKind::Str));
        }
        _ => segs.push(Seg::Val(ValKind::Any)),
    }
    Some(())
}

#[derive(Clone, Debug)]
enum ValState {
    Int { neg: bool, digits: u16 },
    Num { st: NumS, len: u16 },
    Bool { pat: &'static [u8], i: u8 },
    Str { esc: bool, uni: u8, len: u32 },
    Any(JsonGuard),
}

#[derive(Clone, Debug)]
pub struct SchemaGuard {
    plan: std::sync::Arc<Vec<Seg>>,
    seg: usize,
    /// Byte offset inside the current Fixed segment.
    fix: usize,
    val: Option<ValState>,
    done: bool,
}

impl SchemaGuard {
    pub fn new(plan: Vec<Seg>) -> Self {
        let mut g = SchemaGuard {
            plan: std::sync::Arc::new(plan),
            seg: 0,
            fix: 0,
            val: None,
            done: false,
        };
        g.enter_seg();
        g
    }

    fn enter_seg(&mut self) {
        if self.seg >= self.plan.len() {
            self.done = true;
            self.val = None;
            return;
        }
        self.val = match &self.plan[self.seg] {
            Seg::Fixed(_) => None,
            Seg::Val(ValKind::Int) => Some(ValState::Int {
                neg: false,
                digits: 0,
            }),
            Seg::Val(ValKind::Num) => Some(ValState::Num {
                st: NumS::Sign,
                len: 0,
            }),
            Seg::Val(ValKind::Bool) => Some(ValState::Bool { pat: b"", i: 0 }),
            Seg::Val(ValKind::Str) => Some(ValState::Str {
                esc: false,
                uni: 0,
                len: 0,
            }),
            Seg::Val(ValKind::Any) => Some(ValState::Any(JsonGuard::new())),
        };
    }

    fn next_seg(&mut self) {
        self.seg += 1;
        self.fix = 0;
        self.enter_seg();
    }

    /// The scaffold text that must be emitted next, if the guard sits at
    /// the start-or-middle of a Fixed segment. The generation loop encodes
    /// this and force-feeds the tokens instead of sampling.
    pub fn forced_text(&self) -> Option<&str> {
        match self.plan.get(self.seg) {
            Some(Seg::Fixed(t)) if !self.done => Some(&t[self.fix..]),
            _ => None,
        }
    }

    /// Plan fully emitted — the generation-stop condition.
    pub fn finished(&self) -> bool {
        self.done
    }

    /// EOS admissibility: finished, or sitting in a trailing value hole
    /// that could legally end here (a top-level bare-int plan, say).
    pub fn complete(&self) -> bool {
        if self.done {
            return true;
        }
        if self.seg + 1 != self.plan.len() {
            return false;
        }
        self.val_endable()
    }

    fn val_endable(&self) -> bool {
        match &self.val {
            Some(ValState::Int { digits, .. }) => *digits > 0,
            Some(ValState::Num { st, .. }) => st.can_end(),
            Some(ValState::Any(g)) => g.complete(),
            // Bool closes itself on its last byte; Str closes on its quote.
            _ => false,
        }
    }

    /// Same contract as [`JsonGuard::push_bytes`]: false at the first
    /// illegal byte, state then unspecified — clone-probe-commit.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.push_byte(b))
    }

    fn push_byte(&mut self, b: u8) -> bool {
        if self.done {
            return false;
        }
        match self.plan[self.seg].clone() {
            Seg::Fixed(t) => {
                if t.as_bytes()[self.fix] != b {
                    return false;
                }
                self.fix += 1;
                if self.fix == t.len() {
                    self.next_seg();
                }
                true
            }
            Seg::Val(_) => self.push_val_byte(b),
        }
    }

    fn push_val_byte(&mut self, b: u8) -> bool {
        // Borrow dance: operate on a copy, write back or transition.
        let mut st = self.val.clone().expect("val state present in Val segment");
        let mut close = false; // value ended BEFORE b → re-dispatch b
        let mut consumed_close = false; // b consumed AND value ended
        let ok = match &mut st {
            ValState::Int { neg, digits } => match b {
                b'-' if !*neg && *digits == 0 => {
                    *neg = true;
                    true
                }
                b'0'..=b'9' if *digits < 19 => {
                    *digits += 1;
                    true
                }
                _ if *digits > 0 => {
                    close = true;
                    true
                }
                _ => false,
            },
            ValState::Num { st: ns, len } => {
                let next = match (*ns, b) {
                    (NumS::Sign, b'-') => Some(NumS::Sign), // leading '-'
                    (NumS::Sign, b'0') => Some(NumS::IntZero),
                    (NumS::Sign, b'1'..=b'9') => Some(NumS::Int),
                    (NumS::Int, b'0'..=b'9') => Some(NumS::Int),
                    (NumS::IntZero | NumS::Int, b'.') => Some(NumS::DotFirst),
                    (NumS::DotFirst | NumS::Frac, b'0'..=b'9') => Some(NumS::Frac),
                    (NumS::IntZero | NumS::Int | NumS::Frac, b'e' | b'E') => Some(NumS::ExpSign),
                    (NumS::ExpSign, b'+' | b'-') => Some(NumS::ExpFirst),
                    (NumS::ExpSign | NumS::ExpFirst | NumS::Exp, b'0'..=b'9') => Some(NumS::Exp),
                    _ => None,
                };
                match next {
                    Some(n) if *len < 40 => {
                        *ns = n;
                        *len += 1;
                        true
                    }
                    _ if ns.can_end() => {
                        close = true;
                        true
                    }
                    _ => false,
                }
            }
            ValState::Bool { pat, i } => {
                if pat.is_empty() {
                    match b {
                        b't' => *pat = b"rue",
                        b'f' => *pat = b"alse",
                        _ => return false,
                    }
                    true
                } else if b == pat[*i as usize] {
                    if *i as usize + 1 == pat.len() {
                        consumed_close = true;
                    } else {
                        *i += 1;
                    }
                    true
                } else {
                    false
                }
            }
            ValState::Str { esc, uni, len } => {
                if *uni > 0 {
                    if !b.is_ascii_hexdigit() {
                        return false;
                    }
                    *uni -= 1;
                    true
                } else if *esc {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => *esc = false,
                        b'u' => {
                            *esc = false;
                            *uni = 4;
                        }
                        _ => return false,
                    }
                    true
                } else {
                    match b {
                        b'"' => {
                            consumed_close = true;
                            true
                        }
                        // Runaway brake: past 4 KiB only the close is legal.
                        _ if *len > 4096 => false,
                        b'\\' => {
                            *esc = true;
                            *len += 1;
                            true
                        }
                        0x00..=0x1F => false,
                        _ => {
                            *len += 1;
                            true
                        }
                    }
                }
            }
            ValState::Any(g) => {
                if g.push_byte(b) {
                    if g.hard_complete() {
                        consumed_close = true;
                    }
                    true
                } else if g.complete() {
                    // Soft-complete (bare number) hit a non-value byte:
                    // the value ends, `b` belongs to the scaffold.
                    close = true;
                    true
                } else {
                    false
                }
            }
        };
        if !ok {
            return false;
        }
        if close {
            self.next_seg();
            return self.push_byte(b);
        }
        self.val = Some(st);
        if consumed_close {
            self.next_seg();
        }
        true
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    fn plan(schema: &str) -> Vec<Seg> {
        compile_schema(&parse(schema).unwrap()).expect("compilable")
    }

    #[test]
    fn population_schema_end_to_end() {
        let p = plan(
            r#"{"type":"object","properties":{"population":{"type":"integer"}},"required":["population"]}"#,
        );
        let mut g = SchemaGuard::new(p);
        // Scaffold is forced, not sampled:
        assert_eq!(g.forced_text(), Some("{\"population\": "));
        assert!(g.push_bytes(b"{\"population\": "));
        assert_eq!(g.forced_text(), None); // now the model's hole
        assert!(g.push_bytes(b"68000000"));
        assert!(!g.push_bytes(b"x")); // guard state undefined after reject — probe pattern
        let mut g2 = SchemaGuard::new(plan(
            r#"{"type":"object","properties":{"population":{"type":"integer"}},"required":["population"]}"#,
        ));
        assert!(g2.push_bytes(b"{\"population\": 68000000"));
        assert!(g2.push_bytes(b"}")); // closes int, matches scaffold
        assert!(g2.finished());
        assert!(!g2.push_bytes(b" ")); // nothing after the plan
    }

    #[test]
    fn mixed_types_and_required_filter() {
        let p = plan(
            r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"note":{"type":"string"}},"required":["name","age"]}"#,
        );
        let mut g = SchemaGuard::new(p);
        assert_eq!(g.forced_text(), Some("{\"name\": \""));
        assert!(g.push_bytes(b"{\"name\": \"Ada\", \"age\": 36}"));
        assert!(g.finished()); // "note" not required → not emitted
    }

    #[test]
    fn string_escapes_and_bool() {
        let p = plan(
            r#"{"type":"object","properties":{"ok":{"type":"boolean"},"s":{"type":"string"}},"required":["ok","s"]}"#,
        );
        let mut g = SchemaGuard::new(p);
        assert!(g.push_bytes(b"{\"ok\": true, \"s\": \"a\\n\\u00e9b\"}"));
        assert!(g.finished());
    }

    #[test]
    fn any_hole_for_arrays() {
        let p = plan(r#"{"type":"object","properties":{"xs":{"type":"array"}},"required":["xs"]}"#);
        let mut g = SchemaGuard::new(p);
        assert!(g.push_bytes(b"{\"xs\": [1, \"two\", null]}"));
        assert!(g.finished());
    }

    #[test]
    fn uncompilable_falls_back() {
        assert!(compile_schema(&parse(r#"{"type":"object"}"#).unwrap()).is_none());
        assert!(compile_schema(&parse(r#"{"type":"string"}"#).unwrap()).is_some());
        // top-level string OK
    }

    #[test]
    fn wrong_type_rejected() {
        let p = plan(r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#);
        let mut g = SchemaGuard::new(p);
        assert!(g.push_bytes(b"{\"n\": "));
        assert!(!g.clone().push_bytes(b"\"str\"")); // string where int required
        assert!(!g.clone().push_bytes(b"3.5}")); // float where int required
        assert!(g.push_bytes(b"42}"));
    }
}
