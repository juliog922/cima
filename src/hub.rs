//! # hub — Hugging Face Hub downloader (`pull`)
//!
//! TLS HTTP is the one thing not reimplemented from scratch (hand-rolling TLS
//! would be a security defect, not a feature): the engine FFIs directly into
//! the system **libcurl** C library — still zero Rust crate dependencies.
//!
//! Flow for `cima pull org/repo`:
//! 1. `GET https://huggingface.co/api/models/{org}/{repo}` — file manifest.
//! 2. Select the needed files (configs, tokenizer, all `*.safetensors`, index).
//! 3. Stream each file to `./models/{org}__{repo}/` with:
//!    * `O_DIRECT`-friendly sequential writes via `.part` files + atomic rename,
//!    * HTTP `Range` resume of interrupted downloads,
//!    * a real-time progress bar (foreground) or detached daemon (background,
//!      via raw `fork`/`setsid`, logging to `models/.pull-<repo>.log`).

use crate::json;
use crate::traits::Res;
use crate::{err, log};
use std::ffi::{c_char, c_int, c_long, c_void, CString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ===========================================================================
// libcurl FFI (easy interface)
// ===========================================================================

// Mirrors libcurl's own `CURL` handle typedef; the FFI-faithful name is
// clearer than a Rust-cased alias here.
#[allow(clippy::upper_case_acronyms)]
type CURL = *mut c_void;

const CURLOPT_URL: c_int = 10002;
const CURLOPT_WRITEFUNCTION: c_int = 20011;
const CURLOPT_WRITEDATA: c_int = 10001;
const CURLOPT_FOLLOWLOCATION: c_int = 52;
const CURLOPT_HTTPHEADER: c_int = 10023;
const CURLOPT_RANGE: c_int = 10007;
const CURLOPT_FAILONERROR: c_int = 45;
const CURLOPT_USERAGENT: c_int = 10018;
const CURLINFO_RESPONSE_CODE: c_int = 0x200002;
#[allow(dead_code)] // kept: documented curl info code for future HEAD sizing
const CURLINFO_CONTENT_LENGTH_DOWNLOAD_T: c_int = 0x60000F;

extern "C" {
    fn curl_easy_init() -> CURL;
    fn curl_easy_setopt(h: CURL, opt: c_int, ...) -> c_int;
    fn curl_easy_perform(h: CURL) -> c_int;
    fn curl_easy_getinfo(h: CURL, info: c_int, ...) -> c_int;
    fn curl_easy_cleanup(h: CURL);
    fn curl_easy_strerror(code: c_int) -> *const c_char;
    fn curl_slist_append(l: *mut c_void, s: *const c_char) -> *mut c_void;
    fn curl_slist_free_all(l: *mut c_void);
    fn fork() -> c_int;
    fn setsid() -> c_int;
}

/// Sink receiving streamed response bytes from libcurl.
trait Sink {
    fn write(&mut self, chunk: &[u8]);
}

/// curl write callback trampoline: `userdata` is `&mut dyn Sink`.
unsafe extern "C" fn write_cb(
    ptr: *const u8,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let n = size * nmemb;
    let sink = &mut *(userdata as *mut &mut dyn Sink);
    sink.write(std::slice::from_raw_parts(ptr, n));
    n
}

/// One HTTPS request streamed into `sink`. `range_from` enables resume.
fn curl_get(
    url: &str,
    token: Option<&str>,
    range_from: Option<u64>,
    sink: &mut dyn Sink,
) -> Res<u32> {
    unsafe {
        let h = curl_easy_init();
        if h.is_null() {
            return Err(err!("hub", "curl_easy_init failed — is libcurl installed?"));
        }
        let curl = CurlGuard(h);
        let curl_url = CString::new(url).unwrap();
        let ua = CString::new("cima/0.1 (+https://localhost)").unwrap();
        curl_easy_setopt(h, CURLOPT_URL, curl_url.as_ptr());
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1 as c_long);
        curl_easy_setopt(h, CURLOPT_FAILONERROR, 0 as c_long);
        curl_easy_setopt(h, CURLOPT_USERAGENT, ua.as_ptr());

        let mut headers: *mut c_void = std::ptr::null_mut();
        let auth_cstr;
        if let Some(tok) = token {
            auth_cstr = CString::new(format!("Authorization: Bearer {}", tok)).unwrap();
            headers = curl_slist_append(headers, auth_cstr.as_ptr());
            curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
        }
        let range_cstr;
        if let Some(from) = range_from {
            range_cstr = CString::new(format!("{}-", from)).unwrap();
            curl_easy_setopt(h, CURLOPT_RANGE, range_cstr.as_ptr());
        }

        let mut sink_ref: &mut dyn Sink = sink;
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb as *const c_void);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &mut sink_ref as *mut _ as *mut c_void);

        let rc = curl_easy_perform(h);
        if !headers.is_null() {
            curl_slist_free_all(headers);
        }
        if rc != 0 {
            let msg = std::ffi::CStr::from_ptr(curl_easy_strerror(rc)).to_string_lossy();
            return Err(err!(
                "hub",
                "transfer failed for {}: curl error {} ({})",
                url,
                rc,
                msg
            ));
        }
        let mut code: c_long = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &mut code as *mut c_long);
        drop(curl);
        Ok(code as u32)
    }
}

struct CurlGuard(CURL);
impl Drop for CurlGuard {
    fn drop(&mut self) {
        unsafe { curl_easy_cleanup(self.0) };
    }
}

struct VecSink(Vec<u8>);
impl Sink for VecSink {
    fn write(&mut self, c: &[u8]) {
        self.0.extend_from_slice(c);
    }
}

/// Split `org/repo[:revision]` for URL construction — defaults to `main`.
/// Note that user-facing `:TAG` quantization selectors are stripped into an
/// `include` filter by the CLI/API layers before hub functions run; a colon
/// reaching this depth is a genuine git revision.
/// Split `org/repo[@revision]` for URL construction — defaults to `main`.
/// Quant `:TAG` selectors are stripped to an `include` filter upstream in
/// [`pull`]/[`split_selector`] before any hub call, so a colon must not
/// reach here; a genuine git revision is written `@rev`.
fn split_rev(model: &str) -> (&str, &str) {
    match model.split_once('@') {
        Some((r, v)) => (r, v),
        None => (model, "main"),
    }
}

/// Fetch an inclusive byte span of a repo file into memory. The preflight
/// path reads multi-gigabyte checkpoints a few kilobytes at a time: both
/// safetensors and GGUF keep their complete tensor tables in a header at
/// the front of the file.
pub(crate) fn fetch_span(model: &str, name: &str, from: u64, to: u64) -> Res<Vec<u8>> {
    let (repo, rev) = split_rev(model);
    let url = format!("https://huggingface.co/{}/resolve/{}/{}", repo, rev, name);
    let mut sink = VecSink(Vec::new());
    unsafe {
        let h = curl_easy_init();
        if h.is_null() {
            return Err(err!("hub", "curl_easy_init failed — is libcurl installed?"));
        }
        let curl = CurlGuard(h);
        let curl_url = CString::new(url.clone()).unwrap();
        let ua = CString::new("cima/0.1 (+https://localhost)").unwrap();
        curl_easy_setopt(h, CURLOPT_URL, curl_url.as_ptr());
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1 as c_long);
        curl_easy_setopt(h, CURLOPT_FAILONERROR, 0 as c_long);
        curl_easy_setopt(h, CURLOPT_USERAGENT, ua.as_ptr());
        let mut headers: *mut c_void = std::ptr::null_mut();
        let auth_cstr;
        if let Some(tok) = hf_token() {
            auth_cstr = CString::new(format!("Authorization: Bearer {}", tok)).unwrap();
            headers = curl_slist_append(headers, auth_cstr.as_ptr());
            curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
        }
        let range_cstr = CString::new(format!("{}-{}", from, to)).unwrap();
        curl_easy_setopt(h, CURLOPT_RANGE, range_cstr.as_ptr());
        let mut sink_ref: &mut dyn Sink = &mut sink;
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb as *const c_void);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &mut sink_ref as *mut _ as *mut c_void);
        let rc = curl_easy_perform(h);
        if !headers.is_null() {
            curl_slist_free_all(headers);
        }
        if rc != 0 {
            let msg = std::ffi::CStr::from_ptr(curl_easy_strerror(rc)).to_string_lossy();
            return Err(err!(
                "hub",
                "range fetch failed for {}: curl error {} ({})",
                url,
                rc,
                msg
            ));
        }
        let mut code: c_long = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &mut code as *mut c_long);
        drop(curl);
        if !(200..300).contains(&(code as u32)) {
            return Err(err!("hub", "range fetch of {} returned HTTP {}", url, code));
        }
    }
    Ok(sink.0)
}

// ===========================================================================
// Hub API
// ===========================================================================

/// Root of the local weight store. `CIMA_MODELS_DIR` overrides the default
/// `./models` (relative to the working directory) — set it whenever the
/// binary and the data live in different places, e.g. containers mounting
/// a volume at `/data/models`.
pub fn models_dir() -> PathBuf {
    match std::env::var("CIMA_MODELS_DIR") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => PathBuf::from("./models"),
    }
}

/// Map `org/repo` (optionally `org/repo:revision`) to its local directory.
pub fn local_dir(model: &str) -> PathBuf {
    models_dir().join(model.replace('/', "__").replace(':', "@"))
}

/// Is `model` present on local disk and loadable, format-aware?
///
/// A selector `ORG/REPO[:TAG]` is present when its directory holds a
/// complete checkpoint of either format:
///   * safetensors — `config.json` plus at least one `*.safetensors` shard;
///   * GGUF — at least one non-mmproj `*.gguf` file, and when a `:TAG` was
///     given, a file whose name matches that quant tag.
///
/// This is the presence signal an orchestrator polls: it never touches the
/// network and is the same truth `list_local` reports. It does NOT prove
/// the model loads on the GPU (that needs VRAM), only that the bytes are on
/// disk — which is exactly the "pull finished" question.
pub fn is_local(model: &str) -> bool {
    is_local_in(&models_dir(), model)
}

/// [`is_local`] against an explicit store root (testable without touching
/// the process-global `CIMA_MODELS_DIR`).
fn is_local_in(root: &std::path::Path, model: &str) -> bool {
    let (_repo, tag) = split_selector(model);
    // Pull stores under the FULL selector (`ORG/REPO:TAG` → `ORG__REPO@TAG`),
    // so map the whole `model`, not the bare repo.
    let dir = root.join(model.replace('/', "__").replace(':', "@"));
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return false;
    };
    let mut has_config = false;
    let mut has_safetensors = false;
    let mut ggufs: Vec<String> = Vec::new();
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n == "config.json" {
            has_config = true;
        } else if n.ends_with(".safetensors") {
            has_safetensors = true;
        } else if n.ends_with(".gguf") && !n.to_ascii_lowercase().contains("mmproj") {
            ggufs.push(n);
        }
    }
    let gguf_ok = match tag {
        Some(t) => {
            let tl = t.to_ascii_lowercase();
            ggufs.iter().any(|n| n.to_ascii_lowercase().contains(&tl))
        }
        None => !ggufs.is_empty(),
    };
    gguf_ok || (has_config && has_safetensors)
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty())
}

/// Metadata files always downloaded (configs + tokenizer; tiny).
fn is_meta(name: &str) -> bool {
    matches!(
        name,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "preprocessor_config.json"
            | "special_tokens_map.json"
            | "model.safetensors.index.json"
            | "chat_template.json"
    )
}

/// Weight-file selection. Default: every `*.safetensors` shard (skipping
/// pytorch_model.bin duplicates). With an `--include` filter, only weight
/// files whose name contains the (case-insensitive) substring are taken —
/// the mechanism for picking one quantization out of a multi-quant repo,
/// e.g. `--include Q4_K_M` against an unsloth `*-GGUF` repository.
fn wanted(name: &str, include: Option<&str>) -> bool {
    if is_meta(name) {
        return true;
    }
    // mmproj sidecars ride along with any tagged gguf pull: the tag names
    // an LM quantization ("Q4_K_M"), which the multimodal projector file
    // ("mmproj-F16.gguf") never contains — filtering it out strands the
    // model without its vision/audio towers. Preference order mirrors the
    // load-time merge (f16 > bf16 > f32); an untyped mmproj also rides.
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".gguf") && lower.contains("mmproj") {
        // Candidate on any tagged pull; `pull_inner` keeps only the best
        // variant (f16 > bf16 > f32), mirroring the load-time merge.
        return include.is_some();
    }
    match include {
        Some(f) => lower.contains(&f.to_ascii_lowercase()),
        None => name.ends_with(".safetensors"),
    }
}

/// List the repo's files via the Hub model API.
pub fn list_repo(model: &str, include: Option<&str>) -> Res<Vec<(String, u64)>> {
    let (repo, revision) = split_rev(model);
    let url = format!(
        "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
        repo, revision
    );
    let mut body = VecSink(Vec::new());
    let code = curl_get(&url, hf_token().as_deref(), None, &mut body)?;
    if code == 401 || code == 403 {
        // The Hub answers 401 for NONEXISTENT repos too when the request
        // is unauthenticated (existence is not leaked) — a typo'd name
        // and a private repo are indistinguishable from here.
        return Err(err!(
            "hub",
            "HTTP {} for '{}' — repo not found, gated, or private. Check the spelling \
             (https://huggingface.co/{}); if it is gated/private, export HF_TOKEN=<token>",
            code,
            repo,
            repo
        ));
    }
    if code == 404 {
        return Err(err!(
            "hub",
            "model '{}' (revision '{}') not found on the Hugging Face Hub",
            repo,
            revision
        ));
    }
    if code != 200 {
        return Err(err!("hub", "Hub API returned HTTP {} for {}", code, url));
    }
    let txt = String::from_utf8_lossy(&body.0);
    let j = json::parse(&txt).map_err(|e| err!("hub", "Hub API response malformed: {}", e))?;
    let siblings = j
        .arr_of("siblings")
        .ok_or_else(|| err!("hub", "Hub API response for '{}' missing 'siblings'", repo))?;
    let mut out = Vec::new();
    for s in siblings {
        if let Some(name) = s.str_of("rfilename") {
            if wanted(name, include) {
                let size = s.u64_of("size").unwrap_or(0);
                out.push((name.to_string(), size));
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// File download with resume + progress bar
// ===========================================================================

/// Progress-bar sink streaming to a `.part` file.
struct FileSink {
    file: std::fs::File,
    written: u64,
    total: u64,
    name: String,
    started: Instant,
    last_draw: Instant,
    quiet: bool,
}

impl Sink for FileSink {
    fn write(&mut self, chunk: &[u8]) {
        let _ = self.file.write_all(chunk);
        self.written += chunk.len() as u64;
        if !self.quiet && self.last_draw.elapsed().as_millis() > 80 {
            self.draw();
            self.last_draw = Instant::now();
        }
    }
}

impl FileSink {
    fn draw(&self) {
        let frac = if self.total > 0 {
            self.written as f64 / self.total as f64
        } else {
            0.0
        };
        let filled = (frac * 30.0) as usize;
        let mbps = self.written as f64 / 1.0e6 / self.started.elapsed().as_secs_f64().max(0.001);
        eprint!(
            "\r  {} [{}{}] {:5.1}% {:>9} / {:<9} {:6.1} MB/s ",
            pad(&self.name, 34),
            "█".repeat(filled),
            "░".repeat(30 - filled),
            frac * 100.0,
            crate::cuda::fmt_bytes(self.written as usize),
            crate::cuda::fmt_bytes(self.total as usize),
            mbps
        );
        let _ = std::io::stderr().flush();
    }
}

fn pad(s: &str, n: usize) -> String {
    if s.len() >= n {
        format!("…{}", &s[s.len() - n + 1..])
    } else {
        format!("{}{}", s, " ".repeat(n - s.len()))
    }
}

/// Download one repo file with `Range` resume; atomic rename on completion.
fn fetch_file(model: &str, name: &str, size: u64, dir: &Path, quiet: bool) -> Res<()> {
    let (repo, revision) = split_rev(model);
    let dest = dir.join(name);
    if dest.exists() && dest.metadata().map(|m| m.len()).unwrap_or(0) == size && size > 0 {
        if !quiet {
            eprintln!("  {} already complete, skipping", pad(name, 34));
        }
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| err!("hub", "mkdir {}: {}", parent.display(), e))?;
    }
    let part = dir.join(format!("{}.part", name));
    let resume_from = part.metadata().map(|m| m.len()).unwrap_or(0);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part)
        .map_err(|e| err!("hub", "open {}: {}", part.display(), e))?;

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo, revision, name
    );
    let mut sink = FileSink {
        file,
        written: resume_from,
        total: size,
        name: name.to_string(),
        started: Instant::now(),
        last_draw: Instant::now(),
        quiet,
    };
    let code = curl_get(
        &url,
        hf_token().as_deref(),
        if resume_from > 0 {
            Some(resume_from)
        } else {
            None
        },
        &mut sink,
    )?;
    if !(200..300).contains(&code) {
        return Err(err!(
            "hub",
            "HTTP {} downloading {} — partial file kept at {} for resume",
            code,
            url,
            part.display()
        ));
    }
    let _ = sink.file.flush();
    if !quiet {
        sink.draw();
        eprintln!();
    }
    if size > 0 && sink.written != size {
        return Err(err!("hub", "'{}': downloaded {} bytes but Hub manifest says {} — truncated transfer, rerun pull to resume", name, sink.written, size));
    }
    std::fs::rename(&part, &dest).map_err(|e| {
        err!(
            "hub",
            "rename {} -> {}: {}",
            part.display(),
            dest.display(),
            e
        )
    })?;
    Ok(())
}

/// Pull a full model repository into `./models/`.
///
/// `background == true` detaches a daemonized child (`fork` + `setsid`) that
/// performs the download with logs redirected to `models/.pull-<id>.log`,
/// while the parent returns immediately (Ollama's `pull` UX).
/// Download only the repository's metadata files (config.json, tokenizer
/// configs — a few hundred KiB) into the local model directory. This powers
/// the *preflight compatibility gate*: `config.json` is validated against the
/// engine's executable architecture set before any multi-GiB weight download
/// starts.
pub fn pull_config(model: &str) -> Res<std::path::PathBuf> {
    pull_config_to(model, local_dir(model))
}

/// Fetch a repo's metadata files (config.json & friends) into an ARBITRARY
/// directory. `cima check` points this at a self-cleaning temp dir so that a
/// read-only inspection NEVER creates entries under ./models — a folder
/// there implies "pulled" to `cima ls` and to humans.
pub fn pull_config_to(model: &str, dir: std::path::PathBuf) -> Res<std::path::PathBuf> {
    std::fs::create_dir_all(&dir).map_err(|e| err!("hub", "mkdir {}: {}", dir.display(), e))?;
    let files = list_repo(model, None)?;
    for (name, size) in files.iter().filter(|(n, _)| is_meta(n)) {
        fetch_file(model, name, *size, &dir, true)?;
    }
    if !dir.join("config.json").is_file() {
        return Err(err!(
            "hub",
            "repo '{}' has no config.json — cannot determine its architecture; \
             this engine only executes standard Hugging Face transformer checkpoints",
            model
        ));
    }
    Ok(dir)
}

pub fn pull(model: &str, background: bool, include: Option<&str>) -> Res<()> {
    // A `:TAG` on a GGUF repo selects a quantization (a file-name filter),
    // not a git revision. Split it centrally so no caller can leak the tag
    // into a revision URL (which 404s). An explicit `include` argument wins;
    // otherwise the tag becomes the include filter and the bare repo is used
    // for every hub request. Repos with a genuine revision are addressed
    // with `@rev` — see split_selector.
    let (repo, tag) = split_selector(model);
    let include = include.or(tag);
    let dir = local_dir(model); // keeps the tag in the local path, as before
    std::fs::create_dir_all(&dir).map_err(|e| err!("hub", "mkdir {}: {}", dir.display(), e))?;

    if background {
        let logfile = format!("./models/.pull-{}.log", model.replace('/', "__"));
        let pid = unsafe { fork() };
        if pid < 0 {
            return Err(err!(
                "hub",
                "fork() failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        if pid > 0 {
            log::info(&format!(
                "background pull of '{}' started (pid {}), log: {}",
                model, pid, logfile
            ));
            return Ok(());
        }
        // Child: new session, run the download quietly, log result, exit.
        unsafe { setsid() };
        let result = pull_inner(repo, &dir, true, include);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logfile)
            .ok();
        if let Some(f) = f.as_mut() {
            let _ = writeln!(
                f,
                "pull {} => {:?}",
                model,
                result.as_ref().map(|_| "ok").map_err(|e| e.to_string())
            );
        }
        std::process::exit(if result.is_ok() { 0 } else { 1 });
    }
    pull_inner(repo, &dir, false, include)
}

/// Split an `ORG/REPO[:TAG][@REV]` selector into `(repo, tag)`.
///
/// `:TAG` is a GGUF quantization filter (e.g. `:q8_0`, `:Q4_K_M`) — a
/// file-name substring, not a git ref. A genuine git revision is written
/// `@REV` and stays attached to the repo for the hub URL builders. This
/// keeps quant tags out of `resolve/{revision}` URLs, which is the whole
/// point: `.../resolve/q8_0/...` does not exist and 404s.
pub fn split_selector(model: &str) -> (&str, Option<&str>) {
    match model.split_once(':') {
        Some((repo, tag)) => (repo, Some(tag)),
        None => (model, None),
    }
}

fn pull_inner(model: &str, dir: &Path, quiet: bool, include: Option<&str>) -> Res<()> {
    let t0 = Instant::now();
    let mut files = list_repo(model, include)?;
    // Of the mmproj candidates, keep exactly one — the same preference the
    // loader's merge applies (f16 > bf16 > f32 > untyped). Downloading all
    // variants would multiply gigabytes for a file the loader uses once.
    {
        let rank = |n: &str| -> u8 {
            let l = n.to_ascii_lowercase();
            if l.contains("bf16") {
                1
            } else if l.contains("f16") {
                0
            } else if l.contains("f32") {
                2
            } else {
                3
            }
        };
        let best: Option<String> = files
            .iter()
            .filter(|(n, _)| n.to_ascii_lowercase().contains("mmproj") && n.ends_with(".gguf"))
            .min_by_key(|(n, _)| rank(n))
            .map(|(n, _)| n.clone());
        if let Some(keep) = best {
            files.retain(|(n, _)| {
                !(n.to_ascii_lowercase().contains("mmproj") && n.ends_with(".gguf")) || *n == keep
            });
        }
    }
    if !files.iter().any(|(n, _)| !is_meta(n)) {
        return Err(match include {
            Some(f) => err!(
                "hub",
                "repo '{}' has no weight files matching --include '{}'. \
                 Inspect the repository file list on the Hub and pass a substring of the desired file name.",
                model, f
            ),
            None => err!(
                "hub",
                "repo '{}' contains no *.safetensors weights. \
                 For multi-quant repos (e.g. GGUF), select a file explicitly with --include <substring>; \
                 note that executing GGUF requires the gguf ModelLoader (not registered in this build).",
                model
            ),
        });
    }
    let total: u64 = files.iter().map(|(_, s)| s).sum();
    if !quiet {
        eprintln!(
            "pulling {} — {} files, {}",
            model,
            files.len(),
            crate::cuda::fmt_bytes(total as usize)
        );
    }
    for (name, size) in &files {
        fetch_file(model, name, *size, dir, quiet)?;
    }
    log::metric(
        "pull",
        &[
            ("model", model.to_string()),
            ("files", files.len().to_string()),
            ("bytes", total.to_string()),
            ("secs", format!("{:.1}", t0.elapsed().as_secs_f64())),
        ],
    );
    if !quiet {
        eprintln!("✓ {} ready in {}", model, dir.display());
    }
    Ok(())
}

/// Enumerate locally available models (for `/api/tags` and `list`).
///
/// GGUF repos list one row per pulled quantization (`org/repo:Q4_K_M`,
/// size = that tag's shard bytes) — the *pullable/runnable* names —
/// mirroring how `run` addresses them. Multi-part shards
/// (`…-Q4_K_M-00001-of-00002.gguf`) collapse into their tag's row.
/// Non-GGUF repos keep the single whole-directory row.
pub fn list_local() -> Vec<(String, u64, std::time::SystemTime)> {
    list_local_caps()
        .into_iter()
        .map(|(n, s, t, _)| (n, s, t))
        .collect()
}

/// Capability detection, disk-truth only: what could this checkpoint do if
/// loaded? gguf multimodality requires the mmproj sidecar to actually be
/// present; safetensors gemma-4 carries its towers in the main shards, so
/// the config's model_type decides. Anything else is a text model.
fn caps_of(dir: &std::path::Path, has_gguf: bool) -> &'static str {
    let has_mmproj = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|f| {
                let n = f.file_name().to_string_lossy().to_ascii_lowercase();
                n.ends_with(".gguf") && n.contains("mmproj")
            })
        })
        .unwrap_or(false);
    if has_gguf {
        // GGUF text models embed too: embedding is mean-pooling over the
        // final hidden state (embed_tokens -> forward -> rmsnorm -> meanpool),
        // which is format-agnostic — the same forward pass generation uses.
        // The quant weights feed it exactly as they feed decode.
        return if has_mmproj {
            "text+vision+audio+embed"
        } else {
            "text+embed"
        };
    }
    let mm = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .map(|c| {
            c.contains("\"gemma4\"") || c.contains("audio_config") || c.contains("vision_config")
        })
        .unwrap_or(false);
    if mm {
        "text+vision+audio"
    } else {
        "text"
    }
}

/// `list_local` plus a capability string per row.
pub fn list_local_caps() -> Vec<(String, u64, std::time::SystemTime, &'static str)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(models_dir()) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || !e.path().is_dir() {
                continue;
            }
            // The pull path stores under the full selector
            // (`ORG/REPO:TAG` -> `ORG__REPO@TAG`). Split the dir name into
            // its bare repo and any `@tag`. GGUF rows take their tag from the
            // .gguf filenames (the single source of truth), so the bare repo
            // is used for those; keeping the dir tag too would double it
            // (`...GGUF:q8_0:q8_0`). The dir tag is only reattached to a
            // non-gguf (safetensors) bare row, which is the one case where
            // the directory carries the only tag information.
            let (repo_raw, dir_tag) = match name.split_once('@') {
                Some((r, t)) => (r, Some(t.to_string())),
                None => (name.as_str(), None),
            };
            let repo = repo_raw.replace("__", "/");
            // Per-quant rows keyed by tag; the () row is the non-gguf rest.
            let mut tags: std::collections::BTreeMap<Option<String>, (u64, std::time::SystemTime)> =
                std::collections::BTreeMap::new();
            if let Ok(files) = std::fs::read_dir(e.path()) {
                for f in files.flatten() {
                    let fname = f.file_name().to_string_lossy().into_owned();
                    // mmproj tower files are a capability of the model —
                    // sidecars like config.json, not runnable quants: they
                    // ride in the sidecar bucket, never as their own row.
                    let key = if fname.ends_with(".gguf")
                        && !fname.to_ascii_lowercase().contains("mmproj")
                    {
                        Some(quant_tag(&fname))
                    } else {
                        None
                    };
                    if let Ok(md) = f.metadata() {
                        let slot = tags
                            .entry(key)
                            .or_insert((0, std::time::SystemTime::UNIX_EPOCH));
                        slot.0 += md.len();
                        if let Ok(t) = md.modified() {
                            slot.1 = slot.1.max(t);
                        }
                    }
                }
            }
            let has_gguf = tags.keys().any(Option::is_some);
            let caps = caps_of(&e.path(), has_gguf);
            for (tag, (size, mtime)) in tags {
                match tag {
                    Some(t) => out.push((format!("{}:{}", repo, t), size, mtime, caps)),
                    // Sidecars (config/tokenizer) ride with the quant rows;
                    // a bare row only makes sense for non-gguf checkpoints.
                    // If the directory carried a tag (a tagged safetensors
                    // pull), preserve it — here it is the only tag source.
                    None if !has_gguf => {
                        let label = match &dir_tag {
                            Some(t) => format!("{}:{}", repo, t),
                            None => repo.clone(),
                        };
                        out.push((label, size, mtime, caps))
                    }
                    None => {}
                }
            }
        }
    }
    // Two directories can resolve to the same `repo:tag` row — e.g. an older
    // untagged pull (`ORG__REPO`) and a later tagged one (`ORG__REPO@TAG`)
    // that both contain the same quant .gguf. Collapse by the emitted label,
    // keeping the newest (largest mtime) and its size, so `tags` never lists
    // the same model twice.
    let mut by_label: std::collections::BTreeMap<
        String,
        (u64, std::time::SystemTime, &'static str),
    > = std::collections::BTreeMap::new();
    for (label, size, mtime, caps) in out {
        by_label
            .entry(label)
            .and_modify(|slot| {
                if mtime > slot.1 {
                    *slot = (size, mtime, caps);
                }
            })
            .or_insert((size, mtime, caps));
    }
    let mut out: Vec<_> = by_label
        .into_iter()
        .map(|(label, (size, mtime, caps))| (label, size, mtime, caps))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Extract the quantization tag from a gguf filename
/// (`gemma-4-E4B-it-UD-Q4_K_XL.gguf` → `UD-Q4_K_XL`), collapsing
/// multi-part shard suffixes (`…-Q8_0-00001-of-00002.gguf` → `Q8_0`).
/// Shared by `list_local`, `/api/tags`, and gguf tag-resolution errors.
pub fn quant_tag(file: &str) -> String {
    let mut stem = file.strip_suffix(".gguf").unwrap_or(file);
    // strip a -NNNNN-of-NNNNN multi-part suffix
    if let Some(i) = stem.rfind("-of-") {
        let (head, tail) = (&stem[..i], &stem[i + 4..]);
        if !tail.is_empty()
            && tail.bytes().all(|b| b.is_ascii_digit())
            && head
                .rfind('-')
                .map(|j| head[j + 1..].bytes().all(|b| b.is_ascii_digit()) && j + 1 < head.len())
                .unwrap_or(false)
        {
            stem = &head[..head.rfind('-').unwrap()];
        }
    }
    // take the trailing [-][segment] that looks like a quant tag: every
    // published scheme (Q4_K_M, q8_0, IQ4_XS, UD-Q4_K_XL, F16, bf16)
    // contains a digit, which prose segments ("instruct", "it") never do.
    match stem.rfind('-').map(|i| &stem[i + 1..]) {
        Some(t)
            if t.chars()
                .next()
                .map(|c| c.is_ascii_alphanumeric())
                .unwrap_or(false)
                && t.bytes().any(|b| b.is_ascii_digit()) =>
        {
            // include a UD-/IQ- style prefix segment when present
            let with_prev = stem[..stem.len() - t.len() - 1]
                .rfind('-')
                .map(|j| &stem[j + 1..])
                .unwrap_or(t);
            if with_prev.starts_with("UD-") || with_prev.starts_with("IQ") {
                with_prev.to_string()
            } else {
                t.to_string()
            }
        }
        _ => stem.to_string(),
    }
}

pub mod registry {
    //! # registry — the curated model catalog (`available`)
    //!
    //! cima supports *general* config-driven loading, but generality is not a
    //! guarantee: a pulled checkpoint can ship topology surprises, tokenizer
    //! policy quirks, or quantization layouts that load-but-misbehave. The
    //! registry is the curated subset that has **passed `cima vet`** on real
    //! hardware — the same role ollama's library plays, minus the server.
    //!
    //! v1 is embedded in the binary: zero infrastructure, versioned with the
    //! code, auditable in review. The natural upgrade path (a remote
    //! registry.json fetched over the existing libcurl FFI, falling back to
    //! this table offline) keeps the same schema.
    //!
    //! A model EARNS its row here: run `cima vet ORG/REPO`; if every check
    //! passes on your hardware, add the entry in the PR that claims support.

    pub struct RegistryEntry {
        /// Hub id, as given to `cima pull`.
        pub id: &'static str,
        /// Architecture family (`models::Arch` variant that serves it).
        pub family: &'static str,
        /// Approximate download size.
        pub size: &'static str,
        /// Capability summary, in `cima vet` terms.
        pub capabilities: &'static str,
        /// `verified` = full vet pass or recorded A/B battery on reference
        /// hardware; `experimental` = loads and generates but certification is
        /// pending; `avoid` = a known-defective artifact, listed so the
        /// knowledge ships with the product instead of living in a changelog.
        pub status: &'static str,
        /// One-line operator notes (quantization, VRAM floor, quirks).
        pub notes: &'static str,
    }

    /// The catalog. Keep ordered by family, then size.
    pub const REGISTRY: &[RegistryEntry] = &[
    RegistryEntry {
        id: "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        family: "gemma4",
        size: "7.6 GiB",
        capabilities: "generate+embed+vision+audio",
        status: "verified",
        notes: "NF4 text / 16-bit towers; ~4.4 GiB VRAM; 66 tok/s greedy / 45 sampled on RTX 3060 Laptop; full A/B certificates (vision 0.9999, audio char-identical, embed 0.999)",
    },
    RegistryEntry {
        id: "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
        family: "gemma4",
        size: "3.8 GiB",
        capabilities: "generate+vision+audio",
        status: "verified",
        notes: "Q4_K_M + mmproj-F16 (auto-included on pull); fits a 6 GiB card; ~92-97 tok/s text on RTX 3060 Laptop; clean-slate roundtrip 2026-07-02 (text temp-0, speech transcription, tone control, image)",
    },
    RegistryEntry {
        id: "unsloth/gemma-4-E2B-it-GGUF:Q8_0",
        family: "gemma4",
        size: "4.7 GiB",
        capabilities: "generate+vision+audio",
        status: "avoid",
        notes: "defective export: coarser Q4_K_M lands measurably closer to the bf16 reference than this Q8_0 does (per-layer trace 2026-07-02) and text degrades to deflection; use Q4_K_M",
    },
    RegistryEntry {
        id: "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M",
        family: "gemma4",
        size: "8.3 GiB",
        capabilities: "generate+vision+audio",
        status: "verified",
        notes: "Q4_K_M + mmproj-F16; fits a 6 GiB card via the VRAM gate; vision A/B GPU==CPU every stage, audio soft-token cos 1.0 vs bnb reference",
    },
    RegistryEntry {
        id: "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        family: "gemma4",
        size: "14.9 GiB",
        capabilities: "generate+embed+vision+audio",
        status: "verified",
        notes: "NF4 text / 16-bit towers; needs ~10 GiB host RAM to stage (host guard enforces); the E4B ground-truth reference",
    },
    RegistryEntry {
        id: "Qwen/Qwen2.5-0.5B-Instruct",
        family: "transformer",
        size: "1.0 GiB",
        capabilities: "generate+embed",
        status: "verified",
        notes: "f16; 1.25 GiB VRAM; the bench reference model (243 tok/s greedy / 178 sampled on RTX 3060 Laptop)",
    },
    RegistryEntry {
        id: "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0",
        family: "transformer",
        size: "0.63 GiB",
        capabilities: "generate+embed",
        status: "experimental",
        notes: "Q8_0; coherent text verified (temp-0 A/B 2026-07-02); the gguf-kernel control model; embed via mean-pool works but is not yet A/B-verified against the safetensors reference; formal vet pending",
    },
    RegistryEntry {
        id: "bartowski/Qwen2.5-7B-Instruct-GGUF:IQ4_XS",
        family: "transformer",
        size: "3.9 GiB",
        capabilities: "generate+embed",
        status: "experimental",
        notes: "IQ4_XS; loads and generates on a 6 GiB card; embed via mean-pool available (unverified); formal vet pending",
    },
];

    /// Render the catalog as an aligned table.
    pub fn render() -> String {
        let mut out = String::from(
        "models available to pull (curated — verified rows passed certification on real hardware;\navoid rows document known-defective artifacts):\n\n",
    );
        let wid = REGISTRY
            .iter()
            .map(|e| e.id.len())
            .max()
            .unwrap_or(0)
            .max(5);
        let wfam = REGISTRY
            .iter()
            .map(|e| e.family.len())
            .max()
            .unwrap_or(0)
            .max(6);
        let wsz = REGISTRY
            .iter()
            .map(|e| e.size.len())
            .max()
            .unwrap_or(0)
            .max(4);
        let wcap = REGISTRY
            .iter()
            .map(|e| e.capabilities.len())
            .max()
            .unwrap_or(0)
            .max(12);
        let wst = REGISTRY
            .iter()
            .map(|e| e.status.len())
            .max()
            .unwrap_or(0)
            .max(6);
        out.push_str(&format!(
            "{:<wid$}  {:<wfam$}  {:<wsz$}  {:<wcap$}  {:<wst$}  NOTES\n",
            "MODEL", "FAMILY", "SIZE", "CAPABILITIES", "STATUS",
        ));
        for e in REGISTRY {
            out.push_str(&format!(
                "{:<wid$}  {:<wfam$}  {:<wsz$}  {:<wcap$}  {:<wst$}  {}\n",
                e.id, e.family, e.size, e.capabilities, e.status, e.notes,
            ));
        }
        out.push_str(
            "\npull with: cima pull MODEL    |    certify a new model: cima vet ORG/REPO\n",
        );
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Registry hygiene: ids well-formed (org/repo), no duplicates, statuses
        /// from the closed set, and every row renders.
        #[test]
        fn registry_entries_valid() {
            let mut seen = std::collections::HashSet::new();
            for e in REGISTRY {
                assert!(
                    e.id.contains('/') && !e.id.starts_with('/') && !e.id.ends_with('/'),
                    "malformed id: {}",
                    e.id
                );
                assert!(seen.insert(e.id), "duplicate id: {}", e.id);
                assert!(
                    matches!(e.status, "verified" | "experimental" | "avoid"),
                    "unknown status '{}' for {}",
                    e.status,
                    e.id
                );
                assert!(!e.family.is_empty() && !e.capabilities.is_empty());
            }
            let table = render();
            for e in REGISTRY {
                assert!(table.contains(e.id));
            }
        }
    }
}

pub mod vet {
    //! # vet — model certification battery (`cima vet`)
    //!
    //! One command from "found a checkpoint" to "know whether cima serves it".
    //! Each check probes a known failure class: BOS policy, eos tables, media
    //! splice contracts, nondeterminism, and numerically-corrupt-but-fluent
    //! generation. Media checks run on inputs synthesized in memory.
    //!
    //! `--caps a,b,c` declares the EXPECTED capability set: vet fails if the
    //! model doesn't announce a declared capability, and only announced ones
    //! are exercised. Without the flag, whatever the model announces is tested.
    //! A full pass earns the model its `registry.rs` row.

    use crate::models::{LoadedModel, ModelManager};
    use crate::tokenizer::ChatTurn;
    use crate::traits::Capability;
    use crate::traits::{GenOptions, Res, Tokenizer as _};

    use crate::{err, log};

    /// Architecture families with registered execution paths. Kept in sync
    /// with the loader gates in `models/transformer.rs` (safetensors) and
    /// `formats::gguf::translate_name` (GGUF).
    const PREFLIGHT_ARCHS: &[&str] = &[
        "llama",
        "mistral",
        "qwen2",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen2_audio",
        "llava",
        "gemma4",
        "gemma3n",
    ];

    /// Metadata-only certification: validate a checkpoint's complete tensor
    /// table, architecture, and tokenizer WITHOUT downloading weights.
    ///
    /// Both container formats keep their tensor tables in a header at the
    /// front of the file, reachable with HTTP Range requests — a few
    /// kilobytes to a few megabytes against multi-gigabyte checkpoints.
    /// This is the mechanism behind the `verified-by-family` registry
    /// policy: a large model whose small sibling passed the on-GPU battery
    /// is admitted when this check proves it declares the same architecture,
    /// only supported tensor dtypes, and dimensions every kernel accepts.
    ///
    /// `include` selects one quantization of a multi-quant GGUF repo, with
    /// the same substring semantics as `cima pull --include`.
    ///
    /// Returns the number of tensors validated across all files; any
    /// violation is an error naming the file, tensor, and check.
    pub fn preflight_deep(model: &str, include: Option<&str>) -> Res<usize> {
        let files = crate::hub::list_repo(model, Some(""))?;
        let ggufs: Vec<&(String, u64)> = files
            .iter()
            .filter(|(n, _)| {
                let l = n.to_ascii_lowercase();
                l.ends_with(".gguf")
                    && !l.contains("mmproj")
                    && include
                        .map(|t| l.contains(&t.to_ascii_lowercase()))
                        .unwrap_or(true)
            })
            .collect();
        let has_st = files.iter().any(|(n, _)| n.ends_with(".safetensors"));

        if !ggufs.is_empty() {
            let mut total = 0usize;
            for (name, size) in &ggufs {
                total += preflight_gguf(model, name, *size)?;
            }
            log::info(&format!(
                "preflight PASS (gguf): {} — {} tensors validated across {} file(s)",
                model,
                total,
                ggufs.len()
            ));
            return Ok(total);
        }
        if has_st {
            let total = preflight_safetensors(model, &files)?;
            log::info(&format!(
                "preflight PASS (safetensors): {} — {} tensors validated",
                model, total
            ));
            return Ok(total);
        }
        Err(err!(
            "vet",
            "'{}' contains neither .gguf nor .safetensors payloads — nothing to certify",
            model
        ))
    }

    /// GGUF: range-fetch the header (growing the window until the tensor
    /// table parses), then validate architecture, tokenizer family, and
    /// per-tensor dtype and block grain.
    fn preflight_gguf(model: &str, name: &str, size: u64) -> Res<usize> {
        use crate::formats::gguf;
        // The header holds KV metadata (which embeds the tokenizer — up to
        // tens of MB for 256k vocabularies) plus the tensor table. Start
        // small and double; a header larger than the cap is not a plausible
        // checkpoint.
        let mut want: u64 = 4 << 20;
        let cap: u64 = 256 << 20;
        let (meta, tensors) = loop {
            let to = want.min(size.saturating_sub(1));
            let buf = crate::hub::fetch_span(model, name, 0, to)?;
            match gguf::parse_header_bytes(&buf) {
                Ok(parsed) => break parsed,
                Err(_) if want < cap && to < size.saturating_sub(1) => want *= 2,
                Err(e) => {
                    return Err(err!(
                        "vet",
                        "{}: header did not parse within {} MiB: {}",
                        name,
                        want >> 20,
                        e
                    ))
                }
            }
        };
        let arch = meta
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| err!("vet", "{}: metadata is missing general.architecture", name))?
            .to_string();
        if !PREFLIGHT_ARCHS.iter().any(|a| arch.starts_with(a)) {
            return Err(err!(
                "vet",
                "{}: architecture '{}' has no registered execution path (supported: {})",
                name,
                arch,
                PREFLIGHT_ARCHS.join(", ")
            ));
        }
        if let Some(tk) = meta.get("tokenizer.ggml.model").and_then(|v| v.as_str()) {
            if !matches!(tk, "gpt2" | "llama") {
                return Err(err!(
                    "vet",
                    "{}: tokenizer family '{}' is not registered (supported: gpt2, llama)",
                    name,
                    tk
                ));
            }
        }
        let mut checked = 0usize;
        for (tname, dtype, shape) in &tensors {
            let grain = crate::traits::block_elems(*dtype);
            let row = shape.last().copied().unwrap_or(0);
            if grain > 1 && row % grain != 0 {
                return Err(err!(
                    "vet",
                    "{}: tensor '{}' row length {} is not a multiple of the {} block ({} elems) — \
                     this exact quantization cannot execute",
                    name,
                    tname,
                    row,
                    dtype.name(),
                    grain
                ));
            }
            checked += 1;
        }
        let params: usize = tensors
            .iter()
            .map(|(_, _, s)| s.iter().product::<usize>())
            .sum();
        log::info(&format!(
            "preflight {}: arch={} tensors={} params≈{:.2}B ctx={}",
            name,
            arch,
            checked,
            params as f64 / 1e9,
            meta.get(&format!("{}.context_length", arch))
                .and_then(|v| v.as_usize())
                .unwrap_or(0)
        ));
        Ok(checked)
    }

    /// safetensors: fetch config.json plus every shard's JSON header (the
    /// first 8 bytes carry the header length) and validate architecture,
    /// quantization method, and every tensor dtype.
    fn preflight_safetensors(model: &str, files: &[(String, u64)]) -> Res<usize> {
        let cfg_bytes = crate::hub::fetch_span(model, "config.json", 0, 1 << 20)?;
        let cfg = crate::json::parse(&String::from_utf8_lossy(&cfg_bytes))
            .map_err(|e| err!("vet", "config.json did not parse: {}", e))?;
        let mt = cfg
            .get("model_type")
            .and_then(crate::json::Json::as_str)
            .ok_or_else(|| err!("vet", "config.json has no model_type"))?
            .to_string();
        if !PREFLIGHT_ARCHS.contains(&mt.as_str()) {
            return Err(err!(
                "vet",
                "model_type '{}' has no registered execution path (supported: {})",
                mt,
                PREFLIGHT_ARCHS.join(", ")
            ));
        }
        let quant = cfg
            .get("quantization_config")
            .and_then(|q| q.get("quant_method"))
            .and_then(crate::json::Json::as_str)
            .map(str::to_string);
        if let Some(q) = &quant {
            if q != "bitsandbytes" {
                return Err(err!(
                    "vet",
                    "quant_method '{}' has no registered execution path (supported: bitsandbytes, or unquantized)",
                    q
                ));
            }
        }
        let mut checked = 0usize;
        for (name, size) in files.iter().filter(|(n, _)| n.ends_with(".safetensors")) {
            let head = crate::hub::fetch_span(model, name, 0, 7)?;
            if head.len() < 8 {
                return Err(err!(
                    "vet",
                    "{}: could not read the 8-byte header length",
                    name
                ));
            }
            let hlen = u64::from_le_bytes(head[..8].try_into().unwrap());
            if hlen == 0 || hlen > (256 << 20) || hlen + 8 > *size {
                return Err(err!("vet", "{}: implausible header length {}", name, hlen));
            }
            let hdr = crate::hub::fetch_span(model, name, 8, 7 + hlen)?;
            let j = crate::json::parse(&String::from_utf8_lossy(&hdr))
                .map_err(|e| err!("vet", "{}: header JSON did not parse: {}", name, e))?;
            let obj = j
                .as_obj()
                .ok_or_else(|| err!("vet", "{}: header is not a JSON object", name))?;
            for (tname, tmeta) in obj {
                if tname == "__metadata__" {
                    continue;
                }
                let dt = tmeta
                    .get("dtype")
                    .and_then(crate::json::Json::as_str)
                    .unwrap_or("?");
                let ok = matches!(dt, "F32" | "F16" | "BF16")
                    || (dt == "U8" && quant.as_deref() == Some("bitsandbytes"));
                if !ok {
                    return Err(err!(
                        "vet",
                        "{}: tensor '{}' dtype {} has no registered codec{}",
                        name,
                        tname,
                        dt,
                        if dt == "U8" {
                            " (U8 is admissible only under quantization_config.quant_method=bitsandbytes)"
                        } else {
                            ""
                        }
                    ));
                }
                checked += 1;
            }
        }
        log::info(&format!(
            "preflight {}: model_type={} quant={} tensors={}",
            model,
            mt,
            quant.as_deref().unwrap_or("none"),
            checked
        ));
        Ok(checked)
    }

    /// 64×64 solid-red 24-bit BMP, built field by field (the vision probe).
    fn red_bmp() -> Vec<u8> {
        let (w, h) = (64u32, 64u32);
        let row = (w * 3 + 3) & !3; // 4-byte aligned rows
        let data = row * h;
        let mut b = Vec::with_capacity(54 + data as usize);
        b.extend_from_slice(b"BM");
        b.extend_from_slice(&(54 + data).to_le_bytes());
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(&54u32.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&h.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&24u16.to_le_bytes());
        b.extend_from_slice(&[0; 24]); // no compression; defaultable fields
        for _ in 0..h {
            for _ in 0..w {
                b.extend_from_slice(&[0, 0, 255]); // BGR: pure red
            }
            b.extend_from_slice(&vec![0; (row - w * 3) as usize]);
        }
        b
    }

    /// 1 s, 440 Hz, 16 kHz 16-bit mono WAV (the audio-path probe).
    fn tone_wav() -> Vec<u8> {
        let n = 16_000u32;
        let mut b = Vec::with_capacity(44 + n as usize * 2);
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + n * 2).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        for v in [1u16, 1] {
            b.extend_from_slice(&v.to_le_bytes()); // PCM, mono
        }
        b.extend_from_slice(&16_000u32.to_le_bytes());
        b.extend_from_slice(&32_000u32.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(n * 2).to_le_bytes());
        for i in 0..n {
            let s = (24_000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
                as i16;
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    /// Greedy chat generation helper: render one user turn (with media markers)
    /// and return the streamed text.
    fn chat(
        lm: &mut LoadedModel,
        content: &str,
        images: &[Vec<u8>],
        audio: &[Vec<u8>],
        max: usize,
    ) -> Res<String> {
        let rendered = lm.render_chat(&[ChatTurn {
            role: "user".into(),
            content: content.into(),
            n_images: images.len(),
            n_audio: audio.len(),
        }]);
        let prepared = lm.prepare(&rendered, images, audio)?;
        let opts = GenOptions {
            temperature: 0.0,
            repeat_penalty: 1.0, // exercise the greedy device path, not the sampler
            max_tokens: max,
            ..Default::default()
        };
        let mut text = String::new();
        lm.generate(&prepared, &opts, 0.0, |p| text.push_str(p))?;
        Ok(text)
    }

    pub fn run(
        manager: &mut ModelManager,
        model: &str,
        expected: Option<Vec<Capability>>,
    ) -> Res<()> {
        use std::time::Instant;
        let (mut pass, mut fail) = (0usize, 0usize);
        let mut check = |name: &str, ok: bool, detail: String| {
            if ok {
                pass += 1
            } else {
                fail += 1
            };
            println!(
                "VET [{}] {:<22} {}",
                if ok { "ok  " } else { "FAIL" },
                name,
                detail
            );
        };

        let t0 = Instant::now();
        let lm = manager.ensure(model)?;
        let caps = lm.capabilities();
        let cap_str = caps
            .iter()
            .map(|c| format!("{:?}", c))
            .collect::<Vec<_>>()
            .join("+");
        check(
            "load",
            true,
            format!("{:.1}s, announces: {}", t0.elapsed().as_secs_f64(), cap_str),
        );

        // Declared expectations vs announcement.
        if let Some(exp) = &expected {
            let missing: Vec<_> = exp.iter().filter(|c| !caps.contains(c)).collect();
            check(
                "declared-caps",
                missing.is_empty(),
                if missing.is_empty() {
                    format!("all declared capabilities announced ({})", cap_str)
                } else {
                    format!("model does not announce: {:?}", missing)
                },
            );
        }

        // Tokenizer roundtrip (byte-fallback / metaspace failures) and the
        // BOS invariant: at most one, only at position 0.
        let pangram = "The quick brown fox: ¡año nuevo! Καλημέρα 你好 🦀";
        let ids = lm.tokenizer.encode(pangram, true);
        // Accumulate bytes and convert once: byte-level BPE legitimately
        // splits multi-byte characters across tokens, so per-token lossy
        // conversion would corrupt them (the generate loop fuses UTF-8 for
        // the same reason).
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&t| lm.tokenizer.decode_token(t).to_vec())
            .collect();
        let back = String::from_utf8_lossy(&bytes);
        check(
            "tokenizer-roundtrip",
            back.contains("fox") && back.contains("🦀"),
            format!("{} tokens", ids.len()),
        );
        if let Some(bos) = lm.tokenizer.bos() {
            let n_bos = ids.iter().filter(|&&t| t == bos).count();
            check(
                "bos-invariant",
                n_bos <= 1 && !ids.iter().skip(1).any(|&t| t == bos),
                format!(
                    "{} bos in raw encode (tokenizer adds_bos={})",
                    n_bos,
                    lm.tokenizer.adds_bos()
                ),
            );
        }

        if caps.contains(&Capability::Generate) {
            // Determinism: two greedy runs, identical output.
            let a = chat(lm, "Complete: the capital of France is", &[], &[], 24)?;
            let b = chat(lm, "Complete: the capital of France is", &[], &[], 24)?;
            check(
                "greedy-determinism",
                a == b && !a.is_empty(),
                format!("{:?}", a.chars().take(40).collect::<String>()),
            );

            // Knowledge smoke: fluent nonsense from numerical bugs fails this.
            let facts = [
                ("the capital of France", "Paris"),
                ("2 + 2 equals", "4"),
                ("the chemical symbol for water", "H2O"),
            ];
            let hits = facts
                .iter()
                .filter(|(p, w)| {
                    chat(lm, &format!("State only the answer: {}?", p), &[], &[], 24)
                        .map(|t| t.contains(w))
                        .unwrap_or(false)
                })
                .count();
            check("knowledge-smoke", hits >= 2, format!("{}/3 facts", hits));

            // EOS termination: a one-word task must stop well under budget.
            let rendered_len = {
                let t = chat(lm, "Say only: hi", &[], &[], 96)?;
                t.split_whitespace().count()
            };
            check(
                "eos-termination",
                rendered_len < 60,
                format!("{} words before stop", rendered_len),
            );
        }

        if caps.contains(&Capability::Embed) {
            match (lm.embed("a cat on a mat"), lm.embed("stock markets fell")) {
                (Ok(a), Ok(b)) => {
                    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let cos = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>() / (na * nb + 1e-9);
                    check(
                        "embed",
                        a.iter().all(|x| x.is_finite()) && na > 1.0 && na < 1e4 && cos < 0.999,
                        format!("dim={} norm={:.1} cross-cos={:.3}", a.len(), na, cos),
                    );
                }
                _ => check("embed", false, "embed() errored".into()),
            }
        }

        if caps.contains(&Capability::Vision) {
            // A solid-red frame: the answer must mention the color — the
            // cheapest probe that tower + splice produce meaning, not noise.
            // Vision regressions can keep text fluent while failing this.
            let t = chat(
                lm,
                "What is the dominant color of this image? One word.",
                &[red_bmp()],
                &[],
                16,
            )?;
            check(
                "vision-color",
                t.to_lowercase().contains("red"),
                format!("{:?}", t.trim().chars().take(40).collect::<String>()),
            );
        }

        if caps.contains(&Capability::Audio) {
            // A pure tone exercises decode->mel->conformer->splice end to end;
            // any crash/empty/runaway fails. (Content checks need speech.)
            let t = chat(
                lm,
                "Describe this audio in one sentence.",
                &[],
                &[tone_wav()],
                48,
            )?;
            check(
                "audio-path",
                !t.trim().is_empty(),
                format!("{:?}", t.trim().chars().take(50).collect::<String>()),
            );
        }

        // Performance scorecard (informational — vet certifies correctness;
        // `cima profile` judges speed against the bandwidth floor).
        let levers = lm.arch.perf_levers();
        log::info(&format!("vet perf levers: {}", levers.summary()));

        // Throughput snapshot (informational).
        let t1 = Instant::now();
        let n = chat(lm, "Count from one to thirty.", &[], &[], 32)?
            .split_whitespace()
            .count();
        log::info(&format!(
            "vet throughput snapshot: ~{:.1} words/s greedy",
            n as f64 / t1.elapsed().as_secs_f64()
        ));

        println!(
            "\nVET verdict: {} passed, {} failed — {}",
            pass,
            fail,
            if fail == 0 {
                "eligible for registry.rs (add the row in your PR)"
            } else {
                "NOT eligible; fix the failures above"
            }
        );
        if fail == 0 {
            Ok(())
        } else {
            Err(err!("vet", "{} check(s) failed", fail))
        }
    }

    /// Parse `--caps generate,vision,audio,embed`.
    pub fn parse_caps(s: &str) -> Res<Vec<Capability>> {
        s.split(',')
            .map(|c| match c.trim().to_lowercase().as_str() {
                "generate" => Ok(Capability::Generate),
                "embed" => Ok(Capability::Embed),
                "vision" => Ok(Capability::Vision),
                "audio" => Ok(Capability::Audio),
                other => Err(err!(
                    "cli",
                    "unknown capability '{}' (expected generate|embed|vision|audio)",
                    other
                )),
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The synthetic probes must be decodable by our own media layer, and
        /// the caps parser must reject garbage — otherwise vet failures would
        /// blame the model for harness bugs.
        #[test]
        fn probes_and_caps_parse() {
            let m = crate::media::MediaRegistry::standard();
            let img = m
                .decode_image(&red_bmp(), 64, 64, [0.0; 3], [1.0; 3])
                .expect("red bmp decodes");
            // identity-normalized planar CHW: R plane ~1.0, G plane ~0.0.
            assert!(
                img.data[0] > 0.9 && img.data[64 * 64].abs() < 0.1,
                "red plane high, green plane zero"
            );
            let pcm = m
                .decode_audio(&tone_wav(), 16_000)
                .expect("tone wav decodes");
            assert_eq!(pcm.samples.len(), 16_000);
            assert!(
                pcm.samples.iter().any(|s| s.abs() > 0.5),
                "tone has amplitude"
            );
            assert_eq!(parse_caps("generate, vision").unwrap().len(), 2);
            assert!(parse_caps("generate,sorcery").is_err());
        }
    }
}

#[cfg(test)]
mod quant_tag_tests {
    use super::quant_tag;
    #[test]
    fn tags_and_shards() {
        assert_eq!(quant_tag("gemma-4-E4B-it-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_tag("gemma-4-E4B-it-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL");
        assert_eq!(quant_tag("model-IQ4_XS.gguf"), "IQ4_XS");
        assert_eq!(
            quant_tag("qwen2.5-7b-instruct-q8_0-00001-of-00002.gguf"),
            "q8_0"
        );
        assert_eq!(quant_tag("gemma-4-E4B-it-Q8_0-00002-of-00002.gguf"), "Q8_0");
        assert_eq!(quant_tag("plainname.gguf"), "plainname");
    }
}

#[cfg(test)]
mod selector_tests {
    use super::{list_local_caps, local_dir, split_selector};

    // Serialize the env-mutating tests in this module.
    // Serialize model-dir tests: CIMA_MODELS_DIR is process-global, so
    // parallel tests would otherwise clobber each other's fixtures (one
    // test's remove_dir_all wipes another's freshly-written dir, and
    // list_local_caps sees an empty tree). A mutex makes each body's
    // set-var / populate / list / clean sequence atomic.
    fn models_dir_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with_models_dir(f: impl FnOnce(&std::path::Path)) {
        let _guard = models_dir_lock();
        let base = std::env::temp_dir().join(format!(
            "cima-ll-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let prev = std::env::var("CIMA_MODELS_DIR").ok();
        std::env::set_var("CIMA_MODELS_DIR", &base);
        f(&base);
        match prev {
            Some(p) => std::env::set_var("CIMA_MODELS_DIR", p),
            None => std::env::remove_var("CIMA_MODELS_DIR"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn list_local_does_not_double_the_quant_tag() {
        // Regression: a GGUF pulled as ORG/REPO:q8_0 lands in
        // ORG__REPO@q8_0/; list_local must report a single :q8_0, not
        // ...GGUF:q8_0:q8_0 (the dir tag AND the filename tag both applied).
        with_models_dir(|base| {
            let dir = base.join("Qwen__Qwen2.5-0.5B-Instruct-GGUF@q8_0");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("qwen2.5-0.5b-instruct-q8_0.gguf"), b"x").unwrap();
            let names: Vec<String> = list_local_caps().into_iter().map(|(n, ..)| n).collect();
            assert!(
                names
                    .iter()
                    .any(|n| n == "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0"),
                "expected single-tag name, got {names:?}"
            );
            assert!(
                !names.iter().any(|n| n.contains(":q8_0:q8_0")),
                "tag was doubled: {names:?}"
            );
        });
    }

    #[test]
    fn gguf_text_model_advertises_embed() {
        // GGUF models embed via the same format-agnostic mean-pool path as
        // safetensors; list_local_caps must advertise "embed" so /api/tags
        // and clients surface it (a plain-text GGUF, no mmproj).
        with_models_dir(|base| {
            let dir = base.join("Qwen__Qwen2.5-0.5B-Instruct-GGUF@q8_0");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("qwen2.5-0.5b-instruct-q8_0.gguf"), b"x").unwrap();
            let caps: &str = list_local_caps()
                .into_iter()
                .find(|(n, ..)| n == "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0")
                .map(|(_, _, _, c)| c)
                .expect("gguf row present");
            assert!(caps.contains("embed"), "gguf caps missing embed: {caps}");
            assert!(caps.contains("text"), "gguf caps missing text: {caps}");
        });
    }

    #[test]
    fn list_local_collapses_same_repo_tag_from_two_dirs() {
        // Regression: an older untagged pull (ORG__REPO/) and a later tagged
        // pull (ORG__REPO@TAG/) can both hold the same quant .gguf and both
        // resolve to ORG/REPO:TAG — /api/tags showed the model twice. The
        // list must collapse them to a single row (newest kept).
        with_models_dir(|base| {
            let untagged = base.join("unsloth__gemma-4-E4B-it-GGUF");
            let tagged = base.join("unsloth__gemma-4-E4B-it-GGUF@Q4_K_M");
            std::fs::create_dir_all(&untagged).unwrap();
            std::fs::create_dir_all(&tagged).unwrap();
            std::fs::write(untagged.join("gemma-4-E4B-it-Q4_K_M.gguf"), b"xx").unwrap();
            std::fs::write(tagged.join("gemma-4-E4B-it-Q4_K_M.gguf"), b"xx").unwrap();
            let names: Vec<String> = list_local_caps().into_iter().map(|(n, ..)| n).collect();
            let hits = names
                .iter()
                .filter(|n| *n == "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M")
                .count();
            assert_eq!(hits, 1, "duplicate repo:tag not collapsed: {names:?}");
        });
    }

    #[test]
    fn gguf_tag_is_a_filter_not_a_revision() {
        // The reported bug: `:q8_0` must split into an include filter, never
        // reach a resolve/{revision} URL (which 404s).
        let (repo, tag) = split_selector("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0");
        assert_eq!(repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(tag, Some("q8_0"));
    }

    #[test]
    fn bare_repo_has_no_tag() {
        let (repo, tag) = split_selector("Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(repo, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(tag, None);
    }

    #[test]
    fn local_dir_keeps_tag_distinct() {
        // Two quantizations of one repo must land in distinct directories.
        let a = local_dir("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0");
        let b = local_dir("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m");
        assert_ne!(a, b);
        assert!(a.to_string_lossy().contains("q8_0"));
    }

    fn tmp_root() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "cima-islocal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn is_local_recognizes_both_formats_and_tags() {
        use super::is_local_in;
        let root = tmp_root();

        // Absent → false.
        assert!(!is_local_in(&root, "Org/Missing"));

        // safetensors: needs config.json AND a shard.
        let st = root.join("Org__STModel");
        std::fs::create_dir_all(&st).unwrap();
        std::fs::write(st.join("config.json"), "{}").unwrap();
        assert!(
            !is_local_in(&root, "Org/STModel"),
            "config alone is not enough"
        );
        std::fs::write(st.join("model.safetensors"), b"x").unwrap();
        assert!(is_local_in(&root, "Org/STModel"));

        // GGUF: a runnable quant, tag-filtered. Dir uses ':' -> '@'.
        let gg = root.join("Org__GG@q8_0");
        std::fs::create_dir_all(&gg).unwrap();
        std::fs::write(gg.join("model-Q8_0.gguf"), b"x").unwrap();
        assert!(is_local_in(&root, "Org/GG:q8_0"));
        assert!(!is_local_in(&root, "Org/GG:q4_k_m"), "wrong tag must miss");

        // mmproj sidecar alone must NOT count as a runnable model.
        let mm = root.join("Org__MMonly");
        std::fs::create_dir_all(&mm).unwrap();
        std::fs::write(mm.join("mmproj-model.gguf"), b"x").unwrap();
        assert!(!is_local_in(&root, "Org/MMonly"));

        std::fs::remove_dir_all(&root).ok();
    }
}
