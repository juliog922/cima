//! `cima` — the CLI. A thin dispatcher over the [`cima`](crate) library:
//! parses arguments, initializes the CUDA context when a command needs the
//! GPU, and hands off. All engine, protocol, and client logic lives in the
//! library so the binary adds no behavior of its own.

use cima::err;
use cima::{api, cuda, hub, json, log, models, registry, selftest, vet};

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use cima::cuda::CudaCtx;
use cima::models::ModelManager;
use cima::queue::GpuQueue;
use cima::tokenizer::ChatTurn;
use cima::traits::{GenOptions, Res};

/// Ollama-style usage text printed on `help`, no args, or unknown commands.
const USAGE: &str = "\
cima — minimalist CUDA inference engine (Ollama-compatible)

Usage:
  cima serve [HOST] [PORT]        Start the API server (default 127.0.0.1:11435)
  cima run <MODEL> [PROMPT]       Run a model (interactive REPL without PROMPT)
  cima embed <MODEL> <TEXT>       Pooled embedding vector (--out FILE for raw f32)
  cima available                  Curated registry of vetted, pull-ready models
  cima profile <MODEL>            Decode-step anatomy vs the bandwidth floor
  cima selftest [gguf|lm]         GPU numerical self-tests (no downloads)
  cima audio-map <GGUF> <ST>      Recover gguf audio-tensor names by cosine
                                     match against the original safetensors
  cima vet <ORG/REPO> [--caps L]  Pull (if needed) + certification battery;
                                     --caps generate,vision,audio,embed declares
                                     the EXPECTED set (fails if not announced)
  cima pull <MODEL[:QUANT]> [flags]       Pull a model from the Hugging Face Hub
                                     (preflight-validates config.json first)
       --background | -b             daemonize the download
       --include <substr>            only weight files whose name contains
                                     <substr> (pick one quant of a multi-quant
                                     repo, e.g. --include Q4_K_M)
       --force                       download even if the preflight gate
                                     rejects the architecture
  cima check <MODEL>              Validate a Hub model's architecture
                                     against this engine WITHOUT downloading
                                     weights (fetches config.json only)
  cima list                       List locally pulled models
  cima rm <MODEL[:TAG]>           Delete a local model (with :TAG, remove just
                                     that quant's gguf; tag matching identical
                                     to `run`)
  cima stress <MODEL> [--requests N] [--concurrency C]
                                  Concurrent load against a running server
  cima logs [-f] [--level L,..]   Read the shared-memory log ring (--json,
                                  --table for METRIC profiling, -n N)
  cima ps                         Resident model, VRAM, keep-alive expiry
  cima ready [MODEL..] [--wait]   Startup gate: exit 0 when healthy and every
                                  MODEL is pulled (--wait polls; --timeout S)
  cima stop <MODEL>               Release a model from VRAM now (keep_alive 0)
  cima show <MODEL>               Show model configuration
  cima rm <MODEL>                 Remove a pulled model from disk
  cima help                       Show this help

Environment:
  CIMA_GPU      GPU ordinal to bind (default 0)
  CIMA_HOST     Bind host for `serve` (default 127.0.0.1)
  CIMA_PORT     Port for `serve` and for ps/stop/rm (default 11435; ollama keeps 11434)

Examples:
  cima pull Qwen/Qwen2.5-0.5B-Instruct
  cima run Qwen/Qwen2.5-0.5B-Instruct \"Why is the sky blue?\"
  cima serve 0.0.0.0 11435
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(&args) {
        Ok(()) => 0,
        Err(_) => 1, // already logged on construction (see EngineError::new)
    };
    std::process::exit(code);
}

/// Top-level command router. Every subcommand returns a [`Res`] so failures
/// surface as a single, precise error line on stderr with exit code 1.
fn dispatch(args: &[String]) -> Res<()> {
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "serve" => cmd_serve(args.get(1).cloned(), args.get(2).cloned()),
        "logs" => cmd_logs(args.get(1..).unwrap_or(&[])),
        "stress" => cmd_stress(args.get(1..).unwrap_or(&[])),
        "embed" => {
            // cima embed MODEL "text" [--out /path.bin]
            // Prints dim/norm/head; --out writes [u32 dim][f32 LE...] for A/B harnesses.
            let model = args.get(1).cloned().unwrap_or_default();
            let text = args.get(2).cloned().unwrap_or_default();
            if model.is_empty() || text.is_empty() {
                eprintln!("usage: cima embed MODEL \"text\" [--out /path.bin]");
                std::process::exit(2);
            }
            let out_path = args
                .iter()
                .position(|a| a == "--out")
                .and_then(|i| args.get(i + 1))
                .cloned();
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            let mut manager = models::ModelManager::new(ctx);
            let lm = manager.ensure(&model)?;
            let v = lm.embed(&text)?;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let head: Vec<String> = v.iter().take(8).map(|x| format!("{:.5}", x)).collect();
            println!(
                "embedding: dim={} norm={:.4} head=[{}]",
                v.len(),
                norm,
                head.join(", ")
            );
            if let Some(p) = out_path {
                let mut bytes = Vec::with_capacity(4 + v.len() * 4);
                bytes.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in &v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(&p, bytes).map_err(|e| err!("io", "cannot write '{}': {}", p, e))?;
                println!("written to {}", p);
            }
            Ok(())
        }
        "bench" => {
            let model = args
                .get(1)
                .ok_or_else(|| {
                    cima::err!("cli", "usage: cima bench <MODEL> [-n TOKENS] [--iters N]")
                })?
                .clone();
            let mut n = 128usize;
            let mut iters = 5usize;
            let chat = args.iter().any(|a| a == "--chat");
            let rest = args.get(2..).unwrap_or(&[]);
            let mut i = 0;
            while i + 1 < rest.len() + 1 {
                match rest.get(i).map(String::as_str) {
                    Some("-n") | Some("--tokens") if i + 1 < rest.len() => {
                        n = rest[i + 1]
                            .parse()
                            .map_err(|_| cima::err!("cli", "-n expects an integer"))?;
                        i += 2;
                    }
                    Some("--iters") if i + 1 < rest.len() => {
                        iters = rest[i + 1]
                            .parse()
                            .map_err(|_| cima::err!("cli", "--iters expects an integer"))?;
                        i += 2;
                    }
                    Some(_) => i += 1,
                    None => break,
                }
            }
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            let mut manager = ModelManager::new(ctx);
            selftest::run_bench(&mut manager, &model, n, iters, chat)
        }
        // `cima selftest` — full battery; `selftest gguf` / `selftest lm`
        // run one suite. Everything is synthetic: no model download needed.
        "selftest" => {
            let which = args.get(1).map(String::as_str);
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            if which.is_none() || which == Some("gemm") {
                // Native f16 GEMM (the cuBLAS-independence path) vs the
                // f64 host reference — gate for the slim container image.
                selftest::run_gemm(ctx.clone())?;
            }
            if which.is_none() || which == Some("gguf") {
                // Device GGUF kernels (dequant slabs, dp4a GEMVs, embedding
                // gather) vs the tested host decoders.
                selftest::run_gguf_kernels(ctx.clone())?;
            }
            if which.is_none() || which == Some("lm") {
                // GPU integration suite on a synthetic in-memory checkpoint —
                // exercises load→prefill→decode→sample.
                selftest::run_lm(ctx)?;
            }
            Ok(())
        }
        "vision-selftest" => {
            let model = args
                .get(1)
                .ok_or_else(|| cima::err!("cli", "usage: cima vision-selftest <MODEL>"))?;
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            let mut manager = ModelManager::new(ctx);
            let lm = manager.ensure(model)?;
            lm.vision_selftest()
        }
        "audio-map" => {
            // Content-based audio mapping recovery: match gguf mmproj audio
            // tensors to the original safetensors tower by cosine — settles
            // the shape-ambiguous name assignments mechanically.
            let g = args.get(1).ok_or_else(|| {
                cima::err!(
                    "cli",
                    "usage: cima audio-map <GGUF_MODEL[:TAG]> <ORIGINAL_MODEL> [PREFIX]"
                )
            })?;
            let st = args.get(2).ok_or_else(|| {
                cima::err!(
                    "cli",
                    "usage: cima audio-map <GGUF_MODEL[:TAG]> <ORIGINAL_MODEL> [PREFIX]"
                )
            })?;
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            cima::selftest::run_tensor_map(ctx, g, st, args.get(3).map(|s| s.as_str()))
        }
        "run" => {
            if args.iter().any(|a| a == "--help" || a == "-h") {
                println!("usage: cima run <MODEL> [PROMPT] [--image FILE]... [--audio FILE]... [--PARAM VALUE]...");
                println!(
                    "       without PROMPT: interactive REPL (/set PARAM VALUE, /clear, /bye)\n"
                );
                print!("{}", cima::traits::GenOptions::render_help());
                return Ok(());
            }
            let rest = args.get(2..).unwrap_or(&[]);
            let mut opts = GenOptions::default();
            let mut prefix: Option<String> = None;
            let mut images = Vec::new();
            let mut audio = Vec::new();
            let mut words: Vec<String> = Vec::new();
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--image" if i + 1 < rest.len() => {
                        images.push(rest[i + 1].clone());
                        i += 2;
                    }
                    "--audio" if i + 1 < rest.len() => {
                        audio.push(rest[i + 1].clone());
                        i += 2;
                    }
                    "--prefix" if i + 1 < rest.len() => {
                        prefix = Some(rest[i + 1].clone());
                        i += 2;
                    }
                    // Every generation parameter routes through the option
                    // table (GenOptions::set) — new params need no CLI code.
                    flag if flag.starts_with("--") && i + 1 < rest.len() => {
                        opts.set(flag, &rest[i + 1])?;
                        i += 2;
                    }
                    _ => {
                        words.push(rest[i].clone());
                        i += 1;
                    }
                }
            }
            cmd_run(
                args.get(1),
                &words.join(" "),
                &images,
                &audio,
                opts,
                prefix.as_deref(),
            )
        }
        "pull" => {
            // `cima pull ORG/REPO:Q4_K_XL` — the tag selects which .gguf
            // file(s) to fetch from a multi-quantization repo.
            let mut include = flag_value(args, "--include");
            let mut args = args.to_vec();
            if let Some(name) = args.get(1) {
                if let Some((repo, tag)) = name.split_once(':') {
                    if include.is_none() {
                        include = Some(tag.to_string());
                    }
                    args[1] = repo.to_string();
                }
            }
            let args = &args;
            cmd_pull(
                args.get(1),
                args.iter().any(|a| a == "--background" || a == "-b"),
                include,
                args.iter().any(|a| a == "--force"),
            )
        }
        "check" => cmd_check(args.get(1), args.iter().any(|a| a == "--meta")),
        "list" | "ls" => cmd_list(),
        "available" => {
            print!("{}", registry::render());
            Ok(())
        }
        "profile" => {
            let model = args
                .get(1)
                .ok_or_else(|| cima::err!("cli", "usage: cima profile MODEL"))?
                .clone();
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            let mut manager = models::ModelManager::new(ctx);
            selftest::run_profile(&mut manager, &model)
        }
        "vet" => {
            let model = args
                .get(1)
                .ok_or_else(|| {
                    cima::err!(
                        "cli",
                        "usage: cima vet ORG/REPO[:TAG] [--preflight] [--caps generate,vision,audio,embed]"
                    )
                })?
                .clone();
            // Metadata-only certification: validate the complete tensor
            // table, architecture, and tokenizer over HTTP Range requests —
            // no weight download, no GPU. This is the admission check for
            // large family members whose small sibling passed the full
            // battery ("verified-by-family" in the registry).
            if args.iter().any(|a| a == "--preflight") {
                let (repo, tag) = match model.split_once(':') {
                    Some((r, t)) => (r.to_string(), Some(t.to_string())),
                    None => (model.clone(), None),
                };
                let n = vet::preflight_deep(&repo, tag.as_deref())?;
                println!(
                    "preflight PASS: {} ({} tensors validated, no bytes of weights downloaded)",
                    model, n
                );
                return Ok(());
            }
            let expected = args
                .iter()
                .position(|a| a == "--caps")
                .and_then(|i| args.get(i + 1))
                .map(|s| vet::parse_caps(s))
                .transpose()?;
            // Pull first when absent — vet is the one-command path from
            // "found a checkpoint" to "know whether cima serves it".
            // Use the shared locality test: it knows both directory
            // layouts and verifies the quant tag against the files present.
            // A bespoke check here re-downloaded models that were already on
            // disk under the other layout.
            if !hub::is_local(&model) {
                println!("model not local — pulling first…");
                hub::pull(&model, false, None)?;
            }
            let ctx = Arc::new(CudaCtx::init(gpu_index())?);
            let mut manager = models::ModelManager::new(ctx);
            vet::run(&mut manager, &model, expected)
        }
        "ps" => cmd_ps(),
        "ready" => {
            let models: Vec<String> = args[1..]
                .iter()
                .filter(|a| !a.starts_with('-'))
                .cloned()
                .collect();
            let wait = args.iter().any(|a| a == "--wait");
            let timeout = flag_value(args, "--timeout")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(300);
            cmd_ready(&models, wait, timeout)
        }
        "stop" => {
            let model = args
                .get(1)
                .ok_or_else(|| cima::err!("cli", "usage: cima stop MODEL"))?;
            match api::client::Client::local().stop(model) {
                Ok(_) => println!("released {}", model),
                Err(_) => println!("no cima server running — nothing is resident to release"),
            }
            Ok(())
        }
        "show" => cmd_show(args.get(1)),
        "rm" | "remove" => cmd_rm(args.get(1)),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        "--version" | "-v" | "version" => {
            println!(
                "cima {} (Ollama-compatible API {})",
                env!("CARGO_PKG_VERSION"),
                api::server::OLLAMA_COMPAT_VERSION
            );
            Ok(())
        }
        other => {
            // Usage goes to stderr and the exit code is nonzero: a typo'd
            // command in a script must fail the script, not sail past.
            eprint!("{USAGE}");
            Err(cima::err!("cli", "unknown command '{}'", other))
        }
    }
}

/// GPU ordinal from `CIMA_GPU` (default 0).
fn gpu_index() -> u32 {
    std::env::var("CIMA_GPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// `cima serve [HOST] [PORT]` — bring up the CUDA context, model manager
/// and FIFO GPU queue, then block inside the HTTP accept loop forever.
fn cmd_serve(host: Option<String>, port: Option<String>) -> Res<()> {
    let host = host
        .or_else(|| std::env::var("CIMA_HOST").ok())
        .unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = port
        .or_else(|| std::env::var("CIMA_PORT").ok())
        .and_then(|p| p.parse().ok())
        .unwrap_or(api::protocol::DEFAULT_PORT);

    warn_unknown_config();
    let ctx = Arc::new(CudaCtx::init(gpu_index())?);
    let snap = ctx.snapshot();
    // Startup banner: everything a support thread asks for first, in one
    // block — version, device, and the resolved value of every public
    // configuration variable.
    log::info(&format!(
        "cima {} ({}) starting | bind {}:{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("CIMA_GIT_SHA").unwrap_or("unknown"),
        host,
        port
    ));
    log::info(&format!(
        "serve: GPU{} online | VRAM {} free / {} total | util {}%",
        gpu_index(),
        cuda::fmt_bytes(snap.vram_free as usize),
        cuda::fmt_bytes(snap.vram_total as usize),
        snap.util_gpu
    ));
    let models_dir = hub::models_dir();
    let model_count = std::fs::read_dir(&models_dir)
        .map(|r| r.count())
        .unwrap_or(0);
    log::info(&format!(
        "config: models_dir={} ({} entries) | keep_alive={} | max_seq={} | log={} format={} | hf_token={}",
        models_dir.display(),
        model_count,
        std::env::var("CIMA_KEEP_ALIVE").unwrap_or_else(|_| "5m (default)".into()),
        std::env::var("CIMA_MAX_SEQ").unwrap_or_else(|_| "model default".into()),
        std::env::var("CIMA_LOG").unwrap_or_else(|_| "default".into()),
        std::env::var("CIMA_LOG_FORMAT").unwrap_or_else(|_| "text".into()),
        if std::env::var("HF_TOKEN").map(|t| !t.is_empty()).unwrap_or(false) { "set" } else { "unset" },
    ));

    let startup_models = api::server::startup_models_from_env();
    if !startup_models.is_empty() {
        log::info(&format!(
            "config: pull-at-startup = [{}] (server reports ready only once all are present)",
            startup_models.join(", ")
        ));
    }

    let server = api::Server {
        manager: Mutex::new(ModelManager::new(ctx)),
        queue: GpuQueue::new(),
        startup: std::sync::Arc::new(api::server::Startup {
            required: startup_models,
            ..Default::default()
        }),
    };
    println!(
        "logs -> {} (follow with `cima logs -f`)",
        cima::shmlog::default_path().display()
    );
    api::serve(server, &host, port)
}

/// Warn on `CIMA_*` variables the binary does not recognize. A typo'd
/// variable silently doing nothing is the same failure class as an
/// unsupported model: a surprise the operator finds in production.
fn warn_unknown_config() {
    // Public configuration plus every diagnostic lever, per
    // docs/configuration.md. Extend this list when adding a variable.
    const KNOWN: &[&str] = &[
        "CIMA_HOST",
        "CIMA_PORT",
        "CIMA_MODELS_DIR",
        "CIMA_PULL_AT_STARTUP",
        "CIMA_KEEP_ALIVE",
        "CIMA_MAX_SEQ",
        "CIMA_MAX_QUEUE",
        "CIMA_GPU",
        "CIMA_LOG",
        "CIMA_LOG_FORMAT",
        "CIMA_OLLAMA_VERSION",
        "CIMA_LOG_SILENT",
        "CIMA_SHM",
        "CIMA_PIN",
        "CIMA_NO_CUBLAS",
        "CIMA_NO_GRAPH",
        "CIMA_NO_PIPELINE",
        "CIMA_NO_INCR",
        "CIMA_INCR_CHECK",
        "CIMA_VRAM_TRACE",
    ];
    for (key, _) in std::env::vars() {
        if key.starts_with("CIMA_")
            && !KNOWN.contains(&key.as_str())
            && !key.starts_with("CIMA_G4_")
            && !key.starts_with("CIMA_DUMP_")
            && !key.starts_with("CIMA_TRACE_")
        {
            log::warn(&format!(
                "environment variable '{}' is not a recognized cima setting — check docs/configuration.md for the supported list",
                key
            ));
        }
    }
}

/// `cima run <MODEL> [PROMPT]` — load the model on this process's GPU
/// context and either answer a single prompt or drop into an interactive
/// REPL that streams tokens as they are sampled.
fn cmd_run(
    model: Option<&String>,
    prompt: &str,
    images: &[String],
    audio: &[String],
    mut opts: GenOptions,
    prefix: Option<&str>,
) -> Res<()> {
    let model = model.ok_or_else(|| {
        cima::err!(
            "cli",
            "usage: cima run <MODEL> [PROMPT] [--image FILE]... [--audio FILE]... [--temp T] [--top-p P] [--top-k K] [--seed S] [--repeat-penalty R]"
        )
    })?;
    let ctx = Arc::new(CudaCtx::init(gpu_index())?);
    let mut manager = ModelManager::new(ctx);
    let queue = GpuQueue::new();

    let read_all = |paths: &[String]| -> Res<Vec<Vec<u8>>> {
        paths
            .iter()
            .map(|p| {
                std::fs::read(p)
                    .map_err(|e| cima::err!("cli", "cannot read media file '{}': {}", p, e))
            })
            .collect()
    };
    let img_bytes = read_all(images)?;
    let aud_bytes = read_all(audio)?;

    // One-shot mode: a prompt was given on the command line.
    if !prompt.trim().is_empty() {
        return run_once(
            &mut manager,
            &queue,
            model,
            prompt.trim(),
            &img_bytes,
            &aud_bytes,
            &opts,
            prefix,
        );
    }
    if !img_bytes.is_empty() || !aud_bytes.is_empty() {
        return Err(cima::err!(
            "cli",
            "--image/--audio require a one-shot PROMPT (interactive media is not wired yet)"
        ));
    }

    // Interactive REPL (Ollama-style). Multi-turn chat history is kept so the
    // chat template sees the full conversation each turn.
    {
        // Eagerly load before the first prompt so VRAM errors surface now.
        let permit = queue.acquire();
        eprintln!("loading {} …", model);
        manager.ensure(model)?;
        drop(permit);
    }
    println!(">>> Send a message (/bye to exit, /clear to reset history)");
    let stdin = std::io::stdin();
    let mut history: Vec<ChatTurn> = Vec::new();
    loop {
        print!(">>> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| cima::err!("cli", "stdin: {}", e))?
            == 0
        {
            break; // EOF
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/bye" | "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                println!("Cleared session context");
                continue;
            }
            "/help" | "/?" => {
                print!("{}", cima::traits::GenOptions::render_help());
                println!("/set PARAM VALUE   apply one of the above    /clear  reset history    /bye  exit");
                continue;
            }
            _ if line.starts_with("/set ") => {
                let mut it = line[5..].split_whitespace();
                match (it.next(), it.next()) {
                    (Some(p), Some(v)) => {
                        match opts.set(&format!("--{}", p.replace('_', "-")), v) {
                            Ok(_) => println!("set {} = {}", p, v),
                            Err(e) => println!("{}", e),
                        }
                    }
                    _ => println!("usage: /set PARAM VALUE"),
                }
                continue;
            }
            _ => {}
        }

        history.push(ChatTurn {
            role: "user".into(),
            content: line.to_string(),
            n_images: 0,
            n_audio: 0,
        });
        let permit = queue.acquire();
        let wait = permit.wait_ms;
        let lm = manager.ensure(model)?;
        let rendered = lm.render_chat(&history);
        if std::env::var("CIMA_DUMP_RENDER").is_ok() {
            eprintln!("render({} chars): {:?}", rendered.len(), rendered);
        }
        let prepared = lm.prepare_chat(&rendered)?;
        let mut reply = String::new();
        let stats = lm.generate(&prepared, &opts, wait, |tok| {
            reply.push_str(tok);
            print!("{tok}");
            std::io::stdout().flush().ok();
        })?;
        drop(permit);
        lm.note_session_text(format!("{}{}", rendered, reply));
        history.push(ChatTurn {
            role: "assistant".into(),
            content: reply,
            n_images: 0,
            n_audio: 0,
        });
        println!();
        eprintln!(
            "[{} prompt tok | {} gen tok | ttft {:.0} ms | {:.1} tok/s]",
            stats.prompt_tokens, stats.gen_tokens, stats.ttft_ms, stats.tok_per_s
        );
    }
    Ok(())
}

/// Single prompt → single streamed answer, used by `run <MODEL> <PROMPT>`.
fn run_once(
    manager: &mut ModelManager,
    queue: &GpuQueue,
    model: &str,
    prompt: &str,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
    opts: &GenOptions,
    prefix: Option<&str>,
) -> Res<()> {
    let permit = queue.acquire();
    eprintln!("loading {} …", model);
    let lm = manager.ensure(model)?;
    let turns = [ChatTurn {
        role: "user".into(),
        content: prompt.to_string(),
        n_images: images.len(),
        n_audio: audio.len(),
    }];
    let mut rendered = lm.render_chat(&turns);
    // Teacher forcing: continue from a forced assistant prefix — the model
    // generates as if it had already produced `prefix` itself. Diagnostic for
    // separating state quality from sampling-trajectory luck.
    if let Some(p) = prefix {
        rendered.push_str(p);
    }
    let prepared = lm.prepare(&rendered, images, audio)?;
    let stats = lm.generate(&prepared, opts, permit.wait_ms, |tok| {
        print!("{tok}");
        std::io::stdout().flush().ok();
    })?;
    println!();
    eprintln!(
        "[{} prompt tok | {} gen tok | ttft {:.0} ms | {:.1} tok/s]",
        stats.prompt_tokens, stats.gen_tokens, stats.ttft_ms, stats.tok_per_s
    );
    Ok(())
}

/// Value of `--flag <value>` style arguments, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Exact KV bytes for a layer-heterogeneous gemma4 checkpoint, mirroring the
/// per-layer allocation in `models::gemma4`.
///
/// Returns None when this is not gemma4 or no config.json sits beside the
/// weights, in which case the caller keeps the uniform estimate. Reading
/// config.json rather than GGUF metadata is deliberate: the GGUF stores
/// `head_count_kv` as a per-layer array that `meta_usize` cannot decode, and
/// carries no `head_count_kv_swa` at all, so the metadata alone cannot
/// describe a heterogeneous model.
fn gemma4_kv_bytes(weights_path: &std::path::Path, cfg: &models::ModelConfig) -> Option<usize> {
    use cima::models::gemma4::{G4Config, LayerType};
    if cfg.model_type != "gemma4" {
        return None;
    }
    let raw = std::fs::read_to_string(weights_path.parent()?.join("config.json")).ok()?;
    let t = G4Config::parse(&json::parse(&raw).ok()?).ok()?.text;
    let first_shared = t.n_layers.saturating_sub(t.n_kv_shared);
    let mut bytes = 0usize;
    for (i, ty) in t.layer_types.iter().enumerate() {
        if i >= first_shared {
            continue; // shared layers read another layer's cache
        }
        let (d, kvh) = match ty {
            LayerType::Sliding => (t.head_dim, t.n_kv_heads),
            LayerType::Full => (t.global_head_dim, t.n_global_kv_heads),
        };
        bytes += 2 * kvh * t.max_seq * d * 2; // K and V, f16
    }
    Some(bytes)
}

/// Bytes of mmproj tower sidecars beside `weights_path`. They load with the
/// LM, so leaving them out understates the device footprint.
fn mmproj_bytes(weights_path: &std::path::Path) -> usize {
    weights_path
        .parent()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().to_ascii_lowercase();
                    n.ends_with(".gguf") && n.contains("mmproj")
                })
                .filter_map(|e| e.metadata().ok().map(|m| m.len() as usize))
                .sum()
        })
        .unwrap_or(0)
}

/// Shared preflight: download only config.json and run the full architecture
/// validation gate. Returns the parsed config on success so callers can print
/// a verdict without re-reading.
fn preflight(model: &str) -> Res<models::ModelConfig> {
    let dir = hub::pull_config(model)?;
    let cfg = models::ModelConfig::load(&dir)?;
    // ModelConfig::load validates the generic surface (model_type, dtypes).
    // Family-specific gates live in the family's own parser, and until now
    // they only ran at load time — so `cima check` could green-light a
    // checkpoint the loader would refuse, after a multi-gigabyte pull.
    if cfg.model_type == "gemma4" {
        let raw = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| cima::err!("cli", "reading config.json: {}", e))?;
        let j = json::parse(&raw).map_err(|e| cima::err!("cli", "config.json: {}", e))?;
        models::gemma4::G4Config::parse(&j)?;
    }
    Ok(cfg)
}

/// `cima check <MODEL>` — answer "will this run?" without downloading
/// weights: fetch config.json, run the semantic architecture gate, the quant
/// gate, and report geometry + an estimated VRAM footprint from the repo's
/// published shard sizes.
fn cmd_check(model: Option<&String>, meta: bool) -> Res<()> {
    let model = model.ok_or_else(|| cima::err!("cli", "usage: cima check <MODEL> [--meta]"))?;
    // GGUF names carry an `:TAG` quantization selector — route them to
    // the format-aware path (list_repo would parse ':' as a git revision).
    if let Some((repo, tag)) = model.split_once(':') {
        return check_gguf(repo, Some(tag), meta);
    }
    {
        // Tagless name, but a local GGUF snapshot (or a -GGUF repo): same path.
        let dir = hub::local_dir(model);
        let local_gguf = std::fs::read_dir(&dir)
            .map(|d| {
                d.flatten()
                    .any(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
            })
            .unwrap_or(false);
        if local_gguf {
            return check_gguf(model, None, meta);
        }
    }
    // Local-first: an already-pulled snapshot answers from disk — the Hub
    // is only consulted for models we don't have (and Hub ids need the
    // full ORG/REPO form, while local names are whatever was pulled).
    let local = hub::local_dir(model);
    // Never materialize anything under ./models for a read-only check: a
    // folder there makes the model look pulled. Remote checks fetch the
    // metadata into a temp dir that is removed when this function returns.
    struct TmpDir(std::path::PathBuf, bool);
    impl Drop for TmpDir {
        fn drop(&mut self) {
            if self.1 {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    let tmp = if local.join("config.json").is_file() {
        println!("(already pulled — checking the local snapshot)");
        TmpDir(local, false)
    } else {
        let d = std::env::temp_dir().join(format!(
            "cima-check-{}-{}",
            std::process::id(),
            model.replace('/', "__")
        ));
        TmpDir(hub::pull_config_to(model, d)?, true)
    };
    let dir = tmp.0.clone();
    let cfg = models::ModelConfig::load(&dir)?; // incompatibility surfaces here, precisely
                                                // Shard sizes: from local files when pulled, from the Hub API otherwise.
    let weight_bytes: u64 = {
        let local: u64 = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            })
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        if local > 0 {
            local
        } else {
            hub::list_repo(model, None)?
                .iter()
                .filter(|(n, _)| n.ends_with(".safetensors"))
                .map(|(_, s)| s)
                .sum()
        }
    };
    // Host-resident pieces never land in VRAM: the gemma4 per-layer-embedding
    // table (vocab_per_layer × layers × ple_dim) is gathered from the CPU
    // mmap, so the device estimate excludes it.
    let mut host_bytes = 0u64;
    if cfg.model_type == "gemma4" {
        if let Ok(txt) = std::fs::read_to_string(dir.join("config.json")) {
            if let Ok(j) = json::parse(&txt) {
                let tc = j.get("text_config").unwrap_or(&j);
                let g = |k: &str| tc.get(k).and_then(json::Json::as_u64).unwrap_or(0);
                host_bytes = g("vocab_size_per_layer_input")
                    * g("num_hidden_layers")
                    * g("hidden_size_per_layer_input")
                    * 2;
            }
        }
    }
    let device_weights = weight_bytes.saturating_sub(host_bytes);
    let kv = cfg.n_layers * 2 * cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2;
    println!("✓ {} is architecturally executable by this engine", model);
    println!(
        "  model_type={} hidden={} layers={} heads={}/{} head_dim={} vocab={} max_seq={}",
        cfg.model_type,
        cfg.hidden_size,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.max_seq
    );
    println!(
        "  modalities: text{}{}{}",
        if cfg.vision.is_some() {
            " + vision"
        } else {
            ""
        },
        if cfg.audio.is_some() { " + audio" } else { "" },
        if cfg.is_embedding {
            " (embedding model)"
        } else {
            ""
        }
    );
    if host_bytes > 0 {
        println!(
            "  estimated VRAM: ~{} device weights + {} KV cache ({} PLE table stays in host RAM)",
            cuda::fmt_bytes(device_weights as usize),
            cuda::fmt_bytes(kv),
            cuda::fmt_bytes(host_bytes as usize)
        );
    } else {
        println!(
            "  estimated VRAM: ~{} weights (on-disk shard total) + {} KV cache",
            cuda::fmt_bytes(weight_bytes as usize),
            cuda::fmt_bytes(kv)
        );
    }
    // Workspace (logit staging, conversion scratch, graph pools) tracks
    // the largest single tensor — ~4% of weights is the observed envelope.
    let workspace = (device_weights as usize / 25).max(64 << 20);
    let need = device_weights as usize + kv + workspace;
    println!(
        "  + ~{} workspace → total ≈ {}",
        cuda::fmt_bytes(workspace),
        cuda::fmt_bytes(need)
    );
    // The verdict is against the REAL free VRAM when a GPU is visible —
    // an unconditional ✓ above a 6 GiB card's reality is how a 10 GiB
    // download ends in a load-time rejection.
    match cuda::CudaCtx::init(gpu_index()) {
        Ok(ctx) => {
            let snap = ctx.snapshot();
            let free = snap.vram_free as usize;
            let verdict = if need + (256 << 20) <= free {
                "fits in the currently free VRAM"
            } else if need <= free {
                "TIGHT: fits with <256 MiB margin — close other GPU processes first"
            } else {
                "does NOT fit in the currently free VRAM"
            };
            println!(
                "  free VRAM now: {} of {} — {} (the load gate re-checks exactly before allocating)",
                cuda::fmt_bytes(free),
                cuda::fmt_bytes(snap.vram_total as usize),
                verdict
            );
            if need > free {
                return Err(cima::err!(
                    "vram",
                    "estimate {} exceeds free VRAM {} — not pulling this would save {}",
                    cuda::fmt_bytes(need),
                    cuda::fmt_bytes(free),
                    cuda::fmt_bytes(weight_bytes as usize)
                ));
            }
        }
        Err(_) => println!("  (no GPU visible from here — estimate only; the load gate decides on the target machine)"),
    }
    println!("  run `cima pull {}` to download.", model);
    Ok(())
}

/// GGUF-aware `cima check`: with a local file, the EXACT device estimate
/// (this build dequantizes to f16 at load — device bytes = Σ numel × 2,
/// straight from the tensor table); without one, a decisive lower bound
/// from the published file size (Q8_0 is the densest supported block:
/// ×34/32 storage → f16 is ≥ ×1.88 the file; K-quants expand more).
fn check_gguf(repo: &str, tag: Option<&str>, dump_meta: bool) -> Res<()> {
    let dir = hub::local_dir(repo);
    let tagl = tag.map(str::to_ascii_lowercase);
    let local: Option<std::path::PathBuf> = std::fs::read_dir(&dir).ok().and_then(|d| {
        let mut c: Vec<_> = d
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .filter(|p| match &tagl {
                Some(t) => p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains(t),
                None => true,
            })
            .collect();
        c.sort_by_key(|p| p.file_name().unwrap_or_default().len());
        c.into_iter().next()
    });

    let (need, detail) = if let Some(path) = local {
        println!("(already pulled — checking the local gguf exactly)");
        use cima::traits::LoadedWeights as _;
        let w = cima::formats::gguf::GgufWeights::open(&path)?;
        if dump_meta {
            // Architecture onboarding instrument: an unknown family is
            // alias-able only if its hyper-parameter keys are the standard
            // set — novel keys mean novel math, and a blind alias would
            // degrade silently instead of failing loudly.
            let mut keys: Vec<_> = w.meta.iter().collect();
            keys.sort_by(|a, b| a.0.cmp(b.0));
            // Tensor inventory first: onboarding a new architecture is a
            // name-mapping exercise, and the names live here, not in the
            // metadata.
            {
                use cima::traits::LoadedWeights as _;
                let mut names: Vec<_> = w.tensors().keys().cloned().collect();
                names.sort();
                println!("  tensors ({}):", names.len());
                for n in &names {
                    let t = &w.tensors()[n];
                    println!("    {:<58} {:?} {:?}", n, t.shape, t.dtype);
                }
            }
            println!("  metadata ({} keys):", keys.len());
            for (k, v) in keys {
                let val = match v {
                    cima::formats::gguf::Value::Arr(a) => format!("[{} items]", a.len()),
                    cima::formats::gguf::Value::Str(s) if s.len() > 60 => {
                        format!("{:?}…", &s[..60])
                    }
                    other => format!("{:?}", other),
                };
                println!("    {} = {}", k, val);
            }
        }
        let cfg = cima::formats::gguf::model_config(&w)?;
        // The engine executes GGUF quantized-RESIDENT: packed blocks stay
        // packed in VRAM (fused-dequant GEMVs) — device weight bytes are
        // the codec's word, not an f16 expansion.
        let codec = cima::quant::gguf::GgufCodec { resident: true };
        use cima::traits::WeightCodec as _;
        let mut weights: usize = w.tensors().values().map(|t| codec.device_bytes(t)).sum();
        // gemma-4 PLE: the per-layer token table never enters VRAM (the
        // pipeline streams rows from host RAM) — subtract it from the
        // device estimate and say so.
        if let Some(ple) = w.tensors().get("per_layer_token_embd.weight") {
            let b = codec.device_bytes(ple);
            weights = weights.saturating_sub(b);
            println!(
                "  (gemma-4 PLE table {} stays in host RAM — excluded from device weights)",
                cuda::fmt_bytes(b)
            );
        }
        // Uniform sizing is wrong for layer-heterogeneous architectures; use
        // the per-layer figure when the checkpoint lets us compute one.
        let (kv, kv_exact) = match gemma4_kv_bytes(&path, &cfg) {
            Some(b) => (b, true),
            None => (
                cfg.n_layers * 2 * cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2,
                false,
            ),
        };
        let mmproj = mmproj_bytes(&path);
        if mmproj > 0 {
            weights += mmproj;
            println!(
                "  (+ {} mmproj tower sidecar, loaded with the LM)",
                cuda::fmt_bytes(mmproj)
            );
        }
        let ws = (weights / 25).max(64 << 20);
        println!(
            "  arch={} layers={} hidden={} vocab={} — file {}",
            cfg.model_type,
            cfg.n_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        let kv_note = if !kv_exact
            && w.meta_usize(&format!("{}.full_attention_interval", cfg.model_type))
                .is_some()
        {
            " (upper bound: hybrid SSM layers carry recurrent state, not KV)"
        } else if kv_exact {
            " (per layer type)"
        } else {
            ""
        };
        println!(
            "  device (quantized-resident): {} weights + {} KV{} + ~{} workspace",
            cuda::fmt_bytes(weights),
            cuda::fmt_bytes(kv),
            kv_note,
            cuda::fmt_bytes(ws)
        );
        (weights + kv + ws, "exact".to_string())
    } else {
        let files = hub::list_repo(repo, Some(".gguf"))?;
        let mut ggufs: Vec<(String, u64)> = files
            .into_iter()
            .filter(|(n, _)| n.ends_with(".gguf"))
            .collect();
        if let Some(t) = &tagl {
            ggufs.retain(|(n, _)| n.to_ascii_lowercase().contains(t));
        }
        ggufs.sort_by_key(|(n, _)| n.len());
        let Some((name, size)) = ggufs.first().cloned() else {
            return Err(cima::err!(
                "gguf",
                "no .gguf in '{}' matches '{}' on the Hub",
                repo,
                tag.unwrap_or("(any)")
            ));
        };
        // Resident execution: device weights ≈ the file itself (packed
        // blocks stay packed; the few f32 norms are noise).
        let est = size as usize + (size as usize) / 20;
        println!("  {} — {} on the Hub", name, cuda::fmt_bytes(size as usize));
        println!(
            "  quantized-resident execution: device weights ≈ file size ({} + KV + workspace at load)",
            cuda::fmt_bytes(est)
        );
        (est, "size-based".to_string())
    };

    match cuda::CudaCtx::init(gpu_index()) {
        Ok(ctx) => {
            let snap = ctx.snapshot();
            let free = snap.vram_free as usize;
            println!(
                "  free VRAM now: {} of {}",
                cuda::fmt_bytes(free),
                cuda::fmt_bytes(snap.vram_total as usize)
            );
            if need > free {
                return Err(cima::err!(
                    "vram",
                    "{} estimate {} exceeds free VRAM {}",
                    detail,
                    cuda::fmt_bytes(need),
                    cuda::fmt_bytes(free)
                ));
            }
            println!(
                "  verdict: fits ({} estimate {})",
                detail,
                cuda::fmt_bytes(need)
            );
        }
        Err(_) => println!(
            "  (no GPU visible — {} estimate {}; the load gate decides on the target machine)",
            detail,
            cuda::fmt_bytes(need)
        ),
    }
    Ok(())
}

/// `cima pull <MODEL> [--background] [--include <substr>] [--force]` —
/// download a repository from the Hugging Face Hub into `./models/`,
/// resuming partial files. Unless `--force` is given, the architecture
/// preflight gate runs first so an incompatible model never costs the
/// multi-GiB weight download.
fn cmd_pull(
    model: Option<&String>,
    background: bool,
    include: Option<String>,
    force: bool,
) -> Res<()> {
    let model = model.ok_or_else(|| {
        cima::err!(
            "cli",
            "usage: cima pull <MODEL> [--background] [--include <substr>] [--force]"
        )
    })?;
    // GGUF repos carry no config.json: their preflight is the file list
    // itself — confirm .gguf files exist (and that the tag matches one).
    let gguf_preflight = (|| -> cima::traits::Res<Option<(String, String)>> {
        // list_repo's default filter keeps only meta + .safetensors —
        // probe explicitly for gguf payloads.
        let files = hub::list_repo(model, Some(".gguf"))?;
        let ggufs: Vec<&String> = files
            .iter()
            .map(|(n, _)| n)
            .filter(|n| n.ends_with(".gguf"))
            .collect();
        if ggufs.is_empty() {
            return Ok(None);
        }
        match include.as_deref() {
            Some(t) => {
                let tl = t.to_ascii_lowercase();
                if !ggufs.iter().any(|n| n.to_ascii_lowercase().contains(&tl)) {
                    return Err(cima::err!(
                        "hub",
                        "no .gguf in '{}' matches '{}'. Available: {}",
                        model,
                        t,
                        ggufs
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                Ok(Some((
                    t.to_string(),
                    format!("tag '{}' matches; quantization executes at load", t),
                )))
            }
            // No tag: a single file is unambiguous; several need a choice
            // (downloading every quantization of a repo is never intended).
            None if ggufs.len() == 1 => Ok(Some((
                ".gguf".to_string(),
                format!("single file {}", ggufs[0]),
            ))),
            None => Err(cima::err!(
                "hub",
                "'{}' holds {} quantizations — pick one, e.g.: {}",
                model,
                ggufs.len(),
                ggufs
                    .iter()
                    .take(6)
                    .map(|n| format!(
                        "cima pull {}:{}",
                        model,
                        n.rsplit('-').next().unwrap_or(n).trim_end_matches(".gguf")
                    ))
                    .collect::<Vec<_>>()
                    .join(" | ")
            )),
        }
    })();
    match gguf_preflight {
        Ok(Some((inc, msg))) => {
            log::info(&format!("preflight OK (gguf): {}", msg));
            return hub::pull(model, background, Some(&inc));
        }
        Err(e) if !force => return Err(e),
        Err(e) => log::warn(&format!("gguf preflight failed but --force given: {}", e)),
        Ok(None) => {}
    }
    match preflight(model) {
        Ok(cfg) => log::info(&format!(
            "preflight OK: model_type '{}' is executable ({} layers, hidden {})",
            cfg.model_type, cfg.n_layers, cfg.hidden_size
        )),
        Err(e) if force => log::warn(&format!(
            "preflight failed but --force given, downloading anyway: {}",
            e
        )),
        Err(e) => {
            return Err(cima::err!(
                "cli",
                "preflight gate rejected '{}' before downloading any weights: {} \
                 (pass --force to download anyway, e.g. for use with a future loader)",
                model,
                e
            ))
        }
    }
    hub::pull(model, background, include.as_deref())
}

/// `cima list` — table of locally pulled models (name, size, modified).
fn cmd_list() -> Res<()> {
    let rows = hub::list_local_caps();
    println!(
        "{:<48} {:<18} {:>10}  MODIFIED",
        "NAME", "CAPABILITIES", "SIZE"
    );
    for (name, bytes, mtime, caps) in rows {
        let age = mtime.elapsed().map(|d| d.as_secs()).unwrap_or(0);
        println!(
            "{:<48} {:<18} {:>10}  {}",
            name,
            caps,
            cuda::fmt_bytes(bytes as usize),
            fmt_age(age)
        );
    }
    Ok(())
}

/// Human-readable "N units ago" used by `list`.
fn fmt_age(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} seconds ago"),
        60..=3599 => format!("{} minutes ago", secs / 60),
        3600..=86399 => format!("{} hours ago", secs / 3600),
        _ => format!("{} days ago", secs / 86400),
    }
}

/// `cima ps` — query a running server's `/api/ps`; this process holds no
/// GPU state of its own, mirroring Ollama's client/daemon split.
/// `cima logs` — read the shared-memory log ring of any running (or
/// recently exited) cima process. Levels: debug, info, warn, error,
/// metric (profiling). `--table` pivots the METRIC channel into aligned
/// columns; `--json` emits one object per record.
/// `cima stress` — flood a running server with concurrent generate
/// requests through the typed client and report the latency distribution
/// plus aggregate throughput. This measures the REAL serving path: HTTP,
/// queueing, model residency, streaming — not the bare engine.
fn cmd_stress(rest: &[String]) -> Res<()> {
    let model = rest.first().ok_or_else(|| {
        cima::err!("cli", "usage: cima stress MODEL [--requests N] [--concurrency C] [--max-tokens M] [--prompt TEXT]")
    })?.clone();
    let mut requests = 16usize;
    let mut concurrency = 4usize;
    let mut max_tokens = 64usize;
    let mut prompt = "Explain, in two sentences, why the sky is blue.".to_string();
    let mut i = 1;
    while i < rest.len() {
        match rest[i].as_str() {
            "--requests" if i + 1 < rest.len() => {
                requests = rest[i + 1]
                    .parse()
                    .map_err(|_| cima::err!("cli", "--requests expects an integer"))?;
                i += 1;
            }
            "--concurrency" if i + 1 < rest.len() => {
                concurrency = rest[i + 1]
                    .parse()
                    .map_err(|_| cima::err!("cli", "--concurrency expects an integer"))?;
                i += 1;
            }
            "--max-tokens" if i + 1 < rest.len() => {
                max_tokens = rest[i + 1]
                    .parse()
                    .map_err(|_| cima::err!("cli", "--max-tokens expects an integer"))?;
                i += 1;
            }
            "--prompt" if i + 1 < rest.len() => {
                prompt = rest[i + 1].clone();
                i += 1;
            }
            other => return Err(cima::err!("cli", "unknown flag '{}'", other)),
        }
        i += 1;
    }
    // Warm the model first so load time doesn't pollute the distribution.
    let warm = json::Json::obj()
        .set("model", json::Json::s(&model))
        .set("prompt", json::Json::s(""))
        .set("stream", json::Json::b(false));
    api::client::Client::local()
        .generate_stream(&warm, false, |_| {})
        .map_err(|e| {
            cima::err!(
                "cli",
                "warm-up failed ({}); is `cima serve` running with '{}' pulled?",
                e,
                model
            )
        })?;

    println!(
        "stress: {} requests, {} concurrent, {} tokens each",
        requests, concurrency, max_tokens
    );
    let t0 = std::time::Instant::now();
    let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let results = std::sync::Arc::new(Mutex::new(Vec::<(f64, f64, u64, bool)>::new()));
    let mut handles = Vec::new();
    for _ in 0..concurrency {
        let next = next.clone();
        let results = results.clone();
        let model = model.clone();
        let prompt = prompt.clone();
        handles.push(std::thread::spawn(move || {
            let client = api::client::Client::local();
            loop {
                if next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= requests {
                    break;
                }
                let body = json::Json::obj()
                    .set("model", json::Json::s(&model))
                    .set("prompt", json::Json::s(&prompt))
                    .set(
                        "options",
                        json::Json::obj()
                            .set("num_predict", json::Json::n(0.0 + max_tokens as f64))
                            .set("seed", json::Json::n(7.0)),
                    )
                    .set("stream", json::Json::b(true));
                let t = std::time::Instant::now();
                let mut ttft = 0.0f64;
                let res = client.generate_stream(&body, false, |chunk| {
                    if ttft == 0.0 && chunk.get("done").and_then(json::Json::as_bool) != Some(true)
                    {
                        ttft = t.elapsed().as_secs_f64() * 1e3;
                    }
                });
                let total = t.elapsed().as_secs_f64() * 1e3;
                let (tokens, ok) = match res {
                    Ok(fin) => (
                        fin.get("eval_count")
                            .and_then(json::Json::as_f64)
                            .unwrap_or(0.0) as u64,
                        true,
                    ),
                    Err(_) => (0, false),
                };
                results.lock().unwrap().push((total, ttft, tokens, ok));
            }
        }));
    }
    for h in handles {
        h.join()
            .map_err(|_| cima::err!("cli", "stress worker panicked"))?;
    }
    let wall = t0.elapsed().as_secs_f64();
    let mut rows = results.lock().unwrap().clone();
    let failed = rows.iter().filter(|r| !r.3).count();
    rows.retain(|r| r.3);
    if rows.is_empty() {
        return Err(cima::err!("cli", "all {} requests failed", requests));
    }
    let mut lat: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let mut ttfts: Vec<f64> = rows.iter().map(|r| r.1).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[f64], p: f64| v[((v.len() as f64 - 1.0) * p) as usize];
    let tokens: u64 = rows.iter().map(|r| r.2).sum();
    println!(
        "stress: {} ok, {} failed in {:.1}s",
        rows.len(),
        failed,
        wall
    );
    println!(
        "  latency  p50={:.0}ms p95={:.0}ms p99={:.0}ms",
        pct(&lat, 0.50),
        pct(&lat, 0.95),
        pct(&lat, 0.99)
    );
    println!(
        "  ttft     p50={:.0}ms p99={:.0}ms",
        pct(&ttfts, 0.50),
        pct(&ttfts, 0.99)
    );
    println!(
        "  tokens   {} total -> {:.1} tok/s aggregate",
        tokens,
        tokens as f64 / wall
    );
    log::metric(
        "stress",
        &[
            ("model", model),
            ("requests", rows.len().to_string()),
            ("failed", failed.to_string()),
            ("concurrency", concurrency.to_string()),
            ("lat_p50_ms", format!("{:.1}", pct(&lat, 0.50))),
            ("lat_p99_ms", format!("{:.1}", pct(&lat, 0.99))),
            ("ttft_p50_ms", format!("{:.1}", pct(&ttfts, 0.50))),
            ("agg_tok_per_s", format!("{:.1}", tokens as f64 / wall)),
        ],
    );
    Ok(())
}

fn cmd_logs(rest: &[String]) -> Res<()> {
    let mut follow = false;
    let mut json_out = false;
    let mut table = false;
    let mut last_n: Option<usize> = None;
    let mut levels: Vec<cima::shmlog::Level> = Vec::new();
    let mut path = cima::shmlog::default_path();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-f" | "--follow" => follow = true,
            "--json" => json_out = true,
            "--table" => table = true,
            "-n" if i + 1 < rest.len() => {
                last_n = rest[i + 1].parse().ok();
                i += 1;
            }
            "--level" if i + 1 < rest.len() => {
                for part in rest[i + 1].split(',') {
                    levels.push(
                        cima::shmlog::Level::parse(part)
                            .ok_or_else(|| cima::err!("cli", "unknown level '{}' (debug|info|warn|error|metric)", part))?,
                    );
                }
                i += 1;
            }
            "--path" if i + 1 < rest.len() => {
                path = rest[i + 1].clone().into();
                i += 1;
            }
            other => {
                return Err(cima::err!(
                    "cli",
                    "unknown flag '{}' (usage: cima logs [-f] [-n N] [--level L,..] [--json|--table] [--path P])",
                    other
                ))
            }
        }
        i += 1;
    }
    if table {
        levels = vec![cima::shmlog::Level::Metric];
    }
    let ring = cima::shmlog::Ring::open(&path)?;
    let pass = |r: &cima::shmlog::Record| levels.is_empty() || levels.contains(&r.level);

    fn print_text(r: &cima::shmlog::Record) {
        let secs = r.ts_ns / 1_000_000_000;
        let ms = (r.ts_ns / 1_000_000) % 1000;
        println!(
            "{}.{:03} {:6} {}:{} {}",
            log::stamp_at(secs),
            ms,
            r.level.name(),
            r.file,
            r.line,
            r.msg
        );
    }
    fn print_json(r: &cima::shmlog::Record) {
        let j = json::Json::obj()
            .set("ts_ns", json::Json::n(r.ts_ns as f64))
            .set("level", json::Json::s(r.level.name()))
            .set("file", json::Json::s(&r.file))
            .set("line", json::Json::n(r.line as f64))
            .set("msg", json::Json::s(&r.msg));
        println!("{}", j.dump());
    }
    /// METRIC table: subsystem column + the union of keys, aligned.
    fn print_table(recs: &[cima::shmlog::Record]) {
        let mut rows: Vec<(String, Vec<(String, String)>)> = Vec::new();
        let mut keys: Vec<String> = Vec::new();
        for r in recs {
            let mut it = r.msg.split_whitespace();
            let Some(sub) = it.next() else { continue };
            let mut kv = Vec::new();
            for pair in it {
                if let Some((k, v)) = pair.split_once('=') {
                    if !keys.iter().any(|x| x == k) {
                        keys.push(k.to_string());
                    }
                    kv.push((k.to_string(), v.to_string()));
                }
            }
            rows.push((sub.to_string(), kv));
        }
        let mut widths: Vec<usize> = keys.iter().map(|k| k.len()).collect();
        for (_, kv) in &rows {
            for (k, v) in kv {
                let idx = keys.iter().position(|x| x == k).unwrap();
                widths[idx] = widths[idx].max(v.len());
            }
        }
        let sub_w = rows.iter().map(|(s, _)| s.len()).max().unwrap_or(6).max(6);
        print!("{:<w$}", "METRIC", w = sub_w + 2);
        for (k, w) in keys.iter().zip(&widths) {
            print!("{:<w$}", k, w = w + 2);
        }
        println!();
        for (sub, kv) in &rows {
            print!("{:<w$}", sub, w = sub_w + 2);
            for (k, w) in keys.iter().zip(&widths) {
                let v = kv
                    .iter()
                    .find(|(kk, _)| kk == k)
                    .map(|(_, v)| v.as_str())
                    .unwrap_or("-");
                print!("{:<w$}", v, w = w + 2);
            }
            println!();
        }
    }

    let mut recs: Vec<_> = ring.read_since(0).into_iter().filter(&pass).collect();
    if let Some(n) = last_n {
        let skip = recs.len().saturating_sub(n);
        recs.drain(..skip);
    }
    if table {
        print_table(&recs);
    } else {
        for r in &recs {
            if json_out {
                print_json(r)
            } else {
                print_text(r)
            }
        }
    }
    if follow {
        let mut last = recs.last().map(|r| r.seq).unwrap_or(0);
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let fresh: Vec<_> = ring.read_since(last).into_iter().filter(&pass).collect();
            for r in &fresh {
                if table {
                    print_table(std::slice::from_ref(r));
                } else if json_out {
                    print_json(r)
                } else {
                    print_text(r)
                }
                last = r.seq.max(last);
            }
        }
    }
    Ok(())
}

/// `cima ready [MODEL...] [--wait] [--timeout SECS]` — the orchestration
/// probe. Exits 0 when the server is healthy and every named model is
/// present locally, non-zero otherwise. With `--wait` it polls until ready
/// or the timeout elapses — the one-liner a dependent service runs in its
/// start-up gate after kicking off pulls.
fn cmd_ready(models: &[String], wait: bool, timeout_secs: u64) -> Res<()> {
    let client = api::client::Client::local();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match client.ready(models) {
            Ok(j) => {
                let ready = j.bool_of("ready").unwrap_or(false);
                if ready {
                    if models.is_empty() {
                        println!("cima healthy");
                    } else {
                        println!("ready — all {} model(s) present", models.len());
                    }
                    return Ok(());
                }
                if !wait {
                    // Report which are missing, then exit non-zero.
                    if let Some(rows) = j.get("models").and_then(json::Json::as_arr) {
                        for m in rows {
                            let name = m.get("model").and_then(json::Json::as_str).unwrap_or("?");
                            let present = m
                                .get("present")
                                .and_then(json::Json::as_bool)
                                .unwrap_or(false);
                            println!(
                                "  {:<48} {}",
                                name,
                                if present { "present" } else { "MISSING" }
                            );
                        }
                    }
                    return Err(cima::err!("cli", "not ready"));
                }
            }
            Err(e) => {
                if !wait {
                    return Err(cima::err!("cli", "cima not reachable: {}", e));
                }
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(cima::err!(
                "cli",
                "timed out after {}s waiting for readiness",
                timeout_secs
            ));
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn cmd_ps() -> Res<()> {
    // Residency is server state by nature; without a daemon there is
    // nothing resident — that's information, not an error (the CLI never
    // hard-depends on the API: standalone commands stay standalone).
    let ps = match api::client::Client::local().ps() {
        Ok(p) => p,
        Err(_) => {
            println!("no cima server running on this host — models are only resident inside `cima serve`");
            return Ok(());
        }
    };
    let models = ps.get("models").and_then(json::Json::as_arr).unwrap_or(&[]);
    if models.is_empty() {
        println!("no models resident");
        return Ok(());
    }
    println!("{:<48} {:>12}  EXPIRES", "MODEL", "VRAM");
    for m in models {
        let name = m.get("name").and_then(json::Json::as_str).unwrap_or("?");
        let vram = m
            .get("size_vram")
            .and_then(json::Json::as_f64)
            .unwrap_or(0.0) as usize;
        let exp = m
            .get("expires_in")
            .map(|e| match e.as_f64() {
                Some(s) => format!("{}s", s as u64),
                None => e.as_str().unwrap_or("?").to_string(),
            })
            .unwrap_or_default();
        println!("{:<48} {:>12}  {}", name, cuda::fmt_bytes(vram), exp);
    }
    Ok(())
}

/// `cima show <MODEL>` — pretty-print the validated `config.json` of a
/// locally pulled model without touching the GPU.
fn cmd_show(model: Option<&String>) -> Res<()> {
    let model = model.ok_or_else(|| cima::err!("cli", "usage: cima show <MODEL>"))?;
    let dir = hub::local_dir(model);
    let cfg_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).map_err(|e| {
        cima::err!(
            "cli",
            "model {:?} is not pulled (missing {}): {}",
            model,
            cfg_path.display(),
            e
        )
    })?;
    let doc = json::parse(&raw).map_err(|e| cima::err!("cli", "corrupt config.json: {}", e))?;
    println!("{}", doc.dump());
    Ok(())
}

/// `cima rm <MODEL>` — delete the local snapshot directory.
fn cmd_rm(model: Option<&String>) -> Res<()> {
    let model = model.ok_or_else(|| cima::err!("cli", "usage: cima rm <MODEL[:TAG]>"))?;
    let (repo, tag) = match model.split_once(':') {
        Some((r, t)) => (r, Some(t)),
        None => (model.as_str(), None),
    };
    // Whole-model removal goes through the server when one runs (it evicts
    // the model from VRAM first); per-quant (tagged) removal unlinks just
    // the matching LM gguf(s) — safe even under a resident server, since a
    // Linux unlink leaves the mmapped inode alive until unmapped. Tag
    // matching is identical to `run`: case-insensitive filename substring,
    // mmproj excluded.
    if tag.is_none() && api::client::Client::local().delete_quiet(model) {
        println!("deleted {} (via server: evicted from VRAM first)", model);
        return Ok(());
    }
    let dir = hub::local_dir(repo);
    // Legacy layout: older pulls encoded the tag into the directory name
    // (`repo@tag`, one dir per quant). A tagged rm whose repo dir is absent
    // falls back to removing that whole legacy dir; an untagged rm sweeps
    // every `repo@*` legacy sibling after the main dir.
    if !dir.is_dir() {
        if let Some(t) = tag {
            let legacy = hub::local_dir(&format!("{}:{}", repo, t));
            if legacy.is_dir() {
                let freed = {
                    fn sz(d: &std::path::Path) -> u64 {
                        std::fs::read_dir(d)
                            .map(|rd| {
                                rd.filter_map(|e| e.ok())
                                    .map(|e| {
                                        let p = e.path();
                                        if p.is_dir() {
                                            sz(&p)
                                        } else {
                                            std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
                                        }
                                    })
                                    .sum()
                            })
                            .unwrap_or(0)
                    }
                    sz(&legacy)
                };
                std::fs::remove_dir_all(&legacy)?;
                println!("deleted {} (legacy per-tag layout)", model);
                println!("freed {:.2} GiB", freed as f64 / (1u64 << 30) as f64);
                return Ok(());
            }
        }
        return Err(cima::err!(
            "cli",
            "model {:?} is not pulled ({})",
            repo,
            dir.display()
        ));
    }
    fn dir_size(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| {
                        let p = e.path();
                        if p.is_dir() {
                            dir_size(&p)
                        } else {
                            std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
    let is_mmproj = |p: &std::path::PathBuf| {
        p.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase()
            .contains("mmproj")
    };
    let mut freed = 0u64;
    match tag {
        None => {
            freed = dir_size(&dir);
            std::fs::remove_dir_all(&dir)?;
            println!("deleted {}", repo);
            let enc = repo.replace('/', "__");
            if let Ok(rd) = std::fs::read_dir(hub::models_dir()) {
                for e in rd.filter_map(|e| e.ok()) {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if name.starts_with(&format!("{}@", enc)) && e.path().is_dir() {
                        freed += dir_size(&e.path());
                        let _ = std::fs::remove_dir_all(e.path());
                        println!(
                            "deleted {} (legacy per-tag layout)",
                            name.replace("__", "/").replace('@', ":")
                        );
                    }
                }
            }
        }
        Some(t) => {
            let tl = t.to_ascii_lowercase();
            let ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
                .collect();
            let victims: Vec<&std::path::PathBuf> = ggufs
                .iter()
                .filter(|p| !is_mmproj(p))
                .filter(|p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .contains(&tl)
                })
                .collect();
            if victims.is_empty() {
                let have: Vec<String> = ggufs
                    .iter()
                    .filter(|p| !is_mmproj(p))
                    .map(|p| {
                        p.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect();
                return Err(cima::err!(
                    "cli",
                    "no .gguf in {} matches tag '{}'. Available: {}",
                    dir.display(),
                    t,
                    if have.is_empty() {
                        "(none)".to_string()
                    } else {
                        have.join(", ")
                    }
                ));
            }
            for v in &victims {
                freed += std::fs::metadata(v).map(|m| m.len()).unwrap_or(0);
                std::fs::remove_file(v)?;
                println!(
                    "deleted {}",
                    v.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            // Only sidecars (mmproj/config/…) left → retire the directory.
            let lm_left = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
                .any(|p| !is_mmproj(&p));
            if !lm_left {
                freed += dir_size(&dir);
                std::fs::remove_dir_all(&dir)?;
                println!("deleted {} (no LM ggufs left)", repo);
            }
        }
    }
    println!("freed {:.2} GiB", freed as f64 / (1u64 << 30) as f64);
    Ok(())
}
