//! # log — structured logging over the shared-memory ring
//!
//! Every record lands in the [`crate::shmlog`] ring (lock-free, ~100 ns,
//! with the caller's real `file:line` via `#[track_caller]` — call sites
//! stay plain function calls). What reaches the **terminal** is policy:
//!
//! * `WARN`/`ERROR` — always on stderr.
//! * `INFO`/`METRIC` — ring only; `CIMA_LOG=info` restores them on stderr.
//! * `DEBUG` — stderr only with `CIMA_LOG=debug`; always in the ring.
//!
//! `cima logs` reads the ring from any process: follow mode, level
//! filters, JSON output, and a table view for the `METRIC` channel (whose
//! strict `key=value` grammar exists precisely to be machine-parseable).

use std::io::Write;
use std::panic::Location;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use shmlog::Level;

static LOCK: Mutex<()> = Mutex::new(());

/// True when `CIMA_LOG=info` or `debug` — restores INFO/METRIC on stderr.
fn verbose() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CIMA_LOG")
            .map(|v| v.eq_ignore_ascii_case("info") || v.eq_ignore_ascii_case("debug"))
            .unwrap_or(false)
    })
}

/// ISO-8601 UTC, second precision — self-contained civil-time conversion.
pub fn stamp_at(secs: u64) -> String {
    let (mut days, rem) = (secs / 86400, secs % 86400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let mut year = 1970u64;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if days < len {
            break;
        }
        days -= len;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let ml = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 0usize;
    while days >= ml[month] {
        days -= ml[month];
        month += 1;
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year,
        month + 1,
        days + 1,
        h,
        m,
        s
    )
}

fn stamp() -> String {
    stamp_at(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
}

fn emit(level: Level, msg: &str, loc: &Location) {
    if let Some(ring) = shmlog::global() {
        ring.write(level, loc.file(), loc.line(), msg);
    }
    // The crate's unit tests exercise error paths on purpose; their log
    // lines belong in the ring, not interleaved with the test harness.
    if cfg!(test) || std::env::var_os("CIMA_LOG_SILENT").is_some() {
        return;
    }
    // Terminal policy: the CLI speaks through its own println UX; the
    // engine's telemetry lives in the ring. Only anomalies interrupt.
    let to_stderr = match level {
        Level::Warn | Level::Error => true,
        Level::Info | Level::Metric => verbose(),
        Level::Debug => debug_on(),
    };
    if to_stderr {
        let _g = LOCK.lock();
        if json_format() {
            // One JSON object per line for log collectors. Fields are
            // stable: ts, level, file, line, msg.
            let _ = writeln!(
                std::io::stderr(),
                "{{\"ts\":\"{}\",\"level\":\"{}\",\"file\":\"{}\",\"line\":{},\"msg\":\"{}\"}}",
                stamp(),
                level.name().trim(),
                loc.file(),
                loc.line(),
                json_escape(msg)
            );
        } else {
            let _ = writeln!(std::io::stderr(), "{} {:5} {}", stamp(), level.name(), msg);
        }
    }
}

/// True when `CIMA_LOG_FORMAT=json` — emit one JSON object per line
/// instead of the human-readable text format.
fn json_format() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CIMA_LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"))
    })
}

/// Minimal JSON string escaping for log payloads (quotes, backslashes,
/// control characters).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Informational event.
#[track_caller]
pub fn info(msg: &str) {
    emit(Level::Info, msg, Location::caller());
}

/// Developer diagnostics — always in the ring; on stderr only with
/// `CIMA_LOG=debug`.
#[track_caller]
pub fn debug(msg: &str) {
    emit(Level::Debug, msg, Location::caller());
}

/// True when debug logging is active — guards costly message construction.
pub fn debug_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CIMA_LOG").is_ok_and(|v| v.eq_ignore_ascii_case("debug")))
}

/// Recoverable anomaly.
#[track_caller]
pub fn warn(msg: &str) {
    emit(Level::Warn, msg, Location::caller());
}

/// Failure (every `EngineError` routes through here on construction).
#[track_caller]
pub fn error(msg: &str) {
    emit(Level::Error, msg, Location::caller());
}

/// Strict machine-parseable performance metric: `subsystem k=v k=v …`
/// (the profiling channel; `cima logs --table` renders it as columns).
#[track_caller]
pub fn metric(subsystem: &str, kv: &[(&str, String)]) {
    let body: Vec<String> = kv.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    emit(
        Level::Metric,
        &format!("{} {}", subsystem, body.join(" ")),
        Location::caller(),
    );
}

pub mod shmlog {
    //! # shmlog — lock-free shared-memory log ring
    //!
    //! Logging that costs almost nothing on the hot path and never touches the
    //! terminal: a single `mmap`ed ring in `/dev/shm`, written with one atomic
    //! `fetch_add` plus a `memcpy`, read by `cima logs` from any other process.
    //!
    //! ## Layout
    //!
    //! One 64-byte header followed by `slot_count` slots of exactly 64 bytes
    //! (one CPU cache line — two writer threads never contend on a line they
    //! both own). A record occupies one *head* slot and, when the payload
    //! outgrows it, contiguous continuation slots:
    //!
    //! ```text
    //! head slot:  seq u64 | ts_ns u64 | line u32 | level u8 | nslots u8 |
    //!             len u16 | fnv1a u32 | payload[36]   (payload = "file\0msg")
    //! cont slot:  payload[64]
    //! ```
    //!
    //! Records carry an FNV-1a checksum of their payload: beyond the seqlock,
    //! a record is accepted only if its CONTENT verifies. A ring overwrites
    //! oldest-first, so a continuation slot of a newer record legitimately
    //! clobbers older heads byte-for-byte — including their seq field, which
    //! payload bytes can forge. The checksum makes any such forgery, and any
    //! residual torn read, a detected discard instead of corrupt output.
    //!
    //! ## Concurrency contract (seqlock)
    //!
    //! Writers reserve `nslots` with `cursor.fetch_add` (wait-free, multi
    //! producer), fill the payload, then publish by storing `seq = reserved+1`
    //! into the head with `Release`. A head is valid iff `(seq-1) % slot_count
    //! == its own slot index` — a clobbered head from a previous lap fails
    //! this. Readers copy the payload and re-read `seq`; a change means the
    //! ring lapped them mid-copy and the record is discarded, never torn.
    //!
    //! Records cap at [`MAX_RECORD`]; the ring overwrites oldest-first. Nothing
    //! in this module allocates after [`Ring::create`]/[`Ring::open`].

    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::{err, traits::Res};

    /// Slot size: one x86-64/aarch64 L1/L2 cache line.
    pub const SLOT: usize = 64;
    /// Head-slot payload capacity after the 28-byte record header
    /// (seq 8 + ts 8 + line 4 + level 1 + nslots 1 + len 2 + checksum 4).
    pub const HEAD_PAYLOAD: usize = SLOT - 28;
    /// Largest accepted record payload (file path + message); longer messages
    /// are truncated, not split into multiple records.
    pub const MAX_RECORD: usize = 4096;
    /// Default ring: 16384 slots = 1 MiB.
    pub const DEFAULT_SLOTS: usize = 16384;

    const MAGIC: u64 = 0x434f_4c41_4d49_4332; // v2: checksummed records

    #[repr(C, align(64))]
    struct Header {
        magic: u64,
        slot_count: u64,
        /// Monotonic count of slots ever reserved (never wraps logically;
        /// physical slot = index % slot_count).
        cursor: AtomicU64,
        /// Records truncated to MAX_RECORD (diagnostic).
        truncated: AtomicU64,
        pid: u64,
        start_ts_ns: u64,
        _pad: [u8; 16],
    }

    /// Severity / channel of a record. `Metric` is the profiling channel —
    /// `cima logs --table` renders its key=value payloads as columns.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Level {
        Debug = 0,
        Info = 1,
        Warn = 2,
        Error = 3,
        Metric = 4,
    }

    impl Level {
        pub fn name(self) -> &'static str {
            match self {
                Level::Debug => "DEBUG",
                Level::Info => "INFO",
                Level::Warn => "WARN",
                Level::Error => "ERROR",
                Level::Metric => "METRIC",
            }
        }

        pub fn parse(s: &str) -> Option<Level> {
            Some(match s.to_ascii_lowercase().as_str() {
                "debug" => Level::Debug,
                "info" => Level::Info,
                "warn" | "warning" => Level::Warn,
                "error" => Level::Error,
                "metric" | "profiling" | "profile" => Level::Metric,
                _ => return None,
            })
        }

        fn from_u8(v: u8) -> Level {
            match v {
                0 => Level::Debug,
                2 => Level::Warn,
                3 => Level::Error,
                4 => Level::Metric,
                _ => Level::Info,
            }
        }
    }

    /// A decoded record, as returned by [`Ring::read_since`].
    #[derive(Debug)]
    pub struct Record {
        pub seq: u64,
        pub ts_ns: u64,
        pub level: Level,
        pub file: String,
        pub line: u32,
        pub msg: String,
    }

    /// The mapped ring: one global writer handle per process ([`global`]),
    /// ad-hoc reader handles in `cima logs`.
    pub struct Ring {
        base: *mut u8,
        slots: usize,
        map_len: usize,
    }

    // Shared-memory structure with interior atomics; the raw pointer is stable
    // for the mapping's lifetime.
    unsafe impl Send for Ring {}
    unsafe impl Sync for Ring {}

    // Minimal FFI (same zero-crate approach as formats::safetensors).
    extern "C" {
        fn mmap(
            addr: *mut std::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut std::ffi::c_void;
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
        fn getuid() -> u32;
    }
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_SHARED: i32 = 1;

    fn fnv1a(bytes: &[u8]) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for &b in bytes {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }

    fn now_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Default ring path: per-user, overridable with `CIMA_SHM`.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("CIMA_SHM") {
            return p.into();
        }
        format!("/dev/shm/cima-log.{}", unsafe { getuid() }).into()
    }

    impl Ring {
        fn map(file: &std::fs::File, len: usize) -> Res<*mut u8> {
            use std::os::unix::io::AsRawFd;
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                ) as *mut u8
            };
            if ptr as isize == -1 {
                return Err(err!(
                    "shmlog",
                    "mmap failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(ptr)
        }

        /// Create (or reset) a ring at `path` with `slots` slots.
        pub fn create(path: &std::path::Path, slots: usize) -> Res<Ring> {
            let len = SLOT + slots * SLOT;
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .map_err(|e| err!("shmlog", "open '{}': {}", path.display(), e))?;
            file.set_len(len as u64)
                .map_err(|e| err!("shmlog", "set_len: {}", e))?;
            let base = Self::map(&file, len)?;
            let ring = Ring {
                base,
                slots,
                map_len: len,
            };
            let h = ring.header();
            h.magic = MAGIC;
            h.slot_count = slots as u64;
            h.cursor = AtomicU64::new(0);
            h.truncated = AtomicU64::new(0);
            h.pid = std::process::id() as u64;
            h.start_ts_ns = now_ns();
            Ok(ring)
        }

        /// Open an existing ring (readers only load atomics, so read-write
        /// mapping is shared safely with the writer).
        pub fn open(path: &std::path::Path) -> Res<Ring> {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|e| {
                    err!(
                        "shmlog",
                        "open '{}': {} — is a cima process running?",
                        path.display(),
                        e
                    )
                })?;
            let len = file
                .metadata()
                .map_err(|e| err!("shmlog", "stat: {}", e))?
                .len() as usize;
            if len < 2 * SLOT {
                return Err(err!(
                    "shmlog",
                    "'{}' is too small to be a log ring",
                    path.display()
                ));
            }
            let base = Self::map(&file, len)?;
            let ring = Ring {
                base,
                slots: (len - SLOT) / SLOT,
                map_len: len,
            };
            if ring.header().magic != MAGIC {
                return Err(err!(
                    "shmlog",
                    "'{}' is not a cima log ring (bad magic)",
                    path.display()
                ));
            }
            Ok(ring)
        }

        #[allow(clippy::mut_from_ref)]
        fn header(&self) -> &mut Header {
            unsafe { &mut *(self.base as *mut Header) }
        }

        fn slot(&self, idx: u64) -> *mut u8 {
            let phys = (idx % self.slots as u64) as usize;
            unsafe { self.base.add(SLOT + phys * SLOT) }
        }

        /// Monotonic count of slots reserved so far (readers poll this).
        pub fn cursor(&self) -> u64 {
            self.header().cursor.load(Ordering::Acquire)
        }

        /// Records truncated to [`MAX_RECORD`] so far.
        pub fn truncated(&self) -> u64 {
            self.header().truncated.load(Ordering::Relaxed)
        }

        /// Write one record. Hot path: one `fetch_add`, one or more 64-byte
        /// copies, one `Release` store. No locks, no syscalls, no allocation.
        pub fn write(&self, level: Level, file: &str, line: u32, msg: &str) {
            let file = file.as_bytes();
            let msg = msg.as_bytes();
            let mut len = file.len() + 1 + msg.len();
            if len > MAX_RECORD {
                self.header().truncated.fetch_add(1, Ordering::Relaxed);
                len = MAX_RECORD;
            }
            let cont = len.saturating_sub(HEAD_PAYLOAD);
            let nslots = 1 + cont.div_ceil(SLOT);
            if nslots > self.slots {
                return;
            }
            let start = self
                .header()
                .cursor
                .fetch_add(nslots as u64, Ordering::AcqRel);

            // payload = file \0 message (tail-truncated to fit).
            let mut payload = [0u8; MAX_RECORD];
            let f = file.len().min(len.saturating_sub(1));
            payload[..f].copy_from_slice(&file[..f]);
            let m = (len - f - 1).min(msg.len());
            payload[f + 1..f + 1 + m].copy_from_slice(&msg[..m]);

            unsafe {
                let head = self.slot(start);
                let seq_atom = &*(head as *const AtomicU64);
                // Invalidate first: a concurrent reader of the OLD record at
                // this physical slot must fail its seq re-check. The SeqCst
                // fence is the seqlock's load-bearing wall: a plain Release
                // store only orders PRIOR writes, so the payload copies below
                // could be hoisted above the invalidation and a reader holding
                // the stale seq would accept a mixed payload.
                seq_atom.store(0, Ordering::Release);
                std::sync::atomic::fence(Ordering::SeqCst);
                std::ptr::write_unaligned(head.add(8) as *mut u64, now_ns());
                std::ptr::write_unaligned(head.add(16) as *mut u32, line);
                *head.add(20) = level as u8;
                *head.add(21) = nslots as u8;
                std::ptr::write_unaligned(head.add(22) as *mut u16, len as u16);
                std::ptr::write_unaligned(head.add(24) as *mut u32, fnv1a(&payload[..len]));
                let in_head = len.min(HEAD_PAYLOAD);
                std::ptr::copy_nonoverlapping(payload.as_ptr(), head.add(28), in_head);
                let mut off = in_head;
                for s in 1..nslots {
                    let chunk = (len - off).min(SLOT);
                    std::ptr::copy_nonoverlapping(
                        payload.as_ptr().add(off),
                        self.slot(start + s as u64),
                        chunk,
                    );
                    off += chunk;
                }
                // Publish: seq = start+1 (0 means "never written").
                seq_atom.store(start + 1, Ordering::Release);
            }
        }

        /// Decode every record with `seq > since`, oldest first. Torn or
        /// lapped records are skipped, never mis-read (seqlock re-check).
        pub fn read_since(&self, since: u64) -> Vec<Record> {
            let mut out = Vec::new();
            for phys in 0..self.slots as u64 {
                unsafe {
                    let head = self.slot(phys);
                    let seq_atom = &*(head as *const AtomicU64);
                    let seq = seq_atom.load(Ordering::Acquire);
                    if seq == 0 || seq <= since {
                        continue;
                    }
                    let start = seq - 1;
                    if start % self.slots as u64 != phys {
                        continue; // stale head bytes from a previous lap
                    }
                    // The seq field itself can be clobbered by a continuation
                    // slot of a NEWER record (the ring overwrites whole slots):
                    // payload bytes then forge an arbitrary seq. A genuine seq
                    // must lie inside the live window — in particular a forged
                    // FUTURE seq makes `cursor - start` saturate to 0 and
                    // would otherwise sail through the lap guard.
                    let cur = self.header().cursor.load(Ordering::Acquire);
                    if start >= cur || cur - start > self.slots as u64 {
                        continue;
                    }
                    let ts = std::ptr::read_unaligned(head.add(8) as *const u64);
                    let line = std::ptr::read_unaligned(head.add(16) as *const u32);
                    let level = Level::from_u8(*head.add(20));
                    let nslots = *head.add(21) as usize;
                    let len = std::ptr::read_unaligned(head.add(22) as *const u16) as usize;
                    let want = std::ptr::read_unaligned(head.add(24) as *const u32);
                    if len > MAX_RECORD || nslots == 0 || nslots > self.slots {
                        continue;
                    }
                    let mut payload = vec![0u8; len];
                    let in_head = len.min(HEAD_PAYLOAD);
                    std::ptr::copy_nonoverlapping(head.add(28), payload.as_mut_ptr(), in_head);
                    let mut off = in_head;
                    for s in 1..nslots {
                        let chunk = (len - off).min(SLOT);
                        std::ptr::copy_nonoverlapping(
                            self.slot(start + s as u64),
                            payload.as_mut_ptr().add(off),
                            chunk,
                        );
                        off += chunk;
                    }
                    // Seqlock re-check: head unchanged AND our span not
                    // lapped. The fence keeps the payload copies above from
                    // sinking below this load (Acquire alone only constrains
                    // LATER operations).
                    std::sync::atomic::fence(Ordering::SeqCst);
                    if seq_atom.load(Ordering::Acquire) != seq {
                        continue;
                    }
                    let cursor = self.header().cursor.load(Ordering::Acquire);
                    if start >= cursor || cursor - start > self.slots as u64 {
                        continue; // lapped (or forged) while copying
                    }
                    // Content check: the final arbiter. Forged seqs and torn
                    // payloads die here whatever the interleaving was.
                    if fnv1a(&payload) != want {
                        continue;
                    }
                    let nul = payload.iter().position(|&b| b == 0).unwrap_or(0);
                    out.push(Record {
                        seq,
                        ts_ns: ts,
                        level,
                        file: String::from_utf8_lossy(&payload[..nul]).into_owned(),
                        line,
                        msg: String::from_utf8_lossy(
                            &payload[nul.min(payload.len().saturating_sub(1)) + 1..],
                        )
                        .into_owned(),
                    });
                }
            }
            out.sort_by_key(|r| r.seq);
            out
        }
    }

    impl Drop for Ring {
        fn drop(&mut self) {
            unsafe {
                munmap(self.base as *mut std::ffi::c_void, self.map_len);
            }
        }
    }

    /// Per-process writer handle, created lazily. Failure to set up shared
    /// memory silently disables the ring — logging never takes the process
    /// down.
    pub fn global() -> Option<&'static Ring> {
        static RING: std::sync::OnceLock<Option<Ring>> = std::sync::OnceLock::new();
        RING.get_or_init(|| Ring::create(&default_path(), DEFAULT_SLOTS).ok())
            .as_ref()
    }
}
