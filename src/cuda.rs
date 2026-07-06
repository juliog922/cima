//! # cuda — GPU substrate
//!
//! Hand-declared FFI to the Nvidia driver stack (no bindgen, no crates):
//!
//! * **CUDA Driver API** (`libcuda.so`)   — contexts, streams, memory, kernels.
//! * **NVRTC** (`libnvrtc.so`)            — the engine's CUDA C kernels are
//!   embedded as source and JIT-compiled to PTX *for the exact GPU arch* at
//!   startup, then cached on disk. No toolchain needed at build time.
//! * **cuBLAS** (`libcublas.so`)          — peak-throughput FP16/BF16 GEMM.
//! * **NVML** (`libnvidia-ml.so`)         — real-time VRAM / utilization telemetry.
//!
//! ## Zero-copy weight path
//! Weight files are `mmap`ed and page-locked with `cuMemHostRegister`
//! (`CU_MEMHOSTREGISTER_READ_ONLY`), making the page-cache pages directly
//! DMA-able: `cuMemcpyHtoDAsync` streams file -> VRAM with no intermediate
//! staging buffer and no double-buffering on the CPU. Small, latency-critical
//! transfers (token ids, logits) use dedicated pinned bounce buffers.

#![allow(non_camel_case_types, dead_code)]

use crate::traits::Res;
use crate::{err, log};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

// ===========================================================================
// FFI: CUDA Driver API
// ===========================================================================

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;
pub type CUdeviceptr = u64;

const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 0x08;

extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(v: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytes: usize) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    fn cuMemHostRegister_v2(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: CUstream) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuEventCreate(ev: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventRecord(ev: CUevent, stream: CUstream) -> CUresult;
    fn cuEventSynchronize(ev: CUevent) -> CUresult;
    fn cuStreamBeginCapture_v2(s: CUstream, mode: c_uint) -> CUresult;
    fn cuStreamEndCapture(s: CUstream, graph: *mut CUgraph) -> CUresult;
    fn cuGraphInstantiateWithFlags(exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> CUresult;
    fn cuGraphLaunch(exec: CUgraphExec, s: CUstream) -> CUresult;
    fn cuGraphDestroy(graph: CUgraph) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: CUstream) -> CUresult;
    fn cuStreamCreate(s: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamSynchronize(s: CUstream) -> CUresult;
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleLoadDataEx(m: *mut CUmodule, image: *const c_void, n: c_uint, opts: *mut c_uint, vals: *mut *mut c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint,
        shmem: c_uint, stream: CUstream,
        params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
}

// Translate a `CUresult` into a granular `EngineError` (see cu_check below).
// ---------------------------------------------------------------------------
// Fatal-error containment. Sticky CUresults (illegal address/instruction,
// hardware stack fault, launch failure...) poison the ENTIRE primary
// context: every later call on it fails, which turns the server into a
// zombie — HTTP-alive, inference-dead. cu_check marks the poison; the
// ModelManager heals at its admission choke point by resetting the primary
// context and rebuilding. The flight recorder keeps the last device ops so
// the log names the true faulting kernel, not whoever hit the corpse first
// (async faults surface at the NEXT sync point, typically a cuBLAS call).
// ---------------------------------------------------------------------------

static CTX_POISONED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FLIGHT: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

/// Sticky, context-fatal CUresults (CUDA driver docs: "the context cannot
/// be used" after these). 701 (launch out of resources) is NOT sticky.
fn cu_fatal(code: CUresult) -> bool {
    matches!(code, 700 | 702 | 710 | 713 | 714 | 715 | 716 | 717 | 718 | 719)
}

pub fn context_poisoned() -> bool {
    CTX_POISONED.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn record_op(op: String) {
    if let Ok(mut r) = FLIGHT.lock() {
        if r.len() >= 24 {
            r.pop_front();
        }
        r.push_back(op);
    }
}

/// Reset the device's primary context after a fatal error. The driver's
/// fine print makes the naive version fail (field-tested: Retain returned
/// the SAME sticky 715 right after a "successful" reset): the reset only
/// truly clears if (a) our outstanding Retain is released first and (b) no
/// calling thread still has the dead context current. So: release, unbind
/// this thread, reset, and give the driver a few beats before the caller
/// retains fresh. Errors from release are ignored — the context is a
/// corpse and refcount bookkeeping on it is best-effort by definition.
pub fn reset_primary(gpu_index: u32) -> Res<()> {
    unsafe {
        let mut dev: CUdevice = 0;
        cu_check(cuDeviceGet(&mut dev, gpu_index as c_int), "cuDeviceGet")?;
        let _ = cuCtxSetCurrent(std::ptr::null_mut());
        let rel = cuDevicePrimaryCtxRelease_v2(dev);
        if rel != 0 {
            log::warn(&format!("primary ctx release on the dead context: CUresult {} (ignored)", rel));
        }
        for attempt in 0..5 {
            let rc = cuDevicePrimaryCtxReset_v2(dev);
            if rc == 0 {
                CTX_POISONED.store(false, std::sync::atomic::Ordering::Relaxed);
                log::warn("primary CUDA context reset — device state cleared");
                return Ok(());
            }
            log::warn(&format!("cuDevicePrimaryCtxReset attempt {}: CUresult {}", attempt + 1, rc));
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
    Err(err!("cuda", "primary context reset failed after 5 attempts — device requires process restart"))
}

fn cu_check(code: CUresult, what: &str) -> Res<()> {
    if code == 0 {
        return Ok(());
    }
    let mut s: *const c_char = ptr::null();
    let msg = unsafe {
        cuGetErrorString(code, &mut s);
        if s.is_null() { "unknown".to_string() } else { CStr::from_ptr(s).to_string_lossy().into_owned() }
    };
    if cu_fatal(code) && !CTX_POISONED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let recent = FLIGHT
            .lock()
            .map(|r| r.iter().cloned().collect::<Vec<_>>().join(" <- "))
            .unwrap_or_default();
        log::error(&format!(
            "FATAL CUresult {} poisons the CUDA context; last device ops (newest first): {}",
            code, recent
        ));
    }
    Err(err!("cuda", "{} failed: CUresult {} ({})", what, code, msg))
}

// ===========================================================================
// FFI: NVRTC (runtime kernel compilation)
// ===========================================================================

type nvrtcResult = c_int;
type nvrtcProgram = *mut c_void;

extern "C" {
    fn nvrtcCreateProgram(
        p: *mut nvrtcProgram, src: *const c_char, name: *const c_char,
        n_hdr: c_int, hdrs: *const *const c_char, inc: *const *const c_char,
    ) -> nvrtcResult;
    fn nvrtcCompileProgram(p: nvrtcProgram, n: c_int, opts: *const *const c_char) -> nvrtcResult;
    fn nvrtcGetPTXSize(p: nvrtcProgram, n: *mut usize) -> nvrtcResult;
    fn nvrtcGetPTX(p: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    fn nvrtcGetProgramLogSize(p: nvrtcProgram, n: *mut usize) -> nvrtcResult;
    fn nvrtcGetProgramLog(p: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    fn nvrtcDestroyProgram(p: *mut nvrtcProgram) -> nvrtcResult;
}

// ===========================================================================
// FFI: cuBLAS
// ===========================================================================

type cublasHandle = *mut c_void;
type cublasStatus = c_int;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
/// cudaDataType_t
const CUDA_R_16F: c_int = 2;
const CUDA_R_16BF: c_int = 14;
const CUDA_R_32F: c_int = 0;
/// cublasComputeType_t: 32-bit accumulate over 16-bit inputs (tensor cores).
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

// cuBLAS is a runtime OPTION, not a link-time dependency: it is dlopened
// when present (peak tensor-core prefill GEMM) and replaced by the native
// `k_gemm_f16` kernel when absent or disabled with CIMA_NO_CUBLAS=1. This
// is what lets the slim container image ship without ~480 MB of
// libcublas/Lt — the binary's only hard CUDA userspace need is NVRTC.
type CublasGemmExFn = unsafe extern "C" fn(
    h: cublasHandle, ta: c_int, tb: c_int,
    m: c_int, n: c_int, k: c_int,
    alpha: *const c_void,
    a: *const c_void, atype: c_int, lda: c_int,
    b: *const c_void, btype: c_int, ldb: c_int,
    beta: *const c_void,
    c: *mut c_void, ctype: c_int, ldc: c_int,
    compute: c_int, algo: c_int,
) -> cublasStatus;

struct Blas {
    handle: cublasHandle,
    gemm_ex: CublasGemmExFn,
}
// Handle + fn pointer are set once at init and never mutated.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, sym: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

/// Try to bring up cuBLAS from a dlopened library; None ⇒ native GEMM path.
///
/// Determinism-first default: cuBLAS is used ONLY when the operator opts in
/// with `CIMA_CUBLAS=1`. Left off, the prefill GEMM runs the native
/// `k_gemm_f16` kernel, whose reduction order is fixed, so repeated identical
/// greedy requests are bit-reproducible. cuBLAS (even with atomics disabled
/// and an algorithm pinned) selects tensor-core kernels whose accumulation
/// order varies run-to-run on this hardware, which over a long greedy decode
/// flips an argmax tie and changes the output — measured directly (md5
/// differs with cuBLAS on, identical with it off). Enabling `CIMA_CUBLAS=1`
/// trades that reproducibility for faster tensor-core prefill.
/// `CIMA_NO_CUBLAS=1` is still honored as an explicit force-off (it wins over
/// `CIMA_CUBLAS`) so existing scripts keep working.
unsafe fn try_load_cublas(stream: CUstream) -> Option<Blas> {
    let force_off = matches!(
        std::env::var("CIMA_NO_CUBLAS").ok().as_deref().map(str::trim),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    let opt_in = matches!(
        std::env::var("CIMA_CUBLAS").ok().as_deref().map(str::trim),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    if force_off || !opt_in {
        return None;
    }
    let lib = ["libcublas.so.12\0", "libcublas.so.11\0", "libcublas.so\0"]
        .iter()
        .map(|n| dlopen(n.as_ptr() as *const c_char, RTLD_NOW))
        .find(|h| !h.is_null())?;
    let sym = |name: &str| -> *mut c_void {
        let c = std::ffi::CString::new(name).unwrap();
        dlsym(lib, c.as_ptr())
    };
    let create = sym("cublasCreate_v2");
    let set_stream = sym("cublasSetStream_v2");
    let gemm_ex = sym("cublasGemmEx");
    if create.is_null() || set_stream.is_null() || gemm_ex.is_null() {
        return None;
    }
    let create: unsafe extern "C" fn(*mut cublasHandle) -> cublasStatus = std::mem::transmute(create);
    let set_stream: unsafe extern "C" fn(cublasHandle, CUstream) -> cublasStatus = std::mem::transmute(set_stream);
    let mut handle: cublasHandle = std::ptr::null_mut();
    if create(&mut handle) != 0 {
        return None;
    }
    set_stream(handle, stream);
    // Determinism: CUBLAS_GEMM_DEFAULT lets cuBLAS pick atomic /
    // split-K algorithms whose floating-point accumulation ORDER varies
    // launch-to-launch. Over a long greedy decode those sub-ULP differences
    // compound until an argmax tie flips and two runs of the SAME prompt
    // produce different tokens (identical length, different content). Ban
    // atomics so repeated identical requests are bit-reproducible — the
    // documented cuBLAS determinism guarantee. Best-effort: if the symbol
    // is absent (very old cuBLAS) we proceed; the native GEMM fallback is
    // deterministic anyway.
    const CUBLAS_ATOMICS_NOT_ALLOWED: c_int = 0;
    let set_atomics = sym("cublasSetAtomicsMode");
    if !set_atomics.is_null() {
        let set_atomics: unsafe extern "C" fn(cublasHandle, c_int) -> cublasStatus =
            std::mem::transmute(set_atomics);
        set_atomics(handle, CUBLAS_ATOMICS_NOT_ALLOWED);
    }
    Some(Blas { handle, gemm_ex: std::mem::transmute::<*mut c_void, CublasGemmExFn>(gemm_ex) })
}

// ===========================================================================
// FFI: NVML telemetry
// ===========================================================================

type nvmlReturn = c_int;
type nvmlDevice = *mut c_void;

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}
#[repr(C)]
struct NvmlUtilization {
    gpu: c_uint,
    memory: c_uint,
}

extern "C" {
    fn nvmlInit_v2() -> nvmlReturn;
    fn nvmlDeviceGetHandleByIndex_v2(idx: c_uint, dev: *mut nvmlDevice) -> nvmlReturn;
    fn nvmlDeviceGetMemoryInfo(dev: nvmlDevice, mem: *mut NvmlMemory) -> nvmlReturn;
    fn nvmlDeviceGetUtilizationRates(dev: nvmlDevice, u: *mut NvmlUtilization) -> nvmlReturn;
    fn nvmlDeviceGetTemperature(dev: nvmlDevice, sensor: c_uint, t: *mut c_uint) -> nvmlReturn;
}

/// Point-in-time GPU telemetry snapshot (NVML-sourced, driver-truth).
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSnapshot {
    pub vram_total: u64,
    pub vram_used: u64,
    pub vram_free: u64,
    pub util_gpu: u32,
    pub util_mem: u32,
    pub temp_c: u32,
}

// ===========================================================================
// Embedded CUDA C kernels (JIT-compiled via NVRTC at startup)
// ===========================================================================

/// All non-GEMM operators of the transformer, written once in CUDA C.
/// GEMMs go through cuBLAS tensor cores; everything else lives here.
/// `__half` arithmetic is avoided in reductions — accumulation is f32.
/// The CUDA kernel suite, JIT-compiled via NVRTC at startup (PTX cached on
/// disk keyed by source hash and SM version). Kept as a real `.cu` file for
/// tooling and review; `include_str!` keeps the zero-build-step property.
const KERNELS_CU: &str = include_str!("kernels.cu");

// ===========================================================================
// Device / pinned memory RAII wrappers
// ===========================================================================

/// Global counter of live device allocations (engine-side VRAM bookkeeping;
/// NVML provides the driver-truth complement).
static DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// RAII device (VRAM) buffer.
/// An instantiated CUDA graph (see [`CudaCtx::capture_begin`]).
pub struct GraphExec {
    exec: CUgraphExec,
}

unsafe impl Send for GraphExec {}

/// Flash-decode sequence chunk size shared by every family that uses
/// [`CudaCtx::attn_decode_split`]: one block reduces this many KV
/// positions; partials combine per head via log-sum-exp.
pub const ATT_CSZ: usize = 128;

/// Invert the orderable-bits packing used by `k_argmax_softcap` /
/// `k_slot_take`: returns (logit value after any softcap, vocab index).
pub fn unpack_candidate(packed: u64) -> (f32, u32) {
    let o = (packed >> 32) as u32;
    let bits = if o & 0x8000_0000 != 0 { o ^ 0x8000_0000 } else { !o };
    (f32::from_bits(bits), packed as u32)
}

/// See [`CudaCtx::fetch_token_async`]. The packed slot layout is
/// (orderable-float-bits << 32 | index); `wait` returns the index.
pub struct TokenFetch {
    host: *mut c_void,
    event: CUevent,
}

unsafe impl Send for TokenFetch {}

impl TokenFetch {
    pub fn wait(&self) -> Res<u32> {
        cu_check(unsafe { cuEventSynchronize(self.event) }, "cuEventSynchronize")?;
        let packed = unsafe { *(self.host as *const u64) };
        Ok(packed as u32)
    }
}

pub struct DeviceBuf {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

unsafe impl Send for DeviceBuf {}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { cuMemFree_v2(self.ptr) };
            DEVICE_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}

/// RAII pinned (page-locked) host buffer for latency-critical small transfers.
pub struct PinnedBuf {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    /// View as a typed slice. Caller guarantees `T` layout fits.
    pub fn as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.bytes / std::mem::size_of::<T>()) }
    }
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuMemFreeHost(self.ptr) };
        }
    }
}

/// A page-locked registration over an externally-owned mapping (e.g. an mmap
/// of a safetensors file). Unregisters on drop. This is the zero-copy hook.
pub struct HostRegistration {
    ptr: *mut c_void,
    bytes: usize,
}
unsafe impl Send for HostRegistration {}
impl Drop for HostRegistration {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuMemHostUnregister(self.ptr) };
        }
    }
}

// ===========================================================================
// CudaCtx — the one handle everything runs through
// ===========================================================================

/// Owning handle over device, context, stream, cuBLAS and the JIT-compiled
/// kernel module. Exactly one `CudaCtx` exists per process; the inference
/// queue serializes all access to it (single-user GPU constraint).
pub struct CudaCtx {
    /// q8_1 activation scratch for the dp4a resident GEMVs (40 B per 32
    /// elements of the largest k seen), grown only before graphs arm.
    q8_scratch: std::sync::Mutex<Option<DeviceBuf>>,
    dev: CUdevice,
    ctx: CUcontext,
    pub stream: CUstream,
    blas: Option<Blas>,
    module: CUmodule,
    funcs: KernelSet,
    launches: std::sync::atomic::AtomicU64,
    /// CUfunction -> kernel name, for the flight recorder.
    fn_names: std::collections::HashMap<usize, &'static str>,
    nvml: nvmlDevice,
    pub device_name: String,
    pub sm_major: i32,
    pub sm_minor: i32,
    gpu_index: u32,
}

// SAFETY: the CUDA driver API, cuBLAS and NVML are documented thread-safe.
// All kernel-launching code paths are additionally serialized by the strict
// FIFO `GpuQueue` plus the `Mutex<ModelManager>` (single-user GPU constraint),
// and every request thread re-binds the primary context via [`CudaCtx::bind`]
// before touching the device. The raw handles inside are never mutated after
// `init`, so shared references across threads are sound.
unsafe impl Send for CudaCtx {}
unsafe impl Sync for CudaCtx {}

/// Resolved kernel handles (looked up once at startup).
struct KernelSet {
    gemm_f16: CUfunction,
    gather_f16: CUfunction,
    gather_bf16: CUfunction,
    rmsnorm: CUfunction,
    layernorm: CUfunction,
    add: CUfunction,
    swiglu: CUfunction,
    gelu: CUfunction,
    bias: CUfunction,
    rope: CUfunction,
    kv_append: CUfunction,
    attn_decode: CUfunction,
    attn_prefill: CUfunction,
    h2f: CUfunction,
    f2h: CUfunction,
    bf2h: CUfunction,
    meanpool: CUfunction,
    rope2d: CUfunction,
    geglu: CUfunction,
    silu: CUfunction,
    relu: CUfunction,
    glu: CUfunction,
    scalemul: CUfunction,
    mulvec: CUfunction,
    mul_strided: CUfunction,
    clamp: CUfunction,
    dwconv1d: CUfunction,
    audio_attn: CUfunction,
    nf4_dequant: CUfunction,
    gguf_q8_0: CUfunction,
    gguf_q4_0: CUfunction,
    gguf_q4_1: CUfunction,
    gguf_q5_0: CUfunction,
    gguf_q5_1: CUfunction,
    gguf_q4_k: CUfunction,
    gguf_q5_k: CUfunction,
    gguf_q6_k: CUfunction,
    gguf_iq4_xs: CUfunction,
    gguf_q8_0_gemv: CUfunction,
    gguf_q4_0_gemv: CUfunction,
    gguf_q4_1_gemv: CUfunction,
    gguf_q5_0_gemv: CUfunction,
    gguf_q5_1_gemv: CUfunction,
    gguf_q4_k_gemv: CUfunction,
    gguf_q5_k_gemv: CUfunction,
    gguf_q6_k_gemv: CUfunction,
    gguf_iq4_xs_gemv: CUfunction,
    quantize_q8_1: CUfunction,
    gguf_gather: CUfunction,
    nf4_gemv: CUfunction,
    nf4_gemv_ref: CUfunction,
    argmax_softcap: CUfunction,
    argmax_extract: CUfunction,
    pos_bump: CUfunction,
    attn_decode_split: CUfunction,
    attn_reduce: CUfunction,
    gemv_f16: CUfunction,
    apply_penalty: CUfunction,
    slot_take: CUfunction,
    hist_push: CUfunction,
}

impl CudaCtx {
    /// Initialize driver, NVML, primary context, stream, cuBLAS, and JIT the
    /// kernel module for the detected SM architecture.
    /// Make this context current on the *calling* OS thread. Connection
    /// threads in the API server must call this (via `ModelManager::ensure`)
    /// before issuing any driver call; `cuCtxSetCurrent` is a thread-local
    /// pointer swap and costs nanoseconds.
    pub fn bind(&self) {
        unsafe {
            let _ = cuCtxSetCurrent(self.ctx);
        }
    }

    pub fn gpu_index(&self) -> u32 {
        self.gpu_index
    }

    pub fn init(gpu_index: u32) -> Res<CudaCtx> {
        // Eager module loading materializes EVERY kernel in every loaded
        // module at handle-creation time — for cuBLAS that is hundreds of
        // kernels and hundreds of MB of VRAM spent before the first byte
        // of weights. Lazy loading (the documented fix; default only on
        // newer stacks) loads kernels on first call. Respect an explicit
        // user setting; otherwise choose lazy.
        if std::env::var_os("CUDA_MODULE_LOADING").is_none() {
            std::env::set_var("CUDA_MODULE_LOADING", "LAZY");
        }
        let trace = std::env::var_os("CIMA_VRAM_TRACE").is_some();
        let stage = |label: &str| {
            if trace {
                let (mut free, mut total) = (0usize, 0usize);
                if unsafe { cuMemGetInfo_v2(&mut free, &mut total) } == 0 {
                    eprintln!("vram-trace {:<18} free {:>8.3} GiB of {:.3}", label, free as f64 / (1 << 30) as f64, total as f64 / (1 << 30) as f64);
                }
            }
        };
        unsafe {
            cu_check(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            cu_check(cuDeviceGet(&mut dev, gpu_index as c_int), "cuDeviceGet")?;

            let mut name = [0 as c_char; 128];
            cu_check(cuDeviceGetName(name.as_mut_ptr(), 128, dev), "cuDeviceGetName")?;
            let device_name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();

            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75, MINOR=76
            let (mut maj, mut min) = (0, 0);
            cu_check(cuDeviceGetAttribute(&mut maj, 75, dev), "cc major")?;
            cu_check(cuDeviceGetAttribute(&mut min, 76, dev), "cc minor")?;

            let mut ctx: CUcontext = ptr::null_mut();
            cu_check(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            stage("primary-context");

            let mut stream: CUstream = ptr::null_mut();
            cu_check(cuStreamCreate(&mut stream, 1 /*NON_BLOCKING*/), "cuStreamCreate")?;

            let blas = try_load_cublas(stream);
            crate::log::info(match &blas {
                Some(_) => "prefill GEMM: cuBLAS (dlopened, tensor cores — faster, NOT bit-reproducible; opted in via CIMA_CUBLAS=1)",
                None => "prefill GEMM: native k_gemm_f16 kernel (deterministic default; set CIMA_CUBLAS=1 for faster tensor-core prefill)",
            });
            stage("cublas-probe");

            // NVML
            if nvmlInit_v2() != 0 {
                return Err(err!("cuda", "nvmlInit failed — is the Nvidia driver loaded?"));
            }
            let mut nvml: nvmlDevice = ptr::null_mut();
            if nvmlDeviceGetHandleByIndex_v2(gpu_index, &mut nvml) != 0 {
                return Err(err!("cuda", "NVML: no device at index {}", gpu_index));
            }

            // JIT kernels for this exact architecture.
            let ptx = compile_ptx(KERNELS_CU, maj, min)?;
            let mut module: CUmodule = ptr::null_mut();
            {
                // Load with the JIT error buffer attached: a CUresult 218
                // (invalid PTX) without ptxas's own message is undebuggable.
                const CU_JIT_ERROR_LOG_BUFFER: c_uint = 5;
                const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_uint = 6;
                let mut errlog = vec![0u8; 16 * 1024];
                let mut optv: [c_uint; 2] = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
                let mut vals: [*mut c_void; 2] = [errlog.as_mut_ptr() as *mut c_void, errlog.len() as *mut c_void];
                let rc = cuModuleLoadDataEx(&mut module, ptx.as_ptr() as *const c_void, 2, optv.as_mut_ptr(), vals.as_mut_ptr());
                if rc != 0 {
                    let msg = String::from_utf8_lossy(&errlog)
                        .trim_end_matches('\0')
                        .trim()
                        .to_string();
                    return Err(err!(
                        "cuda",
                        "PTX JIT failed (CUresult {}): {}",
                        rc,
                        if msg.is_empty() { "(driver gave no log)".to_string() } else { msg }
                    ));
                }
            }
            stage("jit-module");

            let mut fn_names: std::collections::HashMap<usize, &'static str> = std::collections::HashMap::new();
            let mut f = |n: &'static str| -> Res<CUfunction> {
                let cname = CString::new(n).unwrap();
                let mut func: CUfunction = ptr::null_mut();
                cu_check(cuModuleGetFunction(&mut func, module, cname.as_ptr()), n)?;
                fn_names.insert(func as usize, n);
                Ok(func)
            };
            let funcs = KernelSet {
                gemm_f16: f("k_gemm_f16")?,
                gather_f16: f("k_gather_f16")?,
                gather_bf16: f("k_gather_bf16")?,
                rmsnorm: f("k_rmsnorm")?,
                layernorm: f("k_layernorm")?,
                add: f("k_add")?,
                swiglu: f("k_swiglu")?,
                gelu: f("k_gelu")?,
                bias: f("k_bias")?,
                rope: f("k_rope")?,
                kv_append: f("k_kv_append")?,
                attn_decode: f("k_attn_decode")?,
                attn_prefill: f("k_attn_prefill")?,
                h2f: f("k_h2f")?,
                f2h: f("k_f2h")?,
                bf2h: f("k_bf2h")?,
                meanpool: f("k_meanpool")?,
                rope2d: f("k_rope2d")?,
                geglu: f("k_geglu")?,
                silu: f("k_silu")?,
                relu: f("k_relu")?,
                glu: f("k_glu")?,
                scalemul: f("k_scalemul")?,
                mulvec: f("k_mulvec")?,
                mul_strided: f("k_mul_strided")?,
                clamp: f("k_clamp")?,
                dwconv1d: f("k_dwconv1d")?,
                audio_attn: f("k_audio_attn")?,
                nf4_dequant: f("k_nf4_dequant")?,
                gguf_q8_0: f("k_gguf_q8_0")?,
                gguf_q4_0: f("k_gguf_q4_0")?,
                gguf_q4_1: f("k_gguf_q4_1")?,
                gguf_q5_0: f("k_gguf_q5_0")?,
                gguf_q5_1: f("k_gguf_q5_1")?,
                gguf_q4_k: f("k_gguf_q4_k")?,
                gguf_q5_k: f("k_gguf_q5_k")?,
                gguf_q6_k: f("k_gguf_q6_k")?,
                gguf_iq4_xs: f("k_gguf_iq4_xs")?,
                gguf_q8_0_gemv: f("k_gguf_q8_0_gemv")?,
                gguf_q4_0_gemv: f("k_gguf_q4_0_gemv")?,
                gguf_q4_1_gemv: f("k_gguf_q4_1_gemv")?,
                gguf_q5_0_gemv: f("k_gguf_q5_0_gemv")?,
                gguf_q5_1_gemv: f("k_gguf_q5_1_gemv")?,
                gguf_q4_k_gemv: f("k_gguf_q4_k_gemv")?,
                gguf_q5_k_gemv: f("k_gguf_q5_k_gemv")?,
                gguf_q6_k_gemv: f("k_gguf_q6_k_gemv")?,
                gguf_iq4_xs_gemv: f("k_gguf_iq4_xs_gemv")?,
                quantize_q8_1: f("k_quantize_q8_1")?,
                gguf_gather: f("k_gguf_gather")?,
                argmax_softcap: f("k_argmax_softcap")?,
                argmax_extract: f("k_argmax_extract")?,
                pos_bump: f("k_pos_bump")?,
                nf4_gemv: f("k_nf4_gemv")?,
                nf4_gemv_ref: f("k_nf4_gemv_ref")?,
                attn_decode_split: f("k_attn_decode_split")?,
                attn_reduce: f("k_attn_reduce")?,
                gemv_f16: f("k_gemv_f16")?,
                apply_penalty: f("k_apply_penalty")?,
                slot_take: f("k_slot_take")?,
                hist_push: f("k_hist_push")?,
            };

            log::info(&format!(
                "CUDA ready: {} (sm_{}{}), kernels JIT-compiled, cuBLAS bound",
                device_name, maj, min
            ));

            Ok(CudaCtx {
            gpu_index,
            fn_names,
                q8_scratch: std::sync::Mutex::new(None),
                dev,
                ctx,
                stream,
                blas,
                module,
                funcs,
                launches: std::sync::atomic::AtomicU64::new(0),
                nvml,
                device_name,
                sm_major: maj,
                sm_minor: min,
            })
        }
    }

    // ------------------------------------------------------------- telemetry

    /// Driver-truth telemetry snapshot via NVML.
    pub fn snapshot(&self) -> GpuSnapshot {
        unsafe {
            let mut mem = NvmlMemory { total: 0, free: 0, used: 0 };
            let mut util = NvmlUtilization { gpu: 0, memory: 0 };
            let mut temp: c_uint = 0;
            nvmlDeviceGetMemoryInfo(self.nvml, &mut mem);
            nvmlDeviceGetUtilizationRates(self.nvml, &mut util);
            nvmlDeviceGetTemperature(self.nvml, 0, &mut temp);
            GpuSnapshot {
                vram_total: mem.total,
                vram_used: mem.used,
                vram_free: mem.free,
                util_gpu: util.gpu,
                util_mem: util.memory,
                temp_c: temp,
            }
        }
    }

    /// Free VRAM as reported by the CUDA allocator (slightly stricter than NVML).
    pub fn free_vram(&self) -> Res<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { cu_check(cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo")? };
        Ok((free, total))
    }

    /// Engine-side count of live device allocations.
    pub fn tracked_bytes(&self) -> usize {
        DEVICE_BYTES.load(Ordering::Relaxed)
    }

    // ---------------------------------------------------------------- memory

    /// Allocate VRAM. Errors carry exact requested/free/total numbers.
    pub fn alloc(&self, bytes: usize) -> Res<DeviceBuf> {
        let mut ptr: CUdeviceptr = 0;
        let code = unsafe { cuMemAlloc_v2(&mut ptr, bytes.max(1)) };
        if code != 0 {
            let (free, total) = self.free_vram().unwrap_or((0, 0));
            return Err(err!(
                "cuda",
                "cuMemAlloc({}) failed (CUresult {}). VRAM free={} total={} engine-tracked={}",
                fmt_bytes(bytes), code, fmt_bytes(free), fmt_bytes(total), fmt_bytes(self.tracked_bytes())
            ));
        }
        DEVICE_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Ok(DeviceBuf { ptr, bytes })
    }

    /// Allocate pinned host memory for low-latency H<->D bounce buffers.
    pub fn alloc_pinned(&self, bytes: usize) -> Res<PinnedBuf> {
        let mut p: *mut c_void = ptr::null_mut();
        cu_check(unsafe { cuMemAllocHost_v2(&mut p, bytes.max(1)) }, "cuMemAllocHost")?;
        Ok(PinnedBuf { ptr: p, bytes })
    }

    /// Page-lock an external read-only mapping (the zero-copy weight path).
    ///
    /// # Safety
    /// `ptr` must point to a valid readable mapping of at least `bytes`
    /// bytes that stays alive and unmoved until the returned
    /// [`HostRegistration`] is dropped.
    pub unsafe fn register_host(&self, ptr: *mut c_void, bytes: usize) -> Res<HostRegistration> {
        cu_check(
            unsafe { cuMemHostRegister_v2(ptr, bytes, CU_MEMHOSTREGISTER_READ_ONLY) },
            "cuMemHostRegister",
        )?;
        Ok(HostRegistration { ptr, bytes })
    }

    pub fn htod(&self, dst: &DeviceBuf, src: &[u8]) -> Res<()> {
        debug_assert!(src.len() <= dst.bytes);
        cu_check(
            unsafe { cuMemcpyHtoDAsync_v2(dst.ptr, src.as_ptr() as *const c_void, src.len(), self.stream) },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn dtoh(&self, dst: &mut [u8], src: &DeviceBuf) -> Res<()> {
        debug_assert!(dst.len() <= src.bytes);
        cu_check(
            unsafe { cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.ptr, dst.len(), self.stream) },
            "cuMemcpyDtoHAsync",
        )
    }

    /// dtoh from a raw device address (debug/self-test paths). The copy is
    /// synchronized before returning so the host slice is immediately valid.
    pub fn dtoh_at(&self, dst: &mut [u8], src: CUdeviceptr) -> Res<()> {
        cu_check(
            unsafe { cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src, dst.len(), self.stream) },
            "cuMemcpyDtoHAsync",
        )?;
        self.sync()
    }

    pub fn dtod(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Res<()> {
        cu_check(unsafe { cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream) }, "cuMemcpyDtoD")
    }

    pub fn memset(&self, buf: &DeviceBuf) -> Res<()> {
        cu_check(unsafe { cuMemsetD8Async(buf.ptr, 0, buf.bytes, self.stream) }, "cuMemsetD8")
    }

    /// Capture everything enqueued between begin/end into a replayable
    /// graph — lever 2 of the performance contract ([`crate::traits::PerfLevers`]).
    ///
    /// # Capture rules (violations fail `capture_end`, so fall back gracefully)
    /// - Only stream-ordered work: kernel launches, async memsets/copies.
    ///   No `dtoh`/`htod` (synchronous), no `sync`, no allocation.
    /// - Anything position-dependent must read the position from device
    ///   memory (`pos_dev` parameters) and the captured step must end with
    ///   [`Self::pos_bump`], or the replay would be frozen at the captured
    ///   position.
    /// - cuBLAS calls capture on the bound stream; warm them (prefill)
    ///   before capturing so no lazy workspace allocation happens inside.
    ///
    /// Host-side stages (e.g. a mmap gather) simply stay outside: a
    /// *partial* graph still collapses hundreds of launches into one.
    /// graph: one `cuGraphLaunch` then substitutes hundreds of per-kernel
    /// submissions (the dominant decode cost on high-launch-overhead
    /// stacks such as WSL2). Position-dependent kernels read a device
    /// counter, so a single captured step replays at every position.
    pub fn capture_begin(&self) -> Res<()> {
        cu_check(unsafe { cuStreamBeginCapture_v2(self.stream, 0) }, "cuStreamBeginCapture")
    }

    pub fn capture_end(&self) -> Res<GraphExec> {
        let mut graph: CUgraph = std::ptr::null_mut();
        cu_check(unsafe { cuStreamEndCapture(self.stream, &mut graph) }, "cuStreamEndCapture")?;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let r = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { cuGraphDestroy(graph) };
        cu_check(r, "cuGraphInstantiate")?;
        Ok(GraphExec { exec })
    }

    pub fn graph_launch(&self, g: &GraphExec) -> Res<()> {
        self.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        cu_check(unsafe { cuGraphLaunch(g.exec, self.stream) }, "cuGraphLaunch")
    }

    /// Kernel-submission counter (profiling): graphs count as one.
    pub fn launch_count(&self) -> u64 {
        self.launches.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn sync(&self) -> Res<()> {
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize")
    }

    // --------------------------------------------------------------- kernels

    fn launch(&self, f: CUfunction, grid: (u32, u32, u32), block: u32, shmem: u32, args: &mut [*mut c_void]) -> Res<()> {
        self.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        record_op(format!(
            "{}[{}x{}x{}/{}]",
            self.fn_names.get(&(f as usize)).copied().unwrap_or("?kernel"),
            grid.0, grid.1, grid.2, block
        ));
        cu_check(
            unsafe {
                cuLaunchKernel(
                    f, grid.0, grid.1, grid.2, block, 1, 1, shmem, self.stream,
                    args.as_mut_ptr(), ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// `out[row,:] = table[ids[row],:]` — embedding lookup. `table_bf16`
    /// selects the source dtype; output is always f16 activations.
    pub fn gather(&self, table: u64, table_bf16: bool, ids: u64, out: u64, rows: usize, hidden: usize) -> Res<()> {
        let (mut t, mut i, mut o, mut h) = (table, ids, out, hidden as c_int);
        let mut args = [
            &mut t as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        let f = if table_bf16 { self.funcs.gather_bf16 } else { self.funcs.gather_f16 };
        self.launch(f, (rows as u32, 1, 1), 256, 0, &mut args)
    }

    /// RMSNorm over `rows` rows of width `n`.
    pub fn rmsnorm(&self, x: u64, w: u64, y: u64, rows: usize, n: usize, eps: f32) -> Res<()> {
        let block = 256u32;
        let (mut xa, mut wa, mut ya, mut na, mut ea) = (x, w, y, n as c_int, eps);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.rmsnorm, (rows as u32, 1, 1), block, block * 4, &mut args)
    }

    /// LayerNorm `[rows, n]` with weight + bias (ViT / audio encoder towers).
    pub fn layernorm(&self, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, eps: f32) -> Res<()> {
        let block = 256u32;
        let (mut xa, mut wa, mut ba, mut ya, mut na, mut ea) = (x, w, b, y, n as c_int, eps);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.layernorm, (rows as u32, 1, 1), block, block * 4, &mut args)
    }

    /// `a += b` over `n` f16 elements.
    pub fn add(&self, a: u64, b: u64, n: usize) -> Res<()> {
        let (mut aa, mut ba, mut na) = (a, b, n as c_int);
        let mut args = [
            &mut aa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.add, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// SwiGLU activation: `gate = silu(gate) * up`.
    pub fn swiglu(&self, gate: u64, up: u64, n: usize) -> Res<()> {
        let (mut g, mut u, mut na) = (gate, up, n as c_int);
        let mut args = [
            &mut g as *mut _ as *mut c_void,
            &mut u as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.swiglu, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// GELU in place over `n` f16 elements (vision/audio towers).
    pub fn gelu(&self, x: u64, n: usize) -> Res<()> {
        let (mut xa, mut na) = (x, n as c_int);
        let mut args = [&mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        self.launch(self.funcs.gelu, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Broadcast bias add over `rows` rows of width `n`.
    pub fn bias(&self, x: u64, b: u64, rows: usize, n: usize) -> Res<()> {
        let total = rows * n;
        let (mut xa, mut ba, mut r, mut na) = (x, b, rows as c_int, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.bias, ((total as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Apply rotary position embeddings in place over `[rows, heads, dim]`.
    /// Rotary embedding in place over `[rows, heads*dim]`. `nfreqs` is the
    /// number of nonzero rotary frequencies (`dim/2` for classic RoPE; fewer
    /// for Gemma4 "proportional" RoPE, where the remaining pairs pass through).
    pub fn rope(&self, x: u64, rows: usize, heads: usize, dim: usize, pos0: usize, theta: f32, nfreqs: usize, pos_dev: u64, factors: u64) -> Res<()> {
        let (mut xa, mut ha, mut da, mut pa, mut ta, mut fa, mut pd, mut fc) =
            (x, heads as c_int, dim as c_int, pos0 as c_int, theta, nfreqs as c_int, pos_dev, factors);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut ta as *mut _ as *mut c_void,
            &mut fa as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut fc as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.rope, (rows as u32, heads as u32, 1), ((dim / 2) as u32).max(32), 0, &mut args)
    }


    /// Append `rows` K/V rows into the layer cache at absolute position `pos`.
    pub fn kv_append(&self, k: u64, v: u64, kc: u64, vc: u64, rows: usize, kv_heads: usize, dim: usize, pos: usize, max_seq: usize, pos_dev: u64) -> Res<()> {
        let (mut ka, mut va, mut kca, mut vca, mut pd) = (k, v, kc, vc, pos_dev);
        let (mut h, mut d, mut p, mut m) = (kv_heads as c_int, dim as c_int, pos as c_int, max_seq as c_int);
        let mut args = [
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut kca as *mut _ as *mut c_void,
            &mut vca as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.kv_append, (rows as u32, 1, 1), 256, 0, &mut args)
    }

    /// Warp-per-row decode GEMV with fused epilogues (see kernels.cu).
    /// `bias` 0 = none. Modes: 0 plain, 1 `y[row] += dot`, 2 silu(gate)·up
    /// over a row-concatenated gate|up matrix (`n` = intermediate size).
    pub fn gemv_f16(&self, w: u64, x: u64, y: u64, bias: u64, n: usize, k: usize, mode: i32) -> Res<()> {
        debug_assert!(k % 2 == 0, "half2 loads require even k");
        let (mut wa, mut xa, mut ya, mut ba) = (w, x, y, bias);
        let (mut na, mut ka, mut ma) = (n as c_int, k as c_int, mode as c_int);
        let mut args = [
            &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ba as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut ma as *mut _ as *mut c_void,
        ];
        const WARPS: usize = 4;
        let blocks = ((n + WARPS - 1) / WARPS) as u32;
        self.launch(self.funcs.gemv_f16, (blocks, 1, 1), (WARPS * 32) as u32, 0, &mut args)
    }

    /// Flash-decode pair: per-(head, chunk) partial attention + per-head
    /// log-sum-exp reduction. The grid covers max_seq, so a captured graph
    /// replays at any live length (`pos_dev` gates active chunks).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_split(&self, q: u64, kc: u64, vc: u64, part: u64, q_heads: usize, kv_heads: usize,
        dim: usize, seq: usize, max_seq: usize, csz: usize, n_chunks: usize, scale: f32, window: usize, pos_dev: u64) -> Res<()> {
        let (mut qa, mut ka, mut va, mut pa) = (q, kc, vc, part);
        let (mut qh, mut kh, mut d) = (q_heads as c_int, kv_heads as c_int, dim as c_int);
        let (mut s, mut ms) = (seq as c_int, max_seq as c_int);
        let (mut cs, mut nc, mut sc, mut wi, mut pd) = (csz as c_int, n_chunks as c_int, scale, window as c_int, pos_dev);
        let mut args = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void,
            &mut qh as *mut _ as *mut c_void, &mut kh as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut cs as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
            &mut wi as *mut _ as *mut c_void, &mut pd as *mut _ as *mut c_void,
        ];
        debug_assert!(dim % 32 == 0 && dim <= 256, "warp-register accumulator bounds");
        let threads = 128u32; // 4 warps striding the chunk's positions
        let shmem = (4 * (dim + 2) * 4) as u32;
        self.launch(self.funcs.attn_decode_split, ((q_heads * n_chunks) as u32, 1, 1), threads, shmem, &mut args)
    }

    pub fn attn_reduce(&self, part: u64, out: u64, q_heads: usize, dim: usize, csz: usize,
        n_chunks: usize, seq: usize, window: usize, pos_dev: u64) -> Res<()> {
        let (mut pa, mut oa) = (part, out);
        let (mut d, mut cs, mut nc, mut s) = (dim as c_int, csz as c_int, n_chunks as c_int, seq as c_int);
        let (mut wi, mut pd) = (window as c_int, pos_dev);
        let mut args = [
            &mut pa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void, &mut cs as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut wi as *mut _ as *mut c_void, &mut pd as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.attn_reduce, (q_heads as u32, 1, 1), 64, (dim * 4) as u32, &mut args)
    }

    /// Fused single-token attention over the KV cache (decode path).
    // Each argument is a distinct kernel parameter (q/k/v/out pointers plus
    // head/dim/seq/scale/window geometry); a params struct would only add
    // marshalling on a hot launch path.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(&self, q: u64, kc: u64, vc: u64, out: u64, q_heads: usize, kv_heads: usize, dim: usize, seq: usize, max_seq: usize, scale: f32, window: usize, pos_dev: u64) -> Res<()> {
        let block = 128u32;
        let shmem = (block as usize + dim) * 4;
        let (mut qa, mut ka, mut va, mut oa, mut pd) = (q, kc, vc, out, pos_dev);
        let (mut qh, mut kh, mut da, mut sa, mut ma, mut sc, mut wi) =
            (q_heads as c_int, kv_heads as c_int, dim as c_int, seq as c_int, max_seq as c_int, scale, window as c_int);
        let mut args = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut qh as *mut _ as *mut c_void,
            &mut kh as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut ma as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut wi as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.attn_decode, (q_heads as u32, 1, 1), block, shmem as u32, &mut args)
    }


    /// Causal prefill attention for `rows` new tokens starting at `pos0`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(&self, q: u64, kc: u64, vc: u64, out: u64, rows: usize, q_heads: usize, kv_heads: usize, dim: usize, pos0: usize, max_seq: usize, causal: bool, scale: f32, window: usize, blkid: u64) -> Res<()> {
        let block = 128u32;
        let shmem = (block as usize + dim) * 4;
        let (mut qa, mut ka, mut va, mut oa, mut ba) = (q, kc, vc, out, blkid);
        let (mut qh, mut kh, mut da, mut pa, mut ma, mut na, mut ca, mut sc, mut wi) = (
            q_heads as c_int, kv_heads as c_int, dim as c_int, pos0 as c_int, max_seq as c_int,
            rows as c_int, causal as c_int, scale, window as c_int,
        );
        let mut args = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut qh as *mut _ as *mut c_void,
            &mut kh as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut ma as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut wi as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.attn_prefill, (rows as u32, q_heads as u32, 1), block, shmem as u32, &mut args)
    }

    /// 2-D rotary embedding for the Gemma4 vision tower: first half of each
    /// head rotates with the patch x-coordinate, second half with y.
    pub fn rope2d(&self, x: u64, posx: u64, posy: u64, rows: usize, heads: usize, dim: usize, theta: f32) -> Res<()> {
        let (mut xa, mut pxa, mut pya, mut ha, mut da, mut ta) = (x, posx, posy, heads as c_int, dim as c_int, theta);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut pxa as *mut _ as *mut c_void,
            &mut pya as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut ta as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.rope2d, (rows as u32, heads as u32, 1), ((dim / 2) as u32).max(32), 0, &mut args)
    }

    /// GeGLU in place over `gate`: `gate = gelu_tanh(gate) * up`.
    pub fn geglu(&self, gate: u64, up: u64, n: usize) -> Res<()> {
        let (mut g, mut u, mut na) = (gate, up, n as c_int);
        let mut args = [&mut g as *mut _ as *mut c_void, &mut u as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        self.launch(self.funcs.geglu, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// SiLU in place.
    pub fn silu(&self, x: u64, n: usize) -> Res<()> {
        let (mut xa, mut na) = (x, n as c_int);
        let mut args = [&mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        self.launch(self.funcs.silu, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// ReLU in place.
    pub fn relu(&self, x: u64, n: usize) -> Res<()> {
        let (mut xa, mut na) = (x, n as c_int);
        let mut args = [&mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        self.launch(self.funcs.relu, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// GLU: `y[r, :n] = x[r, :n] * sigmoid(x[r, n:2n])`.
    pub fn glu(&self, x: u64, y: u64, rows: usize, n: usize) -> Res<()> {
        let (mut xa, mut ya, mut ra, mut na) = (x, y, rows as c_int, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.glu, (((rows * n) as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Scalar multiply in place: `x *= s`.
    pub fn scalemul(&self, x: u64, s: f32, n: usize) -> Res<()> {
        let (mut xa, mut sa, mut na) = (x, s, n as c_int);
        let mut args = [&mut xa as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        self.launch(self.funcs.scalemul, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Per-channel vector multiply: `x[r, c] *= s[c]`.
    pub fn mulvec(&self, x: u64, s: u64, rows: usize, n: usize) -> Res<()> {
        let (mut xa, mut sa, mut ra, mut na) = (x, s, rows as c_int, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.mulvec, (((rows * n) as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Strided multiply: `a[r, c] *= b[r*stride + off + c]` (PLE gating).
    pub fn mul_strided(&self, a: u64, b: u64, rows: usize, n: usize, stride: usize, off: usize) -> Res<()> {
        let (mut aa, mut ba, mut ra, mut na, mut st, mut of) =
            (a, b, rows as c_int, n as c_int, stride as c_int, off as c_int);
        let mut args = [
            &mut aa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut of as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.mul_strided, (((rows * n) as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Clamp in place: `x = clamp(x, lo, hi)`.
    pub fn clampk(&self, x: u64, lo: f32, hi: f32, n: usize) -> Res<()> {
        let (mut xa, mut la, mut ha, mut na) = (x, lo, hi, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut la as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.clamp, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Causal depthwise conv1d: `x,y [seq, C]`, `w [C, K]`.
    pub fn dwconv1d(&self, x: u64, w: u64, y: u64, seq: usize, c: usize, k: usize) -> Res<()> {
        let (mut xa, mut wa, mut ya, mut sa, mut ca, mut ka) = (x, w, y, seq as c_int, c as c_int, k as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.dwconv1d, (((seq * c) as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Fused Gemma4 audio attention (chunked local + relative bias + tanh cap).
    #[allow(clippy::too_many_arguments)]
    pub fn audio_attn(&self, q: u64, k: u64, v: u64, relk: u64, out: u64, seq: usize, heads: usize, dim: usize, chunk: usize, past: usize, cap: f32, invalid: f32) -> Res<()> {
        let block = 64u32;
        let shmem = (block as usize + dim) * 4;
        let (mut qa, mut ka, mut va, mut ra, mut oa) = (q, k, v, relk, out);
        let (mut sa, mut ha, mut da, mut ca, mut pa, mut cp, mut inv) =
            (seq as c_int, heads as c_int, dim as c_int, chunk as c_int, past as c_int, cap, invalid);
        let mut args = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut inv as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.audio_attn, (seq as u32, heads as u32, 1), block, shmem as u32, &mut args)
    }

    /// Begin an asynchronous 8-byte fetch of the argmax slot: enqueues the
    /// copy + an event on the stream and returns immediately, so the next
    /// step's kernels can be queued behind it. `TokenFetch::wait` blocks
    /// only until the copy lands (not until the queue drains) — the PCIe
    /// hop hides behind the following step's compute.
    pub fn fetch_token_async(&self, slot: &DeviceBuf, fetch: &TokenFetch) -> Res<()> {
        cu_check(
            unsafe { cuMemcpyDtoHAsync_v2(fetch.host, slot.ptr, 8, self.stream) },
            "cuMemcpyDtoHAsync",
        )?;
        cu_check(unsafe { cuEventRecord(fetch.event, self.stream) }, "cuEventRecord")
    }

    /// Pinned 8-byte landing slot + completion event for [`fetch_token_async`].
    pub fn token_fetch(&self) -> Res<TokenFetch> {
        let mut host: *mut c_void = std::ptr::null_mut();
        cu_check(unsafe { cuMemAllocHost_v2(&mut host, 8) }, "cuMemAllocHost")?;
        let mut event: CUevent = std::ptr::null_mut();
        cu_check(unsafe { cuEventCreate(&mut event, 0) }, "cuEventCreate")?;
        Ok(TokenFetch { host, event })
    }

    /// Fused softcap + argmax over an f16 logits row on device: returns the
    /// winning index after an 8-byte copy (the greedy fast path — the 1 MB
    /// logits row never crosses PCIe). `slot` is any 8-byte device scratch.
    /// Enqueue cap+argmax into `slot` (no readback — compose with
    /// [`Self::argmax_to_ids`] and/or [`Self::fetch_token_async`]).
    pub fn argmax_softcap_enqueue(&self, logits_h: u64, slot: &DeviceBuf, n: usize, cap: f32) -> Res<()> {
        self.memset(slot)?;
        let (mut xa, mut oa) = (logits_h, slot.ptr);
        let (mut na, mut ca) = (n as c_int, cap);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
        ];
        let blocks = (((n + 255) / 256).min(1024)) as u32;
        self.launch(self.funcs.argmax_softcap, (blocks, 1, 1), 256, 0, &mut args)
    }

    /// Unpack the slot's winning index into `ids[0]` on device, enabling the
    /// next step's embedding gather without a host round-trip.
    pub fn argmax_to_ids(&self, slot: &DeviceBuf, ids: &DeviceBuf) -> Res<()> {
        let (mut sa, mut ia) = (slot.ptr, ids.ptr);
        let mut args = [&mut sa as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void];
        self.launch(self.funcs.argmax_extract, (1, 1, 1), 32, 0, &mut args)
    }

    /// Advance the device position counter (closes a captured decode step).
    pub fn pos_bump(&self, pos_dev: &DeviceBuf) -> Res<()> {
        let mut pa = pos_dev.ptr;
        let mut args = [&mut pa as *mut _ as *mut c_void];
        self.launch(self.funcs.pos_bump, (1, 1, 1), 32, 0, &mut args)
    }

    /// Repeat-penalty over the logits row from the device-maintained
    /// occurrence counts (`rp^count`, sign-aware like the host sampler).
    pub fn apply_penalty(&self, logits: u64, counts: u64, rp: f32, n: usize) -> Res<()> {
        let (mut xa, mut ca, mut ra, mut na) = (logits, counts, rp, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.apply_penalty, (((n + 255) / 256) as u32, 1, 1), 256, 0, &mut args)
    }

    /// Enqueue extraction of the top-`k` candidates in descending order:
    /// `k` rounds of (cap+argmax into `slot`, record into `cands[j]`, mask
    /// the winner). All nodes are graph-capturable; `cands` ends holding
    /// `k` packed (orderable-bits, index) pairs — see [`unpack_candidate`].
    pub fn topk_enqueue(&self, logits: u64, slot: &DeviceBuf, cands: u64, n: usize, cap: f32, k: usize) -> Res<()> {
        for j in 0..k {
            self.argmax_softcap_enqueue(logits, slot, n, cap)?;
            let (mut sa, mut oa, mut xa) = (slot.ptr, cands + (j * 8) as u64, logits);
            let mut args = [
                &mut sa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
                &mut xa as *mut _ as *mut c_void,
            ];
            self.launch(self.funcs.slot_take, (1, 1, 1), 32, 0, &mut args)?;
        }
        Ok(())
    }

    /// Slide the device penalty window by one token (ids[0]); `idx` seeds
    /// prompt history when `pos_dev` is 0.
    pub fn hist_push(&self, ring: u64, counts: u64, ids: u64, idx: usize, pos_dev: u64) -> Res<()> {
        let (mut ra, mut ca, mut ia) = (ring, counts, ids);
        let (mut ix, mut pd) = (idx as c_int, pos_dev);
        let mut args = [
            &mut ra as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void, &mut ix as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.hist_push, (1, 1, 1), 32, 0, &mut args)
    }

    /// Synchronous cap+argmax: enqueue + 8-byte readback.
    pub fn argmax_softcap(&self, logits_h: u64, slot: &DeviceBuf, n: usize, cap: f32) -> Res<u32> {
        self.argmax_softcap_enqueue(logits_h, slot, n, cap)?;
        let mut out = [0u8; 8];
        self.dtoh_at(&mut out, slot.ptr)?;
        Ok(u32::from_le_bytes([out[0], out[1], out[2], out[3]]))
    }

    /// Expand a packed NF4/FP4 weight into f16 (prefill scratch path).
    /// Fused resident GEMV over a packed GGUF weight: `y[n] = x[k]·W^T`
    /// with the dequantization happening in registers — the weight is read
    /// at its packed width (~4.5 bits/elem for Q4_K), which is both the
    /// VRAM win and, decode being bandwidth-bound, the speed win.
    /// `mode 0`: write (+bias);  `mode 1`: accumulate into y (+bias).
    /// Grow (or fetch) the q8_1 activation scratch for GEMVs of width `k`.
    /// MUST be called at weight-load time for every gguf linear: growth
    /// allocates, and allocation inside a CUDA-graph capture is illegal —
    /// pre-growing here keeps captured pointers stable.
    pub fn ensure_q8_scratch(&self, k: usize) -> Res<u64> {
        let mut g = self.q8_scratch.lock().unwrap();
        let need = (k / 32) * 40;
        if g.as_ref().map(|b| b.bytes < need).unwrap_or(true) {
            *g = Some(self.alloc(need.max(1 << 16))?);
        }
        Ok(g.as_ref().unwrap().ptr)
    }

    pub fn gguf_gemv(&self, fmt: crate::traits::DType, x: u64, w: u64, bias: u64, y: u64, n: usize, k: usize, mode: i32) -> Res<()> {
        use crate::traits::DType as D;
        let f = match fmt {
            D::GgufQ8_0 => self.funcs.gguf_q8_0_gemv,
            D::GgufQ4_0 => self.funcs.gguf_q4_0_gemv,
            D::GgufQ4_1 => self.funcs.gguf_q4_1_gemv,
            D::GgufQ5_0 => self.funcs.gguf_q5_0_gemv,
            D::GgufQ5_1 => self.funcs.gguf_q5_1_gemv,
            D::GgufQ4K => self.funcs.gguf_q4_k_gemv,
            D::GgufQ5K => self.funcs.gguf_q5_k_gemv,
            D::GgufQ6K => self.funcs.gguf_q6_k_gemv,
            D::GgufIQ4XS => self.funcs.gguf_iq4_xs_gemv,
            other => return Err(err!("cuda", "gguf_gemv: {:?} is not a gguf block format", other)),
        };
        let nb = k / 32;
        let xq = self.ensure_q8_scratch(k)?;
        {
            let (mut xa, mut qa, mut ka) = (x, xq, k as c_int);
            let mut args = [
                &mut xa as *mut _ as *mut c_void,
                &mut qa as *mut _ as *mut c_void,
                &mut ka as *mut _ as *mut c_void,
            ];
            self.launch(self.funcs.quantize_q8_1, (nb as u32, 1, 1), 32, 0, &mut args)?;
        }
        let (mut xa, mut wa, mut ba, mut ya, mut na, mut ka, mut ma) =
            (xq, w, bias, y, n as c_int, k as c_int, mode as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut ma as *mut _ as *mut c_void,
        ];
        self.launch(f, ((n as u32 + 7) / 8, 1, 1), 256, 0, &mut args)
    }

    /// Embedding gather straight from a packed GGUF table    /// Embedding gather straight from a packed GGUF table    /// Embedding gather straight from a packed GGUF table    /// Embedding gather straight from a packed GGUF table: `out[i] =
    /// dequant(table[ids[i]])`, one thread per output element. `fmt` rides
    /// as the ggml type id so a single kernel serves every block format
    /// (uniform branch — no divergence).
    pub fn gguf_gather(&self, fmt: crate::traits::DType, table: u64, ids: u64, out: u64, n: usize, hidden: usize) -> Res<()> {
        use crate::traits::DType as D;
        let (id, row_bytes): (i32, usize) = match fmt {
            D::GgufQ8_0 => (8, hidden / 32 * 34),
            D::GgufQ4_0 => (2, hidden / 32 * 18),
            D::GgufQ4_1 => (3, hidden / 32 * 20),
            D::GgufQ5_0 => (6, hidden / 32 * 22),
            D::GgufQ5_1 => (7, hidden / 32 * 24),
            D::GgufQ4K => (12, hidden / 256 * 144),
            D::GgufQ5K => (13, hidden / 256 * 176),
            D::GgufQ6K => (14, hidden / 256 * 210),
            D::GgufIQ4XS => (23, hidden / 256 * 136),
            other => return Err(err!("cuda", "gguf_gather: {:?} is not a gguf block format", other)),
        };
        let elems = n * hidden;
        let (mut ta, mut ia, mut oa, mut na, mut ha, mut fa, mut ra) =
            (table, ids, out, n as c_int, hidden as c_int, id as c_int, row_bytes as c_int);
        let mut args = [
            &mut ta as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut fa as *mut _ as *mut c_void,
            &mut ra as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.gguf_gather, ((elems as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// GGUF block dequantization to f16 on device — `fmt` selects the
    /// codec (bit-exact mirrors of `quant::gguf`'s host decoders; one
    /// thread per super-block).
    pub fn gguf_dequant(&self, fmt: crate::traits::DType, src: u64, out: u64, nblocks: usize) -> Res<()> {
        use crate::traits::DType as D;
        let f = match fmt {
            D::GgufQ8_0 => self.funcs.gguf_q8_0,
            D::GgufQ4_0 => self.funcs.gguf_q4_0,
            D::GgufQ4_1 => self.funcs.gguf_q4_1,
            D::GgufQ5_0 => self.funcs.gguf_q5_0,
            D::GgufQ5_1 => self.funcs.gguf_q5_1,
            D::GgufQ4K => self.funcs.gguf_q4_k,
            D::GgufQ5K => self.funcs.gguf_q5_k,
            D::GgufQ6K => self.funcs.gguf_q6_k,
            D::GgufIQ4XS => self.funcs.gguf_iq4_xs,
            other => return Err(err!("cuda", "gguf_dequant: {:?} is not a gguf block format", other)),
        };
        let elems = nblocks * crate::traits::block_elems(fmt);
        let (mut sa, mut oa, mut na) = (src, out, nblocks as c_int);
        let mut args = [
            &mut sa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(f, ((elems as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    pub fn nf4_dequant(&self, packed: u64, absmax: u64, qmap: u64, out: u64, n: usize, blocksize: usize) -> Res<()> {
        let (mut pa, mut aa, mut qa, mut oa, mut na, mut ba) =
            (packed, absmax, qmap, out, n as c_int, blocksize as c_int);
        let mut args = [
            &mut pa as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut qa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.nf4_dequant, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// Fused single-row GEMV over a packed NF4/FP4 weight (decode path).
    #[allow(clippy::too_many_arguments)]
    pub fn nf4_gemv(&self, x: u64, packed: u64, absmax: u64, qmap: u64, y: u64, n: usize, k: usize, blocksize: usize) -> Res<()> {
        let (mut xa, mut pa, mut aa, mut qa, mut ya) = (x, packed, absmax, qmap, y);
        let (mut na, mut ka, mut ba) = (n as c_int, k as c_int, blocksize as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void, &mut qa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void, &mut ba as *mut _ as *mut c_void,
        ];
        if k % 8 == 0 && blocksize % 8 == 0 {
            // Warp-per-row fast path (4 warps per block).
            let blocks = ((n + 3) / 4) as u32;
            self.launch(self.funcs.nf4_gemv, (blocks, 1, 1), 128, 0, &mut args)
        } else {
            let block = 256u32;
            self.launch(self.funcs.nf4_gemv_ref, (n as u32, 1, 1), block, block * 4, &mut args)
        }
    }


    /// f16 -> f32 device copy.
    pub fn h2f(&self, x: u64, y: u64, n: usize) -> Res<()> {
        let (mut xa, mut ya, mut na) = (x, y, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.h2f, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// f32 -> f16 device copy.
    pub fn f2h(&self, x: u64, y: u64, n: usize) -> Res<()> {
        let (mut xa, mut ya, mut na) = (x, y, n as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.f2h, ((n as u32 + 255) / 256, 1, 1), 256, 0, &mut args)
    }

    /// bf16 -> f16 device conversion (weight normalization at load time).
    pub fn bf2h(&self, x: u64, y: u64, n: usize) -> Res<()> {
        let (mut xa, mut ya, mut na) = (x, y, n);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.bf2h, (((n + 255) / 256) as u32, 1, 1), 256, 0, &mut args)
    }

    /// Mean-pool `[rows, hidden]` f16 into `hidden` f32 values.
    pub fn meanpool(&self, x: u64, out: u64, rows: usize, hidden: usize) -> Res<()> {
        let (mut xa, mut oa, mut r, mut h) = (x, out, rows as c_int, hidden as c_int);
        let mut args = [
            &mut xa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch(self.funcs.meanpool, (((hidden + 255) / 256) as u32, 1, 1), 256, 0, &mut args)
    }

    // ------------------------------------------------------------------ GEMM

    /// `C[m,n] = A[m,k] · B[n,k]^T` — the canonical "activation × weight^T"
    /// projection. A, B, C are row-major f16 on device; accumulation is f32 on
    /// tensor cores. (Implemented as column-major cuBLAS `B^T·A^T` identity.)
    /// `gemm_f16` writing into a COLUMN RANGE of a wider row-major output:
    /// C_full is [m, ldn] and this call fills columns [0..n) starting at
    /// the pointer (caller offsets to the slab's first column). Same
    /// transpose trick; only ldc changes.
    pub fn gemm_strided_out(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, ldn: usize) -> Res<()> {
        self.gemm_dispatch(a, b, c, m, n, k, ldn)
    }

    /// One dispatcher for both public GEMMs: cuBLAS when dlopened, the
    /// native tiled kernel otherwise. `ldc` is C's row stride in elements
    /// (== n for the plain call, wider for the column-range variant).
    fn gemm_dispatch(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, ldc: usize) -> Res<()> {
        record_op(format!("gemm(m={},n={},k={})", m, n, k));
        self.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(blas) = &self.blas {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            // cuBLAS is the opt-in speed path (CIMA_CUBLAS=1); the default
            // build never reaches here. Determinism is provided by the native
            // kernel default, so here we just use the fastest heuristic.
            let st = unsafe {
                (blas.gemm_ex)(
                    blas.handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n as c_int, m as c_int, k as c_int,
                    &alpha as *const f32 as *const c_void,
                    b as *const c_void, CUDA_R_16F, k as c_int,
                    a as *const c_void, CUDA_R_16F, k as c_int,
                    &beta as *const f32 as *const c_void,
                    c as *mut c_void, CUDA_R_16F, ldc as c_int,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
                )
            };
            if st != 0 {
                return Err(err!("cuda", "cublasGemmEx(m={},n={},k={},ldc={}) failed: status {}", m, n, k, ldc, st));
            }
            return Ok(());
        }
        self.gemm_f16_native(a, b, c, m, n, k, ldc)
    }

    /// The cuBLAS-free path (also directly testable by `selftest gemm`).
    pub fn gemm_f16_native(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, ldc: usize) -> Res<()> {
        let (mut m_, mut n_, mut k_, mut ldc_) = (m as c_int, n as c_int, k as c_int, ldc as c_int);
        let (mut ap, mut bp, mut cp) = (a, b, c);
        let mut args: [*mut c_void; 7] = [
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut m_ as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
            &mut k_ as *mut _ as *mut c_void,
            &mut ldc_ as *mut _ as *mut c_void,
        ];
        let grid = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);
        self.launch(self.funcs.gemm_f16, grid, 256, 0, &mut args)
    }

    pub fn gemm_f16(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize) -> Res<()> {
        self.gemm_dispatch(a, b, c, m, n, k, n)
    }
}

// ===========================================================================
// NVRTC compilation with on-disk PTX cache
// ===========================================================================

/// Locate the CUDA toolkit header directory containing `cuda_fp16.h`.
///
/// NVRTC ships *no* default include search path: `#include <cuda_fp16.h>`
/// fails with "catastrophic error: cannot open source file" unless the
/// toolkit's include dir is passed explicitly via `--include-path=`. Probe
/// order: `$CUDA_HOME`/`$CUDA_PATH`, the conventional install prefixes,
/// versioned `/usr/local/cuda-*` trees, then distro-packaged `/usr/include`.
fn find_cuda_include() -> Res<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(var) {
            candidates.push(std::path::PathBuf::from(root).join("include"));
        }
    }
    candidates.push("/usr/local/cuda/include".into());
    candidates.push("/opt/cuda/include".into());
    if let Ok(rd) = std::fs::read_dir("/usr/local") {
        let mut versioned: Vec<_> = rd
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("cuda-"))
            .map(|e| e.path().join("include"))
            .collect();
        versioned.sort();
        versioned.reverse(); // prefer the newest toolkit
        candidates.extend(versioned);
    }
    candidates.push("/usr/include".into()); // Debian/Ubuntu nvidia-cuda-toolkit

    for dir in &candidates {
        if dir.join("cuda_fp16.h").is_file() {
            return Ok(dir.clone());
        }
    }
    Err(err!(
        "cuda",
        "cuda_fp16.h not found (searched: {}). NVRTC needs the CUDA toolkit \
         headers, not just the driver — install the toolkit (e.g. \
         `apt install nvidia-cuda-toolkit` or the NVIDIA .run/.deb package) \
         or point CUDA_HOME at its prefix, e.g. CUDA_HOME=/usr/local/cuda-12.4",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

/// Compile `src` for `sm_{maj}{min}`, caching PTX under `./models/.ptx-cache`.
fn compile_ptx(src: &str, maj: i32, min: i32) -> Res<CString> {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in src.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    let cache = std::path::PathBuf::from(format!("./models/.ptx-cache/kernels_sm{}{}_{:016x}.ptx", maj, min, h));
    if let Ok(bytes) = std::fs::read(&cache) {
        if let Ok(c) = CString::new(bytes) {
            log::info(&format!("kernel PTX loaded from cache: {}", cache.display()));
            return Ok(c);
        }
    }
    // Resolve the toolkit include dir *before* creating the NVRTC program so
    // a missing-toolkit failure cannot leak the program handle.
    let inc_dir = find_cuda_include()?;
    let t0 = std::time::Instant::now();
    unsafe {
        let csrc = CString::new(src).unwrap();
        let cname = CString::new("cima_kernels.cu").unwrap();
        let mut prog: nvrtcProgram = ptr::null_mut();
        if nvrtcCreateProgram(&mut prog, csrc.as_ptr(), cname.as_ptr(), 0, ptr::null(), ptr::null()) != 0 {
            return Err(err!("cuda", "nvrtcCreateProgram failed"));
        }
        let arch = CString::new(format!("--gpu-architecture=compute_{}{}", maj, min)).unwrap();
        let fast = CString::new("--use_fast_math").unwrap();
        // NVRTC has no default header search path — hand it the toolkit's
        // include dir so `#include <cuda_fp16.h>` resolves (see
        // [`find_cuda_include`] for the probe order and failure guidance).
        let inc = CString::new(format!("--include-path={}", inc_dir.display())).unwrap();
        let opts = [arch.as_ptr(), fast.as_ptr(), inc.as_ptr()];
        let rc = nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr());

        // Always fetch the compiler log so errors are actionable.
        let mut logsz: usize = 0;
        nvrtcGetProgramLogSize(prog, &mut logsz);
        let mut logbuf = vec![0u8; logsz.max(1)];
        nvrtcGetProgramLog(prog, logbuf.as_mut_ptr() as *mut c_char);
        if rc != 0 {
            let logtxt = String::from_utf8_lossy(&logbuf);
            nvrtcDestroyProgram(&mut prog);
            return Err(err!("cuda", "NVRTC compilation failed:\n{}", logtxt));
        }

        let mut n: usize = 0;
        nvrtcGetPTXSize(prog, &mut n);
        let mut ptx = vec![0u8; n];
        nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char);
        nvrtcDestroyProgram(&mut prog);
        // PTX is NUL-terminated from NVRTC.
        while ptx.last() == Some(&0) { ptx.pop(); }

        let _ = std::fs::create_dir_all(cache.parent().unwrap());
        let _ = std::fs::write(&cache, &ptx);
        log::info(&format!("kernels JIT-compiled for sm_{}{} in {:?}", maj, min, t0.elapsed()));
        CString::new(ptx).map_err(|_| err!("cuda", "PTX contained interior NUL"))
    }
}

/// Human-readable byte formatter used across telemetry logs.
pub fn fmt_bytes(b: usize) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 4 {
        v /= 1024.0;
        i += 1;
    }
    format!("{:.2} {}", v, U[i])
}