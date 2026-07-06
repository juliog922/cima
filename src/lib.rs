//! # cima — minimalist CUDA inference engine
//!
//! A from-scratch, zero-dependency Rust inference engine for Nvidia GPUs on
//! Linux. It clones Ollama's CLI and REST API surface 1:1 while pulling
//! models directly from the Hugging Face Hub into `./models/`.
//!
//! ## Module map (linear reading order — one file per responsibility)
//!
//! | file            | inside                                                      |
//! |-----------------|--------------------------------------------------------------|
//! | [`log`]         | structured stderr logging + [`shmlog`] crash-surviving ring  |
//! | [`json`]        | recursive-descent JSON parser/serializer (no serde)          |
//! | [`traits`]      | every extension seam + core types + [`num`] f16/bf16 prims   |
//! | [`cuda`]        | CUDA driver/NVRTC/cuBLAS/NVML FFI, kernels, zero-copy DMA    |
//! | [`formats`]     | weight containers: `safetensors` (mmap) + `gguf`             |
//! | [`quant`]       | codecs: `bnb` (NF4/FP4) + `gguf` block formats               |
//! | [`tokenizer`]   | byte-level BPE (GPT-2 family) + SPM + chat-template renderer |
//! | [`media`]       | `ImageDecoder` / `AudioDecoder` registry (PPM/BMP/WAV)       |
//! | [`imgcodec`]    | baseline JPEG / PNG decoders (std-only)                      |
//! | [`hub`]         | Hub client + local model [`registry`] + checkpoint [`vet`]   |
//! | [`models`]      | `Arch` dispatch, sampler, transformer, towers, gemma4        |
//! | [`api`]         | `protocol` + `queue` + `server` + `client` (Ollama surface)  |
//! | [`selftest`]    | GPU-vs-CPU numerical self-tests (`cima selftest`)            |
//!
//! Absorbed modules stay addressable at their historical crate paths
//! (`crate::shmlog`, `crate::num`, `crate::queue`, `crate::registry`,
//! `crate::vet`) through root re-exports, so downstream code and the
//! integration tests need no changes.
//!
//! Published as a library so the engine, the wire protocol, and the typed
//! client are reusable; the `cima` binary is a thin CLI over this crate.

pub mod api;
pub mod cuda;
pub mod formats;
pub mod hub;
pub mod imgcodec;
pub mod json;
pub mod log;
pub mod media;
pub mod models;
pub use traits::num;
pub mod quant;
pub use api::queue;
pub use hub::registry;
pub mod selftest;
pub use log::shmlog;
pub mod tokenizer;
pub mod traits;
pub use hub::vet;
