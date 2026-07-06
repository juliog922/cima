//! Build script: declare the native libraries the engine's `extern "C"`
//! blocks resolve against, and embed the git commit for the startup banner.
//!
//! Link-time dependencies (the `cu*`, `nvrtc*`, and `nvml*` symbols in
//! `cuda.rs`, and the `curl_*` symbols in `hub.rs`):
//!
//! * `cuda` — CUDA driver API (libcuda.so; from the driver / container runtime)
//! * `nvrtc` — runtime kernel compilation
//! * `nvidia-ml` — NVML device telemetry (libnvidia-ml.so)
//! * `curl` — Hub downloads and the metadata preflight
//!
//! cuBLAS is deliberately absent: it is `dlopen`ed at runtime (see
//! `cuda::try_load_cublas`), so it is not a link-time dependency and the
//! engine runs whether or not it is installed.

use std::process::Command;

fn main() {
    // Search paths for the CUDA stubs/libraries. The driver library
    // (libcuda.so) ships as a stub in the toolkit under stubs/ and is
    // provided for real at runtime by the driver; the others resolve from
    // the standard toolkit lib dir. CUDA_PATH overrides the default.
    let cuda_root = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search=native={cuda_root}/lib64");
    println!("cargo:rustc-link-search=native={cuda_root}/lib64/stubs");
    // Distro package locations (Debian/Ubuntu multiarch, and the driver's
    // own directory) so the driver and NVML resolve without a full toolkit.
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rustc-link-lib=dylib=curl");

    // Embed the short commit for `cima serve`'s startup banner; absent git
    // (release tarball, shallow checkout) falls back to "unknown".
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CIMA_GIT_SHA={sha}");

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}