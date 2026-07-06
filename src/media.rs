//! # media — multimodal input frontends
//!
//! Concrete [`ImageDecoder`] / [`AudioDecoder`] implementations plus the
//! log-mel feature extractor for speech models. Every decoder is fed raw,
//! untrusted bytes from the API's `images`/`audio` fields and must convert
//! hostile input into precise errors, never panics.
//!
//! Shipped decoders (registered in [`MediaRegistry`]):
//! * `jpeg` — baseline JPEG (zero-dependency, see `imgcodec.rs`).
//! * `png` — PNG, all common color types (zero-dependency, see `imgcodec.rs`).
//! * `ppm` — binary PPM (P6), the lossless interchange baseline.
//! * `bmp` — uncompressed 24/32-bit Windows bitmaps.
//! * `wav` — PCM16/PCM32/F32 WAVE, any channel count, any sample rate.
//!
//! Compressed formats (PNG/JPEG/MP3/FLAC) are additional `impl` blocks away —
//! the registry dispatch and the tensor contract don't change.

use crate::traits::{AudioDecoder, AudioPcm, ImageDecoder, ImageTensor, Res};
use crate::{err, log};

// ===========================================================================
// Registry
// ===========================================================================

/// Ordered set of decoders; first whose magic-sniff matches wins.
pub struct MediaRegistry {
    pub images: Vec<Box<dyn ImageDecoder>>,
    pub audio: Vec<Box<dyn AudioDecoder>>,
}

impl MediaRegistry {
    pub fn standard() -> MediaRegistry {
        MediaRegistry {
            images: vec![
                Box::new(JpegDecoder),
                Box::new(PngDecoder),
                Box::new(PpmDecoder),
                Box::new(BmpDecoder),
            ],
            audio: vec![Box::new(WavDecoder)],
        }
    }

    /// Decode an image with whichever registered decoder recognizes it.
    pub fn decode_image(
        &self,
        bytes: &[u8],
        h: usize,
        w: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor> {
        for d in &self.images {
            if d.detect(bytes) {
                log::info(&format!(
                    "image: decoding {} bytes via '{}' decoder",
                    bytes.len(),
                    d.name()
                ));
                return d.decode(bytes, h, w, mean, std);
            }
        }
        Err(err!(
            "media",
            "unrecognized image format (first bytes: {:02x?}); registered decoders: {:?}",
            &bytes[..bytes.len().min(8)],
            self.images.iter().map(|d| d.name()).collect::<Vec<_>>()
        ))
    }

    /// Header-only image dimensions `(width, height)` via whichever decoder
    /// recognizes the format (for aspect-preserving target computation).
    pub fn image_dims(&self, bytes: &[u8]) -> Option<(usize, usize)> {
        self.images
            .iter()
            .find(|d| d.detect(bytes))
            .and_then(|d| d.dims(bytes))
    }

    /// Decode audio with whichever registered decoder recognizes it.
    pub fn decode_audio(&self, bytes: &[u8], rate: u32) -> Res<AudioPcm> {
        for d in &self.audio {
            if d.detect(bytes) {
                log::info(&format!(
                    "audio: decoding {} bytes via '{}' decoder",
                    bytes.len(),
                    d.name()
                ));
                return d.decode(bytes, rate);
            }
        }
        // Diagnose the common impostors before giving up: users regularly
        // feed saved HTML error pages ("<!DOCTYPE…") or MP3s with a .wav
        // extension — naming the actual format turns a mystery into a
        // one-line fix on their side.
        let hint = if bytes.starts_with(b"<!")
            || bytes.starts_with(b"<htm")
            || bytes.starts_with(b"<HTM")
        {
            " — this file is an HTML page, not audio (a saved download-error page?)"
        } else if bytes.starts_with(b"ID3")
            || (bytes.len() > 1 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0)
        {
            " — this is an MP3; convert it first, e.g. `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`"
        } else if bytes.starts_with(b"OggS") {
            " — this is an Ogg container; convert it first, e.g. `ffmpeg -i in.ogg -ar 16000 -ac 1 out.wav`"
        } else if bytes.starts_with(b"fLaC") {
            " — this is a FLAC; convert it first, e.g. `ffmpeg -i in.flac -ar 16000 -ac 1 out.wav`"
        } else if bytes.starts_with(b"RIFF") {
            " — RIFF header found but not a PCM WAVE payload; re-encode with `ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav`"
        } else {
            ""
        };
        Err(err!(
            "media",
            "unrecognized audio format (first bytes: {:02x?}); registered decoders: {:?}{}",
            &bytes[..bytes.len().min(8)],
            self.audio.iter().map(|d| d.name()).collect::<Vec<_>>(),
            hint
        ))
    }
}

// ===========================================================================
// Shared raster helpers
// ===========================================================================

/// Bilinear resize + per-channel normalization of an interleaved RGB buffer
/// into planar CHW (the vision-tower input contract).
fn resize_normalize(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> ImageTensor {
    // Antialiased separable resize with a triangle (bilinear) kernel whose
    // support scales with the downscale ratio — matching PIL / torchvision
    // `resize(..., antialias=True)`. Plain 4-neighbour bilinear aliases badly
    // at 2-4× reduction (typical photo → vision-tower geometry), turning fine
    // detail into structured noise that VLMs then hallucinate over.
    let pass = |src: &[f32], sw: usize, sh: usize, dw: usize, horizontal: bool| -> Vec<f32> {
        let (n_out, n_keep) = if horizontal { (dw, sh) } else { (dw, sw) };
        let src_n = if horizontal { sw } else { sh };
        let ratio = (src_n as f32 / n_out as f32).max(1.0);
        let support = ratio; // triangle kernel radius
        let mut out = vec![0f32; n_out * n_keep * 3];
        // Precompute per-output-coordinate taps (shared across rows/cols).
        let mut taps: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let center = (o as f32 + 0.5) * src_n as f32 / n_out as f32 - 0.5;
            let lo = ((center - support).ceil().max(0.0)) as usize;
            let hi = ((center + support).floor() as isize)
                .min(src_n as isize - 1)
                .max(lo as isize) as usize;
            let mut w: Vec<f32> = (lo..=hi)
                .map(|s| {
                    let d = (s as f32 - center).abs() / ratio;
                    (1.0 - d).max(0.0)
                })
                .collect();
            let sum: f32 = w.iter().sum::<f32>().max(1e-9);
            w.iter_mut().for_each(|v| *v /= sum);
            taps.push((lo, w));
        }
        for k in 0..n_keep {
            for (o, (lo, w)) in taps.iter().enumerate() {
                let mut acc = [0f32; 3];
                for (j, &wt) in w.iter().enumerate() {
                    let s = lo + j;
                    let idx = if horizontal {
                        (k * sw + s) * 3
                    } else {
                        (s * sw + k) * 3
                    };
                    for c in 0..3 {
                        acc[c] += wt * src[idx + c];
                    }
                }
                let oidx = if horizontal {
                    (k * n_out + o) * 3
                } else {
                    (o * sw + k) * 3
                };
                out[oidx..oidx + 3].copy_from_slice(&acc);
            }
        }
        out
    };

    let srcf: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
    let horiz = pass(&srcf, src_w, src_h, dst_w, true); // [src_h, dst_w]
    let full = pass(&horiz, dst_w, src_h, dst_h, false); // [dst_h, dst_w]

    let mut out = vec![0f32; 3 * dst_h * dst_w];
    for y in 0..dst_h {
        for x in 0..dst_w {
            for c in 0..3 {
                let v = full[(y * dst_w + x) * 3 + c].clamp(0.0, 255.0);
                out[c * dst_h * dst_w + y * dst_w + x] = (v / 255.0 - mean[c]) / std[c];
            }
        }
    }
    // Optional decode-fidelity dump: CIMA_DUMP_DECODED=/path/prefix writes the
    // exact resized RGB the vision tower will see, as PPM (P6) — open it to
    // separate "decoder/resize corruption" from "model limitation".
    if let Ok(prefix) = std::env::var("CIMA_DUMP_DECODED") {
        let n = DUMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = format!("{}.{}.ppm", prefix, n);
        let mut ppm = format!("P6\n{} {}\n255\n", dst_w, dst_h).into_bytes();
        for y in 0..dst_h {
            for x in 0..dst_w {
                for c in 0..3 {
                    ppm.push(full[(y * dst_w + x) * 3 + c].clamp(0.0, 255.0) as u8);
                }
            }
        }
        match std::fs::write(&path, ppm) {
            Ok(()) => log::info(&format!(
                "decoded image dumped to {} ({}×{})",
                path, dst_w, dst_h
            )),
            Err(e) => log::warn(&format!("CIMA_DUMP_DECODED: cannot write {}: {}", path, e)),
        }
    }
    ImageTensor {
        data: out,
        channels: 3,
        height: dst_h,
        width: dst_w,
    }
}

static DUMP_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// ===========================================================================
// PPM (P6) decoder
// ===========================================================================

/// Binary PPM decoder — the simplest lossless interchange format; clients
/// can transcode anything to PPM with one ffmpeg/ImageMagick call.
pub struct PpmDecoder;

impl ImageDecoder for PpmDecoder {
    fn name(&self) -> &'static str {
        "ppm"
    }
    fn dims(&self, b: &[u8]) -> Option<(usize, usize)> {
        // P6 header: "P6 <w> <h> <max>" with arbitrary whitespace. Parsed at
        // the byte level — the binary payload right after the header is not
        // UTF-8, so a str conversion of a fixed prefix would fail.
        if !b.starts_with(b"P6") {
            return None;
        }
        let mut i = 2usize;
        let mut fields = [0usize; 2];
        for f in fields.iter_mut() {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            let s = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if s == i {
                return None;
            }
            *f = std::str::from_utf8(&b[s..i]).ok()?.parse().ok()?;
        }
        Some((fields[0], fields[1]))
    }
    fn detect(&self, b: &[u8]) -> bool {
        b.starts_with(b"P6")
    }
    fn decode(
        &self,
        b: &[u8],
        th: usize,
        tw: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor> {
        let mut i = 2usize;
        let mut fields = [0usize; 3]; // width, height, maxval
        for (f, field) in fields.iter_mut().enumerate() {
            // skip whitespace & comments
            loop {
                while i < b.len() && (b[i] as char).is_ascii_whitespace() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'#' {
                    while i < b.len() && b[i] != b'\n' {
                        i += 1;
                    }
                } else {
                    break;
                }
            }
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if start == i {
                return Err(err!(
                    "media",
                    "ppm: header field {} missing (truncated header)",
                    f
                ));
            }
            *field = std::str::from_utf8(&b[start..i])
                .unwrap()
                .parse()
                .map_err(|_| err!("media", "ppm: header field {} not a number", f))?;
        }
        i += 1; // single whitespace after maxval
        let (w, h, maxval) = (fields[0], fields[1], fields[2]);
        if maxval != 255 {
            return Err(err!(
                "media",
                "ppm: maxval {} unsupported (only 8-bit)",
                maxval
            ));
        }
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            return Err(err!("media", "ppm: implausible dimensions {}x{}", w, h));
        }
        let need = w * h * 3;
        if b.len() < i + need {
            return Err(err!(
                "media",
                "ppm: pixel data truncated — need {} bytes, have {}",
                need,
                b.len() - i
            ));
        }
        Ok(resize_normalize(&b[i..i + need], w, h, tw, th, mean, std))
    }
}

// ===========================================================================
// BMP decoder (uncompressed 24/32-bit)
// ===========================================================================

/// Baseline JPEG via the zero-dependency decoder in `imgcodec.rs`
/// (validated pixel-for-pixel against libjpeg/PIL across subsampling modes,
/// restart markers, grayscale and edge sizes).
pub struct JpegDecoder;

impl ImageDecoder for JpegDecoder {
    fn name(&self) -> &'static str {
        "jpeg"
    }
    fn dims(&self, b: &[u8]) -> Option<(usize, usize)> {
        crate::imgcodec::jpeg_dims(b)
    }
    fn detect(&self, b: &[u8]) -> bool {
        b.len() > 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
    }
    fn decode(
        &self,
        b: &[u8],
        th: usize,
        tw: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor> {
        let (w, h, rgb) = crate::imgcodec::decode_jpeg(b).map_err(|e| err!("media", "{}", e))?;
        Ok(resize_normalize(&rgb, w, h, tw, th, mean, std))
    }
}

/// PNG via the zero-dependency inflate + defilter in `imgcodec.rs`
/// (bit-exact against libpng/PIL for gray / RGB / RGBA / palette).
pub struct PngDecoder;

impl ImageDecoder for PngDecoder {
    fn name(&self) -> &'static str {
        "png"
    }
    fn dims(&self, b: &[u8]) -> Option<(usize, usize)> {
        crate::imgcodec::png_dims(b)
    }
    fn detect(&self, b: &[u8]) -> bool {
        b.starts_with(&[0x89, 0x50, 0x4E, 0x47])
    }
    fn decode(
        &self,
        b: &[u8],
        th: usize,
        tw: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor> {
        let (w, h, rgb) = crate::imgcodec::decode_png(b).map_err(|e| err!("media", "{}", e))?;
        Ok(resize_normalize(&rgb, w, h, tw, th, mean, std))
    }
}

pub struct BmpDecoder;

impl ImageDecoder for BmpDecoder {
    fn name(&self) -> &'static str {
        "bmp"
    }
    fn dims(&self, b: &[u8]) -> Option<(usize, usize)> {
        if b.len() < 26 {
            return None;
        }
        let w = i32::from_le_bytes([b[18], b[19], b[20], b[21]]).unsigned_abs() as usize;
        let h = i32::from_le_bytes([b[22], b[23], b[24], b[25]]).unsigned_abs() as usize;
        Some((w, h))
    }
    fn detect(&self, b: &[u8]) -> bool {
        b.starts_with(b"BM")
    }
    fn decode(
        &self,
        b: &[u8],
        th: usize,
        tw: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Res<ImageTensor> {
        let rd_u32 = |o: usize| -> Res<u32> {
            b.get(o..o + 4)
                .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
                .ok_or_else(|| err!("media", "bmp: header truncated at byte {}", o))
        };
        let rd_i32 = |o: usize| rd_u32(o).map(|v| v as i32);
        let data_off = rd_u32(10)? as usize;
        let w = rd_i32(18)?;
        let h_raw = rd_i32(22)?;
        let bpp = b
            .get(28..30)
            .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
            .unwrap_or(0);
        let compression = rd_u32(30)?;
        if compression != 0 && !(compression == 3 && bpp == 32) {
            return Err(err!(
                "media",
                "bmp: compression mode {} unsupported (only BI_RGB)",
                compression
            ));
        }
        if bpp != 24 && bpp != 32 {
            return Err(err!("media", "bmp: {} bpp unsupported (only 24/32)", bpp));
        }
        if w <= 0 || w > 16384 || h_raw == 0 || h_raw.unsigned_abs() > 16384 {
            return Err(err!("media", "bmp: implausible dimensions {}x{}", w, h_raw));
        }
        let (w, h) = (w as usize, h_raw.unsigned_abs() as usize);
        let bottom_up = h_raw > 0;
        let bytes_pp = bpp as usize / 8;
        let stride = (w * bytes_pp + 3) & !3;
        if b.len() < data_off + stride * h {
            return Err(err!(
                "media",
                "bmp: pixel data truncated — need {} bytes from offset {}, file is {}",
                stride * h,
                data_off,
                b.len()
            ));
        }
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            let src_y = if bottom_up { h - 1 - y } else { y };
            let row = &b[data_off + src_y * stride..];
            for x in 0..w {
                let p = &row[x * bytes_pp..];
                let d = &mut rgb[(y * w + x) * 3..(y * w + x) * 3 + 3];
                d[0] = p[2]; // BGR(A) -> RGB
                d[1] = p[1];
                d[2] = p[0];
            }
        }
        Ok(resize_normalize(&rgb, w, h, tw, th, mean, std))
    }
}

// ===========================================================================
// WAV decoder
// ===========================================================================

pub struct WavDecoder;

impl AudioDecoder for WavDecoder {
    fn name(&self) -> &'static str {
        "wav"
    }
    fn detect(&self, b: &[u8]) -> bool {
        b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE"
    }
    fn decode(&self, b: &[u8], target_rate: u32) -> Res<AudioPcm> {
        // chunk walk
        let mut i = 12usize;
        let mut fmt: Option<(u16, u16, u32, u16)> = None; // (codec, channels, rate, bits)
        let mut data: Option<&[u8]> = None;
        while i + 8 <= b.len() {
            let id = &b[i..i + 4];
            let sz = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
            let body = b.get(i + 8..i + 8 + sz).ok_or_else(|| {
                err!(
                    "media",
                    "wav: chunk '{}' truncated",
                    String::from_utf8_lossy(id)
                )
            })?;
            match id {
                b"fmt " => {
                    if sz < 16 {
                        return Err(err!("media", "wav: fmt chunk only {} bytes", sz));
                    }
                    fmt = Some((
                        u16::from_le_bytes(body[0..2].try_into().unwrap()),
                        u16::from_le_bytes(body[2..4].try_into().unwrap()),
                        u32::from_le_bytes(body[4..8].try_into().unwrap()),
                        u16::from_le_bytes(body[14..16].try_into().unwrap()),
                    ));
                }
                b"data" => data = Some(body),
                _ => {}
            }
            i += 8 + sz + (sz & 1);
        }
        let (codec, ch, rate, bits) = fmt.ok_or_else(|| err!("media", "wav: missing fmt chunk"))?;
        let data = data.ok_or_else(|| err!("media", "wav: missing data chunk"))?;
        if ch == 0 {
            return Err(err!("media", "wav: zero channels"));
        }
        let ch = ch as usize;

        // decode to mono f32
        let mono: Vec<f32> = match (codec, bits) {
            (1, 16) => data
                .chunks_exact(2 * ch)
                .map(|fr| {
                    fr.chunks_exact(2)
                        .map(|s| i16::from_le_bytes(s.try_into().unwrap()) as f32 / 32768.0)
                        .sum::<f32>()
                        / ch as f32
                })
                .collect(),
            (1, 32) => data
                .chunks_exact(4 * ch)
                .map(|fr| {
                    fr.chunks_exact(4)
                        .map(|s| i32::from_le_bytes(s.try_into().unwrap()) as f32 / 2147483648.0)
                        .sum::<f32>()
                        / ch as f32
                })
                .collect(),
            (3, 32) => data
                .chunks_exact(4 * ch)
                .map(|fr| {
                    fr.chunks_exact(4)
                        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
                        .sum::<f32>()
                        / ch as f32
                })
                .collect(),
            (c, b) => {
                return Err(err!(
                    "media",
                    "wav: codec {} / {} bits unsupported (PCM16, PCM32, FLOAT32 only)",
                    c,
                    b
                ))
            }
        };

        // linear resample to target rate
        let samples = if rate == target_rate {
            mono
        } else {
            let n_out = (mono.len() as u64 * target_rate as u64 / rate as u64) as usize;
            (0..n_out)
                .map(|j| {
                    let pos = j as f64 * rate as f64 / target_rate as f64;
                    let i0 = pos as usize;
                    let frac = (pos - i0 as f64) as f32;
                    let a = mono.get(i0).copied().unwrap_or(0.0);
                    let b2 = mono.get(i0 + 1).copied().unwrap_or(a);
                    a * (1.0 - frac) + b2 * frac
                })
                .collect()
        };
        Ok(AudioPcm {
            samples,
            sample_rate: target_rate,
        })
    }
}

// ===========================================================================
// Log-mel spectrogram (Whisper-compatible front end)
// ===========================================================================

/// Log-mel spectrogram parameters. The defaults transcribe
/// `Gemma4AudioFeatureExtractor` exactly (verified against the live
/// preprocessor_config.json + feature_extraction_gemma4.py): 20 ms frames
/// zero-padded into a 512-point FFT, MAGNITUDE spectrum (not power), HTK
/// mel scale with unnormalized triangles, natural log with a 1e-3 floor,
/// semicausal left padding so frame 0 is centered at t=0. A future audio
/// family with a different recipe overrides fields instead of forking the
/// function.
pub struct MelParams {
    pub frame_length: usize,
    pub fft_length: usize,
    pub hop: usize,
    pub mel_floor: f32,
}

impl Default for MelParams {
    fn default() -> Self {
        MelParams {
            frame_length: 320,
            fft_length: 512,
            hop: 160,
            mel_floor: 1e-3,
        }
    }
}

/// Returns `[frames, n_mels]` row-major f32, reference-exact (see [`MelParams`]).
pub fn log_mel(pcm: &AudioPcm, n_mels: usize, p: &MelParams) -> Vec<Vec<f32>> {
    // Periodic Hann over the FRAME (not the FFT size): w[n] = 0.5 − 0.5·cos(2πn/L)
    let hann: Vec<f32> = (0..p.frame_length)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / p.frame_length as f32).cos())
        .collect();
    let filters = mel_filterbank(n_mels, p.fft_length, pcm.sample_rate as f32);
    let n_bins = p.fft_length / 2 + 1;

    // Semicausal padding: frame_length/2 zeros prepended so the first frame
    // is centered at t=0 (sl.STFT time_padding='semicausal').
    let pad_left = p.frame_length / 2;
    let padded_len = pcm.samples.len() + pad_left;
    let sample = |idx: usize| -> f32 {
        if idx < pad_left {
            0.0
        } else {
            pcm.samples.get(idx - pad_left).copied().unwrap_or(0.0)
        }
    };
    // unfold(size = frame_length+1, step = hop), preemphasis 0 -> first
    // frame_length samples of each window.
    let win = p.frame_length + 1;
    let n_frames = if padded_len >= win {
        (padded_len - win) / p.hop + 1
    } else {
        0
    };

    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * p.hop;
        // windowed real DFT, zero-padded to fft_length (angle uses fft_length)
        let mut mag = vec![0f32; n_bins];
        for (k, slot) in mag.iter_mut().enumerate() {
            let (mut re, mut im) = (0f32, 0f32);
            // n is the sample offset within the frame; it appears in the
            // twiddle phase (k*n), so the index is intrinsic to the DFT.
            #[allow(clippy::needless_range_loop)]
            for n in 0..p.frame_length {
                let s = sample(start + n) * hann[n];
                let ang = -2.0 * std::f32::consts::PI * (k * n) as f32 / p.fft_length as f32;
                re += s * ang.cos();
                im += s * ang.sin();
            }
            *slot = (re * re + im * im).sqrt(); // MAGNITUDE, not power
        }
        let mut mel = vec![0f32; n_mels];
        for (mm, filt) in filters.iter().enumerate() {
            let mut acc = 0f32;
            for &(bin, w) in filt {
                acc += mag[bin] * w;
            }
            mel[mm] = (acc + p.mel_floor).ln(); // natural log + floor; no compression
        }
        frames.push(mel);
    }
    frames
}

fn mel_filterbank(n_mels: usize, n_fft: usize, rate: f32) -> Vec<Vec<(usize, f32)>> {
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10f32.powf(mel / 2595.0) - 1.0);
    let n_bins = n_fft / 2 + 1;
    let max_mel = hz_to_mel(rate / 2.0);
    let points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(max_mel * i as f32 / (n_mels + 1) as f32) * n_fft as f32 / rate)
        .collect();
    (0..n_mels)
        .map(|m| {
            let (a, b, c) = (points[m], points[m + 1], points[m + 2]);
            let mut filt = Vec::new();
            for bin in a.floor() as usize..(c.ceil() as usize).min(n_bins) {
                let x = bin as f32;
                let w = if x < b {
                    (x - a) / (b - a).max(1e-6)
                } else {
                    (c - x) / (c - b).max(1e-6)
                };
                if w > 0.0 {
                    filt.push((bin, w));
                }
            }
            filt
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    const STD: [f32; 3] = [0.5, 0.5, 0.5];

    /// A valid 2×2 PPM decodes pixel-exactly: white normalizes to +1, black
    /// to −1 with mean/std 0.5/0.5.
    #[test]
    fn ppm_pixel_exact() {
        let ppm = b"P6\n2 2\n255\n\xff\xff\xff\x00\x00\x00\xff\x00\x00\x00\x00\xff";
        let t = PpmDecoder.decode(ppm, 2, 2, MEAN, STD).unwrap();
        assert_eq!((t.width, t.height), (2, 2));
        // CHW f32, plane R: white=+1, black=-1, red=+1, blue=-1
        assert_eq!(&t.data[0..4], &[1.0, -1.0, 1.0, -1.0]);
    }

    /// Malformed PPMs error: bad magic, missing fields, dims overflow,
    /// truncated pixel payload, zero dimensions.
    #[test]
    fn ppm_malformed_errors() {
        for bad in [
            &b"P5\n2 2\n255\n"[..],
            &b"P6\n2\n255\n"[..],
            &b"P6\n999999999 999999999\n255\n"[..],
            &b"P6\n2 2\n255\n\xff\xff"[..],
            &b"P6\n0 2\n255\n"[..],
            &b"P6\n-1 2\n255\n"[..],
            &b""[..],
        ] {
            assert!(
                PpmDecoder.decode(bad, 2, 2, MEAN, STD).is_err(),
                "accepted: {:?}",
                &bad[..bad.len().min(12)]
            );
        }
    }

    /// BMP guards: truncated header, unsupported compression, dims of zero.
    #[test]
    fn bmp_malformed_errors() {
        assert!(BmpDecoder.decode(b"", 2, 2, MEAN, STD).is_err());
        assert!(BmpDecoder.decode(b"BM", 2, 2, MEAN, STD).is_err());
        assert!(BmpDecoder.decode(&[0u8; 54], 2, 2, MEAN, STD).is_err());
    }

    /// WAV guards: truncated RIFF, wrong format tags.
    #[test]
    fn wav_malformed_errors() {
        assert!(WavDecoder.decode(b"", 16000).is_err());
        assert!(WavDecoder.decode(b"RIFF", 16000).is_err());
        assert!(WavDecoder.decode(&[0u8; 44], 16000).is_err());
    }
}
