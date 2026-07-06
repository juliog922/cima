//! # imgcodec — zero-dependency JPEG (baseline) and PNG decoders
//!
//! Self-contained (`std` only, `Result<_, String>`) so the decode logic can
//! be validated standalone against reference codecs. `media.rs` wraps these
//! behind the [`ImageDecoder`] registry.
//!
//! **JPEG**: baseline and extended-sequential DCT (SOF0/SOF1), canonical
//! Huffman per ITU T.81 F.2.2, 8-bit precision, generic h/v chroma
//! subsampling (4:4:4 / 4:2:2 / 4:2:0 / 4:1:1) with nearest-neighbour
//! upsampling, restart markers, grayscale and YCbCr (BT.601 full-range).
//! Progressive (SOF2), arithmetic coding, 12-bit precision and CMYK are
//! rejected with precise errors.
//!
//! **PNG**: full RFC 1951 inflate (stored / fixed / dynamic blocks), all five
//! scanline filters (incl. Paeth), color types gray / RGB / palette /
//! gray+alpha / RGBA at bit depth 8 (palette and gray also at 1/2/4). Alpha
//! is dropped (the RGB samples pass through), matching `Image.convert("RGB")`
//! in reference preprocessors. 16-bit and Adam7 interlace are rejected with
//! precise errors.

type R<T> = Result<T, String>;

// ===========================================================================
// JPEG
// ===========================================================================

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Canonical Huffman table (ITU T.81 F.2.2.3 decode with min/max codes).
struct Huff {
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    vals: Vec<u8>,
}

impl Huff {
    fn build(bits: &[u8; 16], vals: Vec<u8>) -> Huff {
        let (mut mincode, mut maxcode, mut valptr) = ([0i32; 17], [-1i32; 17], [0i32; 17]);
        let mut code = 0i32;
        let mut k = 0i32;
        for l in 1..=16usize {
            valptr[l] = k;
            mincode[l] = code;
            code += bits[l - 1] as i32;
            k += bits[l - 1] as i32;
            maxcode[l] = code - 1;
            code <<= 1;
            if bits[l - 1] == 0 {
                maxcode[l] = -1; // no codes of this length
            }
        }
        Huff {
            mincode,
            maxcode,
            valptr,
            vals,
        }
    }
}

/// Entropy-coded segment bit reader with 0xFF00 byte-stuffing.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u32,
    bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            bitbuf: 0,
            bits: 0,
        }
    }

    /// Next bit; pads with zeros at any non-stuffing marker, which the spec
    /// defines as the end of the entropy-coded segment (T.81 F.2.2.5).
    fn bit(&mut self) -> R<u32> {
        if self.bits == 0 {
            if self.pos >= self.data.len() {
                return Ok(0); // pad past segment end
            }
            let byte = self.data[self.pos];
            if byte == 0xFF {
                match self.data.get(self.pos + 1) {
                    Some(0x00) => {
                        self.bitbuf = 0xFF; // stuffed FF
                        self.pos += 2;
                    }
                    _ => return Ok(0), // real marker: segment over, pad
                }
            } else {
                self.bitbuf = byte as u32;
                self.pos += 1;
            }
            self.bits = 8;
        }
        self.bits -= 1;
        Ok((self.bitbuf >> self.bits) & 1)
    }

    fn bits_n(&mut self, n: u32) -> R<u32> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    fn huff(&mut self, t: &Huff) -> R<u8> {
        let mut code = 0i32;
        for l in 1..=16usize {
            code = (code << 1) | self.bit()? as i32;
            if t.maxcode[l] >= 0 && code <= t.maxcode[l] && code >= t.mincode[l] {
                let idx = (t.valptr[l] + (code - t.mincode[l])) as usize;
                return t.vals.get(idx).copied().ok_or_else(|| {
                    "jpeg: huffman value index out of range (corrupt table)".into()
                });
            }
        }
        Err("jpeg: invalid huffman code (corrupt entropy stream)".into())
    }

    /// RECEIVE+EXTEND (T.81 F.2.2.1).
    fn extend(&mut self, s: u8) -> R<i32> {
        if s == 0 {
            return Ok(0);
        }
        let v = self.bits_n(s as u32)? as i32;
        Ok(if v < (1 << (s - 1)) {
            v - (1 << s) + 1
        } else {
            v
        })
    }

    /// Byte-align and consume an expected RSTn marker.
    fn restart(&mut self) -> R<()> {
        self.bits = 0;
        // scan to the marker (tolerate stray padding bits already dropped)
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF && (0xD0..=0xD7).contains(&self.data[self.pos + 1]) {
                self.pos += 2;
                return Ok(());
            }
            if self.data[self.pos] == 0xFF && self.data[self.pos + 1] != 0x00 {
                return Err(
                    "jpeg: expected restart marker, found another marker (corrupt stream)".into(),
                );
            }
            self.pos += 1;
        }
        Err("jpeg: stream ended where a restart marker was expected".into())
    }
}

/// Float separable 8×8 inverse DCT (within 1 LSB of libjpeg islow).
fn idct8x8(coef: &[f32; 64], out: &mut [u8], stride: usize) {
    let mut tmp = [0f32; 64];
    // orthonormal scale: alpha(0) = sqrt(1/8), alpha(u>0) = sqrt(2/8) = 0.5
    const C: [f32; 8] = [0.353_553_38, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    for y in 0..8 {
        for x in 0..8 {
            let mut s = 0f32;
            for u in 0..8 {
                s += C[u] * coef[y * 8 + u] * COSV[x][u];
            }
            tmp[y * 8 + x] = s;
        }
    }
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0f32;
            for v in 0..8 {
                s += C[v] * tmp[v * 8 + x] * COSV[y][v];
            }
            let p = (s + 128.5).floor();
            out[y * stride + x] = p.clamp(0.0, 255.0) as u8;
        }
    }
}

/// cos((2x+1)·u·π/16) lookup, normalized into the C[] scale above.
const COSV: [[f32; 8]; 8] = {
    // const-fn trig is unavailable; values are precomputed.
    [
        [
            1.0,
            0.980_785_25,
            0.923_879_5,
            0.831_469_6,
            0.707_106_77,
            0.555_570_24,
            0.382_683_43,
            0.195_090_32,
        ],
        [
            1.0,
            0.831_469_6,
            0.382_683_43,
            -0.195_090_32,
            -0.707_106_77,
            -0.980_785_25,
            -0.923_879_5,
            -0.555_570_24,
        ],
        [
            1.0,
            0.555_570_24,
            -0.382_683_43,
            -0.980_785_25,
            -0.707_106_77,
            0.195_090_32,
            0.923_879_5,
            0.831_469_6,
        ],
        [
            1.0,
            0.195_090_32,
            -0.923_879_5,
            -0.555_570_24,
            0.707_106_77,
            0.831_469_6,
            -0.382_683_43,
            -0.980_785_25,
        ],
        [
            1.0,
            -0.195_090_32,
            -0.923_879_5,
            0.555_570_24,
            0.707_106_77,
            -0.831_469_6,
            -0.382_683_43,
            0.980_785_25,
        ],
        [
            1.0,
            -0.555_570_24,
            -0.382_683_43,
            0.980_785_25,
            -0.707_106_77,
            -0.195_090_32,
            0.923_879_5,
            -0.831_469_6,
        ],
        [
            1.0,
            -0.831_469_6,
            0.382_683_43,
            0.195_090_32,
            -0.707_106_77,
            0.980_785_25,
            -0.923_879_5,
            0.555_570_24,
        ],
        [
            1.0,
            -0.980_785_25,
            0.923_879_5,
            -0.831_469_6,
            0.707_106_77,
            -0.555_570_24,
            0.382_683_43,
            -0.195_090_32,
        ],
    ]
};

struct JComp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    /// plane at the component's own sampling resolution
    plane: Vec<u8>,
    pw: usize,
    ph: usize,
    dc_pred: i32,
}

/// Decode a baseline JPEG into `(width, height, interleaved RGB)`.
pub fn decode_jpeg(b: &[u8]) -> R<(usize, usize, Vec<u8>)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return Err("jpeg: missing SOI marker".into());
    }
    let mut qt: [[f32; 64]; 4] = [[0.0; 64]; 4];
    let mut dc_tabs: [Option<Huff>; 4] = [None, None, None, None];
    let mut ac_tabs: [Option<Huff>; 4] = [None, None, None, None];
    let mut comps: Vec<JComp> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut restart_interval = 0usize;
    let mut i = 2usize;

    loop {
        if i + 4 > b.len() {
            return Err("jpeg: truncated before SOS".into());
        }
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = b[i + 1];
        i += 2;
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            continue; // standalone markers
        }
        let len = ((b[i] as usize) << 8 | b[i + 1] as usize).max(2);
        let seg = &b[i + 2..(i + len).min(b.len())];
        match marker {
            0xC0 | 0xC1 => {
                // SOF0 / SOF1
                if seg[0] != 8 {
                    return Err(format!(
                        "jpeg: {}-bit precision is not supported (8-bit baseline only)",
                        seg[0]
                    ));
                }
                height = (seg[1] as usize) << 8 | seg[2] as usize;
                width = (seg[3] as usize) << 8 | seg[4] as usize;
                if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
                    return Err(format!("jpeg: invalid dimensions {}x{}", width, height));
                }
                let nc = seg[5] as usize;
                if nc != 1 && nc != 3 {
                    return Err(format!("jpeg: {} components not supported (grayscale or YCbCr only; CMYK JPEGs are rejected)", nc));
                }
                for c in 0..nc {
                    let o = 6 + c * 3;
                    comps.push(JComp {
                        id: seg[o],
                        h: (seg[o + 1] >> 4) as usize,
                        v: (seg[o + 1] & 0xF) as usize,
                        tq: seg[o + 2] as usize,
                        td: 0,
                        ta: 0,
                        plane: Vec::new(),
                        pw: 0,
                        ph: 0,
                        dc_pred: 0,
                    });
                }
            }
            0xC2 => {
                return Err(
                    "jpeg: progressive JPEG (SOF2) is not supported — re-encode as baseline".into(),
                )
            }
            // SOF family is 0xC0–0xCF *except* C4 (DHT), C8 (reserved) and
            // CC (DAC, arithmetic conditioning — also unsupported).
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCC | 0xCD..=0xCF => {
                return Err(format!(
                    "jpeg: SOF marker 0x{:02X} (non-baseline/arithmetic coding) is not supported",
                    marker
                ))
            }
            0xC4 => {
                // DHT — possibly several tables in one segment
                let mut o = 0usize;
                while o + 17 <= seg.len() {
                    let class = seg[o] >> 4;
                    let id = (seg[o] & 0xF) as usize;
                    if id > 3 {
                        return Err("jpeg: huffman table id > 3".into());
                    }
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(&seg[o + 1..o + 17]);
                    let n: usize = bits.iter().map(|&x| x as usize).sum();
                    if o + 17 + n > seg.len() {
                        return Err("jpeg: DHT segment truncated".into());
                    }
                    let vals = seg[o + 17..o + 17 + n].to_vec();
                    let t = Huff::build(&bits, vals);
                    if class == 0 {
                        dc_tabs[id] = Some(t);
                    } else {
                        ac_tabs[id] = Some(t);
                    }
                    o += 17 + n;
                }
            }
            0xDB => {
                // DQT — 8- or 16-bit tables
                let mut o = 0usize;
                while o < seg.len() {
                    let prec = seg[o] >> 4;
                    let id = (seg[o] & 0xF) as usize;
                    if id > 3 {
                        return Err("jpeg: quant table id > 3".into());
                    }
                    o += 1;
                    for &zz in ZIGZAG.iter() {
                        let q = if prec == 0 {
                            let v = seg[o] as f32;
                            o += 1;
                            v
                        } else {
                            let v = ((seg[o] as u32) << 8 | seg[o + 1] as u32) as f32;
                            o += 2;
                            v
                        };
                        qt[id][zz] = q;
                    }
                }
            }
            0xDD => {
                restart_interval = (seg[0] as usize) << 8 | seg[1] as usize;
            }
            0xDA => {
                // SOS — bind tables, then decode the scan
                let ns = seg[0] as usize;
                if ns != comps.len() {
                    return Err("jpeg: non-interleaved multi-scan files are not supported (baseline single scan only)".into());
                }
                for s in 0..ns {
                    let cid = seg[1 + s * 2];
                    let tt = seg[2 + s * 2];
                    let comp = comps
                        .iter_mut()
                        .find(|c| c.id == cid)
                        .ok_or_else(|| "jpeg: SOS references unknown component".to_string())?;
                    comp.td = (tt >> 4) as usize;
                    comp.ta = (tt & 0xF) as usize;
                }
                let scan_start = i + len;
                return decode_scan(
                    b,
                    scan_start,
                    width,
                    height,
                    &mut comps,
                    &qt,
                    &dc_tabs,
                    &ac_tabs,
                    restart_interval,
                );
            }
            0xD9 => return Err("jpeg: EOI before SOS (no image data)".into()),
            _ => {} // APPn, COM, …
        }
        i += len;
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    b: &[u8],
    start: usize,
    width: usize,
    height: usize,
    comps: &mut [JComp],
    qt: &[[f32; 64]; 4],
    dc_tabs: &[Option<Huff>; 4],
    ac_tabs: &[Option<Huff>; 4],
    restart_interval: usize,
) -> R<(usize, usize, Vec<u8>)> {
    if width == 0 || height == 0 {
        return Err("jpeg: SOS before SOF (no frame header)".into());
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    if hmax == 0 || vmax == 0 || hmax > 4 || vmax > 4 {
        return Err("jpeg: invalid sampling factors".into());
    }
    let mcux = width.div_ceil(8 * hmax);
    let mcuy = height.div_ceil(8 * vmax);
    for c in comps.iter_mut() {
        c.pw = mcux * 8 * c.h;
        c.ph = mcuy * 8 * c.v;
        c.plane = vec![0u8; c.pw * c.ph];
    }

    let mut br = BitReader::new(b, start);
    let mut coef: [f32; 64];
    let mut block = [0u8; 64];
    let mut mcu_count = 0usize;

    for my in 0..mcuy {
        for mx in 0..mcux {
            if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                br.restart()?;
                for c in comps.iter_mut() {
                    c.dc_pred = 0;
                }
            }
            mcu_count += 1;
            // ci indexes comps, which is also mutably indexed (dc_pred,
            // plane) deeper in the body; a by-ref iterator would conflict.
            #[allow(clippy::needless_range_loop)]
            for ci in 0..comps.len() {
                let (h, v, tq, td, ta) = {
                    let c = &comps[ci];
                    (c.h, c.v, c.tq, c.td, c.ta)
                };
                let dc = dc_tabs[td].as_ref().ok_or_else(|| {
                    format!("jpeg: DC huffman table {} referenced but never defined", td)
                })?;
                let ac = ac_tabs[ta].as_ref().ok_or_else(|| {
                    format!("jpeg: AC huffman table {} referenced but never defined", ta)
                })?;
                for by in 0..v {
                    for bx in 0..h {
                        coef = [0f32; 64];
                        // DC
                        let t = br.huff(dc)?;
                        if t > 15 {
                            return Err("jpeg: DC category > 15 (corrupt stream)".into());
                        }
                        let diff = br.extend(t)?;
                        comps[ci].dc_pred += diff;
                        coef[0] = comps[ci].dc_pred as f32 * qt[tq][0];
                        // AC
                        let mut k = 1usize;
                        while k < 64 {
                            let rs = br.huff(ac)?;
                            let (r, s) = ((rs >> 4) as usize, rs & 0xF);
                            if s == 0 {
                                if rs == 0xF0 {
                                    k += 16;
                                    continue;
                                }
                                break; // EOB
                            }
                            k += r;
                            if k > 63 {
                                return Err("jpeg: AC run past block end (corrupt stream)".into());
                            }
                            coef[ZIGZAG[k]] = br.extend(s)? as f32 * qt[tq][ZIGZAG[k]];
                            k += 1;
                        }
                        idct8x8(&coef, &mut block, 8);
                        // place the 8×8 block into the component plane
                        let px0 = (mx * h + bx) * 8;
                        let py0 = (my * v + by) * 8;
                        let (pw, ph) = (comps[ci].pw, comps[ci].ph);
                        for y in 0..8 {
                            if py0 + y >= ph {
                                break;
                            }
                            let row = (py0 + y) * pw + px0;
                            for x in 0..8 {
                                if px0 + x >= pw {
                                    break;
                                }
                                comps[ci].plane[row + x] = block[y * 8 + x];
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- color conversion + nearest-neighbour chroma upsampling ----
    let mut rgb = vec![0u8; width * height * 3];
    if comps.len() == 1 {
        let c = &comps[0];
        for y in 0..height {
            for x in 0..width {
                let g = c.plane[y.min(c.ph - 1) * c.pw + x.min(c.pw - 1)];
                let o = (y * width + x) * 3;
                rgb[o] = g;
                rgb[o + 1] = g;
                rgb[o + 2] = g;
            }
        }
    } else {
        // Centered bilinear upsampling — the float equivalent of libjpeg's
        // "fancy" triangular filter, which reference decoders apply.
        let sample = |c: &JComp, x: usize, y: usize| -> f32 {
            let fx = (x as f32 + 0.5) * c.h as f32 / hmax as f32 - 0.5;
            let fy = (y as f32 + 0.5) * c.v as f32 / vmax as f32 - 0.5;
            let (x0, y0) = (fx.floor(), fy.floor());
            let (tx, ty) = (fx - x0, fy - y0);
            let cl = |v: f32, max: usize| (v.max(0.0) as usize).min(max - 1);
            let (x0i, x1i) = (cl(x0, c.pw), cl(x0 + 1.0, c.pw));
            let (y0i, y1i) = (cl(y0, c.ph), cl(y0 + 1.0, c.ph));
            let p = |xx: usize, yy: usize| c.plane[yy * c.pw + xx] as f32;
            let top = p(x0i, y0i) * (1.0 - tx) + p(x1i, y0i) * tx;
            let bot = p(x0i, y1i) * (1.0 - tx) + p(x1i, y1i) * tx;
            top * (1.0 - ty) + bot * ty
        };
        for y in 0..height {
            for x in 0..width {
                let yy = sample(&comps[0], x, y);
                let cb = sample(&comps[1], x, y) - 128.0;
                let cr = sample(&comps[2], x, y) - 128.0;
                let o = (y * width + x) * 3;
                rgb[o] = (yy + 1.402 * cr).clamp(0.0, 255.0) as u8;
                rgb[o + 1] = (yy - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
                rgb[o + 2] = (yy + 1.772 * cb).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok((width, height, rgb))
}

// ===========================================================================
// PNG (RFC 2083) + DEFLATE (RFC 1951)
// ===========================================================================

/// LSB-first DEFLATE bit reader.
struct ZBits<'a> {
    d: &'a [u8],
    pos: usize,
    bit: u32,
}

impl ZBits<'_> {
    fn take(&mut self, n: u32) -> R<u32> {
        let mut v = 0u32;
        for i in 0..n {
            if self.pos >= self.d.len() {
                return Err("png: deflate stream truncated".into());
            }
            v |= (((self.d[self.pos] >> self.bit) & 1) as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

/// Canonical Huffman decoder built from code lengths (RFC 1951 §3.2.2).
struct ZHuff {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl ZHuff {
    fn build(lengths: &[u8]) -> R<ZHuff> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        // over-subscription check
        let mut left = 1i32;
        // l is the Huffman code length (1..16); it drives the <<= shift and
        // indexes counts by that length — the index is the quantity.
        #[allow(clippy::needless_range_loop)]
        for l in 1..16 {
            left <<= 1;
            left -= counts[l] as i32;
            if left < 0 {
                return Err("png: over-subscribed huffman code lengths".into());
            }
        }
        let mut offs = [0u16; 16];
        for l in 1..15 {
            offs[l + 1] = offs[l] + counts[l];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(ZHuff { counts, symbols })
    }

    fn decode(&self, b: &mut ZBits) -> R<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for l in 1..16usize {
            code |= b.take(1)? as i32;
            let count = self.counts[l] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("png: invalid huffman code in deflate stream".into())
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// RFC 1951 inflate of a zlib stream (RFC 1950 wrapper).
fn inflate_zlib(d: &[u8], size_hint: usize) -> R<Vec<u8>> {
    if d.len() < 2 {
        return Err("png: zlib stream too short".into());
    }
    if d[0] & 0x0F != 8 {
        return Err(format!(
            "png: zlib compression method {} (only deflate=8)",
            d[0] & 0x0F
        ));
    }
    if d[1] & 0x20 != 0 {
        return Err("png: zlib preset dictionary is not supported".into());
    }
    let mut b = ZBits { d, pos: 2, bit: 0 };
    let mut out: Vec<u8> = Vec::with_capacity(size_hint);
    loop {
        let bfinal = b.take(1)?;
        let btype = b.take(2)?;
        match btype {
            0 => {
                b.align();
                if b.pos + 4 > d.len() {
                    return Err("png: stored block header truncated".into());
                }
                let len = d[b.pos] as usize | (d[b.pos + 1] as usize) << 8;
                let nlen = d[b.pos + 2] as usize | (d[b.pos + 3] as usize) << 8;
                if len ^ 0xFFFF != nlen {
                    return Err("png: stored block LEN/NLEN mismatch".into());
                }
                b.pos += 4;
                if b.pos + len > d.len() {
                    return Err("png: stored block data truncated".into());
                }
                out.extend_from_slice(&d[b.pos..b.pos + len]);
                b.pos += len;
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 {
                    // fixed tables
                    let mut ll = [0u8; 288];
                    for (i, l) in ll.iter_mut().enumerate() {
                        *l = match i {
                            0..=143 => 8,
                            144..=255 => 9,
                            256..=279 => 7,
                            _ => 8,
                        };
                    }
                    (ZHuff::build(&ll)?, ZHuff::build(&[5u8; 30])?)
                } else {
                    // dynamic tables
                    let hlit = b.take(5)? as usize + 257;
                    let hdist = b.take(5)? as usize + 1;
                    let hclen = b.take(4)? as usize + 4;
                    const ORDER: [usize; 19] = [
                        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                    ];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) {
                        cl[o] = b.take(3)? as u8;
                    }
                    let clh = ZHuff::build(&cl)?;
                    let mut lengths = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < lengths.len() {
                        let sym = clh.decode(&mut b)?;
                        match sym {
                            0..=15 => {
                                lengths[i] = sym as u8;
                                i += 1;
                            }
                            16 => {
                                if i == 0 {
                                    return Err(
                                        "png: code-length repeat with no previous code".into()
                                    );
                                }
                                let prev = lengths[i - 1];
                                let n = b.take(2)? as usize + 3;
                                for _ in 0..n {
                                    if i >= lengths.len() {
                                        return Err("png: code-length repeat overflow".into());
                                    }
                                    lengths[i] = prev;
                                    i += 1;
                                }
                            }
                            17 | 18 => {
                                let n = if sym == 17 {
                                    b.take(3)? as usize + 3
                                } else {
                                    b.take(7)? as usize + 11
                                };
                                i += n;
                                if i > lengths.len() {
                                    return Err("png: code-length zero-run overflow".into());
                                }
                            }
                            _ => return Err("png: invalid code-length symbol".into()),
                        }
                    }
                    (
                        ZHuff::build(&lengths[..hlit])?,
                        ZHuff::build(&lengths[hlit..])?,
                    )
                };
                loop {
                    let sym = lit.decode(&mut b)?;
                    match sym {
                        0..=255 => out.push(sym as u8),
                        256 => break,
                        257..=285 => {
                            let li = (sym - 257) as usize;
                            let len = LEN_BASE[li] as usize + b.take(LEN_EXTRA[li])? as usize;
                            let ds = dist.decode(&mut b)? as usize;
                            if ds > 29 {
                                return Err("png: invalid distance symbol".into());
                            }
                            let dv = DIST_BASE[ds] as usize + b.take(DIST_EXTRA[ds])? as usize;
                            if dv > out.len() {
                                return Err("png: back-reference before stream start".into());
                            }
                            let from = out.len() - dv;
                            for k in 0..len {
                                let v = out[from + k];
                                out.push(v);
                            }
                        }
                        _ => return Err("png: invalid literal/length symbol".into()),
                    }
                }
            }
            _ => return Err("png: reserved deflate block type 3".into()),
        }
        if bfinal == 1 {
            return Ok(out);
        }
    }
}

/// Decode a PNG into `(width, height, interleaved RGB)`.
pub fn decode_png(b: &[u8]) -> R<(usize, usize, Vec<u8>)> {
    const SIG: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if b.len() < 8 || b[..8] != SIG {
        return Err("png: missing signature".into());
    }
    let (mut w, mut h, mut depth, mut ctype) = (0usize, 0usize, 0u8, 0u8);
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut i = 8usize;
    while i + 8 <= b.len() {
        let len = u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        let ctyp = &b[i + 4..i + 8];
        let data = &b[i + 8..(i + 8 + len).min(b.len())];
        match ctyp {
            b"IHDR" => {
                if data.len() < 13 {
                    return Err("png: IHDR truncated".into());
                }
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                depth = data[8];
                ctype = data[9];
                if data[12] != 0 {
                    return Err(
                        "png: Adam7 interlaced PNGs are not supported — re-save non-interlaced"
                            .into(),
                    );
                }
                if depth == 16 {
                    return Err("png: 16-bit channel depth is not supported (8-bit max)".into());
                }
                crate::log::info(&format!(
                    "png: {}x{} color_type={} ({}) bit_depth={}",
                    w,
                    h,
                    ctype,
                    match ctype {
                        0 => "grayscale",
                        2 => "rgb",
                        3 => "palette",
                        4 => "gray+alpha",
                        6 => "rgba",
                        _ => "?",
                    },
                    depth
                ));
            }
            b"PLTE" => {
                for p in data.chunks_exact(3) {
                    palette.push([p[0], p[1], p[2]]);
                }
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        i += 12 + len; // len + type + crc
    }
    // Bound dimensions BEFORE any size arithmetic: w·h·channels on
    // attacker-controlled u32 dims overflows usize math (wrapping in release
    // → under-allocation). 16384² is far beyond any model input size.
    const MAX_DIM: usize = 16_384;
    if w > MAX_DIM || h > MAX_DIM {
        return Err(format!(
            "png: dimensions {}x{} exceed the {} limit",
            w, h, MAX_DIM
        ));
    }
    if w == 0 || h == 0 {
        return Err("png: missing or empty IHDR".into());
    }
    // channels per pixel and bits per pixel for the filter math
    let channels = match ctype {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return Err(format!("png: color type {} is not valid", ctype)),
    };
    if (ctype == 2 || ctype == 4 || ctype == 6) && depth != 8 {
        return Err(format!(
            "png: color type {} requires 8-bit depth in this decoder (got {})",
            ctype, depth
        ));
    }
    if (ctype == 0 || ctype == 3) && !matches!(depth, 1 | 2 | 4 | 8) {
        return Err(format!(
            "png: bit depth {} is not valid for color type {}",
            depth, ctype
        ));
    }

    let bits_pp = channels * depth as usize;
    let row_bytes = (w * bits_pp).div_ceil(8);
    let raw = inflate_zlib(&idat, (row_bytes + 1) * h)?;
    if raw.len() < (row_bytes + 1) * h {
        return Err(format!(
            "png: decompressed to {} bytes, need {} ({}×{} rows)",
            raw.len(),
            (row_bytes + 1) * h,
            h,
            row_bytes + 1
        ));
    }

    // ---- defilter ----
    let bpp = bits_pp.div_ceil(8).max(1);
    let mut img = vec![0u8; row_bytes * h];
    for y in 0..h {
        let f = raw[y * (row_bytes + 1)];
        let src = &raw[y * (row_bytes + 1) + 1..y * (row_bytes + 1) + 1 + row_bytes];
        for x in 0..row_bytes {
            let a = if x >= bpp {
                img[y * row_bytes + x - bpp]
            } else {
                0
            };
            let bb = if y > 0 {
                img[(y - 1) * row_bytes + x]
            } else {
                0
            };
            let c = if x >= bpp && y > 0 {
                img[(y - 1) * row_bytes + x - bpp]
            } else {
                0
            };
            let v = src[x];
            img[y * row_bytes + x] = match f {
                0 => v,
                1 => v.wrapping_add(a),
                2 => v.wrapping_add(bb),
                3 => v.wrapping_add(((a as u16 + bb as u16) / 2) as u8),
                4 => {
                    // Paeth
                    let p = a as i32 + bb as i32 - c as i32;
                    let (pa, pb, pc) = (
                        (p - a as i32).abs(),
                        (p - bb as i32).abs(),
                        (p - c as i32).abs(),
                    );
                    let pred = if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        bb
                    } else {
                        c
                    };
                    v.wrapping_add(pred)
                }
                _ => return Err(format!("png: unknown scanline filter {}", f)),
            };
        }
    }

    // ---- expand to interleaved RGB ----
    let mut rgb = vec![0u8; w * h * 3];
    let sample_sub8 = |row: &[u8], x: usize| -> u8 {
        // depth 1/2/4 bit extraction with max-value scaling
        let d = depth as usize;
        let per = 8 / d;
        let byte = row[x / per];
        let shift = 8 - d * (x % per + 1);
        let v = (byte >> shift) & ((1 << d) - 1);
        if ctype == 3 {
            v // palette index, no scaling
        } else {
            (v as usize * 255 / ((1 << d) - 1)) as u8
        }
    };
    for y in 0..h {
        let row = &img[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..w {
            let o = (y * w + x) * 3;
            match ctype {
                0 => {
                    let g = if depth == 8 {
                        row[x]
                    } else {
                        sample_sub8(row, x)
                    };
                    rgb[o] = g;
                    rgb[o + 1] = g;
                    rgb[o + 2] = g;
                }
                2 => {
                    rgb[o..o + 3].copy_from_slice(&row[x * 3..x * 3 + 3]);
                }
                3 => {
                    let idx = if depth == 8 {
                        row[x]
                    } else {
                        sample_sub8(row, x)
                    } as usize;
                    let p = palette.get(idx).ok_or_else(|| {
                        format!(
                            "png: palette index {} out of range ({} entries)",
                            idx,
                            palette.len()
                        )
                    })?;
                    rgb[o..o + 3].copy_from_slice(p);
                }
                4 => {
                    let g = row[x * 2]; // alpha dropped (reference convert("RGB"))
                    rgb[o] = g;
                    rgb[o + 1] = g;
                    rgb[o + 2] = g;
                }
                6 => {
                    rgb[o..o + 3].copy_from_slice(&row[x * 4..x * 4 + 3]); // alpha dropped
                }
                _ => unreachable!(),
            }
        }
    }
    Ok((w, h, rgb))
}

// ===========================================================================
// Header-only dimension probes (for aspect-preserving target computation)
// ===========================================================================

/// JPEG dimensions from the first SOF marker (any SOFn — decode proper still
/// rejects non-baseline variants with a precise error).
pub fn jpeg_dims(b: &[u8]) -> Option<(usize, usize)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 9 < b.len() {
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = b[i + 1];
        i += 2;
        if m == 0xD8 || (0xD0..=0xD7).contains(&m) || m == 0x01 {
            continue;
        }
        let len = ((b[i] as usize) << 8 | b[i + 1] as usize).max(2);
        if matches!(m, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) && i + 7 < b.len() {
            let h = (b[i + 3] as usize) << 8 | b[i + 4] as usize;
            let w = (b[i + 5] as usize) << 8 | b[i + 6] as usize;
            return Some((w, h));
        }
        i += len;
    }
    None
}

/// PNG dimensions from IHDR.
pub fn png_dims(b: &[u8]) -> Option<(usize, usize)> {
    if b.len() < 24 || !b.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return None;
    }
    let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]) as usize;
    let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]) as usize;
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Garbage and truncated streams must error, never panic or OOM.
    #[test]
    fn png_garbage_errors() {
        assert!(decode_png(b"").is_err());
        assert!(decode_png(b"\x89PNG\r\n\x1a\n").is_err()); // magic only
        assert!(decode_png(&[0u8; 64]).is_err());
        // valid magic + IHDR claiming insane dims must be rejected before allocation
        let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
        b.extend_from_slice(&13u32.to_be_bytes());
        b.extend_from_slice(b"IHDR");
        b.extend_from_slice(&u32::MAX.to_be_bytes()); // width
        b.extend_from_slice(&u32::MAX.to_be_bytes()); // height
        b.extend_from_slice(&[8, 2, 0, 0, 0]);
        b.extend_from_slice(&[0u8; 4]); // bogus crc
        assert!(decode_png(&b).is_err());
    }

    #[test]
    fn jpeg_garbage_errors() {
        assert!(decode_jpeg(b"").is_err());
        assert!(decode_jpeg(&[0xFF, 0xD8]).is_err()); // SOI only
        assert!(decode_jpeg(&[0u8; 128]).is_err());
    }

    /// Dimension probes on short inputs return None — no slicing panics.
    #[test]
    fn dims_probes_short_inputs() {
        for n in 0..32 {
            let z = vec![0u8; n];
            let _ = png_dims(&z);
            let _ = jpeg_dims(&z);
        }
        assert_eq!(png_dims(b"\x89PNG\r\n\x1a\n"), None);
    }
}
