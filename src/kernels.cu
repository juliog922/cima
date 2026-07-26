
#include <cuda_fp16.h>
typedef unsigned int u32;

// bf16 <-> f32 helpers (bf16 stored as u16; no __nv_bfloat16 dependency).
__device__ inline float bf2f(unsigned short b){ u32 u=((u32)b)<<16; return __uint_as_float(u); }
__device__ inline unsigned short f2bf(float f){ u32 u=__float_as_uint(f); u32 r=(u>>16)&1; return (unsigned short)((u + 0x7fff + r)>>16); }

// ---- embedding gather: out[i,:] = table[ids[i],:] (f16/bf16 -> f16) ----
extern "C" __global__ void k_gather_f16(const __half* table, const u32* ids, __half* out, int hidden){
  int row = blockIdx.x; int t = threadIdx.x;
  const __half* src = table + (size_t)ids[row]*hidden;
  __half* dst = out + (size_t)row*hidden;
  for (int i=t; i<hidden; i+=blockDim.x) dst[i]=src[i];
}
extern "C" __global__ void k_gather_bf16(const unsigned short* table, const u32* ids, __half* out, int hidden){
  int row = blockIdx.x; int t = threadIdx.x;
  const unsigned short* src = table + (size_t)ids[row]*hidden;
  __half* dst = out + (size_t)row*hidden;
  for (int i=t; i<hidden; i+=blockDim.x) dst[i]=__float2half(bf2f(src[i]));
}

// ---- RMSNorm: y = x/rms(x) * w  (w==NULL -> scale-less normalization) ----
// Reads saturate at ±65000: gemma-3n/4 GEMM outputs legitimately spike
// past the f16 ceiling (HF runs bf16, llama.cpp f32) and every GEMM
// output in that architecture flows through exactly one rmsnorm (or the
// geglu, clamped likewise) before anything else — clamping HERE is total
// coverage of the f16 cliff: inf becomes a saturated finite spike instead
// of inf·rsqrt(inf)=NaN poisoning the KV cache. A no-op for healthy
// values (|v| < 65504 by definition of f16).
__device__ inline float g_sat16(float v){
  v = fminf(v,  65000.f);
  v = fmaxf(v, -65000.f);   // CUDA fmin/fmax drop NaN operands: NaN→finite
  return v;
}

extern "C" __global__ void k_rmsnorm(const __half* x, const __half* w, __half* y, int n, float eps){
  extern __shared__ float sh[];
  int row = blockIdx.x; int t = threadIdx.x;
  const __half* xi = x + (size_t)row*n; __half* yo = y + (size_t)row*n;
  float acc=0.f;
  for(int i=t;i<n;i+=blockDim.x){ float v=g_sat16(__half2float(xi[i])); acc+=v*v; }
  sh[t]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  float inv = rsqrtf(sh[0]/n + eps);
  for(int i=t;i<n;i+=blockDim.x){
    float g = w ? __half2float(w[i]) : 1.0f;
    yo[i]=__float2half(g_sat16(__half2float(xi[i]))*inv*g);
  }
}

// ---- LayerNorm: y = (x-mean)/sqrt(var+eps) * w + b  (ViT / audio towers) ----
extern "C" __global__ void k_layernorm(const __half* x, const __half* w, const __half* b,
        __half* y, int n, float eps){
  extern __shared__ float sh[];
  int row = blockIdx.x; int t = threadIdx.x;
  const __half* xi = x + (size_t)row*n; __half* yo = y + (size_t)row*n;
  float acc=0.f;
  for(int i=t;i<n;i+=blockDim.x) acc+=__half2float(xi[i]);
  sh[t]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  float mean = sh[0]/n; __syncthreads();
  acc=0.f;
  for(int i=t;i<n;i+=blockDim.x){ float d=__half2float(xi[i])-mean; acc+=d*d; }
  sh[t]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  float inv = rsqrtf(sh[0]/n + eps);
  for(int i=t;i<n;i+=blockDim.x){
    float bb = b ? __half2float(b[i]) : 0.0f;
    yo[i]=__float2half((__half2float(xi[i])-mean)*inv*__half2float(w[i])+bb);
  }
}

// ---- residual add: a += b ----
extern "C" __global__ void k_add(__half* a, const __half* b, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) a[i]=__float2half(__half2float(a[i])+__half2float(b[i]));
}

// ---- SwiGLU: gate = silu(gate) * up   (in place over gate) ----
extern "C" __global__ void k_swiglu(__half* gate, const __half* up, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float g=__half2float(gate[i]); float s=g/(1.f+expf(-g)); gate[i]=__float2half(s*__half2float(up[i])); }
}

// ---- GELU (tanh approx) for ViT / audio towers ----
extern "C" __global__ void k_gelu(__half* x, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=__half2float(x[i]);
    float c=0.7978845608f*(v+0.044715f*v*v*v);
    x[i]=__float2half(0.5f*v*(1.f+tanhf(c))); }
}

// ---- GeGLU: gate = gelu_tanh(gate) * up  (Gemma4 text/vision MLP) ----
// gemma-3n/4 FFN pre-activations spike past the f16 ceiling (the reason
// HF runs these models in bf16 and llama.cpp in f32): the gate/up GEMMs
// store ±inf, and gelu(inf)·0 births the NaN that poisons the KV cache.
// Saturate the inputs just below the cliff — a clipped spike costs a few
// percent on one channel; an unclipped one costs the whole model.
extern "C" __global__ void k_geglu(__half* gate, const __half* up, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){
    float v = fminf(fmaxf(__half2float(gate[i]), -65000.f), 65000.f);
    float u = fminf(fmaxf(__half2float(up[i]),   -65000.f), 65000.f);
    float c = 0.7978845608f*(v+0.044715f*v*v*v);
    float g = 0.5f*v*(1.f+tanhf(c));
    gate[i] = __float2half(fminf(fmaxf(g*u, -65000.f), 65000.f));
  }
}

// ---- plain SiLU in place (Gemma4 audio feed-forward) ----
extern "C" __global__ void k_silu(__half* x, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=__half2float(x[i]); x[i]=__float2half(v/(1.f+expf(-v))); }
}

// ---- ReLU in place (Gemma4 audio subsampling convs) ----
extern "C" __global__ void k_relu(__half* x, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=__half2float(x[i]); x[i]=__float2half(v>0.f?v:0.f); }
}

// ---- GLU: y[r, 0..n) = x[r, 0..n) * sigmoid(x[r, n..2n))  (audio lconv) ----
extern "C" __global__ void k_glu(const __half* x, __half* y, int rows, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<rows*n){
    int r=i/n, c=i%n;
    float a=__half2float(x[(size_t)r*2*n+c]);
    float b=__half2float(x[(size_t)r*2*n+n+c]);
    y[i]=__float2half(a/(1.f+expf(-b)));
  }
}

// ---- scalar multiply in place: x *= s  (embed scale, residual weights) ----
extern "C" __global__ void k_scalemul(__half* x, float s, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) x[i]=__float2half(__half2float(x[i])*s);
}

// ---- per-channel vector multiply: x[r,c] *= s[c]  (audio per_dim_scale) ----
extern "C" __global__ void k_mulvec(__half* x, const __half* s, int rows, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<rows*n) x[i]=__float2half(__half2float(x[i])*__half2float(s[i%n]));
}

// ---- strided elementwise multiply: a[r,c] *= b[r*stride + off + c]
// (Gemma4 per-layer-embedding gate: PLE slab is [rows, layers*ple_dim]) ----
extern "C" __global__ void k_mul_strided(__half* a, const __half* b, int rows, int n, int stride, int off){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<rows*n){
    int r=i/n, c=i%n;
    a[i]=__float2half(__half2float(a[i])*__half2float(b[(size_t)r*stride+off+c]));
  }
}

// ---- clamp in place (Gemma4 ClippableLinear bounds / gradient clipping) ----
extern "C" __global__ void k_clamp(__half* x, float lo, float hi, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=__half2float(x[i]); v=v<lo?lo:(v>hi?hi:v); x[i]=__float2half(v); }
}

// ---- causal depthwise conv1d (audio light-conv, groups == channels) ----
// x,y: [seq, C]; w: [C, K] (torch depthwise [C,1,K] squeezed). Left pad K-1.
extern "C" __global__ void k_dwconv1d(const __half* x, const __half* w, __half* y,
        int seq, int C, int K){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=seq*C) return;
  int s=i/C, c=i%C;
  float acc=0.f;
  for(int k=0;k<K;k++){
    int src = s - (K-1) + k;
    if(src>=0) acc += __half2float(w[c*K+k])*__half2float(x[(size_t)src*C+c]);
  }
  y[i]=__float2half(acc);
}

// ---- fused Gemma4 audio attention: chunked local attention with relative
// position bias and tanh logit capping (USM-style). One block per (pos, head).
// q is pre-scaled by q_scale*softplus(per_dim_scale); k pre-scaled by k_scale.
// relk: [past+1, H*D] = rel_pos_embedding @ relative_k_proj^T (raw, unscaled).
// Query at pos p (block b = p/chunk, qi = p%chunk) attends context keys
// j = b*chunk - past + kj for kj in [0, chunk+past); rel bias applies when
// r = kj - qi is in [0, past] (i.e. key at or before query, within `past`).
extern "C" __global__ void k_audio_attn(const __half* q, const __half* k, const __half* v,
        const __half* relk, __half* out, int seq, int heads, int dim,
        int chunk, int past, float cap, float invalid){
  extern __shared__ float sh[];
  float* red = sh; float* acc = sh + blockDim.x;
  int p = blockIdx.x; int h = blockIdx.y; int t = threadIdx.x;
  const __half* qh = q + ((size_t)p*heads + h)*dim;
  int blk = p / chunk, qi = p % chunk;
  int ctx = chunk + past;
  for(int i=t;i<dim;i+=blockDim.x) acc[i]=0.f;
  float m=-1e30f, l=0.f; __shared__ float bs;
  for(int kj=0;kj<ctx;kj++){
    int j = blk*chunk - past + kj;
    int r = kj - qi;
    // The block context is COMPUTE TILING only. The logical mask is a
    // per-query sliding window: key j is visible to query p iff
    // p-j in [0, past] (sliding_window_mask_function(context_left-1, 0)),
    // which in window coords is r = kj-qi in [0, past] — exactly the pairs
    // the rel-shift gives a bias row to. Everything else (window slack,
    // future-in-chunk, sequence edges) enters the softmax with
    // logit = attention_invalid_logits_value and V = 0 (reference
    // dilution semantics), not -inf skipping.
    if(j<0 || j>=seq || r<0 || r>past){
      float score = invalid;
      float m2 = fmaxf(m, score);
      float corr = expf(m-m2), pr = expf(score-m2);
      for(int i=t;i<dim;i+=blockDim.x) acc[i] *= corr;
      l = l*corr + pr; m = m2;
      __syncthreads();
      continue;
    }
    float d=0.f;
    const __half* kh = k + ((size_t)j*heads + h)*dim;
    const __half* rh = relk + ((size_t)r*heads + h)*dim;
    for(int i=t;i<dim;i+=blockDim.x){
      float qv=__half2float(qh[i]);
      d += qv*__half2float(kh[i]);
      d += qv*__half2float(rh[i]);
    }
    red[t]=d; __syncthreads();
    for(int s2=blockDim.x/2;s2>0;s2>>=1){ if(t<s2) red[t]+=red[t+s2]; __syncthreads(); }
    if(t==0) bs = cap*tanhf(red[0]/cap); __syncthreads();
    float score = bs;
    float m2 = fmaxf(m, score);
    float corr = expf(m-m2), pr = expf(score-m2);
    const __half* vh = v + ((size_t)j*heads + h)*dim;
    for(int i=t;i<dim;i+=blockDim.x) acc[i] = acc[i]*corr + pr*__half2float(vh[i]);
    l = l*corr + pr; m = m2;
    __syncthreads();
  }
  __half* oh = out + ((size_t)p*heads + h)*dim;
  for(int i=t;i<dim;i+=blockDim.x) oh[i]=__float2half(l>0.f ? acc[i]/l : 0.f);
}

// ---- fused softcap + argmax over f16 logits ----
// Greedy decoding never needs the 1 MB logits row on the host: cap+argmax
// on device and copy back 8 bytes. Result slot packs (orderable float bits
// << 32 | index) so one u64 atomicMax resolves both; float bits are made
// monotonically orderable with the sign-flip trick. Ties resolve to the
// higher index (f32 logit ties are astronomically rare).
extern "C" __global__ void k_argmax_softcap(const __half* x, unsigned long long* out,
        int n, float cap){
  unsigned long long best = 0ULL;
  for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x){
    float v = __half2float(x[i]);
    if(cap > 0.f) v = cap * tanhf(v / cap);
    unsigned int b = __float_as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    unsigned long long packed = ((unsigned long long)b << 32) | (unsigned int)i;
    if(packed > best) best = packed;
  }
  __shared__ unsigned long long red[256];
  red[threadIdx.x] = best; __syncthreads();
  for(int s = blockDim.x/2; s > 0; s >>= 1){
    if(threadIdx.x < s && red[threadIdx.x+s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x+s];
    __syncthreads();
  }
  if(threadIdx.x == 0) atomicMax(out, red[0]);
}

// ---- device sampling support: the sampler chain's heavy half on GPU ----
// The host sampler truncates to top-k (<= 64 by default) BEFORE top-p, so
// shipping the top-64 candidates is exact: repeat penalty applies on
// device (counts maintained by k_hist_push over the same 64-token window
// the host sampler uses), then 64 iterations of argmax + mask record the
// candidates in descending order. 512 bytes cross PCIe instead of the
// vocab-sized logits row, and the whole tail lives inside the CUDA graph.

// rp^count penalty, matching DefaultSampler: positive logits divide,
// negative multiply (count = occurrences in the last-64 window).
extern "C" __global__ void k_apply_penalty(__half* x, const unsigned int* counts, float rp, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  unsigned int c = counts[i];
  if(c == 0u) return;
  float f = powf(rp, (float)c);
  float v = __half2float(x[i]);
  x[i] = __float2half(v > 0.f ? v / f : v * f);
}

// Record the current argmax slot into out[j] and mask the winner with
// -inf so the next argmax finds the runner-up.
extern "C" __global__ void k_slot_take(const unsigned long long* slot, unsigned long long* out_j, __half* x){
  if(blockIdx.x || threadIdx.x) return;
  unsigned long long s = *slot;
  *out_j = s;
  x[(unsigned int)s] = __ushort_as_half((unsigned short)0xFC00u); // -inf f16
}

// Slide the 64-token penalty window: evict the entry this ring slot held,
// admit ids[0]. Slot index comes from the device position counter when
// `pos_dev` is non-null (steady state) or from `idx` (prompt seeding).
extern "C" __global__ void k_hist_push(unsigned int* ring, unsigned int* counts,
        const unsigned int* ids, int idx, const int* pos_dev){
  if(blockIdx.x || threadIdx.x) return;
  int p = (pos_dev ? *pos_dev : idx) & 63;
  unsigned int old = ring[p];
  if(counts[old] > 0u) counts[old]--;
  unsigned int t = ids[0];
  ring[p] = t;
  counts[t]++;
}

// Closes a captured decode step: advances the device position counter so
// the next replay of the same graph operates on the following position.
extern "C" __global__ void k_pos_bump(int* pos_dev){
  if(blockIdx.x == 0 && threadIdx.x == 0) (*pos_dev)++;
}

// Companion to k_argmax_softcap: unpack the winning index into the token
// ids buffer, so the next decode step can gather embeddings without a
// host round-trip (device-resident greedy feedback).
extern "C" __global__ void k_argmax_extract(const unsigned long long* slot, unsigned int* ids){
  if(blockIdx.x == 0 && threadIdx.x == 0) ids[0] = (unsigned int)(*slot);
}

// ---- bitsandbytes NF4/FP4 dequant: packed nibbles -> f16 ----
// w[idx] = qmap[nibble(idx)] * absmax[idx / blocksize]; high nibble = even idx.
extern "C" __global__ void k_nf4_dequant(const unsigned char* packed, const float* absmax,
        const float* qmap, __half* out, int n, int blocksize){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  unsigned char b = packed[i>>1];
  int nib = (i & 1) ? (b & 0x0F) : (b >> 4);
  out[i] = __float2half(qmap[nib] * absmax[i/blocksize]);
}

// ---- fused NF4 GEMV (decode path, m == 1): y[n] = x[k] . W[n,k]^T ----
// One block per output row; threads stride over k, dequantizing in registers.
// Reads n*k/2 weight bytes instead of n*k*2 — the decode GEMV is bandwidth
// bound, so packed residency is a speedup, not a compromise.
// Reference NF4 GEMV (block-per-row, byte loads). Kept for shapes the
// fast path can't take (k or blocksize not a multiple of 8).
extern "C" __global__ void k_nf4_gemv_ref(const __half* x, const unsigned char* packed,
        const float* absmax, const float* qmap, __half* y, int n, int k, int blocksize){
  extern __shared__ float sh[];
  __shared__ float code[16];
  int row = blockIdx.x; int t = threadIdx.x;
  if(t<16) code[t] = qmap[t];
  __syncthreads();
  size_t brow = (size_t)row*k/2;
  float acc = 0.f;
  for(int jb=t;jb<k/2;jb+=blockDim.x){
    unsigned char b = packed[brow + jb];
    float a = absmax[((size_t)row*k + 2*jb)/blocksize];
    acc += __half2float(x[2*jb])   * code[b >> 4]   * a;
    acc += __half2float(x[2*jb+1]) * code[b & 0x0F] * a;
  }
  sh[t]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  if(t==0) y[row]=__float2half(sh[0]);
}

// Fast NF4 GEMV: the house warp-per-row pattern (see k_gemv_f16). Each
// lane consumes one u32 of packed nibbles per iteration — 8 elements per
// load, 128-byte coalesced per warp — with a single absmax fetch per word
// (valid because blocksize % 8 == 0 keeps a word inside one quant block).
// Nibble order matches the reference: element 2j uses the HIGH nibble.
extern "C" __global__ void k_nf4_gemv(const __half* x, const unsigned char* packed,
        const float* absmax, const float* qmap, __half* y, int n, int k, int blocksize){
  __shared__ float code[16];
  int t = threadIdx.x;
  if(t<16) code[t] = qmap[t];
  __syncthreads();
  int row = blockIdx.x*(blockDim.x>>5) + (t>>5);
  if(row >= n) return;
  int lane = t & 31;
  const unsigned int* pw = (const unsigned int*)(packed + (size_t)row*k/2);
  const __half2* x2 = (const __half2*)x;
  int words = k >> 3;                       // u32 words per row
  float acc = 0.f;
  for(int w=lane; w<words; w+=32){
    unsigned int p = pw[w];
    float a = absmax[((size_t)row*k + (size_t)w*8)/blocksize];
    #pragma unroll
    for(int b=0;b<4;b++){
      unsigned int byte = (p >> (8*b)) & 0xFFu;
      float2 xv = __half22float2(x2[w*4 + b]);
      acc += a * (xv.x * code[byte >> 4] + xv.y * code[byte & 0x0F]);
    }
  }
  for(int o=16;o>0;o>>=1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if(lane==0) y[row]=__float2half(acc);
}

// ---- bias add ----
extern "C" __global__ void k_bias(__half* x, const __half* b, int rows, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<rows*n) x[i]=__float2half(__half2float(x[i])+__half2float(b[i%n]));
}

// ---- RoPE (NeoX/Llama interleaved-half layout), applied in place over q or k.
// rows = tokens, heads*dim per row. pos0 = absolute position of row 0.
extern "C" __global__ void k_rope(__half* x, int heads, int dim, int pos0, float theta, int nfreqs,
        const int* pos_dev, const float* factors){
  int row = blockIdx.x;             // token index
  int h   = blockIdx.y;             // head index
  int i   = threadIdx.x;            // rotary pair index < dim/2
  if(i>=dim/2) return;
  if(i>=nfreqs) return;             // proportional RoPE: zero-frequency pairs pass through
  if(pos_dev) pos0 = *pos_dev;      // device counter (CUDA-graph replay)
  __half* v = x + ((size_t)row*heads + h)*dim;
  float p = (float)(pos0+row);
  float freq = powf(theta, -2.0f*(float)i/(float)dim);
  if(factors) freq /= factors[i];      // proportional-RoPE frequency factors
  float c = cosf(p*freq), s = sinf(p*freq);
  float a = __half2float(v[i]);
  float b = __half2float(v[i+dim/2]);
  v[i]        = __float2half(a*c - b*s);
  v[i+dim/2]  = __float2half(a*s + b*c);
}

// ---- 2-D RoPE for the Gemma4 vision tower. The head_dim is split in two
// halves; the first rotates with the patch x-coordinate, the second with the
// y-coordinate. Within each half, classic rotate_half pairing over spatial
// dim = dim/2 with inv_freq exponent over (dim/2)/2 frequencies.
extern "C" __global__ void k_rope2d(__half* x, const int* posx, const int* posy,
        int heads, int dim, float theta){
  int row = blockIdx.x; int h = blockIdx.y; int i = threadIdx.x;
  int half = dim/2;            // per-spatial-dim channels
  int quarter = half/2;        // rotate_half pairs per spatial dim
  if(i>=half) return;
  __half* v = x + ((size_t)row*heads + h)*dim;
  int sd = i / quarter;        // 0 -> x dim, 1 -> y dim
  int j  = i % quarter;        // pair index within the spatial half
  float p = (float)(sd==0 ? posx[row] : posy[row]);
  float freq = powf(theta, -2.0f*(float)j/(float)half);
  float c = cosf(p*freq), s = sinf(p*freq);
  __half* base = v + sd*half;
  float a = __half2float(base[j]);
  float b = __half2float(base[j+quarter]);
  base[j]         = __float2half(a*c - b*s);
  base[j+quarter] = __float2half(a*s + b*c);
}

// ---- KV append: copy current K,V rows into the cache at position `pos` ----
extern "C" __global__ void k_kv_append(const __half* k, const __half* v,
        __half* kcache, __half* vcache, int kv_heads, int dim, int pos, int max_seq,
        const int* pos_dev){
  int row = blockIdx.x;            // token within this micro-batch
  int t = threadIdx.x;
  if(pos_dev) pos = *pos_dev;      // device counter (CUDA-graph replay)
  int n = kv_heads*dim;
  const __half* ks = k + (size_t)row*n; const __half* vs = v + (size_t)row*n;
  for(int i=t;i<n;i+=blockDim.x){
    int h=i/dim, d=i%dim;
    // cache layout: [kv_heads, max_seq, dim] — contiguous per head for GEMM-free attention.
    size_t idx = ((size_t)h*max_seq + (pos+row))*dim + d;
    kcache[idx]=ks[i]; vcache[idx]=vs[i];
  }
}

// ---- fused single-query attention (decode path; GQA-aware) ----
// One block per query head. Online softmax over the cache, f32 accumulators.
extern "C" __global__ void k_attn_decode(const __half* q, const __half* kc, const __half* vc,
        __half* out, int q_heads, int kv_heads, int dim, int seq, int max_seq, float scale, int window,
        const int* pos_dev){
  extern __shared__ float sh[];           // [blockDim.x] partial reductions + [dim] acc
  float* red = sh; float* acc = sh + blockDim.x;
  if(pos_dev) seq = *pos_dev + 1;         // device counter (CUDA-graph replay)
  int h = blockIdx.x; int t = threadIdx.x;
  int kvh = h / (q_heads/kv_heads);
  const __half* qh = q + (size_t)h*dim;
  const __half* kh = kc + (size_t)kvh*max_seq*dim;
  const __half* vh = vc + (size_t)kvh*max_seq*dim;
  for(int i=t;i<dim;i+=blockDim.x) acc[i]=0.f;
  float m=-1e30f, l=0.f;
  __shared__ float bm, bl;
  int s0 = (window>0 && seq>window) ? (seq-window) : 0;
  for(int s=s0;s<seq;s++){
    // dot(q, k_s)
    float d=0.f;
    for(int i=t;i<dim;i+=blockDim.x) d += __half2float(qh[i])*__half2float(kh[(size_t)s*dim+i]);
    red[t]=d; __syncthreads();
    for(int r=blockDim.x/2;r>0;r>>=1){ if(t<r) red[t]+=red[t+r]; __syncthreads(); }
    if(t==0){ bm = red[0]*scale; } __syncthreads();
    float score = bm;
    float m2 = fmaxf(m, score);
    float corr = expf(m-m2), p = expf(score-m2);
    // rescale accumulator and add p * v_s
    for(int i=t;i<dim;i+=blockDim.x) acc[i] = acc[i]*corr + p*__half2float(vh[(size_t)s*dim+i]);
    l = l*corr + p; m = m2;
    __syncthreads();
  }
  if(t==0){ bl=l; } __syncthreads();
  __half* oh = out + (size_t)h*dim;
  for(int i=t;i<dim;i+=blockDim.x) oh[i]=__float2half(acc[i]/bl);
}

// ---- decode GEMV: y[n] = W[n,k] · x[k], one warp per output row ----
// cuBLAS is unbeatable on huge GEMVs (the lm head) but pays a fixed wave
// ramp that dominates the small per-layer matrices of compact models; a
// warp-per-row kernel with half2 loads reaches the bandwidth those sizes
// allow, and epilogues fold the surrounding elementwise kernels away:
//   mode 0: y = dot (+ bias[row] if bias != 0)
//   mode 1: y[row] += dot                       (residual projections)
//   mode 2: gate|up pair — W is [2n,k]; row block computes rows i and n+i
//           and writes y[i] = silu(g + bias_g) * (u + bias_u); bias may be 0
extern "C" __global__ void k_gemv_f16(const __half* W, const __half* x, __half* y,
        const __half* bias, int n, int k, int mode){
  int warps = blockDim.x >> 5;
  int row = blockIdx.x*warps + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if(row >= n) return;
  const __half2* x2 = (const __half2*)x;
  int k2 = k >> 1;
  float acc = 0.f, acc2 = 0.f;
  const __half2* w2 = (const __half2*)(W + (size_t)row*k);
  for(int i=lane;i<k2;i+=32){
    float2 a = __half22float2(w2[i]), b = __half22float2(x2[i]);
    acc += a.x*b.x + a.y*b.y;
  }
  if(mode == 2){
    const __half2* u2 = (const __half2*)(W + (size_t)(row+n)*k);
    for(int i=lane;i<k2;i+=32){
      float2 a = __half22float2(u2[i]), b = __half22float2(x2[i]);
      acc2 += a.x*b.x + a.y*b.y;
    }
  }
  for(int o=16;o>0;o>>=1){
    acc  += __shfl_down_sync(0xffffffffu, acc,  o);
    acc2 += __shfl_down_sync(0xffffffffu, acc2, o);
  }
  if(lane != 0) return;
  if(mode == 0){
    if(bias) acc += __half2float(bias[row]);
    y[row] = __float2half(acc);
  } else if(mode == 1){
    y[row] = __float2half(__half2float(y[row]) + acc);
  } else {
    if(bias){ acc += __half2float(bias[row]); acc2 += __half2float(bias[row+n]); }
    float g = acc / (1.f + expf(-acc));   // silu
    y[row] = __float2half(g * acc2);
  }
}

// ---- split decode attention (flash-decode): one block per (head, chunk) ----
// The monolithic decode kernel walks the whole sequence serially (two block
// syncs per position); here each block reduces one fixed-size chunk of the
// sequence with the same online softmax, emitting an unnormalized partial
// {m, l, acc[dim]}, and k_attn_reduce combines partials per head with the
// log-sum-exp identity. The grid is sized for max_seq so a captured CUDA
// graph replays at any current length: blocks beyond `seq` exit early.
extern "C" __global__ void k_attn_decode_split(const __half* q, const __half* kc, const __half* vc,
        float* part, int q_heads, int kv_heads, int dim, int seq, int max_seq,
        int csz, int n_chunks, float scale, int window, const int* pos_dev){
  // Warp-parallel within the chunk: each of the 4 warps walks every 4th
  // position keeping its online softmax entirely in registers (each lane
  // owns dim/32 components of the accumulator; the q·k dot reduces with a
  // shuffle butterfly so every lane sees the score). No __syncthreads in
  // the position loop — the only block sync is the final 4-way merge.
  // Requires dim % 32 == 0 and dim <= 512. The 512 ceiling covers gemma4's
  // global_head_dim: without it those layers fall back to k_attn_decode,
  // which is one block per head walking every key with ~4 __syncthreads
  // apiece — measured at 8.7 tok/s on a 3164-token context.
  const int R = dim >> 5;                   // accumulator regs per lane
  if(pos_dev) seq = *pos_dev + 1;
  int h = blockIdx.x / n_chunks, ci = blockIdx.x % n_chunks;
  int wlo = (window>0 && seq>window) ? (seq-window) : 0;
  int s0 = max(ci*csz, wlo), s1 = min(ci*csz+csz, seq);
  if(s0 >= s1) return;
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int kvh = h / (q_heads/kv_heads);
  const __half* qh = q + (size_t)h*dim;
  const __half* kh = kc + (size_t)kvh*max_seq*dim;
  const __half* vh = vc + (size_t)kvh*max_seq*dim;
  float qreg[16];
  #pragma unroll
  for(int r=0;r<16;r++) qreg[r] = (r<R) ? __half2float(qh[r*32+lane]) : 0.f;
  float acc[16] = {0.f};
  float m = -1e30f, l = 0.f;
  for(int s=s0+wid;s<s1;s+=nw){
    const __half* ks = kh + (size_t)s*dim;
    float d = 0.f;
    #pragma unroll
    for(int r=0;r<16;r++) if(r<R) d += qreg[r]*__half2float(ks[r*32+lane]);
    #pragma unroll
    for(int o=16;o>0;o>>=1) d += __shfl_xor_sync(0xffffffffu, d, o);
    float score = d*scale;
    float m2 = fmaxf(m, score), corr = expf(m-m2), p = expf(score-m2);
    const __half* vs = vh + (size_t)s*dim;
    #pragma unroll
    for(int r=0;r<16;r++) if(r<R) acc[r] = acc[r]*corr + p*__half2float(vs[r*32+lane]);
    l = l*corr + p; m = m2;
  }
  // Merge the warps' partials (log-sum-exp) through shared memory.
  extern __shared__ float sh[];             // [nw, dim+2]
  float* mine = sh + wid*(dim+2);
  if(lane==0){ mine[0]=m; mine[1]=l; }
  #pragma unroll
  for(int r=0;r<16;r++) if(r<R) mine[2+r*32+lane]=acc[r];
  __syncthreads();
  if(wid != 0) return;
  float M = -1e30f;
  for(int w=0;w<nw;w++) M = fmaxf(M, sh[w*(dim+2)]);
  float L = 0.f, out[16] = {0.f};
  for(int w=0;w<nw;w++){
    float* pw = sh + w*(dim+2);
    float wgt = expf(pw[0]-M);
    L += wgt*pw[1];
    #pragma unroll
    for(int r=0;r<16;r++) if(r<R) out[r] += wgt*pw[2+r*32+lane];
  }
  float* ph = part + ((size_t)h*n_chunks + ci)*(dim+2);
  if(lane==0){ ph[0]=M; ph[1]=L; }
  #pragma unroll
  for(int r=0;r<16;r++) if(r<R) ph[2+r*32+lane]=out[r];
}

// Combine the per-chunk partials of one head: M = max m_i, then
// out = sum(exp(m_i-M) acc_i) / sum(exp(m_i-M) l_i).
extern "C" __global__ void k_attn_reduce(const float* part, __half* out,
        int dim, int csz, int n_chunks, int seq, int window, const int* pos_dev){
  extern __shared__ float accr[];          // [dim]
  if(pos_dev) seq = *pos_dev + 1;
  int h = blockIdx.x, t = threadIdx.x;
  int wlo = (window>0 && seq>window) ? (seq-window) : 0;
  int c0 = wlo / csz;                       // first chunk intersecting the window
  int active = (seq + csz - 1) / csz;
  const float* ph = part + (size_t)h*n_chunks*(dim+2);
  float M=-1e30f;
  for(int c=c0;c<active;c++) M = fmaxf(M, ph[(size_t)c*(dim+2)]);
  for(int i=t;i<dim;i+=blockDim.x) accr[i]=0.f;
  __shared__ float L;
  if(t==0) L=0.f;
  __syncthreads();
  for(int c=c0;c<active;c++){
    const float* pc = ph + (size_t)c*(dim+2);
    float w = expf(pc[0]-M);
    if(t==0) L += w*pc[1];
    for(int i=t;i<dim;i+=blockDim.x) accr[i] += w*pc[2+i];
    __syncthreads();
  }
  __half* oh = out + (size_t)h*dim;
  for(int i=t;i<dim;i+=blockDim.x) oh[i]=__float2half(accr[i]/L);
}

// ---- tiled prefill attention (flash-style) --------------------------------
// One warp per query row with WARPS queries resident per block, and `bk` keys
// staged in shared memory per tile and reused across every resident query.
//
// k_attn_prefill gives one (row, head) pair to a whole block and walks the key
// sequence one key at a time, doing a block-wide tree reduction and ~10
// __syncthreads() per key, with no K/V reuse across queries. Measured on a
// 31B/L40S at 3172 tokens it accounts for ~6.9 s of an 8.6 s prefill.
//
// Here: the dot product is a warp shuffle reduction (5 shuffles, no barrier),
// each K/V tile is loaded once and consumed by all resident queries, and the
// accumulator lives in registers. Barriers drop from ~10 per key to 2 per
// tile. The online softmax, the running (m, l, acc) triple, and the
// causal/window/media-block mask semantics are unchanged.
//
// Shared memory: 2 * bk * dim * sizeof(__half), supplied dynamically.
// Grid: (ceil(nrows / WARPS), q_heads). Requires dim % 32 == 0 && dim <= 512;
// the launcher falls back to k_attn_prefill otherwise.
extern "C" __global__ void k_attn_prefill_tiled(
        const __half* q, const __half* kc, const __half* vc,
        __half* out, int q_heads, int kv_heads, int dim, int pos0,
        int max_seq, int nrows, int causal, float scale, int window,
        const int* blkid, int bk){
  // Distinct name required: every other kernel here declares
  // `extern __shared__ float sh[]`, and NVRTC rejects the same extern
  // __shared__ identifier with a different element type.
  extern __shared__ __half shp[];
  __half* Ks = shp;
  __half* Vs = shp + (size_t)bk * dim;

  const int warps = blockDim.x >> 5;
  const int lane  = threadIdx.x & 31;
  const int warp  = threadIdx.x >> 5;
  const int row   = blockIdx.x * warps + warp;
  const int h     = blockIdx.y;
  const int kvh   = h / (q_heads / kv_heads);
  const int dpl   = dim >> 5;                 // elements per lane, <= 16

  const __half* kh = kc + (size_t)kvh * max_seq * dim;
  const __half* vh = vc + (size_t)kvh * max_seq * dim;

  // Inactive warps still participate in the cooperative tile loads and in
  // every __syncthreads(), so the barriers below stay block-uniform.
  const int active = (row < nrows);
  const int qpos   = pos0 + row;
  const int myblk  = (active && blkid) ? blkid[qpos] : -1;

  float qreg[16], acc[16];
  float m = -1e30f, l = 0.f;
  int seq_q = 0;
  if(active){
    const __half* qh = q + ((size_t)row * q_heads + h) * dim;
    #pragma unroll
    for(int i = 0; i < 16; i++){
      if(i < dpl){ qreg[i] = __half2float(qh[lane + (i << 5)]); acc[i] = 0.f; }
    }
    seq_q = (!causal || myblk >= 0) ? (pos0 + nrows) : (qpos + 1);
  }

  // Block-uniform upper bound: the largest seq_q any resident warp can want.
  // With a media-block map present any query may look forward, so the whole
  // processed prefix is in play.
  int last = blockIdx.x * warps + warps; if(last > nrows) last = nrows;
  const int seq_blk = (!causal || blkid) ? (pos0 + nrows) : (pos0 + last);

  for(int s0 = 0; s0 < seq_blk; s0 += bk){
    int cnt = seq_blk - s0; if(cnt > bk) cnt = bk;
    __syncthreads();
    for(int idx = threadIdx.x; idx < cnt * dim; idx += blockDim.x){
      Ks[idx] = kh[(size_t)s0 * dim + idx];
      Vs[idx] = vh[(size_t)s0 * dim + idx];
    }
    __syncthreads();
    if(!active) continue;

    for(int j = 0; j < cnt; j++){
      const int s = s0 + j;
      if(s >= seq_q) break;                  // warp-uniform
      if(causal){
        const int same_block = (myblk >= 0) && blkid && (blkid[s] == myblk);
        if(s > qpos){
          if(!same_block) continue;          // forward: same block only
        } else {
          const int in_window = (window <= 0) || (qpos - s < window);
          if(!in_window && !same_block) continue;
        }
      }
      const __half* kj = Ks + (size_t)j * dim;
      float d = 0.f;
      #pragma unroll
      for(int i = 0; i < 16; i++)
        if(i < dpl) d += qreg[i] * __half2float(kj[lane + (i << 5)]);
      #pragma unroll
      for(int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
      d *= scale;

      const float m2 = fmaxf(m, d);
      const float corr = __expf(m - m2), p = __expf(d - m2);
      const __half* vj = Vs + (size_t)j * dim;
      #pragma unroll
      for(int i = 0; i < 16; i++)
        if(i < dpl) acc[i] = acc[i] * corr + p * __half2float(vj[lane + (i << 5)]);
      l = l * corr + p; m = m2;
    }
  }

  if(!active) return;
  __half* oh = out + ((size_t)row * q_heads + h) * dim;
  const float inv = (l > 0.f) ? (1.f / l) : 0.f;
  #pragma unroll
  for(int i = 0; i < 16; i++)
    if(i < dpl) oh[lane + (i << 5)] = __float2half(acc[i] * inv);
}

// ---- prefill attention: one block per (token, head) ----
// causal!=0 -> autoregressive mask (decoder prefill); causal==0 -> full
// bidirectional attention over [0, pos0+nrows) (vision / audio encoders).
// window: 0 = unbounded; otherwise sliding window of that many positions.
// blkid: optional per-absolute-position media-block ids (-1 = text). Two
// positions in the same block >= 0 attend bidirectionally (Gemma4 images).
extern "C" __global__ void k_attn_prefill(const __half* q, const __half* kc, const __half* vc,
        __half* out, int q_heads, int kv_heads, int dim, int pos0, int max_seq,
        int nrows, int causal, float scale, int window, const int* blkid){
  extern __shared__ float sh[];
  float* red = sh; float* acc = sh + blockDim.x;
  int row = blockIdx.x;            // token index within batch
  int h = blockIdx.y; int t = threadIdx.x;
  int kvh = h / (q_heads/kv_heads);
  const __half* qh = q + ((size_t)row*q_heads + h)*dim;
  const __half* kh = kc + (size_t)kvh*max_seq*dim;
  const __half* vh = vc + (size_t)kvh*max_seq*dim;
  int qpos = pos0 + row;
  int myblk = blkid ? blkid[qpos] : -1;
  // Media-block queries attend bidirectionally within their block (the
  // reference OR-combines the block mask with the causal/window mask), so
  // they must also scan *forward* keys up to the end of the processed
  // prefix. The chunker guarantees a block never straddles a chunk, so all
  // of the block's keys are in cache by the time its queries run.
  int seq = (!causal || myblk >= 0) ? (pos0 + nrows) : (qpos + 1);
  for(int i=t;i<dim;i+=blockDim.x) acc[i]=0.f;
  float m=-1e30f, l=0.f; __shared__ float bm;
  for(int s=0;s<seq;s++){
    if(causal){
      int same_block = (myblk>=0) && blkid && (blkid[s]==myblk);
      if(s > qpos){
        if(!same_block) continue;          // forward: same-block only
      } else {
        int in_window = (window<=0) || (qpos - s < window);
        if(!in_window && !same_block) continue;  // backward: causal-window OR block
      }
    }
    float d=0.f;
    for(int i=t;i<dim;i+=blockDim.x) d += __half2float(qh[i])*__half2float(kh[(size_t)s*dim+i]);
    red[t]=d; __syncthreads();
    for(int r=blockDim.x/2;r>0;r>>=1){ if(t<r) red[t]+=red[t+r]; __syncthreads(); }
    if(t==0) bm=red[0]*scale; __syncthreads();
    float m2=fmaxf(m,bm), corr=expf(m-m2), p=expf(bm-m2);
    for(int i=t;i<dim;i+=blockDim.x) acc[i]=acc[i]*corr+p*__half2float(vh[(size_t)s*dim+i]);
    l=l*corr+p; m=m2; __syncthreads();
  }
  __half* oh = out + ((size_t)row*q_heads + h)*dim;
  for(int i=t;i<dim;i+=blockDim.x) oh[i]=__float2half(acc[i]/l);
}

// ---- f16 -> f32 copy (logits readback) ----
extern "C" __global__ void k_h2f(const __half* x, float* y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i]=__half2float(x[i]);
}

// ---- f32 -> f16 copy (media tensor upload) ----
extern "C" __global__ void k_f2h(const float* x, __half* y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i]=__float2half(x[i]);
}

// ---- bf16 -> f16 in-place-style conversion for weight normalization ----
extern "C" __global__ void k_bf2h(const unsigned short* x, __half* y, size_t n){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i]=__float2half(bf2f(x[i]));
}

// ---- mean-pool rows: out[hidden] = mean over rows of x[rows,hidden] ----
extern "C" __global__ void k_meanpool(const __half* x, float* out, int rows, int hidden){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=hidden) return;
  float a=0.f;
  for(int r=0;r<rows;r++) a+=__half2float(x[(size_t)r*hidden+i]);
  out[i]=a/rows;
}

// ===========================================================================
// GGUF block formats (device side). The per-element decoders are the single
// source of truth for the math — bit-exact mirrors of quant::gguf's host
// decoders, verified on hardware by `cima selftest gguf`. Three consumers:
//   * k_gguf_<fmt>        element-parallel dequant to f16 (prefill scratch)
//   * k_gguf_<fmt>_gemv   fused decode GEMV: weights stay packed in VRAM,
//                         dequantized in registers while dotting — the
//                         resident path reads ~4.5 bits/weight instead of 16,
//                         which is the whole speedup (decode is bandwidth
//                         bound) and the whole VRAM win.
//   * k_gguf_gather       embedding-row gather straight from packed blocks
// ===========================================================================

__device__ inline float g_dec_q8_0(const unsigned char* blk, int i){
  float d = __half2float(*(const __half*)blk);
  return d * (float)((signed char)blk[2 + i]);
}

// ---- legacy 32-grain formats (llama.cpp's fallback for tensors whose row
// length is not a multiple of 256 — a small model's "Q4_K_M" file contains
// them). Element order per ggml: byte j of qs holds elem j (low nibble) and
// elem j+16 (high nibble); Q5_x add a 5th bit per elem in a 32-bit qh
// (bit j → elem j, bit j+16 → elem j+16).

__device__ inline float g_dec_q4_0(const unsigned char* blk, int r){
  float d = __half2float(*(const __half*)blk);
  unsigned char q = blk[2 + (r & 15)];
  int nib = (r < 16) ? (q & 0x0F) : (q >> 4);
  return d * (float)(nib - 8);
}

__device__ inline float g_dec_q4_1(const unsigned char* blk, int r){
  float d = __half2float(*(const __half*)blk);
  float m = __half2float(*(const __half*)(blk + 2));
  unsigned char q = blk[4 + (r & 15)];
  int nib = (r < 16) ? (q & 0x0F) : (q >> 4);
  return d * (float)nib + m;
}

__device__ inline float g_dec_q5_0(const unsigned char* blk, int r){
  float d = __half2float(*(const __half*)blk);
  unsigned int qh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8)
                  | ((unsigned int)blk[4] << 16) | ((unsigned int)blk[5] << 24);
  unsigned char q = blk[6 + (r & 15)];
  int nib = (r < 16) ? (q & 0x0F) : (q >> 4);
  int hi = (int)((qh >> r) & 1u) << 4;
  return d * (float)((nib | hi) - 16);
}

__device__ inline float g_dec_q5_1(const unsigned char* blk, int r){
  float d = __half2float(*(const __half*)blk);
  float m = __half2float(*(const __half*)(blk + 2));
  unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                  | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
  unsigned char q = blk[8 + (r & 15)];
  int nib = (r < 16) ? (q & 0x0F) : (q >> 4);
  int hi = (int)((qh >> r) & 1u) << 4;
  return d * (float)(nib | hi) + m;
}

__device__ inline void g_scale_min_k4(int j, const unsigned char* s, float* sc, float* m){
  if(j < 4){ *sc = (float)(s[j] & 63); *m = (float)(s[j+4] & 63); }
  else {
    *sc = (float)((s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4));
    *m  = (float)((s[j+4] >> 4)   | ((s[j]   >> 6) << 4));
  }
}

__device__ inline float g_dec_q4_k(const unsigned char* blk, int r){
  int c = r >> 6;
  int lo = (r & 63) < 32;
  int i = r & 31;
  float d    = __half2float(*(const __half*)(blk));
  float dmin = __half2float(*(const __half*)(blk+2));
  float sc, m;
  g_scale_min_k4(2*c + (lo ? 0 : 1), blk + 4, &sc, &m);
  unsigned char q = blk[16 + c*32 + i];
  float nib = lo ? (float)(q & 0x0F) : (float)(q >> 4);
  return d * sc * nib - dmin * m;
}

__device__ inline float g_dec_q5_k(const unsigned char* blk, int r){
  int c = r >> 6;
  int lo = (r & 63) < 32;
  int i = r & 31;
  float d    = __half2float(*(const __half*)(blk));
  float dmin = __half2float(*(const __half*)(blk+2));
  float sc, m;
  g_scale_min_k4(2*c + (lo ? 0 : 1), blk + 4, &sc, &m);
  unsigned char q = blk[48 + c*32 + i];
  unsigned char u = (unsigned char)((lo ? 1 : 2) << (2*c));
  float hi = (blk[16 + i] & u) ? 16.0f : 0.0f;
  float nib = lo ? (float)(q & 0x0F) : (float)(q >> 4);
  return d * sc * (nib + hi) - dmin * m;
}

__device__ inline float g_dec_q6_k(const unsigned char* blk, int r){
  int half = r >> 7;
  int rr = r & 127;
  int g = rr >> 5;
  int l = rr & 31;
  const unsigned char* qlh = blk + half*64;
  const unsigned char* qhh = blk + 128 + half*32;
  const signed char*   sch = (const signed char*)(blk + 192) + half*8;
  int is = l >> 4;
  unsigned char ql = qlh[l + ((g == 1 || g == 3) ? 32 : 0)];
  int nib = (g >= 2) ? (ql >> 4) : (ql & 0x0F);
  int hi2 = (qhh[l] >> (2*g)) & 0x03;
  int q = (nib | (hi2 << 4)) - 32;
  float d = __half2float(*(const __half*)(blk+208));
  return d * (float)sch[is + 2*g] * (float)q;
}

__constant__ float c_iq4nl[16] = {
  -127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f
};

__device__ inline float g_dec_iq4_xs(const unsigned char* blk, int r){
  int ib = r >> 5;
  int rr = r & 31;
  int j = rr & 15;
  unsigned int scales_h = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
  int ls = ((blk[4 + ib/2] >> (4*(ib%2))) & 0x0F) | (((scales_h >> (2*ib)) & 3) << 4);
  float d = __half2float(*(const __half*)blk);
  float dl = d * (float)(ls - 32);
  unsigned char q = blk[8 + ib*16 + j];
  return dl * ((rr < 16) ? c_iq4nl[q & 0x0F] : c_iq4nl[q >> 4]);
}

// ---- element-parallel dequant (one thread per output value) ----

extern "C" __global__ void k_gguf_q8_0(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 32) return;
  out[e] = __float2half(g_dec_q8_0(src + (e >> 5) * 34, (int)(e & 31)));
}

extern "C" __global__ void k_gguf_q4_0(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 32) return;
  out[e] = __float2half(g_dec_q4_0(src + (e >> 5) * 18, (int)(e & 31)));
}

extern "C" __global__ void k_gguf_q4_1(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 32) return;
  out[e] = __float2half(g_dec_q4_1(src + (e >> 5) * 20, (int)(e & 31)));
}

extern "C" __global__ void k_gguf_q5_0(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 32) return;
  out[e] = __float2half(g_dec_q5_0(src + (e >> 5) * 22, (int)(e & 31)));
}

extern "C" __global__ void k_gguf_q5_1(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 32) return;
  out[e] = __float2half(g_dec_q5_1(src + (e >> 5) * 24, (int)(e & 31)));
}

extern "C" __global__ void k_gguf_q4_k(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 256) return;
  out[e] = __float2half(g_dec_q4_k(src + (e >> 8) * 144, (int)(e & 255)));
}

extern "C" __global__ void k_gguf_q5_k(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 256) return;
  out[e] = __float2half(g_dec_q5_k(src + (e >> 8) * 176, (int)(e & 255)));
}

extern "C" __global__ void k_gguf_q6_k(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 256) return;
  out[e] = __float2half(g_dec_q6_k(src + (e >> 8) * 210, (int)(e & 255)));
}

extern "C" __global__ void k_gguf_iq4_xs(const unsigned char* src, __half* out, int nblocks){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)nblocks * 256) return;
  out[e] = __float2half(g_dec_iq4_xs(src + (e >> 8) * 136, (int)(e & 255)));
}

// ---- fused resident GEMVs, v5: the llama.cpp design — integer dp4a.
// x quantizes ONCE per call to q8_1 blocks of 32 (k_quantize_q8_1: f32
// scale d8, f32 s = d8·Σq, s8 qs[32]); every weight format here is
// 32-grain, so each q8 block aligns exactly with one weight sub-block.
// The dot then runs __dp4a (4 signed-byte MACs / instruction) over u32
// words with the per-sub-block float fixups ggml uses:
//   K-quants with mins:  acc += d·sc_j·d8·sumi_j − dmin·m_j·s_j
//   Q8_0 / Q6_K / IQ4_XS: pure scaled integer sums.
// One warp per row, 8 rows per block, u32 loads, shuffle reduction —
// the proven v4 skeleton with ~5× fewer inner-loop instructions.

typedef struct { float d; float s; signed char qs[32]; } q8_1_blk; // 40 B

// NVRTC declares __dp4a in its builtin header but emits an unresolved
// extern call instead of the instruction (ptxas: '_Z6__dp4aiii').
// Sidestep the header entirely: the PTX instruction itself, sm_61+.
__device__ inline int g_dp4a(int a, int b, int c){
  int r;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

extern "C" __global__ void k_quantize_q8_1(const __half* x, q8_1_blk* out, int k){
  int b = blockIdx.x;               // one warp-sized block per 32 elems
  int i = threadIdx.x;              // 0..31
  if(b * 32 + i >= k) return;
  float v = __half2float(x[b * 32 + i]);
  float a = fabsf(v);
  // warp max
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  float d = a / 127.0f;
  int q = (d > 0.f) ? (int)nearbyintf(v / d) : 0;
  q = max(-127, min(127, q));
  out[b].qs[i] = (signed char)q;
  // warp sum of q for the mins fixup
  int s = q;
  for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if(i == 0){ out[b].d = d; out[b].s = d * (float)s; }
}

#define GEMV_HEAD \
  int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); \
  int lane = threadIdx.x & 31; \
  if(row >= n) return; \
  float acc = 0.f;

#define GEMV_TAIL \
  for(int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o); \
  if(lane == 0){ \
    if(bias) acc += __half2float(bias[row]); \
    if(mode == 1) acc += __half2float(y[row]); \
    y[row] = __float2half(acc); \
  }

extern "C" __global__ void k_gguf_q8_0_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 5) * 34);
  int nb = k >> 5;
  for(int b = lane; b < nb; b += 32){
    const unsigned char* blk = wr + b * 34;             // 2-aligned only
    float d = __half2float(*(const __half*)blk);
    const q8_1_blk* xb = xq + b;
    // qs at +2 is 2-aligned: load as u16 pairs into u32 words
    const unsigned short* q16 = (const unsigned short*)(blk + 2);
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 8; i++){
      unsigned int wv = (unsigned int)q16[2*i] | ((unsigned int)q16[2*i+1] << 16);
      sumi = g_dp4a((int)wv, (int)x32[i], sumi);
    }
    acc += d * xb->d * (float)sumi;
  }
  GEMV_TAIL
}

// 4-bit hi-mask spread: distribute the 4 low bits of `nyb` into the four
// byte lanes of a u32, each landing at bit 4 (the "+16" position of a
// 5-bit quant). bit0→bit4, bit1→bit12, bit2→bit20, bit3→bit28.
__device__ inline unsigned int g_spread5(unsigned int nyb){
  return ((nyb & 1u) << 4) | ((nyb & 2u) << 11)
       | ((nyb & 4u) << 18) | ((nyb & 8u) << 25);
}

// Legacy 32-grain GEMVs. One j iteration = ONE full 32-elem weight block
// (both nibble halves), aligned with one q8_1 activation block: x words
// 0..3 dot the low nibbles (elems 0..15), words 4..7 the high nibbles
// (elems 16..31). The per-block float fixups against the q8_1 sums:
//   Q4_0: Σ(q−8)·x  = d·d8·sumi −  8·d·s      (s = d8·Σx_q)
//   Q4_1: Σ q·x·d+m = d·d8·sumi +  m·s
//   Q5_0: Σ(v−16)·x = d·d8·sumi − 16·d·s      (v = nib | bit·16)
//   Q5_1:            d·d8·sumi +  m·s

extern "C" __global__ void k_gguf_q4_0_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 5) * 18);
  int nb = k >> 5;
  for(int b = lane; b < nb; b += 32){
    const unsigned char* blk = wr + b * 18;              // 2-aligned only
    float d = __half2float(*(const __half*)blk);
    const unsigned short* q16 = (const unsigned short*)(blk + 2);
    const q8_1_blk* xb = xq + b;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++){
      unsigned int wv = (unsigned int)q16[2*i] | ((unsigned int)q16[2*i+1] << 16);
      sumi = g_dp4a((int)(wv & 0x0F0F0F0Fu),        (int)x32[i],     sumi);
      sumi = g_dp4a((int)((wv >> 4) & 0x0F0F0F0Fu), (int)x32[i + 4], sumi);
    }
    acc += d * xb->d * (float)sumi - 8.0f * d * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q4_1_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 5) * 20);
  int nb = k >> 5;
  for(int b = lane; b < nb; b += 32){
    const unsigned char* blk = wr + b * 20;              // 4-aligned
    float d = __half2float(*(const __half*)blk);
    float m = __half2float(*(const __half*)(blk + 2));
    const unsigned int* qs = (const unsigned int*)(blk + 4);
    const q8_1_blk* xb = xq + b;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++){
      unsigned int wv = qs[i];
      sumi = g_dp4a((int)(wv & 0x0F0F0F0Fu),        (int)x32[i],     sumi);
      sumi = g_dp4a((int)((wv >> 4) & 0x0F0F0F0Fu), (int)x32[i + 4], sumi);
    }
    acc += d * xb->d * (float)sumi + m * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q5_0_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 5) * 22);
  int nb = k >> 5;
  for(int b = lane; b < nb; b += 32){
    const unsigned char* blk = wr + b * 22;              // 2-aligned only
    float d = __half2float(*(const __half*)blk);
    unsigned int qh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8)
                    | ((unsigned int)blk[4] << 16) | ((unsigned int)blk[5] << 24);
    const unsigned short* q16 = (const unsigned short*)(blk + 6);
    const q8_1_blk* xb = xq + b;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++){
      unsigned int wv = (unsigned int)q16[2*i] | ((unsigned int)q16[2*i+1] << 16);
      unsigned int lo = (wv & 0x0F0F0F0Fu)        + g_spread5((qh >> (4*i))      & 0xFu);
      unsigned int hi = ((wv >> 4) & 0x0F0F0F0Fu) + g_spread5((qh >> (16 + 4*i)) & 0xFu);
      sumi = g_dp4a((int)lo, (int)x32[i],     sumi);
      sumi = g_dp4a((int)hi, (int)x32[i + 4], sumi);
    }
    acc += d * xb->d * (float)sumi - 16.0f * d * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q5_1_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 5) * 24);
  int nb = k >> 5;
  for(int b = lane; b < nb; b += 32){
    const unsigned char* blk = wr + b * 24;              // 4-aligned
    float d = __half2float(*(const __half*)blk);
    float m = __half2float(*(const __half*)(blk + 2));
    unsigned int qh = *(const unsigned int*)(blk + 4);
    const unsigned int* qs = (const unsigned int*)(blk + 8);
    const q8_1_blk* xb = xq + b;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++){
      unsigned int wv = qs[i];
      unsigned int lo = (wv & 0x0F0F0F0Fu)        + g_spread5((qh >> (4*i))      & 0xFu);
      unsigned int hi = ((wv >> 4) & 0x0F0F0F0Fu) + g_spread5((qh >> (16 + 4*i)) & 0xFu);
      sumi = g_dp4a((int)lo, (int)x32[i],     sumi);
      sumi = g_dp4a((int)hi, (int)x32[i + 4], sumi);
    }
    acc += d * xb->d * (float)sumi + m * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q4_k_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 8) * 144);
  int nj = k >> 5;                                       // 32-elem sub-blocks
  for(int j = lane; j < nj; j += 32){
    const unsigned char* blk = wr + (j >> 3) * 144;
    int sj = j & 7;                                      // sub-block in superblock
    int c = sj >> 1;                                     // qs stripe
    int hi = sj & 1;                                     // nibble half
    float d    = __half2float(*(const __half*)(blk));
    float dmin = __half2float(*(const __half*)(blk + 2));
    float sc, m;
    g_scale_min_k4(sj, blk + 4, &sc, &m);
    const unsigned int* qs = (const unsigned int*)(blk + 16 + c * 32);
    const q8_1_blk* xb = xq + j;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 8; i++){
      unsigned int nib = hi ? ((qs[i] >> 4) & 0x0F0F0F0Fu) : (qs[i] & 0x0F0F0F0Fu);
      sumi = g_dp4a((int)nib, (int)x32[i], sumi);
    }
    acc += d * sc * xb->d * (float)sumi - dmin * m * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q5_k_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 8) * 176);
  int nj = k >> 5;
  for(int j = lane; j < nj; j += 32){
    const unsigned char* blk = wr + (j >> 3) * 176;
    int sj = j & 7;
    int c = sj >> 1;
    int hi = sj & 1;
    float d    = __half2float(*(const __half*)(blk));
    float dmin = __half2float(*(const __half*)(blk + 2));
    float sc, m;
    g_scale_min_k4(sj, blk + 4, &sc, &m);
    int sh = 2*c + hi;                                   // 5th-bit position
    const unsigned int* qh = (const unsigned int*)(blk + 16);
    const unsigned int* qs = (const unsigned int*)(blk + 48 + c * 32);
    const q8_1_blk* xb = xq + j;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int i = 0; i < 8; i++){
      unsigned int nib = hi ? ((qs[i] >> 4) & 0x0F0F0F0Fu) : (qs[i] & 0x0F0F0F0Fu);
      // 5th bit per byte lane → +16: shift the bit to position 0, mask,
      // then lift to bit 4 (pure shifts — no per-lane division).
      unsigned int one = (qh[i] >> sh) & 0x01010101u;
      nib += one << 4;
      sumi = g_dp4a((int)nib, (int)x32[i], sumi);
    }
    acc += d * sc * xb->d * (float)sumi - dmin * m * xb->s;
  }
  GEMV_TAIL
}

extern "C" __global__ void k_gguf_q6_k_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 8) * 210);
  int nj = k >> 5;
  for(int j = lane; j < nj; j += 32){
    const unsigned char* blk = wr + (j >> 3) * 210;
    int sj = j & 7;
    int half = sj >> 2;
    int g = sj & 3;
    float d = __half2float(*(const __half*)(blk + 208));
    const unsigned char* qlh = blk + half * 64 + ((g == 1 || g == 3) ? 32 : 0);
    const unsigned char* qhh = blk + 128 + half * 32;
    const signed char*   sch = (const signed char*)(blk + 192) + half * 8 + 2 * g;
    // 210-byte blocks are only 2-aligned: assemble u32 from u16 pairs
    const unsigned short* ql16 = (const unsigned short*)qlh;
    const unsigned short* qh16 = (const unsigned short*)qhh;
    const q8_1_blk* xb = xq + j;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int shift_l = (g >= 2) ? 4 : 0;
    int shift_h = 2 * g;
    int sumi0 = 0, sumi1 = 0;                            // per-16 scales
    #pragma unroll
    for(int i = 0; i < 8; i++){
      unsigned int lw = (unsigned int)ql16[2*i] | ((unsigned int)ql16[2*i+1] << 16);
      unsigned int hw = (unsigned int)qh16[2*i] | ((unsigned int)qh16[2*i+1] << 16);
      unsigned int nib = (lw >> shift_l) & 0x0F0F0F0Fu;
      unsigned int hi2 = (hw >> shift_h) & 0x03030303u;
      unsigned int q = nib | (hi2 << 4);                 // 0..63 per byte
      // subtract 32 per byte via dp4a against ones: sumi(q−32·1)·x =
      // dp4a(q,x) − 32·Σx_bytes; cheaper: bias once per 4-word group.
      int s = g_dp4a((int)q, (int)x32[i], 0);
      int ones = g_dp4a(0x01010101, (int)x32[i], 0);         // Σ of 4 x bytes
      s -= 32 * ones;
      if(i < 4) sumi0 += s; else sumi1 += s;
    }
    acc += d * xb->d * ((float)sch[0] * (float)sumi0 + (float)sch[1] * (float)sumi1);
  }
  GEMV_TAIL
}

// 4-bit indices → s8 codebook values via byte-permute: nibbles ARE the
// hardware's native byte selectors. Two prmt (table halves) + a compare
// blend = 4 values in ~7 ops, vs 8 shared-memory lookups + shift/or packs.
__device__ inline unsigned int g_lut16_sel(unsigned int q, unsigned int t0, unsigned int t1,
        unsigned int t2, unsigned int t3){
  // __byte_perm reads its four selectors as NIBBLES of the low 16 bits —
  // our indices arrive one per BYTE, so compact bytes→nibbles first
  // (the selftest caught the uncompacted version selecting garbage for
  // output bytes 2-3).
  unsigned int sel = (q & 0x00000007u)
                   | ((q >> 4) & 0x00000070u)
                   | ((q >> 8) & 0x00000700u)
                   | ((q >> 12) & 0x00007000u);
  unsigned int lo = __byte_perm(t0, t1, sel);               // table[idx]   for idx < 8
  unsigned int hi = __byte_perm(t2, t3, sel);               // table[idx-8] for idx ≥ 8
  // Byte mask from each lane's bit 3, intrinsic-free: bytes of
  // ((q>>3)&0x01010101) are 0/1; ×0xFF turns each 1 into 0xFF with no
  // cross-byte carry (x·0xFF = (x<<8)−x borrows only through zero bytes).
  unsigned int m = ((q >> 3) & 0x01010101u) * 0xFFu;
  return lo ^ ((lo ^ hi) & m);
}

extern "C" __global__ void k_gguf_iq4_xs_gemv(const q8_1_blk* xq, const unsigned char* w, const __half* bias,
        __half* y, int n, int k, int mode){
  GEMV_HEAD
  // IQ4NL codebook packed into four u32 registers (s8 lanes):
  // bytes: [-127,-104,-83,-65] [-49,-35,-22,-10] [1,13,25,38] [53,69,89,113]
  const unsigned int c0 = (unsigned int)(unsigned char)(-127)
                        | ((unsigned int)(unsigned char)(-104) << 8)
                        | ((unsigned int)(unsigned char)(-83) << 16)
                        | ((unsigned int)(unsigned char)(-65) << 24);
  const unsigned int c1 = (unsigned int)(unsigned char)(-49)
                        | ((unsigned int)(unsigned char)(-35) << 8)
                        | ((unsigned int)(unsigned char)(-22) << 16)
                        | ((unsigned int)(unsigned char)(-10) << 24);
  const unsigned int c2 = 1u | (13u << 8) | (25u << 16) | (38u << 24);
  const unsigned int c3 = 53u | (69u << 8) | (89u << 16) | (113u << 24);
  const unsigned char* wr = w + (long long)row * ((long long)(k >> 8) * 136);
  int nj = k >> 5;
  for(int j = lane; j < nj; j += 32){
    const unsigned char* blk = wr + (j >> 3) * 136;
    int ib = j & 7;
    float d = __half2float(*(const __half*)blk);
    unsigned int scales_h = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
    int ls = ((blk[4 + ib/2] >> (4*(ib%2))) & 0x0F) | (((scales_h >> (2*ib)) & 3) << 4);
    float dl = d * (float)(ls - 32);
    const unsigned int* qs = (const unsigned int*)(blk + 8 + ib * 16);
    const q8_1_blk* xb = xq + j;
    const unsigned int* x32 = (const unsigned int*)xb->qs;
    int sumi = 0;
    #pragma unroll
    for(int wd = 0; wd < 4; wd++){
      unsigned int p = qs[wd];
      unsigned int vl = g_lut16_sel(p & 0x0F0F0F0Fu, c0, c1, c2, c3);
      unsigned int vh = g_lut16_sel((p >> 4) & 0x0F0F0F0Fu, c0, c1, c2, c3);
      sumi = g_dp4a((int)vl, (int)x32[wd], sumi);
      sumi = g_dp4a((int)vh, (int)x32[wd + 4], sumi);
    }
    acc += dl * xb->d * (float)sumi;
  }
  GEMV_TAIL
}

// ---- embedding gather from packed rows (graph-capturable) ----
// fmt is the ggml type id (2, 3, 6, 7, 8, 12, 13, 14, 23); row_bytes precomputed host-side.

extern "C" __global__ void k_gguf_gather(const unsigned char* table, const unsigned int* ids,
        __half* out, int n, int hidden, int fmt, int row_bytes){
  long long e = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= (long long)n * hidden) return;
  int r = (int)(e / hidden);
  int c = (int)(e % hidden);
  const unsigned char* rowp = table + (long long)ids[r] * row_bytes;
  float v;
  switch(fmt){
    case 8:  v = g_dec_q8_0(rowp + (c >> 5) * 34,    c & 31);  break;
    case 2:  v = g_dec_q4_0(rowp + (c >> 5) * 18,    c & 31);  break;
    case 3:  v = g_dec_q4_1(rowp + (c >> 5) * 20,    c & 31);  break;
    case 6:  v = g_dec_q5_0(rowp + (c >> 5) * 22,    c & 31);  break;
    case 7:  v = g_dec_q5_1(rowp + (c >> 5) * 24,    c & 31);  break;
    case 12: v = g_dec_q4_k(rowp + (c >> 8) * 144,   c & 255); break;
    case 13: v = g_dec_q5_k(rowp + (c >> 8) * 176,   c & 255); break;
    case 14: v = g_dec_q6_k(rowp + (c >> 8) * 210,   c & 255); break;
    default: v = g_dec_iq4_xs(rowp + (c >> 8) * 136, c & 255); break;
  }
  out[e] = __float2half(v);
}

// ---- native f16 GEMM: C[m,n] = A[m,k] · B[n,k]^T (row-major, f32 acc) ----
// The cuBLAS-independence kernel: 16x16 shared tiles, one output element
// per thread, 256-thread 1D blocks (grid.x tiles n, grid.y tiles m). C has
// leading dimension ldc >= n so the same kernel serves gemm_f16 (ldc == n)
// and the column-range variant gemm_strided_out (ldc == full row width).
// Prefill-only duty: correctness and cuBLAS-free deploys over peak TFLOPS.

extern "C" __global__ void k_gemm_f16(const __half* A, const __half* B, __half* C,
        int m, int n, int k, int ldc){
  __shared__ float As[16][17];
  __shared__ float Bs[16][17];
  int tx = threadIdx.x & 15;         // column within tile
  int ty = threadIdx.x >> 4;         // row within tile
  int col = blockIdx.x * 16 + tx;    // n index
  int row = blockIdx.y * 16 + ty;    // m index
  float acc = 0.0f;
  for(int t = 0; t < k; t += 16){
    // As tile: A[row, t+tx]; Bs tile: B[col_tile_base+ty, t+tx] — both
    // loaded k-contiguous (coalesced), consumed transposed from shared.
    As[ty][tx] = (row < m && t + tx < k) ? __half2float(A[(long long)row * k + t + tx]) : 0.0f;
    int brow = blockIdx.x * 16 + ty;
    Bs[ty][tx] = (brow < n && t + tx < k) ? __half2float(B[(long long)brow * k + t + tx]) : 0.0f;
    __syncthreads();
    #pragma unroll
    for(int e = 0; e < 16; e++) acc += As[ty][e] * Bs[tx][e];
    __syncthreads();
  }
  if(row < m && col < n) C[(long long)row * ldc + col] = __float2half(acc);
}
