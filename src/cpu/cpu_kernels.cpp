#include "cpu/cpu_kernels.h"
#include "cpu/cpu_iq_tables.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rookery::cpu {

namespace {

struct BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

struct BlockQ5_0 {
    uint16_t d;
    uint8_t  qh[4];
    uint8_t  qs[16];
};
static_assert(sizeof(BlockQ5_0) == 22, "BlockQ5_0 must be 22 bytes");

struct BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

struct BlockIQ4_NL {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(BlockIQ4_NL) == 18, "BlockIQ4_NL must be 18 bytes");

struct BlockIQ2_S {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  qh[8];
    uint8_t  scales[8];
};
static_assert(sizeof(BlockIQ2_S) == 82, "BlockIQ2_S must be 82 bytes");

struct BlockIQ3_S {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  qh[8];
    uint8_t  signs[32];
    uint8_t  scales[4];
};
static_assert(sizeof(BlockIQ3_S) == 110, "BlockIQ3_S must be 110 bytes");

struct BlockQ2_K {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(BlockQ2_K) == 84, "BlockQ2_K must be 84 bytes");

struct BlockQ3_K {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
};
static_assert(sizeof(BlockQ3_K) == 110, "BlockQ3_K must be 110 bytes");

struct BlockIQ4_XS {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS) == 136, "BlockIQ4_XS must be 136 bytes");

struct BlockIQ3_XXS {
    uint16_t d;
    uint8_t  qs[96];
};
static_assert(sizeof(BlockIQ3_XXS) == 98, "BlockIQ3_XXS must be 98 bytes");

inline bool needs_act_q8_0(DType dt) {
    return dt == DType::Q8_0 || dt == DType::Q4_0 || dt == DType::Q5_0 || dt == DType::IQ4_NL;
}

inline bool needs_act_q8_k(DType dt) {
    return dt == DType::Q6_K || dt == DType::Q5_K || dt == DType::Q4_K ||
           dt == DType::Q3_K || dt == DType::Q2_K ||
           dt == DType::IQ4_XS || dt == DType::IQ3_S || dt == DType::IQ3_XXS || dt == DType::IQ2_S;
}

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoefA   = 0.044715f;

inline float gelu_tanh_scalar(float x) {
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * x * (1.0f + kGeluCoefA * x * x)));
}

inline size_t g4_kv_q8_head_stride(int head_dim) {
    const size_t d_bytes = static_cast<size_t>(head_dim >> 5) * sizeof(uint16_t);
    return static_cast<size_t>(head_dim) + ((d_bytes + 31u) & ~size_t(31));
}

inline size_t g4_kv_q4_head_stride(int head_dim) {
    const size_t d_bytes = static_cast<size_t>(head_dim >> 5) * sizeof(uint16_t);
    return static_cast<size_t>(head_dim >> 1) + ((d_bytes + 31u) & ~size_t(31));
}

inline float f16_to_f32_scalar(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t bits = 0;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int exp = -14;
            uint32_t mant = m;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03FFu;
            bits = s | (static_cast<uint32_t>(exp + 127) << 23) | (mant << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint16_t f32_to_f16_scalar(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t t = static_cast<uint32_t>(14 - exp);
        const uint32_t half_m = (mant + (1u << (t - 1))) >> t;
        return static_cast<uint16_t>(sign | half_m);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    const uint32_t rounded = mant + 0x00000FFFu + ((mant >> 13) & 1u);
    if (rounded & 0x00800000u) {
        ++exp;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

inline float f16_to_f32_fast(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f16;
    std::memcpy(&f16, &h, sizeof(f16));
    return static_cast<float>(f16);
#else
    return f16_to_f32_scalar(h);
#endif
}

inline uint16_t f32_to_f16_fast(float f) {
#if defined(__F16C__)
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f16 = static_cast<__fp16>(f);
    uint16_t bits;
    std::memcpy(&bits, &f16, sizeof(bits));
    return bits;
#else
    return f32_to_f16_scalar(f);
#endif
}

inline float bf16_to_f32_fast(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

#if defined(__AVX2__) && defined(__FMA__)
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// Fast 8-wide AVX2+FMA expf (< 1e-6 relative error, Cephes minimax degree-5 + Cody-Waite reduction).
static inline __m256 exp_ps_avx2(__m256 x) {
    const __m256 hi   = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo   = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);

    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);

    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

    const __m256i emm = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm));
}

static inline __m256 silu_ps_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    return _mm256_div_ps(x, _mm256_add_ps(one, exp_ps_avx2(neg)));
}

static inline __m256 sigmoid_ps_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_ps_avx2(neg)));
}

static inline __m256 tanh_ps_avx2(__m256 x) {
    const __m256 hi  = _mm256_set1_ps(9.0f);
    const __m256 lo  = _mm256_set1_ps(-9.0f);
    const __m256 xc  = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    const __m256 e2x = exp_ps_avx2(_mm256_add_ps(xc, xc));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    return _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e2x, one)));
}

static inline __m256 gelu_tanh_ps_avx2(__m256 x) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 c0   = _mm256_set1_ps(kSqrt2OverPi);
    const __m256 ca   = _mm256_set1_ps(kGeluCoefA);
    const __m256 x2   = _mm256_mul_ps(x, x);
    const __m256 arg  = _mm256_mul_ps(_mm256_mul_ps(c0, x), _mm256_fmadd_ps(ca, x2, one));
    return _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, tanh_ps_avx2(arg)));
}

// Expand 32 bits of mask into 32 signed bytes: -1 (0xFF) if bit==1, +1 (0x01) if bit==0.
static inline __m256i unpack_sign32_to_i8_avx2(uint32_t mask) {
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    const __m256i v    = _mm256_set1_epi32(static_cast<int32_t>(mask));
    const __m256i bits = _mm256_and_si256(_mm256_shuffle_epi8(v, shuf), bitm);
    const __m256i neg  = _mm256_cmpeq_epi8(bits, bitm);
    return _mm256_or_si256(neg, _mm256_set1_epi8(1));
}

// Expand 32 bits of qh into 32 bytes: 0x10 if bit==1, 0x00 if bit==0 (for Q5_0).
static inline __m256i unpack_bits32_to_0x10_avx2(uint32_t mask) {
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    const __m256i v    = _mm256_set1_epi32(static_cast<int32_t>(mask));
    const __m256i bits = _mm256_and_si256(_mm256_shuffle_epi8(v, shuf), bitm);
    return _mm256_and_si256(_mm256_cmpeq_epi8(bits, bitm), _mm256_set1_epi8(0x10));
}

static inline void store_i8x32_scaled_f32_avx2(__m256i q8, float scale, float * out) {
    const __m128i lo = _mm256_castsi256_si128(q8);
    const __m128i hi = _mm256_extracti128_si256(q8, 1);
    const __m256  vs = _mm256_set1_ps(scale);
    _mm256_storeu_ps(out +  0, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
    _mm256_storeu_ps(out +  8, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
    _mm256_storeu_ps(out + 16, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
    _mm256_storeu_ps(out + 24, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
}

static inline void store_u8x32_affine_f32_avx2(__m256i q8, float scale, float neg_min, float * out) {
    const __m128i lo = _mm256_castsi256_si128(q8);
    const __m128i hi = _mm256_extracti128_si256(q8, 1);
    const __m256  vs = _mm256_set1_ps(scale);
    const __m256  vm = _mm256_set1_ps(neg_min);
    _mm256_storeu_ps(out +  0, _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), vm));
    _mm256_storeu_ps(out +  8, _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), vm));
    _mm256_storeu_ps(out + 16, _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), vm));
    _mm256_storeu_ps(out + 24, _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), vm));
}

static inline __m256i mul_sum_i8_pairs_to_i32(__m256i w, __m256i x, __m256i ones16) {
    const __m256i aw  = _mm256_abs_epi8(w);
    const __m256i sx  = _mm256_sign_epi8(x, w);
    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    return _mm256_madd_epi16(p16, ones16);
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
// Fast 4-wide NEON exp(x) accurate to < 3e-7 relative error across [-87, 88].
static inline float32x4_t vexpq_f32_fast(float32x4_t x) noexcept {
    const float32x4_t x_hi   = vdupq_n_f32(88.3762626647949f);
    const float32x4_t x_lo   = vdupq_n_f32(-87.3365447504f);
    const float32x4_t log2e  = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
    const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);

    x = vminq_f32(vmaxq_f32(x, x_lo), x_hi);
    const float32x4_t fn = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t r = vfmsq_f32(x, fn, ln2_hi);
    r             = vfmsq_f32(r, fn, ln2_lo);

    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(r, p, vmulq_f32(r, r));
    p = vaddq_f32(p, vdupq_n_f32(1.0f));

    const int32x4_t pow2n = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fn), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(pow2n));
}

// Fast 4-wide NEON tanh(x) accurate to < 1e-6 across all finite float inputs.
static inline float32x4_t vtanhq_f32_fast(float32x4_t x) noexcept {
    const float32x4_t xc  = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-9.0f)), vdupq_n_f32(9.0f));
    const float32x4_t e2x = vexpq_f32_fast(vaddq_f32(xc, xc));
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(vsubq_f32(e2x, one), vaddq_f32(e2x, one));
}

static inline float32x4_t vgelu_tanh_f32_fast(float32x4_t x) noexcept {
    const float32x4_t kSqrt2OverPi = vdupq_n_f32(0.7978845608028654f);
    const float32x4_t kCoeff       = vdupq_n_f32(0.044715f);
    const float32x4_t x2 = vmulq_f32(x, x);
    const float32x4_t inner = vmulq_f32(kSqrt2OverPi, vmulq_f32(x, vfmaq_f32(vdupq_n_f32(1.0f), kCoeff, x2)));
    return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), x), vaddq_f32(vdupq_n_f32(1.0f), vtanhq_f32_fast(inner)));
}

static inline float32x4_t load_4_fp16_scales(uint16_t s0, uint16_t s1, uint16_t s2, uint16_t s3) noexcept {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint16x4_t u = vdup_n_u16(s0);
    u = vset_lane_u16(s1, u, 1);
    u = vset_lane_u16(s2, u, 2);
    u = vset_lane_u16(s3, u, 3);
    return vcvt_f32_f16(vreinterpret_f16_u16(u));
#else
    const float a[4] = {f16_to_f32_fast(s0), f16_to_f32_fast(s1), f16_to_f32_fast(s2), f16_to_f32_fast(s3)};
    return vld1q_f32(a);
#endif
}
#endif

// ============================================================================
// Fast AVX2 Dequantization into L1-resident FP32 Panels (for Haswell FMA GEMM)
// ============================================================================

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j] >> 6) << 4);
    }
}

inline void dequant_slice_to_f32(DType dt, const void * row_ptr, int k0, int kc, float * dst) {
    switch (dt) {
        case DType::F32:
            std::memcpy(dst, static_cast<const float *>(row_ptr) + k0, static_cast<size_t>(kc) * sizeof(float));
            return;
        case DType::BF16: {
            const auto * src = static_cast<const uint16_t *>(row_ptr) + k0;
            int i = 0;
#if defined(__AVX2__)
            for (; i + 7 < kc; i += 8) {
                const __m256 v = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))), 16));
                _mm256_storeu_ps(dst + i, v);
            }
#endif
            for (; i < kc; ++i) dst[i] = bf16_to_f32_fast(src[i]);
            return;
        }
        case DType::F16: {
            const auto * src = static_cast<const uint16_t *>(row_ptr) + k0;
            int i = 0;
#if defined(__AVX2__) && defined(__F16C__)
            for (; i + 7 < kc; i += 8) {
                _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
            }
#endif
            for (; i < kc; ++i) dst[i] = f16_to_f32_fast(src[i]);
            return;
        }
        case DType::Q8_0: {
            const int b0 = k0 >> 5;
            const int nb = kc >> 5;
            const auto * blk = static_cast<const BlockQ8_0 *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            int b = 0;
#if defined(__F16C__)
            for (; b + 7 < nb; b += 8) {
                alignas(32) uint16_t dh[8] = {
                    blk[b + 0].d, blk[b + 1].d, blk[b + 2].d, blk[b + 3].d,
                    blk[b + 4].d, blk[b + 5].d, blk[b + 6].d, blk[b + 7].d
                };
                alignas(32) float df[8];
                _mm256_store_ps(df, _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(dh))));
                for (int bi = 0; bi < 8; ++bi) {
                    const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk[b + bi].qs));
                    store_i8x32_scaled_f32_avx2(q8, df[bi], dst + (b + bi) * 32);
                }
            }
#endif
            for (; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk[b].qs));
                store_i8x32_scaled_f32_avx2(q8, d, dst + b * 32);
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            for (int b = 0; b < nb; ++b) {
                const float32x4_t vd = vdupq_n_f32(f16_to_f32_fast(blk[b].d));
                const int8_t * qs = blk[b].qs;
                float * out = dst + b * 32;
                for (int g = 0; g < 2; ++g) {
                    const int8x16_t qv  = vld1q_s8(qs + g * 16);
                    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                    const int16x8_t qhi = vmovl_s8(vget_high_s8(qv));
                    float * gp = out + g * 16;
                    vst1q_f32(gp +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))),  vd));
                    vst1q_f32(gp +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), vd));
                    vst1q_f32(gp +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))),  vd));
                    vst1q_f32(gp + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), vd));
                }
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                float * out = dst + b * 32;
                for (int i = 0; i < 32; ++i) out[i] = d * static_cast<float>(blk[b].qs[i]);
            }
#endif
            return;
        }
        case DType::Q6_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ6_K *>(row_ptr) + b0;
#if defined(__AVX2__)
            const __m256i m4   = _mm256_set1_epi8(0x0F);
            const __m256i m30  = _mm256_set1_epi8(0x30);
            const __m256i c32  = _mm256_set1_epi8(32);
            auto store_q32 = [](__m256i q8, float s_lo, float s_hi, float * out) {
                const __m128i lo = _mm256_castsi256_si128(q8);
                const __m128i hi = _mm256_extracti128_si256(q8, 1);
                const __m256 vlo = _mm256_set1_ps(s_lo);
                const __m256 vhi = _mm256_set1_ps(s_hi);
                _mm256_storeu_ps(out +  0, _mm256_mul_ps(vlo, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
                _mm256_storeu_ps(out +  8, _mm256_mul_ps(vlo, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
                _mm256_storeu_ps(out + 16, _mm256_mul_ps(vhi, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
                _mm256_storeu_ps(out + 24, _mm256_mul_ps(vhi, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
            };
            for (int b = 0; b < nb; ++b) {
                if (b + 1 < nb) {
                    const char * pf = reinterpret_cast<const char *>(blk + b + 1);
                    _mm_prefetch(pf +   0, _MM_HINT_T0);
                    _mm_prefetch(pf + 128, _MM_HINT_T0);
                }
                const __m256 vd = _mm256_set1_ps(f16_to_f32_fast(blk[b].d));
                const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b].scales));
                alignas(32) float scf[16];
                _mm256_store_ps(scf + 0, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc8))));
                _mm256_store_ps(scf + 8, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sc8, 8)))));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t * ql = blk[b].ql + h * 64;
                    const uint8_t * qh = blk[b].qh + h * 32;
                    const float *   sf = scf + h * 8;
                    float * out = dst + b * 256 + h * 128;

                    const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
                    const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
                    const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

                    const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_and_si256(_mm256_slli_epi16(qhv, 4), m30)), c32);
                    const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_and_si256(_mm256_slli_epi16(qhv, 2), m30)), c32);
                    const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_and_si256(qhv, m30)), c32);
                    const __m256i q4 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_and_si256(_mm256_srli_epi16(qhv, 2), m30)), c32);

                    store_q32(q1, sf[0], sf[1], out + 0);
                    store_q32(q2, sf[2], sf[3], out + 32);
                    store_q32(q3, sf[4], sf[5], out + 64);
                    store_q32(q4, sf[6], sf[7], out + 96);
                }
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                for (int h = 0; h < 2; ++h) {
                    const uint8_t * ql = blk[b].ql + h * 64;
                    const uint8_t * qh = blk[b].qh + h * 32;
                    const int8_t *  sc = blk[b].scales + h * 8;
                    float * out = dst + b * 256 + h * 128;
                    for (int l = 0; l < 32; ++l) {
                        const int is = l >> 4;
                        const int q1 = static_cast<int>((ql[l] & 0xFu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                        const int q2 = static_cast<int>((ql[l + 32] & 0xFu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                        const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                        const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                        out[l + 0]  = d * sc[is + 0] * q1;
                        out[l + 32] = d * sc[is + 2] * q2;
                        out[l + 64] = d * sc[is + 4] * q3;
                        out[l + 96] = d * sc[is + 6] * q4;
                    }
                }
            }
#endif
            return;
        }
        case DType::Q4_0: {
            const int b0 = k0 >> 5;
            const int nb = kc >> 5;
            const auto * blk = static_cast<const BlockQ4_0 *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i c8 = _mm256_set1_epi8(8);
            int b = 0;
#if defined(__F16C__)
            for (; b + 7 < nb; b += 8) {
                alignas(32) uint16_t dh[8] = {
                    blk[b + 0].d, blk[b + 1].d, blk[b + 2].d, blk[b + 3].d,
                    blk[b + 4].d, blk[b + 5].d, blk[b + 6].d, blk[b + 7].d
                };
                alignas(32) float df[8];
                _mm256_store_ps(df, _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(dh))));
                for (int bi = 0; bi < 8; ++bi) {
                    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b + bi].qs));
                    const __m256i q8  = _mm256_sub_epi8(_mm256_and_si256(
                        _mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4), c8);
                    store_i8x32_scaled_f32_avx2(q8, df[bi], dst + (b + bi) * 32);
                }
            }
#endif
            for (; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b].qs));
                const __m256i q8  = _mm256_sub_epi8(_mm256_and_si256(
                    _mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4), c8);
                store_i8x32_scaled_f32_avx2(q8, d, dst + b * 32);
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                float * out = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    out[j]      = d * static_cast<float>(static_cast<int>(blk[b].qs[j] & 0xFu) - 8);
                    out[j + 16] = d * static_cast<float>(static_cast<int>(blk[b].qs[j] >> 4) - 8);
                }
            }
#endif
            return;
        }
        case DType::Q5_0: {
            const int b0 = k0 >> 5;
            const int nb = kc >> 5;
            const auto * blk = static_cast<const BlockQ5_0 *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m128i m4  = _mm_set1_epi8(0x0F);
            const __m256i c16 = _mm256_set1_epi8(16);
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                uint32_t qh = 0;
                std::memcpy(&qh, blk[b].qh, 4);
                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b].qs));
                const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), _mm_and_si128(raw, m4));
                const __m256i q8  = _mm256_sub_epi8(_mm256_or_si256(nib, unpack_bits32_to_0x10_avx2(qh)), c16);
                store_i8x32_scaled_f32_avx2(q8, d, dst + b * 32);
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                uint32_t qh = 0;
                std::memcpy(&qh, blk[b].qh, 4);
                float * out = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
                    const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10u);
                    out[j]      = d * static_cast<float>(static_cast<int>((blk[b].qs[j] & 0x0Fu) | xh0) - 16);
                    out[j + 16] = d * static_cast<float>(static_cast<int>((blk[b].qs[j] >> 4)   | xh1) - 16);
                }
            }
#endif
            return;
        }
        case DType::Q4_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ4_K *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            for (int b = 0; b < nb; ++b) {
                const float d    = f16_to_f32_fast(blk[b].d);
                const float dmin = f16_to_f32_fast(blk[b].dmin);
                const uint8_t * q = blk[b].qs;
                float * out = dst + b * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk[b].scales, sc, m);
                    const float d1 = d * sc, nm1 = -dmin * m;
                    get_scale_min_k4(is + 1, blk[b].scales, sc, m);
                    const float d2 = d * sc, nm2 = -dmin * m;
                    const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
                    store_u8x32_affine_f32_avx2(_mm256_and_si256(qv, m4), d1, nm1, out + j);
                    store_u8x32_affine_f32_avx2(_mm256_and_si256(_mm256_srli_epi16(qv, 4), m4), d2, nm2, out + j + 32);
                    q += 32; is += 2;
                }
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d    = f16_to_f32_fast(blk[b].d);
                const float dmin = f16_to_f32_fast(blk[b].dmin);
                const uint8_t * q = blk[b].qs;
                float * out = dst + b * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk[b].scales, sc, m);
                    const float d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1, blk[b].scales, sc, m);
                    const float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; ++l) {
                        out[j + l]      = d1 * static_cast<float>(q[l] & 0xFu) - m1;
                        out[j + 32 + l] = d2 * static_cast<float>(q[l] >> 4)   - m2;
                    }
                    q += 32; is += 2;
                }
            }
#endif
            return;
        }
        case DType::Q5_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ5_K *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m4  = _mm256_set1_epi8(0x0F);
            const __m256i b16 = _mm256_set1_epi8(16);
            for (int b = 0; b < nb; ++b) {
                const float d    = f16_to_f32_fast(blk[b].d);
                const float dmin = f16_to_f32_fast(blk[b].dmin);
                const uint8_t * ql = blk[b].qs;
                const __m256i qhv  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk[b].qh));
                float * out = dst + b * 256;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk[b].scales, sc, m);
                    const float d1 = d * sc, nm1 = -dmin * m;
                    get_scale_min_k4(is + 1, blk[b].scales, sc, m);
                    const float d2 = d * sc, nm2 = -dmin * m;
                    const __m256i qlv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
                    const __m256i vu1 = _mm256_set1_epi8(static_cast<char>(u1));
                    const __m256i vu2 = _mm256_set1_epi8(static_cast<char>(u2));
                    const __m256i h1  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, vu1), vu1), b16);
                    const __m256i h2  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, vu2), vu2), b16);
                    store_u8x32_affine_f32_avx2(_mm256_or_si256(_mm256_and_si256(qlv, m4), h1), d1, nm1, out + j);
                    store_u8x32_affine_f32_avx2(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4), h2), d2, nm2, out + j + 32);
                    ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
                }
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d    = f16_to_f32_fast(blk[b].d);
                const float dmin = f16_to_f32_fast(blk[b].dmin);
                const uint8_t * ql = blk[b].qs;
                const uint8_t * qh = blk[b].qh;
                float * out = dst + b * 256;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk[b].scales, sc, m);
                    const float d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1, blk[b].scales, sc, m);
                    const float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; ++l) {
                        const int q1 = (ql[l] & 0xFu) + ((qh[l] & u1) ? 16 : 0);
                        const int q2 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                        out[j + l]      = d1 * static_cast<float>(q1) - m1;
                        out[j + 32 + l] = d2 * static_cast<float>(q2) - m2;
                    }
                    ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
                }
            }
#endif
            return;
        }
        case DType::IQ4_NL: {
            const int b0 = k0 >> 5;
            const int nb = kc >> 5;
            const auto * blk = static_cast<const BlockIQ4_NL *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m4   = _mm256_set1_epi8(0x0F);
            const __m128i l128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
            const __m256i lut  = _mm256_broadcastsi128_si256(l128);
            int b = 0;
#if defined(__F16C__)
            for (; b + 7 < nb; b += 8) {
                alignas(32) uint16_t dh[8] = {
                    blk[b + 0].d, blk[b + 1].d, blk[b + 2].d, blk[b + 3].d,
                    blk[b + 4].d, blk[b + 5].d, blk[b + 6].d, blk[b + 7].d
                };
                alignas(32) float df[8];
                _mm256_store_ps(df, _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(dh))));
                for (int bi = 0; bi < 8; ++bi) {
                    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b + bi].qs));
                    const __m256i nib = _mm256_and_si256(
                        _mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                    const __m256i q8  = _mm256_shuffle_epi8(lut, nib);
                    store_i8x32_scaled_f32_avx2(q8, df[bi], dst + (b + bi) * 32);
                }
            }
#endif
            for (; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk[b].qs));
                const __m256i nib = _mm256_and_si256(
                    _mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                const __m256i q8  = _mm256_shuffle_epi8(lut, nib);
                store_i8x32_scaled_f32_avx2(q8, d, dst + b * 32);
            }
#else
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                float * out = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    out[j]      = d * static_cast<float>(kvalues_iq4nl[blk[b].qs[j] & 0xFu]);
                    out[j + 16] = d * static_cast<float>(kvalues_iq4nl[blk[b].qs[j] >> 4]);
                }
            }
#endif
            return;
        }
        case DType::IQ2_S: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockIQ2_S *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i shuf = _mm256_setr_epi8(
                0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
            const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
#endif
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const uint8_t * qs    = blk[b].qs;
                const uint8_t * signs = blk[b].qs + 32;
                const uint8_t * qh    = blk[b].qh;
                float * out = dst + b * 256;
#if defined(__AVX2__) && defined(__FMA__)
                if (b + 1 < nb) {
                    const char * pf = reinterpret_cast<const char *>(blk + b + 1);
                    _mm_prefetch(pf +  0, _MM_HINT_T0);
                    _mm_prefetch(pf + 64, _MM_HINT_T0);
                }
                // Decode all 16 odd scales (0.125 * d * (1 + 2 * nibble)) in 2 AVX2 ops instead of 64 scalar FP ops!
                uint64_t aux64;
                std::memcpy(&aux64, blk[b].scales, 8);
                const uint64_t sc_lo = ((aux64 & 0x0F0F0F0F0F0F0F0FULL) << 1) | 0x0101010101010101ULL;
                const uint64_t sc_hi = (((aux64 >> 4) & 0x0F0F0F0F0F0F0F0FULL) << 1) | 0x0101010101010101ULL;
                const __m256 vd8 = _mm256_set1_ps(d * 0.125f);
                alignas(32) float sf_lo[8], sf_hi[8];
                _mm256_store_ps(sf_lo, _mm256_mul_ps(vd8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(sc_lo))))));
                _mm256_store_ps(sf_hi, _mm256_mul_ps(vd8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(sc_hi))))));
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint32_t qhb = qh[ib32];
                    const uint64_t g0 = iq2s_grid[qs[0] | ((qhb << 8) & 0x300u)];
                    const uint64_t g1 = iq2s_grid[qs[1] | ((qhb << 6) & 0x300u)];
                    const uint64_t g2 = iq2s_grid[qs[2] | ((qhb << 4) & 0x300u)];
                    const uint64_t g3 = iq2s_grid[qs[3] | ((qhb << 2) & 0x300u)];
                    uint32_t smask = 0;
                    std::memcpy(&smask, signs, 4);
                    const __m256i gv  = _mm256_set_epi64x(static_cast<int64_t>(g3), static_cast<int64_t>(g2),
                                                          static_cast<int64_t>(g1), static_cast<int64_t>(g0));
                    const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(smask)), shuf), bitm), bitm);
                    const __m256i q8  = _mm256_sub_epi8(_mm256_xor_si256(gv, neg), neg);
                    const __m128i lo  = _mm256_castsi256_si128(q8);
                    const __m128i hi  = _mm256_extracti128_si256(q8, 1);
                    const __m256  vd0 = _mm256_set1_ps(sf_lo[ib32]);
                    const __m256  vd1 = _mm256_set1_ps(sf_hi[ib32]);
                    _mm256_storeu_ps(out +  0, _mm256_mul_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
                    _mm256_storeu_ps(out +  8, _mm256_mul_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
                    _mm256_storeu_ps(out + 16, _mm256_mul_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
                    _mm256_storeu_ps(out + 24, _mm256_mul_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
                    qs += 4; signs += 4; out += 32;
                }
#else
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const float db0 = d * (0.5f + static_cast<float>(blk[b].scales[ib32] & 0xFu)) * 0.25f;
                    const float db1 = d * (0.5f + static_cast<float>(blk[b].scales[ib32] >> 4))   * 0.25f;
                    const uint32_t qhb = qh[ib32];
                    const uint64_t g0 = iq2s_grid[qs[0] | ((qhb << 8) & 0x300u)];
                    const uint64_t g1 = iq2s_grid[qs[1] | ((qhb << 6) & 0x300u)];
                    const uint64_t g2 = iq2s_grid[qs[2] | ((qhb << 4) & 0x300u)];
                    const uint64_t g3 = iq2s_grid[qs[3] | ((qhb << 2) & 0x300u)];
                    const uint64_t gs[4] = {g0, g1, g2, g3};
                    for (int l = 0; l < 4; ++l) {
                        const float dl = (l < 2) ? db0 : db1;
                        const uint8_t sg = signs[l];
                        for (int j = 0; j < 8; ++j) {
                            const float mag = static_cast<float>((gs[l] >> (8 * j)) & 0xFFu);
                            out[l * 8 + j] = (sg & (1u << j)) ? (-dl * mag) : (dl * mag);
                        }
                    }
                    qs += 4; signs += 4; out += 32;
                }
#endif
            }
            return;
        }
        case DType::IQ3_S: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockIQ3_S *>(row_ptr) + b0;
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const uint8_t * qs    = blk[b].qs;
                const uint8_t * qh    = blk[b].qh;
                const uint8_t * signs = blk[b].signs;
                float * out = dst + b * 256;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t scb = blk[b].scales[ib32 >> 1];
                    const int sc4 = (ib32 & 1) ? (scb >> 4) : (scb & 0xFu);
                    const float db = d * static_cast<float>(1 + 2 * sc4);
                    const uint32_t qhb = qh[ib32];
                    const uint32_t g0 = iq3s_grid[qs[0] | ((qhb << 8) & 256u)];
                    const uint32_t g1 = iq3s_grid[qs[1] | ((qhb << 7) & 256u)];
                    const uint32_t g2 = iq3s_grid[qs[2] | ((qhb << 6) & 256u)];
                    const uint32_t g3 = iq3s_grid[qs[3] | ((qhb << 5) & 256u)];
                    const uint32_t g4 = iq3s_grid[qs[4] | ((qhb << 4) & 256u)];
                    const uint32_t g5 = iq3s_grid[qs[5] | ((qhb << 3) & 256u)];
                    const uint32_t g6 = iq3s_grid[qs[6] | ((qhb << 2) & 256u)];
                    const uint32_t g7 = iq3s_grid[qs[7] | ((qhb << 1) & 256u)];
#if defined(__AVX2__) && defined(__FMA__)
                    uint32_t smask = 0;
                    std::memcpy(&smask, signs, 4);
                    const __m256i gv = _mm256_setr_epi32(
                        static_cast<int32_t>(g0), static_cast<int32_t>(g1), static_cast<int32_t>(g2), static_cast<int32_t>(g3),
                        static_cast<int32_t>(g4), static_cast<int32_t>(g5), static_cast<int32_t>(g6), static_cast<int32_t>(g7));
                    const __m256i q8 = _mm256_sign_epi8(gv, unpack_sign32_to_i8_avx2(smask));
                    store_i8x32_scaled_f32_avx2(q8, db, out);
#else
                    const uint32_t gs[8] = {g0, g1, g2, g3, g4, g5, g6, g7};
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t sg = signs[l];
                        for (int j = 0; j < 4; ++j) {
                            const float m0 = static_cast<float>((gs[2 * l + 0] >> (8 * j)) & 0xFFu);
                            const float m1 = static_cast<float>((gs[2 * l + 1] >> (8 * j)) & 0xFFu);
                            out[l * 8 + j + 0] = (sg & (1u << (j + 0))) ? (-db * m0) : (db * m0);
                            out[l * 8 + j + 4] = (sg & (1u << (j + 4))) ? (-db * m1) : (db * m1);
                        }
                    }
#endif
                    qs += 8; signs += 4; out += 32;
                }
            }
            return;
        }
        case DType::Q2_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ2_K *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m2 = _mm256_set1_epi8(0x03);
#endif
            for (int b = 0; b < nb; ++b) {
                const float d    = f16_to_f32_fast(blk[b].d);
                const float dmin = f16_to_f32_fast(blk[b].dmin);
                float * out = dst + b * 256;
                for (int n = 0; n < 2; ++n) {
                    const uint8_t * q  = blk[b].qs + n * 32;
                    const uint8_t * sc = blk[b].scales + n * 8;
#if defined(__AVX2__) && defined(__FMA__)
                    const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
                    for (int j = 0; j < 4; ++j) {
                        const float dl0 = d * static_cast<float>(sc[2 * j + 0] & 0xFu);
                        const float ml0 = -dmin * static_cast<float>(sc[2 * j + 0] >> 4);
                        const float dl1 = d * static_cast<float>(sc[2 * j + 1] & 0xFu);
                        const float ml1 = -dmin * static_cast<float>(sc[2 * j + 1] >> 4);
                        const __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m2);
                        const __m128i lo = _mm256_castsi256_si128(q2);
                        const __m128i hi = _mm256_extracti128_si256(q2, 1);
                        const __m256 vd0 = _mm256_set1_ps(dl0), vm0 = _mm256_set1_ps(ml0);
                        const __m256 vd1 = _mm256_set1_ps(dl1), vm1 = _mm256_set1_ps(ml1);
                        float * o = out + n * 128 + j * 32;
                        _mm256_storeu_ps(o +  0, _mm256_fmadd_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), vm0));
                        _mm256_storeu_ps(o +  8, _mm256_fmadd_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), vm0));
                        _mm256_storeu_ps(o + 16, _mm256_fmadd_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), vm1));
                        _mm256_storeu_ps(o + 24, _mm256_fmadd_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), vm1));
                    }
#else
                    for (int j = 0; j < 4; ++j) {
                        const int shift = 2 * j;
                        const float dl0 = d * static_cast<float>(sc[2 * j + 0] & 0xFu);
                        const float ml0 = dmin * static_cast<float>(sc[2 * j + 0] >> 4);
                        const float dl1 = d * static_cast<float>(sc[2 * j + 1] & 0xFu);
                        const float ml1 = dmin * static_cast<float>(sc[2 * j + 1] >> 4);
                        float * o = out + n * 128 + j * 32;
                        for (int l = 0; l < 16; ++l) {
                            o[l]      = dl0 * static_cast<float>((q[l]      >> shift) & 3u) - ml0;
                            o[l + 16] = dl1 * static_cast<float>((q[l + 16] >> shift) & 3u) - ml1;
                        }
                    }
#endif
                }
            }
            return;
        }
        case DType::Q3_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ3_K *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i m2 = _mm256_set1_epi8(0x03);
            const __m256i c4 = _mm256_set1_epi8(4);
#endif
            for (int b = 0; b < nb; ++b) {
                const float d_all = f16_to_f32_fast(blk[b].d);
                uint32_t aux[4];
                std::memcpy(aux, blk[b].scales, 12);
                const uint32_t tmp = aux[2];
                constexpr uint32_t kmask1 = 0x03030303u;
                constexpr uint32_t kmask2 = 0x0f0f0f0fu;
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                uint8_t sc_u[16];
                std::memcpy(sc_u, aux, 16);
                float * out = dst + b * 256;
#if defined(__AVX2__) && defined(__FMA__)
                const __m256i hmv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk[b].hmask));
                for (int n = 0; n < 2; ++n) {
                    const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk[b].qs + n * 32));
                    for (int j = 0; j < 4; ++j) {
                        const uint8_t m = static_cast<uint8_t>(1u << (n * 4 + j));
                        const __m256i vm = _mm256_set1_epi8(static_cast<char>(m));
                        const __m256i hb = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hmv, vm), vm), c4);
                        const __m256i q8 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m2), hb), c4);
                        const float dl0 = d_all * static_cast<float>(static_cast<int>(sc_u[n * 8 + 2 * j + 0]) - 32);
                        const float dl1 = d_all * static_cast<float>(static_cast<int>(sc_u[n * 8 + 2 * j + 1]) - 32);
                        const __m128i lo = _mm256_castsi256_si128(q8);
                        const __m128i hi = _mm256_extracti128_si256(q8, 1);
                        const __m256 vd0 = _mm256_set1_ps(dl0), vd1 = _mm256_set1_ps(dl1);
                        float * o = out + n * 128 + j * 32;
                        _mm256_storeu_ps(o +  0, _mm256_mul_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
                        _mm256_storeu_ps(o +  8, _mm256_mul_ps(vd0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
                        _mm256_storeu_ps(o + 16, _mm256_mul_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
                        _mm256_storeu_ps(o + 24, _mm256_mul_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
                    }
                }
#else
                for (int n = 0; n < 2; ++n) {
                    const uint8_t * q = blk[b].qs + n * 32;
                    for (int j = 0; j < 4; ++j) {
                        const int shift = 2 * j;
                        const uint8_t m = static_cast<uint8_t>(1u << (n * 4 + j));
                        const float dl0 = d_all * static_cast<float>(static_cast<int>(sc_u[n * 8 + 2 * j + 0]) - 32);
                        const float dl1 = d_all * static_cast<float>(static_cast<int>(sc_u[n * 8 + 2 * j + 1]) - 32);
                        float * o = out + n * 128 + j * 32;
                        for (int l = 0; l < 16; ++l) {
                            const int v0 = static_cast<int>((q[l]      >> shift) & 3u) - ((blk[b].hmask[l]      & m) ? 0 : 4);
                            const int v1 = static_cast<int>((q[l + 16] >> shift) & 3u) - ((blk[b].hmask[l + 16] & m) ? 0 : 4);
                            o[l]      = dl0 * static_cast<float>(v0);
                            o[l + 16] = dl1 * static_cast<float>(v1);
                        }
                    }
                }
#endif
            }
            return;
        }
        case DType::IQ4_XS: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockIQ4_XS *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m128i m4   = _mm_set1_epi8(0x0F);
            const __m256i lut  = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl)));
#endif
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const uint16_t sh = blk[b].scales_h;
                float * out = dst + b * 256;
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ((blk[b].scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                    const float dl = d * static_cast<float>(ls - 32);
                    const uint8_t * qs = blk[b].qs + ib * 16;
#if defined(__AVX2__) && defined(__FMA__)
                    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));
                    const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), _mm_and_si128(raw, m4));
                    const __m256i q8  = _mm256_shuffle_epi8(lut, nib);
                    store_i8x32_scaled_f32_avx2(q8, dl, out + ib * 32);
#else
                    float * o = out + ib * 32;
                    for (int j = 0; j < 16; ++j) {
                        o[j]      = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xFu]);
                        o[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
                    }
#endif
                }
            }
            return;
        }
        case DType::IQ3_XXS: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockIQ3_XXS *>(row_ptr) + b0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256i shuf = _mm256_setr_epi8(
                0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
            const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
#endif
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                const uint8_t * qs = blk[b].qs;
                const uint8_t * ss = blk[b].qs + 64;
                float * out = dst + b * 256;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    uint32_t aux32 = 0;
                    std::memcpy(&aux32, ss + ib32 * 4, 4);
                    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                    const uint32_t s0 = ksigns_iq2xs[(aux32 >>  0) & 127u];
                    const uint32_t s1 = ksigns_iq2xs[(aux32 >>  7) & 127u];
                    const uint32_t s2 = ksigns_iq2xs[(aux32 >> 14) & 127u];
                    const uint32_t s3 = ksigns_iq2xs[(aux32 >> 21) & 127u];
                    const uint32_t g0 = iq3xxs_grid[qs[0]], g1 = iq3xxs_grid[qs[1]];
                    const uint32_t g2 = iq3xxs_grid[qs[2]], g3 = iq3xxs_grid[qs[3]];
                    const uint32_t g4 = iq3xxs_grid[qs[4]], g5 = iq3xxs_grid[qs[5]];
                    const uint32_t g6 = iq3xxs_grid[qs[6]], g7 = iq3xxs_grid[qs[7]];
#if defined(__AVX2__) && defined(__FMA__)
                    const uint32_t smask = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
                    const __m256i gv = _mm256_setr_epi32(
                        static_cast<int32_t>(g0), static_cast<int32_t>(g1), static_cast<int32_t>(g2), static_cast<int32_t>(g3),
                        static_cast<int32_t>(g4), static_cast<int32_t>(g5), static_cast<int32_t>(g6), static_cast<int32_t>(g7));
                    const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(smask)), shuf), bitm), bitm);
                    const __m256i q8  = _mm256_sub_epi8(_mm256_xor_si256(gv, neg), neg);
                    store_i8x32_scaled_f32_avx2(q8, db, out);
#else
                    const uint32_t gs[8] = {g0, g1, g2, g3, g4, g5, g6, g7};
                    const uint8_t  sg[4] = {static_cast<uint8_t>(s0), static_cast<uint8_t>(s1), static_cast<uint8_t>(s2), static_cast<uint8_t>(s3)};
                    for (int l = 0; l < 4; ++l) {
                        for (int j = 0; j < 4; ++j) {
                            const float m0 = static_cast<float>((gs[2 * l + 0] >> (8 * j)) & 0xFFu);
                            const float m1 = static_cast<float>((gs[2 * l + 1] >> (8 * j)) & 0xFFu);
                            out[l * 8 + j + 0] = (sg[l] & (1u << (j + 0))) ? (-db * m0) : (db * m0);
                            out[l * 8 + j + 4] = (sg[l] & (1u << (j + 4))) ? (-db * m1) : (db * m1);
                        }
                    }
#endif
                    qs += 8; out += 32;
                }
            }
            return;
        }
        default:
            throw std::runtime_error(std::string("CPU dequant_slice_to_f32 unsupported dtype: ") + dtype_traits(dt).name);
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// Static vpshufb masks for Q6_K scale broadcasts: low 128b = sc[2k] x 8, high 128b = sc[2k+1] x 8.
alignas(32) static constexpr uint8_t kQ6ScShuf[4][32] = {
    { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3},
    { 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,  6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7},
    { 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11},
    {12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15},
};
#endif

// ============================================================================
// Haswell/AVX2+FMA L1-Tiled Dual-FMA SGEMM Microkernels (Prefill T >= 12)
// Uses true BLIS contiguous [n_tiles][K][16] panel packing (zero L1 set aliasing!)
// On AVX2 without VNNI, dual-port FP32 FMA achieves ~30 FLOPs/cy/core (~2x INT8)!
// ============================================================================
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)

// In-register 8x8 FP32 transpose helper for AVX2 panel packing (avoids 16-way L1d set aliasing!)
inline void transpose8x8_store_stride(
    const float * r0, const float * r1, const float * r2, const float * r3,
    const float * r4, const float * r5, const float * r6, const float * r7,
    float * dst, int dst_stride) {
    __m256 v0 = _mm256_loadu_ps(r0); __m256 v1 = _mm256_loadu_ps(r1);
    __m256 v2 = _mm256_loadu_ps(r2); __m256 v3 = _mm256_loadu_ps(r3);
    __m256 v4 = _mm256_loadu_ps(r4); __m256 v5 = _mm256_loadu_ps(r5);
    __m256 v6 = _mm256_loadu_ps(r6); __m256 v7 = _mm256_loadu_ps(r7);

    __m256 t0 = _mm256_unpacklo_ps(v0, v1); __m256 t1 = _mm256_unpackhi_ps(v0, v1);
    __m256 t2 = _mm256_unpacklo_ps(v2, v3); __m256 t3 = _mm256_unpackhi_ps(v2, v3);
    __m256 t4 = _mm256_unpacklo_ps(v4, v5); __m256 t5 = _mm256_unpackhi_ps(v4, v5);
    __m256 t6 = _mm256_unpacklo_ps(v6, v7); __m256 t7 = _mm256_unpackhi_ps(v6, v7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    _mm256_storeu_ps(dst + 0 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(dst + 1 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(dst + 2 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(dst + 3 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(dst + 4 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(dst + 5 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(dst + 6 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(dst + 7 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

// Pack X[T, K] into contiguous tile panels:
//   tile tj (covering tokens [tj*16 .. tj*16 + 15]) occupies xp + tj * K * 16,
//   where entry (k, j) is at offset k * 16 + j (100% contiguous 64B cache lines per k!).
inline void pack_x_panels_16_range(const float * x, float * xp, int T, int K, int tj0, int tj1) {
    for (int tj = tj0; tj < tj1; ++tj) {
        const int t0 = tj * 16;
        const int nt = std::min(16, T - t0);
        float * dst_tile = xp + static_cast<size_t>(tj) * K * 16;
        if (nt == 16) {
            const float * x0 = x + static_cast<size_t>(t0 + 0) * K;
            int k = 0;
            for (; k + 7 < K; k += 8) {
                float * dk = dst_tile + static_cast<size_t>(k) * 16;
                transpose8x8_store_stride(
                    x0 + 0 * K + k, x0 + 1 * K + k, x0 + 2 * K + k, x0 + 3 * K + k,
                    x0 + 4 * K + k, x0 + 5 * K + k, x0 + 6 * K + k, x0 + 7 * K + k,
                    dk + 0, 16);
                transpose8x8_store_stride(
                    x0 + 8 * K + k, x0 + 9 * K + k, x0 + 10 * K + k, x0 + 11 * K + k,
                    x0 + 12 * K + k, x0 + 13 * K + k, x0 + 14 * K + k, x0 + 15 * K + k,
                    dk + 8, 16);
            }
            for (; k < K; ++k) {
                float * dk = dst_tile + static_cast<size_t>(k) * 16;
                for (int j = 0; j < 16; ++j) dk[j] = x0[static_cast<size_t>(j) * K + k];
            }
        } else {
            for (int k = 0; k < K; ++k) {
                float * dk = dst_tile + static_cast<size_t>(k) * 16;
                for (int j = 0; j < nt; ++j) dk[j] = x[static_cast<size_t>(t0 + j) * K + k];
                for (int j = nt; j < 16; ++j) dk[j] = 0.0f;
            }
        }
    }
}

// Pack X[nt, K] (nt <= 8) into contiguous [K][8] panel for zero-padding-waste 12x8 MoE FMA microkernel!
inline void pack_x_panels_8(const float * x, float * xp, int nt, int K) {
    alignas(32) float zrow[8] = {};
    if (nt == 8) {
        int k = 0;
        for (; k + 7 < K; k += 8) {
            transpose8x8_store_stride(
                x + 0 * K + k, x + 1 * K + k, x + 2 * K + k, x + 3 * K + k,
                x + 4 * K + k, x + 5 * K + k, x + 6 * K + k, x + 7 * K + k,
                xp + static_cast<size_t>(k) * 8, 8);
        }
        for (; k < K; ++k) {
            float * dk = xp + static_cast<size_t>(k) * 8;
            for (int j = 0; j < 8; ++j) dk[j] = x[static_cast<size_t>(j) * K + k];
        }
    } else {
        int k = 0;
        for (; k + 7 < K; k += 8) {
            const float * rp[8];
            for (int j = 0; j < nt; ++j) rp[j] = x + static_cast<size_t>(j) * K + k;
            for (int j = nt; j < 8; ++j) rp[j] = zrow;
            transpose8x8_store_stride(rp[0], rp[1], rp[2], rp[3], rp[4], rp[5], rp[6], rp[7],
                                      xp + static_cast<size_t>(k) * 8, 8);
        }
        for (; k < K; ++k) {
            float * dk = xp + static_cast<size_t>(k) * 8;
            for (int j = 0; j < nt; ++j) dk[j] = x[static_cast<size_t>(j) * K + k];
            for (int j = nt; j < 8; ++j) dk[j] = 0.0f;
        }
    }
}

inline void pack_x_panels_16(const float * x, float * xp, int T, int K) {
    pack_x_panels_16_range(x, xp, T, K, 0, (T + 15) >> 4);
}

inline void pack_x_panels_16_mt(ThreadPool & pool, const float * x, float * xp, int T, int K) {
    const int n_tiles = (T + 15) >> 4;
    if (n_tiles < 4 || pool.num_threads() <= 1) {
        pack_x_panels_16_range(x, xp, T, K, 0, n_tiles);
        return;
    }
    pool.parallel_for(n_tiles, [&](int tj0, int tj1, int) {
        pack_x_panels_16_range(x, xp, T, K, tj0, tj1);
    });
}

// 6 rows x 16 tokens dual-FMA microkernel on contiguous [KC][16] X tile (15 YMM regs, zero spills, k-unrolled x2!)
inline void fma_kernel_6x16_contig(
    const float * __restrict__ wp, // [6][KC] contiguous per row
    const float * __restrict__ xp, // [KC][16] contiguous (stride 16 = 64B!)
    int KC,
    float * __restrict__ acc_buf, // [6][16] aligned
    bool first_k) {

    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (first_k) {
        c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
        c40 = _mm256_setzero_ps(); c41 = _mm256_setzero_ps();
        c50 = _mm256_setzero_ps(); c51 = _mm256_setzero_ps();
    } else {
        c00 = _mm256_load_ps(acc_buf +  0); c01 = _mm256_load_ps(acc_buf +  8);
        c10 = _mm256_load_ps(acc_buf + 16); c11 = _mm256_load_ps(acc_buf + 24);
        c20 = _mm256_load_ps(acc_buf + 32); c21 = _mm256_load_ps(acc_buf + 40);
        c30 = _mm256_load_ps(acc_buf + 48); c31 = _mm256_load_ps(acc_buf + 56);
        c40 = _mm256_load_ps(acc_buf + 64); c41 = _mm256_load_ps(acc_buf + 72);
        c50 = _mm256_load_ps(acc_buf + 80); c51 = _mm256_load_ps(acc_buf + 88);
    }

    const float * w0 = wp + 0 * KC;
    const float * w1 = wp + 1 * KC;
    const float * w2 = wp + 2 * KC;
    const float * w3 = wp + 3 * KC;
    const float * w4 = wp + 4 * KC;
    const float * w5 = wp + 5 * KC;

    for (int k = 0; k < KC; ++k) {
        const float * xk = xp + static_cast<size_t>(k) * 16;
        const __m256 x0 = _mm256_loadu_ps(xk + 0);
        const __m256 x1 = _mm256_loadu_ps(xk + 8);
        __m256 w;
        w = _mm256_broadcast_ss(w0 + k); c00 = _mm256_fmadd_ps(w, x0, c00); c01 = _mm256_fmadd_ps(w, x1, c01);
        w = _mm256_broadcast_ss(w1 + k); c10 = _mm256_fmadd_ps(w, x0, c10); c11 = _mm256_fmadd_ps(w, x1, c11);
        w = _mm256_broadcast_ss(w2 + k); c20 = _mm256_fmadd_ps(w, x0, c20); c21 = _mm256_fmadd_ps(w, x1, c21);
        w = _mm256_broadcast_ss(w3 + k); c30 = _mm256_fmadd_ps(w, x0, c30); c31 = _mm256_fmadd_ps(w, x1, c31);
        w = _mm256_broadcast_ss(w4 + k); c40 = _mm256_fmadd_ps(w, x0, c40); c41 = _mm256_fmadd_ps(w, x1, c41);
        w = _mm256_broadcast_ss(w5 + k); c50 = _mm256_fmadd_ps(w, x0, c50); c51 = _mm256_fmadd_ps(w, x1, c51);
    }

    _mm256_store_ps(acc_buf +  0, c00); _mm256_store_ps(acc_buf +  8, c01);
    _mm256_store_ps(acc_buf + 16, c10); _mm256_store_ps(acc_buf + 24, c11);
    _mm256_store_ps(acc_buf + 32, c20); _mm256_store_ps(acc_buf + 40, c21);
    _mm256_store_ps(acc_buf + 48, c30); _mm256_store_ps(acc_buf + 56, c31);
    _mm256_store_ps(acc_buf + 64, c40); _mm256_store_ps(acc_buf + 72, c41);
    _mm256_store_ps(acc_buf + 80, c50); _mm256_store_ps(acc_buf + 88, c51);
}

// Compute Y[0..T, r0..r1) = X_packed_16 * W[r0..r1, 0..K]^T using L1-tiled contiguous dual-FMA!
inline void gemm_fma_rows_range(
    const CpuTensor & W,
    const float * xp, // [n_tiles][K][16]
    float * y,        // [T, M]
    int T, int r0, int r1) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    const auto tr = dtype_traits(W.dtype);
    const size_t row_bytes = (static_cast<size_t>(K) / tr.block_elems) * tr.block_bytes;
    const auto * w_base = static_cast<const uint8_t *>(W.ptr);

    // KC=256, TILE_CHUNK=8 (128 tokens): x_tl=128 KB + wp=6 KB + acc_full=3 KB = 137 KB (~53% of 256 KB Haswell L2),
    // leaving ample L2 headroom for streaming quantized weights without thrashing on pp128/pp512.
    constexpr int KC         = 256;
    constexpr int TILE_CHUNK = 8;

    alignas(32) float wp[6 * KC];
    const int n_tiles_tot = (T + 15) >> 4;
    alignas(32) float acc_full[TILE_CHUNK * 6 * 16];

    for (int tile0 = 0; tile0 < n_tiles_tot; tile0 += TILE_CHUNK) {
        const int n_tl = std::min(TILE_CHUNK, n_tiles_tot - tile0);

        int r = r0;
        for (; r + 5 < r1; r += 6) {
            for (int k0 = 0; k0 < K; k0 += KC) {
                const int kc = std::min(KC, K - k0);
                for (int ri = 0; ri < 6; ++ri) {
                    dequant_slice_to_f32(W.dtype, w_base + static_cast<size_t>(r + ri) * row_bytes, k0, kc, wp + ri * kc);
                }
                for (int tl = 0; tl < n_tl; ++tl) {
                    const float * x_tl = xp + (static_cast<size_t>(tile0 + tl) * K + k0) * 16;
                    fma_kernel_6x16_contig(wp, x_tl, kc, acc_full + tl * 96, k0 == 0);
                }
            }
            for (int tl = 0; tl < n_tl; ++tl) {
                const int t_start = (tile0 + tl) * 16;
                const int nt = std::min(16, T - t_start);
                const float * ap = acc_full + tl * 96;
                for (int j = 0; j < nt; ++j) {
                    float * yt = y + static_cast<size_t>(t_start + j) * M + r;
                    yt[0] = ap[0 * 16 + j];
                    yt[1] = ap[1 * 16 + j];
                    yt[2] = ap[2 * 16 + j];
                    yt[3] = ap[3 * 16 + j];
                    yt[4] = ap[4 * 16 + j];
                    yt[5] = ap[5 * 16 + j];
                }
            }
        }
        // Tail rows (< 6)
        for (; r < r1; ++r) {
            alignas(32) float row_acc[TILE_CHUNK * 16] = {};
            for (int k0 = 0; k0 < K; k0 += KC) {
                const int kc = std::min(KC, K - k0);
                dequant_slice_to_f32(W.dtype, w_base + static_cast<size_t>(r) * row_bytes, k0, kc, wp);
                for (int tl = 0; tl < n_tl; ++tl) {
                    const float * x_tl = xp + (static_cast<size_t>(tile0 + tl) * K + k0) * 16;
                    __m256 c0 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_load_ps(row_acc + tl * 16 + 0);
                    __m256 c1 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_load_ps(row_acc + tl * 16 + 8);
                    for (int k = 0; k < kc; ++k) {
                        const __m256 w = _mm256_broadcast_ss(wp + k);
                        c0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x_tl + k * 16 + 0), c0);
                        c1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x_tl + k * 16 + 8), c1);
                    }
                    _mm256_store_ps(row_acc + tl * 16 + 0, c0);
                    _mm256_store_ps(row_acc + tl * 16 + 8, c1);
                }
            }
            for (int tl = 0; tl < n_tl; ++tl) {
                const int t_start = (tile0 + tl) * 16;
                const int nt = std::min(16, T - t_start);
                for (int j = 0; j < nt; ++j) y[static_cast<size_t>(t_start + j) * M + r] = row_acc[tl * 16 + j];
            }
        }
    }
}

// 12 rows x 8 tokens dual-FMA microkernel on contiguous [KC][8] X tile (14 YMMs, ZERO spills, ZERO wasted padding for nt=5..8!)
inline void fma_kernel_12x8_contig(
    const float * __restrict__ wp, // [12][KC]
    const float * __restrict__ xp, // [KC][8]
    int KC,
    float * __restrict__ acc_buf,  // [12][8]
    bool first_k) {

    __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;
    if (first_k) {
        c0 = _mm256_setzero_ps(); c1 = _mm256_setzero_ps(); c2 = _mm256_setzero_ps(); c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps(); c5 = _mm256_setzero_ps(); c6 = _mm256_setzero_ps(); c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps(); c9 = _mm256_setzero_ps(); c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
    } else {
        c0  = _mm256_load_ps(acc_buf +  0); c1  = _mm256_load_ps(acc_buf +  8);
        c2  = _mm256_load_ps(acc_buf + 16); c3  = _mm256_load_ps(acc_buf + 24);
        c4  = _mm256_load_ps(acc_buf + 32); c5  = _mm256_load_ps(acc_buf + 40);
        c6  = _mm256_load_ps(acc_buf + 48); c7  = _mm256_load_ps(acc_buf + 56);
        c8  = _mm256_load_ps(acc_buf + 64); c9  = _mm256_load_ps(acc_buf + 72);
        c10 = _mm256_load_ps(acc_buf + 80); c11 = _mm256_load_ps(acc_buf + 88);
    }

    const float * w0  = wp +  0 * KC; const float * w1  = wp +  1 * KC;
    const float * w2  = wp +  2 * KC; const float * w3  = wp +  3 * KC;
    const float * w4  = wp +  4 * KC; const float * w5  = wp +  5 * KC;
    const float * w6  = wp +  6 * KC; const float * w7  = wp +  7 * KC;
    const float * w8  = wp +  8 * KC; const float * w9  = wp +  9 * KC;
    const float * w10 = wp + 10 * KC; const float * w11 = wp + 11 * KC;

    for (int k = 0; k < KC; ++k) {
        const __m256 xv = _mm256_loadu_ps(xp + static_cast<size_t>(k) * 8);
        c0  = _mm256_fmadd_ps(_mm256_broadcast_ss(w0  + k), xv, c0);
        c1  = _mm256_fmadd_ps(_mm256_broadcast_ss(w1  + k), xv, c1);
        c2  = _mm256_fmadd_ps(_mm256_broadcast_ss(w2  + k), xv, c2);
        c3  = _mm256_fmadd_ps(_mm256_broadcast_ss(w3  + k), xv, c3);
        c4  = _mm256_fmadd_ps(_mm256_broadcast_ss(w4  + k), xv, c4);
        c5  = _mm256_fmadd_ps(_mm256_broadcast_ss(w5  + k), xv, c5);
        c6  = _mm256_fmadd_ps(_mm256_broadcast_ss(w6  + k), xv, c6);
        c7  = _mm256_fmadd_ps(_mm256_broadcast_ss(w7  + k), xv, c7);
        c8  = _mm256_fmadd_ps(_mm256_broadcast_ss(w8  + k), xv, c8);
        c9  = _mm256_fmadd_ps(_mm256_broadcast_ss(w9  + k), xv, c9);
        c10 = _mm256_fmadd_ps(_mm256_broadcast_ss(w10 + k), xv, c10);
        c11 = _mm256_fmadd_ps(_mm256_broadcast_ss(w11 + k), xv, c11);
    }

    _mm256_store_ps(acc_buf +  0, c0);  _mm256_store_ps(acc_buf +  8, c1);
    _mm256_store_ps(acc_buf + 16, c2);  _mm256_store_ps(acc_buf + 24, c3);
    _mm256_store_ps(acc_buf + 32, c4);  _mm256_store_ps(acc_buf + 40, c5);
    _mm256_store_ps(acc_buf + 48, c6);  _mm256_store_ps(acc_buf + 56, c7);
    _mm256_store_ps(acc_buf + 64, c8);  _mm256_store_ps(acc_buf + 72, c9);
    _mm256_store_ps(acc_buf + 80, c10); _mm256_store_ps(acc_buf + 88, c11);
}

// Compute Y[0..T, r0..r1) = X_packed_8 * W[r0..r1, 0..K]^T for 1 <= T <= 8 (MoE small-medium buckets!)
inline void gemm_fma_rows_range_8(
    const CpuTensor & W,
    const float * xp, // [K][8]
    float * y,        // [T, M]
    int T, int r0, int r1) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    const auto tr = dtype_traits(W.dtype);
    const size_t row_bytes = (static_cast<size_t>(K) / tr.block_elems) * tr.block_bytes;
    const auto * w_base = static_cast<const uint8_t *>(W.ptr);

    constexpr int KC = 256;
    alignas(32) float wp[12 * KC];
    alignas(32) float acc[12 * 8];

    int r = r0;
    for (; r + 11 < r1; r += 12) {
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = std::min(KC, K - k0);
            for (int ri = 0; ri < 12; ++ri) {
                dequant_slice_to_f32(W.dtype, w_base + static_cast<size_t>(r + ri) * row_bytes, k0, kc, wp + ri * kc);
            }
            fma_kernel_12x8_contig(wp, xp + static_cast<size_t>(k0) * 8, kc, acc, k0 == 0);
        }
        for (int j = 0; j < T; ++j) {
            float * yt = y + static_cast<size_t>(j) * M + r;
            for (int ri = 0; ri < 12; ++ri) yt[ri] = acc[ri * 8 + j];
        }
    }
    for (; r < r1; ++r) {
        __m256 c0 = _mm256_setzero_ps();
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = std::min(KC, K - k0);
            dequant_slice_to_f32(W.dtype, w_base + static_cast<size_t>(r) * row_bytes, k0, kc, wp);
            const float * xk = xp + static_cast<size_t>(k0) * 8;
            for (int k = 0; k < kc; ++k) {
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(wp + k), _mm256_loadu_ps(xk + k * 8), c0);
            }
        }
        alignas(32) float row_out[8];
        _mm256_store_ps(row_out, c0);
        for (int j = 0; j < T; ++j) y[static_cast<size_t>(j) * M + r] = row_out[j];
    }
}

// Compact zero-spill Q6_K 1-row x 4-token direct kernel for small MoE buckets (nt = 4..11)!
inline void gemm_q6_k_1x4_direct(
    const BlockQ6_K * w_row,
    const CpuActQ8_K * x0, const CpuActQ8_K * x1, const CpuActQ8_K * x2, const CpuActQ8_K * x3,
    int nb,
    float out[4]) {

    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    // Keep the 4 float accumulators in an aligned L1d buffer touched ONCE per block so the inner h-loop has 0 spills!
    alignas(32) float acc_blk[4][8] = {};

    for (int b = 0; b < nb; ++b) {
        const char * pf = reinterpret_cast<const char *>(w_row + b + 2);
        _mm_prefetch(pf +   0, _MM_HINT_T0);
        _mm_prefetch(pf + 128, _MM_HINT_T0);

        const BlockQ6_K & wb = w_row[b];
        const float wd = f16_to_f32_fast(wb.d);
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.scales)));

        __m256i s0 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, _mm256_load_si256(reinterpret_cast<const __m256i *>(x0[b].bsums))), 5));
        __m256i s1 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, _mm256_load_si256(reinterpret_cast<const __m256i *>(x1[b].bsums))), 5));
        __m256i s2 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, _mm256_load_si256(reinterpret_cast<const __m256i *>(x2[b].bsums))), 5));
        __m256i s3 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, _mm256_load_si256(reinterpret_cast<const __m256i *>(x3[b].bsums))), 5));

        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = wb.ql + h * 64;
            const uint8_t * qh = wb.qh + h * 32;
            const int off = h * 128;
            const __m256i sc_h = (h == 0) ? _mm256_permute2x128_si256(sc16, sc16, 0x00)
                                          : _mm256_permute2x128_si256(sc16, sc16, 0x11);
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

            auto step_q = [&](__m256i qv, int qoff, int sh_idx) {
                const __m256i scv = _mm256_shuffle_epi8(sc_h, _mm256_load_si256(reinterpret_cast<const __m256i *>(kQ6ScShuf[sh_idx])));
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x0[b].qs + off + qoff))), scv));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x1[b].qs + off + qoff))), scv));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x2[b].qs + off + qoff))), scv));
                s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x3[b].qs + off + qoff))), scv));
#if defined(__GNUC__) && !defined(__clang__)
                // Keep s0..s3 accumulated eagerly in-register instead of deferring 32 vpaddd temps to stack!
                asm volatile("" : "+x"(s0), "+x"(s1), "+x"(s2), "+x"(s3));
#endif
            };

            // Process ql0 (q1, q3) before loading ql1 (q2, q4) to keep peak live YMMs <= 12 (0 spills!)
            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
            step_q(_mm256_or_si256(_mm256_and_si256(ql0, m4),                    _mm256_and_si256(_mm256_slli_epi16(qhv, 4), m30)),  0, 0);
            step_q(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_and_si256(qhv, m30)),                    64, 2);

            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
            step_q(_mm256_or_si256(_mm256_and_si256(ql1, m4),                    _mm256_and_si256(_mm256_slli_epi16(qhv, 2), m30)), 32, 1);
            step_q(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_and_si256(_mm256_srli_epi16(qhv, 2), m30)), 96, 3);
        }

        _mm256_store_ps(acc_blk[0], _mm256_fmadd_ps(_mm256_set1_ps(wd * x0[b].d), _mm256_cvtepi32_ps(s0), _mm256_load_ps(acc_blk[0])));
        _mm256_store_ps(acc_blk[1], _mm256_fmadd_ps(_mm256_set1_ps(wd * x1[b].d), _mm256_cvtepi32_ps(s1), _mm256_load_ps(acc_blk[1])));
        _mm256_store_ps(acc_blk[2], _mm256_fmadd_ps(_mm256_set1_ps(wd * x2[b].d), _mm256_cvtepi32_ps(s2), _mm256_load_ps(acc_blk[2])));
        _mm256_store_ps(acc_blk[3], _mm256_fmadd_ps(_mm256_set1_ps(wd * x3[b].d), _mm256_cvtepi32_ps(s3), _mm256_load_ps(acc_blk[3])));
    }
    out[0] = hsum256_ps(_mm256_load_ps(acc_blk[0]));
    out[1] = hsum256_ps(_mm256_load_ps(acc_blk[1]));
    out[2] = hsum256_ps(_mm256_load_ps(acc_blk[2]));
    out[3] = hsum256_ps(_mm256_load_ps(acc_blk[3]));
}

inline void gemm_q8_0_2x4_direct(
    const BlockQ8_0 * w0, const BlockQ8_0 * w1,
    const CpuActQ8_0 * x0, const CpuActQ8_0 * x1, const CpuActQ8_0 * x2, const CpuActQ8_0 * x3,
    int nb,
    float out[2][4]) {

    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0[b].qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w1[b].qs));
        const __m256i aw0 = _mm256_abs_epi8(q0);
        const __m256i aw1 = _mm256_abs_epi8(q1);
        const float wd0 = f16_to_f32_fast(w0[b].d);
        const float wd1 = f16_to_f32_fast(w1[b].d);

        const CpuActQ8_0 * xs[4] = {x0 + b, x1 + b, x2 + b, x3 + b};
        __m256 * cs0[4] = {&c00, &c01, &c02, &c03};
        __m256 * cs1[4] = {&c10, &c11, &c12, &c13};
        for (int ti = 0; ti < 4; ++ti) {
            const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xs[ti]->qs));
            const float xd = xs[ti]->d;
            const __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, q0)), ones16);
            const __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, q1)), ones16);
            *cs0[ti] = _mm256_fmadd_ps(_mm256_set1_ps(wd0 * xd), _mm256_cvtepi32_ps(d0), *cs0[ti]);
            *cs1[ti] = _mm256_fmadd_ps(_mm256_set1_ps(wd1 * xd), _mm256_cvtepi32_ps(d1), *cs1[ti]);
        }
    }
    out[0][0] = hsum256_ps(c00); out[0][1] = hsum256_ps(c01); out[0][2] = hsum256_ps(c02); out[0][3] = hsum256_ps(c03);
    out[1][0] = hsum256_ps(c10); out[1][1] = hsum256_ps(c11); out[1][2] = hsum256_ps(c12); out[1][3] = hsum256_ps(c13);
}
#endif // AVX2 && FMA && !AVX512VNNI

// ============================================================================
// Direct Quantized Dot Products (Memory-Bandwidth-Optimal for Decode T=1..4)
// ============================================================================

inline float dot_q8_0_q8_0_row(const BlockQ8_0 * w_row, const CpuActQ8_0 * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w_row[b + 0].qs));
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 0].qs));
        const __m256i d0 = mul_sum_i8_pairs_to_i32(w0, x0, ones16);
        const __m256  s0 = _mm256_set1_ps(f16_to_f32_fast(w_row[b + 0].d) * x_row[b + 0].d);
        acc0 = _mm256_fmadd_ps(s0, _mm256_cvtepi32_ps(d0), acc0);

        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w_row[b + 1].qs));
        const __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 1].qs));
        const __m256i d1 = mul_sum_i8_pairs_to_i32(w1, x1, ones16);
        const __m256  s1 = _mm256_set1_ps(f16_to_f32_fast(w_row[b + 1].d) * x_row[b + 1].d);
        acc1 = _mm256_fmadd_ps(s1, _mm256_cvtepi32_ps(d1), acc1);
    }
    for (; b < nb; ++b) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w_row[b].qs));
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const __m256i d0 = mul_sum_i8_pairs_to_i32(w0, x0, ones16);
        const __m256  s0 = _mm256_set1_ps(f16_to_f32_fast(w_row[b].d) * x_row[b].d);
        acc0 = _mm256_fmadd_ps(s0, _mm256_cvtepi32_ps(d0), acc0);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        if (b + 12 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 12]);
            __builtin_prefetch(pf +  0, 0, 0);
            __builtin_prefetch(pf + 64, 0, 0);
        }
        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_row[b + 0].qs), vld1q_s8(x_row[b + 0].qs)),
                                 vld1q_s8(w_row[b + 0].qs + 16), vld1q_s8(x_row[b + 0].qs + 16));
        int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_row[b + 1].qs), vld1q_s8(x_row[b + 1].qs)),
                                 vld1q_s8(w_row[b + 1].qs + 16), vld1q_s8(x_row[b + 1].qs + 16));
        int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_row[b + 2].qs), vld1q_s8(x_row[b + 2].qs)),
                                 vld1q_s8(w_row[b + 2].qs + 16), vld1q_s8(x_row[b + 2].qs + 16));
        int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_row[b + 3].qs), vld1q_s8(x_row[b + 3].qs)),
                                 vld1q_s8(w_row[b + 3].qs + 16), vld1q_s8(x_row[b + 3].qs + 16));
        const float32x4_t fsum = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)));
        const float32x4_t wsc  = load_4_fp16_scales(w_row[b + 0].d, w_row[b + 1].d, w_row[b + 2].d, w_row[b + 3].d);
        const float xd4[4]     = {x_row[b + 0].d, x_row[b + 1].d, x_row[b + 2].d, x_row[b + 3].d};
        acc = vfmaq_f32(acc, fsum, vmulq_f32(wsc, vld1q_f32(xd4)));
    }
    float sum = vaddvq_f32(acc);
    for (; b < nb; ++b) {
        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_row[b].qs), vld1q_s8(x_row[b].qs)),
                                 vld1q_s8(w_row[b].qs + 16), vld1q_s8(x_row[b].qs + 16));
        sum = std::fma(f16_to_f32_fast(w_row[b].d) * x_row[b].d, static_cast<float>(vaddvq_s32(p0)), sum);
    }
    return sum;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) s += static_cast<int32_t>(w_row[b].qs[i]) * static_cast<int32_t>(x_row[b].qs[i]);
        acc += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(s);
    }
    return acc;
#endif
}

inline void dot_q8_0_q8_0_4rows(
    const BlockQ8_0 * w0, const BlockQ8_0 * w1, const BlockQ8_0 * w2, const BlockQ8_0 * w3,
    const CpuActQ8_0 * x_row, int nb,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        if ((b & 1) == 0) {
            _mm_prefetch(reinterpret_cast<const char *>(w0 + b + 4), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w1 + b + 4), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w2 + b + 4), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w3 + b + 4), _MM_HINT_T0);
        }
        const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const __m128  vxd = _mm_set1_ps(x_row[b].d);
#if defined(__F16C__)
        const uint64_t d0123 =
              static_cast<uint64_t>(w0[b].d)
            | (static_cast<uint64_t>(w1[b].d) << 16)
            | (static_cast<uint64_t>(w2[b].d) << 32)
            | (static_cast<uint64_t>(w3[b].d) << 48);
        const __m128 wxd4 = _mm_mul_ps(_mm_cvtph_ps(_mm_cvtsi64_si128(static_cast<int64_t>(d0123))), vxd);
#else
        const __m128 wxd4 = _mm_mul_ps(_mm_set_ps(f16_to_f32_fast(w3[b].d), f16_to_f32_fast(w2[b].d),
                                                  f16_to_f32_fast(w1[b].d), f16_to_f32_fast(w0[b].d)), vxd);
#endif
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0[b].qs));
        a0 = _mm256_fmadd_ps(_mm256_broadcastss_ps(wxd4),
                             _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q0, xv, ones16)), a0);

        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w1[b].qs));
        a1 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(1, 1, 1, 1))),
                             _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q1, xv, ones16)), a1);

        const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w2[b].qs));
        a2 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(2, 2, 2, 2))),
                             _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q2, xv, ones16)), a2);

        const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w3[b].qs));
        a3 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(3, 3, 3, 3))),
                             _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q3, xv, ones16)), a3);
    }
    y0 = hsum256_ps(a0);
    y1 = hsum256_ps(a1);
    y2 = hsum256_ps(a2);
    y3 = hsum256_ps(a3);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Sequential 1-row streaming keeps 1 contiguous DRAM stream per core (eliminating 16-stream
    // LPDDR4X row-buffer thrashing while x_row stays 100% L1d-resident on Cortex-A76).
    y0 = dot_q8_0_q8_0_row(w0, x_row, nb);
    y1 = dot_q8_0_q8_0_row(w1, x_row, nb);
    y2 = dot_q8_0_q8_0_row(w2, x_row, nb);
    y3 = dot_q8_0_q8_0_row(w3, x_row, nb);
#else
    y0 = dot_q8_0_q8_0_row(w0, x_row, nb);
    y1 = dot_q8_0_q8_0_row(w1, x_row, nb);
    y2 = dot_q8_0_q8_0_row(w2, x_row, nb);
    y3 = dot_q8_0_q8_0_row(w3, x_row, nb);
#endif
}

inline float dot_q6_k_q8_k_row(const BlockQ6_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const char * pf = reinterpret_cast<const char *>(w_row + b + 2);
        _mm_prefetch(pf +   0, _MM_HINT_T0);
        _mm_prefetch(pf + 128, _MM_HINT_T0);

        const BlockQ6_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;

        const __m128i sc8  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.scales));
        const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = wb.ql + h * 64;
            const uint8_t * qh = wb.qh + h * 32;
            const int8_t *  qx = xb.qs + h * 128;
            const __m256i sc_h = (h == 0) ? _mm256_permute2x128_si256(sc16, sc16, 0x00)
                                          : _mm256_permute2x128_si256(sc16, sc16, 0x11);

            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

            // m30 trick eliminates 8 Port-0 shifts/block; vpshufb(sc_h, kQ6ScShuf) eliminates 30 Port-5 shuffles/block!
            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql0, m4),
                                               _mm256_and_si256(_mm256_slli_epi16(qhv, 4), m30));
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(
                _mm256_maddubs_epi16(q1, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 0))),
                _mm256_shuffle_epi8(sc_h, _mm256_load_si256(reinterpret_cast<const __m256i *>(kQ6ScShuf[0])))));

            const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                                               _mm256_and_si256(qhv, m30));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(
                _mm256_maddubs_epi16(q3, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 64))),
                _mm256_shuffle_epi8(sc_h, _mm256_load_si256(reinterpret_cast<const __m256i *>(kQ6ScShuf[2])))));

            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));

            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(ql1, m4),
                                               _mm256_and_si256(_mm256_slli_epi16(qhv, 2), m30));
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(
                _mm256_maddubs_epi16(q2, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 32))),
                _mm256_shuffle_epi8(sc_h, _mm256_load_si256(reinterpret_cast<const __m256i *>(kQ6ScShuf[1])))));

            const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4),
                                               _mm256_and_si256(_mm256_srli_epi16(qhv, 2), m30));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(
                _mm256_maddubs_epi16(q4, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 96))),
                _mm256_shuffle_epi8(sc_h, _mm256_load_si256(reinterpret_cast<const __m256i *>(kQ6ScShuf[3])))));
        }

        const __m256i bs16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.bsums));
        const __m256i sumi = _mm256_sub_epi32(_mm256_add_epi32(sumi0, sumi1),
                                              _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs16), 5));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t m30 = vdupq_n_u8(0x30u);

    auto do_block = [&](const BlockQ6_K & wb, const CpuActQ8_K & xb) -> int32_t {
        // Load all 16 sub-block scales once; subtract 32 * dot(scales, bsums) to eliminate 16x vsubq_s8
        const int8x16_t sc_all = vld1q_s8(wb.scales);
        const int16x8_t sc_lo  = vmovl_s8(vget_low_s8(sc_all));
        const int16x8_t sc_hi  = vmovl_s8(vget_high_s8(sc_all));
        const int16x8_t bs_lo  = vld1q_s16(xb.bsums + 0);
        const int16x8_t bs_hi  = vld1q_s16(xb.bsums + 8);
        int32x4_t bsum_v = vmull_s16(vget_low_s16(sc_lo), vget_low_s16(bs_lo));
        bsum_v = vmlal_s16(bsum_v, vget_high_s16(sc_lo), vget_high_s16(bs_lo));
        bsum_v = vmlal_s16(bsum_v, vget_low_s16(sc_hi),  vget_low_s16(bs_hi));
        bsum_v = vmlal_s16(bsum_v, vget_high_s16(sc_hi), vget_high_s16(bs_hi));

        auto do_half = [&](const uint8_t * ql, const uint8_t * qh, const int8_t * xq, int16x8_t sc16) -> int32x4_t {
            const uint8x16_t ql0 = vld1q_u8(ql +  0);
            const uint8x16_t ql1 = vld1q_u8(ql + 16);
            const uint8x16_t ql2 = vld1q_u8(ql + 32);
            const uint8x16_t ql3 = vld1q_u8(ql + 48);
            const uint8x16_t qh0 = vld1q_u8(qh +  0);
            const uint8x16_t qh1 = vld1q_u8(qh + 16);

            // vbslq_u8(0x30, shifted_qh, masked_ql) cuts Pipe-V0 shifts from 36 -> 20 per block
            int8x16_t w0 = vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh0, 4), vandq_u8(ql0, m4)));
            int8x16_t w1 = vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh1, 4), vandq_u8(ql1, m4)));
            int8x16_t w2 = vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh0, 2), vandq_u8(ql2, m4)));
            int8x16_t w3 = vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh1, 2), vandq_u8(ql3, m4)));

            int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xq +  0));
            int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(xq + 16));
            int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), w2, vld1q_s8(xq + 32));
            int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), w3, vld1q_s8(xq + 48));
            const int32x4_t d0 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            w0 = vreinterpretq_s8_u8(vbslq_u8(m30, qh0,                  vshrq_n_u8(ql0, 4)));
            w1 = vreinterpretq_s8_u8(vbslq_u8(m30, qh1,                  vshrq_n_u8(ql1, 4)));
            w2 = vreinterpretq_s8_u8(vbslq_u8(m30, vshrq_n_u8(qh0, 2),   vshrq_n_u8(ql2, 4)));
            w3 = vreinterpretq_s8_u8(vbslq_u8(m30, vshrq_n_u8(qh1, 2),   vshrq_n_u8(ql3, 4)));

            p0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xq +  64));
            p1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(xq +  80));
            p2 = vdotq_s32(vdupq_n_s32(0), w2, vld1q_s8(xq +  96));
            p3 = vdotq_s32(vdupq_n_s32(0), w3, vld1q_s8(xq + 112));
            const int32x4_t d1 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            return vmlaq_s32(vmulq_s32(d0, vmovl_s16(vget_low_s16(sc16))),
                             d1, vmovl_s16(vget_high_s16(sc16)));
        };

        int32x4_t acc_v = vsubq_s32(
            vaddq_s32(do_half(wb.ql,      wb.qh,      xb.qs,       sc_lo),
                      do_half(wb.ql + 64, wb.qh + 32, xb.qs + 128, sc_hi)),
            vshlq_n_s32(bsum_v, 5));
        return vaddvq_s32(acc_v);
    };

    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * p = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(p +   0, 0, 0);
            __builtin_prefetch(p +  64, 0, 0);
            __builtin_prefetch(p + 128, 0, 0);
            __builtin_prefetch(p + 192, 0, 0);
        }
        const int32_t sumi = do_block(w_row[b], x_row[b]);
        sumf = std::fma(static_cast<float>(sumi) * x_row[b].d, f16_to_f32_fast(w_row[b].d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockQ6_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        int32_t sumi = 0;
        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = wb.ql + h * 64;
            const uint8_t * qh = wb.qh + h * 32;
            const int8_t *  sc = wb.scales + h * 8;
            const int8_t *  qx = xb.qs + h * 128;
            for (int l = 0; l < 32; ++l) {
                const int is = l >> 4;
                const int q1 = static_cast<int>((ql[l] & 0xFu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0xFu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                sumi += sc[is + 0] * q1 * qx[l + 0]
                      + sc[is + 2] * q2 * qx[l + 32]
                      + sc[is + 4] * q3 * qx[l + 64]
                      + sc[is + 6] * q4 * qx[l + 96];
            }
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

inline float dot_f32_row(const float * w, const float * x, int K) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < K; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i),     _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
    }
    for (; i + 7 < K; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a0);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    for (; i < K; ++i) acc += w[i] * x[i];
    return acc;
#elif defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= K; i += 16) {
        if (i + 64 < K) __builtin_prefetch(w + i + 64, 0, 0);
        a0 = vfmaq_f32(a0, vld1q_f32(w + i +  0), vld1q_f32(x + i +  0));
        a1 = vfmaq_f32(a1, vld1q_f32(w + i +  4), vld1q_f32(x + i +  4));
        a2 = vfmaq_f32(a2, vld1q_f32(w + i +  8), vld1q_f32(x + i +  8));
        a3 = vfmaq_f32(a3, vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
    }
    float acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < K; ++i) acc += w[i] * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += w[i] * x[i];
    return acc;
#endif
}

inline void dot_f32_4rows(
    const float * w0, const float * w1, const float * w2, const float * w3,
    const float * x, int K,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < K; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i), xv, a3);
    }
    y0 = hsum256_ps(a0); y1 = hsum256_ps(a1); y2 = hsum256_ps(a2); y3 = hsum256_ps(a3);
    for (; i < K; ++i) {
        const float xv = x[i];
        y0 += w0[i] * xv; y1 += w1[i] * xv; y2 += w2[i] * xv; y3 += w3[i] * xv;
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    // Single sequential stream per core saturates ARM64 HW prefetcher & LPDDR4X bandwidth
    y0 = dot_f32_row(w0, x, K); y1 = dot_f32_row(w1, x, K);
    y2 = dot_f32_row(w2, x, K); y3 = dot_f32_row(w3, x, K);
#else
    y0 = dot_f32_row(w0, x, K); y1 = dot_f32_row(w1, x, K);
    y2 = dot_f32_row(w2, x, K); y3 = dot_f32_row(w3, x, K);
#endif
}

inline float dot_bf16_row(const uint16_t * w, const float * x, int K) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < K; i += 16) {
        const __m256 wv0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i))), 16));
        const __m256 wv1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i + 8))), 16));
        a0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(x + i + 8), a1);
    }
    for (; i + 7 < K; i += 8) {
        const __m256 wv0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i))), 16));
        a0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(x + i), a0);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    for (; i < K; ++i) acc += bf16_to_f32_fast(w[i]) * x[i];
    return acc;
#elif defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= K; i += 8) {
        const uint16x8_t wu = vld1q_u16(w + i);
        const float32x4_t w0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(wu)), 16));
        const float32x4_t w1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(wu)), 16));
        a0 = vfmaq_f32(a0, w0, vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, w1, vld1q_f32(x + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < K; ++i) acc += bf16_to_f32_fast(w[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += bf16_to_f32_fast(w[i]) * x[i];
    return acc;
#endif
}

inline void dot_bf16_4rows(
    const uint16_t * w0, const uint16_t * w1, const uint16_t * w2, const uint16_t * w3,
    const float * x, int K,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps();
    __m256 b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps();
    __m256 b3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < K; i += 16) {
        _mm_prefetch(reinterpret_cast<const char *>(w0 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(w1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(w2 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(w3 + i + 64), _MM_HINT_T0);
        const __m256 xv0 = _mm256_loadu_ps(x + i);
        const __m256 xv1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w0 + i))), 16)), xv0, a0);
        a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w1 + i))), 16)), xv0, a1);
        a2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w2 + i))), 16)), xv0, a2);
        a3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w3 + i))), 16)), xv0, a3);
        b0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w0 + i + 8))), 16)), xv1, b0);
        b1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w1 + i + 8))), 16)), xv1, b1);
        b2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w2 + i + 8))), 16)), xv1, b2);
        b3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w3 + i + 8))), 16)), xv1, b3);
    }
    a0 = _mm256_add_ps(a0, b0); a1 = _mm256_add_ps(a1, b1);
    a2 = _mm256_add_ps(a2, b2); a3 = _mm256_add_ps(a3, b3);
    for (; i + 7 < K; i += 8) {
        const __m256 xv  = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w0 + i))), 16)), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w1 + i))), 16)), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w2 + i))), 16)), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w3 + i))), 16)), xv, a3);
    }
    y0 = hsum256_ps(a0); y1 = hsum256_ps(a1); y2 = hsum256_ps(a2); y3 = hsum256_ps(a3);
    for (; i < K; ++i) {
        const float xv = x[i];
        y0 += bf16_to_f32_fast(w0[i]) * xv; y1 += bf16_to_f32_fast(w1[i]) * xv;
        y2 += bf16_to_f32_fast(w2[i]) * xv; y3 += bf16_to_f32_fast(w3[i]) * xv;
    }
#else
    y0 = dot_bf16_row(w0, x, K); y1 = dot_bf16_row(w1, x, K);
    y2 = dot_bf16_row(w2, x, K); y3 = dot_bf16_row(w3, x, K);
#endif
}

inline float dot_f16_row(const uint16_t * w, const float * x, int K) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < K; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i))),     _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i + 8))), _mm256_loadu_ps(x + i + 8), a1);
    }
    for (; i + 7 < K; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i))), _mm256_loadu_ps(x + i), a0);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    for (; i < K; ++i) acc += f16_to_f32_fast(w[i]) * x[i];
    return acc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= K; i += 8) {
        const float16x8_t wh = vld1q_f16(reinterpret_cast<const __fp16 *>(w + i));
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(wh)),  vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(wh)), vld1q_f32(x + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < K; ++i) acc += f16_to_f32_fast(w[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += f16_to_f32_fast(w[i]) * x[i];
    return acc;
#endif
}

inline void dot_f16_4rows(
    const uint16_t * w0, const uint16_t * w1, const uint16_t * w2, const uint16_t * w3,
    const float * x, int K,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < K; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w0 + i))), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w1 + i))), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w2 + i))), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w3 + i))), xv, a3);
    }
    y0 = hsum256_ps(a0); y1 = hsum256_ps(a1); y2 = hsum256_ps(a2); y3 = hsum256_ps(a3);
    for (; i < K; ++i) {
        const float xv = x[i];
        y0 += f16_to_f32_fast(w0[i]) * xv; y1 += f16_to_f32_fast(w1[i]) * xv;
        y2 += f16_to_f32_fast(w2[i]) * xv; y3 += f16_to_f32_fast(w3[i]) * xv;
    }
#else
    y0 = dot_f16_row(w0, x, K); y1 = dot_f16_row(w1, x, K);
    y2 = dot_f16_row(w2, x, K); y3 = dot_f16_row(w3, x, K);
#endif
}

inline float dot_q4_0_q8_0_row(const BlockQ4_0 * w_row, const CpuActQ8_0 * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float  sacc = 0.0f;
    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b + 0].qs));
        const __m256i q0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4);
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 0].qs));
        const float wd0  = f16_to_f32_fast(w_row[b + 0].d);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(wd0 * x_row[b + 0].d),
                               _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q0, x0), ones16)), acc0);
        sacc += wd0 * x_row[b + 0].s;

        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b + 1].qs));
        const __m256i q1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 4), 1), m4);
        const __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 1].qs));
        const float wd1  = f16_to_f32_fast(w_row[b + 1].d);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(wd1 * x_row[b + 1].d),
                               _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q1, x1), ones16)), acc1);
        sacc += wd1 * x_row[b + 1].s;
    }
    for (; b < nb; ++b) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b].qs));
        const __m256i q0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4);
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const float wd0  = f16_to_f32_fast(w_row[b].d);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(wd0 * x_row[b].d),
                               _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q0, x0), ones16)), acc0);
        sacc += wd0 * x_row[b].s;
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1)) - 8.0f * sacc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    const int8x16_t  b8 = vdupq_n_s8(8);
    for (int b = 0; b < nb; ++b) {
        const uint8x16_t qs = vld1q_u8(w_row[b].qs);
        const int8x16_t  w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, m4)), b8);
        const int8x16_t  w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), b8);
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x_row[b].qs)), w1, vld1q_s8(x_row[b].qs + 16));
        sumf += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(vaddvq_s32(p));
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t s = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = w_row[b].qs[j];
            s += (static_cast<int32_t>(v & 0xFu) - 8) * static_cast<int32_t>(x_row[b].qs[j]);
            s += (static_cast<int32_t>(v >> 4) - 8)   * static_cast<int32_t>(x_row[b].qs[j + 16]);
        }
        acc += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(s);
    }
    return acc;
#endif
}

inline void dot_q4_0_q8_0_4rows(
    const BlockQ4_0 * w0, const BlockQ4_0 * w1, const BlockQ4_0 * w2, const BlockQ4_0 * w3,
    const CpuActQ8_0 * x_row, int nb,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m128 s4 = _mm_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        if ((b & 1) == 0) {
            _mm_prefetch(reinterpret_cast<const char *>(w0 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w1 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w2 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w3 + b + 8), _MM_HINT_T0);
        }
        const __m256i xv  = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
#if defined(__F16C__)
        const uint64_t d0123 =
              static_cast<uint64_t>(w0[b].d)
            | (static_cast<uint64_t>(w1[b].d) << 16)
            | (static_cast<uint64_t>(w2[b].d) << 32)
            | (static_cast<uint64_t>(w3[b].d) << 48);
        const __m128 wd4 = _mm_cvtph_ps(_mm_cvtsi64_si128(static_cast<int64_t>(d0123)));
#else
        const __m128 wd4 = _mm_set_ps(f16_to_f32_fast(w3[b].d), f16_to_f32_fast(w2[b].d),
                                      f16_to_f32_fast(w1[b].d), f16_to_f32_fast(w0[b].d));
#endif
        s4 = _mm_fmadd_ps(wd4, _mm_set1_ps(x_row[b].s), s4);
        const __m128 wxd4 = _mm_mul_ps(wd4, _mm_set1_ps(x_row[b].d));

        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w0[b].qs));
        const __m256i q0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4);
        a0 = _mm256_fmadd_ps(_mm256_broadcastss_ps(wxd4),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q0, xv), ones16)), a0);

        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w1[b].qs));
        const __m256i q1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 4), 1), m4);
        a1 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(1, 1, 1, 1))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q1, xv), ones16)), a1);

        const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w2[b].qs));
        const __m256i q2 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r2), _mm_srli_epi16(r2, 4), 1), m4);
        a2 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(2, 2, 2, 2))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q2, xv), ones16)), a2);

        const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w3[b].qs));
        const __m256i q3 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r3), _mm_srli_epi16(r3, 4), 1), m4);
        a3 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(3, 3, 3, 3))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q3, xv), ones16)), a3);
    }
    alignas(16) float sv[4];
    _mm_store_ps(sv, s4);
    y0 = hsum256_ps(a0) - 8.0f * sv[0];
    y1 = hsum256_ps(a1) - 8.0f * sv[1];
    y2 = hsum256_ps(a2) - 8.0f * sv[2];
    y3 = hsum256_ps(a3) - 8.0f * sv[3];
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    float32x4_t acc  = vdupq_n_f32(0.0f);
    float32x4_t sacc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&w0[b + 8], 0, 0);
            __builtin_prefetch(&w1[b + 8], 0, 0);
            __builtin_prefetch(&w2[b + 8], 0, 0);
            __builtin_prefetch(&w3[b + 8], 0, 0);
        }
        const int8x16_t xlo = vld1q_s8(x_row[b].qs);
        const int8x16_t xhi = vld1q_s8(x_row[b].qs + 16);
        const float32x4_t wd4 = load_4_fp16_scales(w0[b].d, w1[b].d, w2[b].d, w3[b].d);
        sacc = vfmaq_n_f32(sacc, wd4, x_row[b].s);

        auto dot_row = [&](const BlockQ4_0 & wb) -> int32x4_t {
            const uint8x16_t qs = vld1q_u8(wb.qs);
            return vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(qs, m4)), xlo),
                             vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), xhi);
        };
        const int32x4_t p0 = dot_row(w0[b]);
        const int32x4_t p1 = dot_row(w1[b]);
        const int32x4_t p2 = dot_row(w2[b]);
        const int32x4_t p3 = dot_row(w3[b]);
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(sums), vmulq_n_f32(wd4, x_row[b].d));
    }
    const float32x4_t res = vfmsq_n_f32(acc, sacc, 8.0f);
    y0 = vgetq_lane_f32(res, 0); y1 = vgetq_lane_f32(res, 1);
    y2 = vgetq_lane_f32(res, 2); y3 = vgetq_lane_f32(res, 3);
#else
    y0 = dot_q4_0_q8_0_row(w0, x_row, nb); y1 = dot_q4_0_q8_0_row(w1, x_row, nb);
    y2 = dot_q4_0_q8_0_row(w2, x_row, nb); y3 = dot_q4_0_q8_0_row(w3, x_row, nb);
#endif
}

inline float dot_q5_0_q8_0_row(const BlockQ5_0 * w_row, const CpuActQ8_0 * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc  = _mm256_setzero_ps();
    float  sacc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        uint32_t qh = 0;
        std::memcpy(&qh, w_row[b].qh, 4);
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b].qs));
        const __m256i nib = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
        const __m256i q   = _mm256_or_si256(nib, unpack_bits32_to_0x10_avx2(qh));
        const __m256i xv  = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const float   wd  = f16_to_f32_fast(w_row[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(wd * x_row[b].d),
                              _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones16)), acc);
        sacc += wd * x_row[b].s;
    }
    return hsum256_ps(acc) - 16.0f * sacc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const int8x16_t  b16 = vdupq_n_s8(16);
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    for (int b = 0; b < nb; ++b) {
        const uint8x16_t qs  = vld1q_u8(w_row[b].qs);
        const uint8x16_t h01 = vcombine_u8(vdup_n_u8(w_row[b].qh[0]), vdup_n_u8(w_row[b].qh[1]));
        const uint8x16_t h23 = vcombine_u8(vdup_n_u8(w_row[b].qh[2]), vdup_n_u8(w_row[b].qh[3]));
        const uint8x16_t hb0 = vandq_u8(vtstq_u8(h01, vmask), vdupq_n_u8(16));
        const uint8x16_t hb1 = vandq_u8(vtstq_u8(h23, vmask), vdupq_n_u8(16));
        const int8x16_t  w0  = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m4), hb0)), b16);
        const int8x16_t  w1  = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), hb1)), b16);
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x_row[b].qs)), w1, vld1q_s8(x_row[b].qs + 16));
        sumf += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(vaddvq_s32(p));
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        uint32_t qh = 0;
        std::memcpy(&qh, w_row[b].qh, 4);
        int32_t s = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = w_row[b].qs[j];
            const uint8_t xh0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
            const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10u);
            s += (static_cast<int32_t>((v & 0x0Fu) | xh0) - 16) * static_cast<int32_t>(x_row[b].qs[j]);
            s += (static_cast<int32_t>((v >> 4)   | xh1) - 16) * static_cast<int32_t>(x_row[b].qs[j + 16]);
        }
        acc += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(s);
    }
    return acc;
#endif
}

inline void dot_q5_0_q8_0_4rows(
    const BlockQ5_0 * w0, const BlockQ5_0 * w1, const BlockQ5_0 * w2, const BlockQ5_0 * w3,
    const CpuActQ8_0 * x_row, int nb,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m128 s4 = _mm_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const float   xd = x_row[b].d;
        const float  wd0 = f16_to_f32_fast(w0[b].d);
        const float  wd1 = f16_to_f32_fast(w1[b].d);
        const float  wd2 = f16_to_f32_fast(w2[b].d);
        const float  wd3 = f16_to_f32_fast(w3[b].d);
        s4 = _mm_fmadd_ps(_mm_set_ps(wd3, wd2, wd1, wd0), _mm_set1_ps(x_row[b].s), s4);
        auto step_row = [&](const BlockQ5_0 & wb, float wd, __m256 & acc) {
            uint32_t qh = 0;
            std::memcpy(&qh, wb.qh, 4);
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.qs));
            const __m256i nib = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
            const __m256i q   = _mm256_or_si256(nib, unpack_bits32_to_0x10_avx2(qh));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(wd * xd),
                                  _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones16)), acc);
        };
        step_row(w0[b], wd0, a0); step_row(w1[b], wd1, a1); step_row(w2[b], wd2, a2); step_row(w3[b], wd3, a3);
    }
    alignas(16) float sv[4];
    _mm_store_ps(sv, s4);
    y0 = hsum256_ps(a0) - 16.0f * sv[0]; y1 = hsum256_ps(a1) - 16.0f * sv[1];
    y2 = hsum256_ps(a2) - 16.0f * sv[2]; y3 = hsum256_ps(a3) - 16.0f * sv[3];
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t v16 = vdupq_n_u8(16);
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float32x4_t acc  = vdupq_n_f32(0.0f);
    float32x4_t sacc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&w0[b + 8], 0, 0);
            __builtin_prefetch(&w1[b + 8], 0, 0);
            __builtin_prefetch(&w2[b + 8], 0, 0);
            __builtin_prefetch(&w3[b + 8], 0, 0);
        }
        const int8x16_t xlo = vld1q_s8(x_row[b].qs);
        const int8x16_t xhi = vld1q_s8(x_row[b].qs + 16);
        const float32x4_t wd4 = load_4_fp16_scales(w0[b].d, w1[b].d, w2[b].d, w3[b].d);
        sacc = vfmaq_n_f32(sacc, wd4, x_row[b].s);

        auto dot_row = [&](const BlockQ5_0 & wb) -> int32x4_t {
            const uint8x16_t qs  = vld1q_u8(wb.qs);
            const uint8x16_t h01 = vcombine_u8(vdup_n_u8(wb.qh[0]), vdup_n_u8(wb.qh[1]));
            const uint8x16_t h23 = vcombine_u8(vdup_n_u8(wb.qh[2]), vdup_n_u8(wb.qh[3]));
            const uint8x16_t hb0 = vandq_u8(vtstq_u8(h01, vmask), v16);
            const uint8x16_t hb1 = vandq_u8(vtstq_u8(h23, vmask), v16);
            const int8x16_t  q0  = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m4), hb0));
            const int8x16_t  q1  = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), hb1));
            return vdotq_s32(vdotq_s32(vdupq_n_s32(0), q0, xlo), q1, xhi);
        };
        const int32x4_t p0 = dot_row(w0[b]);
        const int32x4_t p1 = dot_row(w1[b]);
        const int32x4_t p2 = dot_row(w2[b]);
        const int32x4_t p3 = dot_row(w3[b]);
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(sums), vmulq_n_f32(wd4, x_row[b].d));
    }
    const float32x4_t res = vfmsq_n_f32(acc, sacc, 16.0f);
    y0 = vgetq_lane_f32(res, 0); y1 = vgetq_lane_f32(res, 1);
    y2 = vgetq_lane_f32(res, 2); y3 = vgetq_lane_f32(res, 3);
#else
    y0 = dot_q5_0_q8_0_row(w0, x_row, nb); y1 = dot_q5_0_q8_0_row(w1, x_row, nb);
    y2 = dot_q5_0_q8_0_row(w2, x_row, nb); y3 = dot_q5_0_q8_0_row(w3, x_row, nb);
#endif
}

inline float dot_iq4_nl_q8_0_row(const BlockIQ4_NL * w_row, const CpuActQ8_0 * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i lut    = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl)));
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b + 0].qs));
        const __m256i q0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4));
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 0].qs));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w_row[b + 0].d) * x_row[b + 0].d),
                               _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q0, x0, ones16)), acc0);

        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b + 1].qs));
        const __m256i q1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 4), 1), m4));
        const __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b + 1].qs));
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w_row[b + 1].d) * x_row[b + 1].d),
                               _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q1, x1, ones16)), acc1);
    }
    for (; b < nb; ++b) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b].qs));
        const __m256i q0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4));
        const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w_row[b].d) * x_row[b].d),
                               _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(q0, x0, ones16)), acc0);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const int8x16_t  lut = vld1q_s8(kvalues_iq4nl);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) __builtin_prefetch(&w_row[b + 8], 0, 0);
        const uint8x16_t qs  = vld1q_u8(w_row[b].qs);
        const int8x16_t  qlo = vqtbl1q_s8(lut, vandq_u8(qs, m4));
        const int8x16_t  qhi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));
        const int32x4_t  p   = vdotq_s32(vdotq_s32(vdupq_n_s32(0), qlo, vld1q_s8(x_row[b].qs)), qhi, vld1q_s8(x_row[b].qs + 16));
        sumf = std::fma(static_cast<float>(vaddvq_s32(p)) * x_row[b].d, f16_to_f32_fast(w_row[b].d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t s = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = w_row[b].qs[j];
            s += static_cast<int32_t>(kvalues_iq4nl[v & 0xFu]) * static_cast<int32_t>(x_row[b].qs[j]);
            s += static_cast<int32_t>(kvalues_iq4nl[v >> 4])   * static_cast<int32_t>(x_row[b].qs[j + 16]);
        }
        acc += (f16_to_f32_fast(w_row[b].d) * x_row[b].d) * static_cast<float>(s);
    }
    return acc;
#endif
}

inline void dot_iq4_nl_q8_0_4rows(
    const BlockIQ4_NL * w0, const BlockIQ4_NL * w1, const BlockIQ4_NL * w2, const BlockIQ4_NL * w3,
    const CpuActQ8_0 * x_row, int nb,
    float & y0, float & y1, float & y2, float & y3) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i lut    = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl)));
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        if ((b & 1) == 0) {
            _mm_prefetch(reinterpret_cast<const char *>(w0 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w1 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w2 + b + 8), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(w3 + b + 8), _MM_HINT_T0);
        }
        const __m256i xv  = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const __m256i axv = _mm256_abs_epi8(xv); // hoisted once for all 4 rows (kvalues_iq4nl min is -127, safe!)
        const __m128  vxd = _mm_set1_ps(x_row[b].d);
#if defined(__F16C__)
        const uint64_t d0123 =
              static_cast<uint64_t>(w0[b].d)
            | (static_cast<uint64_t>(w1[b].d) << 16)
            | (static_cast<uint64_t>(w2[b].d) << 32)
            | (static_cast<uint64_t>(w3[b].d) << 48);
        const __m128 wxd4 = _mm_mul_ps(_mm_cvtph_ps(_mm_cvtsi64_si128(static_cast<int64_t>(d0123))), vxd);
#else
        const __m128 wxd4 = _mm_mul_ps(_mm_set_ps(f16_to_f32_fast(w3[b].d), f16_to_f32_fast(w2[b].d),
                                                  f16_to_f32_fast(w1[b].d), f16_to_f32_fast(w0[b].d)), vxd);
#endif
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w0[b].qs));
        const __m256i q0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 4), 1), m4));
        a0 = _mm256_fmadd_ps(_mm256_broadcastss_ps(wxd4),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axv, _mm256_sign_epi8(q0, xv)), ones16)), a0);

        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w1[b].qs));
        const __m256i q1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 4), 1), m4));
        a1 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(1, 1, 1, 1))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axv, _mm256_sign_epi8(q1, xv)), ones16)), a1);

        const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w2[b].qs));
        const __m256i q2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r2), _mm_srli_epi16(r2, 4), 1), m4));
        a2 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(2, 2, 2, 2))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axv, _mm256_sign_epi8(q2, xv)), ones16)), a2);

        const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w3[b].qs));
        const __m256i q3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r3), _mm_srli_epi16(r3, 4), 1), m4));
        a3 = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_permute_ps(wxd4, _MM_SHUFFLE(3, 3, 3, 3))),
                             _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axv, _mm256_sign_epi8(q3, xv)), ones16)), a3);
    }
    y0 = hsum256_ps(a0); y1 = hsum256_ps(a1); y2 = hsum256_ps(a2); y3 = hsum256_ps(a3);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const int8x16_t  lut = vld1q_s8(kvalues_iq4nl);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&w0[b + 8], 0, 0);
            __builtin_prefetch(&w1[b + 8], 0, 0);
            __builtin_prefetch(&w2[b + 8], 0, 0);
            __builtin_prefetch(&w3[b + 8], 0, 0);
        }
        const int8x16_t xlo = vld1q_s8(x_row[b].qs);
        const int8x16_t xhi = vld1q_s8(x_row[b].qs + 16);
        const float32x4_t wxd = vmulq_n_f32(load_4_fp16_scales(w0[b].d, w1[b].d, w2[b].d, w3[b].d), x_row[b].d);
        auto dot_row = [&](const BlockIQ4_NL & wb) -> int32x4_t {
            const uint8x16_t qs = vld1q_u8(wb.qs);
            return vdotq_s32(vdotq_s32(vdupq_n_s32(0), vqtbl1q_s8(lut, vandq_u8(qs, m4)), xlo),
                             vqtbl1q_s8(lut, vshrq_n_u8(qs, 4)), xhi);
        };
        const int32x4_t p0 = dot_row(w0[b]);
        const int32x4_t p1 = dot_row(w1[b]);
        const int32x4_t p2 = dot_row(w2[b]);
        const int32x4_t p3 = dot_row(w3[b]);
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(sums), wxd);
    }
    y0 = vgetq_lane_f32(acc, 0); y1 = vgetq_lane_f32(acc, 1);
    y2 = vgetq_lane_f32(acc, 2); y3 = vgetq_lane_f32(acc, 3);
#else
    y0 = dot_iq4_nl_q8_0_row(w0, x_row, nb); y1 = dot_iq4_nl_q8_0_row(w1, x_row, nb);
    y2 = dot_iq4_nl_q8_0_row(w2, x_row, nb); y3 = dot_iq4_nl_q8_0_row(w3, x_row, nb);
#endif
}

inline float dot_q4_k_q8_k_row(const BlockQ4_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float min_acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockQ4_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;
        __m256i sumi = _mm256_setzero_si256();
        int32_t sum_m = 0;
        const uint8_t * ql = wb.qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is + 0, wb.scales, sc0, m0);
            get_scale_min_k4(is + 1, wb.scales, sc1, m1);
            sum_m += static_cast<int32_t>(m0) * (xb.bsums[is * 2 + 0] + xb.bsums[is * 2 + 1])
                   + static_cast<int32_t>(m1) * (xb.bsums[is * 2 + 2] + xb.bsums[is * 2 + 3]);

            const __m256i qlv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
            const __m256i q0  = _mm256_and_si256(qlv, m4);
            const __m256i q1  = _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4);
            const __m256i x0  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j));
            const __m256i x1  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j + 32));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q0, x0), _mm256_set1_epi16(sc0)));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q1, x1), _mm256_set1_epi16(sc1)));
            ql += 32; is += 2;
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        min_acc -= dmin * static_cast<float>(sum_m);
    }
    return hsum256_ps(acc) + min_acc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    constexpr uint32_t kmask3 = 0x03030303u;

    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +   0, 0, 0);
            __builtin_prefetch(pf +  64, 0, 0);
            __builtin_prefetch(pf + 128, 0, 0);
        }

        // Branchless 32-bit unpack of 8x 6-bit scales & mins directly into NEON vectors
        uint32_t u0, u1, u2;
        std::memcpy(&u0, w_row[b].scales + 0, 4);
        std::memcpy(&u1, w_row[b].scales + 4, 4);
        std::memcpy(&u2, w_row[b].scales + 8, 4);
        const uint32_t sc_lo4 = u0 & kmask1;
        const uint32_t mn_lo4 = u1 & kmask1;
        const uint32_t sc_hi4 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
        const uint32_t mn_hi4 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);

        const uint16x8_t sc16 = vmovl_u8(vreinterpret_u8_u32(vset_lane_u32(sc_hi4, vdup_n_u32(sc_lo4), 1)));
        const uint16x8_t mn16 = vmovl_u8(vreinterpret_u8_u32(vset_lane_u32(mn_hi4, vdup_n_u32(mn_lo4), 1)));
        const int32x4_t sc_v0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc16)));
        const int32x4_t sc_v1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc16)));

        const int8_t * xq = x_row[b].qs;
        int32x4_t d0, d1;
        {
            const uint8x16_t q0 = vld1q_u8(w_row[b].qs +  0);
            const uint8x16_t q1 = vld1q_u8(w_row[b].qs + 16);
            int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q0, m4)), vld1q_s8(xq +  0)),
                                     vreinterpretq_s8_u8(vandq_u8(q1, m4)), vld1q_s8(xq + 16));
            int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), vld1q_s8(xq + 32)),
                                     vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), vld1q_s8(xq + 48));

            const uint8x16_t q2 = vld1q_u8(w_row[b].qs + 32);
            const uint8x16_t q3 = vld1q_u8(w_row[b].qs + 48);
            int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q2, m4)), vld1q_s8(xq + 64)),
                                     vreinterpretq_s8_u8(vandq_u8(q3, m4)), vld1q_s8(xq + 80));
            int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)), vld1q_s8(xq + 96)),
                                     vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)), vld1q_s8(xq + 112));
            d0 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        }
        {
            const uint8x16_t q0 = vld1q_u8(w_row[b].qs + 64);
            const uint8x16_t q1 = vld1q_u8(w_row[b].qs + 80);
            int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q0, m4)), vld1q_s8(xq + 128)),
                                     vreinterpretq_s8_u8(vandq_u8(q1, m4)), vld1q_s8(xq + 144));
            int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), vld1q_s8(xq + 160)),
                                     vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), vld1q_s8(xq + 176));

            const uint8x16_t q2 = vld1q_u8(w_row[b].qs + 96);
            const uint8x16_t q3 = vld1q_u8(w_row[b].qs + 112);
            int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q2, m4)), vld1q_s8(xq + 192)),
                                     vreinterpretq_s8_u8(vandq_u8(q3, m4)), vld1q_s8(xq + 208));
            int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)), vld1q_s8(xq + 224)),
                                     vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)), vld1q_s8(xq + 240));
            d1 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        }

        const int32_t sum_pos = vaddvq_s32(vmlaq_s32(vmulq_s32(d0, sc_v0), d1, sc_v1));
        const int32x4_t bs0 = vpaddlq_s16(vld1q_s16(x_row[b].bsums + 0));
        const int32x4_t bs1 = vpaddlq_s16(vld1q_s16(x_row[b].bsums + 8));
        const int32_t sum_min = vaddvq_s32(vmlaq_s32(
            vmulq_s32(bs0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mn16)))),
            bs1, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mn16)))));

        const float val = std::fma(-f16_to_f32_fast(w_row[b].dmin), static_cast<float>(sum_min),
                                   f16_to_f32_fast(w_row[b].d) * static_cast<float>(sum_pos));
        sumf = std::fma(val, x_row[b].d, sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockQ4_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;
        int32_t sumi = 0, sum_m = 0;
        const uint8_t * q = wb.qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is + 0, wb.scales, sc0, m0);
            get_scale_min_k4(is + 1, wb.scales, sc1, m1);
            sum_m += static_cast<int32_t>(m0) * (xb.bsums[is * 2 + 0] + xb.bsums[is * 2 + 1])
                   + static_cast<int32_t>(m1) * (xb.bsums[is * 2 + 2] + xb.bsums[is * 2 + 3]);
            int32_t s0 = 0, s1 = 0;
            for (int l = 0; l < 32; ++l) {
                s0 += static_cast<int32_t>(q[l] & 0xFu) * static_cast<int32_t>(xb.qs[j + l]);
                s1 += static_cast<int32_t>(q[l] >> 4)   * static_cast<int32_t>(xb.qs[j + 32 + l]);
            }
            sumi += sc0 * s0 + sc1 * s1;
            q += 32; is += 2;
        }
        acc += d * static_cast<float>(sumi) - dmin * static_cast<float>(sum_m);
    }
    return acc;
#endif
}

inline void dot_q4_k_q8_k_2rows(
    const BlockQ4_K * w0, const BlockQ4_K * w1,
    const CpuActQ8_K * x_row, int nb,
    float & y0, float & y1) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float min0 = 0.0f, min1 = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const CpuActQ8_K & xb = x_row[b];
        __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
        int32_t sm0 = 0, sm1 = 0;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j));
            const __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j + 32));
            const int32_t bs0 = xb.bsums[is * 2 + 0] + xb.bsums[is * 2 + 1];
            const int32_t bs1 = xb.bsums[is * 2 + 2] + xb.bsums[is * 2 + 3];
            auto step = [&](const BlockQ4_K & wb, __m256i & sumi, int32_t & sm) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is + 0, wb.scales, sc0, m0);
                get_scale_min_k4(is + 1, wb.scales, sc1, m1);
                sm += static_cast<int32_t>(m0) * bs0 + static_cast<int32_t>(m1) * bs1;
                const __m256i qlv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb.qs + (j >> 1)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(qlv, m4), x0), _mm256_set1_epi16(sc0)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4), x1), _mm256_set1_epi16(sc1)));
            };
            step(w0[b], sum0, sm0);
            step(w1[b], sum1, sm1);
            is += 2;
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w0[b].d) * xb.d), _mm256_cvtepi32_ps(sum0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w1[b].d) * xb.d), _mm256_cvtepi32_ps(sum1), acc1);
        min0 -= (f16_to_f32_fast(w0[b].dmin) * xb.d) * static_cast<float>(sm0);
        min1 -= (f16_to_f32_fast(w1[b].dmin) * xb.d) * static_cast<float>(sm1);
    }
    y0 = hsum256_ps(acc0) + min0;
    y1 = hsum256_ps(acc1) + min1;
#else
    y0 = dot_q4_k_q8_k_row(w0, x_row, nb);
    y1 = dot_q4_k_q8_k_row(w1, x_row, nb);
#endif
}

inline float dot_q5_k_q8_k_row(const BlockQ5_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i b16 = _mm256_set1_epi8(16);
    __m256 acc = _mm256_setzero_ps();
    float min_acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockQ5_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb.qh));
        __m256i sumi = _mm256_setzero_si256();
        int32_t sum_m = 0;
        const uint8_t * ql = wb.qs;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is + 0, wb.scales, sc0, m0);
            get_scale_min_k4(is + 1, wb.scales, sc1, m1);
            sum_m += static_cast<int32_t>(m0) * (xb.bsums[is * 2 + 0] + xb.bsums[is * 2 + 1])
                   + static_cast<int32_t>(m1) * (xb.bsums[is * 2 + 2] + xb.bsums[is * 2 + 3]);

            const __m256i qlv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
            const __m256i vu1 = _mm256_set1_epi8(static_cast<char>(u1));
            const __m256i vu2 = _mm256_set1_epi8(static_cast<char>(u2));
            const __m256i h1  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, vu1), vu1), b16);
            const __m256i h2  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, vu2), vu2), b16);
            const __m256i q0  = _mm256_or_si256(_mm256_and_si256(qlv, m4), h1);
            const __m256i q1  = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4), h2);
            const __m256i x0  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j));
            const __m256i x1  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + j + 32));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q0, x0), _mm256_set1_epi16(sc0)));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q1, x1), _mm256_set1_epi16(sc1)));
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        min_acc -= dmin * static_cast<float>(sum_m);
    }
    return hsum256_ps(acc) + min_acc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t v16 = vdupq_n_u8(16);
    constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    constexpr uint32_t kmask3 = 0x03030303u;

    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +   0, 0, 0);
            __builtin_prefetch(pf +  64, 0, 0);
            __builtin_prefetch(pf + 128, 0, 0);
        }

        uint32_t u0, u1, u2;
        std::memcpy(&u0, w_row[b].scales + 0, 4);
        std::memcpy(&u1, w_row[b].scales + 4, 4);
        std::memcpy(&u2, w_row[b].scales + 8, 4);
        const uint32_t sc_lo4 = u0 & kmask1;
        const uint32_t mn_lo4 = u1 & kmask1;
        const uint32_t sc_hi4 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
        const uint32_t mn_hi4 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);

        const uint16x8_t sc16 = vmovl_u8(vreinterpret_u8_u32(vset_lane_u32(sc_hi4, vdup_n_u32(sc_lo4), 1)));
        const uint16x8_t mn16 = vmovl_u8(vreinterpret_u8_u32(vset_lane_u32(mn_hi4, vdup_n_u32(mn_lo4), 1)));
        const int32x4_t sc_v0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc16)));
        const int32x4_t sc_v1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc16)));

        const uint8x16_t qh0 = vld1q_u8(w_row[b].qh +  0);
        const uint8x16_t qh1 = vld1q_u8(w_row[b].qh + 16);
        const int8_t * xq = x_row[b].qs;

        auto do_pair = [&](const uint8_t * qlp, const int8_t * xp, uint8_t u_lo, uint8_t u_hi, int32x4_t & out_lo, int32x4_t & out_hi) {
            const uint8x16_t q0  = vld1q_u8(qlp +  0);
            const uint8x16_t q1  = vld1q_u8(qlp + 16);
            const uint8x16_t h0a = vandq_u8(vtstq_u8(qh0, vdupq_n_u8(u_lo)), v16);
            const uint8x16_t h1a = vandq_u8(vtstq_u8(qh1, vdupq_n_u8(u_lo)), v16);
            const uint8x16_t h0b = vandq_u8(vtstq_u8(qh0, vdupq_n_u8(u_hi)), v16);
            const uint8x16_t h1b = vandq_u8(vtstq_u8(qh1, vdupq_n_u8(u_hi)), v16);
            out_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4), h0a)), vld1q_s8(xp +  0)),
                               vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4), h1a)), vld1q_s8(xp + 16));
            out_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q0, 4), h0b)), vld1q_s8(xp + 32)),
                               vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q1, 4), h1b)), vld1q_s8(xp + 48));
        };

        int32x4_t p0, p1, p2, p3;
        do_pair(w_row[b].qs +  0, xq +   0, 0x01u, 0x02u, p0, p1);
        do_pair(w_row[b].qs + 32, xq +  64, 0x04u, 0x08u, p2, p3);
        const int32x4_t d0 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

        do_pair(w_row[b].qs + 64, xq + 128, 0x10u, 0x20u, p0, p1);
        do_pair(w_row[b].qs + 96, xq + 192, 0x40u, 0x80u, p2, p3);
        const int32x4_t d1 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

        const int32_t sum_pos = vaddvq_s32(vmlaq_s32(vmulq_s32(d0, sc_v0), d1, sc_v1));
        const int32x4_t bs0 = vpaddlq_s16(vld1q_s16(x_row[b].bsums + 0));
        const int32x4_t bs1 = vpaddlq_s16(vld1q_s16(x_row[b].bsums + 8));
        const int32_t sum_min = vaddvq_s32(vmlaq_s32(
            vmulq_s32(bs0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mn16)))),
            bs1, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mn16)))));

        const float val = std::fma(-f16_to_f32_fast(w_row[b].dmin), static_cast<float>(sum_min),
                                   f16_to_f32_fast(w_row[b].d) * static_cast<float>(sum_pos));
        sumf = std::fma(val, x_row[b].d, sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockQ5_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;
        int32_t sumi = 0, sum_m = 0;
        const uint8_t * ql = wb.qs;
        const uint8_t * qh = wb.qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is + 0, wb.scales, sc0, m0);
            get_scale_min_k4(is + 1, wb.scales, sc1, m1);
            sum_m += static_cast<int32_t>(m0) * (xb.bsums[is * 2 + 0] + xb.bsums[is * 2 + 1])
                   + static_cast<int32_t>(m1) * (xb.bsums[is * 2 + 2] + xb.bsums[is * 2 + 3]);
            int32_t s0 = 0, s1 = 0;
            for (int l = 0; l < 32; ++l) {
                const int q1 = (ql[l] & 0xFu) + ((qh[l] & u1) ? 16 : 0);
                const int q2 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                s0 += q1 * static_cast<int32_t>(xb.qs[j + l]);
                s1 += q2 * static_cast<int32_t>(xb.qs[j + 32 + l]);
            }
            sumi += sc0 * s0 + sc1 * s1;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
        acc += d * static_cast<float>(sumi) - dmin * static_cast<float>(sum_m);
    }
    return acc;
#endif
}

inline float dot_q2_k_q8_k_row(const BlockQ2_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m2   = _mm256_set1_epi8(0x03);
    const __m128i m4_8 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float min_acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockQ2_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;

        const __m128i sc8  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.scales));
        const __m256i m16  = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), m4_8));
        const __m256i bs16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.bsums));
        const int32_t sum_m = hsum256_epi32(_mm256_madd_epi16(m16, bs16));

        __m256i sumi = _mm256_setzero_si256();
        for (int n = 0; n < 2; ++n) {
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb.qs + n * 32));
            const uint8_t * sc = wb.scales + n * 8;
            for (int j = 0; j < 4; ++j) {
                const __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m2);
                const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + n * 128 + j * 32));
                const __m256i s01 = _mm256_set_m128i(
                    _mm_set1_epi16(static_cast<int16_t>(sc[2 * j + 1] & 0xFu)),
                    _mm_set1_epi16(static_cast<int16_t>(sc[2 * j + 0] & 0xFu)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q2, xv), s01));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        min_acc -= dmin * static_cast<float>(sum_m);
    }
    return hsum256_ps(acc) + min_acc;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m2 = vdupq_n_u8(0x03u);
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +  0, 0, 0);
            __builtin_prefetch(pf + 64, 0, 0);
        }
        const BlockQ2_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];

        const uint8x16_t sc_raw = vld1q_u8(wb.scales);
        const uint8x16_t sc_u8  = vandq_u8(sc_raw, m4);
        const uint8x16_t mn_u8  = vshrq_n_u8(sc_raw, 4);

        // sum_m = dot(mn_u8[0..15], xb.bsums[0..15])
        const int16x8_t mn_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mn_u8)));
        const int16x8_t mn_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mn_u8)));
        const int16x8_t bs_lo = vld1q_s16(xb.bsums + 0);
        const int16x8_t bs_hi = vld1q_s16(xb.bsums + 8);
        int32x4_t vm = vmull_s16(vget_low_s16(mn_lo), vget_low_s16(bs_lo));
        vm = vmlal_s16(vm, vget_high_s16(mn_lo), vget_high_s16(bs_lo));
        vm = vmlal_s16(vm, vget_low_s16(mn_hi),  vget_low_s16(bs_hi));
        vm = vmlal_s16(vm, vget_high_s16(mn_hi), vget_high_s16(bs_hi));
        const int32_t sum_m = vaddvq_s32(vm);

        const uint16x8_t sc_lo16 = vmovl_u8(vget_low_u8(sc_u8));
        const uint16x8_t sc_hi16 = vmovl_u8(vget_high_u8(sc_u8));

        auto do_half = [&](const uint8_t * qp, const int8_t * xp, uint16x8_t sc16) -> int32x4_t {
            const uint8x16_t q0 = vld1q_u8(qp +  0);
            const uint8x16_t q1 = vld1q_u8(qp + 16);
            int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q0, m2)), vld1q_s8(xp +  0));
            int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q1, m2)), vld1q_s8(xp + 16));
            int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 2), m2)), vld1q_s8(xp + 32));
            int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 2), m2)), vld1q_s8(xp + 48));
            const int32x4_t d0 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            p0 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 4), m2)), vld1q_s8(xp +  64));
            p1 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 4), m2)), vld1q_s8(xp +  80));
            p2 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q0, 6)), vld1q_s8(xp +  96));
            p3 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q1, 6)), vld1q_s8(xp + 112));
            const int32x4_t d1 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            return vmlaq_s32(
                vmulq_s32(d0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc16)))),
                d1, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc16))));
        };

        const int32_t sumi = vaddvq_s32(vaddq_s32(
            do_half(wb.qs +  0, xb.qs +   0, sc_lo16),
            do_half(wb.qs + 32, xb.qs + 128, sc_hi16)));
        const float val = std::fma(-f16_to_f32_fast(wb.dmin), static_cast<float>(sum_m),
                                   f16_to_f32_fast(wb.d) * static_cast<float>(sumi));
        sumf = std::fma(val, xb.d, sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockQ2_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d    = f16_to_f32_fast(wb.d) * xb.d;
        const float dmin = f16_to_f32_fast(wb.dmin) * xb.d;
        int32_t sumi = 0, sum_m = 0;
        for (int k = 0; k < 16; ++k) sum_m += static_cast<int32_t>(wb.scales[k] >> 4) * xb.bsums[k];
        for (int n = 0; n < 2; ++n) {
            const uint8_t * q  = wb.qs + n * 32;
            const uint8_t * sc = wb.scales + n * 8;
            for (int j = 0; j < 4; ++j) {
                const int shift = 2 * j;
                int32_t s0 = 0, s1 = 0;
                for (int l = 0; l < 16; ++l) {
                    s0 += static_cast<int32_t>((q[l]      >> shift) & 3u) * static_cast<int32_t>(xb.qs[n * 128 + j * 32 + l]);
                    s1 += static_cast<int32_t>((q[l + 16] >> shift) & 3u) * static_cast<int32_t>(xb.qs[n * 128 + j * 32 + 16 + l]);
                }
                sumi += static_cast<int32_t>(sc[2 * j + 0] & 0xFu) * s0
                      + static_cast<int32_t>(sc[2 * j + 1] & 0xFu) * s1;
            }
        }
        acc += d * static_cast<float>(sumi) - dmin * static_cast<float>(sum_m);
    }
    return acc;
#endif
}

inline float dot_q3_k_q8_k_row(const BlockQ3_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m2 = _mm256_set1_epi8(0x03);
    const __m256i c4 = _mm256_set1_epi8(4);
    const __m128i c32_8 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockQ3_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;

        uint32_t aux[4];
        std::memcpy(aux, wb.scales, 12);
        const uint32_t tmp = aux[2];
        constexpr uint32_t kmask1 = 0x03030303u;
        constexpr uint32_t kmask2 = 0x0f0f0f0fu;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const __m128i sc8_signed = _mm_sub_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(aux)), c32_8);
        const __m256i sc16 = _mm256_cvtepi8_epi16(sc8_signed);
        const __m256i bs16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.bsums));
        // val = q3_u8 - 4 -> subtract 4 * sum_k (sc_signed[k] * bsums[k])
        __m256i sumi = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs16), 2));

        alignas(16) int8_t sc_s[16];
        _mm_store_si128(reinterpret_cast<__m128i *>(sc_s), sc8_signed);
        const __m256i hmv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb.hmask));
        for (int n = 0; n < 2; ++n) {
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb.qs + n * 32));
            for (int j = 0; j < 4; ++j) {
                const uint8_t m = static_cast<uint8_t>(1u << (n * 4 + j));
                const __m256i vm = _mm256_set1_epi8(static_cast<char>(m));
                const __m256i hb = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hmv, vm), vm), c4);
                const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m2), hb);
                const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + n * 128 + j * 32));
                const __m256i s01 = _mm256_set_m128i(
                    _mm_set1_epi16(sc_s[n * 8 + 2 * j + 1]),
                    _mm_set1_epi16(sc_s[n * 8 + 2 * j + 0]));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q3, xv), s01));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m2  = vdupq_n_u8(0x03u);
    const uint8x16_t c4  = vdupq_n_u8(4);
    const int8x16_t  c32 = vdupq_n_s8(32);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +  0, 0, 0);
            __builtin_prefetch(pf + 64, 0, 0);
        }
        const BlockQ3_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];

        uint32_t aux[4];
        std::memcpy(aux, wb.scales, 12);
        const uint32_t tmp = aux[2];
        constexpr uint32_t kmask1 = 0x03030303u;
        constexpr uint32_t kmask2 = 0x0f0f0f0fu;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const int8x16_t sc_s8 = vsubq_s8(vreinterpretq_s8_u32(vld1q_u32(aux)), c32);
        const int16x8_t sc_lo = vmovl_s8(vget_low_s8(sc_s8));
        const int16x8_t sc_hi = vmovl_s8(vget_high_s8(sc_s8));
        const int16x8_t bs_lo = vld1q_s16(xb.bsums + 0);
        const int16x8_t bs_hi = vld1q_s16(xb.bsums + 8);
        int32x4_t bsum_v = vmull_s16(vget_low_s16(sc_lo), vget_low_s16(bs_lo));
        bsum_v = vmlal_s16(bsum_v, vget_high_s16(sc_lo), vget_high_s16(bs_lo));
        bsum_v = vmlal_s16(bsum_v, vget_low_s16(sc_hi),  vget_low_s16(bs_hi));
        bsum_v = vmlal_s16(bsum_v, vget_high_s16(sc_hi), vget_high_s16(bs_hi));

        const uint8x16_t hm0 = vld1q_u8(wb.hmask +  0);
        const uint8x16_t hm1 = vld1q_u8(wb.hmask + 16);

        auto do_half = [&](const uint8_t * qp, const int8_t * xp, int16x8_t sc16, uint8_t m_base) -> int32x4_t {
            const uint8x16_t q0 = vld1q_u8(qp +  0);
            const uint8x16_t q1 = vld1q_u8(qp + 16);

            auto pair_dot = [&](uint8x16_t lo_bits0, uint8x16_t lo_bits1, uint8_t mask_b, const int8_t * xpp,
                                int32x4_t & o0, int32x4_t & o1) {
                const uint8x16_t vm = vdupq_n_u8(mask_b);
                const int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(lo_bits0, vandq_u8(vtstq_u8(hm0, vm), c4)));
                const int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(lo_bits1, vandq_u8(vtstq_u8(hm1, vm), c4)));
                o0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xpp +  0));
                o1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(xpp + 16));
            };

            int32x4_t p0, p1, p2, p3;
            pair_dot(vandq_u8(q0, m2), vandq_u8(q1, m2), m_base, xp + 0, p0, p1);
            pair_dot(vandq_u8(vshrq_n_u8(q0, 2), m2), vandq_u8(vshrq_n_u8(q1, 2), m2), static_cast<uint8_t>(m_base << 1), xp + 32, p2, p3);
            const int32x4_t d0 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            pair_dot(vandq_u8(vshrq_n_u8(q0, 4), m2), vandq_u8(vshrq_n_u8(q1, 4), m2), static_cast<uint8_t>(m_base << 2), xp + 64, p0, p1);
            pair_dot(vshrq_n_u8(q0, 6), vshrq_n_u8(q1, 6), static_cast<uint8_t>(m_base << 3), xp + 96, p2, p3);
            const int32x4_t d1 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));

            return vmlaq_s32(vmulq_s32(d0, vmovl_s16(vget_low_s16(sc16))),
                             d1, vmovl_s16(vget_high_s16(sc16)));
        };

        const int32x4_t acc_v = vsubq_s32(
            vaddq_s32(do_half(wb.qs +  0, xb.qs +   0, sc_lo, 0x01u),
                      do_half(wb.qs + 32, xb.qs + 128, sc_hi, 0x10u)),
            vshlq_n_s32(bsum_v, 2));
        sumf = std::fma(static_cast<float>(vaddvq_s32(acc_v)) * xb.d, f16_to_f32_fast(wb.d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockQ3_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        uint32_t aux[4];
        std::memcpy(aux, wb.scales, 12);
        const uint32_t tmp = aux[2];
        constexpr uint32_t kmask1 = 0x03030303u;
        constexpr uint32_t kmask2 = 0x0f0f0f0fu;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        uint8_t sc_u[16];
        std::memcpy(sc_u, aux, 16);
        int32_t sumi = 0;
        for (int n = 0; n < 2; ++n) {
            const uint8_t * q = wb.qs + n * 32;
            for (int j = 0; j < 4; ++j) {
                const int shift = 2 * j;
                const uint8_t m = static_cast<uint8_t>(1u << (n * 4 + j));
                int32_t s0 = 0, s1 = 0;
                for (int l = 0; l < 16; ++l) {
                    const int v0 = static_cast<int>((q[l]      >> shift) & 3u) - ((wb.hmask[l]      & m) ? 0 : 4);
                    const int v1 = static_cast<int>((q[l + 16] >> shift) & 3u) - ((wb.hmask[l + 16] & m) ? 0 : 4);
                    s0 += v0 * static_cast<int32_t>(xb.qs[n * 128 + j * 32 + l]);
                    s1 += v1 * static_cast<int32_t>(xb.qs[n * 128 + j * 32 + 16 + l]);
                }
                sumi += (static_cast<int32_t>(sc_u[n * 8 + 2 * j + 0]) - 32) * s0
                      + (static_cast<int32_t>(sc_u[n * 8 + 2 * j + 1]) - 32) * s1;
            }
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

inline float dot_iq4_xs_q8_k_row(const BlockIQ4_XS * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl)));
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockIQ4_XS & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint16_t sh = wb.scales_h;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((wb.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.qs + ib * 16));
            const __m256i q8  = _mm256_shuffle_epi8(lut, _mm256_and_si256(
                _mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4));
            const __m256i xv  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + ib * 32));
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(q8), _mm256_sign_epi8(xv, q8));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, _mm256_set1_epi16(static_cast<int16_t>(ls - 32))));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t  lut = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +   0, 0, 0);
            __builtin_prefetch(pf +  64, 0, 0);
            __builtin_prefetch(pf + 128, 0, 0);
        }
        const BlockIQ4_XS & wb = w_row[b];
        const CpuActQ8_K  & xb = x_row[b];
        const uint16_t sh = wb.scales_h;
        int32x4_t sumi_v = vdupq_n_s32(0);

        for (int half = 0; half < 2; ++half) {
            const int ib0 = half * 4;
            int32x4_t p[4];
            int32_t sc4[4];
            for (int k = 0; k < 4; ++k) {
                const int ib = ib0 + k;
                sc4[k] = static_cast<int32_t>(
                    (((wb.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xFu) |
                     (((sh >> (2 * ib)) & 3u) << 4))) - 32;
                const uint8x16_t raw = vld1q_u8(wb.qs + ib * 16);
                const int8x16_t w_lo = vqtbl1q_s8(lut, vandq_u8(raw, m4));
                const int8x16_t w_hi = vqtbl1q_s8(lut, vshrq_n_u8(raw, 4));
                const int8_t * xp = xb.qs + ib * 32;
                p[k] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w_lo, vld1q_s8(xp)),
                                 w_hi, vld1q_s8(xp + 16));
            }
            const int32x4_t d4 = vpaddq_s32(vpaddq_s32(p[0], p[1]), vpaddq_s32(p[2], p[3]));
            sumi_v = vmlaq_s32(sumi_v, d4, vld1q_s32(sc4));
        }
        sumf = std::fma(static_cast<float>(vaddvq_s32(sumi_v)) * xb.d, f16_to_f32_fast(wb.d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockIQ4_XS & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint16_t sh = wb.scales_h;
        int32_t sumi = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((wb.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const uint8_t * qs = wb.qs + ib * 16;
            int32_t s = 0;
            for (int j = 0; j < 16; ++j) {
                s += static_cast<int32_t>(kvalues_iq4nl[qs[j] & 0xFu]) * static_cast<int32_t>(xb.qs[ib * 32 + j]);
                s += static_cast<int32_t>(kvalues_iq4nl[qs[j] >> 4])   * static_cast<int32_t>(xb.qs[ib * 32 + 16 + j]);
            }
            sumi += (ls - 32) * s;
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

inline float dot_iq3_xxs_q8_k_row(const BlockIQ3_XXS * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockIQ3_XXS & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint8_t * qs = wb.qs;
        const uint8_t * ss = wb.qs + 64;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32 = 0;
            std::memcpy(&aux32, ss + ib32 * 4, 4);
            const int16_t sc = static_cast<int16_t>(1 + 2 * (aux32 >> 28));
            const uint32_t sm = static_cast<uint32_t>(ksigns_iq2xs[(aux32 >>  0) & 127u])
                              | (static_cast<uint32_t>(ksigns_iq2xs[(aux32 >>  7) & 127u]) << 8)
                              | (static_cast<uint32_t>(ksigns_iq2xs[(aux32 >> 14) & 127u]) << 16)
                              | (static_cast<uint32_t>(ksigns_iq2xs[(aux32 >> 21) & 127u]) << 24);
            const __m256i gv = _mm256_setr_epi32(
                static_cast<int32_t>(iq3xxs_grid[qs[0]]), static_cast<int32_t>(iq3xxs_grid[qs[1]]),
                static_cast<int32_t>(iq3xxs_grid[qs[2]]), static_cast<int32_t>(iq3xxs_grid[qs[3]]),
                static_cast<int32_t>(iq3xxs_grid[qs[4]]), static_cast<int32_t>(iq3xxs_grid[qs[5]]),
                static_cast<int32_t>(iq3xxs_grid[qs[6]]), static_cast<int32_t>(iq3xxs_grid[qs[7]]));
            const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm)), shuf), bitm), bitm);
            const __m256i xv  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + ib32 * 32));
            const __m256i sx  = _mm256_sub_epi8(_mm256_xor_si256(neg, xv), neg);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(gv, sx), _mm256_set1_epi16(sc)));
            qs += 8;
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.25f * hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +  0, 0, 0);
            __builtin_prefetch(pf + 64, 0, 0);
        }
        const BlockIQ3_XXS & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const uint8_t * qs = wb.qs;
        const uint8_t * ss = wb.qs + 64;
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32 = 0;
            std::memcpy(&aux32, ss + ib32 * 4, 4);
            const int32_t sc = static_cast<int32_t>(1 + 2 * (aux32 >> 28));
            const uint32_t g0[4] = {iq3xxs_grid[qs[0]], iq3xxs_grid[qs[1]], iq3xxs_grid[qs[2]], iq3xxs_grid[qs[3]]};
            const uint32_t g1[4] = {iq3xxs_grid[qs[4]], iq3xxs_grid[qs[5]], iq3xxs_grid[qs[6]], iq3xxs_grid[qs[7]]};
            const int8x16_t gv0 = vreinterpretq_s8_u32(vld1q_u32(g0));
            const int8x16_t gv1 = vreinterpretq_s8_u32(vld1q_u32(g1));
            const uint8_t s0 = ksigns_iq2xs[(aux32 >>  0) & 127u];
            const uint8_t s1 = ksigns_iq2xs[(aux32 >>  7) & 127u];
            const uint8_t s2 = ksigns_iq2xs[(aux32 >> 14) & 127u];
            const uint8_t s3 = ksigns_iq2xs[(aux32 >> 21) & 127u];
            const uint8x16_t sg0 = vtstq_u8(vcombine_u8(vdup_n_u8(s0), vdup_n_u8(s1)), vmask);
            const uint8x16_t sg1 = vtstq_u8(vcombine_u8(vdup_n_u8(s2), vdup_n_u8(s3)), vmask);
            const int8x16_t  w0  = vbslq_s8(sg0, vnegq_s8(gv0), gv0);
            const int8x16_t  w1  = vbslq_s8(sg1, vnegq_s8(gv1), gv1);
            const int32x4_t  p   = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xb.qs + ib32 * 32 + 0)),
                                             w1, vld1q_s8(xb.qs + ib32 * 32 + 16));
            sumi = vmlaq_n_s32(sumi, p, sc);
            qs += 8;
        }
        sumf = std::fma(static_cast<float>(vaddvq_s32(sumi)) * (0.25f * xb.d), f16_to_f32_fast(wb.d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockIQ3_XXS & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * 0.25f * xb.d;
        const uint8_t * qs = wb.qs;
        const uint8_t * ss = wb.qs + 64;
        int32_t sumi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32 = 0;
            std::memcpy(&aux32, ss + ib32 * 4, 4);
            const int32_t sc = static_cast<int32_t>(1 + 2 * (aux32 >> 28));
            int32_t ls = 0;
            for (int l = 0; l < 4; ++l) {
                const uint32_t g0 = iq3xxs_grid[qs[2 * l + 0]];
                const uint32_t g1 = iq3xxs_grid[qs[2 * l + 1]];
                const uint8_t  sg = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
                for (int j = 0; j < 4; ++j) {
                    const int32_t m0 = static_cast<int32_t>((g0 >> (8 * j)) & 0xFFu);
                    const int32_t m1 = static_cast<int32_t>((g1 >> (8 * j)) & 0xFFu);
                    const int32_t x0 = static_cast<int32_t>(xb.qs[ib32 * 32 + l * 8 + j + 0]);
                    const int32_t x1 = static_cast<int32_t>(xb.qs[ib32 * 32 + l * 8 + j + 4]);
                    ls += (sg & (1u << (j + 0))) ? (-m0 * x0) : (m0 * x0);
                    ls += (sg & (1u << (j + 4))) ? (-m1 * x1) : (m1 * x1);
                }
            }
            sumi += sc * ls;
            qs += 8;
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

inline float dot_iq2_s_q8_k_row(const BlockIQ2_S * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const char * pf = reinterpret_cast<const char *>(w_row + b + 2);
        _mm_prefetch(pf +  0, _MM_HINT_T0);
        _mm_prefetch(pf + 64, _MM_HINT_T0);
        const BlockIQ2_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint8_t * qs    = wb.qs;
        const uint8_t * signs = wb.qs + 32;
        const uint8_t * qh    = wb.qh;
        // Pure GPR nibble-to-odd-scale decode: zero XMM constants -> eliminates all 5 stack spills!
        uint64_t aux64;
        std::memcpy(&aux64, wb.scales, 8);
        const uint64_t sc_lo = ((aux64 & 0x0F0F0F0F0F0F0F0FULL) << 1) | 0x0101010101010101ULL;
        const uint64_t sc_hi = (((aux64 >> 4) & 0x0F0F0F0F0F0F0F0FULL) << 1) | 0x0101010101010101ULL;
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_set_epi64x(static_cast<int64_t>(sc_hi), static_cast<int64_t>(sc_lo)));
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            {
                const uint32_t qh0 = qh[ib32 + 0];
                const __m256i gv0 = _mm256_set_epi64x(
                    static_cast<int64_t>(iq2s_grid[qs[3] | ((qh0 << 2) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[2] | ((qh0 << 4) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[1] | ((qh0 << 6) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[0] | ((qh0 << 8) & 0x300u)]));
                uint32_t sm0;
                std::memcpy(&sm0, signs + 0, 4);
                const __m256i neg0 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm0)), shuf), bitm), bitm);
                const __m256i xv0  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + (ib32 + 0) * 32));
                const __m256i sx0  = _mm256_sub_epi8(_mm256_xor_si256(neg0, xv0), neg0);
                const __m256i sh0  = _mm256_set1_epi16(static_cast<int16_t>((2 * (ib32 + 0)) | ((2 * (ib32 + 0) + 1) << 8)));
                sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(_mm256_maddubs_epi16(gv0, sx0), _mm256_shuffle_epi8(sc16, sh0)));
            }
            {
                const uint32_t qh1 = qh[ib32 + 1];
                const __m256i gv1 = _mm256_set_epi64x(
                    static_cast<int64_t>(iq2s_grid[qs[7] | ((qh1 << 2) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[6] | ((qh1 << 4) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[5] | ((qh1 << 6) & 0x300u)]),
                    static_cast<int64_t>(iq2s_grid[qs[4] | ((qh1 << 8) & 0x300u)]));
                uint32_t sm1;
                std::memcpy(&sm1, signs + 4, 4);
                const __m256i neg1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm1)), shuf), bitm), bitm);
                const __m256i xv1  = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + (ib32 + 1) * 32));
                const __m256i sx1  = _mm256_sub_epi8(_mm256_xor_si256(neg1, xv1), neg1);
                const __m256i sh1  = _mm256_set1_epi16(static_cast<int16_t>((2 * (ib32 + 1)) | ((2 * (ib32 + 1) + 1) << 8)));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(gv1, sx1), _mm256_shuffle_epi8(sc16, sh1)));
            }
            qs += 8; signs += 8;
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }
    return 0.125f * hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +   0, 0, 0);
            __builtin_prefetch(pf +  64, 0, 0);
            __builtin_prefetch(pf + 128, 0, 0);
        }
        const BlockIQ2_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const uint8_t * qs    = wb.qs;
        const uint8_t * signs = wb.qs + 32;
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t qhb = wb.qh[ib32];
            const int32_t  s0  = 1 + 2 * (wb.scales[ib32] & 0xFu);
            const int32_t  s1  = 1 + 2 * (wb.scales[ib32] >> 4);

            const uint64x2_t g01 = vcombine_u64(
                vdup_n_u64(iq2s_grid[qs[0] | ((qhb << 8) & 0x300u)]),
                vdup_n_u64(iq2s_grid[qs[1] | ((qhb << 6) & 0x300u)]));
            const uint64x2_t g23 = vcombine_u64(
                vdup_n_u64(iq2s_grid[qs[2] | ((qhb << 4) & 0x300u)]),
                vdup_n_u64(iq2s_grid[qs[3] | ((qhb << 2) & 0x300u)]));

            const uint8x16_t sg01 = vtstq_u8(vcombine_u8(vdup_n_u8(signs[0]), vdup_n_u8(signs[1])), vmask);
            const uint8x16_t sg23 = vtstq_u8(vcombine_u8(vdup_n_u8(signs[2]), vdup_n_u8(signs[3])), vmask);
            const int8x16_t  w01  = vbslq_s8(sg01, vnegq_s8(vreinterpretq_s8_u64(g01)), vreinterpretq_s8_u64(g01));
            const int8x16_t  w23  = vbslq_s8(sg23, vnegq_s8(vreinterpretq_s8_u64(g23)), vreinterpretq_s8_u64(g23));

            const int32x4_t p01 = vdotq_s32(vdupq_n_s32(0), w01, vld1q_s8(xb.qs + ib32 * 32 +  0));
            const int32x4_t p23 = vdotq_s32(vdupq_n_s32(0), w23, vld1q_s8(xb.qs + ib32 * 32 + 16));
            sumi = vmlaq_n_s32(sumi, p01, s0);
            sumi = vmlaq_n_s32(sumi, p23, s1);
            qs += 4; signs += 4;
        }
        sumf = std::fma(static_cast<float>(vaddvq_s32(sumi)) * xb.d, f16_to_f32_fast(wb.d), sumf);
    }
    return 0.125f * sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockIQ2_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * 0.125f * xb.d;
        const uint8_t * qs    = wb.qs;
        const uint8_t * signs = wb.qs + 32;
        const uint8_t * qh    = wb.qh;
        int32_t sumi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t qhb = qh[ib32];
            const int s0 = 1 + 2 * (wb.scales[ib32] & 0xFu);
            const int s1 = 1 + 2 * (wb.scales[ib32] >> 4);
            for (int l = 0; l < 4; ++l) {
                const uint64_t g64 = iq2s_grid[qs[l] | ((qhb << (8 - 2 * l)) & 0x300u)];
                const uint8_t  sg  = signs[l];
                int32_t ls = 0;
                for (int j = 0; j < 8; ++j) {
                    const int32_t mag = static_cast<int32_t>((g64 >> (8 * j)) & 0xFFu);
                    const int32_t xq  = static_cast<int32_t>(xb.qs[ib32 * 32 + l * 8 + j]);
                    ls += (sg & (1u << j)) ? (-mag * xq) : (mag * xq);
                }
                sumi += ((l < 2) ? s0 : s1) * ls;
            }
            qs += 4; signs += 4;
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
// Multi-token (2..4) IQ2_S direct dot: decodes BlockIQ2_S grid & signs ONCE for all nt tokens!
inline void gemm_iq2_s_1xN_direct(
    const BlockIQ2_S * w_row,
    const CpuActQ8_K * const * xs,
    int nt, int nb,
    float * out) {

    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    const __m128i m4   = _mm_set1_epi8(0x0F);
    const __m128i m1   = _mm_set1_epi8(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    for (int b = 0; b < nb; ++b) {
        const BlockIQ2_S & wb = w_row[b];
        const float wd = f16_to_f32_fast(wb.d);
        uint64_t aux64;
        std::memcpy(&aux64, wb.scales, 8);
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(static_cast<int64_t>(aux64 >> 4), static_cast<int64_t>(aux64)), m4), 1), m1));

        __m256i sumi[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * qs = wb.qs + ib32 * 4;
            const uint32_t qhb = wb.qh[ib32];
            const __m256i gv = _mm256_set_epi64x(
                static_cast<int64_t>(iq2s_grid[qs[3] | ((qhb << 2) & 0x300u)]),
                static_cast<int64_t>(iq2s_grid[qs[2] | ((qhb << 4) & 0x300u)]),
                static_cast<int64_t>(iq2s_grid[qs[1] | ((qhb << 6) & 0x300u)]),
                static_cast<int64_t>(iq2s_grid[qs[0] | ((qhb << 8) & 0x300u)]));
            uint32_t sm;
            std::memcpy(&sm, wb.qs + 32 + ib32 * 4, 4);
            const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm)), shuf), bitm), bitm);
            const __m256i scv = _mm256_shuffle_epi8(sc16, _mm256_set1_epi16(static_cast<int16_t>((2 * ib32) | ((2 * ib32 + 1) << 8))));
            for (int ti = 0; ti < nt; ++ti) {
                const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xs[ti][b].qs + ib32 * 32));
                const __m256i sx = _mm256_sub_epi8(_mm256_xor_si256(neg, xv), neg);
                sumi[ti] = _mm256_add_epi32(sumi[ti], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, sx), scv));
            }
        }
        for (int ti = 0; ti < nt; ++ti) {
            acc[ti] = _mm256_fmadd_ps(_mm256_set1_ps(wd * xs[ti][b].d), _mm256_cvtepi32_ps(sumi[ti]), acc[ti]);
        }
    }
    for (int ti = 0; ti < nt; ++ti) out[ti] = 0.125f * hsum256_ps(acc[ti]);
}

// Multi-token (2..4) IQ4_NL direct dot: decodes BlockIQ4_NL LUT ONCE for all nt tokens!
inline void gemm_iq4_nl_1xN_direct(
    const BlockIQ4_NL * w_row,
    const CpuActQ8_0 * const * xs,
    int nt, int nb,
    float * out) {

    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i lut    = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl)));
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    for (int b = 0; b < nb; ++b) {
        const __m128i r  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b].qs));
        const __m256i q  = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r), _mm_srli_epi16(r, 4), 1), m4));
        const __m256i aq = _mm256_abs_epi8(q);
        const float   wd = f16_to_f32_fast(w_row[b].d);
        for (int ti = 0; ti < nt; ++ti) {
            const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xs[ti][b].qs));
            const __m256i dp = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(xv, q)), ones16);
            acc[ti] = _mm256_fmadd_ps(_mm256_set1_ps(wd * xs[ti][b].d), _mm256_cvtepi32_ps(dp), acc[ti]);
        }
    }
    for (int ti = 0; ti < nt; ++ti) out[ti] = hsum256_ps(acc[ti]);
}

// Multi-token (2..4) Q4_0 direct dot: decodes BlockQ4_0 nibbles ONCE for all nt tokens!
inline void gemm_q4_0_1xN_direct(
    const BlockQ4_0 * w_row,
    const CpuActQ8_0 * const * xs,
    int nt, int nb,
    float * out) {

    const __m256i m4     = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    float  sacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nb; ++b) {
        const __m128i r  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_row[b].qs));
        const __m256i q  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r), _mm_srli_epi16(r, 4), 1), m4);
        const float   wd = f16_to_f32_fast(w_row[b].d);
        for (int ti = 0; ti < nt; ++ti) {
            const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(xs[ti][b].qs));
            const __m256i dp = _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones16);
            acc[ti]  = _mm256_fmadd_ps(_mm256_set1_ps(wd * xs[ti][b].d), _mm256_cvtepi32_ps(dp), acc[ti]);
            sacc[ti] += wd * xs[ti][b].s;
        }
    }
    for (int ti = 0; ti < nt; ++ti) out[ti] = hsum256_ps(acc[ti]) - 8.0f * sacc[ti];
}
#endif

inline float dot_iq3_s_q8_k_row(const BlockIQ3_S * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ULL));
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        _mm_prefetch(reinterpret_cast<const char *>(w_row + b + 2), _MM_HINT_T0);
        const BlockIQ3_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint8_t * qs    = wb.qs;
        const uint8_t * qh    = wb.qh;
        const uint8_t * signs = wb.signs;
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const uint32_t qh0 = qh[ib32 + 0];
            const uint32_t qh1 = qh[ib32 + 1];
            const __m256i gv0 = _mm256_setr_epi32(
                static_cast<int32_t>(iq3s_grid[qs[0] | ((qh0 << 8) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[1] | ((qh0 << 7) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[2] | ((qh0 << 6) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[3] | ((qh0 << 5) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[4] | ((qh0 << 4) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[5] | ((qh0 << 3) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[6] | ((qh0 << 2) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[7] | ((qh0 << 1) & 256u)]));
            const __m256i gv1 = _mm256_setr_epi32(
                static_cast<int32_t>(iq3s_grid[qs[8]  | ((qh1 << 8) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[9]  | ((qh1 << 7) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[10] | ((qh1 << 6) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[11] | ((qh1 << 5) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[12] | ((qh1 << 4) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[13] | ((qh1 << 3) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[14] | ((qh1 << 2) & 256u)]),
                static_cast<int32_t>(iq3s_grid[qs[15] | ((qh1 << 1) & 256u)]));

            uint32_t sm0, sm1;
            std::memcpy(&sm0, signs + 0, 4);
            std::memcpy(&sm1, signs + 4, 4);
            const __m256i neg0 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm0)), shuf), bitm), bitm);
            const __m256i neg1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(sm1)), shuf), bitm), bitm);

            const __m256i xv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + (ib32 + 0) * 32));
            const __m256i xv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.qs + (ib32 + 1) * 32));
            const __m256i sx0 = _mm256_sub_epi8(_mm256_xor_si256(neg0, xv0), neg0);
            const __m256i sx1 = _mm256_sub_epi8(_mm256_xor_si256(neg1, xv1), neg1);

            const uint8_t scb = wb.scales[ib32 >> 1];
            const int16_t sc0 = static_cast<int16_t>(1 + 2 * (scb & 0xFu));
            const int16_t sc1 = static_cast<int16_t>(1 + 2 * (scb >> 4));
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(_mm256_maddubs_epi16(gv0, sx0), _mm256_set1_epi16(sc0)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(gv1, sx1), _mm256_set1_epi16(sc1)));
            qs += 16; signs += 8;
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }
    return hsum256_ps(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            const auto * pf = reinterpret_cast<const char *>(&w_row[b + 2]);
            __builtin_prefetch(pf +   0, 0, 0);
            __builtin_prefetch(pf +  64, 0, 0);
            __builtin_prefetch(pf + 128, 0, 0);
        }
        const BlockIQ3_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const uint8_t * qs    = wb.qs;
        const uint8_t * signs = wb.signs;
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t qhb = wb.qh[ib32];
            const uint8_t  scb = wb.scales[ib32 >> 1];
            const int32_t  sc  = 1 + 2 * ((ib32 & 1) ? (scb >> 4) : (scb & 0xFu));

            const uint32_t g0[4] = {
                iq3s_grid[qs[0] | ((qhb << 8) & 256u)],
                iq3s_grid[qs[1] | ((qhb << 7) & 256u)],
                iq3s_grid[qs[2] | ((qhb << 6) & 256u)],
                iq3s_grid[qs[3] | ((qhb << 5) & 256u)]
            };
            const uint32_t g1[4] = {
                iq3s_grid[qs[4] | ((qhb << 4) & 256u)],
                iq3s_grid[qs[5] | ((qhb << 3) & 256u)],
                iq3s_grid[qs[6] | ((qhb << 2) & 256u)],
                iq3s_grid[qs[7] | ((qhb << 1) & 256u)]
            };
            const int8x16_t gv0 = vreinterpretq_s8_u32(vld1q_u32(g0));
            const int8x16_t gv1 = vreinterpretq_s8_u32(vld1q_u32(g1));
            const uint8x16_t sg0 = vtstq_u8(vcombine_u8(vdup_n_u8(signs[0]), vdup_n_u8(signs[1])), vmask);
            const uint8x16_t sg1 = vtstq_u8(vcombine_u8(vdup_n_u8(signs[2]), vdup_n_u8(signs[3])), vmask);
            const int8x16_t  w0  = vbslq_s8(sg0, vnegq_s8(gv0), gv0);
            const int8x16_t  w1  = vbslq_s8(sg1, vnegq_s8(gv1), gv1);

            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xb.qs + ib32 * 32 + 0)),
                                          w1, vld1q_s8(xb.qs + ib32 * 32 + 16));
            sumi = vmlaq_n_s32(sumi, p, sc);
            qs += 8; signs += 4;
        }
        sumf = std::fma(static_cast<float>(vaddvq_s32(sumi)) * xb.d, f16_to_f32_fast(wb.d), sumf);
    }
    return sumf;
#else
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const BlockIQ3_S & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;
        const uint8_t * qs    = wb.qs;
        const uint8_t * qh    = wb.qh;
        const uint8_t * signs = wb.signs;
        int32_t sumi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t qhb = qh[ib32];
            const uint8_t  scb = wb.scales[ib32 >> 1];
            const int32_t  sc  = 1 + 2 * ((ib32 & 1) ? (scb >> 4) : (scb & 0xFu));
            int32_t ls = 0;
            for (int l = 0; l < 4; ++l) {
                const uint32_t g0 = iq3s_grid[qs[2 * l + 0] | ((qhb << (8 - 2 * l)) & 256u)];
                const uint32_t g1 = iq3s_grid[qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256u)];
                const uint8_t  sg = signs[l];
                for (int j = 0; j < 4; ++j) {
                    const int32_t m0 = static_cast<int32_t>((g0 >> (8 * j)) & 0xFFu);
                    const int32_t m1 = static_cast<int32_t>((g1 >> (8 * j)) & 0xFFu);
                    const int32_t x0 = static_cast<int32_t>(xb.qs[ib32 * 32 + l * 8 + j + 0]);
                    const int32_t x1 = static_cast<int32_t>(xb.qs[ib32 * 32 + l * 8 + j + 4]);
                    ls += (sg & (1u << (j + 0))) ? (-m0 * x0) : (m0 * x0);
                    ls += (sg & (1u << (j + 4))) ? (-m1 * x1) : (m1 * x1);
                }
            }
            sumi += sc * ls;
            qs += 8; signs += 4;
        }
        acc += d * static_cast<float>(sumi);
    }
    return acc;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
struct ArmUnpackBuf {
    std::vector<int8_t>  q;
    std::vector<int8_t>  sc8;
    std::vector<uint8_t> u8_sc;
    std::vector<uint8_t> u8_mn;
    std::vector<float>   d;
    std::vector<float>   dmin;
};

#if defined(__ARM_FEATURE_MATMUL_INT8)
// ARMv8.6 i8mm (smmla): computes 4x4 outer dot products over 16 int8 elements with 2x MAC/cycle and zero horizontal adds.
static inline void dot_4x4_i8_16_arm64(
    int8x16_t w0, int8x16_t w1, int8x16_t w2, int8x16_t w3,
    int8x16_t x0, int8x16_t x1, int8x16_t x2, int8x16_t x3,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    const int8x16_t w01_lo = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(w0), vreinterpretq_s64_s8(w1)));
    const int8x16_t w01_hi = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(w0), vreinterpretq_s64_s8(w1)));
    const int8x16_t w23_lo = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(w2), vreinterpretq_s64_s8(w3)));
    const int8x16_t w23_hi = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(w2), vreinterpretq_s64_s8(w3)));

    const int8x16_t x01_lo = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0), vreinterpretq_s64_s8(x1)));
    const int8x16_t x01_hi = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0), vreinterpretq_s64_s8(x1)));
    const int8x16_t x23_lo = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x2), vreinterpretq_s64_s8(x3)));
    const int8x16_t x23_hi = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x2), vreinterpretq_s64_s8(x3)));

    int32x4_t t00 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), w01_lo, x01_lo), w01_hi, x01_hi);
    int32x4_t t01 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), w01_lo, x23_lo), w01_hi, x23_hi);
    int32x4_t t10 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), w23_lo, x01_lo), w23_hi, x01_hi);
    int32x4_t t11 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), w23_lo, x23_lo), w23_hi, x23_hi);

    c0 = vcombine_s32(vget_low_s32(t00),  vget_low_s32(t01));
    c1 = vcombine_s32(vget_high_s32(t00), vget_high_s32(t01));
    c2 = vcombine_s32(vget_low_s32(t10),  vget_low_s32(t11));
    c3 = vcombine_s32(vget_high_s32(t10), vget_high_s32(t11));
}

static inline void dot_4x4_i8_32_arm64(
    int8x16_t w0a, int8x16_t w0b, int8x16_t w1a, int8x16_t w1b,
    int8x16_t w2a, int8x16_t w2b, int8x16_t w3a, int8x16_t w3b,
    int8x16_t x0a, int8x16_t x0b, int8x16_t x1a, int8x16_t x1b,
    int8x16_t x2a, int8x16_t x2b, int8x16_t x3a, int8x16_t x3b,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    int32x4_t a0, a1, a2, a3, b0, b1, b2, b3;
    dot_4x4_i8_16_arm64(w0a, w1a, w2a, w3a, x0a, x1a, x2a, x3a, a0, a1, a2, a3);
    dot_4x4_i8_16_arm64(w0b, w1b, w2b, w3b, x0b, x1b, x2b, x3b, b0, b1, b2, b3);
    c0 = vaddq_s32(a0, b0); c1 = vaddq_s32(a1, b1);
    c2 = vaddq_s32(a2, b2); c3 = vaddq_s32(a3, b3);
}
#else
static inline void dot_4x4_i8_16_arm64(
    int8x16_t w0, int8x16_t w1, int8x16_t w2, int8x16_t w3,
    int8x16_t x0, int8x16_t x1, int8x16_t x2, int8x16_t x3,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    auto row = [&](int8x16_t wv) -> int32x4_t {
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), wv, x0);
        int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), wv, x1);
        int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), wv, x2);
        int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), wv, x3);
        return vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
    };
    c0 = row(w0); c1 = row(w1); c2 = row(w2); c3 = row(w3);
}

static inline void dot_4x4_i8_32_arm64(
    int8x16_t w0a, int8x16_t w0b, int8x16_t w1a, int8x16_t w1b,
    int8x16_t w2a, int8x16_t w2b, int8x16_t w3a, int8x16_t w3b,
    int8x16_t x0a, int8x16_t x0b, int8x16_t x1a, int8x16_t x1b,
    int8x16_t x2a, int8x16_t x2b, int8x16_t x3a, int8x16_t x3b,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    auto row = [&](int8x16_t wa, int8x16_t wb) -> int32x4_t {
        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x0a), wb, x0b);
        int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x1a), wb, x1b);
        int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x2a), wb, x2b);
        int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x3a), wb, x3b);
        return vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
    };
    c0 = row(w0a, w0b); c1 = row(w1a, w1b); c2 = row(w2a, w2b); c3 = row(w3a, w3b);
}

// ============================================================================
// ARMv8.2-A DOTPROD Lane-Indexed 4x4 Prefill GEMM (Zero vpaddq_s32 + Zero 4K-Aliasing)
// Transposes 4 consecutive tokens into 4-byte groups so vdotq_laneq_s32 computes all 4
// token dots directly into one int32x4_t register with zero horizontal reductions.
// ============================================================================

struct alignas(64) ArmQ80Lane4 {
    float  d[4];
    float  s[4];
    int8_t qs[128]; // 8 x 16B: each 16B = {t0[4B], t1[4B], t2[4B], t3[4B]}
};

struct alignas(64) ArmQ8KLane4 {
    float   d[4];
    int16_t bsums8[4][8]; // pre-summed pairwise bsums[2j] + bsums[2j+1] for Q4_K
    int8_t  qs[8][128];   // [sb][k4_grp(0..7)*16]: each 16B = {t0[4B], t1[4B], t2[4B], t3[4B]}
};

static inline void transpose_4x4_i32_to_i8(
    const int8_t * __restrict__ s0, const int8_t * __restrict__ s1,
    const int8_t * __restrict__ s2, const int8_t * __restrict__ s3,
    int8_t * __restrict__ dst) noexcept {
    const int32x4_t r0 = vreinterpretq_s32_s8(vld1q_s8(s0));
    const int32x4_t r1 = vreinterpretq_s32_s8(vld1q_s8(s1));
    const int32x4_t r2 = vreinterpretq_s32_s8(vld1q_s8(s2));
    const int32x4_t r3 = vreinterpretq_s32_s8(vld1q_s8(s3));
    const int32x4_t t01a = vtrn1q_s32(r0, r1);
    const int32x4_t t01b = vtrn2q_s32(r0, r1);
    const int32x4_t t23a = vtrn1q_s32(r2, r3);
    const int32x4_t t23b = vtrn2q_s32(r2, r3);
    vst1q_s8(dst +  0, vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(t01a), vreinterpretq_s64_s32(t23a))));
    vst1q_s8(dst + 16, vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(t01b), vreinterpretq_s64_s32(t23b))));
    vst1q_s8(dst + 32, vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(t01a), vreinterpretq_s64_s32(t23a))));
    vst1q_s8(dst + 48, vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(t01b), vreinterpretq_s64_s32(t23b))));
}

static inline void pack_q8_0_lane4_range(
    const CpuActQ8_0 * __restrict__ xq8,
    ArmQ80Lane4 *      __restrict__ dst,
    int t4_0, int t4_1, int nb) noexcept {
    for (int t4 = t4_0; t4 < t4_1; ++t4) {
        const CpuActQ8_0 * x0 = xq8 + static_cast<size_t>(t4 * 4 + 0) * nb;
        const CpuActQ8_0 * x1 = xq8 + static_cast<size_t>(t4 * 4 + 1) * nb;
        const CpuActQ8_0 * x2 = xq8 + static_cast<size_t>(t4 * 4 + 2) * nb;
        const CpuActQ8_0 * x3 = xq8 + static_cast<size_t>(t4 * 4 + 3) * nb;
        ArmQ80Lane4 * dp = dst + static_cast<size_t>(t4) * nb;
        for (int b = 0; b < nb; ++b) {
            dp[b].d[0] = x0[b].d; dp[b].d[1] = x1[b].d; dp[b].d[2] = x2[b].d; dp[b].d[3] = x3[b].d;
            dp[b].s[0] = x0[b].s; dp[b].s[1] = x1[b].s; dp[b].s[2] = x2[b].s; dp[b].s[3] = x3[b].s;
            transpose_4x4_i32_to_i8(x0[b].qs +  0, x1[b].qs +  0, x2[b].qs +  0, x3[b].qs +  0, dp[b].qs +  0);
            transpose_4x4_i32_to_i8(x0[b].qs + 16, x1[b].qs + 16, x2[b].qs + 16, x3[b].qs + 16, dp[b].qs + 64);
        }
    }
}

static inline void pack_q8_k_lane4_range(
    const CpuActQ8_K * __restrict__ xq8k,
    ArmQ8KLane4 *      __restrict__ dst,
    int t4_0, int t4_1, int nb) noexcept {
    for (int t4 = t4_0; t4 < t4_1; ++t4) {
        const CpuActQ8_K * x0 = xq8k + static_cast<size_t>(t4 * 4 + 0) * nb;
        const CpuActQ8_K * x1 = xq8k + static_cast<size_t>(t4 * 4 + 1) * nb;
        const CpuActQ8_K * x2 = xq8k + static_cast<size_t>(t4 * 4 + 2) * nb;
        const CpuActQ8_K * x3 = xq8k + static_cast<size_t>(t4 * 4 + 3) * nb;
        ArmQ8KLane4 * dp = dst + static_cast<size_t>(t4) * nb;
        for (int b = 0; b < nb; ++b) {
            ArmQ8KLane4 & p = dp[b];
            p.d[0] = x0[b].d; p.d[1] = x1[b].d; p.d[2] = x2[b].d; p.d[3] = x3[b].d;
            vst1q_s16(p.bsums8[0], vpaddq_s16(vld1q_s16(x0[b].bsums + 0), vld1q_s16(x0[b].bsums + 8)));
            vst1q_s16(p.bsums8[1], vpaddq_s16(vld1q_s16(x1[b].bsums + 0), vld1q_s16(x1[b].bsums + 8)));
            vst1q_s16(p.bsums8[2], vpaddq_s16(vld1q_s16(x2[b].bsums + 0), vld1q_s16(x2[b].bsums + 8)));
            vst1q_s16(p.bsums8[3], vpaddq_s16(vld1q_s16(x3[b].bsums + 0), vld1q_s16(x3[b].bsums + 8)));
            for (int sb = 0; sb < 8; ++sb) {
                transpose_4x4_i32_to_i8(x0[b].qs + sb * 32 +  0, x1[b].qs + sb * 32 +  0,
                                        x2[b].qs + sb * 32 +  0, x3[b].qs + sb * 32 +  0, p.qs[sb] +  0);
                transpose_4x4_i32_to_i8(x0[b].qs + sb * 32 + 16, x1[b].qs + sb * 32 + 16,
                                        x2[b].qs + sb * 32 + 16, x3[b].qs + sb * 32 + 16, p.qs[sb] + 64);
            }
        }
    }
}

static inline void dot_4x4_lane_16_arm64(
    int8x16_t w0, int8x16_t w1, int8x16_t w2, int8x16_t w3,
    const int8_t * __restrict__ px,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    const int8x16_t x0 = vld1q_s8(px +  0);
    c0 = vdotq_laneq_s32(vdupq_n_s32(0), x0, w0, 0);
    c1 = vdotq_laneq_s32(vdupq_n_s32(0), x0, w1, 0);
    c2 = vdotq_laneq_s32(vdupq_n_s32(0), x0, w2, 0);
    c3 = vdotq_laneq_s32(vdupq_n_s32(0), x0, w3, 0);
    const int8x16_t x1 = vld1q_s8(px + 16);
    c0 = vdotq_laneq_s32(c0, x1, w0, 1); c1 = vdotq_laneq_s32(c1, x1, w1, 1);
    c2 = vdotq_laneq_s32(c2, x1, w2, 1); c3 = vdotq_laneq_s32(c3, x1, w3, 1);
    const int8x16_t x2 = vld1q_s8(px + 32);
    c0 = vdotq_laneq_s32(c0, x2, w0, 2); c1 = vdotq_laneq_s32(c1, x2, w1, 2);
    c2 = vdotq_laneq_s32(c2, x2, w2, 2); c3 = vdotq_laneq_s32(c3, x2, w3, 2);
    const int8x16_t x3 = vld1q_s8(px + 48);
    c0 = vdotq_laneq_s32(c0, x3, w0, 3); c1 = vdotq_laneq_s32(c1, x3, w1, 3);
    c2 = vdotq_laneq_s32(c2, x3, w2, 3); c3 = vdotq_laneq_s32(c3, x3, w3, 3);
}

static inline void dot_4x4_lane_32_arm64(
    int8x16_t w0a, int8x16_t w0b, int8x16_t w1a, int8x16_t w1b,
    int8x16_t w2a, int8x16_t w2b, int8x16_t w3a, int8x16_t w3b,
    const int8_t * __restrict__ px,
    int32x4_t & c0, int32x4_t & c1, int32x4_t & c2, int32x4_t & c3) noexcept {
    dot_4x4_lane_16_arm64(w0a, w1a, w2a, w3a, px + 0, c0, c1, c2, c3);
    const int8x16_t x4 = vld1q_s8(px +  64);
    c0 = vdotq_laneq_s32(c0, x4, w0b, 0); c1 = vdotq_laneq_s32(c1, x4, w1b, 0);
    c2 = vdotq_laneq_s32(c2, x4, w2b, 0); c3 = vdotq_laneq_s32(c3, x4, w3b, 0);
    const int8x16_t x5 = vld1q_s8(px +  80);
    c0 = vdotq_laneq_s32(c0, x5, w0b, 1); c1 = vdotq_laneq_s32(c1, x5, w1b, 1);
    c2 = vdotq_laneq_s32(c2, x5, w2b, 1); c3 = vdotq_laneq_s32(c3, x5, w3b, 1);
    const int8x16_t x6 = vld1q_s8(px +  96);
    c0 = vdotq_laneq_s32(c0, x6, w0b, 2); c1 = vdotq_laneq_s32(c1, x6, w1b, 2);
    c2 = vdotq_laneq_s32(c2, x6, w2b, 2); c3 = vdotq_laneq_s32(c3, x6, w3b, 2);
    const int8x16_t x7 = vld1q_s8(px + 112);
    c0 = vdotq_laneq_s32(c0, x7, w0b, 3); c1 = vdotq_laneq_s32(c1, x7, w1b, 3);
    c2 = vdotq_laneq_s32(c2, x7, w2b, 3); c3 = vdotq_laneq_s32(c3, x7, w3b, 3);
}

inline void gemm_q8_0_lane_4x4_arm64(
    const BlockQ8_0 * __restrict__ r0, const BlockQ8_0 * __restrict__ r1,
    const BlockQ8_0 * __restrict__ r2, const BlockQ8_0 * __restrict__ r3,
    const ArmQ80Lane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 1) == 0 && b + 4 < nb) {
            __builtin_prefetch(&r0[b + 4], 0, 0); __builtin_prefetch(&r1[b + 4], 0, 0);
            __builtin_prefetch(&r2[b + 4], 0, 0); __builtin_prefetch(&r3[b + 4], 0, 0);
        }
        int32x4_t c0, c1, c2, c3;
        dot_4x4_lane_32_arm64(
            vld1q_s8(r0[b].qs), vld1q_s8(r0[b].qs + 16),
            vld1q_s8(r1[b].qs), vld1q_s8(r1[b].qs + 16),
            vld1q_s8(r2[b].qs), vld1q_s8(r2[b].qs + 16),
            vld1q_s8(r3[b].qs), vld1q_s8(r3[b].qs + 16),
            xp[b].qs, c0, c1, c2, c3);
        const float32x4_t vxd = vld1q_f32(xp[b].d);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void gemm_q4_0_lane_4x4_arm64(
    const BlockQ4_0 * __restrict__ r0, const BlockQ4_0 * __restrict__ r1,
    const BlockQ4_0 * __restrict__ r2, const BlockQ4_0 * __restrict__ r3,
    const ArmQ80Lane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        const uint8x16_t q0 = vld1q_u8(r0[b].qs), q1 = vld1q_u8(r1[b].qs);
        const uint8x16_t q2 = vld1q_u8(r2[b].qs), q3 = vld1q_u8(r3[b].qs);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_lane_32_arm64(
            vreinterpretq_s8_u8(vandq_u8(q0, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
            vreinterpretq_s8_u8(vandq_u8(q1, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
            vreinterpretq_s8_u8(vandq_u8(q2, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)),
            vreinterpretq_s8_u8(vandq_u8(q3, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)),
            xp[b].qs, c0, c1, c2, c3);
        const float32x4_t vxd = vld1q_f32(xp[b].d);
        const float32x4_t vxs = vld1q_f32(xp[b].s);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
        s0 = vfmaq_f32(s0, vxs, vdupq_laneq_f32(wsc, 0));
        s1 = vfmaq_f32(s1, vxs, vdupq_laneq_f32(wsc, 1));
        s2 = vfmaq_f32(s2, vxs, vdupq_laneq_f32(wsc, 2));
        s3 = vfmaq_f32(s3, vxs, vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], vfmsq_n_f32(a0, s0, 8.0f));
    vst1q_f32(out[1], vfmsq_n_f32(a1, s1, 8.0f));
    vst1q_f32(out[2], vfmsq_n_f32(a2, s2, 8.0f));
    vst1q_f32(out[3], vfmsq_n_f32(a3, s3, 8.0f));
}

inline void gemm_q5_0_lane_4x4_arm64(
    const BlockQ5_0 * __restrict__ r0, const BlockQ5_0 * __restrict__ r1,
    const BlockQ5_0 * __restrict__ r2, const BlockQ5_0 * __restrict__ r3,
    const ArmQ80Lane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t v16 = vdupq_n_u8(16);
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    auto unpack = [&](const BlockQ5_0 & wb, int8x16_t & lo, int8x16_t & hi) {
        const uint8x16_t qs  = vld1q_u8(wb.qs);
        const uint8x16_t h01 = vcombine_u8(vdup_n_u8(wb.qh[0]), vdup_n_u8(wb.qh[1]));
        const uint8x16_t h23 = vcombine_u8(vdup_n_u8(wb.qh[2]), vdup_n_u8(wb.qh[3]));
        lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m4), vandq_u8(vtstq_u8(h01, vmask), v16)));
        hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), vandq_u8(vtstq_u8(h23, vmask), v16)));
    };
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        int8x16_t w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b;
        unpack(r0[b], w0a, w0b); unpack(r1[b], w1a, w1b);
        unpack(r2[b], w2a, w2b); unpack(r3[b], w3a, w3b);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_lane_32_arm64(w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b, xp[b].qs, c0, c1, c2, c3);
        const float32x4_t vxd = vld1q_f32(xp[b].d);
        const float32x4_t vxs = vld1q_f32(xp[b].s);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
        s0 = vfmaq_f32(s0, vxs, vdupq_laneq_f32(wsc, 0));
        s1 = vfmaq_f32(s1, vxs, vdupq_laneq_f32(wsc, 1));
        s2 = vfmaq_f32(s2, vxs, vdupq_laneq_f32(wsc, 2));
        s3 = vfmaq_f32(s3, vxs, vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], vfmsq_n_f32(a0, s0, 16.0f));
    vst1q_f32(out[1], vfmsq_n_f32(a1, s1, 16.0f));
    vst1q_f32(out[2], vfmsq_n_f32(a2, s2, 16.0f));
    vst1q_f32(out[3], vfmsq_n_f32(a3, s3, 16.0f));
}

inline void gemm_iq4_nl_lane_4x4_arm64(
    const BlockIQ4_NL * __restrict__ r0, const BlockIQ4_NL * __restrict__ r1,
    const BlockIQ4_NL * __restrict__ r2, const BlockIQ4_NL * __restrict__ r3,
    const ArmQ80Lane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const int8x16_t  lut = vld1q_s8(kvalues_iq4nl);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        const uint8x16_t q0 = vld1q_u8(r0[b].qs), q1 = vld1q_u8(r1[b].qs);
        const uint8x16_t q2 = vld1q_u8(r2[b].qs), q3 = vld1q_u8(r3[b].qs);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_lane_32_arm64(
            vqtbl1q_s8(lut, vandq_u8(q0, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q0, 4)),
            vqtbl1q_s8(lut, vandq_u8(q1, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q1, 4)),
            vqtbl1q_s8(lut, vandq_u8(q2, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q2, 4)),
            vqtbl1q_s8(lut, vandq_u8(q3, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q3, 4)),
            xp[b].qs, c0, c1, c2, c3);
        const float32x4_t vxd = vld1q_f32(xp[b].d);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void gemm_q6_k_lane_4x4_arm64(
    const int8_t * __restrict__ wq0, const int8_t * __restrict__ sc0, const float * __restrict__ wd0,
    const int8_t * __restrict__ wq1, const int8_t * __restrict__ sc1, const float * __restrict__ wd1,
    const int8_t * __restrict__ wq2, const int8_t * __restrict__ sc2, const float * __restrict__ wd2,
    const int8_t * __restrict__ wq3, const int8_t * __restrict__ sc3, const float * __restrict__ wd3,
    const ArmQ8KLane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t i0 = vdupq_n_s32(0), i1 = vdupq_n_s32(0);
        int32x4_t i2 = vdupq_n_s32(0), i3 = vdupq_n_s32(0);
        const int off = b * 256, soff = b * 16;
        const ArmQ8KLane4 & pb = xp[b];
        for (int sb = 0; sb < 8; ++sb) {
            const int s0 = sb * 2;
            const int s1 = sb * 2 + 1;
            int32x4_t c0, c1, c2, c3;
            dot_4x4_lane_16_arm64(
                vld1q_s8(wq0 + off + s0 * 16), vld1q_s8(wq1 + off + s0 * 16),
                vld1q_s8(wq2 + off + s0 * 16), vld1q_s8(wq3 + off + s0 * 16),
                pb.qs[sb] + 0, c0, c1, c2, c3);
            i0 = vmlaq_n_s32(i0, c0, sc0[soff + s0]);
            i1 = vmlaq_n_s32(i1, c1, sc1[soff + s0]);
            i2 = vmlaq_n_s32(i2, c2, sc2[soff + s0]);
            i3 = vmlaq_n_s32(i3, c3, sc3[soff + s0]);

            dot_4x4_lane_16_arm64(
                vld1q_s8(wq0 + off + s1 * 16), vld1q_s8(wq1 + off + s1 * 16),
                vld1q_s8(wq2 + off + s1 * 16), vld1q_s8(wq3 + off + s1 * 16),
                pb.qs[sb] + 64, c0, c1, c2, c3);
            i0 = vmlaq_n_s32(i0, c0, sc0[soff + s1]);
            i1 = vmlaq_n_s32(i1, c1, sc1[soff + s1]);
            i2 = vmlaq_n_s32(i2, c2, sc2[soff + s1]);
            i3 = vmlaq_n_s32(i3, c3, sc3[soff + s1]);
        }
        const float32x4_t vxd = vld1q_f32(pb.d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(i0), vxd), vdupq_n_f32(wd0[b]));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(i1), vxd), vdupq_n_f32(wd1[b]));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(i2), vxd), vdupq_n_f32(wd2[b]));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(i3), vxd), vdupq_n_f32(wd3[b]));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void gemm_q4_k_lane_4x4_arm64(
    const int8_t * __restrict__ wq0, const uint8_t * __restrict__ sc0, const uint8_t * __restrict__ mn0, const float * __restrict__ wd0, const float * __restrict__ wm0,
    const int8_t * __restrict__ wq1, const uint8_t * __restrict__ sc1, const uint8_t * __restrict__ mn1, const float * __restrict__ wd1, const float * __restrict__ wm1,
    const int8_t * __restrict__ wq2, const uint8_t * __restrict__ sc2, const uint8_t * __restrict__ mn2, const float * __restrict__ wd2, const float * __restrict__ wm2,
    const int8_t * __restrict__ wq3, const uint8_t * __restrict__ sc3, const uint8_t * __restrict__ mn3, const float * __restrict__ wd3, const float * __restrict__ wm3,
    const ArmQ8KLane4 * __restrict__ xp, float out[4][4], int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t i0 = vdupq_n_s32(0), i1 = vdupq_n_s32(0);
        int32x4_t i2 = vdupq_n_s32(0), i3 = vdupq_n_s32(0);
        const int off = b * 256, soff = b * 8;
        const ArmQ8KLane4 & pb = xp[b];
        for (int sb = 0; sb < 8; ++sb) {
            int32x4_t c0, c1, c2, c3;
            dot_4x4_lane_32_arm64(
                vld1q_s8(wq0 + off + sb * 32), vld1q_s8(wq0 + off + sb * 32 + 16),
                vld1q_s8(wq1 + off + sb * 32), vld1q_s8(wq1 + off + sb * 32 + 16),
                vld1q_s8(wq2 + off + sb * 32), vld1q_s8(wq2 + off + sb * 32 + 16),
                vld1q_s8(wq3 + off + sb * 32), vld1q_s8(wq3 + off + sb * 32 + 16),
                pb.qs[sb], c0, c1, c2, c3);
            i0 = vmlaq_n_s32(i0, c0, sc0[soff + sb]);
            i1 = vmlaq_n_s32(i1, c1, sc1[soff + sb]);
            i2 = vmlaq_n_s32(i2, c2, sc2[soff + sb]);
            i3 = vmlaq_n_s32(i3, c3, sc3[soff + sb]);
        }
        const int16x8_t bs0 = vld1q_s16(pb.bsums8[0]);
        const int16x8_t bs1 = vld1q_s16(pb.bsums8[1]);
        const int16x8_t bs2 = vld1q_s16(pb.bsums8[2]);
        const int16x8_t bs3 = vld1q_s16(pb.bsums8[3]);
        const float32x4_t vxd = vld1q_f32(pb.d);
        auto fin = [&](int32x4_t sp, const uint8_t * mn, float wd, float wm, float32x4_t & acc) {
            const int16x8_t m16 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn)));
            const int16x4_t mlo = vget_low_s16(m16), mhi = vget_high_s16(m16);
            const int32x4_t sm0 = vmlal_s16(vmull_s16(vget_low_s16(bs0), mlo), vget_high_s16(bs0), mhi);
            const int32x4_t sm1 = vmlal_s16(vmull_s16(vget_low_s16(bs1), mlo), vget_high_s16(bs1), mhi);
            const int32x4_t sm2 = vmlal_s16(vmull_s16(vget_low_s16(bs2), mlo), vget_high_s16(bs2), mhi);
            const int32x4_t sm3 = vmlal_s16(vmull_s16(vget_low_s16(bs3), mlo), vget_high_s16(bs3), mhi);
            const int32x4_t sm  = vpaddq_s32(vpaddq_s32(sm0, sm1), vpaddq_s32(sm2, sm3));
            const float32x4_t val = vfmsq_f32(vmulq_n_f32(vcvtq_f32_s32(sp), wd), vcvtq_f32_s32(sm), vdupq_n_f32(wm));
            acc = vfmaq_f32(acc, val, vxd);
        };
        fin(i0, mn0 + soff, wd0[b], wm0[b], a0);
        fin(i1, mn1 + soff, wd1[b], wm1[b], a1);
        fin(i2, mn2 + soff, wd2[b], wm2[b], a2);
        fin(i3, mn3 + soff, wd3[b], wm3[b], a3);
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}
#endif

inline void gemm_q8_0_4x4_arm64(
    const BlockQ8_0 * __restrict__ r0,
    const BlockQ8_0 * __restrict__ r1,
    const BlockQ8_0 * __restrict__ r2,
    const BlockQ8_0 * __restrict__ r3,
    const CpuActQ8_0 * __restrict__ x0,
    const CpuActQ8_0 * __restrict__ x1,
    const CpuActQ8_0 * __restrict__ x2,
    const CpuActQ8_0 * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 1) == 0 && b + 4 < nb) {
            __builtin_prefetch(&r0[b + 4], 0, 0);
            __builtin_prefetch(&r1[b + 4], 0, 0);
            __builtin_prefetch(&r2[b + 4], 0, 0);
            __builtin_prefetch(&r3[b + 4], 0, 0);
        }
        const int8x16_t t0_lo = vld1q_s8(x0[b].qs), t0_hi = vld1q_s8(x0[b].qs + 16);
        const int8x16_t t1_lo = vld1q_s8(x1[b].qs), t1_hi = vld1q_s8(x1[b].qs + 16);
        const int8x16_t t2_lo = vld1q_s8(x2[b].qs), t2_hi = vld1q_s8(x2[b].qs + 16);
        const int8x16_t t3_lo = vld1q_s8(x3[b].qs), t3_hi = vld1q_s8(x3[b].qs + 16);

        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float32x4_t vxd = vld1q_f32(xd_arr);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);

        int32x4_t c0, c1, c2, c3;
        dot_4x4_i8_32_arm64(
            vld1q_s8(r0[b].qs), vld1q_s8(r0[b].qs + 16),
            vld1q_s8(r1[b].qs), vld1q_s8(r1[b].qs + 16),
            vld1q_s8(r2[b].qs), vld1q_s8(r2[b].qs + 16),
            vld1q_s8(r3[b].qs), vld1q_s8(r3[b].qs + 16),
            t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi,
            c0, c1, c2, c3);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void gemm_q4_0_4x4_arm64(
    const BlockQ4_0 * __restrict__ r0,
    const BlockQ4_0 * __restrict__ r1,
    const BlockQ4_0 * __restrict__ r2,
    const BlockQ4_0 * __restrict__ r3,
    const CpuActQ8_0 * __restrict__ x0,
    const CpuActQ8_0 * __restrict__ x1,
    const CpuActQ8_0 * __restrict__ x2,
    const CpuActQ8_0 * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        const uint8x16_t q0 = vld1q_u8(r0[b].qs), q1 = vld1q_u8(r1[b].qs);
        const uint8x16_t q2 = vld1q_u8(r2[b].qs), q3 = vld1q_u8(r3[b].qs);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_i8_32_arm64(
            vreinterpretq_s8_u8(vandq_u8(q0, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
            vreinterpretq_s8_u8(vandq_u8(q1, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
            vreinterpretq_s8_u8(vandq_u8(q2, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)),
            vreinterpretq_s8_u8(vandq_u8(q3, m4)), vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)),
            vld1q_s8(x0[b].qs), vld1q_s8(x0[b].qs + 16),
            vld1q_s8(x1[b].qs), vld1q_s8(x1[b].qs + 16),
            vld1q_s8(x2[b].qs), vld1q_s8(x2[b].qs + 16),
            vld1q_s8(x3[b].qs), vld1q_s8(x3[b].qs + 16),
            c0, c1, c2, c3);
        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float xs_arr[4] = {x0[b].s, x1[b].s, x2[b].s, x3[b].s};
        const float32x4_t vxd = vld1q_f32(xd_arr);
        const float32x4_t vxs = vld1q_f32(xs_arr);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
        s0 = vfmaq_f32(s0, vxs, vdupq_laneq_f32(wsc, 0));
        s1 = vfmaq_f32(s1, vxs, vdupq_laneq_f32(wsc, 1));
        s2 = vfmaq_f32(s2, vxs, vdupq_laneq_f32(wsc, 2));
        s3 = vfmaq_f32(s3, vxs, vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], vfmsq_n_f32(a0, s0, 8.0f));
    vst1q_f32(out[1], vfmsq_n_f32(a1, s1, 8.0f));
    vst1q_f32(out[2], vfmsq_n_f32(a2, s2, 8.0f));
    vst1q_f32(out[3], vfmsq_n_f32(a3, s3, 8.0f));
}

inline void gemm_q5_0_4x4_arm64(
    const BlockQ5_0 * __restrict__ r0,
    const BlockQ5_0 * __restrict__ r1,
    const BlockQ5_0 * __restrict__ r2,
    const BlockQ5_0 * __restrict__ r3,
    const CpuActQ8_0 * __restrict__ x0,
    const CpuActQ8_0 * __restrict__ x1,
    const CpuActQ8_0 * __restrict__ x2,
    const CpuActQ8_0 * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t v16 = vdupq_n_u8(16);
    static constexpr uint8_t kBitMask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vmask = vld1q_u8(kBitMask);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    auto unpack = [&](const BlockQ5_0 & wb, int8x16_t & lo, int8x16_t & hi) {
        const uint8x16_t qs  = vld1q_u8(wb.qs);
        const uint8x16_t h01 = vcombine_u8(vdup_n_u8(wb.qh[0]), vdup_n_u8(wb.qh[1]));
        const uint8x16_t h23 = vcombine_u8(vdup_n_u8(wb.qh[2]), vdup_n_u8(wb.qh[3]));
        lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m4), vandq_u8(vtstq_u8(h01, vmask), v16)));
        hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), vandq_u8(vtstq_u8(h23, vmask), v16)));
    };
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        int8x16_t w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b;
        unpack(r0[b], w0a, w0b); unpack(r1[b], w1a, w1b);
        unpack(r2[b], w2a, w2b); unpack(r3[b], w3a, w3b);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_i8_32_arm64(
            w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b,
            vld1q_s8(x0[b].qs), vld1q_s8(x0[b].qs + 16),
            vld1q_s8(x1[b].qs), vld1q_s8(x1[b].qs + 16),
            vld1q_s8(x2[b].qs), vld1q_s8(x2[b].qs + 16),
            vld1q_s8(x3[b].qs), vld1q_s8(x3[b].qs + 16),
            c0, c1, c2, c3);
        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float xs_arr[4] = {x0[b].s, x1[b].s, x2[b].s, x3[b].s};
        const float32x4_t vxd = vld1q_f32(xd_arr);
        const float32x4_t vxs = vld1q_f32(xs_arr);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
        s0 = vfmaq_f32(s0, vxs, vdupq_laneq_f32(wsc, 0));
        s1 = vfmaq_f32(s1, vxs, vdupq_laneq_f32(wsc, 1));
        s2 = vfmaq_f32(s2, vxs, vdupq_laneq_f32(wsc, 2));
        s3 = vfmaq_f32(s3, vxs, vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], vfmsq_n_f32(a0, s0, 16.0f));
    vst1q_f32(out[1], vfmsq_n_f32(a1, s1, 16.0f));
    vst1q_f32(out[2], vfmsq_n_f32(a2, s2, 16.0f));
    vst1q_f32(out[3], vfmsq_n_f32(a3, s3, 16.0f));
}

inline void gemm_iq4_nl_4x4_arm64(
    const BlockIQ4_NL * __restrict__ r0,
    const BlockIQ4_NL * __restrict__ r1,
    const BlockIQ4_NL * __restrict__ r2,
    const BlockIQ4_NL * __restrict__ r3,
    const CpuActQ8_0 * __restrict__ x0,
    const CpuActQ8_0 * __restrict__ x1,
    const CpuActQ8_0 * __restrict__ x2,
    const CpuActQ8_0 * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const int8x16_t  lut = vld1q_s8(kvalues_iq4nl);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        if ((b & 3) == 0 && b + 8 < nb) {
            __builtin_prefetch(&r0[b + 8], 0, 0); __builtin_prefetch(&r1[b + 8], 0, 0);
            __builtin_prefetch(&r2[b + 8], 0, 0); __builtin_prefetch(&r3[b + 8], 0, 0);
        }
        const uint8x16_t q0 = vld1q_u8(r0[b].qs), q1 = vld1q_u8(r1[b].qs);
        const uint8x16_t q2 = vld1q_u8(r2[b].qs), q3 = vld1q_u8(r3[b].qs);
        int32x4_t c0, c1, c2, c3;
        dot_4x4_i8_32_arm64(
            vqtbl1q_s8(lut, vandq_u8(q0, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q0, 4)),
            vqtbl1q_s8(lut, vandq_u8(q1, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q1, 4)),
            vqtbl1q_s8(lut, vandq_u8(q2, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q2, 4)),
            vqtbl1q_s8(lut, vandq_u8(q3, m4)), vqtbl1q_s8(lut, vshrq_n_u8(q3, 4)),
            vld1q_s8(x0[b].qs), vld1q_s8(x0[b].qs + 16),
            vld1q_s8(x1[b].qs), vld1q_s8(x1[b].qs + 16),
            vld1q_s8(x2[b].qs), vld1q_s8(x2[b].qs + 16),
            vld1q_s8(x3[b].qs), vld1q_s8(x3[b].qs + 16),
            c0, c1, c2, c3);
        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float32x4_t vxd = vld1q_f32(xd_arr);
        const float32x4_t wsc = load_4_fp16_scales(r0[b].d, r1[b].d, r2[b].d, r3[b].d);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(c0), vxd), vdupq_laneq_f32(wsc, 0));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(c1), vxd), vdupq_laneq_f32(wsc, 1));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(c2), vxd), vdupq_laneq_f32(wsc, 2));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(c3), vxd), vdupq_laneq_f32(wsc, 3));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void unpack_q6_k_row_i8(
    const BlockQ6_K * __restrict__ w,
    int8_t *          __restrict__ dst_q,
    int8_t *          __restrict__ dst_sc,
    float *           __restrict__ dst_d,
    int nb) noexcept {
    const uint8x16_t m4  = vdupq_n_u8(0x0Fu);
    const uint8x16_t m30 = vdupq_n_u8(0x30u);
    const int8x16_t  b32 = vdupq_n_s8(32);
    for (int b = 0; b < nb; ++b) {
        dst_d[b] = f16_to_f32_fast(w[b].d);
        vst1q_s8(dst_sc + b * 16, vld1q_s8(w[b].scales));
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = w[b].ql + half * 64;
            const uint8_t * qh = w[b].qh + half * 32;
            int8_t * dq = dst_q + b * 256 + half * 128;
            const uint8x16_t ql0 = vld1q_u8(ql +  0);
            const uint8x16_t ql1 = vld1q_u8(ql + 16);
            const uint8x16_t ql2 = vld1q_u8(ql + 32);
            const uint8x16_t ql3 = vld1q_u8(ql + 48);
            const uint8x16_t qh0 = vld1q_u8(qh +  0);
            const uint8x16_t qh1 = vld1q_u8(qh + 16);

            vst1q_s8(dq +   0, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh0, 4), vandq_u8(ql0, m4))), b32));
            vst1q_s8(dq +  16, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh1, 4), vandq_u8(ql1, m4))), b32));
            vst1q_s8(dq +  32, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh0, 2), vandq_u8(ql2, m4))), b32));
            vst1q_s8(dq +  48, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshlq_n_u8(qh1, 2), vandq_u8(ql3, m4))), b32));
            vst1q_s8(dq +  64, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, qh0,                vshrq_n_u8(ql0, 4))), b32));
            vst1q_s8(dq +  80, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, qh1,                vshrq_n_u8(ql1, 4))), b32));
            vst1q_s8(dq +  96, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshrq_n_u8(qh0, 2), vshrq_n_u8(ql2, 4))), b32));
            vst1q_s8(dq + 112, vsubq_s8(vreinterpretq_s8_u8(vbslq_u8(m30, vshrq_n_u8(qh1, 2), vshrq_n_u8(ql3, 4))), b32));
        }
    }
}

inline void gemm_q6_k_unpacked_1x4(
    const int8_t *     __restrict__ wq,
    const int8_t *     __restrict__ wsc,
    const float *      __restrict__ wd,
    const CpuActQ8_K * __restrict__ x0,
    const CpuActQ8_K * __restrict__ x1,
    const CpuActQ8_K * __restrict__ x2,
    const CpuActQ8_K * __restrict__ x3,
    float *            __restrict__ out,
    int nb) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t isum = vdupq_n_s32(0);
        const int8_t * wb = wq + b * 256;
        const int8_t * sb = wsc + b * 16;
        const int8_t * q0 = x0[b].qs;
        const int8_t * q1 = x1[b].qs;
        const int8_t * q2 = x2[b].qs;
        const int8_t * q3 = x3[b].qs;
        for (int s = 0; s < 16; ++s) {
            const int8x16_t wv = vld1q_s8(wb + s * 16);
            int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), wv, vld1q_s8(q0 + s * 16));
            int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), wv, vld1q_s8(q1 + s * 16));
            int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), wv, vld1q_s8(q2 + s * 16));
            int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), wv, vld1q_s8(q3 + s * 16));
            const int32x4_t dots = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
            isum = vmlaq_n_s32(isum, dots, static_cast<int32_t>(sb[s]));
        }
        const float d = wd[b];
        const float da[4] = {x0[b].d * d, x1[b].d * d, x2[b].d * d, x3[b].d * d};
        acc = vfmaq_f32(acc, vcvtq_f32_s32(isum), vld1q_f32(da));
    }
    vst1q_f32(out, acc);
}

inline void gemm_q6_k_unpacked_4x4(
    const int8_t *     __restrict__ wq0, const int8_t * __restrict__ sc0, const float * __restrict__ wd0,
    const int8_t *     __restrict__ wq1, const int8_t * __restrict__ sc1, const float * __restrict__ wd1,
    const int8_t *     __restrict__ wq2, const int8_t * __restrict__ sc2, const float * __restrict__ wd2,
    const int8_t *     __restrict__ wq3, const int8_t * __restrict__ sc3, const float * __restrict__ wd3,
    const CpuActQ8_K * __restrict__ x0,
    const CpuActQ8_K * __restrict__ x1,
    const CpuActQ8_K * __restrict__ x2,
    const CpuActQ8_K * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t i0 = vdupq_n_s32(0), i1 = vdupq_n_s32(0);
        int32x4_t i2 = vdupq_n_s32(0), i3 = vdupq_n_s32(0);
        const int off  = b * 256;
        const int soff = b * 16;
        const int8_t * q0 = x0[b].qs, * q1 = x1[b].qs, * q2 = x2[b].qs, * q3 = x3[b].qs;
        for (int s = 0; s < 16; ++s) {
            const int8x16_t xv0 = vld1q_s8(q0 + s * 16);
            const int8x16_t xv1 = vld1q_s8(q1 + s * 16);
            const int8x16_t xv2 = vld1q_s8(q2 + s * 16);
            const int8x16_t xv3 = vld1q_s8(q3 + s * 16);

#if defined(__ARM_FEATURE_MATMUL_INT8)
            int32x4_t c0, c1, c2, c3;
            dot_4x4_i8_16_arm64(
                vld1q_s8(wq0 + off + s * 16), vld1q_s8(wq1 + off + s * 16),
                vld1q_s8(wq2 + off + s * 16), vld1q_s8(wq3 + off + s * 16),
                xv0, xv1, xv2, xv3, c0, c1, c2, c3);
            i0 = vmlaq_n_s32(i0, c0, sc0[soff + s]);
            i1 = vmlaq_n_s32(i1, c1, sc1[soff + s]);
            i2 = vmlaq_n_s32(i2, c2, sc2[soff + s]);
            i3 = vmlaq_n_s32(i3, c3, sc3[soff + s]);
#else
            // Stream 1 weight row at a time so live NEON registers stay <= 18 (zero stack spills on A76/A55)
            auto step_row = [&](const int8_t * wp, int32_t sc, int32x4_t & acc_i) {
                const int8x16_t wv = vld1q_s8(wp);
                int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), wv, xv0);
                int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), wv, xv1);
                int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), wv, xv2);
                int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), wv, xv3);
                acc_i = vmlaq_n_s32(acc_i, vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)), sc);
            };
            step_row(wq0 + off + s * 16, sc0[soff + s], i0);
            step_row(wq1 + off + s * 16, sc1[soff + s], i1);
            step_row(wq2 + off + s * 16, sc2[soff + s], i2);
            step_row(wq3 + off + s * 16, sc3[soff + s], i3);
#endif
        }
        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float32x4_t vxd = vld1q_f32(xd_arr);
        a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(i0), vxd), vdupq_n_f32(wd0[b]));
        a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(i1), vxd), vdupq_n_f32(wd1[b]));
        a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(i2), vxd), vdupq_n_f32(wd2[b]));
        a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(i3), vxd), vdupq_n_f32(wd3[b]));
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void unpack_q4_k_row_i8(
    const BlockQ4_K * __restrict__ w,
    int8_t *          __restrict__ dst_q,
    uint8_t *         __restrict__ dst_sc,
    uint8_t *         __restrict__ dst_mn,
    float *           __restrict__ dst_d,
    float *           __restrict__ dst_dmin,
    int nb) noexcept {
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    constexpr uint32_t kmask3 = 0x03030303u;
    for (int b = 0; b < nb; ++b) {
        dst_d[b]    = f16_to_f32_fast(w[b].d);
        dst_dmin[b] = f16_to_f32_fast(w[b].dmin);
        uint32_t u0, u1, u2;
        std::memcpy(&u0, w[b].scales + 0, 4);
        std::memcpy(&u1, w[b].scales + 4, 4);
        std::memcpy(&u2, w[b].scales + 8, 4);
        const uint32_t sc_lo4 = u0 & kmask1;
        const uint32_t mn_lo4 = u1 & kmask1;
        const uint32_t sc_hi4 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
        const uint32_t mn_hi4 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
        std::memcpy(dst_sc + b * 8 + 0, &sc_lo4, 4);
        std::memcpy(dst_sc + b * 8 + 4, &sc_hi4, 4);
        std::memcpy(dst_mn + b * 8 + 0, &mn_lo4, 4);
        std::memcpy(dst_mn + b * 8 + 4, &mn_hi4, 4);
        int8_t * dq = dst_q + b * 256;
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t q0 = vld1q_u8(w[b].qs + j * 32 +  0);
            const uint8x16_t q1 = vld1q_u8(w[b].qs + j * 32 + 16);
            vst1q_s8(dq + j * 64 +  0, vreinterpretq_s8_u8(vandq_u8(q0, m4)));
            vst1q_s8(dq + j * 64 + 16, vreinterpretq_s8_u8(vandq_u8(q1, m4)));
            vst1q_s8(dq + j * 64 + 32, vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)));
            vst1q_s8(dq + j * 64 + 48, vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)));
        }
    }
}

inline void gemm_q4_k_unpacked_4x4(
    const int8_t *  __restrict__ wq0, const uint8_t * __restrict__ sc0, const uint8_t * __restrict__ mn0, const float * __restrict__ wd0, const float * __restrict__ wm0,
    const int8_t *  __restrict__ wq1, const uint8_t * __restrict__ sc1, const uint8_t * __restrict__ mn1, const float * __restrict__ wd1, const float * __restrict__ wm1,
    const int8_t *  __restrict__ wq2, const uint8_t * __restrict__ sc2, const uint8_t * __restrict__ mn2, const float * __restrict__ wd2, const float * __restrict__ wm2,
    const int8_t *  __restrict__ wq3, const uint8_t * __restrict__ sc3, const uint8_t * __restrict__ mn3, const float * __restrict__ wd3, const float * __restrict__ wm3,
    const CpuActQ8_K * __restrict__ x0,
    const CpuActQ8_K * __restrict__ x1,
    const CpuActQ8_K * __restrict__ x2,
    const CpuActQ8_K * __restrict__ x3,
    float out[4][4],
    int nb) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);

    for (int b = 0; b < nb; ++b) {
        int32x4_t i0 = vdupq_n_s32(0), i1 = vdupq_n_s32(0);
        int32x4_t i2 = vdupq_n_s32(0), i3 = vdupq_n_s32(0);
        const int off  = b * 256;
        const int soff = b * 8;
        const int8_t * q0 = x0[b].qs, * q1 = x1[b].qs, * q2 = x2[b].qs, * q3 = x3[b].qs;

        for (int sb = 0; sb < 8; ++sb) {
            const int8x16_t x0a = vld1q_s8(q0 + sb * 32), x0b = vld1q_s8(q0 + sb * 32 + 16);
            const int8x16_t x1a = vld1q_s8(q1 + sb * 32), x1b = vld1q_s8(q1 + sb * 32 + 16);
            const int8x16_t x2a = vld1q_s8(q2 + sb * 32), x2b = vld1q_s8(q2 + sb * 32 + 16);
            const int8x16_t x3a = vld1q_s8(q3 + sb * 32), x3b = vld1q_s8(q3 + sb * 32 + 16);

#if defined(__ARM_FEATURE_MATMUL_INT8)
            int32x4_t c0, c1, c2, c3;
            dot_4x4_i8_32_arm64(
                vld1q_s8(wq0 + off + sb * 32), vld1q_s8(wq0 + off + sb * 32 + 16),
                vld1q_s8(wq1 + off + sb * 32), vld1q_s8(wq1 + off + sb * 32 + 16),
                vld1q_s8(wq2 + off + sb * 32), vld1q_s8(wq2 + off + sb * 32 + 16),
                vld1q_s8(wq3 + off + sb * 32), vld1q_s8(wq3 + off + sb * 32 + 16),
                x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, c0, c1, c2, c3);
            i0 = vmlaq_n_s32(i0, c0, sc0[soff + sb]);
            i1 = vmlaq_n_s32(i1, c1, sc1[soff + sb]);
            i2 = vmlaq_n_s32(i2, c2, sc2[soff + sb]);
            i3 = vmlaq_n_s32(i3, c3, sc3[soff + sb]);
#else
            // Stream 1 weight row (2 vectors) at a time -> keeps live V-regs <= 22 (zero stack spills!)
            auto step_row = [&](const int8_t * wp, int32_t sc, int32x4_t & acc_i) {
                const int8x16_t wa = vld1q_s8(wp);
                const int8x16_t wb = vld1q_s8(wp + 16);
                int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x0a), wb, x0b);
                int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x1a), wb, x1b);
                int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x2a), wb, x2b);
                int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, x3a), wb, x3b);
                acc_i = vmlaq_n_s32(acc_i, vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)), sc);
            };
            step_row(wq0 + off + sb * 32, sc0[soff + sb], i0);
            step_row(wq1 + off + sb * 32, sc1[soff + sb], i1);
            step_row(wq2 + off + sb * 32, sc2[soff + sb], i2);
            step_row(wq3 + off + sb * 32, sc3[soff + sb], i3);
#endif
        }

        const int32x4_t bs0_lo = vpaddlq_s16(vld1q_s16(x0[b].bsums + 0)), bs0_hi = vpaddlq_s16(vld1q_s16(x0[b].bsums + 8));
        const int32x4_t bs1_lo = vpaddlq_s16(vld1q_s16(x1[b].bsums + 0)), bs1_hi = vpaddlq_s16(vld1q_s16(x1[b].bsums + 8));
        const int32x4_t bs2_lo = vpaddlq_s16(vld1q_s16(x2[b].bsums + 0)), bs2_hi = vpaddlq_s16(vld1q_s16(x2[b].bsums + 8));
        const int32x4_t bs3_lo = vpaddlq_s16(vld1q_s16(x3[b].bsums + 0)), bs3_hi = vpaddlq_s16(vld1q_s16(x3[b].bsums + 8));

        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float32x4_t vxd = vld1q_f32(xd_arr);

        auto fin = [&](int32x4_t sp, const uint8_t * mn, float wd, float wm, float32x4_t & acc) {
            const uint16x8_t m16 = vmovl_u8(vld1_u8(mn));
            const int32x4_t mlo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));
            const int32x4_t mhi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m16)));
            const int32x4_t sm0 = vmlaq_s32(vmulq_s32(bs0_lo, mlo), bs0_hi, mhi);
            const int32x4_t sm1 = vmlaq_s32(vmulq_s32(bs1_lo, mlo), bs1_hi, mhi);
            const int32x4_t sm2 = vmlaq_s32(vmulq_s32(bs2_lo, mlo), bs2_hi, mhi);
            const int32x4_t sm3 = vmlaq_s32(vmulq_s32(bs3_lo, mlo), bs3_hi, mhi);
            const int32x4_t sm  = vpaddq_s32(vpaddq_s32(sm0, sm1), vpaddq_s32(sm2, sm3));
            const float32x4_t val = vfmsq_f32(vmulq_n_f32(vcvtq_f32_s32(sp), wd), vcvtq_f32_s32(sm), vdupq_n_f32(wm));
            acc = vfmaq_f32(acc, val, vxd);
        };
        fin(i0, mn0 + soff, wd0[b], wm0[b], a0);
        fin(i1, mn1 + soff, wd1[b], wm1[b], a1);
        fin(i2, mn2 + soff, wd2[b], wm2[b], a2);
        fin(i3, mn3 + soff, wd3[b], wm3[b], a3);
    }
    vst1q_f32(out[0], a0); vst1q_f32(out[1], a1);
    vst1q_f32(out[2], a2); vst1q_f32(out[3], a3);
}

inline void gemm_q4_k_unpacked_1x4(
    const int8_t *     __restrict__ wq,
    const uint8_t *    __restrict__ wsc,
    const uint8_t *    __restrict__ wmn,
    const float *      __restrict__ wd,
    const float *      __restrict__ wm,
    const CpuActQ8_K * __restrict__ x0,
    const CpuActQ8_K * __restrict__ x1,
    const CpuActQ8_K * __restrict__ x2,
    const CpuActQ8_K * __restrict__ x3,
    float *            __restrict__ out,
    int nb) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t isum = vdupq_n_s32(0);
        const int off  = b * 256;
        const int soff = b * 8;
        const int8_t * q0 = x0[b].qs, * q1 = x1[b].qs, * q2 = x2[b].qs, * q3 = x3[b].qs;
        for (int sb = 0; sb < 8; ++sb) {
            const int8x16_t wa = vld1q_s8(wq + off + sb * 32);
            const int8x16_t wb = vld1q_s8(wq + off + sb * 32 + 16);
            int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, vld1q_s8(q0 + sb * 32)), wb, vld1q_s8(q0 + sb * 32 + 16));
            int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, vld1q_s8(q1 + sb * 32)), wb, vld1q_s8(q1 + sb * 32 + 16));
            int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, vld1q_s8(q2 + sb * 32)), wb, vld1q_s8(q2 + sb * 32 + 16));
            int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa, vld1q_s8(q3 + sb * 32)), wb, vld1q_s8(q3 + sb * 32 + 16));
            isum = vmlaq_n_s32(isum, vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)), static_cast<int32_t>(wsc[soff + sb]));
        }
        const uint16x8_t m16 = vmovl_u8(vld1_u8(wmn + soff));
        const int32x4_t mlo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));
        const int32x4_t mhi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m16)));
        const int32x4_t sm0 = vmlaq_s32(vmulq_s32(vpaddlq_s16(vld1q_s16(x0[b].bsums)), mlo), vpaddlq_s16(vld1q_s16(x0[b].bsums + 8)), mhi);
        const int32x4_t sm1 = vmlaq_s32(vmulq_s32(vpaddlq_s16(vld1q_s16(x1[b].bsums)), mlo), vpaddlq_s16(vld1q_s16(x1[b].bsums + 8)), mhi);
        const int32x4_t sm2 = vmlaq_s32(vmulq_s32(vpaddlq_s16(vld1q_s16(x2[b].bsums)), mlo), vpaddlq_s16(vld1q_s16(x2[b].bsums + 8)), mhi);
        const int32x4_t sm3 = vmlaq_s32(vmulq_s32(vpaddlq_s16(vld1q_s16(x3[b].bsums)), mlo), vpaddlq_s16(vld1q_s16(x3[b].bsums + 8)), mhi);
        const int32x4_t sm  = vpaddq_s32(vpaddq_s32(sm0, sm1), vpaddq_s32(sm2, sm3));
        const float xd_arr[4] = {x0[b].d, x1[b].d, x2[b].d, x3[b].d};
        const float32x4_t val = vfmsq_f32(vmulq_n_f32(vcvtq_f32_s32(isum), wd[b]), vcvtq_f32_s32(sm), vdupq_n_f32(wm[b]));
        acc = vfmaq_f32(acc, val, vld1q_f32(xd_arr));
    }
    vst1q_f32(out, acc);
}
#endif // __aarch64__ && __ARM_NEON && __ARM_FEATURE_DOTPROD

inline void linear_rows_range_direct(
    const CpuTensor & W,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * y,
    int T, int r0, int r1
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
    , const ArmQ80Lane4 * xq8_p4 = nullptr
    , const ArmQ8KLane4 * xq8k_p4 = nullptr
#endif
    ) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);

    if (W.dtype == DType::Q8_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ8_0 *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
            static thread_local std::vector<ArmQ80Lane4> local_p4;
            if (xq8_p4 == nullptr) {
                const int nt4 = T >> 2;
                local_p4.resize(static_cast<size_t>(nt4) * nb);
                pack_q8_0_lane4_range(xq8, local_p4.data(), 0, nt4, nb);
                xq8_p4 = local_p4.data();
            }
#endif
            constexpr int R_BLK = 64;
            constexpr int T_BLK = 64;
            for (int rb = r0; rb < r1; rb += R_BLK) {
                const int re = std::min(rb + R_BLK, r1);
                for (int tb = 0; tb < T; tb += T_BLK) {
                    const int te = std::min(tb + T_BLK, T);
                    int r = rb;
                    for (; r + 4 <= re; r += 4) {
                        const BlockQ8_0 * wr0 = base + static_cast<size_t>(r + 0) * nb;
                        const BlockQ8_0 * wr1 = base + static_cast<size_t>(r + 1) * nb;
                        const BlockQ8_0 * wr2 = base + static_cast<size_t>(r + 2) * nb;
                        const BlockQ8_0 * wr3 = base + static_cast<size_t>(r + 3) * nb;
                        int t = tb;
                        for (; t + 4 <= te; t += 4) {
                            float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                            gemm_q8_0_lane_4x4_arm64(wr0, wr1, wr2, wr3,
                                                     xq8_p4 + static_cast<size_t>(t >> 2) * nb, o, nb);
#else
                            gemm_q8_0_4x4_arm64(wr0, wr1, wr2, wr3,
                                                xq8 + static_cast<size_t>(t + 0) * nb,
                                                xq8 + static_cast<size_t>(t + 1) * nb,
                                                xq8 + static_cast<size_t>(t + 2) * nb,
                                                xq8 + static_cast<size_t>(t + 3) * nb,
                                                o, nb);
#endif
                            for (int k = 0; k < 4; ++k) {
                                float * yp = y + static_cast<size_t>(t + k) * M + r;
                                yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                            }
                        }
                        for (; t < te; ++t) {
                            float * yp = y + static_cast<size_t>(t) * M + r;
                            dot_q8_0_q8_0_4rows(wr0, wr1, wr2, wr3, xq8 + static_cast<size_t>(t) * nb, nb,
                                                yp[0], yp[1], yp[2], yp[3]);
                        }
                    }
                    for (; r < re; ++r) {
                        const BlockQ8_0 * wr = base + static_cast<size_t>(r) * nb;
                        for (int t = tb; t < te; ++t) {
                            y[static_cast<size_t>(t) * M + r] = dot_q8_0_q8_0_row(wr, xq8 + static_cast<size_t>(t) * nb, nb);
                        }
                    }
                }
            }
            return;
        }
#endif
        int t = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
        for (; t + 3 < T; t += 4) {
            const CpuActQ8_0 * xt0 = xq8 + static_cast<size_t>(t + 0) * nb;
            const CpuActQ8_0 * xt1 = xq8 + static_cast<size_t>(t + 1) * nb;
            const CpuActQ8_0 * xt2 = xq8 + static_cast<size_t>(t + 2) * nb;
            const CpuActQ8_0 * xt3 = xq8 + static_cast<size_t>(t + 3) * nb;
            float * yt0 = y + static_cast<size_t>(t + 0) * M;
            float * yt1 = y + static_cast<size_t>(t + 1) * M;
            float * yt2 = y + static_cast<size_t>(t + 2) * M;
            float * yt3 = y + static_cast<size_t>(t + 3) * M;
            int r = r0;
            for (; r + 1 < r1; r += 2) {
                float tile[2][4];
                gemm_q8_0_2x4_direct(base + static_cast<size_t>(r + 0) * nb, base + static_cast<size_t>(r + 1) * nb,
                                     xt0, xt1, xt2, xt3, nb, tile);
                yt0[r + 0] = tile[0][0]; yt1[r + 0] = tile[0][1]; yt2[r + 0] = tile[0][2]; yt3[r + 0] = tile[0][3];
                yt0[r + 1] = tile[1][0]; yt1[r + 1] = tile[1][1]; yt2[r + 1] = tile[1][2]; yt3[r + 1] = tile[1][3];
            }
            for (; r < r1; ++r) {
                const BlockQ8_0 * wr = base + static_cast<size_t>(r) * nb;
                yt0[r] = dot_q8_0_q8_0_row(wr, xt0, nb); yt1[r] = dot_q8_0_q8_0_row(wr, xt1, nb);
                yt2[r] = dot_q8_0_q8_0_row(wr, xt2, nb); yt3[r] = dot_q8_0_q8_0_row(wr, xt3, nb);
            }
        }
#endif
        for (; t < T; ++t) {
            const CpuActQ8_0 * xt = xq8 + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_q8_0_q8_0_4rows(
                    base + static_cast<size_t>(r + 0) * nb,
                    base + static_cast<size_t>(r + 1) * nb,
                    base + static_cast<size_t>(r + 2) * nb,
                    base + static_cast<size_t>(r + 3) * nb,
                    xt, nb, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) {
                yt[r] = dot_q8_0_q8_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
            }
        }
        return;
    }

    if (W.dtype == DType::Q6_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ6_K *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
            static thread_local std::vector<ArmQ8KLane4> local_p4k;
            if (xq8k_p4 == nullptr) {
                const int nt4 = T >> 2;
                local_p4k.resize(static_cast<size_t>(nt4) * nb);
                pack_q8_k_lane4_range(xq8k, local_p4k.data(), 0, nt4, nb);
                xq8k_p4 = local_p4k.data();
            }
#endif
            // K-blocked (NB_BLK=16 -> 16 KB unpacked weights, 100% L1d-resident even for K=18944!)
            // + T-blocked (T_BLK=64 -> L2-resident activations) pure 4x4 sdot GEMM
            constexpr int NB_BLK = 16;
            constexpr int T_BLK  = 64;
            const int nb_cap = std::min(nb, NB_BLK);
            const int k_cap  = nb_cap * 256;
            static thread_local ArmUnpackBuf ub;
            const size_t need_q  = static_cast<size_t>(4) * k_cap;
            const size_t need_sc = static_cast<size_t>(4) * nb_cap * 16;
            const size_t need_d  = static_cast<size_t>(4) * nb_cap;
            if (ub.q.size() < need_q)    ub.q.resize(need_q);
            if (ub.sc8.size() < need_sc) ub.sc8.resize(need_sc);
            if (ub.d.size() < need_d)    ub.d.resize(need_d);

            int r = r0;
            for (; r + 4 <= r1; r += 4) {
                for (int b0 = 0; b0 < nb; b0 += NB_BLK) {
                    const int b_len = std::min(NB_BLK, nb - b0);
                    const int k_len = b_len * 256;
                    for (int ri = 0; ri < 4; ++ri) {
                        unpack_q6_k_row_i8(base + static_cast<size_t>(r + ri) * nb + b0,
                                           ub.q.data()   + static_cast<size_t>(ri) * k_len,
                                           ub.sc8.data() + static_cast<size_t>(ri) * b_len * 16,
                                           ub.d.data()   + static_cast<size_t>(ri) * b_len, b_len);
                    }
                    for (int tb = 0; tb < T; tb += T_BLK) {
                        const int te = std::min(tb + T_BLK, T);
                        int t = tb;
                        for (; t + 4 <= te; t += 4) {
                            float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                            gemm_q6_k_lane_4x4_arm64(
                                ub.q.data() + 0 * k_len, ub.sc8.data() + 0 * b_len * 16, ub.d.data() + 0 * b_len,
                                ub.q.data() + 1 * k_len, ub.sc8.data() + 1 * b_len * 16, ub.d.data() + 1 * b_len,
                                ub.q.data() + 2 * k_len, ub.sc8.data() + 2 * b_len * 16, ub.d.data() + 2 * b_len,
                                ub.q.data() + 3 * k_len, ub.sc8.data() + 3 * b_len * 16, ub.d.data() + 3 * b_len,
                                xq8k_p4 + static_cast<size_t>(t >> 2) * nb + b0, o, b_len);
#else
                            gemm_q6_k_unpacked_4x4(
                                ub.q.data() + 0 * k_len, ub.sc8.data() + 0 * b_len * 16, ub.d.data() + 0 * b_len,
                                ub.q.data() + 1 * k_len, ub.sc8.data() + 1 * b_len * 16, ub.d.data() + 1 * b_len,
                                ub.q.data() + 2 * k_len, ub.sc8.data() + 2 * b_len * 16, ub.d.data() + 2 * b_len,
                                ub.q.data() + 3 * k_len, ub.sc8.data() + 3 * b_len * 16, ub.d.data() + 3 * b_len,
                                xq8k + static_cast<size_t>(t + 0) * nb + b0,
                                xq8k + static_cast<size_t>(t + 1) * nb + b0,
                                xq8k + static_cast<size_t>(t + 2) * nb + b0,
                                xq8k + static_cast<size_t>(t + 3) * nb + b0,
                                o, b_len);
#endif
                            if (b0 == 0) {
                                for (int k = 0; k < 4; ++k) {
                                    float * yp = y + static_cast<size_t>(t + k) * M + r;
                                    yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                                }
                            } else {
                                for (int k = 0; k < 4; ++k) {
                                    float * yp = y + static_cast<size_t>(t + k) * M + r;
                                    yp[0] += o[0][k]; yp[1] += o[1][k]; yp[2] += o[2][k]; yp[3] += o[3][k];
                                }
                            }
                        }
                        for (; t < te; ++t) {
                            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb + b0;
                            if (b0 == 0) {
                                for (int ri = 0; ri < 4; ++ri) {
                                    y[static_cast<size_t>(t) * M + r + ri] = dot_q6_k_q8_k_row(base + static_cast<size_t>(r + ri) * nb + b0, xt, b_len);
                                }
                            } else {
                                for (int ri = 0; ri < 4; ++ri) {
                                    y[static_cast<size_t>(t) * M + r + ri] += dot_q6_k_q8_k_row(base + static_cast<size_t>(r + ri) * nb + b0, xt, b_len);
                                }
                            }
                        }
                    }
                }
            }
            for (; r < r1; ++r) {
                for (int b0 = 0; b0 < nb; b0 += NB_BLK) {
                    const int b_len = std::min(NB_BLK, nb - b0);
                    unpack_q6_k_row_i8(base + static_cast<size_t>(r) * nb + b0, ub.q.data(), ub.sc8.data(), ub.d.data(), b_len);
                    int t = 0;
                    for (; t + 4 <= T; t += 4) {
                        float o[4];
                        gemm_q6_k_unpacked_1x4(ub.q.data(), ub.sc8.data(), ub.d.data(),
                                               xq8k + static_cast<size_t>(t + 0) * nb + b0,
                                               xq8k + static_cast<size_t>(t + 1) * nb + b0,
                                               xq8k + static_cast<size_t>(t + 2) * nb + b0,
                                               xq8k + static_cast<size_t>(t + 3) * nb + b0,
                                               o, b_len);
                        if (b0 == 0) {
                            for (int k = 0; k < 4; ++k) y[static_cast<size_t>(t + k) * M + r] = o[k];
                        } else {
                            for (int k = 0; k < 4; ++k) y[static_cast<size_t>(t + k) * M + r] += o[k];
                        }
                    }
                    for (; t < T; ++t) {
                        const float v = dot_q6_k_q8_k_row(base + static_cast<size_t>(r) * nb + b0,
                                                          xq8k + static_cast<size_t>(t) * nb + b0, b_len);
                        if (b0 == 0) y[static_cast<size_t>(t) * M + r] = v;
                        else         y[static_cast<size_t>(t) * M + r] += v;
                    }
                }
            }
            return;
        }
#endif
        int t = 0;
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
        for (; t + 3 < T; t += 4) {
            const CpuActQ8_K * xt0 = xq8k + static_cast<size_t>(t + 0) * nb;
            const CpuActQ8_K * xt1 = xq8k + static_cast<size_t>(t + 1) * nb;
            const CpuActQ8_K * xt2 = xq8k + static_cast<size_t>(t + 2) * nb;
            const CpuActQ8_K * xt3 = xq8k + static_cast<size_t>(t + 3) * nb;
            float * yt0 = y + static_cast<size_t>(t + 0) * M;
            float * yt1 = y + static_cast<size_t>(t + 1) * M;
            float * yt2 = y + static_cast<size_t>(t + 2) * M;
            float * yt3 = y + static_cast<size_t>(t + 3) * M;
            for (int r = r0; r < r1; ++r) {
                float out4[4];
                gemm_q6_k_1x4_direct(base + static_cast<size_t>(r) * nb, xt0, xt1, xt2, xt3, nb, out4);
                yt0[r] = out4[0]; yt1[r] = out4[1]; yt2[r] = out4[2]; yt3[r] = out4[3];
            }
        }
#endif
        for (; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            // Zero-spill 1-row AVX2 dot (210B/block streams cleanly; 2-row spills 29 YMM regs on AVX2!)
            for (int r = r0; r < r1; ++r) {
                yt[r] = dot_q6_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
            }
        }
        return;
    }

    if (W.dtype == DType::F32) {
        const auto * base = static_cast<const float *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (T >= 4) {
            // L1-resident 4x4 NEON F32 GEMM (vfmaq_laneq_f32): reads each 4-row weight strip ONCE
            // across all T tokens with zero horizontal adds inside the K loop!
            int r = r0;
            for (; r + 4 <= r1; r += 4) {
                const float * w0 = base + static_cast<size_t>(r + 0) * K;
                const float * w1 = base + static_cast<size_t>(r + 1) * K;
                const float * w2 = base + static_cast<size_t>(r + 2) * K;
                const float * w3 = base + static_cast<size_t>(r + 3) * K;
                int t = 0;
                for (; t + 4 <= T; t += 4) {
                    const float * x0 = x + static_cast<size_t>(t + 0) * K;
                    const float * x1 = x + static_cast<size_t>(t + 1) * K;
                    const float * x2 = x + static_cast<size_t>(t + 2) * K;
                    const float * x3 = x + static_cast<size_t>(t + 3) * K;
                    float32x4_t c0 = vdupq_n_f32(0.0f), c1 = vdupq_n_f32(0.0f);
                    float32x4_t c2 = vdupq_n_f32(0.0f), c3 = vdupq_n_f32(0.0f);
                    int k = 0;
                    for (; k + 4 <= K; k += 4) {
                        const float32x4_t r0v = vld1q_f32(x0 + k);
                        const float32x4_t r1v = vld1q_f32(x1 + k);
                        const float32x4_t r2v = vld1q_f32(x2 + k);
                        const float32x4_t r3v = vld1q_f32(x3 + k);
                        // 4x4 transpose so xk_j = {x0[k+j], x1[k+j], x2[k+j], x3[k+j]}
                        const float32x4_t t01a = vtrn1q_f32(r0v, r1v);
                        const float32x4_t t01b = vtrn2q_f32(r0v, r1v);
                        const float32x4_t t23a = vtrn1q_f32(r2v, r3v);
                        const float32x4_t t23b = vtrn2q_f32(r2v, r3v);
                        const float32x4_t xk0  = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01a), vreinterpretq_f64_f32(t23a)));
                        const float32x4_t xk1  = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01b), vreinterpretq_f64_f32(t23b)));
                        const float32x4_t xk2  = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01a), vreinterpretq_f64_f32(t23a)));
                        const float32x4_t xk3  = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01b), vreinterpretq_f64_f32(t23b)));

                        const float32x4_t wv0 = vld1q_f32(w0 + k);
                        c0 = vfmaq_laneq_f32(c0, xk0, wv0, 0); c0 = vfmaq_laneq_f32(c0, xk1, wv0, 1);
                        c0 = vfmaq_laneq_f32(c0, xk2, wv0, 2); c0 = vfmaq_laneq_f32(c0, xk3, wv0, 3);

                        const float32x4_t wv1 = vld1q_f32(w1 + k);
                        c1 = vfmaq_laneq_f32(c1, xk0, wv1, 0); c1 = vfmaq_laneq_f32(c1, xk1, wv1, 1);
                        c1 = vfmaq_laneq_f32(c1, xk2, wv1, 2); c1 = vfmaq_laneq_f32(c1, xk3, wv1, 3);

                        const float32x4_t wv2 = vld1q_f32(w2 + k);
                        c2 = vfmaq_laneq_f32(c2, xk0, wv2, 0); c2 = vfmaq_laneq_f32(c2, xk1, wv2, 1);
                        c2 = vfmaq_laneq_f32(c2, xk2, wv2, 2); c2 = vfmaq_laneq_f32(c2, xk3, wv2, 3);

                        const float32x4_t wv3 = vld1q_f32(w3 + k);
                        c3 = vfmaq_laneq_f32(c3, xk0, wv3, 0); c3 = vfmaq_laneq_f32(c3, xk1, wv3, 1);
                        c3 = vfmaq_laneq_f32(c3, xk2, wv3, 2); c3 = vfmaq_laneq_f32(c3, xk3, wv3, 3);
                    }
                    float o[4][4];
                    vst1q_f32(o[0], c0); vst1q_f32(o[1], c1);
                    vst1q_f32(o[2], c2); vst1q_f32(o[3], c3);
                    for (; k < K; ++k) {
                        const float xv[4] = {x0[k], x1[k], x2[k], x3[k]};
                        for (int ti = 0; ti < 4; ++ti) {
                            o[0][ti] += w0[k] * xv[ti]; o[1][ti] += w1[k] * xv[ti];
                            o[2][ti] += w2[k] * xv[ti]; o[3][ti] += w3[k] * xv[ti];
                        }
                    }
                    for (int ti = 0; ti < 4; ++ti) {
                        float * yp = y + static_cast<size_t>(t + ti) * M + r;
                        yp[0] = o[0][ti]; yp[1] = o[1][ti]; yp[2] = o[2][ti]; yp[3] = o[3][ti];
                    }
                }
                for (; t < T; ++t) {
                    float * yp = y + static_cast<size_t>(t) * M + r;
                    dot_f32_4rows(w0, w1, w2, w3, x + static_cast<size_t>(t) * K, K, yp[0], yp[1], yp[2], yp[3]);
                }
            }
            for (; r < r1; ++r) {
                const float * wr = base + static_cast<size_t>(r) * K;
                for (int t = 0; t < T; ++t) {
                    y[static_cast<size_t>(t) * M + r] = dot_f32_row(wr, x + static_cast<size_t>(t) * K, K);
                }
            }
            return;
        }
#endif
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_f32_4rows(base + static_cast<size_t>(r + 0) * K,
                              base + static_cast<size_t>(r + 1) * K,
                              base + static_cast<size_t>(r + 2) * K,
                              base + static_cast<size_t>(r + 3) * K,
                              xt, K, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_f32_row(base + static_cast<size_t>(r) * K, xt, K);
        }
        return;
    }

    if (W.dtype == DType::BF16) {
        const auto * base = static_cast<const uint16_t *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_bf16_4rows(base + static_cast<size_t>(r + 0) * K,
                               base + static_cast<size_t>(r + 1) * K,
                               base + static_cast<size_t>(r + 2) * K,
                               base + static_cast<size_t>(r + 3) * K,
                               xt, K, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_bf16_row(base + static_cast<size_t>(r) * K, xt, K);
        }
        return;
    }

    if (W.dtype == DType::F16) {
        const auto * base = static_cast<const uint16_t *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_f16_4rows(base + static_cast<size_t>(r + 0) * K,
                              base + static_cast<size_t>(r + 1) * K,
                              base + static_cast<size_t>(r + 2) * K,
                              base + static_cast<size_t>(r + 3) * K,
                              xt, K, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_f16_row(base + static_cast<size_t>(r) * K, xt, K);
        }
        return;
    }

    if (W.dtype == DType::Q4_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ4_0 *>(W.ptr);
        int t = 0;
#if defined(__AVX2__) && defined(__FMA__)
        for (; t + 1 < T; ) {
            const int nt = std::min(4, T - t);
            const CpuActQ8_0 * xs[4];
            float * ys[4];
            for (int ti = 0; ti < nt; ++ti) {
                xs[ti] = xq8 + static_cast<size_t>(t + ti) * nb;
                ys[ti] = y + static_cast<size_t>(t + ti) * M;
            }
            for (int r = r0; r < r1; ++r) {
                float out4[4];
                gemm_q4_0_1xN_direct(base + static_cast<size_t>(r) * nb, xs, nt, nb, out4);
                for (int ti = 0; ti < nt; ++ti) ys[ti][r] = out4[ti];
            }
            t += nt;
        }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (xq8 != nullptr && T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
            static thread_local std::vector<ArmQ80Lane4> local_p4;
            if (xq8_p4 == nullptr) {
                const int nt4 = T >> 2;
                local_p4.resize(static_cast<size_t>(nt4) * nb);
                pack_q8_0_lane4_range(xq8, local_p4.data(), 0, nt4, nb);
                xq8_p4 = local_p4.data();
            }
#endif
            constexpr int R_BLK = 64;
            constexpr int T_BLK = 64;
            for (int rb = r0; rb < r1; rb += R_BLK) {
                const int re = std::min(rb + R_BLK, r1);
                for (int tb = 0; tb < T; tb += T_BLK) {
                    const int te = std::min(tb + T_BLK, T);
                    int r = rb;
                    for (; r + 4 <= re; r += 4) {
                        const BlockQ4_0 * wr0 = base + static_cast<size_t>(r + 0) * nb;
                        const BlockQ4_0 * wr1 = base + static_cast<size_t>(r + 1) * nb;
                        const BlockQ4_0 * wr2 = base + static_cast<size_t>(r + 2) * nb;
                        const BlockQ4_0 * wr3 = base + static_cast<size_t>(r + 3) * nb;
                        int ti = tb;
                        for (; ti + 4 <= te; ti += 4) {
                            float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                            gemm_q4_0_lane_4x4_arm64(wr0, wr1, wr2, wr3,
                                                     xq8_p4 + static_cast<size_t>(ti >> 2) * nb, o, nb);
#else
                            gemm_q4_0_4x4_arm64(wr0, wr1, wr2, wr3,
                                                xq8 + static_cast<size_t>(ti + 0) * nb,
                                                xq8 + static_cast<size_t>(ti + 1) * nb,
                                                xq8 + static_cast<size_t>(ti + 2) * nb,
                                                xq8 + static_cast<size_t>(ti + 3) * nb,
                                                o, nb);
#endif
                            for (int k = 0; k < 4; ++k) {
                                float * yp = y + static_cast<size_t>(ti + k) * M + r;
                                yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                            }
                        }
                        for (; ti < te; ++ti) {
                            float * yp = y + static_cast<size_t>(ti) * M + r;
                            dot_q4_0_q8_0_4rows(wr0, wr1, wr2, wr3, xq8 + static_cast<size_t>(ti) * nb, nb,
                                                yp[0], yp[1], yp[2], yp[3]);
                        }
                    }
                    for (; r < re; ++r) {
                        const BlockQ4_0 * wr = base + static_cast<size_t>(r) * nb;
                        for (int ti = tb; ti < te; ++ti) {
                            y[static_cast<size_t>(ti) * M + r] = dot_q4_0_q8_0_row(wr, xq8 + static_cast<size_t>(ti) * nb, nb);
                        }
                    }
                }
            }
            return;
        }
#endif
        for (; t < T; ++t) {
            const CpuActQ8_0 * xt = xq8 + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_q4_0_q8_0_4rows(
                    base + static_cast<size_t>(r + 0) * nb,
                    base + static_cast<size_t>(r + 1) * nb,
                    base + static_cast<size_t>(r + 2) * nb,
                    base + static_cast<size_t>(r + 3) * nb,
                    xt, nb, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_q4_0_q8_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q5_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ5_0 *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (xq8 != nullptr && T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
            static thread_local std::vector<ArmQ80Lane4> local_p4;
            if (xq8_p4 == nullptr) {
                const int nt4 = T >> 2;
                local_p4.resize(static_cast<size_t>(nt4) * nb);
                pack_q8_0_lane4_range(xq8, local_p4.data(), 0, nt4, nb);
                xq8_p4 = local_p4.data();
            }
#endif
            constexpr int R_BLK = 64;
            constexpr int T_BLK = 64;
            for (int rb = r0; rb < r1; rb += R_BLK) {
                const int re = std::min(rb + R_BLK, r1);
                for (int tb = 0; tb < T; tb += T_BLK) {
                    const int te = std::min(tb + T_BLK, T);
                    int r = rb;
                    for (; r + 4 <= re; r += 4) {
                        const BlockQ5_0 * wr0 = base + static_cast<size_t>(r + 0) * nb;
                        const BlockQ5_0 * wr1 = base + static_cast<size_t>(r + 1) * nb;
                        const BlockQ5_0 * wr2 = base + static_cast<size_t>(r + 2) * nb;
                        const BlockQ5_0 * wr3 = base + static_cast<size_t>(r + 3) * nb;
                        int ti = tb;
                        for (; ti + 4 <= te; ti += 4) {
                            float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                            gemm_q5_0_lane_4x4_arm64(wr0, wr1, wr2, wr3,
                                                     xq8_p4 + static_cast<size_t>(ti >> 2) * nb, o, nb);
#else
                            gemm_q5_0_4x4_arm64(wr0, wr1, wr2, wr3,
                                                xq8 + static_cast<size_t>(ti + 0) * nb,
                                                xq8 + static_cast<size_t>(ti + 1) * nb,
                                                xq8 + static_cast<size_t>(ti + 2) * nb,
                                                xq8 + static_cast<size_t>(ti + 3) * nb,
                                                o, nb);
#endif
                            for (int k = 0; k < 4; ++k) {
                                float * yp = y + static_cast<size_t>(ti + k) * M + r;
                                yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                            }
                        }
                        for (; ti < te; ++ti) {
                            float * yp = y + static_cast<size_t>(ti) * M + r;
                            dot_q5_0_q8_0_4rows(wr0, wr1, wr2, wr3, xq8 + static_cast<size_t>(ti) * nb, nb,
                                                yp[0], yp[1], yp[2], yp[3]);
                        }
                    }
                    for (; r < re; ++r) {
                        const BlockQ5_0 * wr = base + static_cast<size_t>(r) * nb;
                        for (int ti = tb; ti < te; ++ti) {
                            y[static_cast<size_t>(ti) * M + r] = dot_q5_0_q8_0_row(wr, xq8 + static_cast<size_t>(ti) * nb, nb);
                        }
                    }
                }
            }
            return;
        }
#endif
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_0 * xt = xq8 + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_q5_0_q8_0_4rows(
                    base + static_cast<size_t>(r + 0) * nb,
                    base + static_cast<size_t>(r + 1) * nb,
                    base + static_cast<size_t>(r + 2) * nb,
                    base + static_cast<size_t>(r + 3) * nb,
                    xt, nb, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_q5_0_q8_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::IQ4_NL) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockIQ4_NL *>(W.ptr);
        int t = 0;
#if defined(__AVX2__) && defined(__FMA__)
        for (; t + 1 < T; ) {
            const int nt = std::min(4, T - t);
            const CpuActQ8_0 * xs[4];
            float * ys[4];
            for (int ti = 0; ti < nt; ++ti) {
                xs[ti] = xq8 + static_cast<size_t>(t + ti) * nb;
                ys[ti] = y + static_cast<size_t>(t + ti) * M;
            }
            for (int r = r0; r < r1; ++r) {
                float out4[4];
                gemm_iq4_nl_1xN_direct(base + static_cast<size_t>(r) * nb, xs, nt, nb, out4);
                for (int ti = 0; ti < nt; ++ti) ys[ti][r] = out4[ti];
            }
            t += nt;
        }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (xq8 != nullptr && T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
            static thread_local std::vector<ArmQ80Lane4> local_p4;
            if (xq8_p4 == nullptr) {
                const int nt4 = T >> 2;
                local_p4.resize(static_cast<size_t>(nt4) * nb);
                pack_q8_0_lane4_range(xq8, local_p4.data(), 0, nt4, nb);
                xq8_p4 = local_p4.data();
            }
#endif
            constexpr int R_BLK = 64;
            constexpr int T_BLK = 64;
            for (int rb = r0; rb < r1; rb += R_BLK) {
                const int re = std::min(rb + R_BLK, r1);
                for (int tb = 0; tb < T; tb += T_BLK) {
                    const int te = std::min(tb + T_BLK, T);
                    int r = rb;
                    for (; r + 4 <= re; r += 4) {
                        const BlockIQ4_NL * wr0 = base + static_cast<size_t>(r + 0) * nb;
                        const BlockIQ4_NL * wr1 = base + static_cast<size_t>(r + 1) * nb;
                        const BlockIQ4_NL * wr2 = base + static_cast<size_t>(r + 2) * nb;
                        const BlockIQ4_NL * wr3 = base + static_cast<size_t>(r + 3) * nb;
                        int ti = tb;
                        for (; ti + 4 <= te; ti += 4) {
                            float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                            gemm_iq4_nl_lane_4x4_arm64(wr0, wr1, wr2, wr3,
                                                       xq8_p4 + static_cast<size_t>(ti >> 2) * nb, o, nb);
#else
                            gemm_iq4_nl_4x4_arm64(wr0, wr1, wr2, wr3,
                                                  xq8 + static_cast<size_t>(ti + 0) * nb,
                                                  xq8 + static_cast<size_t>(ti + 1) * nb,
                                                  xq8 + static_cast<size_t>(ti + 2) * nb,
                                                  xq8 + static_cast<size_t>(ti + 3) * nb,
                                                  o, nb);
#endif
                            for (int k = 0; k < 4; ++k) {
                                float * yp = y + static_cast<size_t>(ti + k) * M + r;
                                yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                            }
                        }
                        for (; ti < te; ++ti) {
                            float * yp = y + static_cast<size_t>(ti) * M + r;
                            dot_iq4_nl_q8_0_4rows(wr0, wr1, wr2, wr3, xq8 + static_cast<size_t>(ti) * nb, nb,
                                                  yp[0], yp[1], yp[2], yp[3]);
                        }
                    }
                    for (; r < re; ++r) {
                        const BlockIQ4_NL * wr = base + static_cast<size_t>(r) * nb;
                        for (int ti = tb; ti < te; ++ti) {
                            y[static_cast<size_t>(ti) * M + r] = dot_iq4_nl_q8_0_row(wr, xq8 + static_cast<size_t>(ti) * nb, nb);
                        }
                    }
                }
            }
            return;
        }
#endif
        for (; t < T; ++t) {
            const CpuActQ8_0 * xt = xq8 + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 3 < r1; r += 4) {
                dot_iq4_nl_q8_0_4rows(
                    base + static_cast<size_t>(r + 0) * nb,
                    base + static_cast<size_t>(r + 1) * nb,
                    base + static_cast<size_t>(r + 2) * nb,
                    base + static_cast<size_t>(r + 3) * nb,
                    xt, nb, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
            }
            for (; r < r1; ++r) yt[r] = dot_iq4_nl_q8_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q4_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ4_K *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (xq8k != nullptr) {
            if (T >= 4) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                static thread_local std::vector<ArmQ8KLane4> local_p4k;
                if (xq8k_p4 == nullptr) {
                    const int nt4 = T >> 2;
                    local_p4k.resize(static_cast<size_t>(nt4) * nb);
                    pack_q8_k_lane4_range(xq8k, local_p4k.data(), 0, nt4, nb);
                    xq8k_p4 = local_p4k.data();
                }
#endif
                constexpr int NB_BLK = 16;
                constexpr int T_BLK  = 64;
                const int nb_cap = std::min(nb, NB_BLK);
                const int k_cap  = nb_cap * 256;
                static thread_local ArmUnpackBuf ub;
                const size_t need_q  = static_cast<size_t>(4) * k_cap;
                const size_t need_sc = static_cast<size_t>(4) * nb_cap * 8;
                const size_t need_d  = static_cast<size_t>(4) * nb_cap;
                if (ub.q.size() < need_q)      ub.q.resize(need_q);
                if (ub.u8_sc.size() < need_sc) ub.u8_sc.resize(need_sc);
                if (ub.u8_mn.size() < need_sc) ub.u8_mn.resize(need_sc);
                if (ub.d.size() < need_d)      ub.d.resize(need_d);
                if (ub.dmin.size() < need_d)   ub.dmin.resize(need_d);

                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    for (int b0 = 0; b0 < nb; b0 += NB_BLK) {
                        const int b_len = std::min(NB_BLK, nb - b0);
                        const int k_len = b_len * 256;
                        for (int ri = 0; ri < 4; ++ri) {
                            unpack_q4_k_row_i8(base + static_cast<size_t>(r + ri) * nb + b0,
                                               ub.q.data()     + static_cast<size_t>(ri) * k_len,
                                               ub.u8_sc.data() + static_cast<size_t>(ri) * b_len * 8,
                                               ub.u8_mn.data() + static_cast<size_t>(ri) * b_len * 8,
                                               ub.d.data()     + static_cast<size_t>(ri) * b_len,
                                               ub.dmin.data()  + static_cast<size_t>(ri) * b_len, b_len);
                        }
                        for (int tb = 0; tb < T; tb += T_BLK) {
                            const int te = std::min(tb + T_BLK, T);
                            int t = tb;
                            for (; t + 4 <= te; t += 4) {
                                float o[4][4];
#if !defined(__ARM_FEATURE_MATMUL_INT8)
                                gemm_q4_k_lane_4x4_arm64(
                                    ub.q.data() + 0 * k_len, ub.u8_sc.data() + 0 * b_len * 8, ub.u8_mn.data() + 0 * b_len * 8, ub.d.data() + 0 * b_len, ub.dmin.data() + 0 * b_len,
                                    ub.q.data() + 1 * k_len, ub.u8_sc.data() + 1 * b_len * 8, ub.u8_mn.data() + 1 * b_len * 8, ub.d.data() + 1 * b_len, ub.dmin.data() + 1 * b_len,
                                    ub.q.data() + 2 * k_len, ub.u8_sc.data() + 2 * b_len * 8, ub.u8_mn.data() + 2 * b_len * 8, ub.d.data() + 2 * b_len, ub.dmin.data() + 2 * b_len,
                                    ub.q.data() + 3 * k_len, ub.u8_sc.data() + 3 * b_len * 8, ub.u8_mn.data() + 3 * b_len * 8, ub.d.data() + 3 * b_len, ub.dmin.data() + 3 * b_len,
                                    xq8k_p4 + static_cast<size_t>(t >> 2) * nb + b0, o, b_len);
#else
                                gemm_q4_k_unpacked_4x4(
                                    ub.q.data() + 0 * k_len, ub.u8_sc.data() + 0 * b_len * 8, ub.u8_mn.data() + 0 * b_len * 8, ub.d.data() + 0 * b_len, ub.dmin.data() + 0 * b_len,
                                    ub.q.data() + 1 * k_len, ub.u8_sc.data() + 1 * b_len * 8, ub.u8_mn.data() + 1 * b_len * 8, ub.d.data() + 1 * b_len, ub.dmin.data() + 1 * b_len,
                                    ub.q.data() + 2 * k_len, ub.u8_sc.data() + 2 * b_len * 8, ub.u8_mn.data() + 2 * b_len * 8, ub.d.data() + 2 * b_len, ub.dmin.data() + 2 * b_len,
                                    ub.q.data() + 3 * k_len, ub.u8_sc.data() + 3 * b_len * 8, ub.u8_mn.data() + 3 * b_len * 8, ub.d.data() + 3 * b_len, ub.dmin.data() + 3 * b_len,
                                    xq8k + static_cast<size_t>(t + 0) * nb + b0,
                                    xq8k + static_cast<size_t>(t + 1) * nb + b0,
                                    xq8k + static_cast<size_t>(t + 2) * nb + b0,
                                    xq8k + static_cast<size_t>(t + 3) * nb + b0,
                                    o, b_len);
#endif
                                if (b0 == 0) {
                                    for (int k = 0; k < 4; ++k) {
                                        float * yp = y + static_cast<size_t>(t + k) * M + r;
                                        yp[0] = o[0][k]; yp[1] = o[1][k]; yp[2] = o[2][k]; yp[3] = o[3][k];
                                    }
                                } else {
                                    for (int k = 0; k < 4; ++k) {
                                        float * yp = y + static_cast<size_t>(t + k) * M + r;
                                        yp[0] += o[0][k]; yp[1] += o[1][k]; yp[2] += o[2][k]; yp[3] += o[3][k];
                                    }
                                }
                            }
                            for (; t < te; ++t) {
                                const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb + b0;
                                if (b0 == 0) {
                                    for (int ri = 0; ri < 4; ++ri) {
                                        y[static_cast<size_t>(t) * M + r + ri] = dot_q4_k_q8_k_row(base + static_cast<size_t>(r + ri) * nb + b0, xt, b_len);
                                    }
                                } else {
                                    for (int ri = 0; ri < 4; ++ri) {
                                        y[static_cast<size_t>(t) * M + r + ri] += dot_q4_k_q8_k_row(base + static_cast<size_t>(r + ri) * nb + b0, xt, b_len);
                                    }
                                }
                            }
                        }
                    }
                }
                for (; r < r1; ++r) {
                    for (int b0 = 0; b0 < nb; b0 += NB_BLK) {
                        const int b_len = std::min(NB_BLK, nb - b0);
                        unpack_q4_k_row_i8(base + static_cast<size_t>(r) * nb + b0, ub.q.data(), ub.u8_sc.data(), ub.u8_mn.data(), ub.d.data(), ub.dmin.data(), b_len);
                        int t = 0;
                        for (; t + 4 <= T; t += 4) {
                            float o[4];
                            gemm_q4_k_unpacked_1x4(ub.q.data(), ub.u8_sc.data(), ub.u8_mn.data(), ub.d.data(), ub.dmin.data(),
                                                   xq8k + static_cast<size_t>(t + 0) * nb + b0,
                                                   xq8k + static_cast<size_t>(t + 1) * nb + b0,
                                                   xq8k + static_cast<size_t>(t + 2) * nb + b0,
                                                   xq8k + static_cast<size_t>(t + 3) * nb + b0,
                                                   o, b_len);
                            if (b0 == 0) {
                                for (int k = 0; k < 4; ++k) y[static_cast<size_t>(t + k) * M + r] = o[k];
                            } else {
                                for (int k = 0; k < 4; ++k) y[static_cast<size_t>(t + k) * M + r] += o[k];
                            }
                        }
                        for (; t < T; ++t) {
                            const float v = dot_q4_k_q8_k_row(base + static_cast<size_t>(r) * nb + b0,
                                                              xq8k + static_cast<size_t>(t) * nb + b0, b_len);
                            if (b0 == 0) y[static_cast<size_t>(t) * M + r] = v;
                            else         y[static_cast<size_t>(t) * M + r] += v;
                        }
                    }
                }
            } else {
                for (int r = r0; r < r1; ++r) {
                    const BlockQ4_K * wr = base + static_cast<size_t>(r) * nb;
                    for (int t = 0; t < T; ++t) {
                        y[static_cast<size_t>(t) * M + r] = dot_q4_k_q8_k_row(wr, xq8k + static_cast<size_t>(t) * nb, nb);
                    }
                }
            }
            return;
        }
#endif
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            int r = r0;
            for (; r + 1 < r1; r += 2) {
                dot_q4_k_q8_k_2rows(base + static_cast<size_t>(r + 0) * nb,
                                    base + static_cast<size_t>(r + 1) * nb,
                                    xt, nb, yt[r + 0], yt[r + 1]);
            }
            for (; r < r1; ++r) yt[r] = dot_q4_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q5_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ5_K *>(W.ptr);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (xq8k != nullptr) {
            for (int r = r0; r < r1; ++r) {
                const BlockQ5_K * wr = base + static_cast<size_t>(r) * nb;
                for (int t = 0; t < T; ++t) {
                    y[static_cast<size_t>(t) * M + r] = dot_q5_k_q8_k_row(wr, xq8k + static_cast<size_t>(t) * nb, nb);
                }
            }
            return;
        }
#endif
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q5_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::IQ2_S) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockIQ2_S *>(W.ptr);
        int t = 0;
#if defined(__AVX2__) && defined(__FMA__)
        for (; t + 1 < T; ) {
            const int nt = std::min(4, T - t);
            const CpuActQ8_K * xs[4];
            float * ys[4];
            for (int ti = 0; ti < nt; ++ti) {
                xs[ti] = xq8k + static_cast<size_t>(t + ti) * nb;
                ys[ti] = y + static_cast<size_t>(t + ti) * M;
            }
            for (int r = r0; r < r1; ++r) {
                float out4[4];
                gemm_iq2_s_1xN_direct(base + static_cast<size_t>(r) * nb, xs, nt, nb, out4);
                for (int ti = 0; ti < nt; ++ti) ys[ti][r] = out4[ti];
            }
            t += nt;
        }
#endif
        for (; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            // Zero-spill 1-row 2x-ib32-unrolled AVX2 dot (xb fits in L1d; 2-row spills YMM regs on AVX2!)
            for (int r = r0; r < r1; ++r) yt[r] = dot_iq2_s_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::IQ3_S) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockIQ3_S *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_iq3_s_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q2_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ2_K *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q2_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q3_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ3_K *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q3_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::IQ4_XS) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockIQ4_XS *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_iq4_xs_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::IQ3_XXS) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockIQ3_XXS *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const CpuActQ8_K * xt = xq8k + static_cast<size_t>(t) * nb;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_iq3_xxs_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    throw std::runtime_error(std::string("CPU linear unsupported dtype: ") + dtype_traits(W.dtype).name);
}

inline void apply_swiglu_vec(float * g, const float * u, size_t n, float clamp_val) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    if (clamp_val > 0.0f) {
        const __m256 vpos = _mm256_set1_ps(clamp_val);
        const __m256 vneg = _mm256_set1_ps(-clamp_val);
        for (; i + 7 < n; i += 8) {
            __m256 gv = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(g + i), vneg), vpos);
            __m256 uv = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(u + i), vneg), vpos);
            _mm256_storeu_ps(g + i, _mm256_mul_ps(silu_ps_avx2(gv), uv));
        }
    } else {
        for (; i + 7 < n; i += 8) {
            __m256 gv = _mm256_loadu_ps(g + i);
            __m256 uv = _mm256_loadu_ps(u + i);
            _mm256_storeu_ps(g + i, _mm256_mul_ps(silu_ps_avx2(gv), uv));
        }
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const float32x4_t v1 = vdupq_n_f32(1.0f);
    if (clamp_val > 0.0f) {
        const float32x4_t vhi = vdupq_n_f32(clamp_val);
        const float32x4_t vlo = vdupq_n_f32(-clamp_val);
        for (; i + 4 <= n; i += 4) {
            const float32x4_t gv = vminq_f32(vmaxq_f32(vld1q_f32(g + i), vlo), vhi);
            const float32x4_t uv = vminq_f32(vmaxq_f32(vld1q_f32(u + i), vlo), vhi);
            const float32x4_t denom = vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(gv)));
            vst1q_f32(g + i, vmulq_f32(vdivq_f32(gv, denom), uv));
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            const float32x4_t gv = vld1q_f32(g + i);
            const float32x4_t uv = vld1q_f32(u + i);
            const float32x4_t denom = vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(gv)));
            vst1q_f32(g + i, vmulq_f32(vdivq_f32(gv, denom), uv));
        }
    }
#endif
    for (; i < n; ++i) {
        float gv = g[i];
        float uv = u[i];
        if (clamp_val > 0.0f) {
            gv = std::min(std::max(gv, -clamp_val), clamp_val);
            uv = std::min(std::max(uv, -clamp_val), clamp_val);
        }
        g[i] = (gv / (1.0f + std::exp(-gv))) * uv;
    }
}

} // namespace

float f16_to_f32(uint16_t h) { return f16_to_f32_fast(h); }
uint16_t f32_to_f16(float f) { return f32_to_f16_fast(f); }
float bf16_to_f32(uint16_t b) { return bf16_to_f32_fast(b); }

void quantize_row_q8_0(const float * x, CpuActQ8_0 * dst, int K) {
    const int nb = K >> 5;
#if defined(__AVX2__)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256i min127   = _mm256_set1_epi8(-127);
    const __m256i ones8    = _mm256_set1_epi8(1);
    const __m256i ones16   = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        __m256 v0 = _mm256_loadu_ps(xb + 0);
        __m256 v1 = _mm256_loadu_ps(xb + 8);
        __m256 v2 = _mm256_loadu_ps(xb + 16);
        __m256 v3 = _mm256_loadu_ps(xb + 24);
        __m256 m01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1));
        __m256 m23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3));
        __m256 m   = _mm256_max_ps(m01, m23);
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        const float amax = _mm_cvtss_f32(m128);
        const float d    = amax / 127.0f;
        const float id   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;

        const __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
        __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), _MM_SHUFFLE(3, 1, 2, 0));
        __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), _MM_SHUFFLE(3, 1, 2, 0));
        __m256i q8  = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), _MM_SHUFFLE(3, 1, 2, 0));
        q8 = _mm256_max_epi8(q8, min127);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst[b].qs), q8);
        const int32_t bsum = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q8), ones16));
        dst[b].s = d * static_cast<float>(bsum);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const int8x16_t min127 = vdupq_n_s8(-127);
    for (int b = 0; b < nb; ++b) {
        const float * xp = x + b * 32;
        float32x4_t m0 = vabsq_f32(vld1q_f32(xp +  0));
        float32x4_t m1 = vabsq_f32(vld1q_f32(xp +  4));
        float32x4_t m2 = vabsq_f32(vld1q_f32(xp +  8));
        float32x4_t m3 = vabsq_f32(vld1q_f32(xp + 12));
        float32x4_t m4 = vabsq_f32(vld1q_f32(xp + 16));
        float32x4_t m5 = vabsq_f32(vld1q_f32(xp + 20));
        float32x4_t m6 = vabsq_f32(vld1q_f32(xp + 24));
        float32x4_t m7 = vabsq_f32(vld1q_f32(xp + 28));
        m0 = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
        m4 = vmaxq_f32(vmaxq_f32(m4, m5), vmaxq_f32(m6, m7));
        const float amax = vmaxvq_f32(vmaxq_f32(m0, m4));
        const float d    = amax * (1.0f / 127.0f);
        const float id   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;
        const float32x4_t vid = vdupq_n_f32(id);
        int32_t bsum = 0;
        for (int g = 0; g < 2; ++g) {
            const float * gp = xp + g * 16;
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  0), vid));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  4), vid));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  8), vid));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp + 12), vid));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            int8x16_t qv = vmaxq_s8(vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)), min127);
            vst1q_s8(dst[b].qs + g * 16, qv);
            bsum += static_cast<int32_t>(vaddlvq_s8(qv));
        }
        dst[b].s = d * static_cast<float>(bsum);
    }
#else
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d  = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;
        int32_t bsum = 0;
        for (int i = 0; i < 32; ++i) {
            int v = static_cast<int>(std::lrintf(xb[i] * id));
            v = std::max(-127, std::min(127, v));
            dst[b].qs[i] = static_cast<int8_t>(v);
            bsum += v;
        }
        dst[b].s = d * static_cast<float>(bsum);
    }
#endif
}

void quantize_rows_q8_0(ThreadPool & pool, const float * x, CpuActQ8_0 * dst, int T, int K) {
    const int nb = K >> 5;
    if (T <= 2) {
        for (int t = 0; t < T; ++t) quantize_row_q8_0(x + static_cast<size_t>(t) * K, dst + static_cast<size_t>(t) * nb, K);
        return;
    }
    pool.parallel_for(T, [&](int t0, int t1, int) {
        for (int t = t0; t < t1; ++t) {
            quantize_row_q8_0(x + static_cast<size_t>(t) * K, dst + static_cast<size_t>(t) * nb, K);
        }
    });
}

void quantize_row_q8_k(const float * x, CpuActQ8_K * dst, int K) {
    const int nb = K >> 8;
#if defined(__AVX2__)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256i min127   = _mm256_set1_epi8(-127);
    const __m256i ones16   = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 256;
        __m256 m0 = _mm256_setzero_ps();
        __m256 m1 = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 16) {
            m0 = _mm256_max_ps(m0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i)));
            m1 = _mm256_max_ps(m1, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i + 8)));
        }
        __m256 m = _mm256_max_ps(m0, m1);
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        const float amax = _mm_cvtss_f32(m128);
        const float d    = amax / 127.0f;
        const float id   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;
        const __m256 vid = _mm256_set1_ps(id);

        for (int j = 0; j < 8; ++j) {
            const float * p = xb + j * 32;
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 0),  vid));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 8),  vid));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 16), vid));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 24), vid));
            __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), _MM_SHUFFLE(3, 1, 2, 0));
            __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), _MM_SHUFFLE(3, 1, 2, 0));
            __m256i q8  = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), _MM_SHUFFLE(3, 1, 2, 0));
            q8 = _mm256_max_epi8(q8, min127);
            _mm256_store_si256(reinterpret_cast<__m256i *>(dst[b].qs + j * 32), q8);

            __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8));
            __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8, 1));
            dst[b].bsums[2 * j + 0] = static_cast<int16_t>(hsum256_epi32(_mm256_madd_epi16(lo16, ones16)));
            dst[b].bsums[2 * j + 1] = static_cast<int16_t>(hsum256_epi32(_mm256_madd_epi16(hi16, ones16)));
        }
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (int b = 0; b < nb; ++b) {
        const float * xp = x + b * 256;
        float32x4_t vmin0 = vld1q_f32(xp), vmax0 = vmin0;
        float32x4_t vmin1 = vld1q_f32(xp + 4), vmax1 = vmin1;
        for (int i = 8; i < 256; i += 8) {
            float32x4_t v0 = vld1q_f32(xp + i);
            float32x4_t v1 = vld1q_f32(xp + i + 4);
            vmin0 = vminq_f32(vmin0, v0); vmax0 = vmaxq_f32(vmax0, v0);
            vmin1 = vminq_f32(vmin1, v1); vmax1 = vmaxq_f32(vmax1, v1);
        }
        const float mn = vminvq_f32(vminq_f32(vmin0, vmin1));
        const float mx = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        const float signed_max = (-mn > mx) ? mn : mx;
        if (signed_max == 0.0f) {
            dst[b].d = 0.0f;
            std::memset(dst[b].qs, 0, sizeof(dst[b].qs));
            std::memset(dst[b].bsums, 0, sizeof(dst[b].bsums));
            continue;
        }
        const float iscale = -127.0f / signed_max;
        dst[b].d = 1.0f / iscale;
        const float32x4_t vid = vdupq_n_f32(iscale);
        for (int j = 0; j < 16; ++j) {
            const float * gp = xp + j * 16;
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  0), vid));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  4), vid));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp +  8), vid));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gp + 12), vid));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            int8x16_t qv = vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1));
            vst1q_s8(dst[b].qs + j * 16, qv);
            dst[b].bsums[j] = static_cast<int16_t>(vaddlvq_s8(qv));
        }
    }
#else
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d  = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int k = 0; k < 16; ++k) {
                int v = static_cast<int>(std::lrintf(xb[j * 16 + k] * id));
                v = std::max(-127, std::min(127, v));
                dst[b].qs[j * 16 + k] = static_cast<int8_t>(v);
                sum += v;
            }
            dst[b].bsums[j] = static_cast<int16_t>(sum);
        }
    }
#endif
}

void quantize_rows_q8_k(ThreadPool & pool, const float * x, CpuActQ8_K * dst, int T, int K) {
    const int nb = K >> 8;
    if (T <= 2) {
        for (int t = 0; t < T; ++t) quantize_row_q8_k(x + static_cast<size_t>(t) * K, dst + static_cast<size_t>(t) * nb, K);
        return;
    }
    pool.parallel_for(T, [&](int t0, int t1, int) {
        for (int t = t0; t < t1; ++t) {
            quantize_row_q8_k(x + static_cast<size_t>(t) * K, dst + static_cast<size_t>(t) * nb, K);
        }
    });
}

void embedding_lookup(const CpuTensor & W, const int32_t * tokens, float * dst, int n_embd, int T) {
    const auto tr = dtype_traits(W.dtype);
    const size_t row_bytes = (static_cast<size_t>(n_embd) / tr.block_elems) * tr.block_bytes;
    const auto * base = static_cast<const uint8_t *>(W.ptr);
    for (int t = 0; t < T; ++t) {
        const size_t tid = static_cast<size_t>(tokens[t]);
        dequant_slice_to_f32(W.dtype, base + tid * row_bytes, 0, n_embd, dst + static_cast<size_t>(t) * n_embd);
    }
}

void dequant_row_to_f32(DType dt, const void * row_ptr, int K, float * dst) {
    dequant_slice_to_f32(dt, row_ptr, 0, K, dst);
}

void rmsnorm_unweighted(const float * x, float * y, int n_embd, int T, float eps) {
    for (int t = 0; t < T; ++t) {
        const float * xr = x + static_cast<size_t>(t) * n_embd;
        float * yr = y + static_cast<size_t>(t) * n_embd;
        float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 15 < n_embd; i += 16) {
            const __m256 v0 = _mm256_loadu_ps(xr + i);
            const __m256 v1 = _mm256_loadu_ps(xr + i + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        ss = hsum256_ps(_mm256_add_ps(acc0, acc1));
        for (; i < n_embd; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 7 < n_embd; i += 8) _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_loadu_ps(xr + i), vinv));
        for (; i < n_embd; ++i) yr[i] = xr[i] * inv;
#else
        for (int i = 0; i < n_embd; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) yr[i] = xr[i] * inv;
#endif
    }
}

void rmsnorm(const float * x, const float * w, float * y, int n_embd, int T, float eps) {
    for (int t = 0; t < T; ++t) {
        const float * xr = x + static_cast<size_t>(t) * n_embd;
        float * yr = y + static_cast<size_t>(t) * n_embd;
        float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 15 < n_embd; i += 16) {
            const __m256 v0 = _mm256_loadu_ps(xr + i);
            const __m256 v1 = _mm256_loadu_ps(xr + i + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        ss = hsum256_ps(_mm256_add_ps(acc0, acc1));
        for (; i < n_embd; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 7 < n_embd; i += 8) {
            _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vinv), _mm256_loadu_ps(w + i)));
        }
        for (; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#elif defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= n_embd; i += 8) {
            const float32x4_t v0 = vld1q_f32(xr + i);
            const float32x4_t v1 = vld1q_f32(xr + i + 4);
            s0 = vfmaq_f32(s0, v0, v0);
            s1 = vfmaq_f32(s1, v1, v1);
        }
        ss = vaddvq_f32(vaddq_f32(s0, s1));
        for (; i < n_embd; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        const float32x4_t vinv = vdupq_n_f32(inv);
        i = 0;
        for (; i + 8 <= n_embd; i += 8) {
            vst1q_f32(yr + i,     vmulq_f32(vmulq_f32(vld1q_f32(xr + i),     vinv), vld1q_f32(w + i)));
            vst1q_f32(yr + i + 4, vmulq_f32(vmulq_f32(vld1q_f32(xr + i + 4), vinv), vld1q_f32(w + i + 4)));
        }
        for (; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#else
        for (int i = 0; i < n_embd; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#endif
    }
}

void add_rmsnorm(float * x_inout, const float * sub_out, const float * w, float * y, int n_embd, int T, float eps) {
    for (int t = 0; t < T; ++t) {
        float * xr = x_inout + static_cast<size_t>(t) * n_embd;
        const float * sr = sub_out + static_cast<size_t>(t) * n_embd;
        float * yr = y + static_cast<size_t>(t) * n_embd;
        float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < n_embd; i += 8) {
            const __m256 v = _mm256_add_ps(_mm256_loadu_ps(xr + i), _mm256_loadu_ps(sr + i));
            _mm256_storeu_ps(xr + i, v);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        ss = hsum256_ps(acc);
        for (; i < n_embd; ++i) {
            const float v = xr[i] + sr[i];
            xr[i] = v;
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 7 < n_embd; i += 8) {
            _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vinv), _mm256_loadu_ps(w + i)));
        }
        for (; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#elif defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= n_embd; i += 8) {
            const float32x4_t v0 = vaddq_f32(vld1q_f32(xr + i),     vld1q_f32(sr + i));
            const float32x4_t v1 = vaddq_f32(vld1q_f32(xr + i + 4), vld1q_f32(sr + i + 4));
            vst1q_f32(xr + i,     v0);
            vst1q_f32(xr + i + 4, v1);
            s0 = vfmaq_f32(s0, v0, v0);
            s1 = vfmaq_f32(s1, v1, v1);
        }
        ss = vaddvq_f32(vaddq_f32(s0, s1));
        for (; i < n_embd; ++i) {
            const float v = xr[i] + sr[i];
            xr[i] = v;
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        const float32x4_t vinv = vdupq_n_f32(inv);
        i = 0;
        for (; i + 8 <= n_embd; i += 8) {
            vst1q_f32(yr + i,     vmulq_f32(vmulq_f32(vld1q_f32(xr + i),     vinv), vld1q_f32(w + i)));
            vst1q_f32(yr + i + 4, vmulq_f32(vmulq_f32(vld1q_f32(xr + i + 4), vinv), vld1q_f32(w + i + 4)));
        }
        for (; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#else
        for (int i = 0; i < n_embd; ++i) {
            const float v = xr[i] + sr[i];
            xr[i] = v;
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) yr[i] = xr[i] * inv * w[i];
#endif
    }
}

void add_inplace(float * a, const float * b, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(a + i,     vaddq_f32(vld1q_f32(a + i),     vld1q_f32(b + i)));
        vst1q_f32(a + i + 4, vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
#endif
    for (; i < n; ++i) a[i] += b[i];
}

void scale_inplace(float * a, size_t n, float scale) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vs));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(a + i + 0, vmulq_f32(vld1q_f32(a + i + 0), vs));
        vst1q_f32(a + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vs));
    }
#endif
    for (; i < n; ++i) a[i] *= scale;
}

void linear(ThreadPool & pool,
            const CpuTensor & W,
            const float * x,
            const CpuActQ8_0 * xq8,
            const CpuActQ8_K * xq8k,
            float * y,
            int T) {
    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
    if (T >= 12) {
        const int n_tiles = (T + 15) >> 4;
        thread_local std::vector<float> xp_buf;
        xp_buf.resize(static_cast<size_t>(n_tiles) * K * 16);
        float * xp = xp_buf.data();
        pack_x_panels_16_mt(pool, x, xp, T, K);
        pool.parallel_for_blocked(M, 6, [&](int r0, int r1, int) {
            gemm_fma_rows_range(W, xp, y, T, r0, r1);
        });
        return;
    }
#endif

    thread_local std::vector<CpuActQ8_0> local_q8;
    thread_local std::vector<CpuActQ8_K> local_q8k;
    if (needs_act_q8_0(W.dtype) && xq8 == nullptr) {
        local_q8.resize(static_cast<size_t>(T) * (K >> 5));
        quantize_rows_q8_0(pool, x, local_q8.data(), T, K);
        xq8 = local_q8.data();
    } else if (needs_act_q8_k(W.dtype) && xq8k == nullptr) {
        local_q8k.resize(static_cast<size_t>(T) * (K >> 8));
        quantize_rows_q8_k(pool, x, local_q8k.data(), T, K);
        xq8k = local_q8k.data();
    }

#if defined(__aarch64__)
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
    thread_local std::vector<ArmQ80Lane4> p4_q8;
    thread_local std::vector<ArmQ8KLane4> p4_q8k;
    const ArmQ80Lane4 * xq8_p4  = nullptr;
    const ArmQ8KLane4 * xq8k_p4 = nullptr;
    if (T >= 4) {
        const int nt4 = T >> 2;
        if (xq8 != nullptr && (W.dtype == DType::Q8_0 || W.dtype == DType::Q4_0 || W.dtype == DType::Q5_0 || W.dtype == DType::IQ4_NL)) {
            const int nb = K >> 5;
            p4_q8.resize(static_cast<size_t>(nt4) * nb);
            ArmQ80Lane4 * dst = p4_q8.data();
            if (nt4 >= 4 && pool.num_threads() > 1) {
                pool.parallel_for(nt4, [&](int a, int b, int) { pack_q8_0_lane4_range(xq8, dst, a, b, nb); });
            } else {
                pack_q8_0_lane4_range(xq8, dst, 0, nt4, nb);
            }
            xq8_p4 = dst;
        } else if (xq8k != nullptr && (W.dtype == DType::Q4_K || W.dtype == DType::Q6_K)) {
            const int nb = K >> 8;
            p4_q8k.resize(static_cast<size_t>(nt4) * nb);
            ArmQ8KLane4 * dst = p4_q8k.data();
            if (nt4 >= 4 && pool.num_threads() > 1) {
                pool.parallel_for(nt4, [&](int a, int b, int) { pack_q8_k_lane4_range(xq8k, dst, a, b, nb); });
            } else {
                pack_q8_k_lane4_range(xq8k, dst, 0, nt4, nb);
            }
            xq8k_p4 = dst;
        }
    }
#endif
    if (T >= 16 && M >= 512 && pool.num_compute_threads() > 1) {
        // Fine-grained lock-free work stealing across all compute cores (big + LITTLE on RK3588)
        // keeps fast cores 100% fed while eliminating straggler tails during compute-bound prefill.
        const int chunk = (M >= 4096) ? 32 : 16;
        std::atomic<int> next_r{0};
        pool.parallel_compute([&](int, int) {
            while (true) {
                const int r0 = next_r.fetch_add(chunk, std::memory_order_relaxed);
                if (r0 >= M) break;
                const int r1 = std::min(M, r0 + chunk);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1, xq8_p4, xq8k_p4);
#else
                linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1);
#endif
            }
        });
        return;
    }
    if (M >= 1024 && pool.num_threads() > 1) {
        // Dynamic 128-row work stealing absorbs big-core cluster frequency skew (e.g. 2.35 vs 2.26 GHz on RK3588)
        const int chunk = (M >= 32768) ? 512 : 128;
        std::atomic<int> next_r{0};
        pool.parallel([&](int, int) {
            while (true) {
                const int r0 = next_r.fetch_add(chunk, std::memory_order_relaxed);
                if (r0 >= M) break;
                const int r1 = std::min(M, r0 + chunk);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1, xq8_p4, xq8k_p4);
#else
                linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1);
#endif
            }
        });
        return;
    }
    pool.parallel_for_blocked(M, 4, [&](int r0, int r1, int) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
        linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1, xq8_p4, xq8k_p4);
#else
        linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1);
#endif
    });
    return;
#endif
    pool.parallel_for_blocked(M, 4, [&](int r0, int r1, int) {
        linear_rows_range_direct(W, x, xq8, xq8k, y, T, r0, r1);
    });
}

void linear_multi(ThreadPool & pool,
                  const LinearSpec * specs,
                  int n_specs,
                  const float * x,
                  const CpuActQ8_0 * xq8,
                  const CpuActQ8_K * xq8k,
                  int T) {
    if (n_specs <= 0) return;
    const int K = static_cast<int>(specs[0].W->ne[0]);

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
    if (T >= 12) {
        int total_rows = 0;
        int offsets[16];
        for (int i = 0; i < n_specs; ++i) {
            offsets[i] = total_rows;
            total_rows += static_cast<int>(specs[i].W->ne[1]);
        }
        offsets[n_specs] = total_rows;
        const int n_tiles = (T + 15) >> 4;
        thread_local std::vector<float> xp_buf;
        xp_buf.resize(static_cast<size_t>(n_tiles) * K * 16);
        float * xp = xp_buf.data();
        pack_x_panels_16_mt(pool, x, xp, T, K);
        pool.parallel_for_blocked(total_rows, 6, [&](int g0, int g1, int) {
            for (int i = 0; i < n_specs; ++i) {
                const int s0 = offsets[i];
                const int s1 = offsets[i + 1];
                const int r0 = std::max(g0, s0) - s0;
                const int r1 = std::min(g1, s1) - s0;
                if (r0 < r1) {
                    gemm_fma_rows_range(*specs[i].W, xp, specs[i].y, T, r0, r1);
                }
            }
        });
        return;
    }
#endif

    thread_local std::vector<CpuActQ8_0> local_q8;
    thread_local std::vector<CpuActQ8_K> local_q8k;
    bool need_q8 = false, need_q8k = false;
    for (int i = 0; i < n_specs; ++i) {
        if (needs_act_q8_0(specs[i].W->dtype)) need_q8 = true;
        if (needs_act_q8_k(specs[i].W->dtype)) need_q8k = true;
    }
    if (need_q8 && xq8 == nullptr) {
        local_q8.resize(static_cast<size_t>(T) * (K >> 5));
        quantize_rows_q8_0(pool, x, local_q8.data(), T, K);
        xq8 = local_q8.data();
    }
    if (need_q8k && xq8k == nullptr) {
        local_q8k.resize(static_cast<size_t>(T) * (K >> 8));
        quantize_rows_q8_k(pool, x, local_q8k.data(), T, K);
        xq8k = local_q8k.data();
    }

#if defined(__aarch64__)
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
    thread_local std::vector<ArmQ80Lane4> p4_q8;
    thread_local std::vector<ArmQ8KLane4> p4_q8k;
    const ArmQ80Lane4 * xq8_p4  = nullptr;
    const ArmQ8KLane4 * xq8k_p4 = nullptr;
    if (T >= 4) {
        const int nt4 = T >> 2;
        bool want_p4_q8 = false, want_p4_q8k = false;
        for (int i = 0; i < n_specs; ++i) {
            const DType dt = specs[i].W->dtype;
            if (dt == DType::Q8_0 || dt == DType::Q4_0 || dt == DType::Q5_0 || dt == DType::IQ4_NL) want_p4_q8 = true;
            if (dt == DType::Q4_K || dt == DType::Q6_K) want_p4_q8k = true;
        }
        if (want_p4_q8 && xq8 != nullptr) {
            const int nb = K >> 5;
            p4_q8.resize(static_cast<size_t>(nt4) * nb);
            ArmQ80Lane4 * dst = p4_q8.data();
            if (nt4 >= 4 && pool.num_threads() > 1) {
                pool.parallel_for(nt4, [&](int a, int b, int) { pack_q8_0_lane4_range(xq8, dst, a, b, nb); });
            } else {
                pack_q8_0_lane4_range(xq8, dst, 0, nt4, nb);
            }
            xq8_p4 = dst;
        }
        if (want_p4_q8k && xq8k != nullptr) {
            const int nb = K >> 8;
            p4_q8k.resize(static_cast<size_t>(nt4) * nb);
            ArmQ8KLane4 * dst = p4_q8k.data();
            if (nt4 >= 4 && pool.num_threads() > 1) {
                pool.parallel_for(nt4, [&](int a, int b, int) { pack_q8_k_lane4_range(xq8k, dst, a, b, nb); });
            } else {
                pack_q8_k_lane4_range(xq8k, dst, 0, nt4, nb);
            }
            xq8k_p4 = dst;
        }
    }
#endif
    if (T >= 16 && pool.num_compute_threads() > 1) {
        // Flatten all specs into fine-grained 16/32-row dynamic chunks across big + LITTLE cores
        int total_rows = 0;
        for (int i = 0; i < n_specs; ++i) total_rows += static_cast<int>(specs[i].W->ne[1]);
        const int chunk = (total_rows >= 4096) ? 32 : 16;
        struct ChunkItem { int spec; int r0; int r1; };
        thread_local std::vector<ChunkItem> items;
        items.clear();
        for (int i = 0; i < n_specs; ++i) {
            const int M = static_cast<int>(specs[i].W->ne[1]);
            for (int r0 = 0; r0 < M; r0 += chunk) {
                items.push_back({i, r0, std::min(M, r0 + chunk)});
            }
        }
        const int n_items = static_cast<int>(items.size());
        const ChunkItem * items_p = items.data();
        std::atomic<int> next_item{0};
        pool.parallel_compute([&](int, int) {
            while (true) {
                const int idx = next_item.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_items) break;
                const auto & it = items_p[idx];
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                linear_rows_range_direct(*specs[it.spec].W, x, xq8, xq8k, specs[it.spec].y, T, it.r0, it.r1, xq8_p4, xq8k_p4);
#else
                linear_rows_range_direct(*specs[it.spec].W, x, xq8, xq8k, specs[it.spec].y, T, it.r0, it.r1);
#endif
            }
        });
        return;
    }
    pool.parallel([&](int tid, int nth) {
        for (int i = 0; i < n_specs; ++i) {
            const int M = static_cast<int>(specs[i].W->ne[1]);
            const int n_blk = (M + 3) / 4;
            const int base  = n_blk / nth;
            const int rem   = n_blk % nth;
            const int b0    = tid * base + std::min(tid, rem);
            const int b1    = b0 + base + (tid < rem ? 1 : 0);
            const int r0    = b0 * 4;
            const int r1    = std::min(M, b1 * 4);
            if (r0 < r1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                linear_rows_range_direct(*specs[i].W, x, xq8, xq8k, specs[i].y, T, r0, r1, xq8_p4, xq8k_p4);
#else
                linear_rows_range_direct(*specs[i].W, x, xq8, xq8k, specs[i].y, T, r0, r1);
#endif
            }
        }
    });
    return;
#endif
    // Partition EACH spec evenly across all threads inside ONE barrier so mixed-dtype specs
    // (e.g. Q4_K Wq/Wk + Q6_K Wv) never create straggler threads!
    pool.parallel([&](int tid, int nth) {
        for (int i = 0; i < n_specs; ++i) {
            const int M = static_cast<int>(specs[i].W->ne[1]);
            const int n_blk = (M + 3) / 4;
            const int base  = n_blk / nth;
            const int rem   = n_blk % nth;
            const int b0    = tid * base + std::min(tid, rem);
            const int b1    = b0 + base + (tid < rem ? 1 : 0);
            const int r0    = b0 * 4;
            const int r1    = std::min(M, b1 * 4);
            if (r0 < r1) {
                linear_rows_range_direct(*specs[i].W, x, xq8, xq8k, specs[i].y, T, r0, r1);
            }
        }
    });
}

void silu_mul(const float * gate, const float * up, float * out, size_t n) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        const __m256 gv = _mm256_loadu_ps(gate + i);
        const __m256 uv = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(silu_ps_avx2(gv), uv));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const float32x4_t v1 = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t gv = vld1q_f32(gate + i);
        const float32x4_t uv = vld1q_f32(up + i);
        const float32x4_t denom = vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(gv)));
        vst1q_f32(out + i, vmulq_f32(vdivq_f32(gv, denom), uv));
    }
#endif
    for (; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

void geglu_tanh_mul(float * gate_inout, const float * up, size_t n) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        const __m256 gv = _mm256_loadu_ps(gate_inout + i);
        const __m256 uv = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(gate_inout + i, _mm256_mul_ps(gelu_tanh_ps_avx2(gv), uv));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        const float32x4_t gv = vld1q_f32(gate_inout + i);
        const float32x4_t uv = vld1q_f32(up + i);
        vst1q_f32(gate_inout + i, vmulq_f32(vgelu_tanh_f32_fast(gv), uv));
    }
#endif
    for (; i < n; ++i) {
        gate_inout[i] = gelu_tanh_scalar(gate_inout[i]) * up[i];
    }
}

void geglu_tanh_mul_mt(ThreadPool & pool, float * gate_inout, const float * up, size_t n) {
    if (n < 4096 || pool.num_threads() <= 1) {
        geglu_tanh_mul(gate_inout, up, n);
        return;
    }
    const int n_int = static_cast<int>(n);
    pool.parallel_for_blocked(n_int, 64, [&](int i0, int i1, int) {
        geglu_tanh_mul(gate_inout + i0, up + i0, static_cast<size_t>(i1 - i0));
    });
}

void softcap_tanh(float * logits, int n, float cap) {
    if (cap <= 0.0f) return;
    const float inv_cap = 1.0f / cap;
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 vcap  = _mm256_set1_ps(cap);
    const __m256 vicap = _mm256_set1_ps(inv_cap);
    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(logits + i), vicap);
        _mm256_storeu_ps(logits + i, _mm256_mul_ps(vcap, tanh_ps_avx2(v)));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const float32x4_t vcap  = vdupq_n_f32(cap);
    const float32x4_t vicap = vdupq_n_f32(inv_cap);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vmulq_f32(vld1q_f32(logits + i), vicap);
        vst1q_f32(logits + i, vmulq_f32(vcap, vtanhq_f32_fast(v)));
    }
#endif
    for (; i < n; ++i) {
        logits[i] = cap * std::tanh(logits[i] * inv_cap);
    }
}

void silu_mul_mt(ThreadPool & pool, const float * gate, const float * up, float * out, size_t n) {
    if (n < 16384 || pool.num_threads() <= 1) {
        silu_mul(gate, up, out, n);
        return;
    }
    const int n_chunks = static_cast<int>((n + 255) / 256);
    pool.parallel_for(n_chunks, [&](int c0, int c1, int) {
        const size_t off = static_cast<size_t>(c0) * 256;
        const size_t end = std::min(n, static_cast<size_t>(c1) * 256);
        if (off < end) silu_mul(gate + off, up + off, out + off, end - off);
    });
}

void dense_ffn_swiglu_t1(
    ThreadPool & pool,
    const CpuTensor & w_gate,
    const CpuTensor & w_up,
    const CpuTensor & w_down,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * out) {

    const int n_embd = static_cast<int>(w_gate.ne[0]);
    const int n_ff   = static_cast<int>(w_gate.ne[1]);

    const auto tr_g = dtype_traits(w_gate.dtype);
    const auto tr_u = dtype_traits(w_up.dtype);
    const size_t g_row_bytes = (static_cast<size_t>(n_embd) / tr_g.block_elems) * tr_g.block_bytes;
    const size_t u_row_bytes = (static_cast<size_t>(n_embd) / tr_u.block_elems) * tr_u.block_bytes;

    const bool down_q8  = needs_act_q8_0(w_down.dtype);
    const bool down_q8k = needs_act_q8_k(w_down.dtype);
    const bool need_f32 = (!down_q8 && !down_q8k);

    thread_local std::vector<float>      hid_f32;
    thread_local std::vector<CpuActQ8_0> hid_q8;
    thread_local std::vector<CpuActQ8_K> hid_q8k;

    if (need_f32) hid_f32.resize(static_cast<size_t>(n_ff));
    if (down_q8)  hid_q8.resize(static_cast<size_t>(n_ff >> 5));
    if (down_q8k) hid_q8k.resize(static_cast<size_t>(n_ff >> 8));

    float *      hid_f32_p = need_f32 ? hid_f32.data() : nullptr;
    CpuActQ8_0 * hid_q8_p  = down_q8  ? hid_q8.data()  : nullptr;
    CpuActQ8_K * hid_q8k_p = down_q8k ? hid_q8k.data() : nullptr;

    // Barrier 1: Dynamic lock-free 256-row block queue for gate & up -> L1d SiLU + parallel quant
    // Eliminates remainder & clock-tier skew across heterogeneous big-core frequencies
    const int n_blks = n_ff >> 8;
    const auto * g_ptr = static_cast<const uint8_t *>(w_gate.ptr);
    const auto * u_ptr = static_cast<const uint8_t *>(w_up.ptr);
    std::atomic<int> next_blk{0};

    pool.parallel([&](int, int) {
        alignas(64) float g_tmp[256];
        alignas(64) float u_tmp[256];
        while (true) {
            const int blk = next_blk.fetch_add(1, std::memory_order_relaxed);
            if (blk >= n_blks) break;
            const int r0 = blk * 256;
            CpuTensor Wg_sub{g_ptr + static_cast<size_t>(r0) * g_row_bytes, w_gate.dtype,
                             {static_cast<uint64_t>(n_embd), 256, 1, 1}, g_row_bytes * 256};
            CpuTensor Wu_sub{u_ptr + static_cast<size_t>(r0) * u_row_bytes, w_up.dtype,
                             {static_cast<uint64_t>(n_embd), 256, 1, 1}, u_row_bytes * 256};
            linear_rows_range_direct(Wg_sub, x, xq8, xq8k, g_tmp, 1, 0, 256);
            linear_rows_range_direct(Wu_sub, x, xq8, xq8k, u_tmp, 1, 0, 256);
            silu_mul(g_tmp, u_tmp, g_tmp, 256);
            if (down_q8k) quantize_row_q8_k(g_tmp, hid_q8k_p + blk, 256);
            if (down_q8)  quantize_row_q8_0(g_tmp, hid_q8_p + blk * 8, 256);
            if (need_f32) std::memcpy(hid_f32_p + r0, g_tmp, 256 * sizeof(float));
        }
    });

    // Barrier 2: Down projection directly from L1/L2-hot quantized hidden state
    linear(pool, w_down, hid_f32_p, hid_q8_p, hid_q8k_p, out, 1);
}

void dense_ffn_geglu_t1(
    ThreadPool & pool,
    const CpuTensor & w_gate,
    const CpuTensor & w_up,
    const CpuTensor & w_down,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * out) {

    const int n_embd = static_cast<int>(w_gate.ne[0]);
    const int n_ff   = static_cast<int>(w_gate.ne[1]);

    const auto tr_g = dtype_traits(w_gate.dtype);
    const auto tr_u = dtype_traits(w_up.dtype);
    const size_t g_row_bytes = (static_cast<size_t>(n_embd) / tr_g.block_elems) * tr_g.block_bytes;
    const size_t u_row_bytes = (static_cast<size_t>(n_embd) / tr_u.block_elems) * tr_u.block_bytes;

    thread_local std::vector<CpuActQ8_0> in_q8;
    thread_local std::vector<CpuActQ8_K> in_q8k;
    if ((needs_act_q8_0(w_gate.dtype) || needs_act_q8_0(w_up.dtype)) && xq8 == nullptr) {
        in_q8.resize(static_cast<size_t>(n_embd >> 5));
        quantize_row_q8_0(x, in_q8.data(), n_embd);
        xq8 = in_q8.data();
    }
    if ((needs_act_q8_k(w_gate.dtype) || needs_act_q8_k(w_up.dtype)) && xq8k == nullptr) {
        in_q8k.resize(static_cast<size_t>(n_embd >> 8));
        quantize_row_q8_k(x, in_q8k.data(), n_embd);
        xq8k = in_q8k.data();
    }

    const bool down_q8  = needs_act_q8_0(w_down.dtype);
    const bool down_q8k = needs_act_q8_k(w_down.dtype);
    const bool need_f32 = (!down_q8 && !down_q8k);

    thread_local std::vector<float>      hid_f32;
    thread_local std::vector<CpuActQ8_0> hid_q8;
    thread_local std::vector<CpuActQ8_K> hid_q8k;

    if (need_f32) hid_f32.resize(static_cast<size_t>(n_ff));
    if (down_q8)  hid_q8.resize(static_cast<size_t>(n_ff >> 5));
    if (down_q8k) hid_q8k.resize(static_cast<size_t>(n_ff >> 8));

    float *      hid_f32_p = need_f32 ? hid_f32.data() : nullptr;
    CpuActQ8_0 * hid_q8_p  = down_q8  ? hid_q8.data()  : nullptr;
    CpuActQ8_K * hid_q8k_p = down_q8k ? hid_q8k.data() : nullptr;

    // Barrier 1: Dynamic 256-row block queue for gate & up -> L1d-hot GeGLU-tanh + parallel quant
    const int n_blks = (n_ff + 255) >> 8;
    const auto * g_ptr = static_cast<const uint8_t *>(w_gate.ptr);
    const auto * u_ptr = static_cast<const uint8_t *>(w_up.ptr);
    std::atomic<int> next_blk{0};

    pool.parallel([&](int, int) {
        alignas(64) float g_tmp[256];
        alignas(64) float u_tmp[256];
        while (true) {
            const int blk = next_blk.fetch_add(1, std::memory_order_relaxed);
            if (blk >= n_blks) break;
            const int r0 = blk * 256;
            const int nr = std::min(256, n_ff - r0);
            CpuTensor Wg_sub{g_ptr + static_cast<size_t>(r0) * g_row_bytes, w_gate.dtype,
                             {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(nr), 1, 1}, g_row_bytes * static_cast<size_t>(nr)};
            CpuTensor Wu_sub{u_ptr + static_cast<size_t>(r0) * u_row_bytes, w_up.dtype,
                             {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(nr), 1, 1}, u_row_bytes * static_cast<size_t>(nr)};
            linear_rows_range_direct(Wg_sub, x, xq8, xq8k, g_tmp, 1, 0, nr);
            linear_rows_range_direct(Wu_sub, x, xq8, xq8k, u_tmp, 1, 0, nr);
            geglu_tanh_mul(g_tmp, u_tmp, static_cast<size_t>(nr));
            if (down_q8k) quantize_row_q8_k(g_tmp, hid_q8k_p + blk, nr);
            if (down_q8)  quantize_row_q8_0(g_tmp, hid_q8_p + (r0 >> 5), nr);
            if (need_f32) std::memcpy(hid_f32_p + r0, g_tmp, static_cast<size_t>(nr) * sizeof(float));
        }
    });

    // Barrier 2: Down projection directly from L1/L2-hot quantized hidden state
    linear(pool, w_down, hid_f32_p, hid_q8_p, hid_q8k_p, out, 1);
}

void add_bias_rows(float * y, const float * bias, int T, int M) {
    for (int t = 0; t < T; ++t) {
        float * yt = y + static_cast<size_t>(t) * M;
        int j = 0;
#if defined(__AVX2__)
        for (; j + 7 < M; j += 8) {
            _mm256_storeu_ps(yt + j, _mm256_add_ps(_mm256_loadu_ps(yt + j), _mm256_loadu_ps(bias + j)));
        }
#elif defined(__aarch64__) && defined(__ARM_NEON)
        for (; j + 8 <= M; j += 8) {
            vst1q_f32(yt + j + 0, vaddq_f32(vld1q_f32(yt + j + 0), vld1q_f32(bias + j + 0)));
            vst1q_f32(yt + j + 4, vaddq_f32(vld1q_f32(yt + j + 4), vld1q_f32(bias + j + 4)));
        }
#endif
        for (; j < M; ++j) yt[j] += bias[j];
    }
}

void neox_rope_qk(
    float * q, float * k, const int32_t * pos,
    int T, int n_head, int n_head_kv, int head_dim, float rope_freq_base) {

    const int half_d = head_dim >> 1;
    float freqs[128];
    for (int d = 0; d < half_d; ++d) {
        freqs[d] = std::pow(rope_freq_base, -static_cast<float>(2 * d) / static_cast<float>(head_dim));
    }

    alignas(32) float cosv[128];
    alignas(32) float sinv[128];
    for (int t = 0; t < T; ++t) {
        const float p = static_cast<float>(pos[t]);
        for (int d = 0; d < half_d; ++d) {
            const float th = p * freqs[d];
            cosv[d] = std::cos(th);
            sinv[d] = std::sin(th);
        }

        auto apply_heads = [&](float * buf, int nh) {
            for (int h = 0; h < nh; ++h) {
                float * hp = buf + (static_cast<size_t>(t) * nh + h) * head_dim;
                int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                for (; d + 4 <= half_d; d += 4) {
                    const float32x4_t x0 = vld1q_f32(hp + d);
                    const float32x4_t x1 = vld1q_f32(hp + half_d + d);
                    const float32x4_t cv = vld1q_f32(cosv + d);
                    const float32x4_t sv = vld1q_f32(sinv + d);
                    vst1q_f32(hp + d,          vfmsq_f32(vmulq_f32(x0, cv), x1, sv));
                    vst1q_f32(hp + half_d + d, vfmaq_f32(vmulq_f32(x0, sv), x1, cv));
                }
#endif
                for (; d < half_d; ++d) {
                    const float x0 = hp[d];
                    const float x1 = hp[half_d + d];
                    hp[d]          = x0 * cosv[d] - x1 * sinv[d];
                    hp[half_d + d] = x0 * sinv[d] + x1 * cosv[d];
                }
            }
        };
        apply_heads(q, n_head);
        apply_heads(k, n_head_kv);
    }
}

static inline void quantize_128_q4_0_pos(const float * src, CpuGqaKv128Q4Pos & dst) noexcept {
    for (int b = 0; b < 4; ++b) {
        const float * xb = src + b * 32;
        float mn = xb[0], mx = xb[0];
        for (int i = 1; i < 32; ++i) {
            if (xb[i] < mn) mn = xb[i];
            if (xb[i] > mx) mx = xb[i];
        }
        const float smax = (-mn > mx) ? mn : mx;
        const float d    = smax * (-0.125f);
        const float id   = (d != 0.0f) ? (1.0f / d) : 0.0f;
        dst.d[b] = d;
        uint8_t * qp = dst.qs + b * 16;
        for (int j = 0; j < 16; ++j) {
            int q0 = static_cast<int>(xb[j] * id + 8.5f);
            int q1 = static_cast<int>(xb[j + 16] * id + 8.5f);
            q0 = std::max(0, std::min(15, q0));
            q1 = std::max(0, std::min(15, q1));
            qp[j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

void gqa_kv_write_d128(
    const float * k, const float * v, const int32_t * pos,
    void * k_cache, void * v_cache, KvCacheType kv_type,
    int T, int n_head_kv, int max_ctx) {

    if (kv_type == KvCacheType::Q4_0) {
        auto * kc = static_cast<CpuGqaKv128Q4Pos *>(k_cache);
        auto * vc = static_cast<CpuGqaKv128Q4Pos *>(v_cache);
        for (int t = 0; t < T; ++t) {
            const int pidx = pos[t];
            for (int h = 0; h < n_head_kv; ++h) {
                const size_t src_off = (static_cast<size_t>(t) * n_head_kv + h) * 128;
                const size_t dst_idx = static_cast<size_t>(h) * max_ctx + pidx;
                quantize_128_q4_0_pos(k + src_off, kc[dst_idx]);
                quantize_128_q4_0_pos(v + src_off, vc[dst_idx]);
            }
        }
    } else if (kv_type == KvCacheType::Q8_0) {
        auto * kc = static_cast<CpuGqaKv128Q8Pos *>(k_cache);
        auto * vc = static_cast<CpuGqaKv128Q8Pos *>(v_cache);
        alignas(32) CpuActQ8_0 tmp_q8[4];
        for (int t = 0; t < T; ++t) {
            const int pidx = pos[t];
            for (int h = 0; h < n_head_kv; ++h) {
                const size_t src_off = (static_cast<size_t>(t) * n_head_kv + h) * 128;
                const size_t dst_idx = static_cast<size_t>(h) * max_ctx + pidx;

                quantize_row_q8_0(k + src_off, tmp_q8, 128);
                auto & kd = kc[dst_idx];
                for (int b = 0; b < 4; ++b) {
                    kd.d[b] = tmp_q8[b].d;
                    std::memcpy(kd.qs + b * 32, tmp_q8[b].qs, 32);
                }

                quantize_row_q8_0(v + src_off, tmp_q8, 128);
                auto & vd = vc[dst_idx];
                for (int b = 0; b < 4; ++b) {
                    vd.d[b] = tmp_q8[b].d;
                    std::memcpy(vd.qs + b * 32, tmp_q8[b].qs, 32);
                }
            }
        }
    } else {
        auto * kc = static_cast<uint16_t *>(k_cache);
        auto * vc = static_cast<uint16_t *>(v_cache);
        for (int t = 0; t < T; ++t) {
            const int pidx = pos[t];
            for (int h = 0; h < n_head_kv; ++h) {
                const float * ks = k + (static_cast<size_t>(t) * n_head_kv + h) * 128;
                const float * vs = v + (static_cast<size_t>(t) * n_head_kv + h) * 128;
                uint16_t * kd = kc + (static_cast<size_t>(h) * max_ctx + pidx) * 128;
                uint16_t * vd = vc + (static_cast<size_t>(h) * max_ctx + pidx) * 128;
                for (int i = 0; i < 128; ++i) {
                    kd[i] = f32_to_f16_fast(ks[i]);
                    vd[i] = f32_to_f16_fast(vs[i]);
                }
            }
        }
    }
}

void gqa_attn_d128(
    ThreadPool & pool,
    const float * q, const void * k_cache, const void * v_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_head, int n_head_kv, int max_ctx, float scale) {

    const int gqa = n_head / n_head_kv;

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((kv_type == KvCacheType::Q8_0 || kv_type == KvCacheType::Q4_0) && gqa <= 16) {
        // Split each KV head's Q-group when n_head_kv < n_threads so all big cores stay 100% busy
        int g_splits = 1;
        if (T == 1 && n_head_kv < pool.num_threads()) {
            const int want = (pool.num_threads() + n_head_kv - 1) / n_head_kv;
            for (int s = std::min(want, gqa); s >= 1; --s) {
                if ((gqa % s) == 0) { g_splits = s; break; }
            }
        }
        const int grp_sz   = gqa / g_splits;
        const int n_groups = n_head_kv * g_splits;
        const int n_tasks  = T * n_groups;
        const bool is_q4   = (kv_type == KvCacheType::Q4_0);
        const bool use_dyn = (T >= 4 && pool.num_compute_threads() > 1);
        const bool use_hyb = (T >= 16 && pool.num_compute_threads() > pool.num_threads());
        const int  n_big   = pool.num_threads();

        // Double-ended lock-free causal task queue: big cores claim longest kv_len (high t) from tail,
        // LITTLE cores claim shortest kv_len (low t) from head -> eliminates 1.75x causal static imbalance!
        std::atomic<uint64_t> bounds{static_cast<uint64_t>(static_cast<uint32_t>(n_tasks)) << 32};

        auto run_task = [&](int task, std::vector<float> & scores_buf,
                            float (&q_scaled)[16][128], CpuActQ8_0 (&q_q8)[16][4],
                            float (&qd_pre)[16][4], float (&qs_pre)[16][4],
                            float (&ov)[16][128]) {
            const int t      = task / n_groups;
            const int gidx   = task % n_groups;
            const int h_kv   = gidx / g_splits;
            const int h_base = h_kv * gqa + (gidx % g_splits) * grp_sz;
            const int kv_len = pos[t] + 1;

            if (scores_buf.size() < static_cast<size_t>(grp_sz) * kv_len) {
                scores_buf.resize(static_cast<size_t>(grp_sz) * kv_len);
            }

            const float32x4_t vsc = vdupq_n_f32(scale);
            for (int hi = 0; hi < grp_sz; ++hi) {
                const float * qp = q + (static_cast<size_t>(t) * n_head + h_base + hi) * 128;
                for (int d = 0; d < 128; d += 8) {
                    vst1q_f32(q_scaled[hi] + d + 0, vmulq_f32(vld1q_f32(qp + d + 0), vsc));
                    vst1q_f32(q_scaled[hi] + d + 4, vmulq_f32(vld1q_f32(qp + d + 4), vsc));
                }
                quantize_row_q8_0(q_scaled[hi], q_q8[hi], 128);
                for (int b = 0; b < 4; ++b) {
                    qd_pre[hi][b] = q_q8[hi][b].d;
                    qs_pre[hi][b] = 8.0f * q_q8[hi][b].s;
                }
            }

            if (is_q4) {
                const CpuGqaKv128Q4Pos * k_head = static_cast<const CpuGqaKv128Q4Pos *>(k_cache) + static_cast<size_t>(h_kv) * max_ctx;
                const CpuGqaKv128Q4Pos * v_head = static_cast<const CpuGqaKv128Q4Pos *>(v_cache) + static_cast<size_t>(h_kv) * max_ctx;
                const uint8x16_t m4 = vdupq_n_u8(0x0Fu);

                // Pass 1: Read each 80B Q4_0 K position ONCE for all grp_sz query heads (zero vsubq_s8 via q.s!)
                for (int p = 0; p < kv_len; ++p) {
                    const CpuGqaKv128Q4Pos & kp = k_head[p];
                    if (p + 4 < kv_len) __builtin_prefetch(&k_head[p + 4], 0, 0);

                    const uint8x16_t b0 = vld1q_u8(kp.qs +  0), b1 = vld1q_u8(kp.qs + 16);
                    const uint8x16_t b2 = vld1q_u8(kp.qs + 32), b3 = vld1q_u8(kp.qs + 48);
                    const int8x16_t k0a = vreinterpretq_s8_u8(vandq_u8(b0, m4)), k0b = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
                    const int8x16_t k1a = vreinterpretq_s8_u8(vandq_u8(b1, m4)), k1b = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
                    const int8x16_t k2a = vreinterpretq_s8_u8(vandq_u8(b2, m4)), k2b = vreinterpretq_s8_u8(vshrq_n_u8(b2, 4));
                    const int8x16_t k3a = vreinterpretq_s8_u8(vandq_u8(b3, m4)), k3b = vreinterpretq_s8_u8(vshrq_n_u8(b3, 4));
                    const float32x4_t vkd = vld1q_f32(kp.d);

                    for (int hi = 0; hi < grp_sz; ++hi) {
                        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0a, vld1q_s8(q_q8[hi][0].qs)), k0b, vld1q_s8(q_q8[hi][0].qs + 16));
                        int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k1a, vld1q_s8(q_q8[hi][1].qs)), k1b, vld1q_s8(q_q8[hi][1].qs + 16));
                        int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k2a, vld1q_s8(q_q8[hi][2].qs)), k2b, vld1q_s8(q_q8[hi][2].qs + 16));
                        int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k3a, vld1q_s8(q_q8[hi][3].qs)), k3b, vld1q_s8(q_q8[hi][3].qs + 16));
                        const float32x4_t fsum = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)));
                        const float32x4_t term = vfmsq_f32(vmulq_f32(fsum, vld1q_f32(qd_pre[hi])), vdupq_n_f32(1.0f), vld1q_f32(qs_pre[hi]));
                        scores_buf[static_cast<size_t>(hi) * kv_len + p] = vaddvq_f32(vmulq_f32(term, vkd));
                    }
                }

                // Pass 2: Softmax all grp_sz heads, then dequantize each active V position ONCE across all grp_sz heads!
                float inv_l[16], thresh[16];
                for (int hi = 0; hi < grp_sz; ++hi) {
                    float * sp = scores_buf.data() + static_cast<size_t>(hi) * kv_len;
                    float m_max = -FLT_MAX;
                    for (int p = 0; p < kv_len; ++p) m_max = std::max(m_max, sp[p]);
                    const float32x4_t vm = vdupq_n_f32(m_max);
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    int p = 0;
                    for (; p + 4 <= kv_len; p += 4) {
                        const float32x4_t e = vexpq_f32_fast(vsubq_f32(vld1q_f32(sp + p), vm));
                        vst1q_f32(sp + p, e);
                        vsum = vaddq_f32(vsum, e);
                    }
                    float l_sum = vaddvq_f32(vsum);
                    for (; p < kv_len; ++p) { sp[p] = std::exp(sp[p] - m_max); l_sum += sp[p]; }
                    inv_l[hi]  = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                    thresh[hi] = 1e-6f * l_sum;
                    std::memset(ov[hi], 0, 128 * sizeof(float));
                }

                const int8x16_t b8 = vdupq_n_s8(8);
                alignas(64) float vf[128];
                for (int q_idx = 0; q_idx < kv_len; ++q_idx) {
                    bool any = false;
                    for (int hi = 0; hi < grp_sz; ++hi) {
                        if (scores_buf[static_cast<size_t>(hi) * kv_len + q_idx] > thresh[hi]) { any = true; break; }
                    }
                    if (!any) continue;
                    const CpuGqaKv128Q4Pos & vp = v_head[q_idx];
                    for (int b = 0; b < 4; ++b) {
                        const float32x4_t vd = vdupq_n_f32(vp.d[b]);
                        const uint8x16_t packed = vld1q_u8(vp.qs + b * 16);
                        const int8x16_t q_parts[2] = {
                            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, m4)), b8),
                            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), b8)
                        };
                        for (int g = 0; g < 2; ++g) {
                            const int16x8_t qlo = vmovl_s8(vget_low_s8(q_parts[g]));
                            const int16x8_t qhi = vmovl_s8(vget_high_s8(q_parts[g]));
                            float * gp = vf + b * 32 + g * 16;
                            vst1q_f32(gp +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))),  vd));
                            vst1q_f32(gp +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), vd));
                            vst1q_f32(gp +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))),  vd));
                            vst1q_f32(gp + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), vd));
                        }
                    }
                    for (int hi = 0; hi < grp_sz; ++hi) {
                        const float w = scores_buf[static_cast<size_t>(hi) * kv_len + q_idx];
                        if (w <= thresh[hi]) continue;
                        const float32x4_t vpw = vdupq_n_f32(w * inv_l[hi]);
                        float * op = ov[hi];
                        for (int d = 0; d < 128; d += 16) {
                            vst1q_f32(op + d +  0, vfmaq_f32(vld1q_f32(op + d +  0), vld1q_f32(vf + d +  0), vpw));
                            vst1q_f32(op + d +  4, vfmaq_f32(vld1q_f32(op + d +  4), vld1q_f32(vf + d +  4), vpw));
                            vst1q_f32(op + d +  8, vfmaq_f32(vld1q_f32(op + d +  8), vld1q_f32(vf + d +  8), vpw));
                            vst1q_f32(op + d + 12, vfmaq_f32(vld1q_f32(op + d + 12), vld1q_f32(vf + d + 12), vpw));
                        }
                    }
                }
                for (int hi = 0; hi < grp_sz; ++hi) {
                    std::memcpy(out + (static_cast<size_t>(t) * n_head + h_base + hi) * 128, ov[hi], 128 * sizeof(float));
                }
            } else {
                const CpuGqaKv128Q8Pos * k_head = static_cast<const CpuGqaKv128Q8Pos *>(k_cache) + static_cast<size_t>(h_kv) * max_ctx;
                const CpuGqaKv128Q8Pos * v_head = static_cast<const CpuGqaKv128Q8Pos *>(v_cache) + static_cast<size_t>(h_kv) * max_ctx;

                // Pass 1: Read each K position ONCE for all grp_sz query heads
                for (int p = 0; p < kv_len; ++p) {
                    const CpuGqaKv128Q8Pos & kp = k_head[p];
                    if (p + 4 < kv_len) __builtin_prefetch(&k_head[p + 4], 0, 0);

                    const int8x16_t k0a = vld1q_s8(kp.qs +  0), k0b = vld1q_s8(kp.qs +  16);
                    const int8x16_t k1a = vld1q_s8(kp.qs + 32), k1b = vld1q_s8(kp.qs +  48);
                    const int8x16_t k2a = vld1q_s8(kp.qs + 64), k2b = vld1q_s8(kp.qs +  80);
                    const int8x16_t k3a = vld1q_s8(kp.qs + 96), k3b = vld1q_s8(kp.qs + 112);
                    const float32x4_t vkd = vld1q_f32(kp.d);

                    for (int hi = 0; hi < grp_sz; ++hi) {
                        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0a, vld1q_s8(q_q8[hi][0].qs)), k0b, vld1q_s8(q_q8[hi][0].qs + 16));
                        int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k1a, vld1q_s8(q_q8[hi][1].qs)), k1b, vld1q_s8(q_q8[hi][1].qs + 16));
                        int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k2a, vld1q_s8(q_q8[hi][2].qs)), k2b, vld1q_s8(q_q8[hi][2].qs + 16));
                        int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k3a, vld1q_s8(q_q8[hi][3].qs)), k3b, vld1q_s8(q_q8[hi][3].qs + 16));
                        const float32x4_t fsum = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)));
                        scores_buf[static_cast<size_t>(hi) * kv_len + p] = vaddvq_f32(vmulq_f32(fsum, vmulq_f32(vkd, vld1q_f32(qd_pre[hi]))));
                    }
                }

                // Pass 2: Softmax all grp_sz heads, then dequantize each active V position ONCE across all grp_sz heads!
                float inv_l[16], thresh[16];
                for (int hi = 0; hi < grp_sz; ++hi) {
                    float * sp = scores_buf.data() + static_cast<size_t>(hi) * kv_len;
                    float m_max = -FLT_MAX;
                    for (int p = 0; p < kv_len; ++p) m_max = std::max(m_max, sp[p]);
                    const float32x4_t vm = vdupq_n_f32(m_max);
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    int p = 0;
                    for (; p + 4 <= kv_len; p += 4) {
                        const float32x4_t e = vexpq_f32_fast(vsubq_f32(vld1q_f32(sp + p), vm));
                        vst1q_f32(sp + p, e);
                        vsum = vaddq_f32(vsum, e);
                    }
                    float l_sum = vaddvq_f32(vsum);
                    for (; p < kv_len; ++p) { sp[p] = std::exp(sp[p] - m_max); l_sum += sp[p]; }
                    inv_l[hi]  = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                    thresh[hi] = 1e-6f * l_sum;
                    std::memset(ov[hi], 0, 128 * sizeof(float));
                }

                alignas(64) float vf[128];
                for (int q_idx = 0; q_idx < kv_len; ++q_idx) {
                    bool any = false;
                    for (int hi = 0; hi < grp_sz; ++hi) {
                        if (scores_buf[static_cast<size_t>(hi) * kv_len + q_idx] > thresh[hi]) { any = true; break; }
                    }
                    if (!any) continue;
                    const CpuGqaKv128Q8Pos & vp = v_head[q_idx];
                    for (int b = 0; b < 4; ++b) {
                        const float32x4_t vd = vdupq_n_f32(vp.d[b]);
                        const int8_t * vqs = vp.qs + b * 32;
                        for (int g = 0; g < 2; ++g) {
                            const int8x16_t qv  = vld1q_s8(vqs + g * 16);
                            const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                            const int16x8_t qhi = vmovl_s8(vget_high_s8(qv));
                            float * gp = vf + b * 32 + g * 16;
                            vst1q_f32(gp +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))),  vd));
                            vst1q_f32(gp +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), vd));
                            vst1q_f32(gp +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))),  vd));
                            vst1q_f32(gp + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), vd));
                        }
                    }
                    for (int hi = 0; hi < grp_sz; ++hi) {
                        const float w = scores_buf[static_cast<size_t>(hi) * kv_len + q_idx];
                        if (w <= thresh[hi]) continue;
                        const float32x4_t vpw = vdupq_n_f32(w * inv_l[hi]);
                        float * op = ov[hi];
                        for (int d = 0; d < 128; d += 16) {
                            vst1q_f32(op + d +  0, vfmaq_f32(vld1q_f32(op + d +  0), vld1q_f32(vf + d +  0), vpw));
                            vst1q_f32(op + d +  4, vfmaq_f32(vld1q_f32(op + d +  4), vld1q_f32(vf + d +  4), vpw));
                            vst1q_f32(op + d +  8, vfmaq_f32(vld1q_f32(op + d +  8), vld1q_f32(vf + d +  8), vpw));
                            vst1q_f32(op + d + 12, vfmaq_f32(vld1q_f32(op + d + 12), vld1q_f32(vf + d + 12), vpw));
                        }
                    }
                }
                for (int hi = 0; hi < grp_sz; ++hi) {
                    std::memcpy(out + (static_cast<size_t>(t) * n_head + h_base + hi) * 128, ov[hi], 128 * sizeof(float));
                }
            }
        };

        if (use_dyn) {
            auto dyn_worker = [&](int tid, int) {
                static thread_local std::vector<float> scores_buf;
                alignas(64) float      q_scaled[16][128];
                alignas(64) CpuActQ8_0 q_q8[16][4];
                alignas(16) float      qd_pre[16][4];
                alignas(16) float      qs_pre[16][4];
                alignas(64) float      ov[16][128];
                const bool is_little = use_hyb && (tid >= n_big);
                while (true) {
                    uint64_t cur = bounds.load(std::memory_order_relaxed);
                    uint32_t task = 0;
                    bool claimed = false;
                    while (true) {
                        const uint32_t hd = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
                        const uint32_t tl = static_cast<uint32_t>(cur >> 32);
                        if (hd >= tl) break;
                        // Big cores claim highest t (longest kv_len) from tail; LITTLE cores claim lowest t from head!
                        const uint64_t next = is_little
                            ? (static_cast<uint64_t>(hd + 1) | (static_cast<uint64_t>(tl) << 32))
                            : (static_cast<uint64_t>(hd)     | (static_cast<uint64_t>(tl - 1) << 32));
                        if (bounds.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                            task = is_little ? hd : (tl - 1);
                            claimed = true;
                            break;
                        }
                    }
                    if (!claimed) break;
                    run_task(static_cast<int>(task), scores_buf, q_scaled, q_q8, qd_pre, qs_pre, ov);
                }
            };
            if (use_hyb) pool.parallel_compute(dyn_worker);
            else         pool.parallel(dyn_worker);
        } else {
            pool.parallel_for(n_tasks, [&](int task0, int task1, int) {
                static thread_local std::vector<float> scores_buf;
                alignas(64) float      q_scaled[16][128];
                alignas(64) CpuActQ8_0 q_q8[16][4];
                alignas(16) float      qd_pre[16][4];
                alignas(16) float      qs_pre[16][4];
                alignas(64) float      ov[16][128];
                for (int task = task0; task < task1; ++task) {
                    run_task(task, scores_buf, q_scaled, q_q8, qd_pre, qs_pre, ov);
                }
            });
        }
        return;
    }
#endif

    // Portable fallback (F16, Q8_0, or Q4_0 on non-ARM64 targets)
    pool.parallel_for(T * n_head, [&](int task0, int task1, int) {
        alignas(32) float q_scaled[128];
        alignas(32) CpuActQ8_0 q_q8[4];
        alignas(32) float ov[128];
        for (int task = task0; task < task1; ++task) {
            const int t      = task / n_head;
            const int h      = task % n_head;
            const int h_kv   = h / gqa;
            const int kv_len = pos[t] + 1;
            const float * qp = q + (static_cast<size_t>(t) * n_head + h) * 128;
            for (int i = 0; i < 128; ++i) q_scaled[i] = qp[i] * scale;

            float m_i = -FLT_MAX;
            float l_i = 0.0f;
            std::memset(ov, 0, sizeof(ov));

            if (kv_type == KvCacheType::Q4_0) {
                quantize_row_q8_0(q_scaled, q_q8, 128);
                const auto * kh = static_cast<const CpuGqaKv128Q4Pos *>(k_cache) + static_cast<size_t>(h_kv) * max_ctx;
                const auto * vh = static_cast<const CpuGqaKv128Q4Pos *>(v_cache) + static_cast<size_t>(h_kv) * max_ctx;
                for (int p = 0; p < kv_len; ++p) {
                    float score = 0.0f;
                    for (int b = 0; b < 4; ++b) {
                        int32_t dp = 0;
                        const uint8_t * kq = kh[p].qs + b * 16;
                        for (int j = 0; j < 16; ++j) {
                            dp += (static_cast<int32_t>(kq[j] & 0xFu) - 8) * q_q8[b].qs[j];
                            dp += (static_cast<int32_t>(kq[j] >> 4) - 8)   * q_q8[b].qs[j + 16];
                        }
                        score += (kh[p].d[b] * q_q8[b].d) * static_cast<float>(dp);
                    }
                    const float m_new = std::max(m_i, score);
                    const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                    const float beta  = std::exp(score - m_new);
                    l_i = l_i * alpha + beta;
                    m_i = m_new;
                    if (alpha != 1.0f) for (int i = 0; i < 128; ++i) ov[i] *= alpha;
                    if (beta > 1e-8f) {
                        for (int b = 0; b < 4; ++b) {
                            const float c = beta * vh[p].d[b];
                            const uint8_t * vq = vh[p].qs + b * 16;
                            for (int j = 0; j < 16; ++j) {
                                ov[b * 32 + j]      += c * static_cast<float>(static_cast<int32_t>(vq[j] & 0xFu) - 8);
                                ov[b * 32 + j + 16] += c * static_cast<float>(static_cast<int32_t>(vq[j] >> 4) - 8);
                            }
                        }
                    }
                }
            } else if (kv_type == KvCacheType::Q8_0) {
                quantize_row_q8_0(q_scaled, q_q8, 128);
                const auto * kh = static_cast<const CpuGqaKv128Q8Pos *>(k_cache) + static_cast<size_t>(h_kv) * max_ctx;
                const auto * vh = static_cast<const CpuGqaKv128Q8Pos *>(v_cache) + static_cast<size_t>(h_kv) * max_ctx;
                for (int p = 0; p < kv_len; ++p) {
                    float score = 0.0f;
                    for (int b = 0; b < 4; ++b) {
                        int32_t dp = 0;
                        for (int i = 0; i < 32; ++i) dp += static_cast<int32_t>(kh[p].qs[b * 32 + i]) * q_q8[b].qs[i];
                        score += (kh[p].d[b] * q_q8[b].d) * static_cast<float>(dp);
                    }
                    const float m_new = std::max(m_i, score);
                    const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                    const float beta  = std::exp(score - m_new);
                    l_i = l_i * alpha + beta;
                    m_i = m_new;
                    if (alpha != 1.0f) for (int i = 0; i < 128; ++i) ov[i] *= alpha;
                    if (beta > 1e-8f) {
                        for (int b = 0; b < 4; ++b) {
                            const float c = beta * vh[p].d[b];
                            for (int i = 0; i < 32; ++i) ov[b * 32 + i] += c * static_cast<float>(vh[p].qs[b * 32 + i]);
                        }
                    }
                }
            } else {
                const auto * kh = static_cast<const uint16_t *>(k_cache) + static_cast<size_t>(h_kv) * max_ctx * 128;
                const auto * vh = static_cast<const uint16_t *>(v_cache) + static_cast<size_t>(h_kv) * max_ctx * 128;
                for (int p = 0; p < kv_len; ++p) {
                    const float score = dot_f16_row(kh + static_cast<size_t>(p) * 128, q_scaled, 128);
                    const float m_new = std::max(m_i, score);
                    const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                    const float beta  = std::exp(score - m_new);
                    l_i = l_i * alpha + beta;
                    m_i = m_new;
                    const uint16_t * vp = vh + static_cast<size_t>(p) * 128;
                    for (int i = 0; i < 128; ++i) ov[i] = ov[i] * alpha + beta * f16_to_f32_fast(vp[i]);
                }
            }
            const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            float * dst = out + (static_cast<size_t>(t) * n_head + h) * 128;
            for (int i = 0; i < 128; ++i) dst[i] = ov[i] * inv_l;
        }
    });
}

void ssm_conv1d_silu_qkv_l2(
    ThreadPool & pool,
    const float * q, const float * k, const float * v,
    const float * wq, const float * wk, const float * wv,
    float * sq, float * sk, float * sv,
    float * qc, float * kc, float * vc,
    int T, int n_heads, float eps,
    float * sq_snap, float * sk_snap, float * sv_snap) {

    const int C = n_heads * 128;
    const float eps2 = eps * eps;

    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        alignas(32) float w0[128], w1[128], w2[128], w3[128];
        for (int h = h0; h < h1; ++h) {
            const int c_base = h * 128;
            for (int which = 0; which < 3; ++which) {
                const float * x     = (which == 0) ? q  : ((which == 1) ? k  : v);
                const float * w     = (which == 0) ? wq : ((which == 1) ? wk : wv);
                float *       state = (which == 0) ? sq : ((which == 1) ? sk : sv);
                float *       out   = (which == 0) ? qc : ((which == 1) ? kc : vc);
                float *       snap  = (which == 0) ? sq_snap : ((which == 1) ? sk_snap : sv_snap);

                // Deinterleave conv kernel [128, 4] -> 4 contiguous 128-float vectors for AVX2!
                for (int d = 0; d < 128; ++d) {
                    const int c = c_base + d;
                    w0[d] = w[c * 4 + 0];
                    w1[d] = w[c * 4 + 1];
                    w2[d] = w[c * 4 + 2];
                    w3[d] = w[c * 4 + 3];
                }
                float * st0 = state + 0 * C + c_base;
                float * st1 = state + 1 * C + c_base;
                float * st2 = state + 2 * C + c_base;

                for (int t = 0; t < T; ++t) {
                    const float * xt = x + static_cast<size_t>(t) * C + c_base;
                    float * ot = out + static_cast<size_t>(t) * C + c_base;
#if defined(__AVX2__) && defined(__FMA__)
                    __m256 vss0 = _mm256_setzero_ps();
                    __m256 vss1 = _mm256_setzero_ps();
                    for (int d = 0; d < 128; d += 16) {
                        __m256 s0a = _mm256_loadu_ps(st0 + d);
                        __m256 s1a = _mm256_loadu_ps(st1 + d);
                        __m256 s2a = _mm256_loadu_ps(st2 + d);
                        __m256 xva = _mm256_loadu_ps(xt + d);
                        __m256 acca = _mm256_mul_ps(s0a, _mm256_load_ps(w0 + d));
                        acca = _mm256_fmadd_ps(s1a, _mm256_load_ps(w1 + d), acca);
                        acca = _mm256_fmadd_ps(s2a, _mm256_load_ps(w2 + d), acca);
                        acca = _mm256_fmadd_ps(xva, _mm256_load_ps(w3 + d), acca);
                        __m256 vala = silu_ps_avx2(acca);
                        _mm256_storeu_ps(ot + d, vala);
                        _mm256_storeu_ps(st0 + d, s1a);
                        _mm256_storeu_ps(st1 + d, s2a);
                        _mm256_storeu_ps(st2 + d, xva);
                        vss0 = _mm256_fmadd_ps(vala, vala, vss0);

                        __m256 s0b = _mm256_loadu_ps(st0 + d + 8);
                        __m256 s1b = _mm256_loadu_ps(st1 + d + 8);
                        __m256 s2b = _mm256_loadu_ps(st2 + d + 8);
                        __m256 xvb = _mm256_loadu_ps(xt + d + 8);
                        __m256 accb = _mm256_mul_ps(s0b, _mm256_load_ps(w0 + d + 8));
                        accb = _mm256_fmadd_ps(s1b, _mm256_load_ps(w1 + d + 8), accb);
                        accb = _mm256_fmadd_ps(s2b, _mm256_load_ps(w2 + d + 8), accb);
                        accb = _mm256_fmadd_ps(xvb, _mm256_load_ps(w3 + d + 8), accb);
                        __m256 valb = silu_ps_avx2(accb);
                        _mm256_storeu_ps(ot + d + 8, valb);
                        _mm256_storeu_ps(st0 + d + 8, s1b);
                        _mm256_storeu_ps(st1 + d + 8, s2b);
                        _mm256_storeu_ps(st2 + d + 8, xvb);
                        vss1 = _mm256_fmadd_ps(valb, valb, vss1);
                    }
                    if (t == 0 && snap != nullptr) {
                        std::memcpy(snap + 0 * C + c_base, st0, 128 * sizeof(float));
                        std::memcpy(snap + 1 * C + c_base, st1, 128 * sizeof(float));
                        std::memcpy(snap + 2 * C + c_base, st2, 128 * sizeof(float));
                    }
                    if (which < 2) {
                        const float ss = hsum256_ps(_mm256_add_ps(vss0, vss1));
                        const __m256 vinv = _mm256_set1_ps(1.0f / std::sqrt(std::max(ss, eps2)));
                        for (int d = 0; d < 128; d += 8) {
                            _mm256_storeu_ps(ot + d, _mm256_mul_ps(_mm256_loadu_ps(ot + d), vinv));
                        }
                    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
                    const float32x4_t v1 = vdupq_n_f32(1.0f);
                    float32x4_t vss0 = vdupq_n_f32(0.0f), vss1 = vdupq_n_f32(0.0f);
                    for (int d = 0; d < 128; d += 8) {
                        float32x4_t a0 = vmulq_f32(vld1q_f32(st0 + d + 0), vld1q_f32(w0 + d + 0));
                        float32x4_t a1 = vmulq_f32(vld1q_f32(st0 + d + 4), vld1q_f32(w0 + d + 4));
                        float32x4_t vs1_0 = vld1q_f32(st1 + d + 0), vs1_1 = vld1q_f32(st1 + d + 4);
                        float32x4_t vs2_0 = vld1q_f32(st2 + d + 0), vs2_1 = vld1q_f32(st2 + d + 4);
                        float32x4_t vx_0  = vld1q_f32(xt + d + 0),  vx_1  = vld1q_f32(xt + d + 4);
                        a0 = vfmaq_f32(a0, vs1_0, vld1q_f32(w1 + d + 0));
                        a1 = vfmaq_f32(a1, vs1_1, vld1q_f32(w1 + d + 4));
                        a0 = vfmaq_f32(a0, vs2_0, vld1q_f32(w2 + d + 0));
                        a1 = vfmaq_f32(a1, vs2_1, vld1q_f32(w2 + d + 4));
                        a0 = vfmaq_f32(a0, vx_0,  vld1q_f32(w3 + d + 0));
                        a1 = vfmaq_f32(a1, vx_1,  vld1q_f32(w3 + d + 4));
                        vst1q_f32(st0 + d + 0, vs1_0); vst1q_f32(st0 + d + 4, vs1_1);
                        vst1q_f32(st1 + d + 0, vs2_0); vst1q_f32(st1 + d + 4, vs2_1);
                        vst1q_f32(st2 + d + 0, vx_0);  vst1q_f32(st2 + d + 4, vx_1);
                        float32x4_t y0 = vdivq_f32(a0, vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(a0))));
                        float32x4_t y1 = vdivq_f32(a1, vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(a1))));
                        vst1q_f32(ot + d + 0, y0);
                        vst1q_f32(ot + d + 4, y1);
                        vss0 = vfmaq_f32(vss0, y0, y0);
                        vss1 = vfmaq_f32(vss1, y1, y1);
                    }
                    if (t == 0 && snap != nullptr) {
                        std::memcpy(snap + 0 * C + c_base, st0, 128 * sizeof(float));
                        std::memcpy(snap + 1 * C + c_base, st1, 128 * sizeof(float));
                        std::memcpy(snap + 2 * C + c_base, st2, 128 * sizeof(float));
                    }
                    if (which < 2) {
                        const float ss = vaddvq_f32(vaddq_f32(vss0, vss1));
                        const float32x4_t vinv = vdupq_n_f32(1.0f / std::sqrt(std::max(ss, eps2)));
                        for (int d = 0; d < 128; d += 8) {
                            vst1q_f32(ot + d + 0, vmulq_f32(vld1q_f32(ot + d + 0), vinv));
                            vst1q_f32(ot + d + 4, vmulq_f32(vld1q_f32(ot + d + 4), vinv));
                        }
                    }
#else
                    float ss = 0.0f;
                    for (int d = 0; d < 128; ++d) {
                        const float xv = xt[d];
                        const float acc = st0[d] * w0[d] + st1[d] * w1[d] + st2[d] * w2[d] + xv * w3[d];
                        const float val = acc / (1.0f + std::exp(-acc));
                        ot[d] = val;
                        ss += val * val;
                        st0[d] = st1[d];
                        st1[d] = st2[d];
                        st2[d] = xv;
                    }
                    if (t == 0 && snap != nullptr) {
                        std::memcpy(snap + 0 * C + c_base, st0, 128 * sizeof(float));
                        std::memcpy(snap + 1 * C + c_base, st1, 128 * sizeof(float));
                        std::memcpy(snap + 2 * C + c_base, st2, 128 * sizeof(float));
                    }
                    if (which < 2) {
                        const float inv = 1.0f / std::sqrt(std::max(ss, eps2));
                        for (int d = 0; d < 128; ++d) ot[d] *= inv;
                    }
#endif
                }
            }
        }
    });
}

void kda_gate_and_beta(
    ThreadPool & pool,
    float * f_inout, const float * dt_bias, const float * ssm_a,
    float * beta_inout, float lower_bound, int T, int n_heads, int head_dim,
    bool negate_a) {

    const int D = n_heads * head_dim;
    auto work = [&](int t0, int t1, int) {
#if defined(__AVX2__) && defined(__FMA__)
        const __m256 vlb  = _mm256_set1_ps(lower_bound);
        const __m256 vone = _mm256_set1_ps(1.0f);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        const float32x4_t vlb  = vdupq_n_f32(lower_bound);
        const float32x4_t vone = vdupq_n_f32(1.0f);
#endif
        for (int t = t0; t < t1; ++t) {
            float * ft = f_inout + static_cast<size_t>(t) * D;
            float * bt = beta_inout + static_cast<size_t>(t) * n_heads;
            for (int h = 0; h < n_heads; ++h) {
                bt[h] = 1.0f / (1.0f + std::exp(-bt[h]));
                const float a = negate_a ? -ssm_a[h] : ssm_a[h];
                const int base = h * head_dim;
                int d = 0;
#if defined(__AVX2__) && defined(__FMA__)
                const __m256 va = _mm256_set1_ps(-a); // compute exp(-val) = exp((-a)*(f + dt_bias))
                for (; d + 7 < head_dim; d += 8) {
                    const __m256 nv = _mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(ft + base + d), _mm256_loadu_ps(dt_bias + base + d)), va);
                    _mm256_storeu_ps(ft + base + d, _mm256_div_ps(vlb, _mm256_add_ps(vone, exp_ps_avx2(nv))));
                }
#elif defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t va = vdupq_n_f32(-a);
                for (; d + 4 <= head_dim; d += 4) {
                    const float32x4_t nv = vmulq_f32(vaddq_f32(vld1q_f32(ft + base + d), vld1q_f32(dt_bias + base + d)), va);
                    vst1q_f32(ft + base + d, vdivq_f32(vlb, vaddq_f32(vone, vexpq_f32_fast(nv))));
                }
#endif
                for (; d < head_dim; ++d) {
                    const float val = (ft[base + d] + dt_bias[base + d]) * a;
                    ft[base + d] = lower_bound / (1.0f + std::exp(-val));
                }
            }
        }
    };

    if (T <= 2) {
        work(0, T, 0);
    } else {
        pool.parallel_for(T, work);
    }
}

void gated_delta_net_d128(
    ThreadPool & pool,
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    float * state, float * out,
    int T, int n_heads, float scale,
    float * state_snap) {

#if defined(__aarch64__) && defined(__ARM_NEON)
    // Single-pass fused L1d-strip-tiled KDA recurrence:
    // State stored as 8 contiguous 16-col strips: S[h, strip(0..7), row(0..127), col(0..15)].
    // Each strip is 8 KB (100% L1d-resident on both Cortex-A76 64 KB and Cortex-A55 32 KB),
    // keeping d0..d3, o0..o3, and kv0..kv3 simultaneously in 20 NEON registers (zero spills!)
    // so Pass 2 of token t fuses with Pass 1 (decay + K dot) of token t+1 -> halves L1d passes!
    // For T >= 4, precompute exp(g) ONCE in parallel (avoids 8x redundant exp across strips)
    // and use vfmaq_laneq_f32 / vmulq_laneq_f32 (zero scalar-to-vector dup instructions!).
    const int total_tasks = n_heads * 8;
    const float32x4_t vsc = vdupq_n_f32(scale);

    thread_local std::vector<float> decay_all;
    const float * dec_all_p = nullptr;
    if (T >= 4) {
        const size_t n_dec = static_cast<size_t>(T) * n_heads * 128;
        if (decay_all.size() < n_dec) decay_all.resize(n_dec);
        float * dec_dst = decay_all.data();
        const int n_th = T * n_heads;
        pool.parallel_for(n_th, [&](int j0, int j1, int) {
            for (int j = j0; j < j1; ++j) {
                const float * gs = g + static_cast<size_t>(j) * 128;
                float *       dd = dec_dst + static_cast<size_t>(j) * 128;
                for (int r = 0; r < 128; r += 8) {
                    vst1q_f32(dd + r + 0, vexpq_f32_fast(vld1q_f32(gs + r + 0)));
                    vst1q_f32(dd + r + 4, vexpq_f32_fast(vld1q_f32(gs + r + 4)));
                }
            }
        });
        dec_all_p = dec_dst;
    }

    auto run_strip = [&](int task, float (&decay_loc)[128]) {
        const int h     = task >> 3;
        const int strip = task & 7;
        const int c0    = strip * 16;
        const size_t strip_off = (static_cast<size_t>(h) * 8 + strip) * (128 * 16);
        float * S    = state + strip_off;
        float * snap = state_snap ? (state_snap + strip_off) : nullptr;

        auto get_decay = [&](size_t h_off) -> const float * {
            if (dec_all_p != nullptr) return dec_all_p + h_off;
            const float * gt = g + h_off;
            for (int r = 0; r < 128; r += 8) {
                vst1q_f32(decay_loc + r + 0, vexpq_f32_fast(vld1q_f32(gt + r + 0)));
                vst1q_f32(decay_loc + r + 4, vexpq_f32_fast(vld1q_f32(gt + r + 4)));
            }
            return decay_loc;
        };

        // Prologue: Pass 1 for t = 0 (decay S by exp(g[0]) and compute kv0..kv3)
        float32x4_t kv0 = vdupq_n_f32(0.0f), kv1 = vdupq_n_f32(0.0f);
        float32x4_t kv2 = vdupq_n_f32(0.0f), kv3 = vdupq_n_f32(0.0f);
        {
            const size_t h_off0 = static_cast<size_t>(h) * 128;
            const float * kt0   = k + h_off0;
            const float * dec0  = get_decay(h_off0);
            for (int r = 0; r < 128; r += 4) {
                const float32x4_t vd4 = vld1q_f32(dec0 + r);
                const float32x4_t vk4 = vld1q_f32(kt0 + r);
#define KDA_PROL_ROW(L) do { \
                    float * Sr = S + (r + (L)) * 16; \
                    const float32x4_t s0 = vmulq_laneq_f32(vld1q_f32(Sr +  0), vd4, L); vst1q_f32(Sr +  0, s0); kv0 = vfmaq_laneq_f32(kv0, s0, vk4, L); \
                    const float32x4_t s1 = vmulq_laneq_f32(vld1q_f32(Sr +  4), vd4, L); vst1q_f32(Sr +  4, s1); kv1 = vfmaq_laneq_f32(kv1, s1, vk4, L); \
                    const float32x4_t s2 = vmulq_laneq_f32(vld1q_f32(Sr +  8), vd4, L); vst1q_f32(Sr +  8, s2); kv2 = vfmaq_laneq_f32(kv2, s2, vk4, L); \
                    const float32x4_t s3 = vmulq_laneq_f32(vld1q_f32(Sr + 12), vd4, L); vst1q_f32(Sr + 12, s3); kv3 = vfmaq_laneq_f32(kv3, s3, vk4, L); \
                } while (0)
                KDA_PROL_ROW(0); KDA_PROL_ROW(1); KDA_PROL_ROW(2); KDA_PROL_ROW(3);
#undef KDA_PROL_ROW
            }
        }

        for (int t = 0; t < T; ++t) {
            const size_t h_off = (static_cast<size_t>(t) * n_heads + h) * 128;
            const float * qt   = q + h_off;
            const float * kt   = k + h_off;
            const float * vt   = v + h_off + c0;
            const float   bval = beta[static_cast<size_t>(t) * n_heads + h];

            const float32x4_t vb = vdupq_n_f32(bval);
            const float32x4_t d0 = vmulq_f32(vsubq_f32(vld1q_f32(vt +  0), kv0), vb);
            const float32x4_t d1 = vmulq_f32(vsubq_f32(vld1q_f32(vt +  4), kv1), vb);
            const float32x4_t d2 = vmulq_f32(vsubq_f32(vld1q_f32(vt +  8), kv2), vb);
            const float32x4_t d3 = vmulq_f32(vsubq_f32(vld1q_f32(vt + 12), kv3), vb);

            float32x4_t o0 = vdupq_n_f32(0.0f), o1 = vdupq_n_f32(0.0f);
            float32x4_t o2 = vdupq_n_f32(0.0f), o3 = vdupq_n_f32(0.0f);

            if (t + 1 < T) {
                const size_t h_off_n = (static_cast<size_t>(t + 1) * n_heads + h) * 128;
                const float * kt_n   = k + h_off_n;
                const float * dec_n  = get_decay(h_off_n);
                kv0 = vdupq_n_f32(0.0f); kv1 = vdupq_n_f32(0.0f);
                kv2 = vdupq_n_f32(0.0f); kv3 = vdupq_n_f32(0.0f);

                if (t == 0 && snap != nullptr) {
                    for (int r = 0; r < 128; r += 4) {
                        const float32x4_t vk4   = vld1q_f32(kt + r);
                        const float32x4_t vq4   = vld1q_f32(qt + r);
                        const float32x4_t vd4_n = vld1q_f32(dec_n + r);
                        const float32x4_t vk4_n = vld1q_f32(kt_n + r);
#define KDA_FUSED_SNAP_ROW(L) do { \
                            float * Sr = S + (r + (L)) * 16; \
                            float * Sn = snap + (r + (L)) * 16; \
                            const float32x4_t s0 = vfmaq_laneq_f32(vld1q_f32(Sr +  0), d0, vk4, L); vst1q_f32(Sn +  0, s0); o0 = vfmaq_laneq_f32(o0, s0, vq4, L); \
                            const float32x4_t n0 = vmulq_laneq_f32(s0, vd4_n, L); vst1q_f32(Sr +  0, n0); kv0 = vfmaq_laneq_f32(kv0, n0, vk4_n, L); \
                            const float32x4_t s1 = vfmaq_laneq_f32(vld1q_f32(Sr +  4), d1, vk4, L); vst1q_f32(Sn +  4, s1); o1 = vfmaq_laneq_f32(o1, s1, vq4, L); \
                            const float32x4_t n1 = vmulq_laneq_f32(s1, vd4_n, L); vst1q_f32(Sr +  4, n1); kv1 = vfmaq_laneq_f32(kv1, n1, vk4_n, L); \
                            const float32x4_t s2 = vfmaq_laneq_f32(vld1q_f32(Sr +  8), d2, vk4, L); vst1q_f32(Sn +  8, s2); o2 = vfmaq_laneq_f32(o2, s2, vq4, L); \
                            const float32x4_t n2 = vmulq_laneq_f32(s2, vd4_n, L); vst1q_f32(Sr +  8, n2); kv2 = vfmaq_laneq_f32(kv2, n2, vk4_n, L); \
                            const float32x4_t s3 = vfmaq_laneq_f32(vld1q_f32(Sr + 12), d3, vk4, L); vst1q_f32(Sn + 12, s3); o3 = vfmaq_laneq_f32(o3, s3, vq4, L); \
                            const float32x4_t n3 = vmulq_laneq_f32(s3, vd4_n, L); vst1q_f32(Sr + 12, n3); kv3 = vfmaq_laneq_f32(kv3, n3, vk4_n, L); \
                        } while (0)
                        KDA_FUSED_SNAP_ROW(0); KDA_FUSED_SNAP_ROW(1); KDA_FUSED_SNAP_ROW(2); KDA_FUSED_SNAP_ROW(3);
#undef KDA_FUSED_SNAP_ROW
                    }
                } else {
                    for (int r = 0; r < 128; r += 4) {
                        const float32x4_t vk4   = vld1q_f32(kt + r);
                        const float32x4_t vq4   = vld1q_f32(qt + r);
                        const float32x4_t vd4_n = vld1q_f32(dec_n + r);
                        const float32x4_t vk4_n = vld1q_f32(kt_n + r);
#define KDA_FUSED_ROW(L) do { \
                            float * Sr = S + (r + (L)) * 16; \
                            const float32x4_t s0 = vfmaq_laneq_f32(vld1q_f32(Sr +  0), d0, vk4, L); o0 = vfmaq_laneq_f32(o0, s0, vq4, L); \
                            const float32x4_t n0 = vmulq_laneq_f32(s0, vd4_n, L); vst1q_f32(Sr +  0, n0); kv0 = vfmaq_laneq_f32(kv0, n0, vk4_n, L); \
                            const float32x4_t s1 = vfmaq_laneq_f32(vld1q_f32(Sr +  4), d1, vk4, L); o1 = vfmaq_laneq_f32(o1, s1, vq4, L); \
                            const float32x4_t n1 = vmulq_laneq_f32(s1, vd4_n, L); vst1q_f32(Sr +  4, n1); kv1 = vfmaq_laneq_f32(kv1, n1, vk4_n, L); \
                            const float32x4_t s2 = vfmaq_laneq_f32(vld1q_f32(Sr +  8), d2, vk4, L); o2 = vfmaq_laneq_f32(o2, s2, vq4, L); \
                            const float32x4_t n2 = vmulq_laneq_f32(s2, vd4_n, L); vst1q_f32(Sr +  8, n2); kv2 = vfmaq_laneq_f32(kv2, n2, vk4_n, L); \
                            const float32x4_t s3 = vfmaq_laneq_f32(vld1q_f32(Sr + 12), d3, vk4, L); o3 = vfmaq_laneq_f32(o3, s3, vq4, L); \
                            const float32x4_t n3 = vmulq_laneq_f32(s3, vd4_n, L); vst1q_f32(Sr + 12, n3); kv3 = vfmaq_laneq_f32(kv3, n3, vk4_n, L); \
                        } while (0)
                        KDA_FUSED_ROW(0); KDA_FUSED_ROW(1); KDA_FUSED_ROW(2); KDA_FUSED_ROW(3);
#undef KDA_FUSED_ROW
                    }
                }
            } else {
                // Epilogue: final token t == T - 1 leaves state exact without extra decay
                for (int r = 0; r < 128; r += 4) {
                    const float32x4_t vk4 = vld1q_f32(kt + r);
                    const float32x4_t vq4 = vld1q_f32(qt + r);
#define KDA_EPIL_ROW(L) do { \
                        float * Sr = S + (r + (L)) * 16; \
                        const float32x4_t s0 = vfmaq_laneq_f32(vld1q_f32(Sr +  0), d0, vk4, L); vst1q_f32(Sr +  0, s0); o0 = vfmaq_laneq_f32(o0, s0, vq4, L); \
                        const float32x4_t s1 = vfmaq_laneq_f32(vld1q_f32(Sr +  4), d1, vk4, L); vst1q_f32(Sr +  4, s1); o1 = vfmaq_laneq_f32(o1, s1, vq4, L); \
                        const float32x4_t s2 = vfmaq_laneq_f32(vld1q_f32(Sr +  8), d2, vk4, L); vst1q_f32(Sr +  8, s2); o2 = vfmaq_laneq_f32(o2, s2, vq4, L); \
                        const float32x4_t s3 = vfmaq_laneq_f32(vld1q_f32(Sr + 12), d3, vk4, L); vst1q_f32(Sr + 12, s3); o3 = vfmaq_laneq_f32(o3, s3, vq4, L); \
                    } while (0)
                    KDA_EPIL_ROW(0); KDA_EPIL_ROW(1); KDA_EPIL_ROW(2); KDA_EPIL_ROW(3);
#undef KDA_EPIL_ROW
                }
                if (t == 0 && snap != nullptr) {
                    std::memcpy(snap, S, 128 * 16 * sizeof(float));
                }
            }

            float * ot = out + h_off + c0;
            vst1q_f32(ot +  0, vmulq_f32(o0, vsc));
            vst1q_f32(ot +  4, vmulq_f32(o1, vsc));
            vst1q_f32(ot +  8, vmulq_f32(o2, vsc));
            vst1q_f32(ot + 12, vmulq_f32(o3, vsc));
        }
    };

    if (T >= 16 && pool.num_compute_threads() > pool.num_threads()) {
        const int n_big = pool.num_threads();
        // Reserve the final 3*n_big strips for big cores so slower LITTLE cores never create an end-of-barrier tail.
        const int little_limit = std::max(0, total_tasks - 3 * n_big);
        std::atomic<int> next_task{0};
        pool.parallel_compute([&](int tid, int) {
            const bool is_little = (tid >= n_big);
            alignas(64) float decay[128];
            while (true) {
                int task = next_task.load(std::memory_order_relaxed);
                while (true) {
                    if (task >= (is_little ? little_limit : total_tasks)) { task = -1; break; }
                    if (next_task.compare_exchange_weak(task, task + 1, std::memory_order_relaxed)) break;
                }
                if (task < 0) break;
                run_strip(task, decay);
            }
        });
        return;
    }
    pool.parallel_for(total_tasks, [&](int task0, int task1, int) {
        alignas(64) float decay[128];
        for (int task = task0; task < task1; ++task) {
            run_strip(task, decay);
        }
    });
    return;
#endif

#if defined(__AVX2__) && defined(__FMA__)
    // Fused single-pass L1d-tiled 4-column x 16-token KDA recurrence:
    // Pass 2 of token dt is fused with Pass 1 (decay + kv dot) of token dt+1 -> halves state load/store passes!
    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        alignas(32) float dec_blk[16][128];
        for (int h = h0; h < h1; ++h) {
            float * sh = state + static_cast<size_t>(h) * 128 * 128;
            float * sh_snap = state_snap ? (state_snap + static_cast<size_t>(h) * 128 * 128) : nullptr;
            for (int t0 = 0; t0 < T; t0 += 16) {
                const int nt = std::min(16, T - t0);
                for (int dt = 0; dt < nt; ++dt) {
                    const float * gt = g + (static_cast<size_t>(t0 + dt) * n_heads + h) * 128;
                    for (int r = 0; r < 128; r += 8) {
                        _mm256_store_ps(dec_blk[dt] + r, exp_ps_avx2(_mm256_loadu_ps(gt + r)));
                    }
                }
                for (int col = 0; col < 128; col += 4) {
                    float * sc0 = sh + static_cast<size_t>(col + 0) * 128;
                    float * sc1 = sh + static_cast<size_t>(col + 1) * 128;
                    float * sc2 = sh + static_cast<size_t>(col + 2) * 128;
                    float * sc3 = sh + static_cast<size_t>(col + 3) * 128;

                    // Prologue: Pass 1 for dt = 0 (decay S by dec_blk[0] and compute kv0..kv3)
                    __m256 kv0 = _mm256_setzero_ps(), kv1 = _mm256_setzero_ps();
                    __m256 kv2 = _mm256_setzero_ps(), kv3 = _mm256_setzero_ps();
                    {
                        const float * kt0 = k + (static_cast<size_t>(t0) * n_heads + h) * 128;
                        const float * dec0 = dec_blk[0];
                        for (int r = 0; r < 128; r += 8) {
                            const __m256 dv = _mm256_load_ps(dec0 + r);
                            const __m256 kv = _mm256_loadu_ps(kt0 + r);
                            const __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(sc0 + r), dv);
                            const __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(sc1 + r), dv);
                            const __m256 s2 = _mm256_mul_ps(_mm256_loadu_ps(sc2 + r), dv);
                            const __m256 s3 = _mm256_mul_ps(_mm256_loadu_ps(sc3 + r), dv);
                            _mm256_storeu_ps(sc0 + r, s0); _mm256_storeu_ps(sc1 + r, s1);
                            _mm256_storeu_ps(sc2 + r, s2); _mm256_storeu_ps(sc3 + r, s3);
                            kv0 = _mm256_fmadd_ps(s0, kv, kv0); kv1 = _mm256_fmadd_ps(s1, kv, kv1);
                            kv2 = _mm256_fmadd_ps(s2, kv, kv2); kv3 = _mm256_fmadd_ps(s3, kv, kv3);
                        }
                    }

                    for (int dt = 0; dt < nt; ++dt) {
                        const int t = t0 + dt;
                        const size_t h_off = (static_cast<size_t>(t) * n_heads + h) * 128;
                        const float * qt = q + h_off;
                        const float * kt = k + h_off;
                        const float * vt = v + h_off;
                        float * ot = out + h_off;
                        const float bval = beta[static_cast<size_t>(t) * n_heads + h];

                        const float d0 = (vt[col + 0] - hsum256_ps(kv0)) * bval;
                        const float d1 = (vt[col + 1] - hsum256_ps(kv1)) * bval;
                        const float d2 = (vt[col + 2] - hsum256_ps(kv2)) * bval;
                        const float d3 = (vt[col + 3] - hsum256_ps(kv3)) * bval;

                        __m256 oa0 = _mm256_setzero_ps(), oa1 = _mm256_setzero_ps();
                        __m256 oa2 = _mm256_setzero_ps(), oa3 = _mm256_setzero_ps();

                        if (dt + 1 < nt) {
                            const float * dec_n = dec_blk[dt + 1];
                            const float * kt_n  = k + (static_cast<size_t>(t + 1) * n_heads + h) * 128;
                            const bool save_snap = (t == 0 && sh_snap != nullptr);
                            float * sn0 = save_snap ? (sh_snap + static_cast<size_t>(col + 0) * 128) : nullptr;
                            kv0 = _mm256_setzero_ps(); kv1 = _mm256_setzero_ps();
                            kv2 = _mm256_setzero_ps(); kv3 = _mm256_setzero_ps();

                            for (int r = 0; r < 128; r += 8) {
                                const __m256 kv   = _mm256_loadu_ps(kt + r);
                                const __m256 qv   = _mm256_loadu_ps(qt + r);
                                const __m256 dv_n = _mm256_load_ps(dec_n + r);
                                const __m256 kv_n = _mm256_loadu_ps(kt_n + r);

                                const __m256 s0 = _mm256_fmadd_ps(kv, _mm256_set1_ps(d0), _mm256_loadu_ps(sc0 + r));
                                oa0 = _mm256_fmadd_ps(s0, qv, oa0);
                                if (save_snap) _mm256_storeu_ps(sn0 + 0 * 128 + r, s0);
                                const __m256 n0 = _mm256_mul_ps(s0, dv_n);
                                _mm256_storeu_ps(sc0 + r, n0);
                                kv0 = _mm256_fmadd_ps(n0, kv_n, kv0);

                                const __m256 s1 = _mm256_fmadd_ps(kv, _mm256_set1_ps(d1), _mm256_loadu_ps(sc1 + r));
                                oa1 = _mm256_fmadd_ps(s1, qv, oa1);
                                if (save_snap) _mm256_storeu_ps(sn0 + 1 * 128 + r, s1);
                                const __m256 n1 = _mm256_mul_ps(s1, dv_n);
                                _mm256_storeu_ps(sc1 + r, n1);
                                kv1 = _mm256_fmadd_ps(n1, kv_n, kv1);

                                const __m256 s2 = _mm256_fmadd_ps(kv, _mm256_set1_ps(d2), _mm256_loadu_ps(sc2 + r));
                                oa2 = _mm256_fmadd_ps(s2, qv, oa2);
                                if (save_snap) _mm256_storeu_ps(sn0 + 2 * 128 + r, s2);
                                const __m256 n2 = _mm256_mul_ps(s2, dv_n);
                                _mm256_storeu_ps(sc2 + r, n2);
                                kv2 = _mm256_fmadd_ps(n2, kv_n, kv2);

                                const __m256 s3 = _mm256_fmadd_ps(kv, _mm256_set1_ps(d3), _mm256_loadu_ps(sc3 + r));
                                oa3 = _mm256_fmadd_ps(s3, qv, oa3);
                                if (save_snap) _mm256_storeu_ps(sn0 + 3 * 128 + r, s3);
                                const __m256 n3 = _mm256_mul_ps(s3, dv_n);
                                _mm256_storeu_ps(sc3 + r, n3);
                                kv3 = _mm256_fmadd_ps(n3, kv_n, kv3);
                            }
                        } else {
                            const __m256 vd0 = _mm256_set1_ps(d0);
                            const __m256 vd1 = _mm256_set1_ps(d1);
                            const __m256 vd2 = _mm256_set1_ps(d2);
                            const __m256 vd3 = _mm256_set1_ps(d3);
                            for (int r = 0; r < 128; r += 8) {
                                const __m256 kv = _mm256_loadu_ps(kt + r);
                                const __m256 qv = _mm256_loadu_ps(qt + r);
                                const __m256 s0 = _mm256_fmadd_ps(kv, vd0, _mm256_loadu_ps(sc0 + r));
                                const __m256 s1 = _mm256_fmadd_ps(kv, vd1, _mm256_loadu_ps(sc1 + r));
                                const __m256 s2 = _mm256_fmadd_ps(kv, vd2, _mm256_loadu_ps(sc2 + r));
                                const __m256 s3 = _mm256_fmadd_ps(kv, vd3, _mm256_loadu_ps(sc3 + r));
                                _mm256_storeu_ps(sc0 + r, s0); _mm256_storeu_ps(sc1 + r, s1);
                                _mm256_storeu_ps(sc2 + r, s2); _mm256_storeu_ps(sc3 + r, s3);
                                oa0 = _mm256_fmadd_ps(s0, qv, oa0); oa1 = _mm256_fmadd_ps(s1, qv, oa1);
                                oa2 = _mm256_fmadd_ps(s2, qv, oa2); oa3 = _mm256_fmadd_ps(s3, qv, oa3);
                            }
                            if (t == 0 && sh_snap != nullptr) {
                                std::memcpy(sh_snap + static_cast<size_t>(col) * 128, sc0, 4 * 128 * sizeof(float));
                            }
                        }
                        ot[col + 0] = hsum256_ps(oa0) * scale;
                        ot[col + 1] = hsum256_ps(oa1) * scale;
                        ot[col + 2] = hsum256_ps(oa2) * scale;
                        ot[col + 3] = hsum256_ps(oa3) * scale;
                    }
                }
            }
        }
    });
    return;
#endif

    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        alignas(32) float dec[128];
        for (int h = h0; h < h1; ++h) {
            float * sh = state + static_cast<size_t>(h) * 128 * 128;
            float * sh_snap = state_snap ? (state_snap + static_cast<size_t>(h) * 128 * 128) : nullptr;
            for (int t = 0; t < T; ++t) {
                const size_t h_off = (static_cast<size_t>(t) * n_heads + h) * 128;
                const float * qt = q + h_off;
                const float * kt = k + h_off;
                const float * vt = v + h_off;
                const float * gt = g + h_off;
                float * ot = out + h_off;
                const float bval = beta[static_cast<size_t>(t) * n_heads + h];

                for (int r = 0; r < 128; ++r) dec[r] = std::exp(gt[r]);
                for (int col = 0; col < 128; ++col) {
                    float * sc = sh + static_cast<size_t>(col) * 128;
                    float kv = 0.0f;
                    for (int r = 0; r < 128; ++r) {
                        sc[r] *= dec[r];
                        kv += sc[r] * kt[r];
                    }
                    const float delta = (vt[col] - kv) * bval;
                    float attn = 0.0f;
                    for (int r = 0; r < 128; ++r) {
                        sc[r] += kt[r] * delta;
                        attn += sc[r] * qt[r];
                    }
                    ot[col] = attn * scale;
                }
                if (t == 0 && sh_snap != nullptr) {
                    std::memcpy(sh_snap, sh, static_cast<size_t>(128) * 128 * sizeof(float));
                }
            }
        }
    });
}

void rms_norm_sigmoid_gate_d128(
    ThreadPool & pool,
    float * o_inout, const float * norm_w, const float * gate,
    int T, int n_heads, float eps) {

    const int total = T * n_heads;
    auto work = [&](int idx0, int idx1, int) {
        for (int idx = idx0; idx < idx1; ++idx) {
            const size_t off = static_cast<size_t>(idx) * 128;
            float * op = o_inout + off;
            const float * gp = gate + off;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 ss0 = _mm256_setzero_ps();
            __m256 ss1 = _mm256_setzero_ps();
            for (int d = 0; d < 128; d += 16) {
                const __m256 v0 = _mm256_loadu_ps(op + d);
                const __m256 v1 = _mm256_loadu_ps(op + d + 8);
                ss0 = _mm256_fmadd_ps(v0, v0, ss0);
                ss1 = _mm256_fmadd_ps(v1, v1, ss1);
            }
            const float inv = 1.0f / std::sqrt(hsum256_ps(_mm256_add_ps(ss0, ss1)) * (1.0f / 128.0f) + eps);
            const __m256 vinv = _mm256_set1_ps(inv);
            for (int d = 0; d < 128; d += 8) {
                const __m256 ov = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(op + d), vinv), _mm256_loadu_ps(norm_w + d));
                _mm256_storeu_ps(op + d, _mm256_mul_ps(ov, sigmoid_ps_avx2(_mm256_loadu_ps(gp + d))));
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            for (int d = 0; d < 128; d += 8) {
                const float32x4_t v0 = vld1q_f32(op + d);
                const float32x4_t v1 = vld1q_f32(op + d + 4);
                s0 = vfmaq_f32(s0, v0, v0);
                s1 = vfmaq_f32(s1, v1, v1);
            }
            const float inv = 1.0f / std::sqrt(vaddvq_f32(vaddq_f32(s0, s1)) * (1.0f / 128.0f) + eps);
            const float32x4_t vinv = vdupq_n_f32(inv);
            const float32x4_t v1   = vdupq_n_f32(1.0f);
            for (int d = 0; d < 128; d += 4) {
                const float32x4_t ov = vmulq_f32(vmulq_f32(vld1q_f32(op + d), vinv), vld1q_f32(norm_w + d));
                const float32x4_t gv = vld1q_f32(gp + d);
                vst1q_f32(op + d, vdivq_f32(ov, vaddq_f32(v1, vexpq_f32_fast(vnegq_f32(gv)))));
            }
#else
            float ss = 0.0f;
            for (int d = 0; d < 128; ++d) ss += op[d] * op[d];
            const float inv = 1.0f / std::sqrt(ss * (1.0f / 128.0f) + eps);
            for (int d = 0; d < 128; ++d) {
                op[d] = (op[d] * inv * norm_w[d]) / (1.0f + std::exp(-gp[d]));
            }
#endif
        }
    };

    if (total <= 16) {
        work(0, total, 0);
    } else {
        pool.parallel_for(total, work);
    }
}

void mla_rope_split_q(
    const float * q_all, const int32_t * pos,
    float * q_nope, float * q_pe,
    int T, int n_heads, int qk_nope, int rope_dim, float rope_freq_base) {

    const int qk_tot = qk_nope + rope_dim;
    const int half_r = rope_dim >> 1;
    std::vector<float> freqs(static_cast<size_t>(half_r));
    for (int d = 0; d < half_r; ++d) {
        freqs[d] = std::pow(rope_freq_base, -static_cast<float>(2 * d) / static_cast<float>(rope_dim));
    }

    for (int t = 0; t < T; ++t) {
        const float p = static_cast<float>(pos[t]);
        float cosv[64], sinv[64];
        for (int d = 0; d < half_r; ++d) {
            const float theta = p * freqs[d];
            cosv[d] = std::cos(theta);
            sinv[d] = std::sin(theta);
        }
        for (int h = 0; h < n_heads; ++h) {
            const float * src = q_all + (static_cast<size_t>(t) * n_heads + h) * qk_tot;
            float * dn = q_nope + (static_cast<size_t>(t) * n_heads + h) * qk_nope;
            float * dp = q_pe   + (static_cast<size_t>(t) * n_heads + h) * rope_dim;
            std::memcpy(dn, src, static_cast<size_t>(qk_nope) * sizeof(float));
            for (int d = 0; d < half_r; ++d) {
                const float x0 = src[qk_nope + 2 * d + 0];
                const float x1 = src[qk_nope + 2 * d + 1];
                dp[2 * d + 0] = x0 * cosv[d] - x1 * sinv[d];
                dp[2 * d + 1] = x0 * sinv[d] + x1 * cosv[d];
            }
        }
    }
}

// Quantize K (multiple of 32) FP32 elements into SoA Q4_0:
//   dst_qs[b*16 + j] = (q[j] & 0xF) | (q[j+16] << 4), q in [1..15], dequant = d * (q - 8)
//   dst_d[b]         = fp16(amax / 7.0f)
inline void quantize_vec_q4_0_soa(const float * x, uint8_t * dst_qs, uint16_t * dst_d, int K) {
    const int nb = K >> 5;
#if defined(__AVX2__)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256i v8       = _mm256_set1_epi32(8);
    const __m256i vmin1    = _mm256_set1_epi32(1);
    const __m256i vmax15   = _mm256_set1_epi32(15);
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        __m256 v0 = _mm256_loadu_ps(xb + 0);
        __m256 v1 = _mm256_loadu_ps(xb + 8);
        __m256 v2 = _mm256_loadu_ps(xb + 16);
        __m256 v3 = _mm256_loadu_ps(xb + 24);
        __m256 m01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1));
        __m256 m23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3));
        __m256 m   = _mm256_max_ps(m01, m23);
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        const float amax = _mm_cvtss_f32(m128);
        const float d    = amax * (1.0f / 7.0f);
        const float id   = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
        dst_d[b] = f32_to_f16_fast(d);

        const __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v0, vid)), v8), vmin1), vmax15);
        __m256i i1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v1, vid)), v8), vmin1), vmax15);
        __m256i i2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v2, vid)), v8), vmin1), vmax15);
        __m256i i3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v3, vid)), v8), vmin1), vmax15);
        __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), _MM_SHUFFLE(3, 1, 2, 0));
        __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), _MM_SHUFFLE(3, 1, 2, 0));
        __m256i q8  = _mm256_permute4x64_epi64(_mm256_packus_epi16(p01, p23), _MM_SHUFFLE(3, 1, 2, 0));
        __m128i lo  = _mm256_castsi256_si128(q8);
        __m128i hi  = _mm256_extracti128_si256(q8, 1);
        __m128i pk  = _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_qs + b * 16), pk);
    }
#else
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d  = amax * (1.0f / 7.0f);
        const float id = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
        dst_d[b] = f32_to_f16_fast(d);
        uint8_t * qout = dst_qs + b * 16;
        for (int j = 0; j < 16; ++j) {
            int q0 = static_cast<int>(std::lrintf(xb[j] * id)) + 8;
            int q1 = static_cast<int>(std::lrintf(xb[j + 16] * id)) + 8;
            q0 = std::max(1, std::min(15, q0));
            q1 = std::max(1, std::min(15, q1));
            qout[j] = static_cast<uint8_t>((q0 & 0x0F) | (q1 << 4));
        }
    }
#endif
}

void mla_kv_norm_rope_write_576(
    const float * kv_all, const float * norm_w, const int32_t * pos,
    void * kv_cache, KvCacheType kv_type,
    int T, float rms_eps, float rope_freq_base) {

    float freqs[32];
    for (int d = 0; d < 32; ++d) {
        freqs[d] = std::pow(rope_freq_base, -static_cast<float>(2 * d) / 64.0f);
    }

    alignas(32) float buf[576];
    for (int t = 0; t < T; ++t) {
        const float * src = kv_all + static_cast<size_t>(t) * 576;
        float ss = 0.0f;
        for (int i = 0; i < 512; ++i) ss += src[i] * src[i];
        const float inv = 1.0f / std::sqrt(ss * (1.0f / 512.0f) + rms_eps);
        for (int i = 0; i < 512; ++i) buf[i] = src[i] * inv * norm_w[i];

        const float p = static_cast<float>(pos[t]);
        for (int d = 0; d < 32; ++d) {
            const float theta = p * freqs[d];
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float x0 = src[512 + 2 * d + 0];
            const float x1 = src[512 + 2 * d + 1];
            buf[512 + 2 * d + 0] = x0 * c - x1 * s;
            buf[512 + 2 * d + 1] = x0 * s + x1 * c;
        }

        const int pidx = pos[t];
        if (kv_type == KvCacheType::Q8_0) {
            auto & dst = static_cast<CpuMlaKv576Q8Pos *>(kv_cache)[pidx];
            alignas(32) CpuActQ8_0 tmp_q8[18];
            quantize_row_q8_0(buf, tmp_q8, 576);
            for (int b = 0; b < 18; ++b) {
                dst.d[b] = f32_to_f16_fast(tmp_q8[b].d);
                std::memcpy(dst.qs + b * 32, tmp_q8[b].qs, 32);
            }
        } else if (kv_type == KvCacheType::Q4_0) {
            auto & dst = static_cast<CpuMlaKv576Q4Pos *>(kv_cache)[pidx];
            quantize_vec_q4_0_soa(buf, dst.qs, dst.d, 576);
        } else {
            auto * dst = static_cast<uint16_t *>(kv_cache) + static_cast<size_t>(pidx) * 576;
            for (int i = 0; i < 576; ++i) dst[i] = f32_to_f16_fast(buf[i]);
        }
    }
}

void batched_head_proj(
    ThreadPool & pool,
    const CpuTensor & W, const float * x, float * y,
    int T) {

    const int K       = static_cast<int>(W.ne[0]);
    const int M       = static_cast<int>(W.ne[1]);
    const int n_heads = static_cast<int>(W.ne[2]);
    const auto tr     = dtype_traits(W.dtype);
    const size_t head_bytes = (static_cast<size_t>(K) / tr.block_elems) * tr.block_bytes * M;
    const auto * base = static_cast<const uint8_t *>(W.ptr);

    auto run_head = [&](int h, std::vector<float> & xh_buf, std::vector<float> & yh_buf, std::vector<float> & xp_buf,
                        std::vector<CpuActQ8_0> & xh_q8, std::vector<CpuActQ8_K> & xh_q8k) {
        (void) xp_buf;
        CpuTensor Wh{base + static_cast<size_t>(h) * head_bytes, W.dtype,
                     {static_cast<uint64_t>(K), static_cast<uint64_t>(M), 1, 1}, head_bytes};
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
        if (T >= 12) {
            const int n_tiles = (T + 15) >> 4;
            xp_buf.resize(static_cast<size_t>(n_tiles) * K * 16);
            yh_buf.resize(static_cast<size_t>(T) * M);
            xh_buf.resize(static_cast<size_t>(T) * K);
            for (int t = 0; t < T; ++t) {
                std::memcpy(xh_buf.data() + static_cast<size_t>(t) * K,
                            x + (static_cast<size_t>(t) * n_heads + h) * K,
                            static_cast<size_t>(K) * sizeof(float));
            }
            pack_x_panels_16(xh_buf.data(), xp_buf.data(), T, K);
            gemm_fma_rows_range(Wh, xp_buf.data(), yh_buf.data(), T, 0, M);
            for (int t = 0; t < T; ++t) {
                std::memcpy(y + (static_cast<size_t>(t) * n_heads + h) * M,
                            yh_buf.data() + static_cast<size_t>(t) * M,
                            static_cast<size_t>(M) * sizeof(float));
            }
            return;
        }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        // ARM64 dotprod: quantize all T tokens per head and dispatch to linear_rows_range_direct
        // so T >= 4 automatically hits L2-blocked Q8_0 4x4 and L1-unpacked Q6_K/Q4_K 4x4 GEMM
        const bool use_q8  = needs_act_q8_0(W.dtype);
        const bool use_q8k = needs_act_q8_k(W.dtype);
        yh_buf.resize(static_cast<size_t>(T) * M);
        if (use_q8) {
            const int nb = K >> 5;
            xh_q8.resize(static_cast<size_t>(T) * nb);
            for (int t = 0; t < T; ++t) {
                quantize_row_q8_0(x + (static_cast<size_t>(t) * n_heads + h) * K,
                                  xh_q8.data() + static_cast<size_t>(t) * nb, K);
            }
            linear_rows_range_direct(Wh, nullptr, xh_q8.data(), nullptr, yh_buf.data(), T, 0, M);
        } else if (use_q8k) {
            const int nb = K >> 8;
            xh_q8k.resize(static_cast<size_t>(T) * nb);
            for (int t = 0; t < T; ++t) {
                quantize_row_q8_k(x + (static_cast<size_t>(t) * n_heads + h) * K,
                                  xh_q8k.data() + static_cast<size_t>(t) * nb, K);
            }
            linear_rows_range_direct(Wh, nullptr, nullptr, xh_q8k.data(), yh_buf.data(), T, 0, M);
        } else {
            xh_buf.resize(static_cast<size_t>(T) * K);
            for (int t = 0; t < T; ++t) {
                std::memcpy(xh_buf.data() + static_cast<size_t>(t) * K,
                            x + (static_cast<size_t>(t) * n_heads + h) * K,
                            static_cast<size_t>(K) * sizeof(float));
            }
            linear_rows_range_direct(Wh, xh_buf.data(), nullptr, nullptr, yh_buf.data(), T, 0, M);
        }
        for (int t = 0; t < T; ++t) {
            std::memcpy(y + (static_cast<size_t>(t) * n_heads + h) * M,
                        yh_buf.data() + static_cast<size_t>(t) * M,
                        static_cast<size_t>(M) * sizeof(float));
        }
        return;
#endif
        // T < 6: direct zero-copy quantized GEMV per head!
        if (W.dtype == DType::Q8_0) {
            const int nb = K >> 5;
            xh_q8.resize(static_cast<size_t>(nb));
            const auto * wh_q8 = static_cast<const BlockQ8_0 *>(Wh.ptr);
            for (int t = 0; t < T; ++t) {
                quantize_row_q8_0(x + (static_cast<size_t>(t) * n_heads + h) * K, xh_q8.data(), K);
                float * yt = y + (static_cast<size_t>(t) * n_heads + h) * M;
                int r = 0;
                for (; r + 3 < M; r += 4) {
                    dot_q8_0_q8_0_4rows(wh_q8 + (r + 0) * nb, wh_q8 + (r + 1) * nb,
                                        wh_q8 + (r + 2) * nb, wh_q8 + (r + 3) * nb,
                                        xh_q8.data(), nb, yt[r + 0], yt[r + 1], yt[r + 2], yt[r + 3]);
                }
                for (; r < M; ++r) yt[r] = dot_q8_0_q8_0_row(wh_q8 + r * nb, xh_q8.data(), nb);
            }
        } else {
            xh_buf.resize(static_cast<size_t>(T) * K);
            yh_buf.resize(static_cast<size_t>(T) * M);
            for (int t = 0; t < T; ++t) {
                std::memcpy(xh_buf.data() + static_cast<size_t>(t) * K,
                            x + (static_cast<size_t>(t) * n_heads + h) * K,
                            static_cast<size_t>(K) * sizeof(float));
            }
            if (needs_act_q8_0(W.dtype)) {
                xh_q8.resize(static_cast<size_t>(T) * (K >> 5));
                for (int t = 0; t < T; ++t) quantize_row_q8_0(xh_buf.data() + static_cast<size_t>(t) * K, xh_q8.data() + static_cast<size_t>(t) * (K >> 5), K);
            } else if (needs_act_q8_k(W.dtype)) {
                xh_q8k.resize(static_cast<size_t>(T) * (K >> 8));
                for (int t = 0; t < T; ++t) quantize_row_q8_k(xh_buf.data() + static_cast<size_t>(t) * K, xh_q8k.data() + static_cast<size_t>(t) * (K >> 8), K);
            }
            linear_rows_range_direct(Wh, xh_buf.data(), xh_q8.empty() ? nullptr : xh_q8.data(), xh_q8k.empty() ? nullptr : xh_q8k.data(), yh_buf.data(), T, 0, M);
            for (int t = 0; t < T; ++t) {
                std::memcpy(y + (static_cast<size_t>(t) * n_heads + h) * M,
                            yh_buf.data() + static_cast<size_t>(t) * M,
                            static_cast<size_t>(M) * sizeof(float));
            }
        }
    };

    if (T >= 16 && pool.num_compute_threads() > pool.num_threads() && n_heads >= pool.num_threads() * 4) {
        const int n_big = pool.num_threads();
        const int little_limit = std::max(0, n_heads - 3 * n_big);
        std::atomic<int> next_h{0};
        pool.parallel_compute([&](int tid, int) {
            const bool is_little = (tid >= n_big);
            std::vector<float>      xh_buf, yh_buf, xp_buf;
            std::vector<CpuActQ8_0> xh_q8;
            std::vector<CpuActQ8_K> xh_q8k;
            while (true) {
                int h = next_h.load(std::memory_order_relaxed);
                while (true) {
                    if (h >= (is_little ? little_limit : n_heads)) { h = -1; break; }
                    if (next_h.compare_exchange_weak(h, h + 1, std::memory_order_relaxed)) break;
                }
                if (h < 0) break;
                run_head(h, xh_buf, yh_buf, xp_buf, xh_q8, xh_q8k);
            }
        });
        return;
    }

    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        std::vector<float>      xh_buf, yh_buf, xp_buf;
        std::vector<CpuActQ8_0> xh_q8;
        std::vector<CpuActQ8_K> xh_q8k;
        for (int h = h0; h < h1; ++h) run_head(h, xh_buf, yh_buf, xp_buf, xh_q8, xh_q8k);
    });
}

void mla_concat_q576(
    const float * q_abs, const float * q_pe, float * q576,
    int T, int n_heads) {

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const float * a = q_abs + (static_cast<size_t>(t) * n_heads + h) * 512;
            const float * p = q_pe  + (static_cast<size_t>(t) * n_heads + h) * 64;
            float * dst     = q576  + (static_cast<size_t>(t) * n_heads + h) * 576;
            std::memcpy(dst,       a, 512 * sizeof(float));
            std::memcpy(dst + 512, p,  64 * sizeof(float));
        }
    }
}

void mla_attn_d576_v512(
    ThreadPool & pool,
    const float * q576, const void * kv_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_heads, float scale) {

#if defined(__AVX2__) && defined(__FMA__)
    if (kv_type == KvCacheType::Q8_0) {
        const auto * kvc = static_cast<const CpuMlaKv576Q8Pos *>(kv_cache);
        // Partition heads across threads; each thread shares KV cache reads across all its heads (e.g. 4 heads/core)!
        // Uses 8-position tile-softmax with AVX2 exp_ps_avx2 + sparse V skipping + batched F16C scale decode!
        pool.parallel_for(n_heads, [&](int h0, int h1, int) {
            const int nh = h1 - h0;
            constexpr int kMaxH = 16;
            alignas(32) float q_scaled[576];
            alignas(32) CpuActQ8_0 q_q8[kMaxH][18];
            alignas(32) float ov[kMaxH][512];
            float m_i[kMaxH];
            float l_i[kMaxH];
            const __m256i ones16 = _mm256_set1_epi16(1);

            for (int t = 0; t < T; ++t) {
                const int kv_len = pos[t] + 1;
                for (int hi = 0; hi < nh; ++hi) {
                    const float * qsrc = q576 + (static_cast<size_t>(t) * n_heads + (h0 + hi)) * 576;
                    const __m256 vsc = _mm256_set1_ps(scale);
                    for (int i = 0; i < 576; i += 8) _mm256_store_ps(q_scaled + i, _mm256_mul_ps(_mm256_loadu_ps(qsrc + i), vsc));
                    quantize_row_q8_0(q_scaled, q_q8[hi], 576);
                    m_i[hi] = -FLT_MAX;
                    l_i[hi] = 0.0f;
                    std::memset(ov[hi], 0, 512 * sizeof(float));
                }

                constexpr int P_TILE = 8;
                alignas(32) float scores[kMaxH][P_TILE];
                alignas(32) float betas[P_TILE];
                alignas(32) float kv_d_cache[P_TILE][24];

                for (int p0 = 0; p0 < kv_len; p0 += P_TILE) {
                    const int np = std::min(P_TILE, kv_len - p0);
                    if (p0 + P_TILE < kv_len) {
                        const int np_next = std::min(P_TILE, kv_len - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const char * pf = reinterpret_cast<const char *>(kvc + p0 + P_TILE + pi);
                            _mm_prefetch(pf +   0, _MM_HINT_T0);
                            _mm_prefetch(pf + 256, _MM_HINT_T0);
                            _mm_prefetch(pf + 576, _MM_HINT_T0);
                        }
                    }
                    // Pre-decode FP16 scales for this 8-pos tile ONCE for all nh heads!
                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q8Pos & kv = kvc[p0 + pi];
#if defined(__F16C__)
                        _mm256_store_ps(kv_d_cache[pi] + 0, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.d + 0))));
                        _mm256_store_ps(kv_d_cache[pi] + 8, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.d + 8))));
                        kv_d_cache[pi][16] = f16_to_f32_fast(kv.d[16]);
                        kv_d_cache[pi][17] = f16_to_f32_fast(kv.d[17]);
#else
                        for (int b = 0; b < 18; ++b) kv_d_cache[pi][b] = f16_to_f32_fast(kv.d[b]);
#endif
                    }

                    // Score all (hi, pi) in 4-head register-resident batches: load kv & vpabsb(kv) ONCE per 4 heads!
                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q8Pos & kv = kvc[p0 + pi];
                        int hi = 0;
                        for (; hi + 3 < nh; hi += 4) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                            for (int b = 0; b < 18; ++b) {
                                const __m256i kv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(kv.qs + b * 32));
                                const __m256i akv = _mm256_abs_epi8(kv0);
                                const float kd = kv_d_cache[pi][b];
                                const __m256i q0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 0][b].qs));
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q0, kv0)), ones16)), s0);
                                const __m256i q1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 1][b].qs));
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q1, kv0)), ones16)), s1);
                                const __m256i q2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 2][b].qs));
                                s2 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 2][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q2, kv0)), ones16)), s2);
                                const __m256i q3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 3][b].qs));
                                s3 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 3][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q3, kv0)), ones16)), s3);
                            }
                            scores[hi + 0][pi] = hsum256_ps(s0);
                            scores[hi + 1][pi] = hsum256_ps(s1);
                            scores[hi + 2][pi] = hsum256_ps(s2);
                            scores[hi + 3][pi] = hsum256_ps(s3);
                        }
                        for (; hi < nh; ++hi) {
                            __m256 sacc0 = _mm256_setzero_ps();
                            __m256 sacc1 = _mm256_setzero_ps();
                            for (int b = 0; b < 18; b += 2) {
                                const __m256i kv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(kv.qs + (b + 0) * 32));
                                const __m256i qv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi][b + 0].qs));
                                sacc0 = _mm256_fmadd_ps(_mm256_set1_ps(kv_d_cache[pi][b + 0] * q_q8[hi][b + 0].d),
                                                        _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(kv0, qv0, ones16)), sacc0);
                                const __m256i kv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(kv.qs + (b + 1) * 32));
                                const __m256i qv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi][b + 1].qs));
                                sacc1 = _mm256_fmadd_ps(_mm256_set1_ps(kv_d_cache[pi][b + 1] * q_q8[hi][b + 1].d),
                                                        _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(kv1, qv1, ones16)), sacc1);
                            }
                            scores[hi][pi] = hsum256_ps(_mm256_add_ps(sacc0, sacc1));
                        }
                    }

                    // Tile online-softmax + sparse V accumulation per head
                    for (int hi = 0; hi < nh; ++hi) {
                        float m_tile = -FLT_MAX;
                        for (int pi = 0; pi < np; ++pi) m_tile = std::max(m_tile, scores[hi][pi]);
                        if (m_tile <= m_i[hi] - 20.0f) continue;

                        const float m_new = std::max(m_i[hi], m_tile);
                        const float alpha = (m_i[hi] == -FLT_MAX) ? 0.0f : std::exp(m_i[hi] - m_new);
                        l_i[hi] *= alpha;
                        if (alpha != 1.0f) {
                            const __m256 va = _mm256_set1_ps(alpha);
                            for (int i = 0; i < 512; i += 8) _mm256_store_ps(ov[hi] + i, _mm256_mul_ps(_mm256_load_ps(ov[hi] + i), va));
                        }

                        for (int pi = np; pi < P_TILE; ++pi) scores[hi][pi] = -1e30f;
                        const __m256 vb = exp_ps_avx2(_mm256_sub_ps(_mm256_load_ps(scores[hi]), _mm256_set1_ps(m_new)));
                        _mm256_store_ps(betas, vb);

                        for (int pi = 0; pi < np; ++pi) {
                            if (scores[hi][pi] <= m_new - 18.0f) continue;
                            const float beta = betas[pi];
                            l_i[hi] += beta;
                            const CpuMlaKv576Q8Pos & kv = kvc[p0 + pi];
                            for (int b = 0; b < 16; ++b) {
                                const __m256 c = _mm256_set1_ps(beta * kv_d_cache[pi][b]);
                                const int8_t * ks = kv.qs + b * 32;
                                float * ob = ov[hi] + b * 32;
                                const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ks));
                                const __m128i hi8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ks + 16));
                                _mm256_store_ps(ob +  0, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_load_ps(ob + 0)));
                                _mm256_store_ps(ob +  8, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_load_ps(ob + 8)));
                                _mm256_store_ps(ob + 16, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi8)), _mm256_load_ps(ob + 16)));
                                _mm256_store_ps(ob + 24, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi8, 8))), _mm256_load_ps(ob + 24)));
                            }
                        }
                        m_i[hi] = m_new;
                    }
                }

                for (int hi = 0; hi < nh; ++hi) {
                    const float inv_l = (l_i[hi] > 0.0f) ? (1.0f / l_i[hi]) : 0.0f;
                    const __m256 vil = _mm256_set1_ps(inv_l);
                    float * dst = out + (static_cast<size_t>(t) * n_heads + (h0 + hi)) * 512;
                    for (int i = 0; i < 512; i += 8) _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(ov[hi] + i), vil));
                }
            }
        });
        return;
    }
    if (kv_type == KvCacheType::Q4_0) {
        const auto * kvc = static_cast<const CpuMlaKv576Q4Pos *>(kv_cache);
        pool.parallel_for(n_heads, [&](int h0, int h1, int) {
            const int nh = h1 - h0;
            constexpr int kMaxH = 16;
            alignas(32) float q_scaled[576];
            alignas(32) CpuActQ8_0 q_q8[kMaxH][18];
            alignas(32) float ov[kMaxH][512];
            float m_i[kMaxH];
            float l_i[kMaxH];
            const __m256i m4     = _mm256_set1_epi8(0x0F);
            const __m256i c8     = _mm256_set1_epi8(8);
            const __m256i ones16 = _mm256_set1_epi16(1);

            for (int t = 0; t < T; ++t) {
                const int kv_len = pos[t] + 1;
                for (int hi = 0; hi < nh; ++hi) {
                    const float * qsrc = q576 + (static_cast<size_t>(t) * n_heads + (h0 + hi)) * 576;
                    const __m256 vsc = _mm256_set1_ps(scale);
                    for (int i = 0; i < 576; i += 8) _mm256_store_ps(q_scaled + i, _mm256_mul_ps(_mm256_loadu_ps(qsrc + i), vsc));
                    quantize_row_q8_0(q_scaled, q_q8[hi], 576);
                    m_i[hi] = -FLT_MAX;
                    l_i[hi] = 0.0f;
                    std::memset(ov[hi], 0, 512 * sizeof(float));
                }

                constexpr int P_TILE = 8;
                alignas(32) float scores[kMaxH][P_TILE];
                alignas(32) float betas[P_TILE];
                alignas(32) float kv_d_cache[P_TILE][24];

                for (int p0 = 0; p0 < kv_len; p0 += P_TILE) {
                    const int np = std::min(P_TILE, kv_len - p0);
                    if (p0 + P_TILE < kv_len) {
                        const int np_next = std::min(P_TILE, kv_len - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const char * pf = reinterpret_cast<const char *>(kvc + p0 + P_TILE + pi);
                            _mm_prefetch(pf +   0, _MM_HINT_T0);
                            _mm_prefetch(pf + 192, _MM_HINT_T0);
                        }
                    }
                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q4Pos & kv = kvc[p0 + pi];
#if defined(__F16C__)
                        _mm256_store_ps(kv_d_cache[pi] + 0, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.d + 0))));
                        _mm256_store_ps(kv_d_cache[pi] + 8, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.d + 8))));
                        kv_d_cache[pi][16] = f16_to_f32_fast(kv.d[16]);
                        kv_d_cache[pi][17] = f16_to_f32_fast(kv.d[17]);
#else
                        for (int b = 0; b < 18; ++b) kv_d_cache[pi][b] = f16_to_f32_fast(kv.d[b]);
#endif
                    }

                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q4Pos & kv = kvc[p0 + pi];
                        // Register-resident 4-head batches: unpack Q4_0 nibbles ONCE per 4 heads with ZERO stack spills!
                        int hi = 0;
                        for (; hi + 3 < nh; hi += 4) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
                            for (int b = 0; b < 18; ++b) {
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.qs + b * 16));
                                const __m256i qk  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                                const float kd = kv_d_cache[pi][b];
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 0][b].qs))), ones16)), s0);
                                b0 += kd * q_q8[hi + 0][b].s;
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 1][b].qs))), ones16)), s1);
                                b1 += kd * q_q8[hi + 1][b].s;
                                s2 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 2][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 2][b].qs))), ones16)), s2);
                                b2 += kd * q_q8[hi + 2][b].s;
                                s3 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi + 3][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi + 3][b].qs))), ones16)), s3);
                                b3 += kd * q_q8[hi + 3][b].s;
                            }
                            scores[hi + 0][pi] = hsum256_ps(s0) - 8.0f * b0;
                            scores[hi + 1][pi] = hsum256_ps(s1) - 8.0f * b1;
                            scores[hi + 2][pi] = hsum256_ps(s2) - 8.0f * b2;
                            scores[hi + 3][pi] = hsum256_ps(s3) - 8.0f * b3;
                        }
                        for (; hi < nh; ++hi) {
                            __m256 s0 = _mm256_setzero_ps();
                            float sb = 0.0f;
                            for (int b = 0; b < 18; ++b) {
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.qs + b * 16));
                                const __m256i qk  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                                const float kd = kv_d_cache[pi][b];
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[hi][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi][b].qs))), ones16)), s0);
                                sb += kd * q_q8[hi][b].s;
                            }
                            scores[hi][pi] = hsum256_ps(s0) - 8.0f * sb;
                        }
                    }

                    for (int hi = 0; hi < nh; ++hi) {
                        float m_tile = -FLT_MAX;
                        for (int pi = 0; pi < np; ++pi) m_tile = std::max(m_tile, scores[hi][pi]);
                        if (m_tile <= m_i[hi] - 20.0f) continue;

                        const float m_new = std::max(m_i[hi], m_tile);
                        const float alpha = (m_i[hi] == -FLT_MAX) ? 0.0f : std::exp(m_i[hi] - m_new);
                        l_i[hi] *= alpha;
                        if (alpha != 1.0f) {
                            const __m256 va = _mm256_set1_ps(alpha);
                            for (int i = 0; i < 512; i += 8) _mm256_store_ps(ov[hi] + i, _mm256_mul_ps(_mm256_load_ps(ov[hi] + i), va));
                        }

                        for (int pi = np; pi < P_TILE; ++pi) scores[hi][pi] = -1e30f;
                        const __m256 vb = exp_ps_avx2(_mm256_sub_ps(_mm256_load_ps(scores[hi]), _mm256_set1_ps(m_new)));
                        _mm256_store_ps(betas, vb);

                        for (int pi = 0; pi < np; ++pi) {
                            if (scores[hi][pi] <= m_new - 18.0f) continue;
                            const float beta = betas[pi];
                            l_i[hi] += beta;
                            const CpuMlaKv576Q4Pos & kv = kvc[p0 + pi];
                            for (int b = 0; b < 16; ++b) {
                                const __m256 c = _mm256_set1_ps(beta * kv_d_cache[pi][b]);
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kv.qs + b * 16));
                                const __m256i q8  = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4), c8);
                                const __m128i lo  = _mm256_castsi256_si128(q8);
                                const __m128i hi8 = _mm256_extracti128_si256(q8, 1);
                                float * ob = ov[hi] + b * 32;
                                _mm256_store_ps(ob +  0, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_load_ps(ob + 0)));
                                _mm256_store_ps(ob +  8, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_load_ps(ob + 8)));
                                _mm256_store_ps(ob + 16, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi8)), _mm256_load_ps(ob + 16)));
                                _mm256_store_ps(ob + 24, _mm256_fmadd_ps(c, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi8, 8))), _mm256_load_ps(ob + 24)));
                            }
                        }
                        m_i[hi] = m_new;
                    }
                }

                for (int hi = 0; hi < nh; ++hi) {
                    const float inv_l = (l_i[hi] > 0.0f) ? (1.0f / l_i[hi]) : 0.0f;
                    const __m256 vil = _mm256_set1_ps(inv_l);
                    float * dst = out + (static_cast<size_t>(t) * n_heads + (h0 + hi)) * 512;
                    for (int i = 0; i < 512; i += 8) _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(ov[hi] + i), vil));
                }
            }
        });
        return;
    }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (kv_type == KvCacheType::Q8_0 || kv_type == KvCacheType::Q4_0) {
        // Group 4 heads per task -> 4 heads share ONE KV cache read in Q*K^T score pass AND ONE
        // V dequant pass! Double-ended dynamic queue eliminates 1.75x causal imbalance.
        const bool is_q4   = (kv_type == KvCacheType::Q4_0);
        const int h_groups = (n_heads + 3) / 4;
        const int n_tasks  = T * h_groups;
        const auto * kvc_q8 = is_q4 ? nullptr : static_cast<const CpuMlaKv576Q8Pos *>(kv_cache);
        const auto * kvc_q4 = is_q4 ? static_cast<const CpuMlaKv576Q4Pos *>(kv_cache) : nullptr;
        const bool use_dyn = (T >= 4 && pool.num_compute_threads() > 1);
        const bool use_hyb = (T >= 16 && pool.num_compute_threads() > pool.num_threads());
        const int  n_big   = pool.num_threads();
        std::atomic<uint64_t> bounds{static_cast<uint64_t>(static_cast<uint32_t>(n_tasks)) << 32};

        auto run_task = [&](int task, std::vector<float> & scores_buf,
                            float (&q_scaled)[4][576], CpuActQ8_0 (&q_q8)[4][18],
                            float (&ov)[4][512]) {
            const int t      = task / h_groups;
            const int hb     = (task % h_groups) * 4;
            const int hn     = std::min(4, n_heads - hb);
            const int kv_len = pos[t] + 1;

            if (scores_buf.size() < static_cast<size_t>(4) * kv_len) {
                scores_buf.resize(static_cast<size_t>(4) * kv_len);
            }

            const float32x4_t vsc = vdupq_n_f32(scale);
            for (int hi = 0; hi < hn; ++hi) {
                const float * qp = q576 + (static_cast<size_t>(t) * n_heads + hb + hi) * 576;
                for (int d = 0; d < 576; d += 8) {
                    vst1q_f32(q_scaled[hi] + d + 0, vmulq_f32(vld1q_f32(qp + d + 0), vsc));
                    vst1q_f32(q_scaled[hi] + d + 4, vmulq_f32(vld1q_f32(qp + d + 4), vsc));
                }
                quantize_row_q8_0(q_scaled[hi], q_q8[hi], 576);
            }

            // Pass 1: 4-head-shared Q * K^T scores
            if (is_q4) {
                const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
                if (hn == 4) {
                    const float32x4_t v8 = vdupq_n_f32(8.0f);
                    for (int p = 0; p < kv_len; ++p) {
                        const CpuMlaKv576Q4Pos & kv = kvc_q4[p];
                        if (p + 2 < kv_len) __builtin_prefetch(&kvc_q4[p + 2], 0, 0);
                        float32x4_t acc   = vdupq_n_f32(0.0f);
                        float32x4_t sbias = vdupq_n_f32(0.0f);
                        for (int b = 0; b < 18; ++b) {
                            const uint8x16_t raw = vld1q_u8(kv.qs + b * 16);
                            const int8x16_t k0 = vreinterpretq_s8_u8(vandq_u8(raw, m4));
                            const int8x16_t k1 = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                            int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[0][b].qs)), k1, vld1q_s8(q_q8[0][b].qs + 16));
                            int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[1][b].qs)), k1, vld1q_s8(q_q8[1][b].qs + 16));
                            int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[2][b].qs)), k1, vld1q_s8(q_q8[2][b].qs + 16));
                            int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[3][b].qs)), k1, vld1q_s8(q_q8[3][b].qs + 16));
                            const float32x4_t fsum = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)));
                            const float kd = f16_to_f32_fast(kv.d[b]);
                            const float qd[4] = {q_q8[0][b].d * kd, q_q8[1][b].d * kd, q_q8[2][b].d * kd, q_q8[3][b].d * kd};
                            const float qs[4] = {q_q8[0][b].s, q_q8[1][b].s, q_q8[2][b].s, q_q8[3][b].s};
                            acc   = vfmaq_f32(acc, fsum, vld1q_f32(qd));
                            sbias = vfmaq_f32(sbias, vld1q_f32(qs), vdupq_n_f32(kd));
                        }
                        acc = vfmsq_f32(acc, sbias, v8);
                        float sc4[4];
                        vst1q_f32(sc4, acc);
                        scores_buf[0 * kv_len + p] = sc4[0];
                        scores_buf[1 * kv_len + p] = sc4[1];
                        scores_buf[2 * kv_len + p] = sc4[2];
                        scores_buf[3 * kv_len + p] = sc4[3];
                    }
                } else {
                    for (int p = 0; p < kv_len; ++p) {
                        const CpuMlaKv576Q4Pos & kv = kvc_q4[p];
                        if (p + 2 < kv_len) __builtin_prefetch(&kvc_q4[p + 2], 0, 0);
                        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        float sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        for (int b = 0; b < 18; ++b) {
                            const uint8x16_t raw = vld1q_u8(kv.qs + b * 16);
                            const int8x16_t k0 = vreinterpretq_s8_u8(vandq_u8(raw, m4));
                            const int8x16_t k1 = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                            const float kd = f16_to_f32_fast(kv.d[b]);
                            for (int hi = 0; hi < hn; ++hi) {
                                int32x4_t dp = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[hi][b].qs)),
                                                         k1, vld1q_s8(q_q8[hi][b].qs + 16));
                                sc[hi] += (kd * q_q8[hi][b].d) * static_cast<float>(vaddvq_s32(dp));
                                sb[hi] += kd * q_q8[hi][b].s;
                            }
                        }
                        for (int hi = 0; hi < hn; ++hi) scores_buf[hi * kv_len + p] = sc[hi] - 8.0f * sb[hi];
                    }
                }
            } else if (hn == 4) {
                for (int p = 0; p < kv_len; ++p) {
                    const CpuMlaKv576Q8Pos & kv = kvc_q8[p];
                    if (p + 2 < kv_len) __builtin_prefetch(&kvc_q8[p + 2], 0, 0);
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int b = 0; b < 18; ++b) {
                        const int8x16_t k0 = vld1q_s8(kv.qs + b * 32);
                        const int8x16_t k1 = vld1q_s8(kv.qs + b * 32 + 16);
                        int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[0][b].qs)), k1, vld1q_s8(q_q8[0][b].qs + 16));
                        int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[1][b].qs)), k1, vld1q_s8(q_q8[1][b].qs + 16));
                        int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[2][b].qs)), k1, vld1q_s8(q_q8[2][b].qs + 16));
                        int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0, vld1q_s8(q_q8[3][b].qs)), k1, vld1q_s8(q_q8[3][b].qs + 16));
                        const float32x4_t fsum = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)));
                        const float kd = f16_to_f32_fast(kv.d[b]);
                        const float qd[4] = {q_q8[0][b].d * kd, q_q8[1][b].d * kd, q_q8[2][b].d * kd, q_q8[3][b].d * kd};
                        acc = vfmaq_f32(acc, fsum, vld1q_f32(qd));
                    }
                    float sc4[4];
                    vst1q_f32(sc4, acc);
                    scores_buf[0 * kv_len + p] = sc4[0];
                    scores_buf[1 * kv_len + p] = sc4[1];
                    scores_buf[2 * kv_len + p] = sc4[2];
                    scores_buf[3 * kv_len + p] = sc4[3];
                }
            } else {
                for (int p = 0; p < kv_len; ++p) {
                    const CpuMlaKv576Q8Pos & kv = kvc_q8[p];
                    for (int hi = 0; hi < hn; ++hi) {
                        float score = 0.0f;
                        for (int b = 0; b < 18; ++b) {
                            int32x4_t dp = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(kv.qs + b * 32), vld1q_s8(q_q8[hi][b].qs)),
                                                     vld1q_s8(kv.qs + b * 32 + 16), vld1q_s8(q_q8[hi][b].qs + 16));
                            score += (f16_to_f32_fast(kv.d[b]) * q_q8[hi][b].d) * static_cast<float>(vaddvq_s32(dp));
                        }
                        scores_buf[hi * kv_len + p] = score;
                    }
                }
            }

            // Pass 2: Softmax all hn heads, then dequantize each active KV position ONCE across all hn heads!
            float inv_l[4], thresh[4];
            for (int hi = 0; hi < hn; ++hi) {
                float * sp = scores_buf.data() + static_cast<size_t>(hi) * kv_len;
                float m_max = -FLT_MAX;
                for (int p = 0; p < kv_len; ++p) m_max = std::max(m_max, sp[p]);

                const float32x4_t vm = vdupq_n_f32(m_max);
                float32x4_t vsum = vdupq_n_f32(0.0f);
                int p = 0;
                for (; p + 4 <= kv_len; p += 4) {
                    const float32x4_t e = vexpq_f32_fast(vsubq_f32(vld1q_f32(sp + p), vm));
                    vst1q_f32(sp + p, e);
                    vsum = vaddq_f32(vsum, e);
                }
                float l_sum = vaddvq_f32(vsum);
                for (; p < kv_len; ++p) { sp[p] = std::exp(sp[p] - m_max); l_sum += sp[p]; }

                inv_l[hi]  = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                thresh[hi] = 1e-6f * l_sum;
                std::memset(ov[hi], 0, 512 * sizeof(float));
            }

            const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
            const int8x16_t  b8 = vdupq_n_s8(8);
            alignas(64) float vf[512];
            for (int q = 0; q < kv_len; ++q) {
                bool any = false;
                for (int hi = 0; hi < hn; ++hi) {
                    if (scores_buf[static_cast<size_t>(hi) * kv_len + q] > thresh[hi]) { any = true; break; }
                }
                if (!any) continue;
                if (is_q4) {
                    const CpuMlaKv576Q4Pos & kv = kvc_q4[q];
                    for (int b = 0; b < 16; ++b) {
                        const float32x4_t vd = vdupq_n_f32(f16_to_f32_fast(kv.d[b]));
                        const uint8x16_t raw = vld1q_u8(kv.qs + b * 16);
                        const int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), b8);
                        const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), b8);
                        const int16x8_t q0l = vmovl_s8(vget_low_s8(q0)), q0h = vmovl_s8(vget_high_s8(q0));
                        const int16x8_t q1l = vmovl_s8(vget_low_s8(q1)), q1h = vmovl_s8(vget_high_s8(q1));
                        float * gp = vf + b * 32;
                        vst1q_f32(gp +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0l))),  vd));
                        vst1q_f32(gp +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0l))), vd));
                        vst1q_f32(gp +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0h))),  vd));
                        vst1q_f32(gp + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0h))), vd));
                        vst1q_f32(gp + 16, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1l))),  vd));
                        vst1q_f32(gp + 20, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1l))), vd));
                        vst1q_f32(gp + 24, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1h))),  vd));
                        vst1q_f32(gp + 28, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1h))), vd));
                    }
                } else {
                    const CpuMlaKv576Q8Pos & kv = kvc_q8[q];
                    for (int b = 0; b < 16; ++b) {
                        const float32x4_t vd = vdupq_n_f32(f16_to_f32_fast(kv.d[b]));
                        const int8_t * kqs = kv.qs + b * 32;
                        for (int g = 0; g < 2; ++g) {
                            const int8x16_t qv  = vld1q_s8(kqs + g * 16);
                            const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                            const int16x8_t qhi = vmovl_s8(vget_high_s8(qv));
                            float * gp = vf + b * 32 + g * 16;
                            vst1q_f32(gp +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))),  vd));
                            vst1q_f32(gp +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), vd));
                            vst1q_f32(gp +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))),  vd));
                            vst1q_f32(gp + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), vd));
                        }
                    }
                }
                for (int hi = 0; hi < hn; ++hi) {
                    const float w = scores_buf[static_cast<size_t>(hi) * kv_len + q];
                    if (w <= thresh[hi]) continue;
                    const float32x4_t vpw = vdupq_n_f32(w * inv_l[hi]);
                    float * op = ov[hi];
                    for (int d = 0; d < 512; d += 16) {
                        vst1q_f32(op + d +  0, vfmaq_f32(vld1q_f32(op + d +  0), vld1q_f32(vf + d +  0), vpw));
                        vst1q_f32(op + d +  4, vfmaq_f32(vld1q_f32(op + d +  4), vld1q_f32(vf + d +  4), vpw));
                        vst1q_f32(op + d +  8, vfmaq_f32(vld1q_f32(op + d +  8), vld1q_f32(vf + d +  8), vpw));
                        vst1q_f32(op + d + 12, vfmaq_f32(vld1q_f32(op + d + 12), vld1q_f32(vf + d + 12), vpw));
                    }
                }
            }
            for (int hi = 0; hi < hn; ++hi) {
                std::memcpy(out + (static_cast<size_t>(t) * n_heads + hb + hi) * 512, ov[hi], 512 * sizeof(float));
            }
        };

        if (use_dyn) {
            auto dyn_worker = [&](int tid, int) {
                static thread_local std::vector<float> scores_buf;
                alignas(64) float      q_scaled[4][576];
                alignas(64) CpuActQ8_0 q_q8[4][18];
                alignas(64) float      ov[4][512];
                const bool is_little = use_hyb && (tid >= n_big);
                while (true) {
                    uint64_t cur = bounds.load(std::memory_order_relaxed);
                    uint32_t task = 0;
                    bool claimed = false;
                    while (true) {
                        const uint32_t hd = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
                        const uint32_t tl = static_cast<uint32_t>(cur >> 32);
                        if (hd >= tl) break;
                        const uint64_t next = is_little
                            ? (static_cast<uint64_t>(hd + 1) | (static_cast<uint64_t>(tl) << 32))
                            : (static_cast<uint64_t>(hd)     | (static_cast<uint64_t>(tl - 1) << 32));
                        if (bounds.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                            task = is_little ? hd : (tl - 1);
                            claimed = true;
                            break;
                        }
                    }
                    if (!claimed) break;
                    run_task(static_cast<int>(task), scores_buf, q_scaled, q_q8, ov);
                }
            };
            if (use_hyb) pool.parallel_compute(dyn_worker);
            else         pool.parallel(dyn_worker);
        } else {
            pool.parallel_for(n_tasks, [&](int task0, int task1, int) {
                static thread_local std::vector<float> scores_buf;
                alignas(64) float      q_scaled[4][576];
                alignas(64) CpuActQ8_0 q_q8[4][18];
                alignas(64) float      ov[4][512];
                for (int task = task0; task < task1; ++task) {
                    run_task(task, scores_buf, q_scaled, q_q8, ov);
                }
            });
        }
        return;
    }
#endif

    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        alignas(32) float q_scaled[576];
        alignas(32) CpuActQ8_0 q_q8[18];
        alignas(32) float ov[512];

        for (int h = h0; h < h1; ++h) {
            for (int t = 0; t < T; ++t) {
                const int kv_len = pos[t] + 1;
                const float * qsrc = q576 + (static_cast<size_t>(t) * n_heads + h) * 576;
                for (int i = 0; i < 576; ++i) q_scaled[i] = qsrc[i] * scale;

                float m_i = -FLT_MAX;
                float l_i = 0.0f;
                std::memset(ov, 0, sizeof(ov));

                if (kv_type == KvCacheType::Q8_0) {
                    quantize_row_q8_0(q_scaled, q_q8, 576);
                    const auto * kvc = static_cast<const CpuMlaKv576Q8Pos *>(kv_cache);
                    for (int p = 0; p < kv_len; ++p) {
                        const CpuMlaKv576Q8Pos & kv = kvc[p];
                        float score = 0.0f;
                        for (int b = 0; b < 18; ++b) {
                            int32_t dp = 0;
                            const int8_t * ks = kv.qs + b * 32;
                            for (int i = 0; i < 32; ++i) dp += static_cast<int32_t>(ks[i]) * static_cast<int32_t>(q_q8[b].qs[i]);
                            score += (f16_to_f32_fast(kv.d[b]) * q_q8[b].d) * static_cast<float>(dp);
                        }
                        const float m_new = std::max(m_i, score);
                        const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                        const float beta  = std::exp(score - m_new);
                        l_i = l_i * alpha + beta;
                        m_i = m_new;
                        if (alpha != 1.0f) {
                            for (int i = 0; i < 512; ++i) ov[i] *= alpha;
                        }
                        if (beta > 1e-8f) {
                            for (int b = 0; b < 16; ++b) {
                                const float c = beta * f16_to_f32_fast(kv.d[b]);
                                const int8_t * ks = kv.qs + b * 32;
                                float * ob = ov + b * 32;
                                for (int i = 0; i < 32; ++i) ob[i] += c * static_cast<float>(ks[i]);
                            }
                        }
                    }
                } else if (kv_type == KvCacheType::Q4_0) {
                    quantize_row_q8_0(q_scaled, q_q8, 576);
                    const auto * kvc = static_cast<const CpuMlaKv576Q4Pos *>(kv_cache);
                    for (int p = 0; p < kv_len; ++p) {
                        const CpuMlaKv576Q4Pos & kv = kvc[p];
                        float score = 0.0f;
                        for (int b = 0; b < 18; ++b) {
                            int32_t dp = 0;
                            const uint8_t * ks = kv.qs + b * 16;
                            for (int j = 0; j < 16; ++j) {
                                dp += (static_cast<int32_t>(ks[j] & 0x0Fu) - 8) * static_cast<int32_t>(q_q8[b].qs[j]);
                                dp += (static_cast<int32_t>(ks[j] >> 4)    - 8) * static_cast<int32_t>(q_q8[b].qs[j + 16]);
                            }
                            score += (f16_to_f32_fast(kv.d[b]) * q_q8[b].d) * static_cast<float>(dp);
                        }
                        const float m_new = std::max(m_i, score);
                        const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                        const float beta  = std::exp(score - m_new);
                        l_i = l_i * alpha + beta;
                        m_i = m_new;
                        if (alpha != 1.0f) {
                            for (int i = 0; i < 512; ++i) ov[i] *= alpha;
                        }
                        if (beta > 1e-8f) {
                            for (int b = 0; b < 16; ++b) {
                                const float c = beta * f16_to_f32_fast(kv.d[b]);
                                const uint8_t * ks = kv.qs + b * 16;
                                float * ob = ov + b * 32;
                                for (int j = 0; j < 16; ++j) {
                                    ob[j]      += c * static_cast<float>(static_cast<int>(ks[j] & 0x0Fu) - 8);
                                    ob[j + 16] += c * static_cast<float>(static_cast<int>(ks[j] >> 4)    - 8);
                                }
                            }
                        }
                    }
                } else {
                    const auto * kvc = static_cast<const uint16_t *>(kv_cache);
                    for (int p = 0; p < kv_len; ++p) {
                        const uint16_t * kp = kvc + static_cast<size_t>(p) * 576;
                        const float score = dot_f16_row(kp, q_scaled, 576);
                        const float m_new = std::max(m_i, score);
                        const float alpha = (m_i == -FLT_MAX) ? 0.0f : std::exp(m_i - m_new);
                        const float beta  = std::exp(score - m_new);
                        l_i = l_i * alpha + beta;
                        m_i = m_new;
                        for (int i = 0; i < 512; ++i) {
                            ov[i] = ov[i] * alpha + beta * f16_to_f32_fast(kp[i]);
                        }
                    }
                }

                const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
                float * dst = out + (static_cast<size_t>(t) * n_heads + h) * 512;
                for (int i = 0; i < 512; ++i) dst[i] = ov[i] * inv_l;
            }
        }
    });
}

void mla_head_sigmoid_gate(
    float * o_inout, const float * gate,
    int T, int n_heads, int head_dim) {

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const float g = 1.0f / (1.0f + std::exp(-gate[static_cast<size_t>(t) * n_heads + h]));
            float * op = o_inout + (static_cast<size_t>(t) * n_heads + h) * head_dim;
            int d = 0;
#if defined(__AVX2__)
            const __m256 vg = _mm256_set1_ps(g);
            for (; d + 7 < head_dim; d += 8) {
                _mm256_storeu_ps(op + d, _mm256_mul_ps(_mm256_loadu_ps(op + d), vg));
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            const float32x4_t vg = vdupq_n_f32(g);
            for (; d + 4 <= head_dim; d += 4) {
                vst1q_f32(op + d, vmulq_f32(vld1q_f32(op + d), vg));
            }
#endif
            for (; d < head_dim; ++d) op[d] *= g;
        }
    }
}

void moe_route_sigmoid(
    const float * logits, const float * bias,
    int32_t * top_ids, float * top_w,
    int T, int n_expert, int k, bool renorm, float scale,
    int n_expert_groups, int n_group_used) {

    std::vector<float> probs(static_cast<size_t>(n_expert));
    std::vector<float> sel(static_cast<size_t>(n_expert));

    for (int t = 0; t < T; ++t) {
        const float * row = logits + static_cast<size_t>(t) * n_expert;
        for (int e = 0; e < n_expert; ++e) {
            const float p = 1.0f / (1.0f + std::exp(-row[e]));
            probs[e] = p;
            sel[e]   = bias ? (p + bias[e]) : p;
        }

        if (n_expert_groups > 1 && n_group_used > 0 && n_group_used < n_expert_groups) {
            const int n_per_grp = n_expert / n_expert_groups;
            float gscore[64];
            uint8_t gmask[64] = {};
            for (int g = 0; g < n_expert_groups; ++g) {
                const int g0 = g * n_per_grp;
                float t1 = -FLT_MAX, t2 = -FLT_MAX;
                for (int i = 0; i < n_per_grp; ++i) {
                    const float v = sel[g0 + i];
                    if (v > t1) { t2 = t1; t1 = v; }
                    else if (v > t2) { t2 = v; }
                }
                gscore[g] = t1 + t2;
            }
            for (int j = 0; j < n_group_used; ++j) {
                float best_v = -FLT_MAX;
                int best_g = 0;
                for (int g = 0; g < n_expert_groups; ++g) {
                    if (!gmask[g] && gscore[g] > best_v) {
                        best_v = gscore[g];
                        best_g = g;
                    }
                }
                gmask[best_g] = 1;
            }
            for (int e = 0; e < n_expert; ++e) {
                if (!gmask[e / n_per_grp]) sel[e] = -FLT_MAX;
            }
        }

        int32_t * out_i = top_ids + static_cast<size_t>(t) * k;
        float *   out_w = top_w   + static_cast<size_t>(t) * k;
        float w_sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            float best_v = -FLT_MAX;
            int best_e = 0;
            for (int e = 0; e < n_expert; ++e) {
                if (sel[e] > best_v) {
                    best_v = sel[e];
                    best_e = e;
                }
            }
            out_i[j] = best_e;
            out_w[j] = probs[best_e];
            w_sum   += probs[best_e];
            sel[best_e] = -FLT_MAX;
        }

        const float mul = (renorm && w_sum > 0.0f) ? (scale / w_sum) : scale;
        for (int j = 0; j < k; ++j) out_w[j] *= mul;
    }
}

void moe_ffn_with_shared(
    ThreadPool & pool,
    const CpuTensor & gate_exps, const CpuTensor & up_exps, const CpuTensor & down_exps,
    const CpuTensor & gate_shexp, const CpuTensor & up_shexp, const CpuTensor & down_shexp,
    const float * x, const CpuActQ8_0 * xq8, const CpuActQ8_K * xq8k,
    const int32_t * top_ids, const float * top_w,
    float * out,
    int T, int k, float clamp_exp, float clamp_shexp,
    const float * shexp_gate) {

    const int n_embd   = static_cast<int>(gate_exps.ne[0]);
    const int ff_exp   = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_shexp = static_cast<int>(gate_shexp.ne[1]);

    thread_local std::vector<CpuActQ8_0> in_q8_local;
    thread_local std::vector<CpuActQ8_K> in_q8k_local;
    const bool any_in_q8  = needs_act_q8_0(gate_exps.dtype) || needs_act_q8_0(up_exps.dtype) ||
                            needs_act_q8_0(gate_shexp.dtype) || needs_act_q8_0(up_shexp.dtype);
    const bool any_in_q8k = needs_act_q8_k(gate_exps.dtype) || needs_act_q8_k(up_exps.dtype) ||
                            needs_act_q8_k(gate_shexp.dtype) || needs_act_q8_k(up_shexp.dtype);
    if (any_in_q8 && xq8 == nullptr) {
        in_q8_local.resize(static_cast<size_t>(T) * (n_embd >> 5));
        quantize_rows_q8_0(pool, x, in_q8_local.data(), T, n_embd);
        xq8 = in_q8_local.data();
    }
    if (any_in_q8k && xq8k == nullptr) {
        in_q8k_local.resize(static_cast<size_t>(T) * (n_embd >> 8));
        quantize_rows_q8_k(pool, x, in_q8k_local.data(), T, n_embd);
        xq8k = in_q8k_local.data();
    }

    const auto tr_g = dtype_traits(gate_exps.dtype);
    const auto tr_u = dtype_traits(up_exps.dtype);
    const auto tr_d = dtype_traits(down_exps.dtype);
    const size_t gate_exp_bytes = (static_cast<size_t>(n_embd) / tr_g.block_elems) * tr_g.block_bytes * ff_exp;
    const size_t up_exp_bytes   = (static_cast<size_t>(n_embd) / tr_u.block_elems) * tr_u.block_bytes * ff_exp;
    const size_t down_exp_bytes = (static_cast<size_t>(ff_exp) / tr_d.block_elems) * tr_d.block_bytes * n_embd;

    const auto * g_base = static_cast<const uint8_t *>(gate_exps.ptr);
    const auto * u_base = static_cast<const uint8_t *>(up_exps.ptr);
    const auto * d_base = static_cast<const uint8_t *>(down_exps.ptr);

    // ========================================================================
    // Decode path (T == 1):
    //   - Barrier 1: 100% load-balanced across all (k+1) * (ff/32) = 144 blocks!
    //                Computes Gate+Up+SwiGLU and quantizes directly into L1 -> hid_q8_p!
    //   - Barrier 2: Sequential-per-expert stream over thread's [r0, r1) L1-resident out slice!
    // ========================================================================
    if (T == 1) {
        const float sh_scale = (shexp_gate != nullptr) ? (1.0f / (1.0f + std::exp(-shexp_gate[0]))) : 1.0f;
        const int ff_nb_q8  = ff_exp >> 5;
        const int ff_sh_q8  = ff_shexp >> 5;
        const int ff_nb_q8k = ff_exp >> 8;
        const int ff_sh_q8k = ff_shexp >> 8;

        thread_local std::vector<float>      hid_f32;
        thread_local std::vector<CpuActQ8_0> hid_q8;
        thread_local std::vector<CpuActQ8_K> hid_q8k;

        const bool fast_down_q8 = (down_exps.dtype == DType::Q8_0 && down_shexp.dtype == DType::Q8_0);
        if (fast_down_q8 &&
            ((gate_exps.dtype == DType::Q6_K && up_exps.dtype == DType::Q6_K && gate_shexp.dtype == DType::Q6_K && up_shexp.dtype == DType::Q6_K) ||
             (gate_exps.dtype == DType::Q8_0 && up_exps.dtype == DType::Q8_0 && gate_shexp.dtype == DType::Q8_0 && up_shexp.dtype == DType::Q8_0))) {

            const int total_blks = k * ff_nb_q8 + ff_sh_q8;
            hid_q8.resize(static_cast<size_t>(total_blks));
            CpuActQ8_0 * hid_q8_p = hid_q8.data();

            const void * g_ptrs[16];
            const void * u_ptrs[16];
            const BlockQ8_0 * d_ptrs[16];
            for (int s = 0; s < k; ++s) {
                const int e = top_ids[s];
                g_ptrs[s] = g_base + static_cast<size_t>(e) * gate_exp_bytes;
                u_ptrs[s] = u_base + static_cast<size_t>(e) * up_exp_bytes;
                d_ptrs[s] = reinterpret_cast<const BlockQ8_0 *>(d_base + static_cast<size_t>(e) * down_exp_bytes);
            }
            g_ptrs[k] = gate_shexp.ptr;
            u_ptrs[k] = up_shexp.ptr;
            d_ptrs[k] = static_cast<const BlockQ8_0 *>(down_shexp.ptr);

            const bool is_q6k = (gate_exps.dtype == DType::Q6_K);
            const int nb_in   = is_q6k ? (n_embd >> 8) : (n_embd >> 5);

            // Barrier 1: Stream 64-row (2 x 32) chunks sequentially with lock-free dynamic scheduling
            // to absorb heterogeneous big-core clock skew while keeping 100% sequential DRAM streaming.
            const int blk_step       = (ff_nb_q8 >= 4 && (ff_nb_q8 % 2 == 0)) ? 2 : std::max(1, ff_nb_q8);
            const int chunks_per_exp = (ff_nb_q8 + blk_step - 1) / blk_step;
            const int chunks_sh      = (ff_sh_q8 + blk_step - 1) / blk_step;
            const int total_chunks   = k * chunks_per_exp + chunks_sh;

            std::atomic<int> next_gc{0};
            pool.parallel([&](int, int) {
                alignas(64) float gbuf[128];
                alignas(64) float ubuf[128];
                while (true) {
                    const int gc = next_gc.fetch_add(1, std::memory_order_relaxed);
                    if (gc >= total_chunks) break;
                    const bool is_sh = (gc >= k * chunks_per_exp);
                    const int  slot  = is_sh ? k : (gc / chunks_per_exp);
                    const int  lc    = is_sh ? (gc - k * chunks_per_exp) : (gc % chunks_per_exp);
                    const int  slot_ff_blks = is_sh ? ff_sh_q8 : ff_nb_q8;
                    const int  blk0  = lc * blk_step;
                    const int  n_blk = std::min(blk_step, slot_ff_blks - blk0);
                    const int  nrows = n_blk * 32;
                    const int  row0  = blk0 * 32;
                    const int  gb0   = is_sh ? (k * ff_nb_q8 + blk0) : (slot * ff_nb_q8 + blk0);
                    const float cl   = is_sh ? clamp_shexp : clamp_exp;

                    if (is_q6k) {
                        const auto * wgb = static_cast<const BlockQ6_K *>(g_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        const auto * wub = static_cast<const BlockQ6_K *>(u_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        for (int r = 0; r < nrows; ++r) gbuf[r] = dot_q6_k_q8_k_row(wgb + static_cast<size_t>(r) * nb_in, xq8k, nb_in);
                        for (int r = 0; r < nrows; ++r) ubuf[r] = dot_q6_k_q8_k_row(wub + static_cast<size_t>(r) * nb_in, xq8k, nb_in);
                    } else {
                        const auto * wgb = static_cast<const BlockQ8_0 *>(g_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        const auto * wub = static_cast<const BlockQ8_0 *>(u_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        for (int r = 0; r < nrows; r += 4) {
                            dot_q8_0_q8_0_4rows(wgb + (r + 0) * nb_in, wgb + (r + 1) * nb_in, wgb + (r + 2) * nb_in, wgb + (r + 3) * nb_in,
                                                xq8, nb_in, gbuf[r + 0], gbuf[r + 1], gbuf[r + 2], gbuf[r + 3]);
                        }
                        for (int r = 0; r < nrows; r += 4) {
                            dot_q8_0_q8_0_4rows(wub + (r + 0) * nb_in, wub + (r + 1) * nb_in, wub + (r + 2) * nb_in, wub + (r + 3) * nb_in,
                                                xq8, nb_in, ubuf[r + 0], ubuf[r + 1], ubuf[r + 2], ubuf[r + 3]);
                        }
                    }
                    apply_swiglu_vec(gbuf, ubuf, static_cast<size_t>(nrows), cl);
                    quantize_row_q8_0(gbuf, hid_q8_p + gb0, nrows);
                }
            });

            // Barrier 2: Dynamic 128-row work-stealing over n_embd; streams ONE expert sequentially at a time!
            constexpr int kDownChunk = 128;
            std::atomic<int> next_r{0};
            pool.parallel([&](int, int) {
                while (true) {
                    const int r0 = next_r.fetch_add(kDownChunk, std::memory_order_relaxed);
                    if (r0 >= n_embd) break;
                    const int r1 = std::min(n_embd, r0 + kDownChunk);
                    // 1. Shared expert (initializes out[r0..r1))
                    {
                        const BlockQ8_0 * dp = d_ptrs[k];
                        const CpuActQ8_0 * hp = hid_q8_p + static_cast<size_t>(k) * ff_nb_q8;
                        int r = r0;
                        for (; r + 3 < r1; r += 4) {
                            dot_q8_0_q8_0_4rows(dp + (r + 0) * ff_sh_q8, dp + (r + 1) * ff_sh_q8,
                                                dp + (r + 2) * ff_sh_q8, dp + (r + 3) * ff_sh_q8,
                                                hp, ff_sh_q8, out[r + 0], out[r + 1], out[r + 2], out[r + 3]);
                        }
                        for (; r < r1; ++r) out[r] = dot_q8_0_q8_0_row(dp + r * ff_sh_q8, hp, ff_sh_q8);
                        if (shexp_gate != nullptr) scale_inplace(out + r0, static_cast<size_t>(r1 - r0), sh_scale);
                    }
                    // 2. Each routed expert streamed linearly (100% HW prefetcher hit, zero TLB thrashing!)
                    for (int s = 0; s < k; ++s) {
                        const float w = top_w[s];
                        const BlockQ8_0 * dp = d_ptrs[s];
                        const CpuActQ8_0 * hp = hid_q8_p + static_cast<size_t>(s) * ff_nb_q8;
                        int r = r0;
                        for (; r + 3 < r1; r += 4) {
                            float t0, t1, t2, t3;
                            dot_q8_0_q8_0_4rows(dp + (r + 0) * ff_nb_q8, dp + (r + 1) * ff_nb_q8,
                                                dp + (r + 2) * ff_nb_q8, dp + (r + 3) * ff_nb_q8,
                                                hp, ff_nb_q8, t0, t1, t2, t3);
                            out[r + 0] += w * t0; out[r + 1] += w * t1;
                            out[r + 2] += w * t2; out[r + 3] += w * t3;
                        }
                        for (; r < r1; ++r) out[r] += w * dot_q8_0_q8_0_row(dp + r * ff_nb_q8, hp, ff_nb_q8);
                    }
                }
            });
            return;
        }

        // Generic T=1 fallback for other quant combinations (e.g. Q4_K / Q6_K / Q5_0 down on ARM64)
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {
            const bool h_need_q8  = (needs_act_q8_0(down_exps.dtype) || needs_act_q8_0(down_shexp.dtype));
            const bool h_need_q8k = (needs_act_q8_k(down_exps.dtype) || needs_act_q8_k(down_shexp.dtype));
            const int max_ff = std::max(ff_exp, ff_shexp);
            hid_f32.resize(static_cast<size_t>(k + 1) * max_ff);
            if (h_need_q8)  hid_q8.resize(static_cast<size_t>(k) * ff_nb_q8 + ff_sh_q8);
            if (h_need_q8k) hid_q8k.resize(static_cast<size_t>(k) * ff_nb_q8k + ff_sh_q8k);
            float *      hid_f32_p = hid_f32.data();
            CpuActQ8_0 * hid_q8_p  = h_need_q8  ? hid_q8.data()  : nullptr;
            CpuActQ8_K * hid_q8k_p = h_need_q8k ? hid_q8k.data() : nullptr;

            // Barrier 1: Lock-free dynamic 64-row chunks across all (k + 1) experts + parallel in-place quantization!
            constexpr int kChunk = 64;
            const int chunks_exp   = (ff_exp + kChunk - 1) / kChunk;
            const int chunks_sh    = (ff_shexp + kChunk - 1) / kChunk;
            const int total_chunks = k * chunks_exp + chunks_sh;

            const int total_q8k_blks = k * ff_nb_q8k + ff_sh_q8k;
            const bool parallel_q8k  = h_need_q8k && ((ff_exp & 255) == 0) && ((ff_shexp & 255) == 0) && total_q8k_blks <= 256;
            const bool parallel_q8   = h_need_q8  && ((ff_exp & 63) == 0)  && ((ff_shexp & 63) == 0);
            std::atomic<int> q8k_done[256];
            if (parallel_q8k) {
                for (int i = 0; i < total_q8k_blks; ++i) q8k_done[i].store(0, std::memory_order_relaxed);
            }

            std::atomic<int> next_ci{0};
            pool.parallel([&](int, int) {
                alignas(64) float u_tmp[kChunk];
                while (true) {
                    const int ci = next_ci.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= total_chunks) break;
                    const bool is_sh = (ci >= k * chunks_exp);
                    const int  slot  = is_sh ? k : (ci / chunks_exp);
                    const int  r0    = (is_sh ? (ci - k * chunks_exp) : (ci % chunks_exp)) * kChunk;
                    const int  ff    = is_sh ? ff_shexp : ff_exp;
                    const int  r1    = std::min(r0 + kChunk, ff);
                    const float cl   = is_sh ? clamp_shexp : clamp_exp;

                    CpuTensor Wg = is_sh ? gate_shexp : CpuTensor{g_base + static_cast<size_t>(top_ids[slot]) * gate_exp_bytes, gate_exps.dtype, {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, gate_exp_bytes};
                    CpuTensor Wu = is_sh ? up_shexp   : CpuTensor{u_base + static_cast<size_t>(top_ids[slot]) * up_exp_bytes,   up_exps.dtype,   {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, up_exp_bytes};
                    float * hdst = hid_f32_p + static_cast<size_t>(slot) * max_ff;
                    linear_rows_range_direct(Wg, x, xq8, xq8k, hdst, 1, r0, r1);
                    const auto tr_u_loc = dtype_traits(Wu.dtype);
                    const size_t u_row_b = (static_cast<size_t>(n_embd) / tr_u_loc.block_elems) * tr_u_loc.block_bytes;
                    CpuTensor Wu_sub{static_cast<const uint8_t *>(Wu.ptr) + static_cast<size_t>(r0) * u_row_b, Wu.dtype,
                                     {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(r1 - r0), 1, 1}, u_row_b * static_cast<size_t>(r1 - r0)};
                    linear_rows_range_direct(Wu_sub, x, xq8, xq8k, u_tmp, 1, 0, r1 - r0);
                    apply_swiglu_vec(hdst + r0, u_tmp, static_cast<size_t>(r1 - r0), cl);

                    if (parallel_q8) {
                        const size_t q8_off = is_sh ? (static_cast<size_t>(k) * ff_nb_q8 + static_cast<size_t>(r0 >> 5))
                                                    : (static_cast<size_t>(slot) * ff_nb_q8 + static_cast<size_t>(r0 >> 5));
                        quantize_row_q8_0(hdst + r0, hid_q8_p + q8_off, r1 - r0);
                    }
                    if (parallel_q8k) {
                        const int b256 = r0 >> 8;
                        const int q8k_idx = is_sh ? (k * ff_nb_q8k + b256) : (slot * ff_nb_q8k + b256);
                        if (q8k_done[q8k_idx].fetch_add(1, std::memory_order_acq_rel) == 3) {
                            quantize_row_q8_k(hdst + b256 * 256, hid_q8k_p + q8k_idx, 256);
                        }
                    }
                }
            });

            if (h_need_q8 && !parallel_q8) {
                for (int s = 0; s < k; ++s) {
                    quantize_row_q8_0(hid_f32_p + static_cast<size_t>(s) * max_ff, hid_q8_p + static_cast<size_t>(s) * ff_nb_q8, ff_exp);
                }
                quantize_row_q8_0(hid_f32_p + static_cast<size_t>(k) * max_ff, hid_q8_p + static_cast<size_t>(k) * ff_nb_q8, ff_shexp);
            }
            if (h_need_q8k && !parallel_q8k) {
                for (int s = 0; s < k; ++s) {
                    quantize_row_q8_k(hid_f32_p + static_cast<size_t>(s) * max_ff, hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k, ff_exp);
                }
                quantize_row_q8_k(hid_f32_p + static_cast<size_t>(k) * max_ff, hid_q8k_p + static_cast<size_t>(k) * ff_nb_q8k, ff_shexp);
            }

            // Barrier 2: Dynamic 64-row work-stealing over n_embd; sequential per-expert streaming!
            constexpr int kDownChunk = 64;
            const size_t d_row_b = (static_cast<size_t>(ff_exp) / tr_d.block_elems) * tr_d.block_bytes;
            std::atomic<int> next_r{0};
            pool.parallel([&](int, int) {
                alignas(64) float tmp[kDownChunk];
                while (true) {
                    const int r0 = next_r.fetch_add(kDownChunk, std::memory_order_relaxed);
                    if (r0 >= n_embd) break;
                    const int r1 = std::min(n_embd, r0 + kDownChunk);
                    const int rn = r1 - r0;

                    linear_rows_range_direct(down_shexp,
                                             hid_f32_p + static_cast<size_t>(k) * max_ff,
                                             hid_q8_p  ? (hid_q8_p  + static_cast<size_t>(k) * ff_nb_q8)  : nullptr,
                                             hid_q8k_p ? (hid_q8k_p + static_cast<size_t>(k) * ff_nb_q8k) : nullptr,
                                             out, 1, r0, r1);
                    if (shexp_gate != nullptr) scale_inplace(out + r0, static_cast<size_t>(rn), sh_scale);
                    for (int s = 0; s < k; ++s) {
                        const int e = top_ids[s];
                        CpuTensor Wd_sub{d_base + static_cast<size_t>(e) * down_exp_bytes + static_cast<size_t>(r0) * d_row_b,
                                         down_exps.dtype, {static_cast<uint64_t>(ff_exp), static_cast<uint64_t>(rn), 1, 1},
                                         d_row_b * static_cast<size_t>(rn)};
                        linear_rows_range_direct(Wd_sub,
                                                 hid_f32_p + static_cast<size_t>(s) * max_ff,
                                                 hid_q8_p  ? (hid_q8_p  + static_cast<size_t>(s) * ff_nb_q8)  : nullptr,
                                                 hid_q8k_p ? (hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k) : nullptr,
                                                 tmp, 1, 0, rn);
                        const float w = top_w[s];
                        const float32x4_t vw = vdupq_n_f32(w);
                        int j = 0;
                        for (; j + 8 <= rn; j += 8) {
                            vst1q_f32(out + r0 + j + 0, vfmaq_f32(vld1q_f32(out + r0 + j + 0), vld1q_f32(tmp + j + 0), vw));
                            vst1q_f32(out + r0 + j + 4, vfmaq_f32(vld1q_f32(out + r0 + j + 4), vld1q_f32(tmp + j + 4), vw));
                        }
                        for (; j < rn; ++j) out[r0 + j] += w * tmp[j];
                    }
                }
            });
            return;
        }
#endif
        hid_f32.resize(static_cast<size_t>(k) * ff_exp + ff_shexp);
        if (down_exps.dtype == DType::Q8_0 || down_shexp.dtype == DType::Q8_0) {
            hid_q8.resize(static_cast<size_t>(k) * ff_nb_q8 + ff_sh_q8);
        }
        if (down_exps.dtype == DType::Q6_K || down_shexp.dtype == DType::Q6_K) {
            hid_q8k.resize(static_cast<size_t>(k) * ff_nb_q8k + ff_sh_q8k);
        }
        float *      hid_f32_p = hid_f32.data();
        CpuActQ8_0 * hid_q8_p  = hid_q8.empty() ? nullptr : hid_q8.data();
        CpuActQ8_K * hid_q8k_p = hid_q8k.empty() ? nullptr : hid_q8k.data();

        pool.parallel_for(k + 1, [&](int s0, int s1, int) {
            for (int s = s0; s < s1; ++s) {
                const bool is_sh = (s == k);
                const int ff = is_sh ? ff_shexp : ff_exp;
                const float cl = is_sh ? clamp_shexp : clamp_exp;
                float * hdst = is_sh ? (hid_f32_p + static_cast<size_t>(k) * ff_exp)
                                     : (hid_f32_p + static_cast<size_t>(s) * ff_exp);
                CpuTensor Wg = is_sh ? gate_shexp : CpuTensor{g_base + static_cast<size_t>(top_ids[s]) * gate_exp_bytes, gate_exps.dtype, {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, gate_exp_bytes};
                CpuTensor Wu = is_sh ? up_shexp   : CpuTensor{u_base + static_cast<size_t>(top_ids[s]) * up_exp_bytes,   up_exps.dtype,   {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, up_exp_bytes};
                std::vector<float> tmp_u(static_cast<size_t>(ff));
                linear_rows_range_direct(Wg, x, xq8, xq8k, hdst, 1, 0, ff);
                linear_rows_range_direct(Wu, x, xq8, xq8k, tmp_u.data(), 1, 0, ff);
                apply_swiglu_vec(hdst, tmp_u.data(), static_cast<size_t>(ff), cl);
                const DType ddt = is_sh ? down_shexp.dtype : down_exps.dtype;
                if (ddt == DType::Q8_0) {
                    quantize_row_q8_0(hdst, is_sh ? (hid_q8_p + static_cast<size_t>(k) * ff_nb_q8) : (hid_q8_p + static_cast<size_t>(s) * ff_nb_q8), ff);
                } else if (ddt == DType::Q6_K) {
                    quantize_row_q8_k(hdst, is_sh ? (hid_q8k_p + static_cast<size_t>(k) * ff_nb_q8k) : (hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k), ff);
                }
            }
        });

        std::memset(out, 0, static_cast<size_t>(n_embd) * sizeof(float));
        std::vector<float> tmp_out(static_cast<size_t>(n_embd));
        linear(pool, down_shexp, hid_f32_p + static_cast<size_t>(k) * ff_exp,
               hid_q8_p ? (hid_q8_p + static_cast<size_t>(k) * ff_nb_q8) : nullptr,
               hid_q8k_p ? (hid_q8k_p + static_cast<size_t>(k) * ff_nb_q8k) : nullptr, out, 1);
        if (shexp_gate != nullptr) scale_inplace(out, static_cast<size_t>(n_embd), sh_scale);
        for (int s = 0; s < k; ++s) {
            const int e = top_ids[s];
            CpuTensor Wd{d_base + static_cast<size_t>(e) * down_exp_bytes, down_exps.dtype,
                         {static_cast<uint64_t>(ff_exp), static_cast<uint64_t>(n_embd), 1, 1}, down_exp_bytes};
            linear(pool, Wd, hid_f32_p + static_cast<size_t>(s) * ff_exp,
                   hid_q8_p ? (hid_q8_p + static_cast<size_t>(s) * ff_nb_q8) : nullptr,
                   hid_q8k_p ? (hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k) : nullptr, tmp_out.data(), 1);
            const float w = top_w[s];
            for (int r = 0; r < n_embd; ++r) out[r] += w * tmp_out[r];
        }
        return;
    }

    // ========================================================================
    // Prefill path (T >= 2): Expert-grouped L1-tiled FMA GEMM!
    // ========================================================================
    {
        thread_local std::vector<float> sh_g, sh_u;
        sh_g.resize(static_cast<size_t>(T) * ff_shexp);
        sh_u.resize(static_cast<size_t>(T) * ff_shexp);
        float * sh_gp = sh_g.data();
        const float * sh_up = sh_u.data();
        const LinearSpec specs[2] = {{&gate_shexp, sh_gp}, {&up_shexp, sh_u.data()}};
        linear_multi(pool, specs, 2, x, xq8, xq8k, T);
        const size_t sh_tot = static_cast<size_t>(T) * ff_shexp;
        if (sh_tot >= 8192 && pool.num_threads() > 1) {
            pool.parallel_for_blocked(static_cast<int>(sh_tot), 256, [&](int i0, int i1, int) {
                apply_swiglu_vec(sh_gp + i0, sh_up + i0, static_cast<size_t>(i1 - i0), clamp_shexp);
            });
        } else {
            apply_swiglu_vec(sh_gp, sh_up, sh_tot, clamp_shexp);
        }
        linear(pool, down_shexp, sh_gp, nullptr, nullptr, out, T);
        if (shexp_gate != nullptr) {
            for (int t = 0; t < T; ++t) {
                const float sg = 1.0f / (1.0f + std::exp(-shexp_gate[t]));
                scale_inplace(out + static_cast<size_t>(t) * n_embd, static_cast<size_t>(n_embd), sg);
            }
        }
    }

    struct Assign { int32_t tok; float w; };
    thread_local std::vector<std::vector<Assign>> buckets;
    buckets.resize(static_cast<size_t>(n_expert));
    for (int e = 0; e < n_expert; ++e) buckets[e].clear();
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < k; ++j) {
            const int e = top_ids[static_cast<size_t>(t) * k + j];
            buckets[e].push_back({t, top_w[static_cast<size_t>(t) * k + j]});
        }
    }

    std::vector<int> active_exps;
    active_exps.reserve(static_cast<size_t>(n_expert));
    for (int e = 0; e < n_expert; ++e) {
        if (!buckets[e].empty()) active_exps.push_back(e);
    }

    // Sort active experts by descending token count so largest experts start first (LPT load balancing!)
    std::sort(active_exps.begin(), active_exps.end(),
              [&](int a, int b) { return buckets[a].size() > buckets[b].size(); });

    const bool use_hybrid = (T >= 16 && pool.num_compute_threads() > pool.num_threads() && active_exps.size() >= 8);
    const int n_big = pool.num_threads();
    const int nth   = use_hybrid ? pool.num_compute_threads() : n_big;
    const size_t tot = static_cast<size_t>(T) * n_embd;
    thread_local std::vector<std::vector<float>> thr_out;
    thr_out.resize(static_cast<size_t>(nth));
    for (int i = 0; i < nth; ++i) {
        if (thr_out[i].size() < tot) thr_out[i].resize(tot);
    }

    const int K_nb_q8  = n_embd >> 5;
    const int K_nb_q8k = n_embd >> 8;
    const int F_nb_q8  = ff_exp >> 5;
    const int F_nb_q8k = ff_exp >> 8;

    const auto * buckets_p = &buckets;
    auto *       thr_out_p = &thr_out;

    // Lock-free double-ended work queue: big cores claim largest experts from head (LPT),
    // LITTLE cores claim smallest experts from tail (SPT) so slow cores never create end-of-barrier tails.
    const uint32_t n_act = static_cast<uint32_t>(active_exps.size());
    std::atomic<uint64_t> bounds{static_cast<uint64_t>(n_act) << 32};

    auto run_worker = [&](int tid, int) {
        std::vector<float>      g_buf, u_buf, y_buf, xp_buf, hp_buf, gx_f32;
        std::vector<CpuActQ8_0> gx_q8, gh_q8;
        std::vector<CpuActQ8_K> gx_q8k, gh_q8k;
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
        std::vector<ArmQ80Lane4> gx_q8_p4;
        std::vector<ArmQ8KLane4> gx_q8k_p4;
#endif
        float * my_out = (*thr_out_p)[tid].data();
        const bool is_little = use_hybrid && (tid >= n_big);
        std::memset(my_out, 0, tot * sizeof(float));

        while (true) {
            uint64_t cur = bounds.load(std::memory_order_relaxed);
            uint32_t idx = 0;
            bool claimed = false;
            while (true) {
                const uint32_t hd = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
                const uint32_t tl = static_cast<uint32_t>(cur >> 32);
                if (hd >= tl) break;
                const uint64_t next = is_little
                    ? (static_cast<uint64_t>(hd) | (static_cast<uint64_t>(tl - 1) << 32))
                    : (static_cast<uint64_t>(hd + 1) | (static_cast<uint64_t>(tl) << 32));
                if (bounds.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                    idx = is_little ? (tl - 1) : hd;
                    claimed = true;
                    break;
                }
            }
            if (!claimed) break;

            const int e = active_exps[idx];
            const auto & list = (*buckets_p)[e];
            const int nt = static_cast<int>(list.size());

            g_buf.resize(static_cast<size_t>(nt) * ff_exp);
            u_buf.resize(static_cast<size_t>(nt) * ff_exp);
            y_buf.resize(static_cast<size_t>(nt) * n_embd);

            CpuTensor Wg{g_base + static_cast<size_t>(e) * gate_exp_bytes, gate_exps.dtype, {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, gate_exp_bytes};
            CpuTensor Wu{u_base + static_cast<size_t>(e) * up_exp_bytes,   up_exps.dtype,   {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, up_exp_bytes};
            CpuTensor Wd{d_base + static_cast<size_t>(e) * down_exp_bytes, down_exps.dtype, {static_cast<uint64_t>(ff_exp), static_cast<uint64_t>(n_embd), 1, 1}, down_exp_bytes};

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
            if (nt >= 9) {
                const int n_tiles = (nt + 15) >> 4;
                gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                for (int i = 0; i < nt; ++i) {
                    std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                x + static_cast<size_t>(list[i].tok) * n_embd,
                                static_cast<size_t>(n_embd) * sizeof(float));
                }
                xp_buf.resize(static_cast<size_t>(n_tiles) * n_embd * 16);
                pack_x_panels_16(gx_f32.data(), xp_buf.data(), nt, n_embd);
                gemm_fma_rows_range(Wg, xp_buf.data(), g_buf.data(), nt, 0, ff_exp);
                gemm_fma_rows_range(Wu, xp_buf.data(), u_buf.data(), nt, 0, ff_exp);
                apply_swiglu_vec(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp, clamp_exp);

                hp_buf.resize(static_cast<size_t>(n_tiles) * ff_exp * 16);
                pack_x_panels_16(g_buf.data(), hp_buf.data(), nt, ff_exp);
                gemm_fma_rows_range(Wd, hp_buf.data(), y_buf.data(), nt, 0, n_embd);
            } else if (nt >= 7) {
                gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                for (int i = 0; i < nt; ++i) {
                    std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                x + static_cast<size_t>(list[i].tok) * n_embd,
                                static_cast<size_t>(n_embd) * sizeof(float));
                }
                xp_buf.resize(static_cast<size_t>(n_embd) * 8);
                pack_x_panels_8(gx_f32.data(), xp_buf.data(), nt, n_embd);
                gemm_fma_rows_range_8(Wg, xp_buf.data(), g_buf.data(), nt, 0, ff_exp);
                gemm_fma_rows_range_8(Wu, xp_buf.data(), u_buf.data(), nt, 0, ff_exp);
                apply_swiglu_vec(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp, clamp_exp);

                hp_buf.resize(static_cast<size_t>(ff_exp) * 8);
                pack_x_panels_8(g_buf.data(), hp_buf.data(), nt, ff_exp);
                gemm_fma_rows_range_8(Wd, hp_buf.data(), y_buf.data(), nt, 0, n_embd);
            } else
#endif
            {
                const bool need_q8  = (needs_act_q8_0(gate_exps.dtype) || needs_act_q8_0(up_exps.dtype));
                const bool need_q8k = (needs_act_q8_k(gate_exps.dtype) || needs_act_q8_k(up_exps.dtype));
                const bool need_f32 = (!need_q8 && !need_q8k);
                if (need_q8) {
                    gx_q8.resize(static_cast<size_t>(nt) * K_nb_q8);
                    for (int i = 0; i < nt; ++i) {
                        std::memcpy(gx_q8.data() + static_cast<size_t>(i) * K_nb_q8,
                                    xq8 + static_cast<size_t>(list[i].tok) * K_nb_q8,
                                    static_cast<size_t>(K_nb_q8) * sizeof(CpuActQ8_0));
                    }
                }
                if (need_q8k) {
                    gx_q8k.resize(static_cast<size_t>(nt) * K_nb_q8k);
                    for (int i = 0; i < nt; ++i) {
                        std::memcpy(gx_q8k.data() + static_cast<size_t>(i) * K_nb_q8k,
                                    xq8k + static_cast<size_t>(list[i].tok) * K_nb_q8k,
                                    static_cast<size_t>(K_nb_q8k) * sizeof(CpuActQ8_K));
                    }
                }
                if (need_f32) {
                    gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                    for (int i = 0; i < nt; ++i) {
                        std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                    x + static_cast<size_t>(list[i].tok) * n_embd,
                                    static_cast<size_t>(n_embd) * sizeof(float));
                    }
                }

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                const ArmQ80Lane4 * p4_q8  = nullptr;
                const ArmQ8KLane4 * p4_q8k = nullptr;
                if (nt >= 4) {
                    const int nt4 = nt >> 2;
                    if (need_q8) {
                        gx_q8_p4.resize(static_cast<size_t>(nt4) * K_nb_q8);
                        pack_q8_0_lane4_range(gx_q8.data(), gx_q8_p4.data(), 0, nt4, K_nb_q8);
                        p4_q8 = gx_q8_p4.data();
                    }
                    if (need_q8k) {
                        gx_q8k_p4.resize(static_cast<size_t>(nt4) * K_nb_q8k);
                        pack_q8_k_lane4_range(gx_q8k.data(), gx_q8k_p4.data(), 0, nt4, K_nb_q8k);
                        p4_q8k = gx_q8k_p4.data();
                    }
                }
                linear_rows_range_direct(Wg, gx_f32.data(), gx_q8.data(), gx_q8k.data(), g_buf.data(), nt, 0, ff_exp, p4_q8, p4_q8k);
                linear_rows_range_direct(Wu, gx_f32.data(), gx_q8.data(), gx_q8k.data(), u_buf.data(), nt, 0, ff_exp, p4_q8, p4_q8k);
#else
                linear_rows_range_direct(Wg, gx_f32.data(), gx_q8.data(), gx_q8k.data(), g_buf.data(), nt, 0, ff_exp);
                linear_rows_range_direct(Wu, gx_f32.data(), gx_q8.data(), gx_q8k.data(), u_buf.data(), nt, 0, ff_exp);
#endif
                apply_swiglu_vec(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp, clamp_exp);

                const bool d_q8  = needs_act_q8_0(down_exps.dtype);
                const bool d_q8k = needs_act_q8_k(down_exps.dtype);
                if (d_q8) {
                    gh_q8.resize(static_cast<size_t>(nt) * F_nb_q8);
                    for (int i = 0; i < nt; ++i) quantize_row_q8_0(g_buf.data() + static_cast<size_t>(i) * ff_exp, gh_q8.data() + static_cast<size_t>(i) * F_nb_q8, ff_exp);
                } else if (d_q8k) {
                    gh_q8k.resize(static_cast<size_t>(nt) * F_nb_q8k);
                    for (int i = 0; i < nt; ++i) quantize_row_q8_k(g_buf.data() + static_cast<size_t>(i) * ff_exp, gh_q8k.data() + static_cast<size_t>(i) * F_nb_q8k, ff_exp);
                }
                linear_rows_range_direct(Wd, g_buf.data(), gh_q8.data(), gh_q8k.data(), y_buf.data(), nt, 0, n_embd);
            }

            for (int i = 0; i < nt; ++i) {
                const float w = list[i].w;
                float * dst = my_out + static_cast<size_t>(list[i].tok) * n_embd;
                const float * src = y_buf.data() + static_cast<size_t>(i) * n_embd;
                int d = 0;
#if defined(__AVX2__) && defined(__FMA__)
                const __m256 vw = _mm256_set1_ps(w);
                for (; d + 7 < n_embd; d += 8) {
                    _mm256_storeu_ps(dst + d, _mm256_fmadd_ps(vw, _mm256_loadu_ps(src + d), _mm256_loadu_ps(dst + d)));
                }
#elif defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t vw = vdupq_n_f32(w);
                for (; d + 8 <= n_embd; d += 8) {
                    vst1q_f32(dst + d + 0, vfmaq_f32(vld1q_f32(dst + d + 0), vld1q_f32(src + d + 0), vw));
                    vst1q_f32(dst + d + 4, vfmaq_f32(vld1q_f32(dst + d + 4), vld1q_f32(src + d + 4), vw));
                }
#endif
                for (; d < n_embd; ++d) dst[d] += w * src[d];
            }
        }
    };
    if (use_hybrid) pool.parallel_compute(run_worker);
    else            pool.parallel(run_worker);

    if (tot >= 8192 && nth > 1) {
        pool.parallel_for_blocked(static_cast<int>(tot), 256, [&](int i0, int i1, int) {
            const size_t cnt = static_cast<size_t>(i1 - i0);
            for (int ti = 0; ti < nth; ++ti) add_inplace(out + i0, (*thr_out_p)[ti].data() + i0, cnt);
        });
    } else {
        for (int i = 0; i < nth; ++i) {
            add_inplace(out, thr_out[i].data(), tot);
        }
    }
}

// ============================================================================
// Gemma 4 Kernels (Dense+PLE E2B/E4B & MoE 26B-A4B)
// ============================================================================

void gemma4_ple_combine(
    const float * proj, const float * raw_ple, const float * proj_norm_w,
    float * ple_out, int T, int n_layers, int ple_dim, float rms_eps, int n_embd) {

    const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(n_embd));
    const float tok_scale  = std::sqrt(static_cast<float>(ple_dim));
    const float out_scale  = 1.0f / std::sqrt(2.0f);
    const float proj_eps   = rms_eps * static_cast<float>(n_embd); // equivalent to scaling proj by 1/sqrt(n_embd) before rmsnorm

    for (int t = 0; t < T; ++t) {
        for (int il = 0; il < n_layers; ++il) {
            const size_t off = (static_cast<size_t>(t) * n_layers + il) * ple_dim;
            const float * pr = proj + off;
            const float * rr = raw_ple + off;
            float *       yr = ple_out + off;

            float ss = 0.0f;
#if defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            int si = 0;
            for (; si + 8 <= ple_dim; si += 8) {
                const float32x4_t v0 = vld1q_f32(pr + si);
                const float32x4_t v1 = vld1q_f32(pr + si + 4);
                s0 = vfmaq_f32(s0, v0, v0);
                s1 = vfmaq_f32(s1, v1, v1);
            }
            ss = vaddvq_f32(vaddq_f32(s0, s1));
            for (; si < ple_dim; ++si) ss += pr[si] * pr[si];
#else
            for (int i = 0; i < ple_dim; ++i) ss += pr[i] * pr[i];
#endif
            // (pr * inv_sqrt_h) / sqrt(mean((pr * inv_sqrt_h)^2) + rms_eps)
            //   = pr / sqrt(mean(pr^2) + rms_eps * n_embd)
            (void) inv_sqrt_h;
            const float inv_rms = (1.0f / std::sqrt(ss / static_cast<float>(ple_dim) + proj_eps)) * out_scale;
            const float r_scale = tok_scale * out_scale;
            int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256 vinv = _mm256_set1_ps(inv_rms);
            const __m256 vrsc = _mm256_set1_ps(r_scale);
            for (; i + 7 < ple_dim; i += 8) {
                const __m256 p = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(pr + i), vinv), _mm256_loadu_ps(proj_norm_w + i));
                _mm256_storeu_ps(yr + i, _mm256_fmadd_ps(_mm256_loadu_ps(rr + i), vrsc, p));
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            const float32x4_t vinv = vdupq_n_f32(inv_rms);
            const float32x4_t vrsc = vdupq_n_f32(r_scale);
            for (; i + 8 <= ple_dim; i += 8) {
                const float32x4_t p0 = vmulq_f32(vmulq_f32(vld1q_f32(pr + i + 0), vinv), vld1q_f32(proj_norm_w + i + 0));
                const float32x4_t p1 = vmulq_f32(vmulq_f32(vld1q_f32(pr + i + 4), vinv), vld1q_f32(proj_norm_w + i + 4));
                vst1q_f32(yr + i + 0, vfmaq_f32(p0, vld1q_f32(rr + i + 0), vrsc));
                vst1q_f32(yr + i + 4, vfmaq_f32(p1, vld1q_f32(rr + i + 4), vrsc));
            }
#endif
            for (; i < ple_dim; ++i) {
                yr[i] = pr[i] * inv_rms * proj_norm_w[i] + rr[i] * r_scale;
            }
        }
    }
}

void gemma4_qk_norm_rope(
    float * qk, const float * norm_w, const float * rope_freqs, const int32_t * pos,
    int T, int n_head, int head_dim, int rope_dim, float freq_base, float eps) {

    const int half_d = head_dim >> 1;
    const int half_r = rope_dim >> 1;

    // Precompute base inverse frequencies (<= 256 floats)
    float inv_freq[256];
    for (int i = 0; i < half_r; ++i) {
        const float ffac = rope_freqs ? rope_freqs[i] : 1.0f;
        inv_freq[i] = (ffac > 1e20f) ? 0.0f : (std::pow(freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim)) / ffac);
    }

    float cosv[256], sinv[256];
    for (int t = 0; t < T; ++t) {
        const float p = static_cast<float>(pos[t]);
        for (int i = 0; i < half_r; ++i) {
            if (inv_freq[i] == 0.0f) {
                cosv[i] = 1.0f;
                sinv[i] = 0.0f;
            } else {
                const float theta = p * inv_freq[i];
                cosv[i] = std::cos(theta);
                sinv[i] = std::sin(theta);
            }
        }
        for (int h = 0; h < n_head; ++h) {
            float * hp = qk + (static_cast<size_t>(t) * n_head + h) * head_dim;
            float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            for (int i = 0; i < head_dim; i += 16) {
                const __m256 v0 = _mm256_loadu_ps(hp + i);
                const __m256 v1 = _mm256_loadu_ps(hp + i + 8);
                acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            }
            ss = hsum256_ps(_mm256_add_ps(acc0, acc1));
#elif defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            int si = 0;
            for (; si + 8 <= head_dim; si += 8) {
                const float32x4_t v0 = vld1q_f32(hp + si + 0);
                const float32x4_t v1 = vld1q_f32(hp + si + 4);
                acc0 = vfmaq_f32(acc0, v0, v0);
                acc1 = vfmaq_f32(acc1, v1, v1);
            }
            ss = vaddvq_f32(vaddq_f32(acc0, acc1));
            for (; si < head_dim; ++si) ss += hp[si] * hp[si];
#else
            for (int i = 0; i < head_dim; ++i) ss += hp[i] * hp[i];
#endif
            const float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
            int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256 vinv = _mm256_set1_ps(inv_rms);
            if (norm_w) {
                for (; i + 7 < half_r; i += 8) {
                    const __m256 x0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(hp + i),          vinv), _mm256_loadu_ps(norm_w + i));
                    const __m256 x1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(hp + i + half_d), vinv), _mm256_loadu_ps(norm_w + i + half_d));
                    const __m256 cv = _mm256_loadu_ps(cosv + i);
                    const __m256 sv = _mm256_loadu_ps(sinv + i);
                    _mm256_storeu_ps(hp + i,          _mm256_fnmadd_ps(x1, sv, _mm256_mul_ps(x0, cv)));
                    _mm256_storeu_ps(hp + i + half_d, _mm256_fmadd_ps(x0, sv, _mm256_mul_ps(x1, cv)));
                }
                for (; i < half_r; ++i) {
                    const float x0 = hp[i]          * inv_rms * norm_w[i];
                    const float x1 = hp[i + half_d] * inv_rms * norm_w[i + half_d];
                    hp[i]          = x0 * cosv[i] - x1 * sinv[i];
                    hp[i + half_d] = x0 * sinv[i] + x1 * cosv[i];
                }
                for (; i + 7 < half_d; i += 8) {
                    _mm256_storeu_ps(hp + i,          _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(hp + i),          vinv), _mm256_loadu_ps(norm_w + i)));
                    _mm256_storeu_ps(hp + i + half_d, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(hp + i + half_d), vinv), _mm256_loadu_ps(norm_w + i + half_d)));
                }
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            const float32x4_t vinv = vdupq_n_f32(inv_rms);
            if (norm_w) {
                for (; i + 4 <= half_r; i += 4) {
                    const float32x4_t x0 = vmulq_f32(vmulq_f32(vld1q_f32(hp + i),          vinv), vld1q_f32(norm_w + i));
                    const float32x4_t x1 = vmulq_f32(vmulq_f32(vld1q_f32(hp + i + half_d), vinv), vld1q_f32(norm_w + i + half_d));
                    const float32x4_t cv = vld1q_f32(cosv + i);
                    const float32x4_t sv = vld1q_f32(sinv + i);
                    vst1q_f32(hp + i,          vfmsq_f32(vmulq_f32(x0, cv), x1, sv));
                    vst1q_f32(hp + i + half_d, vfmaq_f32(vmulq_f32(x1, cv), x0, sv));
                }
                for (; i < half_r; ++i) {
                    const float x0 = hp[i]          * inv_rms * norm_w[i];
                    const float x1 = hp[i + half_d] * inv_rms * norm_w[i + half_d];
                    hp[i]          = x0 * cosv[i] - x1 * sinv[i];
                    hp[i + half_d] = x0 * sinv[i] + x1 * cosv[i];
                }
                for (; i + 4 <= half_d; i += 4) {
                    vst1q_f32(hp + i,          vmulq_f32(vmulq_f32(vld1q_f32(hp + i),          vinv), vld1q_f32(norm_w + i)));
                    vst1q_f32(hp + i + half_d, vmulq_f32(vmulq_f32(vld1q_f32(hp + i + half_d), vinv), vld1q_f32(norm_w + i + half_d)));
                }
            }
#endif
            for (; i < half_d; ++i) {
                const float x0 = hp[i]          * inv_rms * (norm_w ? norm_w[i]          : 1.0f);
                const float x1 = hp[i + half_d] * inv_rms * (norm_w ? norm_w[i + half_d] : 1.0f);
                if (i < half_r) {
                    hp[i]          = x0 * cosv[i] - x1 * sinv[i];
                    hp[i + half_d] = x0 * sinv[i] + x1 * cosv[i];
                } else {
                    hp[i]          = x0;
                    hp[i + half_d] = x1;
                }
            }
        }
    }
}

void gemma4_vnorm_kv_write(
    const float * k_in, const float * v_in,
    void * k_cache, void * v_cache, KvCacheType kv_type,
    const int32_t * pos, int T, int n_kv, int head_dim, int max_slots, bool is_swa, float eps) {

    const int nb = head_dim >> 5;
    const size_t q8_stride = g4_kv_q8_head_stride(head_dim);
    const size_t q4_stride = g4_kv_q4_head_stride(head_dim);
    alignas(32) float v_norm[512];
    alignas(32) CpuActQ8_0 tmp_q8[16];

    for (int t = 0; t < T; ++t) {
        const int p = pos[t];
        const int slot = is_swa ? (p % max_slots) : p;
        for (int kvh = 0; kvh < n_kv; ++kvh) {
            const float * kr = k_in + (static_cast<size_t>(t) * n_kv + kvh) * head_dim;
            const float * vr = v_in + (static_cast<size_t>(t) * n_kv + kvh) * head_dim;

            float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 vs0 = _mm256_setzero_ps(), vs1 = _mm256_setzero_ps();
            for (int i = 0; i < head_dim; i += 16) {
                const __m256 v0 = _mm256_loadu_ps(vr + i);
                const __m256 v1 = _mm256_loadu_ps(vr + i + 8);
                vs0 = _mm256_fmadd_ps(v0, v0, vs0);
                vs1 = _mm256_fmadd_ps(v1, v1, vs1);
            }
            ss = hsum256_ps(_mm256_add_ps(vs0, vs1));
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
            const __m256 vinv = _mm256_set1_ps(inv);
            for (int i = 0; i < head_dim; i += 8) {
                _mm256_store_ps(v_norm + i, _mm256_mul_ps(_mm256_loadu_ps(vr + i), vinv));
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
            int si = 0;
            for (; si + 8 <= head_dim; si += 8) {
                const float32x4_t v0 = vld1q_f32(vr + si + 0);
                const float32x4_t v1 = vld1q_f32(vr + si + 4);
                vs0 = vfmaq_f32(vs0, v0, v0);
                vs1 = vfmaq_f32(vs1, v1, v1);
            }
            ss = vaddvq_f32(vaddq_f32(vs0, vs1));
            for (; si < head_dim; ++si) ss += vr[si] * vr[si];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
            const float32x4_t vinv = vdupq_n_f32(inv);
            int i = 0;
            for (; i + 8 <= head_dim; i += 8) {
                vst1q_f32(v_norm + i + 0, vmulq_f32(vld1q_f32(vr + i + 0), vinv));
                vst1q_f32(v_norm + i + 4, vmulq_f32(vld1q_f32(vr + i + 4), vinv));
            }
            for (; i < head_dim; ++i) v_norm[i] = vr[i] * inv;
#else
            for (int i = 0; i < head_dim; ++i) ss += vr[i] * vr[i];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
            for (int i = 0; i < head_dim; ++i) v_norm[i] = vr[i] * inv;
#endif

            const size_t h_idx = static_cast<size_t>(slot) * n_kv + kvh;
            if (kv_type == KvCacheType::Q8_0) {
                auto * k_dst = static_cast<uint8_t *>(k_cache) + h_idx * q8_stride;
                auto * v_dst = static_cast<uint8_t *>(v_cache) + h_idx * q8_stride;
                auto * k_qs  = reinterpret_cast<int8_t *>(k_dst);
                auto * k_d   = reinterpret_cast<uint16_t *>(k_dst + head_dim);
                auto * v_qs  = reinterpret_cast<int8_t *>(v_dst);
                auto * v_d   = reinterpret_cast<uint16_t *>(v_dst + head_dim);

                quantize_row_q8_0(kr, tmp_q8, head_dim);
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(k_qs + b * 32, tmp_q8[b].qs, 32);
                    k_d[b] = f32_to_f16_fast(tmp_q8[b].d);
                }
                quantize_row_q8_0(v_norm, tmp_q8, head_dim);
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(v_qs + b * 32, tmp_q8[b].qs, 32);
                    v_d[b] = f32_to_f16_fast(tmp_q8[b].d);
                }
            } else if (kv_type == KvCacheType::Q4_0) {
                auto * k_dst = static_cast<uint8_t *>(k_cache) + h_idx * q4_stride;
                auto * v_dst = static_cast<uint8_t *>(v_cache) + h_idx * q4_stride;
                quantize_vec_q4_0_soa(kr,     k_dst, reinterpret_cast<uint16_t *>(k_dst + (head_dim >> 1)), head_dim);
                quantize_vec_q4_0_soa(v_norm, v_dst, reinterpret_cast<uint16_t *>(v_dst + (head_dim >> 1)), head_dim);
            } else {
                auto * k_dst = static_cast<uint16_t *>(k_cache) + h_idx * head_dim;
                auto * v_dst = static_cast<uint16_t *>(v_cache) + h_idx * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    k_dst[i] = f32_to_f16_fast(kr[i]);
                    v_dst[i] = f32_to_f16_fast(v_norm[i]);
                }
            }
        }
    }
}

void gemma4_gqa_attn(
    ThreadPool & pool,
    const float * q, const void * k_cache, const void * v_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_head, int n_kv, int head_dim, int max_slots, int swa_window) {

    const int gqa = n_head / n_kv;
    const int nb  = head_dim >> 5;
    const size_t q8_stride = g4_kv_q8_head_stride(head_dim);
    const size_t q4_stride = g4_kv_q4_head_stride(head_dim);
    const bool is_swa = (swa_window > 0);

    // Parallelize across all (t, h) pairs (or heads when T==1) so even n_kv=1..2 uses all 4 cores!
    const int group_sz = (T * n_kv >= pool.num_threads()) ? gqa : std::max(1, gqa / std::max(1, pool.num_threads() / std::max(1, T * n_kv)));
    const int groups_per_kv = (gqa + group_sz - 1) / group_sz;
    const int n_work = T * n_kv * groups_per_kv;

    const bool use_dyn = (T >= 4 && pool.num_compute_threads() > 1);
    const bool use_hyb = (T >= 16 && pool.num_compute_threads() > pool.num_threads());
    const int  n_big   = pool.num_threads();
    std::atomic<uint64_t> bounds{static_cast<uint64_t>(static_cast<uint32_t>(n_work)) << 32};

    auto run_work_item = [&](int w,
                             CpuActQ8_0 (&q_q8)[16][16], float (&ov)[16][512],
                             float (&scores)[16][8], float (&betas)[8],
                             float (&kd_cache)[8][16], float (&vd_tmp)[16],
                             float (&m_i)[16], float (&l_i)[16]) {
        constexpr int kMaxGrp = 16;
        constexpr int P_TILE  = 8;
        (void) kMaxGrp;
#if defined(__AVX2__) && defined(__FMA__)
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i m4     = _mm256_set1_epi8(0x0F);
        const __m256i c8     = _mm256_set1_epi8(8);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        const uint8x16_t m4_arm = vdupq_n_u8(0x0Fu);
        const int8x16_t  b8_arm = vdupq_n_s8(8);
#endif
        {
            const int t     = w / (n_kv * groups_per_kv);
            const int rem   = w % (n_kv * groups_per_kv);
            const int kvh   = rem / groups_per_kv;
            const int g_blk = rem % groups_per_kv;
            const int g0    = g_blk * group_sz;
            const int ng    = std::min(group_sz, gqa - g0);
            const int h_base = kvh * gqa + g0;

            const int q_pos = pos[t];
            const int p_min = (is_swa && q_pos - swa_window + 1 > 0) ? (q_pos - swa_window + 1) : 0;
            const int p_end = q_pos + 1;

            for (int gi = 0; gi < ng; ++gi) {
                const float * qsrc = q + (static_cast<size_t>(t) * n_head + (h_base + gi)) * head_dim;
                if (kv_type != KvCacheType::F16) quantize_row_q8_0(qsrc, q_q8[gi], head_dim);
                m_i[gi] = -FLT_MAX;
                l_i[gi] = 0.0f;
                std::memset(ov[gi], 0, static_cast<size_t>(head_dim) * sizeof(float));
            }

            if (kv_type == KvCacheType::Q8_0) {
                const auto * kb = static_cast<const uint8_t *>(k_cache);
                const auto * vb = static_cast<const uint8_t *>(v_cache);
                for (int p0 = p_min; p0 < p_end; p0 += P_TILE) {
                    const int np = std::min(P_TILE, p_end - p0);
                    int slots[P_TILE];
                    // Prefetch next K tile (never touch V cache unless sparse softmax passes!)
#if defined(__AVX2__)
                    if (p0 + P_TILE < p_end) {
                        const int np_next = std::min(P_TILE, p_end - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const int nslot = is_swa ? ((p0 + P_TILE + pi) % max_slots) : (p0 + P_TILE + pi);
                            const char * kpf = reinterpret_cast<const char *>(kb + (static_cast<size_t>(nslot) * n_kv + kvh) * q8_stride);
                            _mm_prefetch(kpf + 0, _MM_HINT_T0);
                            _mm_prefetch(kpf + head_dim, _MM_HINT_T0);
                        }
                    }
#elif defined(__aarch64__)
                    if (p0 + P_TILE < p_end) {
                        const int np_next = std::min(P_TILE, p_end - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const int nslot = is_swa ? ((p0 + P_TILE + pi) % max_slots) : (p0 + P_TILE + pi);
                            const char * kpf = reinterpret_cast<const char *>(kb + (static_cast<size_t>(nslot) * n_kv + kvh) * q8_stride);
                            __builtin_prefetch(kpf + 0, 0, 0);
                            __builtin_prefetch(kpf + head_dim, 0, 0);
                        }
                    }
#endif
                    for (int pi = 0; pi < np; ++pi) {
                        const int slot = is_swa ? ((p0 + pi) % max_slots) : (p0 + pi);
                        slots[pi] = slot;
                        const size_t h_idx = static_cast<size_t>(slot) * n_kv + kvh;
                        const auto * kd = reinterpret_cast<const uint16_t *>(kb + h_idx * q8_stride + head_dim);
#if defined(__AVX2__) && defined(__F16C__)
                        for (int b = 0; b < nb; b += 8) {
                            _mm256_store_ps(kd_cache[pi] + b, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kd + b))));
                        }
#else
                        for (int b = 0; b < nb; ++b) kd_cache[pi][b] = f16_to_f32_fast(kd[b]);
#endif
                    }

                    for (int pi = 0; pi < np; ++pi) {
                        const size_t h_idx = static_cast<size_t>(slots[pi]) * n_kv + kvh;
                        const auto * k_qs = reinterpret_cast<const int8_t *>(kb + h_idx * q8_stride);
#if defined(__AVX2__) && defined(__FMA__)
                        int gi = 0;
                        for (; gi + 3 < ng; gi += 4) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                            for (int b = 0; b < nb; ++b) {
                                const __m256i kv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_qs + b * 32));
                                const __m256i akv = _mm256_abs_epi8(kv0);
                                const float kd = kd_cache[pi][b];
                                const __m256i q0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 0][b].qs));
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q0, kv0)), ones16)), s0);
                                const __m256i q1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 1][b].qs));
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q1, kv0)), ones16)), s1);
                                const __m256i q2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 2][b].qs));
                                s2 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 2][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q2, kv0)), ones16)), s2);
                                const __m256i q3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 3][b].qs));
                                s3 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 3][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q3, kv0)), ones16)), s3);
                            }
                            scores[gi + 0][pi] = hsum256_ps(s0);
                            scores[gi + 1][pi] = hsum256_ps(s1);
                            scores[gi + 2][pi] = hsum256_ps(s2);
                            scores[gi + 3][pi] = hsum256_ps(s3);
                        }
                        for (; gi + 1 < ng; gi += 2) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            for (int b = 0; b < nb; ++b) {
                                const __m256i kv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_qs + b * 32));
                                const __m256i akv = _mm256_abs_epi8(kv0);
                                const float kd = kd_cache[pi][b];
                                const __m256i q0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 0][b].qs));
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q0, kv0)), ones16)), s0);
                                const __m256i q1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 1][b].qs));
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(akv, _mm256_sign_epi8(q1, kv0)), ones16)), s1);
                            }
                            scores[gi + 0][pi] = hsum256_ps(s0);
                            scores[gi + 1][pi] = hsum256_ps(s1);
                        }
                        for (; gi < ng; ++gi) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            for (int b = 0; b < nb; b += 2) {
                                const __m256i kv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_qs + (b + 0) * 32));
                                const __m256i qv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi][b + 0].qs));
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd_cache[pi][b + 0] * q_q8[gi][b + 0].d),
                                                     _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(kv0, qv0, ones16)), s0);
                                const __m256i kv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_qs + (b + 1) * 32));
                                const __m256i qv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi][b + 1].qs));
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd_cache[pi][b + 1] * q_q8[gi][b + 1].d),
                                                     _mm256_cvtepi32_ps(mul_sum_i8_pairs_to_i32(kv1, qv1, ones16)), s1);
                            }
                            scores[gi][pi] = hsum256_ps(_mm256_add_ps(s0, s1));
                        }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                        for (int gi = 0; gi < ng; ++gi) {
                            float32x4_t sv = vdupq_n_f32(0.0f);
                            int b = 0;
                            for (; b + 4 <= nb; b += 4) {
                                int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(k_qs + (b + 0) * 32), vld1q_s8(q_q8[gi][b + 0].qs)),
                                                         vld1q_s8(k_qs + (b + 0) * 32 + 16), vld1q_s8(q_q8[gi][b + 0].qs + 16));
                                int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(k_qs + (b + 1) * 32), vld1q_s8(q_q8[gi][b + 1].qs)),
                                                         vld1q_s8(k_qs + (b + 1) * 32 + 16), vld1q_s8(q_q8[gi][b + 1].qs + 16));
                                int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(k_qs + (b + 2) * 32), vld1q_s8(q_q8[gi][b + 2].qs)),
                                                         vld1q_s8(k_qs + (b + 2) * 32 + 16), vld1q_s8(q_q8[gi][b + 2].qs + 16));
                                int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(k_qs + (b + 3) * 32), vld1q_s8(q_q8[gi][b + 3].qs)),
                                                         vld1q_s8(k_qs + (b + 3) * 32 + 16), vld1q_s8(q_q8[gi][b + 3].qs + 16));
                                const float qd4[4] = {q_q8[gi][b + 0].d, q_q8[gi][b + 1].d, q_q8[gi][b + 2].d, q_q8[gi][b + 3].d};
                                const int32x4_t dp = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
                                sv = vfmaq_f32(sv, vcvtq_f32_s32(dp), vmulq_f32(vld1q_f32(kd_cache[pi] + b), vld1q_f32(qd4)));
                            }
                            float sc = vaddvq_f32(sv);
                            for (; b < nb; ++b) {
                                int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(k_qs + b * 32), vld1q_s8(q_q8[gi][b].qs)),
                                                        vld1q_s8(k_qs + b * 32 + 16), vld1q_s8(q_q8[gi][b].qs + 16));
                                sc += (kd_cache[pi][b] * q_q8[gi][b].d) * static_cast<float>(vaddvq_s32(p));
                            }
                            scores[gi][pi] = sc;
                        }
#else
                        for (int gi = 0; gi < ng; ++gi) {
                            float sc = 0.0f;
                            for (int b = 0; b < nb; ++b) {
                                int32_t dp = 0;
                                const int8_t * ks = k_qs + b * 32;
                                for (int i = 0; i < 32; ++i) dp += static_cast<int32_t>(ks[i]) * static_cast<int32_t>(q_q8[gi][b].qs[i]);
                                sc += (kd_cache[pi][b] * q_q8[gi][b].d) * static_cast<float>(dp);
                            }
                            scores[gi][pi] = sc;
                        }
#endif
                    }

                    for (int gi = 0; gi < ng; ++gi) {
                        float m_tile = -FLT_MAX;
                        for (int pi = 0; pi < np; ++pi) m_tile = std::max(m_tile, scores[gi][pi]);
                        if (m_tile <= m_i[gi] - 20.0f) continue;

                        const float m_new = std::max(m_i[gi], m_tile);
                        const float alpha = (m_i[gi] == -FLT_MAX) ? 0.0f : std::exp(m_i[gi] - m_new);
                        l_i[gi] *= alpha;
                        if (alpha != 1.0f) {
                            scale_inplace(ov[gi], static_cast<size_t>(head_dim), alpha);
                        }

#if defined(__AVX2__) && defined(__FMA__)
                        for (int pi = np; pi < P_TILE; ++pi) scores[gi][pi] = -1e30f;
                        _mm256_store_ps(betas, exp_ps_avx2(_mm256_sub_ps(_mm256_load_ps(scores[gi]), _mm256_set1_ps(m_new))));
#elif defined(__aarch64__) && defined(__ARM_NEON)
                        for (int pi = np; pi < P_TILE; ++pi) scores[gi][pi] = -1e30f;
                        const float32x4_t vmn = vdupq_n_f32(m_new);
                        vst1q_f32(betas + 0, vexpq_f32_fast(vsubq_f32(vld1q_f32(scores[gi] + 0), vmn)));
                        vst1q_f32(betas + 4, vexpq_f32_fast(vsubq_f32(vld1q_f32(scores[gi] + 4), vmn)));
#else
                        for (int pi = 0; pi < np; ++pi) betas[pi] = std::exp(scores[gi][pi] - m_new);
#endif
                        for (int pi = 0; pi < np; ++pi) {
                            if (scores[gi][pi] <= m_new - 18.0f) continue;
                            const float beta = betas[pi];
                            l_i[gi] += beta;
                            const size_t h_idx = static_cast<size_t>(slots[pi]) * n_kv + kvh;
                            const auto * v_qs = reinterpret_cast<const int8_t *>(vb + h_idx * q8_stride);
                            const auto * v_d  = reinterpret_cast<const uint16_t *>(vb + h_idx * q8_stride + head_dim);
#if defined(__AVX2__) && defined(__F16C__)
                            for (int b = 0; b < nb; b += 8) {
                                _mm256_store_ps(vd_tmp + b, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v_d + b))));
                            }
#else
                            for (int b = 0; b < nb; ++b) vd_tmp[b] = f16_to_f32_fast(v_d[b]);
#endif
                            for (int b = 0; b < nb; ++b) {
                                const float c = beta * vd_tmp[b];
                                const int8_t * vs = v_qs + b * 32;
                                float * ob = ov[gi] + b * 32;
#if defined(__AVX2__) && defined(__FMA__)
                                const __m256 vc = _mm256_set1_ps(c);
                                const __m128i lo  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vs));
                                const __m128i hi8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vs + 16));
                                _mm256_store_ps(ob +  0, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_load_ps(ob + 0)));
                                _mm256_store_ps(ob +  8, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_load_ps(ob + 8)));
                                _mm256_store_ps(ob + 16, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi8)), _mm256_load_ps(ob + 16)));
                                _mm256_store_ps(ob + 24, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi8, 8))), _mm256_load_ps(ob + 24)));
#elif defined(__aarch64__) && defined(__ARM_NEON)
                                const float32x4_t vc = vdupq_n_f32(c);
                                for (int g = 0; g < 2; ++g) {
                                    const int8x16_t qv  = vld1q_s8(vs + g * 16);
                                    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                                    const int16x8_t qhi = vmovl_s8(vget_high_s8(qv));
                                    float * gp = ob + g * 16;
                                    vst1q_f32(gp +  0, vfmaq_f32(vld1q_f32(gp +  0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))),  vc));
                                    vst1q_f32(gp +  4, vfmaq_f32(vld1q_f32(gp +  4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), vc));
                                    vst1q_f32(gp +  8, vfmaq_f32(vld1q_f32(gp +  8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))),  vc));
                                    vst1q_f32(gp + 12, vfmaq_f32(vld1q_f32(gp + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), vc));
                                }
#else
                                for (int i = 0; i < 32; ++i) ob[i] += c * static_cast<float>(vs[i]);
#endif
                            }
                        }
                        m_i[gi] = m_new;
                    }
                }
            } else if (kv_type == KvCacheType::Q4_0) {
                const auto * kb = static_cast<const uint8_t *>(k_cache);
                const auto * vb = static_cast<const uint8_t *>(v_cache);
                const int half_hd = head_dim >> 1;
                for (int p0 = p_min; p0 < p_end; p0 += P_TILE) {
                    const int np = std::min(P_TILE, p_end - p0);
                    int slots[P_TILE];
#if defined(__AVX2__)
                    if (p0 + P_TILE < p_end) {
                        const int np_next = std::min(P_TILE, p_end - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const int nslot = is_swa ? ((p0 + P_TILE + pi) % max_slots) : (p0 + P_TILE + pi);
                            const char * kpf = reinterpret_cast<const char *>(kb + (static_cast<size_t>(nslot) * n_kv + kvh) * q4_stride);
                            _mm_prefetch(kpf + 0, _MM_HINT_T0);
                            _mm_prefetch(kpf + half_hd, _MM_HINT_T0);
                        }
                    }
#elif defined(__aarch64__)
                    if (p0 + P_TILE < p_end) {
                        const int np_next = std::min(P_TILE, p_end - (p0 + P_TILE));
                        for (int pi = 0; pi < np_next; ++pi) {
                            const int nslot = is_swa ? ((p0 + P_TILE + pi) % max_slots) : (p0 + P_TILE + pi);
                            const char * kpf = reinterpret_cast<const char *>(kb + (static_cast<size_t>(nslot) * n_kv + kvh) * q4_stride);
                            __builtin_prefetch(kpf + 0, 0, 0);
                            __builtin_prefetch(kpf + half_hd, 0, 0);
                        }
                    }
#endif
                    for (int pi = 0; pi < np; ++pi) {
                        const int slot = is_swa ? ((p0 + pi) % max_slots) : (p0 + pi);
                        slots[pi] = slot;
                        const size_t h_idx = static_cast<size_t>(slot) * n_kv + kvh;
                        const auto * kd = reinterpret_cast<const uint16_t *>(kb + h_idx * q4_stride + half_hd);
#if defined(__AVX2__) && defined(__F16C__)
                        for (int b = 0; b < nb; b += 8) {
                            _mm256_store_ps(kd_cache[pi] + b, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kd + b))));
                        }
#else
                        for (int b = 0; b < nb; ++b) kd_cache[pi][b] = f16_to_f32_fast(kd[b]);
#endif
                    }

                    for (int pi = 0; pi < np; ++pi) {
                        const size_t h_idx = static_cast<size_t>(slots[pi]) * n_kv + kvh;
                        const uint8_t * k_qs = kb + h_idx * q4_stride;
#if defined(__AVX2__) && defined(__FMA__)
                        int gi = 0;
                        for (; gi + 3 < ng; gi += 4) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
                            for (int b = 0; b < nb; ++b) {
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_qs + b * 16));
                                const __m256i qk  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                                const float kd = kd_cache[pi][b];
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 0][b].qs))), ones16)), s0);
                                b0 += kd * q_q8[gi + 0][b].s;
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 1][b].qs))), ones16)), s1);
                                b1 += kd * q_q8[gi + 1][b].s;
                                s2 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 2][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 2][b].qs))), ones16)), s2);
                                b2 += kd * q_q8[gi + 2][b].s;
                                s3 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 3][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 3][b].qs))), ones16)), s3);
                                b3 += kd * q_q8[gi + 3][b].s;
                            }
                            scores[gi + 0][pi] = hsum256_ps(s0) - 8.0f * b0;
                            scores[gi + 1][pi] = hsum256_ps(s1) - 8.0f * b1;
                            scores[gi + 2][pi] = hsum256_ps(s2) - 8.0f * b2;
                            scores[gi + 3][pi] = hsum256_ps(s3) - 8.0f * b3;
                        }
                        for (; gi + 1 < ng; gi += 2) {
                            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                            float b0 = 0.0f, b1 = 0.0f;
                            for (int b = 0; b < nb; ++b) {
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_qs + b * 16));
                                const __m256i qk  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                                const float kd = kd_cache[pi][b];
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 0][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 0][b].qs))), ones16)), s0);
                                b0 += kd * q_q8[gi + 0][b].s;
                                s1 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi + 1][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi + 1][b].qs))), ones16)), s1);
                                b1 += kd * q_q8[gi + 1][b].s;
                            }
                            scores[gi + 0][pi] = hsum256_ps(s0) - 8.0f * b0;
                            scores[gi + 1][pi] = hsum256_ps(s1) - 8.0f * b1;
                        }
                        for (; gi < ng; ++gi) {
                            __m256 s0 = _mm256_setzero_ps();
                            float sb = 0.0f;
                            for (int b = 0; b < nb; ++b) {
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_qs + b * 16));
                                const __m256i qk  = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4);
                                const float kd = kd_cache[pi][b];
                                s0 = _mm256_fmadd_ps(_mm256_set1_ps(kd * q_q8[gi][b].d),
                                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qk, _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[gi][b].qs))), ones16)), s0);
                                sb += kd * q_q8[gi][b].s;
                            }
                            scores[gi][pi] = hsum256_ps(s0) - 8.0f * sb;
                        }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                        float32x4_t sv[kMaxGrp];
                        float32x4_t sbv[kMaxGrp];
                        for (int gi = 0; gi < ng; ++gi) { sv[gi] = vdupq_n_f32(0.0f); sbv[gi] = vdupq_n_f32(0.0f); }
                        int b = 0;
                        for (; b + 4 <= nb; b += 4) {
                            const uint8x16_t r0 = vld1q_u8(k_qs + (b + 0) * 16), r1 = vld1q_u8(k_qs + (b + 1) * 16);
                            const uint8x16_t r2 = vld1q_u8(k_qs + (b + 2) * 16), r3 = vld1q_u8(k_qs + (b + 3) * 16);
                            const int8x16_t k0a = vreinterpretq_s8_u8(vandq_u8(r0, m4_arm)), k0b = vreinterpretq_s8_u8(vshrq_n_u8(r0, 4));
                            const int8x16_t k1a = vreinterpretq_s8_u8(vandq_u8(r1, m4_arm)), k1b = vreinterpretq_s8_u8(vshrq_n_u8(r1, 4));
                            const int8x16_t k2a = vreinterpretq_s8_u8(vandq_u8(r2, m4_arm)), k2b = vreinterpretq_s8_u8(vshrq_n_u8(r2, 4));
                            const int8x16_t k3a = vreinterpretq_s8_u8(vandq_u8(r3, m4_arm)), k3b = vreinterpretq_s8_u8(vshrq_n_u8(r3, 4));
                            const float32x4_t vkd = vld1q_f32(kd_cache[pi] + b);
                            for (int gi = 0; gi < ng; ++gi) {
                                int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k0a, vld1q_s8(q_q8[gi][b + 0].qs)), k0b, vld1q_s8(q_q8[gi][b + 0].qs + 16));
                                int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k1a, vld1q_s8(q_q8[gi][b + 1].qs)), k1b, vld1q_s8(q_q8[gi][b + 1].qs + 16));
                                int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k2a, vld1q_s8(q_q8[gi][b + 2].qs)), k2b, vld1q_s8(q_q8[gi][b + 2].qs + 16));
                                int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), k3a, vld1q_s8(q_q8[gi][b + 3].qs)), k3b, vld1q_s8(q_q8[gi][b + 3].qs + 16));
                                const int32x4_t dp = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
                                const float qd4[4] = {q_q8[gi][b + 0].d, q_q8[gi][b + 1].d, q_q8[gi][b + 2].d, q_q8[gi][b + 3].d};
                                const float qs4[4] = {q_q8[gi][b + 0].s, q_q8[gi][b + 1].s, q_q8[gi][b + 2].s, q_q8[gi][b + 3].s};
                                sv[gi]  = vfmaq_f32(sv[gi],  vcvtq_f32_s32(dp), vmulq_f32(vkd, vld1q_f32(qd4)));
                                sbv[gi] = vfmaq_f32(sbv[gi], vkd, vld1q_f32(qs4));
                            }
                        }
                        for (int gi = 0; gi < ng; ++gi) {
                            scores[gi][pi] = vaddvq_f32(vfmsq_f32(sv[gi], sbv[gi], vdupq_n_f32(8.0f)));
                        }
                        for (; b < nb; ++b) {
                            const uint8x16_t raw = vld1q_u8(k_qs + b * 16);
                            const int8x16_t ka = vreinterpretq_s8_u8(vandq_u8(raw, m4_arm));
                            const int8x16_t kb_v = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                            const float kd = kd_cache[pi][b];
                            for (int gi = 0; gi < ng; ++gi) {
                                int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), ka, vld1q_s8(q_q8[gi][b].qs)), kb_v, vld1q_s8(q_q8[gi][b].qs + 16));
                                scores[gi][pi] += (kd * q_q8[gi][b].d) * static_cast<float>(vaddvq_s32(p)) - 8.0f * kd * q_q8[gi][b].s;
                            }
                        }
#else
                        for (int gi = 0; gi < ng; ++gi) {
                            float sc = 0.0f;
                            for (int b = 0; b < nb; ++b) {
                                int32_t dp = 0;
                                const uint8_t * ks = k_qs + b * 16;
                                for (int j = 0; j < 16; ++j) {
                                    dp += (static_cast<int32_t>(ks[j] & 0x0Fu) - 8) * static_cast<int32_t>(q_q8[gi][b].qs[j]);
                                    dp += (static_cast<int32_t>(ks[j] >> 4)    - 8) * static_cast<int32_t>(q_q8[gi][b].qs[j + 16]);
                                }
                                sc += (kd_cache[pi][b] * q_q8[gi][b].d) * static_cast<float>(dp);
                            }
                            scores[gi][pi] = sc;
                        }
#endif
                    }

                    for (int gi = 0; gi < ng; ++gi) {
                        float m_tile = -FLT_MAX;
                        for (int pi = 0; pi < np; ++pi) m_tile = std::max(m_tile, scores[gi][pi]);
                        if (m_tile <= m_i[gi] - 20.0f) continue;

                        const float m_new = std::max(m_i[gi], m_tile);
                        const float alpha = (m_i[gi] == -FLT_MAX) ? 0.0f : std::exp(m_i[gi] - m_new);
                        l_i[gi] *= alpha;
                        if (alpha != 1.0f) {
                            scale_inplace(ov[gi], static_cast<size_t>(head_dim), alpha);
                        }

#if defined(__AVX2__) && defined(__FMA__)
                        for (int pi = np; pi < P_TILE; ++pi) scores[gi][pi] = -1e30f;
                        _mm256_store_ps(betas, exp_ps_avx2(_mm256_sub_ps(_mm256_load_ps(scores[gi]), _mm256_set1_ps(m_new))));
#elif defined(__aarch64__) && defined(__ARM_NEON)
                        for (int pi = np; pi < P_TILE; ++pi) scores[gi][pi] = -1e30f;
                        const float32x4_t vmn = vdupq_n_f32(m_new);
                        vst1q_f32(betas + 0, vexpq_f32_fast(vsubq_f32(vld1q_f32(scores[gi] + 0), vmn)));
                        vst1q_f32(betas + 4, vexpq_f32_fast(vsubq_f32(vld1q_f32(scores[gi] + 4), vmn)));
#else
                        for (int pi = 0; pi < np; ++pi) betas[pi] = std::exp(scores[gi][pi] - m_new);
#endif
                        for (int pi = 0; pi < np; ++pi) {
                            if (scores[gi][pi] <= m_new - 18.0f) continue;
                            const float beta = betas[pi];
                            l_i[gi] += beta;
                            const size_t h_idx = static_cast<size_t>(slots[pi]) * n_kv + kvh;
                            const uint8_t * v_qs = vb + h_idx * q4_stride;
                            const auto * v_d     = reinterpret_cast<const uint16_t *>(vb + h_idx * q4_stride + half_hd);
#if defined(__AVX2__) && defined(__F16C__)
                            for (int b = 0; b < nb; b += 8) {
                                _mm256_store_ps(vd_tmp + b, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v_d + b))));
                            }
#else
                            for (int b = 0; b < nb; ++b) vd_tmp[b] = f16_to_f32_fast(v_d[b]);
#endif
                            for (int b = 0; b < nb; ++b) {
                                const float c = beta * vd_tmp[b];
                                const uint8_t * vs = v_qs + b * 16;
                                float * ob = ov[gi] + b * 32;
#if defined(__AVX2__) && defined(__FMA__)
                                const __m256 vc = _mm256_set1_ps(c);
                                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vs));
                                const __m256i q8  = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1), m4), c8);
                                const __m128i lo  = _mm256_castsi256_si128(q8);
                                const __m128i hi8 = _mm256_extracti128_si256(q8, 1);
                                _mm256_store_ps(ob +  0, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_load_ps(ob + 0)));
                                _mm256_store_ps(ob +  8, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_load_ps(ob + 8)));
                                _mm256_store_ps(ob + 16, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi8)), _mm256_load_ps(ob + 16)));
                                _mm256_store_ps(ob + 24, _mm256_fmadd_ps(vc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi8, 8))), _mm256_load_ps(ob + 24)));
#elif defined(__aarch64__) && defined(__ARM_NEON)
                                const float32x4_t vc = vdupq_n_f32(c);
                                const uint8x16_t raw = vld1q_u8(vs);
                                const int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4_arm)), b8_arm);
                                const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), b8_arm);
                                const int16x8_t q0l = vmovl_s8(vget_low_s8(q0)), q0h = vmovl_s8(vget_high_s8(q0));
                                const int16x8_t q1l = vmovl_s8(vget_low_s8(q1)), q1h = vmovl_s8(vget_high_s8(q1));
                                vst1q_f32(ob +  0, vfmaq_f32(vld1q_f32(ob +  0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0l))),  vc));
                                vst1q_f32(ob +  4, vfmaq_f32(vld1q_f32(ob +  4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0l))), vc));
                                vst1q_f32(ob +  8, vfmaq_f32(vld1q_f32(ob +  8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0h))),  vc));
                                vst1q_f32(ob + 12, vfmaq_f32(vld1q_f32(ob + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0h))), vc));
                                vst1q_f32(ob + 16, vfmaq_f32(vld1q_f32(ob + 16), vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1l))),  vc));
                                vst1q_f32(ob + 20, vfmaq_f32(vld1q_f32(ob + 20), vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1l))), vc));
                                vst1q_f32(ob + 24, vfmaq_f32(vld1q_f32(ob + 24), vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1h))),  vc));
                                vst1q_f32(ob + 28, vfmaq_f32(vld1q_f32(ob + 28), vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1h))), vc));
#else
                                for (int j = 0; j < 16; ++j) {
                                    ob[j]      += c * static_cast<float>(static_cast<int>(vs[j] & 0x0Fu) - 8);
                                    ob[j + 16] += c * static_cast<float>(static_cast<int>(vs[j] >> 4)    - 8);
                                }
#endif
                            }
                        }
                        m_i[gi] = m_new;
                    }
                }
            } else {
                const auto * kb = static_cast<const uint16_t *>(k_cache);
                const auto * vb = static_cast<const uint16_t *>(v_cache);
                for (int p = p_min; p < p_end; ++p) {
                    const int slot = is_swa ? (p % max_slots) : p;
                    const size_t h_idx = static_cast<size_t>(slot) * n_kv + kvh;
                    const uint16_t * kp = kb + h_idx * head_dim;
                    const uint16_t * vp = vb + h_idx * head_dim;
                    for (int gi = 0; gi < ng; ++gi) {
                        const float * qsrc = q + (static_cast<size_t>(t) * n_head + (h_base + gi)) * head_dim;
                        const float sc = dot_f16_row(kp, qsrc, head_dim);
                        const float m_new = std::max(m_i[gi], sc);
                        const float alpha = (m_i[gi] == -FLT_MAX) ? 0.0f : std::exp(m_i[gi] - m_new);
                        const float beta  = std::exp(sc - m_new);
                        l_i[gi] = l_i[gi] * alpha + beta;
                        m_i[gi] = m_new;
                        for (int i = 0; i < head_dim; ++i) ov[gi][i] = ov[gi][i] * alpha + beta * f16_to_f32_fast(vp[i]);
                    }
                }
            }

            for (int gi = 0; gi < ng; ++gi) {
                const float inv_l = (l_i[gi] > 0.0f) ? (1.0f / l_i[gi]) : 0.0f;
                float * dst = out + (static_cast<size_t>(t) * n_head + (h_base + gi)) * head_dim;
                int i = 0;
#if defined(__AVX2__)
                const __m256 vil = _mm256_set1_ps(inv_l);
                for (; i + 7 < head_dim; i += 8) _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(ov[gi] + i), vil));
#elif defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t vil = vdupq_n_f32(inv_l);
                for (; i + 8 <= head_dim; i += 8) {
                    vst1q_f32(dst + i + 0, vmulq_f32(vld1q_f32(ov[gi] + i + 0), vil));
                    vst1q_f32(dst + i + 4, vmulq_f32(vld1q_f32(ov[gi] + i + 4), vil));
                }
#endif
                for (; i < head_dim; ++i) dst[i] = ov[gi][i] * inv_l;
            }
        }
    };

    if (use_dyn) {
        auto dyn_worker = [&](int tid, int) {
            alignas(32) CpuActQ8_0 q_q8[16][16];
            alignas(32) float ov[16][512];
            alignas(32) float scores[16][8];
            alignas(32) float betas[8];
            alignas(32) float kd_cache[8][16];
            alignas(32) float vd_tmp[16];
            float m_i[16], l_i[16];
            const bool is_little = use_hyb && (tid >= n_big);
            while (true) {
                uint64_t cur = bounds.load(std::memory_order_relaxed);
                uint32_t w = 0;
                bool claimed = false;
                while (true) {
                    const uint32_t hd = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
                    const uint32_t tl = static_cast<uint32_t>(cur >> 32);
                    if (hd >= tl) break;
                    const uint64_t next = is_little
                        ? (static_cast<uint64_t>(hd + 1) | (static_cast<uint64_t>(tl) << 32))
                        : (static_cast<uint64_t>(hd)     | (static_cast<uint64_t>(tl - 1) << 32));
                    if (bounds.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                        w = is_little ? hd : (tl - 1);
                        claimed = true;
                        break;
                    }
                }
                if (!claimed) break;
                run_work_item(static_cast<int>(w), q_q8, ov, scores, betas, kd_cache, vd_tmp, m_i, l_i);
            }
        };
        if (use_hyb) pool.parallel_compute(dyn_worker);
        else         pool.parallel(dyn_worker);
    } else {
        pool.parallel_for(n_work, [&](int w0, int w1, int) {
            alignas(32) CpuActQ8_0 q_q8[16][16];
            alignas(32) float ov[16][512];
            alignas(32) float scores[16][8];
            alignas(32) float betas[8];
            alignas(32) float kd_cache[8][16];
            alignas(32) float vd_tmp[16];
            float m_i[16], l_i[16];
            for (int w = w0; w < w1; ++w) {
                run_work_item(w, q_q8, ov, scores, betas, kd_cache, vd_tmp, m_i, l_i);
            }
        });
    }
}

void moe_route_softmax(
    const float * logits, int32_t * top_ids, float * top_w,
    int T, int n_expert, int k, bool renorm, float scale) {

    std::vector<float> probs(static_cast<size_t>(n_expert));
    for (int t = 0; t < T; ++t) {
        const float * row = logits + static_cast<size_t>(t) * n_expert;
        float mx = -FLT_MAX;
        for (int e = 0; e < n_expert; ++e) mx = std::max(mx, row[e]);
        float sum = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            probs[e] = std::exp(row[e] - mx);
            sum += probs[e];
        }
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int e = 0; e < n_expert; ++e) probs[e] *= inv;

        int32_t * out_i = top_ids + static_cast<size_t>(t) * k;
        float *   out_w = top_w   + static_cast<size_t>(t) * k;
        float w_sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            float best_v = -FLT_MAX;
            int best_e = 0;
            for (int e = 0; e < n_expert; ++e) {
                if (probs[e] > best_v) { best_v = probs[e]; best_e = e; }
            }
            out_i[j] = best_e;
            out_w[j] = best_v;
            w_sum += best_v;
            probs[best_e] = -FLT_MAX;
        }
        const float denom = std::max(w_sum, 6.103515625e-5f);
        const float mul = renorm ? (scale / denom) : scale;
        for (int j = 0; j < k; ++j) out_w[j] *= mul;
    }
}

void gemma4_moe_ffn(
    ThreadPool & pool,
    const CpuTensor & gate_up_exps,
    const CpuTensor & gate_exps, const CpuTensor & up_exps,
    const CpuTensor & down_exps, const float * down_exps_scale,
    const float * x, const CpuActQ8_0 * xq8, const CpuActQ8_K * xq8k,
    const int32_t * top_ids, const float * top_w,
    float * out, int T, int k) {

    const bool fused_gu = gate_up_exps.valid();
    const int n_embd   = static_cast<int>(fused_gu ? gate_up_exps.ne[0] : gate_exps.ne[0]);
    const int ff_exp   = static_cast<int>(fused_gu ? (gate_up_exps.ne[1] / 2) : gate_exps.ne[1]);
    const int n_expert = static_cast<int>(fused_gu ? gate_up_exps.ne[2] : gate_exps.ne[2]);
    const DType gu_dt  = fused_gu ? gate_up_exps.dtype : gate_exps.dtype;

    const auto tr_gu = dtype_traits(gu_dt);
    const auto tr_d  = dtype_traits(down_exps.dtype);
    const size_t gu_row_bytes   = (static_cast<size_t>(n_embd) / tr_gu.block_elems) * tr_gu.block_bytes;
    const size_t gu_half_bytes  = gu_row_bytes * ff_exp;
    const size_t gu_exp_bytes   = fused_gu ? (gu_half_bytes * 2) : gu_half_bytes;
    const size_t down_row_bytes = (static_cast<size_t>(ff_exp) / tr_d.block_elems) * tr_d.block_bytes;
    const size_t down_exp_bytes = down_row_bytes * n_embd;

    const auto * gu_base = static_cast<const uint8_t *>(fused_gu ? gate_up_exps.ptr : gate_exps.ptr);
    const auto * u_base  = static_cast<const uint8_t *>(fused_gu ? gate_up_exps.ptr : up_exps.ptr);
    const auto * d_base  = static_cast<const uint8_t *>(down_exps.ptr);

    auto get_gu_ptrs = [&](int e, const uint8_t *& g_ptr, const uint8_t *& u_ptr) {
        if (fused_gu) {
            g_ptr = gu_base + static_cast<size_t>(e) * gu_exp_bytes;
            u_ptr = g_ptr + gu_half_bytes;
        } else {
            g_ptr = gu_base + static_cast<size_t>(e) * gu_half_bytes;
            u_ptr = u_base  + static_cast<size_t>(e) * gu_half_bytes;
        }
    };

    thread_local std::vector<CpuActQ8_0> local_q8;
    thread_local std::vector<CpuActQ8_K> local_q8k;
    if (T < 12) {
        if (needs_act_q8_0(gu_dt) && xq8 == nullptr) {
            local_q8.resize(static_cast<size_t>(T) * (n_embd >> 5));
            quantize_rows_q8_0(pool, x, local_q8.data(), T, n_embd);
            xq8 = local_q8.data();
        }
        if (needs_act_q8_k(gu_dt) && xq8k == nullptr) {
            local_q8k.resize(static_cast<size_t>(T) * (n_embd >> 8));
            quantize_rows_q8_k(pool, x, local_q8k.data(), T, n_embd);
            xq8k = local_q8k.data();
        }
    }

    // ========================================================================
    // Decode path (T == 1): 2-barrier load-balanced parallel MoE
    // ========================================================================
    if (T == 1) {
        const int ff_nb_q8  = ff_exp >> 5;
        const int ff_nb_q8k = ff_exp >> 8;
        const bool down_uses_q8  = needs_act_q8_0(down_exps.dtype);
        const bool down_uses_q8k = needs_act_q8_k(down_exps.dtype);

        thread_local std::vector<float>      hid_f32;
        thread_local std::vector<CpuActQ8_0> hid_q8;
        thread_local std::vector<CpuActQ8_K> hid_q8k;

        hid_f32.resize(static_cast<size_t>(k) * ff_exp);
        if (down_uses_q8)  hid_q8.resize(static_cast<size_t>(k) * ff_nb_q8);
        if (down_uses_q8k) hid_q8k.resize(static_cast<size_t>(k) * ff_nb_q8k);
        float *      hid_f32_p = hid_f32.data();
        CpuActQ8_0 * hid_q8_p  = down_uses_q8  ? hid_q8.data()  : nullptr;
        CpuActQ8_K * hid_q8k_p = down_uses_q8k ? hid_q8k.data() : nullptr;

        // Barrier 1: Split each active expert into 64-row chunks (e.g. 704 -> 11 chunks x 8 experts = 88 chunks!)
        // Keeps all 4 cores 100% load-balanced!
        constexpr int kChunkRows = 64;
        const int chunks_per_exp = (ff_exp + kChunkRows - 1) / kChunkRows;
        const int total_chunks   = k * chunks_per_exp;

        pool.parallel_for(total_chunks, [&](int c0, int c1, int) {
            alignas(32) float gbuf[kChunkRows];
            alignas(32) float ubuf[kChunkRows];
            for (int gc = c0; gc < c1; ++gc) {
                const int s    = gc / chunks_per_exp;
                const int lc   = gc % chunks_per_exp;
                const int r0   = lc * kChunkRows;
                const int r1   = std::min(ff_exp, r0 + kChunkRows);
                const int nr   = r1 - r0;
                const int e    = top_ids[s];

                const uint8_t * gp = nullptr;
                const uint8_t * up = nullptr;
                get_gu_ptrs(e, gp, up);

                float * gdst = down_uses_q8 ? gbuf : (hid_f32_p + static_cast<size_t>(s) * ff_exp + r0);
                if (gu_dt == DType::IQ2_S) {
                    const int nb_in = n_embd >> 8;
                    const auto * wgb = reinterpret_cast<const BlockIQ2_S *>(gp) + static_cast<size_t>(r0) * nb_in;
                    const auto * wub = reinterpret_cast<const BlockIQ2_S *>(up) + static_cast<size_t>(r0) * nb_in;
                    for (int ri = 0; ri < nr; ++ri) gdst[ri] = dot_iq2_s_q8_k_row(wgb + static_cast<size_t>(ri) * nb_in, xq8k, nb_in);
                    for (int ri = 0; ri < nr; ++ri) ubuf[ri] = dot_iq2_s_q8_k_row(wub + static_cast<size_t>(ri) * nb_in, xq8k, nb_in);
                } else {
                    CpuTensor Wg_sub{gp + static_cast<size_t>(r0) * gu_row_bytes, gu_dt,
                                     {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(nr), 1, 1},
                                     static_cast<size_t>(nr) * gu_row_bytes};
                    CpuTensor Wu_sub{up + static_cast<size_t>(r0) * gu_row_bytes, gu_dt,
                                     {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(nr), 1, 1},
                                     static_cast<size_t>(nr) * gu_row_bytes};
                    linear_rows_range_direct(Wg_sub, x, xq8, xq8k, gdst, 1, 0, nr);
                    linear_rows_range_direct(Wu_sub, x, xq8, xq8k, ubuf, 1, 0, nr);
                }
                geglu_tanh_mul(gdst, ubuf, static_cast<size_t>(nr));
                if (down_uses_q8) {
                    quantize_row_q8_0(gdst, hid_q8_p + static_cast<size_t>(s) * ff_nb_q8 + (r0 >> 5), nr);
                }
            }
        });

        if (down_uses_q8k) {
            for (int s = 0; s < k; ++s) {
                quantize_row_q8_k(hid_f32_p + static_cast<size_t>(s) * ff_exp,
                                  hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k, ff_exp);
            }
        }

        // Precompute effective weight = top_w[s] * down_exps_scale[e]
        float eff_w[32];
        const uint8_t * d_ptrs[32];
        for (int s = 0; s < k; ++s) {
            const int e = top_ids[s];
            eff_w[s]  = top_w[s] * (down_exps_scale ? down_exps_scale[e] : 1.0f);
            d_ptrs[s] = d_base + static_cast<size_t>(e) * down_exp_bytes;
        }

        // Barrier 2: Stream one expert sequentially at a time over thread's L1-resident out[r0..r1)!
        pool.parallel_for_blocked(n_embd, 4, [&](int r0, int r1, int) {
            std::memset(out + r0, 0, static_cast<size_t>(r1 - r0) * sizeof(float));
            if (down_exps.dtype == DType::IQ4_NL) {
                for (int s = 0; s < k; ++s) {
                    const float w = eff_w[s];
                    const auto * dp = reinterpret_cast<const BlockIQ4_NL *>(d_ptrs[s]);
                    const CpuActQ8_0 * hp = hid_q8_p + static_cast<size_t>(s) * ff_nb_q8;
                    int r = r0;
                    for (; r + 3 < r1; r += 4) {
                        float t0, t1, t2, t3;
                        dot_iq4_nl_q8_0_4rows(dp + static_cast<size_t>(r + 0) * ff_nb_q8,
                                              dp + static_cast<size_t>(r + 1) * ff_nb_q8,
                                              dp + static_cast<size_t>(r + 2) * ff_nb_q8,
                                              dp + static_cast<size_t>(r + 3) * ff_nb_q8,
                                              hp, ff_nb_q8, t0, t1, t2, t3);
                        out[r + 0] += w * t0; out[r + 1] += w * t1;
                        out[r + 2] += w * t2; out[r + 3] += w * t3;
                    }
                    for (; r < r1; ++r) out[r] += w * dot_iq4_nl_q8_0_row(dp + static_cast<size_t>(r) * ff_nb_q8, hp, ff_nb_q8);
                }
            } else {
                std::vector<float> tmp(static_cast<size_t>(r1 - r0));
                for (int s = 0; s < k; ++s) {
                    const float w = eff_w[s];
                    CpuTensor Wd_sub{d_ptrs[s] + static_cast<size_t>(r0) * down_row_bytes, down_exps.dtype,
                                     {static_cast<uint64_t>(ff_exp), static_cast<uint64_t>(r1 - r0), 1, 1},
                                     static_cast<size_t>(r1 - r0) * down_row_bytes};
                    linear_rows_range_direct(Wd_sub, hid_f32_p + static_cast<size_t>(s) * ff_exp,
                                             hid_q8_p ? (hid_q8_p + static_cast<size_t>(s) * ff_nb_q8) : nullptr,
                                             hid_q8k_p ? (hid_q8k_p + static_cast<size_t>(s) * ff_nb_q8k) : nullptr,
                                             tmp.data(), 1, 0, r1 - r0);
                    const int rn = r1 - r0;
                    int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                    const float32x4_t vw = vdupq_n_f32(w);
                    for (; i + 8 <= rn; i += 8) {
                        vst1q_f32(out + r0 + i + 0, vfmaq_f32(vld1q_f32(out + r0 + i + 0), vld1q_f32(tmp.data() + i + 0), vw));
                        vst1q_f32(out + r0 + i + 4, vfmaq_f32(vld1q_f32(out + r0 + i + 4), vld1q_f32(tmp.data() + i + 4), vw));
                    }
#endif
                    for (; i < rn; ++i) out[r0 + i] += w * tmp[i];
                }
            }
        });
        return;
    }

    // ========================================================================
    // Prefill path (T >= 2): Expert-grouped dynamic LPT work queue!
    // ========================================================================
    struct Assign { int32_t tok; float w; };
    thread_local std::vector<std::vector<Assign>> buckets;
    buckets.resize(static_cast<size_t>(n_expert));
    for (int e = 0; e < n_expert; ++e) buckets[e].clear();
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < k; ++j) {
            const int e = top_ids[static_cast<size_t>(t) * k + j];
            const float ew = top_w[static_cast<size_t>(t) * k + j] * (down_exps_scale ? down_exps_scale[e] : 1.0f);
            buckets[e].push_back({t, ew});
        }
    }

    std::vector<int> active_exps;
    active_exps.reserve(static_cast<size_t>(n_expert));
    for (int e = 0; e < n_expert; ++e) {
        if (!buckets[e].empty()) active_exps.push_back(e);
    }
    std::sort(active_exps.begin(), active_exps.end(),
              [&](int a, int b) { return buckets[a].size() > buckets[b].size(); });

    const bool use_hybrid = (T >= 16 && pool.num_compute_threads() > pool.num_threads() && active_exps.size() >= 8);
    const int n_big = pool.num_threads();
    const int nth   = use_hybrid ? pool.num_compute_threads() : n_big;
    const size_t tot = static_cast<size_t>(T) * n_embd;
    thread_local std::vector<std::vector<float>> thr_out;
    thr_out.resize(static_cast<size_t>(nth));
    for (int i = 0; i < nth; ++i) {
        if (thr_out[i].size() < tot) thr_out[i].resize(tot);
    }

    const int K_nb_q8  = n_embd >> 5;
    const int K_nb_q8k = n_embd >> 8;
    const int F_nb_q8  = ff_exp >> 5;
    const int F_nb_q8k = ff_exp >> 8;

    // Pre-quantize input activations ONCE in parallel if any bucket uses Q8_0/Q8_K (always on ARM64 dotprod,
    // and whenever any small bucket nt < 5 exists on AVX2+FMA)!
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const bool has_small_bucket = true;
#else
    bool has_small_bucket = false;
    for (int e : active_exps) {
        if (buckets[e].size() < 5) { has_small_bucket = true; break; }
    }
#endif
    if (has_small_bucket) {
        if (needs_act_q8_0(gu_dt) && xq8 == nullptr) {
            local_q8.resize(static_cast<size_t>(T) * K_nb_q8);
            quantize_rows_q8_0(pool, x, local_q8.data(), T, n_embd);
            xq8 = local_q8.data();
        }
        if (needs_act_q8_k(gu_dt) && xq8k == nullptr) {
            local_q8k.resize(static_cast<size_t>(T) * K_nb_q8k);
            quantize_rows_q8_k(pool, x, local_q8k.data(), T, n_embd);
            xq8k = local_q8k.data();
        }
    }

    const auto * buckets_p = &buckets;
    auto *       thr_out_p = &thr_out;
    const uint32_t n_act = static_cast<uint32_t>(active_exps.size());
    std::atomic<uint64_t> bounds{static_cast<uint64_t>(n_act) << 32};

    auto run_worker = [&](int tid, int) {
        std::vector<float>      g_buf, u_buf, y_buf, xp_buf, hp_buf, gx_f32;
        std::vector<CpuActQ8_0> gx_q8, gh_q8;
        std::vector<CpuActQ8_K> gx_q8k, gh_q8k;
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
        std::vector<ArmQ80Lane4> gx_q8_p4;
        std::vector<ArmQ8KLane4> gx_q8k_p4;
#endif
        float * my_out = (*thr_out_p)[tid].data();
        const bool is_little = use_hybrid && (tid >= n_big);
        std::memset(my_out, 0, tot * sizeof(float));

        while (true) {
            uint64_t cur = bounds.load(std::memory_order_relaxed);
            uint32_t idx = 0;
            bool claimed = false;
            while (true) {
                const uint32_t hd = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
                const uint32_t tl = static_cast<uint32_t>(cur >> 32);
                if (hd >= tl) break;
                const uint64_t next = is_little
                    ? (static_cast<uint64_t>(hd) | (static_cast<uint64_t>(tl - 1) << 32))
                    : (static_cast<uint64_t>(hd + 1) | (static_cast<uint64_t>(tl) << 32));
                if (bounds.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                    idx = is_little ? (tl - 1) : hd;
                    claimed = true;
                    break;
                }
            }
            if (!claimed) break;

            const int e = active_exps[idx];
            const auto & list = (*buckets_p)[e];
            const int nt = static_cast<int>(list.size());

            g_buf.resize(static_cast<size_t>(nt) * ff_exp);
            u_buf.resize(static_cast<size_t>(nt) * ff_exp);
            y_buf.resize(static_cast<size_t>(nt) * n_embd);

            const uint8_t * gp = nullptr;
            const uint8_t * up = nullptr;
            get_gu_ptrs(e, gp, up);
            CpuTensor Wg{gp, gu_dt, {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, gu_half_bytes};
            CpuTensor Wu{up, gu_dt, {static_cast<uint64_t>(n_embd), static_cast<uint64_t>(ff_exp), 1, 1}, gu_half_bytes};
            CpuTensor Wd{d_base + static_cast<size_t>(e) * down_exp_bytes, down_exps.dtype,
                         {static_cast<uint64_t>(ff_exp), static_cast<uint64_t>(n_embd), 1, 1}, down_exp_bytes};

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
            if (nt >= 9) {
                const int n_tiles = (nt + 15) >> 4;
                gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                for (int i = 0; i < nt; ++i) {
                    std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                x + static_cast<size_t>(list[i].tok) * n_embd,
                                static_cast<size_t>(n_embd) * sizeof(float));
                }
                xp_buf.resize(static_cast<size_t>(n_tiles) * n_embd * 16);
                pack_x_panels_16(gx_f32.data(), xp_buf.data(), nt, n_embd);
                gemm_fma_rows_range(Wg, xp_buf.data(), g_buf.data(), nt, 0, ff_exp);
                gemm_fma_rows_range(Wu, xp_buf.data(), u_buf.data(), nt, 0, ff_exp);
                geglu_tanh_mul(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp);

                hp_buf.resize(static_cast<size_t>(n_tiles) * ff_exp * 16);
                pack_x_panels_16(g_buf.data(), hp_buf.data(), nt, ff_exp);
                gemm_fma_rows_range(Wd, hp_buf.data(), y_buf.data(), nt, 0, n_embd);
            } else if (nt >= 5) {
                gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                for (int i = 0; i < nt; ++i) {
                    std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                x + static_cast<size_t>(list[i].tok) * n_embd,
                                static_cast<size_t>(n_embd) * sizeof(float));
                }
                xp_buf.resize(static_cast<size_t>(n_embd) * 8);
                pack_x_panels_8(gx_f32.data(), xp_buf.data(), nt, n_embd);
                gemm_fma_rows_range_8(Wg, xp_buf.data(), g_buf.data(), nt, 0, ff_exp);
                gemm_fma_rows_range_8(Wu, xp_buf.data(), u_buf.data(), nt, 0, ff_exp);
                geglu_tanh_mul(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp);

                hp_buf.resize(static_cast<size_t>(ff_exp) * 8);
                pack_x_panels_8(g_buf.data(), hp_buf.data(), nt, ff_exp);
                gemm_fma_rows_range_8(Wd, hp_buf.data(), y_buf.data(), nt, 0, n_embd);
            } else
#endif
            {
                const bool gu_q8  = needs_act_q8_0(gu_dt);
                const bool gu_q8k = needs_act_q8_k(gu_dt);
                const bool need_f32 = (!gu_q8 && !gu_q8k) || (gu_q8 && xq8 == nullptr) || (gu_q8k && xq8k == nullptr);
                if (need_f32) {
                    gx_f32.resize(static_cast<size_t>(nt) * n_embd);
                    for (int i = 0; i < nt; ++i) {
                        std::memcpy(gx_f32.data() + static_cast<size_t>(i) * n_embd,
                                    x + static_cast<size_t>(list[i].tok) * n_embd,
                                    static_cast<size_t>(n_embd) * sizeof(float));
                    }
                }
                if (gu_q8) {
                    gx_q8.resize(static_cast<size_t>(nt) * K_nb_q8);
                    if (xq8 != nullptr) {
                        for (int i = 0; i < nt; ++i) {
                            std::memcpy(gx_q8.data() + static_cast<size_t>(i) * K_nb_q8,
                                        xq8 + static_cast<size_t>(list[i].tok) * K_nb_q8,
                                        static_cast<size_t>(K_nb_q8) * sizeof(CpuActQ8_0));
                        }
                    } else {
                        for (int i = 0; i < nt; ++i) quantize_row_q8_0(gx_f32.data() + static_cast<size_t>(i) * n_embd, gx_q8.data() + static_cast<size_t>(i) * K_nb_q8, n_embd);
                    }
                }
                if (gu_q8k) {
                    gx_q8k.resize(static_cast<size_t>(nt) * K_nb_q8k);
                    if (xq8k != nullptr) {
                        for (int i = 0; i < nt; ++i) {
                            std::memcpy(gx_q8k.data() + static_cast<size_t>(i) * K_nb_q8k,
                                        xq8k + static_cast<size_t>(list[i].tok) * K_nb_q8k,
                                        static_cast<size_t>(K_nb_q8k) * sizeof(CpuActQ8_K));
                        }
                    } else {
                        for (int i = 0; i < nt; ++i) quantize_row_q8_k(gx_f32.data() + static_cast<size_t>(i) * n_embd, gx_q8k.data() + static_cast<size_t>(i) * K_nb_q8k, n_embd);
                    }
                }

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
                const ArmQ80Lane4 * p4_q8  = nullptr;
                const ArmQ8KLane4 * p4_q8k = nullptr;
                if (nt >= 4) {
                    const int nt4 = nt >> 2;
                    if (gu_q8) {
                        gx_q8_p4.resize(static_cast<size_t>(nt4) * K_nb_q8);
                        pack_q8_0_lane4_range(gx_q8.data(), gx_q8_p4.data(), 0, nt4, K_nb_q8);
                        p4_q8 = gx_q8_p4.data();
                    }
                    if (gu_q8k) {
                        gx_q8k_p4.resize(static_cast<size_t>(nt4) * K_nb_q8k);
                        pack_q8_k_lane4_range(gx_q8k.data(), gx_q8k_p4.data(), 0, nt4, K_nb_q8k);
                        p4_q8k = gx_q8k_p4.data();
                    }
                }
                linear_rows_range_direct(Wg, gx_f32.data(), gx_q8.empty() ? nullptr : gx_q8.data(), gx_q8k.empty() ? nullptr : gx_q8k.data(), g_buf.data(), nt, 0, ff_exp, p4_q8, p4_q8k);
                linear_rows_range_direct(Wu, gx_f32.data(), gx_q8.empty() ? nullptr : gx_q8.data(), gx_q8k.empty() ? nullptr : gx_q8k.data(), u_buf.data(), nt, 0, ff_exp, p4_q8, p4_q8k);
#else
                linear_rows_range_direct(Wg, gx_f32.data(), gx_q8.empty() ? nullptr : gx_q8.data(), gx_q8k.empty() ? nullptr : gx_q8k.data(), g_buf.data(), nt, 0, ff_exp);
                linear_rows_range_direct(Wu, gx_f32.data(), gx_q8.empty() ? nullptr : gx_q8.data(), gx_q8k.empty() ? nullptr : gx_q8k.data(), u_buf.data(), nt, 0, ff_exp);
#endif
                geglu_tanh_mul(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp);

                if (needs_act_q8_0(down_exps.dtype)) {
                    gh_q8.resize(static_cast<size_t>(nt) * F_nb_q8);
                    for (int i = 0; i < nt; ++i) quantize_row_q8_0(g_buf.data() + static_cast<size_t>(i) * ff_exp, gh_q8.data() + static_cast<size_t>(i) * F_nb_q8, ff_exp);
                } else if (needs_act_q8_k(down_exps.dtype)) {
                    gh_q8k.resize(static_cast<size_t>(nt) * F_nb_q8k);
                    for (int i = 0; i < nt; ++i) quantize_row_q8_k(g_buf.data() + static_cast<size_t>(i) * ff_exp, gh_q8k.data() + static_cast<size_t>(i) * F_nb_q8k, ff_exp);
                }
                linear_rows_range_direct(Wd, g_buf.data(), gh_q8.empty() ? nullptr : gh_q8.data(), gh_q8k.empty() ? nullptr : gh_q8k.data(), y_buf.data(), nt, 0, n_embd);
            }

            for (int i = 0; i < nt; ++i) {
                const float w = list[i].w;
                float * dst = my_out + static_cast<size_t>(list[i].tok) * n_embd;
                const float * src = y_buf.data() + static_cast<size_t>(i) * n_embd;
                int d = 0;
#if defined(__AVX2__) && defined(__FMA__)
                const __m256 vw = _mm256_set1_ps(w);
                for (; d + 7 < n_embd; d += 8) {
                    _mm256_storeu_ps(dst + d, _mm256_fmadd_ps(vw, _mm256_loadu_ps(src + d), _mm256_loadu_ps(dst + d)));
                }
#elif defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t vw = vdupq_n_f32(w);
                for (; d + 8 <= n_embd; d += 8) {
                    vst1q_f32(dst + d + 0, vfmaq_f32(vld1q_f32(dst + d + 0), vld1q_f32(src + d + 0), vw));
                    vst1q_f32(dst + d + 4, vfmaq_f32(vld1q_f32(dst + d + 4), vld1q_f32(src + d + 4), vw));
                }
#endif
                for (; d < n_embd; ++d) dst[d] += w * src[d];
            }
        }
    };
    if (use_hybrid) pool.parallel_compute(run_worker);
    else            pool.parallel(run_worker);

    if (tot >= 8192 && nth > 1) {
        pool.parallel_for_blocked(static_cast<int>(tot), 256, [&](int i0, int i1, int) {
            const size_t cnt = static_cast<size_t>(i1 - i0);
            std::memcpy(out + i0, (*thr_out_p)[0].data() + i0, cnt * sizeof(float));
            for (int ti = 1; ti < nth; ++ti) add_inplace(out + i0, (*thr_out_p)[ti].data() + i0, cnt);
        });
    } else {
        std::memcpy(out, thr_out[0].data(), tot * sizeof(float));
        for (int i = 1; i < nth; ++i) add_inplace(out, thr_out[i].data(), tot);
    }
}

int32_t argmax_f32(const float * logits, int n_vocab) {
    int32_t best_i = 0;
    float   best_v = -FLT_MAX;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_i = i;
        }
    }
    return best_i;
}

} // namespace rookery::cpu
