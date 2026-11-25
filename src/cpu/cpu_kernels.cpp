#include "cpu/cpu_kernels.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
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
#else
    return f16_to_f32_scalar(h);
#endif
}

inline uint16_t f32_to_f16_fast(float f) {
#if defined(__F16C__)
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
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

static inline __m256i mul_sum_i8_pairs_to_i32(__m256i w, __m256i x, __m256i ones16) {
    const __m256i aw  = _mm256_abs_epi8(w);
    const __m256i sx  = _mm256_sign_epi8(x, w);
    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    return _mm256_madd_epi16(p16, ones16);
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
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                float * out = dst + b * 32;
#if defined(__AVX2__)
                const __m256 vd = _mm256_set1_ps(d);
                const int8_t * qs = blk[b].qs;
                for (int g = 0; g < 32; g += 8) {
                    const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs + g));
                    _mm256_storeu_ps(out + g, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8))));
                }
#else
                for (int i = 0; i < 32; ++i) out[i] = d * static_cast<float>(blk[b].qs[i]);
#endif
            }
            return;
        }
        case DType::Q6_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ6_K *>(row_ptr) + b0;
#if defined(__AVX2__)
            const __m256i m4   = _mm256_set1_epi8(0x0F);
            const __m256i m2   = _mm256_set1_epi8(0x03);
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
                const float d = f16_to_f32_fast(blk[b].d);
                for (int h = 0; h < 2; ++h) {
                    const uint8_t * ql = blk[b].ql + h * 64;
                    const uint8_t * qh = blk[b].qh + h * 32;
                    const int8_t *  sc = blk[b].scales + h * 8;
                    float * out = dst + b * 256 + h * 128;

                    const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
                    const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
                    const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

                    const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4)), c32);
                    const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4)), c32);
                    const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4)), c32);
                    const __m256i q4 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4)), c32);

                    store_q32(q1, d * sc[0], d * sc[1], out + 0);
                    store_q32(q2, d * sc[2], d * sc[3], out + 32);
                    store_q32(q3, d * sc[4], d * sc[5], out + 64);
                    store_q32(q4, d * sc[6], d * sc[7], out + 96);
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
            for (int b = 0; b < nb; ++b) {
                const float d = f16_to_f32_fast(blk[b].d);
                float * out = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    out[j]      = d * static_cast<float>(static_cast<int>(blk[b].qs[j] & 0xFu) - 8);
                    out[j + 16] = d * static_cast<float>(static_cast<int>(blk[b].qs[j] >> 4) - 8);
                }
            }
            return;
        }
        case DType::Q5_0: {
            const int b0 = k0 >> 5;
            const int nb = kc >> 5;
            const auto * blk = static_cast<const BlockQ5_0 *>(row_ptr) + b0;
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
            return;
        }
        case DType::Q4_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ4_K *>(row_ptr) + b0;
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
            return;
        }
        case DType::Q5_K: {
            const int b0 = k0 >> 8;
            const int nb = kc >> 8;
            const auto * blk = static_cast<const BlockQ5_K *>(row_ptr) + b0;
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
            return;
        }
        default:
            throw std::runtime_error(std::string("CPU dequant_slice_to_f32 unsupported dtype: ") + dtype_traits(dt).name);
    }
}

// ============================================================================
// Haswell/AVX2+FMA L1-Tiled Dual-FMA SGEMM Microkernels (Prefill T >= 12)
// Uses true BLIS contiguous [n_tiles][K][16] panel packing (zero L1 set aliasing!)
// On AVX2 without VNNI, dual-port FP32 FMA achieves ~30 FLOPs/cy/core (~2x INT8)!
// ============================================================================
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)

// Pack X[T, K] into contiguous tile panels:
//   tile tj (covering tokens [tj*16 .. tj*16 + 15]) occupies xp + tj * K * 16,
//   where entry (k, j) is at offset k * 16 + j (100% contiguous 64B cache lines per k!).
inline void pack_x_panels_16_range(const float * x, float * xp, int T, int K, int tj0, int tj1) {
    for (int tj = tj0; tj < tj1; ++tj) {
        const int t0 = tj * 16;
        const int nt = std::min(16, T - t0);
        float * dst_tile = xp + static_cast<size_t>(tj) * K * 16;
        if (nt == 16) {
            for (int k = 0; k < K; ++k) {
                float * dk = dst_tile + static_cast<size_t>(k) * 16;
                for (int j = 0; j < 16; ++j) {
                    dk[j] = x[static_cast<size_t>(t0 + j) * K + k];
                }
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

    int k = 0;
    for (; k + 1 < KC; k += 2) {
        const float * xk0 = xp + static_cast<size_t>(k) * 16;
        const __m256 x0 = _mm256_loadu_ps(xk0 + 0);
        const __m256 x1 = _mm256_loadu_ps(xk0 + 8);
        __m256 w;
        w = _mm256_broadcast_ss(w0 + k); c00 = _mm256_fmadd_ps(w, x0, c00); c01 = _mm256_fmadd_ps(w, x1, c01);
        w = _mm256_broadcast_ss(w1 + k); c10 = _mm256_fmadd_ps(w, x0, c10); c11 = _mm256_fmadd_ps(w, x1, c11);
        w = _mm256_broadcast_ss(w2 + k); c20 = _mm256_fmadd_ps(w, x0, c20); c21 = _mm256_fmadd_ps(w, x1, c21);
        w = _mm256_broadcast_ss(w3 + k); c30 = _mm256_fmadd_ps(w, x0, c30); c31 = _mm256_fmadd_ps(w, x1, c31);
        w = _mm256_broadcast_ss(w4 + k); c40 = _mm256_fmadd_ps(w, x0, c40); c41 = _mm256_fmadd_ps(w, x1, c41);
        w = _mm256_broadcast_ss(w5 + k); c50 = _mm256_fmadd_ps(w, x0, c50); c51 = _mm256_fmadd_ps(w, x1, c51);

        const __m256 x2 = _mm256_loadu_ps(xk0 + 16);
        const __m256 x3 = _mm256_loadu_ps(xk0 + 24);
        w = _mm256_broadcast_ss(w0 + k + 1); c00 = _mm256_fmadd_ps(w, x2, c00); c01 = _mm256_fmadd_ps(w, x3, c01);
        w = _mm256_broadcast_ss(w1 + k + 1); c10 = _mm256_fmadd_ps(w, x2, c10); c11 = _mm256_fmadd_ps(w, x3, c11);
        w = _mm256_broadcast_ss(w2 + k + 1); c20 = _mm256_fmadd_ps(w, x2, c20); c21 = _mm256_fmadd_ps(w, x3, c21);
        w = _mm256_broadcast_ss(w3 + k + 1); c30 = _mm256_fmadd_ps(w, x2, c30); c31 = _mm256_fmadd_ps(w, x3, c31);
        w = _mm256_broadcast_ss(w4 + k + 1); c40 = _mm256_fmadd_ps(w, x2, c40); c41 = _mm256_fmadd_ps(w, x3, c41);
        w = _mm256_broadcast_ss(w5 + k + 1); c50 = _mm256_fmadd_ps(w, x2, c50); c51 = _mm256_fmadd_ps(w, x3, c51);
    }
    for (; k < KC; ++k) {
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

    constexpr int KC = 256;
    alignas(32) float wp[6 * KC];

    // Process up to 8 tiles (128 tokens) per W-panel sweep -> acc_full = 6 * 128 * 4 = 3 KB (L1d!)
    const int n_tiles_tot = (T + 15) >> 4;
    alignas(32) float acc_full[8 * 6 * 16];

    for (int tile0 = 0; tile0 < n_tiles_tot; tile0 += 8) {
        const int n_tl = std::min(8, n_tiles_tot - tile0);

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
            alignas(32) float row_acc[8 * 16] = {};
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

// Compact zero-spill Q6_K 1-row x 4-token direct kernel for small MoE buckets (nt = 4..11)!
inline void gemm_q6_k_1x4_direct(
    const BlockQ6_K * w_row,
    const CpuActQ8_K * x0, const CpuActQ8_K * x1, const CpuActQ8_K * x2, const CpuActQ8_K * x3,
    int nb,
    float out[4]) {

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
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
            const int8_t *  sc = wb.scales + h * 8;
            const int off = h * 128;

            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

            // Process one q vector at a time -> only 1 q + 1 sc live -> ZERO YMM register spills!
            auto step_q = [&](__m256i qv, int qoff, int8_t sl, int8_t sh) {
                const __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sh), _mm_set1_epi16(sl));
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x0[b].qs + off + qoff))), scv));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x1[b].qs + off + qoff))), scv));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x2[b].qs + off + qoff))), scv));
                s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(qv, _mm256_load_si256(reinterpret_cast<const __m256i *>(x3[b].qs + off + qoff))), scv));
            };

            step_q(_mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4)), 0, sc[0], sc[1]);
            step_q(_mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4)), 32, sc[2], sc[3]);
            step_q(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4)), 64, sc[4], sc[5]);
            step_q(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4)), 96, sc[6], sc[7]);
        }

        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wd * x0[b].d), _mm256_cvtepi32_ps(s0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wd * x1[b].d), _mm256_cvtepi32_ps(s1), a1);
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wd * x2[b].d), _mm256_cvtepi32_ps(s2), a2);
        a3 = _mm256_fmadd_ps(_mm256_set1_ps(wd * x3[b].d), _mm256_cvtepi32_ps(s3), a3);
    }
    out[0] = hsum256_ps(a0); out[1] = hsum256_ps(a1);
    out[2] = hsum256_ps(a2); out[3] = hsum256_ps(a3);
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
        const __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i *>(x_row[b].qs));
        const float   xd = x_row[b].d;

        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0[b].qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w1[b].qs));
        const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w2[b].qs));
        const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w3[b].qs));

        const __m256i d0 = mul_sum_i8_pairs_to_i32(q0, xv, ones16);
        const __m256i d1 = mul_sum_i8_pairs_to_i32(q1, xv, ones16);
        const __m256i d2 = mul_sum_i8_pairs_to_i32(q2, xv, ones16);
        const __m256i d3 = mul_sum_i8_pairs_to_i32(q3, xv, ones16);

        a0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w0[b].d) * xd), _mm256_cvtepi32_ps(d0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w1[b].d) * xd), _mm256_cvtepi32_ps(d1), a1);
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w2[b].d) * xd), _mm256_cvtepi32_ps(d2), a2);
        a3 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(w3[b].d) * xd), _mm256_cvtepi32_ps(d3), a3);
    }
    y0 = hsum256_ps(a0);
    y1 = hsum256_ps(a1);
    y2 = hsum256_ps(a2);
    y3 = hsum256_ps(a3);
#else
    y0 = dot_q8_0_q8_0_row(w0, x_row, nb);
    y1 = dot_q8_0_q8_0_row(w1, x_row, nb);
    y2 = dot_q8_0_q8_0_row(w2, x_row, nb);
    y3 = dot_q8_0_q8_0_row(w3, x_row, nb);
#endif
}

inline float dot_q6_k_q8_k_row(const BlockQ6_K * w_row, const CpuActQ8_K * x_row, int nb) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const BlockQ6_K & wb = w_row[b];
        const CpuActQ8_K & xb = x_row[b];
        const float d = f16_to_f32_fast(wb.d) * xb.d;

        const __m128i sc8  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wb.scales));
        const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
        const __m256i bs16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.bsums));
        __m256i sumi = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs16), 5));

        for (int h = 0; h < 2; ++h) {
            const uint8_t * ql = wb.ql + h * 64;
            const uint8_t * qh = wb.qh + h * 32;
            const int8_t *  sc = wb.scales + h * 8;
            const int8_t *  qx = xb.qs + h * 128;

            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql0, m4),
                                               _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4));
            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(ql1, m4),
                                               _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4));
            const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                                               _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4));
            const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4),
                                               _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4));

            const __m256i p1 = _mm256_maddubs_epi16(q1, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 0)));
            const __m256i p2 = _mm256_maddubs_epi16(q2, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 32)));
            const __m256i p3 = _mm256_maddubs_epi16(q3, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 64)));
            const __m256i p4 = _mm256_maddubs_epi16(q4, _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 96)));

            const __m256i s01 = _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            const __m256i s23 = _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            const __m256i s45 = _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            const __m256i s67 = _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));

            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p1, s01));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p2, s23));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p3, s45));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p4, s67));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256_ps(acc);
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

// 2-row Q6_K x Q8_K dot product: loads xq8k ONCE for 2 rows (e.g. gate[r] and up[r], or 2 adjacent rows)!
inline void dot_q6_k_q8_k_2rows(
    const BlockQ6_K * w0, const BlockQ6_K * w1,
    const CpuActQ8_K * x_row, int nb,
    float & y0, float & y1) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const BlockQ6_K & wb0 = w0[b];
        const BlockQ6_K & wb1 = w1[b];
        const CpuActQ8_K & xb = x_row[b];
        const float xd = xb.d;
        const __m256i bs16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(xb.bsums));

        const __m256i sc16_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(wb0.scales)));
        const __m256i sc16_1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(wb1.scales)));
        __m256i sum0 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16_0, bs16), 5));
        __m256i sum1 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(_mm256_madd_epi16(sc16_1, bs16), 5));

        for (int h = 0; h < 2; ++h) {
            const int8_t * qx = xb.qs + h * 128;
            const __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 0));
            const __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 32));
            const __m256i x2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 64));
            const __m256i x3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(qx + 96));

            auto do_half = [&](const BlockQ6_K & wb, __m256i & sumi) {
                const uint8_t * ql = wb.ql + h * 64;
                const uint8_t * qh = wb.qh + h * 32;
                const int8_t *  sc = wb.scales + h * 8;
                const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 0));
                const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
                const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

                const __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4));
                const __m256i q2 = _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4));
                const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4));
                const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4));

                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q1, x0), _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]))));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q2, x1), _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]))));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q3, x2), _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]))));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q4, x3), _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]))));
            };
            do_half(wb0, sum0);
            do_half(wb1, sum1);
        }

        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(wb0.d) * xd), _mm256_cvtepi32_ps(sum0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(wb1.d) * xd), _mm256_cvtepi32_ps(sum1), acc1);
    }
    y0 = hsum256_ps(acc0);
    y1 = hsum256_ps(acc1);
#else
    y0 = dot_q6_k_q8_k_row(w0, x_row, nb);
    y1 = dot_q6_k_q8_k_row(w1, x_row, nb);
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
    int i = 0;
    for (; i + 7 < K; i += 8) {
        const __m256 xv  = _mm256_loadu_ps(x + i);
        const __m256 wv0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w0 + i))), 16));
        const __m256 wv1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w1 + i))), 16));
        const __m256 wv2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w2 + i))), 16));
        const __m256 wv3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w3 + i))), 16));
        a0 = _mm256_fmadd_ps(wv0, xv, a0);
        a1 = _mm256_fmadd_ps(wv1, xv, a1);
        a2 = _mm256_fmadd_ps(wv2, xv, a2);
        a3 = _mm256_fmadd_ps(wv3, xv, a3);
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
#else
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += f16_to_f32_fast(w[i]) * x[i];
    return acc;
#endif
}

inline float dot_q4_0_row(const BlockQ4_0 * w_row, const float * x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = f16_to_f32_fast(w_row[b].d);
        const float * xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = w_row[b].qs[j];
            s += static_cast<float>(static_cast<int>(v & 0xFu) - 8) * xb[j];
            s += static_cast<float>(static_cast<int>(v >> 4) - 8) * xb[j + 16];
        }
        acc += d * s;
    }
    return acc;
}

inline float dot_q5_0_row(const BlockQ5_0 * w_row, const float * x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = f16_to_f32_fast(w_row[b].d);
        uint32_t qh = 0;
        std::memcpy(&qh, w_row[b].qh, 4);
        const float * xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = w_row[b].qs[j];
            const uint8_t xh0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
            const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10u);
            s += static_cast<float>(static_cast<int>((v & 0x0Fu) | xh0) - 16) * xb[j];
            s += static_cast<float>(static_cast<int>((v >> 4)   | xh1) - 16) * xb[j + 16];
        }
        acc += d * s;
    }
    return acc;
}

inline float dot_q4_k_row(const BlockQ4_K * w_row, const float * x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d    = f16_to_f32_fast(w_row[b].d);
        const float dmin = f16_to_f32_fast(w_row[b].dmin);
        const uint8_t * q = w_row[b].qs;
        const float * xb  = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, w_row[b].scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, w_row[b].scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            float s1 = 0.0f, s2 = 0.0f;
            for (int l = 0; l < 32; ++l) {
                s1 += (d1 * static_cast<float>(q[l] & 0xFu) - m1) * xb[j + l];
                s2 += (d2 * static_cast<float>(q[l] >> 4)   - m2) * xb[j + 32 + l];
            }
            acc += s1 + s2;
            q  += 32;
            is += 2;
        }
    }
    return acc;
}

inline float dot_q5_k_row(const BlockQ5_K * w_row, const float * x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d    = f16_to_f32_fast(w_row[b].d);
        const float dmin = f16_to_f32_fast(w_row[b].dmin);
        const uint8_t * ql = w_row[b].qs;
        const uint8_t * qh = w_row[b].qh;
        const float * xb   = x + b * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, w_row[b].scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, w_row[b].scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            float s1 = 0.0f, s2 = 0.0f;
            for (int l = 0; l < 32; ++l) {
                const int q1 = (ql[l] & 0xFu) + ((qh[l] & u1) ? 16 : 0);
                const int q2 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                s1 += (d1 * static_cast<float>(q1) - m1) * xb[j + l];
                s2 += (d2 * static_cast<float>(q2) - m2) * xb[j + 32 + l];
            }
            acc += s1 + s2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return acc;
}

inline void linear_rows_range_direct(
    const CpuTensor & W,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * y,
    int T, int r0, int r1) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);

    if (W.dtype == DType::Q8_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ8_0 *>(W.ptr);
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
            int r = r0;
            for (; r + 1 < r1; r += 2) {
                dot_q6_k_q8_k_2rows(base + static_cast<size_t>(r + 0) * nb,
                                    base + static_cast<size_t>(r + 1) * nb,
                                    xt, nb, yt[r + 0], yt[r + 1]);
            }
            for (; r < r1; ++r) {
                yt[r] = dot_q6_k_q8_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
            }
        }
        return;
    }

    if (W.dtype == DType::F32) {
        const auto * base = static_cast<const float *>(W.ptr);
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
            for (int r = r0; r < r1; ++r) yt[r] = dot_f16_row(base + static_cast<size_t>(r) * K, xt, K);
        }
        return;
    }

    if (W.dtype == DType::Q4_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ4_0 *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q4_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q5_0) {
        const int nb = K >> 5;
        const auto * base = static_cast<const BlockQ5_0 *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q5_0_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q4_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ4_K *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q4_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
        }
        return;
    }

    if (W.dtype == DType::Q5_K) {
        const int nb = K >> 8;
        const auto * base = static_cast<const BlockQ5_K *>(W.ptr);
        for (int t = 0; t < T; ++t) {
            const float * xt = x + static_cast<size_t>(t) * K;
            float * yt = y + static_cast<size_t>(t) * M;
            for (int r = r0; r < r1; ++r) yt[r] = dot_q5_k_row(base + static_cast<size_t>(r) * nb, xt, nb);
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
    }
#else
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d  = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        dst[b].d = d;
        for (int i = 0; i < 32; ++i) {
            int v = static_cast<int>(std::lrintf(xb[i] * id));
            v = std::max(-127, std::min(127, v));
            dst[b].qs[i] = static_cast<int8_t>(v);
        }
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
#endif
    for (; i < n; ++i) a[i] += b[i];
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
    if (W.dtype == DType::Q8_0 && xq8 == nullptr) {
        local_q8.resize(static_cast<size_t>(T) * (K >> 5));
        quantize_rows_q8_0(pool, x, local_q8.data(), T, K);
        xq8 = local_q8.data();
    } else if (W.dtype == DType::Q6_K && xq8k == nullptr) {
        local_q8k.resize(static_cast<size_t>(T) * (K >> 8));
        quantize_rows_q8_k(pool, x, local_q8k.data(), T, K);
        xq8k = local_q8k.data();
    }

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
    int total_rows = 0;
    int offsets[16];
    for (int i = 0; i < n_specs; ++i) {
        offsets[i] = total_rows;
        total_rows += static_cast<int>(specs[i].W->ne[1]);
    }
    offsets[n_specs] = total_rows;

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
    if (T >= 12) {
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

    (void) K;
    pool.parallel_for_blocked(total_rows, 4, [&](int g0, int g1, int) {
        for (int i = 0; i < n_specs; ++i) {
            const int s0 = offsets[i];
            const int s1 = offsets[i + 1];
            const int r0 = std::max(g0, s0) - s0;
            const int r1 = std::min(g1, s1) - s0;
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
#endif
    for (; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
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

#if defined(__AVX2__) && defined(__FMA__)
                for (int r = 0; r < 128; r += 8) {
                    _mm256_store_ps(dec + r, exp_ps_avx2(_mm256_loadu_ps(gt + r)));
                }
                for (int col = 0; col < 128; col += 2) {
                    float * sc0 = sh + static_cast<size_t>(col + 0) * 128;
                    float * sc1 = sh + static_cast<size_t>(col + 1) * 128;

                    __m256 kv0 = _mm256_setzero_ps();
                    __m256 kv1 = _mm256_setzero_ps();
                    for (int r = 0; r < 128; r += 8) {
                        const __m256 dv = _mm256_load_ps(dec + r);
                        const __m256 kv = _mm256_loadu_ps(kt + r);
                        const __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(sc0 + r), dv);
                        const __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(sc1 + r), dv);
                        _mm256_storeu_ps(sc0 + r, s0);
                        _mm256_storeu_ps(sc1 + r, s1);
                        kv0 = _mm256_fmadd_ps(s0, kv, kv0);
                        kv1 = _mm256_fmadd_ps(s1, kv, kv1);
                    }
                    const __m256 vd0 = _mm256_set1_ps((vt[col + 0] - hsum256_ps(kv0)) * bval);
                    const __m256 vd1 = _mm256_set1_ps((vt[col + 1] - hsum256_ps(kv1)) * bval);

                    __m256 oa0 = _mm256_setzero_ps();
                    __m256 oa1 = _mm256_setzero_ps();
                    for (int r = 0; r < 128; r += 8) {
                        const __m256 kv = _mm256_loadu_ps(kt + r);
                        const __m256 qv = _mm256_loadu_ps(qt + r);
                        const __m256 s0 = _mm256_fmadd_ps(kv, vd0, _mm256_loadu_ps(sc0 + r));
                        const __m256 s1 = _mm256_fmadd_ps(kv, vd1, _mm256_loadu_ps(sc1 + r));
                        _mm256_storeu_ps(sc0 + r, s0);
                        _mm256_storeu_ps(sc1 + r, s1);
                        oa0 = _mm256_fmadd_ps(s0, qv, oa0);
                        oa1 = _mm256_fmadd_ps(s1, qv, oa1);
                    }
                    ot[col + 0] = hsum256_ps(oa0) * scale;
                    ot[col + 1] = hsum256_ps(oa1) * scale;
                }
#else
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
#endif
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

void mla_kv_norm_rope_write_576(
    const float * kv_all, const float * norm_w, const int32_t * pos,
    void * kv_cache, bool kv_q8_0,
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
        if (kv_q8_0) {
            auto & dst = static_cast<CpuMlaKv576Q8Pos *>(kv_cache)[pidx];
            alignas(32) CpuActQ8_0 tmp_q8[18];
            quantize_row_q8_0(buf, tmp_q8, 576);
            for (int b = 0; b < 18; ++b) {
                dst.d[b] = f32_to_f16_fast(tmp_q8[b].d);
                std::memcpy(dst.qs + b * 32, tmp_q8[b].qs, 32);
            }
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

    pool.parallel_for(n_heads, [&](int h0, int h1, int) {
        std::vector<float>      xh_buf, yh_buf, xp_buf;
        std::vector<CpuActQ8_0> xh_q8;
        std::vector<CpuActQ8_K> xh_q8k;
        for (int h = h0; h < h1; ++h) {
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
                continue;
            }
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
                if (W.dtype == DType::Q6_K) {
                    xh_q8k.resize(static_cast<size_t>(T) * (K >> 8));
                    for (int t = 0; t < T; ++t) quantize_row_q8_k(xh_buf.data() + static_cast<size_t>(t) * K, xh_q8k.data() + static_cast<size_t>(t) * (K >> 8), K);
                }
                linear_rows_range_direct(Wh, xh_buf.data(), nullptr, xh_q8k.empty() ? nullptr : xh_q8k.data(), yh_buf.data(), T, 0, M);
                for (int t = 0; t < T; ++t) {
                    std::memcpy(y + (static_cast<size_t>(t) * n_heads + h) * M,
                                yh_buf.data() + static_cast<size_t>(t) * M,
                                static_cast<size_t>(M) * sizeof(float));
                }
            }
        }
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
    const float * q576, const void * kv_cache, bool kv_q8_0,
    const int32_t * pos, float * out,
    int T, int n_heads, float scale) {

#if defined(__AVX2__) && defined(__FMA__)
    if (kv_q8_0) {
        const auto * kvc = static_cast<const CpuMlaKv576Q8Pos *>(kv_cache);
        // Partition heads across threads; each thread shares KV cache reads across all its heads (e.g. 4 heads/core)!
        // Uses 8-position tile-softmax with AVX2 exp_ps_avx2 + sparse V skipping (score > m_new - 18.0f)!
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
                float kv_d_cache[P_TILE][18];

                for (int p0 = 0; p0 < kv_len; p0 += P_TILE) {
                    const int np = std::min(P_TILE, kv_len - p0);
                    // Pre-decode FP16 scales for this 8-pos tile ONCE for all nh heads!
                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q8Pos & kv = kvc[p0 + pi];
                        for (int b = 0; b < 18; ++b) kv_d_cache[pi][b] = f16_to_f32_fast(kv.d[b]);
                    }

                    // Score all (hi, pi) while the 8 KV positions (5 KB) sit hot in L1d!
                    for (int pi = 0; pi < np; ++pi) {
                        const CpuMlaKv576Q8Pos & kv = kvc[p0 + pi];
                        for (int hi = 0; hi < nh; ++hi) {
                            __m256 sacc0 = _mm256_setzero_ps();
                            __m256 sacc1 = _mm256_setzero_ps();
                            for (int b = 0; b < 18; b += 2) {
                                const __m256i kv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(kv.qs + (b + 0) * 32));
                                const __m256i qv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi][b + 0].qs));
                                const __m256i dp0 = mul_sum_i8_pairs_to_i32(kv0, qv0, ones16);
                                sacc0 = _mm256_fmadd_ps(_mm256_set1_ps(kv_d_cache[pi][b + 0] * q_q8[hi][b + 0].d), _mm256_cvtepi32_ps(dp0), sacc0);

                                const __m256i kv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(kv.qs + (b + 1) * 32));
                                const __m256i qv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(q_q8[hi][b + 1].qs));
                                const __m256i dp1 = mul_sum_i8_pairs_to_i32(kv1, qv1, ones16);
                                sacc1 = _mm256_fmadd_ps(_mm256_set1_ps(kv_d_cache[pi][b + 1] * q_q8[hi][b + 1].d), _mm256_cvtepi32_ps(dp1), sacc1);
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

                        // Compute all 8 exp(score - m_new) in ONE AVX2 vector call!
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

                if (kv_q8_0) {
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
    int T, int k, float clamp_exp, float clamp_shexp) {

    const int n_embd   = static_cast<int>(gate_exps.ne[0]);
    const int ff_exp   = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_shexp = static_cast<int>(gate_shexp.ne[1]);

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

            // Barrier 1: Stream 128-row (4 x 32) chunks sequentially (gate then up -> 100% HW prefetch + balanced across 4 cores!)
            const int chunks_per_exp = std::max(1, ff_nb_q8 / 4);
            const int chunks_sh      = std::max(1, ff_sh_q8 / 4);
            const int total_chunks   = k * chunks_per_exp + chunks_sh;

            pool.parallel_for(total_chunks, [&](int c0, int c1, int) {
                alignas(32) float gbuf[128];
                alignas(32) float ubuf[128];
                for (int gc = c0; gc < c1; ++gc) {
                    const bool is_sh = (gc >= k * chunks_per_exp);
                    const int  slot  = is_sh ? k : (gc / chunks_per_exp);
                    const int  lc    = is_sh ? (gc - k * chunks_per_exp) : (gc % chunks_per_exp);
                    const int  slot_ff_blks = is_sh ? ff_sh_q8 : ff_nb_q8;
                    const int  blk0  = lc * 4;
                    const int  n_blk = std::min(4, slot_ff_blks - blk0);
                    const int  nrows = n_blk * 32;
                    const int  row0  = blk0 * 32;
                    const int  gb0   = is_sh ? (k * ff_nb_q8 + blk0) : (slot * ff_nb_q8 + blk0);
                    const float cl   = is_sh ? clamp_shexp : clamp_exp;

                    if (is_q6k) {
                        const auto * wgb = static_cast<const BlockQ6_K *>(g_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        const auto * wub = static_cast<const BlockQ6_K *>(u_ptrs[slot]) + static_cast<size_t>(row0) * nb_in;
                        for (int r = 0; r < nrows; r += 2) {
                            dot_q6_k_q8_k_2rows(wgb + (r + 0) * nb_in, wgb + (r + 1) * nb_in, xq8k, nb_in, gbuf[r + 0], gbuf[r + 1]);
                        }
                        for (int r = 0; r < nrows; r += 2) {
                            dot_q6_k_q8_k_2rows(wub + (r + 0) * nb_in, wub + (r + 1) * nb_in, xq8k, nb_in, ubuf[r + 0], ubuf[r + 1]);
                        }
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

            // Barrier 2: Stream ONE expert sequentially at a time over thread's L1-resident out[r0..r1)!
            pool.parallel_for_blocked(n_embd, 4, [&](int r0, int r1, int) {
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
            });
            return;
        }

        // Generic T=1 fallback for other quant combinations
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
        const LinearSpec specs[2] = {{&gate_shexp, sh_g.data()}, {&up_shexp, sh_u.data()}};
        linear_multi(pool, specs, 2, x, xq8, xq8k, T);
        apply_swiglu_vec(sh_g.data(), sh_u.data(), static_cast<size_t>(T) * ff_shexp, clamp_shexp);
        linear(pool, down_shexp, sh_g.data(), nullptr, nullptr, out, T);
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

    const int nth = pool.num_threads();
    thread_local std::vector<std::vector<float>> thr_out;
    thr_out.resize(static_cast<size_t>(nth));
    for (int i = 0; i < nth; ++i) {
        thr_out[i].assign(static_cast<size_t>(T) * n_embd, 0.0f);
    }

    const int K_nb_q8  = n_embd >> 5;
    const int K_nb_q8k = n_embd >> 8;
    const int F_nb_q8  = ff_exp >> 5;
    const int F_nb_q8k = ff_exp >> 8;

    const auto * buckets_p = &buckets;
    auto *       thr_out_p = &thr_out;

    // Atomic dynamic work queue across active experts for near-perfect load balance across varying bucket sizes!
    std::atomic<size_t> next_exp{0};

    pool.parallel([&](int tid, int) {
        std::vector<float>      g_buf, u_buf, y_buf, xp_buf, hp_buf, gx_f32;
        std::vector<CpuActQ8_0> gx_q8, gh_q8;
        std::vector<CpuActQ8_K> gx_q8k, gh_q8k;
        float * my_out = (*thr_out_p)[tid].data();

        while (true) {
            const size_t idx = next_exp.fetch_add(1, std::memory_order_relaxed);
            if (idx >= active_exps.size()) break;

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
            } else
#endif
            {
                const bool need_q8  = (gate_exps.dtype == DType::Q8_0 || up_exps.dtype == DType::Q8_0);
                const bool need_q8k = (gate_exps.dtype == DType::Q6_K || up_exps.dtype == DType::Q6_K);
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

                linear_rows_range_direct(Wg, gx_f32.data(), gx_q8.data(), gx_q8k.data(), g_buf.data(), nt, 0, ff_exp);
                linear_rows_range_direct(Wu, gx_f32.data(), gx_q8.data(), gx_q8k.data(), u_buf.data(), nt, 0, ff_exp);
                apply_swiglu_vec(g_buf.data(), u_buf.data(), static_cast<size_t>(nt) * ff_exp, clamp_exp);

                if (down_exps.dtype == DType::Q8_0) {
                    gh_q8.resize(static_cast<size_t>(nt) * F_nb_q8);
                    for (int i = 0; i < nt; ++i) quantize_row_q8_0(g_buf.data() + static_cast<size_t>(i) * ff_exp, gh_q8.data() + static_cast<size_t>(i) * F_nb_q8, ff_exp);
                } else if (down_exps.dtype == DType::Q6_K) {
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
#endif
                for (; d < n_embd; ++d) dst[d] += w * src[d];
            }
        }
    });

    const size_t tot = static_cast<size_t>(T) * n_embd;
    for (int i = 0; i < nth; ++i) {
        add_inplace(out, thr_out[i].data(), tot);
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
