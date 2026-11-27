#include "cpu/qwen35_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rookery::cpu {

namespace {

#if defined(__aarch64__) && defined(__ARM_NEON)
inline float32x4_t q35_vexpq_f32_fast(float32x4_t x) {
    const float32x4_t c_hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t c_lo = vdupq_n_f32(-88.3762626647949f);
    x = vminq_f32(vmaxq_f32(x, c_lo), c_hi);

    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
    const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);

    float32x4_t fx = vrndaq_f32(vmulq_f32(x, log2e));
    int32x4_t   ni = vcvtq_s32_f32(fx);
    float32x4_t r  = vfmsq_f32(vfmsq_f32(x, fx, ln2_hi), fx, ln2_lo);

    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, vmulq_f32(r, r));

    int32x4_t pow2n = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(pow2n));
}
#endif

inline size_t q35_kv_head_stride_bytes(int head_dim, bool kv_q8_0) {
    if (kv_q8_0) {
        const size_t d_bytes = static_cast<size_t>(head_dim >> 5) * sizeof(uint16_t);
        return static_cast<size_t>(head_dim) + ((d_bytes + 31u) & ~size_t(31));
    }
    return static_cast<size_t>(head_dim) * sizeof(uint16_t);
}

// 1. GDN Gate (ssm_a * softplus(alpha + dt_bias)) + Beta (sigmoid(beta))
void q35_gdn_gate_beta(float * alpha, const float * dt_bias, const float * ssm_a,
                       float * beta, int T, int n_v_heads) {
    for (int t = 0; t < T; ++t) {
        float * at = alpha + static_cast<size_t>(t) * n_v_heads;
        float * bt = beta  + static_cast<size_t>(t) * n_v_heads;
        for (int h = 0; h < n_v_heads; ++h) {
            const float x  = at[h] + dt_bias[h];
            const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
            at[h] = sp * ssm_a[h];
            bt[h] = 1.0f / (1.0f + std::exp(-bt[h]));
        }
    }
}

// 2. Causal Depthwise Conv1D (d_conv=4) + SiLU + L2-norm on packed [Q, K, V]
//    Tiled 64 channels per worker with t-outer, c-inner contiguous NEON sweeps!
void q35_conv1d_silu_l2(
    ThreadPool & pool,
    const float * x, const float * w, float * conv_state, float * out,
    int T, int n_k_heads, int n_v_heads, int d_state, float eps) {

    const int conv_dim = (2 * n_k_heads + n_v_heads) * d_state;
    pool.parallel_for_blocked(conv_dim, 64, [&](int c0, int c1, int) {
        const int nc = c1 - c0;
        alignas(64) float s0[64], s1[64], s2[64];
        alignas(64) float w0[64], w1[64], w2[64], w3[64];
        for (int i = 0; i < nc; ++i) {
            const int c = c0 + i;
            s0[i] = conv_state[0 * conv_dim + c];
            s1[i] = conv_state[1 * conv_dim + c];
            s2[i] = conv_state[2 * conv_dim + c];
            w0[i] = w[c * 4 + 0];
            w1[i] = w[c * 4 + 1];
            w2[i] = w[c * 4 + 2];
            w3[i] = w[c * 4 + 3];
        }
#if defined(__aarch64__) && defined(__ARM_NEON)
        const float32x4_t v1 = vdupq_n_f32(1.0f);
#endif
        for (int t = 0; t < T; ++t) {
            const float * xt = x   + static_cast<size_t>(t) * conv_dim + c0;
            float *       ot = out + static_cast<size_t>(t) * conv_dim + c0;
            int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            for (; i + 8 <= nc; i += 8) {
                const float32x4_t vs0a = vld1q_f32(s0 + i + 0), vs0b = vld1q_f32(s0 + i + 4);
                const float32x4_t vs1a = vld1q_f32(s1 + i + 0), vs1b = vld1q_f32(s1 + i + 4);
                const float32x4_t vs2a = vld1q_f32(s2 + i + 0), vs2b = vld1q_f32(s2 + i + 4);
                const float32x4_t vxa  = vld1q_f32(xt + i + 0), vxb  = vld1q_f32(xt + i + 4);
                float32x4_t aa = vmulq_f32(vs0a, vld1q_f32(w0 + i + 0));
                float32x4_t ab = vmulq_f32(vs0b, vld1q_f32(w0 + i + 4));
                aa = vfmaq_f32(aa, vs1a, vld1q_f32(w1 + i + 0));
                ab = vfmaq_f32(ab, vs1b, vld1q_f32(w1 + i + 4));
                aa = vfmaq_f32(aa, vs2a, vld1q_f32(w2 + i + 0));
                ab = vfmaq_f32(ab, vs2b, vld1q_f32(w2 + i + 4));
                aa = vfmaq_f32(aa, vxa,  vld1q_f32(w3 + i + 0));
                ab = vfmaq_f32(ab, vxb,  vld1q_f32(w3 + i + 4));
                vst1q_f32(s0 + i + 0, vs1a); vst1q_f32(s0 + i + 4, vs1b);
                vst1q_f32(s1 + i + 0, vs2a); vst1q_f32(s1 + i + 4, vs2b);
                vst1q_f32(s2 + i + 0, vxa);  vst1q_f32(s2 + i + 4, vxb);
                vst1q_f32(ot + i + 0, vdivq_f32(aa, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(aa)))));
                vst1q_f32(ot + i + 4, vdivq_f32(ab, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(ab)))));
            }
#endif
            for (; i < nc; ++i) {
                const float xv  = xt[i];
                const float acc = s0[i] * w0[i] + s1[i] * w1[i] + s2[i] * w2[i] + xv * w3[i];
                ot[i] = acc / (1.0f + std::exp(-acc));
                s0[i] = s1[i]; s1[i] = s2[i]; s2[i] = xv;
            }
        }
        for (int i = 0; i < nc; ++i) {
            const int c = c0 + i;
            conv_state[0 * conv_dim + c] = s0[i];
            conv_state[1 * conv_dim + c] = s1[i];
            conv_state[2 * conv_dim + c] = s2[i];
        }
    });

    const int n_qk_heads = 2 * n_k_heads;
    const float eps2 = eps * eps;
    pool.parallel_for(T * n_qk_heads, [&](int job0, int job1, int) {
        for (int job = job0; job < job1; ++job) {
            const int t  = job / n_qk_heads;
            const int qh = job % n_qk_heads;
            float * p = out + static_cast<size_t>(t) * conv_dim + qh * d_state;
            float ss = 0.0f;
#if defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 8 <= d_state; i += 8) {
                const float32x4_t v0 = vld1q_f32(p + i);
                const float32x4_t v1 = vld1q_f32(p + i + 4);
                a0 = vfmaq_f32(a0, v0, v0);
                a1 = vfmaq_f32(a1, v1, v1);
            }
            ss = vaddvq_f32(vaddq_f32(a0, a1));
            for (; i < d_state; ++i) ss += p[i] * p[i];
            const float inv = 1.0f / std::sqrt(std::max(ss, eps2));
            const float32x4_t vinv = vdupq_n_f32(inv);
            i = 0;
            for (; i + 8 <= d_state; i += 8) {
                vst1q_f32(p + i + 0, vmulq_f32(vld1q_f32(p + i + 0), vinv));
                vst1q_f32(p + i + 4, vmulq_f32(vld1q_f32(p + i + 4), vinv));
            }
            for (; i < d_state; ++i) p[i] *= inv;
#else
            for (int i = 0; i < d_state; ++i) ss += p[i] * p[i];
            const float inv = 1.0f / std::sqrt(std::max(ss, eps2));
            for (int i = 0; i < d_state; ++i) p[i] *= inv;
#endif
        }
    });
}

// 3. Gated DeltaNet Recurrence: Single-pass fused 4-column L1d-strip-tiled NEON + big.LITTLE prefill!
void q35_gdn_recurrence(
    ThreadPool & pool,
    const float * qkv_conv, const float * g, const float * beta,
    float * state, float * out,
    int T, int n_k_heads, int n_v_heads, int d_state, float scale) {

    const int conv_dim = (2 * n_k_heads + n_v_heads) * d_state;
    constexpr int COL_STRIP = 16;
    const int n_strips = (d_state + COL_STRIP - 1) / COL_STRIP;
    const int total_tasks = n_v_heads * n_strips;

    auto run_strip = [&](int task) {
        const int h      = task / n_strips;
        const int col0   = (task % n_strips) * COL_STRIP;
        const int col1   = std::min(d_state, col0 + COL_STRIP);
        const int kh     = h % n_k_heads;
        const int q_base = kh * d_state;
        const int k_base = (n_k_heads + kh) * d_state;
        const int v_base = (2 * n_k_heads + h) * d_state;
        float * sh = state + static_cast<size_t>(h) * d_state * d_state;

#if defined(__aarch64__) && defined(__ARM_NEON)
        int col = col0;
        for (; col + 4 <= col1; col += 4) {
            float * sc0 = sh + static_cast<size_t>(col + 0) * d_state;
            float * sc1 = sh + static_cast<size_t>(col + 1) * d_state;
            float * sc2 = sh + static_cast<size_t>(col + 2) * d_state;
            float * sc3 = sh + static_cast<size_t>(col + 3) * d_state;

            // Prologue: Pass 1 for t = 0 (decay 4 state columns and compute kv0..kv3 with shared kt loads)
            float kvf0 = 0.0f, kvf1 = 0.0f, kvf2 = 0.0f, kvf3 = 0.0f;
            {
                const float * kt0 = qkv_conv + k_base;
                const float d0    = std::exp(g[h]);
                const float32x4_t vd0 = vdupq_n_f32(d0);
                float32x4_t kv0 = vdupq_n_f32(0.0f), kv1 = vdupq_n_f32(0.0f);
                float32x4_t kv2 = vdupq_n_f32(0.0f), kv3 = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i + 4 <= d_state; i += 4) {
                    const float32x4_t vk = vld1q_f32(kt0 + i);
                    const float32x4_t s0 = vmulq_f32(vld1q_f32(sc0 + i), vd0); vst1q_f32(sc0 + i, s0); kv0 = vfmaq_f32(kv0, s0, vk);
                    const float32x4_t s1 = vmulq_f32(vld1q_f32(sc1 + i), vd0); vst1q_f32(sc1 + i, s1); kv1 = vfmaq_f32(kv1, s1, vk);
                    const float32x4_t s2 = vmulq_f32(vld1q_f32(sc2 + i), vd0); vst1q_f32(sc2 + i, s2); kv2 = vfmaq_f32(kv2, s2, vk);
                    const float32x4_t s3 = vmulq_f32(vld1q_f32(sc3 + i), vd0); vst1q_f32(sc3 + i, s3); kv3 = vfmaq_f32(kv3, s3, vk);
                }
                kvf0 = vaddvq_f32(kv0); kvf1 = vaddvq_f32(kv1);
                kvf2 = vaddvq_f32(kv2); kvf3 = vaddvq_f32(kv3);
                for (; i < d_state; ++i) {
                    const float kv = kt0[i];
                    sc0[i] *= d0; kvf0 += sc0[i] * kv;
                    sc1[i] *= d0; kvf1 += sc1[i] * kv;
                    sc2[i] *= d0; kvf2 += sc2[i] * kv;
                    sc3[i] *= d0; kvf3 += sc3[i] * kv;
                }
            }

            for (int t = 0; t < T; ++t) {
                const float * row = qkv_conv + static_cast<size_t>(t) * conv_dim;
                const float * qt  = row + q_base;
                const float * kt  = row + k_base;
                const float * vt  = row + v_base;
                const float bval  = beta[static_cast<size_t>(t) * n_v_heads + h];
                float * oh        = out + (static_cast<size_t>(t) * n_v_heads + h) * d_state;

                const float del0 = (vt[col + 0] - kvf0) * bval;
                const float del1 = (vt[col + 1] - kvf1) * bval;
                const float del2 = (vt[col + 2] - kvf2) * bval;
                const float del3 = (vt[col + 3] - kvf3) * bval;
                const float32x4_t vdel0 = vdupq_n_f32(del0);
                const float32x4_t vdel1 = vdupq_n_f32(del1);
                const float32x4_t vdel2 = vdupq_n_f32(del2);
                const float32x4_t vdel3 = vdupq_n_f32(del3);
                float32x4_t at0 = vdupq_n_f32(0.0f), at1 = vdupq_n_f32(0.0f);
                float32x4_t at2 = vdupq_n_f32(0.0f), at3 = vdupq_n_f32(0.0f);

                if (t + 1 < T) {
                    const float * kt_n = qkv_conv + static_cast<size_t>(t + 1) * conv_dim + k_base;
                    const float d_n    = std::exp(g[static_cast<size_t>(t + 1) * n_v_heads + h]);
                    const float32x4_t vd_n = vdupq_n_f32(d_n);
                    float32x4_t kv0 = vdupq_n_f32(0.0f), kv1 = vdupq_n_f32(0.0f);
                    float32x4_t kv2 = vdupq_n_f32(0.0f), kv3 = vdupq_n_f32(0.0f);
                    int i = 0;
                    for (; i + 4 <= d_state; i += 4) {
                        const float32x4_t vk   = vld1q_f32(kt + i);
                        const float32x4_t vq   = vld1q_f32(qt + i);
                        const float32x4_t vk_n = vld1q_f32(kt_n + i);

                        const float32x4_t s0 = vfmaq_f32(vld1q_f32(sc0 + i), vk, vdel0); at0 = vfmaq_f32(at0, s0, vq);
                        const float32x4_t n0 = vmulq_f32(s0, vd_n); vst1q_f32(sc0 + i, n0); kv0 = vfmaq_f32(kv0, n0, vk_n);

                        const float32x4_t s1 = vfmaq_f32(vld1q_f32(sc1 + i), vk, vdel1); at1 = vfmaq_f32(at1, s1, vq);
                        const float32x4_t n1 = vmulq_f32(s1, vd_n); vst1q_f32(sc1 + i, n1); kv1 = vfmaq_f32(kv1, n1, vk_n);

                        const float32x4_t s2 = vfmaq_f32(vld1q_f32(sc2 + i), vk, vdel2); at2 = vfmaq_f32(at2, s2, vq);
                        const float32x4_t n2 = vmulq_f32(s2, vd_n); vst1q_f32(sc2 + i, n2); kv2 = vfmaq_f32(kv2, n2, vk_n);

                        const float32x4_t s3 = vfmaq_f32(vld1q_f32(sc3 + i), vk, vdel3); at3 = vfmaq_f32(at3, s3, vq);
                        const float32x4_t n3 = vmulq_f32(s3, vd_n); vst1q_f32(sc3 + i, n3); kv3 = vfmaq_f32(kv3, n3, vk_n);
                    }
                    float af0 = vaddvq_f32(at0), af1 = vaddvq_f32(at1), af2 = vaddvq_f32(at2), af3 = vaddvq_f32(at3);
                    kvf0 = vaddvq_f32(kv0); kvf1 = vaddvq_f32(kv1); kvf2 = vaddvq_f32(kv2); kvf3 = vaddvq_f32(kv3);
                    for (; i < d_state; ++i) {
                        const float kv = kt[i], qv = qt[i], kvn = kt_n[i];
                        float s0 = sc0[i] + kv * del0; af0 += s0 * qv; s0 *= d_n; sc0[i] = s0; kvf0 += s0 * kvn;
                        float s1 = sc1[i] + kv * del1; af1 += s1 * qv; s1 *= d_n; sc1[i] = s1; kvf1 += s1 * kvn;
                        float s2 = sc2[i] + kv * del2; af2 += s2 * qv; s2 *= d_n; sc2[i] = s2; kvf2 += s2 * kvn;
                        float s3 = sc3[i] + kv * del3; af3 += s3 * qv; s3 *= d_n; sc3[i] = s3; kvf3 += s3 * kvn;
                    }
                    oh[col + 0] = af0 * scale; oh[col + 1] = af1 * scale;
                    oh[col + 2] = af2 * scale; oh[col + 3] = af3 * scale;
                } else {
                    int i = 0;
                    for (; i + 4 <= d_state; i += 4) {
                        const float32x4_t vk = vld1q_f32(kt + i);
                        const float32x4_t vq = vld1q_f32(qt + i);
                        const float32x4_t s0 = vfmaq_f32(vld1q_f32(sc0 + i), vk, vdel0); vst1q_f32(sc0 + i, s0); at0 = vfmaq_f32(at0, s0, vq);
                        const float32x4_t s1 = vfmaq_f32(vld1q_f32(sc1 + i), vk, vdel1); vst1q_f32(sc1 + i, s1); at1 = vfmaq_f32(at1, s1, vq);
                        const float32x4_t s2 = vfmaq_f32(vld1q_f32(sc2 + i), vk, vdel2); vst1q_f32(sc2 + i, s2); at2 = vfmaq_f32(at2, s2, vq);
                        const float32x4_t s3 = vfmaq_f32(vld1q_f32(sc3 + i), vk, vdel3); vst1q_f32(sc3 + i, s3); at3 = vfmaq_f32(at3, s3, vq);
                    }
                    float af0 = vaddvq_f32(at0), af1 = vaddvq_f32(at1), af2 = vaddvq_f32(at2), af3 = vaddvq_f32(at3);
                    for (; i < d_state; ++i) {
                        const float kv = kt[i], qv = qt[i];
                        sc0[i] += kv * del0; af0 += sc0[i] * qv;
                        sc1[i] += kv * del1; af1 += sc1[i] * qv;
                        sc2[i] += kv * del2; af2 += sc2[i] * qv;
                        sc3[i] += kv * del3; af3 += sc3[i] * qv;
                    }
                    oh[col + 0] = af0 * scale; oh[col + 1] = af1 * scale;
                    oh[col + 2] = af2 * scale; oh[col + 3] = af3 * scale;
                }
            }
        }
        for (; col < col1; ++col) {
            float * s_col = sh + static_cast<size_t>(col) * d_state;
            for (int t = 0; t < T; ++t) {
                const float * row = qkv_conv + static_cast<size_t>(t) * conv_dim;
                const float * qt = row + q_base, * kt = row + k_base, * vt = row + v_base;
                const float d = std::exp(g[static_cast<size_t>(t) * n_v_heads + h]);
                const float bval = beta[static_cast<size_t>(t) * n_v_heads + h];
                float kv = 0.0f;
                for (int i = 0; i < d_state; ++i) { s_col[i] *= d; kv += s_col[i] * kt[i]; }
                const float delta = (vt[col] - kv) * bval;
                float attn = 0.0f;
                for (int i = 0; i < d_state; ++i) { s_col[i] += kt[i] * delta; attn += s_col[i] * qt[i]; }
                out[(static_cast<size_t>(t) * n_v_heads + h) * d_state + col] = attn * scale;
            }
        }
#else
        for (int t = 0; t < T; ++t) {
            const float * row = qkv_conv + static_cast<size_t>(t) * conv_dim;
            const float * qt  = row + q_base;
            const float * kt  = row + k_base;
            const float * vt  = row + v_base;
            const float d     = std::exp(g[static_cast<size_t>(t) * n_v_heads + h]);
            const float bval  = beta[static_cast<size_t>(t) * n_v_heads + h];
            float * oh        = out + (static_cast<size_t>(t) * n_v_heads + h) * d_state;
            for (int col = col0; col < col1; ++col) {
                float * s_col = sh + static_cast<size_t>(col) * d_state;
                float kv = 0.0f;
                for (int i = 0; i < d_state; ++i) { s_col[i] *= d; kv += s_col[i] * kt[i]; }
                const float delta = (vt[col] - kv) * bval;
                float attn = 0.0f;
                for (int i = 0; i < d_state; ++i) { s_col[i] += kt[i] * delta; attn += s_col[i] * qt[i]; }
                oh[col] = attn * scale;
            }
        }
#endif
    };

    if (T >= 16 && pool.num_compute_threads() > pool.num_threads()) {
        const int n_big = pool.num_threads();
        const int little_limit = std::max(0, total_tasks - 3 * n_big);
        std::atomic<int> next_task{0};
        pool.parallel_compute([&](int tid, int) {
            const bool is_little = (tid >= n_big);
            while (true) {
                int task = next_task.load(std::memory_order_relaxed);
                while (true) {
                    if (task >= (is_little ? little_limit : total_tasks)) { task = -1; break; }
                    if (next_task.compare_exchange_weak(task, task + 1, std::memory_order_relaxed)) break;
                }
                if (task < 0) break;
                run_strip(task);
            }
        });
        return;
    }

    pool.parallel_for(total_tasks, [&](int task0, int task1, int) {
        for (int task = task0; task < task1; ++task) run_strip(task);
    });
}

// 4. Per-head RMSNorm(o, ssm_norm, eps) * SiLU(z) — parallel & NEON-vectorized!
void q35_rms_norm_silu_gate(ThreadPool & pool, float * o, const float * norm_w, const float * z,
                            int T, int n_v_heads, int d_state, float eps) {
    const float inv_n = 1.0f / static_cast<float>(d_state);
    const int total = T * n_v_heads;
    auto work = [&](int j0, int j1, int) {
#if defined(__aarch64__) && defined(__ARM_NEON)
        const float32x4_t v1 = vdupq_n_f32(1.0f);
#endif
        for (int j = j0; j < j1; ++j) {
            const size_t off = static_cast<size_t>(j) * d_state;
            float * op = o + off;
            const float * zp = z + off;
            float ss = 0.0f;
            int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            for (; i + 8 <= d_state; i += 8) {
                const float32x4_t v0 = vld1q_f32(op + i + 0);
                const float32x4_t v1v = vld1q_f32(op + i + 4);
                s0 = vfmaq_f32(s0, v0, v0);
                s1 = vfmaq_f32(s1, v1v, v1v);
            }
            ss = vaddvq_f32(vaddq_f32(s0, s1));
#endif
            for (; i < d_state; ++i) ss += op[i] * op[i];
            const float inv = 1.0f / std::sqrt(ss * inv_n + eps);
            i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            const float32x4_t vinv = vdupq_n_f32(inv);
            for (; i + 8 <= d_state; i += 8) {
                const float32x4_t z0 = vld1q_f32(zp + i + 0), z1 = vld1q_f32(zp + i + 4);
                const float32x4_t g0 = vdivq_f32(z0, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(z0))));
                const float32x4_t g1 = vdivq_f32(z1, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(z1))));
                const float32x4_t r0 = vmulq_f32(vmulq_f32(vld1q_f32(op + i + 0), vinv), vld1q_f32(norm_w + i + 0));
                const float32x4_t r1 = vmulq_f32(vmulq_f32(vld1q_f32(op + i + 4), vinv), vld1q_f32(norm_w + i + 4));
                vst1q_f32(op + i + 0, vmulq_f32(r0, g0));
                vst1q_f32(op + i + 4, vmulq_f32(r1, g1));
            }
#endif
            for (; i < d_state; ++i) {
                const float zv = zp[i];
                op[i] = op[i] * inv * norm_w[i] * (zv / (1.0f + std::exp(-zv)));
            }
        }
    };
    if (T >= 4 && pool.num_threads() > 1) pool.parallel_for(total, work);
    else                                  work(0, total, 0);
}

// 5. Split interleaved [Q_h, gate_h] + per-head RMSNorm + partial NeoX RoPE
void q35_qgate_split_norm_rope(
    ThreadPool & pool,
    const float * q_full, const float * q_norm_w, const int32_t * pos,
    float * q_out, float * gate_out,
    int T, int n_head, int head_dim, int rope_dim, float rms_eps, float rope_freq_base) {

    const int half_r = rope_dim >> 1;
    std::vector<float> freqs(static_cast<size_t>(half_r));
    for (int d = 0; d < half_r; ++d) {
        freqs[d] = std::pow(rope_freq_base, -2.0f * static_cast<float>(d) / static_cast<float>(rope_dim));
    }
    const float inv_hd = 1.0f / static_cast<float>(head_dim);

    auto work = [&](int t0, int t1, int) {
        alignas(32) float cosv[256], sinv[256];
        for (int t = t0; t < t1; ++t) {
            const float p = static_cast<float>(pos[t]);
            for (int d = 0; d < half_r; ++d) {
                const float a = p * freqs[d];
                cosv[d] = std::cos(a);
                sinv[d] = std::sin(a);
            }
            for (int h = 0; h < n_head; ++h) {
                const float * src_q = q_full + (static_cast<size_t>(t) * n_head + h) * (2 * head_dim);
                const float * src_g = src_q + head_dim;
                float * dst_q = q_out    + (static_cast<size_t>(t) * n_head + h) * head_dim;
                float * dst_g = gate_out + (static_cast<size_t>(t) * n_head + h) * head_dim;
                std::memcpy(dst_g, src_g, static_cast<size_t>(head_dim) * sizeof(float));

                float ss = 0.0f;
                int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
                for (; d + 8 <= head_dim; d += 8) {
                    const float32x4_t v0 = vld1q_f32(src_q + d + 0);
                    const float32x4_t v1 = vld1q_f32(src_q + d + 4);
                    s0 = vfmaq_f32(s0, v0, v0);
                    s1 = vfmaq_f32(s1, v1, v1);
                }
                ss = vaddvq_f32(vaddq_f32(s0, s1));
#endif
                for (; d < head_dim; ++d) ss += src_q[d] * src_q[d];
                const float inv = 1.0f / std::sqrt(ss * inv_hd + rms_eps);
                d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t vinv = vdupq_n_f32(inv);
                for (; d + 8 <= head_dim; d += 8) {
                    vst1q_f32(dst_q + d + 0, vmulq_f32(vmulq_f32(vld1q_f32(src_q + d + 0), vinv), vld1q_f32(q_norm_w + d + 0)));
                    vst1q_f32(dst_q + d + 4, vmulq_f32(vmulq_f32(vld1q_f32(src_q + d + 4), vinv), vld1q_f32(q_norm_w + d + 4)));
                }
#endif
                for (; d < head_dim; ++d) dst_q[d] = src_q[d] * inv * q_norm_w[d];

                d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                for (; d + 4 <= half_r; d += 4) {
                    const float32x4_t x0 = vld1q_f32(dst_q + d);
                    const float32x4_t x1 = vld1q_f32(dst_q + half_r + d);
                    const float32x4_t cv = vld1q_f32(cosv + d);
                    const float32x4_t sv = vld1q_f32(sinv + d);
                    vst1q_f32(dst_q + d,          vfmsq_f32(vmulq_f32(x0, cv), x1, sv));
                    vst1q_f32(dst_q + half_r + d, vfmaq_f32(vmulq_f32(x1, cv), x0, sv));
                }
#endif
                for (; d < half_r; ++d) {
                    const float x0 = dst_q[d];
                    const float x1 = dst_q[half_r + d];
                    dst_q[d]          = x0 * cosv[d] - x1 * sinv[d];
                    dst_q[half_r + d] = x0 * sinv[d] + x1 * cosv[d];
                }
            }
        }
    };
    if (T >= 4 && pool.num_threads() > 1) pool.parallel_for(T, work);
    else                                  work(0, T, 0);
}

void q35_k_norm_rope(
    ThreadPool & pool,
    float * k_buf, const float * k_norm_w, const int32_t * pos,
    int T, int n_kv, int head_dim, int rope_dim, float rms_eps, float rope_freq_base) {

    const int half_r = rope_dim >> 1;
    std::vector<float> freqs(static_cast<size_t>(half_r));
    for (int d = 0; d < half_r; ++d) {
        freqs[d] = std::pow(rope_freq_base, -2.0f * static_cast<float>(d) / static_cast<float>(rope_dim));
    }
    const float inv_hd = 1.0f / static_cast<float>(head_dim);

    auto work = [&](int t0, int t1, int) {
        alignas(32) float cosv[256], sinv[256];
        for (int t = t0; t < t1; ++t) {
            const float p = static_cast<float>(pos[t]);
            for (int d = 0; d < half_r; ++d) {
                const float a = p * freqs[d];
                cosv[d] = std::cos(a);
                sinv[d] = std::sin(a);
            }
            for (int h = 0; h < n_kv; ++h) {
                float * kp = k_buf + (static_cast<size_t>(t) * n_kv + h) * head_dim;
                float ss = 0.0f;
                int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
                for (; d + 8 <= head_dim; d += 8) {
                    const float32x4_t v0 = vld1q_f32(kp + d + 0);
                    const float32x4_t v1 = vld1q_f32(kp + d + 4);
                    s0 = vfmaq_f32(s0, v0, v0);
                    s1 = vfmaq_f32(s1, v1, v1);
                }
                ss = vaddvq_f32(vaddq_f32(s0, s1));
#endif
                for (; d < head_dim; ++d) ss += kp[d] * kp[d];
                const float inv = 1.0f / std::sqrt(ss * inv_hd + rms_eps);
                d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                const float32x4_t vinv = vdupq_n_f32(inv);
                for (; d + 8 <= head_dim; d += 8) {
                    vst1q_f32(kp + d + 0, vmulq_f32(vmulq_f32(vld1q_f32(kp + d + 0), vinv), vld1q_f32(k_norm_w + d + 0)));
                    vst1q_f32(kp + d + 4, vmulq_f32(vmulq_f32(vld1q_f32(kp + d + 4), vinv), vld1q_f32(k_norm_w + d + 4)));
                }
#endif
                for (; d < head_dim; ++d) kp[d] = kp[d] * inv * k_norm_w[d];

                d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                for (; d + 4 <= half_r; d += 4) {
                    const float32x4_t x0 = vld1q_f32(kp + d);
                    const float32x4_t x1 = vld1q_f32(kp + half_r + d);
                    const float32x4_t cv = vld1q_f32(cosv + d);
                    const float32x4_t sv = vld1q_f32(sinv + d);
                    vst1q_f32(kp + d,          vfmsq_f32(vmulq_f32(x0, cv), x1, sv));
                    vst1q_f32(kp + half_r + d, vfmaq_f32(vmulq_f32(x1, cv), x0, sv));
                }
#endif
                for (; d < half_r; ++d) {
                    const float x0 = kp[d];
                    const float x1 = kp[half_r + d];
                    kp[d]          = x0 * cosv[d] - x1 * sinv[d];
                    kp[half_r + d] = x0 * sinv[d] + x1 * cosv[d];
                }
            }
        }
    };
    if (T >= 4 && pool.num_threads() > 1) pool.parallel_for(T, work);
    else                                  work(0, T, 0);
}

void q35_kv_write(
    ThreadPool & pool,
    const float * k, const float * v, void * k_cache, void * v_cache, bool kv_q8_0,
    const int32_t * pos, int T, int n_kv, int head_dim) {

    const int nb = head_dim >> 5;
    const size_t stride = q35_kv_head_stride_bytes(head_dim, kv_q8_0);
    auto work = [&](int t0, int t1, int) {
        if (kv_q8_0) {
            auto * kb = static_cast<uint8_t *>(k_cache);
            auto * vb = static_cast<uint8_t *>(v_cache);
            alignas(32) CpuActQ8_0 tmp[16];
            for (int t = t0; t < t1; ++t) {
                const int slot = pos[t];
                for (int h = 0; h < n_kv; ++h) {
                    const size_t src_off = (static_cast<size_t>(t) * n_kv + h) * head_dim;
                    const size_t h_idx   = static_cast<size_t>(slot) * n_kv + h;

                    quantize_row_q8_0(k + src_off, tmp, head_dim);
                    uint8_t * kd_base = kb + h_idx * stride;
                    auto *    kd_sc   = reinterpret_cast<uint16_t *>(kd_base + head_dim);
                    for (int b = 0; b < nb; ++b) {
                        std::memcpy(kd_base + b * 32, tmp[b].qs, 32);
                        kd_sc[b] = f32_to_f16(tmp[b].d);
                    }

                    quantize_row_q8_0(v + src_off, tmp, head_dim);
                    uint8_t * vd_base = vb + h_idx * stride;
                    auto *    vd_sc   = reinterpret_cast<uint16_t *>(vd_base + head_dim);
                    for (int b = 0; b < nb; ++b) {
                        std::memcpy(vd_base + b * 32, tmp[b].qs, 32);
                        vd_sc[b] = f32_to_f16(tmp[b].d);
                    }
                }
            }
        } else {
            auto * kb = static_cast<uint16_t *>(k_cache);
            auto * vb = static_cast<uint16_t *>(v_cache);
            for (int t = t0; t < t1; ++t) {
                const int slot = pos[t];
                for (int h = 0; h < n_kv; ++h) {
                    const float * ks = k + (static_cast<size_t>(t) * n_kv + h) * head_dim;
                    const float * vs = v + (static_cast<size_t>(t) * n_kv + h) * head_dim;
                    uint16_t * kd = kb + (static_cast<size_t>(slot) * n_kv + h) * head_dim;
                    uint16_t * vd = vb + (static_cast<size_t>(slot) * n_kv + h) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        kd[d] = f32_to_f16(ks[d]);
                        vd[d] = f32_to_f16(vs[d]);
                    }
                }
            }
        }
    };
    if (T >= 4 && pool.num_threads() > 1) pool.parallel_for(T, work);
    else                                  work(0, T, 0);
}

void q35_sigmoid_gate_mul(ThreadPool & pool, float * out, const float * gate, size_t n) {
    auto work = [&](int i0, int i1, int) {
        size_t i = static_cast<size_t>(i0);
        const size_t end = static_cast<size_t>(i1);
#if defined(__aarch64__) && defined(__ARM_NEON)
        const float32x4_t v1 = vdupq_n_f32(1.0f);
        for (; i + 8 <= end; i += 8) {
            const float32x4_t g0 = vdivq_f32(v1, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(vld1q_f32(gate + i + 0)))));
            const float32x4_t g1 = vdivq_f32(v1, vaddq_f32(v1, q35_vexpq_f32_fast(vnegq_f32(vld1q_f32(gate + i + 4)))));
            vst1q_f32(out + i + 0, vmulq_f32(vld1q_f32(out + i + 0), g0));
            vst1q_f32(out + i + 4, vmulq_f32(vld1q_f32(out + i + 4), g1));
        }
#endif
        for (; i < end; ++i) {
            out[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
        }
    };
    if (n >= 4096 && pool.num_threads() > 1) pool.parallel_for_blocked(static_cast<int>(n), 256, work);
    else                                     work(0, static_cast<int>(n), 0);
}

} // namespace

CpuTensor Qwen35CpuModel::bind_tensor(const std::string & name, bool required) const {
    const auto loc = sg_.find_tensor(name);
    if (!loc.info) {
        if (required) throw std::runtime_error("Qwen35CpuModel: missing tensor '" + name + "'");
        return CpuTensor{};
    }
    CpuTensor t;
    t.ptr    = loc.data();
    t.dtype  = loc.info->dtype;
    t.nbytes = loc.info->nbytes;
    for (int i = 0; i < 4; ++i) t.ne[i] = loc.info->ne[i];
    return t;
}

void Qwen35CpuModel::load(const std::string & gguf_path,
                          int max_ctx,
                          bool kv_q8_0,
                          int n_threads) {
    const auto t0 = std::chrono::steady_clock::now();
    pool_.init(n_threads);
    print_cpu_caps(pool_.num_threads());

    sg_ = ShardedGguf::open(gguf_path);
    const std::string arch = sg_.get_str("general.architecture");
    if (arch != "qwen35" && arch != "qwen35moe") {
        throw std::runtime_error("Expected qwen35 or qwen35moe GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.is_dense             = (arch == "qwen35");
    cfg_.n_blocks             = static_cast<int>(sg_.get_u64(pfx + "block_count", cfg_.is_dense ? 32 : 41));
    cfg_.n_nextn              = static_cast<int>(sg_.get_u64(pfx + "nextn_predict_layers", 0));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_embd               = static_cast<int>(sg_.get_u64(pfx + "embedding_length", 2048));
    cfg_.n_head               = static_cast<int>(sg_.get_u64(pfx + "attention.head_count", 16));
    cfg_.n_head_kv            = static_cast<int>(sg_.get_u64(pfx + "attention.head_count_kv", 2));
    cfg_.head_dim             = static_cast<int>(sg_.get_u64(pfx + "attention.key_length", 256));
    cfg_.rope_dim             = static_cast<int>(sg_.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.rope_freq_base       = static_cast<float>(sg_.get_f64(pfx + "rope.freq_base", 10000000.0));
    cfg_.rms_eps              = static_cast<float>(sg_.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.ssm_d_conv           = static_cast<int>(sg_.get_u64(pfx + "ssm.conv_kernel", 4));
    cfg_.ssm_d_state          = static_cast<int>(sg_.get_u64(pfx + "ssm.state_size", 128));
    cfg_.ssm_n_group          = static_cast<int>(sg_.get_u64(pfx + "ssm.group_count", 16));
    cfg_.ssm_dt_rank          = static_cast<int>(sg_.get_u64(pfx + "ssm.time_step_rank", 32));
    cfg_.ssm_d_inner          = static_cast<int>(sg_.get_u64(pfx + "ssm.inner_size", 4096));
    cfg_.full_attn_interval   = static_cast<int>(sg_.get_u64(pfx + "full_attention_interval", 4));
    if (cfg_.is_dense) {
        cfg_.n_ff                 = static_cast<int>(sg_.get_u64(pfx + "feed_forward_length", 12288));
        cfg_.n_expert             = 0;
        cfg_.n_expert_used        = 0;
        cfg_.n_ff_exp             = 0;
        cfg_.n_ff_shexp           = 0;
        cfg_.expert_weights_scale = 1.0f;
    } else {
        cfg_.n_ff                 = 0;
        cfg_.n_expert             = static_cast<int>(sg_.get_u64(pfx + "expert_count", 256));
        cfg_.n_expert_used        = static_cast<int>(sg_.get_u64(pfx + "expert_used_count", 8));
        cfg_.n_ff_exp             = static_cast<int>(sg_.get_u64(pfx + "expert_feed_forward_length", 512));
        cfg_.n_ff_shexp           = static_cast<int>(sg_.get_u64(pfx + "expert_shared_feed_forward_length", 512));
        cfg_.expert_weights_scale = static_cast<float>(sg_.get_f64(pfx + "expert_weights_scale", 1.0));
    }
    cfg_.max_ctx = max_ctx;
    cfg_.kv_q8_0 = kv_q8_0;

    cfg_.is_recr.resize(static_cast<size_t>(cfg_.n_layers));
    for (int il = 0; il < cfg_.n_layers; ++il) {
        cfg_.is_recr[il] = ((il + 1) % cfg_.full_attn_interval != 0);
    }

    tok_embd_    = bind_tensor("token_embd.weight");
    output_norm_ = bind_tensor("output_norm.weight");
    output_      = bind_tensor("output.weight", false);
    if (!output_.valid()) output_ = tok_embd_;
    cfg_.n_vocab = static_cast<int>(output_.ne[1]);

    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;

    layers_.resize(static_cast<size_t>(cfg_.n_layers));
    gdn_conv_.resize(static_cast<size_t>(cfg_.n_layers));
    gdn_state_.resize(static_cast<size_t>(cfg_.n_layers));
    k_cache_.resize(static_cast<size_t>(cfg_.n_layers));
    v_cache_.resize(static_cast<size_t>(cfg_.n_layers));

    kv_cap_ = std::min(cfg_.max_ctx, 2048);
    const size_t kv_bytes = static_cast<size_t>(kv_cap_) * cfg_.n_head_kv *
                            q35_kv_head_stride_bytes(cfg_.head_dim, cfg_.kv_q8_0);
    size_t kv_total = 0;
    size_t gdn_total = 0;

    for (int il = 0; il < cfg_.n_layers; ++il) {
        auto & L = layers_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.is_recr        = cfg_.is_recr[il];
        L.attn_norm      = bind_tensor(p + "attn_norm.weight");
        L.post_attn_norm = bind_tensor(p + "post_attention_norm.weight");

        if (L.is_recr) {
            L.wqkv       = bind_tensor(p + "attn_qkv.weight");
            L.wqkv_gate  = bind_tensor(p + "attn_gate.weight");
            L.ssm_conv1d = bind_tensor(p + "ssm_conv1d.weight");
            L.ssm_dt     = bind_tensor(p + "ssm_dt.bias");
            L.ssm_a      = bind_tensor(p + "ssm_a");
            L.ssm_alpha  = bind_tensor(p + "ssm_alpha.weight");
            L.ssm_beta   = bind_tensor(p + "ssm_beta.weight");
            L.ssm_norm   = bind_tensor(p + "ssm_norm.weight");
            L.ssm_out    = bind_tensor(p + "ssm_out.weight");
            gdn_conv_[il].assign(conv_n, 0.0f);
            gdn_state_[il].assign(ssm_n, 0.0f);
            gdn_total += (conv_n + ssm_n) * sizeof(float);
        } else {
            L.wq          = bind_tensor(p + "attn_q.weight");
            L.wk          = bind_tensor(p + "attn_k.weight");
            L.wv          = bind_tensor(p + "attn_v.weight");
            L.wo          = bind_tensor(p + "attn_output.weight");
            L.attn_q_norm = bind_tensor(p + "attn_q_norm.weight");
            L.attn_k_norm = bind_tensor(p + "attn_k_norm.weight");
            k_cache_[il].assign(kv_bytes, 0);
            v_cache_[il].assign(kv_bytes, 0);
            kv_total += 2 * kv_bytes;
        }

        if (cfg_.is_dense) {
            L.ffn_gate = bind_tensor(p + "ffn_gate.weight");
            L.ffn_up   = bind_tensor(p + "ffn_up.weight");
            L.ffn_down = bind_tensor(p + "ffn_down.weight");
        } else {
            L.ffn_gate_inp       = bind_tensor(p + "ffn_gate_inp.weight");
            L.ffn_gate_exps      = bind_tensor(p + "ffn_gate_exps.weight");
            L.ffn_up_exps        = bind_tensor(p + "ffn_up_exps.weight");
            L.ffn_down_exps      = bind_tensor(p + "ffn_down_exps.weight");
            L.ffn_gate_inp_shexp = bind_tensor(p + "ffn_gate_inp_shexp.weight");
            L.ffn_gate_shexp     = bind_tensor(p + "ffn_gate_shexp.weight");
            L.ffn_up_shexp       = bind_tensor(p + "ffn_up_shexp.weight");
            L.ffn_down_shexp     = bind_tensor(p + "ffn_down_shexp.weight");
        }
    }

    logits_buf_.resize(static_cast<size_t>(cfg_.n_vocab));
    ensure_batch_buffers(64);

    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[Qwen35-CPU] Loaded %s (%s, L=%d, H=%d, heads=%d/%d d=%d rope=%d, gdn=%d/%d, vocab=%d, ctx=%d, kv=%s %.1f MiB, gdn=%.1f MiB, mmap=%.2f GiB, threads=%d) in %.2f s\n",
                sg_.get_str("general.name", arch).c_str(),
                cfg_.is_dense ? "dense" : "moe",
                cfg_.n_layers, cfg_.n_embd, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, cfg_.rope_dim,
                cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.n_vocab, cfg_.max_ctx,
                cfg_.kv_q8_0 ? "Q8_0" : "F16", kv_total / (1024.0 * 1024.0),
                gdn_total / (1024.0 * 1024.0),
                static_cast<double>(sg_.total_bytes()) / (1024.0 * 1024.0 * 1024.0),
                pool_.num_threads(), load_s);
}

void Qwen35CpuModel::reset_state() {
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (cfg_.is_recr[il]) {
            std::fill(gdn_conv_[il].begin(), gdn_conv_[il].end(), 0.0f);
            std::fill(gdn_state_[il].begin(), gdn_state_[il].end(), 0.0f);
        } else {
            std::fill(k_cache_[il].begin(), k_cache_[il].end(), 0);
            std::fill(v_cache_[il].begin(), v_cache_[il].end(), 0);
        }
    }
}

void Qwen35CpuModel::save_gdn_checkpoint(GdnCheckpoint & out, int pos) const {
    out.pos = pos;
    out.conv.resize(static_cast<size_t>(cfg_.n_layers));
    out.ssm.resize(static_cast<size_t>(cfg_.n_layers));
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (cfg_.is_recr[il]) {
            out.conv[il] = gdn_conv_[il];
            out.ssm[il]  = gdn_state_[il];
        }
    }
}

void Qwen35CpuModel::restore_gdn_checkpoint(const GdnCheckpoint & cp) {
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (cfg_.is_recr[il] && il < static_cast<int>(cp.conv.size()) && !cp.conv[il].empty()) {
            gdn_conv_[il]  = cp.conv[il];
            gdn_state_[il] = cp.ssm[il];
        }
    }
}

void Qwen35CpuModel::ensure_kv_capacity(int needed_pos) {
    if (needed_pos < kv_cap_) return;
    int new_cap = std::min(cfg_.max_ctx, std::max(needed_pos + 1, kv_cap_ * 2));
    if (needed_pos >= new_cap) {
        throw std::runtime_error("Qwen35CpuModel: position exceeds max_ctx");
    }
    const size_t new_bytes = static_cast<size_t>(new_cap) * cfg_.n_head_kv *
                             q35_kv_head_stride_bytes(cfg_.head_dim, cfg_.kv_q8_0);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!cfg_.is_recr[il]) {
            k_cache_[il].resize(new_bytes, 0);
            v_cache_[il].resize(new_bytes, 0);
        }
    }
    kv_cap_ = new_cap;
}

void Qwen35CpuModel::ensure_batch_buffers(int T) {
    if (T <= max_alloc_T_) return;
    max_alloc_T_ = T;

    const int H        = cfg_.n_embd;
    const int d_q      = cfg_.n_head * cfg_.head_dim;
    const int d_kv     = cfg_.n_head_kv * cfg_.head_dim;
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const int max_k    = std::max({H, d_q, cfg_.ssm_d_inner, cfg_.n_ff, cfg_.n_ff_exp, cfg_.n_ff_shexp});

    cur_.resize(static_cast<size_t>(T) * H);
    x_norm_.resize(static_cast<size_t>(T) * H);
    sub_out_.resize(static_cast<size_t>(T) * H);
    x_q8_.resize(static_cast<size_t>(T) * (max_k >> 5));
    x_q8k_.resize(static_cast<size_t>(T) * (max_k >> 8));

    qkv_raw_.resize(static_cast<size_t>(T) * conv_dim);
    qkv_conv_.resize(static_cast<size_t>(T) * conv_dim);
    z_buf_.resize(static_cast<size_t>(T) * cfg_.ssm_d_inner);
    alpha_buf_.resize(static_cast<size_t>(T) * cfg_.ssm_dt_rank);
    beta_buf_.resize(static_cast<size_t>(T) * cfg_.ssm_dt_rank);
    gdn_out_.resize(static_cast<size_t>(T) * cfg_.ssm_d_inner);

    q_full_.resize(static_cast<size_t>(T) * d_q * 2);
    q_buf_.resize(static_cast<size_t>(T) * d_q);
    gate_buf_.resize(static_cast<size_t>(T) * d_q);
    k_buf_.resize(static_cast<size_t>(T) * d_kv);
    v_buf_.resize(static_cast<size_t>(T) * d_kv);
    attn_out_.resize(static_cast<size_t>(T) * d_q);

    if (cfg_.is_dense) {
        ffn_g_.resize(static_cast<size_t>(T) * cfg_.n_ff);
        ffn_u_.resize(static_cast<size_t>(T) * cfg_.n_ff);
    } else {
        router_logits_.resize(static_cast<size_t>(T) * cfg_.n_expert);
        shexp_gate_.resize(static_cast<size_t>(T));
        top_ids_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
        top_w_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
        shexp_out_.resize(static_cast<size_t>(T) * H);
        ffn_g_.resize(static_cast<size_t>(T) * std::max(cfg_.n_ff_exp, cfg_.n_ff_shexp));
        ffn_u_.resize(static_cast<size_t>(T) * std::max(cfg_.n_ff_exp, cfg_.n_ff_shexp));
    }
}

void Qwen35CpuModel::run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits) {
    ensure_batch_buffers(T);
    int max_p = 0;
    for (int t = 0; t < T; ++t) max_p = std::max(max_p, positions[t]);
    ensure_kv_capacity(max_p);

    const int   H          = cfg_.n_embd;
    const int   d_q        = cfg_.n_head * cfg_.head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const float gdn_scale  = 1.0f / std::sqrt(static_cast<float>(cfg_.ssm_d_state));

    embedding_lookup(tok_embd_, tokens, cur_.data(), H, T);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const auto & L = layers_[il];

        rmsnorm(cur_.data(), static_cast<const float *>(L.attn_norm.ptr), x_norm_.data(), H, T, cfg_.rms_eps);

        if (L.is_recr) {
            const LinearSpec specs[4] = {
                {&L.wqkv,      qkv_raw_.data()},
                {&L.wqkv_gate, z_buf_.data()},
                {&L.ssm_alpha, alpha_buf_.data()},
                {&L.ssm_beta,  beta_buf_.data()},
            };
            linear_multi(pool_, specs, 4, x_norm_.data(), nullptr, nullptr, T);
            q35_gdn_gate_beta(alpha_buf_.data(),
                              static_cast<const float *>(L.ssm_dt.ptr),
                              static_cast<const float *>(L.ssm_a.ptr),
                              beta_buf_.data(), T, cfg_.ssm_dt_rank);
            q35_conv1d_silu_l2(pool_, qkv_raw_.data(), static_cast<const float *>(L.ssm_conv1d.ptr),
                               gdn_conv_[il].data(), qkv_conv_.data(),
                               T, cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.ssm_d_state, cfg_.rms_eps);
            q35_gdn_recurrence(pool_, qkv_conv_.data(), alpha_buf_.data(), beta_buf_.data(),
                               gdn_state_[il].data(), gdn_out_.data(),
                               T, cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.ssm_d_state, gdn_scale);
            q35_rms_norm_silu_gate(pool_, gdn_out_.data(), static_cast<const float *>(L.ssm_norm.ptr),
                                   z_buf_.data(), T, cfg_.ssm_dt_rank, cfg_.ssm_d_state, cfg_.rms_eps);
            linear(pool_, L.ssm_out, gdn_out_.data(), nullptr, nullptr, sub_out_.data(), T);
        } else {
            const LinearSpec specs[3] = {
                {&L.wq, q_full_.data()},
                {&L.wk, k_buf_.data()},
                {&L.wv, v_buf_.data()},
            };
            linear_multi(pool_, specs, 3, x_norm_.data(), nullptr, nullptr, T);
            q35_qgate_split_norm_rope(pool_, q_full_.data(), static_cast<const float *>(L.attn_q_norm.ptr),
                                      positions, q_buf_.data(), gate_buf_.data(),
                                      T, cfg_.n_head, cfg_.head_dim, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base);
            q35_k_norm_rope(pool_, k_buf_.data(), static_cast<const float *>(L.attn_k_norm.ptr),
                            positions, T, cfg_.n_head_kv, cfg_.head_dim, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base);
            q35_kv_write(pool_, k_buf_.data(), v_buf_.data(), k_cache_[il].data(), v_cache_[il].data(),
                         cfg_.kv_q8_0, positions, T, cfg_.n_head_kv, cfg_.head_dim);

            // Scale Q by 1/sqrt(head_dim) and reuse vectorized gemma4_gqa_attn (exact same KV cache layout!)
            scale_inplace(q_buf_.data(), static_cast<size_t>(T) * d_q, attn_scale);
            gemma4_gqa_attn(pool_, q_buf_.data(), k_cache_[il].data(), v_cache_[il].data(),
                            cfg_.kv_q8_0 ? KvCacheType::Q8_0 : KvCacheType::F16,
                            positions, attn_out_.data(),
                            T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, kv_cap_, 0);
            q35_sigmoid_gate_mul(pool_, attn_out_.data(), gate_buf_.data(), static_cast<size_t>(T) * d_q);
            linear(pool_, L.wo, attn_out_.data(), nullptr, nullptr, sub_out_.data(), T);
        }

        add_rmsnorm(cur_.data(), sub_out_.data(), static_cast<const float *>(L.post_attn_norm.ptr),
                    x_norm_.data(), H, T, cfg_.rms_eps);

        if (cfg_.is_dense) {
            if (T == 1 && (cfg_.n_ff % 256) == 0) {
                auto dt_q8  = [](DType dt) { return dt == DType::Q8_0 || dt == DType::Q5_0 || dt == DType::Q4_0 || dt == DType::IQ4_NL; };
                auto dt_q8k = [](DType dt) {
                    return dt == DType::Q6_K || dt == DType::Q5_K || dt == DType::Q4_K ||
                           dt == DType::Q3_K || dt == DType::Q2_K ||
                           dt == DType::IQ4_XS || dt == DType::IQ3_S || dt == DType::IQ3_XXS || dt == DType::IQ2_S;
                };
                const bool need_q8  = dt_q8(L.ffn_gate.dtype)  || dt_q8(L.ffn_up.dtype);
                const bool need_q8k = dt_q8k(L.ffn_gate.dtype) || dt_q8k(L.ffn_up.dtype);
                if (need_q8)  quantize_row_q8_0(x_norm_.data(), x_q8_.data(),  H);
                if (need_q8k) quantize_row_q8_k(x_norm_.data(), x_q8k_.data(), H);
                dense_ffn_swiglu_t1(pool_, L.ffn_gate, L.ffn_up, L.ffn_down,
                                    x_norm_.data(),
                                    need_q8 ? x_q8_.data() : nullptr,
                                    need_q8k ? x_q8k_.data() : nullptr,
                                    sub_out_.data());
            } else {
                const LinearSpec specs[2] = {{&L.ffn_gate, ffn_g_.data()}, {&L.ffn_up, ffn_u_.data()}};
                linear_multi(pool_, specs, 2, x_norm_.data(), nullptr, nullptr, T);
                silu_mul_mt(pool_, ffn_g_.data(), ffn_u_.data(), ffn_g_.data(), static_cast<size_t>(T) * cfg_.n_ff);
                linear(pool_, L.ffn_down, ffn_g_.data(), nullptr, nullptr, sub_out_.data(), T);
            }
        } else {
            // MoE: softmax top-k routed experts + sigmoid-gated shared expert via fused expert-grouped moe_ffn_with_shared!
            const LinearSpec rtr_specs[2] = {
                {&L.ffn_gate_inp,       router_logits_.data()},
                {&L.ffn_gate_inp_shexp, shexp_gate_.data()},
            };
            linear_multi(pool_, rtr_specs, 2, x_norm_.data(), nullptr, nullptr, T);
            moe_route_softmax(router_logits_.data(), top_ids_.data(), top_w_.data(),
                              T, cfg_.n_expert, cfg_.n_expert_used, true, cfg_.expert_weights_scale);
            moe_ffn_with_shared(pool_,
                                L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                                L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                                x_norm_.data(), nullptr, nullptr,
                                top_ids_.data(), top_w_.data(),
                                sub_out_.data(),
                                T, cfg_.n_expert_used, 0.0f, 0.0f,
                                shexp_gate_.data());
        }

        add_inplace(cur_.data(), sub_out_.data(), static_cast<size_t>(T) * H);
    }

    const float * last_h = cur_.data() + static_cast<size_t>(T - 1) * H;
    rmsnorm(last_h, static_cast<const float *>(output_norm_.ptr), x_norm_.data(), H, 1, cfg_.rms_eps);
    linear(pool_, output_, x_norm_.data(), nullptr, nullptr, out_logits, 1);
}

void Qwen35CpuModel::forward(const std::vector<int32_t> & tokens,
                             const std::vector<int32_t> & positions,
                             std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || positions.size() != tokens.size()) {
        throw std::runtime_error("Qwen35CpuModel::forward invalid T");
    }
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(tokens.data(), positions.data(), T, out_logits.data());
}

int32_t Qwen35CpuModel::forward_decode_greedy(int32_t token, int32_t pos) {
    run_forward(&token, &pos, 1, logits_buf_.data());
    return argmax_f32(logits_buf_.data(), cfg_.n_vocab);
}

void Qwen35CpuModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(&token, &pos, 1, out_logits.data());
}

} // namespace rookery::cpu
