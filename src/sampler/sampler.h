#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace rookery {

struct SamplingParams {
    float    temperature = 0.6f;
    float    top_p       = 0.95f;
    int      top_k       = 20;
    uint64_t seed        = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplingParams & p) : params_(p), rng_(p.seed) {}

    bool is_greedy() const { return params_.temperature <= 0.0f || params_.top_k == 1; }

    struct SparseDist {
        std::vector<int32_t> ids;
        std::vector<float>   probs; // sums to 1.0
        float prob_of(int32_t tok) const {
            for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == tok) return probs[i];
            return 0.0f;
        }
    };

    SparseDist filtered_distribution(const float * logits, int V) const {
        SparseDist sd;
        if (is_greedy()) {
            int32_t best = static_cast<int32_t>(std::max_element(logits, logits + V) - logits);
            sd.ids   = {best};
            sd.probs = {1.0f};
            return sd;
        }

        const int K = (params_.top_k > 0 && params_.top_k < V) ? params_.top_k : std::min(V, 64);
        std::vector<std::pair<float, int32_t>> top;
        top.reserve(K);
        float min_v = -1e30f;
        int   min_i = 0;
        for (int i = 0; i < V; ++i) {
            const float v = logits[i];
            if (static_cast<int>(top.size()) < K) {
                top.emplace_back(v, i);
                if (static_cast<int>(top.size()) == K) {
                    min_v = top[0].first; min_i = 0;
                    for (int j = 1; j < K; ++j) {
                        if (top[j].first < min_v) { min_v = top[j].first; min_i = j; }
                    }
                }
            } else if (v > min_v) {
                top[min_i] = {v, i};
                min_v = top[0].first; min_i = 0;
                for (int j = 1; j < K; ++j) {
                    if (top[j].first < min_v) { min_v = top[j].first; min_i = j; }
                }
            }
        }
        std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

        const float inv_t = 1.0f / params_.temperature;
        const float max_l = top[0].first;
        std::vector<float> probs(K);
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            probs[i] = std::exp((top[i].first - max_l) * inv_t);
            sum += probs[i];
        }
        for (float & p : probs) p /= sum;

        int cutoff = K;
        if (params_.top_p > 0.0f && params_.top_p < 1.0f) {
            float cum = 0.0f;
            for (int i = 0; i < K; ++i) {
                cum += probs[i];
                if (cum >= params_.top_p) { cutoff = i + 1; break; }
            }
        }
        float renorm = 0.0f;
        for (int i = 0; i < cutoff; ++i) renorm += probs[i];
        const float inv_r = (renorm > 0.0f) ? (1.0f / renorm) : 1.0f;
        sd.ids.resize(cutoff);
        sd.probs.resize(cutoff);
        for (int i = 0; i < cutoff; ++i) {
            sd.ids[i]   = top[i].second;
            sd.probs[i] = probs[i] * inv_r;
        }
        return sd;
    }

    int32_t sample_dist(const SparseDist & sd) {
        if (sd.ids.size() <= 1) return sd.ids.empty() ? 0 : sd.ids[0];
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const float r = u(rng_);
        float acc = 0.0f;
        for (size_t i = 0; i < sd.ids.size(); ++i) {
            acc += sd.probs[i];
            if (r <= acc) return sd.ids[i];
        }
        return sd.ids.back();
    }

    // Exact lossless Leviathan et al. speculative verification:
    // Returns true if `draft_tok` is accepted; otherwise returns false and writes resampled token to `corrected_tok`!
    bool verify_speculative(int32_t draft_tok,
                            const SparseDist & target_p,
                            const SparseDist & draft_q,
                            int32_t & corrected_tok) {
        const float p = target_p.prob_of(draft_tok);
        const float q = draft_q.prob_of(draft_tok);
        if (q > 0.0f && p >= q) return true;
        if (q > 0.0f && p > 0.0f) {
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            if (u(rng_) <= (p / q)) return true;
        }
        // Reject: sample from norm(max(0, target_p - draft_q))
        SparseDist residual;
        float rsum = 0.0f;
        for (size_t i = 0; i < target_p.ids.size(); ++i) {
            float diff = target_p.probs[i] - draft_q.prob_of(target_p.ids[i]);
            if (diff > 0.0f) {
                residual.ids.push_back(target_p.ids[i]);
                residual.probs.push_back(diff);
                rsum += diff;
            }
        }
        if (rsum > 0.0f) {
            const float inv = 1.0f / rsum;
            for (float & v : residual.probs) v *= inv;
            corrected_tok = sample_dist(residual);
        } else {
            corrected_tok = sample_dist(target_p);
        }
        return false;
    }

    int32_t sample(const std::vector<float> & logits) {
        return sample_dist(filtered_distribution(logits.data(), static_cast<int>(logits.size())));
    }

private:
    SamplingParams params_;
    std::mt19937_64 rng_;
};

} // namespace rookery
