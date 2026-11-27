#include "core/device.h"
#include "cpu/bailingmoe3_cpu.h"
#include "cpu/gemma4_cpu.h"
#include "cpu/qwen2_cpu.h"
#include "cpu/qwen35_cpu.h"
#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "gguf/sharded.h"
#ifdef ROOKERY_ENABLE_CUDA
#include "models/bailingmoe3.h"
#include "models/gemma4.h"
#include "models/glm5_next.h"
#include "models/inkling2_loop.h"
#include "models/maxwell_moe.h"
#include "models/qwen35_moe.h"
#include "models/qwen4exp.h"
#include "models/raon1_mini.h"
#include "models/sensenova_loop.h"
#endif
#include "sampler/sampler.h"
#include "server/chat_template.h"
#include "tokenizer/bpe.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<float> parse_csv_floats(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
    return out;
}

static std::vector<int> parse_csv_ints(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

// Returns the largest prefix length <= max_len that does NOT split a multi-byte UTF-8 sequence.
static size_t utf8_safe_prefix_len(const std::string & s, size_t max_len) {
    size_t n = std::min(max_len, s.size());
    if (n == 0) return 0;
    // Scan back up to 3 bytes to see if we're ending in the middle of a multi-byte UTF-8 lead/continuation sequence.
    size_t i = n;
    while (i > 0 && (n - i) < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
        --i;
    }
    if (i > 0) {
        unsigned char lead = static_cast<unsigned char>(s[i - 1]);
        size_t need = 1;
        if ((lead & 0xE0) == 0xC0)      need = 2;
        else if ((lead & 0xF0) == 0xE0) need = 3;
        else if ((lead & 0xF8) == 0xF0) need = 4;
        if (need > 1 && (n - (i - 1)) < need) {
            return i - 1;
        }
    }
    return n;
}

static std::string safe_json_dump(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

static std::vector<std::string> parse_stop_strings(const json & j) {
    std::vector<std::string> out;
    if (!j.contains("stop")) return out;
    const auto & s = j["stop"];
    if (s.is_string()) {
        auto v = s.get<std::string>();
        if (!v.empty()) out.push_back(std::move(v));
    } else if (s.is_array()) {
        for (const auto & item : s) {
            if (item.is_string()) {
                auto v = item.get<std::string>();
                if (!v.empty()) out.push_back(std::move(v));
            }
        }
    }
    return out;
}

static size_t find_earliest_stop(const std::string & s, const std::vector<std::string> & stops) {
    size_t best = std::string::npos;
    for (const auto & st : stops) {
        size_t pos = s.find(st);
        if (pos < best) best = pos;
    }
    return best;
}

// Unified runtime interface over GPU models and CPU backends.
struct ServerEngine {
    std::string arch;
#ifdef ROOKERY_ENABLE_CUDA
    std::unique_ptr<rookery::MaxwellMoeModel>   maxwell;
    std::unique_ptr<rookery::Glm5NextModel>     glm5;
    std::unique_ptr<rookery::Inkling2LoopModel> inkling2;
    std::unique_ptr<rookery::Qwen35MoeModel>    qwen35;
    std::unique_ptr<rookery::Qwen4ExpModel>     qwen4;
    std::unique_ptr<rookery::BailingMoe3Model>  bailing3;
    std::unique_ptr<rookery::Gemma4Model>       gemma4;
    std::unique_ptr<rookery::Raon1MiniModel>     raon1;
    std::unique_ptr<rookery::SenseNovaLoopModel> sensenova;
#endif
    std::unique_ptr<rookery::cpu::BailingMoe3CpuModel> bailing3_cpu;
    std::unique_ptr<rookery::cpu::Gemma4CpuModel>      gemma4_cpu;
    std::unique_ptr<rookery::cpu::Qwen2CpuModel>       qwen2_cpu;
    std::unique_ptr<rookery::cpu::Qwen35CpuModel>      qwen35_cpu;

    bool is_glm5() const {
#ifdef ROOKERY_ENABLE_CUDA
        return glm5 != nullptr;
#else
        return false;
#endif
    }
    bool is_inkling2() const {
#ifdef ROOKERY_ENABLE_CUDA
        return inkling2 != nullptr;
#else
        return false;
#endif
    }
    bool is_qwen35() const {
        if (qwen35_cpu != nullptr) return true;
#ifdef ROOKERY_ENABLE_CUDA
        return qwen35 != nullptr;
#else
        return false;
#endif
    }
    bool is_qwen4() const {
#ifdef ROOKERY_ENABLE_CUDA
        return qwen4 != nullptr;
#else
        return false;
#endif
    }
    bool is_bailing3() const {
        if (bailing3_cpu != nullptr) return true;
#ifdef ROOKERY_ENABLE_CUDA
        return bailing3 != nullptr;
#else
        return false;
#endif
    }
    bool is_gemma4() const {
        if (gemma4_cpu != nullptr) return true;
#ifdef ROOKERY_ENABLE_CUDA
        return gemma4 != nullptr;
#else
        return false;
#endif
    }
    bool is_raon1() const {
#ifdef ROOKERY_ENABLE_CUDA
        return raon1 != nullptr;
#else
        return false;
#endif
    }
    bool is_qwen2() const { return qwen2_cpu != nullptr; }
    bool is_sensenova() const {
#ifdef ROOKERY_ENABLE_CUDA
        return sensenova != nullptr;
#else
        return false;
#endif
    }
    bool has_state_checkpoints() const { return is_glm5() || is_inkling2() || is_qwen35() || is_qwen4() || is_bailing3() || is_raon1() || is_sensenova(); }

    int max_ctx() const {
        if (bailing3_cpu) return bailing3_cpu->config().max_ctx;
        if (gemma4_cpu)   return gemma4_cpu->config().max_ctx;
        if (qwen2_cpu)    return qwen2_cpu->config().max_ctx;
        if (qwen35_cpu)   return qwen35_cpu->config().max_ctx;
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->config().max_ctx;
        if (inkling2) return inkling2->config().max_ctx;
        if (qwen35)   return qwen35->config().max_ctx;
        if (qwen4)    return qwen4->config().max_ctx;
        if (bailing3) return bailing3->config().max_ctx;
        if (gemma4)   return gemma4->config().max_ctx;
        if (raon1)     return raon1->config().max_ctx;
        if (sensenova) return sensenova->config().max_ctx;
        return maxwell->config().max_ctx;
#else
        return 0;
#endif
    }
    int n_embd() const {
        if (bailing3_cpu) return bailing3_cpu->config().n_embd;
        if (gemma4_cpu)   return gemma4_cpu->config().n_embd;
        if (qwen2_cpu)    return qwen2_cpu->config().n_embd;
        if (qwen35_cpu)   return qwen35_cpu->config().n_embd;
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->config().n_embd;
        if (inkling2) return inkling2->config().n_embd;
        if (qwen35)   return qwen35->config().n_embd;
        if (qwen4)    return qwen4->config().n_embd;
        if (bailing3) return bailing3->config().n_embd;
        if (gemma4)   return gemma4->config().n_embd;
        if (raon1)     return raon1->config().n_embd;
        if (sensenova) return sensenova->config().n_embd;
        return maxwell->config().n_embd;
#else
        return 0;
#endif
    }
    int hc_mult() const {
        if (bailing3_cpu || gemma4_cpu || qwen2_cpu || qwen35_cpu) return 1;
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->config().hc_mult;
        if (qwen4)    return qwen4->config().hc_mult;
        if (raon1)    return raon1->config().hc_mult;
        if (inkling2 || qwen35 || bailing3 || gemma4 || sensenova) return 1;
        return maxwell->config().hc_mult;
#else
        return 1;
#endif
    }
    size_t prefill_chunk() const {
        if (bailing3_cpu || gemma4_cpu || qwen2_cpu || qwen35_cpu) return 256;
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5 || inkling2) return 256;
        if (bailing3 && bailing3->config().n_expert >= 256) return 1024;
#endif
        return 512;
    }
    bool has_mtp() const {
        if (bailing3_cpu) return bailing3_cpu->has_mtp();
        if (gemma4_cpu)   return gemma4_cpu->has_mtp();
        if (qwen2_cpu || qwen35_cpu) return false;
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->has_mtp();
        if (inkling2 || qwen4 || gemma4 || raon1) return false;
        if (qwen35)   return qwen35->has_mtp();
        if (bailing3) return bailing3->has_mtp();
        if (sensenova) return sensenova->has_mtp();
        return maxwell->has_mtp();
#else
        return false;
#endif
    }

    std::vector<int> layer_devices() const {
        if (bailing3_cpu) return bailing3_cpu->layer_devices();
        if (gemma4_cpu)   return gemma4_cpu->layer_devices();
        if (qwen2_cpu)    return qwen2_cpu->layer_devices();
        if (qwen35_cpu)   return qwen35_cpu->layer_devices();
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->layer_devices();
        if (inkling2) return inkling2->layer_devices();
        if (qwen35)   return qwen35->layer_devices();
        if (qwen4)    return qwen4->layer_devices();
        if (bailing3) return bailing3->layer_devices();
        if (gemma4)   return gemma4->layer_devices();
        if (raon1)     return raon1->layer_devices();
        if (sensenova) return sensenova->layer_devices();
        return maxwell->layer_devices();
#else
        return {};
#endif
    }
    void set_layer_hook(rookery::LayerHook * h) {
        if (bailing3_cpu) { bailing3_cpu->set_layer_hook(h); return; }
        if (gemma4_cpu)   { gemma4_cpu->set_layer_hook(h);   return; }
        if (qwen2_cpu)    { qwen2_cpu->set_layer_hook(h);    return; }
        if (qwen35_cpu)   { qwen35_cpu->set_layer_hook(h);   return; }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->set_layer_hook(h);
        else if (inkling2) inkling2->set_layer_hook(h);
        else if (qwen35)   qwen35->set_layer_hook(h);
        else if (qwen4)    qwen4->set_layer_hook(h);
        else if (bailing3) bailing3->set_layer_hook(h);
        else if (gemma4)   gemma4->set_layer_hook(h);
        else if (raon1)     raon1->set_layer_hook(h);
        else if (sensenova) sensenova->set_layer_hook(h);
        else               maxwell->set_layer_hook(h);
#endif
    }
    void warmup_decode_graphs() {
        if (bailing3_cpu) { bailing3_cpu->warmup_decode_graphs(); return; }
        if (gemma4_cpu)   { gemma4_cpu->warmup_decode_graphs();   return; }
        if (qwen2_cpu)    { qwen2_cpu->warmup_decode_graphs();    return; }
        if (qwen35_cpu)   { qwen35_cpu->warmup_decode_graphs();   return; }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->warmup_decode_graphs();
        else if (inkling2) inkling2->warmup_decode_graphs();
        else if (qwen35)   qwen35->warmup_decode_graphs();
        else if (qwen4)    qwen4->warmup_decode_graphs();
        else if (bailing3) bailing3->warmup_decode_graphs();
        else if (gemma4)   gemma4->warmup_decode_graphs();
        else if (raon1)     raon1->warmup_decode_graphs();
        else if (sensenova) sensenova->warmup_decode_graphs();
        else               maxwell->warmup_decode_graphs();
#endif
    }
    void reset_state() {
        if (bailing3_cpu) { bailing3_cpu->reset_state(); return; }
        if (gemma4_cpu)   { gemma4_cpu->reset_state();   return; }
        if (qwen2_cpu)    { qwen2_cpu->reset_state();    return; }
        if (qwen35_cpu)   { qwen35_cpu->reset_state();   return; }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->reset_state();
        else if (inkling2) inkling2->reset_state();
        else if (qwen35)   qwen35->reset_state();
        else if (qwen4)    qwen4->reset_state();
        else if (bailing3) bailing3->reset_state();
        else if (gemma4)   gemma4->reset_state();
        else if (raon1)     raon1->reset_state();
        else if (sensenova) sensenova->reset_state();
#endif
    }
    void forward(const std::vector<int32_t> & t, const std::vector<int32_t> & p, std::vector<float> & l, bool compute_logits = true) {
        if (bailing3_cpu) { bailing3_cpu->forward(t, p, l); return; }
        if (gemma4_cpu)   { gemma4_cpu->forward(t, p, l, compute_logits); return; }
        if (qwen2_cpu)    { qwen2_cpu->forward(t, p, l);    return; }
        if (qwen35_cpu)   { qwen35_cpu->forward(t, p, l);   return; }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->forward(t, p, l);
        else if (inkling2) inkling2->forward(t, p, l);
        else if (qwen35)   qwen35->forward(t, p, l);
        else if (qwen4)    qwen4->forward(t, p, l);
        else if (bailing3) bailing3->forward(t, p, l);
        else if (gemma4)   gemma4->forward(t, p, l, compute_logits);
        else if (raon1)     raon1->forward(t, p, l, compute_logits);
        else if (sensenova) sensenova->forward(t, p, l, compute_logits);
        else               maxwell->forward(t, p, l);
#else
        (void) compute_logits;
#endif
    }
    int32_t forward_decode_greedy(int32_t t, int32_t p) {
        if (bailing3_cpu) return bailing3_cpu->forward_decode_greedy(t, p);
        if (gemma4_cpu)   return gemma4_cpu->forward_decode_greedy(t, p);
        if (qwen2_cpu)    return qwen2_cpu->forward_decode_greedy(t, p);
        if (qwen35_cpu)   return qwen35_cpu->forward_decode_greedy(t, p);
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->forward_decode_greedy(t, p);
        if (inkling2) return inkling2->forward_decode_greedy(t, p);
        if (qwen35)   return qwen35->forward_decode_greedy(t, p);
        if (qwen4)    return qwen4->forward_decode_greedy(t, p);
        if (bailing3) return bailing3->forward_decode_greedy(t, p);
        if (gemma4)   return gemma4->forward_decode_greedy(t, p);
        if (raon1)     return raon1->forward_decode_greedy(t, p);
        if (sensenova) return sensenova->forward_decode_greedy(t, p);
        return maxwell->forward_decode_greedy(t, p);
#else
        return 0;
#endif
    }
    void forward_decode_logits(int32_t t, int32_t p, std::vector<float> & l) {
        if (bailing3_cpu) { bailing3_cpu->forward_decode_logits(t, p, l); return; }
        if (gemma4_cpu)   { gemma4_cpu->forward_decode_logits(t, p, l);   return; }
        if (qwen2_cpu)    { qwen2_cpu->forward_decode_logits(t, p, l);    return; }
        if (qwen35_cpu)   { qwen35_cpu->forward_decode_logits(t, p, l);   return; }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->forward_decode_logits(t, p, l);
        else if (inkling2) inkling2->forward_decode_logits(t, p, l);
        else if (qwen35)   qwen35->forward_decode_logits(t, p, l);
        else if (qwen4)    qwen4->forward_decode_logits(t, p, l);
        else if (bailing3) bailing3->forward_decode_logits(t, p, l);
        else if (gemma4)   gemma4->forward_decode_logits(t, p, l);
        else if (raon1)     raon1->forward_decode_logits(t, p, l);
        else if (sensenova) sensenova->forward_decode_logits(t, p, l);
        else               maxwell->forward_decode_logits(t, p, l);
#endif
    }
    int32_t mtp_prefill_chunk(const std::vector<int32_t> & nt, const std::vector<int32_t> & p) {
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)     return glm5->mtp_prefill_chunk(nt, p);
        if (qwen35)   return qwen35->mtp_prefill_chunk(nt, p);
        if (bailing3) return bailing3->mtp_prefill_chunk(nt, p);
        if (sensenova) return sensenova->mtp_prefill_chunk(nt, p);
        return maxwell->mtp_prefill_chunk(nt, p);
#else
        (void) nt; (void) p; return 0;
#endif
    }
    std::vector<int32_t> cached_tokens;
#ifdef ROOKERY_ENABLE_CUDA
    std::vector<rookery::Glm5NextModel::KdaCheckpoint>      kda_checkpoints;
    std::vector<rookery::Inkling2LoopModel::ConvCheckpoint> conv_checkpoints;
    std::vector<rookery::Qwen35MoeModel::GdnCheckpoint>     gdn_checkpoints;
    std::vector<rookery::Qwen4ExpModel::StateCheckpoint>    qwen4_checkpoints;
    std::vector<rookery::BailingMoe3Model::KdaCheckpoint>   bailing3_checkpoints;
    std::vector<rookery::Raon1MiniModel::RdaCheckpoint>     raon1_checkpoints;
    float   cached_raon_effort_alpha = -1.0f;
    std::vector<rookery::SenseNovaLoopModel::GdnCheckpoint> sensenova_checkpoints;
#endif
    std::vector<rookery::cpu::BailingMoe3CpuModel::KdaCheckpoint> bailing3_cpu_checkpoints;
    std::vector<rookery::cpu::Qwen35CpuModel::GdnCheckpoint>      qwen35_cpu_checkpoints;
    size_t  mtp_kv_valid_len  = 0;
    int32_t mtp_last_next_tok = -1;
    bool    has_cached_engram_params = false;
    std::vector<rookery::EngramRuntimeParams> cached_engram_params;
    int     engram_seq_id            = 0;

    void prune_state_checkpoints_after(int pos) {
#ifdef ROOKERY_ENABLE_CUDA
        kda_checkpoints.erase(
            std::remove_if(kda_checkpoints.begin(), kda_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            kda_checkpoints.end());
        conv_checkpoints.erase(
            std::remove_if(conv_checkpoints.begin(), conv_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            conv_checkpoints.end());
        gdn_checkpoints.erase(
            std::remove_if(gdn_checkpoints.begin(), gdn_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            gdn_checkpoints.end());
        qwen4_checkpoints.erase(
            std::remove_if(qwen4_checkpoints.begin(), qwen4_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            qwen4_checkpoints.end());
        bailing3_checkpoints.erase(
            std::remove_if(bailing3_checkpoints.begin(), bailing3_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            bailing3_checkpoints.end());
        raon1_checkpoints.erase(
            std::remove_if(raon1_checkpoints.begin(), raon1_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            raon1_checkpoints.end());
        sensenova_checkpoints.erase(
            std::remove_if(sensenova_checkpoints.begin(), sensenova_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            sensenova_checkpoints.end());
#endif
        bailing3_cpu_checkpoints.erase(
            std::remove_if(bailing3_cpu_checkpoints.begin(), bailing3_cpu_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            bailing3_cpu_checkpoints.end());
        qwen35_cpu_checkpoints.erase(
            std::remove_if(qwen35_cpu_checkpoints.begin(), qwen35_cpu_checkpoints.end(),
                           [pos](const auto & cp) { return cp.pos > pos; }),
            qwen35_cpu_checkpoints.end());
    }

    void save_state_checkpoint(int pos) {
        if (pos <= 0) return;
        if (bailing3_cpu) {
            for (auto & cp : bailing3_cpu_checkpoints) {
                if (cp.pos == pos) { bailing3_cpu->save_kda_checkpoint(cp, pos); return; }
            }
            if (bailing3_cpu_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < bailing3_cpu_checkpoints.size(); ++i) {
                    int gap = bailing3_cpu_checkpoints[i + 1].pos - bailing3_cpu_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                bailing3_cpu_checkpoints.erase(bailing3_cpu_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::cpu::BailingMoe3CpuModel::KdaCheckpoint cp;
            bailing3_cpu->save_kda_checkpoint(cp, pos);
            auto it = std::lower_bound(bailing3_cpu_checkpoints.begin(), bailing3_cpu_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            bailing3_cpu_checkpoints.insert(it, std::move(cp));
            return;
        }
        if (qwen35_cpu) {
            for (auto & cp : qwen35_cpu_checkpoints) {
                if (cp.pos == pos) { qwen35_cpu->save_gdn_checkpoint(cp, pos); return; }
            }
            if (qwen35_cpu_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < qwen35_cpu_checkpoints.size(); ++i) {
                    int gap = qwen35_cpu_checkpoints[i + 1].pos - qwen35_cpu_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                qwen35_cpu_checkpoints.erase(qwen35_cpu_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::cpu::Qwen35CpuModel::GdnCheckpoint cp;
            qwen35_cpu->save_gdn_checkpoint(cp, pos);
            auto it = std::lower_bound(qwen35_cpu_checkpoints.begin(), qwen35_cpu_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            qwen35_cpu_checkpoints.insert(it, std::move(cp));
            return;
        }
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5) {
            for (auto & cp : kda_checkpoints) {
                if (cp.pos == pos) { glm5->save_kda_checkpoint(cp, pos); return; }
            }
            if (kda_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < kda_checkpoints.size(); ++i) {
                    int gap = kda_checkpoints[i + 1].pos - kda_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                kda_checkpoints.erase(kda_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::Glm5NextModel::KdaCheckpoint cp;
            glm5->save_kda_checkpoint(cp, pos);
            auto it = std::lower_bound(kda_checkpoints.begin(), kda_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            kda_checkpoints.insert(it, std::move(cp));
        } else if (inkling2) {
            for (auto & cp : conv_checkpoints) {
                if (cp.pos == pos) { inkling2->save_conv_checkpoint(cp, pos); return; }
            }
            if (conv_checkpoints.size() >= 64) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < conv_checkpoints.size(); ++i) {
                    int gap = conv_checkpoints[i + 1].pos - conv_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                conv_checkpoints.erase(conv_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::Inkling2LoopModel::ConvCheckpoint cp;
            inkling2->save_conv_checkpoint(cp, pos);
            auto it = std::lower_bound(conv_checkpoints.begin(), conv_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            conv_checkpoints.insert(it, std::move(cp));
        } else if (qwen35) {
            for (auto & cp : gdn_checkpoints) {
                if (cp.pos == pos) { qwen35->save_gdn_checkpoint(cp, pos); return; }
            }
            if (gdn_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < gdn_checkpoints.size(); ++i) {
                    int gap = gdn_checkpoints[i + 1].pos - gdn_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                gdn_checkpoints.erase(gdn_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::Qwen35MoeModel::GdnCheckpoint cp;
            qwen35->save_gdn_checkpoint(cp, pos);
            auto it = std::lower_bound(gdn_checkpoints.begin(), gdn_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            gdn_checkpoints.insert(it, std::move(cp));
        } else if (qwen4) {
            for (auto & cp : qwen4_checkpoints) {
                if (cp.pos == pos) { qwen4->save_checkpoint(cp, pos); return; }
            }
            if (qwen4_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < qwen4_checkpoints.size(); ++i) {
                    int gap = qwen4_checkpoints[i + 1].pos - qwen4_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                qwen4_checkpoints.erase(qwen4_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::Qwen4ExpModel::StateCheckpoint cp;
            qwen4->save_checkpoint(cp, pos);
            auto it = std::lower_bound(qwen4_checkpoints.begin(), qwen4_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            qwen4_checkpoints.insert(it, std::move(cp));
        } else if (bailing3) {
            for (auto & cp : bailing3_checkpoints) {
                if (cp.pos == pos) { bailing3->save_kda_checkpoint(cp, pos); return; }
            }
            if (bailing3_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < bailing3_checkpoints.size(); ++i) {
                    int gap = bailing3_checkpoints[i + 1].pos - bailing3_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                bailing3_checkpoints.erase(bailing3_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::BailingMoe3Model::KdaCheckpoint cp;
            bailing3->save_kda_checkpoint(cp, pos);
            auto it = std::lower_bound(bailing3_checkpoints.begin(), bailing3_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            bailing3_checkpoints.insert(it, std::move(cp));
        } else if (raon1) {
            for (auto & cp : raon1_checkpoints) {
                if (cp.pos == pos) { raon1->save_rda_checkpoint(cp, pos); return; }
            }
            if (raon1_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < raon1_checkpoints.size(); ++i) {
                    int gap = raon1_checkpoints[i + 1].pos - raon1_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                raon1_checkpoints.erase(raon1_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::Raon1MiniModel::RdaCheckpoint cp;
            raon1->save_rda_checkpoint(cp, pos);
            auto it = std::lower_bound(raon1_checkpoints.begin(), raon1_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            raon1_checkpoints.insert(it, std::move(cp));
        } else if (sensenova) {
            for (auto & cp : sensenova_checkpoints) {
                if (cp.pos == pos) { sensenova->save_gdn_checkpoint(cp, pos); return; }
            }
            if (sensenova_checkpoints.size() >= 32) {
                size_t drop_idx = 0; int min_gap = INT32_MAX;
                for (size_t i = 1; i + 1 < sensenova_checkpoints.size(); ++i) {
                    int gap = sensenova_checkpoints[i + 1].pos - sensenova_checkpoints[i - 1].pos;
                    if (gap < min_gap) { min_gap = gap; drop_idx = i; }
                }
                sensenova_checkpoints.erase(sensenova_checkpoints.begin() + static_cast<ptrdiff_t>(drop_idx));
            }
            rookery::SenseNovaLoopModel::GdnCheckpoint cp;
            sensenova->save_gdn_checkpoint(cp, pos);
            auto it = std::lower_bound(sensenova_checkpoints.begin(), sensenova_checkpoints.end(), pos,
                                       [](const auto & a, int p) { return a.pos < p; });
            sensenova_checkpoints.insert(it, std::move(cp));
        }
#endif
    }

    bool supports_sample_mtp() const {
#ifdef ROOKERY_ENABLE_CUDA
        return (glm5 && glm5->has_mtp()) || (qwen35 && qwen35->has_mtp()) ||
               (bailing3 && bailing3->has_mtp()) || (sensenova && sensenova->has_mtp());
#else
        return false;
#endif
    }

    void mtp_prefill_chunk_logits(const std::vector<int32_t> & nt, const std::vector<int32_t> & p, std::vector<float> & l) {
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)          glm5->mtp_prefill_chunk_logits(nt, p, l);
        else if (qwen35)   qwen35->mtp_prefill_chunk_logits(nt, p, l);
        else if (sensenova) sensenova->mtp_prefill_chunk_logits(nt, p, l);
        else               bailing3->mtp_prefill_chunk_logits(nt, p, l);
#else
        (void) nt; (void) p; (void) l;
#endif
    }

    void decode_mtp_greedy_step(int32_t & ct, int32_t & dt, int32_t & cp,
                                std::vector<int32_t> & out, rookery::MtpStats & st) {
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)           glm5->decode_mtp_greedy_step(ct, dt, cp, out, st);
        else if (qwen35)    qwen35->decode_mtp_greedy_step(ct, dt, cp, out, st);
        else if (bailing3)  bailing3->decode_mtp_greedy_step(ct, dt, cp, out, st);
        else if (sensenova) sensenova->decode_mtp_greedy_step(ct, dt, cp, out, st);
        else                maxwell->decode_mtp_greedy_step(ct, dt, cp, out, st);
#else
        (void) ct; (void) dt; (void) cp; (void) out; (void) st;
#endif
    }

    void decode_mtp_sample_step(int32_t & ct, int32_t & dt, rookery::Sampler::SparseDist & dq,
                                int32_t & cp, rookery::Sampler & sampler,
                                std::vector<int32_t> & out, rookery::MtpStats & st) {
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)           glm5->decode_mtp_sample_step(ct, dt, dq, cp, sampler, out, st);
        else if (qwen35)    qwen35->decode_mtp_sample_step(ct, dt, dq, cp, sampler, out, st);
        else if (sensenova) sensenova->decode_mtp_sample_step(ct, dt, dq, cp, sampler, out, st);
        else                bailing3->decode_mtp_sample_step(ct, dt, dq, cp, sampler, out, st);
#else
        (void) ct; (void) dt; (void) dq; (void) cp; (void) sampler; (void) out; (void) st;
#endif
    }

    void rollback_mtp_1tok() {
#ifdef ROOKERY_ENABLE_CUDA
        if (glm5)           glm5->rollback_kda_states();
        else if (qwen35)    qwen35->rollback_gdn_states();
        else if (bailing3)  bailing3->rollback_kda_states();
        else if (sensenova) sensenova->rollback_gdn_states();
#endif
    }
};

struct GenResult {
    std::vector<int32_t> tokens;
    bool   hit_max_tokens   = false;
    size_t reused_tokens    = 0;
    size_t prefilled_tokens = 0;
    double pp_ms            = 0.0;
    double tg_ms            = 0.0;
};

static std::vector<rookery::EngramRuntimeParams> parse_engram_params(
    const json & j,
    const std::vector<rookery::EngramSidecar *> & engrams) {
    std::vector<rookery::EngramRuntimeParams> eps;
    eps.reserve(engrams.size());
    for (size_t i = 0; i < engrams.size(); ++i) {
        rookery::EngramRuntimeParams ep = engrams[i] ? engrams[i]->default_params() : rookery::EngramRuntimeParams{};
        const char * k_sc = (i == 0) ? "engram_scale" : "engram_scale2";
        const char * k_gb = (i == 0) ? "engram_gate_bias" : "engram_gate_bias2";
        const char * k_dc = (i == 0) ? "engram_delta_clip" : "engram_delta_clip2";
        if (j.contains(k_sc) && j[k_sc].is_number()) ep.scale = j[k_sc].get<float>();
        if (j.contains(k_gb) && j[k_gb].is_number()) ep.gate_bias = j[k_gb].get<float>();
        if (j.contains(k_dc) && j[k_dc].is_number()) ep.delta_clip_ratio = j[k_dc].get<float>();
        eps.push_back(ep);
    }
    return eps;
}

static GenResult run_generation(
    ServerEngine & engine,
    const rookery::BpeTokenizer & tokenizer,
    const std::vector<rookery::EngramSidecar *> & engrams,
    const std::vector<rookery::EngramRuntimeParams> & engram_params,
    const std::vector<int32_t> & prompt_ids,
    const rookery::SamplingParams & sp,
    int max_tokens,
    bool use_mtp,
    const std::function<bool(int32_t)> & on_token,
    const std::function<bool()> & on_prefill_progress = nullptr) {

    GenResult gr;
    const size_t chunk = engine.prefill_chunk();
    rookery::Sampler sampler(sp);
    const bool run_mtp = use_mtp && engine.has_mtp() && !prompt_ids.empty() &&
                         (sampler.is_greedy() || engine.supports_sample_mtp());

    // Prompt Prefix Cache: find longest matching prefix with cached KV (+ recurrent/Engram checkpoints).
    // Safe with Engram whenever runtime params (scale, gate_bias, delta_clip) match the cached request,
    // because EngramSeqState::rollback_to(reuse_pos) keeps tokenizer/hash state exact!
    bool engram_cache_ok = engrams.empty();
    if (!engrams.empty() && engine.has_cached_engram_params &&
        engine.cached_engram_params.size() == engram_params.size()) {
        engram_cache_ok = true;
        for (size_t i = 0; i < engram_params.size(); ++i) {
            if (engine.cached_engram_params[i].scale != engram_params[i].scale ||
                engine.cached_engram_params[i].gate_bias != engram_params[i].gate_bias ||
                engine.cached_engram_params[i].delta_clip_ratio != engram_params[i].delta_clip_ratio) {
                engram_cache_ok = false;
                break;
            }
        }
    }
    size_t common = 0;
#ifdef ROOKERY_ENABLE_CUDA
    const bool raon_cache_ok = !engine.raon1 ||
        (engine.cached_raon_effort_alpha == engine.raon1->reasoning_effort_alpha());
#else
    const bool raon_cache_ok = true;
#endif
    if (engram_cache_ok && raon_cache_ok && !prompt_ids.empty()) {
        const size_t lim = std::min(engine.cached_tokens.size(), prompt_ids.size());
        while (common < lim && engine.cached_tokens[common] == prompt_ids[common]) ++common;
    }

    // Always run at least 1 token to produce fresh logits; for MTP, position p also depends on token[p+1].
    size_t max_reuse = 0;
    if (!prompt_ids.empty() && common > 0) {
        max_reuse = std::min(common, prompt_ids.size() - 1);
        if (run_mtp) {
            const bool next_matches_mtp =
                (common == engine.cached_tokens.size() && common < prompt_ids.size() &&
                 engine.mtp_last_next_tok >= 0 && prompt_ids[common] == engine.mtp_last_next_tok);
            const size_t mtp_lim = next_matches_mtp ? common : ((common >= 2) ? (common - 1) : 0);
            max_reuse = std::min(max_reuse, std::min(mtp_lim, engine.mtp_kv_valid_len));
        }
    }

    size_t reuse_pos = 0;
    if (engine.bailing3_cpu) {
        const rookery::cpu::BailingMoe3CpuModel::KdaCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.bailing3_cpu_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.bailing3_cpu->restore_kda_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.qwen35_cpu) {
        const rookery::cpu::Qwen35CpuModel::GdnCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.qwen35_cpu_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.qwen35_cpu->restore_gdn_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    }
#ifdef ROOKERY_ENABLE_CUDA
    else if (engine.is_glm5()) {
        const rookery::Glm5NextModel::KdaCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.kda_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.glm5->restore_kda_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_inkling2()) {
        const size_t safe_lim = static_cast<size_t>(engine.inkling2->swa_safe_rollback_limit());
        const size_t cached_n = engine.cached_tokens.size();
        const rookery::Inkling2LoopModel::ConvCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.conv_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse &&
                (cached_n <= static_cast<size_t>(cp.pos) || (cached_n - static_cast<size_t>(cp.pos)) < safe_lim)) {
                best_cp = &cp;
            }
        }
        if (best_cp != nullptr) {
            engine.inkling2->restore_conv_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_qwen35()) {
        const rookery::Qwen35MoeModel::GdnCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.gdn_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.qwen35->restore_gdn_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_sensenova()) {
        const rookery::SenseNovaLoopModel::GdnCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.sensenova_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.sensenova->restore_gdn_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_qwen4()) {
        const rookery::Qwen4ExpModel::StateCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.qwen4_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.qwen4->restore_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_bailing3()) {
        const rookery::BailingMoe3Model::KdaCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.bailing3_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.bailing3->restore_kda_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    } else if (engine.is_raon1()) {
        const rookery::Raon1MiniModel::RdaCheckpoint * best_cp = nullptr;
        for (const auto & cp : engine.raon1_checkpoints) {
            if (cp.pos > 0 && static_cast<size_t>(cp.pos) <= max_reuse) best_cp = &cp;
        }
        if (best_cp != nullptr) {
            engine.raon1->restore_rda_checkpoint(*best_cp);
            reuse_pos = static_cast<size_t>(best_cp->pos);
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
        engine.prune_state_checkpoints_after(static_cast<int>(reuse_pos));
    }
#endif
    else if (engine.gemma4_cpu) {
        const size_t safe_lim = static_cast<size_t>(engine.gemma4_cpu->swa_safe_rollback_limit());
        const size_t cached_n = engine.cached_tokens.size();
        if (max_reuse > 0 && (cached_n <= max_reuse || (cached_n - max_reuse) < safe_lim)) {
            reuse_pos = max_reuse;
        } else {
            engine.reset_state();
            reuse_pos = 0;
        }
    } else {
        reuse_pos = max_reuse;
    }

    gr.reused_tokens    = reuse_pos;
    gr.prefilled_tokens = prompt_ids.size() - reuse_pos;

    // Sequence-state lifecycle for EngramSidecar: release evicted sequence state on cold start
    // (reuse_pos == 0) and assign a fresh seq_id; on prefix-cache reuse keep the active seq_id
    // (prepare_step rolls back EngramSeqState cleanly to reuse_pos).
    if (!engrams.empty() && reuse_pos == 0) {
        for (auto * sc : engrams) sc->release_sequence(engine.engram_seq_id);
        engine.engram_seq_id += 1;
    }
    const int cur_engram_seq_id = engine.engram_seq_id;

    std::vector<int32_t> pos(prompt_ids.size());
    std::iota(pos.begin(), pos.end(), 0);

    std::vector<float> logits;
    const auto t_pp0 = std::chrono::steady_clock::now();
    size_t last_off = reuse_pos;
    for (size_t off = reuse_pos; off < prompt_ids.size(); off += chunk) {
        last_off = off;
        size_t len = std::min(chunk, prompt_ids.size() - off);
        std::vector<int32_t> pt(prompt_ids.begin() + static_cast<ptrdiff_t>(off),
                                prompt_ids.begin() + static_cast<ptrdiff_t>(off + len));
        std::vector<int32_t> pp(pos.begin() + static_cast<ptrdiff_t>(off),
                                pos.begin() + static_cast<ptrdiff_t>(off + len));
        for (size_t ei = 0; ei < engrams.size(); ++ei) {
            engrams[ei]->prepare_step(pt, static_cast<int>(off), engram_params[ei], cur_engram_seq_id);
        }
        engine.forward(pt, pp, logits, /*compute_logits=*/(off + len >= prompt_ids.size()));
        if (run_mtp && off + len < prompt_ids.size()) {
            std::vector<int32_t> next_t(len);
            for (size_t k = 0; k < len; ++k) next_t[k] = prompt_ids[off + k + 1];
            engine.mtp_prefill_chunk(next_t, pp);
        }
        // Periodic recurrent/conv checkpoints every 1024 tokens + at the penultimate chunk boundary
        // (guarantees a checkpoint strictly <= prompt_ids.size()-1 for MTP strip-<think> reuse at ANY prompt length!)
        if (engine.has_state_checkpoints() && off + len < prompt_ids.size() &&
            (((off + len) % 1024 == 0) || (off + len + chunk >= prompt_ids.size()))) {
            engine.save_state_checkpoint(static_cast<int>(off + len));
        }
        // Keep SSE connections alive during multi-second cold prefills; preserve partial cache on disconnect!
        if (on_prefill_progress && !on_prefill_progress()) {
            const size_t done_pos = off + len;
            engine.cached_tokens.assign(prompt_ids.begin(), prompt_ids.begin() + static_cast<ptrdiff_t>(done_pos));
            engine.has_cached_engram_params = !engrams.empty();
            engine.cached_engram_params     = engram_params;
            if (engine.has_state_checkpoints()) engine.save_state_checkpoint(static_cast<int>(done_pos));
            engine.mtp_kv_valid_len = (run_mtp && done_pos < prompt_ids.size()) ? done_pos : std::min(engine.mtp_kv_valid_len, reuse_pos);
            gr.pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pp0).count();
            return gr;
        }
    }
    // Always save a state checkpoint at exact end of prompt (guarantees 100% cache hit even when clients strip <think>!)
    if (engine.has_state_checkpoints() && !prompt_ids.empty()) {
        engine.save_state_checkpoint(static_cast<int>(prompt_ids.size()));
    }
    const auto t_pp1 = std::chrono::steady_clock::now();
    gr.pp_ms = std::chrono::duration<double, std::milli>(t_pp1 - t_pp0).count();

    std::vector<int32_t> processed_tokens = prompt_ids;
    int32_t cur_tok = sampler.sample(logits);
    int32_t cur_pos = static_cast<int32_t>(prompt_ids.size());
    bool mtp_ran_decode = false;

    const auto t_tg0 = std::chrono::steady_clock::now();
    if (!tokenizer.is_stop_token(cur_tok) && max_tokens > 0) {
        gr.tokens.push_back(cur_tok);
        bool keep_going = !on_token || on_token(cur_tok);
        if (static_cast<int>(gr.tokens.size()) >= max_tokens) {
            gr.hit_max_tokens = true;
            keep_going = false;
        }

        if (keep_going && run_mtp) {
            mtp_ran_decode = true;
            const size_t last_len = prompt_ids.size() - last_off;
            std::vector<int32_t> next_t(last_len);
            std::vector<int32_t> last_p(last_len);
            for (size_t k = 0; k + 1 < last_len; ++k) {
                next_t[k] = prompt_ids[last_off + k + 1];
                last_p[k] = static_cast<int32_t>(last_off + k);
            }
            next_t[last_len - 1] = cur_tok;
            last_p[last_len - 1] = static_cast<int32_t>(prompt_ids.size() - 1);

            int32_t draft = 0;
            rookery::Sampler::SparseDist draft_q;
            if (sampler.is_greedy()) {
                draft = engine.mtp_prefill_chunk(next_t, last_p);
            } else {
                std::vector<float> mtp_logits;
                engine.mtp_prefill_chunk_logits(next_t, last_p, mtp_logits);
                draft_q = sampler.filtered_distribution(mtp_logits.data(), static_cast<int>(mtp_logits.size()));
                draft   = sampler.sample_dist(draft_q);
            }

            rookery::MtpStats ms{};
            size_t next_gen_ckpt = 128;
            while (static_cast<int>(gr.tokens.size()) < max_tokens && cur_pos + 2 < engine.max_ctx()) {
                if (engine.has_state_checkpoints() &&
                    (static_cast<int>(gr.tokens.size()) + 2 >= max_tokens || gr.tokens.size() >= next_gen_ckpt)) {
                    engine.save_state_checkpoint(cur_pos);
                    if (gr.tokens.size() >= next_gen_ckpt) next_gen_ckpt = gr.tokens.size() + 128;
                }
                for (size_t ei = 0; ei < engrams.size(); ++ei) {
                    engrams[ei]->prepare_step({cur_tok, draft}, cur_pos, engram_params[ei], cur_engram_seq_id);
                }
                const int32_t old_cur   = cur_tok;
                const int32_t old_draft = draft;
                const int32_t old_pos   = cur_pos;
                std::vector<int32_t> step_out;
                if (sampler.is_greedy()) {
                    engine.decode_mtp_greedy_step(cur_tok, draft, cur_pos, step_out, ms);
                } else {
                    engine.decode_mtp_sample_step(cur_tok, draft, draft_q, cur_pos, sampler, step_out, ms);
                }
                processed_tokens.push_back(old_cur);
                // If MTP accepted a draft stop token on slot 0, roll back the 2nd-slot overshoot so
                // processed_tokens ends cleanly before the stop token and mtp_last_next_tok == stop token!
                if (cur_pos == old_pos + 2 && !step_out.empty() && tokenizer.is_stop_token(step_out[0])) {
                    engine.rollback_mtp_1tok();
                    cur_pos = old_pos + 1;
                    cur_tok = step_out[0];
                    break;
                }
                if (cur_pos == old_pos + 2) processed_tokens.push_back(old_draft);
                bool stopped = false;
                for (int32_t tid : step_out) {
                    if (tokenizer.is_stop_token(tid)) { stopped = true; break; }
                    gr.tokens.push_back(tid);
                    if (on_token && !on_token(tid)) { stopped = true; break; }
                    if (static_cast<int>(gr.tokens.size()) >= max_tokens) {
                        gr.hit_max_tokens = true;
                        stopped = true;
                        break;
                    }
                }
                if (stopped) break;
            }
        } else if (keep_going) {
            while (static_cast<int>(gr.tokens.size()) < max_tokens && cur_pos + 1 < engine.max_ctx()) {
                for (size_t ei = 0; ei < engrams.size(); ++ei) {
                    engrams[ei]->prepare_step({cur_tok}, cur_pos, engram_params[ei], cur_engram_seq_id);
                }
                const int32_t fed_tok = cur_tok;
                if (sampler.is_greedy()) {
                    cur_tok = engine.forward_decode_greedy(cur_tok, cur_pos++);
                } else {
                    engine.forward_decode_logits(cur_tok, cur_pos++, logits);
                    cur_tok = sampler.sample(logits);
                }
                processed_tokens.push_back(fed_tok);
                if (tokenizer.is_stop_token(cur_tok)) break;
                gr.tokens.push_back(cur_tok);
                if (on_token && !on_token(cur_tok)) break;
                if (static_cast<int>(gr.tokens.size()) >= max_tokens) {
                    gr.hit_max_tokens = true;
                    break;
                }
            }
        }
    }
    const auto t_tg1 = std::chrono::steady_clock::now();
    gr.tg_ms = std::chrono::duration<double, std::milli>(t_tg1 - t_tg0).count();

    // Update prompt prefix cache & end-of-generation state checkpoint
    engine.cached_tokens            = std::move(processed_tokens);
    engine.has_cached_engram_params = !engrams.empty();
    engine.cached_engram_params     = engram_params;
#ifdef ROOKERY_ENABLE_CUDA
    if (engine.raon1) engine.cached_raon_effort_alpha = engine.raon1->reasoning_effort_alpha();
#endif
    if (engine.has_state_checkpoints() && cur_pos > static_cast<int32_t>(prompt_ids.size())) {
        engine.save_state_checkpoint(cur_pos);
    }
    if (run_mtp) {
        engine.mtp_kv_valid_len  = mtp_ran_decode ? static_cast<size_t>(cur_pos) : last_off;
        engine.mtp_last_next_tok = mtp_ran_decode ? cur_tok : -1;
    } else {
        engine.mtp_kv_valid_len  = std::min(engine.mtp_kv_valid_len, reuse_pos);
        engine.mtp_last_next_tok = -1;
    }

    const double pp_tps = (gr.pp_ms > 0.0) ? (gr.prefilled_tokens * 1000.0 / gr.pp_ms) : 0.0;
    const double tg_tps = (gr.tg_ms > 0.0) ? (gr.tokens.size() * 1000.0 / gr.tg_ms) : 0.0;
    std::printf("[req] prompt=%zu common=%zu max_reuse=%zu reused=%zu prefilled=%zu (%.1f ms, %.1f tok/s) | gen=%zu (%.1f ms, %.1f tok/s)\n",
                prompt_ids.size(), common, max_reuse, gr.reused_tokens, gr.prefilled_tokens, gr.pp_ms, pp_tps,
                gr.tokens.size(), gr.tg_ms, tg_tps);
    std::fflush(stdout);
    return gr;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string engram_path;
    std::string engram_adapter_path;
    float       engram_scale = -1.0f;
    float       engram_gate_bias = 0.0f;
    bool        engram_gate_bias_set = false;
    std::vector<int> engram_layers;
    std::string engram2_path;
    std::string engram2_adapter_path;
    float       engram2_scale = -1.0f;
    float       engram2_gate_bias = 0.0f;
    bool        engram2_gate_bias_set = false;
    std::vector<int> engram2_layers;
    std::string host  = "127.0.0.1";
    int         port  = 8080;
    std::string alias;
    int                  max_ctx       = 32768;
    rookery::KvCacheType kv_type       = rookery::KvCacheType::Q8_0;
    bool                 kv_type_k_set = false;
    float                rope_scale    = 1.0f;
    bool        use_mtp   = true;
    int         n_threads = 0;
    bool        force_cpu = false;
    std::vector<int>   gpu_ids;
    std::vector<float> tensor_split;
    rookery::SamplingParams default_sampling;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

        if (a == "-m" || a == "--model")          model_path = next();
        else if (a == "--engram")                 engram_path = next();
        else if (a == "--engram-adapter")         engram_adapter_path = next();
        else if (a == "--engram-scale")           engram_scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--engram-gate-bias")       { engram_gate_bias = std::strtof(next().c_str(), nullptr); engram_gate_bias_set = true; }
        else if (a == "--engram-layers")          engram_layers = parse_csv_ints(next());
        else if (a == "--engram2")                engram2_path = next();
        else if (a == "--engram-adapter2")        engram2_adapter_path = next();
        else if (a == "--engram-scale2")          engram2_scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--engram-gate-bias2")      { engram2_gate_bias = std::strtof(next().c_str(), nullptr); engram2_gate_bias_set = true; }
        else if (a == "--engram-layers2")         engram2_layers = parse_csv_ints(next());
        else if (a == "--host")                   host = next();
        else if (a == "--port")                   port = std::atoi(next().c_str());
        else if (a == "--alias")                  alias = next();
        else if (a == "-c" || a == "--ctx-size")  max_ctx = std::atoi(next().c_str());
        else if (a == "-t" || a == "--threads")   n_threads = std::atoi(next().c_str());
        else if (a == "--cpu")                    force_cpu = true;
        else if (a == "--tensor-split" || a == "-ts") tensor_split = parse_csv_floats(next());
        else if (a == "--gpus")                   gpu_ids = parse_csv_ints(next());
        else if (a == "--cache-type-k" || a == "-ctk") {
            kv_type = rookery::parse_kv_cache_type(next());
            kv_type_k_set = true;
        } else if (a == "--cache-type-v" || a == "-ctv") {
            const auto vt = rookery::parse_kv_cache_type(next());
            if (!kv_type_k_set) kv_type = vt;
        }
        else if (a == "--temp")                   default_sampling.temperature = std::strtof(next().c_str(), nullptr);
        else if (a == "--top-p")                  default_sampling.top_p = std::strtof(next().c_str(), nullptr);
        else if (a == "--top-k")                  default_sampling.top_k = std::atoi(next().c_str());
        else if (a == "--rope-scale")             rope_scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--no-mtp")                 use_mtp = false;
        else if (a == "--mtp")                    use_mtp = true;
        // Accepted drop-in llama-swap compatibility flags:
        else if (a == "-ngl" || a == "--n-gpu-layers" || a == "-np" || a == "--parallel" ||
                 a == "--flash-attn" || a == "-fa" || a == "--reasoning-format" || a == "--timeout" ||
                 a == "--spec-type" || a == "--spec-draft-n-max" || a == "-ot" || a == "-ub" || a == "-b") {
            (void) next();
        } else if (a == "--jinja" || a == "--no-mmproj" || a == "--no-warmup") {
            // no-arg flag
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <model.gguf> [--host 127.0.0.1] [--port 8080] [--alias name] [-c 32768] [--tensor-split 11,12,12,10]\n", argv[0]);
        return 1;
    }

    auto topo = rookery::discover_topology();
    rookery::print_topology(topo);

    if (gpu_ids.empty()) {
        const size_t n = tensor_split.empty() ? topo.gpus.size() : std::min(tensor_split.size(), topo.gpus.size());
        for (size_t i = 0; i < n; ++i) gpu_ids.push_back(static_cast<int>(i));
    }

    std::atomic<bool> ready{false};
    std::mutex        infer_mu;

    httplib::Server svr;
    svr.set_read_timeout(600, 0);
    svr.set_write_timeout(600, 0);
    svr.set_idle_interval(0, 100000);
    rookery::BpeTokenizer        tokenizer;
    ServerEngine                 engine;
    rookery::EngramSidecar       engram_sidecar;
    rookery::EngramSidecar       engram_sidecar2;
    rookery::CompositeLayerHook  engram_composite;
    std::vector<rookery::EngramSidecar *> engram_ptrs;
    std::string model_alias = alias;

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        if (!ready.load()) {
            res.status = 503;
            res.set_content(R"({"status":"loading model"})", "application/json");
            return;
        }
        json body = {{"status", "ok"}, {"arch", engine.arch}};
        auto describe_engram = [](const rookery::EngramSidecar * sc) -> json {
            return {
                {"provider_arch", sc->table().cfg().arch},
                {"provider_fingerprint", sc->table().fingerprint_hex()},
                {"rkad_version", sc->manifest().format_version},
                {"calibration", sc->manifest().calibration},
                {"default_scale", sc->default_params().scale},
                {"hooks", sc->num_hooks()}
            };
        };
        if (!engram_ptrs.empty()) {
            body["engram"] = describe_engram(engram_ptrs[0]);
            if (engram_ptrs.size() > 1) {
                body["engram2"] = describe_engram(engram_ptrs[1]);
            }
        }
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json body = {
            {"object", "list"},
            {"data", json::array({{
                {"id", model_alias},
                {"object", "model"},
                {"owned_by", "rookery"}
            }})}
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/tokenize", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            auto j = json::parse(req.body);
            auto ids = tokenizer.encode(j.value("content", ""), j.value("add_special", tokenizer.add_bos_default()), true);
            res.set_content(json({{"tokens", ids}}).dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ready.load()) {
            res.status = 503;
            res.set_content(R"({"error":"model loading"})", "application/json");
            return;
        }
        try {
            const auto j = json::parse(req.body);
            const std::string prompt_str = (j.contains("prompt") && j["prompt"].is_string()) ? j["prompt"].get<std::string>() : "";
            const int max_tokens = (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) ? j["max_tokens"].get<int>() : 256;
            rookery::SamplingParams sp = default_sampling;
            sp.seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            if (j.contains("temperature") && j["temperature"].is_number()) sp.temperature = j["temperature"].get<float>();
            if (j.contains("top_p") && j["top_p"].is_number())             sp.top_p = j["top_p"].get<float>();
            if (j.contains("top_k") && j["top_k"].is_number())             sp.top_k = j["top_k"].get<int>();
            if (j.contains("seed") && j["seed"].is_number_integer())       sp.seed = j["seed"].get<uint64_t>();

            const auto prompt_ids = tokenizer.encode(prompt_str, tokenizer.add_bos_default(), true);
            const auto stop_strs  = parse_stop_strings(j);
            const auto ep         = parse_engram_params(j, engram_ptrs);
            std::string acc_text;
            std::lock_guard<std::mutex> lock(infer_mu);
            auto gr = run_generation(engine, tokenizer, engram_ptrs, ep, prompt_ids, sp, max_tokens, use_mtp,
                stop_strs.empty() ? std::function<bool(int32_t)>(nullptr) : std::function<bool(int32_t)>([&](int32_t tid) -> bool {
                    acc_text += tokenizer.decode_token(tid);
                    return find_earliest_stop(acc_text, stop_strs) == std::string::npos;
                }));

            std::string text = stop_strs.empty() ? tokenizer.decode(gr.tokens, false) : acc_text;
            if (!stop_strs.empty()) {
                size_t sp_pos = find_earliest_stop(text, stop_strs);
                if (sp_pos != std::string::npos) text.resize(sp_pos);
            }
            const auto now_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            json out = {
                {"id", "cmpl-rookery"},
                {"object", "text_completion"},
                {"created", now_s},
                {"model", model_alias},
                {"choices", json::array({{
                    {"index", 0},
                    {"text", text},
                    {"finish_reason", gr.hit_max_tokens ? "length" : "stop"}
                }})},
                {"usage", {
                    {"prompt_tokens", prompt_ids.size()},
                    {"completion_tokens", gr.tokens.size()},
                    {"total_tokens", prompt_ids.size() + gr.tokens.size()},
                    {"prompt_tokens_details", {{"cached_tokens", gr.reused_tokens}}}
                }},
                {"timings", {
                    {"cache_n", gr.reused_tokens},
                    {"prompt_n", gr.prefilled_tokens},
                    {"prompt_ms", gr.pp_ms},
                    {"prompt_per_second", gr.pp_ms > 0 ? gr.prefilled_tokens * 1000.0 / gr.pp_ms : 0.0},
                    {"predicted_n", gr.tokens.size()},
                    {"predicted_ms", gr.tg_ms},
                    {"predicted_per_second", gr.tg_ms > 0 ? gr.tokens.size() * 1000.0 / gr.tg_ms : 0.0}
                }}
            };
            res.set_content(safe_json_dump(out), "application/json");
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ready.load()) {
            res.status = 503;
            res.set_content(R"({"error":"model loading"})", "application/json");
            return;
        }
        try {
            const auto j = json::parse(req.body);
            const bool stream = (j.contains("stream") && j["stream"].is_boolean()) ? j["stream"].get<bool>() : false;
            int max_tokens = 8192;
            if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
                max_tokens = j["max_tokens"].get<int>();
            } else if (j.contains("max_completion_tokens") && j["max_completion_tokens"].is_number_integer()) {
                max_tokens = j["max_completion_tokens"].get<int>();
            }
            bool enable_thinking = !engine.is_qwen2();
            std::string reasoning_effort = engine.is_inkling2() ? "medium" : (engine.is_raon1() ? "high" : "max");
            if (j.contains("reasoning_effort") && j["reasoning_effort"].is_string()) {
                reasoning_effort = j["reasoning_effort"].get<std::string>();
            }
            if (j.contains("chat_template_kwargs") && j["chat_template_kwargs"].is_object()) {
                if (j["chat_template_kwargs"].contains("enable_thinking") && j["chat_template_kwargs"]["enable_thinking"].is_boolean()) {
                    enable_thinking = j["chat_template_kwargs"]["enable_thinking"].get<bool>();
                }
                if (j["chat_template_kwargs"].contains("reasoning_effort") && j["chat_template_kwargs"]["reasoning_effort"].is_string()) {
                    reasoning_effort = j["chat_template_kwargs"]["reasoning_effort"].get<std::string>();
                }
            }
            if (engine.is_raon1()) {
                if (reasoning_effort == "none") enable_thinking = false;
                else if (!enable_thinking)      reasoning_effort = "none";
            }

            rookery::SamplingParams sp = default_sampling;
            sp.seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            if (j.contains("temperature") && j["temperature"].is_number()) sp.temperature = j["temperature"].get<float>();
            if (j.contains("top_p") && j["top_p"].is_number())             sp.top_p = j["top_p"].get<float>();
            if (j.contains("top_k") && j["top_k"].is_number())             sp.top_k = j["top_k"].get<int>();
            if (j.contains("seed") && j["seed"].is_number_integer())       sp.seed = j["seed"].get<uint64_t>();

            const json tools = j.contains("tools") ? j["tools"] : json::array();
            const auto stop_strs = parse_stop_strings(j);
            const std::string prompt_str = engine.is_glm5()
                ? rookery::render_glm5_chat_prompt(j.at("messages"), tools, enable_thinking, reasoning_effort)
                : (engine.is_inkling2()
                    ? rookery::render_inkling2_chat_prompt(j.at("messages"), tools, enable_thinking, reasoning_effort)
                    : (engine.is_bailing3()
                        ? rookery::render_bailing3_chat_prompt(j.at("messages"), tools, enable_thinking)
                        : (engine.is_gemma4()
                            ? rookery::render_gemma4_chat_prompt(j.at("messages"), enable_thinking)
                            : (engine.is_qwen2()
                                ? rookery::render_qwen2_chat_prompt(j.at("messages"), tools)
                                : (engine.is_raon1()
                                    ? rookery::render_raon1_chat_prompt(j.at("messages"), tools, enable_thinking)
                                    : ((engine.is_qwen35() || engine.is_qwen4() || engine.is_sensenova())
                                        ? rookery::render_qwen3_chat_prompt(j.at("messages"), tools, enable_thinking)
                                        : rookery::render_maxwell_chat_prompt(j.at("messages"), tools, enable_thinking)))))));
            const auto prompt_ids = tokenizer.encode(prompt_str, tokenizer.add_bos_default(), true);
            const auto now_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            auto parse_out = [&](const std::string & raw, bool in_think) {
                if (engine.is_glm5() || engine.is_bailing3()) return rookery::parse_glm5_output(raw, in_think);
                if (engine.is_inkling2()) return rookery::parse_inkling2_output(in_think ? raw : ("<|content_text|>" + raw));
                if (engine.is_gemma4())   return rookery::parse_gemma4_output(raw, in_think);
                if (engine.is_qwen2())    return rookery::parse_qwen2_output(raw);
                return rookery::parse_maxwell_output(raw, in_think);
            };

            const auto ep = parse_engram_params(j, engram_ptrs);
            if (!stream) {
                std::lock_guard<std::mutex> lock(infer_mu);
#ifdef ROOKERY_ENABLE_CUDA
                if (engine.raon1) engine.raon1->set_reasoning_effort(reasoning_effort);
#endif
                std::string acc_raw;
                auto gr = run_generation(engine, tokenizer, engram_ptrs, ep, prompt_ids, sp, max_tokens, use_mtp,
                    stop_strs.empty() ? std::function<bool(int32_t)>(nullptr) : std::function<bool(int32_t)>([&](int32_t tid) -> bool {
                        acc_raw += tokenizer.decode_token(tid);
                        return find_earliest_stop(acc_raw, stop_strs) == std::string::npos;
                    }));

                std::string raw_text = stop_strs.empty() ? tokenizer.decode(gr.tokens, false) : acc_raw;
                if (!stop_strs.empty()) {
                    size_t sp_pos = find_earliest_stop(raw_text, stop_strs);
                    if (sp_pos != std::string::npos) raw_text.resize(sp_pos);
                }
                auto parsed = parse_out(raw_text, enable_thinking);
                std::string finish = !parsed.tool_calls.empty() ? "tool_calls"
                                   : (gr.hit_max_tokens ? "length" : "stop");

                json msg = {{"role", "assistant"}, {"content", parsed.content}};
                if (!parsed.reasoning_content.empty()) msg["reasoning_content"] = parsed.reasoning_content;
                if (!parsed.tool_calls.empty())        msg["tool_calls"]        = parsed.tool_calls;

                json out = {
                    {"id", "chatcmpl-rookery"},
                    {"object", "chat.completion"},
                    {"created", now_s},
                    {"model", model_alias},
                    {"choices", json::array({{
                        {"index", 0},
                        {"message", msg},
                        {"finish_reason", finish}
                    }})},
                    {"usage", {
                        {"prompt_tokens", prompt_ids.size()},
                        {"completion_tokens", gr.tokens.size()},
                        {"total_tokens", prompt_ids.size() + gr.tokens.size()},
                        {"prompt_tokens_details", {{"cached_tokens", gr.reused_tokens}}}
                    }},
                    {"timings", {
                        {"cache_n", gr.reused_tokens},
                        {"prompt_n", gr.prefilled_tokens},
                        {"prompt_ms", gr.pp_ms},
                        {"prompt_per_second", gr.pp_ms > 0 ? gr.prefilled_tokens * 1000.0 / gr.pp_ms : 0.0},
                        {"predicted_n", gr.tokens.size()},
                        {"predicted_ms", gr.tg_ms},
                        {"predicted_per_second", gr.tg_ms > 0 ? gr.tokens.size() * 1000.0 / gr.tg_ms : 0.0}
                    }}
                };
                res.set_content(safe_json_dump(out), "application/json");
            } else {
                // Real-time token-by-token SSE streaming (UTF-8 boundary-safe + prefill keepalive!)
                res.set_header("Cache-Control", "no-cache");
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&, prompt_ids, sp, max_tokens, enable_thinking, reasoning_effort, now_s, stop_strs, ep](size_t, httplib::DataSink & sink) -> bool {
                        try {
                            std::lock_guard<std::mutex> lock(infer_mu);
#ifdef ROOKERY_ENABLE_CUDA
                            if (engine.raon1) engine.raon1->set_reasoning_effort(reasoning_effort);
#endif

                            auto send_chunk = [&](const json & delta, const json & finish_r) -> bool {
                                json ch = {
                                    {"id", "chatcmpl-rookery"},
                                    {"object", "chat.completion.chunk"},
                                    {"created", now_s},
                                    {"model", model_alias},
                                    {"choices", json::array({{
                                        {"index", 0},
                                        {"delta", delta},
                                        {"finish_reason", finish_r}
                                    }})}
                                };
                                std::string s = "data: " + safe_json_dump(ch) + "\n\n";
                                return sink.write(s.data(), s.size());
                            };

                            if (!send_chunk({{"role", "assistant"}}, nullptr)) { sink.done(); return true; }

                            bool in_think = enable_thinking;
                            bool in_tool  = false;
                            bool has_tools = false;
                            bool hit_stop_str = false;
                            int  ink_state = enable_thinking ? 0 : 2; // 0=IDLE, 1=THINK, 2=TEXT, 3=TOOL
                            int  ink_tool_idx = 0;
                            rookery::Gemma4StreamSplitter gemma4_splitter(enable_thinking);
                            size_t max_stop_len = 0;
                            for (const auto & st : stop_strs) max_stop_len = std::max(max_stop_len, st.size());
                            std::string pending;

                            auto flush_stream = [&](bool is_final) -> bool {
                                if (engine.is_inkling2()) {
                                    static const std::string kThink = "<|content_thinking|>";
                                    static const std::string kText  = "<|content_text|>";
                                    static const std::string kTool  = "<|content_invoke_tool_json|>";
                                    static const std::string kEnd   = "<|end_message|>";
                                    while (true) {
                                        if (ink_state == 0) {
                                            size_t p_th = pending.find(kThink);
                                            size_t p_tx = pending.find(kText);
                                            size_t p_tl = pending.find(kTool);
                                            size_t p_min = std::min(p_th, std::min(p_tx, p_tl));
                                            if (p_min != std::string::npos) {
                                                if (p_min == p_th) { pending.erase(0, p_min + kThink.size()); ink_state = 1; }
                                                else if (p_min == p_tx) { pending.erase(0, p_min + kText.size()); ink_state = 2; }
                                                else { pending.erase(0, p_min + kTool.size()); ink_state = 3; }
                                                continue;
                                            }
                                            if (!is_final && !sink.is_writable()) return false;
                                            return true;
                                        }
                                        if (ink_state == 1 || ink_state == 2) {
                                            if (ink_state == 2 && !stop_strs.empty()) {
                                                size_t sp_pos = find_earliest_stop(pending, stop_strs);
                                                if (sp_pos != std::string::npos) {
                                                    std::string chunk_s = pending.substr(0, sp_pos);
                                                    pending.clear();
                                                    hit_stop_str = true;
                                                    if (!chunk_s.empty() && !send_chunk({{"content", chunk_s}}, nullptr)) return false;
                                                    return false;
                                                }
                                            }
                                            size_t ep = pending.find(kEnd);
                                            const char * field = (ink_state == 1) ? "reasoning_content" : "content";
                                            if (ep != std::string::npos) {
                                                std::string seg = pending.substr(0, ep);
                                                pending.erase(0, ep + kEnd.size());
                                                ink_state = 0;
                                                if (!seg.empty() && !send_chunk({{field, seg}}, nullptr)) return false;
                                                continue;
                                            }
                                            if (is_final) {
                                                if (!pending.empty() && !send_chunk({{field, pending}}, nullptr)) return false;
                                                pending.clear();
                                                return !hit_stop_str;
                                            }
                                            const size_t hold = std::max<size_t>(kEnd.size(), max_stop_len);
                                            if (pending.size() > hold) {
                                                size_t n = utf8_safe_prefix_len(pending, pending.size() - hold);
                                                if (n > 0) {
                                                    if (!send_chunk({{field, pending.substr(0, n)}}, nullptr)) return false;
                                                    pending.erase(0, n);
                                                }
                                            }
                                            return true;
                                        }
                                        if (ink_state == 3) {
                                            size_t ep = pending.find(kEnd);
                                            if (ep != std::string::npos || is_final) {
                                                std::string body = (ep != std::string::npos) ? pending.substr(0, ep) : pending;
                                                if (ep != std::string::npos) pending.erase(0, ep + kEnd.size());
                                                else pending.clear();
                                                ink_state = 0;
                                                try {
                                                    json tj = json::parse(body);
                                                    std::string fname = tj.value("name", "");
                                                    json args = tj.contains("args") ? tj["args"] : (tj.contains("arguments") ? tj["arguments"] : json::object());
                                                    std::string args_s = args.is_string() ? args.get<std::string>() : args.dump();
                                                    has_tools = true;
                                                    json tc = {
                                                        {"index", ink_tool_idx++},
                                                        {"id", rookery::next_tool_call_id()},
                                                        {"type", "function"},
                                                        {"function", {{"name", fname}, {"arguments", args_s}}}
                                                    };
                                                    if (!send_chunk({{"tool_calls", json::array({tc})}}, nullptr)) return false;
                                                } catch (...) {}
                                                if (ep != std::string::npos) continue;
                                            }
                                            if (!is_final && !sink.is_writable()) return false;
                                            return true;
                                        }
                                    }
                                }
                                if (!stop_strs.empty() && !in_tool) {
                                    size_t sp_pos = find_earliest_stop(pending, stop_strs);
                                    if (sp_pos != std::string::npos) {
                                        pending.resize(sp_pos);
                                        hit_stop_str = true;
                                        is_final = true;
                                    }
                                }
                                if (engine.is_gemma4()) {
                                    const bool ok = gemma4_splitter.flush(
                                        pending, is_final, max_stop_len,
                                        [&](const char * field, const std::string & s) -> bool {
                                            return send_chunk({{field, s}}, nullptr);
                                        },
                                        utf8_safe_prefix_len);
                                    return ok && !hit_stop_str;
                                }
                                if (in_think) {
                                    const char * end_tag = engine.is_gemma4() ? "<channel|>" : "</think>";
                                    const size_t end_len = engine.is_gemma4() ? 10 : 8;
                                    size_t end_t = pending.find(end_tag);
                                    size_t tc_t  = pending.find("<tool_call>");
                                    const size_t hold = std::max<size_t>(11, max_stop_len);
                                    if (end_t != std::string::npos && (tc_t == std::string::npos || end_t < tc_t)) {
                                        std::string r = pending.substr(0, end_t);
                                        if (!r.empty() && !send_chunk({{"reasoning_content", r}}, nullptr)) return false;
                                        pending.erase(0, end_t + end_len);
                                        if (!engine.is_glm5() && !engine.is_bailing3()) {
                                            while (!pending.empty() && pending.front() == '\n') pending.erase(pending.begin());
                                        }
                                        in_think = false;
                                    } else if (tc_t != std::string::npos) {
                                        // Recover gracefully if model skips </think> and emits <tool_call> directly!
                                        if (tc_t > 0 && !send_chunk({{"reasoning_content", pending.substr(0, tc_t)}}, nullptr)) return false;
                                        pending.erase(0, tc_t);
                                        in_think = false;
                                        in_tool  = true;
                                    } else if (is_final) {
                                        if (!pending.empty() && !send_chunk({{"reasoning_content", pending}}, nullptr)) return false;
                                        pending.clear();
                                        return !hit_stop_str;
                                    } else if (pending.size() > hold) {
                                        size_t n = utf8_safe_prefix_len(pending, pending.size() - hold);
                                        if (n > 0) {
                                            if (!send_chunk({{"reasoning_content", pending.substr(0, n)}}, nullptr)) return false;
                                            pending.erase(0, n);
                                        }
                                        return true;
                                    }
                                }
                                if (!in_think && !in_tool) {
                                    size_t tc = pending.find("<tool_call>");
                                    const size_t hold = std::max<size_t>(11, max_stop_len);
                                    if (tc != std::string::npos) {
                                        if (tc > 0 && !send_chunk({{"content", pending.substr(0, tc)}}, nullptr)) return false;
                                        pending.erase(0, tc);
                                        in_tool = true;
                                    } else if (is_final) {
                                        if (!pending.empty() && !send_chunk({{"content", pending}}, nullptr)) return false;
                                        pending.clear();
                                        return !hit_stop_str;
                                    } else if (pending.size() > hold) {
                                        size_t n = utf8_safe_prefix_len(pending, pending.size() - hold);
                                        if (n > 0) {
                                            if (!send_chunk({{"content", pending.substr(0, n)}}, nullptr)) return false;
                                            pending.erase(0, n);
                                        }
                                    }
                                }
                                if (in_tool && !is_final && !sink.is_writable()) return false;
                                return !hit_stop_str;
                            };

                            auto gr = run_generation(engine, tokenizer, engram_ptrs, ep, prompt_ids, sp, max_tokens, use_mtp,
                                [&](int32_t tid) -> bool {
                                    if (!sink.is_writable()) return false;
                                    pending += tokenizer.decode_token(tid);
                                    return flush_stream(false);
                                },
                                [&]() -> bool {
                                    if (!sink.is_writable()) return false;
                                    static const char kPing[] = ": keepalive\n\n";
                                    return sink.write(kPing, sizeof(kPing) - 1);
                                });

                            flush_stream(true);
                            if (in_tool && !pending.empty()) {
                                auto parsed = (engine.is_glm5() || engine.is_bailing3())
                                    ? rookery::parse_glm5_output(pending, false)
                                    : (engine.is_qwen2()
                                        ? rookery::parse_qwen2_output(pending)
                                        : rookery::parse_maxwell_output(pending, false));
                                if (!parsed.tool_calls.empty()) {
                                    has_tools = true;
                                    json indexed = json::array();
                                    for (size_t ti = 0; ti < parsed.tool_calls.size(); ++ti) {
                                        json tc = parsed.tool_calls[ti];
                                        tc["index"] = static_cast<int>(ti);
                                        indexed.push_back(std::move(tc));
                                    }
                                    send_chunk({{"tool_calls", indexed}}, nullptr);
                                }
                            }

                            std::string finish = has_tools ? "tool_calls" : (gr.hit_max_tokens ? "length" : "stop");
                            json final_ch = {
                                {"id", "chatcmpl-rookery"},
                                {"object", "chat.completion.chunk"},
                                {"created", now_s},
                                {"model", model_alias},
                                {"choices", json::array({{
                                    {"index", 0},
                                    {"delta", json::object()},
                                    {"finish_reason", finish}
                                }})},
                                {"usage", {
                                    {"prompt_tokens", prompt_ids.size()},
                                    {"completion_tokens", gr.tokens.size()},
                                    {"total_tokens", prompt_ids.size() + gr.tokens.size()},
                                    {"prompt_tokens_details", {{"cached_tokens", gr.reused_tokens}}}
                                }},
                                {"timings", {
                                    {"cache_n", gr.reused_tokens},
                                    {"prompt_n", gr.prefilled_tokens},
                                    {"prompt_ms", gr.pp_ms},
                                    {"prompt_per_second", gr.pp_ms > 0 ? gr.prefilled_tokens * 1000.0 / gr.pp_ms : 0.0},
                                    {"predicted_n", gr.tokens.size()},
                                    {"predicted_ms", gr.tg_ms},
                                    {"predicted_per_second", gr.tg_ms > 0 ? gr.tokens.size() * 1000.0 / gr.tg_ms : 0.0}
                                }}
                            };
                            std::string fs = "data: " + safe_json_dump(final_ch) + "\n\n";
                            sink.write(fs.data(), fs.size());
                            static const char kDone[] = "data: [DONE]\n\n";
                            sink.write(kDone, sizeof(kDone) - 1);
                            sink.done();
                        } catch (const std::exception & e) {
                            std::fprintf(stderr, "[sse error] %s\n", e.what());
                            std::fflush(stderr);
                            sink.done();
                        }
                        return true;
                    });
            }
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    {
        auto sg = rookery::ShardedGguf::open(model_path);
        engine.arch = sg.get_str("general.architecture");
        if (model_alias.empty()) model_alias = sg.get_str("general.name", "rookery-model");
        tokenizer.load(sg.meta());
    }

    const bool kv_q8 = (kv_type != rookery::KvCacheType::F16);
    const bool use_cpu_backend = force_cpu || gpu_ids.empty();
#ifdef ROOKERY_ENABLE_CUDA
    if (engine.arch == "glm5-next" || engine.arch == "glm5next") {
        tokenizer.add_stop_token("<|user|>");
        tokenizer.add_stop_token("<|observation|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.glm5 = std::make_unique<rookery::Glm5NextModel>();
        engine.glm5->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8, use_mtp);
    } else if (engine.arch == "inkling2_loop" || engine.arch == "inkling2") {
        engine.inkling2 = std::make_unique<rookery::Inkling2LoopModel>();
        engine.inkling2->load(model_path, gpu_ids, max_ctx, kv_type);
    } else if (engine.arch == "qwen35moe" || engine.arch == "qwen35") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        if (use_cpu_backend) {
            engine.qwen35_cpu = std::make_unique<rookery::cpu::Qwen35CpuModel>();
            engine.qwen35_cpu->load(model_path, max_ctx, kv_q8, n_threads);
        } else {
            engine.qwen35 = std::make_unique<rookery::Qwen35MoeModel>();
            engine.qwen35->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8, use_mtp);
        }
    } else if (engine.arch == "sensenova_loop") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.sensenova = std::make_unique<rookery::SenseNovaLoopModel>();
        engine.sensenova->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8, use_mtp);
    } else if (engine.arch == "qwen4exp") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.qwen4 = std::make_unique<rookery::Qwen4ExpModel>();
        engine.qwen4->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8);
    } else if (engine.arch == "bailingmoe3") {
        tokenizer.add_stop_token("<|role_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        if (use_cpu_backend) {
            engine.bailing3_cpu = std::make_unique<rookery::cpu::BailingMoe3CpuModel>();
            engine.bailing3_cpu->load(model_path, max_ctx, kv_type, n_threads);
        } else {
            engine.bailing3 = std::make_unique<rookery::BailingMoe3Model>();
            engine.bailing3->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8, use_mtp);
        }
    } else if (engine.arch == "gemma4") {
        tokenizer.add_stop_token("<turn|>");
        tokenizer.add_stop_token("<eos>");
        if (use_cpu_backend) {
            engine.gemma4_cpu = std::make_unique<rookery::cpu::Gemma4CpuModel>();
            engine.gemma4_cpu->load(model_path, max_ctx, kv_type, n_threads);
        } else {
            engine.gemma4 = std::make_unique<rookery::Gemma4Model>();
            engine.gemma4->load(model_path, gpu_ids, tensor_split, max_ctx, kv_type, rope_scale);
        }
    } else if (engine.arch == "qwen2") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.qwen2_cpu = std::make_unique<rookery::cpu::Qwen2CpuModel>();
        engine.qwen2_cpu->load(model_path, max_ctx, kv_type, n_threads);
    } else if (engine.arch == "raon1" || engine.arch == "raon1_mini") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.raon1 = std::make_unique<rookery::Raon1MiniModel>();
        engine.raon1->load(model_path, gpu_ids, tensor_split, max_ctx, kv_q8);
    } else if (engine.arch == "maxwell-moe") {
        engine.maxwell = std::make_unique<rookery::MaxwellMoeModel>();
        engine.maxwell->load(model_path, gpu_ids, tensor_split, max_ctx, kv_type, use_mtp);
    } else {
        throw std::runtime_error("Unsupported model architecture: " + engine.arch);
    }
#else
    (void) use_cpu_backend;
    if (engine.arch == "bailingmoe3") {
        tokenizer.add_stop_token("<|role_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.bailing3_cpu = std::make_unique<rookery::cpu::BailingMoe3CpuModel>();
        engine.bailing3_cpu->load(model_path, max_ctx, kv_type, n_threads);
    } else if (engine.arch == "gemma4") {
        tokenizer.add_stop_token("<turn|>");
        tokenizer.add_stop_token("<eos>");
        engine.gemma4_cpu = std::make_unique<rookery::cpu::Gemma4CpuModel>();
        engine.gemma4_cpu->load(model_path, max_ctx, kv_type, n_threads);
    } else if (engine.arch == "qwen2") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.qwen2_cpu = std::make_unique<rookery::cpu::Qwen2CpuModel>();
        engine.qwen2_cpu->load(model_path, max_ctx, kv_type, n_threads);
    } else if (engine.arch == "qwen35" || engine.arch == "qwen35moe") {
        tokenizer.add_stop_token("<|im_end|>");
        tokenizer.add_stop_token("<|endoftext|>");
        engine.qwen35_cpu = std::make_unique<rookery::cpu::Qwen35CpuModel>();
        engine.qwen35_cpu->load(model_path, max_ctx, kv_q8, n_threads);
    } else {
        throw std::runtime_error("CPU-only build currently supports bailingmoe3, gemma4, qwen2, and qwen35/qwen35moe; unsupported architecture: " + engine.arch);
    }
#endif

    if (!engram_path.empty()) {
        const float eff_scale = (engram_scale >= 0.0f) ? engram_scale : (engram_adapter_path.empty() ? 0.0f : 1.0f);
        engram_sidecar.init(engram_path, &tokenizer, engine.layer_devices(),
                            engine.n_embd(), engine.hc_mult(),
                            512, engram_adapter_path, eff_scale, engram_layers, engine.arch);
        if (engram_gate_bias_set) engram_sidecar.set_gate_bias(engram_gate_bias);
        engram_ptrs.push_back(&engram_sidecar);
    }
    if (!engram2_path.empty()) {
        const float eff_scale2 = (engram2_scale >= 0.0f) ? engram2_scale : (engram2_adapter_path.empty() ? 0.0f : 1.0f);
        engram_sidecar2.init(engram2_path, &tokenizer, engine.layer_devices(),
                             engine.n_embd(), engine.hc_mult(),
                             512, engram2_adapter_path, eff_scale2, engram2_layers, engine.arch);
        if (engram2_gate_bias_set) engram_sidecar2.set_gate_bias(engram2_gate_bias);
        engram_ptrs.push_back(&engram_sidecar2);
    }
    if (engram_ptrs.size() == 1) {
        engine.set_layer_hook(engram_ptrs[0]);
    } else if (engram_ptrs.size() > 1) {
        for (auto * sc : engram_ptrs) engram_composite.add(sc);
        engine.set_layer_hook(&engram_composite);
    }
    engine.warmup_decode_graphs();
    ready.store(true);

    std::string engram_summary = "off";
    if (engram_ptrs.size() == 1) {
        engram_summary = engram_ptrs[0]->table().cfg().arch;
    } else if (engram_ptrs.size() > 1) {
        engram_summary = engram_ptrs[0]->table().cfg().arch + "+" + engram_ptrs[1]->table().cfg().arch;
    }
    std::printf("rookery-server listening on http://%s:%d (arch=%s, alias=%s, ctx=%d, mtp=%s, engram=%s)\n",
                host.c_str(), port, engine.arch.c_str(), model_alias.c_str(), max_ctx,
                (use_mtp && engine.has_mtp()) ? "on" : "off",
                engram_summary.c_str());
    std::fflush(stdout);
    svr.listen(host, port);
    return 0;
}
