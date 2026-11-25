#pragma once

#include "gguf/reader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rookery {

class BpeTokenizer {
public:
    void load(const GgufFile & gguf);

    std::vector<int32_t> encode(const std::string & text, bool add_bos = true, bool parse_special = true) const;
    std::string          decode(const std::vector<int32_t> & ids, bool skip_special = false) const;
    std::string          decode_token(int32_t id) const;
    size_t               token_byte_len(int32_t id) const {
        return (id >= 0 && static_cast<size_t>(id) < decoded_tokens_.size()) ? decoded_tokens_[id].size() : 0;
    }

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t eot_id() const { return eot_id_; }
    bool    add_bos_default() const { return add_bos_default_; }
    void    add_stop_token(const std::string & tok);
    bool    is_stop_token(int32_t id) const;
    bool    is_special_token(int32_t id) const {
        return (id >= 0 && static_cast<size_t>(id) < token_types_.size()) &&
               (token_types_[id] == 3 || token_types_[id] == 4);
    }
    int32_t token_type(int32_t id) const {
        return (id >= 0 && static_cast<size_t>(id) < token_types_.size()) ? token_types_[id] : 1;
    }
    size_t  vocab_size() const { return id_to_token_.size(); }

private:
    void init_byte_tables();
    std::vector<std::string> pretokenize_maxwell(const std::string & text) const;
    std::vector<std::string> pretokenize_llama3(const std::string & text) const;
    void bpe_encode_piece(const std::string & piece, std::vector<int32_t> & out) const;
    void bpe_encode_spm_segment(const std::string & seg, std::vector<int32_t> & out) const;

    std::vector<std::string>                id_to_token_;
    std::vector<std::string>                decoded_tokens_;
    std::vector<int32_t>                    token_types_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_; // sorted longest-first
    std::vector<int32_t>                    extra_stop_ids_;

    std::string                              byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t eot_id_ = 200001;
    bool    add_bos_default_ = true;
    int     max_digits_ = 3;
    bool    use_llama3_pre_ = false;
    bool    use_spm_bpe_ = false;
};

} // namespace rookery
