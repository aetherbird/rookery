#include "tokenizer/bpe.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace rookery {

namespace {

std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

bool decode_utf8_at(const std::string & s, size_t pos, uint32_t & cp, size_t & len) {
    if (pos >= s.size()) return false;
    uint8_t c0 = static_cast<uint8_t>(s[pos]);
    if (c0 < 0x80) {
        cp = c0; len = 1; return true;
    } else if ((c0 & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        cp = ((c0 & 0x1F) << 6) | (static_cast<uint8_t>(s[pos + 1]) & 0x3F);
        len = 2; return true;
    } else if ((c0 & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        cp = ((c0 & 0x0F) << 12) | ((static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6) |
             (static_cast<uint8_t>(s[pos + 2]) & 0x3F);
        len = 3; return true;
    } else if ((c0 & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        cp = ((c0 & 0x07) << 18) | ((static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12) |
             ((static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[pos + 3]) & 0x3F);
        len = 4; return true;
    }
    cp = c0; len = 1; return true;
}

inline bool is_han(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

inline bool is_sea(uint32_t cp) {
    return (cp >= 0x0E00 && cp <= 0x0E7F) || (cp >= 0x0E80 && cp <= 0x0EFF) ||
           (cp >= 0x1000 && cp <= 0x109F) || (cp >= 0x1780 && cp <= 0x17FF);
}

inline bool is_ascii_upper(uint32_t cp) { return cp >= 'A' && cp <= 'Z'; }
inline bool is_ascii_lower(uint32_t cp) { return cp >= 'a' && cp <= 'z'; }
inline bool is_letter(uint32_t cp) {
    return is_ascii_upper(cp) || is_ascii_lower(cp) ||
           (cp >= 0x00C0 && cp <= 0x024F) || (cp >= 0x0370 && cp <= 0x052F) ||
           (cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0900 && cp <= 0x097F) ||
           (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0xAC00 && cp <= 0xD7AF) ||
           is_han(cp) || is_sea(cp);
}
inline bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_space(uint32_t cp) { return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f'; }

} // namespace

void BpeTokenizer::init_byte_tables() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = in_bs[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
        std::string u = utf8_encode(cp);
        byte_to_unicode_[b] = u;
        unicode_to_byte_[u] = static_cast<uint8_t>(b);
    }
}

void BpeTokenizer::load(const GgufFile & gguf) {
    init_byte_tables();

    bos_id_          = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.bos_token_id", 1));
    eos_id_          = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.eos_token_id", 2));
    eot_id_          = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.eot_token_id", 200001));
    add_bos_default_ = gguf.get_bool("tokenizer.ggml.add_bos_token", true);
    const std::string tok_model = gguf.get_str("tokenizer.ggml.model", "");
    const std::string pre = gguf.get_str("tokenizer.ggml.pre", "");
    is_spm_bpe_ = (tok_model == "gemma4" || tok_model == "spm" || tok_model == "llama");
    max_digits_ = (pre == "qwen35" || pre == "qwen2" || pre == "bailingmoe" || pre == "bailingmoe2") ? 1 : 3;
    use_llama3_pre_ = (pre == "qwen35" || pre == "qwen2" || pre == "glm4" || pre == "glm5" ||
                       pre == "llama3" || pre == "llama-bpe" || pre == "bailingmoe" || pre == "bailingmoe2");

    const auto & tok_val = gguf.kv("tokenizer.ggml.tokens");
    const auto * tok_vec = std::get_if<std::vector<std::string>>(&tok_val.data);
    if (!tok_vec) throw std::runtime_error("tokenizer.ggml.tokens is not string array");

    id_to_token_ = *tok_vec;
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size() * 2);
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<int32_t>(i);
    }

    token_types_.assign(id_to_token_.size(), 1);
    if (gguf.has_kv("tokenizer.ggml.token_type")) {
        const auto & tt = gguf.kv("tokenizer.ggml.token_type");
        if (auto * v = std::get_if<std::vector<int64_t>>(&tt.data)) {
            for (size_t i = 0; i < v->size() && i < token_types_.size(); ++i) token_types_[i] = static_cast<int32_t>((*v)[i]);
        } else if (auto * v = std::get_if<std::vector<uint64_t>>(&tt.data)) {
            for (size_t i = 0; i < v->size() && i < token_types_.size(); ++i) token_types_[i] = static_cast<int32_t>((*v)[i]);
        }
    }

    special_tokens_.clear();
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        // Control (3) or User-Defined (4) tokens like <|maxwell_start|>, <think>, <tool_call>
        if ((token_types_[i] == 3 || token_types_[i] == 4) && !id_to_token_[i].empty()) {
            special_tokens_.emplace_back(id_to_token_[i], static_cast<int32_t>(i));
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });

    merge_ranks_.clear();
    if (gguf.has_kv("tokenizer.ggml.merges")) {
        const auto & mv = gguf.kv("tokenizer.ggml.merges");
        if (auto * merges = std::get_if<std::vector<std::string>>(&mv.data)) {
            merge_ranks_.reserve(merges->size() * 2);
            for (size_t i = 0; i < merges->size(); ++i) {
                merge_ranks_.emplace((*merges)[i], static_cast<int32_t>(i));
            }
        }
    }
    extra_stop_ids_.clear();

    // Precompute O(1) decoded UTF-8 byte strings for every token ID (eliminates per-token UTF-8 + hash lookup in decode_token)
    decoded_tokens_.resize(id_to_token_.size());
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        if (token_types_[i] == 3 || token_types_[i] == 4) {
            decoded_tokens_[i] = id_to_token_[i];
            continue;
        }
        const std::string & tok = id_to_token_[i];
        if (is_spm_bpe_) {
            if (token_types_[i] == 6 && tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
                auto hexv = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                    return 0;
                };
                char b = static_cast<char>((hexv(tok[3]) << 4) | hexv(tok[4]));
                decoded_tokens_[i] = std::string(1, b);
                continue;
            }
            std::string out;
            out.reserve(tok.size());
            for (size_t pos = 0; pos < tok.size();) {
                if (pos + 2 < tok.size() &&
                    static_cast<uint8_t>(tok[pos]) == 0xE2 &&
                    static_cast<uint8_t>(tok[pos + 1]) == 0x96 &&
                    static_cast<uint8_t>(tok[pos + 2]) == 0x81) {
                    out.push_back(' ');
                    pos += 3;
                } else {
                    out.push_back(tok[pos]);
                    ++pos;
                }
            }
            decoded_tokens_[i] = std::move(out);
            continue;
        }
        std::string out;
        out.reserve(tok.size());
        for (size_t pos = 0; pos < tok.size();) {
            uint32_t cp = 0; size_t len = 0;
            decode_utf8_at(tok, pos, cp, len);
            std::string ch = tok.substr(pos, len);
            auto it = unicode_to_byte_.find(ch);
            if (it != unicode_to_byte_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                out.append(ch);
            }
            pos += len;
        }
        decoded_tokens_[i] = std::move(out);
    }
}

void BpeTokenizer::add_stop_token(const std::string & tok) {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) {
        if (std::find(extra_stop_ids_.begin(), extra_stop_ids_.end(), it->second) == extra_stop_ids_.end()) {
            extra_stop_ids_.push_back(it->second);
        }
    }
}

bool BpeTokenizer::is_stop_token(int32_t id) const {
    if (id == eos_id_ || id == eot_id_) return true;
    for (int32_t sid : extra_stop_ids_) {
        if (id == sid) return true;
    }
    return false;
}

// Maxwell pre-tokenizer: splits newlines, Han runs, SEA runs, and o200k-style word/number/punct/space runs.
std::vector<std::string> BpeTokenizer::pretokenize_maxwell(const std::string & text) const {
    struct CpSpan { uint32_t cp; size_t off; size_t len; };
    std::vector<CpSpan> cps;
    cps.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        uint32_t cp = 0; size_t len = 0;
        decode_utf8_at(text, pos, cp, len);
        cps.push_back({cp, pos, len});
        pos += len;
    }

    auto slice = [&](size_t a, size_t b) {
        size_t s = cps[a].off;
        size_t e = cps[b - 1].off + cps[b - 1].len;
        return text.substr(s, e - s);
    };

    std::vector<std::string> pieces;
    size_t i = 0, n = cps.size();
    while (i < n) {
        uint32_t c = cps[i].cp;

        // 1. Standalone newline
        if (c == '\n') {
            pieces.push_back("\n");
            ++i;
            continue;
        }

        // 2. Han run with optional leading non-letter/non-digit (not \n)
        if (is_han(c) || (!is_letter(c) && !is_digit(c) && c != '\n' && i + 1 < n && is_han(cps[i + 1].cp))) {
            size_t j = is_han(c) ? i : i + 1;
            while (j < n && is_han(cps[j].cp)) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 3. SEA run
        if (is_sea(c)) {
            size_t j = i + 1;
            while (j < n && is_sea(cps[j].cp)) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 4. Digits (1..3 digits per chunk for o200k/GPT4O; 1 digit for Qwen2/Qwen3.5)
        if (is_digit(c)) {
            size_t j = i + 1;
            while (j < n && static_cast<int>(j - i) < max_digits_ && is_digit(cps[j].cp)) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 5. Optional single non-letter/non-digit prefix (e.g. space) + word + optional English contraction
        bool has_word_start = is_letter(c) ||
            (!is_letter(c) && !is_digit(c) && c != '\n' && c != '\r' && i + 1 < n && is_letter(cps[i + 1].cp) && !is_han(cps[i + 1].cp) && !is_sea(cps[i + 1].cp));
        if (has_word_start) {
            size_t j = is_letter(c) ? i : i + 1;
            // Leading uppercase run (if followed by lowercase, split CamelCase only when appropriate; o200k keeps Title/lower together)
            while (j < n && is_ascii_upper(cps[j].cp)) ++j;
            while (j < n && is_letter(cps[j].cp) && !is_ascii_upper(cps[j].cp) && !is_han(cps[j].cp) && !is_sea(cps[j].cp)) ++j;
            // Contraction suffix: 's|'t|'re|'ve|'m|'ll|'d (case-insensitive)
            if (j + 1 < n && cps[j].cp == '\'') {
                char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(cps[j + 1].cp)));
                if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                    j += 2;
                } else if (j + 2 < n) {
                    char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(cps[j + 2].cp)));
                    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                        j += 3;
                    }
                }
            }
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 6. Optional single space + punctuation/symbol run (+ trailing / or \r) per o200k:
        //    ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
        bool space_punct = (c == ' ' && i + 1 < n && !is_space(cps[i + 1].cp) &&
                            !is_letter(cps[i + 1].cp) && !is_digit(cps[i + 1].cp));
        if (space_punct || (!is_space(c) && !is_letter(c) && !is_digit(c))) {
            size_t j = space_punct ? (i + 2) : (i + 1);
            while (j < n && !is_space(cps[j].cp) && !is_letter(cps[j].cp) && !is_digit(cps[j].cp)) ++j;
            while (j < n && (cps[j].cp == '\r' || cps[j].cp == '/')) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 7. Whitespace run (excluding \n which is handled above)
        if (is_space(c)) {
            size_t j = i + 1;
            while (j < n && is_space(cps[j].cp) && cps[j].cp != '\n') ++j;
            // o200k rule: \s+(?!\S) keeps trailing space attached to next word/punct if len > 1
            if (j - i > 1 && j < n && !is_space(cps[j].cp)) {
                --j;
            }
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }
    }
    return pieces;
}

// Exact Llama3 / Qwen2 / Qwen3.5 / GLM4 / GLM5 pre-tokenizer regex:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max_digits}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::string> BpeTokenizer::pretokenize_llama3(const std::string & text) const {
    struct CpSpan { uint32_t cp; size_t off; size_t len; };
    std::vector<CpSpan> cps;
    cps.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        uint32_t cp = 0; size_t len = 0;
        decode_utf8_at(text, pos, cp, len);
        cps.push_back({cp, pos, len});
        pos += len;
    }

    auto slice = [&](size_t a, size_t b) {
        size_t s = cps[a].off;
        size_t e = cps[b - 1].off + cps[b - 1].len;
        return text.substr(s, e - s);
    };

    std::vector<std::string> pieces;
    size_t i = 0, n = cps.size();
    while (i < n) {
        uint32_t c = cps[i].cp;

        // 1. Contraction: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n) {
            char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(cps[i + 1].cp)));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pieces.push_back(slice(i, i + 2));
                i += 2;
                continue;
            }
            if (i + 2 < n) {
                char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(cps[i + 2].cp)));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pieces.push_back(slice(i, i + 3));
                    i += 3;
                    continue;
                }
            }
        }

        // 2. Word: [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(c) || (c != '\r' && c != '\n' && !is_digit(c) && i + 1 < n && is_letter(cps[i + 1].cp))) {
            size_t j = is_letter(c) ? (i + 1) : (i + 2);
            while (j < n && is_letter(cps[j].cp)) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 3. Digits: \p{N}{1,max_digits}
        if (is_digit(c)) {
            size_t j = i + 1;
            while (j < n && static_cast<int>(j - i) < max_digits_ && is_digit(cps[j].cp)) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 4. Punctuation + trailing newlines:  ?[^\s\p{L}\p{N}]+[\r\n]*
        bool sp_punct = (c == ' ' && i + 1 < n && !is_space(cps[i + 1].cp) &&
                         !is_letter(cps[i + 1].cp) && !is_digit(cps[i + 1].cp));
        if (sp_punct || (!is_space(c) && !is_letter(c) && !is_digit(c))) {
            size_t j = sp_punct ? (i + 2) : (i + 1);
            while (j < n && !is_space(cps[j].cp) && !is_letter(cps[j].cp) && !is_digit(cps[j].cp)) ++j;
            while (j < n && (cps[j].cp == '\r' || cps[j].cp == '\n')) ++j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        // 5/6/7. Whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
        if (is_space(c)) {
            size_t j = i;
            size_t last_nl = std::string::npos;
            while (j < n && is_space(cps[j].cp)) {
                if (cps[j].cp == '\r' || cps[j].cp == '\n') last_nl = j;
                ++j;
            }
            if (last_nl != std::string::npos) {
                // \s*[\r\n]+ consumes up through the last newline in this whitespace run
                pieces.push_back(slice(i, last_nl + 1));
                i = last_nl + 1;
                continue;
            }
            // \s+(?!\S) keeps last single space for the next non-space token if run len > 1
            if (j - i > 1 && j < n) --j;
            pieces.push_back(slice(i, j));
            i = j;
            continue;
        }

        pieces.push_back(slice(i, i + 1));
        ++i;
    }
    return pieces;
}

void BpeTokenizer::bpe_encode_piece(const std::string & piece, std::vector<int32_t> & out) const {
    if (piece.empty()) return;

    std::vector<std::string> syms;
    syms.reserve(piece.size());
    if (is_spm_bpe_) {
        static const char kHex[] = "0123456789ABCDEF";
        for (size_t pos = 0; pos < piece.size();) {
            if (piece[pos] == ' ') {
                syms.emplace_back("\xE2\x96\x81");
                ++pos;
                continue;
            }
            uint32_t cp = 0; size_t len = 0;
            decode_utf8_at(piece, pos, cp, len);
            std::string ch = piece.substr(pos, len);
            if (token_to_id_.find(ch) != token_to_id_.end()) {
                syms.push_back(std::move(ch));
            } else {
                for (size_t k = 0; k < len; ++k) {
                    uint8_t b = static_cast<uint8_t>(piece[pos + k]);
                    std::string hex = "<0x00>";
                    hex[3] = kHex[b >> 4];
                    hex[4] = kHex[b & 0xF];
                    syms.push_back(std::move(hex));
                }
            }
            pos += len;
        }
    } else {
        for (unsigned char b : piece) {
            syms.push_back(byte_to_unicode_[b]);
        }
    }

    while (syms.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t  best_idx  = std::string::npos;

        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            std::string pair = syms[i] + " " + syms[i + 1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
            }
        }
        if (best_idx == std::string::npos) break;

        syms[best_idx] += syms[best_idx + 1];
        syms.erase(syms.begin() + static_cast<ptrdiff_t>(best_idx + 1));
    }

    for (const auto & s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        }
    }
}

std::vector<int32_t> BpeTokenizer::encode(const std::string & text, bool add_bos, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos && add_bos_default_) {
        out.push_back(bos_id_);
    }

    size_t pos = 0;
    size_t chunk_start = 0;
    while (pos < text.size()) {
        int32_t matched_id = -1;
        size_t  matched_len = 0;

        if (parse_special) {
            for (const auto & [st, sid] : special_tokens_) {
                if (pos + st.size() <= text.size() && text.compare(pos, st.size(), st) == 0) {
                    matched_id  = sid;
                    matched_len = st.size();
                    break;
                }
            }
        }

        if (matched_id >= 0) {
            if (pos > chunk_start) {
                const std::string seg = text.substr(chunk_start, pos - chunk_start);
                const auto pcs = use_llama3_pre_ ? pretokenize_llama3(seg) : pretokenize_maxwell(seg);
                for (const auto & p : pcs) bpe_encode_piece(p, out);
            }
            out.push_back(matched_id);
            pos += matched_len;
            chunk_start = pos;
        } else {
            ++pos;
        }
    }

    if (chunk_start < text.size()) {
        const std::string seg = text.substr(chunk_start);
        const auto pcs = use_llama3_pre_ ? pretokenize_llama3(seg) : pretokenize_maxwell(seg);
        for (const auto & p : pcs) bpe_encode_piece(p, out);
    }
    return out;
}

std::string BpeTokenizer::decode_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= decoded_tokens_.size()) return "";
    return decoded_tokens_[id];
}

std::string BpeTokenizer::decode(const std::vector<int32_t> & ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        if (skip_special && (token_types_[id] == 3 || token_types_[id] == 4)) continue;
        out.append(decode_token(id));
    }
    return out;
}

} // namespace rookery
