#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rookery {

// Matches GGML / GGUF v3 ggml_type numeric IDs exactly.
enum class DType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    NVFP4   = 40,
    Q1_0    = 41,
    Q2_0    = 42,
    F8_E4M3 = 43,
};

struct DTypeTraits {
    const char * name;
    uint32_t     block_elems;
    uint32_t     block_bytes;
};

inline DTypeTraits dtype_traits(DType t) {
    switch (t) {
        case DType::F32:     return {"F32",     1,   4};
        case DType::F16:     return {"F16",     1,   2};
        case DType::BF16:    return {"BF16",    1,   2};
        case DType::Q4_0:    return {"Q4_0",   32,  18};
        case DType::Q4_1:    return {"Q4_1",   32,  20};
        case DType::Q5_0:    return {"Q5_0",   32,  22};
        case DType::Q5_1:    return {"Q5_1",   32,  24};
        case DType::Q8_0:    return {"Q8_0",   32,  34};
        case DType::Q8_1:    return {"Q8_1",   32,  36};
        case DType::Q2_K:    return {"Q2_K",  256,  84};
        case DType::Q3_K:    return {"Q3_K",  256, 110};
        case DType::Q4_K:    return {"Q4_K",  256, 144};
        case DType::Q5_K:    return {"Q5_K",  256, 176};
        case DType::Q6_K:    return {"Q6_K",  256, 210};
        case DType::Q8_K:    return {"Q8_K",  256, 292};
        case DType::IQ2_XXS: return {"IQ2_XXS",256, 66};
        case DType::IQ2_XS:  return {"IQ2_XS", 256, 74};
        case DType::IQ3_XXS: return {"IQ3_XXS",256, 98};
        case DType::IQ1_S:   return {"IQ1_S",  256, 50};
        case DType::IQ4_NL:  return {"IQ4_NL",  32, 18};
        case DType::IQ3_S:   return {"IQ3_S",  256,110};
        case DType::IQ2_S:   return {"IQ2_S",  256, 82};
        case DType::IQ4_XS:  return {"IQ4_XS", 256,136};
        case DType::IQ1_M:   return {"IQ1_M",  256, 56};
        case DType::TQ1_0:   return {"TQ1_0",  256, 54};
        case DType::TQ2_0:   return {"TQ2_0",  256, 66};
        case DType::MXFP4:   return {"MXFP4",   32, 17};
        case DType::NVFP4:   return {"NVFP4",  64, 36};
        case DType::F8_E4M3: return {"F8_E4M3",  1,  1};
        case DType::I8:      return {"I8",      1,   1};
        case DType::I16:     return {"I16",     1,   2};
        case DType::I32:     return {"I32",     1,   4};
        case DType::I64:     return {"I64",     1,   8};
        case DType::F64:     return {"F64",     1,   8};
        default:             return {"UNKNOWN", 1,   0};
    }
}

inline size_t tensor_nbytes(DType t, uint64_t ne0, uint64_t ne1 = 1, uint64_t ne2 = 1, uint64_t ne3 = 1) {
    const auto tr = dtype_traits(t);
    if (tr.block_bytes == 0) {
        throw std::runtime_error("Unsupported DType id=" + std::to_string(static_cast<uint32_t>(t)));
    }
    if (ne0 % tr.block_elems != 0) {
        throw std::runtime_error(std::string("ne0=") + std::to_string(ne0) +
                                 " not divisible by block_elems for " + tr.name);
    }
    const uint64_t row_bytes = (ne0 / tr.block_elems) * tr.block_bytes;
    return static_cast<size_t>(row_bytes * ne1 * ne2 * ne3);
}

// Quant block structs matching ggml on-disk layout verbatim.
struct BlockQ8_0 {
    uint16_t d;       // fp16 scale
    int8_t   qs[32];  // int8 quants
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

// Pair-packed Q4_0 block for KV & indexer caches:
//   qs[j] stores elem 2*j in low nibble (bits 0..3) and elem 2*j+1 in high nibble (bits 4..7),
//   each nibble in [0..15], dequant = d * (nibble - 8).
struct BlockQ4_0 {
    uint16_t d;       // fp16 scale
    uint8_t  qs[16];  // 32 x 4-bit unsigned nibbles
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

enum class KvCacheType : uint8_t {
    F16  = 0,
    Q8_0 = 1,
    Q4_0 = 2,
};

inline const char * kv_cache_type_name(KvCacheType t) {
    switch (t) {
        case KvCacheType::F16:  return "F16";
        case KvCacheType::Q8_0: return "Q8_0";
        case KvCacheType::Q4_0: return "Q4_0";
    }
    return "UNKNOWN";
}

inline KvCacheType parse_kv_cache_type(const std::string & s) {
    if (s == "f16" || s == "F16" || s == "fp16" || s == "FP16") return KvCacheType::F16;
    if (s == "q4_0" || s == "Q4_0" || s == "q4" || s == "Q4")   return KvCacheType::Q4_0;
    return KvCacheType::Q8_0;
}

} // namespace rookery
