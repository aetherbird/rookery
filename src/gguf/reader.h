#pragma once

#include "core/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rookery {

enum class GgufValType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct GgufValue {
    GgufValType type = GgufValType::UINT32;
    std::variant<
        uint64_t,
        int64_t,
        double,
        bool,
        std::string,
        std::vector<uint64_t>,
        std::vector<int64_t>,
        std::vector<double>,
        std::vector<bool>,
        std::vector<std::string>
    > data;

    uint64_t    as_u64()  const;
    int64_t     as_i64()  const;
    double      as_f64()  const;
    bool        as_bool() const;
    std::string as_str()  const;
};

struct GgufTensorInfo {
    std::string           name;
    uint32_t              n_dims = 0;
    std::array<uint64_t, 4> ne   = {1, 1, 1, 1};
    DType                 dtype  = DType::F32;
    uint64_t              offset = 0;   // relative to data_offset
    size_t                nbytes = 0;
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();

    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;
    GgufFile(GgufFile && other) noexcept;
    GgufFile & operator=(GgufFile && other) noexcept;

    static GgufFile open(const std::string & path);

    uint32_t version()     const { return version_; }
    size_t   file_size()   const { return file_size_; }
    size_t   data_offset() const { return data_offset_; }

    bool has_kv(const std::string & key) const { return kv_.count(key) > 0; }
    const GgufValue & kv(const std::string & key) const;

    uint64_t    get_u64(const std::string & key, uint64_t def = 0) const;
    int64_t     get_i64(const std::string & key, int64_t def = 0) const;
    double      get_f64(const std::string & key, double def = 0.0) const;
    bool        get_bool(const std::string & key, bool def = false) const;
    std::string get_str(const std::string & key, const std::string & def = "") const;

    const std::vector<GgufTensorInfo> & tensors() const { return tensors_; }
    const GgufTensorInfo * find_tensor(const std::string & name) const;
    const void *           tensor_data(const GgufTensorInfo & info) const;

    const std::unordered_map<std::string, GgufValue> & all_kv() const { return kv_; }

private:
    void close_mmap() noexcept;

    int      fd_          = -1;
    uint8_t *base_        = nullptr;
    size_t   file_size_   = 0;
    uint32_t version_     = 0;
    size_t   data_offset_ = 0;
    uint64_t alignment_   = 32;

    std::unordered_map<std::string, GgufValue> kv_;
    std::vector<GgufTensorInfo>                tensors_;
    std::unordered_map<std::string, size_t>    tensor_idx_;
};

} // namespace rookery
