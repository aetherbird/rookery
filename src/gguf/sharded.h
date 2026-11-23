#pragma once

#include "gguf/reader.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace rookery {

// Transparent multi-shard GGUF v3 reader (also works on single-file GGUFs).
// Auto-discovers sibling shards `*-00001-of-0000N.gguf` .. `*-0000N-of-0000N.gguf`,
// reads KV metadata from shard 0, and maps every tensor name across all shards in O(1).
class ShardedGguf {
public:
    ShardedGguf() = default;

    static ShardedGguf open(const std::string & path);

    const GgufFile & meta() const { return shards_.front(); }
    size_t num_shards() const { return shards_.size(); }

    bool has_kv(const std::string & key) const { return meta().has_kv(key); }
    const GgufValue & kv(const std::string & key) const { return meta().kv(key); }

    uint64_t    get_u64(const std::string & key, uint64_t def = 0) const { return meta().get_u64(key, def); }
    int64_t     get_i64(const std::string & key, int64_t def = 0) const { return meta().get_i64(key, def); }
    double      get_f64(const std::string & key, double def = 0.0) const { return meta().get_f64(key, def); }
    bool        get_bool(const std::string & key, bool def = false) const { return meta().get_bool(key, def); }
    std::string get_str(const std::string & key, const std::string & def = "") const { return meta().get_str(key, def); }

    struct TensorLoc {
        const GgufFile *       shard = nullptr;
        const GgufTensorInfo * info  = nullptr;
        const void * data() const { return shard->tensor_data(*info); }
    };

    TensorLoc find_tensor(const std::string & name) const;
    size_t total_tensors() const { return index_.size(); }
    size_t total_bytes() const;

private:
    std::vector<GgufFile>                     shards_;
    std::unordered_map<std::string, size_t>   index_; // tensor name -> shard index
};

} // namespace rookery
