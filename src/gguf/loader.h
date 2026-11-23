#pragma once

#include "core/device.h"
#include "core/types.h"
#include "gguf/reader.h"
#include "gguf/sharded.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rookery {

struct DeviceTensor {
    void *   ptr        = nullptr; // Q8_0 SoA: qs[nb*32]; Q6_K SoA: ql[nb*128]
    void *   scales_ptr = nullptr; // Q8_0 SoA: __half[nb]; Q6_K SoA: int8_t scales[nb*16]
    void *   aux_ptr    = nullptr; // Q6_K SoA: uint8_t qh[nb*64]
    void *   d_ptr      = nullptr; // Q6_K SoA: uint16_t d[nb]
    DType    dtype      = DType::F32;
    int      device     = -1;
    uint64_t ne[4]      = {1, 1, 1, 1};
    size_t   nbytes     = 0;
    bool     is_soa     = false;

    bool valid() const { return ptr != nullptr; }
};

class WeightPool {
public:
    WeightPool() = default;
    ~WeightPool();

    WeightPool(const WeightPool &) = delete;
    WeightPool & operator=(const WeightPool &) = delete;

    // Upload a single GGUF tensor to target GPU `dev`.
    // Q8_0 tensors are automatically repacked on-GPU into Structure-of-Arrays (SoA)
    // for 100% coalesced 128-bit LDG.128 loads with ZERO extra VRAM!
    DeviceTensor upload(const GgufFile & gguf,
                        const std::string & name,
                        int dev,
                        bool convert_bf16_to_f16 = false,
                        bool required = true);

    DeviceTensor upload(const ShardedGguf & gguf,
                        const std::string & name,
                        int dev,
                        bool convert_bf16_to_f16 = false,
                        bool required = true);

    DeviceTensor upload_raw(const GgufTensorInfo & info,
                            const void * src,
                            int dev,
                            bool convert_bf16_to_f16 = false);

    void free_all() noexcept;
    void release_staging() noexcept;

    size_t bytes_on_device(int dev) const;

private:
    struct Alloc {
        int    device;
        void * ptr;
        size_t nbytes;
    };
    std::vector<Alloc> allocs_;
    std::unordered_map<int, size_t> dev_bytes_;

    // Reusable per-GPU staging buffer for fast Q8_0 / Q6_K -> SoA GPU unpack during load
    std::unordered_map<int, std::pair<void *, size_t>> staging_;
    mutable std::mutex mu_;
};

} // namespace rookery
