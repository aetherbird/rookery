#include "gguf/loader.h"

#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline size_t align256(size_t n) {
    return (n + 255) & ~size_t(255);
}

inline uint16_t bf16_to_f16_bits(uint16_t b) {
    uint32_t f32_bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    __half h = __float2half_rn(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(uint16_t));
    return out;
}

// Fast GPU unpack: packed 34-byte BlockQ8_0[n_blocks] -> SoA (qs[n_blocks*32], scales[n_blocks]).
// 1 warp per block -> 32 threads write 32 coalesced int8 bytes + lane 0 writes __half scale.
__global__ void unpack_q8_0_to_soa_kernel(
    const BlockQ8_0 * __restrict__ src,
    int8_t *          __restrict__ dst_qs,
    uint16_t *        __restrict__ dst_scales,
    int64_t n_blocks) {

    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x;
    dst_qs[b * 32 + lane] = src[b].qs[lane];
    if (lane == 0) {
        dst_scales[b] = src[b].d;
    }
}

// Packed on-disk 210-byte BlockQ6_K (QK_K = 256)
struct PackedBlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(PackedBlockQ6_K) == 210, "PackedBlockQ6_K size");

// Fast GPU unpack: packed 210-byte BlockQ6_K[n_blocks] -> 256B-aligned SoA (ql, qh, scales, d).
// Enables 100% coalesced 128-bit LDG.128 loads in gemv_q6_k_soa_dp4a_kernel with ZERO extra VRAM!
__global__ void unpack_q6_k_to_soa_kernel(
    const PackedBlockQ6_K * __restrict__ src,
    uint8_t *               __restrict__ dst_ql,
    uint8_t *               __restrict__ dst_qh,
    int8_t *                __restrict__ dst_sc,
    uint16_t *              __restrict__ dst_d,
    int64_t n_blocks) {

    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x;
    const PackedBlockQ6_K & s = src[b];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst_ql[b * 128 + i * 32 + lane] = s.ql[i * 32 + lane];
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        dst_qh[b * 64 + i * 32 + lane] = s.qh[i * 32 + lane];
    }
    if (lane < 16) dst_sc[b * 16 + lane] = s.scales[lane];
    if (lane == 0) dst_d[b] = s.d;
}

} // namespace

WeightPool::~WeightPool() {
    free_all();
}

void WeightPool::release_staging() noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto & [dev, ps] : staging_) {
        if (ps.first) {
            cudaSetDevice(dev);
            cudaFree(ps.first);
        }
    }
    staging_.clear();
}

void WeightPool::free_all() noexcept {
    release_staging();
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto & a : allocs_) {
        if (a.ptr) {
            cudaSetDevice(a.device);
            cudaFree(a.ptr);
        }
    }
    allocs_.clear();
    dev_bytes_.clear();
}

size_t WeightPool::bytes_on_device(int dev) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = dev_bytes_.find(dev);
    return it == dev_bytes_.end() ? 0 : it->second;
}

DeviceTensor WeightPool::upload(const GgufFile & gguf,
                                const std::string & name,
                                int dev,
                                bool convert_bf16_to_f16,
                                bool required) {
    const GgufTensorInfo * info = gguf.find_tensor(name);
    if (!info) {
        if (required) {
            throw std::runtime_error("Required tensor missing from GGUF: " + name);
        }
        return {};
    }
    return upload_raw(*info, gguf.tensor_data(*info), dev, convert_bf16_to_f16);
}

DeviceTensor WeightPool::upload(const ShardedGguf & gguf,
                                const std::string & name,
                                int dev,
                                bool convert_bf16_to_f16,
                                bool required) {
    auto loc = gguf.find_tensor(name);
    if (!loc.info) {
        if (required) {
            throw std::runtime_error("Required tensor missing from sharded GGUF: " + name);
        }
        return {};
    }
    return upload_raw(*loc.info, loc.data(), dev, convert_bf16_to_f16);
}

DeviceTensor WeightPool::upload_raw(const GgufTensorInfo & info_ref,
                                    const void * src,
                                    int dev,
                                    bool convert_bf16_to_f16) {
    const GgufTensorInfo * info = &info_ref;
    const std::string & name = info->name;
    DType dst_dtype  = info->dtype;
    size_t dst_bytes = info->nbytes;

    cuda_ok(cudaSetDevice(dev), "WeightPool::upload cudaSetDevice");

    auto ensure_staging = [&](size_t req_bytes) -> void * {
        void * cur_ptr = nullptr;
        size_t cur_sz  = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = staging_.find(dev);
            if (it != staging_.end()) {
                cur_ptr = it->second.first;
                cur_sz  = it->second.second;
            }
        }
        if (cur_sz < req_bytes) {
            if (cur_ptr) cudaFree(cur_ptr);
            cur_ptr = nullptr;
            cuda_ok(cudaMalloc(&cur_ptr, req_bytes), ("cudaMalloc staging for " + name).c_str());
            std::lock_guard<std::mutex> lk(mu_);
            staging_[dev] = {cur_ptr, req_bytes};
        }
        return cur_ptr;
    };

    if (info->dtype == DType::Q8_0) {
        const int64_t n_blocks = static_cast<int64_t>(info->nbytes / sizeof(BlockQ8_0));
        const size_t  qs_bytes = align256(static_cast<size_t>(n_blocks) * 32);
        const size_t  sc_bytes = align256(static_cast<size_t>(n_blocks) * sizeof(uint16_t));
        const size_t  tot_bytes = qs_bytes + sc_bytes;

        void * st_ptr = ensure_staging(info->nbytes);
        void * dptr = nullptr;
        cuda_ok(cudaMalloc(&dptr, tot_bytes), ("cudaMalloc SoA Q8_0 for " + name).c_str());
        cuda_ok(cudaMemcpy(st_ptr, src, info->nbytes, cudaMemcpyHostToDevice),
                ("cudaMemcpy staging Q8_0 for " + name).c_str());

        auto * d_qs = static_cast<int8_t *>(dptr);
        auto * d_sc = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(dptr) + qs_bytes);

        constexpr int wpb = 8;
        const int64_t grid = (n_blocks + wpb - 1) / wpb;
        unpack_q8_0_to_soa_kernel<<<static_cast<unsigned int>(grid), dim3(32, wpb)>>>(
            static_cast<const BlockQ8_0 *>(st_ptr), d_qs, d_sc, n_blocks);
        cuda_ok(cudaGetLastError(), ("unpack_q8_0_to_soa launch for " + name).c_str());

        {
            std::lock_guard<std::mutex> lk(mu_);
            allocs_.push_back({dev, dptr, tot_bytes});
            dev_bytes_[dev] += tot_bytes;
        }

        DeviceTensor dt;
        dt.ptr        = d_qs;
        dt.scales_ptr = d_sc;
        dt.dtype      = DType::Q8_0;
        dt.device     = dev;
        for (int i = 0; i < 4; ++i) dt.ne[i] = info->ne[i];
        dt.nbytes     = tot_bytes;
        dt.is_soa     = true;
        return dt;
    }

    if (info->dtype == DType::Q6_K) {
        const int64_t n_blocks  = static_cast<int64_t>(info->nbytes / sizeof(PackedBlockQ6_K));
        const size_t  ql_bytes  = align256(static_cast<size_t>(n_blocks) * 128);
        const size_t  qh_bytes  = align256(static_cast<size_t>(n_blocks) * 64);
        const size_t  sc_bytes  = align256(static_cast<size_t>(n_blocks) * 16);
        const size_t  d_bytes   = align256(static_cast<size_t>(n_blocks) * sizeof(uint16_t));
        const size_t  tot_bytes = ql_bytes + qh_bytes + sc_bytes + d_bytes;

        void * st_ptr = ensure_staging(info->nbytes);
        void * dptr = nullptr;
        cuda_ok(cudaMalloc(&dptr, tot_bytes), ("cudaMalloc SoA Q6_K for " + name).c_str());
        cuda_ok(cudaMemcpy(st_ptr, src, info->nbytes, cudaMemcpyHostToDevice),
                ("cudaMemcpy staging Q6_K for " + name).c_str());

        auto * base = static_cast<uint8_t *>(dptr);
        auto * d_ql = base;
        auto * d_qh = base + ql_bytes;
        auto * d_sc = reinterpret_cast<int8_t *>(base + ql_bytes + qh_bytes);
        auto * d_d  = reinterpret_cast<uint16_t *>(base + ql_bytes + qh_bytes + sc_bytes);

        constexpr int wpb = 8;
        const int64_t grid = (n_blocks + wpb - 1) / wpb;
        unpack_q6_k_to_soa_kernel<<<static_cast<unsigned int>(grid), dim3(32, wpb)>>>(
            static_cast<const PackedBlockQ6_K *>(st_ptr), d_ql, d_qh, d_sc, d_d, n_blocks);
        cuda_ok(cudaGetLastError(), ("unpack_q6_k_to_soa launch for " + name).c_str());

        {
            std::lock_guard<std::mutex> lk(mu_);
            allocs_.push_back({dev, dptr, tot_bytes});
            dev_bytes_[dev] += tot_bytes;
        }

        DeviceTensor dt;
        dt.ptr        = d_ql;
        dt.aux_ptr    = d_qh;
        dt.scales_ptr = d_sc;
        dt.d_ptr      = d_d;
        dt.dtype      = DType::Q6_K;
        dt.device     = dev;
        for (int i = 0; i < 4; ++i) dt.ne[i] = info->ne[i];
        dt.nbytes     = tot_bytes;
        dt.is_soa     = true;
        return dt;
    }

    void * dptr = nullptr;
    cuda_ok(cudaMalloc(&dptr, dst_bytes), ("cudaMalloc failed for " + name).c_str());

    if (info->dtype == DType::BF16 && convert_bf16_to_f16) {
        const size_t n_elems = info->nbytes / 2;
        std::vector<uint16_t> tmp(n_elems);
        const auto * s16 = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n_elems; ++i) {
            tmp[i] = bf16_to_f16_bits(s16[i]);
        }
        cuda_ok(cudaMemcpy(dptr, tmp.data(), dst_bytes, cudaMemcpyHostToDevice),
                ("cudaMemcpy BF16->F16 failed for " + name).c_str());
        dst_dtype = DType::F16;
    } else {
        cuda_ok(cudaMemcpy(dptr, src, dst_bytes, cudaMemcpyHostToDevice),
                ("cudaMemcpy failed for " + name).c_str());
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        allocs_.push_back({dev, dptr, dst_bytes});
        dev_bytes_[dev] += dst_bytes;
    }

    DeviceTensor dt;
    dt.ptr    = dptr;
    dt.dtype  = dst_dtype;
    dt.device = dev;
    for (int i = 0; i < 4; ++i) dt.ne[i] = info->ne[i];
    dt.nbytes = dst_bytes;
    return dt;
}

namespace {

inline float f16_bits_to_f32(uint16_t h) {
    __half hh;
    std::memcpy(static_cast<void *>(&hh), &h, 2);
    return __half2float(hh);
}

inline uint16_t f32_to_f16_bits(float f) {
    __half hh = __float2half_rn(f);
    uint16_t out;
    std::memcpy(&out, &hh, 2);
    return out;
}

inline void get_scale_min_k4_host(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

void dequant_to_f16_host(DType dtype, const void * src, size_t n_elems, uint16_t * dst) {
    const auto * p = static_cast<const uint8_t *>(src);
    if (dtype == DType::F16) {
        std::memcpy(dst, p, n_elems * 2);
        return;
    }
    if (dtype == DType::F32) {
        const auto * s = reinterpret_cast<const float *>(p);
        for (size_t i = 0; i < n_elems; ++i) dst[i] = f32_to_f16_bits(s[i]);
        return;
    }
    if (dtype == DType::BF16) {
        const auto * s = reinterpret_cast<const uint16_t *>(p);
        for (size_t i = 0; i < n_elems; ++i) dst[i] = bf16_to_f16_bits(s[i]);
        return;
    }
    if (dtype == DType::Q8_0) {
        const size_t nb = n_elems / 32;
        const auto * blk = reinterpret_cast<const BlockQ8_0 *>(p);
        for (size_t b = 0; b < nb; ++b) {
            const float d = f16_bits_to_f32(blk[b].d);
            for (int i = 0; i < 32; ++i) dst[b * 32 + i] = f32_to_f16_bits(d * blk[b].qs[i]);
        }
        return;
    }
    if (dtype == DType::Q5_0) {
        const size_t nb = n_elems / 32;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t * bp = p + b * 22;
            uint16_t dh; std::memcpy(&dh, bp, 2);
            uint32_t qh; std::memcpy(&qh, bp + 2, 4);
            const uint8_t * qs = bp + 6;
            const float d = f16_bits_to_f32(dh);
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                const int x0 = static_cast<int>((qs[j] & 0x0F) | xh_0) - 16;
                const int x1 = static_cast<int>((qs[j] >> 4)   | xh_1) - 16;
                dst[b * 32 + j]      = f32_to_f16_bits(x0 * d);
                dst[b * 32 + j + 16] = f32_to_f16_bits(x1 * d);
            }
        }
        return;
    }
    if (dtype == DType::Q4_K) {
        const size_t nb = n_elems / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t * bp = p + b * 144;
            uint16_t dh, dminh;
            std::memcpy(&dh, bp, 2);
            std::memcpy(&dminh, bp + 2, 2);
            const float d = f16_bits_to_f32(dh);
            const float min_v = f16_bits_to_f32(dminh);
            const uint8_t * sc = bp + 4;
            const uint8_t * q  = bp + 16;
            uint16_t * y = dst + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                get_scale_min_k4_host(is + 0, sc, s1, m1);
                get_scale_min_k4_host(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = min_v * m1;
                const float d2 = d * s2, mm2 = min_v * m2;
                for (int l = 0; l < 32; ++l) y[j + l]      = f32_to_f16_bits(d1 * (q[l] & 0xF) - mm1);
                for (int l = 0; l < 32; ++l) y[j + 32 + l] = f32_to_f16_bits(d2 * (q[l] >> 4)  - mm2);
                q += 32;
                is += 2;
            }
        }
        return;
    }
    if (dtype == DType::Q5_K) {
        const size_t nb = n_elems / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t * bp = p + b * 176;
            uint16_t dh, dminh;
            std::memcpy(&dh, bp, 2);
            std::memcpy(&dminh, bp + 2, 2);
            const float d = f16_bits_to_f32(dh);
            const float min_v = f16_bits_to_f32(dminh);
            const uint8_t * sc = bp + 4;
            const uint8_t * qh = bp + 16;
            const uint8_t * ql = bp + 48;
            uint16_t * y = dst + b * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                get_scale_min_k4_host(is + 0, sc, s1, m1);
                get_scale_min_k4_host(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = min_v * m1;
                const float d2 = d * s2, mm2 = min_v * m2;
                for (int l = 0; l < 32; ++l) {
                    y[j + l]      = f32_to_f16_bits(d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1);
                    y[j + 32 + l] = f32_to_f16_bits(d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2);
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        return;
    }
    if (dtype == DType::Q6_K) {
        const size_t nb = n_elems / 256;
        const auto * blk = reinterpret_cast<const PackedBlockQ6_K *>(p);
        for (size_t b = 0; b < nb; ++b) {
            const float d = f16_bits_to_f32(blk[b].d);
            const uint8_t * ql = blk[b].ql;
            const uint8_t * qh = blk[b].qh;
            const int8_t  * sc = blk[b].scales;
            uint16_t * y = dst + b * 256;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int8_t q1 = static_cast<int8_t>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = static_cast<int8_t>((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = static_cast<int8_t>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l +  0] = f32_to_f16_bits(d * sc[is + 0] * q1);
                    y[l + 32] = f32_to_f16_bits(d * sc[is + 2] * q2);
                    y[l + 64] = f32_to_f16_bits(d * sc[is + 4] * q3);
                    y[l + 96] = f32_to_f16_bits(d * sc[is + 6] * q4);
                }
                y  += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    }
    throw std::runtime_error(std::string("WeightPool::upload_as_f16 unsupported dtype: ") + dtype_traits(dtype).name);
}

} // namespace

DeviceTensor WeightPool::upload_as_f16(const ShardedGguf & gguf,
                                       const std::string & name,
                                       int dev,
                                       bool required) {
    auto loc = gguf.find_tensor(name);
    if (!loc.info) {
        if (required) throw std::runtime_error("Required tensor missing from GGUF: " + name);
        return {};
    }
    const auto & info = *loc.info;
    const size_t n_elems = static_cast<size_t>(info.ne[0]) * info.ne[1] * info.ne[2] * info.ne[3];
    std::vector<uint16_t> h_f16(n_elems);
    dequant_to_f16_host(info.dtype, loc.data(), n_elems, h_f16.data());

    const size_t dst_bytes = n_elems * sizeof(uint16_t);
    cuda_ok(cudaSetDevice(dev), "upload_as_f16 setDevice");
    void * dptr = nullptr;
    cuda_ok(cudaMalloc(&dptr, dst_bytes), ("cudaMalloc F16 for " + name).c_str());
    cuda_ok(cudaMemcpy(dptr, h_f16.data(), dst_bytes, cudaMemcpyHostToDevice), ("cudaMemcpy F16 for " + name).c_str());

    {
        std::lock_guard<std::mutex> lk(mu_);
        allocs_.push_back({dev, dptr, dst_bytes});
        dev_bytes_[dev] += dst_bytes;
    }

    DeviceTensor dt;
    dt.ptr    = dptr;
    dt.dtype  = DType::F16;
    dt.device = dev;
    for (int i = 0; i < 4; ++i) dt.ne[i] = info.ne[i];
    dt.nbytes = dst_bytes;
    return dt;
}

} // namespace rookery
