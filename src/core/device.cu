#include "core/device.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <stdexcept>

namespace rookery {

namespace {

constexpr int kMaxCachedDevices = 64;
std::atomic<int> g_device_sm_cache[kMaxCachedDevices];

int query_device_sm(int dev) {
    if (dev < 0) {
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) {
            return 0;
        }
    }
    if (dev < kMaxCachedDevices) {
        int cached = g_device_sm_cache[dev].load(std::memory_order_relaxed);
        if (cached > 0) return cached;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return 0;
    }
    const int sm = major * 10 + minor;
    if (dev < kMaxCachedDevices && sm > 0) {
        g_device_sm_cache[dev].store(sm, std::memory_order_relaxed);
    }
    return sm;
}

} // namespace

int device_sm_version(int dev) {
    return query_device_sm(dev);
}

bool device_has_tensor_cores(int dev) {
    return query_device_sm(dev) >= 70;
}

bool device_has_fast_fp16(int dev) {
    const int sm = query_device_sm(dev);
    return (sm >= 70) || (sm == 60) || (sm == 62);
}

bool device_has_dp4a(int dev) {
    return query_device_sm(dev) >= 61;
}

bool device_has_bf16(int dev) {
    return query_device_sm(dev) >= 80;
}

static void cuda_check(cudaError_t err, const char * ctx) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
    }
}

Topology discover_topology() {
    Topology topo;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        return topo;
    }

    topo.gpus.reserve(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        cuda_check(cudaSetDevice(i), "cudaSetDevice");

        size_t free_b = 0, total_b = 0;
        cuda_check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");

        const int sm = prop.major * 10 + prop.minor;
        if (i < kMaxCachedDevices && sm > 0) {
            g_device_sm_cache[i].store(sm, std::memory_order_relaxed);
        }

        GpuInfo g;
        g.index            = i;
        g.name             = prop.name;
        g.sm_major         = prop.major;
        g.sm_minor         = prop.minor;
        g.vram_total       = total_b;
        g.vram_free        = free_b;
        g.pcie_bus_id      = prop.pciBusID;
        g.has_tensor_cores = (sm >= 70);
        g.has_fast_fp16    = (sm >= 70) || (sm == 60) || (sm == 62);
        g.has_bf16         = (sm >= 80);
        g.native_bf16      = g.has_bf16;
        g.has_dp4a         = (sm >= 61);

        // Estimate memory bandwidth tier: HBM2 GPUs (sm_60 P100, sm_70 V100) or >=20 GiB VRAM.
        g.tier = (total_b >= (20ULL << 30) || sm == 60 || sm == 70) ? 1 : 2;
        topo.gpus.push_back(std::move(g));
    }

    topo.p2p_access.assign(count, std::vector<bool>(count, false));
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if (i == j) {
                topo.p2p_access[i][j] = true;
                continue;
            }
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess) {
                topo.p2p_access[i][j] = (can != 0);
            }
        }
    }

    return topo;
}

void print_topology(const Topology & topo) {
    std::printf("CUDA Topology: %zu GPU(s)\n", topo.gpus.size());
    double total_gb = 0.0, free_gb = 0.0;
    for (const auto & g : topo.gpus) {
        double tg = static_cast<double>(g.vram_total) / (1024.0 * 1024.0 * 1024.0);
        double fg = static_cast<double>(g.vram_free)  / (1024.0 * 1024.0 * 1024.0);
        total_gb += tg;
        free_gb  += fg;
        std::printf("  GPU%d: %-26s sm_%d%d  VRAM: %5.1f / %5.1f GiB free  tier=%d  tc=%s  fp16=%s  dp4a=%s  bf16=%s\n",
                    g.index, g.name.c_str(), g.sm_major, g.sm_minor, fg, tg,
                    g.tier,
                    g.has_tensor_cores ? "yes" : "no",
                    g.has_fast_fp16 ? "fast" : "emu",
                    g.has_dp4a ? "yes" : "no",
                    g.has_bf16 ? "native" : "conv->f16");
    }
    std::printf("  Total VRAM: %.1f GiB free / %.1f GiB total\n", free_gb, total_gb);
}

} // namespace rookery
