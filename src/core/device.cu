#include "core/device.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace rookery {

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

        GpuInfo g;
        g.index       = i;
        g.name        = prop.name;
        g.sm_major    = prop.major;
        g.sm_minor    = prop.minor;
        g.vram_total  = total_b;
        g.vram_free   = free_b;
        g.pcie_bus_id = prop.pciBusID;
        g.native_bf16 = (prop.major >= 8);
        g.has_dp4a    = (prop.major > 6) || (prop.major == 6 && prop.minor >= 1);

        // Estimate memory bandwidth (GB/s) from memoryClockRate (kHz) and bus width (bits)
        // Note: CUDA 13 deprecated memoryClockRate, still present in CUDA 12.x; fallback by VRAM >= 20GB.
        g.tier = (total_b >= (20ULL << 30)) ? 1 : 2;
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
        std::printf("  GPU%d: %-26s sm_%d%d  VRAM: %5.1f / %5.1f GiB free  tier=%d  bf16=%s  dp4a=%s\n",
                    g.index, g.name.c_str(), g.sm_major, g.sm_minor, fg, tg,
                    g.tier, g.native_bf16 ? "native" : "conv->f16", g.has_dp4a ? "yes" : "no");
    }
    std::printf("  Total VRAM: %.1f GiB free / %.1f GiB total\n", free_gb, total_gb);
}

} // namespace rookery
