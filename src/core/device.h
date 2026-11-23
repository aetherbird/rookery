#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct GpuInfo {
    int         index            = -1;
    std::string name;
    int         sm_major         = 0;
    int         sm_minor         = 0;
    size_t      vram_total       = 0;
    size_t      vram_free        = 0;
    int         pcie_bus_id      = 0;
    bool        has_tensor_cores = false; // sm_70+
    bool        has_fast_fp16    = false; // sm_60, sm_62, sm_70+
    bool        has_bf16         = false; // sm_80+
    bool        native_bf16      = false; // alias for has_bf16 (sm_80+)
    bool        has_dp4a         = false; // sm_61+
    int         tier             = 2;     // 1 = high-BW (>=700 GB/s), 2 = secondary

    int sm_version() const { return sm_major * 10 + sm_minor; }
};

struct Topology {
    std::vector<GpuInfo>            gpus;
    std::vector<std::vector<bool>>  p2p_access;
};

Topology discover_topology();
void     print_topology(const Topology & topo);

// Cached per-device capability queries (pass dev < 0 to query active CUDA device).
int  device_sm_version(int dev = -1);
bool device_has_tensor_cores(int dev = -1);
bool device_has_fast_fp16(int dev = -1);
bool device_has_dp4a(int dev = -1);
bool device_has_bf16(int dev = -1);

} // namespace rookery
