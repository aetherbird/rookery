#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct GpuInfo {
    int         index       = -1;
    std::string name;
    int         sm_major    = 0;
    int         sm_minor    = 0;
    size_t      vram_total  = 0;
    size_t      vram_free   = 0;
    int         pcie_bus_id = 0;
    bool        native_bf16 = false; // sm_80+
    bool        has_dp4a    = false; // sm_61+
    int         tier        = 2;     // 1 = high-BW (>=700 GB/s), 2 = secondary
};

struct Topology {
    std::vector<GpuInfo>            gpus;
    std::vector<std::vector<bool>>  p2p_access;
};

Topology discover_topology();
void     print_topology(const Topology & topo);

} // namespace rookery
