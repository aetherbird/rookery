#include "core/device.h"
#include "cpu/cpu_pool.h"

#include <cstdio>

namespace rookery {

int  device_sm_version(int)       { return 0; }
bool device_has_tensor_cores(int) { return false; }
bool device_has_fast_fp16(int)    { return false; }
bool device_has_dp4a(int)         { return false; }
bool device_has_bf16(int)         { return false; }

Topology discover_topology() {
    return Topology{};
}

void print_topology(const Topology &) {
    const auto & caps = cpu::detect_cpu_caps();
    std::printf("Execution Mode: CPU-only (%d physical / %d logical cores | ISA:%s%s%s%s%s)\n",
                caps.physical_cores, caps.logical_cores,
                caps.has_avx2 ? " AVX2" : "",
                caps.has_fma ? " FMA" : "",
                caps.has_f16c ? " F16C" : "",
                caps.has_avx512f ? " AVX512F" : "",
                caps.has_avx512vnni ? " AVX512VNNI" : "");
}

} // namespace rookery
