#include "core/device.h"
#include "gguf/reader.h"

#include <chrono>
#include <cstdio>
#include <map>

int main(int argc, char ** argv) {
    auto topo = rookery::discover_topology();
    rookery::print_topology(topo);

    if (argc < 2) {
        std::printf("\nUsage: %s <model.gguf>\n", argv[0]);
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto gguf = rookery::GgufFile::open(argv[1]);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const std::string arch = gguf.get_str("general.architecture", "unknown");
    const std::string name = gguf.get_str("general.name", "unknown");
    std::printf("\nGGUF v%u opened in %.2f ms: %s (%s)\n", gguf.version(), ms, name.c_str(), arch.c_str());
    std::printf("  File size:   %.2f GiB\n", static_cast<double>(gguf.file_size()) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  Data offset: %zu bytes\n", gguf.data_offset());
    std::printf("  KV entries:  %zu\n", gguf.all_kv().size());
    std::printf("  Tensors:     %zu\n", gguf.tensors().size());

    std::map<uint32_t, std::pair<size_t, size_t>> by_dtype; // count, bytes
    size_t total_tensor_bytes = 0;
    for (const auto & t : gguf.tensors()) {
        auto & e = by_dtype[static_cast<uint32_t>(t.dtype)];
        e.first  += 1;
        e.second += t.nbytes;
        total_tensor_bytes += t.nbytes;
        // Touch bounds check
        (void) gguf.tensor_data(t);
    }

    std::printf("  Tensor data: %.2f GiB\n", static_cast<double>(total_tensor_bytes) / (1024.0 * 1024.0 * 1024.0));
    for (const auto & [id, cb] : by_dtype) {
        auto tr = rookery::dtype_traits(static_cast<rookery::DType>(id));
        std::printf("    %-8s : %4zu tensors, %8.2f MiB\n",
                    tr.name, cb.first, static_cast<double>(cb.second) / (1024.0 * 1024.0));
    }

    if (arch == "maxwell-moe") {
        std::printf("\nMaxwell-MoE verified params:\n");
        std::printf("  blocks=%llu (dense_lead=%llu, nextn=%llu)  embd=%llu  vocab=%llu\n",
                    (unsigned long long) gguf.get_u64("maxwell-moe.block_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.leading_dense_block_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.nextn_predict_layers"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.embedding_length"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.vocab_size"));
        std::printf("  heads=%llu/%llu  experts=%llu (used=%llu, shared=%llu)  hc=%llu\n",
                    (unsigned long long) gguf.get_u64("maxwell-moe.attention.head_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.attention.head_count_kv"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.expert_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.expert_used_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.expert_shared_count"),
                    (unsigned long long) gguf.get_u64("maxwell-moe.hyper_connection.count"));
    }

    return 0;
}
