#include "gguf/sharded.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace rookery {

namespace {

std::vector<std::string> discover_gguf_shards(const std::string & path) {
    namespace fs = std::filesystem;
    const std::string fname = fs::path(path).filename().string();
    const size_t of_pos = fname.rfind("-of-");
    if (of_pos == std::string::npos || of_pos < 6 || fname[of_pos - 6] != '-') {
        return {path};
    }
    const std::string prefix = fname.substr(0, of_pos - 6);
    const std::string suffix = fname.substr(of_pos);
    const fs::path dir = fs::path(path).parent_path();

    std::vector<std::string> out;
    for (const auto & ent : fs::directory_iterator(dir.empty() ? fs::path(".") : dir)) {
        if (!ent.is_regular_file()) continue;
        const std::string n = ent.path().filename().string();
        if (n.size() == fname.size() &&
            n.compare(0, prefix.size(), prefix) == 0 &&
            n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            out.push_back(ent.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) out.push_back(path);
    return out;
}

} // namespace

ShardedGguf ShardedGguf::open(const std::string & path) {
    ShardedGguf sg;
    const auto paths = discover_gguf_shards(path);
    sg.shards_.reserve(paths.size());
    for (size_t si = 0; si < paths.size(); ++si) {
        sg.shards_.push_back(GgufFile::open(paths[si]));
        for (const auto & ti : sg.shards_.back().tensors()) {
            sg.index_[ti.name] = si;
        }
    }
    return sg;
}

ShardedGguf::TensorLoc ShardedGguf::find_tensor(const std::string & name) const {
    auto it = index_.find(name);
    if (it == index_.end()) return {};
    const auto & sh = shards_[it->second];
    return {&sh, sh.find_tensor(name)};
}

size_t ShardedGguf::total_bytes() const {
    size_t sum = 0;
    for (const auto & sh : shards_) sum += sh.file_size();
    return sum;
}

} // namespace rookery
