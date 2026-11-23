#include "gguf/reader.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rookery {

namespace {

struct Cursor {
    const uint8_t * ptr;
    const uint8_t * end;

    void require(size_t n) const {
        if (static_cast<size_t>(end - ptr) < n) {
            throw std::runtime_error("GGUF truncated unexpectedly");
        }
    }

    template <typename T>
    T read() {
        require(sizeof(T));
        T v;
        std::memcpy(&v, ptr, sizeof(T));
        ptr += sizeof(T);
        return v;
    }

    std::string read_string() {
        const uint64_t len = read<uint64_t>();
        require(len);
        std::string s(reinterpret_cast<const char *>(ptr), len);
        ptr += len;
        return s;
    }
};

GgufValue read_value(Cursor & c, GgufValType t) {
    GgufValue out;
    out.type = t;
    switch (t) {
        case GgufValType::UINT8:   out.data = static_cast<uint64_t>(c.read<uint8_t>());  break;
        case GgufValType::INT8:    out.data = static_cast<int64_t>(c.read<int8_t>());    break;
        case GgufValType::UINT16:  out.data = static_cast<uint64_t>(c.read<uint16_t>()); break;
        case GgufValType::INT16:   out.data = static_cast<int64_t>(c.read<int16_t>());   break;
        case GgufValType::UINT32:  out.data = static_cast<uint64_t>(c.read<uint32_t>()); break;
        case GgufValType::INT32:   out.data = static_cast<int64_t>(c.read<int32_t>());   break;
        case GgufValType::UINT64:  out.data = c.read<uint64_t>();                        break;
        case GgufValType::INT64:   out.data = c.read<int64_t>();                         break;
        case GgufValType::FLOAT32: out.data = static_cast<double>(c.read<float>());      break;
        case GgufValType::FLOAT64: out.data = c.read<double>();                          break;
        case GgufValType::BOOL:    out.data = (c.read<uint8_t>() != 0);                  break;
        case GgufValType::STRING:  out.data = c.read_string();                           break;
        case GgufValType::ARRAY: {
            const auto elem_type = static_cast<GgufValType>(c.read<uint32_t>());
            const uint64_t len   = c.read<uint64_t>();
            switch (elem_type) {
                case GgufValType::UINT8:
                case GgufValType::UINT16:
                case GgufValType::UINT32:
                case GgufValType::UINT64: {
                    std::vector<uint64_t> arr;
                    arr.reserve(len);
                    for (uint64_t i = 0; i < len; ++i) arr.push_back(read_value(c, elem_type).as_u64());
                    out.data = std::move(arr);
                    break;
                }
                case GgufValType::INT8:
                case GgufValType::INT16:
                case GgufValType::INT32:
                case GgufValType::INT64: {
                    std::vector<int64_t> arr;
                    arr.reserve(len);
                    for (uint64_t i = 0; i < len; ++i) arr.push_back(read_value(c, elem_type).as_i64());
                    out.data = std::move(arr);
                    break;
                }
                case GgufValType::FLOAT32:
                case GgufValType::FLOAT64: {
                    std::vector<double> arr;
                    arr.reserve(len);
                    for (uint64_t i = 0; i < len; ++i) arr.push_back(read_value(c, elem_type).as_f64());
                    out.data = std::move(arr);
                    break;
                }
                case GgufValType::BOOL: {
                    std::vector<bool> arr;
                    arr.reserve(len);
                    for (uint64_t i = 0; i < len; ++i) arr.push_back(c.read<uint8_t>() != 0);
                    out.data = std::move(arr);
                    break;
                }
                case GgufValType::STRING: {
                    std::vector<std::string> arr;
                    arr.reserve(len);
                    for (uint64_t i = 0; i < len; ++i) arr.push_back(c.read_string());
                    out.data = std::move(arr);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported GGUF array element type");
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported GGUF value type");
    }
    return out;
}

} // namespace

uint64_t GgufValue::as_u64() const {
    if (auto * v = std::get_if<uint64_t>(&data)) return *v;
    if (auto * v = std::get_if<int64_t>(&data))  return static_cast<uint64_t>(*v);
    if (auto * v = std::get_if<bool>(&data))     return *v ? 1ULL : 0ULL;
    throw std::runtime_error("GgufValue is not uint64-convertible");
}

int64_t GgufValue::as_i64() const {
    if (auto * v = std::get_if<int64_t>(&data))  return *v;
    if (auto * v = std::get_if<uint64_t>(&data)) return static_cast<int64_t>(*v);
    if (auto * v = std::get_if<bool>(&data))     return *v ? 1LL : 0LL;
    throw std::runtime_error("GgufValue is not int64-convertible");
}

double GgufValue::as_f64() const {
    if (auto * v = std::get_if<double>(&data))   return *v;
    if (auto * v = std::get_if<uint64_t>(&data)) return static_cast<double>(*v);
    if (auto * v = std::get_if<int64_t>(&data))  return static_cast<double>(*v);
    throw std::runtime_error("GgufValue is not float-convertible");
}

bool GgufValue::as_bool() const {
    if (auto * v = std::get_if<bool>(&data))     return *v;
    if (auto * v = std::get_if<uint64_t>(&data)) return *v != 0;
    if (auto * v = std::get_if<int64_t>(&data))  return *v != 0;
    throw std::runtime_error("GgufValue is not bool-convertible");
}

std::string GgufValue::as_str() const {
    if (auto * v = std::get_if<std::string>(&data)) return *v;
    throw std::runtime_error("GgufValue is not string");
}

GgufFile::~GgufFile() {
    close_mmap();
}

void GgufFile::close_mmap() noexcept {
    if (base_ != nullptr && base_ != MAP_FAILED) {
        munmap(base_, file_size_);
    }
    base_ = nullptr;
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

GgufFile::GgufFile(GgufFile && other) noexcept
    : fd_(other.fd_),
      base_(other.base_),
      file_size_(other.file_size_),
      version_(other.version_),
      data_offset_(other.data_offset_),
      alignment_(other.alignment_),
      kv_(std::move(other.kv_)),
      tensors_(std::move(other.tensors_)),
      tensor_idx_(std::move(other.tensor_idx_)) {
    other.fd_   = -1;
    other.base_ = nullptr;
    other.file_size_ = 0;
}

GgufFile & GgufFile::operator=(GgufFile && other) noexcept {
    if (this != &other) {
        close_mmap();
        fd_          = other.fd_;
        base_        = other.base_;
        file_size_   = other.file_size_;
        version_     = other.version_;
        data_offset_ = other.data_offset_;
        alignment_   = other.alignment_;
        kv_          = std::move(other.kv_);
        tensors_     = std::move(other.tensors_);
        tensor_idx_  = std::move(other.tensor_idx_);
        other.fd_        = -1;
        other.base_      = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

GgufFile GgufFile::open(const std::string & path) {
    GgufFile f;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) {
        throw std::runtime_error("Failed to open GGUF: " + path);
    }

    struct stat st{};
    if (fstat(f.fd_, &st) != 0 || st.st_size < 24) {
        throw std::runtime_error("Failed to stat GGUF or file too small: " + path);
    }
    f.file_size_ = static_cast<size_t>(st.st_size);

    void * mem = mmap(nullptr, f.file_size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (mem == MAP_FAILED) {
        throw std::runtime_error("mmap failed for GGUF: " + path);
    }
    f.base_ = static_cast<uint8_t *>(mem);

    Cursor c{f.base_, f.base_ + f.file_size_};
    if (std::memcmp(c.ptr, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF magic in: " + path);
    }
    c.ptr += 4;

    f.version_ = c.read<uint32_t>();
    if (f.version_ < 2 || f.version_ > 3) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(f.version_));
    }

    const uint64_t n_tensors = c.read<uint64_t>();
    const uint64_t n_kv      = c.read<uint64_t>();

    f.kv_.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.read_string();
        auto vtype      = static_cast<GgufValType>(c.read<uint32_t>());
        f.kv_.emplace(std::move(key), read_value(c, vtype));
    }

    f.alignment_ = f.get_u64("general.alignment", 32);

    f.tensors_.reserve(n_tensors);
    f.tensor_idx_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;
        info.name   = c.read_string();
        info.n_dims = c.read<uint32_t>();
        if (info.n_dims > 4) {
            throw std::runtime_error("Tensor n_dims > 4: " + info.name);
        }
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            info.ne[d] = c.read<uint64_t>();
        }
        info.dtype  = static_cast<DType>(c.read<uint32_t>());
        info.offset = c.read<uint64_t>();
        info.nbytes = tensor_nbytes(info.dtype, info.ne[0], info.ne[1], info.ne[2], info.ne[3]);

        f.tensor_idx_.emplace(info.name, f.tensors_.size());
        f.tensors_.push_back(std::move(info));
    }

    const size_t pos = static_cast<size_t>(c.ptr - f.base_);
    const size_t rem = pos % f.alignment_;
    f.data_offset_   = rem == 0 ? pos : pos + (f.alignment_ - rem);

    return f;
}

const GgufValue & GgufFile::kv(const std::string & key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) {
        throw std::runtime_error("Missing GGUF key: " + key);
    }
    return it->second;
}

uint64_t GgufFile::get_u64(const std::string & key, uint64_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.as_u64();
}

int64_t GgufFile::get_i64(const std::string & key, int64_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.as_i64();
}

double GgufFile::get_f64(const std::string & key, double def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.as_f64();
}

bool GgufFile::get_bool(const std::string & key, bool def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.as_bool();
}

std::string GgufFile::get_str(const std::string & key, const std::string & def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.as_str();
}

const GgufTensorInfo * GgufFile::find_tensor(const std::string & name) const {
    auto it = tensor_idx_.find(name);
    if (it == tensor_idx_.end()) return nullptr;
    return &tensors_[it->second];
}

const void * GgufFile::tensor_data(const GgufTensorInfo & info) const {
    const size_t abs_off = data_offset_ + info.offset;
    if (abs_off + info.nbytes > file_size_) {
        throw std::runtime_error("Tensor data out of file bounds: " + info.name);
    }
    return base_ + abs_off;
}

} // namespace rookery
