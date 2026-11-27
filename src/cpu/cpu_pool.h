#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#endif
#endif

namespace rookery::cpu {

struct CpuCaps {
    bool has_avx2       = false;
    bool has_fma        = false;
    bool has_f16c       = false;
    bool has_avx512f    = false;
    bool has_avx512vnni = false;
    bool has_neon       = false;
    bool has_dotprod    = false;
    bool has_fp16       = false;
    bool has_i8mm       = false;
    int  logical_cores  = 1;
    int  physical_cores = 1;
    int  big_cores      = 1;
    std::vector<int> fast_core_ids; // sorted fastest-first via sysfs cpu_capacity / max_freq
};

inline const CpuCaps & detect_cpu_caps() {
    static const CpuCaps caps = []() {
        CpuCaps c;
        unsigned hc = std::thread::hardware_concurrency();
        c.logical_cores = (hc > 0) ? static_cast<int>(hc) : 1;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        c.has_avx2       = __builtin_cpu_supports("avx2") != 0;
        c.has_fma        = __builtin_cpu_supports("fma") != 0;
        c.has_f16c       = __builtin_cpu_supports("f16c") != 0;
        c.has_avx512f    = __builtin_cpu_supports("avx512f") != 0;
        c.has_avx512vnni = __builtin_cpu_supports("avx512vnni") != 0;
#elif defined(__aarch64__)
#if defined(__ARM_NEON)
        c.has_neon = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
        c.has_dotprod = true;
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        c.has_fp16 = true;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
        c.has_i8mm = true;
#endif
#if defined(__linux__)
        const unsigned long hwcap = getauxval(AT_HWCAP);
        // Linux ARM64 HWCAP bit positions (stable kernel ABI):
        // bit 1 = ASIMD (NEON), bit 10 = ASIMDHP (FP16 vec), bit 20 = ASIMDDP (DOTPROD)
        if (hwcap & (1UL << 1))  c.has_neon    = true;
        if (hwcap & (1UL << 10)) c.has_fp16    = true;
        if (hwcap & (1UL << 20)) c.has_dotprod = true;
#ifdef AT_HWCAP2
        const unsigned long hwcap2 = getauxval(AT_HWCAP2);
        // bit 13 = I8MM
        if (hwcap2 & (1UL << 13)) c.has_i8mm = true;
#endif
#endif
#endif
        // Estimate physical cores on Linux via /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
        int smt = 1;
#if defined(__linux__)
        if (FILE * f = std::fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
            char buf[64] = {};
            if (std::fgets(buf, sizeof(buf), f)) {
                int commas = 0;
                for (char * p = buf; *p; ++p) {
                    if (*p == ',') ++commas;
                    else if (*p == '-') {
                        int a = 0, b = 0;
                        if (std::sscanf(buf, "%d-%d", &a, &b) == 2 && b >= a) {
                            smt = b - a + 1;
                            commas = -1;
                            break;
                        }
                    }
                }
                if (commas >= 0) smt = commas + 1;
            }
            std::fclose(f);
        }
#endif
        c.physical_cores = std::max(1, c.logical_cores / std::max(1, smt));
        c.big_cores      = c.physical_cores;

#if defined(__linux__)
        struct CoreRank {
            int      id;
            uint64_t score;
        };
        std::vector<CoreRank> ranks;
        ranks.reserve(static_cast<size_t>(c.logical_cores));
        uint64_t max_score = 0;
        for (int cpu = 0; cpu < c.logical_cores; ++cpu) {
            uint64_t cap  = 0;
            uint64_t freq = 0;
            char path[128];
            std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
            if (FILE * f = std::fopen(path, "r")) {
                unsigned long long v = 0;
                if (std::fscanf(f, "%llu", &v) == 1) cap = static_cast<uint64_t>(v);
                std::fclose(f);
            }
            std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            if (FILE * f = std::fopen(path, "r")) {
                unsigned long long v = 0;
                if (std::fscanf(f, "%llu", &v) == 1) freq = static_cast<uint64_t>(v);
                std::fclose(f);
            }
            const uint64_t score = (cap > 0) ? (cap * 1000000ULL + freq) : (freq > 0 ? freq : 1000ULL);
            ranks.push_back({cpu, score});
            if (score > max_score) max_score = score;
        }
        std::stable_sort(ranks.begin(), ranks.end(), [](const CoreRank & a, const CoreRank & b) {
            return a.score > b.score;
        });
        c.fast_core_ids.reserve(ranks.size());
        int n_big = 0;
        for (const auto & r : ranks) {
            c.fast_core_ids.push_back(r.id);
            if (max_score > 0 && r.score * 100 >= max_score * 85) ++n_big;
        }
        if (n_big > 0) c.big_cores = std::min(c.physical_cores, n_big);
#endif
        return c;
    }();
    return caps;
}

inline void print_cpu_caps(int n_threads) {
    const auto & c = detect_cpu_caps();
    if (c.big_cores > 0 && c.big_cores < c.physical_cores) {
        std::printf("[CPU] cores=%d big / %d physical (%d logical), threads=%d | ISA:%s%s%s%s%s%s%s%s%s\n",
                    c.big_cores, c.physical_cores, c.logical_cores, n_threads,
                    c.has_avx2 ? " AVX2" : "",
                    c.has_fma ? " FMA" : "",
                    c.has_f16c ? " F16C" : "",
                    c.has_avx512f ? " AVX512F" : "",
                    c.has_avx512vnni ? " AVX512VNNI" : "",
                    c.has_neon ? " NEON" : "",
                    c.has_dotprod ? " DOTPROD" : "",
                    c.has_fp16 ? " FP16" : "",
                    c.has_i8mm ? " I8MM" : "");
    } else {
        std::printf("[CPU] cores=%d physical (%d logical), threads=%d | ISA:%s%s%s%s%s%s%s%s%s\n",
                    c.physical_cores, c.logical_cores, n_threads,
                    c.has_avx2 ? " AVX2" : "",
                    c.has_fma ? " FMA" : "",
                    c.has_f16c ? " F16C" : "",
                    c.has_avx512f ? " AVX512F" : "",
                    c.has_avx512vnni ? " AVX512VNNI" : "",
                    c.has_neon ? " NEON" : "",
                    c.has_dotprod ? " DOTPROD" : "",
                    c.has_fp16 ? " FP16" : "",
                    c.has_i8mm ? " I8MM" : "");
    }
}

// Low-latency hybrid spin-then-park thread pool for per-layer CPU inference.
// Thread 0 executes inline on the caller thread; threads 1..N-1 are persistent workers.
// Spins briefly between closely-spaced dispatches (<0.5 us barrier) and sleeps on condvar when idle.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        init(n_threads);
    }

    ~ThreadPool() {
        shutdown();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool & operator=(const ThreadPool &) = delete;

    void init(int n_threads) {
        shutdown();
        const auto & caps = detect_cpu_caps();
        const int req_threads = n_threads;
        if (n_threads <= 0) {
            n_threads = (caps.big_cores > 0 && caps.big_cores < caps.physical_cores)
                            ? caps.big_cores
                            : caps.physical_cores;
        }
        n_threads_ = std::max(1, n_threads);
        n_compute_threads_ = n_threads_;
        active_nth_ = n_threads_;
        stop_.store(false, std::memory_order_relaxed);
        gen_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        parked_.store(0, std::memory_order_relaxed);
        aux_gen_.store(0, std::memory_order_relaxed);
        aux_done_.store(0, std::memory_order_relaxed);
        aux_parked_.store(0, std::memory_order_relaxed);

#if defined(__aarch64__) && defined(__linux__)
        use_affinity_ = (std::getenv("ROOKERY_NO_AFFINITY") == nullptr);
        core_ids_     = caps.fast_core_ids;
        if (use_affinity_ && !core_ids_.empty()) {
            pin_caller_if_needed(core_ids_[0]);
        }
        // On heterogeneous big.LITTLE ARM64 SoCs where LITTLE cores share dotprod (e.g. RK3588
        // 4x A76 + 4x A55), keep decode strictly on big cores (peak DRAM bandwidth) while spawning
        // auxiliary LITTLE workers woken ONLY for compute-bound dynamic prefill (T >= 16).
        const bool want_hybrid = (caps.big_cores > 0 && caps.big_cores < caps.physical_cores &&
                                  caps.has_dotprod &&
                                  (req_threads <= 0 || req_threads == caps.big_cores) &&
                                  std::getenv("ROOKERY_NO_HYBRID_PREFILL") == nullptr);
        if (want_hybrid) {
            n_compute_threads_ = caps.physical_cores;
        }
#else
        (void) req_threads;
#endif

        const int n_workers = n_threads_ - 1;
        workers_.reserve(static_cast<size_t>(n_workers));
        for (int i = 1; i <= n_workers; ++i) {
#if defined(__aarch64__) && defined(__linux__)
            const int core_id = (use_affinity_ && i < static_cast<int>(core_ids_.size())) ? core_ids_[i] : -1;
            workers_.emplace_back([this, i, core_id]() {
                if (core_id >= 0) pin_this_thread(core_id);
                worker_loop(i);
            });
#else
            workers_.emplace_back([this, i]() { worker_loop(i); });
#endif
        }

        const int n_aux = n_compute_threads_ - n_threads_;
        if (n_aux > 0) {
            aux_workers_.reserve(static_cast<size_t>(n_aux));
            for (int i = n_threads_; i < n_compute_threads_; ++i) {
#if defined(__aarch64__) && defined(__linux__)
                const int core_id = (use_affinity_ && i < static_cast<int>(core_ids_.size())) ? core_ids_[i] : -1;
                aux_workers_.emplace_back([this, i, core_id]() {
                    if (core_id >= 0) pin_this_thread(core_id);
                    aux_worker_loop(i);
                });
#else
                aux_workers_.emplace_back([this, i]() { aux_worker_loop(i); });
#endif
            }
        }
    }

    void shutdown() {
        if (!workers_.empty() || !aux_workers_.empty()) {
            stop_.store(true, std::memory_order_release);
            if (!workers_.empty()) {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    gen_.fetch_add(1, std::memory_order_seq_cst);
                }
                cv_.notify_all();
                for (auto & th : workers_) {
                    if (th.joinable()) th.join();
                }
                workers_.clear();
            }
            if (!aux_workers_.empty()) {
                {
                    std::lock_guard<std::mutex> lk(aux_mu_);
                    aux_gen_.fetch_add(1, std::memory_order_seq_cst);
                }
                aux_cv_.notify_all();
                for (auto & th : aux_workers_) {
                    if (th.joinable()) th.join();
                }
                aux_workers_.clear();
            }
        }
        n_threads_ = 1;
        n_compute_threads_ = 1;
        active_nth_ = 1;
    }

    int num_threads() const { return n_threads_; }
    int num_compute_threads() const { return n_compute_threads_; }

    // Dispatch fn(int tid, int n_threads) across all n_threads_ (big) threads.
    template <typename Fn>
    void parallel(Fn && fn) {
        if (n_threads_ <= 1) {
            fn(0, 1);
            return;
        }
        using FnT = std::remove_reference_t<Fn>;
        struct Wrapper {
            FnT * f;
            static void call(void * ctx, int tid, int nth) {
                (*static_cast<Wrapper *>(ctx)->f)(tid, nth);
            }
        } w{&fn};
        dispatch_raw(&Wrapper::call, &w);
    }

    // Dispatch fn(int tid, int n_compute_threads) across all compute threads (big + LITTLE on
    // heterogeneous big.LITTLE CPUs). Intended ONLY for compute-bound dynamic work-stealing loops
    // (prefill T >= 16); auxiliary LITTLE threads remain parked during normal parallel() decode calls.
    template <typename Fn>
    void parallel_compute(Fn && fn) {
        if (n_compute_threads_ <= n_threads_) {
            parallel(std::forward<Fn>(fn));
            return;
        }
        using FnT = std::remove_reference_t<Fn>;
        struct Wrapper {
            FnT * f;
            static void call(void * ctx, int tid, int nth) {
                (*static_cast<Wrapper *>(ctx)->f)(tid, nth);
            }
        } w{&fn};
        dispatch_compute_raw(&Wrapper::call, &w);
    }

    // Dispatch fn(int begin, int end, int tid) partitioning [0, total) across threads.
    template <typename Fn>
    void parallel_for(int total, Fn && fn) {
        if (total <= 0) return;
        if (n_threads_ <= 1 || total == 1) {
            fn(0, total, 0);
            return;
        }
        const int active = std::min(n_threads_, total);
        if (active == 1) {
            fn(0, total, 0);
            return;
        }
        parallel([&](int tid, int nth) {
            const int base = total / nth;
            const int rem  = total % nth;
            const int b    = tid * base + std::min(tid, rem);
            const int e    = b + base + (tid < rem ? 1 : 0);
            if (b < e) fn(b, e, tid);
        });
    }

    // Partition [0, total) in chunks of `granularity` (e.g., 4 rows for 4-row-unrolled GEMV/GEMM).
    template <typename Fn>
    void parallel_for_blocked(int total, int granularity, Fn && fn) {
        if (total <= 0) return;
        if (n_threads_ <= 1 || total <= granularity) {
            fn(0, total, 0);
            return;
        }
        const int n_blocks = (total + granularity - 1) / granularity;
        parallel([&](int tid, int nth) {
            const int base = n_blocks / nth;
            const int rem  = n_blocks % nth;
            const int b0   = tid * base + std::min(tid, rem);
            const int b1   = b0 + base + (tid < rem ? 1 : 0);
            const int i0   = b0 * granularity;
            const int i1   = std::min(total, b1 * granularity);
            if (i0 < i1) fn(i0, i1, tid);
        });
    }

private:
    using TaskFn = void (*)(void *, int, int);

    static inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

    static inline void pin_this_thread(int core_id) noexcept {
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core_id, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
        (void) core_id;
#endif
    }

    static inline void pin_caller_if_needed(int core_id) noexcept {
        thread_local int pinned_core = -1;
        if (pinned_core != core_id) {
            pin_this_thread(core_id);
            pinned_core = core_id;
        }
    }

    void dispatch_raw(TaskFn fn, void * ctx) {
#if defined(__aarch64__) && defined(__linux__)
        if (use_affinity_ && !core_ids_.empty()) {
            pin_caller_if_needed(core_ids_[0]);
        }
#endif
        fn_         = fn;
        ctx_        = ctx;
        active_nth_ = n_threads_;
        const int n_workers = n_threads_ - 1;
        done_.store(0, std::memory_order_relaxed);
        // Full StoreLoad barrier (seq_cst) prevents ARM64 lost-wakeup between gen_ write and parked_ read
        gen_.fetch_add(1, std::memory_order_seq_cst);

        if (parked_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }

        // Run shard 0 on caller thread
        fn(ctx, 0, n_threads_);

        // Wait for all worker threads
        while (done_.load(std::memory_order_acquire) < n_workers) {
            cpu_relax();
        }
    }

    void dispatch_compute_raw(TaskFn fn, void * ctx) {
#if defined(__aarch64__) && defined(__linux__)
        if (use_affinity_ && !core_ids_.empty()) {
            pin_caller_if_needed(core_ids_[0]);
        }
#endif
        fn_         = fn;
        ctx_        = ctx;
        active_nth_ = n_compute_threads_;
        const int n_workers = n_threads_ - 1;
        const int n_aux     = n_compute_threads_ - n_threads_;
        done_.store(0, std::memory_order_relaxed);
        aux_done_.store(0, std::memory_order_relaxed);

        if (n_workers > 0) {
            gen_.fetch_add(1, std::memory_order_seq_cst);
            if (parked_.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(mu_);
                cv_.notify_all();
            }
        }
        if (n_aux > 0) {
            aux_gen_.fetch_add(1, std::memory_order_seq_cst);
            if (aux_parked_.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(aux_mu_);
                aux_cv_.notify_all();
            }
        }

        fn(ctx, 0, n_compute_threads_);

        while (done_.load(std::memory_order_acquire) < n_workers ||
               aux_done_.load(std::memory_order_acquire) < n_aux) {
            cpu_relax();
        }
    }

    void worker_loop(int tid) {
        uint64_t my_gen = 0;
        constexpr int kSpinIters = 120000; // ~0.4-1 ms spin before parking on condvar
        while (true) {
            uint64_t g = gen_.load(std::memory_order_acquire);
            if (g == my_gen) {
                int spin = 0;
                while ((g = gen_.load(std::memory_order_acquire)) == my_gen && spin < kSpinIters) {
                    cpu_relax();
                    ++spin;
                }
                if (g == my_gen) {
                    std::unique_lock<std::mutex> lk(mu_);
                    parked_.fetch_add(1, std::memory_order_seq_cst);
                    cv_.wait(lk, [&]() {
                        return gen_.load(std::memory_order_seq_cst) != my_gen ||
                               stop_.load(std::memory_order_acquire);
                    });
                    parked_.fetch_sub(1, std::memory_order_seq_cst);
                    g = gen_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_acquire)) return;
            my_gen = g;
            fn_(ctx_, tid, active_nth_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    void aux_worker_loop(int tid) {
        uint64_t my_gen = 0;
        constexpr int kAuxSpinIters = 15000; // brief spin during consecutive prefill layers, then sleep cleanly for decode
        while (true) {
            uint64_t g = aux_gen_.load(std::memory_order_acquire);
            if (g == my_gen) {
                int spin = 0;
                while ((g = aux_gen_.load(std::memory_order_acquire)) == my_gen && spin < kAuxSpinIters) {
                    cpu_relax();
                    ++spin;
                }
                if (g == my_gen) {
                    std::unique_lock<std::mutex> lk(aux_mu_);
                    aux_parked_.fetch_add(1, std::memory_order_seq_cst);
                    aux_cv_.wait(lk, [&]() {
                        return aux_gen_.load(std::memory_order_seq_cst) != my_gen ||
                               stop_.load(std::memory_order_acquire);
                    });
                    aux_parked_.fetch_sub(1, std::memory_order_seq_cst);
                    g = aux_gen_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_acquire)) return;
            my_gen = g;
            fn_(ctx_, tid, active_nth_);
            aux_done_.fetch_add(1, std::memory_order_release);
        }
    }

    int                      n_threads_         = 1;
    int                      n_compute_threads_ = 1;
    int                      active_nth_        = 1;
#if defined(__aarch64__) && defined(__linux__)
    bool                     use_affinity_ = true;
    std::vector<int>         core_ids_;
#endif
    std::vector<std::thread> workers_;
    std::vector<std::thread> aux_workers_;
    TaskFn                   fn_        = nullptr;
    void *                   ctx_       = nullptr;

    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int>      done_{0};
    alignas(64) std::atomic<int>      parked_{0};
    alignas(64) std::atomic<uint64_t> aux_gen_{0};
    alignas(64) std::atomic<int>      aux_done_{0};
    alignas(64) std::atomic<int>      aux_parked_{0};
    alignas(64) std::atomic<bool>     stop_{false};

    std::mutex              mu_;
    std::condition_variable cv_;
    std::mutex              aux_mu_;
    std::condition_variable aux_cv_;
};

} // namespace rookery::cpu
