#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#endif

namespace rookery::cpu {

struct CpuCaps {
    bool has_avx2       = false;
    bool has_fma        = false;
    bool has_f16c       = false;
    bool has_avx512f    = false;
    bool has_avx512vnni = false;
    int  logical_cores  = 1;
    int  physical_cores = 1;
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
        return c;
    }();
    return caps;
}

inline void print_cpu_caps(int n_threads) {
    const auto & c = detect_cpu_caps();
    std::printf("[CPU] cores=%d physical (%d logical), threads=%d | ISA:%s%s%s%s%s\n",
                c.physical_cores, c.logical_cores, n_threads,
                c.has_avx2 ? " AVX2" : "",
                c.has_fma ? " FMA" : "",
                c.has_f16c ? " F16C" : "",
                c.has_avx512f ? " AVX512F" : "",
                c.has_avx512vnni ? " AVX512VNNI" : "");
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
        if (n_threads <= 0) {
            n_threads = detect_cpu_caps().physical_cores;
        }
        n_threads_ = std::max(1, n_threads);
        stop_.store(false, std::memory_order_relaxed);
        gen_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);

        const int n_workers = n_threads_ - 1;
        workers_.reserve(static_cast<size_t>(n_workers));
        for (int i = 1; i <= n_workers; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    void shutdown() {
        if (!workers_.empty()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_.store(true, std::memory_order_release);
                gen_.fetch_add(1, std::memory_order_release);
            }
            cv_.notify_all();
            for (auto & th : workers_) {
                if (th.joinable()) th.join();
            }
            workers_.clear();
        }
        n_threads_ = 1;
    }

    int num_threads() const { return n_threads_; }

    // Dispatch fn(int tid, int n_threads) across all n_threads_ threads.
    template <typename Fn>
    void parallel(Fn && fn) {
        if (n_threads_ <= 1) {
            fn(0, 1);
            return;
        }
        struct Wrapper {
            Fn * f;
            static void call(void * ctx, int tid, int nth) {
                (*static_cast<Wrapper *>(ctx)->f)(tid, nth);
            }
        } w{&fn};
        dispatch_raw(&Wrapper::call, &w);
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

    void dispatch_raw(TaskFn fn, void * ctx) {
        fn_  = fn;
        ctx_ = ctx;
        const int n_workers = n_threads_ - 1;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);

        if (parked_.load(std::memory_order_acquire) > 0) {
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
                    parked_.fetch_add(1, std::memory_order_acq_rel);
                    cv_.wait(lk, [&]() {
                        return gen_.load(std::memory_order_acquire) != my_gen ||
                               stop_.load(std::memory_order_acquire);
                    });
                    parked_.fetch_sub(1, std::memory_order_acq_rel);
                    g = gen_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_acquire)) return;
            my_gen = g;
            fn_(ctx_, tid, n_threads_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    int                      n_threads_ = 1;
    std::vector<std::thread> workers_;
    TaskFn                   fn_        = nullptr;
    void *                   ctx_       = nullptr;

    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int>      done_{0};
    alignas(64) std::atomic<int>      parked_{0};
    alignas(64) std::atomic<bool>     stop_{false};

    std::mutex              mu_;
    std::condition_variable cv_;
};

} // namespace rookery::cpu
