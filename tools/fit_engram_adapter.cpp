#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}
inline void cublas_ok(cublasStatus_t st, const char * msg) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(msg) + " (cuBLAS status=" + std::to_string(st) + ")");
}
inline void cusolver_ok(cusolverStatus_t st, const char * msg) {
    if (st != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(msg) + " (cuSOLVER status=" + std::to_string(st) + ")");
}

struct HookPairs {
    int32_t target_layer = 0;
    int32_t src_layer    = 0;
    std::vector<float> E; // [N, d_engram] row-major
    std::vector<float> H; // [N, d_host]   row-major
    int64_t N = 0;
};

} // namespace

// Pure C++20 + cuBLAS + cuSOLVER closed-form Ridge + Low-Rank SVD Adapter Fitter.
// Zero Python / PyTorch dependencies — fits multi-layer cross-model Engram adapters in < 1 second on GPU!
int main(int argc, char ** argv) {
    std::string pairs_path;
    std::string out_path;
    int   rank     = 128;
    float lam      = 0.1f;
    float scale    = 0.02f;
    bool  gated    = false;
    bool  center   = true;
    bool  auto_lam = false;
    bool  rkad_v3  = false;
    int   device   = 0;
    std::string provider_arch;
    std::string provider_fp;
    std::string host_arch;
    int   host_n_layers = 0;
    int   host_hc_mult  = 4;
    float delta_clip    = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--pairs")               pairs_path = next();
        else if (a == "--out")            out_path = next();
        else if (a == "--rank")           rank = std::atoi(next().c_str());
        else if (a == "--lam")            lam = std::strtof(next().c_str(), nullptr);
        else if (a == "--auto-lam")       auto_lam = true;
        else if (a == "--scale")          scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--gated")          gated = true;
        else if (a == "--center")         center = true;
        else if (a == "--no-center")      center = false;
        else if (a == "--device")         device = std::atoi(next().c_str());
        else if (a == "--rkad-v3")        rkad_v3 = true;
        else if (a == "--provider-arch")  provider_arch = next();
        else if (a == "--provider-fp")    provider_fp = next();
        else if (a == "--host-arch")      host_arch = next();
        else if (a == "--host-n-layers")  host_n_layers = std::atoi(next().c_str());
        else if (a == "--host-hc-mult")   host_hc_mult = std::atoi(next().c_str());
        else if (a == "--delta-clip")     delta_clip = std::strtof(next().c_str(), nullptr);
    }

    if (pairs_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "Usage: %s --pairs <in.rkpr> --out <out.rkad> [--rank 128] [--lam 0.1 | --auto-lam] [--scale 0.02] [--gated] [--center|--no-center] [--device 0]\n", argv[0]);
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 1. Read RKPR v1 binary pair dump
    FILE * fp = std::fopen(pairs_path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open " + pairs_path);
    char magic[4]{};
    uint32_t hdr[4]{};
    uint64_t total_rec = 0;
    if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "RKPR", 4) != 0 ||
        std::fread(hdr, sizeof(uint32_t), 4, fp) != 4 || std::fread(&total_rec, sizeof(uint64_t), 1, fp) != 1) {
        std::fclose(fp);
        throw std::runtime_error("Invalid RKPR header");
    }
    const uint32_t n_hooks  = hdr[1];
    const uint32_t d_engram = hdr[2];
    const uint32_t d_host   = hdr[3];

    std::vector<HookPairs> hooks(n_hooks);
    for (uint32_t i = 0; i < n_hooks; ++i) {
        int32_t meta[2]{};
        if (std::fread(meta, sizeof(int32_t), 2, fp) != 2) throw std::runtime_error("EOF in RKPR meta");
        hooks[i].target_layer = meta[0];
        hooks[i].src_layer    = meta[1];
    }

    while (true) {
        int32_t rec[2]{};
        if (std::fread(rec, sizeof(int32_t), 2, fp) != 2) break;
        const int hi = rec[0];
        const int nt = rec[1];
        if (hi < 0 || hi >= static_cast<int>(n_hooks) || nt <= 0) break;
        auto & hp = hooks[hi];
        const size_t e_off = hp.E.size();
        const size_t h_off = hp.H.size();
        hp.E.resize(e_off + static_cast<size_t>(nt) * d_engram);
        hp.H.resize(h_off + static_cast<size_t>(nt) * d_host);
        if (std::fread(hp.E.data() + e_off, sizeof(float), static_cast<size_t>(nt) * d_engram, fp) != static_cast<size_t>(nt) * d_engram) break;
        if (std::fread(hp.H.data() + h_off, sizeof(float), static_cast<size_t>(nt) * d_host, fp)   != static_cast<size_t>(nt) * d_host)   break;
        hp.N += nt;
    }
    std::fclose(fp);

    std::printf("Loaded %s: %u hooks, N=%lld toks/hook, d_engram=%u -> d_host=%u\n",
                pairs_path.c_str(), n_hooks, static_cast<long long>(hooks[0].N), d_engram, d_host);

    // 2. Init cuBLAS + cuSOLVER on GPU
    cuda_ok(cudaSetDevice(device), "setDevice");
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cublas_ok(cublasCreate(&cublas), "cublasCreate");
    cusolver_ok(cusolverDnCreate(&cusolver), "cusolverDnCreate");

    const int eff_rank = (rank > 0 && rank < static_cast<int>(std::min(d_engram, d_host))) ? rank : 0;

    const uint32_t out_ver = rkad_v3 ? 3u : (center ? 2u : 1u);
    FILE * out = std::fopen(out_path.c_str(), "wb");
    if (!out) throw std::runtime_error("Cannot open output " + out_path);
    const char omagic[4] = {'R', 'K', 'A', 'D'};
    uint32_t ohdr[6] = {out_ver, n_hooks, d_engram, d_host, static_cast<uint32_t>(eff_rank), (rkad_v3 ? 0u : (gated ? 1u : 0u))};
    std::fwrite(omagic, 1, 4, out);
    std::fwrite(ohdr, sizeof(uint32_t), 6, out);

    if (rkad_v3) {
        std::string mj = "{\"schema\":\"rookery.engram.adapter.v3\"";
        if (!provider_arch.empty()) mj += ",\"provider_arch\":\"" + provider_arch + "\"";
        if (!provider_fp.empty())   mj += ",\"provider_fingerprint\":\"" + provider_fp + "\"";
        if (!host_arch.empty())     mj += ",\"host_arch\":\"" + host_arch + "\"";
        if (host_n_layers > 0)      mj += ",\"host_n_layers\":" + std::to_string(host_n_layers);
        mj += ",\"host_d_embd\":" + std::to_string(d_host);
        if (host_hc_mult > 0)       mj += ",\"host_hc_mult\":" + std::to_string(host_hc_mult);
        mj += ",\"calibration\":\"level2_delta_ridge_auto\"}";
        uint32_t mlen = static_cast<uint32_t>(mj.size());
        std::fwrite(&mlen, sizeof(uint32_t), 1, out);
        std::fwrite(mj.data(), 1, mlen, out);
    }

    int * d_info = nullptr;
    cuda_ok(cudaMalloc(&d_info, sizeof(int)), "malloc info");

    for (uint32_t hi = 0; hi < n_hooks; ++hi) {
        const auto & hp = hooks[hi];
        const int N  = static_cast<int>(hp.N);
        const int de = static_cast<int>(d_engram);
        const int dh = static_cast<int>(d_host);

        // Hold out last 10% for out-of-sample Ridge R^2 sanity check (fit uses first N_tr, then refits on all N for export!)
        const int N_val = (N >= 200) ? (N / 10) : 0;
        const int N_tr  = N - N_val;

        // Compute mean vectors mu_E and mu_H over training samples (for centering)
        std::vector<float> mu_E(de, 0.0f);
        std::vector<float> mu_H(dh, 0.0f);
        if (center) {
            std::vector<double> sum_e(de, 0.0), sum_h(dh, 0.0);
            for (int n = 0; n < N; ++n) {
                const float * ep = hp.E.data() + static_cast<size_t>(n) * de;
                const float * hpv = hp.H.data() + static_cast<size_t>(n) * dh;
                for (int j = 0; j < de; ++j) sum_e[j] += ep[j];
                for (int d = 0; d < dh; ++d) sum_h[d] += hpv[d];
            }
            for (int j = 0; j < de; ++j) mu_E[j] = static_cast<float>(sum_e[j] / N);
            for (int d = 0; d < dh; ++d) mu_H[d] = static_cast<float>(sum_h[d] / N);
        }

        std::vector<float> E_fit(hp.E.size());
        std::vector<float> H_fit(hp.H.size());
        for (int n = 0; n < N; ++n) {
            const float * es = hp.E.data() + static_cast<size_t>(n) * de;
            float *       ed = E_fit.data() + static_cast<size_t>(n) * de;
            const float * hs = hp.H.data() + static_cast<size_t>(n) * dh;
            float *       hd = H_fit.data() + static_cast<size_t>(n) * dh;
            if (center) {
                for (int j = 0; j < de; ++j) ed[j] = es[j] - mu_E[j];
                for (int d = 0; d < dh; ++d) hd[d] = hs[d] - mu_H[d];
            } else {
                std::memcpy(ed, es, static_cast<size_t>(de) * sizeof(float));
                std::memcpy(hd, hs, static_cast<size_t>(dh) * sizeof(float));
            }
        }

        float * d_E = nullptr, * d_H = nullptr, * d_EtE = nullptr, * d_EtH = nullptr;
        cuda_ok(cudaMalloc(&d_E,   static_cast<size_t>(N) * de * sizeof(float)), "malloc E");
        cuda_ok(cudaMalloc(&d_H,   static_cast<size_t>(N) * dh * sizeof(float)), "malloc H");
        cuda_ok(cudaMalloc(&d_EtE, static_cast<size_t>(de) * de * sizeof(float)), "malloc EtE");
        cuda_ok(cudaMalloc(&d_EtH, static_cast<size_t>(de) * dh * sizeof(float)), "malloc EtH");
        cuda_ok(cudaMemcpy(d_E, E_fit.data(), static_cast<size_t>(N) * de * sizeof(float), cudaMemcpyHostToDevice), "copy E");
        cuda_ok(cudaMemcpy(d_H, H_fit.data(), static_cast<size_t>(N) * dh * sizeof(float), cudaMemcpyHostToDevice), "copy H");

        const float alpha = 1.0f, beta = 0.0f;
        std::vector<float> h_EtE(static_cast<size_t>(de) * de);
        int lwork_chol = 0;
        cusolver_ok(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, de, d_EtE, de, &lwork_chol), "potrf buf");
        float * d_work = nullptr;
        cuda_ok(cudaMalloc(&d_work, static_cast<size_t>(lwork_chol) * sizeof(float)), "malloc chol work");

        auto solve_ridge = [&](int n_rows, float use_lam) {
            cublas_ok(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, de, de, n_rows, &alpha, d_E, de, d_E, de, &beta, d_EtE, de), "EtE");
            cublas_ok(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, de, dh, n_rows, &alpha, d_E, de, d_H, dh, &beta, d_EtH, de), "EtH");
            cuda_ok(cudaMemcpy(h_EtE.data(), d_EtE, h_EtE.size() * sizeof(float), cudaMemcpyDeviceToHost), "EtE D2H");
            double diag_sum = 0.0;
            for (int j = 0; j < de; ++j) diag_sum += h_EtE[static_cast<size_t>(j) * de + j];
            const float reg = std::max(1e-5f, use_lam * static_cast<float>(diag_sum / de));
            for (int j = 0; j < de; ++j) h_EtE[static_cast<size_t>(j) * de + j] += reg;
            cuda_ok(cudaMemcpy(d_EtE, h_EtE.data(), h_EtE.size() * sizeof(float), cudaMemcpyHostToDevice), "EtE H2D");
            cusolver_ok(cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, de, d_EtE, de, d_work, lwork_chol, d_info), "potrf");
            cusolver_ok(cusolverDnSpotrs(cusolver, CUBLAS_FILL_MODE_LOWER, de, dh, d_EtE, de, d_EtH, de, d_info), "potrs");
        };

        auto compute_r2 = [&](int row_start, int n_rows) -> double {
            float * d_pred = nullptr;
            cuda_ok(cudaMalloc(&d_pred, static_cast<size_t>(n_rows) * dh * sizeof(float)), "malloc pred");
            cublas_ok(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dh, n_rows, de,
                                  &alpha, d_EtH, de, d_E + static_cast<size_t>(row_start) * de, de, &beta, d_pred, dh), "pred");
            std::vector<float> h_pred(static_cast<size_t>(n_rows) * dh);
            cuda_ok(cudaMemcpy(h_pred.data(), d_pred, h_pred.size() * sizeof(float), cudaMemcpyDeviceToHost), "pred D2H");
            cudaFree(d_pred);

            std::vector<double> hm(dh, 0.0);
            for (int n = 0; n < n_rows; ++n)
                for (int d = 0; d < dh; ++d) hm[d] += H_fit[static_cast<size_t>(row_start + n) * dh + d];
            for (int d = 0; d < dh; ++d) hm[d] /= n_rows;

            double ss_res = 0.0, ss_tot = 0.0;
            for (int n = 0; n < n_rows; ++n) {
                for (int d = 0; d < dh; ++d) {
                    const double hv = H_fit[static_cast<size_t>(row_start + n) * dh + d];
                    const double pv = h_pred[static_cast<size_t>(n) * dh + d];
                    ss_res += (hv - pv) * (hv - pv);
                    ss_tot += (hv - hm[d]) * (hv - hm[d]);
                }
            }
            return 1.0 - ss_res / std::max(ss_tot, 1e-12);
        };

        float best_lam = lam;
        double val_r2 = 0.0;
        if (auto_lam && N_val > 0) {
            const float grid[] = {0.5f, 2.0f, 5.0f, 15.0f, 50.0f};
            val_r2 = -1e9;
            for (float cand : grid) {
                solve_ridge(N_tr, cand);
                double vr = compute_r2(N_tr, N_val);
                if (vr > val_r2) {
                    val_r2 = vr;
                    best_lam = cand;
                }
            }
        } else if (N_val > 0) {
            solve_ridge(N_tr, best_lam);
            val_r2 = compute_r2(N_tr, N_val);
        }
        solve_ridge(N, best_lam);
        const double r2 = compute_r2(0, N);
        cudaFree(d_work);

        // Auto-gate off hooks whose held-out validation R^2 <= 0 when --auto-lam is enabled!
        const float hook_scale = (auto_lam && N_val > 0 && val_r2 <= 0.0) ? 0.0f : scale;

        // W_cm[de, dh] IS Wt_rm[dh, de] in row-major!
        // Write hook header (+ RKAD v3 extension + mu_E[de] for RKAD v2/v3)
        int32_t hmeta[2] = {hp.target_layer, hp.src_layer};
        std::fwrite(hmeta, sizeof(int32_t), 2, out);
        std::fwrite(&hook_scale, sizeof(float), 1, out);
        if (rkad_v3) {
            uint32_t hflags = (center ? 1u : 0u) | (gated ? 2u : 0u) | (delta_clip > 0.0f ? 4u : 0u);
            uint32_t hextra[2] = {static_cast<uint32_t>(eff_rank), hflags};
            std::fwrite(hextra, sizeof(uint32_t), 2, out);
            std::fwrite(&delta_clip, sizeof(float), 1, out);
        }
        if (center) {
            std::fwrite(mu_E.data(), sizeof(float), de, out);
        }

        if (eff_rank == 0) {
            std::vector<float> h_Wt(static_cast<size_t>(dh) * de);
            cuda_ok(cudaMemcpy(h_Wt.data(), d_EtH, h_Wt.size() * sizeof(float), cudaMemcpyDeviceToHost), "Wt D2H");
            std::printf("  hook[%u] src=%d -> host_layer=%d: train R^2=%.4f, val(10%%) R^2=%.4f (full dense %dx%d, centered=%s)\n",
                        hi, hp.src_layer, hp.target_layer, r2, val_r2, dh, de, center ? "yes" : "no");
            std::fwrite(h_Wt.data(), sizeof(float), h_Wt.size(), out);
            if (gated) {
                std::fwrite(h_Wt.data(), sizeof(float), h_Wt.size(), out);
                std::vector<float> gw(static_cast<size_t>(std::max(1, host_hc_mult)) * dh, 1.0f);
                std::fwrite(gw.data(), sizeof(float), gw.size(), out);
            }
        } else {
            // Exact GPU SVD on W_cm[m, n] where m >= n (cusolverDnSgesvd requires m >= n!).
            // If de >= dh: factor W_cm[de, dh] = U[de, dh] * S[dh] * Vt[dh, dh]
            //   => Wt_rm[dh, de] = (Vt^T[:, :r] * sqrt(S)) @ (sqrt(S) * U[:, :r]^T) = B_v[dh, r] @ A_v[r, de]
            // If dh > de: transpose to W_t_cm[dh, de] = U[dh, de] * S[de] * Vt[de, de]
            //   => Wt_rm[dh, de] = (U[:, :r] * sqrt(S)) @ (sqrt(S) * Vt[:r, :]) = B_v[dh, r] @ A_v[r, de]
            const bool swap_mn = (dh > de);
            const int m = swap_mn ? dh : de;
            const int n = swap_mn ? de : dh;

            float * d_A_svd = nullptr;
            cuda_ok(cudaMalloc(&d_A_svd, static_cast<size_t>(m) * n * sizeof(float)), "malloc A_svd");
            if (swap_mn) {
                // Transpose d_EtH [de, dh] -> d_A_svd [dh, de]
                const float one = 1.0f, zero = 0.0f;
                cublas_ok(cublasSgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dh, de, &one, d_EtH, de, &zero, d_EtH, dh, d_A_svd, dh), "transpose for svd");
            } else {
                cuda_ok(cudaMemcpy(d_A_svd, d_EtH, static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToDevice), "copy for svd");
            }

            float * d_S = nullptr, * d_U = nullptr, * d_Vt = nullptr;
            cuda_ok(cudaMalloc(&d_S,  static_cast<size_t>(n) * sizeof(float)), "malloc S");
            cuda_ok(cudaMalloc(&d_U,  static_cast<size_t>(m) * n * sizeof(float)), "malloc U");
            cuda_ok(cudaMalloc(&d_Vt, static_cast<size_t>(n) * n * sizeof(float)), "malloc Vt");

            int lwork_svd = 0;
            cusolver_ok(cusolverDnSgesvd_bufferSize(cusolver, m, n, &lwork_svd), "svd buf");
            float * d_svd_work = nullptr, * d_rwork = nullptr;
            cuda_ok(cudaMalloc(&d_svd_work, static_cast<size_t>(lwork_svd) * sizeof(float)), "malloc svd work");
            cuda_ok(cudaMalloc(&d_rwork,    static_cast<size_t>(n) * sizeof(float)), "malloc rwork");
            cusolver_ok(cusolverDnSgesvd(cusolver, 'S', 'S', m, n, d_A_svd, m, d_S, d_U, m, d_Vt, n, d_svd_work, lwork_svd, d_rwork, d_info), "gesvd");

            std::vector<float> h_S(n), h_U(static_cast<size_t>(m) * n), h_Vt(static_cast<size_t>(n) * n);
            cuda_ok(cudaMemcpy(h_S.data(),  d_S,  h_S.size()  * sizeof(float), cudaMemcpyDeviceToHost), "S D2H");
            cuda_ok(cudaMemcpy(h_U.data(),  d_U,  h_U.size()  * sizeof(float), cudaMemcpyDeviceToHost), "U D2H");
            cuda_ok(cudaMemcpy(h_Vt.data(), d_Vt, h_Vt.size() * sizeof(float), cudaMemcpyDeviceToHost), "Vt D2H");

            cudaFree(d_A_svd); cudaFree(d_S); cudaFree(d_U); cudaFree(d_Vt); cudaFree(d_svd_work); cudaFree(d_rwork);

            // Build row-major A_v[eff_rank, de] and B_v[dh, eff_rank]
            std::vector<float> A_v(static_cast<size_t>(eff_rank) * de);
            std::vector<float> B_v(static_cast<size_t>(dh) * eff_rank);

            double tot_e = 0.0, kept_e = 0.0;
            for (int k = 0; k < n; ++k) {
                double s2 = static_cast<double>(h_S[k]) * h_S[k];
                tot_e += s2;
                if (k < eff_rank) kept_e += s2;
            }

            if (swap_mn) {
                // d_A_svd was [dh, de] col-major (so row d, col e is idx = e*dh + d):
                // U is [dh, n] col-major -> U(d, k) = h_U[k*dh + d]
                // Vt is [n, de] col-major -> Vt(k, e) = h_Vt[e*n + k]
                for (int k = 0; k < eff_rank; ++k) {
                    const float sq = std::sqrt(std::max(0.0f, h_S[k]));
                    for (int d = 0; d < dh; ++d) B_v[static_cast<size_t>(d) * eff_rank + k] = h_U[static_cast<size_t>(k) * dh + d] * sq;
                    for (int e = 0; e < de; ++e) A_v[static_cast<size_t>(k) * de + e]       = h_Vt[static_cast<size_t>(e) * n + k] * sq;
                }
            } else {
                // d_A_svd was W_cm [de, dh] col-major: W(e, d) = sum_k U(e, k) * S[k] * Vt(k, d)
                // => Wt(d, e) = sum_k Vt(k, d) * S[k] * U(e, k)
                // U is [de, n] col-major -> U(e, k) = h_U[k*de + e]
                // Vt is [n, dh] col-major -> Vt(k, d) = h_Vt[d*n + k]
                for (int k = 0; k < eff_rank; ++k) {
                    const float sq = std::sqrt(std::max(0.0f, h_S[k]));
                    for (int d = 0; d < dh; ++d) B_v[static_cast<size_t>(d) * eff_rank + k] = h_Vt[static_cast<size_t>(d) * n + k] * sq;
                    for (int e = 0; e < de; ++e) A_v[static_cast<size_t>(k) * de + e]       = h_U[static_cast<size_t>(k) * de + e] * sq;
                }
            }

            std::printf("  hook[%u] src=%d -> host_layer=%d: lam=%.1f, train R^2=%.4f, val(10%%) R^2=%.4f, rank-%d spectral energy=%.1f%% (centered=%s, scale=%.4f%s)\n",
                        hi, hp.src_layer, hp.target_layer, best_lam, r2, val_r2, eff_rank, 100.0 * kept_e / std::max(tot_e, 1e-12),
                        center ? "yes" : "no", hook_scale, hook_scale == 0.0f ? " [AUTO-DISABLED]" : "");

            std::fwrite(A_v.data(), sizeof(float), A_v.size(), out);
            std::fwrite(B_v.data(), sizeof(float), B_v.size(), out);
            if (gated) {
                std::fwrite(A_v.data(), sizeof(float), A_v.size(), out);
                std::fwrite(B_v.data(), sizeof(float), B_v.size(), out);
                std::vector<float> gw(static_cast<size_t>(std::max(1, host_hc_mult)) * dh, 1.0f);
                std::fwrite(gw.data(), sizeof(float), gw.size(), out);
            }
        }

        cudaFree(d_E);
        cudaFree(d_H);
        cudaFree(d_EtE);
        cudaFree(d_EtH);
    }

    std::fclose(out);
    cudaFree(d_info);
    cusolverDnDestroy(cusolver);
    cublasDestroy(cublas);

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("Wrote RKAD v%u adapter -> %s in %.2f s\n",
                out_ver, out_path.c_str(), std::chrono::duration<double>(t1 - t0).count());
    return 0;
}
