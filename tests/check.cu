// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& data) { if (data.size() != n) throw std::runtime_error("upload size"); cuda_check(cudaMemcpy(p, data.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> data(n); cuda_check(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return data; }
};
struct Case { std::string name; bool pass; };

int main() {
    std::vector<Case> cases;
    auto record = [&](const char* name, bool pass) { cases.push_back({name, pass}); };
    cudaStream_t stream = nullptr;
    try {
        cuda_check(cudaStreamCreate(&stream));
        const int rows = 19, columns = 37;
        std::vector<int> offsets(1, 0), indices;
        std::vector<float> values, x(columns), initial(rows);
        for (int j = 0; j < columns; ++j) x[j] = (j % 7 - 3) * 0.125f;
        for (int i = 0; i < rows; ++i) {
            const int length = i == 0 ? 0 : (i == 1 ? 1057 : 1 + i % 13);
            for (int k = 0; k < length; ++k) { indices.push_back((i * 11 + k * 7) % columns); values.push_back((k % 11 - 5) * 0.0625f); }
            offsets.push_back(static_cast<int>(values.size())); initial[i] = i * 0.25f;
        }
        Device<int> d_offsets(offsets.size()), d_indices(indices.size());
        Device<float> d_values(values.size()), d_x(x.size()), d_y(initial.size());
        d_offsets.put(offsets); d_indices.put(indices); d_values.put(values); d_x.put(x); d_y.put(initial);
        auto parity = [&](float alpha, float beta) {
            d_y.put(initial);
            cuda_check(sfspmv::multiply(rows, columns, static_cast<int>(values.size()), d_offsets.p,
                d_indices.p, d_values.p, d_x.p, d_y.p, alpha, beta, stream));
            cuda_check(cudaStreamSynchronize(stream));
            const auto actual = d_y.get();
            bool pass = true;
            for (int i = 0; i < rows; ++i) {
                double sum = 0.0;
                for (int k = offsets[i]; k < offsets[i + 1]; ++k) sum += static_cast<double>(values[k]) * x[indices[k]];
                const double expected = alpha * sum + beta * initial[i];
                pass &= std::isfinite(actual[i]) && std::abs(actual[i] - expected) <= 2e-5 * (1.0 + std::abs(expected));
            }
            return pass;
        };
        record("csr-empty-long-duplicate-and-tail-rows", parity(1.0f, 0.0f));
        record("alpha-beta-vs-independent-scalar", parity(1.75f, -0.5f));
        record("zero-alpha", parity(0.0f, 2.0f));
        record("reject-invalid-dimensions-and-alias",
            sfspmv::multiply(0, columns, 0, d_offsets.p, nullptr, nullptr, d_x.p, d_y.p) == cudaErrorInvalidValue &&
            sfspmv::multiply(rows, columns, -1, d_offsets.p, d_indices.p, d_values.p, d_x.p, d_y.p) == cudaErrorInvalidValue &&
            sfspmv::multiply(rows, columns, 0, d_offsets.p, nullptr, nullptr, d_x.p, d_x.p) == cudaErrorInvalidValue);
        std::vector<int> zero_offsets(rows + 1, 0);
        d_offsets.put(zero_offsets);
        cuda_check(sfspmv::multiply(rows, columns, 0, d_offsets.p, nullptr, nullptr, d_x.p, d_y.p, 1.0f, 0.0f, stream));
        cuda_check(cudaStreamSynchronize(stream));
        bool all_zero = true; for (float value : d_y.get()) all_zero &= value == 0.0f;
        record("zero-nnz-null-storage", all_zero);
        offsets[2] = static_cast<int>(values.size()) + 1;
        d_offsets.put(offsets);
        cuda_check(sfspmv::multiply(rows, columns, static_cast<int>(values.size()), d_offsets.p, d_indices.p,
            d_values.p, d_x.p, d_y.p, 1.0f, 0.0f, stream));
        cuda_check(cudaStreamSynchronize(stream));
        record("malformed-csr-row-is-nan", std::isnan(d_y.get()[1]));
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); record("cuda-runtime", false); }
    if (stream) cudaStreamDestroy(stream);
    bool pass = !cases.empty(); for (const auto& item : cases) pass &= item.pass;
    std::printf("{\"package\":\"sfSpMV\",\"status\":\"%s\",\"cases\":[", pass ? "PASS" : "FAIL");
    for (size_t i = 0; i < cases.size(); ++i) std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].name.c_str(), cases[i].pass ? "PASS" : "FAIL");
    std::puts("]}"); return pass ? 0 : 1;
}
