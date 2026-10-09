// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace sfspmv {

static __global__ void csr_kernel(int rows, int columns, int nnz, const int* offsets,
    const int* indices, const float* values, const float* x, float* y, float alpha, float beta) {
    const int lane = threadIdx.x & 31;
    const long long row = static_cast<long long>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int begin = offsets[row], end = offsets[row + 1];
    float sum = 0.0f;
    if (begin < 0 || end < begin || end > nnz || (row == 0 && begin != 0) ||
        (row == rows - 1 && end != nnz)) {
        if (lane == 0) y[row] = nanf("");
        return;
    }
    for (long long k = static_cast<long long>(begin) + lane; k < end; k += 32) {
        const int column = indices[k];
        sum = column >= 0 && column < columns ? fmaf(values[k], x[column], sum) : nanf("");
    }
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[row] = beta == 0.0f ? alpha * sum : fmaf(alpha, sum, beta * y[row]);
}

inline cudaError_t multiply(int rows, int columns, int nnz, const int* offsets,
    const int* indices, const float* values, const float* x, float* y,
    float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = nullptr) {
    if (rows <= 0 || columns <= 0 || nnz < 0 || !offsets || !x || !y ||
        (nnz && (!indices || !values)) || x == y || !std::isfinite(alpha) || !std::isfinite(beta))
        return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((static_cast<long long>(rows) + 3) / 4);
    csr_kernel<<<blocks, 128, 0, stream>>>(rows, columns, nnz, offsets, indices, values, x, y, alpha, beta);
    return cudaGetLastError();
}
}
