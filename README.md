# sfSpMV

Standalone CUDA C++17 real-f32 compressed sparse row matrix-vector product:
`y = alpha A x + beta y`. The implementation is a fresh MIT mathematical
kernel, not an extraction or relicensing of proprietary solidSF or rvllm code.
Its only library dependency is the CUDA runtime.

`src/kernel.cuh` exports `sfspmv::multiply(rows, columns, nnz, offsets,
indices, values, x, y, alpha, beta, stream)`. Offsets and column indices are
zero-based signed 32-bit integers; data and vectors are f32. A 128-thread
block assigns one warp to each of four rows. Lanes stride over a row's
nonzeros and reduce with shuffle instructions. Empty rows, repeated column
indices, non-multiples of four rows and long rows are supported. This is
linear work in rows plus nonzeros; it does not scan every nonzero for every row.

Rows and columns must be positive, and `nnz` nonnegative. Device buffers must
hold `rows + 1`, `nnz`, `nnz`, `columns` and `rows` entries respectively.
Indices/values may be null when `nnz == 0`. Offsets must begin at zero, end
at `nnz`, and be monotone. Column indices must be in range. A malformed row
or out-of-range column produces a NaN for that row without indexing outside
the declared CSR/vector bounds. The buffer extents and pointer residency are
the caller's responsibility. Input/output vector overlap is unsupported.
When beta is zero, the previous contents of y are not read.

The API validates host dimensions, null pointers, exact x/y aliasing and
finite alpha/beta. It launches on the supplied stream and returns
`cudaGetLastError()` without synchronization. The caller must check stream
completion before reading outputs. Partial buffer overlap cannot be checked
from the supplied device pointers. Non-finite matrix/vector inputs propagate
through IEEE arithmetic; this package does not sanitize them.

Required device targets are H100 and H200 (`sm_90`), B200 (`sm_100`), B300/GB300
(`sm_103`) and GB10 (`sm_121`). It uses portable CUDA cores and warp shuffles,
with no device-family-specific tensor instructions. Build with a CUDA toolkit
that supports the selected target. Compilation and numerical hardware
qualification are separate; this source alone proves neither speed nor a
device qualification.

`tests/check.cu` emits one JSON object and exits zero only if every case
passes. It compares against independent scalar f64 CSR arithmetic, and
checks alpha/beta, duplicate columns, empty/long/tail rows, zero nonzeros,
domain rejection and malformed CSR. The parent package tooling supplies the
build and qualification commands. `make -j1 ARCHS='90 100 103 121'
BUILD_DIR=<durable-path>` builds all target images; `make test` runs
`BUILD_DIR/check`. A minimal direct compile is
`nvcc -std=c++17 -O3 -arch=sm_121 tests/check.cu -o <retained-staging-path>/check`.
Run only within the approved resource and duration limits.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfSpMV](https://github.com/solidsf-inc/sfSpMV). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| spmv f32; 4096 x 4096, nnz=65536 | 5.984 | 14.240 | 2.380x |

References: cuSPARSE cusparseSpMV CSR ALG2.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 6 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
