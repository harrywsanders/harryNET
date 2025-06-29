#ifndef HARRYNET_BLAS_H
#define HARRYNET_BLAS_H

#include <cstdint>
#include <algorithm>
#include <cstring>
#include "simd.h"

namespace harrynet {
namespace blas {

// cache-friendly tile sizes (tuned for typical L1/L2 cache sizes)
constexpr int64_t TILE_M = 64;  // rows of A and C
constexpr int64_t TILE_N = 64;  // cols of B and C
constexpr int64_t TILE_K = 64;  // cols of A and rows of B

// micro-kernel sizes for register blocking
constexpr int64_t MR = 8;  // register block size for M dimension
constexpr int64_t NR = 8;  // register block size for N dimension

inline void gemm_micro_kernel(
    int64_t m, int64_t n, int64_t k,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float* C, int64_t ldc
);

// C = alpha * A * B + beta * C
// where A is M x K, B is K x N, C is M x N
inline void gemm(
    bool transA, bool transB,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float beta,
    float* C, int64_t ldc
) {
    if (transA || transB) {
        throw std::runtime_error("Transposed GEMM not yet implemented");
    }
    
    // edge cases
    if (M == 0 || N == 0 || K == 0) return;
    
    // scale C by beta if needed
    if (beta != 1.0f) {
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // early exit if alpha is zero
    if (alpha == 0.0f) return;
    
    // main tiled computation
    #pragma omp parallel for collapse(2)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            // determine actual tile size (handle edge tiles)
            int64_t tile_m = std::min(TILE_M, M - i0);
            int64_t tile_n = std::min(TILE_N, N - j0);
            
            // allocate tile buffer for C (in L1 cache)
            alignas(64) float C_tile[TILE_M * TILE_N] = {0};
            
            // loop over K dimension in tiles
            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t tile_k = std::min(TILE_K, K - k0);
                
                // micro-kernel: compute C_tile += A_tile * B_tile
                gemm_micro_kernel(
                    tile_m, tile_n, tile_k,
                    &A[i0 * lda + k0], lda,
                    &B[k0 * ldb + j0], ldb,
                    C_tile, TILE_N
                );
            }
            
            // write tile back to C with alpha scaling
            for (int64_t i = 0; i < tile_m; ++i) {
                for (int64_t j = 0; j < tile_n; ++j) {
                    C[(i0 + i) * ldc + j0 + j] += alpha * C_tile[i * TILE_N + j];
                }
            }
        }
    }
}

// optimized micro-kernel for small matrix multiplication
inline void gemm_micro_kernel(
    int64_t m, int64_t n, int64_t k,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float* C, int64_t ldc
) {
#ifdef HARRYNET_AVX2
    // process MR x NR blocks w SIMD
    for (int64_t i = 0; i < m; i += MR) {
        for (int64_t j = 0; j < n; j += NR) {
            int64_t mr = std::min(MR, m - i);
            int64_t nr = std::min(NR, n - j);
            
            // load and compute MR x NR block
            __m256 c_regs[MR];
            
            // init accumulators
            for (int64_t ii = 0; ii < mr; ++ii) {
                if (nr == NR) {
                    c_regs[ii] = _mm256_loadu_ps(&C[(i + ii) * ldc + j]);
                } else {
                    // handle edge case with partial load
                    alignas(32) float temp[NR] = {0};
                    std::memcpy(temp, &C[(i + ii) * ldc + j], nr * sizeof(float));
                    c_regs[ii] = _mm256_loadu_ps(temp);
                }
            }
            
            // compute dot products
            for (int64_t kk = 0; kk < k; ++kk) {
                // load B vector
                __m256 b_vec;
                if (nr == NR) {
                    b_vec = _mm256_loadu_ps(&B[kk * ldb + j]);
                } else {
                    alignas(32) float temp[NR] = {0};
                    std::memcpy(temp, &B[kk * ldb + j], nr * sizeof(float));
                    b_vec = _mm256_loadu_ps(temp);
                }
                
                // multiply-accumulate for each row of A
                for (int64_t ii = 0; ii < mr; ++ii) {
                    __m256 a_scalar = _mm256_set1_ps(A[(i + ii) * lda + kk]);
                    c_regs[ii] = _mm256_fmadd_ps(a_scalar, b_vec, c_regs[ii]);
                }
            }
            
            // store results
            for (int64_t ii = 0; ii < mr; ++ii) {
                if (nr == NR) {
                    _mm256_storeu_ps(&C[(i + ii) * ldc + j], c_regs[ii]);
                } else {
                    // edge case with partial store
                    alignas(32) float temp[NR];
                    _mm256_storeu_ps(temp, c_regs[ii]);
                    std::memcpy(&C[(i + ii) * ldc + j], temp, nr * sizeof(float));
                }
            }
        }
    }
#else
    constexpr int64_t BLOCK_SIZE = 4;
    
    for (int64_t i = 0; i < m; i += BLOCK_SIZE) {
        for (int64_t j = 0; j < n; j += BLOCK_SIZE) {
            int64_t block_m = std::min(BLOCK_SIZE, m - i);
            int64_t block_n = std::min(BLOCK_SIZE, n - j);
            
            // load C block into registers
            float c_block[BLOCK_SIZE][BLOCK_SIZE];
            for (int64_t ii = 0; ii < block_m; ++ii) {
                for (int64_t jj = 0; jj < block_n; ++jj) {
                    c_block[ii][jj] = C[(i + ii) * ldc + j + jj];
                }
            }
            
            // compute matrix multiplication for this block
            for (int64_t kk = 0; kk < k; ++kk) {
                for (int64_t ii = 0; ii < block_m; ++ii) {
                    float a_val = A[(i + ii) * lda + kk];
                    for (int64_t jj = 0; jj < block_n; ++jj) {
                        c_block[ii][jj] += a_val * B[kk * ldb + j + jj];
                    }
                }
            }
            
            // store C block back
            for (int64_t ii = 0; ii < block_m; ++ii) {
                for (int64_t jj = 0; jj < block_n; ++jj) {
                    C[(i + ii) * ldc + j + jj] = c_block[ii][jj];
                }
            }
        }
    }
#endif
}

// specialized fn for matrix-vector multiplication (GEMV)
// y = alpha * A * x + beta * y
inline void gemv(
    bool transA,
    int64_t M, int64_t N,
    float alpha,
    const float* A, int64_t lda,
    const float* x,
    float beta,
    float* y
) {
    if (transA) {
        // y = alpha * A^T * x + beta * y
        // result size is N
        #pragma omp parallel for
        for (int64_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < M; ++j) {
                sum += A[j * lda + i] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        // y = alpha * A * x + beta * y
        // result size is M
        #pragma omp parallel for
        for (int64_t i = 0; i < M; ++i) {
            float sum = simd::dot_product_vec(&A[i * lda], x, N);
            y[i] = alpha * sum + beta * y[i];
        }
    }
}

// batched matmul for tensor ops
inline void batched_gemm(
    int64_t batch_size,
    bool transA, bool transB,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t strideA,
    const float* B, int64_t strideB,
    float beta,
    float* C, int64_t strideC
) {
    #pragma omp parallel for
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        gemm(transA, transB, M, N, K, alpha,
             A + batch * strideA, K,
             B + batch * strideB, N,
             beta,
             C + batch * strideC, N);
    }
}

} // namespace blas
} // namespace harrynet

#endif // HARRYNET_BLAS_H