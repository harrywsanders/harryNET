#ifndef HARRYNET_SIMD_H
#define HARRYNET_SIMD_H

#include <cstdint>
#include <cmath>
#include <algorithm>

// platform detection and SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define HARRYNET_X86
    #include <immintrin.h>
    #ifdef __AVX2__
        #define HARRYNET_AVX2
    #endif
    #ifdef __AVX512F__
        #define HARRYNET_AVX512
    #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define HARRYNET_NEON
    #include <arm_neon.h>
#endif

namespace harrynet {
namespace simd {

// SIMD vector width consts
#ifdef HARRYNET_AVX512
    constexpr int SIMD_WIDTH = 16;  // 512 bits / 32 bits per float
#elif defined(HARRYNET_AVX2)
    constexpr int SIMD_WIDTH = 8;   // 256 bits / 32 bits per float
#elif defined(HARRYNET_NEON)
    constexpr int SIMD_WIDTH = 4;   // 128 bits / 32 bits per float
#else
    constexpr int SIMD_WIDTH = 1;   // Scalar fallback
#endif

// aligned memory allocation
inline void* aligned_malloc(size_t size, size_t alignment = 64) {
    #ifdef _WIN32
        return _aligned_malloc(size, alignment);
    #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
    #endif
}

inline void aligned_free(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// SIMD ops for different archs
#ifdef HARRYNET_AVX2

// AVX2 impls
inline void add_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void multiply_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void subtract_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

inline void divide_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!   
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] / b[i];
    }
}

// fused multiply-add: result = a * b + c
inline void fma_vec(const float* a, const float* b, const float* c, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// ReLU activation
inline void relu_vec(const float* input, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    __m256 zero = _mm256_setzero_ps();
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 vi = _mm256_loadu_ps(&input[i]);
        __m256 vr = _mm256_max_ps(vi, zero);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = std::max(0.0f, input[i]);
    }
}

// sigmoid activation (approx for perf)
inline void sigmoid_vec(const float* input, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    __m256 one = _mm256_set1_ps(1.0f);
    
    // SIMD loop - using fast approx
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 vi = _mm256_loadu_ps(&input[i]);
        __m256 neg_vi = _mm256_sub_ps(_mm256_setzero_ps(), vi);
        
        // fast exp approx (less accurate but much faster)
        // exp(-x) ≈ 1 / (1 + x + 0.5*x^2) for small x
        // for larger x, clamp to avoid overflow
        __m256 clamped = _mm256_max_ps(_mm256_set1_ps(-10.0f), 
                                       _mm256_min_ps(_mm256_set1_ps(10.0f), neg_vi));
        
        // polynomial approx of exp
        __m256 x2 = _mm256_mul_ps(clamped, clamped);
        __m256 exp_approx = _mm256_add_ps(one, 
                           _mm256_add_ps(clamped, 
                           _mm256_mul_ps(_mm256_set1_ps(0.5f), x2)));
        
        // sigmoid = 1 / (1 + exp(-x))
        __m256 vr = _mm256_div_ps(one, exp_approx);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // scalar remainder with accurate computation
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

#elif defined(HARRYNET_NEON)

// ARM NEON impls
inline void add_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void multiply_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

// add other NEON impls...
inline void subtract_vec(const float* a, const float* b, float* result, int64_t size) {
    int64_t simd_size = size - (size % SIMD_WIDTH);
    
    // SIMD loop!
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vsubq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
    
    // scalar remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

inline void divide_vec(const float* a, const float* b, float* result, int64_t size) {
    // TODO: implement optimized NEON version
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] / b[i];
    }
}

inline void fma_vec(const float* a, const float* b, const float* c, float* result, int64_t size) {
    // TODO: Implement optimized NEON version
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

inline void relu_vec(const float* input, float* result, int64_t size) {
    // TODO: Implement optimized NEON version
    for (int64_t i = 0; i < size; ++i) {
        result[i] = std::max(0.0f, input[i]);
    }
}

inline void sigmoid_vec(const float* input, float* result, int64_t size) {
    // TODO: Implement optimized NEON version
    for (int64_t i = 0; i < size; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

#else

// scalar fallback impls
inline void add_vec(const float* a, const float* b, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void multiply_vec(const float* a, const float* b, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void subtract_vec(const float* a, const float* b, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

inline void divide_vec(const float* a, const float* b, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] / b[i];
    }
}

inline void fma_vec(const float* a, const float* b, const float* c, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

inline void relu_vec(const float* input, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = std::max(0.0f, input[i]);
    }
}

inline void sigmoid_vec(const float* input, float* result, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

#endif

// reduction ops with SIMD
inline float sum_vec(const float* data, int64_t size) {
    float sum = 0.0f;
    
#ifdef HARRYNET_AVX2
    int64_t simd_size = size - (size % SIMD_WIDTH);
    __m256 vsum = _mm256_setzero_ps();
    
    // SIMD accum
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 vdata = _mm256_loadu_ps(&data[i]);
        vsum = _mm256_add_ps(vsum, vdata);
    }
    
    // horizontal sum of SIMD reg
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum = _mm_cvtss_f32(sums);
    
    // add remainder
    for (int64_t i = simd_size; i < size; ++i) {
        sum += data[i];
    }
#else
    // scalar fallback
    for (int64_t i = 0; i < size; ++i) {
        sum += data[i];
    }
#endif
    
    return sum;
}

// dot product with SIMD
inline float dot_product_vec(const float* a, const float* b, int64_t size) {
    float result = 0.0f;
    
#ifdef HARRYNET_AVX2
    int64_t simd_size = size - (size % SIMD_WIDTH);
    __m256 vsum = _mm256_setzero_ps();
    
    // SIMD dot prod
    for (int64_t i = 0; i < simd_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    
    // horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    result = _mm_cvtss_f32(sums);
    
    // add remainder
    for (int64_t i = simd_size; i < size; ++i) {
        result += a[i] * b[i];
    }
#else
    // scalar fallback
    for (int64_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
#endif
    
    return result;
}

} // namespace simd
} // namespace harrynet

#endif // HARRYNET_SIMD_H