#include "harryNET/tensor.h"
#include "harryNET/ops.h"
#include "harryNET/module.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace harrynet;
using namespace std::chrono;

//  timer utility
class Timer {
public:
    Timer() : start_time(high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end_time = high_resolution_clock::now();
        return duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
    }
    
    void reset() {
        start_time = high_resolution_clock::now();
    }
    
private:
    high_resolution_clock::time_point start_time;
};

// run a benchmark and return average time
template<typename Func>
double benchmark(const std::string& name, Func func, int warmup = 10, int iterations = 100) {
    std::cout << "Benchmarking: " << name << std::endl;
    
    // warmup runs
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    
    // timed runs
    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    double total_time = timer.elapsed_ms();
    double avg_time = total_time / iterations;
    
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) 
              << avg_time << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) 
              << (1000.0 / avg_time) << " ops/sec" << std::endl;
    
    return avg_time;
}

void benchmark_elementwise_ops() {
    std::cout << "\n=== Element-wise Operations Benchmark ===" << std::endl;
    
    std::vector<Shape> shapes = {
        Shape({1000}),        // 1K elements
        Shape({100, 100}),    // 10K elements
        Shape({1000, 1000}),  // 1M elements
        Shape({100, 100, 100}) // 1M elements (3D)
    };
    
    for (const auto& shape : shapes) {
        std::cout << "\nShape: " << shape.to_string() 
                  << " (" << shape.numel() << " elements)" << std::endl;
        
        auto a = Tensor::randn(shape);
        auto b = Tensor::randn(shape);
        
        // addition
        benchmark("Addition", [&]() {
            auto c = add(a, b);
        });
        
        // multiplication
        benchmark("Multiplication", [&]() {
            auto c = multiply(a, b);
        });
        
        // ReLU
        benchmark("ReLU", [&]() {
            auto c = relu(a);
        });
        
        // Sigmoid
        benchmark("Sigmoid", [&]() {
            auto c = sigmoid(a);
        });
        
        // Inplace operations
        auto temp = a->clone();
        benchmark("Inplace Add", [&]() {
            temp->add_(b);
            temp = a->clone(); // reset for next iteration
        });
    }
}

void benchmark_matmul() {
    std::cout << "\n=== Matrix Multiplication Benchmark ===" << std::endl;
    
    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 128},    // small
        {512, 512, 512},    // Medium
        {1024, 1024, 1024}, // Large
        {2048, 2048, 2048}  // Very large
    };
    
    for (const auto& [m, n, k] : sizes) {
        std::cout << "\nMatrix sizes: " << m << "x" << k << " @ " << k << "x" << n << std::endl;
        
        auto a = Tensor::randn(Shape({m, k}));
        auto b = Tensor::randn(Shape({k, n}));
        
        double time = benchmark("MatMul", [&]() {
            auto c = matmul(a, b);
        }, 5, 20);  // fewer iterations for large matrices
        
        // calculate GFLOPS
        double flops = 2.0 * m * n * k;  // 2 ops per multiply-add
        double gflops = (flops / 1e9) / (time / 1000.0);
        std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                  << gflops << " GFLOPS" << std::endl;
    }
}

void benchmark_broadcasting() {
    std::cout << "\n=== Broadcasting Operations Benchmark ===" << std::endl;
    
    auto a = Tensor::randn(Shape({100, 100, 100}));  // 1M elements
    auto b1 = Tensor::randn(Shape({100}));           // Vector
    auto b2 = Tensor::randn(Shape({100, 1}));        // Column vector
    auto b3 = Tensor::randn(Shape({1, 100}));        // Row vector
    
    benchmark("Broadcast 3D + 1D", [&]() {
        auto c = add(a, b1);
    });
    
    benchmark("Broadcast 3D + column", [&]() {
        auto c = add(a, b2);
    });
    
    benchmark("Broadcast 3D + row", [&]() {
        auto c = add(a, b3);
    });
}

void benchmark_neural_network() {
    std::cout << "\n=== Neural Network Layer Benchmark ===" << std::endl;
    
    // typical layer sizes
    std::vector<std::tuple<int, int, int>> configs = {
        {128, 784, 256},   // MNIST hidden layer
        {64, 1024, 512},   // Medium layer
        {32, 2048, 1024},  // Large layer
    };
    
    for (const auto& [batch_size, in_features, out_features] : configs) {
        std::cout << "\nLinear layer: batch=" << batch_size 
                  << ", in=" << in_features << ", out=" << out_features << std::endl;
        
        auto linear = std::make_shared<Linear>(in_features, out_features);
        auto input = Tensor::randn(Shape({batch_size, in_features}));
        
        benchmark("Forward pass", [&]() {
            auto output = linear->forward(input);
        });
        
        // With activation
        benchmark("Linear + ReLU", [&]() {
            auto output = linear->forward(input);
            auto activated = relu(output);
        });
    }
}

void benchmark_autograd() {
    std::cout << "\n=== Autograd Performance Benchmark ===" << std::endl;
    
    // simple network
    auto x = Tensor::randn(Shape({64, 784}), true);
    auto w1 = Tensor::randn(Shape({784, 256}), true);
    auto w2 = Tensor::randn(Shape({256, 10}), true);
    
    benchmark("Forward + Backward", [&]() {
        // forward
        auto h = relu(matmul(x, w1));
        auto y = matmul(h, w2);
        auto loss = sum(multiply(y, y));  //  L2 loss!
        
        // backward
        loss->backward();
        
        // clear gradients for next iteration
        x->zero_grad();
        w1->zero_grad();
        w2->zero_grad();
    }, 5, 50);
}

void print_system_info() {
    std::cout << "=== System Information ===" << std::endl;
    
#ifdef HARRYNET_AVX2
    std::cout << "SIMD: AVX2 enabled (256-bit vectors)" << std::endl;
#elif defined(HARRYNET_NEON)
    std::cout << "SIMD: NEON enabled (128-bit vectors)" << std::endl;
#else
    std::cout << "SIMD: Disabled (scalar fallback)" << std::endl;
#endif
    
#ifdef _OPENMP
    std::cout << "OpenMP: Enabled" << std::endl;
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif
    
    std::cout << "Tensor alignment: " << alignof(Tensor) << " bytes" << std::endl;
}

int main() {
    std::cout << "HarryNET Performance Benchmark Suite" << std::endl;
    std::cout << "====================================" << std::endl;
    
    print_system_info();
    
    benchmark_elementwise_ops();
    benchmark_matmul();
    benchmark_broadcasting();
    benchmark_neural_network();
    benchmark_autograd();
    
    std::cout << "\nBenchmark complete!" << std::endl;
    
    return 0;
}