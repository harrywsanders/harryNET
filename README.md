# harryNET

> A medium-performance, N-dim tensor library with automatic differentiation built in C++17

## What's harryNET?

harryNET is my tensor computation library, which I designed for building and training neural networks from scratch. It features:

- **N-dimensional tensor operations** with broadcasting support
- **Autograd** for gradient computation!
- **SIMD vectorization** (AVX2/NEON) for pretty fast element-wise operations!
- **Cache-efficieny optimizations** for matrix multiplication
- **Memory-efficient operations** like inplace tensor modifications

## Key Features

### N-Dimensional Tensor Support
```cpp
// create tensors of any dim!
auto tensor_1d = Tensor::create({100});
auto tensor_2d = Tensor::create({64, 128});
auto tensor_3d = Tensor::create({32, 64, 128});
auto tensor_4d = Tensor::create({16, 3, 224, 224});  // batch, channels, height, width

//  broadcasting works automatically
auto a = Tensor::create({5, 1, 4});
auto b = Tensor::create({1, 3, 4});
auto c = add(a, b);  // res shape: {5, 3, 4}

// generalized matrix multiplication
auto batch_matmul = matmul(tensor_3d, tensor_3d.transpose(-1, -2));
```

### High-Performance Operations
```cpp
// SIMD-accelerated element-wise operations (4-8x faster than naive impl! )
auto result = add(a, b);      // uses AVX2/NEON automatically
auto product = multiply(a, b); // vectorized computation

// cache-efficient matrix multiplication (3-5x faster than naive impl!)
auto output = matmul(input, weight);  // tiled algorithm for cache efficiency

tensor->add_(1.0f);      // no memalloc
tensor->multiply_(2.0f); // modifies in-place
```

### Automatic Differentiation
```cpp
// enable gradient tracking
auto x = Tensor::create({2, 3}, true);  // requires_grad = true
auto w = Tensor::create({3, 4}, true);
auto b = Tensor::create({4}, true);

// fwd pass with autograd
auto y = matmul(x, w);
auto z = add(y, b);
auto loss = mean(square(z));

// backward pass
loss->backward();

// get gradients
auto x_grad = x->grad();  // ∂loss/∂x
auto w_grad = w->grad();  // ∂loss/∂w
```

### Neural Network Modules
```cpp
// build models!
auto model = std::make_shared<Sequential>();
model->add(std::make_shared<Linear>(784, 256));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Dropout>(0.5));
model->add(std::make_shared<Linear>(256, 128));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Linear>(128, 10));

// fwd pass
auto output = model->forward(input);

// some optimizers with modern features
auto optimizer = Adam(model->parameter_list(), 0.001f);
optimizer.zero_grad();
loss->backward();
optimizer.step();
```

## Installation

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- OpenMP (optional, for parallelization)
- AVX2 support (optional, for x86_64 SIMD)
- NEON support (optional, for ARM SIMD)

### Building
```bash
git clone https://github.com/harrywsanders/harryNET.git
cd harryNET
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```


## Performance Benchmarks
We get some cool speedups from our tricks! 

| Operation | Naive Implementation | harryNET | Speedup |
|-----------|---------------------|----------|---------|
| Element-wise Add (1M elements) | 2.1ms | 0.3ms | 7x |
| Matrix Multiply (512x512) | 145ms | 28ms | 5.2x |
| ReLU Activation (1M elements) | 1.8ms | 0.2ms | 9x |
| Batch Normalization | 4.5ms | 0.9ms | 5x |


## Future Goals

- [ ] **Operation Fusion**: Combine multiple operations for better cache utilization
- [ ] **Einsum Operation**: Flexible tensor contractions with Einstein notation
- [ ] **Convolutional Layers**: Extend to support CNNs with efficient im2col
- [ ] **Memory Pool Allocator**: Reduce allocation overhead for temporary tensors
- [ ] **Distributed Training**: Multi-GPU and distributed training support?

