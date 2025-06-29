#ifndef HARRYNET_OPS_H
#define HARRYNET_OPS_H

#include "tensor.h"
#include "autograd.h"
#include "simd.h"
#include "blas.h"
#include <cmath>
#include <random>

namespace harrynet {


// tensor operations with autograd support

// element-wise operations
inline TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw std::runtime_error("Cannot add null tensors");
    }
    
    // check broadcasting
    auto result_shape = Shape::broadcast_shape(a->shape(), b->shape());
    
    // create result tensor
    auto result = Tensor::create(result_shape, a->requires_grad() || b->requires_grad());
    
    float* r_data = result->data();
    const float* a_data = a->data();
    const float* b_data = b->data();
    
    // Fast path: same shape tensors - use SIMD
    if (a->shape() == b->shape() && a->shape() == result_shape) {
        simd::add_vec(a_data, b_data, r_data, result->numel());
    } else {
        // suuuuper slow path: broadcasting required
        // get strides for all tensors
        auto result_strides = result_shape.strides();
        auto a_strides = a->shape().strides();
        auto b_strides = b->shape().strides();
        
        // prepare broadcast dimensions
        int result_ndim = result_shape.ndim();
        std::vector<int64_t> a_shape_padded(result_ndim, 1);
        std::vector<int64_t> b_shape_padded(result_ndim, 1);
        std::vector<int64_t> a_strides_padded(result_ndim, 0);
        std::vector<int64_t> b_strides_padded(result_ndim, 0);
        
        // pad shapes and strides from the right (align to the right)
        int a_offset = result_ndim - a->ndim();
        int b_offset = result_ndim - b->ndim();
        
        for (int i = 0; i < a->ndim(); ++i) {
            a_shape_padded[a_offset + i] = a->shape()[i];
            a_strides_padded[a_offset + i] = a_strides[i];
        }
        
        for (int i = 0; i < b->ndim(); ++i) {
            b_shape_padded[b_offset + i] = b->shape()[i];
            b_strides_padded[b_offset + i] = b_strides[i];
        }
        
        // compute the result with broadcasting
        int64_t total_size = result->numel();
        
        #pragma omp parallel for
        for (int64_t idx = 0; idx < total_size; ++idx) {
            // convert linear index to multi-dim indices
            int64_t temp_idx = idx;
            int64_t a_idx = 0;
            int64_t b_idx = 0;
            
            for (int dim = result_ndim - 1; dim >= 0; --dim) {
                int64_t coord = temp_idx % result_shape[dim];
                temp_idx /= result_shape[dim];
                
                // get indices for a and b considering broadcasting
                if (a_shape_padded[dim] != 1) {
                    a_idx += coord * a_strides_padded[dim];
                }
                if (b_shape_padded[dim] != 1) {
                    b_idx += coord * b_strides_padded[dim];
                }
            }
            
            r_data[idx] = a_data[a_idx] + b_data[b_idx];
        }
    }
    
    // setup backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<AddBackward>(a->shape(), b->shape());
        grad_fn->add_next_edge(a, 0);
        grad_fn->add_next_edge(b, 1);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

inline TensorPtr multiply(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw std::runtime_error("Cannot multiply null tensors");
    }
    
    // check broadcasting
    auto result_shape = Shape::broadcast_shape(a->shape(), b->shape());
    
    // create res tensor
    auto result = Tensor::create(result_shape, a->requires_grad() || b->requires_grad());
    
    float* r_data = result->data();
    const float* a_data = a->data();
    const float* b_data = b->data();
    
    // fast path: same shape tensors - use SIMD
    if (a->shape() == b->shape() && a->shape() == result_shape) {
        simd::multiply_vec(a_data, b_data, r_data, result->numel());
    } else {
        // slow path: broadcasting required
        // get strides for all tensors
        auto result_strides = result_shape.strides();
        auto a_strides = a->shape().strides();
        auto b_strides = b->shape().strides();
        
        // prepare broadcast dimensions
        int result_ndim = result_shape.ndim();
        std::vector<int64_t> a_shape_padded(result_ndim, 1);
        std::vector<int64_t> b_shape_padded(result_ndim, 1);
        std::vector<int64_t> a_strides_padded(result_ndim, 0);
        std::vector<int64_t> b_strides_padded(result_ndim, 0);
        
        // pad shapes and strides from the right (align to the right)
        int a_offset = result_ndim - a->ndim();
        int b_offset = result_ndim - b->ndim();
        
        for (int i = 0; i < a->ndim(); ++i) {
            a_shape_padded[a_offset + i] = a->shape()[i];
            a_strides_padded[a_offset + i] = a_strides[i];
        }
        
        for (int i = 0; i < b->ndim(); ++i) {
            b_shape_padded[b_offset + i] = b->shape()[i];
            b_strides_padded[b_offset + i] = b_strides[i];
        }
        
        // compute the result with broadcasting
        int64_t total_size = result->numel();
        
        #pragma omp parallel for
        for (int64_t idx = 0; idx < total_size; ++idx) {
            // convert linear index to multi-dim indices
            int64_t temp_idx = idx;
            int64_t a_idx = 0;
            int64_t b_idx = 0;
            
            for (int dim = result_ndim - 1; dim >= 0; --dim) {
                int64_t coord = temp_idx % result_shape[dim];
                temp_idx /= result_shape[dim];
                
                // get indices for a and b w/ broadcasting
                if (a_shape_padded[dim] != 1) {
                    a_idx += coord * a_strides_padded[dim];
                }
                if (b_shape_padded[dim] != 1) {
                    b_idx += coord * b_strides_padded[dim];
                }
            }
            
            r_data[idx] = a_data[a_idx] * b_data[b_idx];
        }
    }
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<MulBackward>(a, b);
        grad_fn->add_next_edge(a, 0);
        grad_fn->add_next_edge(b, 1);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

inline TensorPtr subtract(const TensorPtr& a, const TensorPtr& b) {
    return add(a, multiply(b, Tensor::full(b->shape(), -1.0f)));
}

inline TensorPtr divide(const TensorPtr& a, const TensorPtr& b) {
    // division as multiplication by reciprocal
    auto reciprocal = Tensor::create(b->shape(), b->requires_grad());
    
    float* r_data = reciprocal->data();
    const float* b_data = b->data();
    int64_t size = b->numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        r_data[i] = 1.0f / b_data[i];
    }
    
    return multiply(a, reciprocal);
}

// matrix ops
inline TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw std::runtime_error("Cannot matmul null tensors");
    }
    
    // support for N-D tensors:
    // - 1D x 1D -> scalar (dot product)
    // - 2D x 2D -> 2D (matrix multiplication)
    // - 1D x 2D -> 1D (matrix-vector multiplication)
    // - 2D x 1D -> 1D (matrix-vector multiplication)
    // - ND x ND -> ND (batched matrix multiplication on last 2 dims)
    
    int a_ndim = a->ndim();
    int b_ndim = b->ndim();
    
    if (a_ndim == 0 || b_ndim == 0) {
        throw std::runtime_error("Scalar inputs not supported for matmul");
    }
    
    // handle 1D x 1D -> scalar (dot product)
    if (a_ndim == 1 && b_ndim == 1) {
        if (a->size(0) != b->size(0)) {
            throw std::runtime_error("Vector dimensions incompatible for dot product");
        }
        
        auto result = Tensor::create(Shape({1}), a->requires_grad() || b->requires_grad());
        
        const float* a_data = a->data();
        const float* b_data = b->data();
        float sum = 0.0f;
        int64_t size = a->size(0);
        
        #pragma omp parallel for reduction(+:sum)
        for (int64_t i = 0; i < size; ++i) {
            sum += a_data[i] * b_data[i];
        }
        
        result->data()[0] = sum;
        
        if (result->requires_grad()) {
            auto grad_fn = std::make_shared<MatMulBackward>(a, b);
            grad_fn->add_next_edge(a, 0);
            grad_fn->add_next_edge(b, 1);
            result->set_grad_fn(grad_fn);
        }
        
        return result;
    }
    
    // promote 1D to 2D for matrix-vector ops
    TensorPtr a_2d = a;
    TensorPtr b_2d = b;
    bool a_was_1d = false;
    bool b_was_1d = false;
    
    if (a_ndim == 1) {
        a_2d = a->unsqueeze(0);  // [n] -> [1, n]
        a_was_1d = true;
    }
    if (b_ndim == 1) {
        b_2d = b->unsqueeze(1);  // [n] -> [n, 1]
        b_was_1d = true;
    }
    
    // now the general N-D case!
    auto a_shape = a_2d->shape().dims();
    auto b_shape = b_2d->shape().dims();
    
    // check matrix multiplication compatibility on last 2 dims
    if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    // get output shape
    std::vector<int64_t> out_shape;
    
    // handle broadcasting for batch dims
    size_t batch_dims = std::max(a_shape.size() - 2, b_shape.size() - 2);
    for (size_t i = 0; i < batch_dims; ++i) {
        int64_t a_dim = (i < a_shape.size() - 2) ? a_shape[i] : 1;
        int64_t b_dim = (i < b_shape.size() - 2) ? b_shape[i] : 1;
        
        if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
            throw std::runtime_error("Batch dimensions are not broadcastable");
        }
        
        out_shape.push_back(std::max(a_dim, b_dim));
    }
    
    // add matrix dims
    out_shape.push_back(a_shape[a_shape.size() - 2]);  // m
    out_shape.push_back(b_shape[b_shape.size() - 1]);  // n
    
    // create res tensor
    auto result = Tensor::create(Shape(out_shape), a->requires_grad() || b->requires_grad());
    
    // get matrix dims
    int64_t m = a_shape[a_shape.size() - 2];
    int64_t k = a_shape[a_shape.size() - 1];
    int64_t n = b_shape[b_shape.size() - 1];
    
    // compute batch size and strides
    int64_t batch_size = 1;
    for (size_t i = 0; i < out_shape.size() - 2; ++i) {
        batch_size *= out_shape[i];
    }
    
    // perform batched matrix multiplication
    float* c_data = result->data();
    const float* a_data = a_2d->data();
    const float* b_data = b_2d->data();
    
    #pragma omp parallel for
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        // compute batch indices for broadcasting
        int64_t a_batch_offset = 0;
        int64_t b_batch_offset = 0;
        int64_t temp_batch = batch;
        
        for (int i = batch_dims - 1; i >= 0; --i) {
            int64_t coord = temp_batch % out_shape[i];
            temp_batch /= out_shape[i];
            
            int64_t a_dim = (i < a_shape.size() - 2) ? a_shape[i] : 1;
            int64_t b_dim = (i < b_shape.size() - 2) ? b_shape[i] : 1;
            
            if (a_dim != 1) {
                int64_t a_stride = m * k;
                for (int j = i + 1; j < a_shape.size() - 2; ++j) {
                    a_stride *= a_shape[j];
                }
                a_batch_offset += coord * a_stride;
            }
            
            if (b_dim != 1) {
                int64_t b_stride = k * n;
                for (int j = i + 1; j < b_shape.size() - 2; ++j) {
                    b_stride *= b_shape[j];
                }
                b_batch_offset += coord * b_stride;
            }
        }
        
        // perform matmul for this batch w optimized BLAS
        const float* a_batch = a_data + a_batch_offset;
        const float* b_batch = b_data + b_batch_offset;
        float* c_batch = c_data + batch * m * n;
        
        // optimized GEMM: C = 1.0 * A * B + 0.0 * C
        blas::gemm(false, false, m, n, k, 1.0f, a_batch, k, b_batch, n, 0.0f, c_batch, n);
    }
    
    // remove added dims if inputs were 1D
    if (a_was_1d && !b_was_1d) {
        result = result->squeeze();  // remove first dim [1, n] -> [n]
    } else if (!a_was_1d && b_was_1d) {
        result = result->squeeze();  // remove last dim [..., n, 1] -> [..., n]
    }
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<MatMulBackward>(a, b);
        grad_fn->add_next_edge(a, 0);
        grad_fn->add_next_edge(b, 1);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

// activation functions

// relu grad fn
class ReluBackward : public GradientFunction {
public:
    ReluBackward(const TensorPtr& input) 
        : input_(input->clone()->detach()) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        auto result = Tensor::create(grad->shape());
        
        float* r_data = result->data();
        const float* g_data = grad->data();
        const float* i_data = input_->data();
        int64_t size = grad->numel();
        
        #pragma omp parallel for
        for (int64_t i = 0; i < size; ++i) {
            r_data[i] = i_data[i] > 0 ? g_data[i] : 0.0f;
        }
        
        return {result};
    }
    
    std::string name() const override { return "ReluBackward"; }
    
private:
    TensorPtr input_;
};

inline TensorPtr relu(const TensorPtr& input) {
    if (!input) {
        throw std::runtime_error("Cannot apply relu to null tensor");
    }
    
    auto result = Tensor::create(input->shape(), input->requires_grad());
    
    // forward comp w SIMD
    simd::relu_vec(input->data(), result->data(), input->numel());
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<ReluBackward>(input);
        grad_fn->add_next_edge(input, 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

// sigmoid grad fn
class SigmoidBackward : public GradientFunction {
public:
    SigmoidBackward(const TensorPtr& output) 
        : output_(output->clone()->detach()) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        auto result = Tensor::create(grad->shape());
        
        float* r_data = result->data();
        const float* g_data = grad->data();
        const float* o_data = output_->data();
        int64_t size = grad->numel();
        
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        #pragma omp parallel for
        for (int64_t i = 0; i < size; ++i) {
            float s = o_data[i];
            r_data[i] = g_data[i] * s * (1.0f - s);
        }
        
        return {result};
    }
    
    std::string name() const override { return "SigmoidBackward"; }
    
private:
    TensorPtr output_;
};

inline TensorPtr sigmoid(const TensorPtr& input) {
    if (!input) {
        throw std::runtime_error("Cannot apply sigmoid to null tensor");
    }
    
    auto result = Tensor::create(input->shape(), input->requires_grad());
    
    // forward comp w SIMD
    simd::sigmoid_vec(input->data(), result->data(), input->numel());
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<SigmoidBackward>(result);
        grad_fn->add_next_edge(input, 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

// tanh grad fn
class TanhBackward : public GradientFunction {
public:
    TanhBackward(const TensorPtr& output) 
        : output_(output->clone()->detach()) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        auto result = Tensor::create(grad->shape());
        
        float* r_data = result->data();
        const float* g_data = grad->data();
        const float* o_data = output_->data();
        int64_t size = grad->numel();
        
        // tanh'(x) = 1 - tanh(x)^2
        #pragma omp parallel for
        for (int64_t i = 0; i < size; ++i) {
            float t = o_data[i];
            r_data[i] = g_data[i] * (1.0f - t * t);
        }
        
        return {result};
    }
    
    std::string name() const override { return "TanhBackward"; }
    
private:
    TensorPtr output_;
};

inline TensorPtr tanh(const TensorPtr& input) {
    if (!input) {
        throw std::runtime_error("Cannot apply tanh to null tensor");
    }
    
    auto result = Tensor::create(input->shape(), input->requires_grad());
    
    // forward comp
    float* r_data = result->data();
    const float* i_data = input->data();
    int64_t size = input->numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        r_data[i] = std::tanh(i_data[i]);
    }
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<TanhBackward>(result);
        grad_fn->add_next_edge(input, 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

// sum backward grad fn
class SumBackward : public GradientFunction {
public:
    SumBackward(const Shape& input_shape) : input_shape_(input_shape) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        
        // grad of sum is 1 for all elements
        // so we just broadcast the incoming grad to all positions
        auto result = Tensor::full(input_shape_, grad->data()[0]);
        
        return {result};
    }
    
    std::string name() const override { return "SumBackward"; }
    
private:
    Shape input_shape_;
};

// reduction ops
inline TensorPtr sum(const TensorPtr& input, bool keepdim = false) {
    if (!input) {
        throw std::runtime_error("Cannot sum null tensor");
    }
    
    // for now, sum all elements
    auto result = Tensor::create(Shape({1}), input->requires_grad());
    
    float sum_val = 0.0f;
    const float* i_data = input->data();
    int64_t size = input->numel();
    
    #pragma omp parallel for reduction(+:sum_val)
    for (int64_t i = 0; i < size; ++i) {
        sum_val += i_data[i];
    }
    
    result->data()[0] = sum_val;
    
    // backward if needed
    if (result->requires_grad()) {
        auto grad_fn = std::make_shared<SumBackward>(input->shape());
        grad_fn->add_next_edge(input, 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

inline TensorPtr mean(const TensorPtr& input, bool keepdim = false) {
    auto sum_result = sum(input, keepdim);
    return divide(sum_result, Tensor::full(Shape({1}), static_cast<float>(input->numel())));
}

// loss functions

// MSE loss
inline TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target) {
    if (!input || !target) {
        throw std::runtime_error("Cannot compute MSE with null tensors");
    }
    
    if (input->shape() != target->shape()) {
        throw std::runtime_error("Input and target must have the same shape");
    }
    
    auto diff = subtract(input, target);
    auto squared = multiply(diff, diff);
    return mean(squared);
}

// cross entropy loss
inline TensorPtr cross_entropy_loss(const TensorPtr& logits, const TensorPtr& target) {
    // todo: implement cross entropy with log_softmax
    throw std::runtime_error("cross entropy not yet implemented");
}


} // namespace harrynet

#endif // HARRYNET_OPS_H