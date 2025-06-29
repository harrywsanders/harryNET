#include "harryNET/tensor.h"
#include "harryNET/autograd.h"
#include "harryNET/simd.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace harrynet {

// Random number generation
static std::random_device rd;
static std::mt19937 gen(rd());

// Factory methods implementation
TensorPtr Tensor::randn(const Shape& shape, bool requires_grad) {
    auto tensor = create(shape, requires_grad);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float* data = tensor->data();
    for (int64_t i = 0; i < shape.numel(); ++i) {
        data[i] = dist(gen);
    }
    
    return tensor;
}

TensorPtr Tensor::rand(const Shape& shape, bool requires_grad) {
    auto tensor = create(shape, requires_grad);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float* data = tensor->data();
    for (int64_t i = 0; i < shape.numel(); ++i) {
        data[i] = dist(gen);
    }
    
    return tensor;
}

TensorPtr Tensor::arange(float start, float end, float step, bool requires_grad) {
    int64_t size = static_cast<int64_t>((end - start) / step);
    auto tensor = create(Shape({size}), requires_grad);
    
    float* data = tensor->data();
    for (int64_t i = 0; i < size; ++i) {
        data[i] = start + i * step;
    }
    
    return tensor;
}

// Autograd methods
void Tensor::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
    if (requires_grad) {
        init_autograd();
    }
}

void Tensor::backward(const TensorPtr& gradient) {
    AutogradEngine::backward(shared_from_this(), gradient);
}

void Tensor::zero_grad() {
    if (autograd_meta_ && autograd_meta_->grad_) {
        autograd_meta_->grad_->storage().zero();
    }
}

// View operations
TensorPtr Tensor::view(const Shape& shape) const {
    if (shape.numel() != shape_.numel()) {
        throw std::runtime_error("Cannot view tensor with different number of elements");
    }
    
    // Create new tensor sharing storage
    auto result = std::make_shared<Tensor>(shape, storage_, requires_grad_);
    
    // Setup view backward if needed
    if (requires_grad_) {
        auto grad_fn = std::make_shared<ViewBackward>(shape_);
        grad_fn->add_next_edge(std::const_pointer_cast<Tensor>(shared_from_this()), 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

TensorPtr Tensor::flatten() const {
    return view(Shape({shape_.numel()}));
}

TensorPtr Tensor::squeeze() const {
    return view(shape_.squeeze());
}

TensorPtr Tensor::unsqueeze(int dim) const {
    return view(shape_.unsqueeze(dim));
}

TensorPtr Tensor::transpose(int dim0, int dim1) const {
    // normalize dimensions
    int nd = ndim();
    if (nd == 0) {
        throw std::runtime_error("Cannot transpose scalar (0-d tensor)");
    }
    
    // handle negative dimensions
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;
    
    // check dimension bounds
    if (dim0 < 0 || dim0 >= nd || dim1 < 0 || dim1 >= nd) {
        throw std::runtime_error("Dimension out of range for transpose");
    }
    
    // if transposing same dimension, just return clone
    if (dim0 == dim1) {
        return clone();
    }
    
    // create transposed shape
    std::vector<int64_t> new_dims = shape_.dims();
    std::swap(new_dims[dim0], new_dims[dim1]);
    Shape new_shape(new_dims);
    auto result = create(new_shape, requires_grad_);
    
    // compute strides for source and destination
    auto src_strides = shape_.strides();
    auto dst_strides = new_shape.strides();
    
    // swap strides for transposed dimensions
    std::vector<int64_t> transposed_src_strides = src_strides;
    std::swap(transposed_src_strides[dim0], transposed_src_strides[dim1]);
    
    // copy data in transposed order
    const float* src = data();
    float* dst = result->data();
    int64_t total_size = numel();
    
    #pragma omp parallel for
    for (int64_t dst_idx = 0; dst_idx < total_size; ++dst_idx) {
        // convert destination linear index to coordinates
        int64_t temp_idx = dst_idx;
        int64_t src_idx = 0;
        
        for (int dim = nd - 1; dim >= 0; --dim) {
            int64_t coord = temp_idx % new_dims[dim];
            temp_idx /= new_dims[dim];
            
            // map coordinate through transpose
            int src_dim = dim;
            if (dim == dim0) src_dim = dim1;
            else if (dim == dim1) src_dim = dim0;
            
            src_idx += coord * src_strides[src_dim];
        }
        
        dst[dst_idx] = src[src_idx];
    }
    
    // setup backward if needed
    if (requires_grad_) {
        auto grad_fn = std::make_shared<TransposeBackward>(dim0, dim1);
        grad_fn->add_next_edge(std::const_pointer_cast<Tensor>(shared_from_this()), 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

// memory operations
TensorPtr Tensor::clone() const {
    auto result = create(shape_, requires_grad_);
    std::copy(data(), data() + numel(), result->data());
    
    // setup clone backward if needed
    if (requires_grad_) {
        auto grad_fn = std::make_shared<CloneBackward>();
        grad_fn->add_next_edge(std::const_pointer_cast<Tensor>(shared_from_this()), 0);
        result->set_grad_fn(grad_fn);
    }
    
    return result;
}

TensorPtr Tensor::detach() const {
    // Create new tensor with same data but no gradient
    auto result = create(shape_, false);
    std::copy(data(), data() + numel(), result->data());
    return result;
}

TensorPtr Tensor::contiguous() const {
    // For now, all tensors are contiguous
    return std::const_pointer_cast<Tensor>(shared_from_this());
}

// Inplace operations
Tensor& Tensor::add_(const TensorPtr& other) {
    if (!other) {
        throw std::runtime_error("Cannot add null tensor");
    }
    
    if (shape_ != other->shape()) {
        throw std::runtime_error("Inplace operations require same shape tensors");
    }
    
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    // Perform inplace addition using SIMD
    simd::add_vec(data(), other->data(), data(), numel());
    
    // Bump version for gradient tracking
    bump_version();
    
    return *this;
}

Tensor& Tensor::add_(float scalar) {
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    float* data_ptr = data();
    int64_t size = numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        data_ptr[i] += scalar;
    }
    
    bump_version();
    return *this;
}

Tensor& Tensor::mul_(const TensorPtr& other) {
    if (!other) {
        throw std::runtime_error("Cannot multiply by null tensor");
    }
    
    if (shape_ != other->shape()) {
        throw std::runtime_error("Inplace operations require same shape tensors");
    }
    
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    // Perform inplace multiplication using SIMD
    simd::multiply_vec(data(), other->data(), data(), numel());
    
    bump_version();
    return *this;
}

Tensor& Tensor::mul_(float scalar) {
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    float* data_ptr = data();
    int64_t size = numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        data_ptr[i] *= scalar;
    }
    
    bump_version();
    return *this;
}

Tensor& Tensor::sub_(const TensorPtr& other) {
    if (!other) {
        throw std::runtime_error("Cannot subtract null tensor");
    }
    
    if (shape_ != other->shape()) {
        throw std::runtime_error("Inplace operations require same shape tensors");
    }
    
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    // Perform inplace subtraction using SIMD
    simd::subtract_vec(data(), other->data(), data(), numel());
    
    bump_version();
    return *this;
}

Tensor& Tensor::sub_(float scalar) {
    return add_(-scalar);
}

Tensor& Tensor::div_(const TensorPtr& other) {
    if (!other) {
        throw std::runtime_error("Cannot divide by null tensor");
    }
    
    if (shape_ != other->shape()) {
        throw std::runtime_error("Inplace operations require same shape tensors");
    }
    
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    // Perform inplace division using SIMD
    simd::divide_vec(data(), other->data(), data(), numel());
    
    bump_version();
    return *this;
}

Tensor& Tensor::div_(float scalar) {
    if (scalar == 0.0f) {
        throw std::runtime_error("Division by zero");
    }
    return mul_(1.0f / scalar);
}

Tensor& Tensor::zero_() {
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    storage_.zero();
    bump_version();
    return *this;
}

Tensor& Tensor::fill_(float value) {
    if (requires_grad_ && is_leaf()) {
        throw std::runtime_error("Cannot perform inplace operation on leaf tensor that requires grad");
    }
    
    storage_.fill(value);
    bump_version();
    return *this;
}

// Utility methods
std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_ << ", requires_grad=" << requires_grad_;
    
    if (grad_fn()) {
        oss << ", grad_fn=" << grad_fn()->name();
    }
    
    oss << ")";
    return oss.str();
}

void Tensor::print() const {
    std::cout << "Tensor(" << std::endl;
    
    // Print data based on dimensions
    if (ndim() == 0) {
        // Scalar
        std::cout << data()[0] << std::endl;
    } else if (ndim() == 1) {
        // Vector
        std::cout << "[";
        for (int64_t i = 0; i < size(0); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::setw(8) << std::setprecision(4) << data()[i];
        }
        std::cout << "]" << std::endl;
    } else if (ndim() == 2) {
        // Matrix
        for (int64_t i = 0; i < size(0); ++i) {
            std::cout << "[";
            for (int64_t j = 0; j < size(1); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::setw(8) << std::setprecision(4) 
                         << data()[i * size(1) + j];
            }
            std::cout << "]" << std::endl;
        }
    } else {
        // Higher dimensional - just print shape and summary
        std::cout << "  Shape: " << shape_ << std::endl;
        std::cout << "  Min: " << *std::min_element(data(), data() + numel()) << std::endl;
        std::cout << "  Max: " << *std::max_element(data(), data() + numel()) << std::endl;
        
        float sum = 0.0f;
        for (int64_t i = 0; i < numel(); ++i) {
            sum += data()[i];
        }
        std::cout << "  Mean: " << sum / numel() << std::endl;
    }
    
    std::cout << "  requires_grad=" << requires_grad_;
    if (grad_fn()) {
        std::cout << ", grad_fn=" << grad_fn()->name();
    }
    std::cout << ")" << std::endl;
}

} // namespace harrynet