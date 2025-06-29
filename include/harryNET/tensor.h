#ifndef HARRYNET_TENSOR_H
#define HARRYNET_TENSOR_H

#include "storage.h"
#include "shape.h"
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <string>

namespace harrynet {

// forward declarations
class Tensor;
class GradientFunction;
class AutogradMeta;

using TensorPtr = std::shared_ptr<Tensor>;
using TensorList = std::vector<TensorPtr>;
using GradFnPtr = std::shared_ptr<GradientFunction>;

// edge in computation graph
struct Edge {
    TensorPtr tensor;
    uint32_t input_nr;  // which input of the gradient function
    
    Edge(TensorPtr t, uint32_t nr) : tensor(t), input_nr(nr) {}
};

// autograd metadata attached to tensors
class AutogradMeta {
public:
    AutogradMeta() : version_(0) {}
    
    // gradient function that created this tensor
    GradFnPtr grad_fn_;
    
    // gradient accumulator
    TensorPtr grad_;
    
    // version counter for detecting in-place modifications
    std::atomic<uint32_t> version_;
    
    // whether gradient is required
    bool requires_grad_ = false;
    
    // whether this is a leaf tensor (created by user)
    bool is_leaf_ = true;
    
    // output number if this tensor is output of a GradFn
    uint32_t output_nr_ = 0;
    
    void increment_version() {
        version_.fetch_add(1, std::memory_order_relaxed);
    }
};

// base class for gradient functions
class GradientFunction : public std::enable_shared_from_this<GradientFunction> {
public:
    virtual ~GradientFunction() = default;
    
    // apply gradient function
    virtual TensorList apply(const TensorList& grad_outputs) = 0;
    
    // get function name for debugging
    virtual std::string name() const = 0;
    
    // input edges
    std::vector<Edge> next_edges_;
    
    // add next edge
    void add_next_edge(const TensorPtr& tensor, uint32_t input_nr) {
        next_edges_.emplace_back(tensor, input_nr);
    }
};

// main tensor class
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // constructors
    Tensor() = default;
    
    Tensor(const Shape& shape, bool requires_grad = false)
        : shape_(shape), 
          storage_(shape.numel()),
          requires_grad_(requires_grad) {
        if (requires_grad) {
            init_autograd();
        }
    }
    
    Tensor(const Shape& shape, const Storage& storage, bool requires_grad = false)
        : shape_(shape), 
          storage_(storage),
          requires_grad_(requires_grad) {
        if (shape.numel() != storage.size()) {
            throw std::runtime_error("Shape and storage size mismatch");
        }
        if (requires_grad) {
            init_autograd();
        }
    }
    
    // factory methods
    static TensorPtr create(const Shape& shape, bool requires_grad = false);
    static TensorPtr zeros(const Shape& shape, bool requires_grad = false);
    static TensorPtr ones(const Shape& shape, bool requires_grad = false);
    static TensorPtr full(const Shape& shape, float value, bool requires_grad = false);
    static TensorPtr randn(const Shape& shape, bool requires_grad = false);
    static TensorPtr rand(const Shape& shape, bool requires_grad = false);
    static TensorPtr arange(float start, float end, float step = 1.0f, bool requires_grad = false);
    
    // shape and storage access
    const Shape& shape() const { return shape_; }
    const Storage& storage() const { return storage_; }
    Storage& storage() { return storage_; }
    
    int64_t numel() const { return shape_.numel(); }
    size_t ndim() const { return shape_.ndim(); }
    int64_t size(int dim) const { return shape_[dim]; }
    
    // data access
    float* data() { return storage_.data(); }
    const float* data() const { return storage_.data(); }
    
    float& operator[](int64_t idx) { return storage_[idx]; }
    const float& operator[](int64_t idx) const { return storage_[idx]; }
    
    // autograd properties
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad);
    
    bool is_leaf() const { 
        return !autograd_meta_ || autograd_meta_->is_leaf_; 
    }
    
    TensorPtr grad() const {
        return autograd_meta_ ? autograd_meta_->grad_ : nullptr;
    }
    
    void set_grad(const TensorPtr& grad) {
        if (!autograd_meta_) {
            init_autograd();
        }
        autograd_meta_->grad_ = grad;
    }
    
    GradFnPtr grad_fn() const {
        return autograd_meta_ ? autograd_meta_->grad_fn_ : nullptr;
    }
    
    void set_grad_fn(const GradFnPtr& grad_fn) {
        if (!autograd_meta_) {
            init_autograd();
        }
        autograd_meta_->grad_fn_ = grad_fn;
        autograd_meta_->is_leaf_ = false;
    }
    
    // version tracking for in-place ops
    uint32_t version() const {
        return autograd_meta_ ? autograd_meta_->version_.load() : 0;
    }
    
    void bump_version() {
        if (autograd_meta_) {
            autograd_meta_->increment_version();
        }
    }
    
    // gradient ops
    void backward(const TensorPtr& gradient = nullptr);
    void zero_grad();
    
    // view ops (share storage)
    TensorPtr view(const Shape& shape) const;
    TensorPtr reshape(const Shape& shape) const { return view(shape); }
    TensorPtr flatten() const;
    TensorPtr squeeze() const;
    TensorPtr unsqueeze(int dim) const;
    TensorPtr transpose(int dim0, int dim1) const;
    
    // memory ops
    TensorPtr clone() const;
    TensorPtr detach() const;
    TensorPtr contiguous() const;
    
    // inplace ops
    Tensor& add_(const TensorPtr& other);
    Tensor& add_(float scalar);
    Tensor& mul_(const TensorPtr& other);
    Tensor& mul_(float scalar);
    Tensor& sub_(const TensorPtr& other);
    Tensor& sub_(float scalar);
    Tensor& div_(const TensorPtr& other);
    Tensor& div_(float scalar);
    Tensor& zero_();
    Tensor& fill_(float value);
    
    // utility
    std::string to_string() const;
    void print() const;
    
private:
    Shape shape_;
    Storage storage_;
    bool requires_grad_ = false;
    std::unique_ptr<AutogradMeta> autograd_meta_;
    
    void init_autograd() {
        if (!autograd_meta_) {
            autograd_meta_ = std::make_unique<AutogradMeta>();
            autograd_meta_->requires_grad_ = requires_grad_;
        }
    }
    
    // private constructor for creating result tensors
    Tensor(const Shape& shape, Storage&& storage, bool requires_grad)
        : shape_(shape), 
          storage_(std::move(storage)),
          requires_grad_(requires_grad) {
        if (requires_grad) {
            init_autograd();
        }
    }
    
    friend class TensorFactory;
};

// factory impls
inline TensorPtr Tensor::create(const Shape& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

inline TensorPtr Tensor::zeros(const Shape& shape, bool requires_grad) {
    auto tensor = create(shape, requires_grad);
    tensor->storage().zero();
    return tensor;
}

inline TensorPtr Tensor::ones(const Shape& shape, bool requires_grad) {
    auto tensor = create(shape, requires_grad);
    tensor->storage().fill(1.0f);
    return tensor;
}

inline TensorPtr Tensor::full(const Shape& shape, float value, bool requires_grad) {
    auto tensor = create(shape, requires_grad);
    tensor->storage().fill(value);
    return tensor;
}

} // namespace harrynet

#endif // HARRYNET_TENSOR_H