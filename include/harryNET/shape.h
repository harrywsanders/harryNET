#ifndef HARRYNET_SHAPE_H
#define HARRYNET_SHAPE_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace harrynet {


class Shape {
public:
    Shape() = default;
    
    explicit Shape(std::initializer_list<int64_t> dims) 
        : dims_(dims) {
        validate();
    }
    
    explicit Shape(std::vector<int64_t> dims) 
        : dims_(std::move(dims)) {
        validate();
    }
    
    // get total numel
    int64_t numel() const {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), 
                              int64_t(1), std::multiplies<int64_t>());
    }
    
    // get number of dims
    size_t ndim() const { return dims_.size(); }
    
    // check if shape is empty
    bool empty() const { return dims_.empty(); }
    
    // access dim
    int64_t operator[](size_t idx) const {
        if (idx >= dims_.size()) {
            throw std::out_of_range("Shape index out of range");
        }
        return dims_[idx];
    }
    
    int64_t& operator[](size_t idx) {
        if (idx >= dims_.size()) {
            throw std::out_of_range("Shape index out of range");
        }
        return dims_[idx];
    }
    
    // get dims
    const std::vector<int64_t>& dims() const { return dims_; }
    
    // comparison operators
    bool operator==(const Shape& other) const {
        return dims_ == other.dims_;
    }
    
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
    
    // string repr
    std::string to_string() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << dims_[i];
        }
        oss << ")";
        return oss.str();
    }
    
    // compute strides for row-major
    std::vector<int64_t> strides() const {
        if (dims_.empty()) return {};
        
        std::vector<int64_t> result(dims_.size());
        int64_t stride = 1;
        
        for (int i = dims_.size() - 1; i >= 0; --i) {
            result[i] = stride;
            stride *= dims_[i];
        }
        
        return result;
    }
    
    // check if shapes can be broadcasted
    static bool are_broadcastable(const Shape& a, const Shape& b) {
        size_t ndim = std::max(a.ndim(), b.ndim());
        
        for (size_t i = 0; i < ndim; ++i) {
            int64_t dim_a = (i < a.ndim()) ? a.dims_[a.ndim() - 1 - i] : 1;
            int64_t dim_b = (i < b.ndim()) ? b.dims_[b.ndim() - 1 - i] : 1;
            
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }
        
        return true;
    }
    
    // compute broadcast shape
    static Shape broadcast_shape(const Shape& a, const Shape& b) {
        if (!are_broadcastable(a, b)) {
            throw std::runtime_error("Shapes " + a.to_string() + " and " + 
                                   b.to_string() + " are not broadcastable");
        }
        
        size_t ndim = std::max(a.ndim(), b.ndim());
        std::vector<int64_t> result(ndim);
        
        for (size_t i = 0; i < ndim; ++i) {
            int64_t dim_a = (i < a.ndim()) ? a.dims_[a.ndim() - 1 - i] : 1;
            int64_t dim_b = (i < b.ndim()) ? b.dims_[b.ndim() - 1 - i] : 1;
            result[ndim - 1 - i] = std::max(dim_a, dim_b);
        }
        
        return Shape(result);
    }
    
    // reshape (returns new shape, validates compatibility)
    Shape reshape(std::initializer_list<int64_t> new_dims) const {
        return reshape(std::vector<int64_t>(new_dims));
    }
    
    Shape reshape(std::vector<int64_t> new_dims) const {
        int64_t infer_idx = -1;
        int64_t known_size = 1;
        
        // handle -1 dim (infer)
        for (size_t i = 0; i < new_dims.size(); ++i) {
            if (new_dims[i] == -1) {
                if (infer_idx >= 0) {
                    throw std::runtime_error("Can only infer one dimension");
                }
                infer_idx = i;
            } else if (new_dims[i] <= 0) {
                throw std::runtime_error("Invalid dimension size");
            } else {
                known_size *= new_dims[i];
            }
        }
        
        // infer dim if needed
        if (infer_idx >= 0) {
            int64_t total_size = numel();
            if (total_size % known_size != 0) {
                throw std::runtime_error("Cannot reshape " + to_string() + 
                                       " to requested shape");
            }
            new_dims[infer_idx] = total_size / known_size;
        }
        
        Shape result(new_dims);
        if (result.numel() != numel()) {
            throw std::runtime_error("Cannot reshape " + to_string() + 
                                   " to " + result.to_string());
        }
        
        return result;
    }
    
    // squeeze (remove dims of size 1)
    Shape squeeze() const {
        std::vector<int64_t> new_dims;
        for (auto dim : dims_) {
            if (dim != 1) {
                new_dims.push_back(dim);
            }
        }
        return Shape(new_dims);
    }
    
    // unsqueeze (add dim of size 1)
    Shape unsqueeze(int axis) const {
        int ndim = dims_.size();
        if (axis < -ndim - 1 || axis > ndim) {
            throw std::out_of_range("Axis out of range");
        }
        
        if (axis < 0) {
            axis += ndim + 1;
        }
        
        std::vector<int64_t> new_dims = dims_;
        new_dims.insert(new_dims.begin() + axis, 1);
        return Shape(new_dims);
    }
    
private:
    std::vector<int64_t> dims_;
    
    void validate() {
        for (auto dim : dims_) {
            if (dim <= 0) {
                throw std::invalid_argument("Shape dimensions must be positive");
            }
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.to_string();
    return os;
}


} // namespace harrynet

#endif // HARRYNET_SHAPE_H