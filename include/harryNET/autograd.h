#ifndef HARRYNET_AUTOGRAD_H
#define HARRYNET_AUTOGRAD_H

#include "tensor.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace harrynet {

// forward declaration of matmul from ops.h
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);


// node in the execution graph
struct Node {
    GradFnPtr fn;
    TensorPtr grad;
    int dependencies = 0;
    
    Node(GradFnPtr f) : fn(f) {}
};

// autograd engine for computing gradients
class AutogradEngine {
public:
    // main backward function
    static void backward(const TensorPtr& root, const TensorPtr& grad_output = nullptr) {
        if (!root) {
            throw std::runtime_error("Cannot run backward on null tensor");
        }
        
        if (root->shape().numel() != 1 && !grad_output) {
            throw std::runtime_error("grad_output must be specified for non-scalar outputs");
        }
        
        // init grad
        TensorPtr initial_grad = grad_output;
        if (!initial_grad) {
            initial_grad = Tensor::ones(root->shape(), false);
        }
        
        if (initial_grad->shape() != root->shape()) {
            throw std::runtime_error("grad_output shape must match tensor shape");
        }
        
        // build exec graph
        std::unordered_map<GradientFunction*, Node*> graph;
        std::queue<std::pair<GradFnPtr, TensorPtr>> queue;
        
        // handle root tensor (leaf)
        if (root->grad_fn()) {
            queue.push({root->grad_fn(), initial_grad});
        } else if (root->requires_grad()) {
            // leaf - accumulate grad directly
            accumulate_grad(root, initial_grad);
        }
        
        // topo sort and grad computation (non-leaf)
        while (!queue.empty()) {
            auto [grad_fn, grad] = queue.front();
            queue.pop();
            
            if (!grad_fn) continue;
            
            // get or create node (non-leaf)
            Node* node = nullptr;
            bool is_new_node = false;
            auto it = graph.find(grad_fn.get());
            if (it == graph.end()) {
                node = new Node(grad_fn);
                graph[grad_fn.get()] = node;
                is_new_node = true;
                
                // count dependencies - how many times this grad_fn will be called
                node->dependencies = 1;
            } else {
                node = it->second;
                node->dependencies++;
            }
            
            // accumulate grad (non-leaf)
            if (!node->grad) {
                node->grad = grad->clone();
            } else {
                // in-place add
                add_inplace(node->grad, grad);
            }
            
            // process if all deps are satisfied
            node->dependencies--;
            if (node->dependencies == 0) {
                // apply grad fn (non-leaf)
                TensorList grad_inputs = grad_fn->apply({node->grad});
                
                // propagate to next nodes (non-leaf)   
                for (size_t i = 0; i < grad_fn->next_edges_.size(); ++i) {
                    const auto& edge = grad_fn->next_edges_[i];
                    if (!edge.tensor) continue;
                    
                    if (i < grad_inputs.size() && grad_inputs[i]) {
                        if (edge.tensor->grad_fn()) {
                            queue.push({edge.tensor->grad_fn(), grad_inputs[i]});
                        } else if (edge.tensor->requires_grad()) {
                            // leaf tensor 
                            accumulate_grad(edge.tensor, grad_inputs[i]);
                        }
                    }
                }
            }
        }
        
        // cleanup
        for (auto& [_, node] : graph) {
            delete node;
        }
    }
    
private:
    // accumulate grad to tensor
    static void accumulate_grad(const TensorPtr& tensor, const TensorPtr& grad) {
        if (!tensor->requires_grad()) return;
        
        if (!tensor->grad()) {
            tensor->set_grad(grad->clone());
        } else {
            // in-place add
            add_inplace(tensor->grad(), grad);
        }
    }
    
    // in-place tensor add
    static void add_inplace(const TensorPtr& a, const TensorPtr& b) {
        if (a->shape() != b->shape()) {
            throw std::runtime_error("Cannot add tensors with different shapes");
        }
        
        float* a_data = a->data();
        const float* b_data = b->data();
        int64_t size = a->numel();
        
        #pragma omp parallel for
        for (int64_t i = 0; i < size; ++i) {
            a_data[i] += b_data[i];
        }
    }
};

// grad fn implementations

// add backward
class AddBackward : public GradientFunction {
public:
    AddBackward(const Shape& self_shape, const Shape& other_shape)
        : self_shape_(self_shape), other_shape_(other_shape) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        
        // handle broadcast
        auto self_grad = sum_to_shape(grad, self_shape_);
        auto other_grad = sum_to_shape(grad, other_shape_);
        
        return {self_grad, other_grad};
    }
    
    std::string name() const override { return "AddBackward"; }
    
private:
    Shape self_shape_;
    Shape other_shape_;
    
    TensorPtr sum_to_shape(const TensorPtr& tensor, const Shape& target_shape) {
        if (tensor->shape() == target_shape) {
            return tensor;
        }
        
        // we need to sum over dims that were broadcast
        auto result = tensor->clone();
        auto current_shape = tensor->shape().dims();
        auto target_dims = target_shape.dims();
        
        // first, sum over leading dims if tensor has more dims than target
        while (current_shape.size() > target_dims.size()) {
            // sum over the first dim
            auto new_shape = std::vector<int64_t>(current_shape.begin() + 1, current_shape.end());
            auto temp = Tensor::create(Shape(new_shape));
            
            float* temp_data = temp->data();
            const float* result_data = result->data();
            int64_t outer_size = current_shape[0];
            int64_t inner_size = temp->numel();
            
            // init to zero
            std::fill(temp_data, temp_data + inner_size, 0.0f);
            
            // sum along first dim
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t j = 0; j < inner_size; ++j) {
                    temp_data[j] += result_data[i * inner_size + j];
                }
            }
            
            result = temp;
            current_shape = new_shape;
        }
        
        // now sum over dims where target has size 1 but current doesn't
        for (size_t i = 0; i < target_dims.size(); ++i) {
            if (target_dims[i] == 1 && current_shape[i] != 1) {
                // sum along this dim
                std::vector<int64_t> new_shape = current_shape;
                new_shape[i] = 1;
                
                auto temp = Tensor::create(Shape(new_shape));
                float* temp_data = temp->data();
                const float* result_data = result->data();
                
                // calc strides
                int64_t outer_stride = 1;
                for (size_t j = 0; j < i; ++j) {
                    outer_stride *= current_shape[j];
                }
                int64_t dim_size = current_shape[i];
                int64_t inner_stride = 1;
                for (size_t j = i + 1; j < current_shape.size(); ++j) {
                    inner_stride *= current_shape[j];
                }
                
                // sum along dim i
                #pragma omp parallel for
                for (int64_t outer = 0; outer < outer_stride; ++outer) {
                    for (int64_t inner = 0; inner < inner_stride; ++inner) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < dim_size; ++k) {
                            sum += result_data[outer * dim_size * inner_stride + k * inner_stride + inner];
                        }
                        temp_data[outer * inner_stride + inner] = sum;
                    }
                }
                
                result = temp;
                current_shape = new_shape;
            }
        }
        
        // reshape to remove dims of size 1 if needed
        if (result->shape() != target_shape) {
            result = result->view(target_shape);
        }
        
        return result;
    }
};

// mul backward
class MulBackward : public GradientFunction {
public:
    MulBackward(const TensorPtr& self, const TensorPtr& other)
        : self_(self->clone()->detach()), 
          other_(other->clone()->detach()),
          self_shape_(self->shape()),
          other_shape_(other->shape()) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        
        // grad of mul: d(a*b) = b*da + a*db (non-leaf)
        auto self_grad = multiply(grad, other_);
        auto other_grad = multiply(grad, self_);
        
        // handle broadcast
        self_grad = sum_to_shape(self_grad, self_shape_);
        other_grad = sum_to_shape(other_grad, other_shape_);
        
        return {self_grad, other_grad};
    }
    
    std::string name() const override { return "MulBackward"; }
    
private:
    TensorPtr self_;
    TensorPtr other_;
    Shape self_shape_;
    Shape other_shape_;
    
    TensorPtr multiply(const TensorPtr& a, const TensorPtr& b) {
        // element-wise mul
        auto result = Tensor::create(a->shape());
        
        float* r_data = result->data();
        const float* a_data = a->data();
        const float* b_data = b->data();
        int64_t size = a->numel();
        
        #pragma omp parallel for
        for (int64_t i = 0; i < size; ++i) {
            r_data[i] = a_data[i] * b_data[i];
        }
        
        return result;
    }
    
    TensorPtr sum_to_shape(const TensorPtr& tensor, const Shape& target_shape) {
        if (tensor->shape() == target_shape) {
            return tensor;
        }
        
        // we need to sum over dims that were broadcast
        auto result = tensor->clone();
        auto current_shape = tensor->shape().dims();
        auto target_dims = target_shape.dims();
        
        // first we sum over leading dims if tensor has more dims than target
        while (current_shape.size() > target_dims.size()) {
            // sum over the first dim
            auto new_shape = std::vector<int64_t>(current_shape.begin() + 1, current_shape.end());
            auto temp = Tensor::create(Shape(new_shape));
            
            float* temp_data = temp->data();
            const float* result_data = result->data();
            int64_t outer_size = current_shape[0];
            int64_t inner_size = temp->numel();
            
            // init to zero
            std::fill(temp_data, temp_data + inner_size, 0.0f);
            
            // sum along first dim
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t j = 0; j < inner_size; ++j) {
                    temp_data[j] += result_data[i * inner_size + j];
                }
            }
            
            result = temp;
            current_shape = new_shape;
        }
        
        // now sum over dims where target has size 1 but current doesn't
        for (size_t i = 0; i < target_dims.size(); ++i) {
            if (target_dims[i] == 1 && current_shape[i] != 1) {
                // sum along this dim
                std::vector<int64_t> new_shape = current_shape;
                new_shape[i] = 1;
                
                auto temp = Tensor::create(Shape(new_shape));
                float* temp_data = temp->data();
                const float* result_data = result->data();
                
                // calc strides
                int64_t outer_stride = 1;
                for (size_t j = 0; j < i; ++j) {
                    outer_stride *= current_shape[j];
                }
                int64_t dim_size = current_shape[i];
                int64_t inner_stride = 1;
                for (size_t j = i + 1; j < current_shape.size(); ++j) {
                    inner_stride *= current_shape[j];
                }
                
                // sum along dim i
                #pragma omp parallel for
                for (int64_t outer = 0; outer < outer_stride; ++outer) {
                    for (int64_t inner = 0; inner < inner_stride; ++inner) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < dim_size; ++k) {
                            sum += result_data[outer * dim_size * inner_stride + k * inner_stride + inner];
                        }
                        temp_data[outer * inner_stride + inner] = sum;
                    }
                }
                
                result = temp;
                current_shape = new_shape;
            }
        }
        
        // reshape to remove dims of size 1 if needed
        if (result->shape() != target_shape) {
            result = result->view(target_shape);
        }
        
        return result;
    }
};

// view backward - just reshapes grad back to original shape
class ViewBackward : public GradientFunction {
public:
    ViewBackward(const Shape& original_shape) : original_shape_(original_shape) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        // simply reshape the grad back to the original shape
        return {grad->view(original_shape_)};
    }
    
    std::string name() const override { return "ViewBackward"; }
    
private:
    Shape original_shape_;
};

// transpose backward - transposes grad back
class TransposeBackward : public GradientFunction {
public:
    TransposeBackward(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        // transpose the gradient back
        return {grad->transpose(dim1_, dim0_)};
    }
    
    std::string name() const override { return "TransposeBackward"; }
    
private:
    int dim0_, dim1_;
};

// clone backward - just passes gradient through
class CloneBackward : public GradientFunction {
public:
    TensorList apply(const TensorList& grad_outputs) override {
        // gradient flows through unchanged
        return grad_outputs;
    }
    
    std::string name() const override { return "CloneBackward"; }
};

// matmul backward
class MatMulBackward : public GradientFunction {
public:
    MatMulBackward(const TensorPtr& self, const TensorPtr& other)
        : self_(self->clone()->detach()), 
          other_(other->clone()->detach()) {}
    
    TensorList apply(const TensorList& grad_outputs) override {
        auto grad = grad_outputs[0];
        
        // For C = A @ B:
        // dA = dC @ B^T
        // dB = A^T @ dC
        
        auto self_grad = matmul(grad, transpose(other_));
        auto other_grad = matmul(transpose(self_), grad);
        
        return {self_grad, other_grad};
    }
    
    std::string name() const override { return "MatMulBackward"; }
    
private:
    TensorPtr self_;
    TensorPtr other_;
    
    TensorPtr transpose(const TensorPtr& tensor) {
        // for 2D tensors (most common case in matmul), transpose last two dims
        // for higher dim tensors, this will be handled by the generalized matmul
        if (tensor->ndim() < 2) {
            throw std::runtime_error("transpose requires at least 2D tensor");
        }
        
        return tensor->transpose(-2, -1);
    }
};


} // namespace harrynet

#endif // HARRYNET_AUTOGRAD_H