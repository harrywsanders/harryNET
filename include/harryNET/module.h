#ifndef HARRYNET_MODULE_H
#define HARRYNET_MODULE_H

#include "tensor.h"
#include "ops.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace harrynet {


// param container
using ParameterDict = std::unordered_map<std::string, TensorPtr>;
using ModuleDict = std::unordered_map<std::string, std::shared_ptr<class Module>>;

// base class for all nn modules
class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    // forward pass - must be implemented by subclasses
    virtual TensorPtr forward(const TensorPtr& input) = 0;
    
    // training mode management
    void train(bool mode = true) {
        training_ = mode;
        for (auto& [name, module] : modules_) {
            module->train(mode);
        }
    }
    
    void eval() { train(false); }
    bool training() const { return training_; }
    
    // param access
    ParameterDict parameters(bool recurse = true) const {
        ParameterDict params = parameters_;
        
        if (recurse) {
            for (const auto& [prefix, module] : modules_) {
                auto sub_params = module->parameters(true);
                for (const auto& [name, param] : sub_params) {
                    params[prefix + "." + name] = param;
                }
            }
        }
        
        return params;
    }
    
    // named params for optimizer
    std::vector<TensorPtr> parameter_list(bool recurse = true) const {
        std::vector<TensorPtr> params;
        auto param_dict = parameters(recurse);
        
        for (const auto& [name, param] : param_dict) {
            params.push_back(param);
        }
        
        return params;
    }
    
    // zero grads
    void zero_grad() {
        for (auto& [name, param] : parameters_) {
            if (param) {
                param->zero_grad();
            }
        }
        
        for (auto& [name, module] : modules_) {
            module->zero_grad();
        }
    }
    
    // state dict for serialization
    std::unordered_map<std::string, TensorPtr> state_dict() const {
        return parameters(true);
    }
    
    void load_state_dict(const std::unordered_map<std::string, TensorPtr>& state) {
        for (const auto& [name, tensor] : state) {
            // find param and copy data
            auto params = parameters(true);
            auto it = params.find(name);
            if (it != params.end()) {
                // copy data in-place to preserve autograd graph
                std::copy(tensor->data(), 
                         tensor->data() + tensor->numel(),
                         it->second->data());
            }
        }
    }
    
protected:
    // register a param
    void register_parameter(const std::string& name, const TensorPtr& param) {
        if (param) {
            param->set_requires_grad(true);
        }
        parameters_[name] = param;
    }
    
    // register a submodule
    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        modules_[name] = module;
    }
    
    bool training_ = true;
    ParameterDict parameters_;
    ModuleDict modules_;
};

// linear (fully connected) layer
class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
        
        // init weight with Kaiming uniform
        float k = std::sqrt(1.0f / in_features);
        weight_ = Tensor::create(Shape({out_features, in_features}), true);
        
        // random init
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-k, k);
        
        float* w_data = weight_->data();
        for (int64_t i = 0; i < weight_->numel(); ++i) {
            w_data[i] = dist(gen);
        }
        
        register_parameter("weight", weight_);
        
        if (use_bias_) {
            bias_ = Tensor::create(Shape({out_features}), true);
            float* b_data = bias_->data();
            for (int64_t i = 0; i < out_features; ++i) {
                b_data[i] = dist(gen);
            }
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& input) override {
        // support N-dimensional inputs where last dim is in_features
        // input shape: [..., in_features] -> output shape: [..., out_features]
        
        if (input->ndim() == 0) {
            throw std::runtime_error("Linear layer cannot process scalar input");
        }
        
        if (input->size(input->ndim() - 1) != in_features_) {
            throw std::runtime_error("Input size mismatch. Expected last dimension to be " + 
                                   std::to_string(in_features_) + " but got " + 
                                   std::to_string(input->size(input->ndim() - 1)));
        }
        
        // for inputs with > 2 dims, we need to reshape
        TensorPtr reshaped_input = input;
        std::vector<int64_t> original_shape = input->shape().dims();
        bool needs_reshape = input->ndim() > 2;
        
        if (needs_reshape) {
            // flatten all dims except the last one
            int64_t batch_size = 1;
            for (int i = 0; i < input->ndim() - 1; ++i) {
                batch_size *= original_shape[i];
            }
            reshaped_input = input->view(Shape({batch_size, in_features_}));
        }
        
        // compute output = input @ weight^T + bias
        // weight is stored as [out_features, in_features], so we transpose it
        auto output = matmul(reshaped_input, weight_->transpose(0, 1));
        
        if (use_bias_) {
            // broadcast bias across all batch dims
            output = add(output, bias_);
        }
        
        // reshape output back to original batch dims if needed
        if (needs_reshape) {
            std::vector<int64_t> output_shape(original_shape.begin(), original_shape.end() - 1);
            output_shape.push_back(out_features_);
            output = output->view(Shape(output_shape));
        }
        
        return output;
    }
    
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    
private:
    int64_t in_features_;
    int64_t out_features_;
    bool use_bias_;
    TensorPtr weight_;
    TensorPtr bias_;
};

// ReLU activation module 
class ReLU : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override {
        return relu(input);
    }
};

// Sigmoid activation module
class Sigmoid : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override {
        return sigmoid(input);
    }
};

// Tanh activation module  
class Tanh : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override {
        return tanh(input);
    }
};

// Dropout 
class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5) : p_(p) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Dropout probability must be in [0, 1]");
        }
    }
    
    TensorPtr forward(const TensorPtr& input) override {
        if (!training_ || p_ == 0.0) {
            return input;
        }
        
        // create dropout mask
        auto mask = Tensor::create(input->shape(), false);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        float* m_data = mask->data();
        float scale = 1.0f / (1.0f - p_);
        
        for (int64_t i = 0; i < mask->numel(); ++i) {
            m_data[i] = dist(gen) ? scale : 0.0f;
        }
        
        return multiply(input, mask);
    }
    
private:
    float p_;
};

// sequential container
class Sequential : public Module {
public:
    Sequential() = default;
    
    // add modules
    void add(std::shared_ptr<Module> module) {
        std::string name = "layer" + std::to_string(layers_.size());
        register_module(name, module);
        layers_.push_back(module);
    }
    
    // forward through all layers
    TensorPtr forward(const TensorPtr& input) override {
        TensorPtr output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }
    
        // access layers
    size_t size() const { return layers_.size(); }
    std::shared_ptr<Module> operator[](size_t idx) { return layers_[idx]; }
    
private:
    std::vector<std::shared_ptr<Module>> layers_;
};


} // namespace harrynet

#endif // HARRYNET_MODULE_H