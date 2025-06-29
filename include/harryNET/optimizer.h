#ifndef HARRYNET_OPTIMIZER_H
#define HARRYNET_OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace harrynet {


// base optimizer class
class Optimizer {
public:
    explicit Optimizer(const std::vector<TensorPtr>& parameters)
        : parameters_(parameters) {}
    
    virtual ~Optimizer() = default;
    
    // perform optimization step
    virtual void step() = 0;
    
    // zero all grads
    void zero_grad() {
        for (auto& param : parameters_) {
            if (param) {
                param->zero_grad();
            }
        }
    }
    
protected:
    std::vector<TensorPtr> parameters_;
};

// SGD optimizer with momentum
class SGD : public Optimizer {
public:
    SGD(const std::vector<TensorPtr>& parameters, 
        float lr = 0.01f, 
        float momentum = 0.0f,
        float weight_decay = 0.0f)
        : Optimizer(parameters), 
          lr_(lr), 
          momentum_(momentum),
          weight_decay_(weight_decay) {
        
        // init momentum buffers
        if (momentum_ > 0) {
            for (size_t i = 0; i < parameters_.size(); ++i) {
                if (parameters_[i]) {
                    momentum_buffers_.push_back(
                        Tensor::zeros(parameters_[i]->shape(), false)
                    );
                }
            }
        }
    }
    
    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param || !param->grad()) continue;
            
            float* p_data = param->data();
            const float* g_data = param->grad()->data();
            int64_t size = param->numel();
            
            // apply weight decay (L2 regularization)
            if (weight_decay_ > 0) {
                #pragma omp parallel for
                for (int64_t j = 0; j < size; ++j) {
                    const_cast<float*>(g_data)[j] += weight_decay_ * p_data[j];
                }
            }
            
            // apply momentum
            if (momentum_ > 0) {
                float* m_data = momentum_buffers_[i]->data();
                
                #pragma omp parallel for
                for (int64_t j = 0; j < size; ++j) {
                    m_data[j] = momentum_ * m_data[j] + g_data[j];
                    p_data[j] -= lr_ * m_data[j];
                }
            } else {
                // standard SGD update
                #pragma omp parallel for
                for (int64_t j = 0; j < size; ++j) {
                    p_data[j] -= lr_ * g_data[j];
                }
            }
        }
    }
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    
private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<TensorPtr> momentum_buffers_;
};

// adam optimizer
class Adam : public Optimizer {
public:
    Adam(const std::vector<TensorPtr>& parameters,
         float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f,
         float weight_decay = 0.0f)
        : Optimizer(parameters),
          lr_(lr),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps),
          weight_decay_(weight_decay),
          step_count_(0) {
        
        // init moment buffers
        for (const auto& param : parameters_) {
            if (param) {
                m_buffers_.push_back(Tensor::zeros(param->shape(), false));
                v_buffers_.push_back(Tensor::zeros(param->shape(), false));
            }
        }
    }
    
    void step() override {
        step_count_++;
        
        // bias correction
        float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
        float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param || !param->grad()) continue;
            
            float* p_data = param->data();
            const float* g_data = param->grad()->data();
            float* m_data = m_buffers_[i]->data();
            float* v_data = v_buffers_[i]->data();
            int64_t size = param->numel();
            
            #pragma omp parallel for
            for (int64_t j = 0; j < size; ++j) {
                float grad = g_data[j];
                
                // apply weight decay
                if (weight_decay_ > 0) {
                    grad += weight_decay_ * p_data[j];
                }
                
                // update biased first moment estimate
                m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad;
                
                // update biased second raw moment estimate  
                v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * grad * grad;
                
                // compute bias-corrected moments
                float m_hat = m_data[j] / bias_correction1;
                float v_hat = v_data[j] / bias_correction2;
                
                // update params
                p_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_;
    std::vector<TensorPtr> m_buffers_;  // first moment
    std::vector<TensorPtr> v_buffers_;  // second moment
};

// learning rate scheduler base class
class LRScheduler {
public:
    explicit LRScheduler(Optimizer* optimizer) : optimizer_(optimizer) {}
    virtual ~LRScheduler() = default;
    
    virtual void step() = 0;
    
protected:
    Optimizer* optimizer_;
};

// step learning rate scheduler
class StepLR : public LRScheduler {
public:
    StepLR(SGD* optimizer, int step_size, float gamma = 0.1f)
        : LRScheduler(optimizer), 
          sgd_optimizer_(optimizer),
          step_size_(step_size), 
          gamma_(gamma),
          current_step_(0) {}
    
    void step() override {
        current_step_++;
        if (current_step_ % step_size_ == 0) {
            float new_lr = sgd_optimizer_->get_lr() * gamma_;
            sgd_optimizer_->set_lr(new_lr);
        }
    }
    
private:
    SGD* sgd_optimizer_;
    int step_size_;
    float gamma_;
    int current_step_;
};

// grad clipping utils
inline void clip_grad_norm(const std::vector<TensorPtr>& parameters, float max_norm) {
    // calculate total norm
    float total_norm = 0.0f;
    
    for (const auto& param : parameters) {
        if (param && param->grad()) {
            const float* g_data = param->grad()->data();
            int64_t size = param->numel();
            
            float param_norm = 0.0f;
            #pragma omp parallel for reduction(+:param_norm)
            for (int64_t i = 0; i < size; ++i) {
                param_norm += g_data[i] * g_data[i];
            }
            
            total_norm += param_norm;
        }
    }
    
    total_norm = std::sqrt(total_norm);
    
    // clip if necessary
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        
        for (const auto& param : parameters) {
            if (param && param->grad()) {
                float* g_data = param->grad()->data();
                int64_t size = param->numel();
                
                #pragma omp parallel for
                for (int64_t i = 0; i < size; ++i) {
                    g_data[i] *= scale;
                }
            }
        }
    }
}

inline void clip_grad_value(const std::vector<TensorPtr>& parameters, float clip_value) {
    for (const auto& param : parameters) {
        if (param && param->grad()) {
            float* g_data = param->grad()->data();
            int64_t size = param->numel();
            
            #pragma omp parallel for
            for (int64_t i = 0; i < size; ++i) {
                g_data[i] = std::max(-clip_value, std::min(clip_value, g_data[i]));
            }
        }
    }
}


} // namespace harrynet

#endif // HARRYNET_OPTIMIZER_H