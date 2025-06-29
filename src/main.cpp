#include "harryNET/tensor.h"
#include "harryNET/ops.h"
#include "harryNET/module.h"
#include "harryNET/optimizer.h"
#include <iostream>
#include <iomanip>

using namespace harrynet;

int main() {
    std::cout << "=== HarryNET Deep Learning Library ===" << std::endl;
    std::cout << "A clean, modern C++ implementation with automatic differentiation\n" << std::endl;

    // Create some tensors
    std::cout << "1. Creating tensors:" << std::endl;
    auto x = Tensor::randn(Shape({2, 3}), true);
    auto y = Tensor::randn(Shape({3, 4}), true);
    
    std::cout << "x = " << x->to_string() << std::endl;
    std::cout << "y = " << y->to_string() << std::endl;
    
    // Perform operations
    std::cout << "\n2. Matrix multiplication:" << std::endl;
    auto z = matmul(x, y);
    std::cout << "z = x @ y" << std::endl;
    z->print();
    
    // Compute loss
    std::cout << "\n3. Computing loss:" << std::endl;
    auto target = Tensor::randn(Shape({2, 4}), false);
    auto loss = mse_loss(z, target);
    std::cout << "loss = MSE(z, target) = " << loss->data()[0] << std::endl;
    
    // Backward pass
    std::cout << "\n4. Backward pass:" << std::endl;
    loss->backward();
    std::cout << "Gradients computed!" << std::endl;
    
    // Create a simple neural network
    std::cout << "\n5. Creating a neural network:" << std::endl;
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 64));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(64, 32));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(32, 1));
    
    std::cout << "Model created with " << model->size() << " layers" << std::endl;
    
    // Create optimizer
    auto params = model->parameter_list();
    Adam optimizer(params, 0.001f);
    std::cout << "Adam optimizer initialized with " << params.size() << " parameters" << std::endl;
    
    // Mini training loop
    std::cout << "\n6. Training for 10 iterations:" << std::endl;
    auto input = Tensor::randn(Shape({8, 10}), false);  // Batch of 8
    auto labels = Tensor::randn(Shape({8, 1}), false);
    
    for (int i = 0; i < 10; ++i) {
        // Forward pass
        auto output = model->forward(input);
        auto loss_val = mse_loss(output, labels);
        
        // Backward pass
        optimizer.zero_grad();
        loss_val->backward();
        
        // Update weights
        optimizer.step();
        
        std::cout << "Iteration " << std::setw(2) << i + 1 
                  << ", Loss: " << std::setprecision(6) << loss_val->data()[0] << std::endl;
    }
    
    std::cout << "\nHarryNET is ready for deep learning!" << std::endl;
    std::cout << "\nKey features:" << std::endl;
    std::cout << "- Automatic differentiation with correct topological sorting" << std::endl;
    std::cout << "- Memory-safe design with smart pointers" << std::endl;
    std::cout << "- PyTorch-like API for ease of use" << std::endl;
    std::cout << "- Optimized for performance with aligned memory and SIMD support" << std::endl;
    
    return 0;
}