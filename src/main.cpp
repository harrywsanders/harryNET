#include "harryNET/tensor.h"
#include "harryNET/ops.h"
#include "harryNET/module.h"
#include "harryNET/optimizer.h"
#include "harryNET/dataset.h"
#include "harryNET/dataloader.h"
#include "harryNET/data_utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace harrynet;

// helper function to generate synthetic dataset
std::pair<TensorPtr, TensorPtr> generate_synthetic_data(size_t n_samples, size_t n_features) {
    // generate random input data
    auto X = Tensor::randn(Shape({static_cast<int64_t>(n_samples), static_cast<int64_t>(n_features)}));
    
    // generate labels with some pattern (for demonstration)
    auto y = Tensor::create(Shape({static_cast<int64_t>(n_samples), 1}));
    
    //  linear rel w noise
    for (size_t i = 0; i < n_samples; ++i) {
        float sum = 0;
        for (size_t j = 0; j < n_features; ++j) {
            sum += X->data()[i * n_features + j] * (j + 1) * 0.1f;
        }
        y->data()[i] = sum + 0.1f * (rand() / float(RAND_MAX) - 0.5f);
    }
    
    return {X, y};
}

int main() {
    // 1. gen synthetic dataset
    std::cout << "1. Creating synthetic dataset..." << std::endl;
    size_t n_samples = 1000;
    size_t n_features = 10;
    auto [X, y] = generate_synthetic_data(n_samples, n_features);
    std::cout << "   Generated " << n_samples << " samples with " << n_features << " features each\n" << std::endl;

    // 2. create tensordataset
    std::cout << "2. Creating TensorDataset..." << std::endl;
    auto dataset = std::make_shared<TensorDataset>(X, y);
    std::cout << "   Dataset size: " << dataset->size() << " samples" << std::endl;
    
    // show a sample
    auto [sample_x, sample_y] = dataset->get_item(0);
    std::cout << "   First sample shape - X: " << sample_x->shape().to_string() 
              << ", y: " << sample_y->shape().to_string() << "\n" << std::endl;

    // 3. split dataset into train/val/test
    std::cout << "3. Splitting dataset..." << std::endl;
    auto splits = train_val_test_split(dataset, 0.7, 0.15);
    std::cout << "   Train size: " << splits.train->size() << " samples" << std::endl;
    std::cout << "   Val size: " << splits.val->size() << " samples" << std::endl;
    std::cout << "   Test size: " << splits.test->size() << " samples\n" << std::endl;

    // 4. create dataloader
    std::cout << "4. Creating DataLoaders..." << std::endl;
    size_t batch_size = 32;
    auto train_loader = make_dataloader(splits.train, batch_size, true, 0, false);  // shuffle=true, single-threaded
    auto val_loader = make_dataloader(splits.val, batch_size, false, 0, false);    // shuffle=false
    std::cout << "   Batch size: " << batch_size << std::endl;
    std::cout << "   Train batches: " << train_loader->size() << std::endl;
    std::cout << "   Val batches: " << val_loader->size() << "\n" << std::endl;

    // 5. data normalization
    std::cout << "5. Normalizing data..." << std::endl;
    Normalizer normalizer;
    normalizer.fit(X);
    std::cout << "   Mean: " << normalizer.mean() << ", Std: " << normalizer.std() << "\n" << std::endl;

    // 6. create nn
    std::cout << "6. Creating neural network..." << std::endl;
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(n_features, 64));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(64, 32));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(32, 1));
    std::cout << "   Model created with " << model->size() << " layers\n" << std::endl;

    // 7. setup training
    auto params = model->parameter_list();
    Adam optimizer(params, 0.001f);
    
    // 8. training loop with dataloader
    std::cout << "7. Training with DataLoader..." << std::endl;
    int n_epochs = 5;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int batch_count = 0;
        
        // training loop
        for (auto [batch_x, batch_y] : *train_loader) {
            // normalize batch
            normalizer.normalize_inplace(batch_x);
            
            // fwd pass
            auto output = model->forward(batch_x);
            auto loss = mse_loss(output, batch_y);
            
            // bwd pass
            optimizer.zero_grad();
            loss->backward();
            
            // update weights
            optimizer.step();
            
            epoch_loss += loss->data()[0];
            batch_count++;
        }
        
        // val loop
        float val_loss = 0.0f;
        int val_batch_count = 0;
        
        for (auto [batch_x, batch_y] : *val_loader) {
            // normalize batch
            normalizer.normalize_inplace(batch_x);
            
            // fwd pass (no gradients needed)
            auto output = model->forward(batch_x);
            auto loss = mse_loss(output, batch_y);
            
            val_loss += loss->data()[0];
            val_batch_count++;
        }
        
        std::cout << "   Epoch " << std::setw(2) << epoch + 1 << "/" << n_epochs
                  << " - Train Loss: " << std::fixed << std::setprecision(4) 
                  << epoch_loss / batch_count
                  << ", Val Loss: " << val_loss / val_batch_count << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\n   Training completed in " << duration.count() << "ms\n" << std::endl;

    // 9. test multi-threaded dataloader performance
    std::cout << "8. Testing multi-threaded DataLoader..." << std::endl;
    
    // create multi-threaded loader with 4 workers
    auto parallel_loader = make_dataloader(splits.train, batch_size, true, 4, false);
    
    start_time = std::chrono::high_resolution_clock::now();
    int loaded_batches = 0;
    for (auto [batch_x, batch_y] : *parallel_loader) {
        loaded_batches++;
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "   Loaded " << loaded_batches << " batches with 4 workers in " 
              << duration.count() << "ms\n" << std::endl;

    // 10. demonstrate transforms
    std::cout << "9. Demonstrating data transforms..." << std::endl;
    
    // create a transform that adds noise
    auto noise_transform = [](const Sample& sample) -> Sample {
        auto [x, y] = sample;
        auto noisy_x = x->clone();
        
        // add gaussian noise
        for (int64_t i = 0; i < noisy_x->numel(); ++i) {
            noisy_x->data()[i] += 0.01f * (rand() / float(RAND_MAX) - 0.5f);
        }
        
        return {noisy_x, y};
    };
    
    auto augmented_dataset = make_transform_dataset(splits.train, noise_transform);
    std::cout << "   Created augmented dataset with noise transform\n" << std::endl;
    
    return 0;
}