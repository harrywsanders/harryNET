#ifndef HARRYNET_DATASET_H
#define HARRYNET_DATASET_H

#include "tensor.h"
#include <vector>
#include <utility>
#include <memory>
#include <exception>
#include <random>
#include <algorithm>
#include <numeric>

namespace harrynet {

// sample type - pair of input and target tensors
using Sample = std::pair<TensorPtr, TensorPtr>;

// abstract base class for datasets
class Dataset {
public:
    virtual ~Dataset() = default;
    
    // get a single sample by index
    virtual Sample get_item(size_t index) const = 0;
    
    // get total number of samples in dataset
    virtual size_t size() const = 0;
    
    // optional: get batch of samples (default implementation)
    virtual std::vector<Sample> get_batch(const std::vector<size_t>& indices) const {
        std::vector<Sample> batch;
        batch.reserve(indices.size());
        for (size_t idx : indices) {
            batch.push_back(get_item(idx));
        }
        return batch;
    }
    
    // check if index is valid
    void check_index(size_t index) const {
        if (index >= size()) {
            throw std::out_of_range("Dataset index " + std::to_string(index) + 
                                  " out of range [0, " + std::to_string(size()) + ")");
        }
    }
};

// dataset that holds tensors in memory
class TensorDataset : public Dataset {
public:
    TensorDataset(const TensorPtr& data, const TensorPtr& targets)
        : data_(data), targets_(targets) {
        if (data_->size(0) != targets_->size(0)) {
            throw std::invalid_argument("Data and targets must have same number of samples");
        }
    }
    
    TensorDataset(const std::vector<TensorPtr>& data_list, 
                  const std::vector<TensorPtr>& target_list) {
        if (data_list.size() != target_list.size()) {
            throw std::invalid_argument("Data and target lists must have same length");
        }
        if (data_list.empty()) {
            throw std::invalid_argument("Dataset cannot be empty");
        }
        
        // stack tensors along first dimension
        data_ = stack_tensors(data_list);
        targets_ = stack_tensors(target_list);
    }
    
    Sample get_item(size_t index) const override {
        check_index(index);
        
        // extract single sample from batched tensors
        auto data_shape = data_->shape();
        auto target_shape = targets_->shape();
        
        // create shape for single sample (remove batch dimension)
        std::vector<int64_t> single_data_dims(data_shape.dims().begin() + 1, data_shape.dims().end());
        std::vector<int64_t> single_target_dims(target_shape.dims().begin() + 1, target_shape.dims().end());
        Shape single_data_shape(single_data_dims);
        Shape single_target_shape(single_target_dims);
        
        // calculate offsets
        int64_t data_offset = index * single_data_shape.numel();
        int64_t target_offset = index * single_target_shape.numel();
        
        // create new tensors for single samples
        auto data_sample = Tensor::create(single_data_shape);
        auto target_sample = Tensor::create(single_target_shape);
        
        // copy data
        std::copy(data_->data() + data_offset,
                  data_->data() + data_offset + single_data_shape.numel(),
                  data_sample->data());
                  
        std::copy(targets_->data() + target_offset,
                  targets_->data() + target_offset + single_target_shape.numel(),
                  target_sample->data());
        
        return {data_sample, target_sample};
    }
    
    size_t size() const override {
        return static_cast<size_t>(data_->size(0));
    }
    
    // get all data
    TensorPtr get_data() const { return data_; }
    TensorPtr get_targets() const { return targets_; }
    
private:
    TensorPtr data_;
    TensorPtr targets_;
    
    // helper to stack list of tensors
    TensorPtr stack_tensors(const std::vector<TensorPtr>& tensors) {
        if (tensors.empty()) return nullptr;
        
        // get shape of first tensor and add batch dimension
        auto first_shape = tensors[0]->shape();
        std::vector<int64_t> stacked_dims = {static_cast<int64_t>(tensors.size())};
        stacked_dims.insert(stacked_dims.end(), 
                           first_shape.dims().begin(), 
                           first_shape.dims().end());
        
        Shape stacked_shape(stacked_dims);
        auto result = Tensor::create(stacked_shape);
        
        // copy each tensor into result
        int64_t offset = 0;
        int64_t sample_size = first_shape.numel();
        
        for (const auto& tensor : tensors) {
            if (tensor->shape() != first_shape) {
                throw std::invalid_argument("All tensors must have same shape");
            }
            std::copy(tensor->data(), 
                     tensor->data() + sample_size,
                     result->data() + offset);
            offset += sample_size;
        }
        
        return result;
    }
};

// dataset wrapper for applying transforms
template<typename TransformFunc>
class TransformDataset : public Dataset {
public:
    TransformDataset(std::shared_ptr<Dataset> dataset, TransformFunc transform)
        : dataset_(dataset), transform_(transform) {}
    
    Sample get_item(size_t index) const override {
        auto sample = dataset_->get_item(index);
        return transform_(sample);
    }
    
    size_t size() const override {
        return dataset_->size();
    }
    
private:
    std::shared_ptr<Dataset> dataset_;
    TransformFunc transform_;
};

// helper function to create transform dataset
template<typename TransformFunc>
std::shared_ptr<TransformDataset<TransformFunc>> 
make_transform_dataset(std::shared_ptr<Dataset> dataset, TransformFunc transform) {
    return std::make_shared<TransformDataset<TransformFunc>>(dataset, transform);
}

// subset dataset for train/val/test splits
class SubsetDataset : public Dataset {
public:
    SubsetDataset(std::shared_ptr<Dataset> dataset, 
                  const std::vector<size_t>& indices)
        : dataset_(dataset), indices_(indices) {
        for (size_t idx : indices) {
            if (idx >= dataset->size()) {
                throw std::out_of_range("Subset index out of range");
            }
        }
    }
    
    Sample get_item(size_t index) const override {
        check_index(index);
        return dataset_->get_item(indices_[index]);
    }
    
    size_t size() const override {
        return indices_.size();
    }
    
private:
    std::shared_ptr<Dataset> dataset_;
    std::vector<size_t> indices_;
};

// random split utility
inline std::vector<std::shared_ptr<Dataset>> 
random_split(std::shared_ptr<Dataset> dataset,
             const std::vector<double>& ratios,
             unsigned int seed = 42) {
    size_t total_size = dataset->size();
    std::vector<size_t> all_indices(total_size);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // shuffle indices
    std::mt19937 rng(seed);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    
    // split according to ratios
    std::vector<std::shared_ptr<Dataset>> splits;
    size_t start_idx = 0;
    
    for (size_t i = 0; i < ratios.size(); ++i) {
        size_t split_size = (i == ratios.size() - 1) 
            ? total_size - start_idx
            : static_cast<size_t>(total_size * ratios[i]);
            
        std::vector<size_t> split_indices(
            all_indices.begin() + start_idx,
            all_indices.begin() + start_idx + split_size
        );
        
        splits.push_back(std::make_shared<SubsetDataset>(dataset, split_indices));
        start_idx += split_size;
    }
    
    return splits;
}

} // namespace harrynet

#endif // HARRYNET_DATASET_H