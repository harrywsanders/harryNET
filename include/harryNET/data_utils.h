#ifndef HARRYNET_DATA_UTILS_H
#define HARRYNET_DATA_UTILS_H

#include "tensor.h"
#include "dataset.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <unordered_map>

namespace harrynet {

// Data normalization utilities
class Normalizer {
public:
    // Compute mean and std from data
    void fit(const TensorPtr& data) {
        if (!data || data->numel() == 0) {
            throw std::invalid_argument("Cannot fit normalizer on empty data");
        }
        
        // Compute mean
        float sum = 0.0f;
        const float* ptr = data->data();
        int64_t n = data->numel();
        
        for (int64_t i = 0; i < n; ++i) {
            sum += ptr[i];
        }
        mean_ = sum / n;
        
        // Compute std
        float var_sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float diff = ptr[i] - mean_;
            var_sum += diff * diff;
        }
        std_ = std::sqrt(var_sum / n);
        
        // Prevent division by zero
        if (std_ < 1e-7f) {
            std_ = 1.0f;
        }
    }
    
    // Normalize data using computed statistics
    TensorPtr transform(const TensorPtr& data) const {
        auto result = data->clone();
        normalize_inplace(result);
        return result;
    }
    
    // Normalize in-place
    void normalize_inplace(TensorPtr& data) const {
        float* ptr = data->data();
        int64_t n = data->numel();
        
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] = (ptr[i] - mean_) / std_;
        }
    }
    
    // Inverse transform
    TensorPtr inverse_transform(const TensorPtr& data) const {
        auto result = data->clone();
        float* ptr = result->data();
        int64_t n = result->numel();
        
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] = ptr[i] * std_ + mean_;
        }
        
        return result;
    }
    
    float mean() const { return mean_; }
    float std() const { return std_; }
    
private:
    float mean_ = 0.0f;
    float std_ = 1.0f;
};

// Min-Max scaler
class MinMaxScaler {
public:
    void fit(const TensorPtr& data, float feature_min = 0.0f, float feature_max = 1.0f) {
        if (!data || data->numel() == 0) {
            throw std::invalid_argument("Cannot fit scaler on empty data");
        }
        
        feature_min_ = feature_min;
        feature_max_ = feature_max;
        
        const float* ptr = data->data();
        int64_t n = data->numel();
        
        data_min_ = *std::min_element(ptr, ptr + n);
        data_max_ = *std::max_element(ptr, ptr + n);
        
        // Prevent division by zero
        if (data_max_ - data_min_ < 1e-7f) {
            data_max_ = data_min_ + 1.0f;
        }
    }
    
    TensorPtr transform(const TensorPtr& data) const {
        auto result = data->clone();
        scale_inplace(result);
        return result;
    }
    
    void scale_inplace(TensorPtr& data) const {
        float* ptr = data->data();
        int64_t n = data->numel();
        float scale = (feature_max_ - feature_min_) / (data_max_ - data_min_);
        
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] = feature_min_ + (ptr[i] - data_min_) * scale;
        }
    }
    
    TensorPtr inverse_transform(const TensorPtr& data) const {
        auto result = data->clone();
        float* ptr = result->data();
        int64_t n = result->numel();
        float scale = (data_max_ - data_min_) / (feature_max_ - feature_min_);
        
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] = data_min_ + (ptr[i] - feature_min_) * scale;
        }
        
        return result;
    }
    
private:
    float data_min_ = 0.0f;
    float data_max_ = 1.0f;
    float feature_min_ = 0.0f;
    float feature_max_ = 1.0f;
};

// One-hot encoding
inline TensorPtr one_hot(const TensorPtr& labels, int64_t num_classes) {
    if (labels->ndim() != 1) {
        throw std::invalid_argument("Labels must be 1D tensor");
    }
    
    int64_t n = labels->numel();
    auto result = Tensor::zeros(Shape({n, num_classes}));
    
    const float* label_ptr = labels->data();
    float* result_ptr = result->data();
    
    for (int64_t i = 0; i < n; ++i) {
        int64_t label = static_cast<int64_t>(label_ptr[i]);
        if (label < 0 || label >= num_classes) {
            throw std::out_of_range("Label out of range for one-hot encoding");
        }
        result_ptr[i * num_classes + label] = 1.0f;
    }
    
    return result;
}

// Label encoder
class LabelEncoder {
public:
    void fit(const std::vector<std::string>& labels) {
        label_to_idx_.clear();
        idx_to_label_.clear();
        
        for (const auto& label : labels) {
            if (label_to_idx_.find(label) == label_to_idx_.end()) {
                int idx = static_cast<int>(label_to_idx_.size());
                label_to_idx_[label] = idx;
                idx_to_label_.push_back(label);
            }
        }
    }
    
    TensorPtr transform(const std::vector<std::string>& labels) const {
        auto result = Tensor::create(Shape({static_cast<int64_t>(labels.size())}));
        float* ptr = result->data();
        
        for (size_t i = 0; i < labels.size(); ++i) {
            auto it = label_to_idx_.find(labels[i]);
            if (it == label_to_idx_.end()) {
                throw std::invalid_argument("Unknown label: " + labels[i]);
            }
            ptr[i] = static_cast<float>(it->second);
        }
        
        return result;
    }
    
    std::vector<std::string> inverse_transform(const TensorPtr& encoded) const {
        if (encoded->ndim() != 1) {
            throw std::invalid_argument("Encoded labels must be 1D tensor");
        }
        
        std::vector<std::string> labels;
        labels.reserve(encoded->numel());
        
        const float* ptr = encoded->data();
        for (int64_t i = 0; i < encoded->numel(); ++i) {
            int idx = static_cast<int>(ptr[i]);
            if (idx < 0 || idx >= static_cast<int>(idx_to_label_.size())) {
                throw std::out_of_range("Encoded label out of range");
            }
            labels.push_back(idx_to_label_[idx]);
        }
        
        return labels;
    }
    
    size_t num_classes() const { return label_to_idx_.size(); }
    
private:
    std::unordered_map<std::string, int> label_to_idx_;
    std::vector<std::string> idx_to_label_;
};

// Data augmentation transforms
namespace transforms {

// Random horizontal flip
class RandomHorizontalFlip {
public:
    RandomHorizontalFlip(float p = 0.5) : p_(p), rng_(std::random_device{}()) {}
    
    TensorPtr operator()(const TensorPtr& image) {
        if (image->ndim() < 2) {
            throw std::invalid_argument("Image must have at least 2 dimensions");
        }
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng_) > p_) {
            return image;  // No flip
        }
        
        // Flip along width dimension (assuming HWC or CHW format)
        auto result = image->clone();
        float* data = result->data();
        
        // Get dimensions
        int64_t height = image->size(image->ndim() - 2);
        int64_t width = image->size(image->ndim() - 1);
        int64_t channels = 1;
        
        if (image->ndim() == 3) {
            // Could be CHW or HWC
            channels = image->size(0);  // Assuming CHW
        }
        
        // Flip each row
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width / 2; ++w) {
                    int64_t idx1 = (c * height + h) * width + w;
                    int64_t idx2 = (c * height + h) * width + (width - 1 - w);
                    std::swap(data[idx1], data[idx2]);
                }
            }
        }
        
        return result;
    }
    
private:
    float p_;
    mutable std::mt19937 rng_;
};

// Random crop
class RandomCrop {
public:
    RandomCrop(int64_t crop_height, int64_t crop_width)
        : crop_height_(crop_height), 
          crop_width_(crop_width),
          rng_(std::random_device{}()) {}
    
    TensorPtr operator()(const TensorPtr& image) {
        if (image->ndim() < 2) {
            throw std::invalid_argument("Image must have at least 2 dimensions");
        }
        
        int64_t height = image->size(image->ndim() - 2);
        int64_t width = image->size(image->ndim() - 1);
        
        if (height < crop_height_ || width < crop_width_) {
            throw std::invalid_argument("Image smaller than crop size");
        }
        
        // Random crop position
        std::uniform_int_distribution<int64_t> h_dist(0, height - crop_height_);
        std::uniform_int_distribution<int64_t> w_dist(0, width - crop_width_);
        
        int64_t h_start = h_dist(rng_);
        int64_t w_start = w_dist(rng_);
        
        // Create cropped tensor
        std::vector<int64_t> crop_dims = image->shape().dims();
        crop_dims[crop_dims.size() - 2] = crop_height_;
        crop_dims[crop_dims.size() - 1] = crop_width_;
        
        auto result = Tensor::create(Shape(crop_dims));
        
        // Copy cropped region
        const float* src = image->data();
        float* dst = result->data();
        
        if (image->ndim() == 2) {
            // 2D image
            for (int64_t h = 0; h < crop_height_; ++h) {
                for (int64_t w = 0; w < crop_width_; ++w) {
                    dst[h * crop_width_ + w] = 
                        src[(h_start + h) * width + (w_start + w)];
                }
            }
        } else if (image->ndim() == 3) {
            // 3D image (CHW format)
            int64_t channels = image->size(0);
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t h = 0; h < crop_height_; ++h) {
                    for (int64_t w = 0; w < crop_width_; ++w) {
                        dst[(c * crop_height_ + h) * crop_width_ + w] = 
                            src[(c * height + h_start + h) * width + (w_start + w)];
                    }
                }
            }
        }
        
        return result;
    }
    
private:
    int64_t crop_height_;
    int64_t crop_width_;
    mutable std::mt19937 rng_;
};

// Compose multiple transforms
template<typename... Transforms>
class Compose {
public:
    Compose(Transforms... transforms) : transforms_(transforms...) {}
    
    TensorPtr operator()(const TensorPtr& input) {
        return apply_transforms(input, std::index_sequence_for<Transforms...>{});
    }
    
private:
    std::tuple<Transforms...> transforms_;
    
    template<size_t... Is>
    TensorPtr apply_transforms(const TensorPtr& input, std::index_sequence<Is...>) {
        TensorPtr result = input;
        ((result = std::get<Is>(transforms_)(result)), ...);
        return result;
    }
};

// Helper to create compose transform
template<typename... Transforms>
auto compose(Transforms... transforms) {
    return Compose<Transforms...>(transforms...);
}

} // namespace transforms

// Train/validation/test split utilities
struct DataSplit {
    std::shared_ptr<Dataset> train;
    std::shared_ptr<Dataset> val;
    std::shared_ptr<Dataset> test;
};

inline DataSplit train_val_test_split(std::shared_ptr<Dataset> dataset,
                                     float train_ratio = 0.7f,
                                     float val_ratio = 0.15f,
                                     unsigned int seed = 42) {
    float test_ratio = 1.0f - train_ratio - val_ratio;
    
    if (train_ratio <= 0 || val_ratio <= 0 || test_ratio <= 0) {
        throw std::invalid_argument("Split ratios must be positive and sum to 1");
    }
    
    auto splits = random_split(dataset, {train_ratio, val_ratio, test_ratio}, seed);
    
    return {
        .train = splits[0],
        .val = splits[1],
        .test = splits[2]
    };
}

// K-fold cross validation
inline std::vector<std::pair<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>>>
k_fold_split(std::shared_ptr<Dataset> dataset, size_t k, unsigned int seed = 42) {
    if (k < 2) {
        throw std::invalid_argument("k must be at least 2");
    }
    
    size_t total_size = dataset->size();
    std::vector<size_t> all_indices(total_size);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // Shuffle indices
    std::mt19937 rng(seed);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    
    std::vector<std::pair<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>>> folds;
    size_t fold_size = total_size / k;
    
    for (size_t i = 0; i < k; ++i) {
        // Validation indices for this fold
        size_t val_start = i * fold_size;
        size_t val_end = (i == k - 1) ? total_size : (i + 1) * fold_size;
        
        std::vector<size_t> val_indices(
            all_indices.begin() + val_start,
            all_indices.begin() + val_end
        );
        
        // Training indices (everything else)
        std::vector<size_t> train_indices;
        train_indices.reserve(total_size - val_indices.size());
        
        for (size_t j = 0; j < val_start; ++j) {
            train_indices.push_back(all_indices[j]);
        }
        for (size_t j = val_end; j < total_size; ++j) {
            train_indices.push_back(all_indices[j]);
        }
        
        auto train_dataset = std::make_shared<SubsetDataset>(dataset, train_indices);
        auto val_dataset = std::make_shared<SubsetDataset>(dataset, val_indices);
        
        folds.emplace_back(train_dataset, val_dataset);
    }
    
    return folds;
}

} // namespace harrynet

#endif // HARRYNET_DATA_UTILS_H