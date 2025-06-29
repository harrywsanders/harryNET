#ifndef HARRYNET_DATALOADER_H
#define HARRYNET_DATALOADER_H

#include "dataset.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <numeric>
#include <atomic>
#include <unordered_map>
#include <iostream>
#include <chrono>

namespace harrynet {

// batch type - pair of batched input and target tensors
using Batch = std::pair<TensorPtr, TensorPtr>;

// default collate function - stacks samples into batches
class DefaultCollate {
public:
    Batch operator()(const std::vector<Sample>& samples) const {
        if (samples.empty()) {
            throw std::invalid_argument("Cannot collate empty batch");
        }
        
        // get shapes from first sample
        auto first_data_shape = samples[0].first->shape();
        auto first_target_shape = samples[0].second->shape();
        
        // create batch shapes
        std::vector<int64_t> batch_data_dims = {static_cast<int64_t>(samples.size())};
        batch_data_dims.insert(batch_data_dims.end(), 
                              first_data_shape.dims().begin(), 
                              first_data_shape.dims().end());
        
        std::vector<int64_t> batch_target_dims = {static_cast<int64_t>(samples.size())};
        batch_target_dims.insert(batch_target_dims.end(),
                                first_target_shape.dims().begin(),
                                first_target_shape.dims().end());
        
        Shape batch_data_shape(batch_data_dims);
        Shape batch_target_shape(batch_target_dims);
        
        // allocate batch tensors
        auto batch_data = Tensor::create(batch_data_shape);
        auto batch_targets = Tensor::create(batch_target_shape);
        
        // copy samples into batch
        int64_t data_sample_size = first_data_shape.numel();
        int64_t target_sample_size = first_target_shape.numel();
        
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& [data, target] = samples[i];
            
            // verify shapes match
            if (data->shape() != first_data_shape || 
                target->shape() != first_target_shape) {
                throw std::invalid_argument("All samples in batch must have same shape");
            }
            
            // copy data
            std::copy(data->data(),
                     data->data() + data_sample_size,
                     batch_data->data() + i * data_sample_size);
                     
            std::copy(target->data(),
                     target->data() + target_sample_size,
                     batch_targets->data() + i * target_sample_size);
        }
        
        return {batch_data, batch_targets};
    }
};

// sampler interface for controlling iteration order
class Sampler {
public:
    virtual ~Sampler() = default;
    virtual std::vector<size_t> indices() = 0;
    virtual size_t size() const = 0;
};

// sequential sampler - returns indices in order
class SequentialSampler : public Sampler {
public:
    SequentialSampler(size_t dataset_size) : size_(dataset_size) {}
    
    std::vector<size_t> indices() override {
        std::vector<size_t> idx(size_);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }
    
    size_t size() const override { return size_; }
    
private:
    size_t size_;
};

// random sampler - returns shuffled indices
class RandomSampler : public Sampler {
public:
    RandomSampler(size_t dataset_size, unsigned int seed = 0)
        : size_(dataset_size), rng_(seed ? seed : std::random_device{}()) {}
    
    std::vector<size_t> indices() override {
        std::vector<size_t> idx(size_);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng_);
        return idx;
    }
    
    size_t size() const override { return size_; }
    
private:
    size_t size_;
    std::mt19937 rng_;
};

// batch sampler - groups indices into batches
class BatchSampler {
public:
    BatchSampler(std::shared_ptr<Sampler> sampler, 
                 size_t batch_size,
                 bool drop_last = false)
        : sampler_(sampler), batch_size_(batch_size), drop_last_(drop_last) {}
    
    std::vector<std::vector<size_t>> batches() {
        auto all_indices = sampler_->indices();
        std::vector<std::vector<size_t>> batch_indices;
        
        size_t num_batches = all_indices.size() / batch_size_;
        if (!drop_last_ && all_indices.size() % batch_size_ != 0) {
            num_batches++;
        }
        
        batch_indices.reserve(num_batches);
        
        for (size_t i = 0; i < num_batches; ++i) {
            size_t start = i * batch_size_;
            size_t end = std::min(start + batch_size_, all_indices.size());
            
            if (drop_last_ && end - start < batch_size_) {
                break;
            }
            
            batch_indices.emplace_back(all_indices.begin() + start, 
                                      all_indices.begin() + end);
        }
        
        return batch_indices;
    }
    
    size_t size() const {
        size_t total = sampler_->size();
        if (drop_last_) {
            return total / batch_size_;
        } else {
            return (total + batch_size_ - 1) / batch_size_;
        }
    }
    
private:
    std::shared_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool drop_last_;
};

// Worker for parallel data loading
template<typename CollateFn>
class DataLoaderWorker {
public:
    DataLoaderWorker(std::shared_ptr<Dataset> dataset,
                     const CollateFn& collate_fn,
                     size_t worker_id)
        : dataset_(dataset), 
          collate_fn_(collate_fn), 
          worker_id_(worker_id),
          should_stop_(false) {}
    
    ~DataLoaderWorker() {
        stop();
    }
    
    void start() {
        worker_thread_ = std::thread([this]() { this->worker_loop(); });
    }
    
    void stop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            should_stop_ = true;
            cv_task_.notify_all();
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    // Queue a batch task
    void queue_batch(const std::vector<size_t>& indices, size_t batch_id) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_queue_.push({indices, batch_id});
            cv_task_.notify_one();
        }
    }
    
    // Try to get a completed batch
    bool try_get_batch(size_t batch_id, Batch& out_batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = completed_batches_.find(batch_id);
        if (it != completed_batches_.end()) {
            out_batch = std::move(it->second);
            completed_batches_.erase(it);
            return true;
        }
        return false;
    }
    
    // Wait for a specific batch
    bool wait_for_batch(size_t batch_id, Batch& out_batch, 
                       std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto pred = [this, batch_id]() {
            return completed_batches_.find(batch_id) != completed_batches_.end() || should_stop_;
        };
        
        if (cv_result_.wait_for(lock, timeout, pred)) {
            auto it = completed_batches_.find(batch_id);
            if (it != completed_batches_.end()) {
                out_batch = std::move(it->second);
                completed_batches_.erase(it);
                return true;
            }
        }
        return false;
    }
    
private:
    struct Task {
        std::vector<size_t> indices;
        size_t batch_id;
    };
    
    void worker_loop() {
        while (true) {
            Task task;
            
            // Get next task
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_task_.wait(lock, [this]() { 
                    return !task_queue_.empty() || should_stop_; 
                });
                
                if (should_stop_ && task_queue_.empty()) {
                    break;
                }
                
                if (!task_queue_.empty()) {
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
            }
            
            // Process task
            if (!task.indices.empty()) {
                try {
                    // Load samples
                    std::vector<Sample> samples;
                    samples.reserve(task.indices.size());
                    
                    for (size_t idx : task.indices) {
                        samples.push_back(dataset_->get_item(idx));
                    }
                    
                    // Collate into batch
                    auto batch = collate_fn_(samples);
                    
                    // Store completed batch
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        completed_batches_[task.batch_id] = std::move(batch);
                        cv_result_.notify_all();
                    }
                } catch (const std::exception& e) {
                    // Log error but continue
                    std::cerr << "Worker " << worker_id_ << " error: " << e.what() << std::endl;
                }
            }
        }
    }
    
    std::shared_ptr<Dataset> dataset_;
    CollateFn collate_fn_;
    size_t worker_id_;
    
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_result_;
    
    std::queue<Task> task_queue_;
    std::unordered_map<size_t, Batch> completed_batches_;
    std::atomic<bool> should_stop_;
};

// Main DataLoader class
template<typename CollateFn = DefaultCollate>
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset,
               size_t batch_size = 1,
               bool shuffle = false,
               size_t num_workers = 0,
               bool drop_last = false,
               CollateFn collate_fn = CollateFn(),
               unsigned int seed = 0)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          drop_last_(drop_last),
          collate_fn_(collate_fn) {
        
        // Create sampler
        if (shuffle) {
            sampler_ = std::make_shared<RandomSampler>(dataset->size(), seed);
        } else {
            sampler_ = std::make_shared<SequentialSampler>(dataset->size());
        }
        
        // Create batch sampler
        batch_sampler_ = std::make_unique<BatchSampler>(sampler_, batch_size, drop_last);
        
        // Initialize workers if using multiprocessing
        if (num_workers_ > 0) {
            workers_.reserve(num_workers_);
            for (size_t i = 0; i < num_workers_; ++i) {
                workers_.emplace_back(
                    std::make_unique<DataLoaderWorker<CollateFn>>(dataset_, collate_fn_, i)
                );
                workers_.back()->start();
            }
        }
    }
    
    ~DataLoader() {
        // Workers are automatically stopped in their destructors
    }
    
    // Iterator class
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t start_idx)
            : loader_(loader), 
              current_idx_(start_idx),
              batch_id_(0) {
            if (loader_) {
                batches_ = loader_->batch_sampler_->batches();
                total_batches_ = batches_.size();
                
                // For multi-threaded loading, prefetch initial batches
                if (loader_->num_workers_ > 0 && current_idx_ < total_batches_) {
                    prefetch_batches();
                }
            }
        }
        
        Iterator& operator++() {
            if (current_idx_ < total_batches_) {
                ++current_idx_;
                ++batch_id_;
            }
            return *this;
        }
        
        bool operator!=(const Iterator& other) const {
            return current_idx_ != other.current_idx_;
        }
        
        Batch operator*() {
            if (current_idx_ >= total_batches_) {
                throw std::out_of_range("Iterator out of range");
            }
            
            if (loader_->num_workers_ > 0) {
                // Multi-threaded loading
                size_t worker_idx = current_idx_ % loader_->num_workers_;
                Batch batch;
                
                // Wait for the batch from the worker
                if (!loader_->workers_[worker_idx]->wait_for_batch(batch_id_, batch)) {
                    throw std::runtime_error("Failed to load batch from worker");
                }
                
                // Prefetch next batch
                size_t next_idx = current_idx_ + loader_->num_workers_;
                if (next_idx < total_batches_) {
                    size_t next_worker = next_idx % loader_->num_workers_;
                    loader_->workers_[next_worker]->queue_batch(batches_[next_idx], 
                                                               batch_id_ + loader_->num_workers_);
                }
                
                return batch;
            } else {
                // Single-threaded loading
                std::vector<Sample> samples;
                const auto& batch_indices = batches_[current_idx_];
                samples.reserve(batch_indices.size());
                
                for (size_t idx : batch_indices) {
                    samples.push_back(loader_->dataset_->get_item(idx));
                }
                
                return loader_->collate_fn_(samples);
            }
        }
        
    private:
        void prefetch_batches() {
            // Queue initial batches to workers
            size_t num_prefetch = std::min(loader_->num_workers_, 
                                         total_batches_ - current_idx_);
            
            for (size_t i = 0; i < num_prefetch; ++i) {
                size_t batch_idx = current_idx_ + i;
                size_t worker_idx = batch_idx % loader_->num_workers_;
                loader_->workers_[worker_idx]->queue_batch(batches_[batch_idx], batch_id_ + i);
            }
        }
        
        DataLoader* loader_;
        size_t current_idx_;
        size_t total_batches_;
        size_t batch_id_;  // Unique ID for tracking batches across workers
        std::vector<std::vector<size_t>> batches_;
    };
    
    Iterator begin() { 
        return Iterator(this, 0); 
    }
    
    Iterator end() { 
        auto batches = batch_sampler_->batches();
        return Iterator(this, batches.size()); 
    }
    
    size_t size() const { 
        return batch_sampler_->size(); 
    }
    
private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    bool drop_last_;
    CollateFn collate_fn_;
    
    std::shared_ptr<Sampler> sampler_;
    std::unique_ptr<BatchSampler> batch_sampler_;
    std::vector<std::unique_ptr<DataLoaderWorker<CollateFn>>> workers_;
};

// helper function to create dataloader with default collate
inline std::shared_ptr<DataLoader<DefaultCollate>>
make_dataloader(std::shared_ptr<Dataset> dataset,
                size_t batch_size = 1,
                bool shuffle = false,
                size_t num_workers = 0,
                bool drop_last = false,
                unsigned int seed = 0) {
    return std::make_shared<DataLoader<DefaultCollate>>(
        dataset, batch_size, shuffle, num_workers, drop_last, DefaultCollate(), seed
    );
}

} // namespace harrynet

#endif // HARRYNET_DATALOADER_H