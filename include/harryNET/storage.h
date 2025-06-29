#ifndef HARRYNET_STORAGE_H
#define HARRYNET_STORAGE_H

#include <memory>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace harrynet {

// aligned allocation for SIMD ops
inline void* aligned_alloc(size_t size, size_t alignment = 32) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

inline void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(float* ptr) const {
        aligned_free(ptr);
    }
};

// efficient storage class with ref counting and alignment
class Storage {
public:
    using DataPtr = std::shared_ptr<float[]>;
    
    Storage() : size_(0) {}
    
    explicit Storage(size_t size) 
        : size_(size) {
        if (size > 0) {
            float* raw_ptr = static_cast<float*>(aligned_alloc(size * sizeof(float)));
            data_ = DataPtr(raw_ptr, AlignedDeleter());
            std::fill(raw_ptr, raw_ptr + size, 0.0f);
        }
    }
    
    Storage(size_t size, float value) 
        : size_(size) {
        if (size > 0) {
            float* raw_ptr = static_cast<float*>(aligned_alloc(size * sizeof(float)));
            data_ = DataPtr(raw_ptr, AlignedDeleter());
            std::fill(raw_ptr, raw_ptr + size, value);
        }
    }
    
    // create storage from existing data
    Storage(const float* data, size_t size) 
        : size_(size) {
        if (size > 0 && data) {
            float* raw_ptr = static_cast<float*>(aligned_alloc(size * sizeof(float)));
            data_ = DataPtr(raw_ptr, AlignedDeleter());
            std::copy(data, data + size, raw_ptr);
        }
    }
    
    // copy constructor (shares data)
    Storage(const Storage& other) = default;
    
    // move constructor
    Storage(Storage&& other) noexcept
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }
    
    // assignment operators
    Storage& operator=(const Storage& other) = default;
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
    
    // clone storage (deep copy)
    Storage clone() const {
        if (size_ == 0) return Storage();
        return Storage(data_.get(), size_);
    }
    
    // element access
    float& operator[](size_t idx) {
        return data_[idx];
    }
    
    const float& operator[](size_t idx) const {
        return data_[idx];
    }
    
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    // check if storage is shared
    bool is_shared() const {
        return data_.use_count() > 1;
    }
    
    // get number of references
    long use_count() const {
        return data_.use_count();
    }
    
    // fill with value
    void fill(float value) {
        if (data_) {
            std::fill(data_.get(), data_.get() + size_, value);
        }
    }
    
    // zero out storage
    void zero() {
        fill(0.0f);
    }
    
private:
    DataPtr data_;
    size_t size_;
};

} // namespace harrynet

#endif // HARRYNET_STORAGE_H