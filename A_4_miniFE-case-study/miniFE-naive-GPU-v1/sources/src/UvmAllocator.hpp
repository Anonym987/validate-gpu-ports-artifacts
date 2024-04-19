#pragma once

#include <cuda_runtime.h>

template <class T>
struct UvmAllocator
{
    using value_type = T;
    UvmAllocator() noexcept {};
    template <class U>
    UvmAllocator(const UvmAllocator<U> &) noexcept {};
    T *allocate(std::size_t n)
    {
        T *mem;
        cudaError_t err = cudaMallocManaged(&mem, n * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error in UVM Allocator "
                      << std::endl;
            std::cerr << cudaGetErrorString(err) << " allocate" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return mem;
    };
    void deallocate(T *p, std::size_t n)
    {
        cudaFree(p);
    };
};

template <class T, class U>
constexpr bool operator==(const UvmAllocator<T> &, const UvmAllocator<U> &) noexcept
{
    return true;
}

template <class T, class U>
constexpr bool operator!=(const UvmAllocator<T> &, const UvmAllocator<U> &) noexcept
{
    return false;
}