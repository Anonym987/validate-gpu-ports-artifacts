#include <limits>
namespace desul {
namespace Impl {
#if defined(__CUDA_ARCH__) || (defined(__clang__) && !defined(__NVCC__))
// Choose the variant of atomics we are using later
#if !defined(DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE) && \
    !defined(DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL)
#if (__CUDACC_VER_MAJOR__ > 11) || \
    ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ > 1))
#define DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL
#else
#define DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE
#endif
#endif
#include <desul/atomics/cuda/cuda_cc7_asm.inc>

#endif
}  // namespace Impl
}  // namespace desul
