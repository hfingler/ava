#ifndef _AVA_COMMON_EXTENSIONS_CUDA_H_
#define _AVA_COMMON_EXTENSIONS_CUDA_H_
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t __helper_cuda_memcpy_async_host_to_host(void *dst, const void *src,
    size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_host_to_device(void *dst, const void *src,
    size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_device_to_host(void *dst, const void *src,
    size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_device_to_device(void *dst, const void *src,
    size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_default(void *dst, const void *src,
    size_t count, cudaStream_t stream, bool dst_is_gpu, bool src_is_gpu);


#ifdef __cplusplus
}
#endif

#endif // _AVA_COMMON_EXTENSIONS_CUDA_H_
