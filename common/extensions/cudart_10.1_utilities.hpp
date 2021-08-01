#ifndef AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
#define AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glib.h>

#include <algorithm>
#include <absl/container/flat_hash_map.h>

#define MAX_KERNEL_ARG 30
#define MAX_KERNEL_NAME_LEN 1024
#define MAX_ASYNC_BUFFER_NUM 16

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int deference_int_pointer(int *p);

#define cu_in_out_buffer(x, y)                                                \
  ({                                                                          \
    if (ava_is_in)                                                            \
      ava_buffer(x);                                                          \
    else                                                                      \
      ava_buffer(std::min(x, y == (void *)0 ? x : deference_int_pointer(y))); \
  })

struct fatbin_wrapper {
  uint32_t magic;
  uint32_t seq;
  uint64_t ptr;
  uint64_t data_ptr;
};

struct kernel_arg {
  char is_handle;
  uint32_t size;
};

struct fatbin_function {
  int argc;
  struct kernel_arg args[MAX_KERNEL_ARG];

  //this is barely any space, so leave 4 as default
  CUfunction cufunc[4];
  void *hostfunc[4];
  CUmodule module[4];
};

size_t __helper_fatbin_size(const void *cubin);

void __helper_print_kernel_info(struct fatbin_function *func, void **args);

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream);

int __helper_cubin_num(void **cubin_handle);

void __helper_print_fatcubin_info(void *fatCubin, void **ret);

void __helper_unregister_fatbin(void **fatCubinHandle);

void __helper_parse_function_args(const char *name, struct kernel_arg *args);

size_t __helper_launch_extra_size(void **extra);

void *__helper_cu_mem_host_alloc_portable(size_t size);

void __helper_cu_mem_host_free(void *ptr);

void __helper_assosiate_function_dump(GHashTable *funcs, struct fatbin_function **func, void *local,
                                      const char *deviceName);

void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule* module,
                                const char *deviceName);

/* Async buffer address list */
struct async_buffer_list {
  int num_buffers;
  void *buffers[MAX_ASYNC_BUFFER_NUM]; /* array of buffer addresses */
  size_t buffer_sizes[MAX_ASYNC_BUFFER_NUM];
};

void __helper_register_async_buffer(struct async_buffer_list *buffers, void *buffer, size_t size);

struct async_buffer_list *__helper_load_async_buffer_list(struct async_buffer_list *buffers);

int __helper_a_last_dim_size(cublasOperation_t transa, int k, int m);

int __helper_b_last_dim_size(cublasOperation_t transb, int k, int n);

int __helper_type_size(cudaDataType dataType);

cudaError_t __helper_func_get_attributes(struct cudaFuncAttributes *attr, struct fatbin_function *func,
                                         const void *hostFun);

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor(int *numBlocks, struct fatbin_function *func,
                                                                    const void *hostFun, int blockSize,
                                                                    size_t dynamicSMemSize);

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(int *numBlocks,
                                                                               struct fatbin_function *func,
                                                                               const void *hostFun, int blockSize,
                                                                               size_t dynamicSMemSize,
                                                                               unsigned int flags);

void __helper_print_pointer_attributes(const struct cudaPointerAttributes *attributes, const void *ptr);

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

void __helper_record_module_path(CUmodule module, const char* fname);
void __helper_parse_module_function_args(CUmodule module, const char *name, struct fatbin_function **func);
void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle, CUmodule *module);
CUresult __helper_cuModuleLoad(CUmodule *module, const char *fname);
cudaError_t __helper_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t __helper_cudaMemset(void *devPtr, int value, size_t count);
CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size);
cudaError_t __internal_cudaFree(void *devPtr);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
