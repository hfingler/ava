#include "common/declaration.h"
#include "common/logging.h"
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/extensions/memory_server/client.hpp"
#include <iostream>


cudaError_t __helper_cuda_memcpy_async_host_to_host(void *dst, const void *src,
    size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_host_to_device(void *dst, const void *src,
    size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), src, count, cudaMemcpyHostToDevice, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_host(void *dst, const void *src,
    size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst, __translate_ptr(src), count, cudaMemcpyDeviceToHost, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_device(void *dst, const void *src,
    size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), __translate_ptr(src), count, cudaMemcpyDeviceToDevice, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_default(void *dst, const void *src,
    size_t count, cudaStream_t stream, bool dst_is_gpu, bool src_is_gpu) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(
      dst_is_gpu ? __translate_ptr(dst) : dst,
      src_is_gpu ? __translate_ptr(src) : src,
      count, cudaMemcpyDefault, stream);
  return ret;
}

void __helper_print_kernel_info(struct fatbin_function *func, void **args) {
  std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc[0] << ", cufunc "
            << (void *)func->cufunc[0] << ", argc " << func->argc;
  int i;
  for (i = 0; i < func->argc; i++) {
    LOG_DEBUG << "arg[" << i << "] is " << (func->args[i].is_handle ? "" : "not ")
              << "a handle, size = " << func->args[i].size << ", ptr = " << args[i]
              << ", content = " << *((void **)args[i]);
  }
}


cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;

  uint32_t cur_dvc = __internal_getCurrentDevice();

  if (func == NULL) {
    return (cudaError_t)CUDA_ERROR_INVALID_PTX;
  }

  if (func->hostfunc[cur_dvc] != hostFun) {
    std::cerr << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc[cur_dvc] << " (device func "
              << (void *)func->cufunc[cur_dvc] << ")";
  } else {
    std::cerr << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc[cur_dvc] << std::endl;
  }
  //__helper_print_kernel_info(func, args);

  std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc[cur_dvc] << ", cufunc "
            << (void *)func->cufunc[cur_dvc] << ", argc " << func->argc << std::endl;

  /*
  //this is the default stuff
  ret = (cudaError_t)cuLaunchKernel(func->cufunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    sharedMem, (CUstream)stream, args, NULL);
  return ret;
  */

  for (int i = 0; i < func->argc; i++) {
    //TODO: I'm just throwing pointers at the dict. there is a probability that pointers collide and we mess up
    printf("  arg %d is handle? %d   size %d\n", i, func->args[i].is_handle, func->args[i].size);
    printf("p1 %p   p2 %p \n", args[i], *((void **)args[i]));
    
    //TODO: we need something that says if something is a pointer or not
    //if (func->args[i].is_handle) {    
    if (func->args[i].size == 8) {
      *((void**)args[i]) = __translate_ptr(*((void **)args[i]));
    }
  }

  for (int i = 0; i < func->argc; i++) {
    printf("    arg [%d]:  %p\n", i, *((void **)args[i]));
  }

  //BIG TODOs: need to map streams on new GPU when migrating
  //      (maybe) need to figure out the replay mechanism so we actually have the kernel in the new GPU
  int current_gpu;
  cudaGetDevice(&current_gpu);
  printf(">>> __helper_launch_kernel on GPU [%d], current device: %d\n", current_gpu, cur_dvc);

  ret = (cudaError_t)cuLaunchKernel(func->cufunc[cur_dvc], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    0, 0, args, NULL);
                                    //sharedMem, (CUstream)stream, args, NULL);

  printf(">>> cuLaunchKernel returned %d\n", ret);
  return ret;

}
