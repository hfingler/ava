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
  std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc << ", cufunc "
            << (void *)func->cufunc << ", argc " << func->argc;
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

  if (func == NULL) {
    return (cudaError_t)CUDA_ERROR_INVALID_PTX;
  }

  if (func->hostfunc != hostFun) {
    std::cerr << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc << " (device func "
              << (void *)func->cufunc << ")";
  } else {
    std::cerr << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc;
  }
  __helper_print_kernel_info(func, args);

  //this is the default stuff
  ret = (cudaError_t)cuLaunchKernel(func->cufunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    sharedMem, (CUstream)stream, args, NULL);
  return ret;

/*
  //possibly translate pointers
  for (int i = 0; i < func->argc; i++) {
    //TODO: I'm just throwing pointers at the dict. there is a probability that pointers collide and we mess up
    //printf("  arg %d is handle? %d   size %d\n", i, func->args[i].is_handle, func->args[i].size);
    //std::cout << "  content:  " << *((void **)args[i]) << std::endl;
    //if (func->args[i].is_handle) {
    //  args[i] = __translate_ptr(args[i]);
    //}
    *((char*)args[i]) = (char*)__translate_ptr(*((void **)args[i]));
  }

  //BIG TODOs: need to map streams on new GPU when migrating
  //      (maybe) need to figure out the replay mechanism so we actually have the kernel in the new GPU
  cudaFree(0);
  int current_gpu;
  cudaGetDevice(&current_gpu);
  printf(">>> __helper_launch_kernel on GPU [%d]\n", current_gpu);

  cudaStream_t s2;
  ret = cudaStreamCreate(&s2);
  printf(">>> cudaStreamCreate returned %d\n", ret);

  CUcontext cuctx1, cuctx2;
  cuStreamGetCtx(s2, &cuctx1); 
  cuCtxGetCurrent(&cuctx2);


  printf("%p == %p ?  %d\n", (void*)cuctx1, (void*)cuctx2, cuctx1==cuctx2);

  cudaDeviceSynchronize();
  //ret = (cudaError_t)cuLaunchKernel(func->cufunc, 1, 1, 1, 1, 1, 1,
  //                                  0, 0, 0, NULL);
                                    //sharedMem, (CUstream)stream, args, NULL);

  ret = cudaLaunchKernel((void *)func->cufunc, gridDim, blockDim, 0, 0, 0);

  printf(">>> cuLaunchKernel returned %d\n", ret);
  return ret;
*/

}
