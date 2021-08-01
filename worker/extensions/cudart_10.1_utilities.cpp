#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudart_nw_internal.h"
#include "common/declaration.h"
#include "common/logging.h"
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "extensions/memory_server/client.hpp"

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

#ifdef WITH_SVLESS_MIGRATION
  uint32_t cur_dvc = __internal_getCurrentDevice();
#else
  uint32_t cur_dvc = 0;
#endif

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

#ifdef WITH_SVLESS_MIGRATION
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
  //BIG TODOs: need to map streams on new GPU when migrating
  ret = (cudaError_t)cuLaunchKernel(func->cufunc[cur_dvc], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    0, 0, args, NULL);
                                    //sharedMem, (CUstream)stream, args, NULL);
  printf(">>> cuLaunchKernel returned %d\n", ret);
  return ret;
// if not with migration, just get over it and do
#else
  ret = (cudaError_t)cuLaunchKernel(func->cufunc[0], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                                    0, 0, args, 0);
                                    //sharedMem, (CUstream)stream, args, NULL);
  printf(">>> cuLaunchKernel returned %d\n", ret);
  return ret;
#endif
}

void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle, CUmodule *module) {
  int ret;

#ifdef WITH_SVLESS_MIGRATION
  for (int i = 0 ; i < __internal_getDeviceCount() ; i++) {
    printf("setting device to %d\n", i);
    cudaSetDevice(i);

    __cudaInitModule(handle);
    module[i] = NULL;
    ret = cuModuleLoadData(&module[i], (void *)fatCubin->ptr);
    printf("loaded module data into ctx %d : %p\n", i, module[i]);
    assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
    (void)ret;
  }
  //reset back
  printf("resetting device to %d\n", __internal_getCurrentDevice());
  cudaSetDevice(__internal_getCurrentDevice());
#else
  __cudaInitModule(handle);
  module[0] = NULL;
  ret = cuModuleLoadData(&module[0], (void *)fatCubin->ptr);
  assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
#endif
}

CUresult __helper_cuModuleLoad(CUmodule *module, const char *fname) {
#ifdef WITH_SVLESS_MIGRATION
  CUresult ret;
  CUresult other_ret;
  CUmodule other_module;
  for (int i = 0; i < __internal_getDeviceCount(); i++) {
    cudaSetDevice(i);
    if (i != __internal_getCurrentDevice()) {
      other_ret = cuModuleLoad(&other_module, fname);
    } else {
      ret = cuModuleLoad(module, fname);
    }
  }
  fprintf(stderr, "resetting device to %d\n", __internal_getCurrentDevice());
  cudaSetDevice(__internal_getCurrentDevice());
  return ret;
#else
  return cuModuleLoad(module, fname);
#endif
}

cudaError_t __helper_cudaLaunchKernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                      void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret;
  __internal_kernelIn();
  ret = __helper_launch_kernel(func, hostFun, gridDim, blockDim, args, sharedMem, stream);
  __internal_kernelOut();
  return ret;
}

cudaError_t __helper_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  return cudaMemcpy(__translate_ptr(dst), __translate_ptr(src), count, kind);
}

cudaError_t __helper_cudaMemset(void *devPtr, int value, size_t count) {
  return cudaMemset(__translate_ptr(devPtr), value, count);
}

/**
 * Look up the CUDA kernel function and save it in the list.
 */
void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule* module,
                                const char *deviceName) {

#ifdef WITH_SVLESS_MIGRATION
  for (int i = 0 ; i < __internal_getDeviceCount() ; i++) {
#else
  for (int i = 0 ; i < 1 ; i++) {
#endif
    // Empty fatbinary
    if (!module[i]) {
      LOG_DEBUG << "Register a fat binary from a empty module";
      return;
    }

    if (func == NULL) {
      LOG_FATAL << "fatbin_function is NULL";
      throw std::invalid_argument("received empty fatbin_function");
    }

    // Only register the first host function
    //if (func->hostfunc != NULL) return;
    if (func->hostfunc[i] != NULL) continue;

#ifdef WITH_SVLESS_MIGRATION
    //this call needs to be done in each context
    printf("setting device to %d\n", i);
    cudaSetDevice(i);

    CUcontext cuCtx;
    cuCtxGetCurrent(&cuCtx);
    CUdevice cuDev;
    cuCtxGetDevice(&cuDev);

    printf("Curently on device %d, ctx %p\n", cuDev, cuCtx);
#endif

    CUresult ret = cuModuleGetFunction(&func->cufunc[i], module[i], deviceName);
    if (ret != CUDA_SUCCESS) {
      LOG_ERROR << "cuModuleGetFunction fail with " << ret;
      throw std::runtime_error("failed to get module function");
    }
    std::cerr << "*** __helper_register_function kernel at device " << i << " host func " << std::hex << (uintptr_t)hostFun << " -> device func " << (uintptr_t)func->cufunc[i]
      << " deviceName " << deviceName << std::endl;
    func->hostfunc[i] = (void *)hostFun;
    func->module[i] = module[i];
  }

#ifdef WITH_SVLESS_MIGRATION
  //reset back
  cudaSetDevice(__internal_getCurrentDevice());
#endif
}

cudaError_t __helper_func_get_attributes(struct cudaFuncAttributes *attr, struct fatbin_function *func,
                                         const void *hostFun) {
#ifdef WITH_SVLESS_MIGRATION
  uint32_t cur_dvc = __internal_getCurrentDevice();
#else
  uint32_t cur_dvc = 0;
#endif

  if (func == NULL) {
    LOG_DEBUG << "func is NULL";
    return static_cast<cudaError_t>(cudaErrorInvalidDeviceFunction);
  }

  if (func->hostfunc[cur_dvc] != hostFun) {
    LOG_ERROR << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc[cur_dvc] << " (device func "
              << (void *)func->cufunc[cur_dvc] << ")";
  } else {
    LOG_DEBUG << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc[cur_dvc];
  }

  CUresult ret;
  ret = cuFuncGetAttribute((int *)&attr->sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute((int *)&attr->constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute((int *)&attr->localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute(&attr->maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute(&attr->numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute(&attr->ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute(&attr->binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, func->cufunc[cur_dvc]);
  attr->cacheModeCA = 0;
  ret = cuFuncGetAttribute(&attr->maxDynamicSharedSizeBytes, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           func->cufunc[cur_dvc]);
  ret = cuFuncGetAttribute(&attr->preferredShmemCarveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                           func->cufunc[cur_dvc]);

  return static_cast<cudaError_t>(ret);
}

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor(int *numBlocks, struct fatbin_function *func,
                                                                    const void *hostFun, int blockSize,
                                                                    size_t dynamicSMemSize) {
#ifdef WITH_SVLESS_MIGRATION
  uint32_t cur_dvc = __internal_getCurrentDevice();
#else
  uint32_t cur_dvc = 0;
#endif

  if (func == NULL) {
    LOG_DEBUG << "func is NULL";
    return (cudaError_t)cudaErrorInvalidDeviceFunction;
  }

  if (func->hostfunc[cur_dvc] != hostFun) {
    LOG_ERROR << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc[cur_dvc] << " (device func "
              << (void *)func->cufunc[cur_dvc] << ")";
  } else {
    LOG_DEBUG << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc[cur_dvc];
  }
  return static_cast<cudaError_t>(
      cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func->cufunc[cur_dvc], blockSize, dynamicSMemSize));
}

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(int *numBlocks,
                                                                               struct fatbin_function *func,
                                                                               const void *hostFun, int blockSize,
                                                                               size_t dynamicSMemSize,
                                                                               unsigned int flags) {
#ifdef WITH_SVLESS_MIGRATION
  uint32_t cur_dvc = __internal_getCurrentDevice();
#else
  uint32_t cur_dvc = 0;
#endif

  if (func == NULL) {
    LOG_DEBUG << "func is NULL";
    return (cudaError_t)cudaErrorInvalidDeviceFunction;
  }

  if (func->hostfunc[cur_dvc] != hostFun) {
    LOG_ERROR << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc[cur_dvc] << " (device func "
              << (void *)func->cufunc[cur_dvc] << ")";
  } else {
    LOG_DEBUG << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc[cur_dvc];
  }

  return static_cast<cudaError_t>(
      cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func->cufunc[cur_dvc], blockSize, dynamicSMemSize, flags));
}
