#include "common/extensions/cudart_10.1_utilities.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fmt/core.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include <chrono>
#include <thread>


#include "common/declaration.h"
#include "common/logging.h"
#include "cudart_nw_internal.h"
#include "extensions/memory_server/client.hpp"

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

uint32_t __internal_getCurrentDeviceIndex() { return __internal_getCurrentDevice(); }

cudaError_t __helper_cuda_memcpy_async_host_to_host(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_host_to_device(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), src, count, cudaMemcpyHostToDevice, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_host(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret; 
  ret = cudaMemcpyAsync(dst, __translate_ptr(src), count, cudaMemcpyDeviceToHost, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_device(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), __translate_ptr(src), count, cudaMemcpyDeviceToDevice, stream);
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_default(void *dst, const void *src, size_t count, cudaStream_t stream,
                                               bool dst_is_gpu, bool src_is_gpu) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst_is_gpu ? __translate_ptr(dst) : dst, src_is_gpu ? __translate_ptr(src) : src, count,
                        cudaMemcpyDefault, stream);
  return ret;
}

void __helper_print_kernel_info(struct fatbin_function *func, void **args) {
  std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc[0] << ", cufunc "
            << (void *)func->cufunc[0] << ", argc " << func->argc << std::endl;
  int i;
  for (i = 0; i < func->argc; i++) {
    std::cerr << "arg[" << i << "] is " << (func->args[i].is_handle ? "" : "not ")
              << "a handle, size = " << func->args[i].size << ", ptr = " << args[i]
              << ", content = " << *((void **)args[i]) << std::endl;
  }
}

cublasStatus_t __helper_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                       int n, int k, const float *alpha, /* host or device pointer */
                                       const float *A, int lda, const float *B, int ldb,
                                       const float *beta, /* host or device pointer */
                                       float *C, int ldc, bool alpha_is_gpu, bool beta_is_gpu) {
  auto ret = cublasSgemm(handle, transa, transb, m, n, k, alpha_is_gpu ? __translate_ptr(alpha) : alpha, A, lda, B, ldb,
                         beta_is_gpu ? __translate_ptr(beta) : beta, C, ldc);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_create_stream(cudaStream_t *pStream, unsigned int flags, int priority) {
  cudaError_t ret;

  if (__internal_allContextsEnabled()) {
    uint32_t cur_dvc = __internal_getCurrentDevice();
    cudaStreamCreateWithPriority(pStream, flags, priority);
    // add current text
    GPUMemoryServer::Client::getInstance().streams_map[*pStream][cur_dvc] = *pStream;

    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      if (i == cur_dvc) continue;
      cudaSetDevice(i);
      cudaStream_t new_stream;
      cudaStreamCreateWithPriority(&new_stream, flags, priority);
      GPUMemoryServer::Client::getInstance().streams_map[*pStream][i] = new_stream;
    }
    // reset back device and return OK
    cudaSetDevice(__internal_getCurrentDevice());
    return (cudaError_t)0;
  } else {
    cudaError_t err = cudaStreamCreateWithPriority(pStream, flags, priority);
    return err;
  }
}

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret2 = cudaGetLastError();
  printf(" culaunch peek error:  %d\n", ret2);

  // this might trigger and to migration
  __internal_kernelIn();

  uint32_t cur_dvc;
  if (__internal_allContextsEnabled())
    cur_dvc = __internal_getCurrentDevice();
  else
    cur_dvc = 0;

#ifndef NDEBUG
  printf("__helper_launch_kernel on device slot [%d]\n", cur_dvc);
#endif

  if (func == NULL) {
    return (cudaError_t)CUDA_ERROR_INVALID_PTX;
  }

  if (func->hostfunc[cur_dvc] != hostFun) {
    // std::cerr << "search host func " << hostFun << " -> stored " << (void *)func->hostfunc[cur_dvc] << " (device func
    // "
    //          << (void *)func->cufunc[cur_dvc] << ")";
  } else {
#ifndef NDEBUG
    std::cerr << "matched host func " << hostFun << " -> device func " << (void *)func->cufunc[cur_dvc] << std::endl;
#endif
  }
  __helper_print_kernel_info(func, args);
  // std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc[cur_dvc] << ", cufunc "
  //          << (void *)func->cufunc[cur_dvc] << ", argc " << func->argc << std::endl;

  cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;
  // if we need to figure out the correct context to get function
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < func->argc; i++) {
      //printf("  arg %d is handle? %d   size %d  ptr  %p\n", i, func->args[i].is_handle, func->args[i].size, *((void **)args[i]));
      // TODO: we need something that says if something is a pointer or not
      if (func->args[i].size == 8) {
          *((void **)args[i]) = __translate_ptr(*((void **)args[i]));
      }
    }
    // BIG TODOs: need to map streams on new GPU when migrating
    ret = (cudaError_t)cuLaunchKernel(func->cufunc[cur_dvc], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                      blockDim.z, sharedMem, (CUstream)  __helper_translate_stream(stream), args, NULL);
    printf(">>> cuLaunchKernel returned %d\n", ret);

    // TODO: fix
    __internal_kernelOut();
    return ret;
  }
  // if not with migration, just get over it and do
  else {
    ret = (cudaError_t)cuLaunchKernel(func->cufunc[0], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                      blockDim.z, sharedMem,  (CUstream)stream, args, NULL);
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif

    return ret;
  }
}

void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle, CUmodule *module) {
  int ret;

  // opt is incompatible with allContextsEnabled, so assume it is opt (handle is 0)
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      void **hdl = __cudaRegisterFatBinary(fatCubin);
      __cudaInitModule(hdl);
      module[i] = NULL;
      ret = cuModuleLoadData(&(module[i]), (void *)fatCubin->ptr);
      assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
    }
    // reset back
    cudaSetDevice(__internal_getCurrentDevice());
  } else {
    // opt passes no handles, so we have to register
    if (handle == 0) {
      handle = __cudaRegisterFatBinary(fatCubin);
    }
    __cudaInitModule(handle);
    module[0] = NULL;
    ret = cuModuleLoadData(&module[0], (void *)fatCubin->ptr);
    assert((ret == CUDA_SUCCESS || ret == CUDA_ERROR_NO_BINARY_FOR_GPU) && "Module load failed");
  }
}

CUresult __helper_cuModuleLoad(CUmodule *module, const char *fname) {
  if (__internal_allContextsEnabled()) {
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
  } else {
    return cuModuleLoad(module, fname);
  }
}

cudaError_t __helper_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  /*
  cudaDeviceSynchronize();
  ret = cudaGetLastError();
  printf("peek error:  %d\n", ret);
  if(ret) {
    printf("\n\n\n ### MEMCPY ERROR \n\n\n");
  }
  */

  int device;
  cudaGetDevice(&device);
  printf("__helper_cudaMemcpy at device [%d]\n", device);
  
  struct cudaPointerAttributes at;
  ret = cudaPointerGetAttributes(&at, __translate_ptr(dst));
  printf("curdvc %d  dst attr ret %d  type %d  device  %d    dvcptr %p hostptr %p\n", __internal_getCurrentDevice(), ret, at.type, 
      at.device, at.devicePointer, at.hostPointer);

  ret = cudaPointerGetAttributes(&at, __translate_ptr(src));
  printf("curdvc %d  src attr ret %d  type %d  device  %d    dvcptr %p hostptr %p\n", __internal_getCurrentDevice(), ret, at.type, 
      at.device, at.devicePointer, at.hostPointer);
  cudaGetLastError();



  ret = cudaMemcpy(__translate_ptr(dst), __translate_ptr(src), count, kind);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_cudaMemset(void *devPtr, int value, size_t count) {
  cudaError_t ret2;
  ret2 = cudaGetLastError();
  printf("__helper_cudaMemset  peek error:  %d\n", ret2);

  cudaError_t ret = cudaMemset(__translate_ptr(devPtr), value, count);
  printf("memset ret %d   input %p  val %d  count %u\n", ret, __translate_ptr(devPtr), value, count);

#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

/**
 * Look up the CUDA kernel function and save it in the list.
 */
void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule *module,
                                const char *deviceName) {
  uint32_t loopc;
  if (__internal_allContextsEnabled())
    loopc = __internal_getDeviceCount();
  else
    loopc = 1;

  for (int i = 0; i < loopc; i++) {
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
    // if (func->hostfunc != NULL) return;
    if (func->hostfunc[i] != NULL) {
      std::cerr << "----------------------- func->hostfunc[i] != NULL in __helper_register_function------------"
                << std::endl;
      continue;
    }

    // this call needs to be done in each context
    if (__internal_allContextsEnabled()) {
      // printf("  __helper_register_function  setting device to %d\n", i);
      cudaSetDevice(i);
    } 

    CUresult ret = cuModuleGetFunction(&func->cufunc[i], module[i], deviceName);
    if (ret != CUDA_SUCCESS) {
      LOG_ERROR << "cuModuleGetFunction fail with " << ret;
      throw std::runtime_error("failed to get module function");
    }

    // std::cerr << "*** __helper_register_function kernel at device slot " << i << " host func " << std::hex <<
    // (uintptr_t)hostFun << " -> device func " << (uintptr_t)func->cufunc[i]
    //  << " deviceName " << deviceName << std::endl;
    func->hostfunc[i] = (void *)hostFun;
    func->module[i] = module[i];
  }

  // reset back if necessary
  if (__internal_allContextsEnabled()) {
    cudaSetDevice(__internal_getCurrentDevice());
  }
}

cudaError_t __helper_func_get_attributes(struct cudaFuncAttributes *attr, struct fatbin_function *func,
                                         const void *hostFun) {
  uint32_t cur_dvc;
  if (__internal_allContextsEnabled())
    cur_dvc = __internal_getDeviceCount();
  else
    cur_dvc = 0;

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
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return static_cast<cudaError_t>(ret);
}

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor(int *numBlocks, struct fatbin_function *func,
                                                                    const void *hostFun, int blockSize,
                                                                    size_t dynamicSMemSize) {
  uint32_t cur_dvc;
  if (__internal_allContextsEnabled())
    cur_dvc = __internal_getDeviceCount();
  else
    cur_dvc = 0;

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
  cudaError_t ret = static_cast<cudaError_t>(
      cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func->cufunc[cur_dvc], blockSize, dynamicSMemSize));
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(int *numBlocks,
                                                                               struct fatbin_function *func,
                                                                               const void *hostFun, int blockSize,
                                                                               size_t dynamicSMemSize,
                                                                               unsigned int flags) {
  uint32_t cur_dvc;
  if (__internal_allContextsEnabled())
    cur_dvc = __internal_getDeviceCount();
  else
    cur_dvc = 0;

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

  cudaError_t ret = static_cast<cudaError_t>(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, func->cufunc[cur_dvc], blockSize, dynamicSMemSize, flags));
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDevicePrimaryCtxGetState(d, flags, active);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDevicePrimaryCtxRetain(pctx, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceGetName(char *name, int len, CUdevice dev) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDeviceGetName(name, len, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDeviceGetAttribute(pi, attrib, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDeviceGetUuid(uuid, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDevicePrimaryCtxSetFlags(d, flags);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  CUdevice d;
  cuDeviceGet(&d, __internal_getCurrentDeviceIndex());
  CUresult ret = cuDeviceTotalMem(bytes, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDeviceGetPCIBusId(pciBusId, len, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceComputeCapability(int *major, int *minor, CUdevice device) {
  CUdevice d;
  cuCtxGetDevice(&d);
  CUresult ret = cuDeviceComputeCapability(major, minor, d);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  cudaError_t ret = cudaGetDeviceProperties(prop, __internal_getCurrentDeviceIndex());
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
  cudaError_t ret = cudaDeviceGetAttribute(value, attr, __internal_getCurrentDeviceIndex());
#ifndef NDEBUG
  auto tid = gsl::narrow_cast<int>(syscall(SYS_gettid));
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

CUresult __helper_cuDeviceGet(CUdevice *device, int ordinal) {
  
  if (ordinal != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  } else {
    CUresult ret = cuDeviceGet(device, __internal_getCurrentDeviceIndex());
#ifndef NDEBUG
    auto tid = gsl::narrow_cast<int>(syscall(SYS_gettid));
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
    return ret;
  }
}

cudaError_t __helper_cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  if (__internal_allContextsEnabled()) {
    /*
    cudaEventCreateWithFlags(event, flags);
    printf("------ return event %p\n", *event);
    
    // add current
    uint32_t cur_dvc = __internal_getCurrentDevice();
    GPUMemoryServer::Client::getInstance().events_map[*event][cur_dvc] = *event;
    printf("------event dev [%d] :  %p\n", cur_dvc, *event);

    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      if (i == cur_dvc) continue;
      cudaSetDevice(i);
      cudaEvent_t new_event;
      cudaEventCreateWithFlags(&new_event, flags);
      GPUMemoryServer::Client::getInstance().events_map[*event][i] = new_event;

      printf("------event dev [%d] :  %p\n", i, new_event);
    }
    // reset back device and return OK
    cudaSetDevice(cur_dvc);
    */
    return (cudaError_t)0;
  }
  else
    return cudaEventCreateWithFlags(event, flags);
}

cudaError_t __helper_cudaEventDestroy(cudaEvent_t event) {
  if (__internal_allContextsEnabled()) {
    /*
    auto v = GPUMemoryServer::Client::getInstance().events_map[event];
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      cudaEventDestroy(v[i]);
    }
    GPUMemoryServer::Client::getInstance().events_map.erase(event);
    cudaSetDevice(__internal_getCurrentDevice());
    */
    return (cudaError_t)0;
  }
  else
    return cudaEventDestroy(event);
}

cudaError_t  __helper_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  if (__internal_allContextsEnabled()) {
    /*
    printf("input event:  %p\n", event);

    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      printf("------ record at %d :  %p -> %p\n", i, event, GPUMemoryServer::Client::getInstance().events_map[event][i]);
      cudaEventRecord(GPUMemoryServer::Client::getInstance().events_map[event][i], GPUMemoryServer::Client::getInstance().streams_map[stream][i]);
      cudaError_t  ret = cudaGetLastError();
      if (ret)
        printf("__helper_cudaEventRecord last error:  %d\n", ret);
    }
    cudaSetDevice(__internal_getCurrentDevice());
    */
    return (cudaError_t)0;
  }
  else
    return cudaEventRecord(event, stream);
}

cudaError_t  __helper_cudaEventSynchronize(cudaEvent_t event) {
  if (__internal_allContextsEnabled()) {
    cudaDeviceSynchronize();
    return (cudaError_t)0;
  }
  else {
    return cudaEventSynchronize(event);
  }
}

cudaEvent_t __helper_translate_event(cudaEvent_t event) {
  return __translate_event(event);
}

cudaStream_t __helper_translate_stream(cudaStream_t stream) {
  return __translate_stream(stream);
}

void* __helper_translate_ptr(void* ptr) {
  return __translate_ptr(ptr);
}

cudaError_t __helper_cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
  return cudaPointerGetAttributes(attributes, ptr);
}

cudnnStatus_t __helper_cudnnConvolutionForward_double(cudnnHandle_t handle, const double *alpha,
                                                      const cudnnTensorDescriptor_t xDesc, const void *x,
                                                      const cudnnFilterDescriptor_t wDesc, const void *w,
                                                      const cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                      size_t workSpaceSizeInBytes, const double *beta,
                                                      const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
}

cudnnStatus_t __helper_cudnnConvolutionForward_float(cudnnHandle_t handle, const float *alpha,
                                                     const cudnnTensorDescriptor_t xDesc, const void *x,
                                                     const cudnnFilterDescriptor_t wDesc, const void *w,
                                                     const cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                     size_t workSpaceSizeInBytes, const float *beta,
                                                     const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
}