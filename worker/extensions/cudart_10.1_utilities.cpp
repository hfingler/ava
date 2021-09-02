#include "common/extensions/cudart_10.1_utilities.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fmt/core.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <gsl/gsl>
#include <iostream>
#include <thread>

#include "common/declaration.h"
#include "common/logging.h"
#include "cudart_nw_internal.h"
#include "extensions/memory_server/client.hpp"
#include "common/extensions/cudnn_optimization.h"

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

uint32_t __internal_getCurrentDeviceIndex() { return __internal_getCurrentDevice(); }

cudaError_t __helper_cuda_memcpy_async_host_to_host(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, __translate_stream(stream));
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_host_to_device(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), src, count, cudaMemcpyHostToDevice, __translate_stream(stream));
  
#ifndef NDEBUG
  //cudaDeviceSynchronize();
  if(ret) 
    std::cerr << "\n ### __helper_cuda_memcpy_async_host_to_device ERROR after " << ret << "\n";
#endif
  
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_host(void *dst, const void *src, size_t count, cudaStream_t stream) {
#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret1 = cudaGetLastError();
    if(ret1) 
      std::cerr << "\n ### __helper_cuda_memcpy_async_device_to_host error before " << ret1 << "\n";
#endif
  
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst, __translate_ptr(src), count, cudaMemcpyDeviceToHost, __translate_stream(stream));

#ifndef NDEBUG
    //cudaDeviceSynchronize();
    if(ret) 
      std::cerr << "\n ### __helper_cuda_memcpy_async_device_to_host error after " << ret << "\n";
#endif

  return ret;
}

cudaError_t __helper_cuda_memcpy_async_device_to_device(void *dst, const void *src, size_t count, cudaStream_t stream) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(__translate_ptr(dst), __translate_ptr(src), count, cudaMemcpyDeviceToDevice, __translate_stream(stream));
  
#ifndef NDEBUG
  //cudaDeviceSynchronize();
  if(ret) 
    std::cerr << "\n ### __helper_cuda_memcpy_async_host_to_device ERROR after " << ret << "\n";
#endif
  
  return ret;
}

cudaError_t __helper_cuda_memcpy_async_default(void *dst, const void *src, size_t count, cudaStream_t stream,
                                               bool dst_is_gpu, bool src_is_gpu) {
  cudaError_t ret;
  ret = cudaMemcpyAsync(dst_is_gpu ? __translate_ptr(dst) : dst, src_is_gpu ? __translate_ptr(src) : src, count,
                        cudaMemcpyDefault, __translate_stream(stream));
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

//nothing is translated at this point
cublasStatus_t  __helper_cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
  if (__internal_allContextsEnabled()) {
    uint32_t real_cur_dvc = GPUMemoryServer::Client::getInstance().current_device;
    //save real current gpu
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      //we need to fakely switch current_device because that's what the translation
      //functions use
      GPUMemoryServer::Client::getInstance().current_device = i;
      cublasStatus_t err = cublasSetStream(__get_cublas_handle(handle), __translate_stream(streamId));
      if (err > 0)
        std::cerr << " ### error on cublas set stream for gpu " << i << std::endl;
    }
    // reset back device and return OK
    GPUMemoryServer::Client::getInstance().current_device = real_cur_dvc;
    cudaSetDevice(__internal_getCurrentDevice());
    return (cublasStatus_t)0;
  } else {
    cublasStatus_t err = cublasSetStream(handle, streamId);
    return err;
  }
}

cudnnStatus_t __helper_cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  if (__internal_allContextsEnabled()) {
    uint32_t real_cur_dvc = GPUMemoryServer::Client::getInstance().current_device;
    //save real current gpu
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      //we need to fakely switch current_device because that's what the translation
      //functions use
      GPUMemoryServer::Client::getInstance().current_device = i;
      cudnnStatus_t err = cudnnSetStream(__get_cudnn_handle(handle), __translate_stream(streamId));
      if (err > 0)
        std::cerr << " ### error on cublas set stream for gpu " << i << std::endl;
    }
    // reset back device and return OK
    GPUMemoryServer::Client::getInstance().current_device = real_cur_dvc;
    cudaSetDevice(__internal_getCurrentDevice());
    return (cudnnStatus_t)0;
  } else {
    cudnnStatus_t err = cudnnSetStream(handle, streamId);
    return err;
  }
}

cublasStatus_t __helper_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                       int n, int k, const float *alpha, /* host or device pointer */
                                       const float *A, int lda, const float *B, int ldb,
                                       const float *beta, /* host or device pointer */
                                       float *C, int ldc, bool alpha_is_gpu, bool beta_is_gpu) {

#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret1 = cudaGetLastError();
    if(ret1) 
      std::cerr << "\n ### __helper_cublasSgemm_v2 error before " << ret1 << "\n";
#endif
  
  auto ret = cublasSgemm(handle, transa, transb, m, n, k, 
                    alpha_is_gpu ? __translate_ptr(alpha) : alpha, 
                    __translate_ptr(A), lda, __translate_ptr(B), ldb,
                    beta_is_gpu ? __translate_ptr(beta) : beta, 
                    __translate_ptr(C), ldc);

#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);

  //cudaDeviceSynchronize();
  cudaError_t ret2 = cudaGetLastError();
  if(ret2) 
    std::cerr << "\n ### __helper_cublasSgemm_v2 ERROR " << ret2 << "\n";
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
      if(cudaStreamCreateWithPriority(&new_stream, flags, priority) > 0 )
        printf("### ERROR ON __helper_create_stream\n");
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

CUresult __helper_destroy_custream(CUstream stream) {
  CUresult ret;
  ret = (CUresult) __helper_destroy_stream((cudaStream_t) stream);
}

cudaError_t __helper_destroy_stream(cudaStream_t stream) {
  if (stream == 0) return 0;
  
  cudaError_t ret;
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      //TODO: check if stream is in the map
      cudaStreamDestroy(GPUMemoryServer::Client::getInstance().streams_map[stream][i]); 
    }
    // reset back device and return OK
    cudaSetDevice(__internal_getCurrentDevice());
    
    GPUMemoryServer::Client::getInstance().streams_map.erase(stream);
    return (cudaError_t)0;
  } else {
    return cudaStreamDestroy(stream);
  }
}

cudaError_t __helper_cudaStreamSynchronize_sync(cudaStream_t stream) {
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      //cudaDeviceSynchronize();
      cudaStreamSynchronize(GPUMemoryServer::Client::getInstance().streams_map[stream][i]);
      return 0;
    }
  } else {
    return cudaStreamSynchronize(stream);
  }
}

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream) {
#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret2 = cudaGetLastError();
    if(ret2) 
      std::cerr << "\n ### __helper_launch_kernel ERROR BEFORE " << ret2 << "\n";
#endif

  // this might trigger and to migration
  __internal_kernelIn();

  uint32_t cur_dvc;
  if (__internal_allContextsEnabled())
    cur_dvc = __internal_getCurrentDevice();
  else
    cur_dvc = 0;

#ifndef NDEBUG
  std::cerr << "__helper_launch_kernel on device slot " << cur_dvc << std::endl;
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

#ifndef NDEBUG
  __helper_print_kernel_info(func, args);
#endif

  // std::cerr << "function metadata (" << (void *)func << ") for local " << func->hostfunc[cur_dvc] << ", cufunc "
  //          << (void *)func->cufunc[cur_dvc] << ", argc " << func->argc << std::endl;

  cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;
  // if we need to figure out the correct context to get function
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < func->argc; i++) {
      // TODO: we need something that says if something is a pointer or not
      if (func->args[i].size == 8) {
        *((void **)args[i]) = __translate_ptr(*((void **)args[i]));
      }
    }

    // auto start = std::chrono::steady_clock::now();
    ret = (cudaError_t)cuLaunchKernel(func->cufunc[cur_dvc], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                      blockDim.z, sharedMem, (CUstream)__translate_stream(stream), args, NULL);
    // auto end = std::chrono::steady_clock::now();
    // std::cerr << "???" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

#ifndef NDEBUG
    std::cerr << ">>> cuLaunchKernel returned " << ret << std::endl;
    cudaDeviceSynchronize();
    cudaError_t ret2 = cudaGetLastError();
    if(ret2) 
      std::cerr << "\n ### __helper_launch_kernel ERROR after sync " << ret2 << "\n";
#endif

    // TODO: fix
    __internal_kernelOut();
    return ret;
  }

  // if not with migration, just get over it and do
  else {
    // auto start = std::chrono::steady_clock::now();
    ret = (cudaError_t)cuLaunchKernel(func->cufunc[0], gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                      blockDim.z, sharedMem, (CUstream)stream, args, NULL);
    // auto end = std::chrono::steady_clock::now();
    // std::cerr << "???" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

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
    // fprintf(stderr, "resetting device to %d\n", __internal_getCurrentDevice());
    cudaSetDevice(__internal_getCurrentDevice());
    return ret;
  } else {
    return cuModuleLoad(module, fname);
  }
}

cudaError_t __helper_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;

#ifndef NDEBUG
    //cudaDeviceSynchronize();
    //int d;
    //cudaGetDevice(&d);
    //std::cerr << "\n ### __helper_cudaMemcpy at DEVICE " << d << " src/dest: " << std::hex << src << " " << std::hex << dst << "\n";
    cudaError_t ret2 = cudaGetLastError();
    if(ret2) 
      std::cerr << "\n ### __helper_cudaMemcpy LAST ERROR " << ret2 << "\n";

#endif

  // auto start = std::chrono::steady_clock::now();
  ret = cudaMemcpy(__translate_ptr(dst), __translate_ptr(src), count, kind);
  // auto end = std::chrono::steady_clock::now();
  // std::cerr << ";;;" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

#ifndef NDEBUG

  //cudaDeviceSynchronize();
  cudaError_t ret3 = cudaGetLastError();
  if(ret3) 
    std::cerr << "\n ### __helper_cudaMemcpy CAUSED ERROR " << ret3 << "\n";

#endif

#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
  return ret;
}

cudaError_t __helper_cudaMemset(void *devPtr, int value, size_t count) {

  // auto start = std::chrono::steady_clock::now();
  cudaError_t ret = cudaMemset(__translate_ptr(devPtr), value, count);
  // printf("memset ret %d   input %p  val %d  count %u\n", ret, __translate_ptr(devPtr), value, count);
  // auto end = std::chrono::steady_clock::now();
  // std::cerr << ":::" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

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
      //std::cerr << "----------------------- func->hostfunc[i] != NULL in __helper_register_function------------"
      //          << std::endl;
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
    cudaEventCreateWithFlags(event, flags);

    // add current
    uint32_t cur_dvc = __internal_getCurrentDevice();
    GPUMemoryServer::Client::getInstance().events_map[*event][cur_dvc] = *event;

    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      if (i == cur_dvc) continue;
      cudaSetDevice(i);
      cudaEvent_t new_event;
      cudaEventCreateWithFlags(&new_event, flags);
      GPUMemoryServer::Client::getInstance().events_map[*event][i] = new_event;
    }
    // reset back device and return OK
    cudaSetDevice(cur_dvc);

    return (cudaError_t)0;
  } else
    return cudaEventCreateWithFlags(event, flags);
}

cudaError_t __helper_cudaEventDestroy(cudaEvent_t event) {
  if (__internal_allContextsEnabled()) {
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      cudaSetDevice(i);
      cudaEventDestroy(GPUMemoryServer::Client::getInstance().events_map[event][i]);
    }
    GPUMemoryServer::Client::getInstance().events_map[event].clear();
    GPUMemoryServer::Client::getInstance().events_map.erase(event);
    cudaSetDevice(__internal_getCurrentDevice());
    return (cudaError_t)0;
  } else
    return cudaEventDestroy(event);
}

cudaError_t __helper_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  if (__internal_allContextsEnabled()) {
    uint32_t cur_dvc = __internal_getCurrentDevice();
    cudaEvent_t real_event = GPUMemoryServer::Client::getInstance().events_map[event][cur_dvc];
    return cudaEventRecord(real_event, __translate_stream(stream));
  } else
    return cudaEventRecord(event, stream);
}

cudaError_t __helper_cudaEventSynchronize(cudaEvent_t event) {
  if (__internal_allContextsEnabled()) {

    //testing
    /*
    for (int i = 0; i < __internal_getDeviceCount(); i++) {
      //testing
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }
    */
    cudaSetDevice(__internal_getCurrentDevice());
    //cudaDeviceSynchronize();
    uint32_t cur_dvc = __internal_getCurrentDevice();
    cudaEvent_t real_event = GPUMemoryServer::Client::getInstance().events_map[event][cur_dvc];
    cudaEventSynchronize(real_event);
  
    return (cudaError_t)0;
  } else {
    return cudaEventSynchronize(event);
  }
}

cudaEvent_t __helper_translate_event(cudaEvent_t event) { return __translate_event(event); }

cudaStream_t __helper_translate_stream(cudaStream_t stream) { return __translate_stream(stream); }

void *__helper_translate_ptr(void *ptr) { return __translate_ptr(ptr); }

const void *__helper_translate_const_ptr(const void *ptr) { return __translate_ptr(ptr); }

cudaError_t __helper_cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret2 = cudaGetLastError();
    if(ret2) 
      std::cerr << "\n### __helper_cudaPointerGetAttributes BEFORE ERROR " << ret2 << "\n";
#endif

  cudaError_t err = cudaPointerGetAttributes(attributes, __translate_ptr(ptr));
 
#ifndef NDEBUG
    /*
    bool c1 = GPUMemoryServer::Client::getInstance().pointerIsMapped(ptr);
    bool c2 = GPUMemoryServer::Client::getInstance().pointerIsMapped(__translate_ptr(ptr));
    std::cerr << " ### __helper_cudaPointerGetAttributes mapped in current / mapped translated: " << 
        c1 << " / " << c2 << "\n";
    if (err) {
        std::cerr << "\n ### __helper_cudaPointerGetAttributes error: " << err << "\n";
    }
    */
    //cudaDeviceSynchronize();
    cudaError_t ret3 = cudaGetLastError();
    if(ret3) 
      std::cerr << "__helper_cudaPointerGetAttributes AFTER ERROR (if 1, probably fine) " << ret3 << "\n";
#endif

  return err;
}

//handle translated
cudnnStatus_t __helper_cudnnConvolutionForward_double(cudnnHandle_t handle, const double *alpha,
                                                      const cudnnTensorDescriptor_t xDesc, const void *x,
                                                      const cudnnFilterDescriptor_t wDesc, const void *w,
                                                      const cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                      size_t workSpaceSizeInBytes, const double *beta,
                                                      const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnConvolutionForward(handle, alpha, xDesc, __translate_ptr(x), wDesc, __translate_ptr(w), 
            convDesc, algo, __translate_ptr(workSpace), workSpaceSizeInBytes, beta, yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t __helper_cudnnConvolutionForward_float(cudnnHandle_t handle, const float *alpha,
                                                     const cudnnTensorDescriptor_t xDesc, const void *x,
                                                     const cudnnFilterDescriptor_t wDesc, const void *w,
                                                     const cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                     size_t workSpaceSizeInBytes, const float *beta,
                                                     const cudnnTensorDescriptor_t yDesc, void *y) {

#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret2 = cudaGetLastError();
    if(ret2) 
      std::cerr << "\n### __helper_cudnnConvolutionForward_float BEFORE ERROR " << ret2 << "\n";
#endif
  cudnnStatus_t ret = cudnnConvolutionForward(handle, alpha, xDesc, __translate_ptr(x), wDesc, __translate_ptr(w), convDesc, 
            algo, __translate_ptr(workSpace), workSpaceSizeInBytes, beta, yDesc, __translate_ptr(y));

#ifndef NDEBUG
    //cudaDeviceSynchronize();
    cudaError_t ret3 = cudaGetLastError();
    if(ret3) 
      std::cerr << "\n### __helper_cudnnConvolutionForward_float BEFORE ERROR " << ret3 << "\n";
#endif

  return ret;
}

//handle translated
cudnnStatus_t cudnnBatchNormalizationForwardInference_float(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const float *alpha, /* alpha[0] = result blend factor */
    const float *beta,                                                   /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x,                  /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,                        /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon) {
  
#ifndef NDEBUG
  //cudaDeviceSynchronize();
  cudaError_t ret3 = cudaGetLastError();
  if(ret3) 
    std::cerr << "\n ### cudnnBatchNormalizationForwardInference_float  before " << ret3 << "\n";
#endif
  cudnnStatus_t ret;
  ret =  cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, __translate_ptr(x), yDesc, 
          __translate_ptr(y), __translate_ptr(bnScaleBiasMeanVarDesc), __translate_ptr(bnScale), __translate_ptr(bnBias), 
          __translate_ptr(estimatedMean), __translate_ptr(estimatedVariance), epsilon);

#ifndef NDEBUG
  //cudaDeviceSynchronize();
  cudaError_t ret2 = cudaGetLastError();
  if(ret2) 
    std::cerr << "\n ### cudnnBatchNormalizationForwardInference_float  after " << ret2 << "\n";
#endif

  return ret;
}

//handle translated
cudnnStatus_t cudnnBatchNormalizationForwardInference_double(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const double *alpha, /* alpha[0] = result blend factor */
    const double *beta,                                                   /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x,                   /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,                         /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon) {
  return cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, __translate_ptr(x), yDesc, __translate_ptr(y), 
                bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
}

//handle translated
cudnnStatus_t cudnnPoolingForward_float(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                        const float *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                        const float *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, __translate_ptr(x), beta, yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t cudnnPoolingForward_double(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                         const double *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                         const double *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, __translate_ptr(x), beta, yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t cudnnAddTensor_double(cudnnHandle_t handle, const double *alpha, const cudnnTensorDescriptor_t aDesc,
                                    const void *A, const double *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  return cudnnAddTensor(handle, alpha, aDesc, __translate_ptr(A), beta, cDesc, __translate_ptr(C));
}

//handle translated
cudnnStatus_t cudnnAddTensor_float(cudnnHandle_t handle, const float *alpha, const cudnnTensorDescriptor_t aDesc,
                                   const void *A, const float *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
#ifndef NDEBUG
  //cudaDeviceSynchronize();
  cudaError_t ret2 = cudaGetLastError();
  if(ret2) 
    std::cerr << "\n ### cudnnAddTensor_float Before " << ret2 << "\n";
#endif

  cudnnStatus_t ret = cudnnAddTensor(handle, alpha, aDesc, __translate_ptr(A), beta, cDesc, __translate_ptr(C));

#ifndef NDEBUG
  //cudaDeviceSynchronize();
  cudaError_t ret3 = cudaGetLastError();
  if(ret3) 
    std::cerr << "\n ### cudnnAddTensor_float After " << ret3 << "\n";
#endif
}

//handle translated
cudnnStatus_t cudnnReduceTensor_double(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                       void *indices, size_t indicesSizeInBytes, void *workspace,
                                       size_t workspaceSizeInBytes, const double *alpha,
                                       const cudnnTensorDescriptor_t aDesc, const void *A, const double *beta,
                                       const cudnnTensorDescriptor_t cDesc, void *C) {
  return cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes,
                           alpha, aDesc, A, beta, cDesc, C);
}

//handle translated
cudnnStatus_t cudnnReduceTensor_float(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                      void *indices, size_t indicesSizeInBytes, void *workspace,
                                      size_t workspaceSizeInBytes, const float *alpha,
                                      const cudnnTensorDescriptor_t aDesc, const void *A, const float *beta,
                                      const cudnnTensorDescriptor_t cDesc, void *C) {
  return cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, __translate_ptr(workspace), workspaceSizeInBytes,
                           alpha, aDesc, __translate_ptr(A), beta, cDesc, __translate_ptr(C));
}

//handle translated
cudnnStatus_t cudnnConvolutionBiasActivationForward_double(
    cudnnHandle_t handle, const double *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const double *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, __translate_ptr(x), wDesc, __translate_ptr(w), convDesc, 
              algo, __translate_ptr(workSpace), workSpaceSizeInBytes, alpha2, zDesc, __translate_ptr(z), biasDesc, bias, activationDesc,
              yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t cudnnConvolutionBiasActivationForward_float(
    cudnnHandle_t handle, const float *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const float *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, __translate_ptr(x), wDesc, __translate_ptr(w), convDesc, algo, 
              __translate_ptr(workSpace), workSpaceSizeInBytes, alpha2, zDesc, __translate_ptr(z), biasDesc, bias, activationDesc,
              yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t cudnnScaleTensor_float(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                     const float *alpha) {
  return cudnnScaleTensor(handle, yDesc, __translate_ptr(y), alpha);
}

//handle translated
cudnnStatus_t cudnnScaleTensor_double(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                      const double *alpha) {
  return cudnnScaleTensor(handle, yDesc, __translate_ptr(y), alpha);
}

//handle translated
cudnnStatus_t cudnnSoftmaxForward_double(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                         const double *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                         const double *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, __translate_ptr(x), beta, yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t cudnnSoftmaxForward_float(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                        const float *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                        const float *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  return cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, __translate_ptr(x), beta, yDesc, __translate_ptr(y));
}

//handle translated
cudnnStatus_t __helper_cudnnFindConvolutionForwardAlgorithmEx(
            cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc,
            const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y,
            const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
            size_t workSpaceSizeInBytes) {

#ifndef NDEBUG
  cudaDeviceSynchronize();
  cudaError_t ret3 = cudaGetLastError();
  if(ret3) 
    std::cerr << "\n ### __helper_cudnnFindConvolutionForwardAlgorithmEx  before " << ret3 << "\n";
#endif

  cudnnStatus_t err = cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, __translate_ptr(x), wDesc, 
    __translate_ptr(w), convDesc, yDesc, __translate_ptr(y), requestedAlgoCount, returnedAlgoCount, 
    perfResults, __translate_ptr(workSpace), workSpaceSizeInBytes);

#ifndef NDEBUG
  std::cerr << "__helper_cudnnFindConvolutionForwardAlgorithmEx returned  " << err << "\n";
  if( err > 0) {
    std::cerr << " ### ERROR ON __helper_cudnnFindConvolutionForwardAlgorithmEx  \n";
  }

  cudaDeviceSynchronize();
  cudaError_t ret2 = cudaGetLastError();
  if(ret2) 
    std::cerr << "\n ### __helper_cudnnFindConvolutionForwardAlgorithmEx  before " << ret2 << "\n";
#endif

  return err;
}

cublasStatus_t __helper_cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float *alpha,                                                /* host or device pointer */
    const float *A, int lda, long long int strideA,                    /* purposely signed */
    const float *B, int ldb, long long int strideB, const float *beta, /* host or device pointer */
    float *C, int ldc, long long int strideC, int batchCount,  bool alpha_is_gpu, bool beta_is_gpu) {

  return cublasSgemmStridedBatched(__get_cublas_handle(handle), transa, transb, m, n, k,
    alpha_is_gpu ? __translate_ptr(alpha) : alpha, 
    (const float*)__helper_translate_ptr((const void*)A), 
    lda, strideA, 
    (const float*)__helper_translate_ptr((const void*)B), ldb, 
    strideB, 
    beta_is_gpu ? __translate_ptr(beta) : beta, 
    (const float*)__helper_translate_ptr((const void*)C), 
    ldc, strideC, batchCount);
}