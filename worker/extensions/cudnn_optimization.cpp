#include "common/extensions/cudnn_optimization.h"
#include "worker/extensions/memory_server/client.hpp"
#include <fmt/core.h>
#include <glib.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include "common/endpoint_lib.hpp"

//GQueue *cudnn_handles;
//GQueue *cublas_handles;

std::map<uint32_t, GQueue*> cudnn_handles;
std::map<uint32_t, GQueue*> cublas_handles;

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
void guestlib_cudnn_opt_init(void) {}
void guestlib_cudnn_opt_fini(void) {}

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

/*
 *  cleanup functions
 */
void destroy_cublas_handle(gpointer key, gpointer value, gpointer userdata) {
  cublasDestroy((cublasHandle_t) value);
}

void destroy_cudnn_handle(gpointer key, gpointer value, gpointer userdata) {
  cudnnDestroy((cudnnHandle_t) value);
}

void worker_cudnn_opt_cleanup(void) {
  for (auto el : cudnn_handles) {
    g_queue_foreach(el.second, destroy_cudnn_handle, NULL);
    g_queue_free(el.second);
  }

  for (auto el : cublas_handles) {
    g_queue_foreach(el.second, destroy_cublas_handle, NULL);
    g_queue_free(el.second);
  }

  cudnn_handles.clear();
  cublas_handles.clear();
}

/*
 *  pre creation function
 */

void worker_cudnn_opt_init(uint32_t n_handles) {
  cudnnHandle_t cudnn_handle;
  cudnnStatus_t cudnn_ret;
  cublasHandle_t cublas_handle;
  cublasStatus_t cublas_ret;

  //better be explicit and have duplicate code than have a total mess
  if (__internal_allContextsEnabled()) {
    //create all queues
    for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
      cudnn_handles[gpuid] = g_queue_new();
      cublas_handles[gpuid] = g_queue_new();
    }
    
    for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
      for (int i = 0; i < n_handles; i++) {
        cudnn_ret = cudnnCreate(&cudnn_handle);
        if (cudnn_ret == CUDNN_STATUS_SUCCESS)
          g_queue_push_tail(cudnn_handles[gpuid], (gpointer)cudnn_handle);
        else {
          fprintf(stderr, "Failed to create CUDNN handle\n");
          break;
        }
      }
    }

    for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
      for (int i = 0; i < n_handles; i++) {
        cublas_ret = cublasCreate(&cublas_handle);
        if (cublas_ret == CUBLAS_STATUS_SUCCESS)
          g_queue_push_tail(cublas_handles[gpuid], (gpointer)cublas_handle);
        else {
          fprintf(stderr, "Failed to create CUBLAS handle\n");
          break;
        }
      }
    }
  }
  //normal case, only one context will be used
  else {
    //create only one position on map
    uint32_t gpuid = __internal_getCurrentDevice();
    cudnn_handles[gpuid] = g_queue_new();
    cublas_handles[gpuid] = g_queue_new();

    for (int i = 0; i < n_handles; i++) {
      cudnn_ret = cudnnCreate(&cudnn_handle);
      if (cudnn_ret == CUDNN_STATUS_SUCCESS)
        g_queue_push_tail(cudnn_handles[gpuid], (gpointer)cudnn_handle);
      else {
        fprintf(stderr, "Failed to create CUDNN handle\n");
        break;
      }
    }

    for (int i = 0; i < n_handles; i++) {
      cublas_ret = cublasCreate(&cublas_handle);
      if (cublas_ret == CUBLAS_STATUS_SUCCESS)
        g_queue_push_tail(cublas_handles[gpuid], (gpointer)cublas_handle);
      else {
        fprintf(stderr, "Failed to create CUBLAS handle\n");
        break;
      }
    }
  }

  return;
}

cudnnStatus_t __pool_cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  size_t i;
  cudnnConvolutionDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &convDesc[i];
    res = cudnnCreateConvolutionDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyConvolutionDescriptor(convDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  size_t i;
  cudnnPoolingDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &poolingDesc[i];
    res = cudnnCreatePoolingDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyPoolingDescriptor(poolingDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  size_t i;
  cudnnTensorDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &tensorDesc[i];
    res = cudnnCreateTensorDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyTensorDescriptor(tensorDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  size_t i;
  cudnnFilterDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &filterDesc[i];
    res = cudnnCreateFilterDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyFilterDescriptor(filterDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count) {
  size_t i;
  cudnnReduceTensorDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &reduceTensorDesc[i];
    res = cudnnCreateReduceTensorDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) {
      return res;
    }
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) {
      return res;
    }
  }

  return res;
}

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle) {
  //printf("### ### ### __cudnnCreate\n");
  uint32_t gpuid = __internal_getCurrentDevice();
  
  if (g_queue_is_empty(cudnn_handles[gpuid])) {
    cudnnStatus_t ret = cudnnCreate(handle);
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
    printf("  queue empty, creating, returned %d\n", ret);
    return ret;
  }

  *handle = (cudnnHandle_t)g_queue_pop_head(cudnn_handles[gpuid]);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUDNN_STATUS_SUCCESS);
#endif
  return CUDNN_STATUS_SUCCESS;
}

cublasStatus_t __cublasCreate(cublasHandle_t *handle) {
  //printf("### ### ### __cublasCreate\n");
  uint32_t gpuid = __internal_getCurrentDevice();
  
  if (g_queue_is_empty(cublas_handles[gpuid])) {
    cublasStatus_t ret = cublasCreate(handle);
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
printf("  queue empty, creating, returned %d\n", ret);
    return ret;
  }

  *handle = (cublasHandle_t)g_queue_pop_head(cublas_handles[gpuid]);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUBLAS_STATUS_SUCCESS);
#endif
  return CUBLAS_STATUS_SUCCESS;
}
