#include "common/extensions/cudnn_optimization.h"

#include <fmt/core.h>
#include <glib.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include "common/endpoint_lib.hpp"

GQueue *cudnn_handles;
GQueue *cublas_handles;

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
void guestlib_cudnn_opt_init(void) {}
void guestlib_cudnn_opt_fini(void) {}

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

void worker_cudnn_opt_init(void) {
  cudnnHandle_t cudnn_handle;
  cudnnStatus_t cudnn_ret;
  cublasHandle_t cublas_handle;
  cublasStatus_t cublas_ret;
  int i;

  cuInit(0);

  /* Cache cudnn and cublas handles */
  cudnn_handles = g_queue_new();
  cublas_handles = g_queue_new();

  for (i = 0; i < CUDNN_HANDLE_POOL_SIZE; i++) {
    cudnn_ret = cudnnCreate(&cudnn_handle);
    if (cudnn_ret == CUDNN_STATUS_SUCCESS)
      g_queue_push_tail(cudnn_handles, (gpointer)cudnn_handle);
    else {
      fprintf(stderr, "Failed to create CUDNN handle\n");
      break;
    }
  }

  for (i = 0; i < CUBLAS_HANDLE_POOL_SIZE; i++) {
    cublas_ret = cublasCreate(&cublas_handle);
    if (cublas_ret == CUBLAS_STATUS_SUCCESS)
      g_queue_push_tail(cublas_handles, (gpointer)cublas_handle);
    else {
      fprintf(stderr, "Failed to create CUBLAS handle\n");
      break;
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

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle) {
  if (g_queue_is_empty(cudnn_handles)) {
    cudnnStatus_t ret = cudnnCreate(handle);
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
    return ret;
  }

  *handle = (cudnnHandle_t)g_queue_pop_head(cudnn_handles);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUDNN_STATUS_SUCCESS);
#endif
  return CUDNN_STATUS_SUCCESS;
}

cublasStatus_t __cublasCreate(cublasHandle_t *handle) {
  if (g_queue_is_empty(cublas_handles)) {
    cublasStatus_t ret = cublasCreate(handle);
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, ret);
#endif
    return ret;
  }

  *handle = (cublasHandle_t)g_queue_pop_head(cublas_handles);
#ifndef NDEBUG
  auto tid = __gettid();
  std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUBLAS_STATUS_SUCCESS);
#endif
  return CUBLAS_STATUS_SUCCESS;
}
