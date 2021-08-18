#include "common/extensions/cudnn_optimization.h"

#include <absl/synchronization/mutex.h>
#include <fmt/core.h>
#include <glib.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include "common/endpoint_lib.hpp"
#include "worker/extensions/memory_server/client.hpp"

//WANTED: make this generic
template <class handle_type>
struct cudnnHandleSet {
    std::vector<handle_type> handles;

    void cleanup() {
        for (auto &h : handles) cudnnDestroy(h);
    }

    bool containsHandle(handle_type guest_handle) {
        if (__internal_allContextsEnabled()) {
            for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
                if (handles[gpuid] == guest_handle) return true;
            }
            return false;
        } else {
            return guest_handle == handles[0];
        }
    }

    handle_type getCurrentGPUHandle() {
        if (__internal_allContextsEnabled()) {
            auto current = __internal_getCurrentDevice();
            return handles[current];
        } else {
            return handles[0];
        }
    }

    ~cudnnHandleSet() { cleanup(); }

    static gint finder(gpointer a, gpointer b) {
        cudnnHandleSet* set = a;
        handle_type handle = b;
        //return 0 if it contains
        return a->containsHandle(b) ? 0 : 1;
    }
}

template <>
cudnnHandleSet<cudnnHandle_t>::cudnnHandleSet() {
// if we need to create one per gpu
    if (__internal_allContextsEnabled()) {
        handles.reserve(__internal_getDeviceCount());
        for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
            cudnnHandle_t cudnn_handle;
            cudnn_ret = cudnnCreate(&cudnn_handle);
            if (cudnn_ret == CUDNN_STATUS_SUCCESS)
                handles[gpuid] = cudnn_handle;
            else
                fprintf(stderr, "### Failed to create CUDNN handle!\n");
        }
    }
    // otherwise keep it simple
    else {
        cudnnHandle_t cudnn_handle;
        cudnn_ret = cudnnCreate(&cudnn_handle);
        if (cudnn_ret == CUDNN_STATUS_SUCCESS)
            handles[0] = cudnn_handle;
        else
            fprintf(stderr, "### Failed to create CUDNN handle!\n");
    }
}



absl::Mutex cudnn_handles_mu;
GQueue *used_cudnn_handles ABSL_GUARDED_BY(cudnn_handles_mu);
GQueue *idle_cudnn_handles ABSL_GUARDED_BY(cudnn_handles_mu);

absl::Mutex cublas_handles_mu;
GQueue *used_cublas_handles ABSL_GUARDED_BY(cublas_handles_mu);
GQueue *idle_cublas_handles ABSL_GUARDED_BY(cublas_handles_mu);

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
void guestlib_cudnn_opt_init(void) {}
void guestlib_cudnn_opt_fini(void) {}

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

/*
 *  cleanup functions
 */
void gq_delete(gpointer key, gpointer value, gpointer userdata) { delete value; }

void worker_cudnn_opt_cleanup(void) {
  // delete each element from queue
    {
        absl::MutexLock lk(&cudnn_handles_mu);
        g_queue_foreach(used_cudnn_handles, gq_delete, NULL);
        g_queue_free(used_cudnn_handles);

        g_queue_foreach(idle_cudnn_handles, gq_delete, NULL);
        g_queue_free(idle_cudnn_handles);
    }

    //TODO add cublas
}

/*
 *  pre creation function
 */

void worker_cudnn_opt_init(uint32_t n_handles) {
    used_cudnn_handles = g_queue_new();
    idle_cudnn_handles = g_queue_new();

    // create all queues
    {
        absl::MutexLock lk(&cudnn_handles_mu);
        for (int i = 0; i < n_handles; i++) {
            auto handleset = new cudnnHandleSet<cudnnHandle_t>();
            g_queue_push_tail(idle_cudnn_handles, (gpointer)handleset);
        }
    }

    // TODO: add cublas

    return;
}

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle) {
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUDNN_STATUS_SUCCESS);
#endif
  
    absl::MutexLock lk(&cudnn_handles_mu);

    if (g_queue_is_empty(idle_cudnn_handles)) {
        auto handleset = new cudnnHandleSet<cudnnHandle_t>();
        *handle = handleset->getCurrentGPUHandle();
        g_queue_push_tail(used_cudnn_handles, (gpointer) handleset);
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnHandleSet<cudnnHandle_t> *handleset = g_queue_pop_head(idle_cudnn_handles);
    *handle = handleset->getCurrentGPUHandle();
    g_queue_push_tail(used_cudnn_handles, (gpointer) handleset);
    return CUDNN_STATUS_SUCCESS;
}

cublasStatus_t __cublasCreate(cublasHandle_t *handle) {
    //TBD
}

// TODO: the handle is not freed
cublasStatus_t __helper_cublasDestroy(cublasHandle_t handle) {

}

// TODO: the handle is not freed
cudnnStatus_t __helper_cudnnDestroy(cudnnHandle_t handle) {
    absl::MutexLock lk(&cudnn_handles_mu);

    GList* el = g_queue_find_custom(used_cudnn_handles, handle, cudnnHandleSet<cudnnHandle_t>::finder);
    if (el == NULL) {
        std::cerr << " ### Tried to destroy a non-existent cudnn handle, I guess." << std::endl;
        return 1;
    }
    cudnnHandleSet<cudnnHandle_t>* set = el->data;
    //AFAIK this does not delete the pointer that is in the list, just the list element itself
    g_queue_delete_link(used_cudnn_handles, el);
    g_queue_push_tail(idle_cudnn_handles, (gpointer)set);
}

/*
 *      Pool cudnn descriptors below
 */ 
 

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
