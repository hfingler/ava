#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <string.h>
#include <stdlib.h>
#include <zmq.h>
#include <errno.h>
#include <memory>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/common_context.h"
#include "extensions/memory_server/client.hpp"
#include "extensions/memory_server/common.hpp"

/**************************************
 * 
 *    Global functions, used by spec and helper functions. Mostly translate a global call to a client call
 * 
 * ************************************/
std::atomic<uint32_t> pending_cmds(0);
std::atomic<bool> thread_block(false);
std::mutex thread_block_mutex;
std::condition_variable thread_block_cv;

void wait_for_cmd_drain() {
    thread_block = true;
    //we count as a cmd, so wait for 1 pending
    while (pending_cmds != 1) ;
}

void release_cmd_handlers() {
  std::unique_lock<std::mutex> lk(thread_block_mutex);
  thread_block = false;
  thread_block_cv.notify_all();
}

void __cmd_handle_in() {
    if (thread_block) {
        std::unique_lock<std::mutex> lk(thread_block_mutex);
        while (thread_block)
            thread_block_cv.wait(lk);
    }
    pending_cmds++;
#ifndef NDEBUG
    //std::cerr << "in: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

void __cmd_handle_out() {
    pending_cmds--;
#ifndef NDEBUG
    //std::cerr << "out: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

bool allContextsEnabled;
bool __internal_allContextsEnabled() {
    return allContextsEnabled;
}
bool __internal_setAllContextsEnabled(bool f) {
    allContextsEnabled = f;
}

int32_t __internal_getDeviceCount() {
    return GPUMemoryServer::Client::getInstance().device_count;
}

void __internal_setDeviceCount(uint32_t dc) {
    GPUMemoryServer::Client::getInstance().device_count = dc;
}

uint32_t __internal_getCurrentDevice() {
    return GPUMemoryServer::Client::getInstance().current_device;
}

void* __translate_ptr(void* ptr) {
    if (__internal_allContextsEnabled())
        return GPUMemoryServer::Client::getInstance().translateDevicePointer(ptr);
    else
        return ptr;
}

const void* __translate_ptr(const void* ptr) {
    if (__internal_allContextsEnabled()) 
        return const_cast<const void*>(GPUMemoryServer::Client::getInstance().translateDevicePointer(ptr));
    else
        return ptr;
}

void __internal_kernelIn() {
    GPUMemoryServer::Client::getInstance().kernelIn();
}

void __internal_kernelOut() {
    GPUMemoryServer::Client::getInstance().kernelOut();
}

CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    CUresult cur;
    GPUMemoryServer::Client::getInstance().kernelIn();
    cur = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
    GPUMemoryServer::Client::getInstance().kernelOut();
    return cur;
}

cudaError_t __internal_cudaMalloc(void **devPtr, size_t size) {
    return GPUMemoryServer::Client::getInstance().localMalloc(devPtr, size);
}

cudaError_t __internal_cudaFree(void *devPtr) {
    return GPUMemoryServer::Client::getInstance().localFree(devPtr);
}

cudaStream_t __translate_stream(cudaStream_t key) {
    if (__internal_allContextsEnabled()) {
        if (key == 0) return 0;
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().streams_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

cudaEvent_t __translate_event(cudaEvent_t key) {
    if (__internal_allContextsEnabled()) {
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().events_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

/**************************************
 * 
 *    Functions for GPUMemoryServer namespace, Client class
 * 
 * ************************************/

namespace GPUMemoryServer {

    //uint32_t kc = 0;
    Client::Client() {
        og_device = -1;
        context = zmq_ctx_new();
        for (int i = 0 ; i < 4; i++) {
            sockets[i] = 0;
        }
        migrated_type = Migration::NOPE;
     }

     Client::~Client() {
        for (int i = 0 ; i < device_count ; i++) {
            zmq_close(sockets[i]);
        }
        //zmq_ctx_destroy(context);
    }

    void Client::connectToGPU(uint32_t gpuid) {
        std::ostringstream stringStream;
        stringStream << GPUMemoryServer::get_base_socket_path() << gpuid;
        sockets[gpuid] = zmq_socket(context, ZMQ_REQ);

        while (1) { 
            int ret = zmq_connect(sockets[gpuid], stringStream.str().c_str());
            if (ret == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            printf(" !!! GPU Client couldn't connect to server [%d], this is WRONG! The server probably died due to a bug\n", gpuid);
        }
        printf("GPU Client succesfully connected to server [%d]\n", gpuid);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        int d;
        cudaGetDevice(&d);
        cudaError_t err = cudaMalloc(devPtr, size);
        local_allocs.emplace((uint64_t)*devPtr, std::make_unique<Client::LocalAlloc>(*devPtr, size, current_device));
        // printf("  malloc %p  size %lu  at device %d, ret=%d\n", *devPtr, size, d, err);
        //TODO add memory if we do it
        if (migrated_type == Migration::TOTAL) {
            //here's the issue.. if we have migrated memory, cudaMalloc may return a pointer that was
            //previously returned for the previous GPU. That's a conflict that messes us up pretty bad
            if (isInPointerMap(*devPtr)) {
                std::cerr << "### Pointer conflict after memory migration ..." << std::endl;
                uint64_t optr = *devPtr;
                
                //lets flip a high order bit, this will not be dereferenced anyway
                optr = optr ^ (1 << 62);
                if (isInPointerMap(optr)) {
                    printf("\n\n### FLIPING THE BIT WAS NOT ENOUGH, FIX IT!\n\n\n");
                }

                //add to the map
                pointer_map[optr].dstPtr = *devPtr;
                pointer_map[optr].size = size;
                //we now have a pointer that is not in the map
                *devPtr = optr;
            }
        }
        //reportMalloc(size); 
        return err;
    }

    void Client::reportMalloc(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    cudaError_t Client::localFree(void* devPtr) {
        //there is a use-after-free bug with execution (KERNEL) migration, where
        //if the app frees a buffer immediately after a kernel launch, an error 700
        //can occur due to the array being freed before the kernel finishes.
        //this will only happen in badly written code (like the kmeans we're testing with)
        //if it migrated, don't free now, it will be freed on exit
        if (migrated_type == Migration::KERNEL) {
            return cudaSuccess;
        }

        // cudaFree with nullptr does no operation
        if (devPtr == nullptr) {
            return cudaSuccess;
        }

        auto it = local_allocs.find((uint64_t)devPtr);
        //found it, remove and get out
        if (it != local_allocs.end()) {
            tryRemoveFromPointerMap(devPtr);
            //GPUMemoryServer::Client::getInstance().reportFree(it->second->size); 
            local_allocs.erase(it); //this calls cudaFree
            return cudaSuccess;   
        }

        //we could have migrated memory, check
        void* optr = translateDevicePointer(devPtr);
        it = local_allocs.find((uint64_t)optr);
        if (it != local_allocs.end()) {
            fprintf(stderr, "### ERASE BY MAP\n");
            tryRemoveFromPointerMap(devPtr);
            //GPUMemoryServer::Client::getInstance().reportFree(it->second->size); 
            local_allocs.erase(it); //this calls cudaFree
            return cudaSuccess;   
        }

        fprintf(stderr, "### ILLEGAL cudaFree call on devPtr %x\n", (uint64_t)devPtr);
        return cudaErrorInvalidValue;   
    }

    void Client::reportFree(uint64_t size) {
        Request req;
        req.type = RequestType::FREE;
        req.data.size = size;
        sendRequest(req);
    }

    void Client::reportMemoryRequested(uint64_t mem_mb) {
        Request req;
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        //sendRequest(req);
    }

    void Client::sendRequest(Request &req) {
        //just quit if we are not reporting to gpu server
        if (enable_reporting == false) {
            return;
        }

        sockmtx.lock();
        strncpy(req.worker_id, uuid.c_str(), MAX_UUID_LEN);
        if (strlen(uuid.c_str()) > MAX_UUID_LEN) {
            printf(" ### uuid %S IS LONGER THAN %d, THIS IS AN ISSUE\n", uuid.c_str(), MAX_UUID_LEN);
        }
        //if not connected yet, do it
        if (sockets[current_device] == 0) {
            connectToGPU(current_device);
        }
        if (zmq_send(sockets[current_device], &req, sizeof(Request), 0) == -1) {
            printf(" ### zmq_send errno %d\n", errno);
        }
        Reply rep;
        zmq_recv(sockets[current_device], &rep, sizeof(Reply), 0);
        sockmtx.unlock();
        //unlock and handle it
        handleReply(rep);
    }

    void Client::handleReply(Reply& reply) {
        //only migrate if server told us so and we haven't migrated before
        if (reply.migrate != Migration::NOPE && current_device == og_device)
            migrateToGPU(reply.target_device, reply.migrate);
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
        if (err) {
            printf("### CUDA err on cudaSetDevice(%d): %d\n", id, err);
            exit(1);
        }
        printf("Worker [%s] setting current device to %d\n", uuid.c_str(), id);
        if (og_device == -1) 
            og_device = id;
        current_device = id;

        //set on context so shadow threads also change context
        auto ccontext = ava::CommonContext::instance();
        ccontext->current_device = id;
    }

    void Client::resetCurrentGPU() {
        if (og_device != current_device) {
            setCurrentGPU(og_device);
            current_device = og_device;
        }
        migrated_type = Migration::NOPE;
    }

    void Client::kernelIn() {
        //printf("%d kernels\n", ++kc);
        if (migrated_type != Migration::NOPE) {
            //if we have migrated, lets just skip..
            return;
        }
        Request req;
        req.type = RequestType::KERNEL_IN;
        sendRequest(req);
    }
    
    void Client::kernelOut() {
        Request req;
        req.type = RequestType::KERNEL_OUT;
        //sendRequest(req);
    }

    bool Client::isInPointerMap(void* ptr) {
        auto it = pointer_map.upper_bound((uint64_t)ptr);
        //if there is no upper bound or previous, return
        if (it == pointer_map.end() || it == pointer_map.begin()) {
            return false;
        }

        it = std::prev(it);
        if( (uint64_t)ptr < it->first  &&  (uint64_t)ptr >= (it->first + it->second.size) ) {
            return false;
        }

        return true;
    }

    void Client::tryRemoveFromPointerMap(void* ptr) {
        auto it = pointer_map.find((uint64_t)ptr);
        if (it != pointer_map.end()) {
            pointer_map.erase(it);
        }
    }

    void* Client::translateDevicePointer(void* ptr) {
        //need to implement interval structure here. easiest idea is to
        //use std::map, which has a value with {size, other_ptr}.
        //this map can be queried with 
        // auto it = pointer_map.upper_bound(ptr)
        //then we go back one, since upper bound return the first
        //higher ptr
        // it = std::prev(it)
        //and check if it is range
        // if (ptr >= key)

        if (pointer_map.empty()) return ptr;

        auto it = pointer_map.upper_bound((uint64_t)ptr);
        //if there is no upper bound or previous, return
        if (it == pointer_map.begin()) return ptr;

        //there is an upper bound or it returned after last element, check if it really is in the range
        it = std::prev(it);
        if( (uint64_t)ptr < it->first  ||  (uint64_t)ptr >= (it->first + it->second.size) ) {
            return ptr;
        }

        //offset is always >= 0
        uint64_t offset = (uint64_t)ptr - it->first;
        //std::cerr << "  in translation, offset is " << offset << "\n";
        void* optr = (void*) (it->second.dstPtr + offset);
        //lets double check we need to translate
        cudaError_t ret;
        struct cudaPointerAttributes at;
        ret = cudaPointerGetAttributes(&at, optr);
        //if it really is a device pointer
        if (ret == 0 && at.type == 2) {
            std::cerr << "translated " << ptr << " to " << optr  << std::endl;
            return optr;
        }
        else {
            std::cerr << "### WE DODGED A BULLET" << std::endl;
            return ptr;
        }
    }

    void Client::migrateToGPU(uint32_t new_gpuid, Migration migration_type) {
        std::cerr << "[[[MIGRATION]]]\n" << std::endl;

        //wait for all cmd handlers to stop
        wait_for_cmd_drain();
        std::cerr << "All cmd handlers are stopped.. migrating\n" << std::endl;
        //mark that we migrated
        migrated_type = migration_type;

        if (migration_type == Migration::KERNEL) {
            printf("Migration by kernel mode, changing device to [%d]\n", new_gpuid);
            //cudaDeviceEnablePeerAccess(new_gpuid, 0);
            cudaDeviceSynchronize();
            setCurrentGPU(new_gpuid);
            printf("cudaDeviceEnablePeerAccess: new device [%d] can access memory on old [%d]\n", new_gpuid, og_device);
            cudaDeviceEnablePeerAccess(og_device, 0);
        }
        else if (migration_type == Migration::TOTAL) {
            std::cerr << "Full migration, changing device to" << new_gpuid << std::endl;
            setCurrentGPU(new_gpuid);

            //make a copy since we will modify ours during migration
            std::map<uint64_t, uint64_t> current_allocs;
            for (auto& al : local_allocs) {
                current_allocs[al.first] = al.second->size;
                //printf( "   [%p] -> sized %p\n", al.first, al.second->size);
            }

            for (auto const& al : current_allocs) {
                void* devPtr;
                //malloc on new device
                localMalloc(&devPtr, al.second);
                //update map
                pointer_map[al.first].dstPtr = devPtr;
                pointer_map[al.first].size = al.second;
                //cudaMemcpyPeer(devPtr, new_gpuid, al.first, og_device, al.second);
                cudaMemcpyPeerAsync(devPtr, new_gpuid, al.first, og_device, al.second);
                printf("  [%s] copying %d bytes GPUs [%d]  ->  [%d]\n", uuid.c_str(), al.second, og_device, new_gpuid);
                printf("      [%p] -> %p\n", al.first, devPtr);
            }
            cudaDeviceSynchronize();

            //now that data was moved, cleanup
            cleanup(og_device);
            cudaSetDevice(new_gpuid);
            //printf("Local migration: cleaned up data on old GPU\n");
        }

        //release handlers
        release_cmd_handlers();
    }

    //clean up all memory that are in current_device ONLY
    void Client::cleanup(uint32_t cd) {
        cudaSetDevice(cd);
        //report to server
        reportCleanup(cd);
        //erase only memory in GPU current_device
        for (auto it = local_allocs.begin(); it != local_allocs.end(); ) {
            if (it->second->device_id == cd)
                it = local_allocs.erase(it);
            else
                ++it;
        }
        cudaSetDevice(current_device);
    }

    void Client::reportCleanup(uint32_t gpuid) {
        Request req;
        req.type = RequestType::FINISHED;
        //TODO: report cleanup only device
        //sendRequest(req);
    }

    void Client::fullCleanup() {
        reportCleanup(0);  //TODO

        for (int i = 0 ; i < device_count ; i++) 
            cleanup(i);
        local_allocs.clear();

        pointer_map.clear();

        //iterate over map of maps, destroying streams
        while (!streams_map.empty())
            __helper_destroy_stream((streams_map.begin())->first);

        streams_map.clear();

        //clear leftover events
        while (!events_map.empty()) 
            __helper_cudaEventDestroy((events_map.begin())->first);
        events_map.clear();

        cudaSetDevice(current_device);
    }

}
