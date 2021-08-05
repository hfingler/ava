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
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/common_context.h"
#include "extensions/memory_server/client.hpp"
#include "extensions/memory_server/common.hpp"

/**************************************
 * 
 *    Global functions, used by spec and helper functions. Mostly translate a global call to a client call
 * 
 * ************************************/

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
    return GPUMemoryServer::Client::getInstance().translate_ptr(ptr);
}

const void* __translate_ptr(const void* ptr) {
    return const_cast<const void*>(GPUMemoryServer::Client::getInstance().translate_ptr(ptr));
}

void __internal_kernelIn() {
    //printf("__internal_kernelIn\n");
    GPUMemoryServer::Client::getInstance().kernelIn();
}

void __internal_kernelOut() {
    //printf("__internal_kernelOut\n");
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

/**************************************
 * 
 *    Functions for GPUMemoryServer namespace, Client class
 * 
 * ************************************/

namespace GPUMemoryServer {
    Client::Client() {
        og_device = -1;
        context = zmq_ctx_new();
        for (int i = 0 ; i < 4; i++) {
             sockets[i] = 0;
         }
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
        matchCurrentGPU();
        cudaError_t err = cudaMalloc(devPtr, size);
        local_allocs.emplace((uint64_t)*devPtr, std::make_unique<Client::LocalAlloc>(*devPtr, size, current_device));
        //report to server
        //printf("malloc: %p\n", *devPtr);
        GPUMemoryServer::Client::getInstance().reportMalloc(size); 
        return err;
    }

    void Client::reportMalloc(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    cudaError_t Client::localFree(void* devPtr) {
        // cudaFree with nullptr does no operation
        if (devPtr == nullptr) {
            return cudaSuccess;
        }
        auto it = local_allocs.find((uint64_t)devPtr);
        if (it == local_allocs.end()) {
            fprintf(stderr, "ILLEGAL cudaFree call on devPtr %x\n", (uint64_t)devPtr);
            cudaFree(devPtr);
            return (cudaError_t)0;
        }
        //report to server
        GPUMemoryServer::Client::getInstance().reportFree(it->second->size); 
        local_allocs.erase(it);
        return (cudaError_t)0;   
    }

    void Client::reportFree(uint64_t size) {
        Request req;
        req.type = RequestType::FREE;
        req.data.size = size;
        sendRequest(req);
    }

    void Client::fullCleanup() {
        reportCleanup(0);  //TODO
        local_allocs.clear();

        //iterate over map of maps, destroying events
        for (auto& kv : streams_map) {
            for (auto& idx_st : kv.second) {
                cudaStreamDestroy(idx_st.second);
            }
        }
        streams_map.clear();
    }

    //clean up all memory that are in current_device ONLY
    void Client::cleanup(uint32_t cd) {
        //report to server
        reportCleanup(cd);
        //erase only memory in GPU current_device
        for (auto it = local_allocs.begin(); it != local_allocs.end(); ) {
            if (it->second->device_id == cd)
                it = local_allocs.erase(it);
            else
                ++it;
        }
    }

    void Client::reportCleanup(uint32_t gpuid) {
        Request req;
        req.type = RequestType::FINISHED;
        //TODO: report cleanup only device
        sendRequest(req);
    }

    void Client::reportMemoryRequested(uint64_t mem_mb) {
        Request req;
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        sendRequest(req);
    }

    void Client::sendRequest(Request &req) {
        //just quit if we are not reporting to gpu server
        if (enable_reporting == false) {
            return;
        }

        sockmtx.lock();

        strncpy(req.worker_id, uuid.c_str(), MAX_UUID_LEN);
        if (strlen(uuid.c_str()) > MAX_UUID_LEN) {
            printf(" @@@@ uuid %S IS LONGER THAN %d, THIS IS AN ISSUE\n", uuid.c_str(), MAX_UUID_LEN);
        }

        //if not connected yet, do it
        if (sockets[current_device] == 0) {
            connectToGPU(current_device);
        }

        //printf(" !! sending request type [%d]  to %d\n", req.type, current_device);
        int rc = zmq_send(sockets[current_device], &req, sizeof(Request), 0);
        if (rc == -1) {
            printf(" !!!!!!!! zmq_send errno %d\n", errno);
        }

        Reply rep;
        zmq_recv(sockets[current_device], &rep, sizeof(Reply), 0);
        handleReply(rep);
        sockmtx.unlock();
    }

    void Client::handleReply(Reply& reply) {
        if (reply.migrate != Migration::NOPE)
            migrateToGPU(reply.target_device, reply.migrate);
    }

    void Client::matchCurrentGPU() {
        /*
        int d;
        cudaGetDevice(&d);
        if (current_device != d) {
            printf("  ###WRONG### !!! Worker [%s] is at the wrong GPU somehow cuda %d  client %d\n", uuid.c_str(), d, current_device);
            setCurrentGPU(current_device);
        }
        */
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
        cudaFree(0);
        if (err) {
            printf("CUDA err on cudaSetDevice(%d): %d\n", id, err);
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

    void Client::setOriginalGPU() {
        if (og_device != current_device)
            setCurrentGPU(og_device);
    }

    void Client::kernelIn() {
        Request req;
        req.type = RequestType::KERNEL_IN;
        sendRequest(req);
    }
    
    void Client::kernelOut() {
        Request req;
        req.type = RequestType::KERNEL_OUT;
        sendRequest(req);
    }

    void* Client::translate_ptr(void* ptr) {
        auto it = pointer_map.find((uint64_t)ptr);
        if (it == pointer_map.end()) {
            //printf("**pointer NOT found in map, returning  %p\n", ptr);
            return ptr;
        }
        else {
            printf("**pointer found in map, translating: %p  ->  %p\n", ptr, it->second);
            return it->second;
        }
    }

    void Client::migrateToGPU(uint32_t new_gpuid, Migration migration_type) {
        //printf("GPU server told us to ask for migration\n");
       
        if (migration_type == Migration::KERNEL) {
            printf("Migration by kernel mode, changing device to [%d]\n", new_gpuid);
            setCurrentGPU(new_gpuid);
            printf("cudaDeviceEnablePeerAccess: new device [%d] can access memory on old [%d]\n", new_gpuid, og_device);
            cudaDeviceEnablePeerAccess(og_device, 0);
        }
        else if (migration_type == Migration::MEMORY) {
            printf("Migration by memory mode, changing device to [%d]\n", new_gpuid);
            setCurrentGPU(new_gpuid);

            //make a copy since we will modify ours during migration
            std::map<uint64_t, uint64_t> current_allocs;
            for (auto& al : local_allocs) {
                current_allocs[al.first] = al.second->size;
            }

            for (auto const& al : current_allocs) {
                void* devPtr;
                //malloc on new device
                __internal_cudaMalloc(&devPtr, al.second);
                //update map
                pointer_map[al.first] = devPtr;
                //async might not help much since they are not parallel
                cudaMemcpyPeer(devPtr, new_gpuid, al.first, og_device, al.second);
                printf("  [%s] copying %d bytes GPUs [%d]  ->  [%d]\n", uuid.c_str(), al.second, og_device, new_gpuid);
                printf("      [%p] -> %p\n", al.first, devPtr);
                
                /*
                char b[8];
                cudaMemcpy((void*)b, devPtr, 8, cudaMemcpyDeviceToHost);
                printf("first 8 bytes of %p on new device\n      ", devPtr);
                for (int i = 0 ; i < 8 ; i++) {
                    printf("%#02x ", b[i]);
                }
                printf("\n");
                */
            }

            //now that data was moved, cleanup
            cleanup(og_device);
            printf("Local migration: cleaned up data on old GPU\n");
            setCurrentGPU(new_gpuid);
        }
    }
}
