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
    void Client::connectToGPUs() {
        context = zmq_ctx_new();
        sockets = new void*[device_count];
        for (int i = 0 ; i < device_count; i++) {
            sockets[i] = 0;
        }
    }

    void Client::connectToGPU(uint32_t gpuid) {
        int ret;
        std::ostringstream stringStream;
        stringStream << GPUMemoryServer::get_base_socket_path() << gpuid;
        sockets[gpuid] = zmq_socket(context, ZMQ_REQ);
        while (ret != 0) { 
            ret = zmq_connect(sockets[gpuid], stringStream.str().c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        printf("GPU Client succesfully connected to server [%d] at %p\n", gpuid, sockets[gpuid]);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        cudaError_t err = cudaMalloc(devPtr, size);
        local_allocs.emplace((uint64_t)*devPtr, std::make_unique<Client::LocalAlloc>(*devPtr, size, current_device));
        //report to server
        GPUMemoryServer::Client::getInstance().sendMallocRequest(size); 
        return err;
    }

    void Client::sendMallocRequest(uint64_t size) {
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    void Client::sendFreeRequest(uint64_t size) {
        req.type = RequestType::FREE;
        req.data.size = size;
        sendRequest(req);
    }
    
    void Client::sendCleanupRequest() {
        req.type = RequestType::FINISHED;
        sendRequest(req);
    }

    void Client::sendMemoryRequestedValue(uint64_t mem_mb) {
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        sendRequest(req);
    }

    void Client::sendRequest(Request &req) {
        /*
        strcpy(req.worker_id, uuid.c_str());
        printf(" !! sending to %d  at %p\n",current_device, sockets[current_device] );

        if (sockets[current_device] == 0) {
            connectToGPU(current_device);
        }

        int rc = zmq_send(sockets[current_device], (char*)&req, sizeof(Request), 0);
        printf(" !!! zmq_send ret %d\n", rc);


        printf(" !!! waiting for receive\n");
        zmq_recv(sockets[current_device], buffer, sizeof(Reply), 0);
        memcpy(&reply, buffer, sizeof(Reply));
        printf(" !!! received\n");

        handleReply();
    */
    }

    Client::~Client() {
        for (int i = 0 ; i < device_count ; i++) {
            zmq_close(sockets[i]);
        }
        //zmq_ctx_destroy(context);
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
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
        req.type = RequestType::KERNEL_IN;
        sendRequest(req);
    }
    
    void Client::kernelOut() {
        req.type = RequestType::KERNEL_OUT;
        sendRequest(req);
    }

    

    cudaError_t Client::localFree(void* devPtr) {
        auto it = local_allocs.find((uint64_t)devPtr);
        if (it == local_allocs.end()) {
            printf("ILLEGAL cudaFree call!\n");
            return 1;
        }
        //report to server
        GPUMemoryServer::Client::getInstance().sendFreeRequest(it->second->size); 
        local_allocs.erase(it);
        return (cudaError_t)0;   
    }

    //clean up all memory that are in current_device ONLY
    void Client::cleanup() {
        //report to server
        sendCleanupRequest();
        //erase only memory in GPU current_device
        uint32_t cd = current_device;
        for (auto it = local_allocs.begin(); it != local_allocs.end(); ) {
            if (it->second->device_id == cd)
                it = local_allocs.erase(it);
            else
                ++it;
        }
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

    void Client::handleReply() {
        if (reply.migrate != Migration::NOPE)
            migrateToGPU(reply.target_device, reply.migrate);
    }

    void Client::migrateToGPU(uint32_t new_gpuid, Migration migration_type) {
        printf("Worker [%s] talked to GPU server and it told us to ask for migration\n");
       
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
                
                char b[8];
                cudaMemcpy((void*)b, devPtr, 8, cudaMemcpyDeviceToHost);
                printf("first 8 bytes of %p on new device\n      ", devPtr);
                for (int i = 0 ; i < 8 ; i++) {
                    printf("%#02x ", b[i]);
                }
                printf("\n");
            }

            //now that data was moved, cleanup
            setOriginalGPU();
            cleanup();
            printf("Local migration: cleaned up data on old GPU\n");
            setCurrentGPU(new_gpuid);
        }
    }
}
