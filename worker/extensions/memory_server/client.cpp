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
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/extensions/memory_server/common.hpp"
#include "common/extensions/memory_server/client.hpp"
#include "common/common_context.h"

/**************************************
 * 
 *    Global functions, used by spec and helper functions. Mostly translate a global call to a client call
 * 
 * ************************************/

static uint32_t device_count;

int32_t __internal_getDeviceCount() {
    return device_count;
}

void __internal_setDeviceCount(uint32_t dc) {
    printf("set device count %d\n", dc);
    device_count = dc;
}

uint32_t __internal_getCurrentDevice() {
    return GPUMemoryServer::Client::getInstance().current_device;
}

void* __translate_ptr(void* ptr) {
    return GPUMemoryServer::Client::getInstance().translate_ptr(ptr);
}

const void* __translate_ptr(const void* ptr) {
    return 
        const_cast<const void*>(GPUMemoryServer::Client::getInstance().translate_ptr(ptr));
}

void __internal_kernelIn() {
    printf("__internal_kernelIn\n");
    GPUMemoryServer::Client::getInstance().kernelIn();
}

void __internal_kernelOut() {
    printf("__internal_kernelOut\n");
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
    //if we are not in mem server mode, use normal cudaMalloc
    if(!GPUMemoryServer::Client::getInstance().isMemoryServerMode()) {
        cudaError_t err = GPUMemoryServer::Client::getInstance().localMalloc(devPtr, size);
        return err;
    }

    GPUMemoryServer::Client::getInstance().sendMallocRequest(size); 
    GPUMemoryServer::Reply* rep = GPUMemoryServer::Client::getInstance().getReply(); 

    void* ourDevPtr;
    if (rep->returnErr == 0) {
        //if it wasn't managed, it already allocated, so map here
        cudaError_t err = cudaIpcOpenMemHandle(&ourDevPtr, rep->data.memHandle, cudaIpcMemLazyEnablePeerAccess);
    }
    //even if the malloc failed, this is fine
    *devPtr = ourDevPtr;
    return rep->returnErr;
}

cudaError_t __internal_cudaFree(void *devPtr) {
    if(!GPUMemoryServer::Client::getInstance().isMemoryServerMode()) {
        return GPUMemoryServer::Client::getInstance().localFree(devPtr);
    }

    GPUMemoryServer::Client::getInstance().sendFreeRequest(devPtr); 
    GPUMemoryServer::Reply* rep = GPUMemoryServer::Client::getInstance().getReply();
    return rep->returnErr;
}

/**************************************
 * 
 *    Functions for GPUMemoryServer namespace, Client class
 * 
 * ************************************/

namespace GPUMemoryServer {
    int Client::connectToGPU(uint16_t gpuId) {
        context = zmq_ctx_new();
        socket = zmq_socket(context, ZMQ_REQ);
        std::ostringstream stringStream;
        stringStream << get_base_socket_path() << gpuId;
        int ret = zmq_connect(socket, stringStream.str().c_str());
        if (ret == -1) {
            printf("connect failed, errno: %d\n", errno);
        }
        return ret;
    }

    void Client::sendMallocRequest(uint64_t size) {
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    void Client::sendFreeRequest(void* devPtr) {
        req.type = RequestType::FREE;
        req.data.devPtr = devPtr;
        sendRequest(req);
    }
    
    //this is called when we are done, so cleanup too
    void Client::sendCleanupRequest() {
        req.type = RequestType::FINISHED;
        sendRequest(req);
    }

    void Client::sendMemoryRequestedValue(uint64_t mem_mb) {
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        sendRequest(req);
    }

    void Client::sendRequest(Request &req, void* sock) {
        void* socket;
        if (sock == 0)
            socket = this->socket;
        else
            socket = sock;

        strcpy(req.worker_id, uuid.c_str());
        zmq_send(socket, (char*)&req, sizeof(Request), 0);

        zmq_recv(socket, buffer, sizeof(Reply), 0);
        memcpy(&reply, buffer, sizeof(Reply));
    }

    Client::~Client() {
        zmq_close(socket);
        //zmq_ctx_destroy(context);
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
        if (err) {
            printf("CUDA err on cudaSetDevice(%d): %d\n", id, err);
            exit(1);
        }
        cudaFree(0);
        printf("Worker [%s] setting current device to %d\n", uuid.c_str(), id);
        if (og_device == -1) 
            og_device = id;
        current_device = id;

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
        
        if (reply.data.migrate == 1) {
            printf("Worker [%s] talked to GPU server and it told us to ask for migration\n");

    
        migrateToGPU(current_device == 2 ? 3 : 2);
        }
    }
    
    void Client::kernelOut() {
        req.type = RequestType::KERNEL_OUT;
        sendRequest(req);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        cudaError_t err = cudaMalloc(devPtr, size);
        local_allocs.push_back(std::make_unique<Client::LocalAlloc>(*devPtr, size, current_device));
        return err;
    }

    cudaError_t Client::localFree(void* devPtr) {
        //TODO: crap.. this should've been a map
        for (auto it = local_allocs.begin() ; it != local_allocs.end(); ++it) {
            if((*it)->devPtr == devPtr) {
                local_allocs.erase(it);
                return (cudaError_t)0;
            }
        }
        return cudaFree(devPtr);
    }

    //clean up all memorys that are in current_device ONLY
    void Client::cleanup() {
        //if we are connected to server, inform it that we are done
        if (isMemoryServerMode()) {
            sendCleanupRequest();
        }
        else {
            //erase only memory in GPU current_device
            uint32_t cd = current_device;
            local_allocs.erase(
                std::remove_if(
                    local_allocs.begin(), 
                    local_allocs.end(),
                    [cd](auto const& al)-> bool
                        { return al->device_id == cd; }
                ), 
                local_allocs.end()
            ); 
        }
    }

    void* Client::translate_ptr(void* ptr) {
        auto it = pointer_map.find((uint64_t)ptr);

        if (it == pointer_map.end()) {
            printf("**pointer NOt found in map, returning  %p\n", ptr);
            return ptr;
        }
        else {
            printf("**pointer found in map, translating: %p  ->  %p\n", ptr, it->second);
            return it->second;
        }
    }

    void Client::sendGetAllPointersRequest() {
        req.type = RequestType::FETCH_ALL_PTRS;
        sendRequest(req);
    }

    void Client::migrateToGPU(uint32_t new_gpuid) {
        std::map<uint64_t, uint64_t> current_allocs;

        //on server mode the server knows our allocations
        if (isMemoryServerMode()) {
            //first we need to get all information of our memory
            sendGetAllPointersRequest();
            char* buf = reply.piggyback;
            MigrationHeader* header = (MigrationHeader*) buf;
            PointerPair* pps = (PointerPair*) buf+sizeof(MigrationHeader);

            for (int i = 0 ; i < header->n_buffers ; i++) {
                current_allocs[pps[i].ptr] = pps[i].size;
            }
        }
        //on normal mode we have all the allocs
        else {
            for (auto& al : local_allocs) {
                current_allocs[(uint64_t)al->devPtr] = al->size;
            }
        }

        //let's connect to the new GPU server
        void* old_socket = socket;
        void* new_socket = zmq_socket(context, ZMQ_REQ);
        std::ostringstream stringStream;
        stringStream << get_base_socket_path() << new_gpuid;
        int ret = zmq_connect(new_socket, stringStream.str().c_str());
        printf("migration: connected to new worker on gpu %d\n", new_gpuid);
        if (ret == -1) {
            printf("connect to new gpu failed, errno: %d\n", errno);
        }

        //switch connection to new one
        socket = new_socket;
        if (!isMemoryServerMode()) {
            printf("Migration on local mode, changing device to [%d] temporarily\n", new_gpuid);
            setCurrentGPU(new_gpuid);

            printf("cudaDeviceEnablePeerAccess: device [%d] can access memory on [%d]\n", new_gpuid, og_device);
            cudaDeviceEnablePeerAccess(og_device, 0);
        }

        /*
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

        //set information to old so we can cleanup
        //TODO: cleanup deletes EVERYTHING, including what we just did, so fix it.
        if (isMemoryServerMode()) {
            socket = old_socket;
            cleanup();
        }
        else {
            setOriginalGPU();
            cleanup();
            printf("Local migration: cleaned up data on old GPU\n");
            printf("Migration on local mode, changing device to [%d] until finish\n", new_gpuid);
            setCurrentGPU(new_gpuid);
        }
        */

        //update socket
        socket = new_socket;
        zmq_close(old_socket);


    }
}
