#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "common.hpp"
#include "client.hpp"
#include <zmq.h>
#include <errno.h>
#include <memory>
#include "common/extensions/cudart_10.1_utilities.hpp"

//for migration we might need
// cudaMemcpyPeer or cudaMemcpyPeerAsync 
// https://stackoverflow.com/questions/31628041/how-to-copy-memory-between-different-gpus-in-cuda


void* __translate_ptr(void* ptr) {
    return GPUMemoryServer::Client::getInstance().translate_ptr(ptr);
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
    GPUMemoryServer::Client::getInstance().sendFreeRequest(devPtr); 
    GPUMemoryServer::Reply* rep = GPUMemoryServer::Client::getInstance().getReply();
    return rep->returnErr;
}


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

    void Client::sendRequest(Request &req, uint32_t size, void* sock) {
        void* socket;
        if (sock == 0)
            socket = this->socket;
        else
            socket = sock;

        strcpy(req.worker_id, uuid.c_str());
        zmq_send(socket, (char*)&req, size, 0);

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
        printf("Worker [%s] setting current device to %d\n", uuid.c_str(), id);
        if (og_device == -1) 
            og_device = id;
        current_device = id;
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
        }
    }
    
    void Client::kernelOut() {
        req.type = RequestType::KERNEL_OUT;
        sendRequest(req);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        cudaError_t err = cudaMalloc(devPtr, size);
        local_allocs.push_back(std::make_unique<Client::LocalAlloc>(*devPtr));
        return err;
    }

    void Client::cleanup() {
        //if we are connected to server, inform it that we are done
        if (isMemoryServerMode()) {
            sendCleanupRequest();
        }
        //free all local mallocs
        local_allocs.clear();
    }

    void* Client::translate_ptr(void* ptr) {
        auto it = pointer_map.find((uint64_t)ptr);

        if (it == pointer_map.end())
            return ptr;
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
        //first we need to get all information of our memory
        char* buf = reply.piggyback;
        MigrationHeader* header = (MigrationHeader*) buf;
        PointerPair* pps = (PointerPair*) buf+sizeof(MigrationHeader);

        uint32_t size = sizeof(MigrationHeader) + (header->n_buffers * sizeof(PointerPair));

        //let's connect to the new GPU server
        void* new_socket = zmq_socket(context, ZMQ_REQ);
        std::ostringstream stringStream;
        stringStream << get_base_socket_path() << new_gpuid;
        int ret = zmq_connect(new_socket, stringStream.str().c_str());
        if (ret == -1) {
            printf("connect to new gpu failed, errno: %d\n", errno);
        }

        req.type = RequestType::MIGRATE;
        memcpy(req.piggyback, buf, size);
        size += offsetof(Request, guard); //add the request size to the piggyback size
        sendRequest(req, size, new_socket);

        //TODO:
        //cleanup on previous gpu
        //switch connection
        //make sure we will switch back when done

    }
}