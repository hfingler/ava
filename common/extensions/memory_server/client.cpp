#include <stdint.h>
#include <string>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "common.hpp"
#include "client.hpp"
#include <zmq.h>
#include <errno.h>

//I'll just dumpt it here so I don't have to link yet another .cpp
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size) {
    //if we are not connected, use normal cudaMalloc
    if(!GPUMemoryServer::Client::getInstance().isConnected()) {
        return cudaMalloc(devPtr, size);
    }

    GPUMemoryServer::Reply rep =
            GPUMemoryServer::Client::getInstance().sendMallocRequest(size); 
    printf("Done, err code is %d\n", rep.returnErr);

    void* ourDevPtr;
    if (rep.returnErr == 0) {
        //if it wasn't managed, it already allocated, so map here
        if (rep.data.is_managed == 0) {
            printf("Mapping dev ptr in this process\n");
            cudaError_t err = cudaIpcOpenMemHandle(&ourDevPtr, rep.data.alloc.memHandle, cudaIpcMemLazyEnablePeerAccess);
        }
        //managed doesn't support Ipc, so we do it ourselves
        else {
            //update error
            rep.returnErr = cudaMallocManaged(&ourDevPtr, size);
            if(rep.returnErr) {
                printf("Something went wrong on cudaMallocManaged, returning err.");
                return rep.returnErr;
            }
            GPUMemoryServer::Client::getInstance().managed_allocations.push_back(ourDevPtr);
        }
    }
    
    //even if the malloc failed, this is fine
    *devPtr = ourDevPtr;
    return rep.returnErr;
}

cudaError_t __internal_cudaFree(void *devPtr) {
    GPUMemoryServer::Reply rep =
            GPUMemoryServer::Client::getInstance().sendFreeRequest(devPtr); 
    return rep.returnErr;
}

namespace GPUMemoryServer {

    int Client::connectToGPU(uint16_t gpuId) {
        context = zmq_ctx_new();
        socket = zmq_socket(context, ZMQ_REQ);
        std::ostringstream stringStream;
        stringStream << get_base_socket_path() << gpuId;
        is_connected = true;
        int ret = zmq_connect(socket, stringStream.str().c_str());
        if (ret == -1) {
            printf("connect failed, errno: %d\n", errno);
        }
        return ret;
    }

    Reply Client::sendMallocRequest(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;
        return sendRequest(req);
    }
    
    Reply Client::sendFreeRequest(void* devPtr) {
        Request req;
        req.type = RequestType::FREE;
        req.data.devPtr = devPtr;
        return sendRequest(req);
    }
    
    //this is called when we are done, so cleanup too
    Reply Client::sendCleanupRequest() {
        Request req;
        req.type = RequestType::FINISHED;
        return sendRequest(req);
    }

    Reply Client::sendMemoryRequestedValue(uint64_t mem_mb) {
        Request req;
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        return sendRequest(req);
    }

    Reply Client::sendRequest(Request &req) {
        req.worker_id = this->uuid;
        memcpy(buffer, &req, sizeof(Request));
        zmq_send(socket, buffer, sizeof(Request), 0);
        zmq_recv(socket, buffer, sizeof(Reply), 0);

        Reply rep;
        memcpy(&rep, buffer, sizeof(Reply));
        return rep;
    }

    Client::~Client() {
        zmq_close(socket);
        //zmq_ctx_destroy(context);

        for (auto &ptr : managed_allocations) {
            cudaFree(ptr);
        }
    }
}