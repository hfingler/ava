#include <stdint.h>
#include <string>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "common.hpp"
#include "client.hpp"
#include <zmq.h>
#include <errno.h>

//for migration we might need
// cudaMemcpyPeer or cudaMemcpyPeerAsync 
// https://stackoverflow.com/questions/31628041/how-to-copy-memory-between-different-gpus-in-cuda

//  GPUMemoryServer::Client::getInstance().kernelIn();
//   GPUMemoryServer::Client::getInstance().kernelOut();

//I'll just dumpt it here so I don't have to link yet another .cpp
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size) {
    //if we are not connected, use normal cudaMalloc
    if(!GPUMemoryServer::Client::getInstance().isConnected()) {
        return cudaMalloc(devPtr, size);
    }

    GPUMemoryServer::Reply rep =
            GPUMemoryServer::Client::getInstance().sendMallocRequest(size); 
    void* ourDevPtr;
    if (rep.returnErr == 0) {
        //if it wasn't managed, it already allocated, so map here
        //printf("Mapping dev ptr in this process\n");
        cudaError_t err = cudaIpcOpenMemHandle(&ourDevPtr, rep.data.memHandle, cudaIpcMemLazyEnablePeerAccess);
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
        strcpy(req.worker_id, uuid.c_str());
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
        Request req;
        req.type = RequestType::KERNEL_IN;
        Reply rep = sendRequest(req);
        
        if (rep.data.migrate == 1) {
            printf("Worker [%s] talked to GPU server and it told us to ask for migration\n");
        }
    }
    
    void Client::kernelOut() {
        Request req;
        req.type = RequestType::KERNEL_OUT;
        sendRequest(req);
    }
}