#include <map>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cuda_helper.hpp"
#include <zmq.h>
#include "server.hpp"
#include "common.hpp"
#include <unistd.h>

namespace GPUMemoryServer {

//The default value for flags is cudaMemAttachGlobal. If cudaMemAttachGlobal is specified, then this memory is accessible from any stream on any device.
//cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal ) 

//https://forums.developer.nvidia.com/t/gpu-inter-process-communications-ipc-question/35936/5

//when we need to enable peer access
//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleIPC/simpleIPC.cu

void Server::run() {
    //create zmq listener
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    //make sure it doesn't exist
    unlink(unix_socket_path.c_str());
    int rc = zmq_bind(responder, unix_socket_path.c_str());
    if (rc == -1) {
        printf("BIND FAILED\n");
    }
    //go to our gpu
    checkCudaErrors( cudaSetDevice(gpu) );

    char* buffer = new char[BUF_SIZE];
    printf("Memory server of GPU %d ready, bound to %s\n", gpu, unix_socket_path.c_str());
    Reply rep;
    while(1) {
        zmq_recv(responder, buffer, BUF_SIZE, 0);
        printf("got a request on gpu %d!\n", gpu);
        handleRequest(buffer, &rep);
        memcpy(buffer, &rep, sizeof(Reply));
        zmq_send(responder, buffer, sizeof(Reply), 0);
    }
}

void Server::handleRequest(char* buffer, Reply* rep) {
    Request req;
    cudaError_t err;

    //TODO: better (de)serialization
    memcpy(&req, buffer, sizeof(Request));

    /*************************
     *   cudaMalloc request
     *************************/
    if (req.type == RequestType::ALLOC) {
        //allocate and get the ipc handle
        void *ptr;
        cudaIpcMemHandle_t memHandle;

        uint64_t requested_in_mb = req.data.size / (1024*1024);
        uint64_t requested = req.data.size;

        //if user is requesting more than it asked for
        if (used_memory[req.worker_id] + requested_in_mb > requested_memory[req.worker_id]) {
            printf("Worker [%lu] requested %lu but is allocating total of %lu, using managed\n", 
                    req.worker_id, requested_memory[req.worker_id], used_memory[req.worker_id] + requested_in_mb);
            
            err = cudaMallocManaged(&ptr, requested);
            if(err) {
                printf("Something went wrong on cudaMallocManaged, returning err.");
                rep->returnErr = err;
                return;
            }

            rep->data.is_managed = 1;
            rep->data.alloc.devPtr = ptr;
            rep->returnErr = err;
        }
        else {
            err = cudaMalloc(&ptr, requested);
            //if malloc fails.. malloc fails
            if(err) {
                printf("Something went wrong on malloc, returning err.");
                rep->returnErr = err;
                return;
            }
            //TODO: double check if cudaMallocManaged works with cudaIpcGetMemHandle 
            checkCudaErrors( cudaIpcGetMemHandle(&memHandle, ptr) );
            rep->data.is_managed = 0;
            rep->data.alloc.memHandle = memHandle;
            rep->returnErr = err;
        }

        //store on our metadata maps
        uint64_t key = (uint64_t) ptr;
        allocs[req.worker_id][key] = std::make_unique<Allocation>(requested, ptr);

        //c++ wizardy:  https://stackoverflow.com/questions/2667355/mapint-int-default-values
        used_memory[req.worker_id] += requested_in_mb;        
    }
    /*************************
     *       cudaFree
     *************************/
    else if (req.type == RequestType::FREE) {
        uint64_t ptr = (uint64_t) req.data.devPtr;
        auto alloc = allocs[req.worker_id].find(ptr);

        //not found, illegal free
        if (alloc == allocs[req.worker_id].end()) {
            printf("Worker %d tried to free %p, which wasn't allocated\n", req.worker_id, req.data.devPtr);
        }
        else {
            //update used memory
            used_memory[req.worker_id] -= alloc->second->size;
            //this frees the handle and memory
            allocs[req.worker_id].erase(alloc);
        }

        //construct reply
        memset(&(rep), 0, sizeof(Reply));
        rep->returnErr = cudaError::cudaSuccess;
    }
    /*************************
     * worker finished, cleanup
     *************************/
    else if (req.type == RequestType::FINISHED) {
        used_memory[req.worker_id] = 0;
        allocs[req.worker_id].clear();
        //construct reply
        memset(&(rep), 0, sizeof(Reply));
    }
    /*************************
     * store requested amount of memory
     *************************/
    else if (req.type == RequestType::MEMREQUESTED) {
        requested_memory[req.worker_id] = req.data.size;
        //construct reply
        memset(&(rep), 0, sizeof(Reply));
    }
}   

Allocation::~Allocation() {
    checkCudaErrors( cudaIpcCloseMemHandle(devPtr) );  //TODO: double check if we are opening for all requests
    checkCudaErrors( cudaFree(devPtr) );
}

//end of namespace
};

