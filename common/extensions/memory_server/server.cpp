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

static uint32_t debug_kernel_count = 0;

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

    char* buffer[BUF_SIZE];
    printf("Memory server of GPU %d ready, bound to %s\n", gpu, unix_socket_path.c_str());
    
    while(1) {
        zmq_recv(responder, buffer, BUF_SIZE, 0);
        handleRequest((char*)buffer, responder);
    }
}

void Server::handleRequest(char* buffer, void *responder) {
    Request req;
    Reply rep;
    uint32_t reply_len;

    //TODO: better (de)serialization
    memcpy(&req, buffer, sizeof(Request));

    /*************************
     *   cudaMalloc request
     *************************/
    if (req.type == RequestType::ALLOC) {
        reply_len = handleMalloc(req, rep);
    }
    /*************************
     *       cudaFree
     *************************/
    else if (req.type == RequestType::FREE) {
        reply_len = handleFree(req, rep);
    }
    /*************************
     * worker finished, cleanup
     *************************/
    else if (req.type == RequestType::FINISHED) {
        reply_len = handleFinish(req, rep);
    }
    /*************************
     * store requested amount of memory
     *************************/
    else if (req.type == RequestType::MEMREQUESTED) {
        std::string worker_id(req.worker_id);
        requested_memory[worker_id] = req.data.size;
        //construct reply
        memset(&(rep), 0, sizeof(Reply));
        reply_len = sizeof(Reply);
    }
    /*************************
     * kernel asking to run
     *************************/
    else if (req.type == RequestType::KERNEL_IN) {
        reply_len = handleKernelIn(req, rep);
    }
    /*************************
     * kernel done
     *************************/
    else if (req.type == RequestType::KERNEL_OUT) {
        reply_len = handleKernelOut(req, rep);
    }
    /*************************
     * Client requesting info about all allocations
     *************************/
    else if (req.type == RequestType::FETCH_ALL_PTRS) {
        reply_len = handleGetAllPointers(req, rep);
    }
    /*************************
     * Client requesting migration to us
     *************************/
    else if (req.type == RequestType::MIGRATE) {
        reply_len = handleMigrate(req, rep);
    }

    //dont send the piggyback part if not necessary
    zmq_send(responder, (char*)&rep, reply_len, 0);
}   

uint32_t Server::handleMalloc(Request& req, Reply& rep) {
    //allocate and get the ipc handle
    void *ptr;
    cudaIpcMemHandle_t memHandle;
    uint64_t requested_in_mb = req.data.size / (1024*1024);
    uint64_t requested = req.data.size;
    std::string worker_id(req.worker_id);

    cudaError_t err = cudaMalloc(&ptr, requested);
    //if malloc fails.. malloc fails
    if(err) {
        printf("Something went wrong on malloc, returning err.");
        rep.returnErr = err;
        return offsetof(Reply, guard);;
    }
    checkCudaErrors( cudaIpcGetMemHandle(&memHandle, ptr) );
    rep.data.memHandle = memHandle;
    rep.returnErr = err;

    //store on our metadata maps
    uint64_t key = (uint64_t) ptr;
    allocs[worker_id][key] = std::make_unique<Allocation>(requested, ptr);

    //c++ wizardy:  https://stackoverflow.com/questions/2667355/mapint-int-default-values
    used_memory[worker_id] += requested_in_mb;

    return offsetof(Reply, guard);
}

uint32_t Server::handleFree(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    uint64_t ptr = (uint64_t) req.data.devPtr;
    auto alloc = allocs[worker_id].find(ptr);

    //not found, illegal free
    if (alloc == allocs[worker_id].end()) {
        printf("Worker %s tried to free %p, which wasn't allocated\n", worker_id.c_str(), req.data.devPtr);
    }
    else {
        //update used memory
        used_memory[worker_id] -= alloc->second->size;
        //this frees the handle and memory
        allocs[worker_id].erase(alloc);
    }

    //construct reply
    memset(&(rep), 0, sizeof(Reply));
    rep.returnErr = cudaError::cudaSuccess;

    return offsetof(Reply, guard);
}

uint32_t Server::handleFinish(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    used_memory[worker_id] = 0;
    allocs[worker_id].clear();
    //don't need reply
    (void)rep;

    return offsetof(Reply, guard);
}

uint32_t Server::handleKernelIn(Request& req, Reply& rep) {
    printf("   handleKernelIn %s\n", std::getenv("SG_DEBUG_MIGRATION"));
    //check if we are just debugging
    if (std::getenv("SG_DEBUG_MIGRATION")) {
        debug_kernel_count += 1;
        if (debug_kernel_count % 2 == 0) {
            printf("SG_DEBUG_MIGRATION:  kernel #%d, setting migration on\n", debug_kernel_count);
            rep.data.migrate = 1;
        }
        else 
            rep.data.migrate = 0;
        return offsetof(Reply, guard); 
    }
    (void)req;

    return offsetof(Reply, guard);
}

uint32_t Server::handleKernelOut(Request& req, Reply& rep) {
    (void)req; (void)rep;

    return offsetof(Reply, guard);
}

uint32_t Server::handleGetAllPointers(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    
    char* buffer = rep.piggyback;
    uint32_t size = sizeof(MigrationHeader) + (allocs[worker_id].size() * sizeof(PointerPair));
    MigrationHeader* header = (MigrationHeader*) buffer;
    PointerPair* pps = (PointerPair*) buffer+sizeof(MigrationHeader);

    header->n_buffers = allocs[worker_id].size();

    uint32_t count = 0;
    for (auto &al : allocs[worker_id]) {
        pps[count].ptr = (uint64_t)al.second->devPtr;
        pps[count].data = al.second->size;
        count++;
    }

    return offsetof(Reply, guard) + size;
}

uint32_t Server::handleMigrate(Request& req, Reply& rep) {
    (void)req; 
    char* buf = rep.piggyback;
    MigrationHeader* header = (MigrationHeader*) buf;
    PointerPair* pps = (PointerPair*) buf+sizeof(MigrationHeader);

    //TODO: migrate.. duh

    return offsetof(Reply, guard);
}

Allocation::~Allocation() {
    cudaError_t err = cudaIpcCloseMemHandle(devPtr);
    if (err) {
        printf("CUDA err on cudaIpcCloseMemHandle: %d\n", err);
    }
    checkCudaErrors( cudaFree(devPtr) );
}

//end of namespace
};

