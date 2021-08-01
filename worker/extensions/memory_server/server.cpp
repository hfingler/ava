#include <map>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zmq.h>
#include "server.hpp"
#include "extensions/memory_server/common.hpp"


namespace GPUMemoryServer {

static uint32_t debug_kernel_count = 0;

Server::Server(uint16_t gpu, uint64_t total_memory, std::string unix_socket_path, std::string resmngr_address) {
    this->gpu = gpu;
    this->gpu_memory_total = total_memory;  
    this->gpu_memory_used = 0;
    this->kernels_queued = 0;
    this->unix_socket_path = unix_socket_path;

    if (resmngr_address != "") {
        printf("Server connecting to resmngr at %s\n", resmngr_address.c_str());
        this->resmngr_client = new ResMngrClient(grpc::CreateChannel(resmngr_address, grpc::InsecureChannelCredentials())); 
    }
    else {
        printf("Server did not get resmngr_address, so not connecting..\n");
        this->resmngr_client = 0;
    }
}

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

    printf("Memory server of GPU %d ready, bound to %s\n", gpu, unix_socket_path.c_str());
    Reply rep;
    Request req;
    while(1) {
        printf(" >> server %d waiting for request\n", gpu);
        zmq_recv(responder, &req, sizeof(Request), 0);
        printf(" >> server %d got a request\n", gpu);
        handleRequest(req, rep);
        printf(" >> sending response\n");
        zmq_send(responder, &rep, sizeof(Reply), 0);
    }
}

void Server::handleRequest(Request& req, Reply& rep) {
    //reset reply
    rep.migrate = Migration::NOPE;

    /*************************
     *   cudaMalloc request
     *************************/
    if (req.type == RequestType::ALLOC) {
        handleMalloc(req, rep);
    }
    /*************************
     *       cudaFree
     *************************/
    else if (req.type == RequestType::FREE) {
        handleFree(req, rep);
    }
    /*************************
     * worker finished, cleanup
     *************************/
    else if (req.type == RequestType::FINISHED) {
        handleFinish(req, rep);
    }
    /*************************
     * store requested amount of memory
     *************************/
    else if (req.type == RequestType::MEMREQUESTED) {
        handleRequestedMemory(req, rep);
    }
    /*************************
     * kernel asking to run
     *************************/
    else if (req.type == RequestType::KERNEL_IN) {
        handleKernelIn(req, rep);
    }
    /*************************
     * kernel done
     *************************/
    else if (req.type == RequestType::KERNEL_OUT) {
        handleKernelOut(req, rep);
    }
}

void Server::handleMalloc(Request& req, Reply& rep) {
    uint64_t requested = req.data.size;
    std::string worker_id(req.worker_id);

    workers_info[worker_id].mem_used += requested;
    (void)rep;
    
    if (requested > gpu_memory_total-gpu_memory_used) {
        //TODO: maybe check not good citizens and move them.
        //TODO: request migration by memory
    }
}

void Server::handleFree(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    workers_info[worker_id].mem_used -= req.data.size;
    (void)rep;
}

void Server::handleRequestedMemory(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    workers_info[worker_id].requested_memory = req.data.size;
    (void)rep;
}

void Server::handleFinish(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    workers_info.erase(worker_id);
    (void)rep;
}

void Server::handleKernelIn(Request& req, Reply& rep) {
    kernels_queued++;
    printf(" >>in: there are now %d kernels queued\n", kernels_queued);

    //check if we are just debugging
    char* dbg_mig = std::getenv("SG_DEBUG_MIGRATION");
    if (dbg_mig) {
        debug_kernel_count += 1;
        if (debug_kernel_count % 2 == 0) {
            printf("SG_DEBUG_MIGRATION:  kernel #%d, setting migration on\n", debug_kernel_count);

            if (!strcmp(dbg_mig, "1")) {
                printf("SG_DEBUG_MIGRATION:  setting EXEECUTION migration on\n");
                rep.migrate = Migration::KERNEL;
                rep.target_device = gpu == 2 ? 3 : 2;
            }
            else if (!strcmp(dbg_mig, "2")) {
                printf("SG_DEBUG_MIGRATION:  setting MEMORY migration on\n");
                rep.migrate = Migration::MEMORY;
                rep.target_device = gpu == 2 ? 3 : 2;
            }
        }
        else 
            rep.migrate = Migration::NOPE;
    }

    (void)req;
}

void Server::handleKernelOut(Request& req, Reply& rep) {
    (void)req; (void)rep;
    kernels_queued--;
    printf(" <<out: there are now %d kernels queued\n", kernels_queued);
}

//end of namespace
};

