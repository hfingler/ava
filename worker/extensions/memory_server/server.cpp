#include <map>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <random>
#include <zmq.h>
#include "server.hpp"
#include "extensions/memory_server/common.hpp"
#include <iostream>

namespace GPUMemoryServer {

static uint32_t debug_kernel_count = 0;
//std::random_device rdd;
//std::mt19937 rgen(rdd());
std::mt19937 rgen{0};
std::uniform_real_distribution<float> dis01(0, 1);
std::uniform_int_distribution<int> intdist(0,1);

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
        std::cerr << " >> server " << gpu <<  " waiting for request" << std::endl;
        zmq_recv(responder, &req, sizeof(Request), 0);
        std::cerr << " >> server " << gpu <<  " got a request" << std::endl;
        handleRequest(req, rep);
        std::cerr << " >> sending response" << std::endl;
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
    std::cerr << " >>in: there are now" <<  kernels_queued << "kernels queued" <<std::endl;

    //check if we are just debugging
    char* dbg_mig = std::getenv("SG_DEBUG_MIGRATION");
    if (dbg_mig) {
        debug_kernel_count += 1;

        //if debug type 1 or 2 and multiple of 2
        if (!strcmp(dbg_mig, "1") && !strcmp(dbg_mig, "2")  &&  debug_kernel_count % 2 == 0) {
            printf("SG_DEBUG_MIGRATION:  kernel #%d, setting migration on\n", debug_kernel_count);

            if (!strcmp(dbg_mig, "1")) {
                printf("SG_DEBUG_MIGRATION:  setting EXECUTION migration on\n");
                rep.migrate = Migration::KERNEL;
                rep.target_device = gpu == 1 ? 2 : 1;
            }
            else if (!strcmp(dbg_mig, "2")) {
                printf("SG_DEBUG_MIGRATION:  setting MEMORY migration on\n");
                rep.migrate = Migration::TOTAL;
                rep.target_device = gpu == 1 ? 2 : 1;
            }
        }
        //3 means randomly chosen over a %
        else if (!strcmp(dbg_mig, "3")) {
            //10% change of migration
            if (dis01(rgen) <= 0.01) { 
            //if (1) {
                //rep.migrate = Migration::KERNEL;
                rep.migrate = Migration::TOTAL;
                uint32_t dg;
                //while (1) {
                //    dg = intdist(rgen);
                //    if (dg != gpu) break;
                //}
                dg = gpu == 0 ? 1 : 0;

                rep.target_device = dg;
                std::cerr << " SG_DEBUG_MIGRATION: random migration triggered:  " << gpu  << " -> " << dg << std::endl;
            }  
        }
    }

    (void)req;
}

void Server::handleKernelOut(Request& req, Reply& rep) {
    (void)req; (void)rep;
    kernels_queued--;
    //printf(" <<out: there are now %d kernels queued\n", kernels_queued);
}

//end of namespace
};

