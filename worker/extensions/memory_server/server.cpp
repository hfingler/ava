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
//std::uniform_int_distribution<int> intdist(0,1);

Server::Server(uint16_t gpu, std::string self_unix_socket_path, std::string central_server_unix_socket_path) {
    this->gpu = gpu;
    //this->gpu_memory_total = total_memory;  
    this->gpu_memory_used = 0;
    this->kernels_queued = 0;
    this->self_unix_socket_path = self_unix_socket_path;
    this->central_server_unix_socket_path = central_server_unix_socket_path;
}

void Server::run() {
    //create zmq listener
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    //make sure it doesn't exist
    unlink(self_unix_socket_path.c_str());
    int rc = zmq_bind(responder, self_unix_socket_path.c_str());
    if (rc == -1) {
        printf("BIND FAILED\n");
    }
    printf("Memory server of GPU %d ready, bound to %s\n", gpu, self_unix_socket_path.c_str());
    
    central_socket = zmq_socket(context, ZMQ_REQ);
    while (1) { 
        int ret = zmq_connect(central_socket, central_server_unix_socket_path.c_str());
        if (ret == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        printf(" !!! GPU Client couldn't connect to central server\n");
    }

    Reply rep;
    Request req;
    while(1) {
        zmq_recv(responder, &req, sizeof(Request), 0);
        handleRequest(req, rep);
        zmq_send(responder, &rep, sizeof(Reply), 0);
    }
}

void Server::handleRequest(Request& req, Reply& rep) {
    //reset reply
    rep.data.migrate = Migration::NOPE;

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
    /*************************
     *  worker ready
     *************************/
    else if (req.type == RequestType::READY) {
        handleReady(req, rep);
    }
    else {
        std::cerr << " Server got an unknown request type!\n";
    }
}

void Server::handleReady(Request& req, Reply& rep) {
    //just forward to central
    if (zmq_send(central_socket, &req, sizeof(Request), 0) == -1) {
        printf(" ### zmq_send errno %d\n", errno);
    }
}

void Server::handleMalloc(Request& req, Reply& rep) {
    uint64_t requested = req.data.size;
    std::string worker_id(req.worker_id);

    workers_info[worker_id].mem_used += requested;
    (void)rep;
    
    //if (requested > gpu_memory_total-gpu_memory_used) {
        //TODO: maybe check not good citizens and move them.
        //TODO: request migration by memory
    //}
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
    //std::cerr << " >>in: there are now" <<  kernels_queued << "kernels queued" <<std::endl;

    //check if we are just debugging
    char* dbg_mig = std::getenv("SG_DEBUG_MIGRATION");
    if (dbg_mig) {
        debug_kernel_count += 1;
        uint32_t dbgi = atoi(std::getenv("SG_DEBUG_MIGRATION"));

        //if debug type 1 or 2 and multiple of 2
        if ((dbgi == 1 || dbgi == 2)  &&  debug_kernel_count % 2 == 0) {
            printf("SG_DEBUG_MIGRATION:  kernel #%d, setting migration on\n", debug_kernel_count);

            if (!strcmp(dbg_mig, "1")) {
                printf("SG_DEBUG_MIGRATION:  setting EXECUTION migration on\n");
                rep.data.migrate = Migration::KERNEL;
                rep.target_device = gpu == 1 ? 2 : 1;
            }
            else if (!strcmp(dbg_mig, "2")) {
                printf("SG_DEBUG_MIGRATION:  setting MEMORY migration on\n");
                rep.data.migrate = Migration::TOTAL;
                rep.target_device = gpu == 1 ? 2 : 1;
            }
        }
        //if a multiple of 10, divide 1 by it and that's the prob
        else if (dbgi >= 10 && dbgi % 10 == 0) {
            float prob = 1.0 / dbgi;
            if (dis01(rgen) <= prob) { 
            //if (1) {
                rep.data.migrate = Migration::TOTAL;
                uint32_t dg = gpu == 0 ? 1 : 0;
                rep.target_device = dg;
                std::cerr << " SG_DEBUG_MIGRATION: TOTAL random migration triggered:  " << gpu  << " -> " << dg << " with prob " << prob << std::endl;
            }
        }
        //if a multiple of 10 after -1, divide 1 by it and that's the prob, use kernel migration
        else if (dbgi >= 11 && (dbgi-1) % 10 == 0) {
            float prob = 1.0 / dbgi;
            if (dis01(rgen) <= prob) { 
                rep.data.migrate = Migration::KERNEL;
                uint32_t dg = gpu == 0 ? 1 : 0;
                rep.target_device = dg;
                std::cerr << " SG_DEBUG_MIGRATION: KERNEL random migration triggered:  " << gpu  << " -> " << dg << " with prob " << prob << std::endl;
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

