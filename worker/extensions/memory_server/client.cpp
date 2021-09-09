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
#include <condition_variable>
#include <mutex>
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/common_context.h"
#include "extensions/memory_server/client.hpp"
#include "extensions/memory_server/common.hpp"

/**************************************
 * 
 *    Global functions, used by spec and helper functions. Mostly translate a global call to a client call
 * 
 * ************************************/
std::atomic<uint32_t> pending_cmds(0);
std::atomic<bool> thread_block(false);
std::mutex thread_block_mutex;
std::condition_variable thread_block_cv;

void wait_for_cmd_drain() {
    thread_block = true;
    //we count as a cmd, so wait for 1 pending
    while (pending_cmds != 1) ;
}

void release_cmd_handlers() {
  std::unique_lock<std::mutex> lk(thread_block_mutex);
  thread_block = false;
  thread_block_cv.notify_all();
}

void __cmd_handle_in() {
    if (thread_block) {
        std::unique_lock<std::mutex> lk(thread_block_mutex);
        while (thread_block)
            thread_block_cv.wait(lk);
    }
    pending_cmds++;
#ifndef NDEBUG
    //std::cerr << "in: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

void __cmd_handle_out() {
    pending_cmds--;
#ifndef NDEBUG
    //std::cerr << "out: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

bool allContextsEnabled;
bool __internal_allContextsEnabled() {
    return allContextsEnabled;
}
bool __internal_setAllContextsEnabled(bool f) {
    allContextsEnabled = f;
}

int32_t __internal_getDeviceCount() {
    return GPUMemoryServer::Client::getInstance().device_count;
}

void __internal_setDeviceCount(uint32_t dc) {
    GPUMemoryServer::Client::getInstance().device_count = dc;
}

uint32_t __internal_getCurrentDevice() {
    return GPUMemoryServer::Client::getInstance().current_device;
}

void __internal_kernelIn() {
    GPUMemoryServer::Client::getInstance().kernelIn();
}

void __internal_kernelOut() {
    GPUMemoryServer::Client::getInstance().kernelOut();
}

CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    // TODO for TF
    CUresult cur;
    GPUMemoryServer::Client::getInstance().kernelIn();
    cur = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
    GPUMemoryServer::Client::getInstance().kernelOut();
    return cur;
}

CUresult __internal_cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    void* ptr;
    cudaError_t err = GPUMemoryServer::Client::getInstance().localMalloc(&ptr, bytesize);
    *dptr = ptr;
    return (CUresult) err;
}

CUresult __internal_cuMemFree(CUdeviceptr dptr) {
    cudaError_t err = GPUMemoryServer::Client::getInstance().localFree((void*)dptr);
    return (CUresult) err;
}

cudaError_t __internal_cudaMalloc(void **devPtr, size_t size) {
    return GPUMemoryServer::Client::getInstance().localMalloc(devPtr, size);
}

cudaError_t __internal_cudaFree(void *devPtr) {
    return GPUMemoryServer::Client::getInstance().localFree(devPtr);
}

cudaStream_t __translate_stream(cudaStream_t key) {
    if (__internal_allContextsEnabled()) {
        if (key == 0) return 0;
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().streams_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

cudaEvent_t __translate_event(cudaEvent_t key) {
    if (__internal_allContextsEnabled()) {
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().events_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

/**************************************
 * 
 *    Functions for GPUMemoryServer namespace, Client class
 * 
 * ************************************/

namespace GPUMemoryServer {

    //uint32_t kc = 0;
    Client::Client() {
        og_device = -1;
        context = zmq_ctx_new();
        migrated_type = Migration::NOPE;

        connectToCentral();
     }

     Client::~Client() {
        zmq_close(central_socket);
    }

    void Client::connectToCentral() {
        central_socket = zmq_socket(context, ZMQ_REQ);
        while (1) { 
            int ret = zmq_connect(central_socket, GPUMemoryServer::get_central_socket_path().c_str());
            if (ret == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            printf(" !!! GPU Client couldn't connect to central server [%d]\n");
        }
        //printf("GPU Client succesfully connected to server [%d]\n", gpuid);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        auto al = std::make_unique<Client::LocalAlloc>(current_device);

        int ret = al->physAlloc(size);
        if (ret) return cudaErrorMemoryAllocation;
        ret = al->reserveVaddr();
        if (ret) return cudaErrorMemoryAllocation;
        ret = al->map();
        if (ret) return cudaErrorMemoryAllocation;

        *devPtr = (void*)al->devptr;
        local_allocs.push_back(std::move(al));
        //reportMalloc(size); 
        return cudaSuccess;
    }

    cudaError_t Client::localFree(void* devPtr) {
        // cudaFree with nullptr does no operation
        if (devPtr == nullptr) {
            return cudaSuccess;
        }

        for (auto it = local_allocs.begin(); it != local_allocs.end(); ++it) {
            if ((*it)->devptr == devPtr) {
                local_allocs.erase(it);
                //GPUMemoryServer::Client::getInstance().reportFree(it->second->size); 
                return cudaSuccess;
            }
        }
        
        fprintf(stderr, "### ILLEGAL cudaFree call on devPtr %x\n", (uint64_t)devPtr);
        return cudaErrorInvalidValue;   
    }

    void Client::sendRequest(Request &req) {
        //just quit if we are not reporting to gpu server, except READY, which must go through
        if (enable_reporting == false && req.type != READY) {
            return;
        }

        req.gpu = current_device;
        strncpy(req.worker_id, uuid.c_str(), MAX_UUID_LEN);
        if (strlen(uuid.c_str()) > MAX_UUID_LEN) {
            printf(" ### uuid %S IS LONGER THAN %d, THIS IS AN ISSUE\n", uuid.c_str(), MAX_UUID_LEN);
        }

        sockmtx.lock();
        if (zmq_send(central_socket, &req, sizeof(Request), 0) == -1) {
            printf(" ### zmq_send errno %d\n", errno);
        }
        Reply rep;
        zmq_recv(central_socket, &rep, sizeof(Reply), 0);
        sockmtx.unlock();

        //unlock and handle it
        handleReply(rep);
    }

    void Client::handleReply(Reply& reply) {
        //only migrate if server told us so and we haven't migrated before
        if (reply.code == ReplyCode::MIGRATE && current_device == og_device)
            migrateToGPU(reply.data.migration.target_device, reply.data.migration.type);
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
        if (err) {
            printf("### CUDA err on cudaSetDevice(%d): %d\n", id, err);
            exit(1);
        }
        //printf("Worker [%s] setting current device to %d\n", uuid.c_str(), id);
        //this if triggers once, when worker is setting up
        if (og_device == -1) 
            og_device = id;
        current_device = id;

        //set on context so shadow threads also change context
        auto ccontext = ava::CommonContext::instance();
        ccontext->current_device = id;
    }

    void Client::resetCurrentGPU() {
        current_device = og_device;
        setCurrentGPU(current_device);
        auto ccontext = ava::CommonContext::instance();
        ccontext->current_device = current_device;
        migrated_type = Migration::NOPE;
    }

    void Client::migrateToGPU(uint32_t new_gpuid, Migration migration_type) {
        std::cerr << "[[[MIGRATION]]]\n" << std::endl;

        //wait for all cmd handlers to stop
        wait_for_cmd_drain();
        std::cerr << "All cmd handlers are stopped.. migrating\n" << std::endl;
        //mark that we migrated
        migrated_type = migration_type;

        cudaDeviceSynchronize();
        setCurrentGPU(new_gpuid);

        if (migration_type == Migration::KERNEL) {
            printf("Migration by kernel mode, changing device to [%d]\n", new_gpuid);
            //cudaDeviceEnablePeerAccess(new_gpuid, 0);
            setCurrentGPU(new_gpuid);
            printf("cudaDeviceEnablePeerAccess: new device [%d] can access memory on old [%d]\n", new_gpuid, og_device);
            cudaDeviceEnablePeerAccess(og_device, 0);
        }
        else if (migration_type == Migration::TOTAL) {
            std::cerr << "Full migration, changing device to" << new_gpuid << std::endl;
            //all we gotta do is change all allocations to new device
            for (auto const& al : local_allocs) {
                al->moveToGPU(new_gpuid);
            }
            cudaDeviceSynchronize();
        }

        //release handlers
        release_cmd_handlers();
    }

    //clean up all memory that are in cd ONLY
    void Client::cleanup(uint32_t cd) {
        cudaSetDevice(cd);
        //erase only memory in GPU current_device
        for (auto it = local_allocs.begin(); it != local_allocs.end(); ) {
            if ((*it)->device_id == cd)
                it = local_allocs.erase(it);
            else
                ++it;
        }
        //report to server
        reportCleanup(cd);
        //reset to device
        cudaSetDevice(current_device);
    }

    void Client::fullCleanup() {
        reportCleanup(0);  //TODO

        for (int i = 0 ; i < device_count ; i++) 
            cleanup(i);
        local_allocs.clear();

        //iterate over map of maps, destroying streams
        while (!streams_map.empty())
            __helper_destroy_stream((streams_map.begin())->first);

        streams_map.clear();

        //clear leftover events
        while (!events_map.empty()) 
            __helper_cudaEventDestroy((events_map.begin())->first);
        events_map.clear();

        //reset
        cudaSetDevice(current_device);
    }

    void Client::notifyReady() {
        Request req;
        req.type = RequestType::READY;
        req.data.ready.port = listen_port;
        std::cerr << "  worker on port " << listen_port << " sending ready message" << std::endl;
        sendRequest(req);
    }

    void Client::reportCleanup(uint32_t gpuid) {
        Request req;
        req.type = RequestType::FINISHED;
        //TODO: report cleanup only device
        //sendRequest(req);
    }

    void Client::kernelIn() {
        //printf("%d kernels\n", ++kc);
        if (migrated_type != Migration::NOPE) {
            //if we have migrated, lets just skip..
            return;
        }
        Request req;
        req.type = RequestType::KERNEL_IN;
        sendRequest(req);
    }
    
    void Client::kernelOut() {
        Request req;
        req.type = RequestType::KERNEL_OUT;
        //sendRequest(req);
    }

        void Client::reportFree(uint64_t size) {
        Request req;
        req.type = RequestType::FREE;
        req.data.size = size;
        sendRequest(req);
    }

    void Client::reportMemoryRequested(uint64_t mem_mb) {
        Request req;
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        //sendRequest(req);
    }

    void Client::reportMalloc(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    Client::LocalAlloc::LocalAlloc(uint32_t device) {
        devptr = 0;
        device_id = device;
    }

    Client::LocalAlloc::~LocalAlloc() {
        cuMemUnmap(devptr, size);
        cuMemAddressFree(devptr, size);
        cuMemRelease(phys_mem_handle);
    }

    int Client::LocalAlloc::physAlloc(uint32_t req_size) {
        size_t aligned_sz;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUresult err;
        err = cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (err) return 1;
        
        size = ((req_size + aligned_sz - 1) / aligned_sz) * aligned_sz;
        printf("cuMemCreate of %lu\n", size);
        err = cuMemCreate(&phys_mem_handle, size, &prop, 0);
        if (err) return 1;

        return 0;
    }

    int Client::LocalAlloc::reserveVaddr() {
        uint64_t addr = vasBitmask();
        CUresult err = cuMemAddressReserve(&devptr, size, 0ULL, addr, 0ULL);
        if (err) return 1;
        printf("VA reserve at %lu, got back %lu\n", addr, devptr);

        return 0;
    }

    int Client::LocalAlloc::map() {
        CUresult err = cuMemMap(devptr, size, 0ULL, phys_mem_handle, 0ULL);
        if (err) return 1;
        err = cuMemSetAccess(devptr, size, &accessDesc, 1ULL);
        if (err) return 1;
        return 0;
    }

    int Client::LocalAlloc::moveToGPU(uint32_t new_device) {
        //save current state
        CUmemGenericAllocationHandle old_handle = phys_mem_handle;
        uint32_t old_device = device_id;
        CUresult err;
        int err2;
        //get new temporary vaddr
        uint64_t temp_devptr;
        err = cuMemAddressReserve(&temp_devptr, size, 0ULL, vasBitmask(), 0ULL);
        if (err) return 1;
        //unlink current mapping       
        err = cuMemUnmap(devptr, size);
        if (err) return 1;
        //link temp to current phys allocation
        err = cuMemMap(temp_devptr, size, 0ULL, phys_mem_handle, 0ULL);
        if (err) return 1;
        err = cuMemSetAccess(temp_devptr, size, &accessDesc, 1ULL);
        if (err) return 1;

        //change current device
        device_id = new_device;
        //aloc on new gpu, this changes data in this struct
        err2 = physAlloc(size);
        if (err2) return 1;
        //map new allocation to original vaddr (devptr hasnt changed)
        err2 = map();
        if (err2) return 1;
        //copy data from old to new
        cudaError_t err3 = cudaMemcpyPeer((void*) devptr, new_device, (void*)temp_devptr, old_device, size);
        if (err3) return 1;

        //cleanup
        err = cuMemUnmap(temp_devptr, size);
        if (err) return 1;
        err = cuMemAddressFree(temp_devptr, size);
        if (err) return 1;
        err = cuMemRelease(old_handle);
        if (err) return 1;

        return 0;
    }

//end of GPUMemoryServer nameserver
}