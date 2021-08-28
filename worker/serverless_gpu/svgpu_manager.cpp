#include "svgpu_manager.hpp"
#include <nvml.h>
#include <sys/wait.h>
#include <boost/algorithm/string/join.hpp>
#include "declaration.h"
#include <string>
#include <memory>
#include <zmq.h>
#include "extensions/memory_server/common.hpp"
#include "resmngr.grpc.pb.h"
#include "resmngr.pb.h"

#include "scheduling/first_fit.hpp"

/*************************************
 *
 *    gRPC methods
 *
 *************************************/

std::string SVGPUManager::ResMngrClient::registerSelf() {
    resmngr::RegisterGPUNodeRequest request;
    std::cout << "[SVLESS-MNGR]: registerSelf to resource manager.. " << std::endl;
    ClientContext context;
    resmngr::RegisterGPUNodeResponse reply;
    Status status = stub_->RegisterGPUNode(&context, request, &reply);
    if (!status.ok()) {
        std::cerr << "Error registering self with resmngr:" << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::exit(1);
    }

    return reply.uuid();
}

void SVGPUManager::ResMngrClient::addGPUWorker(std::string uuid) {
    resmngr::AddGPUWorkerRequest request;
    request.set_uuid(uuid);
    request.set_workers(1);

    std::cout << "[SVLESS-MNGR]: adding gpu worker to resource manager.. " << std::endl;
    ClientContext context;
    resmngr::AddGPUWorkerResponse reply;
    Status status = stub_->AddGPUWorker(&context, request, &reply);
    if (!status.ok()) {
        std::cerr << "Error registering self with resmngr:" << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::exit(1);
    }
}

void SVGPUManager::registerSelf() {
    resmngr_client = new ResMngrClient(grpc::CreateChannel(resmngr_address, grpc::InsecureChannelCredentials()));
    uuid = resmngr_client->registerSelf();
}

/*************************************
 *
 *    AvA HandleRequest override
 *
 *************************************/

ava_proto::WorkerAssignReply SVGPUManager::HandleRequest(const ava_proto::WorkerAssignRequest &request) {
    ava_proto::WorkerAssignReply reply;

    if (request.gpu_count() > 1) {
        std::cerr << "ERR: someone requested more than 1 GPU, no bueno" << std::endl;
        return reply;
    }

    std::cerr << "[SVLESS-MNGR]: API server request arrived, asking for schedule.." << std::endl;
    while (true) {
        GPUMemoryServer::Request req;
        req.type = GPUMemoryServer::RequestType::SCHEDULE;
        if (zmq_send(zmq_central_socket, &req, sizeof(GPUMemoryServer::Request), 0) == -1) {
            printf(" ### zmq_send errno %d\n", errno);
        }

        GPUMemoryServer::Reply rep;
        zmq_recv(zmq_central_socket, &rep, sizeof(GPUMemoryServer::Reply), 0);
        
        if (rep.code == GPUMemoryServer::ReplyCode::OK) {
            std::cerr << "[SVLESS-MNGR]: scheduled at port " << rep.data.ready.port << std::endl;
            reply.worker_address().push_back("0.0.0.0:" + std::to_string(rep.data.ready.port));
            return reply;
        }
        else if (rep.code == GPUMemoryServer::ReplyCode::RETRY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}

/*************************************
 *
 *    General stuff
 *
 *************************************/

void SVGPUManager::launchReportServers() {
    // launch central
    central_server_thread = std::thread(&SVGPUManager::centralManagerLoop, this);
    std::cerr << "[SVLESS-MNGR]: Launched central server, sleeping to give them time to spin up" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //lets connect ourself to the central server, so that when we need to schedule we can talk to it
    zmq_context = zmq_ctx_new();
    zmq_central_socket = zmq_socket(zmq_context, ZMQ_REQ);
    while (1) { 
        int ret = zmq_connect(zmq_central_socket, GPUMemoryServer::get_central_socket_path().c_str());
        if (ret == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        printf(" !!! Manager couldn't connect to central server\n");
    }
}

SVGPUManager::SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
            std::vector<std::string> &worker_env, uint16_t ngpus, uint16_t gpu_offset, std::string resmngr_address,
            std::string scheduler_name, uint32_t precreated_workers)
    : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {

    
    this->n_gpus = ngpus;
    this->gpu_offset = gpu_offset;
    this->uuid_counter = 0;
    this->resmngr_address = resmngr_address;
    //update to real values using nvml
    setRealGPUOffsetCount();

    launchReportServers();
    
    if (!resmngr_address.empty())
        registerSelf();

    //now that everything is up, we can launch workers
    this->precreated_workers = precreated_workers;
    for (unsigned int gpu = gpu_offset; gpu < gpu_offset+n_gpus; gpu++) {
        for (int i = 0 ; i < precreated_workers ; i++) {
            launchWorker(gpu);
        }
    }
};

uint32_t SVGPUManager::launchWorker(uint32_t gpu_id) {
    // Start from input environment variables
    std::vector<std::string> environments(worker_env_);

    for (std::string &e : environments) {
        printf("   %s", e.c_str());
    }

    std::string visible_devices = "GPU_DEVICE=" + std::to_string(gpu_id);
    environments.push_back(visible_devices);

    // Let API server use TCP channel
    environments.push_back("AVA_CHANNEL=TCP");

    std::string worker_uuid = "AVA_WORKER_UUID=" + std::to_string(uuid_counter);
    environments.push_back(worker_uuid);
    uuid_counter++;

    // Pass port to API server
    auto port = worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::string> parameters;
    parameters.push_back(std::to_string(port));

    // Append custom API server arguments
    for (const auto &argv : worker_argv_) {
        parameters.push_back(argv);
    }

    for (auto& element : environments) {
        printf("  > %s\n", element.c_str());
    }

    std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\"" << boost::algorithm::join(environments, " ")
                << " " << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

    auto child_pid = SpawnWorker(environments, parameters);

    auto child_monitor = std::make_shared<std::thread>(
        [](pid_t child_pid, uint32_t port, std::map<pid_t, std::shared_ptr<std::thread>> *worker_monitor_map) {
            pid_t ret = waitpid(child_pid, NULL, 0);
            std::cerr << "[pid=" << child_pid << "] API server at ::" << port << " has exit (waitpid=" << ret << ")"
                    << std::endl;
            worker_monitor_map->erase(port);
        },
        child_pid, port, &worker_monitor_map_);
    child_monitor->detach();
    worker_monitor_map_.insert({port, child_monitor});

    return port;
}

void SVGPUManager::setRealGPUOffsetCount() {
    nvmlReturn_t result;
    uint32_t device_count;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to query device count NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    // bounds check requested gpus, use all gpus if n_gpus == 0
    if (n_gpus == 0) {
        n_gpus = device_count - gpu_offset;
    } 

    std::cout << "[SVLESS-MNGR]: set GPU offset to " << gpu_offset << " and GPU count to " << n_gpus << std::endl;
}

void SVGPUManager::createScheduler(std::string name) {
    if (name == "firstfit") {
        this->scheduler = new FirstFit(&gpu_workers, &gpu_states);
    }

}