#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Parallelism (2D: TP x PP)
    int tp_rank = 0;
    int pp_rank = 0;
    int tp_degree = 1;
    int pp_degree = 1;
    uint32_t stage_layer_start = 0; // inclusive
    uint32_t stage_layer_end = 0;   // exclusive
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    // For PP activation handoff (host-side bounce in first version)
    void **activation_in_host_arr;   // array size = pp_degree-1, index = stage-1
    void **activation_out_host_arr;  // array size = pp_degree-1, index = stage
    uint32_t activation_ntok;   // equals ntok
    // host-side synchronization between PP stages (size = pp_degree - 1)
    std::mutex **act_mtx_arr;                // index: boundary
    std::condition_variable **act_cv_arr;    // index: boundary
    uint8_t *act_ready_arr;                  // index: boundary
};

struct JiugeModel {
    JiugeMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;
    // 2D parallelism
    int tp_degree = 0;
    int pp_degree = 0;
    // host activation bounce buffers and sync for PP boundaries
    std::vector<std::shared_ptr<Storage>> act_buffers; // size = max(pp_degree-1, 0)
    std::vector<std::unique_ptr<std::mutex>> act_mtx;
    std::vector<std::unique_ptr<std::condition_variable>> act_cv;
    std::vector<uint8_t> act_ready;
    std::vector<void *> act_in_ptrs;
    std::vector<void *> act_out_ptrs;
    std::vector<std::mutex *> act_mtx_ptrs;
    std::vector<std::condition_variable *> act_cv_ptrs;

    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif
