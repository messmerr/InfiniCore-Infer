#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
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

    // Reusable base buffers to reduce frequent allocations during inference
    // They are oversized 1-D buffers; actual tensors are lightweight views via memShare
    std::shared_ptr<Tensor> base_dt_logits_buf; // float/bf16/fp16 depending on model logits dtype
    std::shared_ptr<Tensor> base_i64_buf;
    std::shared_ptr<Tensor> base_u32_buf;
    size_t cap_dt_logits_elems = 0;
    size_t cap_i64_elems = 0;
    size_t cap_u32_elems = 0;

    // Reusable pinned host storages for small H2D/D2H transfers
    std::shared_ptr<Storage> host_u32_storage;
    std::shared_ptr<Storage> host_i64_storage;
    size_t host_u32_elems = 0;
    size_t host_i64_elems = 0;

    // Additional streams for per-request intra-layer concurrency
    std::vector<infinirtStream_t> substreams;
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
};

struct JiugeModel {
    JiugeMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif
