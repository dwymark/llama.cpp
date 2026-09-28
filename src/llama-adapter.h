#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    // steering mode, entered by steer_configure: per-layer flags choose which layers add the
    // direction (scaled per token by `scale`, F32 [1, n_tokens]), read the projection onto the
    // layer's unit direction, and cap it from below; projections read are appended to `readout`
    ggml_tensor * apply_steer(ggml_context * ctx, ggml_tensor * cur, int il, ggml_tensor * scale,
                              std::vector<std::pair<int, ggml_tensor *>> & readout) const;

    bool steer_configure(const llama_model & model, const float * unit, const uint8_t * flags, int32_t n_layer);
    void steer_set_tau(const float * tau, int32_t n_layer);

    bool steering() const { return !steer_flags.empty(); }
    float scale_for(llama_seq_id seq_id) const;

    // bumped by every steer_configure; graph reuse requires it to match
    uint32_t version = 0;

    float scale_default = 1.0f;
    std::map<llama_seq_id, float> seq_scale;

    std::vector<uint8_t> steer_flags; // per layer, empty outside steering mode
    std::vector<int>     read_layers; // layers with STEER_READ, ascending

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
    std::vector<ggml_tensor *> units;   // per layer, unit directions for reading and capping
    std::vector<ggml_tensor *> taus;    // per layer, F32 [1] capping thresholds
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
