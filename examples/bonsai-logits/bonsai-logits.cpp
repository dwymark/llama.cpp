#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void log_message(ggml_log_level level, const char * text, void *) {
    if (level <= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

int main(int argc, char ** argv) {
    if (argc != 8 && argc != 9) {
        std::fprintf(stderr, "usage: llama-bonsai-logits model backend-dir corpus output prompt-tokens decode-tokens interval [batch]\n");
        return 2;
    }
    int prompt_count, decode_count, interval, batch;
    try {
        prompt_count = std::stoi(argv[5]);
        decode_count = std::stoi(argv[6]);
        interval = std::stoi(argv[7]);
        batch = argc == 9 ? std::stoi(argv[8]) : 512;
    } catch (const std::exception &) {
        std::fprintf(stderr, "token counts, interval, and batch must be integers\n");
        return 2;
    }
    if (batch < 1 || batch > 4096) return 2;
    if (prompt_count < 1 || decode_count < 1 || interval < 1 || decode_count % interval ||
        prompt_count > INT_MAX - decode_count) return 2;
    std::ifstream corpus_file(argv[3], std::ios::binary);
    if (!corpus_file) return 3;
    const std::string corpus((std::istreambuf_iterator<char>(corpus_file)), std::istreambuf_iterator<char>());
    if (corpus.size() > INT_MAX) return 3;
    llama_log_set(log_message, nullptr);
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 4;
    auto cp = llama_context_default_params();
    cp.n_ctx = prompt_count + decode_count; cp.n_batch = batch; cp.n_ubatch = batch;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 5;
    const auto * vocab = llama_model_get_vocab(model);
    const int needed = -llama_tokenize(vocab, corpus.data(), int(corpus.size()), nullptr, 0, true, true);
    if (needed < prompt_count + decode_count) return 6;
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, corpus.data(), int(corpus.size()), tokens.data(), needed, true, true) != needed) return 6;
    auto * out = std::fopen(argv[4], "wb");
    if (!out) return 7;
    const int32_t nv = llama_vocab_n_tokens(vocab), steps = 1 + decode_count / interval;
    // The comparison script reads two int32 dimensions followed by full FP32 vocabulary rows.
    if (std::fwrite(&nv, sizeof(nv), 1, out) != 1 || std::fwrite(&steps, sizeof(steps), 1, out) != 1) return 9;
    auto dump = [&]() {
        const float * logits = llama_get_logits(ctx);
        for (int i = 0; i < nv; ++i) if (!std::isfinite(logits[i])) return false;
        return std::fwrite(logits, sizeof(float), nv, out) == size_t(nv);
    };
    std::fprintf(stderr, "prefill batch=%d\n", batch);
    const auto begin = std::chrono::steady_clock::now();
    for (int offset = 0; offset < prompt_count; offset += batch) {
        const int count = std::min(batch, prompt_count - offset);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, count))) return 8;
    }
    if (!dump()) return 9;
    for (int i = 0; i < decode_count; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + prompt_count + i, 1))) return 8;
        if ((i + 1) % interval == 0) {
            if (!dump()) return 9;
            std::fprintf(stderr, "checked position=%d elapsed=%.3f seconds\n", prompt_count + i + 1,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
            std::fflush(stderr);
        }
    }
    if (std::fclose(out)) return 10;
    std::fprintf(stderr, "complete prompt=%d decode=%d positions=%d vocabulary=%d\n", prompt_count, decode_count, steps, nv);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
