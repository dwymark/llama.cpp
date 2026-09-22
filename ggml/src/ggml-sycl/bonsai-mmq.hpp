#pragma once

#include "common.hpp"

bool ggml_sycl_try_pq2_prefill(ggml_backend_sycl_context & ctx, const void * weights,
                              const float * input, float * output, int64_t k, int64_t m,
                              int64_t n, int ldc, const dpct::queue_ptr & stream);
