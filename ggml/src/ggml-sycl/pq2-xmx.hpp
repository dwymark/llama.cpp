#pragma once

#include "common.hpp"

// PQ2_0 weights times a batch of FP32 activation rows on the XMX engines: the activations are quantized to
// int8 with one scale per 128 values, and each 128-value weight block is decoded to int8 in registers.
// Returns false when the shapes or the device do not fit the kernel.
bool ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const void * src0_pq2, const float * src1,
                               float * dst, int64_t nrows, int64_t ncols_k, int64_t ntokens, int64_t ldc,
                               dpct::queue_ptr stream);
