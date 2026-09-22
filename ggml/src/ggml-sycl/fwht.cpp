#include "fwht.hpp"

#include <cmath>

template <int N, bool signed_input = false>
static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const float * signs, int64_t sign_count, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();
    const int64_t sign_base = signed_input ? (r % (sign_count / N)) * N : 0;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        float value = src[i * WARP_SIZE + lane];
        if constexpr (signed_input) {
            value *= signs[sign_base + i * WARP_SIZE + lane];
        }
        reg[i] = value * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N, bool signed_input = false>
static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream, const float * signs = nullptr, int64_t sign_count = 0) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, signed_input>(src, dst, n_rows, scale, signs, sign_count, item);
                         });
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float *   src_d  = (const float *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            launch_fwht<1024>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

int ggml_sycl_try_signed_fwht(ggml_backend_sycl_context & ctx, ggml_cgraph * graph, int index) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FWHT_SIGNS", 0);
    if (!enabled || !g_ggml_sycl_enable_fusion ||
        !ggml_can_fuse_subgraph(graph, index, { GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { index + 2 })) {
        return 0;
    }
    const ggml_tensor * mul = graph->nodes[index];
    const ggml_tensor * reshaped = graph->nodes[index + 1];
    ggml_tensor * dst = graph->nodes[index + 2];
    const ggml_tensor * src = mul->src[0];
    const ggml_tensor * signs = mul->src[1];
    if (reshaped->src[0] != mul || dst->src[1] != reshaped ||
        ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
        src->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst) ||
        signs->ne[0] != src->ne[0] || ggml_nrows(signs) != 1 ||
        !ggml_are_same_shape(src, mul) || !ggml_are_same_shape(reshaped, dst)) {
        return 0;
    }
    const int n = int(dst->ne[0]);
    if (signs->ne[0] % n != 0) {
        return 0;
    }
    const uintptr_t input_begin = reinterpret_cast<uintptr_t>(src->data);
    const uintptr_t output_begin = reinterpret_cast<uintptr_t>(dst->data);
    const size_t bytes = ggml_nbytes(dst);
    if (input_begin < output_begin + bytes && output_begin < input_begin + bytes) {
        return 0;
    }
    const int64_t rows = ggml_nrows(dst);
    const float scale = 1.0f / std::sqrt(float(n));
    auto stream = ctx.stream();
    const auto * input = static_cast<const float *>(src->data);
    const auto * sign_data = static_cast<const float *>(signs->data);
    auto * output = static_cast<float *>(dst->data);
    switch (n) {
        case 64: launch_fwht<64, true>(input, output, rows, scale, stream, sign_data, signs->ne[0]); break;
        case 128: launch_fwht<128, true>(input, output, rows, scale, stream, sign_data, signs->ne[0]); break;
        case 256: launch_fwht<256, true>(input, output, rows, scale, stream, sign_data, signs->ne[0]); break;
        case 512: launch_fwht<512, true>(input, output, rows, scale, stream, sign_data, signs->ne[0]); break;
        case 1024: launch_fwht<1024, true>(input, output, rows, scale, stream, sign_data, signs->ne[0]); break;
        default: return 0;
    }
    GGML_SYCL_DEBUG("Fused signed FWHT: n=%d rows=%lld\n", n, (long long) rows);
    return 2;
}
