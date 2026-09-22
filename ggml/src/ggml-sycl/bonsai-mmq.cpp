#include "bonsai-mmq.hpp"
#include "convert.hpp"

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

template <int tiles>
static void pq2_prefill(sycl::queue & queue, const block_pq2_0 * weights, const sycl::half * input,
                  float * output, int k, int m, int n, int ldc) {
    constexpr int token_rows = 8 * tiles;
    const int row_tiles = (m + 15) / 16;
    const int token_tiles = (n + token_rows - 1) / token_rows;
    const int work_items = row_tiles * token_tiles;
    const sycl::range<1> local(4), global(((work_items + 3) / 4) * 4);
    queue.parallel_for(sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
            using namespace sycl::ext::intel::esimd;
            const int id = item.get_global_id(0);
            if (id >= work_items) return;
            const int row_base = (id / token_tiles) * 16;
            const int token_base = (id % token_tiles) * token_rows;
            const simd<uint32_t, 16> rows = simd<uint32_t, 16>(0, 1) + row_base;
            const simd_mask<16> valid = rows < uint32_t(m);
            const simd<uint32_t, 16> row_offsets = rows * (k / 128) * sizeof(block_pq2_0);
            simd<float, 128> accumulators[tiles];
#pragma unroll
            for (int tile = 0; tile < tiles; ++tile) accumulators[tile] = 0.0f;
            for (int block = 0; block < k / 128; ++block) {
                const simd<uint32_t, 16> wb = row_offsets + block * sizeof(block_pq2_0);
                const simd<uint32_t, 16> word_offsets = wb + ((block & 1) ? 2 : 0);
                simd<uint32_t, 128> packed = gather<uint32_t, 128, 8>(
                    reinterpret_cast<const uint32_t *>(weights), word_offsets, valid, simd<uint32_t, 128>(0));
                const simd<sycl::half, 16> scale_half = gather<sycl::half, 16, 1>(
                    reinterpret_cast<const sycl::half *>(weights), wb, valid, simd<sycl::half, 16>(sycl::half(0)));
                const simd<float, 16> scales = scale_half;
                if ((block & 1) == 0) {
                    const simd<uint32_t, 16> tail_offsets = wb + 32;
                    const simd<uint32_t, 16> tail = gather<uint16_t, 16, 1>(
                        reinterpret_cast<const uint16_t *>(weights), tail_offsets, valid, simd<uint16_t, 16>(0));
#pragma unroll
                    for (int word = 0; word < 7; ++word) {
                        const simd<uint32_t, 16> low = packed.select<16, 1>(word * 16);
                        const simd<uint32_t, 16> high = packed.select<16, 1>((word + 1) * 16);
                        packed.select<16, 1>(word * 16) = (low >> 16) | (high << 16);
                    }
                    const simd<uint32_t, 16> low = packed.select<16, 1>(7 * 16);
                    packed.select<16, 1>(7 * 16) = (low >> 16) | (tail << 16);
                }
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    const simd<uint32_t, 16> bits = packed.select<16, 1>(step * 16);
                    simd<sycl::half, 256> b;
#pragma unroll
                    for (int pair = 0; pair < 8; ++pair) {
                        const simd<int, 16> first = simd<int, 16>((bits >> (4 * pair)) & 3) - 1;
                        const simd<int, 16> second = simd<int, 16>((bits >> (4 * pair + 2)) & 3) - 1;
                        b.select<16, 2>(pair * 32) = simd<sycl::half, 16>(scales * simd<float, 16>(first));
                        b.select<16, 2>(pair * 32 + 1) = simd<sycl::half, 16>(scales * simd<float, 16>(second));
                    }
#pragma unroll
                    for (int tile = 0; tile < tiles; ++tile) {
                        const simd<sycl::half, 128> a = load_2d<sycl::half, 16, 8>(
                            input, k * sizeof(sycl::half) - 1, n - 1, k * sizeof(sycl::half) - 1,
                            block * 128 + step * 16, token_base + tile * 8);
                        accumulators[tile] = xmx::dpas<8, 8, float>(accumulators[tile], b, a);
                    }
                }
            }
#pragma unroll
            for (int tile = 0; tile < tiles; ++tile) {
#pragma unroll
                for (int token = 0; token < 8; ++token) {
                    const int t = token_base + tile * 8 + token;
                    if (t < n) {
                        const simd<uint32_t, 16> offsets = (rows + t * ldc) * sizeof(float);
                        const simd<float, 16> values = accumulators[tile].template select<16, 1>(token * 16);
                        scatter<float, 16>(output, offsets, values, valid);
                    }
                }
            }
        });
}

#endif

bool ggml_sycl_try_pq2_prefill(ggml_backend_sycl_context & ctx, const void * weights,
                              const float * input, float * output, int64_t k, int64_t m,
                              int64_t n, int ldc, const dpct::queue_ptr & stream) {
#if defined(__INTEL_LLVM_COMPILER)
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_PQ2_PREFILL", 0);
    if (!enabled || !g_ggml_sycl_enable_esimd || k < 256 || k % 256 || n < 9 || n > 256 ||
        m <= 0 || k > INT32_MAX || m > INT32_MAX ||
        uint64_t(m) * (k / QK_PQ2_0) * sizeof(block_pq2_0) > UINT32_MAX ||
        uint64_t(n) * ldc * sizeof(float) > UINT32_MAX) {
        return false;
    }
    namespace exp = sycl::ext::oneapi::experimental;
    if (stream->get_device().get_info<exp::info::device::architecture>() != exp::architecture::intel_gpu_lnl_m) {
        return false;
    }
    ggml_sycl_pool_alloc<sycl::half> converted(ctx.pool(), k * n + 31);
    auto * half_input = reinterpret_cast<sycl::half *>((reinterpret_cast<uintptr_t>(converted.get()) + 63) & ~uintptr_t(63));
    ggml_get_to_fp16_sycl(GGML_TYPE_F32, nullptr)(input, half_input, k * n, stream);
    GGML_SYCL_DEBUG("PQ2 fused prefill: k=%lld m=%lld n=%lld\n", (long long) k, (long long) m, (long long) n);
    const auto * packed = static_cast<const block_pq2_0 *>(weights);
    if (n <= 16) {
        pq2_prefill<2>(*stream, packed, half_input, output, k, m, n, ldc);
    } else if (n <= 32) {
        pq2_prefill<4>(*stream, packed, half_input, output, k, m, n, ldc);
    } else {
        pq2_prefill<8>(*stream, packed, half_input, output, k, m, n, ldc);
    }
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_UNUSED(k); GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(ldc); GGML_UNUSED(stream);
    return false;
#endif
}
