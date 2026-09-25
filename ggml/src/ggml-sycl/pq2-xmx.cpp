#include "pq2-xmx.hpp"

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#define GGML_SYCL_PQ2_XMX_AVAILABLE
#endif

#ifdef GGML_SYCL_PQ2_XMX_AVAILABLE

static constexpr int PQ2_XMX_QK = 128;  // values per PQ2_0 block and per activation scale

// One sub-group of 16 lanes quantizes one 128-value slice of one activation row, 8 values per lane.
static void pq2_xmx_quantize_act(const float * x, int8_t * q, float * s, int64_t K, int64_t ntokens,
                                 dpct::queue_ptr stream) {
    const int64_t nblk = ntokens * (K / PQ2_XMX_QK);
    stream->parallel_for(sycl::nd_range<1>(nblk * 16, 16), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t blk  = item.get_group(0);
        const int     lane = item.get_local_id(0);
        const float * xb   = x + blk * PQ2_XMX_QK + lane * 8;
        float v[8];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = xb[j];
            amax = sycl::fmax(amax, sycl::fabs(v[j]));
        }
        amax = sycl::reduce_over_group(item.get_sub_group(), amax, sycl::maximum<float>());
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int8_t * qb = q + blk * PQ2_XMX_QK + lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            qb[j] = (int8_t) sycl::rint(v[j] * id);
        }
        if (lane == 0) {
            s[blk] = d;
        }
    });
}

// Each thread computes 8*RM weight rows for 16 tokens (the dpas execution width). Weights are the dpas A
// operand: per 128-value block, each group of 8 rows is decoded from its 32 packed bytes into four int8 tiles.
// Activations are the VNNI B operand: one gather fetches 32 int8 values for each of the 16 tokens per step.
// Four dpas steps accumulate a block into int32, which is folded into FP32 with both scales.
template <int RM, int ABL = 0>
static void pq2_xmx_gemm(const uint8_t * vx, const int8_t * qa, const float * sa, float * dst, int64_t nrows,
                         int64_t K, int64_t ntokens, int64_t ldc, dpct::queue_ptr stream) {
    constexpr int ntok = 16;
    constexpr int rows = 8 * RM;
    const int64_t nrb  = nrows / rows;
    const int64_t ntb  = (ntokens + ntok - 1) / ntok;
    const int64_t nkb  = K / PQ2_XMX_QK;
    const uint32_t row_bytes = static_cast<uint32_t>(nkb * sizeof(block_pq2_0));
    constexpr int wg = 8;
    const int64_t nthreads = nrb * ntb;
    stream->parallel_for(sycl::nd_range<1>(((nthreads + wg - 1) / wg) * wg, wg),
        [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
            using namespace sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            const int64_t g = item.get_global_id(0);
            if (g >= nthreads) {
                return;
            }
            const int64_t rb   = g % nrb;
            const int64_t tb   = g / nrb;
            const int64_t row0 = rb * rows;
            const int64_t tok0 = tb * ntok;

            const uint8_t * wbase = vx + row0 * row_bytes;
            const simd<uint32_t, 8> r8(0, 1);
            const simd<uint32_t, ntok> n16(0, 1);
            const simd<uint32_t, ntok> tok = n16 + static_cast<uint32_t>(tok0);
            const simd_mask<ntok> tok_ok = tok < static_cast<uint32_t>(ntokens);
            simd<uint32_t, ntok> tokc(static_cast<uint32_t>(ntokens - 1));
            tokc.merge(tok, tok_ok);
            const simd<uint32_t, ntok> a_row = tokc * static_cast<uint32_t>(K);
            const simd<uint32_t, ntok> s_row = tokc * static_cast<uint32_t>(nkb);

            simd<float, 8 * ntok> acc[RM];
#pragma unroll
            for (int m = 0; m < RM; ++m) {
                acc[m] = 0.0f;
            }

            for (int64_t kb = 0; kb < nkb; ++kb) {
                const simd<float, ntok> as = gather<float, ntok>(sa, (s_row + static_cast<uint32_t>(kb)) * sizeof(float));
                simd<int, 8 * ntok> bt[4];
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    if constexpr (ABL & 2) {
                        bt[s] = simd<int, 8 * ntok>(s + (int) kb, 1);
                    } else {
                        bt[s] = gather<int, ntok * 8, 8>(reinterpret_cast<const int *>(qa),
                                                         a_row + static_cast<uint32_t>(kb * PQ2_XMX_QK + s * 32));
                    }
                }
#pragma unroll
                for (int m = 0; m < RM; ++m) {
                    const simd<uint32_t, 8> off = (r8 + 8 * m) * row_bytes + static_cast<uint32_t>(kb * sizeof(block_pq2_0));
                    const simd<float, 8> wd = simd<float, 8>(gather<sycl::half, 8, 1>(
                        reinterpret_cast<const sycl::half *>(wbase), off));
                    const simd<uint32_t, 8> qs_off = off + 2;
                    const simd<uint32_t, 8> shift = qs_off & 3;
                    const simd<uint32_t, 8> aligned = qs_off - shift;
                    simd<uint32_t, 8 * 8> raw = gather<uint32_t, 8 * 8, 8>(
                        reinterpret_cast<const uint32_t *>(wbase), aligned);
                    const simd_mask<8> sh = shift != 0;
                    const simd<uint32_t, 8> tail = gather<uint32_t, 8, 1>(
                        reinterpret_cast<const uint32_t *>(wbase), aligned + 32, sh, simd<uint32_t, 8>(0));
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        simd<uint32_t, 8> cur = raw.template select<8, 1>(i * 8);
                        const simd<uint32_t, 8> nxt = i < 7 ? simd<uint32_t, 8>(raw.template select<8, 1>((i + 1) * 8)) : tail;
                        cur.merge((cur >> 16) | (nxt << 16), sh);
                        raw.template select<8, 1>(i * 8) = cur;
                    }
                    simd<int, 8 * ntok> c = 0;
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        // A tile [row t][dword kk]: K positions 32s + 4kk .. +3, from packed byte 8s + kk of row t
                        simd<int, 64> at;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const simd<uint32_t, 8> dw = raw.template select<8, 1>((2 * s + h) * 8);
#pragma unroll
                            for (int bi = 0; bi < 4; ++bi) {
                                simd<uint32_t, 8> v;
                                if constexpr (ABL & 4) {
                                    v = dw + bi;
                                } else {
                                    v = (dw >> (8 * bi)) & 0xff;
                                    v = (v | (v << 12)) & 0x000f000f;
                                    v = (v | (v << 6)) & 0x03030303;
                                    v = ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
                                }
                                at.template select<8, 8>(4 * h + bi) = v.template bit_cast_view<int>();
                            }
                        }
                        if constexpr (ABL & 1) {
                            const int a0 = at[s];
                            c += bt[s] + a0;
                        } else {
                            c = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8,
                                          xmx::dpas_argument_type::s8>(c, bt[s], at);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const float wt = wd[t];
                        const simd<float, ntok> ci = c.template select<ntok, 1>(t * ntok);
                        acc[m].template select<ntok, 1>(t * ntok) += ci * (as * wt);
                    }
                }
            }

            const simd<uint32_t, ntok> out_off = n16 * static_cast<uint32_t>(ldc * sizeof(float));
            float * dbase = dst + tok0 * ldc + row0;
#pragma unroll
            for (int m = 0; m < RM; ++m) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    simd<float, ntok> out = acc[m].template select<ntok, 1>(t * ntok);
                    scatter<float, ntok>(dbase + 8 * m + t, out_off, out, tok_ok);
                }
            }
        });
}

bool ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const void * src0_pq2, const float * src1,
                               float * dst, int64_t nrows, int64_t K, int64_t ntokens, int64_t ldc,
                               dpct::queue_ptr stream) {
    if (nrows % 16 != 0 || K % PQ2_XMX_QK != 0 || ntokens < 1 || ntokens * K > INT32_MAX) {
        return false;
    }
    ggml_sycl_pool_alloc<int8_t> qa(ctx.pool(), ntokens * K);
    ggml_sycl_pool_alloc<float>  sa(ctx.pool(), ntokens * (K / PQ2_XMX_QK));
    pq2_xmx_quantize_act(src1, qa.get(), sa.get(), K, ntokens, stream);
    static const int rm = ggml_sycl_get_env("GGML_SYCL_PQ2_XMX_RM", 4);
    const uint8_t * w = static_cast<const uint8_t *>(src0_pq2);
    static const int abl = ggml_sycl_get_env("GGML_SYCL_PQ2_XMX_ABL", 0);
    if (rm == 2 || nrows % 32 != 0) {
        pq2_xmx_gemm<2>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream);
    } else {
        switch (abl) {
            case 1: pq2_xmx_gemm<4, 1>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream); break;
            case 2: pq2_xmx_gemm<4, 2>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream); break;
            case 4: pq2_xmx_gemm<4, 4>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream); break;
            case 6: pq2_xmx_gemm<4, 6>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream); break;
            default: pq2_xmx_gemm<4>(w, qa.get(), sa.get(), dst, nrows, K, ntokens, ldc, stream); break;
        }
    }
    return true;
}

#else

bool ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context &, const void *, const float *, float *, int64_t, int64_t,
                               int64_t, int64_t, dpct::queue_ptr) {
    return false;
}

#endif
