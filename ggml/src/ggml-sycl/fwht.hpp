#ifndef GGML_SYCL_FWHT_HPP
#define GGML_SYCL_FWHT_HPP

#include "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// The sign flip, reshape, and Hadamard matmul of a folded rotation in one pass: x is the unsigned activation and
// signs its per-column sign vector. Returns false when the widths do not fit the wide kernel.
bool ggml_sycl_op_fwht_signed(ggml_backend_sycl_context & ctx, const ggml_tensor * x, const ggml_tensor * signs, ggml_tensor * dst);

#endif  // GGML_SYCL_FWHT_HPP
