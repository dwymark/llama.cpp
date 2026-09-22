#ifndef GGML_SYCL_FWHT_HPP
#define GGML_SYCL_FWHT_HPP

#include "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// Returns the number of following nodes consumed by sign-multiply and Hadamard fusion.
int ggml_sycl_try_signed_fwht(ggml_backend_sycl_context & ctx, ggml_cgraph * graph, int index);

#endif  // GGML_SYCL_FWHT_HPP
