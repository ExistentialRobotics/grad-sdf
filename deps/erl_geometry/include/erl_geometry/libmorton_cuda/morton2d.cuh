#pragma once

#include "morton2d_lut.cuh"

#include <cuda_runtime.h>

namespace libmorton_cuda {

    // 32-bit encoding

    __global__ void
    Morton2dEncodeKernel32(
        const uint16_t *__restrict__ x,
        const uint16_t *__restrict__ y,
        uint32_t *__restrict__ codes,
        size_t n);

    __global__ void
    Morton2dEncodeKernel32(const uint16_t *__restrict__ xy, uint32_t *__restrict__ codes, size_t n);

    // 64-bit encoding

    __global__ void
    Morton2dEncodeKernel64(
        const uint32_t *__restrict__ x,
        const uint32_t *__restrict__ y,
        uint64_t *__restrict__ codes,
        size_t n);

    __global__ void
    Morton2dEncodeKernel64(const uint32_t *__restrict__ xy, uint64_t *__restrict__ codes, size_t n);

    // 32-bit decoding

    __global__ void
    Morton2dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ x,
        uint16_t *__restrict__ y,
        size_t n);

    __global__ void
    Morton2dDecodeKernel32(const uint32_t *__restrict__ codes, uint16_t *__restrict__ xy, size_t n);

    // 64-bit decoding

    __global__ void
    Morton2dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ x,
        uint32_t *__restrict__ y,
        size_t n);

    __global__ void
    Morton2dDecodeKernel64(const uint64_t *__restrict__ codes, uint32_t *__restrict__ xy, size_t n);
}  // namespace libmorton_cuda
