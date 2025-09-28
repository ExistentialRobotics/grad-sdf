#pragma once

#include "morton3d_lut.cuh"

#include <cuda_runtime.h>

namespace libmorton_cuda {

    // encoding

    __global__ void
    Morton3dEncodeKernel32(
        const uint16_t *__restrict__ x,
        const uint16_t *__restrict__ y,
        const uint16_t *__restrict__ z,
        uint32_t *__restrict__ codes,
        size_t n);

    __global__ void
    Morton3dEncodeKernel64(
        const uint32_t *__restrict__ x,
        const uint32_t *__restrict__ y,
        const uint32_t *__restrict__ z,
        uint64_t *__restrict__ codes,
        size_t n);

    __global__ void
    Morton3dEncodeKernel32(
        const uint16_t *__restrict__ xyz,
        uint32_t *__restrict__ codes,
        size_t n);

    __global__ void
    Morton3dEncodeKernel64(
        const uint32_t *__restrict__ xyz,
        uint64_t *__restrict__ codes,
        size_t n);

    // decoding

    __global__ void
    Morton3dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ x,
        uint16_t *__restrict__ y,
        uint16_t *__restrict__ z,
        size_t n);

    __global__ void
    Morton3dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ xyz,
        size_t n);

    __global__ void
    Morton3dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ x,
        uint32_t *__restrict__ y,
        uint32_t *__restrict__ z,
        size_t n);

    __global__ void
    Morton3dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ xyz,
        size_t n);

}  // namespace libmorton_cuda
