#include "erl_geometry/libmorton_cuda/morton3d.cuh"

namespace libmorton_cuda {

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ morton_t
    Morton3dEncode(coord_t x, coord_t y, coord_t z) {
        morton_t code = 0;
        constexpr coord_t mask = 0xFF;
        for (auto i = sizeof(coord_t); i > 0; --i) {
            coord_t shift = (i - 1) << 3;  // * 8
            code = (code << 24) |          //
                   Morton3dEncodeX256[(x >> shift) & mask] |
                   Morton3dEncodeY256[(y >> shift) & mask] |
                   Morton3dEncodeZ256[(z >> shift) & mask];
        }
        return code;
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton3dEncode(
        const coord_t *__restrict__ x,
        const coord_t *__restrict__ y,
        const coord_t *__restrict__ z,
        morton_t *__restrict__ codes,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { codes[i] = Morton3dEncode<morton_t, coord_t>(x[i], y[i], z[i]); }
    }

    __global__ void
    Morton3dEncodeKernel32(
        const uint16_t *__restrict__ x,
        const uint16_t *__restrict__ y,
        const uint16_t *__restrict__ z,
        uint32_t *__restrict__ codes,
        const size_t n) {
        Morton3dEncode<uint32_t, uint16_t>(x, y, z, codes, n);
    }

    __global__ void
    Morton3dEncodeKernel64(
        const uint32_t *__restrict__ x,
        const uint32_t *__restrict__ y,
        const uint32_t *__restrict__ z,
        uint64_t *__restrict__ codes,
        const size_t n) {
        Morton3dEncode<uint64_t, uint32_t>(x, y, z, codes, n);
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton3dEncode(const coord_t *__restrict__ xyz, morton_t *__restrict__ codes, size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xyz += i * 3;
            codes[i] = Morton3dEncode<morton_t, coord_t>(xyz[0], xyz[1], xyz[2]);
        }
    }

    __global__ void
    Morton3dEncodeKernel32(
        const uint16_t *__restrict__ xyz,
        uint32_t *__restrict__ codes,
        const size_t n) {
        Morton3dEncode<uint32_t, uint16_t>(xyz, codes, n);
    }

    __global__ void
    Morton3dEncodeKernel64(
        const uint32_t *__restrict__ xyz,
        uint64_t *__restrict__ codes,
        const size_t n) {
        Morton3dEncode<uint64_t, uint32_t>(xyz, codes, n);
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ coord_t
    Morton3dDecodeX(morton_t code, const uint8_t *__restrict__ LUT) {
        coord_t x = 0;
        constexpr int loops = (sizeof(morton_t) <= 4) ? 4 : 7;
        constexpr morton_t mask = 0x1FF;
        for (int i = 0; i < loops; ++i) {
            morton_t j = i * 9;
            morton_t k = i * 3;
            x |= static_cast<coord_t>(LUT[(code >> j) & mask] << k);
        }
        return x;
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton3dDecode(morton_t code, coord_t &x, coord_t &y, coord_t &z) {
        x = Morton3dDecodeX<morton_t, coord_t>(code, Morton3dDecodeX512);
        y = Morton3dDecodeX<morton_t, coord_t>(code, Morton3dDecodeY512);
        z = Morton3dDecodeX<morton_t, coord_t>(code, Morton3dDecodeZ512);
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton3dDecode(
        const morton_t *__restrict__ codes,
        coord_t *__restrict__ x,
        coord_t *__restrict__ y,
        coord_t *__restrict__ z,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { Morton3dDecode(codes[i], x[i], y[i], z[i]); }
    }

    __global__ void
    Morton3dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ x,
        uint16_t *__restrict__ y,
        uint16_t *__restrict__ z,
        const size_t n) {
        Morton3dDecode<uint32_t, uint16_t>(codes, x, y, z, n);
    }

    __global__ void
    Morton3dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ x,
        uint32_t *__restrict__ y,
        uint32_t *__restrict__ z,
        const size_t n) {
        Morton3dDecode<uint64_t, uint32_t>(codes, x, y, z, n);
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton3dDecode(const morton_t *__restrict__ codes, coord_t *__restrict__ xyz, size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xyz += i * 3;
            Morton3dDecode(codes[i], xyz[0], xyz[1], xyz[2]);
        }
    }

    __global__ void
    Morton3dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ xyz,
        const size_t n) {
        Morton3dDecode<uint32_t, uint16_t>(codes, xyz, n);
    }

    __global__ void
    Morton3dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ xyz,
        const size_t n) {
        Morton3dDecode<uint64_t, uint32_t>(codes, xyz, n);
    }

}  // namespace libmorton_cuda
