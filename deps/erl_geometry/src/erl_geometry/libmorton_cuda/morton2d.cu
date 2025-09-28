#include "erl_geometry/libmorton_cuda/morton2d.cuh"

namespace libmorton_cuda {
    __device__ __forceinline__ uint32_t
    Morton2dEncode32(const uint16_t x, const uint16_t y) {
        // put Y in upper 32 bits, X in lower 32 bits
        uint64_t code = x | (static_cast<uint64_t>(y) << 32);
        code = (code | (code << 8)) & MagicBit2dMasks64[2];
        code = (code | (code << 4)) & MagicBit2dMasks64[3];
        code = (code | (code << 2)) & MagicBit2dMasks64[4];
        code = (code | (code << 1)) & MagicBit2dMasks64[5];
        code = code | (code >> 31);  // merge X and Y back together
        code = code & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(code);
    }

    __global__ void
    Morton2dEncodeKernel32(
        const uint16_t *__restrict__ x,
        const uint16_t *__restrict__ y,
        uint32_t *__restrict__ codes,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { codes[i] = Morton2dEncode32(x[i], y[i]); }
    }

    __global__ void
    Morton2dEncodeKernel32(
        const uint16_t *__restrict__ xy,
        uint32_t *__restrict__ codes,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xy += (i << 1);
            codes[i] = Morton2dEncode32(xy[0], xy[1]);
        }
    }

    __device__ uint64_t
    Morton2dEncode64(const uint32_t x, const uint32_t y) {
        uint64_t code = 0;
        for (uint32_t i = sizeof(uint32_t); i > 0; --i) {
            constexpr uint32_t mask = 0xFF;
            const uint32_t shift = (i - 1) << 3;
            code = (code << 16) |  //
                   Morton2dEncodeY256[(y >> shift) & mask] |
                   Morton2dEncodeX256[(x >> shift) & mask];
        }
        return code;
    }

    __global__ void
    Morton2dEncodeKernel64(
        const uint32_t *__restrict__ x,
        const uint32_t *__restrict__ y,
        uint64_t *__restrict__ codes,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { codes[i] = Morton2dEncode64(x[i], y[i]); }
    }

    __global__ void
    Morton2dEncodeKernel64(
        const uint32_t *__restrict__ xy,
        uint64_t *__restrict__ codes,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xy += (i << 1);
            codes[i] = Morton2dEncode64(xy[0], xy[1]);
        }
    }

    __device__ __forceinline__ void
    Morton2dDecode32(const uint32_t code, uint16_t &x, uint16_t &y) {
        uint64_t res = (code | (static_cast<uint64_t>(code) << 31)) & MagicBit2dMasks64[5];
        res = (res | (res >> 1)) & MagicBit2dMasks64[4];
        res = (res | (res >> 2)) & MagicBit2dMasks64[3];
        res = (res | (res >> 4)) & MagicBit2dMasks64[2];
        res = res | (res >> 8);
        x = static_cast<uint16_t>(res) & 0xFFFF;
        y = static_cast<uint16_t>(res >> 32) & 0xFFFF;
    }

    __global__ void
    Morton2dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ x,
        uint16_t *__restrict__ y,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { Morton2dDecode32(codes[i], x[i], y[i]); }
    }

    __global__ void
    Morton2dDecodeKernel32(
        const uint32_t *__restrict__ codes,
        uint16_t *__restrict__ xy,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xy += (i << 1);
            Morton2dDecode32(codes[i], xy[0], xy[1]);
        }
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ coord_t
    Morton2dDecodeX(const morton_t code, const uint8_t *__restrict__ LUT) {
        coord_t x = 0;
        constexpr int loops = sizeof(morton_t);
        for (int i = 0; i < loops; ++i) {
            x |= (static_cast<coord_t>(LUT[(code >> (i << 3)) & 0xFF]) << (i << 2));
        }
        return x;
    }

    template<typename morton_t, typename coord_t>
    __device__ __forceinline__ void
    Morton2dDecode(const morton_t code, coord_t &x, coord_t &y) {
        x = Morton2dDecodeX<morton_t, coord_t>(code, Morton2dDecodeX256);
        y = Morton2dDecodeX<morton_t, coord_t>(code, Morton2dDecodeY256);
    }

    __global__ void
    Morton2dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ x,
        uint32_t *__restrict__ y,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { Morton2dDecode<uint64_t, uint32_t>(codes[i], x[i], y[i]); }
    }

    __global__ void
    Morton2dDecodeKernel64(
        const uint64_t *__restrict__ codes,
        uint32_t *__restrict__ xy,
        const size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            xy += (i << 1);
            Morton2dDecode<uint64_t, uint32_t>(codes[i], xy[0], xy[1]);
        }
    }

}  // namespace libmorton_cuda
