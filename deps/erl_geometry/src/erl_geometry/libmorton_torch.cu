#ifdef ERL_USE_LIBTORCH

    #include "erl_geometry/libmorton_cuda/morton.cuh"
    #include "erl_geometry/libmorton_torch.cuh"

    #include <c10/cuda/CUDAException.h>

    #define CHECK_CUDA(x)       TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
    #define CHECK_LONG(x) \
        TORCH_CHECK((x).scalar_type() == torch::kInt64, #x " must be a Long tensor")
    #define WARP_SIZE      32l
    #define MAX_BLOCK_SIZE 256l

namespace erl::geometry {
    void
    MortonEncodeTorchCUDA(const torch::Tensor &coords, torch::Tensor &codes) {
        CHECK_CUDA(coords);
        CHECK_CONTIGUOUS(coords);
        TORCH_CHECK(coords.numel() > 0, "coords must not be empty.");

        const int64_t n = coords.ndimension();
        const int64_t d = coords.size(n - 1);
        TORCH_CHECK(d == 2 || d == 3, "coords must be of shape (..., 2) or (..., 3).");

        const int64_t b = coords.numel() / d;
        const int64_t warps = (b + WARP_SIZE - 1) / WARP_SIZE;
        const int64_t threads = std::min(MAX_BLOCK_SIZE, warps * WARP_SIZE);
        const int64_t blocks = (b + threads - 1) / threads;

        if (d == 2) {
            AT_DISPATCH_SWITCH(
                coords.scalar_type(),
                "morton2d_encode",
                AT_DISPATCH_CASE(
                    torch::kUInt16,
                    [&] {
                        codes = torch::empty(
                            coords.sizes().slice(0, n - 1),
                            torch::TensorOptions().dtype(torch::kUInt32).device(coords.device()));
                        libmorton_cuda::Morton2dEncodeKernel32<<<blocks, threads>>>(
                            coords.data_ptr<uint16_t>(),
                            codes.data_ptr<uint32_t>(),
                            b);
                    })  //
                AT_DISPATCH_CASE(
                    torch::kUInt32,
                    [&] {
                        codes = torch::empty(
                            coords.sizes().slice(0, n - 1),
                            torch::TensorOptions().dtype(torch::kUInt64).device(coords.device()));
                        libmorton_cuda::Morton2dEncodeKernel64<<<blocks, threads>>>(
                            coords.data_ptr<uint32_t>(),
                            codes.data_ptr<uint64_t>(),
                            b);
                    })  //
            );
        } else {
            AT_DISPATCH_SWITCH(
                coords.scalar_type(),
                "morton3d_encode",
                AT_DISPATCH_CASE(
                    torch::kUInt16,
                    [&] {
                        codes = torch::empty(
                            coords.sizes().slice(0, n - 1),
                            torch::TensorOptions().dtype(torch::kUInt32).device(coords.device()));
                        libmorton_cuda::Morton3dEncodeKernel32<<<blocks, threads>>>(
                            coords.data_ptr<uint16_t>(),
                            codes.data_ptr<uint32_t>(),
                            b);
                    })  //
                AT_DISPATCH_CASE(
                    torch::kUInt32,
                    [&] {
                        codes = torch::empty(
                            coords.sizes().slice(0, n - 1),
                            torch::TensorOptions().dtype(torch::kUInt64).device(coords.device()));
                        libmorton_cuda::Morton3dEncodeKernel64<<<blocks, threads>>>(
                            coords.data_ptr<uint32_t>(),
                            codes.data_ptr<uint64_t>(),
                            b);
                    })  //
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void
    MortonDecodeTorchCUDA(const torch::Tensor &codes, const int dims, torch::Tensor &coords) {
        TORCH_CHECK(dims == 2 || dims == 3, "dims must be 2 or 3.");
        TORCH_CHECK(
            codes.scalar_type() == torch::kUInt32 || codes.scalar_type() == torch::kUInt64,
            "codes must be of dtype torch::kUInt32 or torch::kUInt64.");

        CHECK_CUDA(codes);
        CHECK_CONTIGUOUS(codes);

        const int64_t b = codes.numel();
        const int64_t warps = (b + WARP_SIZE - 1) / WARP_SIZE;
        const int64_t threads = std::min(MAX_BLOCK_SIZE, warps * WARP_SIZE);
        const int64_t blocks = (b + threads - 1) / threads;

        auto coords_size = codes.sizes().vec();
        coords_size.push_back(dims);

        if (dims == 2) {
            AT_DISPATCH_SWITCH(
                codes.scalar_type(),
                "morton2d_decode",
                AT_DISPATCH_CASE(
                    torch::kUInt32,
                    [&] {
                        coords = torch::empty(
                            coords_size,
                            torch::TensorOptions().dtype(torch::kUInt16).device(codes.device()));
                        libmorton_cuda::Morton2dDecodeKernel32<<<blocks, threads>>>(
                            codes.data_ptr<uint32_t>(),
                            coords.data_ptr<uint16_t>(),
                            b);
                    })  //
                AT_DISPATCH_CASE(
                    torch::kUInt64,
                    [&] {
                        coords = torch::empty(
                            coords_size,
                            torch::TensorOptions().dtype(torch::kUInt32).device(codes.device()));
                        libmorton_cuda::Morton2dDecodeKernel64<<<blocks, threads>>>(
                            codes.data_ptr<uint64_t>(),
                            coords.data_ptr<uint32_t>(),
                            b);
                    })  //
            );
        } else {
            AT_DISPATCH_SWITCH(
                codes.scalar_type(),
                "morton3d_decode",
                AT_DISPATCH_CASE(
                    torch::kUInt32,
                    [&] {
                        coords = torch::empty(
                            coords_size,
                            torch::TensorOptions().dtype(torch::kUInt16).device(codes.device()));
                        libmorton_cuda::Morton3dDecodeKernel32<<<blocks, threads>>>(
                            codes.data_ptr<uint32_t>(),
                            coords.data_ptr<uint16_t>(),
                            b);
                    })  //
                AT_DISPATCH_CASE(
                    torch::kUInt64,
                    [&] {
                        coords = torch::empty(
                            coords_size,
                            torch::TensorOptions().dtype(torch::kUInt32).device(codes.device()));
                        libmorton_cuda::Morton3dDecodeKernel64<<<blocks, threads>>>(
                            codes.data_ptr<uint64_t>(),
                            coords.data_ptr<uint32_t>(),
                            b);
                    })  //
            );
        }
    }
}  // namespace erl::geometry
#endif
