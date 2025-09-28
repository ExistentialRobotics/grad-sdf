#ifdef ERL_USE_LIBTORCH

    #include "erl_geometry/libmorton_torch.hpp"

    #include "erl_geometry/libmorton/morton.h"
    #include "erl_geometry/libmorton_torch.cuh"

    #define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace erl::geometry {

    void
    MortonEncodeTorch(const torch::Tensor &coords, torch::Tensor &codes) {

        if (coords.is_cuda()) {
            MortonEncodeTorchCUDA(coords, codes);
            return;
        }
        if (!coords.is_cpu()) { AT_ERROR("Unsupported device for MortonEncodeTorch."); }

        CHECK_CONTIGUOUS(coords);

        const int64_t n = coords.ndimension();
        const int64_t d = coords.size(n - 1);
        const int64_t b = coords.numel() / d;

        TORCH_CHECK(d == 2 || d == 3, "coords must be of shape (..., 2) or (..., 3).");

        if (d == 2) {
            switch (coords.scalar_type()) {
                case torch::kUInt16: {
                    codes = torch::empty(
                        coords.sizes().slice(0, n - 1),
                        torch::TensorOptions().dtype(torch::kUInt32).device(coords.device()));
                    const uint16_t *coords_ptr = coords.data_ptr<uint16_t>();
                    auto *codes_ptr = codes.data_ptr<uint32_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 2) {
                        codes_ptr[i] = libmorton::morton2D_32_encode(coords_ptr[0], coords_ptr[1]);
                    }
                    break;
                }
                case torch::kUInt32: {
                    codes = torch::empty(
                        coords.sizes().slice(0, n - 1),
                        torch::TensorOptions().dtype(torch::kUInt64).device(coords.device()));
                    const uint32_t *coords_ptr = coords.data_ptr<uint32_t>();
                    auto *codes_ptr = codes.data_ptr<uint64_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 2) {
                        codes_ptr[i] = libmorton::morton2D_64_encode(coords_ptr[0], coords_ptr[1]);
                    }
                    break;
                }
                default: {
                    AT_ERROR("Unsupported data type for MortonEncodeTorch.");
                }
            }
        } else {
            switch (coords.scalar_type()) {
                case torch::kUInt16: {
                    codes = torch::empty(
                        coords.sizes().slice(0, n - 1),
                        torch::TensorOptions().dtype(torch::kUInt32).device(coords.device()));
                    const uint16_t *coords_ptr = coords.data_ptr<uint16_t>();
                    auto *codes_ptr = codes.data_ptr<uint32_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 3) {
                        codes_ptr[i] = libmorton::morton3D_32_encode(
                            coords_ptr[0],
                            coords_ptr[1],
                            coords_ptr[2]);
                    }
                    break;
                }
                case torch::kUInt32: {
                    codes = torch::empty(
                        coords.sizes().slice(0, n - 1),
                        torch::TensorOptions().dtype(torch::kUInt64).device(coords.device()));
                    const uint32_t *coords_ptr = coords.data_ptr<uint32_t>();
                    auto *codes_ptr = codes.data_ptr<uint64_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 3) {
                        codes_ptr[i] = libmorton::morton3D_64_encode(
                            coords_ptr[0],
                            coords_ptr[1],
                            coords_ptr[2]);
                    }
                    break;
                }
                default: {
                    AT_ERROR("Unsupported data type for MortonEncodeTorch.");
                }
            }
        }
    }

    void
    MortonDecodeTorch(const torch::Tensor &codes, int dims, torch::Tensor &coords) {
        if (codes.is_cuda()) {
            MortonDecodeTorchCUDA(codes, dims, coords);
            return;
        }
        if (!codes.is_cpu()) { AT_ERROR("Unsupported device for MortonDecodeTorch."); }

        TORCH_CHECK(dims == 2 || dims == 3, "dims must be 2 or 3.");
        TORCH_CHECK(
            codes.scalar_type() == torch::kUInt32 || codes.scalar_type() == torch::kUInt64,
            "codes must be of dtype torch::kUInt32 or torch::kUInt64.");

        CHECK_CONTIGUOUS(codes);

        const int64_t b = codes.numel();
        auto coords_size = codes.sizes().vec();
        coords_size.push_back(dims);

        if (dims == 2) {
            switch (codes.scalar_type()) {
                case torch::kUInt32: {
                    coords = torch::empty(
                        coords_size,
                        torch::TensorOptions().dtype(torch::kUInt16).device(codes.device()));
                    const uint32_t *codes_ptr = codes.data_ptr<uint32_t>();
                    auto *coords_ptr = coords.data_ptr<uint16_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 2) {
                        uint_fast16_t x, y;
                        libmorton::morton2D_32_decode(codes_ptr[i], x, y);
                        coords_ptr[0] = x;
                        coords_ptr[1] = y;
                    }
                    break;
                }
                case torch::kUInt64: {
                    coords = torch::empty(
                        coords_size,
                        torch::TensorOptions().dtype(torch::kUInt32).device(codes.device()));
                    const uint64_t *codes_ptr = codes.data_ptr<uint64_t>();
                    auto *coords_ptr = coords.data_ptr<uint32_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 2) {
                        uint_fast32_t x, y;
                        libmorton::morton2D_64_decode(codes_ptr[i], x, y);
                        coords_ptr[0] = x;
                        coords_ptr[1] = y;
                    }
                    break;
                }
                default: {
                    AT_ERROR("Unsupported data type for MortonDecodeTorch.");
                }
            }
        } else {
            switch (codes.scalar_type()) {
                case torch::kUInt32: {
                    coords = torch::empty(
                        coords_size,
                        torch::TensorOptions().dtype(torch::kUInt16).device(codes.device()));
                    const uint32_t *codes_ptr = codes.data_ptr<uint32_t>();
                    auto *coords_ptr = coords.data_ptr<uint16_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 3) {
                        uint_fast16_t x, y, z;
                        libmorton::morton3D_32_decode(codes_ptr[i], x, y, z);
                        coords_ptr[0] = x;
                        coords_ptr[1] = y;
                        coords_ptr[2] = z;
                    }
                    break;
                }
                case torch::kUInt64: {
                    coords = torch::empty(
                        coords_size,
                        torch::TensorOptions().dtype(torch::kUInt32).device(codes.device()));
                    const uint64_t *codes_ptr = codes.data_ptr<uint64_t>();
                    auto *coords_ptr = coords.data_ptr<uint32_t>();
                    for (int64_t i = 0; i < b; ++i, coords_ptr += 3) {
                        uint_fast32_t x, y, z;
                        libmorton::morton3D_64_decode(codes_ptr[i], x, y, z);
                        coords_ptr[0] = x;
                        coords_ptr[1] = y;
                        coords_ptr[2] = z;
                    }
                    break;
                }
                default: {
                    AT_ERROR("Unsupported data type for MortonDecodeTorch.");
                }
            }
        }
    }

}  // namespace erl::geometry

#endif
