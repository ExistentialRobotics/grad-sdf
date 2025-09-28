#pragma once

#ifdef ERL_USE_LIBTORCH

    #include <torch/torch.h>

namespace erl::geometry {

    /**
     * Encode coordinates to morton codes.
     * @param coords tensor of shape (D1, ... D2, dims) with dtype torch::kUInt16 or torch::kUInt32.
     * @param codes output tensor of shape (D1, ... D2) with dtype torch::kUInt32 or torch::kUInt64.
     */
    void
    MortonEncodeTorch(const torch::Tensor &coords, torch::Tensor &codes);

    /**
     * Decode morton codes to coordinates.
     * @param codes tensor of morton code with dtype torch::kUInt32 or torch::kUInt64.
     * @param dims space dimension, 2 or 3.
     * @param coords output tensor of shape (..., dims) with dtype torch::kUInt16 or torch::kUInt32.
     */
    void
    MortonDecodeTorch(const torch::Tensor &codes, int dims, torch::Tensor &coords);

}  // namespace erl::geometry

#endif
