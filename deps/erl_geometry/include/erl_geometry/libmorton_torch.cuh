#pragma once

#include <torch/torch.h>

namespace erl::geometry {

    void
    MortonEncodeTorchCUDA(const torch::Tensor &coords, torch::Tensor &codes);

    void
    MortonDecodeTorchCUDA(const torch::Tensor &codes, int dims, torch::Tensor &coords);

}  // namespace erl::geometry
