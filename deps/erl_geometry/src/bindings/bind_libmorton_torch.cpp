#include "erl_common/pybind11.hpp"
#include "erl_geometry/libmorton_torch.hpp"

#ifdef ERL_USE_LIBTORCH
    #include <torch/extension.h>
#endif

void
BindLibmortonTorch(py::module &m) {
#ifdef ERL_USE_LIBTORCH
    m.def(
         "morton_encode",
         [](const torch::Tensor &coords) {
             torch::Tensor codes;
             erl::geometry::MortonEncodeTorch(coords, codes);
             return codes;
         },
         py::arg("coords").noconvert(),
         py::call_guard<py::gil_scoped_release>(),
         R"pbdoc(
Encode coordinates to morton codes.

Args:
    coords (torch.Tensor): Tensor of shape (D1, ..., D2, dims) with dtype torch.uint16 or torch.uint32.

Returns:
    torch.Tensor: Output tensor of shape (D1, ..., D2) with dtype torch.uint32 or torch.uint64.)pbdoc")
        .def(
            "morton_decode",
            [](const torch::Tensor &codes, const int dims) {
                torch::Tensor coords;
                erl::geometry::MortonDecodeTorch(codes, dims, coords);
                return coords;
            },
            py::arg("codes").noconvert(),
            py::arg("dims"),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
Decode morton codes to coordinates.

Args:
    codes (torch.Tensor): Tensor of morton code with dtype torch.uint32 or torch.uint64.
    dims (int): Space dimension, 2 or 3.

Returns:
    torch.Tensor: Output tensor of shape (..., dims) with dtype torch.uint16 or torch.uint32.)pbdoc");
#endif
}
