#include "erl_common/pybind11.hpp"
#include "erl_geometry/marching_squares.hpp"

void
BindMarchingSquares(py::module &m) {
    using namespace erl::geometry;
    m.def(
        "marching_square",
        [](const Eigen::Ref<const Eigen::MatrixXd> &img, const double iso_value) {
            Eigen::Matrix2Xd vertices;
            Eigen::Matrix2Xi lines_to_vertices;
            Eigen::Matrix2Xi objects_to_lines;
            MarchingSquares::Run(img, iso_value, vertices, lines_to_vertices, objects_to_lines);

            return py::make_tuple(vertices, lines_to_vertices, objects_to_lines);
        },
        py::arg("img"),
        py::arg("iso_value"));
}
