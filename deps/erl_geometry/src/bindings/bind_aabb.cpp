#include "erl_common/pybind11.hpp"
#include "erl_geometry/aabb.hpp"

template<typename Scalar, int Dim>
static std::enable_if_t<Dim <= 3>
BindAabbTemplate(const py::module &m, const char *py_class_name) {
    using namespace erl::geometry;
    using Cls = Aabb<Scalar, Dim>;

    auto py_aabb = py::class_<Cls>(m, py_class_name)
                       .def(py::init<typename Cls::Point, typename Cls::Scalar>(), py::arg("center"), py::arg("half_size"))
                       .def(py::init<typename Cls::Point, typename Cls::Point>(), py::arg("min"), py::arg("max"));
    if (Dim == 1) {
        py::enum_<typename Cls::CornerType>(py_aabb, "CornerType", py::arithmetic(), "Type of corners.")
            .value("kMin", Cls::CornerType::Min)
            .value("kMax", Cls::CornerType::Max)
            .export_values();
    } else if (Dim == 2) {
        py::enum_<typename Cls::CornerType>(py_aabb, "CornerType", py::arithmetic(), "Type of corners.")
            .value("kBottomLeft", Cls::CornerType::BottomLeft)
            .value("kBottomRight", Cls::CornerType::BottomRight)
            .value("kTopLeft", Cls::CornerType::TopLeft)
            .value("kTopRight", Cls::CornerType::TopRight)
            .export_values();
    } else {
        py::enum_<typename Cls::CornerType>(py_aabb, "CornerType", py::arithmetic(), "Type of corners.")
            .value("kBottomLeftFloor", Cls::CornerType::BottomLeftFloor)
            .value("kBottomRightFloor", Cls::CornerType::BottomRightFloor)
            .value("kTopLeftFloor", Cls::CornerType::TopLeftFloor)
            .value("kTopRightFloor", Cls::CornerType::TopRightFloor)
            .value("kBottomLeftCeil", Cls::CornerType::BottomLeftCeil)
            .value("kBottomRightCeil", Cls::CornerType::BottomRightCeil)
            .value("kTopLeftCeil", Cls::CornerType::TopLeftCeil)
            .value("kTopRightCeil", Cls::CornerType::TopRightCeil)
            .export_values();
    }

    ERL_PYBIND_WRAP_PROPERTY_AS_READONLY(py_aabb, Cls, center);
    ERL_PYBIND_WRAP_PROPERTY_AS_READONLY(py_aabb, Cls, half_sizes);

    py_aabb.def_property_readonly("min", py::overload_cast<>(&Cls::min, py::const_))
        .def_property_readonly("max", py::overload_cast<>(&Cls::max, py::const_))
        .def(
            "__contains__",
            [](const Cls &aabb, const Eigen::Vector<Scalar, Dim> &point) { return aabb.contains(point); },
            py::arg("point"))
        .def(
            "__contains__",
            [](const Cls &aabb_1, const Cls &aabb_2) { return aabb_1.contains(aabb_2); },
            py::arg("another_aabb"))
        .def("padding", py::overload_cast<const typename Cls::Point &>(&Cls::Padding, py::const_), py::arg("padding"))
        .def("padding", py::overload_cast<typename Cls::Scalar>(&Cls::Padding, py::const_), py::arg("padding"))
        .def("corner", &Cls::corner, py::arg("corner_type"))
        .def("intersects", &Cls::intersects, py::arg("another_aabb"));
}

void
BindAabb(const py::module &m) {
    BindAabbTemplate<double, 2>(m, "Aabb2Dd");
    BindAabbTemplate<double, 3>(m, "Aabb3Dd");
    BindAabbTemplate<float, 2>(m, "Aabb2Df");
    BindAabbTemplate<float, 3>(m, "Aabb3Df");
}
