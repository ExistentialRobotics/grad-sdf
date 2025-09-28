#include "erl_common/pybind11.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/nd_tree_setting.hpp"

void
BindNdTreeSetting(const py::module& m) {
    using namespace erl::common;
    using namespace erl::geometry;
    py::class_<NdTreeSetting, YamlableBase, std::shared_ptr<NdTreeSetting>>(m, "NdTreeSetting")
        .def(py::init<>())
        .def_readwrite("resolution", &NdTreeSetting::resolution)
        .def_readwrite("tree_depth", &NdTreeSetting::tree_depth)
        .def("__eq__", (&NdTreeSetting::operator==), py::arg("other"))
        .def("__ne__", (&NdTreeSetting::operator!=), py::arg("other"));
}
