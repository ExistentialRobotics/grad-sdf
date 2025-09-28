#include "erl_common/pybind11.hpp"
#include "erl_geometry/quadtree_key.hpp"

void
BindQuadtreeKey(const py::module &m) {
    using namespace erl::geometry;

    py::class_<QuadtreeKey>(m, "QuadtreeKey")
        .def(py::init<>())
        .def(py::init<QuadtreeKey::KeyType, QuadtreeKey::KeyType>(), py::arg("a"), py::arg("b"))
        .def("__repr__", [](const QuadtreeKey &self) { return std::string(self); })
        .def(
            "__eq__",
            [](const QuadtreeKey &self, const QuadtreeKey &other) { return self == other; })
        .def(
            "__ne__",
            [](const QuadtreeKey &self, const QuadtreeKey &other) { return self != other; })
        .def("__getitem__", [](const QuadtreeKey &self, const int idx) { return self[idx]; })
        .def("__hash__", [](const QuadtreeKey &self) { return QuadtreeKey::KeyHash()(self); })
        .def("to_list", [](const QuadtreeKey &self) {
            py::list list;
            list.append(self[0]);
            list.append(self[1]);
            return list;
        });
}
