#include "erl_common/pybind11.hpp"
#include "erl_geometry/octree_key.hpp"

void
BindOctreeKey(const py::module &m) {
    using namespace erl::geometry;

    py::class_<OctreeKey>(m, "OctreeKey")
        .def(py::init<>())
        .def(
            py::init<OctreeKey::KeyType, OctreeKey::KeyType, OctreeKey::KeyType>(),
            py::arg("a"),
            py::arg("b"),
            py::arg("c"))
        .def("__repr__", [](const OctreeKey &self) { return std::string(self); })
        .def("__eq__", [](const OctreeKey &self, const OctreeKey &other) { return self == other; })
        .def("__ne__", [](const OctreeKey &self, const OctreeKey &other) { return self != other; })
        .def("__getitem__", [](const OctreeKey &self, const int idx) { return self[idx]; })
        .def("__hash__", [](const OctreeKey &self) { return OctreeKey::KeyHash()(self); })
        .def("to_list", [](const OctreeKey &self) {
            py::list list;
            list.append(self[0]);
            list.append(self[1]);
            list.append(self[2]);
            return list;
        });
}
