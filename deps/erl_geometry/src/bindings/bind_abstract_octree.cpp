#include "erl_common/pybind11.hpp"
#include "erl_common/serialization.hpp"
#include "erl_geometry/abstract_octree.hpp"

template<typename Dtype>
void
BindAbstractOctreeImpl(const py::module& m, const char* name) {
    using namespace erl::geometry;
    using T = AbstractOctree<Dtype>;

    py::class_<T, std::shared_ptr<T>> tree(m, name);
    tree.def_property_readonly("tree_type", &T::GetTreeType)
        .def("apply_setting", &T::ApplySetting)
        .def("read_setting", &T::ReadSetting)
        .def("write_setting", &T::WriteSetting)
        .def(
            "write",
            [](const T* self, const std::string& filename) -> bool {
                return erl::common::Serialization<T>::Write(filename, self);
            },
            py::arg("filename"))
        .def(
            "read",
            [](T* self, const std::string& filename) -> bool {
                return erl::common::Serialization<T>::Read(filename, self);
            },
            py::arg("filename"))
        .def(
            "search_node",
            py::overload_cast<Dtype, Dtype, Dtype, uint32_t>(&T::SearchNode, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_depth"))
        .def(
            "search_node",
            py::overload_cast<const OctreeKey&, uint32_t>(&T::SearchNode, py::const_),
            py::arg("key"),
            py::arg("max_depth"));
    py::class_<typename T::OctreeNodeIterator>(tree, "OctreeNodeIterator")
        .def_property_readonly("x", &T::OctreeNodeIterator::GetX)
        .def_property_readonly("y", &T::OctreeNodeIterator::GetY)
        .def_property_readonly("z", &T::OctreeNodeIterator::GetZ)
        .def_property_readonly("node_size", &T::OctreeNodeIterator::GetNodeSize)
        .def_property_readonly("depth", &T::OctreeNodeIterator::GetDepth)
        .def("next", &T::OctreeNodeIterator::Next)
        .def_property_readonly("is_valid", &T::OctreeNodeIterator::IsValid)
        .def("get_node", &T::OctreeNodeIterator::GetNode)
        .def("get_key", &T::OctreeNodeIterator::GetKey)
        .def("get_index_key", &T::OctreeNodeIterator::GetIndexKey);
}

void
BindAbstractOctree(const py::module& m) {
    BindAbstractOctreeImpl<double>(m, "AbstractOctreeD");
    BindAbstractOctreeImpl<float>(m, "AbstractOctreeF");
}
