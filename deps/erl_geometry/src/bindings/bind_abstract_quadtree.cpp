#include "erl_common/pybind11.hpp"
#include "erl_common/serialization.hpp"
#include "erl_geometry/abstract_quadtree.hpp"

template<typename Dtype>
void
BindAbstractQuadtreeImpl(const py::module& m, const char* name) {
    using namespace erl::geometry;
    using T = AbstractQuadtree<Dtype>;

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
            py::overload_cast<Dtype, Dtype, uint32_t>(&T::SearchNode, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("max_depth"))
        .def(
            "search_node",
            py::overload_cast<const QuadtreeKey&, uint32_t>(&T::SearchNode, py::const_),
            py::arg("key"),
            py::arg("max_depth"));
    py::class_<typename T::QuadtreeNodeIterator>(tree, "QuadtreeNodeIterator")
        .def_property_readonly("x", &T::QuadtreeNodeIterator::GetX)
        .def_property_readonly("y", &T::QuadtreeNodeIterator::GetY)
        .def_property_readonly("node_size", &T::QuadtreeNodeIterator::GetNodeSize)
        .def_property_readonly("depth", &T::QuadtreeNodeIterator::GetDepth)
        .def("next", &T::QuadtreeNodeIterator::Next)
        .def_property_readonly("is_valid", &T::QuadtreeNodeIterator::IsValid)
        .def("get_node", &T::QuadtreeNodeIterator::GetNode)
        .def("get_key", &T::QuadtreeNodeIterator::GetKey)
        .def("get_index_key", &T::QuadtreeNodeIterator::GetIndexKey);
}

void
BindAbstractQuadtree(const py::module& m) {
    BindAbstractQuadtreeImpl<double>(m, "AbstractQuadtreeD");
    BindAbstractQuadtreeImpl<float>(m, "AbstractQuadtreeF");
}
