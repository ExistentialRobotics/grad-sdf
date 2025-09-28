#include "erl_common/pybind11.hpp"
#include "erl_geometry/abstract_quadtree_node.hpp"

void
BindAbstractQuadtreeNode(const py::module& m) {
    using namespace erl::geometry;
    py::class_<AbstractQuadtreeNode, py::RawPtrWrapper<AbstractQuadtreeNode>>(m, "AbstractQuadtreeNode")
        .def_property_readonly("node_type", &AbstractQuadtreeNode::GetNodeType)
        .def_property_readonly("depth", &AbstractQuadtreeNode::GetDepth)
        .def_property_readonly("child_index", &AbstractQuadtreeNode::GetChildIndex)
        .def_property_readonly("num_children", &AbstractQuadtreeNode::GetNumChildren)
        .def_property_readonly("has_any_child", &AbstractQuadtreeNode::HasAnyChild)
        .def("has_child", &AbstractQuadtreeNode::HasChild, py::arg("child_idx"));
}
