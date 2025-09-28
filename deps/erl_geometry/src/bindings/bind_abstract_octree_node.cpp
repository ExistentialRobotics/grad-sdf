#include "erl_common/pybind11.hpp"
#include "erl_geometry/abstract_octree_node.hpp"

void
BindAbstractOctreeNode(const py::module& m) {
    using namespace erl::geometry;
    py::class_<AbstractOctreeNode, py::RawPtrWrapper<AbstractOctreeNode>>(m, "AbstractOctreeNode")
        .def_property_readonly("node_type", &AbstractOctreeNode::GetNodeType)
        .def_property_readonly("depth", &AbstractOctreeNode::GetDepth)
        .def_property_readonly("child_index", &AbstractOctreeNode::GetChildIndex)
        .def_property_readonly("num_children", &AbstractOctreeNode::GetNumChildren)
        .def_property_readonly("has_any_child", &AbstractOctreeNode::HasAnyChild)
        .def("has_child", &AbstractOctreeNode::HasChild, py::arg("child_idx"));
}
