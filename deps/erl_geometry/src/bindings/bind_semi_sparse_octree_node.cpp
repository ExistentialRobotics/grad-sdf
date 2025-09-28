#include "erl_common/pybind11.hpp"
#include "erl_geometry/semi_sparse_octree_node.hpp"

void
BindSemiSparseOctreeNode(const py::module &m) {
    using namespace erl::geometry;
    py::class_<SemiSparseOctreeNode, AbstractOctreeNode, py::RawPtrWrapper<SemiSparseOctreeNode>>(
        m,
        "SemiSparseOctreeNode")
        .def_property_readonly("node_index", &SemiSparseOctreeNode::GetNodeIndex);
}
