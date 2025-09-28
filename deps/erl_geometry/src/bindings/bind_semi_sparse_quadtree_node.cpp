#include "erl_common/pybind11.hpp"
#include "erl_geometry/semi_sparse_quadtree_node.hpp"

void
BindSemiSparseQuadtreeNode(const py::module &m) {
    using namespace erl::geometry;
    py::class_<
        SemiSparseQuadtreeNode,
        AbstractQuadtreeNode,
        py::RawPtrWrapper<SemiSparseQuadtreeNode>>(m, "SemiSparseQuadtreeNode")
        .def_property_readonly("node_index", &SemiSparseQuadtreeNode::GetNodeIndex);
}
