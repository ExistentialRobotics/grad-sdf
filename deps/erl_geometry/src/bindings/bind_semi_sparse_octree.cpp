#include "erl_geometry/pybind11_semi_sparse_octree.hpp"
#include "erl_geometry/semi_sparse_octree.hpp"

void
BindSemiSparseOctree(const py::module &m) {
    using namespace erl::geometry;
    BindSemiSparseOctree<SemiSparseOctreeD, SemiSparseOctreeNode>(m, "SemiSparseOctreeD");
    BindSemiSparseOctree<SemiSparseOctreeF, SemiSparseOctreeNode>(m, "SemiSparseOctreeF");
}
