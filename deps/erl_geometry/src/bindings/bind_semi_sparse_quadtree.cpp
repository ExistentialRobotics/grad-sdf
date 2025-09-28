#include "erl_geometry/pybind11_semi_sparse_quadtree.hpp"
#include "erl_geometry/semi_sparse_quadtree.hpp"

void
BindSemiSparseQuadtree(const py::module &m) {
    using namespace erl::geometry;
    BindSemiSparseQuadtree<SemiSparseQuadtreeD, SemiSparseQuadtreeNode>(m, "SemiSparseQuadtreeD");
    BindSemiSparseQuadtree<SemiSparseQuadtreeF, SemiSparseQuadtreeNode>(m, "SemiSparseQuadtreeF");
}
