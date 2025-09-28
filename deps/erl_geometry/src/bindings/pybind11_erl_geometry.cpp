#include "erl_common/pybind11.hpp"
#include "erl_common/pybind11_yaml.hpp"
#include "erl_geometry/init.hpp"

void
BindAabb(const py::module &m);

void
BindNdTreeSetting(const py::module &m);

void
BindSemiSparseNdTreeSetting(const py::module &m);

void
BindQuadtreeKey(const py::module &m);

void
BindAbstractQuadtreeNode(const py::module &m);

void
BindAbstractQuadtree(const py::module &m);

void
BindSemiSparseQuadtreeNode(const py::module &m);

void
BindSemiSparseQuadtree(const py::module &m);

void
BindOctreeKey(const py::module &m);

void
BindAbstractOctreeNode(const py::module &m);

void
BindAbstractOctree(const py::module &m);

void
BindSemiSparseOctreeNode(const py::module &m);

void
BindSemiSparseOctree(const py::module &m);

void
BindFindVoxelIndicesTorch(py::module &m);

void
BindLibmortonTorch(py::module &m);

void
BindMarchingSquares(py::module &m);

void
BindMarchingCubes(const py::module &m);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    erl::geometry::Init();

    m.doc() = "Python 3 Interface of erl_geometry";

    BindYamlableBase(m);

    BindAabb(m);

    BindNdTreeSetting(m);
    BindSemiSparseNdTreeSetting(m);

    BindQuadtreeKey(m);

    BindAbstractQuadtreeNode(m);
    BindSemiSparseQuadtreeNode(m);

    BindAbstractQuadtree(m);
    BindSemiSparseQuadtree(m);

    BindOctreeKey(m);

    BindAbstractOctreeNode(m);
    BindSemiSparseOctreeNode(m);

    BindAbstractOctree(m);
    BindSemiSparseOctree(m);

    BindFindVoxelIndicesTorch(m);
    BindLibmortonTorch(m);
    BindMarchingSquares(m);
    BindMarchingCubes(m);
}
