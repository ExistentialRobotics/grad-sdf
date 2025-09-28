#include "erl_geometry/init.hpp"

#include "erl_geometry/semi_sparse_octree.hpp"
#include "erl_geometry/semi_sparse_quadtree.hpp"

namespace erl::geometry {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = Init();

    bool
    Init() {
        static bool initialized_ = false;

        if (initialized_) { return true; }

        REGISTER(NdTreeSetting);
        REGISTER(SemiSparseNdTreeSetting);

        REGISTER(SemiSparseOctreeD);
        REGISTER(SemiSparseOctreeF);
        REGISTER(SemiSparseQuadtreeD);
        REGISTER(SemiSparseQuadtreeF);

        ERL_INFO("erl_geometry initialized");
        initialized_ = true;
        return true;
    }
}  // namespace erl::geometry
