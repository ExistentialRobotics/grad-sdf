#include "erl_geometry/semi_sparse_nd_tree_setting.hpp"

bool
erl::geometry::SemiSparseNdTreeSetting::operator==(const NdTreeSetting &other) const {
    if (NdTreeSetting::operator==(other)) {
        const auto that = reinterpret_cast<const SemiSparseNdTreeSetting &>(other);
        return semi_sparse_depth == that.semi_sparse_depth &&
               init_voxel_num == that.init_voxel_num &&
               cache_voxel_centers == that.cache_voxel_centers;
    }
    return false;
}
