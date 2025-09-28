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

YAML::Node
YAML::convert<erl::geometry::SemiSparseNdTreeSetting>::encode(
    const erl::geometry::SemiSparseNdTreeSetting &setting) {
    Node node = convert<erl::geometry::NdTreeSetting>::encode(setting);
    ERL_YAML_SAVE_ATTR(node, setting, semi_sparse_depth);
    ERL_YAML_SAVE_ATTR(node, setting, init_voxel_num);
    ERL_YAML_SAVE_ATTR(node, setting, cache_voxel_centers);
    return node;
}

bool
YAML::convert<erl::geometry::SemiSparseNdTreeSetting>::decode(
    const Node &node,
    erl::geometry::SemiSparseNdTreeSetting &setting) {
    if (!node.IsMap()) { return false; }
    if (!convert<erl::geometry::NdTreeSetting>::decode(node, setting)) { return false; }
    ERL_YAML_LOAD_ATTR(node, setting, semi_sparse_depth);
    ERL_YAML_LOAD_ATTR(node, setting, init_voxel_num);
    ERL_YAML_LOAD_ATTR(node, setting, cache_voxel_centers);
    return true;
}
