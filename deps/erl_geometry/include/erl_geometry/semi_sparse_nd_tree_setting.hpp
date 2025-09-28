#pragma once

#include "nd_tree_setting.hpp"

#include "erl_common/yaml.hpp"

namespace erl::geometry {

    struct SemiSparseNdTreeSetting : common::Yamlable<SemiSparseNdTreeSetting, NdTreeSetting> {

        // depth up to which all child nodes are always allocated when a child is created
        uint32_t semi_sparse_depth = 2;
        std::size_t init_voxel_num = 200000;  // initial number of voxels to allocate memory for
        bool cache_voxel_centers = false;     // whether to cache voxel centers

        bool
        operator==(const NdTreeSetting& other) const override;
    };
}  // namespace erl::geometry

template<>
struct YAML::convert<erl::geometry::SemiSparseNdTreeSetting> {
    static Node
    encode(const erl::geometry::SemiSparseNdTreeSetting& setting);

    static bool
    decode(const Node& node, erl::geometry::SemiSparseNdTreeSetting& setting);
};  // namespace YAML
