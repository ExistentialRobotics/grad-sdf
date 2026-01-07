#pragma once

#include "nd_tree_setting.hpp"

#include "erl_common/yaml.hpp"

namespace erl::geometry {

    struct SemiSparseNdTreeSetting : common::Yamlable<SemiSparseNdTreeSetting, NdTreeSetting> {

        // depth up to which all child nodes are always allocated when a child is created
        uint32_t semi_sparse_depth = 2;
        std::size_t init_voxel_num = 200000;  // initial number of voxels to allocate memory for
        bool cache_voxel_centers = false;     // whether to cache voxel centers

        ERL_REFLECT_SCHEMA(
            SemiSparseNdTreeSetting,
            ERL_REFLECT_MEMBER(SemiSparseNdTreeSetting, semi_sparse_depth),
            ERL_REFLECT_MEMBER(SemiSparseNdTreeSetting, init_voxel_num),
            ERL_REFLECT_MEMBER(SemiSparseNdTreeSetting, cache_voxel_centers));

        bool
        operator==(const NdTreeSetting &other) const override;
    };
}  // namespace erl::geometry
