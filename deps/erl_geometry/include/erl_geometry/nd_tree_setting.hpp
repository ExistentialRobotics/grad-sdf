#pragma once

#include "erl_common/yaml.hpp"

namespace erl::geometry {

    /**
     * NDTreeSetting is a base class for all n-d tree settings.
     */
    class NdTreeSetting : public common::Yamlable<NdTreeSetting> {
    public:
        float resolution = 0.1;
        uint32_t tree_depth = 16;

        ERL_REFLECT_SCHEMA(
            NdTreeSetting,
            ERL_REFLECT_MEMBER(NdTreeSetting, resolution),
            ERL_REFLECT_MEMBER(NdTreeSetting, tree_depth));

        virtual bool
        operator==(const NdTreeSetting &other) const;

        bool
        operator!=(const NdTreeSetting &rhs) const {
            return !(*this == rhs);
        }
    };
}  // namespace erl::geometry
