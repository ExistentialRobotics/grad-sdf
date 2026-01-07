#include "erl_geometry/nd_tree_setting.hpp"

bool
erl::geometry::NdTreeSetting::operator==(const NdTreeSetting &other) const {
    if (typeid(*this) != typeid(other)) { return false; }
    return resolution == other.resolution && tree_depth == other.tree_depth;
}
