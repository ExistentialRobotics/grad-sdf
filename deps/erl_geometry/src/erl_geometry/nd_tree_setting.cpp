#include "erl_geometry/nd_tree_setting.hpp"

bool
erl::geometry::NdTreeSetting::operator==(const NdTreeSetting &other) const {
    if (typeid(*this) != typeid(other)) { return false; }
    return resolution == other.resolution && tree_depth == other.tree_depth;
}

YAML::Node
YAML::convert<erl::geometry::NdTreeSetting>::encode(const erl::geometry::NdTreeSetting &setting) {
    Node node;
    ERL_YAML_SAVE_ATTR(node, setting, resolution);
    ERL_YAML_SAVE_ATTR(node, setting, tree_depth);
    return node;
}

bool
YAML::convert<erl::geometry::NdTreeSetting>::decode(
    const Node &node,
    erl::geometry::NdTreeSetting &setting) {
    if (!node.IsMap()) { return false; }
    ERL_YAML_LOAD_ATTR(node, setting, resolution);
    ERL_YAML_LOAD_ATTR(node, setting, tree_depth);
    return true;
}
