#pragma once

#include "abstract_quadtree.hpp"
#include "quadtree_key.hpp"

#include "erl_common/pybind11.hpp"

template<class PyClass, class Dtype, class Quadtree, class Node>
void
BindQuadtreeImpl(PyClass& tree) {
    using namespace erl::geometry;
    using Vector2 = Eigen::Vector2<Dtype>;

    tree.def_property_readonly("number_of_nodes", &Quadtree::GetSize)
        .def_property_readonly("resolution", &Quadtree::GetResolution)
        .def_property_readonly("tree_depth", &Quadtree::GetTreeDepth)
        .def_property_readonly("tree_center", &Quadtree::GetTreeCenter)
        .def_property_readonly("tree_center_key", &Quadtree::GetTreeCenterKey)
        .def_property_readonly("tree_max_half_size", &Quadtree::GetTreeMaxHalfSize)
        .def_property_readonly("metric_min", [](Quadtree& self) { return self.GetMetricMin(); })
        .def_property_readonly("metric_max", [](Quadtree& self) { return self.GetMetricMax(); })
        .def_property_readonly(
            "metric_min_max",
            [](Quadtree& self) { return self.GetMetricMinMax(); })
        .def_property_readonly("metric_aabb", [](Quadtree& self) { return self.GetMetricAabb(); })
        .def_property_readonly("metric_size", [](Quadtree& self) { return self.GetMetricSize(); })
        .def("get_node_size", &Quadtree::GetNodeSize, py::arg("depth"))
        .def_property_readonly("number_of_leaf_nodes", &Quadtree::ComputeNumberOfLeafNodes)
        .def_property_readonly("memory_usage", &Quadtree::GetMemoryUsage)
        .def_property_readonly("memory_usage_per_node", &Quadtree::GetMemoryUsagePerNode)
        .def(
            "coord_to_key",
            py::overload_cast<Dtype>(&Quadtree::CoordToKey, py::const_),
            py::arg("coordinate"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, uint32_t>(&Quadtree::CoordToKey, py::const_),
            py::arg("coordinate"),
            py::arg("depth"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, Dtype>(&Quadtree::CoordToKey, py::const_),
            py::arg("x"),
            py::arg("y"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, Dtype, uint32_t>(&Quadtree::CoordToKey, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("depth"))
        .def(
            "coord_to_key_checked",
            [](const Quadtree& self, Dtype coordinate) {
                if (QuadtreeKey::KeyType key; self.CoordToKeyChecked(coordinate, key)) {
                    return std::optional<QuadtreeKey::KeyType>(key);
                }
                return std::optional<QuadtreeKey::KeyType>();
            },
            py::arg("coordinate"))
        .def(
            "coord_to_key_checked",
            [](const Quadtree& self, Dtype coordinate, uint32_t depth) {
                if (QuadtreeKey::KeyType key; self.CoordToKeyChecked(coordinate, depth, key)) {
                    return std::optional<QuadtreeKey::KeyType>(key);
                }
                return std::optional<QuadtreeKey::KeyType>();
            },
            py::arg("coordinate"),
            py::arg("depth"))
        .def(
            "coord_to_key_checked",
            [](const Quadtree& self, Dtype x, Dtype y) {
                if (QuadtreeKey key; self.CoordToKeyChecked(x, y, key)) {
                    return std::optional<QuadtreeKey>(key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("x"),
            py::arg("y"))
        .def(
            "coord_to_key_checked",
            [](const Quadtree& self, Dtype x, Dtype y, uint32_t depth) {
                if (QuadtreeKey key; self.CoordToKeyChecked(x, y, depth, key)) {
                    return std::optional<QuadtreeKey>(key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("depth"))
        .def(
            "adjust_key_to_depth",
            py::overload_cast<QuadtreeKey::KeyType, uint32_t>(
                &Quadtree::AdjustKeyToDepth,
                py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "adjust_key_to_depth",
            py::overload_cast<const QuadtreeKey&, uint32_t>(
                &Quadtree::AdjustKeyToDepth,
                py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_common_ancestor_key",
            [](const Quadtree& self, const QuadtreeKey& key1, const QuadtreeKey& key2) {
                QuadtreeKey key;
                uint32_t ancestor_depth;
                self.ComputeCommonAncestorKey(key1, key2, key, ancestor_depth);
                return std::make_tuple(key, ancestor_depth);
            })
        .def(
            "compute_west_neighbor_key",
            [](const Quadtree& self, const QuadtreeKey& key, uint32_t depth) {
                if (QuadtreeKey neighbor_key;
                    self.ComputeWestNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<QuadtreeKey>(neighbor_key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_east_neighbor_key",
            [](const Quadtree& self, const QuadtreeKey& key, uint32_t depth) {
                if (QuadtreeKey neighbor_key;
                    self.ComputeEastNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<QuadtreeKey>(neighbor_key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_north_neighbor_key",
            [](const Quadtree& self, const QuadtreeKey& key, uint32_t depth) {
                if (QuadtreeKey neighbor_key;
                    self.ComputeNorthNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<QuadtreeKey>(neighbor_key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_south_neighbor_key",
            [](const Quadtree& self, const QuadtreeKey& key, uint32_t depth) {
                if (QuadtreeKey neighbor_key;
                    self.ComputeSouthNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<QuadtreeKey>(neighbor_key);
                }
                return std::optional<QuadtreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "key_to_coord",
            py::overload_cast<QuadtreeKey::KeyType>(&Quadtree::KeyToCoord, py::const_),
            py::arg("key"))
        .def(
            "key_to_coord",
            py::overload_cast<QuadtreeKey::KeyType, uint32_t>(&Quadtree::KeyToCoord, py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "key_to_coord",
            [](const Quadtree& self, const QuadtreeKey& key) {
                Dtype x, y;
                self.KeyToCoord(key, x, y);
                return std::make_tuple(x, y);
            },
            py::arg("key"))
        .def(
            "key_to_coord",
            [](const Quadtree& self, const QuadtreeKey& key, uint32_t depth) {
                Dtype x, y;
                self.KeyToCoord(key, depth, x, y);
                return std::make_tuple(x, y);
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_ray_keys",
            [](const Quadtree& self, Dtype sx, Dtype sy, Dtype ex, Dtype ey) {
                if (QuadtreeKeyRay ray; self.ComputeRayKeys(sx, sy, ex, ey, ray)) {
                    return std::optional<QuadtreeKeyRay>(ray);
                }
                return std::optional<QuadtreeKeyRay>();
            },
            py::arg("sx"),
            py::arg("sy"),
            py::arg("ex"),
            py::arg("ey"))
        .def(
            "compute_ray_coords",
            [](const Quadtree& self, Dtype sx, Dtype sy, Dtype ex, Dtype ey) {
                if (std::vector<Vector2> ray; self.ComputeRayCoords(sx, sy, ex, ey, ray)) {
                    return std::optional<std::vector<Vector2>>(ray);
                }
                return std::optional<std::vector<Vector2>>();
            },
            py::arg("sx"),
            py::arg("sy"),
            py::arg("ex"),
            py::arg("ey"))
        .def("create_node_child", &Quadtree::CreateNodeChild, py::arg("node"), py::arg("child_idx"))
        .def(
            "delete_node_child",
            &Quadtree::DeleteNodeChild,
            py::arg("node"),
            py::arg("child_idx"),
            py::arg("key"))
        .def(
            "get_node_child",
            py::overload_cast<Node*, uint32_t>(&Quadtree::GetNodeChild),
            py::arg("node"),
            py::arg("child_idx"))
        .def("is_node_collapsible", &Quadtree::IsNodeCollapsible, py::arg("node"))
        .def("expand_node", &Quadtree::ExpandNode, py::arg("node"))
        .def("prune_node", &Quadtree::PruneNode, py::arg("node"))
        .def(
            "delete_node",
            py::overload_cast<Dtype, Dtype, uint32_t>(&Quadtree::DeleteNode),
            py::arg("x"),
            py::arg("y"),
            py::arg("depth"))
        .def(
            "delete_node",
            py::overload_cast<const QuadtreeKey&, uint32_t>(&Quadtree::DeleteNode),
            py::arg("key"),
            py::arg("depth"))
        .def("clear", &Quadtree::Clear)
        .def("prune", &Quadtree::Prune)
        .def("expand", &Quadtree::Expand)
        .def_property_readonly("root", &Quadtree::GetRoot)
        .def(
            "search",
            py::overload_cast<Dtype, Dtype, uint32_t>(&Quadtree::Search, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("max_depth") = 0)
        .def(
            "search",
            py::overload_cast<const QuadtreeKey&, uint32_t>(&Quadtree::Search, py::const_),
            py::arg("key"),
            py::arg("max_depth") = 0)
        .def(
            "insert_node",
            py::overload_cast<Dtype, Dtype, uint32_t>(&Quadtree::InsertNode),
            py::arg("x"),
            py::arg("y"),
            py::arg("depth"))
        .def(
            "insert_node",
            py::overload_cast<const QuadtreeKey&, uint32_t>(&Quadtree::InsertNode),
            py::arg("key"),
            py::arg("depth"));

    // Iterators defined in QuadtreeImpl
    py::class_<
        typename Quadtree::IteratorBase,
        typename AbstractQuadtree<typename Quadtree::DataType>::QuadtreeNodeIterator>(
        tree,
        "IteratorBase")
        .def(
            "__eq__",
            [](const typename Quadtree::IteratorBase& self,
               const typename Quadtree::IteratorBase& other) { return self == other; })
        .def(
            "__ne__",
            [](const typename Quadtree::IteratorBase& self,
               const typename Quadtree::IteratorBase& other) { return self != other; })
        .def_property_readonly("node_aabb", &Quadtree::IteratorBase::GetNodeAabb)
        .def_property_readonly("node", py::overload_cast<>(&Quadtree::IteratorBase::GetNode))
        .def_property_readonly("key", &Quadtree::IteratorBase::GetKey)
        .def_property_readonly("index_key", &Quadtree::IteratorBase::GetIndexKey);

    (void) py::class_<typename Quadtree::TreeIterator, typename Quadtree::IteratorBase>(
        tree,
        "TreeIterator");
    (void) py::class_<typename Quadtree::TreeInAabbIterator, typename Quadtree::IteratorBase>(
        tree,
        "TreeInAabbIterator");
    (void) py::class_<typename Quadtree::LeafIterator, typename Quadtree::IteratorBase>(
        tree,
        "LeafIterator");
    (void) py::class_<typename Quadtree::LeafOfNodeIterator, typename Quadtree::IteratorBase>(
        tree,
        "LeafOfNodeIterator");
    (void) py::class_<typename Quadtree::LeafInAabbIterator, typename Quadtree::IteratorBase>(
        tree,
        "LeafInAabbIterator");
    (void) py::class_<typename Quadtree::WestLeafNeighborIterator, typename Quadtree::IteratorBase>(
        tree,
        "WestLeafNeighborIterator");
    (void) py::class_<typename Quadtree::EastLeafNeighborIterator, typename Quadtree::IteratorBase>(
        tree,
        "EastLeafNeighborIterator");
    (void)
        py::class_<typename Quadtree::NorthLeafNeighborIterator, typename Quadtree::IteratorBase>(
            tree,
            "NorthLeafNeighborIterator");
    (void)
        py::class_<typename Quadtree::SouthLeafNeighborIterator, typename Quadtree::IteratorBase>(
            tree,
            "SouthLeafNeighborIterator");
    py::class_<typename Quadtree::NodeOnRayIterator, typename Quadtree::IteratorBase>(
        tree,
        "NodeOnRayIterator")
        .def_property_readonly("distance", &Quadtree::NodeOnRayIterator::GetDistance);

    tree.def(
            "iter_leaf",
            [](Quadtree& self, const uint32_t max_depth) {
                return py::wrap_iterator(self.BeginLeaf(max_depth), self.EndLeaf());
            },
            py::arg("max_depth") = 0)
        .def(
            "iter_leaf_of_node",
            [](Quadtree& self,
               const QuadtreeKey& node_key,
               const uint32_t node_depth,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginLeafOfNode(node_key, node_depth, max_depth),
                    self.EndLeafOfNode());
            },
            py::arg("node_key"),
            py::arg("node_depth"),
            py::arg("max_depth") = 0)
        .def(
            "iter_leaf_in_aabb",
            [](Quadtree& self,
               const Dtype aabb_min_x,
               const Dtype aabb_min_y,
               const Dtype aabb_max_x,
               const Dtype aabb_max_y,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginLeafInAabb(aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y, max_depth),
                    self.EndLeafInAabb());
            },
            py::arg("aabb_min_x"),
            py::arg("aabb_min_y"),
            py::arg("aabb_max_x"),
            py::arg("aabb_max_y"),
            py::arg("max_depth") = 0)
        .def(
            "iter_leaf_in_aabb",
            [](Quadtree& self,
               const QuadtreeKey& aabb_min_key,
               const QuadtreeKey& aabb_max_key,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginLeafInAabb(aabb_min_key, aabb_max_key, max_depth),
                    self.EndLeafInAabb());
            },
            py::arg("aabb_min_key"),
            py::arg("aabb_max_key"),
            py::arg("max_depth") = 0)
        .def(
            "iter_node",
            [](Quadtree& self, const uint32_t max_depth) {
                return py::wrap_iterator(self.BeginTree(max_depth), self.EndTree());
            },
            py::arg("max_depth") = 0)
        .def(
            "iter_node_in_aabb",
            [](Quadtree& self,
               const Dtype aabb_min_x,
               const Dtype aabb_min_y,
               const Dtype aabb_max_x,
               const Dtype aabb_max_y,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginTreeInAabb(aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y, max_depth),
                    self.EndTreeInAabb());
            },
            py::arg("aabb_min_x"),
            py::arg("aabb_min_y"),
            py::arg("aabb_max_x"),
            py::arg("aabb_max_y"),
            py::arg("max_depth") = 0)
        .def(
            "iter_node_in_aabb",
            [](Quadtree& self,
               const QuadtreeKey& aabb_min_key,
               const QuadtreeKey& aabb_max_key,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginTreeInAabb(aabb_min_key, aabb_max_key, max_depth),
                    self.EndTreeInAabb());
            },
            py::arg("aabb_min_key"),
            py::arg("aabb_max_key"),
            py::arg("max_depth") = 0)
        .def(
            "iter_west_leaf_neighbor",
            [](Quadtree& self, const Dtype x, const Dtype y, const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginWestLeafNeighbor(x, y, max_leaf_depth),
                    self.EndWestLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_west_leaf_neighbor",
            [](Quadtree& self,
               const QuadtreeKey& key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginWestLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndWestLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_east_leaf_neighbor",
            [](Quadtree& self, const Dtype x, const Dtype y, const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginEastLeafNeighbor(x, y, max_leaf_depth),
                    self.EndEastLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_east_leaf_neighbor",
            [](Quadtree& self,
               const QuadtreeKey& key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginEastLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndEastLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_north_leaf_neighbor",
            [](Quadtree& self, const Dtype x, const Dtype y, const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginNorthLeafNeighbor(x, y, max_leaf_depth),
                    self.EndNorthLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_north_leaf_neighbor",
            [](Quadtree& self,
               const QuadtreeKey& key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginNorthLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndNorthLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_south_leaf_neighbor",
            [](Quadtree& self, const Dtype x, const Dtype y, const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginSouthLeafNeighbor(x, y, max_leaf_depth),
                    self.EndSouthLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_south_leaf_neighbor",
            [](Quadtree& self,
               const QuadtreeKey& key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginSouthLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndSouthLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_node_on_ray",
            [](Quadtree& self,
               const Dtype px,
               const Dtype py,
               const Dtype vx,
               const Dtype vy,
               const Dtype max_range,
               const Dtype node_padding,
               const bool bidirectional,
               const bool leaf_only,
               const uint32_t min_node_depth,
               const uint32_t max_node_depth) {
                return py::wrap_iterator(
                    self.BeginNodeOnRay(
                        px,
                        py,
                        vx,
                        vy,
                        max_range,
                        node_padding,
                        bidirectional,
                        leaf_only,
                        min_node_depth,
                        max_node_depth),
                    self.EndNodeOnRay());
            },
            py::arg("px"),
            py::arg("py"),
            py::arg("vx"),
            py::arg("vy"),
            py::arg("max_range") = -1,
            py::arg("node_padding") = 0,
            py::arg("bidirectional") = false,
            py::arg("leaf_only") = false,
            py::arg("min_node_depth") = 0,
            py::arg("max_node_depth") = 0);
}
