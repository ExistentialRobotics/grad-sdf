#pragma once

#include "abstract_octree.hpp"
#include "octree_key.hpp"

#include "erl_common/pybind11.hpp"

template<class PyClass, class Dtype, class Octree, class Node>
void
BindOctreeImpl(PyClass &tree) {
    using namespace erl::geometry;
    using Vector3 = Eigen::Vector3<Dtype>;

    tree.def_property_readonly("number_of_nodes", &Octree::GetSize)
        .def_property_readonly("resolution", &Octree::GetResolution)
        .def_property_readonly("tree_depth", &Octree::GetTreeDepth)
        .def_property_readonly("tree_center", &Octree::GetTreeCenter)
        .def_property_readonly("tree_center_key", &Octree::GetTreeCenterKey)
        .def_property_readonly("tree_max_half_size", &Octree::GetTreeMaxHalfSize)
        .def_property_readonly("metric_min", [](Octree &self) { return self.GetMetricMin(); })
        .def_property_readonly("metric_max", [](Octree &self) { return self.GetMetricMax(); })
        .def_property_readonly(
            "metric_min_max",
            [](Octree &self) { return self.GetMetricMinMax(); })
        .def_property_readonly("metric_aabb", [](Octree &self) { return self.GetMetricAabb(); })
        .def_property_readonly("metric_size", [](Octree &self) { return self.GetMetricSize(); })
        .def("get_node_size", &Octree::GetNodeSize, py::arg("depth"))
        .def_property_readonly("number_of_leaf_nodes", &Octree::ComputeNumberOfLeafNodes)
        .def_property_readonly("memory_usage", &Octree::GetMemoryUsage)
        .def_property_readonly("memory_usage_per_node", &Octree::GetMemoryUsagePerNode)
        .def(
            "coord_to_key",
            py::overload_cast<Dtype>(&Octree::CoordToKey, py::const_),
            py::arg("coordinate"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, uint32_t>(&Octree::CoordToKey, py::const_),
            py::arg("coordinate"),
            py::arg("depth"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, Dtype, Dtype>(&Octree::CoordToKey, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def(
            "coord_to_key",
            py::overload_cast<Dtype, Dtype, Dtype, uint32_t>(&Octree::CoordToKey, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("depth"))
        .def(
            "coord_to_key_checked",
            [](const Octree &self, Dtype coordinate) {
                if (OctreeKey::KeyType key; self.CoordToKeyChecked(coordinate, key)) {
                    return std::optional<OctreeKey::KeyType>(key);
                }
                return std::optional<OctreeKey::KeyType>();
            },
            py::arg("coordinate"))
        .def(
            "coord_to_key_checked",
            [](const Octree &self, Dtype coordinate, uint32_t depth) {
                if (OctreeKey::KeyType key; self.CoordToKeyChecked(coordinate, depth, key)) {
                    return std::optional<OctreeKey::KeyType>(key);
                }
                return std::optional<OctreeKey::KeyType>();
            },
            py::arg("coordinate"),
            py::arg("depth"))
        .def(
            "coord_to_key_checked",
            [](const Octree &self, Dtype x, Dtype y, Dtype z) {
                if (OctreeKey key; self.CoordToKeyChecked(x, y, z, key)) {
                    return std::optional<OctreeKey>(key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def(
            "coord_to_key_checked",
            [](const Octree &self, Dtype x, Dtype y, Dtype z, uint32_t depth) {
                if (OctreeKey key; self.CoordToKeyChecked(x, y, z, depth, key)) {
                    return std::optional<OctreeKey>(key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("depth"))
        .def(
            "adjust_key_to_depth",
            py::overload_cast<OctreeKey::KeyType, uint32_t>(&Octree::AdjustKeyToDepth, py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "adjust_key_to_depth",
            py::overload_cast<const OctreeKey &, uint32_t>(&Octree::AdjustKeyToDepth, py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_common_ancestor_key",
            [](const Octree &self, const OctreeKey &key1, const OctreeKey &key2) {
                OctreeKey key;
                uint32_t ancestor_depth;
                self.ComputeCommonAncestorKey(key1, key2, key, ancestor_depth);
                return std::make_tuple(key, ancestor_depth);
            })
        .def(
            "compute_west_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key; self.ComputeWestNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_east_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key; self.ComputeEastNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_north_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key;
                    self.ComputeNorthNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_south_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key;
                    self.ComputeSouthNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_top_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key; self.ComputeTopNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_bottom_neighbor_key",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                if (OctreeKey neighbor_key;
                    self.ComputeBottomNeighborKey(key, depth, neighbor_key)) {
                    return std::optional<OctreeKey>(neighbor_key);
                }
                return std::optional<OctreeKey>();
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "key_to_coord",
            py::overload_cast<OctreeKey::KeyType>(&Octree::KeyToCoord, py::const_),
            py::arg("key"))
        .def(
            "key_to_coord",
            py::overload_cast<OctreeKey::KeyType, uint32_t>(&Octree::KeyToCoord, py::const_),
            py::arg("key"),
            py::arg("depth"))
        .def(
            "key_to_coord",
            [](const Octree &self, const OctreeKey &key) {
                Dtype x, y, z;
                self.KeyToCoord(key, x, y, z);
                return std::make_tuple(x, y, z);
            },
            py::arg("key"))
        .def(
            "key_to_coord",
            [](const Octree &self, const OctreeKey &key, uint32_t depth) {
                Dtype x, y, z;
                self.KeyToCoord(key, depth, x, y, z);
                return std::make_tuple(x, y, z);
            },
            py::arg("key"),
            py::arg("depth"))
        .def(
            "compute_ray_keys",
            [](const Octree &self, Dtype sx, Dtype sy, Dtype sz, Dtype ex, Dtype ey, Dtype ez) {
                if (OctreeKeyRay ray; self.ComputeRayKeys(sx, sy, sz, ex, ey, ez, ray)) {
                    return std::optional<OctreeKeyRay>(ray);
                }
                return std::optional<OctreeKeyRay>();
            },
            py::arg("sx"),
            py::arg("sy"),
            py::arg("sz"),
            py::arg("ex"),
            py::arg("ey"),
            py::arg("ez"))
        .def(
            "compute_ray_coords",
            [](const Octree &self, Dtype sx, Dtype sy, Dtype sz, Dtype ex, Dtype ey, Dtype ez) {
                if (std::vector<Vector3> ray; self.ComputeRayCoords(sx, sy, sz, ex, ey, ez, ray)) {
                    return std::optional<std::vector<Vector3>>(ray);
                }
                return std::optional<std::vector<Vector3>>();
            },
            py::arg("sx"),
            py::arg("sy"),
            py::arg("sz"),
            py::arg("ex"),
            py::arg("ey"),
            py::arg("ez"))
        .def("create_node_child", &Octree::CreateNodeChild, py::arg("node"), py::arg("child_idx"))
        .def(
            "delete_node_child",
            &Octree::DeleteNodeChild,
            py::arg("node"),
            py::arg("child_idx"),
            py::arg("key"))
        .def(
            "get_node_child",
            py::overload_cast<Node *, uint32_t>(&Octree::GetNodeChild),
            py::arg("node"),
            py::arg("child_idx"))
        .def("is_node_collapsible", &Octree::IsNodeCollapsible, py::arg("node"))
        .def("expand_node", &Octree::ExpandNode, py::arg("node"))
        .def("prune_node", &Octree::PruneNode, py::arg("node"))
        .def(
            "delete_node",
            py::overload_cast<Dtype, Dtype, Dtype, uint32_t>(&Octree::DeleteNode),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("depth"))
        .def(
            "delete_node",
            py::overload_cast<const OctreeKey &, uint32_t>(&Octree::DeleteNode),
            py::arg("key"),
            py::arg("depth"))
        .def("clear", &Octree::Clear)
        .def("prune", &Octree::Prune)
        .def("expand", &Octree::Expand)
        .def_property_readonly("root", &Octree::GetRoot)
        .def(
            "search",
            py::overload_cast<Dtype, Dtype, Dtype, uint32_t>(&Octree::Search, py::const_),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_depth") = 0)
        .def(
            "search",
            py::overload_cast<const OctreeKey &, uint32_t>(&Octree::Search, py::const_),
            py::arg("key"),
            py::arg("max_depth") = 0)
        .def(
            "insert_node",
            py::overload_cast<Dtype, Dtype, Dtype, uint32_t>(&Octree::InsertNode),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("depth"))
        .def(
            "insert_node",
            py::overload_cast<const OctreeKey &, uint32_t>(&Octree::InsertNode),
            py::arg("key"),
            py::arg("depth"));

    // Iterators defined in OctreeImpl
    py::class_<typename Octree::IteratorBase, typename AbstractOctree<Dtype>::OctreeNodeIterator>(
        tree,
        "IteratorBase")
        .def(
            "__eq__",
            [](const typename Octree::IteratorBase &self,
               const typename Octree::IteratorBase &other) { return self == other; })
        .def(
            "__ne__",
            [](const typename Octree::IteratorBase &self,
               const typename Octree::IteratorBase &other) { return self != other; })
        .def_property_readonly("node", py::overload_cast<>(&Octree::IteratorBase::GetNode))
        .def_property_readonly("node_aabb", &Octree::IteratorBase::GetNodeAabb)
        .def_property_readonly("key", &Octree::IteratorBase::GetKey)
        .def_property_readonly("index_key", &Octree::IteratorBase::GetIndexKey);

    (void) py::class_<typename Octree::TreeIterator, typename Octree::IteratorBase>(
        tree,
        "TreeIterator");
    (void) py::class_<typename Octree::TreeInAabbIterator, typename Octree::IteratorBase>(
        tree,
        "TreeInAabbIterator");
    (void) py::class_<typename Octree::LeafIterator, typename Octree::IteratorBase>(
        tree,
        "LeafIterator");
    (void) py::class_<typename Octree::LeafOfNodeIterator, typename Octree::IteratorBase>(
        tree,
        "LeafOfNodeIterator");
    (void) py::class_<typename Octree::LeafInAabbIterator, typename Octree::IteratorBase>(
        tree,
        "LeafInAabbIterator");
    (void) py::class_<typename Octree::WestLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "WestLeafNeighborIterator");
    (void) py::class_<typename Octree::EastLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "EastLeafNeighborIterator");
    (void) py::class_<typename Octree::NorthLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "NorthLeafNeighborIterator");
    (void) py::class_<typename Octree::SouthLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "SouthLeafNeighborIterator");
    (void) py::class_<typename Octree::TopLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "TopLeafNeighborIterator");
    (void) py::class_<typename Octree::BottomLeafNeighborIterator, typename Octree::IteratorBase>(
        tree,
        "BottomLeafNeighborIterator");
    py::class_<typename Octree::NodeOnRayIterator, typename Octree::IteratorBase>(
        tree,
        "NodeOnRayIterator")
        .def_property_readonly("distance", &Octree::NodeOnRayIterator::GetDistance);

    tree.def(
            "iter_leaf",
            [](Octree &self, const uint32_t max_depth) {
                return py::wrap_iterator(self.BeginLeaf(max_depth), self.EndLeaf());
            },
            py::arg("max_depth") = 0)
        .def(
            "iter_leaf_of_node",
            [](Octree &self,
               const OctreeKey &node_key,
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
            [](Octree &self,
               const Dtype aabb_min_x,
               const Dtype aabb_min_y,
               const Dtype aabb_min_z,
               const Dtype aabb_max_x,
               const Dtype aabb_max_y,
               const Dtype aabb_max_z,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginLeafInAabb(
                        aabb_min_x,
                        aabb_min_y,
                        aabb_min_z,
                        aabb_max_x,
                        aabb_max_y,
                        aabb_max_z,
                        max_depth),
                    self.EndLeafInAabb());
            },
            py::arg("aabb_min_x"),
            py::arg("aabb_min_y"),
            py::arg("aabb_min_z"),
            py::arg("aabb_max_x"),
            py::arg("aabb_max_y"),
            py::arg("aabb_max_z"),
            py::arg("max_depth") = 0)
        .def(
            "iter_leaf_in_aabb",
            [](Octree &self,
               const OctreeKey &aabb_min_key,
               const OctreeKey &aabb_max_key,
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
            [](Octree &self, const uint32_t max_depth) {
                return py::wrap_iterator(self.BeginTree(max_depth), self.EndTree());
            },
            py::arg("max_depth") = 0)
        .def(
            "iter_node_in_aabb",
            [](Octree &self,
               const Dtype aabb_min_x,
               const Dtype aabb_min_y,
               const Dtype aabb_min_z,
               const Dtype aabb_max_x,
               const Dtype aabb_max_y,
               const Dtype aabb_max_z,
               const uint32_t max_depth) {
                return py::wrap_iterator(
                    self.BeginTreeInAabb(
                        aabb_min_x,
                        aabb_min_y,
                        aabb_min_z,
                        aabb_max_x,
                        aabb_max_y,
                        aabb_max_z,
                        max_depth),
                    self.EndTreeInAabb());
            },
            py::arg("aabb_min_x"),
            py::arg("aabb_min_y"),
            py::arg("aabb_min_z"),
            py::arg("aabb_max_x"),
            py::arg("aabb_max_y"),
            py::arg("aabb_max_z"),
            py::arg("max_depth") = 0)
        .def(
            "iter_node_in_aabb",
            [](Octree &self,
               const OctreeKey &aabb_min_key,
               const OctreeKey &aabb_max_key,
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
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginWestLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndWestLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_west_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
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
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginEastLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndEastLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_east_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
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
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginNorthLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndNorthLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_north_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
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
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginSouthLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndSouthLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_south_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
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
            "iter_top_leaf_neighbor",
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginTopLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndTopLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_top_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginTopLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndTopLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_bottom_leaf_neighbor",
            [](Octree &self,
               const Dtype x,
               const Dtype y,
               const Dtype z,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginBottomLeafNeighbor(x, y, z, max_leaf_depth),
                    self.EndBottomLeafNeighbor());
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_bottom_leaf_neighbor",
            [](Octree &self,
               const OctreeKey &key,
               const uint32_t key_depth,
               const uint32_t max_leaf_depth) {
                return py::wrap_iterator(
                    self.BeginBottomLeafNeighbor(key, key_depth, max_leaf_depth),
                    self.EndBottomLeafNeighbor());
            },
            py::arg("key"),
            py::arg("key_depth"),
            py::arg("max_leaf_depth") = 0)
        .def(
            "iter_node_on_ray",
            [](Octree &self,
               const Dtype px,
               const Dtype py,
               const Dtype pz,
               const Dtype vx,
               const Dtype vy,
               const Dtype vz,
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
                        pz,
                        vx,
                        vy,
                        vz,
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
            py::arg("pz"),
            py::arg("vx"),
            py::arg("vy"),
            py::arg("vz"),
            py::arg("max_range") = -1,
            py::arg("node_padding") = 0,
            py::arg("bidirectional") = false,
            py::arg("leaf_only") = true,
            py::arg("min_node_depth") = 0,
            py::arg("max_node_depth") = 0);
}
