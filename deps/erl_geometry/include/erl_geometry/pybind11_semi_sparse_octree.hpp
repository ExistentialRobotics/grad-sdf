#pragma once

#ifdef ERL_USE_OPEN3D
    #include "open3d_visualizer_wrapper.hpp"
    #include "pybind11_semi_sparse_octree_drawer.hpp"
#endif

#include "pybind11_octree_impl.hpp"
#include "semi_sparse_nd_tree_setting.hpp"
#include "semi_sparse_octree_base.hpp"

#ifdef ERL_USE_LIBTORCH
    #include <torch/extension.h>
#endif

template<class Octree, class Node>
auto
BindSemiSparseOctree(
    const py::module &m,
    const char *tree_name,
    std::function<void(py::class_<
                       Octree,
                       erl::geometry::AbstractOctree<typename Octree::DataType>,
                       std::shared_ptr<Octree>> &)> additional_bindings = nullptr) {

    using namespace erl::common;
    using namespace erl::geometry;
    using Dtype = typename Octree::DataType;
    using Matrix3X = Eigen::Matrix3X<Dtype>;

    py::class_<Octree, AbstractOctree<Dtype>, std::shared_ptr<Octree>> tree(m, tree_name);

    if (additional_bindings) { additional_bindings(tree); }
    if (std::is_same_v<typename Octree::Setting, SemiSparseNdTreeSetting> &&
        !py::hasattr(tree, "Setting")) {
        ERL_DEBUG("Bind default Setting type to {}", tree_name);
        tree.def("Setting", []() { return std::make_shared<typename Octree::Setting>(); });
    }

    // AbstractOctree methods are defined in bind_abstract_octree.cpp

    // SemiSparseOctree
    tree.def(py::init<>())
        .def(
            py::init<>([](const std::shared_ptr<typename Octree::Setting> &setting) {
                return std::make_shared<Octree>(setting);
            }),
            py::arg("setting"))
        .def(
            py::init<>(
                [](const std::string &filename) { return std::make_shared<Octree>(filename); }),
            py::arg("filename"))
        .def_property_readonly("setting", &Octree::template GetSetting<typename Octree::Setting>)
        .def_property_readonly("parents", &Octree::GetParents)
        .def_property_readonly("children", &Octree::GetChildren)
        .def_property_readonly("voxels", &Octree::GetVoxels)
        .def_property_readonly("voxel_centers", &Octree::GetVoxelCenters)
        .def_property_readonly("vertices", &Octree::GetVertices)
        .def_property_readonly("num_vertices", &Octree::GetVertexCount)
        .def_property_readonly("vertex_keys", &Octree::GetVertexKeys)
        .def(
            "insert_points",
            py::overload_cast<const Matrix3X &>(&Octree::InsertPoints),
            py::arg("points"))
        .def(
            "insert_keys",
            [](Octree &self, const Eigen::Matrix3X<OctreeKey::KeyType> &keys) {
                auto keys_ptr = reinterpret_cast<const OctreeKey *>(keys.data());
                self.InsertKeys(keys_ptr, keys.cols());
            },
            py::arg("keys"))
        .def("insert_key", &Octree::InsertKey, py::arg("key"), py::arg("max_depth"))
        .def(
            "find_voxel_indices",
            py::overload_cast<const Matrix3X &, bool>(&Octree::FindVoxelIndices, py::const_),
            py::arg("points"),
            py::arg("parallel"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "find_voxel_indices",
            [](const Octree &self, const Eigen::Matrix3X<OctreeKey::KeyType> &keys, bool parallel) {
                auto keys_ptr = reinterpret_cast<const OctreeKey *>(keys.data());
                return self.FindVoxelIndices(keys_ptr, keys.cols(), parallel);
            },
            py::arg("keys"),
            py::arg("parallel"))
        .def("find_voxel_index", &Octree::FindVoxelIndex, py::arg("key"));

#ifdef ERL_USE_LIBTORCH
    // extra bindings to support libtorch tensors
    tree.def_property_readonly(
            "parents_tensor",
            [](const Octree &self) {
                const auto &parents = self.GetParents();
                torch::Tensor parents_tensor =
                    torch::from_blob(
                        const_cast<typename Octree::NodeIndex *>(parents.data()),
                        {static_cast<long>(parents.size())},
                        c10::CppTypeToScalarType<typename Octree::NodeIndex>::value)
                        .clone();
                return parents_tensor;
            })
        .def_property_readonly(
            "children_tensor",
            [](const Octree &self) {
                const auto &children = self.GetChildren();
                torch::Tensor children_tensor =
                    torch::from_blob(
                        const_cast<typename Octree::NodeIndex *>(children.data()),
                        {static_cast<long>(children.cols()), 8},
                        c10::CppTypeToScalarType<typename Octree::NodeIndex>::value)
                        .clone();
                return children_tensor;
            })
        .def_property_readonly(
            "voxels_tensor",
            [](const Octree &self) {
                const auto &voxels = self.GetVoxels();
                torch::Tensor voxels_tensor =
                    torch::from_blob(
                        const_cast<OctreeKey::KeyType *>(voxels.data()),
                        {static_cast<long>(voxels.cols()), 4},  // (x, y, z, size)
                        c10::CppTypeToScalarType<OctreeKey::KeyType>::value)
                        .clone();
                return voxels_tensor;
            })
        .def_property_readonly(
            "voxel_centers_tensor",
            [](const Octree &self) {
                const auto &voxel_centers = self.GetVoxelCenters();
                torch::Tensor voxel_centers_tensor =
                    torch::from_blob(
                        const_cast<Dtype *>(voxel_centers.data()),
                        {static_cast<long>(voxel_centers.cols()), 3},  // (x, y, z)
                        c10::CppTypeToScalarType<Dtype>::value)
                        .clone();
                return voxel_centers_tensor;
            })
        .def_property_readonly(
            "vertices_tensor",
            [](const Octree &self) {
                const auto &vertices = self.GetVertices();
                torch::Tensor vertices_tensor =
                    torch::from_blob(
                        const_cast<typename Octree::NodeIndex *>(vertices.data()),
                        {static_cast<long>(vertices.cols()), 8},
                        c10::CppTypeToScalarType<typename Octree::NodeIndex>::value)
                        .clone();
                return vertices_tensor;
            })
        .def(
            "insert_points",
            [](Octree &self, const torch::Tensor &points) {
                ERL_ASSERTM(points.is_floating_point(), "points must be floating point tensor.");
                ERL_ASSERTM(points.ndimension() >= 2, "points must be (..., 3) tensor.");
                ERL_ASSERTM(
                    points.size(points.ndimension() - 1) == 3,
                    "points must be (..., 3) tensor.");

                const auto setting = self.template GetSetting<typename Octree::Setting>();
                const uint32_t key_offset = 1 << (setting->tree_depth - 1);
                torch::Tensor keys = (points / setting->resolution + key_offset).cpu();
                keys = keys.to(c10::CppTypeToScalarType<OctreeKey::KeyType>::value).contiguous();

                const auto key_val_ptr = keys.const_data_ptr<OctreeKey::KeyType>();
                const auto key_ptr = reinterpret_cast<const OctreeKey *>(key_val_ptr);
                const long num_keys = keys.numel() / 3;

                auto indices = self.InsertKeys(key_ptr, num_keys);
                torch::Tensor indices_tensor =
                    torch::from_blob(
                        indices.data(),
                        keys.sizes().slice(0, keys.ndimension() - 1),
                        c10::CppTypeToScalarType<typename Octree::NodeIndex>::value)
                        .clone()
                        .to(keys.device());
                return indices_tensor;
            },
            py::arg("points"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "insert_keys",
            [](Octree &self, const torch::Tensor &keys) {
                ERL_ASSERTM(!keys.is_floating_point(), "keys must be integer tensor.");
                ERL_ASSERTM(keys.ndimension() >= 2, "keys must be (..., 3) tensor.");
                ERL_ASSERTM(keys.size(keys.ndimension() - 1) == 3, "keys must be (..., 3) tensor.");

                const torch::Tensor keys_ =
                    keys.cpu().to(c10::CppTypeToScalarType<OctreeKey::KeyType>::value).contiguous();

                const auto key_val_ptr = keys_.const_data_ptr<OctreeKey::KeyType>();
                const auto key_ptr = reinterpret_cast<const OctreeKey *>(key_val_ptr);
                const long num_keys = keys_.numel() / 3;

                auto indices = self.InsertKeys(key_ptr, num_keys);
                torch::Tensor indices_tensor =
                    torch::from_blob(
                        indices.data(),
                        keys.sizes().slice(0, keys.ndimension() - 1),
                        c10::CppTypeToScalarType<typename Octree::NodeIndex>::value)
                        .clone()
                        .to(keys.device());
                return indices_tensor;
            },
            py::arg("keys"),
            py::call_guard<py::gil_scoped_release>());
#endif

    BindOctreeImpl<decltype(tree), Dtype, Octree, Node>(tree);

#ifdef ERL_USE_OPEN3D
    BindSemiSparseOctreeDrawer<Octree>(tree, "Drawer");

    tree.def(
        "visualize",
        [](std::shared_ptr<Octree> &self,
           const bool leaf_only,
           float scaling,
           const Eigen::Vector3d &area_min,
           const Eigen::Vector3d &area_max,
           const Eigen::Vector3d &border_color,
           const int window_width,
           const int window_height,
           const int window_left,
           const int window_top) {
            using Drawer = SemiSparseOctreeDrawer<Octree>;
            auto drawer_setting = std::make_shared<typename Drawer::Setting>();
            drawer_setting->scaling = scaling;
            drawer_setting->area_min = area_min;
            drawer_setting->area_max = area_max;
            drawer_setting->border_color = border_color;

            auto drawer = std::make_shared<Drawer>(drawer_setting, self);
            auto visualizer_setting = std::make_shared<Open3dVisualizerWrapper::Setting>();
            visualizer_setting->window_width = window_width;
            visualizer_setting->window_height = window_height;
            visualizer_setting->window_left = window_left;
            visualizer_setting->window_top = window_top;
            const auto visualizer = std::make_shared<Open3dVisualizerWrapper>(visualizer_setting);
            std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;
            if (leaf_only) {
                drawer->DrawLeaves(geometries);
            } else {
                drawer->DrawTree(geometries);
            }
            visualizer->AddGeometries(geometries);
            visualizer->Show();
        },
        py::arg("leaf_only") = false,
        py::arg("scaling") = 1.0f,
        py::arg("area_min") = Eigen::Vector3f(-1.0, -1.0, -1.0),
        py::arg("area_max") = Eigen::Vector3f(1.0, 1.0, 1.0),
        py::arg("border_color") = Eigen::Vector3f(0.0, 0.0, 0.0),
        py::arg("window_width") = 1920,
        py::arg("window_height") = 1080,
        py::arg("window_left") = 50,
        py::arg("window_top") = 50);
#endif

    return tree;
}
