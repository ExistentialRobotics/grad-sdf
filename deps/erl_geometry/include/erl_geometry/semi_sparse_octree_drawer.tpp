#pragma once

#include "semi_sparse_octree_drawer.hpp"

namespace erl::geometry {

    template<typename SemiSparseOctreeType>
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::SemiSparseOctreeDrawer(
        std::shared_ptr<Setting> setting,
        std::shared_ptr<const SemiSparseOctreeType> octree)
        : AbstractOctreeDrawer(std::static_pointer_cast<AbstractOctreeDrawer::Setting>(setting)),
          m_setting_(std::move(setting)),
          m_octree_(std::move(octree)) {
        ERL_ASSERTM(m_setting_, "setting is nullptr.");
    }

    template<typename SemiSparseOctreeType>
    [[nodiscard]] std::shared_ptr<typename SemiSparseOctreeDrawer<SemiSparseOctreeType>::Setting>
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::GetSetting() const {
        return m_setting_;
    }

    template<typename SemiSparseOctreeType>
    void
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::SetOctree(
        std::shared_ptr<const SemiSparseOctreeType> octree) {
        m_octree_ = std::move(octree);
    }

    template<typename SemiSparseOctreeType>
    void
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::SetDrawTreeCallback(DrawTreeCallback draw_tree) {
        m_draw_tree_ = std::move(draw_tree);
    }

    template<typename SemiSparseOctreeType>
    void
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::SetDrawLeafCallback(DrawLeafCallback draw_leaf) {
        m_draw_leaf_ = std::move(draw_leaf);
    }

    template<typename SemiSparseOctreeType>
    void
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::DrawTree(
        std::vector<std::shared_ptr<open3d::geometry::Geometry>> &geometries) const {
        if (geometries.empty()) { geometries = GetBlankGeometries(); }
        ERL_ASSERTM(
            geometries.size() >= 2,
            "geometries should be empty or contain at least 2 elements: triangle mesh and line "
            "set.");

        const std::shared_ptr<open3d::geometry::VoxelGrid> boxes =
            std::dynamic_pointer_cast<open3d::geometry::VoxelGrid>(geometries[0]);
        ERL_ASSERTM(boxes, "the first element of geometries should be a triangle mesh.");
        const std::shared_ptr<open3d::geometry::LineSet> node_border =
            std::dynamic_pointer_cast<open3d::geometry::LineSet>(geometries[1]);
        ERL_ASSERTM(node_border, "the second element of geometries should be a line set.");

        if (m_octree_ == nullptr) {
            ERL_WARN("no occupancy octree is set.");
            return;
        }

        // draw
        const double scaling = m_setting_->scaling;
        boxes->Clear();
        boxes->voxel_size_ = m_octree_->GetResolution() * scaling;
        boxes->origin_ = (m_setting_->area_max + m_setting_->area_min) / 2.0 * scaling;
        node_border->Clear();

        auto it = m_octree_->BeginTreeInAabb(
            m_setting_->area_min[0],
            m_setting_->area_min[1],
            m_setting_->area_min[2],
            m_setting_->area_max[0],
            m_setting_->area_max[1],
            m_setting_->area_max[2]);
        auto end = m_octree_->EndTreeInAabb();

        const double area_size = (m_setting_->area_max - m_setting_->area_min).maxCoeff();
        for (; it != end; ++it) {
            const double node_size = it.GetNodeSize();
            if (node_size > area_size) { continue; }  // skip nodes that are too large
            const double half_size = node_size / 2.0 * scaling;
            const double x = it.GetX() * scaling;
            const double y = it.GetY() * scaling;
            const double z = it.GetZ() * scaling;

            const auto n = static_cast<int>(node_border->points_.size());
            node_border->points_.emplace_back(x - half_size, y - half_size, z - half_size);
            node_border->points_.emplace_back(x + half_size, y - half_size, z - half_size);
            node_border->points_.emplace_back(x + half_size, y + half_size, z - half_size);
            node_border->points_.emplace_back(x - half_size, y + half_size, z - half_size);
            node_border->points_.emplace_back(x - half_size, y - half_size, z + half_size);
            node_border->points_.emplace_back(x + half_size, y - half_size, z + half_size);
            node_border->points_.emplace_back(x + half_size, y + half_size, z + half_size);
            node_border->points_.emplace_back(x - half_size, y + half_size, z + half_size);
            node_border->lines_.emplace_back(n + 0, n + 1);
            node_border->lines_.emplace_back(n + 1, n + 2);
            node_border->lines_.emplace_back(n + 2, n + 3);
            node_border->lines_.emplace_back(n + 3, n + 0);
            node_border->lines_.emplace_back(n + 4, n + 5);
            node_border->lines_.emplace_back(n + 5, n + 6);
            node_border->lines_.emplace_back(n + 6, n + 7);
            node_border->lines_.emplace_back(n + 7, n + 4);
            node_border->lines_.emplace_back(n + 0, n + 4);
            node_border->lines_.emplace_back(n + 1, n + 5);
            node_border->lines_.emplace_back(n + 2, n + 6);
            node_border->lines_.emplace_back(n + 3, n + 7);

            if (m_draw_tree_) { m_draw_tree_(this, geometries, it); }
        }
        node_border->PaintUniformColor(m_setting_->border_color);
    }

    template<typename SemiSparseOctreeType>
    void
    SemiSparseOctreeDrawer<SemiSparseOctreeType>::DrawLeaves(
        std::vector<std::shared_ptr<open3d::geometry::Geometry>> &geometries) const {
        if (geometries.empty()) { geometries = GetBlankGeometries(); }
        ERL_ASSERTM(
            geometries.size() >= 2,
            "geometries should be empty or contain at least 2 elements: triangle mesh and line "
            "set.");

        const std::shared_ptr<open3d::geometry::VoxelGrid> boxes =
            std::dynamic_pointer_cast<open3d::geometry::VoxelGrid>(geometries[0]);
        ERL_ASSERTM(boxes, "the first element of geometries should be a triangle mesh.");
        const std::shared_ptr<open3d::geometry::LineSet> node_border =
            std::dynamic_pointer_cast<open3d::geometry::LineSet>(geometries[1]);
        ERL_ASSERTM(node_border, "the second element of geometries should be a line set.");

        std::shared_ptr<const SemiSparseOctreeType> octree =
            std::static_pointer_cast<const SemiSparseOctreeType>(m_octree_);
        if (octree == nullptr) {
            ERL_WARN("octree is not an occupancy octree.");
            return;
        }

        auto it = octree->BeginLeafInAabb(
            m_setting_->area_min[0],
            m_setting_->area_min[1],
            m_setting_->area_min[2],
            m_setting_->area_max[0],
            m_setting_->area_max[1],
            m_setting_->area_max[2]);
        auto end = octree->EndLeafInAabb();

        const double scaling = m_setting_->scaling;
        boxes->Clear();
        boxes->voxel_size_ = m_octree_->GetResolution() * scaling;
        boxes->origin_ = (m_setting_->area_max + m_setting_->area_min) / 2.0 * scaling;
        node_border->Clear();
        for (; it != end; ++it) {
            ERL_DEBUG_ASSERT(!it->HasAnyChild(), "the iterator visits an inner node!");

            const double half_size = it.GetNodeSize() / 2.0 * scaling;
            const double x = it.GetX() * scaling;
            const double y = it.GetY() * scaling;
            const double z = it.GetZ() * scaling;

            const auto n = static_cast<int>(node_border->points_.size());
            node_border->points_.emplace_back(x - half_size, y - half_size, z - half_size);
            node_border->points_.emplace_back(x + half_size, y - half_size, z - half_size);
            node_border->points_.emplace_back(x + half_size, y + half_size, z - half_size);
            node_border->points_.emplace_back(x - half_size, y + half_size, z - half_size);
            node_border->points_.emplace_back(x - half_size, y - half_size, z + half_size);
            node_border->points_.emplace_back(x + half_size, y - half_size, z + half_size);
            node_border->points_.emplace_back(x + half_size, y + half_size, z + half_size);
            node_border->points_.emplace_back(x - half_size, y + half_size, z + half_size);
            node_border->lines_.emplace_back(n + 0, n + 1);
            node_border->lines_.emplace_back(n + 1, n + 2);
            node_border->lines_.emplace_back(n + 2, n + 3);
            node_border->lines_.emplace_back(n + 3, n + 0);
            node_border->lines_.emplace_back(n + 4, n + 5);
            node_border->lines_.emplace_back(n + 5, n + 6);
            node_border->lines_.emplace_back(n + 6, n + 7);
            node_border->lines_.emplace_back(n + 7, n + 4);
            node_border->lines_.emplace_back(n + 0, n + 4);
            node_border->lines_.emplace_back(n + 1, n + 5);
            node_border->lines_.emplace_back(n + 2, n + 6);
            node_border->lines_.emplace_back(n + 3, n + 7);
            if (m_draw_leaf_) { m_draw_leaf_(this, geometries, it); }
        }
        node_border->PaintUniformColor(m_setting_->border_color);
    }
}  // namespace erl::geometry
