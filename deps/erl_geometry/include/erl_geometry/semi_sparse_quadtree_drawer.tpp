#pragma once

#include "semi_sparse_quadtree_drawer.hpp"

namespace erl::geometry {
    template<typename SemiSparseQuadtreeType>
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::SemiSparseQuadtreeDrawer(
        std::shared_ptr<Setting> setting,
        std::shared_ptr<const SemiSparseQuadtreeType> quadtree)
        : AbstractQuadtreeDrawer(
              std::static_pointer_cast<AbstractQuadtreeDrawer::Setting>(setting)),
          m_setting_(std::move(setting)),
          m_quadtree_(std::move(quadtree)) {
        ERL_ASSERTM(m_setting_, "setting is nullptr.");
    }

    template<typename SemiSparseQuadtreeType>
    [[nodiscard]] std::shared_ptr<
        typename SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::Setting>
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::GetSetting() const {
        return m_setting_;
    }

    template<typename SemiSparseQuadtreeType>
    std::shared_ptr<const SemiSparseQuadtreeType>
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::GetQuadtree() const {
        return m_quadtree_;
    }

    template<typename SemiSparseQuadtreeType>
    void
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::SetQuadtree(
        std::shared_ptr<const SemiSparseQuadtreeType> quadtree) {
        m_quadtree_ = std::move(quadtree);
    }

    template<typename SemiSparseQuadtreeType>
    void
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::SetDrawTreeCallback(
        DrawTreeCallback draw_tree) {
        m_draw_tree_ = std::move(draw_tree);
    }

    template<typename SemiSparseQuadtreeType>
    void
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::SetDrawLeafCallback(
        DrawLeafCallback draw_leaf) {
        m_draw_leaf_ = std::move(draw_leaf);
    }

    template<typename SemiSparseQuadtreeType>
    void
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::DrawTree(cv::Mat &mat) const {
        if (!mat.total()) {
            mat = cv::Mat(
                std::vector<int>{m_grid_map_info_->Height(), m_grid_map_info_->Width()},
                CV_8UC4,
                m_setting_->bg_color);
        }
        if (m_quadtree_ == nullptr) { return; }

        auto it = m_quadtree_->BeginTreeInAabb(
            m_setting_->area_min[0],
            m_setting_->area_min[1],
            m_setting_->area_max[0],
            m_setting_->area_max[1]);
        auto end = m_quadtree_->EndTreeInAabb();
        Eigen::Matrix2f area;
        for (; it != end; ++it) {
            const auto node_size = static_cast<float>(it.GetNodeSize());
            const float half_size = node_size * 0.5f;
            const auto x = static_cast<float>(it.GetX());
            const auto y = static_cast<float>(it.GetY());

            area << x - half_size, x + half_size, y - half_size, y + half_size;
            Eigen::Matrix2i area_px = GetPixelCoordsForPositions(area, true);

            cv::rectangle(
                mat,
                cv::Point(area_px(0, 0), area_px(1, 0)),  // min
                cv::Point(area_px(0, 1), area_px(1, 1)),  // max
                m_setting_->border_color,
                m_setting_->border_thickness);

            if (m_draw_tree_) { m_draw_tree_(this, mat, it); }
        }
    }

    template<typename SemiSparseQuadtreeType>
    void
    SemiSparseQuadtreeDrawer<SemiSparseQuadtreeType>::DrawLeaves(cv::Mat &mat) const {
        if (!mat.total()) {
            mat = cv::Mat(
                std::vector<int>{m_grid_map_info_->Height(), m_grid_map_info_->Width()},
                CV_8UC4,
                m_setting_->bg_color);
        }
        if (m_quadtree_ == nullptr) { return; }

        auto it = m_quadtree_->BeginLeafInAabb(
            m_setting_->area_min[0],
            m_setting_->area_min[1],
            m_setting_->area_max[0],
            m_setting_->area_max[1]);
        auto end = m_quadtree_->EndLeafInAabb();
        Eigen::Matrix2f area;
        for (; it != end; ++it) {

            ERL_DEBUG_ASSERT(!it->HasAnyChild(), "the iterator visits an inner node!");

            const auto node_size = static_cast<float>(it.GetNodeSize());
            const float half_size = node_size * 0.5f;
            const auto x = static_cast<float>(it.GetX());
            const auto y = static_cast<float>(it.GetY());

            area << x - half_size, x + half_size, y - half_size, y + half_size;
            Eigen::Matrix2i area_px = GetPixelCoordsForPositions(area, true);

            cv::rectangle(
                mat,
                cv::Point(area_px(0, 0), area_px(1, 0)),  // min
                cv::Point(area_px(0, 1), area_px(1, 1)),  // max
                m_setting_->border_color,
                m_setting_->border_thickness);

            if (m_draw_leaf_) { m_draw_leaf_(this, mat, it); }
        }
    }
}  // namespace erl::geometry
