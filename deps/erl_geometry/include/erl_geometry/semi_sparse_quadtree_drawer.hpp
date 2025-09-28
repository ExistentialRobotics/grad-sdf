#pragma once

#include "abstract_quadtree_drawer.hpp"

#include "erl_common/grid_map_info.hpp"

namespace erl::geometry {
    template<typename SemiSparseQuadtreeType>
    class SemiSparseQuadtreeDrawer : public AbstractQuadtreeDrawer {
    public:
        using Tree = SemiSparseQuadtreeType;
        using Dtype = typename Tree::DataType;
        using Setting = AbstractQuadtreeDrawer::Setting;
        using DrawTreeCallback = std::function<
            void(const SemiSparseQuadtreeDrawer *, cv::Mat &, typename Tree::TreeInAabbIterator &)>;
        using DrawLeafCallback = std::function<
            void(const SemiSparseQuadtreeDrawer *, cv::Mat &, typename Tree::LeafInAabbIterator &)>;

    private:
        std::shared_ptr<Setting> m_setting_ = {};
        std::shared_ptr<const Tree> m_quadtree_ = nullptr;
        DrawTreeCallback m_draw_tree_ = {};
        DrawLeafCallback m_draw_leaf_ = {};

    public:
        explicit SemiSparseQuadtreeDrawer(
            std::shared_ptr<Setting> setting,
            std::shared_ptr<const Tree> quadtree = nullptr);

        using AbstractQuadtreeDrawer::DrawLeaves;
        using AbstractQuadtreeDrawer::DrawTree;

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetQuadtree() const;

        void
        SetQuadtree(std::shared_ptr<const Tree> quadtree);

        void
        SetDrawTreeCallback(DrawTreeCallback draw_tree);

        void
        SetDrawLeafCallback(DrawLeafCallback draw_leaf);

        void
        DrawTree(cv::Mat &mat) const override;

        void
        DrawLeaves(cv::Mat &mat) const override;
    };
}  // namespace erl::geometry

#include "semi_sparse_quadtree_drawer.tpp"
