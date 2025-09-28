#pragma once

#include "abstract_octree_drawer.hpp"

namespace erl::geometry {
    template<typename SemiSparseOctreeType>
    class SemiSparseOctreeDrawer : public AbstractOctreeDrawer {
    public:
        using Setting = AbstractOctreeDrawer::Setting;
        using DrawTreeCallback = std::function<void(
            const SemiSparseOctreeDrawer *,                              // this
            std::vector<std::shared_ptr<open3d::geometry::Geometry>> &,  // geometries
            typename SemiSparseOctreeType::TreeInAabbIterator &)>;
        using DrawLeafCallback = std::function<void(
            const SemiSparseOctreeDrawer *,                              // this
            std::vector<std::shared_ptr<open3d::geometry::Geometry>> &,  // geometries
            typename SemiSparseOctreeType::LeafInAabbIterator &)>;

    private:
        std::shared_ptr<Setting> m_setting_ = {};
        std::shared_ptr<const SemiSparseOctreeType> m_octree_ = nullptr;
        DrawTreeCallback m_draw_tree_ = {};
        DrawLeafCallback m_draw_leaf_ = {};

    public:
        explicit SemiSparseOctreeDrawer(
            std::shared_ptr<Setting> setting,
            std::shared_ptr<const SemiSparseOctreeType> octree = nullptr);

        using AbstractOctreeDrawer::DrawLeaves;
        using AbstractOctreeDrawer::DrawTree;

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const;

        void
        SetOctree(std::shared_ptr<const SemiSparseOctreeType> octree);

        void
        SetDrawTreeCallback(DrawTreeCallback draw_tree);

        void
        SetDrawLeafCallback(DrawLeafCallback draw_leaf);

        void
        DrawTree(
            std::vector<std::shared_ptr<open3d::geometry::Geometry>> &geometries) const override;

        void
        DrawLeaves(
            std::vector<std::shared_ptr<open3d::geometry::Geometry>> &geometries) const override;
    };
}  // namespace erl::geometry

#include "semi_sparse_octree_drawer.tpp"
