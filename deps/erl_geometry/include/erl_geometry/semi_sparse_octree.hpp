#pragma once

#include "semi_sparse_nd_tree_setting.hpp"
#include "semi_sparse_octree_base.hpp"

#include "erl_common/serialization.hpp"

namespace erl::geometry {

    template<typename Dtype>
    class SemiSparseOctree
        : public SemiSparseOctreeBase<Dtype, SemiSparseOctreeNode, SemiSparseNdTreeSetting> {
    public:
        using Setting = SemiSparseNdTreeSetting;
        using Super = SemiSparseOctreeBase<Dtype, SemiSparseOctreeNode, Setting>;

        explicit SemiSparseOctree(const std::shared_ptr<Setting> &setting)
            : Super(setting) {}

        SemiSparseOctree()
            : SemiSparseOctree(std::make_shared<Setting>()) {}

        explicit SemiSparseOctree(const std::string &filename)
            : SemiSparseOctree() {
            ERL_ASSERTM(
                common::Serialization<SemiSparseOctree>::Read(filename, this),
                "Failed to read SemiSparseOctree from file: {}",
                filename);
        }

        SemiSparseOctree(const SemiSparseOctree &other) = default;
        SemiSparseOctree &
        operator=(const SemiSparseOctree &other) = default;
        SemiSparseOctree(SemiSparseOctree &&other) noexcept = default;
        SemiSparseOctree &
        operator=(SemiSparseOctree &&other) noexcept = default;

    protected:
        [[nodiscard]] std::shared_ptr<AbstractOctree<Dtype>>
        Create(const std::shared_ptr<NdTreeSetting> &setting) const override {
            auto tree_setting = std::dynamic_pointer_cast<Setting>(setting);
            if (tree_setting == nullptr) {
                ERL_DEBUG_ASSERT(
                    setting == nullptr,
                    "setting is not the type for OccupancyOctree.");
                tree_setting = std::make_shared<Setting>();
            }
            return std::make_shared<SemiSparseOctree>(tree_setting);
        }
    };

    using SemiSparseOctreeD = SemiSparseOctree<double>;
    using SemiSparseOctreeF = SemiSparseOctree<float>;
}  // namespace erl::geometry
