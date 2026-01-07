#pragma once

#include "semi_sparse_nd_tree_setting.hpp"
#include "semi_sparse_quadtree_base.hpp"

#include "erl_common/serialization.hpp"

namespace erl::geometry {

    template<typename Dtype>
    class SemiSparseQuadtree
        : public SemiSparseQuadtreeBase<Dtype, SemiSparseQuadtreeNode, SemiSparseNdTreeSetting> {
    public:
        using Setting = SemiSparseNdTreeSetting;
        using Super = SemiSparseQuadtreeBase<Dtype, SemiSparseQuadtreeNode, Setting>;

        explicit SemiSparseQuadtree(const std::shared_ptr<Setting> &setting)
            : Super(setting) {}

        SemiSparseQuadtree()
            : SemiSparseQuadtree(std::make_shared<Setting>()) {}

        explicit SemiSparseQuadtree(const std::string &filename)
            : SemiSparseQuadtree() {
            ERL_ASSERTM(
                common::serialization::Serialization<SemiSparseQuadtree>::Read(filename, this),
                "Failed to read SemiSparseQuadtree from file: {}",
                filename);
        }

        SemiSparseQuadtree(const SemiSparseQuadtree &other) = default;
        SemiSparseQuadtree &
        operator=(const SemiSparseQuadtree &other) = default;
        SemiSparseQuadtree(SemiSparseQuadtree &&other) noexcept = default;
        SemiSparseQuadtree &
        operator=(SemiSparseQuadtree &&other) noexcept = default;

    protected:
        [[nodiscard]] std::shared_ptr<AbstractQuadtree<Dtype>>
        Create(const std::shared_ptr<NdTreeSetting> &setting) const override {
            auto tree_setting = std::dynamic_pointer_cast<Setting>(setting);
            if (tree_setting == nullptr) {
                ERL_DEBUG_ASSERT(
                    setting == nullptr,
                    "setting is not the type for OccupancyQuadtree.");
                tree_setting = std::make_shared<Setting>();
            }
            return std::make_shared<SemiSparseQuadtree>(tree_setting);
        }
    };

    using SemiSparseQuadtreeD = SemiSparseQuadtree<double>;
    using SemiSparseQuadtreeF = SemiSparseQuadtree<float>;
}  // namespace erl::geometry
