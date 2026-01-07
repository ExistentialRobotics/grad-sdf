#pragma once

#include "abstract_quadtree_node.hpp"

#include "erl_common/template_helper.hpp"

namespace erl::geometry {
    class SemiSparseQuadtreeNode : public AbstractQuadtreeNode {
    public:
        using NodeIndex = int64_t;

    protected:
        NodeIndex m_node_index_ = -1;

    public:
        explicit SemiSparseQuadtreeNode(
            const uint32_t depth = 0,
            const int child_index = -1,
            const NodeIndex node_index = -1)
            : AbstractQuadtreeNode(depth, child_index), m_node_index_(node_index) {}

        SemiSparseQuadtreeNode(const SemiSparseQuadtreeNode &other) = default;
        SemiSparseQuadtreeNode &
        operator=(const SemiSparseQuadtreeNode &other) = default;
        SemiSparseQuadtreeNode(SemiSparseQuadtreeNode &&other) noexcept = default;
        SemiSparseQuadtreeNode &
        operator=(SemiSparseQuadtreeNode &&other) noexcept = default;

        bool
        operator==(const AbstractQuadtreeNode &other) const override {
            if (AbstractQuadtreeNode::operator==(other)) {
                const auto &other_node = reinterpret_cast<const SemiSparseQuadtreeNode &>(other);
                return m_node_index_ == other_node.m_node_index_;
            }
            return false;
        }

        [[nodiscard]] NodeIndex
        GetNodeIndex() const {
            return m_node_index_;
        }

        void
        SetNodeIndex(const NodeIndex node_index) {
            m_node_index_ = node_index;
        }

        [[nodiscard]] std::unique_ptr<AbstractQuadtreeNode>
        Create(const uint32_t depth, const int child_index) const override {
            CheckRuntimeType<SemiSparseQuadtreeNode>(this, /*debug_only*/ true);
            return std::make_unique<SemiSparseQuadtreeNode>(depth, child_index, /*node_index*/ -1);
        }

        [[nodiscard]] std::unique_ptr<AbstractQuadtreeNode>
        Clone() const override {
            CheckRuntimeType<SemiSparseQuadtreeNode>(this, /*debug_only*/ true);
            return std::make_unique<SemiSparseQuadtreeNode>(*this);
        }

        //-- file IO
        std::istream &
        ReadData(std::istream &s) override {
            s.read(reinterpret_cast<char *>(&m_node_index_), sizeof(NodeIndex));
            return s;
        }

        std::ostream &
        WriteData(std::ostream &s) const override {
            s.write(reinterpret_cast<const char *>(&m_node_index_), sizeof(NodeIndex));
            return s;
        }
    };
}  // namespace erl::geometry
