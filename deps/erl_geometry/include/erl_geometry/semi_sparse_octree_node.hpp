#pragma once

#include "abstract_octree_node.hpp"

#include "erl_common/template_helper.hpp"

namespace erl::geometry {
    class SemiSparseOctreeNode : public AbstractOctreeNode {
    public:
        using NodeIndex = int64_t;

    protected:
        NodeIndex m_node_index_ = -1;

    public:
        explicit SemiSparseOctreeNode(
            const uint32_t depth = 0,
            const int child_index = -1,
            const NodeIndex node_index = -1)
            : AbstractOctreeNode(depth, child_index), m_node_index_(node_index) {}

        SemiSparseOctreeNode(const SemiSparseOctreeNode &other) = default;
        SemiSparseOctreeNode &
        operator=(const SemiSparseOctreeNode &other) = default;
        SemiSparseOctreeNode(SemiSparseOctreeNode &&other) noexcept = default;
        SemiSparseOctreeNode &
        operator=(SemiSparseOctreeNode &&other) noexcept = default;

        bool
        operator==(const AbstractOctreeNode &other) const override {
            if (AbstractOctreeNode::operator==(other)) {
                const auto &other_node = reinterpret_cast<const SemiSparseOctreeNode &>(other);
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

        [[nodiscard]] std::unique_ptr<AbstractOctreeNode>
        Create(const uint32_t depth, const int child_index) const override {
            CheckRuntimeType<SemiSparseOctreeNode>(this, /*debug_only*/ true);
            return std::make_unique<SemiSparseOctreeNode>(depth, child_index, /*node_index*/ -1);
        }

        [[nodiscard]] std::unique_ptr<AbstractOctreeNode>
        Clone() const override {
            CheckRuntimeType<SemiSparseOctreeNode>(this, /*debug_only*/ true);
            return std::make_unique<SemiSparseOctreeNode>(*this);
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
