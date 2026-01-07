#include "erl_geometry/abstract_octree_node.hpp"

#include "erl_common/string_utils.hpp"

namespace erl::geometry {

    AbstractOctreeNode::AbstractOctreeNode(const uint32_t depth, const int child_index)
        : m_depth_(depth), m_child_index_(child_index) {}

    AbstractOctreeNode::AbstractOctreeNode(const AbstractOctreeNode &other)
        : m_depth_(other.m_depth_),
          m_child_index_(other.m_child_index_),
          m_num_children_(other.m_num_children_) {
        if (other.m_num_children_ == 0) { return; }
        for (int i = 0; i < 8; ++i) {
            const auto &child = other.m_children_[i];
            if (child == nullptr) { continue; }
            m_children_[i] = child->Clone();
        }
    }

    AbstractOctreeNode &
    AbstractOctreeNode::operator=(const AbstractOctreeNode &other) {
        if (this == &other) { return *this; }
        m_depth_ = other.m_depth_;
        m_child_index_ = other.m_child_index_;
        m_num_children_ = other.m_num_children_;
        if (other.m_num_children_ == 0) { return *this; }
        for (int i = 0; i < 8; ++i) {
            if (const auto &child = other.m_children_[i]; child == nullptr) {
                m_children_[i] = nullptr;
            } else {
                m_children_[i] = child->Clone();
            }
        }
        return *this;
    }

    AbstractOctreeNode::AbstractOctreeNode(AbstractOctreeNode &&other) noexcept
        : m_depth_(other.m_depth_),
          m_child_index_(other.m_child_index_),
          m_num_children_(other.m_num_children_),
          m_children_(std::move(other.m_children_)) {
        other.m_depth_ = 0;
        other.m_child_index_ = -1;
        other.m_num_children_ = 0;
    }

    AbstractOctreeNode &
    AbstractOctreeNode::operator=(AbstractOctreeNode &&other) noexcept {
        if (this == &other) { return *this; }
        m_depth_ = other.m_depth_;
        m_child_index_ = other.m_child_index_;
        m_num_children_ = other.m_num_children_;
        m_children_ = std::move(other.m_children_);
        other.m_depth_ = 0;
        other.m_child_index_ = -1;
        other.m_num_children_ = 0;
        return *this;
    }

    std::string
    AbstractOctreeNode::GetNodeType() const {
        return demangle(typeid(*this).name());
    }

    std::unique_ptr<AbstractOctreeNode>
    AbstractOctreeNode::CreateNode(
        const std::string &node_type,
        const uint32_t depth,
        const int child_index) {
        return Factory::GetInstance().Create(node_type, depth, child_index);
    }

    std::ostream &
    AbstractOctreeNode::Print(std::ostream &os) const {
        os                                         //
            << "NodeType: " << GetNodeType()       //
            << ", Depth: " << m_depth_             //
            << ", ChildIndex: " << m_child_index_  //
            << ", NumChildren: " << m_num_children_;
        return os;
    }

    bool
    AbstractOctreeNode::operator==(const AbstractOctreeNode &other) const {
        // we don't do polymorphic check because it is expensive to do so here.
        // The tree should do polymorphic check: if two trees are the same type, their nodes
        // should be the same type. Unless we hack it by assigning nodes of a wrong type to the
        // tree, which is not supposed to happen.
        if (m_depth_ != other.m_depth_ || m_child_index_ != other.m_child_index_ ||
            m_num_children_ != other.m_num_children_) {
            return false;
        }
        if (m_num_children_ == 0) { return true; }
        for (int i = 0; i < 8; ++i) {
            if (m_children_[i] == nullptr && other.m_children_[i] == nullptr) { continue; }
            if (m_children_[i] == nullptr || other.m_children_[i] == nullptr) { return false; }
            if (*m_children_[i] != *other.m_children_[i]) { return false; }
        }
        return true;
    }

    bool
    AbstractOctreeNode::HasChild(const uint32_t index) const {
        ERL_DEBUG_ASSERT_LT(index, 8);
        return m_children_[index] != nullptr;
    }

    AbstractOctreeNode *
    AbstractOctreeNode::CreateChild(const uint32_t child_index) {
        ERL_DEBUG_ASSERT_LT(child_index, 8);
        ERL_DEBUG_ASSERT_NULL(m_children_[child_index]);
        m_children_[child_index] = this->Create(m_depth_ + 1, static_cast<int>(child_index));
        ++m_num_children_;
        return m_children_[child_index].get();
    }

    void
    AbstractOctreeNode::RemoveChild(const uint32_t child_index) {
        ERL_DEBUG_ASSERT_LT(child_index, 8);
        ERL_DEBUG_ASSERT_PTR(m_children_[child_index]);
        m_children_[child_index] = nullptr;
        --m_num_children_;
    }

    void
    AbstractOctreeNode::Prune() {
        ERL_DEBUG_ASSERT(
            m_num_children_ == 8,
            "Prune() can only be called when all children are present.");
        for (int i = 0; i < 8; ++i) { m_children_[i] = nullptr; }
        m_num_children_ = 0;
    }

    void
    AbstractOctreeNode::Expand() {
        ERL_DEBUG_ASSERT(
            m_num_children_ == 0,
            "Expand() can only be called when no children are present.");
        for (int i = 0; i < 8; ++i) { m_children_[i] = this->Create(m_depth_ + 1, i); }
        m_num_children_ = 8;
    }

}  // namespace erl::geometry
