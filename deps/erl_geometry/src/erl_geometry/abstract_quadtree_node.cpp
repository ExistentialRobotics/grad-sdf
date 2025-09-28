#include "erl_geometry/abstract_quadtree_node.hpp"

namespace erl::geometry {

    AbstractQuadtreeNode::AbstractQuadtreeNode(const uint32_t depth, const int child_index)
        : m_depth_(depth),
          m_child_index_(child_index) {}

    AbstractQuadtreeNode::AbstractQuadtreeNode(const AbstractQuadtreeNode &other)
        : m_depth_(other.m_depth_),
          m_child_index_(other.m_child_index_),
          m_num_children_(other.m_num_children_) {
        if (other.m_children_ == nullptr) { return; }
        this->AllocateChildrenPtr();
        ERL_ASSERTM(m_children_ != nullptr, "Failed to allocate memory.");
        for (int i = 0; i < 4; ++i) {
            const AbstractQuadtreeNode *child = other.m_children_[i];
            if (child == nullptr) { continue; }
            m_children_[i] = child->Clone();
        }
    }

    AbstractQuadtreeNode &
    AbstractQuadtreeNode::operator=(const AbstractQuadtreeNode &other) {
        if (this == &other) { return *this; }
        m_depth_ = other.m_depth_;
        m_child_index_ = other.m_child_index_;
        m_num_children_ = other.m_num_children_;
        if (other.m_children_ == nullptr) {
            this->DeleteChildrenPtr();
            return *this;
        }
        this->AllocateChildrenPtr();
        ERL_ASSERTM(m_children_ != nullptr, "Failed to allocate memory.");
        for (int i = 0; i < 4; ++i) {
            const AbstractQuadtreeNode *child = other.m_children_[i];
            if (child == nullptr) {
                m_children_[i] = nullptr;
            } else {
                m_children_[i] = child->Clone();
            }
        }
        return *this;
    }

    AbstractQuadtreeNode::AbstractQuadtreeNode(AbstractQuadtreeNode &&other) noexcept
        : m_depth_(other.m_depth_),
          m_child_index_(other.m_child_index_),
          m_children_(other.m_children_),
          m_num_children_(other.m_num_children_) {
        other.m_depth_ = 0;
        other.m_child_index_ = -1;
        other.m_children_ = nullptr;
        other.m_num_children_ = 0;
    }

    AbstractQuadtreeNode &
    AbstractQuadtreeNode::operator=(AbstractQuadtreeNode &&other) noexcept {
        if (this == &other) { return *this; }
        m_depth_ = other.m_depth_;
        m_child_index_ = other.m_child_index_;
        m_children_ = other.m_children_;
        m_num_children_ = other.m_num_children_;
        other.m_depth_ = 0;
        other.m_child_index_ = -1;
        other.m_children_ = nullptr;
        other.m_num_children_ = 0;
        return *this;
    }

    std::string
    AbstractQuadtreeNode::GetNodeType() const {
        return demangle(typeid(*this).name());
    }

    std::shared_ptr<AbstractQuadtreeNode>
    AbstractQuadtreeNode::CreateNode(
        const std::string &node_type,
        const uint32_t depth,
        const int child_index) {
        return Factory::GetInstance().Create(node_type, depth, child_index);
    }

    bool
    AbstractQuadtreeNode::operator==(const AbstractQuadtreeNode &other) const {
        // We don't do polymorphic check because it is expensive to do so here.
        // The tree should do polymorphic check: if two trees are the same type, their nodes should
        // be the same type. Unless we hack it by assigning nodes of a wrong type to the tree, which
        // is not supposed to happen.
        if (m_depth_ != other.m_depth_ || m_child_index_ != other.m_child_index_ ||
            m_num_children_ != other.m_num_children_) {
            return false;
        }
        if (m_num_children_ == 0) { return true; }
        for (int i = 0; i < 4; ++i) {
            if (m_children_[i] == nullptr && other.m_children_[i] == nullptr) { continue; }
            if (m_children_[i] == nullptr || other.m_children_[i] == nullptr) { return false; }
            if (*m_children_[i] != *other.m_children_[i]) { return false; }
        }
        return true;
    }

    void
    AbstractQuadtreeNode::AllocateChildrenPtr() {
        if (m_children_ != nullptr) { return; }
        m_children_ = new AbstractQuadtreeNode *[4];
        for (int i = 0; i < 4; ++i) { m_children_[i] = nullptr; }
    }

    void
    AbstractQuadtreeNode::DeleteChildrenPtr() {
        if (m_children_ == nullptr) { return; }
        if (m_num_children_ > 0) {
            for (int i = 0; i < 4; ++i) {
                if (m_children_[i] != nullptr) { delete m_children_[i]; }
            }
            m_num_children_ = 0;
        }
        delete[] m_children_;
        m_children_ = nullptr;
    }

    bool
    AbstractQuadtreeNode::HasChild(const uint32_t index) const {
        if (m_children_ == nullptr) { return false; }
        ERL_DEBUG_ASSERT(index < 4, "Index must be in [0, 3], but got %u.", index);
        return m_children_[index] != nullptr;
    }

    AbstractQuadtreeNode *
    AbstractQuadtreeNode::CreateChild(const uint32_t child_index) {
        ERL_DEBUG_ASSERT(
            child_index < 4,
            "Child index must be in [0, 3], but got %u.",
            child_index);
        ERL_DEBUG_ASSERT(
            m_children_[child_index] == nullptr,
            "Child %u already exists.",
            child_index);
        AbstractQuadtreeNode *child = this->Create(m_depth_ + 1, static_cast<int>(child_index));
        m_children_[child_index] = child;
        m_num_children_++;
        return child;
    }

    void
    AbstractQuadtreeNode::RemoveChild(const uint32_t child_index) {
        ERL_DEBUG_ASSERT(
            child_index < 4,
            "Child index must be in [0, 3], but got %u.",
            child_index);
        ERL_DEBUG_ASSERT(
            m_children_[child_index] != nullptr,
            "Child %u does not exist.",
            child_index);
        delete m_children_[child_index];
        m_children_[child_index] = nullptr;
        m_num_children_--;
    }

    void
    AbstractQuadtreeNode::Prune() {
        ERL_DEBUG_ASSERT(
            m_num_children_ == 4,
            "Prune() can only be called when all children are present.");
        for (int i = 0; i < 4; ++i) {
            delete m_children_[i];
            m_children_[i] = nullptr;
        }
        m_num_children_ = 0;
    }

    void
    AbstractQuadtreeNode::Expand() {
        ERL_DEBUG_ASSERT(
            m_num_children_ == 0,
            "Expand() can only be called when no children are present.");
        if (m_children_ == nullptr) { m_children_ = new AbstractQuadtreeNode *[4]; }
        for (int i = 0; i < 4; ++i) { m_children_[i] = this->Create(m_depth_ + 1, i); }
        m_num_children_ = 4;
    }

}  // namespace erl::geometry
