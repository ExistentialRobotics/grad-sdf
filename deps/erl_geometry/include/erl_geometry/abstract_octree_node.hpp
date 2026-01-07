#pragma once

#include "erl_common/factory_pattern.hpp"
#include "erl_common/logging.hpp"

#include <array>
#include <memory>
#include <string>

namespace erl::geometry {
    /**
     * AbstractOctreeNode is a base class for all octree node implementations. It provides a common
     * interface for file I/O and children management.
     */
    class AbstractOctreeNode {
    protected:
        uint32_t m_depth_ = 0;
        int m_child_index_ = -1;
        uint32_t m_num_children_ = 0;

        // array of 8 children pointers. we use unique_ptr for automatic memory management without
        // worrying about memory leaks and overhead of shared_ptr.
        std::array<std::unique_ptr<AbstractOctreeNode>, 8> m_children_ =
            {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    public:
        using Factory = common::FactoryPattern<AbstractOctreeNode, true, false, uint32_t, int>;

        // rules of five: https://www.youtube.com/watch?v=juAZDfsaMvY
        // except for user-defined constructor,
        // always define: destructor, copy constructor, copy assignment, move constructor, move
        // assignment

        AbstractOctreeNode() = delete;

        explicit AbstractOctreeNode(uint32_t depth, int child_index = -1);

        /**
         * Copy constructor, deep copy. If you want to do a shallow copy, please wrap it in a smart
         * pointer. AbstractOctreeNode uses raw pointers internally and is responsible for memory
         * management. So, shallow copy is impossible, which will lead to double free.
         * @param other
         */
        AbstractOctreeNode(const AbstractOctreeNode &other);

        // copy assignment
        AbstractOctreeNode &
        operator=(const AbstractOctreeNode &other);

        // move constructor
        AbstractOctreeNode(AbstractOctreeNode &&other) noexcept;

        // move assignment
        AbstractOctreeNode &
        operator=(AbstractOctreeNode &&other) noexcept;

        // destructor
        virtual ~AbstractOctreeNode() = default;

        //-- factory pattern
        [[nodiscard]] std::string
        GetNodeType() const;

        /**
         * Implemented by derived classes to create a new node of the same type.
         * @return a new node of the same type.
         */
        [[nodiscard]] virtual std::unique_ptr<AbstractOctreeNode>
        Create(uint32_t depth, int child_index) const = 0;

        static std::unique_ptr<AbstractOctreeNode>
        CreateNode(const std::string &node_type, uint32_t depth, int child_index);

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractOctreeNode, Derived>, bool>
        Register(std::string node_type = "") {
            return Factory::GetInstance().Register<Derived>(
                node_type,
                [](uint32_t depth, int child_index) {
                    return std::make_unique<Derived>(depth, child_index);
                });
        }

        /**
         * Deep copy of the node. Used for copy constructor and copy assignment.
         * @return deep copy of the node.
         */
        [[nodiscard]] virtual std::unique_ptr<AbstractOctreeNode>
        Clone() const = 0;

        //-- attributes

        [[nodiscard]] const uint32_t &
        GetDepth() const {
            return m_depth_;
        }

        [[nodiscard]] const int &
        GetChildIndex() const {
            return m_child_index_;
        }

        //-- file IO
        virtual std::istream &
        ReadData(std::istream &s) = 0;

        virtual std::ostream &
        WriteData(std::ostream &s) const = 0;

        virtual std::ostream &
        Print(std::ostream &os) const;

        //-- comparison

        [[nodiscard]] virtual bool
        operator==(const AbstractOctreeNode &other) const;

        bool
        operator!=(const AbstractOctreeNode &other) const {  // NOLINT(*-no-recursion)
            return !(*this == other);
        }

        //-- children

        [[nodiscard]] uint32_t
        GetNumChildren() const {
            return m_num_children_;
        }

        [[nodiscard]] bool
        HasAnyChild() const {
            return m_num_children_ > 0;
        }

        [[nodiscard]] bool
        HasChild(uint32_t index) const;

        [[nodiscard]] AbstractOctreeNode *
        CreateChild(uint32_t child_index);

        void
        RemoveChild(uint32_t child_index);

        template<typename Derived>
        Derived *
        GetChild(const uint32_t child_index) {
            ERL_DEBUG_ASSERT(
                child_index < 8,
                "Child index must be in [0, 7], but got {}.",
                child_index);
            return static_cast<Derived *>(m_children_[child_index].get());
        }

        template<typename Derived>
        [[nodiscard]] const Derived *
        GetChild(const uint32_t child_index) const {
            ERL_DEBUG_ASSERT(
                child_index < 8,
                "Child index must be in [0, 7], but got {}.",
                child_index);
            return static_cast<const Derived *>(m_children_[child_index].get());
        }

        [[nodiscard]] virtual bool
        AllowMerge(const AbstractOctreeNode *other) const {
            return m_num_children_ == 0 && other->m_num_children_ == 0;
        }

        virtual void
        Prune();

        virtual void
        Expand();
    };
}  // namespace erl::geometry
