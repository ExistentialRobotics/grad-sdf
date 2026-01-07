#pragma once

#include "aabb.hpp"
#include "abstract_quadtree.hpp"
#include "quadtree_key.hpp"

#include <stack>

namespace erl::geometry {

    /**
     * QuadtreeImpl is a template class that implements generic quadtree functionality. The tree is
     * centered at the origin.
     * @tparam Node
     * @tparam Interface
     */
    template<class Node, class Interface, class InterfaceSetting>
    class QuadtreeImpl : public Interface {
        using Dtype = typename Interface::DataType;
        static_assert(
            std::is_base_of_v<AbstractQuadtreeNode, Node>,
            "Node must be derived from AbstractQuadtreeNode");
        static_assert(
            std::is_base_of_v<AbstractQuadtree<Dtype>, Interface>,
            "Interface must be derived from AbstractQuadtree");

        std::shared_ptr<InterfaceSetting> m_setting_ = nullptr;

    protected:
        // root node of the quadtree, nullptr if the quadtree is empty
        std::shared_ptr<Node> m_root_ = nullptr;
        // inverse of the resolution
        Dtype m_resolution_inv_ = 0;
        // offset of the key, = 1 << (tree_depth - 1)
        uint32_t m_tree_key_offset_ = 0;
        // number of nodes in the tree
        std::size_t m_tree_size_ = 0;
        // flag indicating if the metric size of the tree has changed
        bool m_size_changed_ = false;
        // max metric coordinate of x and y
        Dtype m_metric_max_[2] = {
            std::numeric_limits<Dtype>::lowest(),
            std::numeric_limits<Dtype>::lowest()};
        // min metric coordinate of x and y
        Dtype m_metric_min_[2] = {
            std::numeric_limits<Dtype>::max(),
            std::numeric_limits<Dtype>::max()};
        // the size of a quadrant at depth i (0: root node, tree_depth: smallest leaf node)
        std::vector<Dtype> m_size_lookup_table_;
        // data structure for parallel ray casting
        std::vector<QuadtreeKeyRay> m_key_rays_;
        std::vector<QuadtreeKeyLongMap> m_key_long_maps_;
        std::vector<QuadtreeKeyVector> m_key_vectors_;

        // taking one dimension, tree_depth = 3 as an example:
        // depth 0 (level 3): root, key = `(           4          ) = 1 << (tree_depth - 1)`
        // depth 1 (level 2): 2 children, `(     2,          6    )`
        // depth 2 (level 1): 4 children, `(  1,    3,    5,    7 )`
        // depth 3 (level 0): 8 children, `(0, 1, 2, 3, 4, 5, 6, 7)`

    public:
        using NodeType = Node;
        using KeyType = QuadtreeKey;
        using Vector2 = Eigen::Vector2<Dtype>;

        QuadtreeImpl() = delete;  // no default constructor

        explicit QuadtreeImpl(std::shared_ptr<InterfaceSetting> setting);

        QuadtreeImpl(const QuadtreeImpl &other) = default;

        QuadtreeImpl &
        operator=(const QuadtreeImpl &other) = default;

        QuadtreeImpl(QuadtreeImpl &&other) noexcept = default;

        QuadtreeImpl &
        operator=(QuadtreeImpl &&other) noexcept = default;

        /**
         * deep copy the octree
         * @return cloned octree
         */
        [[nodiscard]] virtual std::shared_ptr<AbstractQuadtree<Dtype>>
        Clone() const;

        //-- comparison operators
        [[nodiscard]] bool
        operator==(const AbstractQuadtree<Dtype> &other) const override;

        //-- get tree info

        [[nodiscard]] std::size_t
        GetSize() const override;

        [[nodiscard]] Vector2
        GetTreeCenter() const;

        [[nodiscard]] QuadtreeKey
        GetTreeCenterKey() const;

        [[nodiscard]] Vector2
        GetTreeMaxHalfSize() const;

        void
        ApplySetting() override;

    protected:
        void
        ApplySettingToQuadtreeImpl();

    public:
        using Interface::GetMetricMin;

        void
        GetMetricMin(Dtype &min_x, Dtype &min_y) override;

        void
        GetMetricMin(Dtype &min_x, Dtype &min_y) const override;

        using Interface::GetMetricMax;

        void
        GetMetricMax(Dtype &max_x, Dtype &max_y) override;

        void
        GetMetricMax(Dtype &max_x, Dtype &max_y) const override;

        using Interface::GetMetricMinMax;

        void
        GetMetricMinMax(Dtype &min_x, Dtype &min_y, Dtype &max_x, Dtype &max_y) override;

        void
        GetMetricMinMax(Dtype &min_x, Dtype &min_y, Dtype &max_x, Dtype &max_y) const override;

        using Interface::GetMetricSize;

        void
        GetMetricSize(Dtype &x, Dtype &y) override;

        void
        GetMetricSize(Dtype &x, Dtype &y) const override;

        [[nodiscard]] Dtype
        GetNodeSize(uint32_t depth) const override;

        [[nodiscard]] Aabb<Dtype, 2>
        GetNodeAabb(const QuadtreeKey &key, uint32_t depth) const;

        [[nodiscard]] std::size_t
        ComputeNumberOfLeafNodes() const;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const override;

        [[nodiscard]] std::size_t
        GetMemoryUsagePerNode() const override;

        [[nodiscard]] std::size_t
        ComputeNumberOfNodes() const;

        //-- key / coordinate operations
        // ref:
        // https://www.tandfonline.com/doi/abs/10.1080/10867651.2002.10487560?journalCode=ujgt19
        /**
         * Convert 1-dim coordinate to key at depth N.
         * @param coordinate
         * @return
         */
        [[nodiscard]] QuadtreeKey::KeyType
        CoordToKey(Dtype coordinate) const;

        /**
         * Convert 1-dim coordinate to key at a given depth.
         * @param coordinate
         * @param depth
         * @return
         */
        [[nodiscard]] QuadtreeKey::KeyType
        CoordToKey(Dtype coordinate, uint32_t depth) const;

        [[nodiscard]] QuadtreeKey
        CoordToKey(const Eigen::Ref<const Vector2> &coord) const;

        [[nodiscard]] QuadtreeKey
        CoordToKey(const Eigen::Ref<const Vector2> &coord, uint32_t depth) const;

        /**
         * Convert 2-dim coordinate to key at depth N.
         * @param x
         * @param y
         * @return
         */
        [[nodiscard]] QuadtreeKey
        CoordToKey(Dtype x, Dtype y) const;

        /**
         * Convert 2-dim coordinate to key at a given depth.
         * @param x
         * @param y
         * @param depth
         * @return
         */
        [[nodiscard]] QuadtreeKey
        CoordToKey(Dtype x, Dtype y, uint32_t depth) const;

        /**
         * Convert 1-dim coordinate to a key at depth N with the boundary check.
         * @param coordinate
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype coordinate, QuadtreeKey::KeyType &key) const;

        /**
         * Convert 1-dim coordinate to a key at a given depth with the boundary check.
         * @param coordinate
         * @param depth
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype coordinate, uint32_t depth, QuadtreeKey::KeyType &key) const;

        [[nodiscard]] bool
        CoordToKeyChecked(const Eigen::Ref<const Vector2> &coord, QuadtreeKey &key) const;

        [[nodiscard]] bool
        CoordToKeyChecked(const Eigen::Ref<const Vector2> &coord, uint32_t depth, QuadtreeKey &key)
            const;

        /**
         * Convert 2-dim coordinate to a key at depth N with the boundary check.
         * @param x
         * @param y
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype x, Dtype y, QuadtreeKey &key) const;

        /**
         * Convert 2-dim coordinate to a key at a given depth with the boundary check.
         * @param x
         * @param y
         * @param depth
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype x, Dtype y, uint32_t depth, QuadtreeKey &key) const;

        /**
         * Adjust a 1-dim key from the lowest level (max depth) to a given depth.
         * @param key the key at the lowest level
         * @param depth the target depth
         * @return
         */
        [[nodiscard]] QuadtreeKey::KeyType
        AdjustKeyToDepth(QuadtreeKey::KeyType key, uint32_t depth) const;

        /**
         * Adjust a 2-dim key from the lowest level (max depth) to a given depth.
         * @param key the key at the lowest level
         * @param depth the target depth
         * @return
         */
        [[nodiscard]] QuadtreeKey
        AdjustKeyToDepth(const QuadtreeKey &key, uint32_t depth) const;

        void
        ComputeCommonAncestorKey(
            const QuadtreeKey &key1,
            const QuadtreeKey &key2,
            QuadtreeKey &ancestor_key,
            uint32_t &ancestor_depth) const;

        /**
         * Compute the key of the west(left) neighbor.
         * @param key
         * @param depth 0 means root
         * @param neighbor_key
         */
        bool
        ComputeWestNeighborKey(const QuadtreeKey &key, uint32_t depth, QuadtreeKey &neighbor_key)
            const;

        bool
        ComputeEastNeighborKey(const QuadtreeKey &key, uint32_t depth, QuadtreeKey &neighbor_key)
            const;

        bool
        ComputeNorthNeighborKey(const QuadtreeKey &key, uint32_t depth, QuadtreeKey &neighbor_key)
            const;

        bool
        ComputeSouthNeighborKey(const QuadtreeKey &key, uint32_t depth, QuadtreeKey &neighbor_key)
            const;

        /**
         * Convert 1-dim key to vertex coordinate.
         * @param key
         * @return
         */
        [[nodiscard]] Dtype
        KeyToVertexCoord(QuadtreeKey::KeyType key) const;

        /**
         * Convert a 2-dim key to vertex coordinate at a given depth.
         * @param key
         * @param x
         * @param y
         */
        void
        KeyToVertexCoord(const QuadtreeKey &key, Dtype &x, Dtype &y) const;

        void
        KeyToVertexCoord(const QuadtreeKey &key, Vector2 &vertex_coord) const;

        [[nodiscard]] Vector2
        KeyToVertexCoord(const QuadtreeKey &key) const;

        /**
         * Convert 1-dim key to coordinate.
         * @param key
         * @return
         */
        [[nodiscard]] Dtype
        KeyToCoord(QuadtreeKey::KeyType key) const;

        /**
         * Convert a 1-dim key to coordinate at a given depth.
         * @param key
         * @param depth
         * @return
         */
        [[nodiscard]] Dtype
        KeyToCoord(QuadtreeKey::KeyType key, uint32_t depth) const;

        /**
         * Convert 2-dim key to coordinate.
         * @param key
         * @param x
         * @param y
         */
        void
        KeyToCoord(const QuadtreeKey &key, Dtype &x, Dtype &y) const;

        /**
         * Convert a 2-dim key to coordinate at a given depth.
         * @param key
         * @param depth
         * @param x
         * @param y
         */
        void
        KeyToCoord(const QuadtreeKey &key, uint32_t depth, Dtype &x, Dtype &y) const;

        void
        KeyToCoord(const QuadtreeKey &key, uint32_t depth, Vector2 &coord) const;

        [[nodiscard]] Vector2
        KeyToCoord(const QuadtreeKey &key, uint32_t depth) const;

        //-- iterator implementation
        class IteratorBase : public AbstractQuadtree<Dtype>::QuadtreeNodeIterator {
        public:
            struct StackElement {
                const Node *node = nullptr;
                QuadtreeKey key = {};
                std::shared_ptr<void> data = nullptr;  // data pointer for derived classes

                StackElement() = default;

                StackElement(const Node *node, QuadtreeKey key);

                template<typename T>
                const T &
                GetData() const;
            };

        protected:
            const QuadtreeImpl *m_tree_;             // the tree this iterator is working on
            uint32_t m_max_node_depth_;              // the maximum depth to query
            std::deque<StackElement> m_stack_ = {};  // stack for depth first traversal

        public:
            /**
             * Default constructor, only used for the end-iterator.
             */
            IteratorBase();

            explicit IteratorBase(const QuadtreeImpl *tree, uint32_t max_node_depth);

            [[nodiscard]] bool
            operator==(const IteratorBase &other) const;

            [[nodiscard]] bool
            operator!=(const IteratorBase &other) const;

            Node *
            operator->();

            [[nodiscard]] const Node *
            operator->() const;

            Node *
            operator*();

            [[nodiscard]] const Node *
            operator*() const;

            [[nodiscard]] Dtype
            GetX() const override;

            [[nodiscard]] Dtype
            GetY() const override;

            [[nodiscard]] Vector2
            GetCenter() const override;

            [[nodiscard]] Dtype
            GetNodeSize() const override;

            [[nodiscard]] uint32_t
            GetDepth() const override;

            [[nodiscard]] bool
            IsValid() const override;

            [[nodiscard]] const Node *
            GetNode() const override;

            [[nodiscard]] const Node *
            GetNode();

            [[nodiscard]] Aabb<Dtype, 2>
            GetNodeAabb() const;

            [[nodiscard]] const QuadtreeKey &
            GetKey() const override;

            [[nodiscard]] QuadtreeKey
            GetIndexKey() const override;

        protected:
            /**
             * One step of depth-first tree traversal.
             */
            void
            SingleIncrement1();

            [[nodiscard]] bool
            IsLeaf() const;

            void
            Terminate();

            /**
             * @brief Get the biggest in-tree axis-aligned bounding box (AABB) that is inside the
             * given AABB.
             *
             * @param aabb_min_x
             * @param aabb_min_y
             * @param aabb_max_x
             * @param aabb_max_y
             * @return true if such an AABB exists, false otherwise.
             */
            bool
            GetInTreeAabb(
                Dtype &aabb_min_x,
                Dtype &aabb_min_y,
                Dtype &aabb_max_x,
                Dtype &aabb_max_y) const;
        };

        class TreeIterator : public IteratorBase {
        public:
            TreeIterator() = default;

            explicit TreeIterator(const QuadtreeImpl *tree, uint32_t max_node_depth = 0);

            // post-increment
            TreeIterator
            operator++(int);

            // pre-increment
            TreeIterator &
            operator++();

            void
            Next() override;
        };

        class InAabbIteratorBase : public IteratorBase {
            QuadtreeKey m_aabb_min_key_;
            QuadtreeKey m_aabb_max_key_;

        public:
            InAabbIteratorBase() = default;

            InAabbIteratorBase(
                Dtype aabb_min_x,
                Dtype aabb_min_y,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                const QuadtreeImpl *tree,
                uint32_t max_node_depth = 0);

            InAabbIteratorBase(
                QuadtreeKey aabb_min_key,
                QuadtreeKey aabb_max_key,
                const QuadtreeImpl *tree,
                uint32_t max_node_depth = 0);

        protected:
            void
            SingleIncrement2();
        };

        class TreeInAabbIterator : public InAabbIteratorBase {
        public:
            TreeInAabbIterator() = default;

            TreeInAabbIterator(
                Dtype aabb_min_x,
                Dtype aabb_min_y,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                const QuadtreeImpl *tree,
                uint32_t max_node_depth = 0);

            TreeInAabbIterator(
                QuadtreeKey aabb_min_key,
                QuadtreeKey aabb_max_key,
                const QuadtreeImpl *tree,
                uint32_t max_node_depth = 0);

            // post-increment
            TreeInAabbIterator
            operator++(int);

            // pre-increment
            TreeInAabbIterator &
            operator++();

            void
            Next() override;
        };

        /**
         * Iterate over all leaf nodes that intersect with the given axis-aligned bounding box.
         */
        class LeafInAabbIterator : public InAabbIteratorBase {
        public:
            LeafInAabbIterator() = default;

            LeafInAabbIterator(
                Dtype aabb_min_x,
                Dtype aabb_min_y,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth = 0);

            LeafInAabbIterator(
                QuadtreeKey aabb_min_key,
                QuadtreeKey aabb_max_key,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth = 0);

            // post-increment
            LeafInAabbIterator
            operator++(int);

            // pre-increment
            LeafInAabbIterator &
            operator++();

            void
            Next() override;
        };

        class LeafIterator : public IteratorBase {
        public:
            LeafIterator() = default;

            explicit LeafIterator(const QuadtreeImpl *tree, uint32_t depth = 0);

            // post-increment
            LeafIterator
            operator++(int);

            // pre-increment
            LeafIterator &
            operator++();

            void
            Next() override;
        };

        class LeafOfNodeIterator : public IteratorBase {
        public:
            LeafOfNodeIterator() = default;

            LeafOfNodeIterator(
                QuadtreeKey key,
                uint32_t cluster_depth,
                const QuadtreeImpl *tree,
                uint32_t max_node_depth = 0);

            // post-increment
            LeafOfNodeIterator
            operator++(int);

            // pre-increment
            LeafOfNodeIterator &
            operator++();

            void
            Next() override;
        };

        class LeafNeighborIteratorBase : public IteratorBase {
        public:
            LeafNeighborIteratorBase() = default;

            LeafNeighborIteratorBase(const QuadtreeImpl *tree, uint32_t max_leaf_depth);

        protected:
            QuadtreeKey m_neighbor_key_ = {};
            QuadtreeKey::KeyType m_max_key_changing_dim_ = 0;

            /**
             *
             * @param key
             * @param key_depth
             * @param changing_dim the dimension that is changing during the search
             * @param unchanged_dim the dimension that is unchanged during the search
             * @param increase
             */
            void
            Init(
                QuadtreeKey key,
                uint32_t key_depth,
                int changing_dim,
                int unchanged_dim,
                bool increase);

            void
            SingleIncrementOf(int changing_dim);
        };

        class WestLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_UnchangedDim_ = 0;
            static constexpr bool sk_Increase_ = false;  // decrease the x-dim key

        public:
            WestLeafNeighborIterator() = default;

            WestLeafNeighborIterator(
                Dtype x,
                Dtype y,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            WestLeafNeighborIterator(
                const QuadtreeKey &key,
                uint32_t key_depth,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            WestLeafNeighborIterator
            operator++(int);

            // pre-increment
            WestLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class EastLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_UnchangedDim_ = 0;
            static constexpr bool sk_Increase_ = true;  // increase the x-dim key

        public:
            EastLeafNeighborIterator() = default;

            EastLeafNeighborIterator(
                Dtype x,
                Dtype y,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            EastLeafNeighborIterator(
                const QuadtreeKey &key,
                uint32_t key_depth,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            EastLeafNeighborIterator
            operator++(int);

            // pre-increment
            EastLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class NorthLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_UnchangedDim_ = 1;
            static constexpr bool sk_Increase_ = true;  // increase the y-dim key

        public:
            NorthLeafNeighborIterator() = default;

            NorthLeafNeighborIterator(
                Dtype x,
                Dtype y,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            NorthLeafNeighborIterator(
                const QuadtreeKey &key,
                uint32_t key_depth,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            NorthLeafNeighborIterator
            operator++(int);

            // pre-increment
            NorthLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class SouthLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_UnchangedDim_ = 1;
            static constexpr bool sk_Increase_ = false;  // decrease the y-dim key

        public:
            SouthLeafNeighborIterator() = default;

            SouthLeafNeighborIterator(
                Dtype x,
                Dtype y,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            SouthLeafNeighborIterator(
                const QuadtreeKey &key,
                uint32_t key_depth,
                const QuadtreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            SouthLeafNeighborIterator
            operator++(int);

            // pre-increment
            SouthLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class NodeOnRayIterator : public IteratorBase {
            Vector2 m_origin_ = {};
            Vector2 m_dir_ = {};
            Vector2 m_dir_inv_ = {};
            Dtype m_max_range_ = 0.0f;
            Dtype m_node_padding_ = 0.0f;
            bool m_bidirectional_ = false;
            bool m_leaf_only_ = false;
            uint32_t m_min_node_depth_ = 0;

        public:
            NodeOnRayIterator() = default;

            /**
             *
             * @param px
             * @param py
             * @param vx
             * @param vy
             * @param max_range
             * @param node_padding
             * @param bidirectional
             * @param tree
             * @param leaf_only
             * @param min_node_depth
             * @param max_node_depth 0 means using the tree's max depth
             */
            NodeOnRayIterator(
                Dtype px,
                Dtype py,
                Dtype vx,
                Dtype vy,
                Dtype max_range,
                Dtype node_padding,
                bool bidirectional,
                const QuadtreeImpl *tree,
                bool leaf_only = false,
                uint32_t min_node_depth = 0,
                uint32_t max_node_depth = 0);

            [[nodiscard]] Dtype
            GetDistance() const;

            // post-increment
            NodeOnRayIterator
            operator++(int);

            // pre-increment
            NodeOnRayIterator &
            operator++();

            void
            Next() override;

        protected:
            struct StackElementCompare {  // for min-heap

                bool
                operator()(
                    const typename IteratorBase::StackElement &lhs,
                    const typename IteratorBase::StackElement &rhs) const;
            };

            void
            SingleIncrement2();
        };

        /**
         * Iterator that iterates over all leaf nodes.
         * @param max_depth
         * @return
         */
        [[nodiscard]] LeafIterator
        BeginLeaf(uint32_t max_depth = 0) const;

        /**
         * End iterator of LeafIterator.
         */
        [[nodiscard]] LeafIterator
        EndLeaf() const;

        [[nodiscard]] LeafOfNodeIterator
        BeginLeafOfNode(QuadtreeKey key, uint32_t node_depth, uint32_t max_depth = 0) const;

        [[nodiscard]] LeafOfNodeIterator
        EndLeafOfNode() const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(const Aabb<Dtype, 2> &aabb, uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(
            Dtype aabb_min_x,
            Dtype aabb_min_y,
            Dtype aabb_max_x,
            Dtype aabb_max_y,
            uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(
            const QuadtreeKey &aabb_min_key,
            const QuadtreeKey &aabb_max_key,
            uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        EndLeafInAabb() const;

        [[nodiscard]] std::shared_ptr<typename AbstractQuadtree<Dtype>::QuadtreeNodeIterator>
        GetLeafInAabbIterator(const Aabb<Dtype, 2> &aabb, uint32_t max_depth) const override;

        [[nodiscard]] TreeIterator
        BeginTree(uint32_t max_depth = 0) const;

        [[nodiscard]] TreeIterator
        EndTree() const;

        [[nodiscard]] std::shared_ptr<typename AbstractQuadtree<Dtype>::QuadtreeNodeIterator>
        GetTreeIterator(uint32_t max_depth) const override;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(const Aabb<Dtype, 2> &aabb, uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(
            Dtype aabb_min_x,
            Dtype aabb_min_y,
            Dtype aabb_max_x,
            Dtype aabb_max_y,
            uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(
            const QuadtreeKey &aabb_min_key,
            const QuadtreeKey &aabb_max_key,
            uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        EndTreeInAabb() const;

        [[nodiscard]] WestLeafNeighborIterator
        BeginWestLeafNeighbor(Dtype x, Dtype y, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] WestLeafNeighborIterator
        BeginWestLeafNeighbor(
            const QuadtreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] WestLeafNeighborIterator
        EndWestLeafNeighbor() const;

        [[nodiscard]] EastLeafNeighborIterator
        BeginEastLeafNeighbor(Dtype x, Dtype y, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] EastLeafNeighborIterator
        BeginEastLeafNeighbor(
            const QuadtreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] EastLeafNeighborIterator
        EndEastLeafNeighbor() const;

        [[nodiscard]] NorthLeafNeighborIterator
        BeginNorthLeafNeighbor(Dtype x, Dtype y, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] NorthLeafNeighborIterator
        BeginNorthLeafNeighbor(
            const QuadtreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] NorthLeafNeighborIterator
        EndNorthLeafNeighbor() const;

        [[nodiscard]] SouthLeafNeighborIterator
        BeginSouthLeafNeighbor(Dtype x, Dtype y, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] SouthLeafNeighborIterator
        BeginSouthLeafNeighbor(
            const QuadtreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] SouthLeafNeighborIterator
        EndSouthLeafNeighbor() const;

        [[nodiscard]] NodeOnRayIterator
        BeginNodeOnRay(
            Dtype px,
            Dtype py,
            Dtype vx,
            Dtype vy,
            Dtype max_range = -1,
            Dtype node_padding = 0.0,
            bool bidirectional = false,
            bool leaf_only = false,
            uint32_t min_node_depth = 0,
            uint32_t max_node_depth = 0) const;

        [[nodiscard]] NodeOnRayIterator
        BeginNodeOnRay(
            const Eigen::Ref<const Vector2> &origin,
            const Eigen::Ref<const Vector2> &direction,
            Dtype max_range = -1,
            Dtype node_padding = 0.0,
            bool bidirectional = false,
            bool leaf_only = false,
            uint32_t min_node_depth = 0,
            uint32_t max_node_depth = 0) const;

        [[nodiscard]] NodeOnRayIterator
        EndNodeOnRay() const;

        //-- ray tracing
        /**
         * Trace a ray from origin to end (excluded), returning a QuadtreeKeyRay of all nodes'
         * QuadtreeKey traversed by the ray. For each key, the corresponding node may not exist. You
         * can use Search() to check if the node exists.
         * @param sx x coordinate of the origin
         * @param sy y coordinate of the origin
         * @param ex x coordinate of the end (excluded)
         * @param ey y coordinate of the end (excluded)
         * @param ray
         * @return
         */
        [[nodiscard]] bool
        ComputeRayKeys(Dtype sx, Dtype sy, Dtype ex, Dtype ey, QuadtreeKeyRay &ray) const;

        /**
         * Trace a ray from origin to end (excluded), returning a list of all nodes' coordinates
         * traversed by the ray. For each coordinate, the corresponding node may not exist. You can
         * use Search() to check if the node exists.
         * @param sx x coordinate of the origin
         * @param sy y coordinate of the origin
         * @param ex x coordinate of the end (excluded)
         * @param ey y coordinate of the end (excluded)
         * @param ray
         * @return
         */
        [[nodiscard]] bool
        ComputeRayCoords(Dtype sx, Dtype sy, Dtype ex, Dtype ey, std::vector<Vector2> &ray) const;

        /**
         * Clear KeyRay vector to minimize unneeded memory. This is only useful for the
         * StaticMemberInitializer classes, don't call it for a quadtree that is actually used.
         */
        void
        ClearKeyRays();

        //-- tree structure operations
        /**
         * Create a new child node for the given node.
         * @param node
         * @param child_idx
         * @return
         */
        Node *
        CreateNodeChild(Node *node, uint32_t child_idx);

        uint32_t
        DeleteNodeChild(Node *node, uint32_t child_idx, const QuadtreeKey &key);

        /**
         * Get a child node of the given node. Before calling this function, make sure
         * node->HasChildrenPtr() or node->HasAnyChild() returns true.
         * @param node
         * @param child_idx
         * @return
         */
        Node *
        GetNodeChild(Node *node, uint32_t child_idx);

        /**
         * Get a child node of the given node. Before calling this function, make sure
         * node->HasChildrenPtr() or node->HasAnyChild() returns true.
         * @param node
         * @param child_idx
         * @return
         */
        const Node *
        GetNodeChild(const Node *node, uint32_t child_idx) const;

        /**
         * Check if a node is collapsible. For example, for occupancy quadtree, a node is
         * collapsible if all its children exist, none of them have its own children, and they all
         * have the same occupancy value.
         * @param node
         * @return
         */
        virtual bool
        IsNodeCollapsible(const Node *node) const;

        /**
         * Expand a node: all children are created, and their data is copied from the parent.
         * @param node
         * @return
         */
        void
        ExpandNode(Node *node);

        /**
         * Prune a node: delete all children if the node is collapsible.
         * @param node
         * @return
         */
        bool
        PruneNode(Node *node);

        /**
         * Delete nodes that contain the given point, at or deeper than the given delete_depth.
         * @param x
         * @param y
         * @param delete_depth delete_depth to start deleting nodes, nodes at delete_depth >=
         * delete_depth will be deleted. If delete_depth == 0, delete at the lowest level.
         * @return
         */
        uint32_t
        DeleteNode(Dtype x, Dtype y, uint32_t delete_depth = 0);

        /**
         * Delete nodes that contain the given key, at or deeper than the given delete_depth.
         * @param key
         * @param delete_depth delete_depth to start deleting nodes, nodes at delete_depth >=
         * delete_depth will be deleted. If delete_depth == 0, delete at the lowest level.
         * @return
         */
        void
        DeleteNode(const QuadtreeKey &key, uint32_t delete_depth = 0);

    protected:
        /**
         * Callback before deleting a child node.
         * @param node parent node, which may be nullptr if the child is the root node
         * @param child child node to be deleted
         * @param key key of the child node
         */
        virtual void
        OnDeleteNodeChild(Node *node, Node *child, const QuadtreeKey &key);

        /**
         * Delete child nodes down to min_depth matching the given key of the given node that is at
         * the given depth.
         * @param node node at depth, which must not be nullptr, and it will not be deleted
         * @param key
         * @param min_depth
         * @return
         */
        bool
        DeleteNodeRecurs(Node *node, const QuadtreeKey &key, uint32_t min_depth);

        /**
         * Delete all descendants of the given node.
         * @param node
         * @param key
         * @return
         */
        void
        DeleteNodeDescendants(Node *node, const QuadtreeKey &key);

    public:
        /**
         * Delete the whole tree.
         * @return
         */
        void
        Clear() override;

        /**
         * Lossless compression of the tree: a node will replace all of its children if the node is
         * collapsible.
         */
        void
        Prune() override;

    protected:
        void
        PruneRecurs(Node *node, uint32_t max_depth);

    public:
        /**
         * Expand all pruned nodes (reverse operation of Prune).
         * @attention This is an expensive operation, especially when the tree is nearly empty!
         */
        virtual void
        Expand();

    protected:
        void
        ExpandRecurs(Node *node, uint32_t depth, uint32_t max_depth);

    public:
        const std::shared_ptr<Node> &
        GetRoot() const;

        //-- Search functions
        [[nodiscard]] const AbstractQuadtreeNode *
        SearchNode(Dtype x, Dtype y, uint32_t max_depth) const override;

        [[nodiscard]] const AbstractQuadtreeNode *
        SearchNode(const QuadtreeKey &key, uint32_t max_depth) const override;

        [[nodiscard]] const Node *
        Search(const Eigen::Ref<const Vector2> &position, uint32_t max_depth = 0) const;

        /**
         * Search node given a point.
         * @param x
         * @param y
         * @param max_depth Max depth to search. However, max_depth=0 means searching from the root.
         * @return
         */
        [[nodiscard]] const Node *
        Search(Dtype x, Dtype y, uint32_t max_depth = 0) const;

        /**
         * Search node at specified max_depth given a key.
         * @param key
         * @param max_depth Max max_depth to search. However, max_depth=0 means searching from the
         * root.
         * @return
         */
        [[nodiscard]] const Node *
        Search(const QuadtreeKey &key, uint32_t max_depth = 0) const;

        /**
         * Search nodes that match the key at different depths.
         * @param key The key to locate the nodes.
         * @param max_depth The deepest node to search. 0 means searching until the deepest.
         * @return A vector of pointers to the nodes that match the key at depths <= max_depth.
         * @note The last node in the vector is the deepest node that matches the key. But this node
         * may be shallower than the requested depth. If the last node is not a leaf node, it means
         * a child node should be created to reach the requested depth.
         */
        [[nodiscard]] std::vector<const Node *>
        SearchNodes(const QuadtreeKey &key, uint32_t max_depth = 0) const;

        Node *
        InsertNode(Dtype x, Dtype y, uint32_t depth = 0);

        Node *
        InsertNode(const QuadtreeKey &key, uint32_t depth = 0);

    protected:
        //-- file IO
        /**
         * Read all nodes from the input stream (without the file header). For general file IO, use
         * AbstractQuadtree::Read.
         */
        std::istream &
        ReadData(std::istream &s) override;

        std::ostream &
        WriteData(std::ostream &s) const override;

    public:
        std::ostream &
        Print(std::ostream &os) const override;

    protected:
        /**
         * Get the minimum and maximum coordinates of the octree. This is an expensive operation, so
         * it is only called when necessary.
         */
        void
        ComputeMinMax();
    };
}  // namespace erl::geometry

#include "quadtree_impl.tpp"
