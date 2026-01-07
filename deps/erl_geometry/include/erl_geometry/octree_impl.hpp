#pragma once

#include "aabb.hpp"
#include "abstract_octree.hpp"
#include "octree_key.hpp"

#include <deque>

namespace erl::geometry {

    /**
     * OctreeImpl is a template class that implements generic octree functionality. The tree is
     * centered at the origin.
     * @tparam Node
     * @tparam Interface
     */
    template<class Node, class Interface, class InterfaceSetting>
    class OctreeImpl : public Interface {
        using Dtype = typename Interface::DataType;
        static_assert(
            std::is_base_of_v<AbstractOctreeNode, Node>,
            "Node must be derived from AbstractOctreeNode");
        static_assert(
            std::is_base_of_v<AbstractOctree<Dtype>, Interface>,
            "Interface must be derived from AbstractOctree");

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
        // max metric coordinate of x, y and z
        Dtype m_metric_max_[3] = {
            std::numeric_limits<Dtype>::lowest(),
            std::numeric_limits<Dtype>::lowest(),
            std::numeric_limits<Dtype>::lowest()};
        // min metric coordinate of x, y and z
        Dtype m_metric_min_[3] = {
            std::numeric_limits<Dtype>::max(),
            std::numeric_limits<Dtype>::max(),
            std::numeric_limits<Dtype>::max()};
        // the size of a quadrant at depth i (0: root node, tree_depth: smallest leaf node)
        std::vector<Dtype> m_size_lookup_table_;
        // data structure for parallel ray casting
        std::vector<OctreeKeyRay> m_key_rays_;
        std::vector<OctreeKeyLongMap> m_key_long_maps_;
        std::vector<OctreeKeyVector> m_key_vectors_;

    public:
        using NodeType = Node;
        using KeyType = OctreeKey;
        using Vector3 = Eigen::Vector3<Dtype>;

        OctreeImpl() = delete;  // no default constructor

        explicit OctreeImpl(std::shared_ptr<InterfaceSetting> setting);

        OctreeImpl(const OctreeImpl &other) = default;

        OctreeImpl &
        operator=(const OctreeImpl &other) = default;

        OctreeImpl(OctreeImpl &&other) noexcept = default;

        OctreeImpl &
        operator=(OctreeImpl &&other) noexcept = default;

        /**
         * deep copy the octree
         * @return cloned octree
         */
        [[nodiscard]] virtual std::shared_ptr<AbstractOctree<Dtype>>
        Clone() const;

        //-- comparison operators
        [[nodiscard]] bool
        operator==(const AbstractOctree<Dtype> &other) const override;

        //-- get tree info

        [[nodiscard]] std::size_t
        GetSize() const override;

        [[nodiscard]] Vector3
        GetTreeCenter() const;

        [[nodiscard]] OctreeKey
        GetTreeCenterKey() const;

        [[nodiscard]] Vector3
        GetTreeMaxHalfSize() const;

        void
        ApplySetting() override;

    protected:
        void
        ApplySettingToOctreeImpl();

    public:
        using Interface::GetMetricMin;

        void
        GetMetricMin(Dtype &min_x, Dtype &min_y, Dtype &min_z) override;

        void
        GetMetricMin(Dtype &min_x, Dtype &min_y, Dtype &min_z) const override;

        using Interface::GetMetricMax;

        void
        GetMetricMax(Dtype &max_x, Dtype &max_y, Dtype &max_z) override;

        void
        GetMetricMax(Dtype &max_x, Dtype &max_y, Dtype &max_z) const override;

        using Interface::GetMetricMinMax;

        void
        GetMetricMinMax(
            Dtype &min_x,
            Dtype &min_y,
            Dtype &min_z,
            Dtype &max_x,
            Dtype &max_y,
            Dtype &max_z) override;

        void
        GetMetricMinMax(
            Dtype &min_x,
            Dtype &min_y,
            Dtype &min_z,
            Dtype &max_x,
            Dtype &max_y,
            Dtype &max_z) const override;

        using Interface::GetMetricSize;

        void
        GetMetricSize(Dtype &x, Dtype &y, Dtype &z) override;

        void
        GetMetricSize(Dtype &x, Dtype &y, Dtype &z) const override;

        [[nodiscard]] Dtype
        GetNodeSize(uint32_t depth) const override;

        [[nodiscard]] Aabb<Dtype, 3>
        GetNodeAabb(const OctreeKey &key, uint32_t depth) const;

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
        [[nodiscard]] OctreeKey::KeyType
        CoordToKey(Dtype coordinate) const;

        /**
         * Convert 1-dim coordinate to key at a given depth.
         * @param coordinate
         * @param depth
         * @return
         */
        [[nodiscard]] OctreeKey::KeyType
        CoordToKey(Dtype coordinate, uint32_t depth) const;

        [[nodiscard]] OctreeKey
        CoordToKey(const Eigen::Ref<const Vector3> &coord) const;

        [[nodiscard]] OctreeKey
        CoordToKey(const Eigen::Ref<const Vector3> &coord, uint32_t depth) const;

        /**
         * Convert 3-dim coordinate to key at depth N.
         * @param x
         * @param y
         * @param z
         * @return
         */
        [[nodiscard]] OctreeKey
        CoordToKey(Dtype x, Dtype y, Dtype z) const;

        /**
         * Convert 3-dim coordinate to key at a given depth.
         * @param x
         * @param y
         * @param z
         * @param depth
         * @return
         */
        [[nodiscard]] OctreeKey
        CoordToKey(Dtype x, Dtype y, Dtype z, uint32_t depth) const;

        /**
         * Convert 1-dim coordinate to a key at depth N with the boundary check.
         * @param coordinate
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype coordinate, OctreeKey::KeyType &key) const;

        /**
         * Convert 1-dim coordinate to a key at a given depth with the boundary check.
         * @param coordinate
         * @param depth
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype coordinate, uint32_t depth, OctreeKey::KeyType &key) const;

        [[nodiscard]] bool
        CoordToKeyChecked(const Eigen::Ref<const Vector3> &coord, OctreeKey &key) const;

        [[nodiscard]] bool
        CoordToKeyChecked(const Eigen::Ref<const Vector3> &coord, uint32_t depth, OctreeKey &key)
            const;

        /**
         * Convert 3-dim coordinate to a key at depth N with the boundary check.
         * @param x
         * @param y
         * @param z
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype x, Dtype y, Dtype z, OctreeKey &key) const;

        /**
         * Convert 3-dim coordinate to a key at a given depth with the boundary check.
         * @param x
         * @param y
         * @param z
         * @param depth
         * @param key
         * @return
         */
        [[nodiscard]] bool
        CoordToKeyChecked(Dtype x, Dtype y, Dtype z, uint32_t depth, OctreeKey &key) const;

        /**
         * Adjust the 1-dim key from the lowest level (max depth) to a given depth.
         * @param key the key at the lowest level
         * @param depth the target depth
         * @return
         */
        [[nodiscard]] OctreeKey::KeyType
        AdjustKeyToDepth(OctreeKey::KeyType key, uint32_t depth) const;

        /**
         * Adjust the 3-dim key from the lowest level (max depth) to a given depth.
         * @param key the key at the lowest level
         * @param depth the target depth
         * @return
         */
        [[nodiscard]] OctreeKey
        AdjustKeyToDepth(const OctreeKey &key, uint32_t depth) const;

        void
        ComputeCommonAncestorKey(
            const OctreeKey &key1,
            const OctreeKey &key2,
            OctreeKey &ancestor_key,
            uint32_t &ancestor_depth) const;

        /**
         * Compute the key of the west (left) neighbor.
         * @param key
         * @param depth 0 means root
         * @param neighbor_key
         */
        bool
        ComputeWestNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key) const;

        bool
        ComputeEastNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key) const;

        bool
        ComputeNorthNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key)
            const;

        bool
        ComputeSouthNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key)
            const;

        bool
        ComputeBottomNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key)
            const;

        bool
        ComputeTopNeighborKey(const OctreeKey &key, uint32_t depth, OctreeKey &neighbor_key) const;

        /**
         * Convert 1-dim key to vertex coordinate.
         * @param key
         * @return
         */
        [[nodiscard]] Dtype
        KeyToVertexCoord(OctreeKey::KeyType key) const;

        /**
         * Convert 3-dim key to vertex coordinate.
         * @param key
         * @param x
         * @param y
         * @param z
         */
        void
        KeyToVertexCoord(const OctreeKey &key, Dtype &x, Dtype &y, Dtype &z) const;

        void
        KeyToVertexCoord(const OctreeKey &key, Vector3 &vertex_coord) const;

        [[nodiscard]] Vector3
        KeyToVertexCoord(const OctreeKey &key) const;

        /**
         * Convert 1-dim key to coordinate.
         * @param key
         * @return
         */
        [[nodiscard]] Dtype
        KeyToCoord(OctreeKey::KeyType key) const;

        /**
         * Convert a 1-dim key to coordinate at a given depth.
         * @param key
         * @param depth
         * @return
         */
        [[nodiscard]] Dtype
        KeyToCoord(OctreeKey::KeyType key, uint32_t depth) const;

        /**
         * Convert 3-dim key to coordinate.
         * @param key
         * @param x
         * @param y
         * @param z
         */
        void
        KeyToCoord(const OctreeKey &key, Dtype &x, Dtype &y, Dtype &z) const;

        /**
         * Convert a 3-dim key to coordinate at a given depth.
         * @param key
         * @param depth
         * @param x
         * @param y
         * @param z
         */
        void
        KeyToCoord(const OctreeKey &key, uint32_t depth, Dtype &x, Dtype &y, Dtype &z) const;

        void
        KeyToCoord(const OctreeKey &key, uint32_t depth, Vector3 &coord) const;

        [[nodiscard]] Vector3
        KeyToCoord(const OctreeKey &key, uint32_t depth) const;

        //-- iterator implementation
        class IteratorBase : public AbstractOctree<Dtype>::OctreeNodeIterator {
        public:
            struct StackElement {
                const Node *node = nullptr;
                OctreeKey key = {};
                std::shared_ptr<void> data = nullptr;  // data pointer for derived classes

                StackElement() = default;

                StackElement(const Node *node, OctreeKey key);

                template<typename T>
                const T &
                GetData() const;
            };

        protected:
            const OctreeImpl *m_tree_;               // the tree this iterator is working on
            uint32_t m_max_node_depth_;              // the maximum depth to query
            std::deque<StackElement> m_stack_ = {};  // stack for depth first traversal

        public:
            /**
             * Default constructor, only used for the end-iterator.
             */
            IteratorBase();

            explicit IteratorBase(const OctreeImpl *tree, uint32_t max_node_depth);

            [[nodiscard]] bool
            operator==(const IteratorBase &other) const;

            [[nodiscard]] bool
            operator!=(const IteratorBase &other) const {
                return !operator==(other);
            }

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

            [[nodiscard]] Dtype
            GetZ() const override;

            [[nodiscard]] Vector3
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

            [[nodiscard]] Aabb<Dtype, 3>
            GetNodeAabb() const;

            [[nodiscard]] const OctreeKey &
            GetKey() const override;

            [[nodiscard]] OctreeKey
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
             * @param aabb_min_z
             * @param aabb_max_x
             * @param aabb_max_y
             * @param aabb_max_z
             * @return true if such an AABB exists, false otherwise.
             */
            bool
            GetInTreeAabb(
                Dtype &aabb_min_x,
                Dtype &aabb_min_y,
                Dtype &aabb_min_z,
                Dtype &aabb_max_x,
                Dtype &aabb_max_y,
                Dtype &aabb_max_z) const;
        };

        class TreeIterator : public IteratorBase {
        public:
            TreeIterator() = default;

            explicit TreeIterator(const OctreeImpl *tree, uint32_t max_node_depth = 0);

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
            OctreeKey m_aabb_min_key_;
            OctreeKey m_aabb_max_key_;

        public:
            InAabbIteratorBase() = default;

            InAabbIteratorBase(
                Dtype aabb_min_x,
                Dtype aabb_min_y,
                Dtype aabb_min_z,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                Dtype aabb_max_z,
                const OctreeImpl *tree,
                uint32_t max_node_depth = 0);

            InAabbIteratorBase(
                OctreeKey aabb_min_key,
                OctreeKey aabb_max_key,
                const OctreeImpl *tree,
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
                Dtype aabb_min_z,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                Dtype aabb_max_z,
                const OctreeImpl *tree,
                uint32_t max_node_depth = 0);

            TreeInAabbIterator(
                OctreeKey aabb_min_key,
                OctreeKey aabb_max_key,
                const OctreeImpl *tree,
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
                Dtype aabb_min_z,
                Dtype aabb_max_x,
                Dtype aabb_max_y,
                Dtype aabb_max_z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth = 0);

            LeafInAabbIterator(
                OctreeKey aabb_min_key,
                OctreeKey aabb_max_key,
                const OctreeImpl *tree,
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

            explicit LeafIterator(const OctreeImpl *tree, uint32_t depth = 0);

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
                OctreeKey key,
                uint32_t cluster_depth,
                const OctreeImpl *tree,
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

            LeafNeighborIteratorBase(const OctreeImpl *tree, uint32_t max_leaf_depth);

        protected:
            OctreeKey m_neighbor_key_ = {};
            OctreeKey::KeyType m_max_key_changing_dim1_ = 0;
            OctreeKey::KeyType m_min_key_changing_dim2_ = 0;
            OctreeKey::KeyType m_max_key_changing_dim2_ = 0;

            /**
             *
             * @param key
             * @param key_depth
             * @param changing_dim1 the dimension that is changing during the search
             * @param changing_dim2 the dimension that is changing during the search
             * @param unchanged_dim the dimension that is unchanged during the search
             * @param increase
             */
            void
            Init(
                OctreeKey key,
                uint32_t key_depth,
                int changing_dim1,
                int changing_dim2,
                int unchanged_dim,
                bool increase);

            void
            SingleIncrementOf(int changing_dim1, int changing_dim2);
        };

        class WestLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim1_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_ChangingDim2_ = 2;  // changing the z-dim key during the search
            static constexpr int sk_UnchangedDim_ = 0;
            static constexpr bool sk_Increase_ = false;  // decrease the x-dim key

        public:
            WestLeafNeighborIterator() = default;

            WestLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            WestLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
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
            static constexpr int sk_ChangingDim1_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_ChangingDim2_ = 2;  // changing the z-dim key during the search
            static constexpr int sk_UnchangedDim_ = 0;
            static constexpr bool sk_Increase_ = true;  // increase the x-dim key

        public:
            EastLeafNeighborIterator() = default;

            EastLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            EastLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
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
            static constexpr int sk_ChangingDim1_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_ChangingDim2_ = 2;  // changing the z-dim key during the search
            static constexpr int sk_UnchangedDim_ = 1;
            static constexpr bool sk_Increase_ = true;  // increase the y-dim key

        public:
            NorthLeafNeighborIterator() = default;

            NorthLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            NorthLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
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
            static constexpr int sk_ChangingDim1_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_ChangingDim2_ = 2;  // changing the z-dim key during the search
            static constexpr int sk_UnchangedDim_ = 1;
            static constexpr bool sk_Increase_ = false;  // decrease the y-dim key

        public:
            SouthLeafNeighborIterator() = default;

            SouthLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            SouthLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
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

        class TopLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim1_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_ChangingDim2_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_UnchangedDim_ = 2;
            static constexpr bool sk_Increase_ = true;  // increase the z-dim key

        public:
            TopLeafNeighborIterator() = default;

            TopLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            TopLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            auto
            operator++(int);

            // pre-increment
            TopLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class BottomLeafNeighborIterator : public LeafNeighborIteratorBase {
            static constexpr int sk_ChangingDim1_ = 0;  // changing the x-dim key during the search
            static constexpr int sk_ChangingDim2_ = 1;  // changing the y-dim key during the search
            static constexpr int sk_UnchangedDim_ = 2;
            static constexpr bool sk_Increase_ = false;  // decrease the z-dim key

        public:
            BottomLeafNeighborIterator() = default;

            BottomLeafNeighborIterator(
                Dtype x,
                Dtype y,
                Dtype z,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            BottomLeafNeighborIterator(
                const OctreeKey &key,
                uint32_t key_depth,
                const OctreeImpl *tree,
                uint32_t max_leaf_depth);

            // post-increment
            auto
            operator++(int);

            // pre-increment
            BottomLeafNeighborIterator &
            operator++();

            void
            Next() override;
        };

        class NodeOnRayIterator : public IteratorBase {
            Vector3 m_origin_ = {};
            Vector3 m_dir_ = {};
            Vector3 m_dir_inv_ = {};
            Dtype m_max_range_ = 0.f;
            Dtype m_node_padding_ = 0.0f;  // padding for node size
            bool m_bidirectional_ = false;
            bool m_leaf_only_ = false;
            uint32_t m_min_node_depth_ = 0;

        public:
            NodeOnRayIterator() = default;

            /**
             *
             * @param px
             * @param py
             * @param pz
             * @param vx
             * @param vy
             * @param vz
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
                Dtype pz,
                Dtype vx,
                Dtype vy,
                Dtype vz,
                Dtype max_range,
                Dtype node_padding,
                const bool &bidirectional,
                const OctreeImpl *tree,
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
        BeginLeafOfNode(const OctreeKey &key, uint32_t node_depth, uint32_t max_depth = 0) const;

        [[nodiscard]] LeafOfNodeIterator
        EndLeafOfNode() const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(const Aabb<Dtype, 3> &aabb, uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(
            Dtype aabb_min_x,
            Dtype aabb_min_y,
            Dtype aabb_min_z,
            Dtype aabb_max_x,
            Dtype aabb_max_y,
            Dtype aabb_max_z,
            uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        BeginLeafInAabb(
            const OctreeKey &aabb_min_key,
            const OctreeKey &aabb_max_key,
            uint32_t max_depth = 0) const;

        [[nodiscard]] LeafInAabbIterator
        EndLeafInAabb() const;

        [[nodiscard]] std::shared_ptr<typename AbstractOctree<Dtype>::OctreeNodeIterator>
        GetLeafInAabbIterator(const Aabb<Dtype, 3> &aabb, uint32_t max_depth) const override;

        [[nodiscard]] TreeIterator
        BeginTree(uint32_t max_depth = 0) const;

        [[nodiscard]] TreeIterator
        EndTree() const;

        [[nodiscard]] std::shared_ptr<typename AbstractOctree<Dtype>::OctreeNodeIterator>
        GetTreeIterator(uint32_t max_depth) const override;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(const Aabb<Dtype, 3> &aabb, uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(
            Dtype aabb_min_x,
            Dtype aabb_min_y,
            Dtype aabb_min_z,
            Dtype aabb_max_x,
            Dtype aabb_max_y,
            Dtype aabb_max_z,
            uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        BeginTreeInAabb(
            const OctreeKey &aabb_min_key,
            const OctreeKey &aabb_max_key,
            uint32_t max_depth = 0) const;

        [[nodiscard]] TreeInAabbIterator
        EndTreeInAabb() const;

        [[nodiscard]] WestLeafNeighborIterator
        BeginWestLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] WestLeafNeighborIterator
        BeginWestLeafNeighbor(const OctreeKey &key, uint32_t key_depth, uint32_t max_leaf_depth = 0)
            const;

        [[nodiscard]] WestLeafNeighborIterator
        EndWestLeafNeighbor() const;

        [[nodiscard]] EastLeafNeighborIterator
        BeginEastLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] EastLeafNeighborIterator
        BeginEastLeafNeighbor(const OctreeKey &key, uint32_t key_depth, uint32_t max_leaf_depth = 0)
            const;

        [[nodiscard]] EastLeafNeighborIterator
        EndEastLeafNeighbor() const;

        [[nodiscard]] NorthLeafNeighborIterator
        BeginNorthLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] NorthLeafNeighborIterator
        BeginNorthLeafNeighbor(
            const OctreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] NorthLeafNeighborIterator
        EndNorthLeafNeighbor() const;

        [[nodiscard]] SouthLeafNeighborIterator
        BeginSouthLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] SouthLeafNeighborIterator
        BeginSouthLeafNeighbor(
            const OctreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] SouthLeafNeighborIterator
        EndSouthLeafNeighbor() const;

        [[nodiscard]] TopLeafNeighborIterator
        BeginTopLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] TopLeafNeighborIterator
        BeginTopLeafNeighbor(const OctreeKey &key, uint32_t key_depth, uint32_t max_leaf_depth = 0)
            const;

        [[nodiscard]] TopLeafNeighborIterator
        EndTopLeafNeighbor() const;

        [[nodiscard]] BottomLeafNeighborIterator
        BeginBottomLeafNeighbor(Dtype x, Dtype y, Dtype z, uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] BottomLeafNeighborIterator
        BeginBottomLeafNeighbor(
            const OctreeKey &key,
            uint32_t key_depth,
            uint32_t max_leaf_depth = 0) const;

        [[nodiscard]] BottomLeafNeighborIterator
        EndBottomLeafNeighbor() const;

        [[nodiscard]] NodeOnRayIterator
        BeginNodeOnRay(
            Dtype px,
            Dtype py,
            Dtype pz,
            Dtype vx,
            Dtype vy,
            Dtype vz,
            Dtype max_range = -1,
            Dtype node_padding = 0.0,  // padding for the node size
            bool bidirectional = false,
            bool leaf_only = false,
            uint32_t min_node_depth = 0,
            uint32_t max_node_depth = 0) const;

        [[nodiscard]] NodeOnRayIterator
        BeginNodeOnRay(
            const Eigen::Ref<const Vector3> &origin,
            const Eigen::Ref<const Vector3> &direction,
            Dtype max_range = -1,
            Dtype node_padding = 0.0,  // padding for the node size
            bool bidirectional = false,
            bool leaf_only = false,
            uint32_t min_node_depth = 0,
            uint32_t max_node_depth = 0) const;

        [[nodiscard]] NodeOnRayIterator
        EndNodeOnRay() const;

        //-- ray tracing
        /**
         * Trace a ray from origin to end (excluded), returning the OctreeKeyRay of all nodes'
         * OctreeKey traversed by the ray. For each key, the corresponding node may not exist. You
         * can use Search() to check if the node exists.
         * @param sx x coordinate of the origin
         * @param sy y coordinate of the origin
         * @param sz z coordinate of the origin
         * @param ex x coordinate of the end (excluded)
         * @param ey y coordinate of the end (excluded)
         * @param ez z coordinate of the end (excluded)
         * @param ray
         * @return
         */
        [[nodiscard]] bool
        ComputeRayKeys(
            Dtype sx,
            Dtype sy,
            Dtype sz,
            Dtype ex,
            Dtype ey,
            Dtype ez,
            OctreeKeyRay &ray) const;

        /**
         * Trace a ray from origin to end (excluded), returning a list of all nodes' coordinates
         * traversed by the ray. For each coordinate, the corresponding node may not exist. You can
         * use Search() to check if the node exists.
         * @param sx x coordinate of the origin
         * @param sy y coordinate of the origin
         * @param sz z coordinate of the origin
         * @param ex x coordinate of the end (excluded)
         * @param ey y coordinate of the end (excluded)
         * @param ez z coordinate of the end (excluded)
         * @param ray
         * @return
         */
        [[nodiscard]] bool
        ComputeRayCoords(
            Dtype sx,
            Dtype sy,
            Dtype sz,
            Dtype ex,
            Dtype ey,
            Dtype ez,
            std::vector<Vector3> &ray) const;

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
        DeleteNodeChild(Node *node, uint32_t child_idx, const OctreeKey &key);

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
         * @param z
         * @param delete_depth delete_depth to start deleting nodes, nodes at delete_depth >=
         * delete_depth will be deleted. If delete_depth == 0, delete at the lowest level.
         * @return
         */
        uint32_t
        DeleteNode(Dtype x, Dtype y, Dtype z, uint32_t delete_depth = 0);

        /**
         * Delete nodes that contain the given key, at or deeper than the given delete_depth.
         * @param key
         * @param delete_depth delete_depth to start deleting nodes, nodes at delete_depth >=
         * delete_depth will be deleted. If delete_depth == 0, delete at the lowest level.
         * @return
         */
        void
        DeleteNode(const OctreeKey &key, uint32_t delete_depth = 0);

    protected:
        /**
         * Callback before deleting a child node.
         * @param node parent node, which may be nullptr if the child is the root node
         * @param child child node to be deleted
         * @param key the key of the child node
         */
        virtual void
        OnDeleteNodeChild(Node *node, Node *child, const OctreeKey &key);

        /**
         * Delete child nodes down to min_depth matching the given key of the given node that is at
         * the given depth.
         * @param node node at depth, which must not be nullptr, and it will not be deleted
         * @param key
         * @param min_depth
         * @return
         */
        bool
        DeleteNodeRecurs(Node *node, const OctreeKey &key, uint32_t min_depth);

        /**
         * Delete all descendants of the given node.
         * @param node
         * @param key
         * @return
         */
        void
        DeleteNodeDescendants(Node *node, const OctreeKey &key);

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
        [[nodiscard]] const AbstractOctreeNode *
        SearchNode(Dtype x, Dtype y, Dtype z, uint32_t max_depth) const override;

        [[nodiscard]] const AbstractOctreeNode *
        SearchNode(const OctreeKey &key, uint32_t max_depth) const override;

        [[nodiscard]] const Node *
        Search(const Eigen::Ref<const Vector3> &position, uint32_t max_depth = 0) const;

        /**
         * Search node given a point.
         * @param x
         * @param y
         * @param z
         * @param max_depth Max depth to search. However, max_depth=0 means searching from the root.
         * @return
         */
        [[nodiscard]] const Node *
        Search(Dtype x, Dtype y, Dtype z, uint32_t max_depth = 0) const;

        /**
         * Search node at specified depth given a key.
         * @param key
         * @param max_depth Max depth to search. 0 means searching until the deepest.
         * @return A pointer to the deepest node that matches the key.
         */
        [[nodiscard]] const Node *
        Search(const OctreeKey &key, uint32_t max_depth = 0) const;

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
        SearchNodes(const OctreeKey &key, uint32_t max_depth = 0) const;

        Node *
        InsertNode(Dtype x, Dtype y, Dtype z, uint32_t depth = 0);

        Node *
        InsertNode(const OctreeKey &key, uint32_t depth = 0);

    protected:
        //-- file IO
        /**
         * Read all nodes from the input stream (without the file header). For general file IO, use
         * AbstractOctree::Read.
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

#include "octree_impl.tpp"
