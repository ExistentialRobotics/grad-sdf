#pragma once

#include "aabb.hpp"
#include "abstract_octree_node.hpp"
#include "nd_tree_setting.hpp"
#include "octree_key.hpp"

#include "erl_common/factory_pattern.hpp"

#include <memory>
#include <string>

namespace erl::geometry {

    /**
     * AbstractOctree is a base class for all octree implementations. It provides a common interface
     * for factory pattern and file I/O.
     */
    template<typename Dtype>
    class AbstractOctree {
        std::shared_ptr<NdTreeSetting> m_setting_ = std::make_shared<NdTreeSetting>();

    public:
        using DataType = Dtype;
        using Factory =
            common::FactoryPattern<AbstractOctree, false, false, std::shared_ptr<NdTreeSetting>>;
        using Vector3 = Eigen::Vector3<Dtype>;

        AbstractOctree() = delete;  // no default constructor

        explicit AbstractOctree(std::shared_ptr<NdTreeSetting> setting);

        AbstractOctree(const AbstractOctree& other) = default;
        AbstractOctree&
        operator=(const AbstractOctree& other) = default;
        AbstractOctree(AbstractOctree&& other) = default;
        AbstractOctree&
        operator=(AbstractOctree&& other) = default;

        virtual ~AbstractOctree() = default;

        //-- factory pattern
        /**
         * returns the actual class name as string for identification
         * @return The type of the tree.
         */
        [[nodiscard]] std::string
        GetTreeType() const;

        /**
         * Implemented by derived classes to create a new tree of the same type.
         * @return A new tree of the same type.
         */
        [[nodiscard]] virtual std::shared_ptr<AbstractOctree>
        Create(const std::shared_ptr<NdTreeSetting>& setting) const = 0;

        /**
         * Create a new tree of the given type.
         * @param tree_id
         * @param setting
         * @return
         */
        static std::shared_ptr<AbstractOctree>
        CreateTree(const std::string& tree_id, const std::shared_ptr<NdTreeSetting>& setting);

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractOctree, Derived>, bool>
        Register(const std::string& tree_type = "") {
            return Factory::GetInstance().template Register<Derived>(
                tree_type,
                [](const std::shared_ptr<NdTreeSetting>& setting) {
                    auto tree_setting =
                        std::dynamic_pointer_cast<typename Derived::Setting>(setting);
                    if (setting == nullptr) {
                        tree_setting = std::make_shared<typename Derived::Setting>();
                    }
                    ERL_ASSERTM(tree_setting != nullptr, "setting is nullptr.");
                    return std::make_shared<Derived>(tree_setting);
                });
        }

        //-- setting
        /**
         * Get the setting of the tree.
         * @tparam T The type of the setting.
         * @return
         */
        template<typename T>
        std::shared_ptr<T>
        GetSetting() const {
            return std::reinterpret_pointer_cast<T>(m_setting_);
        }

        /**
         * This function should be called when the setting is changed.
         */
        virtual void
        ApplySetting() = 0;

        [[nodiscard]] bool
        ReadSetting(std::istream& s) const;

        void
        WriteSetting(std::ostream& s) const;

        //-- comparison
        [[nodiscard]] virtual bool
        operator==(const AbstractOctree& other) const = 0;

        [[nodiscard]] bool
        operator!=(const AbstractOctree& other) const;

        //-- get tree information
        [[nodiscard]] uint32_t
        GetTreeDepth() const;

        [[nodiscard]] Dtype
        GetResolution() const;

        [[nodiscard]] virtual std::size_t
        GetSize() const = 0;

        [[nodiscard]] virtual std::size_t
        GetMemoryUsage() const = 0;

        [[nodiscard]] virtual std::size_t
        GetMemoryUsagePerNode() const = 0;

        Vector3
        GetMetricMin();

        [[nodiscard]] Vector3
        GetMetricMin() const;

        void
        GetMetricMin(Vector3& min);

        void
        GetMetricMin(Vector3& min) const;

        virtual void
        GetMetricMin(Dtype& x, Dtype& y, Dtype& z) = 0;

        virtual void
        GetMetricMin(Dtype& x, Dtype& y, Dtype& z) const = 0;

        Vector3
        GetMetricMax();

        [[nodiscard]] Vector3
        GetMetricMax() const;

        void
        GetMetricMax(Vector3& max);

        void
        GetMetricMax(Vector3& max) const;

        virtual void
        GetMetricMax(Dtype& x, Dtype& y, Dtype& z) = 0;

        virtual void
        GetMetricMax(Dtype& x, Dtype& y, Dtype& z) const = 0;

        Aabb<Dtype, 3>
        GetMetricAabb();

        [[nodiscard]] Aabb<Dtype, 3>
        GetMetricAabb() const;

        std::pair<Vector3, Vector3>
        GetMetricMinMax();

        [[nodiscard]] std::pair<Vector3, Vector3>
        GetMetricMinMax() const;

        void
        GetMetricMinMax(Vector3& min, Vector3& max);

        void
        GetMetricMinMax(Vector3& min, Vector3& max) const;

        virtual void
        GetMetricMinMax(
            Dtype& min_x,
            Dtype& min_y,
            Dtype& min_z,
            Dtype& max_x,
            Dtype& max_y,
            Dtype& max_z) = 0;

        virtual void
        GetMetricMinMax(
            Dtype& min_x,
            Dtype& min_y,
            Dtype& min_z,
            Dtype& max_x,
            Dtype& max_y,
            Dtype& max_z) const = 0;

        Vector3
        GetMetricSize();

        [[nodiscard]] Vector3
        GetMetricSize() const;

        void
        GetMetricSize(Vector3& size);

        void
        GetMetricSize(Vector3& size) const;

        virtual void
        GetMetricSize(Dtype& x, Dtype& y, Dtype& z) = 0;

        virtual void
        GetMetricSize(Dtype& x, Dtype& y, Dtype& z) const = 0;

        [[nodiscard]] virtual Dtype
        GetNodeSize(uint32_t depth) const = 0;

        //-- IO
        virtual void
        Clear() = 0;
        virtual void
        Prune() = 0;
        /**
         * Write the tree as raw data to a stream.
         * @param s
         * @return
         */
        [[nodiscard]] bool
        Write(std::ostream& s) const;

        /**
         * Write all nodes to the output stream (without the file header) for a created tree.
         * Pruning the tree first produces smaller files and faster loading.
         * @param s
         * @return
         */
        virtual std::ostream&
        WriteData(std::ostream& s) const = 0;

        /**
         * Generic read function to read an octree from a stream.
         * @param s
         * @return An octree derived from AbstractOctree
         */
        bool
        Read(std::istream& s);

        /**
         * Read all nodes from the input steam (without the file header) for a created tree.
         */
        virtual std::istream&
        ReadData(std::istream& s) = 0;

        //-- search node
        [[nodiscard]] virtual const AbstractOctreeNode*
        SearchNode(Dtype x, Dtype y, Dtype z, uint32_t max_depth) const = 0;

        [[nodiscard]] virtual const AbstractOctreeNode*
        SearchNode(const OctreeKey& key, uint32_t max_depth) const = 0;

        //-- iterators
        struct OctreeNodeIterator {
            virtual ~OctreeNodeIterator() = default;

            [[nodiscard]] virtual Dtype
            GetX() const = 0;
            [[nodiscard]] virtual Dtype
            GetY() const = 0;
            [[nodiscard]] virtual Dtype
            GetZ() const = 0;
            [[nodiscard]] virtual Vector3
            GetCenter() const = 0;
            [[nodiscard]] virtual Dtype
            GetNodeSize() const = 0;
            [[nodiscard]] virtual uint32_t
            GetDepth() const = 0;
            virtual void
            Next() = 0;
            [[nodiscard]] virtual bool
            IsValid() const = 0;
            [[nodiscard]] virtual const AbstractOctreeNode*
            GetNode() const = 0;
            [[nodiscard]] virtual const OctreeKey&
            GetKey() const = 0;
            [[nodiscard]] virtual OctreeKey
            GetIndexKey() const = 0;
        };

        [[nodiscard]] virtual std::shared_ptr<OctreeNodeIterator>
        GetTreeIterator(uint32_t max_depth) const = 0;

        [[nodiscard]] virtual std::shared_ptr<OctreeNodeIterator>
        GetLeafInAabbIterator(const Aabb<Dtype, 3>& aabb, uint32_t max_depth) const = 0;
    };

    extern template class AbstractOctree<double>;
    extern template class AbstractOctree<float>;
}  // namespace erl::geometry
