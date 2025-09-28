#pragma once

#include "aabb.hpp"
#include "abstract_quadtree_node.hpp"
#include "nd_tree_setting.hpp"
#include "quadtree_key.hpp"

#include "erl_common/factory_pattern.hpp"

#include <memory>
#include <string>

namespace erl::geometry {

    /**
     * AbstractQuadtree is a base class for all quadtree implementations. It provides a common
     * interface for factory pattern and file I/O.
     */
    template<typename Dtype>
    class AbstractQuadtree {
        std::shared_ptr<NdTreeSetting> m_setting_ = std::make_shared<NdTreeSetting>();

    public:
        using DataType = Dtype;
        using Factory =
            common::FactoryPattern<AbstractQuadtree, false, false, std::shared_ptr<NdTreeSetting>>;
        using Vector2 = Eigen::Vector2<Dtype>;

        AbstractQuadtree() = delete;  // no default constructor

        explicit AbstractQuadtree(std::shared_ptr<NdTreeSetting> setting);

        AbstractQuadtree(const AbstractQuadtree& other) = default;
        AbstractQuadtree&
        operator=(const AbstractQuadtree& other) = default;
        AbstractQuadtree(AbstractQuadtree&& other) = default;
        AbstractQuadtree&
        operator=(AbstractQuadtree&& other) = default;

        virtual ~AbstractQuadtree() = default;

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
        [[nodiscard]] virtual std::shared_ptr<AbstractQuadtree>
        Create(const std::shared_ptr<NdTreeSetting>& setting) const = 0;

        /**
         * Create a new tree of the given type.
         * @param tree_id
         * @param setting
         * @return
         */
        static std::shared_ptr<AbstractQuadtree>
        CreateTree(const std::string& tree_id, const std::shared_ptr<NdTreeSetting>& setting);

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractQuadtree, Derived>, bool>
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
        operator==(const AbstractQuadtree& other) const = 0;

        [[nodiscard]] bool
        operator!=(const AbstractQuadtree& other) const;

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

        Vector2
        GetMetricMin();

        [[nodiscard]] Vector2
        GetMetricMin() const;

        void
        GetMetricMin(Vector2& min);

        void
        GetMetricMin(Vector2& min) const;

        virtual void
        GetMetricMin(Dtype& x, Dtype& y) = 0;

        virtual void
        GetMetricMin(Dtype& x, Dtype& y) const = 0;

        Vector2
        GetMetricMax();

        [[nodiscard]] Vector2
        GetMetricMax() const;

        void
        GetMetricMax(Vector2& max);

        void
        GetMetricMax(Vector2& max) const;

        virtual void
        GetMetricMax(Dtype& x, Dtype& y) = 0;

        virtual void
        GetMetricMax(Dtype& x, Dtype& y) const = 0;

        Aabb<Dtype, 2>
        GetMetricAabb();

        [[nodiscard]] Aabb<Dtype, 2>
        GetMetricAabb() const;

        std::pair<Vector2, Vector2>
        GetMetricMinMax();

        [[nodiscard]] std::pair<Vector2, Vector2>
        GetMetricMinMax() const;

        void
        GetMetricMinMax(Vector2& min, Vector2& max);

        void
        GetMetricMinMax(Vector2& min, Vector2& max) const;

        virtual void
        GetMetricMinMax(Dtype& min_x, Dtype& min_y, Dtype& max_x, Dtype& max_y) = 0;

        virtual void
        GetMetricMinMax(Dtype& min_x, Dtype& min_y, Dtype& max_x, Dtype& max_y) const = 0;

        Vector2
        GetMetricSize();

        [[nodiscard]] Vector2
        GetMetricSize() const;

        void
        GetMetricSize(Vector2& size);

        void
        GetMetricSize(Vector2& size) const;

        virtual void
        GetMetricSize(Dtype& x, Dtype& y) = 0;

        virtual void
        GetMetricSize(Dtype& x, Dtype& y) const = 0;

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

        bool
        Read(std::istream& s);

        /**
         * Read all nodes from the input steam (without the file header) for a created tree.
         */
        virtual std::istream&
        ReadData(std::istream& s) = 0;

        //-- search node
        [[nodiscard]] virtual const AbstractQuadtreeNode*
        SearchNode(Dtype x, Dtype y, uint32_t max_depth) const = 0;

        [[nodiscard]] virtual const AbstractQuadtreeNode*
        SearchNode(const QuadtreeKey& key, uint32_t max_depth) const = 0;

        //-- iterators
        struct QuadtreeNodeIterator {
            virtual ~QuadtreeNodeIterator() = default;

            [[nodiscard]] virtual Dtype
            GetX() const = 0;
            [[nodiscard]] virtual Dtype
            GetY() const = 0;
            [[nodiscard]] virtual Vector2
            GetCenter() const = 0;
            [[nodiscard]] virtual Dtype
            GetNodeSize() const = 0;
            [[nodiscard]] virtual uint32_t
            GetDepth() const = 0;
            virtual void
            Next() = 0;
            [[nodiscard]] virtual bool
            IsValid() const = 0;
            [[nodiscard]] virtual const AbstractQuadtreeNode*
            GetNode() const = 0;
            [[nodiscard]] virtual const QuadtreeKey&
            GetKey() const = 0;
            [[nodiscard]] virtual QuadtreeKey
            GetIndexKey() const = 0;
        };

        [[nodiscard]] virtual std::shared_ptr<QuadtreeNodeIterator>
        GetTreeIterator(uint32_t max_depth) const = 0;

        [[nodiscard]] virtual std::shared_ptr<QuadtreeNodeIterator>
        GetLeafInAabbIterator(const Aabb<Dtype, 2>& aabb, uint32_t max_depth) const = 0;
    };

    extern template class AbstractQuadtree<double>;
    extern template class AbstractQuadtree<float>;
}  // namespace erl::geometry
