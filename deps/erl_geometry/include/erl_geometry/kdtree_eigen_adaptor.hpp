#pragma once

#include "erl_common/eigen.hpp"
#include "erl_common/logging.hpp"

#include <nanoflann.hpp>

#include <memory>

namespace erl::geometry {

    /**
     * KdTreeEigenAdaptor has internal storage of data  points compared to
     * nanoflann::KDTreeEigenMatrixAdaptor.
     * @tparam T The type of the data points
     * @tparam Dim The dimension of the data points
     * @tparam Metric The metric to use for the KD-tree, default is nanoflann::metric_L2_Simple
     * @tparam IndexType The type of the index, default is long
     */
    template<
        typename T,
        int Dim,
        typename Metric = nanoflann::metric_L2_Simple,
        typename IndexType = long>
    class KdTreeEigenAdaptor {

        using EigenMatrix = Eigen::Matrix<T, Dim, Eigen::Dynamic>;
        using Self = KdTreeEigenAdaptor<T, Dim, Metric, IndexType>;
        using NumType = typename EigenMatrix::Scalar;
        using MetricType = typename Metric::template traits<NumType, Self>::distance_t;
        using TreeType = nanoflann::KDTreeSingleIndexAdaptor<MetricType, Self, Dim, IndexType>;

        std::shared_ptr<TreeType> m_tree_ = nullptr;
        EigenMatrix m_data_matrix_{};
        const int m_leaf_max_size_;

    public:
        explicit KdTreeEigenAdaptor(const int leaf_max_size = 10)
            : m_leaf_max_size_(leaf_max_size) {}

        explicit KdTreeEigenAdaptor(
            EigenMatrix mat,
            const bool build = true,
            const int leaf_max_size = 10)
            : m_data_matrix_(std::move(mat)),
              m_leaf_max_size_(leaf_max_size) {

            if (build) { Build(); }
        }

        explicit KdTreeEigenAdaptor(
            const T *data,
            long num_points,
            const bool build = true,
            const int leaf_max_size = 10)
            : m_data_matrix_(Eigen::Map<const EigenMatrix>(data, Dim, num_points)),
              m_leaf_max_size_(leaf_max_size) {

            if (build) { Build(); }
        }

        [[nodiscard]] const EigenMatrix &
        GetDataMatrix() const {
            return m_data_matrix_;
        }

        [[nodiscard]] Eigen::Vector<T, Dim>
        GetPoint(IndexType idx) const {
            return m_data_matrix_.col(idx);
        }

        void
        SetDataMatrix(EigenMatrix mat, const bool build = true) {
            m_data_matrix_ = std::move(mat);
            m_tree_ = nullptr;  // invalidate the tree
            if (build) { Build(); }
        }

        void
        SetDataMatrix(const T *data, long num_points, const bool build = true) {
            m_data_matrix_ = Eigen::Map<const EigenMatrix>(data, Dim, num_points);
            m_tree_ = nullptr;  // invalidate the tree
            if (build) { Build(); }
        }

        void
        Clear() {
            m_tree_ = nullptr;
        }

        [[nodiscard]] bool
        Ready() const {
            return m_tree_ != nullptr;
        }

        // Rebuild the KD tree from scratch
        void
        Build() {
            ERL_ASSERTM(m_data_matrix_.cols() > 0, "no data. cannot build tree.");
            m_tree_ = std::make_shared<TreeType>(
                Dim,
                *this,
                nanoflann::KDTreeSingleIndexAdaptorParams(m_leaf_max_size_));
            m_tree_->buildIndex();
        }

        void
        Knn(size_t k,
            const Eigen::Ref<const Eigen::Vector<T, Dim>> &point,
            Eigen::VectorX<IndexType> &indices_out,
            Eigen::VectorX<NumType> &metric_out) {
            ERL_ASSERTM(m_tree_ != nullptr, "tree is not ready yet. Please call Build() first.");
            m_tree_->knnSearch(point.data(), k, indices_out.data(), metric_out.data());
        }

        void
        Nearest(
            const Eigen::Ref<const Eigen::Vector<T, Dim>> &point,
            IndexType &index,
            NumType &metric) const {
            ERL_ASSERTM(m_tree_ != nullptr, "tree is not ready yet. Please call Build() first.");
            m_tree_->knnSearch(point.data(), 1, &index, &metric);
        }

        void
        RadiusSearch(
            const Eigen::Ref<const Eigen::Vector<T, Dim>> &point,
            NumType radius,
            std::vector<nanoflann::ResultItem<IndexType, NumType>> &indices_dists,
            const bool sorted) {
            ERL_ASSERTM(m_tree_ != nullptr, "tree is not ready yet. Please call Build() first.");
            nanoflann::SearchParameters params;
            params.sorted = sorted;
            m_tree_->radiusSearch(point.data(), radius, indices_dists, params);
        }

        // Returns the number of points: used by TreeType
        [[nodiscard]] size_t
        kdtree_get_point_count() const {
            return m_data_matrix_.cols();
        }

        // Returns the dim-th component of the idx-th point in the class, used by TreeType
        [[nodiscard]] NumType
        kdtree_get_pt(const size_t idx, int dim) const {
            return m_data_matrix_(dim, idx);
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation
        // loop.
        template<class BBOX>
        static bool
        kdtree_get_bbox(BBOX &) {
            return false;
        }
    };

    using KdTree3d = KdTreeEigenAdaptor<double, 3>;
    using KdTree2d = KdTreeEigenAdaptor<double, 2>;
    using KdTree3f = KdTreeEigenAdaptor<float, 3>;
    using KdTree2f = KdTreeEigenAdaptor<float, 2>;
}  // namespace erl::geometry

// ReSharper restore CppInconsistentNaming
