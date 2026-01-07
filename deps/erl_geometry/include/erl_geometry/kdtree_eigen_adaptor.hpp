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
    public:
        using DataMatrix = Eigen::Matrix<T, Dim, Eigen::Dynamic>;
        using Self = KdTreeEigenAdaptor;
        using NumType = typename DataMatrix::Scalar;
        using MetricType = typename Metric::template traits<NumType, Self>::distance_t;
        using TreeType = nanoflann::KDTreeSingleIndexAdaptor<MetricType, Self, Dim, IndexType>;
        using ResultItem = nanoflann::ResultItem<IndexType, NumType>;

    private:
        std::shared_ptr<TreeType> m_tree_ = nullptr;
        DataMatrix m_data_matrix_{};
        const int m_leaf_max_size_;

    public:
        explicit KdTreeEigenAdaptor(const int leaf_max_size = 10)
            : m_leaf_max_size_(leaf_max_size) {}

        explicit KdTreeEigenAdaptor(
            DataMatrix mat,
            const bool build = true,
            const int leaf_max_size = 10)
            : m_data_matrix_(std::move(mat)), m_leaf_max_size_(leaf_max_size) {

            if (build) { Build(); }
        }

        explicit KdTreeEigenAdaptor(
            const T *data,
            long num_points,
            const bool build = true,
            const int leaf_max_size = 10)
            : m_data_matrix_(Eigen::Map<const DataMatrix>(data, Dim, num_points)),
              m_leaf_max_size_(leaf_max_size) {

            if (build) { Build(); }
        }

        [[nodiscard]] const DataMatrix &
        GetDataMatrix() const {
            return m_data_matrix_;
        }

        [[nodiscard]] Eigen::Vector<T, Dim>
        GetPoint(IndexType idx) const {
            return m_data_matrix_.col(idx);
        }

        void
        SetDataMatrix(DataMatrix mat, const bool build = true) {
            m_data_matrix_ = std::move(mat);
            m_tree_ = nullptr;  // invalidate the tree
            if (build) { Build(); }
        }

        void
        SetDataMatrix(const T *data, long num_points, const bool build = true) {
            m_data_matrix_ = Eigen::Map<const DataMatrix>(data, Dim, num_points);
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

        [[nodiscard]] long
        Size() const {
            return m_data_matrix_.cols();
        }

        /**
         * K nearest neighbor search. The output distances are sorted.
         * @param k The number of nearest neighbors to search.
         * @param point The query point.
         * @param indices_out The output indices of the nearest neighbors.
         * @param metric_out The output distances to the nearest neighbors. If L2, squared distances
         * are returned.
         * @return The number of neighbors found. Smaller than k if there are not enough points in
         * the tree.
         */
        [[nodiscard]] IndexType
        Knn(IndexType k,
            const Eigen::Vector<T, Dim> &point,
            Eigen::VectorX<IndexType> &indices_out,
            Eigen::VectorX<NumType> &metric_out) {
            ERL_ASSERT_GT(k, 0);
            ERL_ASSERT_PTR(m_tree_);

            if (indices_out.size() < k) {
                indices_out.setConstant(k, -1);
            } else {
                indices_out.setConstant(-1);
            }
            if (metric_out.size() < k) { metric_out.resize(k); }
            k = m_tree_->knnSearch(point.data(), k, indices_out.data(), metric_out.data());
            return k;
        }

        /**
         * K nearest neighbor search. The output distances are sorted.
         * @param k The number of nearest neighbors to search.
         * @param point The query point.
         * @param indices_out The output indices of the nearest neighbors.
         * @param metric_out The output distances to the nearest neighbors. If L2, squared distances
         * are returned.
         * @return The number of neighbors found. Smaller than k if there are not enough points in
         * the tree.
         */
        [[nodiscard]] IndexType
        Knn(IndexType k,
            const Eigen::Vector<T, Dim> &point,
            std::vector<IndexType> &indices_out,
            std::vector<NumType> &metric_out) const {
            ERL_ASSERT_GT(k, 0);
            ERL_ASSERT_PTR(m_tree_);
            indices_out.resize(k, -1);
            metric_out.resize(k);
            k = m_tree_->knnSearch(point.data(), k, indices_out.data(), metric_out.data());
            return k;
        }

        [[nodiscard]] bool
        Nearest(const Eigen::Vector<T, Dim> &point, IndexType &index, NumType &metric) const {
            ERL_ASSERT_PTR(m_tree_);
            return m_tree_->knnSearch(point.data(), 1, &index, &metric);
        }

        void
        RadiusSearch(
            const Eigen::Vector<T, Dim> &point,
            const NumType radius,
            const bool sorted,
            std::vector<ResultItem> &indices_dists) const {
            ERL_ASSERT_PTR(m_tree_);
            nanoflann::SearchParameters params;
            params.sorted = sorted;
            m_tree_->radiusSearch(point.data(), radius * radius, indices_dists, params);
        }

        [[nodiscard]] IndexType
        RadiusKnn(
            IndexType k,
            const Eigen::Vector<T, Dim> &point,
            const NumType radius,
            Eigen::VectorX<IndexType> &indices_out,
            Eigen::VectorX<NumType> &metric_out) const {
            ERL_ASSERT_GT(k, 0);
            ERL_ASSERT_PTR(m_tree_);

            if (indices_out.size() < k) {
                indices_out.setConstant(k, -1);
            } else {
                indices_out.setConstant(-1);
            }
            if (metric_out.size() < k) { metric_out.resize(k); }
            k = m_tree_->rknnSearch(
                point.data(),
                k,
                indices_out.data(),
                metric_out.data(),
                radius * radius);
            return k;
        }

        [[nodiscard]] IndexType
        RadiusKnn(
            IndexType k,
            const Eigen::Vector<T, Dim> &point,
            const NumType radius,
            std::vector<IndexType> &indices_out,
            std::vector<NumType> &metric_out) const {
            ERL_ASSERT_GT(k, 0);
            ERL_ASSERT_PTR(m_tree_);
            indices_out.resize(k, -1);
            metric_out.resize(k);
            k = m_tree_->rknnSearch(
                point.data(),
                k,
                indices_out.data(),
                metric_out.data(),
                radius * radius);
            return k;
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
