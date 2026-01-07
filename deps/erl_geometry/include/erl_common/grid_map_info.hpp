#pragma once

#include "storage_order.hpp"

#ifdef ERL_USE_OPENCV
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>

    #include <algorithm>
#endif

namespace erl::common {

    template<typename Dtype, typename Index>
    Index
    MeterToGrid(const Dtype meter, const Dtype meter_min, const Dtype resolution) {
        return static_cast<Index>(std::floor((meter - meter_min) / resolution));
    }

    template<typename Dtype, typename Index>
    Dtype
    GridToMeter(const Index grid, const Dtype meter_min, const Dtype resolution) {
        // The grid should cover a valid range, e.g., if the grid is 0, the valid range is
        // [meter_min, meter_min + resolution). If we use rounding when converting from meter to
        // grid, the covered range will be [meter_min - 0.5 * resolution, meter_min + 0.5 *
        // resolution). Half of the range is invalid. Therefore, we need to add 0.5 to the grid
        // index to get the correct range. If we use `floor` instead, the covered range will be
        // [meter_min, meter_min + resolution).
        return (static_cast<Dtype>(grid) + 0.5f) * resolution + meter_min;
    }

    /**
     * Convert discrete coordinate to vertex coordinate in meters. This function is used when we are
     * dealing with the grid points in the grid map. If we want to deal with the grid cells, we
     * should use the `GridToMeter` function.
     * @tparam Dtype data type.
     * @param index discrete coordinate.
     * @param meter_min min vertex coordinate in meters.
     * @param resolution grid resolution in meters.
     * @return vertex coordinate in meters.
     */
    template<typename Dtype, typename Index>
    Dtype
    VertexIndexToMeter(const Index index, const Dtype meter_min, const Dtype resolution) {
        return static_cast<Dtype>(index) * resolution + meter_min;
    }

    template<typename Index, int Dim, bool RowMajor>
    Eigen::Matrix<Index, Dim, Eigen::Dynamic>
    CalculateGridCoordinates(const Eigen::Vector<Index, Dim> &grid_shape) {
        const Index size = grid_shape.prod();
        if (size == 0) { return {}; }

        Eigen::Vector<Index, Dim> strides;
        if constexpr (RowMajor) {
            strides = ComputeCStrides<Index>(grid_shape, 1);
        } else {
            strides = ComputeFStrides<Index>(grid_shape, 1);
        }

        const long n_dims = grid_shape.size();
        Eigen::Matrix<Index, Dim, Eigen::Dynamic> grid_coords(n_dims, size);
        for (long i = 0; i < n_dims; ++i) {
            const Index stride = strides[i];
            const Index dim_size = grid_shape[i];
            const Index n_copies = size / dim_size;
            Eigen::MatrixX<Index> coords =
                Eigen::VectorX<Index>::LinSpaced(dim_size, 0, dim_size - 1)
                    .transpose()
                    .replicate(stride, n_copies / stride);
            Eigen::Map<Eigen::Matrix<Index, 1, Eigen::Dynamic>> coords_reshaped(
                coords.data(),
                1,
                size);
            grid_coords.row(i) << coords_reshaped;
        }
        return grid_coords;
    }

    /**
     * Generate the meter coordinates of all grid/vertex points in the grid map.
     * @tparam Dtype data type.
     * @tparam Dim dimension.
     * @tparam RowMajor whether the grid map is stored in row-major order.
     * @tparam GridCoords whether to compute grid coordinates or vertex coordinates. If true, grid
     * coordinates will be computed; otherwise, vertex coordinates will be computed.
     * @param grid_shape shape of the grid map.
     * @param grid_min min vertex coordinate of the grid map in meters.
     * @param grid_max max vertex coordinate of the grid map in meters.
     * @param resolution grid resolution in meters.
     * @return meter coordinates of all grid points in the grid map. The size is (Dim, N), where
     * N is the number of grid points.
     */
    template<typename Dtype, typename Index, int Dim, bool RowMajor, bool GridCoords>
    Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>
    CalculateMeterCoordinates(
        Eigen::Vector<Index, Dim> grid_shape,
        const Eigen::Vector<Dtype, Dim> &grid_min,
        const Eigen::Vector<Dtype, Dim> &grid_max,
        const Eigen::Vector<Dtype, Dim> &resolution) {

        Index size = grid_shape.prod();
        if (size == 0) { return {}; }

        if constexpr (!GridCoords) {
            grid_shape.array() += 1;
            size = grid_shape.prod();
        }

        Eigen::Vector<Index, Dim> strides;
        if constexpr (RowMajor) {
            // row-major, e.g.
            // coords: [x, y], shape: [r, c], axis0=x, axis1=y
            // strides: [c, 1], when c = 3
            // x: 1, 1, 1, 2, 2, 2, ...
            // y: 1, 2, 3, 1, 2, 3, ...
            strides = ComputeCStrides<Index>(grid_shape, 1);
        } else {
            strides = ComputeFStrides<Index>(grid_shape, 1);
        }

        const long n_dims = grid_shape.size();
        Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> meter_coords(n_dims, size);
        for (long i = 0; i < n_dims; ++i) {
            const Index stride = strides[i];
            const Index dim_size = grid_shape[i];
            const Index n_copies = size / dim_size;
            Dtype min = grid_min[i];
            Dtype max = grid_max[i];
            if constexpr (GridCoords) {
                const Dtype half_res = 0.5f * resolution[i];
                min += half_res;
                max -= half_res;
            }
            Eigen::MatrixX<Dtype> coords = Eigen::VectorX<Dtype>::LinSpaced(dim_size, min, max)
                                               .transpose()
                                               .replicate(stride, n_copies / stride);
            Eigen::Map<Eigen::Matrix<Dtype, 1, Eigen::Dynamic>> coords_reshaped(
                coords.data(),
                1,
                size);
            meter_coords.row(i) << coords_reshaped;
        }

        return meter_coords;
    }

    /**
     * GridMapInfo defines
     * 1. the mapping between right-handed n-dim world system and right-handed n-dim grid map
     * (xy-indexing).
     * 2. hashing of grid coordinates with respect to the specific storing order (row-major or
     * column-major).
     */
    template<typename Dtype, int Dim, typename Index = int>
    class GridMapInfo {

        Eigen::Vector<Index, Dim> m_map_shape_;
        Eigen::Vector<Dtype, Dim> m_resolution_;
        Eigen::Vector<Dtype, Dim> m_min_;
        Eigen::Vector<Dtype, Dim> m_max_;
        Eigen::Vector<Dtype, Dim> m_center_;
        Eigen::Vector<Index, Dim> m_center_grid_;

    public:
        GridMapInfo(
            const Eigen::Vector<Dtype, Dim> &min,
            const Eigen::Vector<Dtype, Dim> &max,
            const Eigen::Vector<Dtype, Dim> &resolution,
            const Eigen::Vector<Index, Dim> &padding)
            : m_map_shape_(
                  Eigen::Vector<Index, Dim>(
                      ((max - min).array() / resolution.array())
                          .ceil()
                          .template cast<Index>()
                          .unaryExpr([](Index x) { return x % 2 ? x + 1 : x; })
                          .array() +
                      1 + 2 * padding.array())),
              m_resolution_(
                  (max - min).array() /
                  (m_map_shape_.array() - 2 * padding.array()).template cast<Dtype>().array()),
              m_min_(min.array() - m_resolution_.array() * padding.template cast<Dtype>().array()),
              m_max_(max.array() + m_resolution_.array() * padding.template cast<Dtype>().array()),
              m_center_((m_min_ + m_max_) * 0.5),
              m_center_grid_(m_map_shape_.array() / 2) {}

        GridMapInfo(
            const Eigen::Vector<Index, Dim> &map_shape,
            const Eigen::Vector<Dtype, Dim> &min,
            const Eigen::Vector<Dtype, Dim> &max)
            : m_map_shape_(Eigen::Vector<Index, Dim>(map_shape.unaryExpr([](Index x) {
                  return x % 2 ? x : x + 1;
              }))),
              m_resolution_((max - min).array() / m_map_shape_.template cast<Dtype>().array()),
              m_min_(min),
              m_max_(max),
              m_center_((m_min_ + m_max_) * 0.5),
              m_center_grid_(m_map_shape_.array() / 2) {
            if (Dim == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(m_map_shape_.size() > 0, "0-dim map is not allowed!");
                ERL_DEBUG_ASSERT(Size() > 0, "0-element map is not allowed!");
            }
        }

        GridMapInfo(
            const Eigen::Vector<Dtype, Dim> &origin,
            const Eigen::Vector<Dtype, Dim> &resolution,
            const Eigen::Vector<Index, Dim> &map_shape)
            : m_map_shape_(Eigen::Vector<Index, Dim>(map_shape.unaryExpr([](Index x) {
                  return x % 2 ? x : x + 1;
              }))),
              m_resolution_(resolution),
              m_min_(origin),
              m_max_(
                  origin.array() + resolution.array() * map_shape.template cast<Dtype>().array()),
              m_center_((m_min_ + m_max_) * 0.5),
              m_center_grid_(m_map_shape_.array() / 2) {}

        explicit GridMapInfo(const GridMapInfo<Dtype, Eigen::Dynamic> &info)
            : m_map_shape_(info.Shape()),
              m_resolution_(info.Resolution()),
              m_min_(info.Min()),
              m_max_(info.Max()),
              m_center_(info.Center()),
              m_center_grid_(info.CenterGrid()) {}

        [[nodiscard]] bool
        operator==(const GridMapInfo &other) const {
            return (m_map_shape_ == other.m_map_shape_) && (m_resolution_ == other.m_resolution_) &&
                   (m_min_ == other.m_min_) && (m_max_ == other.m_max_) &&
                   (m_center_ == other.m_center_) && (m_center_grid_ == other.m_center_grid_);
        }

        [[nodiscard]] bool
        operator!=(const GridMapInfo &other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(std::ostream &s) const {
            const auto n_dims = static_cast<Index>(Dims());
            s.write(reinterpret_cast<const char *>(&n_dims), sizeof(Index));
            s.write(reinterpret_cast<const char *>(m_map_shape_.data()), sizeof(Index) * n_dims);
            s.write(reinterpret_cast<const char *>(m_resolution_.data()), sizeof(Dtype) * n_dims);
            s.write(reinterpret_cast<const char *>(m_min_.data()), sizeof(Dtype) * n_dims);
            s.write(reinterpret_cast<const char *>(m_max_.data()), sizeof(Dtype) * n_dims);
            s.write(reinterpret_cast<const char *>(m_center_.data()), sizeof(Dtype) * n_dims);
            s.write(reinterpret_cast<const char *>(m_center_grid_.data()), sizeof(Index) * n_dims);
            return s.good();
        }

        [[nodiscard]] bool
        Read(std::istream &s) {
            Index n_dims = 0;
            s.read(reinterpret_cast<char *>(&n_dims), sizeof(Index));
            ERL_DEBUG_ASSERT(
                Dim == Eigen::Dynamic || n_dims == Dim,
                "The number of dimensions read from the stream (%d) does not match the template "
                "parameter Dim (%d).",
                n_dims,
                Dim);
            if constexpr (Dim == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(n_dims > 0, "0-dim map is not allowed!");
                m_map_shape_.resize(n_dims);
                m_resolution_.resize(n_dims);
                m_min_.resize(n_dims);
                m_max_.resize(n_dims);
                m_center_.resize(n_dims);
                m_center_grid_.resize(n_dims);
            }
            s.read(reinterpret_cast<char *>(m_map_shape_.data()), sizeof(Index) * n_dims);
            s.read(reinterpret_cast<char *>(m_resolution_.data()), sizeof(Dtype) * n_dims);
            s.read(reinterpret_cast<char *>(m_min_.data()), sizeof(Dtype) * n_dims);
            s.read(reinterpret_cast<char *>(m_max_.data()), sizeof(Dtype) * n_dims);
            s.read(reinterpret_cast<char *>(m_center_.data()), sizeof(Dtype) * n_dims);
            s.read(reinterpret_cast<char *>(m_center_grid_.data()), sizeof(Index) * n_dims);
            return s.good();
        }

        template<typename Dtype2>
        GridMapInfo<Dtype2, Dim>
        Cast() const {
            return {
                m_map_shape_,
                m_min_.template cast<Dtype2>(),
                m_max_.template cast<Dtype2>(),
            };
        }

        template<typename Dtype2>
        std::shared_ptr<GridMapInfo<Dtype2, Dim>>
        CastSharedPtr() const {
            return std::make_shared<GridMapInfo<Dtype2, Dim>>(
                m_map_shape_,
                m_min_.template cast<Dtype2>(),
                m_max_.template cast<Dtype2>());
        }

        [[nodiscard]] GridMapInfo<Dtype, Eigen::Dynamic>
        Extend(Index size, const Dtype min, const Dtype max, const Index dim) const {

            const long n_dims = Dims();
            ERL_DEBUG_ASSERT(
                dim >= 0 && dim <= n_dims,
                "dim = %d is out of range [%d, %d].",
                dim,
                0,
                n_dims);

            Eigen::VectorX<Index> new_shape(n_dims + 1);
            Eigen::VectorX<Dtype> new_min(n_dims + 1);
            Eigen::VectorX<Dtype> new_max(n_dims + 1);

            for (long i = 0; i <= n_dims; ++i) {
                if (i < dim) {
                    new_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_max[i] = m_max_[i];
                } else if (i == dim) {
                    if (size % 2 == 0) { ++size; }  // make sure size is odd
                    new_shape[i] = size;
                    new_min[i] = min;
                    new_max[i] = max;
                } else {
                    new_shape[i] = m_map_shape_[i - 1];
                    new_min[i] = m_min_[i - 1];
                    new_max[i] = m_max_[i - 1];
                }
            }

            return {new_shape, new_min, new_max};
        }

        [[nodiscard]] GridMapInfo<Dtype, Eigen::Dynamic>
        Extend(
            const Dtype min,
            const Dtype max,
            const Dtype resolution,
            const Index padding,
            const Index dim) const {

            const long n_dims = Dims();
            ERL_DEBUG_ASSERT(
                dim >= 0 && dim <= n_dims,
                "dim = %d is out of range [%d, %d]",
                dim,
                0,
                n_dims);

            Eigen::VectorX<Index> new_map_shape(n_dims + 1);
            Eigen::VectorX<Dtype> new_min(n_dims + 1);
            Eigen::VectorX<Dtype> new_resolution(n_dims + 1);

            for (long i = 0; i <= n_dims; ++i) {
                if (i < dim) {
                    new_map_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_resolution[i] = m_resolution_[i];
                } else if (i == dim) {
                    new_map_shape[i] = static_cast<Index>(std::ceil((max - min) / resolution));
                    if (new_map_shape[i] % 2) { ++new_map_shape[i]; }
                    new_resolution[i] = (max - min) / new_map_shape[i];
                    new_map_shape[i] += 1 + 2 * padding;
                    new_min[i] = min - new_resolution[i] * padding;
                } else {
                    new_map_shape[i] = m_map_shape_[i - 1];
                    new_min[i] = m_min_[i - 1];
                    new_resolution[i] = m_resolution_[i - 1];
                }
            }

            return {new_map_shape, new_min, new_resolution};
        }

        [[nodiscard]] GridMapInfo<Dtype, Eigen::Dynamic>
        Squeeze(const long dim) const {
            const long n_dims = Dims();
            ERL_DEBUG_ASSERT(
                dim >= 0 && dim < n_dims,
                "dim = %d is out of range [%d, %d)",
                dim,
                0,
                n_dims);

            Eigen::VectorX<Index> new_map_shape(n_dims - 1);
            Eigen::VectorX<Dtype> new_min(n_dims - 1);
            Eigen::VectorX<Dtype> new_resolution(n_dims - 1);

            for (long i = 0; i < n_dims; ++i) {
                if (i < dim) {
                    new_map_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_resolution[i] = m_resolution_[i];
                }
            }

            return {new_map_shape, new_min, new_resolution};
        }

        [[nodiscard]] long
        Dims() const {
            return m_map_shape_.size();
        }

        [[nodiscard]] const Eigen::Vector<Index, Dim> &
        Shape() const {
            return m_map_shape_;
        }

        [[nodiscard]] Index
        Shape(long dim) const {
            return m_map_shape_[dim];
        }

        [[nodiscard]] Index
        Size() const {
            if (Dims()) { return m_map_shape_.prod(); }
            return 0;
        }

        [[nodiscard]] Index
        Rows() const {
            return m_map_shape_[0];
        }

        [[nodiscard]] Index
        Cols() const {
            return m_map_shape_[1];
        }

        [[nodiscard]] Index
        Width() const {
            return m_map_shape_[0];
        }

        [[nodiscard]] Index
        Height() const {
            return m_map_shape_[1];
        }

        [[nodiscard]] Index
        Length() const {
            if (Dims() >= 3) { return m_map_shape_[2]; }
            return 0;
        }

        [[nodiscard]] const Eigen::Vector<Dtype, Dim> &
        Min() const {
            return m_min_;
        }

        [[nodiscard]] Dtype
        Min(long dim) const {
            return m_min_[dim];
        }

        [[nodiscard]] Eigen::Vector<Dtype, Dim>
        GetMinMeterCoords() const {
            return m_min_ + 0.5f * m_resolution_;
        }

        [[nodiscard]] Dtype
        GetMinMeterCoord(long dim) const {
            return m_min_[dim] + 0.5f * m_resolution_[dim];
        }

        [[nodiscard]] const Eigen::Vector<Dtype, Dim> &
        Max() const {
            return m_max_;
        }

        [[nodiscard]] Dtype
        Max(long dim) const {
            return m_max_[dim];
        }

        [[nodiscard]] Eigen::Vector<Dtype, Dim>
        GetMaxMeterCoords() const {
            return m_max_ - 0.5f * m_resolution_;
        }

        [[nodiscard]] Dtype
        GetMaxMeterCoord(long dim) const {
            return m_max_[dim] - 0.5f * m_resolution_[dim];
        }

        [[nodiscard]] const Eigen::Vector<Dtype, Dim> &
        Resolution() const {
            return m_resolution_;
        }

        [[nodiscard]] Dtype
        Resolution(long dim) const {
            return m_resolution_[dim];
        }

        [[nodiscard]] const Eigen::Vector<Dtype, Dim> &
        Center() const {
            return m_center_;
        }

        [[nodiscard]] const Eigen::Vector<Index, Dim> &
        CenterGrid() const {
            return m_center_grid_;
        }

        [[nodiscard]] Eigen::VectorX<Dtype>
        GetDimLinSpace(const long dim) const {
            return Eigen::VectorX<Dtype>::LinSpaced(
                Shape(dim),
                Min(dim),
                Max(dim) - Resolution(dim));
        }

        [[nodiscard]] Dtype
        GridToMeterAtDim(const Index grid_value, const long dim) const {
            return GridToMeter(grid_value, m_min_[dim], m_resolution_[dim]);
        }

        [[nodiscard]] Eigen::VectorX<Dtype>
        GridToMeterAtDim(const Eigen::Ref<const Eigen::VectorX<Index>> &grid_values, long dim)
            const {

            const Dtype &min = m_min_[dim];
            const Dtype &res = m_resolution_[dim];
            return grid_values.unaryExpr(
                [&](const Index v) -> Dtype { return GridToMeter(v, min, res); });
        }

        [[nodiscard]] Index
        MeterToGridAtDim(const Dtype meter_value, long dim) const {
            return MeterToGrid<Dtype, Index>(meter_value, m_min_[dim], m_resolution_[dim]);
        }

        [[nodiscard]] Eigen::VectorX<Index>
        MeterToGridAtDim(const Eigen::Ref<const Eigen::VectorX<Dtype>> &meter_values, long dim)
            const {
            const Dtype &min = m_min_[dim];
            const Dtype &res = m_resolution_[dim];
            return meter_values.unaryExpr(
                [&](const Dtype v) -> Index { return MeterToGrid<Dtype, Index>(v, min, res); });
        }

        [[nodiscard]] Eigen::Vector<Dtype, Dim>
        GridToMeterForPoint(const Eigen::Ref<const Eigen::Vector<Index, Dim>> &grid_point) const {
            Eigen::Vector<Dtype, Dim> meter_point;
            if constexpr (Dim == Eigen::Dynamic) { meter_point.resize(grid_point.size()); }
            for (long i = 0; i < Dim; ++i) {
                meter_point[i] = GridToMeter(grid_point[i], m_min_[i], m_resolution_[i]);
            }
            return meter_point;
        }

        [[nodiscard]] Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>
        GridToMeterForPoints(
            const Eigen::Ref<const Eigen::Matrix<Index, Dim, Eigen::Dynamic>> &grid_points) const {
            const long n_rows = grid_points.rows();
            const long n_cols = grid_points.cols();
            Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> meter_points(n_rows, n_cols);
            for (long i = 0; i < n_rows; ++i) {
                for (long j = 0; j < n_cols; ++j) {
                    meter_points(i, j) =
                        GridToMeter(grid_points(i, j), m_min_[i], m_resolution_[i]);
                }
            }
            return meter_points;
        }

        [[nodiscard]] Eigen::Vector<Index, Dim>
        MeterToGridForPoint(const Eigen::Ref<const Eigen::Vector<Dtype, Dim>> &meter_point) const {
            Eigen::Vector<Index, Dim> grid_point;
            if constexpr (Dim == Eigen::Dynamic) { grid_point.resize(meter_point.size()); }
            for (int i = 0; i < Dim; ++i) {
                grid_point[i] =
                    MeterToGrid<Dtype, Index>(meter_point[i], m_min_[i], m_resolution_[i]);
            }
            return grid_point;
        }

        [[nodiscard]] Eigen::Matrix<Index, Dim, Eigen::Dynamic>
        MeterToGridForPoints(
            const Eigen::Ref<const Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>> &meter_points) const {
            const long n_dims = meter_points.rows();
            const long n_cols = meter_points.cols();
            Eigen::Matrix<Index, Dim, Eigen::Dynamic> grid_points(n_dims, n_cols);
            for (long j = 0; j < n_cols; ++j) {
                const Dtype *meter = meter_points.col(j).data();
                Index *grid = grid_points.col(j).data();
                for (long i = 0; i < n_dims; ++i) {
                    grid[i] = MeterToGrid<Dtype, Index>(meter[i], m_min_[i], m_resolution_[i]);
                }
            }
            return grid_points;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Index>>
        GridToPixelForPoint(const Eigen::Ref<const Eigen::Vector2<Index>> &grid_point) const {
            Eigen::Vector2<Index> pixel;
            pixel[0] = grid_point[0];
            pixel[1] = m_map_shape_[1] - grid_point[1];
            return pixel;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        GridToPixelForPoints(const Eigen::Ref<const Eigen::Matrix2X<Index>> &grid_points) const {
            if constexpr (D == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(Dims() == 2, "Not available when Dims() != 2");
            }

            Eigen::Matrix2X<Index> pixel_points(2, grid_points.cols());
            // m_map_shape_[0] <--> width
            // m_map_shape_[1] <--> height
            // [x, y] -> [height - y, x] (ij indexing) -> [x, height - y] (xy indexing, used by
            // OpenCV)
            const long n_cols = grid_points.cols();
            for (long j = 0; j < n_cols; ++j) {
                pixel_points(0, j) = grid_points(0, j);
                pixel_points(1, j) = m_map_shape_[1] - grid_points(1, j);
            }
            return pixel_points;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Index>>
        PixelToGridForPoint(const Eigen::Ref<const Eigen::Vector2<Index>> &pixel_point) const {
            return GridToPixelForPoint(pixel_point);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Dtype>>
        SubPixelToGridForPoint(const Eigen::Ref<const Eigen::Vector2<Dtype>> &pixel_point) const {
            Eigen::Vector2<Dtype> grid;
            grid[0] = pixel_point[0];
            grid[1] = m_map_shape_[1] - pixel_point[1];
            return grid;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        PixelToGridForPoints(const Eigen::Ref<const Eigen::Matrix2X<Index>> &pixel_points) const {
            return GridToPixelForPoints(pixel_points);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Index>>
        MeterToPixelForPoint(const Eigen::Ref<const Eigen::Vector2<Dtype>> &meter_point) const {
            return GridToPixelForPoint(MeterToGridForPoint(meter_point));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        MeterToPixelForPoints(const Eigen::Ref<const Eigen::Matrix2X<Dtype>> &meter_points) const {
            return GridToPixelForPoints(MeterToGridForPoints(meter_points));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Dtype>>
        PixelToMeterForPoint(const Eigen::Ref<const Eigen::Vector2<Index>> &pixel_point) const {
            return GridToMeterForPoint(PixelToGridForPoint(pixel_point));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector2<Dtype>>
        SubPixelToMeterForPoint(const Eigen::Ref<const Eigen::Vector2<Dtype>> &pixel_point) const {
            Eigen::Vector2<Dtype> meter;
            meter[0] = pixel_point[0] * m_resolution_[0] + m_min_[0];
            meter[1] = (m_map_shape_[1] - pixel_point[1]) * m_resolution_[1] + m_min_[1];
            return meter;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Dtype>>
        PixelToMeterForPoints(const Eigen::Ref<const Eigen::Matrix2X<Index>> &pixel_points) const {
            return GridToMeterForPoints(PixelToGridForPoints(pixel_points));
        }

        [[nodiscard]] Eigen::MatrixX<Dtype>
        GridToMeterForVectors(const Eigen::Ref<const Eigen::MatrixX<Index>> &grid_vectors) const {
            return grid_vectors.template cast<Dtype>().array().colwise() * m_resolution_.array();
        }

        [[nodiscard]] Eigen::MatrixX<Index>
        MeterToGridForVectors(const Eigen::Ref<const Eigen::MatrixX<Dtype>> &meter_vectors) const {
            return (meter_vectors.array().colwise() / m_resolution_.array())
                .floor()
                .template cast<Index>();
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        GridToPixelForVectors(const Eigen::Ref<const Eigen::Matrix2X<Index>> &grid_vectors) const {
            if (D == Eigen::Dynamic) { ERL_ASSERT_EQ(Dims(), 2); }

            Eigen::Matrix2X<Index> pixel_vectors(2, grid_vectors.cols());
            const long n_cols = grid_vectors.cols();
            for (long j = 0; j < n_cols; ++j) {
                pixel_vectors(0, j) = grid_vectors(0, j);
                pixel_vectors(1, j) = -grid_vectors(1, j);
            }

            return pixel_vectors;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        PixelToGridForVectors(const Eigen::Ref<const Eigen::Matrix2X<Index>> &pixel_vectors) const {
            return GridToPixelForVectors(pixel_vectors);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Index>>
        MeterToPixelForVectors(
            const Eigen::Ref<const Eigen::Matrix2X<Dtype>> &meter_vectors) const {
            return GridToPixelForVectors(MeterToGridForVectors(meter_vectors));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2X<Dtype>>
        PixelToMeterForVectors(
            const Eigen::Ref<const Eigen::Matrix2X<Index>> &pixel_vectors) const {
            return GridToMeterForVectors(PixelToGridForVectors(pixel_vectors));
        }

        [[nodiscard]] bool
        InMap(const Eigen::Ref<const Eigen::Vector<Dtype, Dim>> &meter_point) const {
            ERL_DEBUG_ASSERT(
                meter_point.size() == m_map_shape_.size(),
                "meter_point is {}-dim but the map is {}-dim.",
                meter_point.size(),
                m_map_shape_.size());
            for (long i = 0; i < meter_point.size(); ++i) {
                if (meter_point[i] < m_min_[i] || meter_point[i] > m_max_[i]) { return false; }
            }
            return true;
        }

        [[nodiscard]] bool
        InGrids(const Eigen::Ref<const Eigen::Vector<Index, Dim>> &grid_point) const {
            // clang-format off
            if (Dim == 2) {
                return grid_point[0] >= 0 && grid_point[0] < m_map_shape_[0] &&
                       grid_point[1] >= 0 && grid_point[1] < m_map_shape_[1];
            }
            if (Dim == 3) {
                return grid_point[0] >= 0 && grid_point[0] < m_map_shape_[0] &&
                       grid_point[1] >= 0 && grid_point[1] < m_map_shape_[1] &&
                       grid_point[2] >= 0 && grid_point[2] < m_map_shape_[2];
            }
            // clang-format on

            for (long i = 0; i < Dim; ++i) {
                if (grid_point[i] < 0 || grid_point[i] >= m_map_shape_[i]) { return false; }
            }
            return true;
        }

        [[nodiscard]] Index
        GridToIndex(const Eigen::Ref<const Eigen::Vector<Index, Dim>> &grid, bool c_stride) const {
            ERL_DEBUG_ASSERT(
                InGrids(grid),
                "{} is out of map.\n",
                EigenToNumPyFmtString(grid.transpose()));
            return CoordsToIndex<Index, Dim>(m_map_shape_, grid, c_stride);
        }

        [[nodiscard]] Eigen::Vector<Index, Dim>
        IndexToGrid(Index index, bool c_stride) const {
            return IndexToCoords<Dim>(m_map_shape_, index, c_stride);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Index>
        PixelToIndex(const Eigen::Ref<const Eigen::Vector<Index, D>> &pixel, const bool c_stride)
            const {
            return GridToIndex(PixelToGridForPoints(pixel), c_stride);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector<Index, D>>
        IndexToPixel(const Index index, const bool c_stride) const {
            return GridToPixelForPoints(IndexToGrid(index, c_stride));
        }

        [[nodiscard]] Eigen::Matrix<Index, Dim, Eigen::Dynamic>
        GenerateGridCoordinates(const bool c_stride) const {
            if (c_stride) { return CalculateGridCoordinates<Index, Dim, true>(m_map_shape_); }
            return CalculateGridCoordinates<Index, Dim, false>(m_map_shape_);
        }

        [[nodiscard]] Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>
        GenerateMeterCoordinates(const bool c_stride) const {
            if (c_stride) {
                return CalculateMeterCoordinates<Dtype, Index, Dim, true, true>(
                    m_map_shape_,
                    m_min_,
                    m_max_,
                    m_resolution_);
            }

            return CalculateMeterCoordinates<Dtype, Index, Dim, false, true>(
                m_map_shape_,
                m_min_,
                m_max_,
                m_resolution_);
        }

        [[nodiscard]] Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>
        GenerateVoxelVertices(const bool c_stride) const {
            const long n_dims = Dims();
            // compute the metric coordinates of voxel vertices
            Eigen::Vector<Index, Dim> vertex_grid_shape = Shape() + 1;  // even, not compatible
            Eigen::VectorX<Index> strides;
            if (c_stride) {
                strides = ComputeCStrides<Index>(vertex_grid_shape, 1);
            } else {
                strides = ComputeFStrides<Index>(vertex_grid_shape, 1);
            }
            const Index n_vertices = vertex_grid_shape.prod();
            Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> vertex_meter_coords(n_dims, n_vertices);
            for (long i = 0; i < n_dims; ++i) {
                const Index stride = strides[i];
                const Index dim_size = vertex_grid_shape[i];
                const Index n_copies = n_vertices / dim_size;
                Dtype min = Min(i);
                Dtype max = Max(i);
                Eigen::MatrixX<Dtype> coords = Eigen::VectorX<Dtype>::LinSpaced(dim_size, min, max)
                                                   .transpose()
                                                   .replicate(stride, n_copies / stride);
                Eigen::Map<Eigen::Matrix<Dtype, 1, Eigen::Dynamic>> coords_reshaped(
                    coords.data(),
                    1,
                    n_vertices);
                vertex_meter_coords.row(i) << coords_reshaped;
            }
            return vertex_meter_coords;
        }

#ifdef ERL_USE_OPENCV

        template<int D = Dim>
        [[nodiscard]] std::
            enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<Dtype, D, Eigen::Dynamic>>
            GetMetricCoordinatesOfFilledMetricPolygon(
                const Eigen::Ref<const Eigen::Matrix<Dtype, 2, Eigen::Dynamic>>
                    &polygon_metric_vertices) const {
            return PixelToMeterForPoints(
                GetPixelCoordinatesOfFilledMetricPolygon<D>(polygon_metric_vertices));
        }

        template<int D = Dim>
        [[nodiscard]] std::
            enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<Index, D, Eigen::Dynamic>>
            GetGridCoordinatesOfFilledMetricPolygon(
                const Eigen::Ref<const Eigen::Matrix<Dtype, 2, Eigen::Dynamic>>
                    &polygon_metric_vertices) const {
            return PixelToGridForPoints(
                GetPixelCoordinatesOfFilledMetricPolygon<D>(polygon_metric_vertices));
        }

        template<int D = Dim>
        [[nodiscard]] std::
            enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<Index, D, Eigen::Dynamic>>
            GetPixelCoordinatesOfFilledMetricPolygon(
                const Eigen::Ref<const Eigen::Matrix<Dtype, 2, Eigen::Dynamic>>
                    &polygon_metric_vertices) const {
            if (D == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(
                    polygon_metric_vertices.rows() == Dims(),
                    "polygon_metric_vertices is {}-dim but the map is {}-dim.",
                    polygon_metric_vertices.rows(),
                    Dims());
            }
            ERL_DEBUG_ASSERT(
                polygon_metric_vertices.cols() >= 3,
                "polygon_metric_vertices must have at least 3 vertices.");

            Index u_min = std::numeric_limits<Index>::max();
            Index u_max = -std::numeric_limits<Index>::max();
            Index v_min = std::numeric_limits<Index>::max();
            Index v_max = -std::numeric_limits<Index>::max();
            long num_vertices = polygon_metric_vertices.cols();
            Eigen::Matrix2X<Index> polygon_pixel_points(2, num_vertices);
            for (long i = 0; i < num_vertices; ++i) {
                polygon_pixel_points.col(i) =
                    MeterToPixelForPoints<D>(polygon_metric_vertices.col(i));
                const Index &u = polygon_pixel_points(0, i);
                const Index &v = polygon_pixel_points(1, i);
                u_min = std::min(u, u_min);
                u_max = std::max(u, u_max);
                v_min = std::min(v, v_min);
                v_max = std::max(v, v_max);
            }

            cv::Mat canvas(v_max - v_min + 1, u_max - u_min + 1, CV_8UC1, cv::Scalar(0));
            std::vector<std::vector<cv::Point>> polygon_points(1);
            auto &points = polygon_points[0];
            for (long i = 0; i < num_vertices; ++i) {  // make sure all points are in the canvas
                points.emplace_back(
                    polygon_pixel_points(0, i) - u_min,
                    polygon_pixel_points(1, i) - v_min);
            }
            cv::drawContours(canvas, polygon_points, 0, cv::Scalar(255), cv::FILLED, cv::LINE_8);

            polygon_pixel_points.conservativeResize(
                Eigen::NoChange,
                static_cast<long>(canvas.rows) * static_cast<long>(canvas.cols));
            num_vertices = 0;
            for (int i = 0; i < canvas.rows; ++i) {
                for (int j = 0; j < canvas.cols; ++j) {  // pick filled pixels
                    if (canvas.at<uint8_t>(i, j) == 255) {
                        polygon_pixel_points.col(num_vertices++) =
                            Eigen::Vector2<Index>(j + u_min, i + v_min);
                    }
                }
            }
            polygon_pixel_points.conservativeResize(Eigen::NoChange, num_vertices);

            return polygon_pixel_points;
        }
#endif

        [[nodiscard]] Eigen::MatrixX<Index>
        RayCasting(
            const Eigen::Ref<const Eigen::VectorX<Dtype>> &start,
            const Eigen::Ref<const Eigen::VectorX<Dtype>> &end) const {
            if (!InMap(start)) {
                ERL_WARN("start point ({}, {}, {}) is out of map.", start[0], start[1], start[2]);
                return {};
            }
            if (!InMap(end)) {
                ERL_WARN("end point ({}, {}, {}) is out of map.", end[0], end[1], end[2]);
                return {};
            }
            Eigen::VectorX<Index> cur_grid = MeterToGridForPoints(start);
            const Eigen::VectorX<Index> end_grid = MeterToGridForPoints(end);
            if (cur_grid == end_grid) { return Eigen::MatrixX<Index>(cur_grid); }

            // initialize
            const long dim = Dims();
            Eigen::VectorX<Dtype> direction = end - start;
            const Dtype length = direction.norm();
            direction /= length;

            // compute step direction
            Eigen::VectorX<Index> step(dim);
            for (long i = 0; i < dim; ++i) {
                if (direction[i] > 0) {
                    step[i] = 1;
                } else if (direction[i] < 0) {
                    step[i] = -1;
                } else {
                    step[i] = 0;
                }
            }
            if (step.isZero()) {
                ERL_WARN("Ray casting in direction (0, 0, 0) is impossible!");
                return {};
            }

            // compute t_max and t_delta
            Eigen::VectorX<Dtype> t_max(dim);
            Eigen::VectorX<Dtype> t_delta(dim);
            for (long i = 0; i < dim; ++i) {
                if (step[i] == 0) {
                    t_max[i] = std::numeric_limits<Dtype>::infinity();
                    t_delta[i] = std::numeric_limits<Dtype>::infinity();
                } else {
                    const Dtype voxel_border = GridToMeterAtDim(cur_grid[i], i) +
                                               static_cast<Dtype>(step[i]) * 0.5 * m_resolution_[i];
                    t_max[i] = (voxel_border - start[i]) / direction[i];
                    t_delta[i] = m_resolution_[i] / std::abs(direction[i]);
                }
            }

            // incremental phase
            Eigen::MatrixX<Index> points(
                dim,
                (3 + dim) * (end_grid - cur_grid).cwiseAbs().maxCoeff());
            long cnt = 0;
            points.col(cnt++) = cur_grid;
            while (true) {
                long min_dim = 0;
                for (long i = 1; i < dim; ++i) {
                    if (t_max[i] < t_max[min_dim]) { min_dim = i; }
                }

                t_max[min_dim] += t_delta[min_dim];
                cur_grid[min_dim] += step[min_dim];
                ERL_DEBUG_ASSERT(
                    cur_grid[min_dim] >= 0 && cur_grid[min_dim] < m_map_shape_[min_dim],
                    "cur_grid[{}] = {} is out of range [0, {})",
                    min_dim,
                    cur_grid[min_dim],
                    m_map_shape_[min_dim]);

                if (cur_grid == end_grid) {
                    points.col(cnt++) = cur_grid;
                    break;
                }

                // this seems to be unlikely to happen
                Dtype dist_from_origin = t_max.minCoeff();
                if (dist_from_origin > length) { break; }  // this happens due to numerical error
                points.col(cnt++) = cur_grid;
                ERL_ASSERT_LT(cnt, points.cols() - 1);
            }
            points.conservativeResize(Eigen::NoChange, cnt);
            return points;
        }
    };

    template<typename Dtype>
    using GridMapInfo2D = GridMapInfo<Dtype, 2>;

    template<typename Dtype>
    using GridMapInfo3D = GridMapInfo<Dtype, 3>;

    using GridMapInfo2Dd = GridMapInfo<double, 2>;
    using GridMapInfo2Df = GridMapInfo<float, 2>;
    using GridMapInfo3Dd = GridMapInfo<double, 3>;
    using GridMapInfo3Df = GridMapInfo<float, 3>;
    using GridMapInfoXDd = GridMapInfo<double, Eigen::Dynamic>;
    using GridMapInfoXDf = GridMapInfo<float, Eigen::Dynamic>;
}  // namespace erl::common
