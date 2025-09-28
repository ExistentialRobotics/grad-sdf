#pragma once

#include "eigen.hpp"
#include "logging.hpp"

#include <vector>

namespace erl::common {

    template<typename T>
    std::vector<T>
    ComputeCStrides(const std::vector<T> &shape, const T item_size) {
        const auto ndim = static_cast<T>(shape.size());
        std::vector<T> strides(ndim, item_size);
        for (T i = ndim - 1; i > 0; --i) { strides[i - 1] = strides[i] * shape[i]; }
        return strides;
    }

    template<typename T, int Dim>
    Eigen::VectorX<T>
    ComputeCStrides(const Eigen::Vector<T, Dim> &shape, const T item_size) {
        const auto ndim = Dim == Eigen::Dynamic ? static_cast<T>(shape.size()) : Dim;
        Eigen::VectorX<T> strides = Eigen::VectorX<T>::Constant(ndim, item_size);
        for (T i = ndim - 1; i > 0; --i) { strides[i - 1] = strides[i] * shape[i]; }
        return strides;
    }

    template<typename T>
    std::vector<T>
    ComputeFStrides(const std::vector<T> &shape, const T item_size) {
        const auto ndim = static_cast<T>(shape.size());
        std::vector<T> strides(ndim, item_size);
        for (T i = 1; i < ndim; ++i) { strides[i] = strides[i - 1] * shape[i - 1]; }
        return strides;
    }

    template<typename T, int Dim>
    Eigen::VectorX<T>
    ComputeFStrides(const Eigen::Vector<T, Dim> &shape, const T item_size) {
        const auto ndim = Dim == Eigen::Dynamic ? static_cast<T>(shape.size()) : Dim;
        Eigen::VectorX<T> strides = Eigen::VectorX<T>::Constant(ndim, item_size);
        for (T i = 1; i < ndim; ++i) { strides[i] = strides[i - 1] * shape[i - 1]; }
        return strides;
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] Dtype
    CoordsToIndex(
        const Eigen::Vector<Dtype, Dim> &shape,
        const Eigen::Vector<Dtype, Dim> &coords,
        const bool c_stride) {
        const auto ndim = static_cast<int>(shape.size());

        if (Dim == 2) {
            if (c_stride) { return coords[1] + coords[0] * shape[1]; }
            return coords[0] + coords[1] * shape[0];
        }

        if (Dim == 3) {
            if (c_stride) { return coords[2] + shape[2] * (coords[1] + shape[1] * coords[0]); }
            return coords[0] + shape[0] * (coords[1] + shape[1] * coords[2]);
        }

        if (Dim == 4) {
            if (c_stride) {
                return coords[3] +
                       shape[3] * (coords[2] + shape[2] * (coords[1] + shape[1] * coords[0]));
            }
            return coords[0] +
                   shape[0] * (coords[1] + shape[1] * (coords[2] + shape[2] * coords[3]));
        }

        if (c_stride) {
            int index = coords[0];
            for (int i = 1; i < ndim; ++i) { index = index * shape[i] + coords[i]; }
            return index;
        }

        int index = coords[ndim - 1];
        for (int i = ndim - 1; i > 0; --i) { index = index * shape[i - 1] + coords[i - 1]; }

        return index;
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] Dtype
    CoordsToIndex(
        const Eigen::Vector<Dtype, Dim> &strides,
        const Eigen::Vector<Dtype, Dim> &coords) {
        ERL_DEBUG_ASSERT((coords.array() >= 0).all(), "Coords must be non-negative.");
        return strides.dot(coords);
    }

    template<int Dim>
    [[nodiscard]] Eigen::Vector<int, Dim>
    IndexToCoords(const Eigen::Vector<int, Dim> &shape, int index, const bool c_stride) {
        const auto ndim = Dim == Eigen::Dynamic ? static_cast<int>(shape.size()) : Dim;
        Eigen::Vector<int, Dim> coords;
        coords.setZero(ndim);

        if (Dim == 2) {
            if (c_stride) {
                coords[0] = index / shape[1];
                coords[1] = index - coords[0] * shape[1];
            } else {
                coords[1] = index / shape[0];
                coords[0] = index - coords[1] * shape[0];
            }
            return coords;
        }

        if (Dim == 3) {
            if (c_stride) {  // coords[2] + shape[2] * (coords[1] + shape[1] * coords[0])
                int prod = shape[1] * shape[2];
                coords[0] = index / prod;
                index -= coords[0] * prod;
                coords[1] = index / shape[2];
                coords[2] = index - coords[1] * shape[2];
            } else {  // coords[0] + shape[0] * (coords[1] + shape[1] * coords[2])
                int prod = shape[0] * shape[1];
                coords[2] = index / prod;
                index -= coords[2] * prod;
                coords[1] = index / shape[0];
                coords[0] = index - coords[1] * shape[0];
            }
            return coords;
        }

        if (Dim == 4) {
            if (c_stride) {
                // coords[3] + shape[3] * (coords[2] + shape[2] *
                // (coords[1] + shape[1] * coords[0]))
                int prod_23 = shape[2] * shape[3];
                int prod_123 = shape[1] * prod_23;
                coords[0] = index / prod_123;
                index -= coords[0] * prod_123;
                coords[1] = index / prod_23;
                index -= coords[1] * prod_23;
                coords[2] = index / shape[3];
                coords[3] = index - coords[2] * shape[3];
            } else {
                // coords[0] + shape[0] * (coords[1] + shape[1] *
                // (coords[2] + shape[2] * coords[3]))
                int prod_12 = shape[1] * shape[2];
                int prod_012 = shape[0] * prod_12;
                coords[3] = index / prod_012;
                index -= coords[3] * prod_012;
                coords[2] = index / prod_12;
                index -= coords[2] * prod_12;
                coords[1] = index / shape[2];
                coords[0] = index - coords[1] * shape[2];
            }
            return coords;
        }

        if (c_stride) {
            for (int i = ndim - 1; i >= 0; --i) {
                coords[i] = index % shape[i];
                index -= coords[i];
                index /= shape[i];
            }
            return coords;
        }

        for (int i = 0; i < ndim; ++i) {
            coords[i] = index % shape[i];
            index -= coords[i];
            index /= shape[i];
        }
        return coords;
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] Eigen::Vector<Dtype, Dim>
    IndexToCoordsWithStrides(
        const Eigen::Vector<Dtype, Dim> &strides,
        Dtype index,
        const bool c_stride) {

        const auto ndim = Dim == Eigen::Dynamic ? static_cast<int>(strides.size()) : Dim;
        Eigen::Vector<Dtype, Dim> coords;
        coords.setZero(ndim);

        if (Dim == 2) {
            if (c_stride) {
                coords[0] = index / strides[0];
                coords[1] = index - coords[0] * strides[0];
            } else {
                coords[1] = index / strides[1];
                coords[0] = index - coords[1] * strides[1];
            }
            return coords;
        }

        if (Dim == 3) {
            if (c_stride) {  // (shape[1] * shape[2], shape[2], 1)
                coords[0] = index / strides[0];
                index -= coords[0] * strides[0];
                coords[1] = index / strides[1];
                coords[2] = index - coords[1] * strides[1];
            } else {  // (1, shape[0], shape[0] * shape[1])
                coords[2] = index / strides[2];
                index -= coords[2] * strides[2];
                coords[1] = index / strides[1];
                coords[0] = index - coords[1] * strides[1];
            }
            return coords;
        }

        if (Dim == 4) {
            if (c_stride) {
                coords[0] = index / strides[0];
                index -= coords[0] * strides[0];
                coords[1] = index / strides[1];
                index -= coords[1] * strides[1];
                coords[2] = index / strides[2];
                coords[3] = index - coords[2] * strides[2];
            } else {
                coords[3] = index / strides[3];
                index -= coords[3] * strides[3];
                coords[2] = index / strides[2];
                index -= coords[2] * strides[2];
                coords[1] = index / strides[1];
                coords[0] = index - coords[1] * strides[1];
            }
            return coords;
        }

        if (c_stride) {
            for (int i = 0; i < ndim; ++i) {
                coords[i] = index / strides[i];
                index -= coords[i] * strides[i];
            }
            return coords;
        }

        for (int i = ndim - 1; i >= 0; --i) {
            coords[i] = index / strides[i];
            index -= coords[i] * strides[i];
        }
        return coords;
    }
}  // namespace erl::common
