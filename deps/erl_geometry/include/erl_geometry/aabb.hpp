#pragma once

#include "erl_common/yaml.hpp"

#include <utility>

namespace erl::geometry {

    struct AabbBase {};

    template<typename ScalarType, int Dim>
    struct Aabb : Eigen::AlignedBox<ScalarType, Dim>, AabbBase {

        using Scalar = ScalarType;
        using Point = Eigen::Vector<Scalar, Dim>;

        Point center = {};
        Point half_sizes = {};

        Aabb() = default;

        Aabb(Point center, Scalar half_size)
            : Eigen::AlignedBox<Scalar, Dim>(
                  center.array() - half_size,
                  center.array() + half_size),
              center(std::move(center)),
              half_sizes(Point::Constant(half_size)) {}

        Aabb(const Point &min, const Point &max)
            : Eigen::AlignedBox<Scalar, Dim>(min, max),
              center((min + max) / 2),
              half_sizes((max - min) / 2) {}

        [[nodiscard]] Aabb
        Padding(const Point &padding) const {
            return {this->m_min - padding, this->m_max + padding};
        }

        [[nodiscard]] Aabb
        Padding(Scalar padding) const {
            return {this->m_min.array() - padding, this->m_max.array() + padding};
        }

        bool
        operator==(const Aabb &rhs) const {
            return center == rhs.center && half_sizes == rhs.half_sizes;
        }

        bool
        operator!=(const Aabb &rhs) const {
            return !(*this == rhs);
        }

        [[nodiscard]] bool
        IsValid() const {
            for (int i = 0; i < Dim; ++i) {
                if (half_sizes[i] < 0) { return false; }
            }
            return true;
        }

        [[nodiscard]] Aabb
        Intersection(const Aabb &rhs) const {
            return {this->m_min.cwiseMax(rhs.m_min), this->m_max.cwiseMin(rhs.m_max)};
        }

        template<typename Dtype>
        Aabb<Dtype, Dim>
        Cast() const {
            return {this->m_min.template cast<Dtype>(), this->m_max.template cast<Dtype>()};
        }
    };

    using Aabb2Dd = Aabb<double, 2>;
    using Aabb3Dd = Aabb<double, 3>;
    using Aabb2Df = Aabb<float, 2>;
    using Aabb3Df = Aabb<float, 3>;
}  // namespace erl::geometry

namespace YAML {
    template<typename AABB>
    struct ConvertAabb {
        static_assert(
            std::is_base_of_v<erl::geometry::AabbBase, AABB>,
            "AABB must be derived from AabbBase");

        static Node
        encode(const AABB &aabb) {
            Node node;
            ERL_YAML_SAVE_ATTR(node, aabb, center);
            ERL_YAML_SAVE_ATTR(node, aabb, half_sizes);
            return node;
        }

        static bool
        decode(const Node &node, AABB &aabb) {
            if (!node.IsMap()) { return false; }
            ERL_YAML_LOAD_ATTR(node, aabb, center);
            ERL_YAML_LOAD_ATTR(node, aabb, half_sizes);
            return true;
        }
    };

    template<>
    struct convert<erl::geometry::Aabb2Dd> : ConvertAabb<erl::geometry::Aabb2Dd> {};

    template<>
    struct convert<erl::geometry::Aabb3Dd> : ConvertAabb<erl::geometry::Aabb3Dd> {};

    template<>
    struct convert<erl::geometry::Aabb2Df> : ConvertAabb<erl::geometry::Aabb2Df> {};

    template<>
    struct convert<erl::geometry::Aabb3Df> : ConvertAabb<erl::geometry::Aabb3Df> {};
}  // namespace YAML
