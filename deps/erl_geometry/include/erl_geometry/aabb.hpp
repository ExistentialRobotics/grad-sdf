#pragma once

#include "erl_common/yaml.hpp"

#include <utility>

namespace erl::geometry {

    template<typename Dtype, int Dim>
    struct Aabb : Eigen::AlignedBox<Dtype, Dim> {

        using Scalar = Dtype;
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

        template<typename Dtype2>
        Aabb<Dtype2, Dim>
        Cast() const {
            return {this->m_min.template cast<Dtype2>(), this->m_max.template cast<Dtype2>()};
        }
    };

    using Aabb2Dd = Aabb<double, 2>;
    using Aabb3Dd = Aabb<double, 3>;
    using Aabb2Df = Aabb<float, 2>;
    using Aabb3Df = Aabb<float, 3>;
}  // namespace erl::geometry

template<typename Dtype, int Dim>
struct YAML::convert<erl::geometry::Aabb<Dtype, Dim>> {
    static Node
    encode(const erl::geometry::Aabb<Dtype, Dim> &aabb) {
        Node node;
        node["center"] = aabb.center;
        node["half_sizes"] = aabb.half_sizes;
        return node;
    }

    static bool
    decode(const Node &node, erl::geometry::Aabb<Dtype, Dim> &aabb) {
        if (!node.IsMap()) { return false; }
        using Point = typename erl::geometry::Aabb<Dtype, Dim>::Point;
        aabb.center = node["center"].as<Point>();
        aabb.half_sizes = node["half_sizes"].as<Point>();
        aabb = erl::geometry::Aabb<Dtype, Dim>(aabb.center, aabb.half_sizes);
        return true;
    }
};

#ifdef ERL_USE_BOOST
template<typename Dtype, int Dim>
struct erl::common::program_options::ParseOption<erl::geometry::Aabb<Dtype, Dim>>
    : ParseOptionBase {

    using T = erl::geometry::Aabb<Dtype, Dim>;

    ParseOption<typename T::Point> center_parser;
    ParseOption<typename T::Point> half_sizes_parser;

    ParseOption(std::string option_name_in, ProgramOptionsData *po_data_in, T *member_ptr_in)
        : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in),
          center_parser(GetBoostOptionName(option_name, "center"), po_data, &member_ptr_in->center),
          half_sizes_parser(
              GetBoostOptionName(option_name, "half_sizes"),
              po_data,
              &member_ptr_in->half_sizes) {}

    void
    Run() override {

        center_parser.Run();
        half_sizes_parser.Run();

        T &member = *static_cast<T *>(member_ptr);

        for (int i = 0; i < Dim; ++i) {
            ERL_ASSERTM(
                member.half_sizes[i] > 0,
                "Half size must be non-negative for {}.half_sizes[{}], got {}",
                option_name,
                i,
                member.half_sizes[i]);
        }

        member = erl::geometry::Aabb<Dtype, Dim>(member.center, member.half_sizes);
    }
};
#endif

#ifdef ERL_ROS_VERSION_1
template<typename Dtype, int Dim>
struct erl::common::ros_params::LoadRos1Param<erl::geometry::Aabb<Dtype, Dim>> {
    using Aabb = erl::geometry::Aabb<Dtype, Dim>;

    static void
    Run(ros::NodeHandle &nh, const std::string &param_name, Aabb &member) {
        using Point = typename Aabb::Point;
        LoadRos1Param<Point>::Run(nh, GetRos1ParamPath(param_name, "center"), member.center);
        LoadRos1Param<Point>::Run(
            nh,
            GetRos1ParamPath(param_name, "half_sizes"),
            member.half_sizes);
        member = Aabb(member.center, member.half_sizes);
    }
};
#endif

#ifdef ERL_ROS_VERSION_2
template<typename Dtype, int Dim>
struct erl::common::ros_params::LoadRos2Param<erl::geometry::Aabb<Dtype, Dim>> {
    using Aabb = erl::geometry::Aabb<Dtype, Dim>;

    static void
    Run(rclcpp::Node *node, const std::string &param_name, Aabb &member) {
        using Point = typename Aabb::Point;
        LoadRos2Param<Point>::Run(node, GetRos2ParamPath(param_name, "center"), member.center);
        LoadRos2Param<Point>::Run(
            node,
            GetRos2ParamPath(param_name, "half_sizes"),
            member.half_sizes);
        member = Aabb(member.center, member.half_sizes);
    }
};
#endif
