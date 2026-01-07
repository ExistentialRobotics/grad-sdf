#pragma once

#include "logging.hpp"

#include <filesystem>

#ifdef ERL_USE_OPENCV
    #include <opencv2/core.hpp>
#endif

#ifdef ERL_ROS_VERSION_1
    #include <ros/ros.h>

namespace erl::common::ros_params {
    // Functionality for loading ROS parameters

    [[nodiscard]] inline std::string
    GetRos1ParamPath(const std::string &prefix, const std::string &name) {
        if (prefix.empty()) {
            return name;
        } else if (prefix.back() == '/') {
            return prefix + name;
        } else {
            return prefix + "/" + name;
        }
    }

    template<typename T>
    struct LoadRos1Param {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, T &member) {
            nh.param<T>(param_name, member, member);
        }
    };

    template<>
    struct LoadRos1Param<long> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, long &member) {
            std::string value_str = std::to_string(member);
            if (!nh.param<std::string>(param_name, value_str, value_str)) { return; }
            member = std::stol(value_str);
        }
    };

    template<>
    struct LoadRos1Param<uint32_t> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, uint32_t &member) {
            std::string value_str = std::to_string(member);
            if (!nh.param<std::string>(param_name, value_str, value_str)) { return; }
            long temp = std::stol(value_str);
            if (temp < 0) {  // print warning but still assign the value
                ERL_WARN("Parameter {} has negative value {} for type uint32_t", param_name, temp);
            }
            member = static_cast<uint32_t>(temp);
        }
    };

    template<>
    struct LoadRos1Param<uint64_t> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, uint64_t &member) {
            std::string value_str = std::to_string(member);
            if (!nh.param<std::string>(param_name, value_str, value_str)) { return; }
            long long temp = std::stoll(value_str);
            if (temp < 0) {  // print warning but still assign the value
                ERL_WARN("Parameter {} has negative value {} for type uint64_t", param_name, temp);
            }
            member = static_cast<uint64_t>(temp);
        }
    };

    template<>
    struct LoadRos1Param<float> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, float &member) {
            double temp = static_cast<double>(member);
            if (!nh.param<double>(param_name, temp, temp)) { return; }
            if (temp < -std::numeric_limits<float>::max() ||
                temp > std::numeric_limits<float>::max()) {
                ERL_WARN(
                    "Parameter {} has value {} outside the range of float type",
                    param_name,
                    temp);
            }
            member = static_cast<float>(temp);
        }
    };

    template<>
    struct LoadRos1Param<std::filesystem::path> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, std::filesystem::path &member) {
            std::string path_str = member.string();
            if (!nh.param<std::string>(param_name, path_str, path_str)) { return; }
            member = std::filesystem::path(path_str);
        }
    };

    template<>
    struct LoadRos1Param<std::vector<long>> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, std::vector<long> &member) {
            std::vector<std::string> temp;
            temp.reserve(member.size());
            std::transform(member.begin(), member.end(), temp.begin(), [](long v) {
                return std::to_string(v);
            });
            if (!nh.param<std::vector<std::string>>(param_name, temp, temp)) { return; }
            member.resize(temp.size());
            for (std::size_t i = 0; i < temp.size(); ++i) { member[i] = std::stol(temp[i]); }
        }
    };

    template<>
    struct LoadRos1Param<std::vector<uint64_t>> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, std::vector<uint64_t> &member) {
            std::vector<std::string> temp;
            temp.reserve(member.size());
            std::transform(member.begin(), member.end(), temp.begin(), [](uint64_t v) {
                return std::to_string(v);
            });
            if (!nh.param<std::vector<std::string>>(param_name, temp, temp)) { return; }
            member.resize(temp.size());
            for (std::size_t i = 0; i < temp.size(); ++i) {
                member[i] = static_cast<uint64_t>(std::stoull(temp[i]));
            }
        }
    };

    template<>
    struct LoadRos1Param<std::vector<float>> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, std::vector<float> &member) {
            std::vector<double> temp;
            temp.reserve(member.size());
            std::transform(member.begin(), member.end(), temp.begin(), [](float v) {
                return static_cast<double>(v);
            });
            if (!nh.param<std::vector<double>>(param_name, temp, temp)) { return; }
            member.resize(temp.size());
            for (std::size_t i = 0; i < temp.size(); ++i) {
                member[i] = static_cast<float>(temp[i]);
            }
        }
    };

    template<typename T1, typename T2>
    struct LoadRos1Param<std::pair<T1, T2>> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, std::pair<T1, T2> &member) {
            using namespace erl::common::ros_params;
            std::string first_param = GetRos1ParamPath(param_name, "first");
            std::string second_param = GetRos1ParamPath(param_name, "second");
            LoadRos1Param<T1>::Run(nh, first_param, member.first);
            LoadRos1Param<T2>::Run(nh, second_param, member.second);
        }
    };

    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct LoadRos1Param<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
        using Mat = Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>;

        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, Mat &member) {
            long n_rows = Rows_;
            long n_cols = Cols_;
            if (Rows_ == Eigen::Dynamic) {
                LoadRos1Param<long>::Run(nh, GetRos1ParamPath(param_name, "rows"), n_rows);
            }
            if (Cols_ == Eigen::Dynamic) {
                LoadRos1Param<long>::Run(nh, GetRos1ParamPath(param_name, "cols"), n_cols);
            }

            std::vector<Scalar_> values;
            LoadRos1Param<std::vector<Scalar_>>::Run(nh, param_name, values);
            if (values.empty()) { return; }

            if (Rows_ > 0 && Cols_ > 0) {
                ERL_ASSERTM(
                    values.size() == static_cast<std::size_t>(Rows_ * Cols_),
                    "expecting {} values for param {}, got {}",
                    Rows_ * Cols_,
                    param_name,
                    values.size());
                member = Eigen::Map<Mat>(values.data(), Rows_, Cols_);
                return;
            }
            if (Rows_ > 0) {
                ERL_ASSERTM(
                    values.size() % Rows_ == 0,
                    "expecting multiple of {} values for param {} ({} x -1), got {}",
                    Rows_,
                    param_name,
                    Rows_,
                    values.size());
                const int cols = static_cast<int>(values.size()) / Rows_;
                ERL_ASSERTM(
                    n_cols <= 0 || cols == n_cols,
                    "mismatched number of columns: {} vs {}",
                    cols,
                    n_cols);
                member.resize(Rows_, cols);
                member = Eigen::Map<Mat>(values.data(), Rows_, cols);
                return;
            }
            if (Cols_ > 0) {
                ERL_ASSERTM(
                    values.size() % Cols_ == 0,
                    "expecting multiple of {} values for param {} (-1 x {}), got {}",
                    Cols_,
                    param_name,
                    Cols_,
                    values.size());
                const int rows = static_cast<int>(values.size()) / Cols_;
                ERL_ASSERTM(
                    n_rows <= 0 || rows == n_rows,
                    "mismatched number of rows: {} vs {}",
                    rows,
                    n_rows);
                member.resize(rows, Cols_);
                member = Eigen::Map<Mat>(values.data(), rows, Cols_);
                return;
            }
            ERL_ASSERTM(
                n_rows > 0 && n_cols > 0,
                "For param {} with fully dynamic size (-1 x -1), both rows and cols must be "
                "specified and > 0",
                param_name);
            const int size = static_cast<int>(values.size());
            ERL_ASSERTM(
                size == n_rows * n_cols,
                "expecting {} values for param {} ({} x {}), got {}",
                n_rows * n_cols,
                param_name,
                n_rows,
                n_cols,
                size);
            member.resize(n_rows, n_cols);
            member = Eigen::Map<Mat>(values.data(), n_rows, n_cols);
            return;
        }
    };

    #ifdef ERL_USE_OPENCV

    template<>
    struct LoadRos1Param<cv::Scalar> {
        static void
        Run(ros::NodeHandle &nh, const std::string &param_name, cv::Scalar &member) {
            std::vector<double> values;
            if (!nh.param<std::vector<double>>(param_name, values, values)) { return; }
            if (values.empty()) { return; }
            ERL_ASSERTM(
                values.size() <= 4,
                "expecting up to 4 values for param {}, got {}",
                param_name,
                values.size());
            for (std::size_t i = 0; i < values.size(); ++i) { member[i] = values[i]; }
        }
    };

    #endif

}  // namespace erl::common::ros_params

    #define ERL_LOAD_ROS1_PARAM_ENUM(T)                                                   \
        template<>                                                                        \
        struct erl::common::ros_params::LoadRos1Param<T> {                                \
            static void                                                                   \
            Run(ros::NodeHandle &nh, const std::string &param_name, T &member) {          \
                using yaml_convert = YAML::convert<T>;                                    \
                std::string value_str = yaml_convert::encode(member).as<std::string>();   \
                if (!nh.param<std::string>(param_name, value_str, value_str)) { return; } \
                ERL_ASSERT(yaml_convert::decode(YAML::Node(value_str), member));          \
            }                                                                             \
        }
#else
    #define ERL_LOAD_ROS1_PARAM_ENUM(T)
#endif

#ifdef ERL_ROS_VERSION_2
    #include <rclcpp/rclcpp.hpp>

namespace erl::common::ros_params {
    // Functionality for loading ROS2 parameters

    [[nodiscard]] inline std::string
    GetRos2ParamPath(const std::string &prefix, const std::string &name) {
        if (prefix.empty()) { return name; }
        return prefix + "." + name;
    }

    template<typename T>
    struct LoadRos2Param {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, T &member) {
            node->declare_parameter<T>(param_name, member);
            node->get_parameter<T>(param_name, member);
        }
    };

    template<>
    struct LoadRos2Param<uint32_t> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, uint32_t &member) {
            std::string value_str = std::to_string(member);
            node->declare_parameter<std::string>(param_name, value_str);
            if (!node->get_parameter(param_name, value_str)) { return; }
            long temp = std::stol(value_str);
            if (temp < 0) {  // print warning but still assign the value
                ERL_WARN("Parameter {} has negative value {} for type uint32_t", param_name, temp);
            }
            member = static_cast<uint32_t>(temp);
        }
    };

    template<>
    struct LoadRos2Param<uint64_t> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, uint64_t &member) {
            std::string value_str = std::to_string(member);
            node->declare_parameter<std::string>(param_name, value_str);
            if (!node->get_parameter(param_name, value_str)) { return; }
            long long temp = std::stoll(value_str);
            if (temp < 0) {  // print warning but still assign the value
                ERL_WARN("Parameter {} has negative value {} for type uint64_t", param_name, temp);
            }
            member = static_cast<uint64_t>(temp);
        }
    };

    template<>
    struct LoadRos2Param<std::filesystem::path> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, std::filesystem::path &member) {
            std::string path_str = member.string();
            node->declare_parameter<std::string>(param_name, path_str);
            if (!node->get_parameter(param_name, path_str)) { return; }
            member = std::filesystem::path(path_str);
        }
    };

    template<>
    struct LoadRos2Param<std::vector<float>> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, std::vector<float> &member) {
            std::vector<double> temp;
            temp.reserve(member.size());
            std::transform(member.begin(), member.end(), temp.begin(), [](float v) {
                return static_cast<double>(v);
            });
            node->declare_parameter<std::vector<double>>(param_name, temp);
            if (!node->get_parameter(param_name, temp)) { return; }
            member.resize(temp.size());
            for (std::size_t i = 0; i < temp.size(); ++i) {
                member[i] = static_cast<float>(temp[i]);
            }
        }
    };

    template<typename T1, typename T2>
    struct LoadRos2Param<std::pair<T1, T2>> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, std::pair<T1, T2> &member) {
            using namespace erl::common::ros_params;
            std::string first_param = GetRos2ParamPath(param_name, "first");
            std::string second_param = GetRos2ParamPath(param_name, "second");
            LoadRos2Param<T1>::Run(node, first_param, member.first);
            LoadRos2Param<T2>::Run(node, second_param, member.second);
        }
    };

    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct LoadRos2Param<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
        using Mat = Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>;

        static void
        Run(rclcpp::Node *node, const std::string &param_name, Mat &member) {
            long n_rows = Rows_;
            long n_cols = Cols_;
            if (Rows_ == Eigen::Dynamic) {
                LoadRos2Param<long>::Run(node, GetRos2ParamPath(param_name, "rows"), n_rows);
            }
            if (Cols_ == Eigen::Dynamic) {
                LoadRos2Param<long>::Run(node, GetRos2ParamPath(param_name, "cols"), n_cols);
            }

            std::vector<Scalar_> values;
            LoadRos2Param<std::vector<Scalar_>>::Run(node, param_name, values);
            if (values.empty()) { return; }

            if (Rows_ > 0 && Cols_ > 0) {
                ERL_ASSERTM(
                    values.size() == static_cast<std::size_t>(Rows_ * Cols_),
                    "expecting {} values for param {}, got {}",
                    Rows_ * Cols_,
                    param_name,
                    values.size());
                member = Eigen::Map<Mat>(values.data(), Rows_, Cols_);
                return;
            }
            if (Rows_ > 0) {
                ERL_ASSERTM(
                    values.size() % Rows_ == 0,
                    "expecting multiple of {} values for param {} ({} x -1), got {}",
                    Rows_,
                    param_name,
                    Rows_,
                    values.size());
                const int cols = static_cast<int>(values.size()) / Rows_;
                ERL_ASSERTM(
                    n_cols <= 0 || cols == n_cols,
                    "mismatched number of columns: {} vs {}",
                    cols,
                    n_cols);
                member.resize(Rows_, cols);
                member = Eigen::Map<Mat>(values.data(), Rows_, cols);
                return;
            }
            if (Cols_ > 0) {
                ERL_ASSERTM(
                    values.size() % Cols_ == 0,
                    "expecting multiple of {} values for param {} (-1 x {}), got {}",
                    Cols_,
                    param_name,
                    Cols_,
                    values.size());
                const int rows = static_cast<int>(values.size()) / Cols_;
                ERL_ASSERTM(
                    n_rows <= 0 || rows == n_rows,
                    "mismatched number of rows: {} vs {}",
                    rows,
                    n_rows);
                member.resize(rows, Cols_);
                member = Eigen::Map<Mat>(values.data(), rows, Cols_);
                return;
            }
            ERL_ASSERTM(
                n_rows > 0 && n_cols > 0,
                "For param {} with fully dynamic size (-1 x -1), both rows and cols must be "
                "specified and > 0",
                param_name);
            const int size = static_cast<int>(values.size());
            ERL_ASSERTM(
                size == n_rows * n_cols,
                "expecting {} values for param {} ({} x {}), got {}",
                n_rows * n_cols,
                param_name,
                n_rows,
                n_cols,
                size);
            member.resize(n_rows, n_cols);
            member = Eigen::Map<Mat>(values.data(), n_rows, n_cols);
            return;
        }
    };

    #ifdef ERL_USE_OPENCV

    template<>
    struct LoadRos2Param<cv::Scalar> {
        static void
        Run(rclcpp::Node *node, const std::string &param_name, cv::Scalar &member) {
            std::vector<double> values;
            node->declare_parameter<std::vector<double>>(param_name, values);
            if (!node->get_parameter<std::vector<double>>(param_name, values)) { return; }
            if (values.empty()) { return; }
            ERL_ASSERTM(
                values.size() <= 4,
                "expecting up to 4 values for param {}, got {}",
                param_name,
                values.size());
            for (std::size_t i = 0; i < values.size(); ++i) { member[i] = values[i]; }
        }
    };

    #endif

}  // namespace erl::common::ros_params

    #define ERL_LOAD_ROS2_PARAM_ENUM(T)                                                   \
        template<>                                                                        \
        struct erl::common::ros_params::LoadRos2Param<T> {                                \
            static void                                                                   \
            Run(rclcpp::Node *node, const std::string &param_name, T &member) {           \
                using yaml_convert = YAML::convert<T>;                                    \
                std::string value_str = yaml_convert::encode(member).as<std::string>();   \
                node->declare_parameter<std::string>(param_name, value_str);              \
                if (!node->get_parameter<std::string>(param_name, value_str)) { return; } \
                ERL_ASSERT(yaml_convert::decode(YAML::Node(value_str), member));          \
            }                                                                             \
        }
#else
    #define ERL_LOAD_ROS2_PARAM_ENUM(T)
#endif
