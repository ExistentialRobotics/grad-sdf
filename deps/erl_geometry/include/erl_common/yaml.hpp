#pragma once

#include "eigen.hpp"
#include "factory_pattern.hpp"
#include "logging.hpp"
#include "opencv.hpp"
#include "template_helper.hpp"
#include "version_check.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <optional>

// https://yaml.org/spec/1.2.2/
// https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started

template<>
struct YAML::convert<std::filesystem::path> {
    static Node
    encode(const std::filesystem::path& path) {
        return Node(path.string());
    }

    static bool
    decode(const Node& node, std::filesystem::path& path) {
        path = node.as<std::string>();
        return true;
    }
};

namespace erl::common {

    struct YamlableBase {
        using Factory = FactoryPattern<YamlableBase>;

        virtual ~YamlableBase() = default;

        template<typename Derived>
        static bool
        Register(const std::string& yamlable_type = "") {
            return Factory::GetInstance().Register<Derived>(yamlable_type, [] {
                return std::make_shared<Derived>();
            });
        }

        template<typename Derived>
        static std::shared_ptr<Derived>
        Create(const std::string& yamlable_type) {
            return std::dynamic_pointer_cast<Derived>(Factory::GetInstance().Create(yamlable_type));
        }

        [[nodiscard]] bool
        operator==(const YamlableBase& other) const;

        [[nodiscard]] bool
        operator!=(const YamlableBase& other) const;

        [[nodiscard]] virtual bool
        FromYamlNode(const YAML::Node& node) = 0;

        [[nodiscard]] bool
        FromYamlString(const std::string& yaml_string);

        [[nodiscard]] virtual YAML::Node
        AsYamlNode() const = 0;

        [[nodiscard]] virtual std::string
        AsYamlString() const;

        [[nodiscard]] bool
        FromYamlFile(const std::string& yaml_file);

        void
        AsYamlFile(const std::string& yaml_file) const;

        [[nodiscard]] bool
        Write(std::ostream& s) const;

        [[nodiscard]] bool
        Read(std::istream& s);

        void
        FromCommandLine(int argc, const char* argv[]);
    };

    template<typename T, typename Base = YamlableBase>
    struct Yamlable : Base {

        [[nodiscard]] bool
        FromYamlNode(const YAML::Node& node) override {
            return YAML::convert<T>::decode(node, *static_cast<T*>(this));
        }

        [[nodiscard]] YAML::Node
        AsYamlNode() const override {
            return YAML::convert<T>::encode(*static_cast<const T*>(this));
        }
    };
}  // namespace erl::common

template<
    typename T,
    std::enable_if_t<
        IsSmartPtr<T>::value &&
            std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
void
SaveToNode(YAML::Node& node, const char* name, const T& obj) {
    if (obj == nullptr) {
        node[name] = YAML::Node(YAML::NodeType::Null);
        return;
    }
    node[name] = obj->AsYamlNode();
}

template<
    typename T,
    std::enable_if_t<
        IsSmartPtr<T>::value &&
            !std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
void
SaveToNode(YAML::Node& node, const char* name, const T& obj) {
    node[name] = obj;  // this will try to call YAML::convert<T>::encode
}

template<
    typename T,
    std::enable_if_t<
        IsWeakPtr<T>::value &&
            std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
void
SaveToNode(YAML::Node& node, const char* name, const T& obj) {
    if (obj == nullptr) {
        node[name] = YAML::Node(YAML::NodeType::Null);
        return;
    }
    ERL_DEBUG_ASSERT(!obj.expired(), "Weak pointer is expired, cannot save to YAML node.");
    node[name] = obj.lock()->AsYamlNode();
}

template<
    typename T,
    std::enable_if_t<std::is_base_of_v<erl::common::YamlableBase, T>, void>* = nullptr>
void
SaveToNode(YAML::Node& node, const char* name, const T& obj) {
    node[name] = obj.AsYamlNode();
}

template<
    typename T,
    std::enable_if_t<
        // don't allow raw pointers additionally. if T is a raw pointer, no SaveToNode will work.
        !IsSmartPtr<T>::value && !IsWeakPtr<T>::value && !std::is_pointer_v<T> &&
            !std::is_base_of_v<erl::common::YamlableBase, T>,
        void>* = nullptr>
void
SaveToNode(YAML::Node& node, const char* name, const T& obj) {
    node[name] = obj;
}

template<
    typename T,
    std::enable_if_t<
        IsSmartPtr<T>::value &&
            std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
[[nodiscard]] bool
LoadFromNode(const YAML::Node& node, const char* name, const T& obj) {
    // we should not call LoadFromNode(node, name, *obj) here.
    // otherwise, we might slice the object by mistake.
    // we cannot call `obj = node[name].as<T>()` here, which might create a new object of wrong type
    // because `T` is a smart pointer, and we don't know the exact derived type we need to load.
    // here `obj` must point to a valid object created earlier when the exact type is known.
    ERL_DEBUG_ASSERT(obj != nullptr, "Smart pointer is null, cannot load from YAML node.");
    return obj->FromYamlNode(node[name]);
}

template<
    typename T,
    std::enable_if_t<
        IsSmartPtr<T>::value &&
            !std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
void
LoadFromNode(const YAML::Node& node, const char* name, T& obj) {
    // we assume T::element_type is the exact type we want to load.
    // we might get nullptr here if `node[name]` is null.
    obj = node[name].as<T>();
}

template<
    typename T,
    std::enable_if_t<
        IsWeakPtr<T>::value &&
            std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
[[nodiscard]] bool
LoadFromNode(const YAML::Node& node, const char* name, const T& obj) {
    ERL_DEBUG_ASSERT(obj != nullptr, "Weak pointer is null, cannot load from YAML node.");
    ERL_DEBUG_ASSERT(!obj.expired(), "Weak pointer is expired, cannot load from YAML node.");
    return obj.lock()->FromYamlNode(node[name]);
}

template<
    typename T,
    std::enable_if_t<
        IsWeakPtr<T>::value &&
            !std::is_base_of_v<erl::common::YamlableBase, typename T::element_type>,
        void>* = nullptr>
void
LoadFromNode(const YAML::Node& node, const char* name, T& obj) {
    ERL_DEBUG_ASSERT(obj != nullptr, "Weak pointer is null, cannot load from YAML node.");
    ERL_DEBUG_ASSERT(!obj.expired(), "Weak pointer is expired, cannot load from YAML node.");
    auto& locked_obj = *obj.lock();
    locked_obj = node[name].as<decltype(locked_obj)>();
}

template<
    typename T,
    std::enable_if_t<std::is_base_of_v<erl::common::YamlableBase, T>, void>* = nullptr>
[[nodiscard]] bool
LoadFromNode(const YAML::Node& node, const char* name, T& obj) {
    return obj.FromYamlNode(node[name]);
}

template<typename T, std::enable_if_t<std::is_floating_point_v<T>, void>* = nullptr>
void
LoadFromNode(const YAML::Node& node, const char* name, T& obj) {
#if ERL_CHECK_VERSION_GE(   \
    YAML_CPP_VERSION_MAJOR, \
    YAML_CPP_VERSION_MINOR, \
    YAML_CPP_VERSION_PATCH, \
    0,                      \
    6,                      \
    3)
    obj = node[name].as<T>();
#else
    // YAML 0.6.2 fails to parse "inf".
    obj = static_cast<T>(std::stod(node[name].as<std::string>()));
#endif
}

template<
    typename T,
    std::enable_if_t<
        // don't allow raw pointers additionally. if T is a raw pointer, no LoadFromNode will work.
        !IsSmartPtr<T>::value && !IsWeakPtr<T>::value && !std::is_pointer_v<T> &&
            !std::is_base_of_v<erl::common::YamlableBase, T> && !std::is_floating_point_v<T>,
        void>* = nullptr>
void
LoadFromNode(const YAML::Node& node, const char* name, T& obj) {
    obj = node[name].as<T>();
}

#define ERL_YAML_SAVE_ATTR(node, obj, attr) SaveToNode(node, #attr, (obj).attr)
#define ERL_YAML_LOAD_ATTR(node, obj, attr) LoadFromNode(node, #attr, (obj).attr)

namespace YAML {
    template<
        typename T,
        int Rows = Eigen::Dynamic,
        int Cols = Eigen::Dynamic,
        int Order = Eigen::ColMajor>
    struct ConvertEigenMatrix {
        static Node
        encode(const Eigen::Matrix<T, Rows, Cols, Order>& rhs) {
            Node node(NodeType::Sequence);
            const int rows = Rows == Eigen::Dynamic ? rhs.rows() : Rows;
            const int cols = Cols == Eigen::Dynamic ? rhs.cols() : Cols;

            for (int i = 0; i < rows; ++i) {
                Node row_node(NodeType::Sequence);
                for (int j = 0; j < cols; ++j) { row_node.push_back(rhs(i, j)); }
                node.push_back(row_node);
            }

            return node;
        }

        static bool
        decode(const Node& node, Eigen::Matrix<T, Rows, Cols, Order>& rhs) {
            if (node.IsNull() && (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic)) {
                return true;
            }
            if (!node.IsSequence()) { return false; }
            if (!node[0].IsSequence()) { return false; }

            int rows = Rows == Eigen::Dynamic ? node.size() : Rows;
            int cols = Cols == Eigen::Dynamic ? node[0].size() : Cols;
            rhs.resize(rows, cols);
            ERL_DEBUG_ASSERT(
                rows == static_cast<int>(node.size()),
                "expecting rows: {}, get node.size(): {}",
                rows,
                node.size());
            for (int i = 0; i < rows; ++i) {
                ERL_DEBUG_ASSERT(
                    cols == static_cast<int>(node[i].size()),
                    "expecting cols: {}, get node[0].size(): {}",
                    cols,
                    node[i].size());
                auto& row_node = node[i];
                for (int j = 0; j < cols; ++j) { rhs(i, j) = row_node[j].as<T>(); }
            }

            return true;
        }
    };

    template<>
    struct convert<Eigen::Matrix2i> : ConvertEigenMatrix<int, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2f> : ConvertEigenMatrix<float, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2d> : ConvertEigenMatrix<double, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xi> : ConvertEigenMatrix<int, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xf> : ConvertEigenMatrix<float, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xd> : ConvertEigenMatrix<double, 2> {};

    template<>
    struct convert<Eigen::MatrixX2i> : ConvertEigenMatrix<int, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::MatrixX2f> : ConvertEigenMatrix<float, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::MatrixX2d> : ConvertEigenMatrix<double, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::Matrix3i> : ConvertEigenMatrix<int, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3f> : ConvertEigenMatrix<float, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3d> : ConvertEigenMatrix<double, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xi> : ConvertEigenMatrix<int, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xf> : ConvertEigenMatrix<float, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xd> : ConvertEigenMatrix<double, 3> {};

    template<>
    struct convert<Eigen::MatrixX3i> : ConvertEigenMatrix<int, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::MatrixX3f> : ConvertEigenMatrix<float, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::MatrixX3d> : ConvertEigenMatrix<double, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::Matrix4i> : ConvertEigenMatrix<int, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4f> : ConvertEigenMatrix<float, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4d> : ConvertEigenMatrix<double, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xi> : ConvertEigenMatrix<int, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xf> : ConvertEigenMatrix<float, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xd> : ConvertEigenMatrix<double, 4> {};

    template<>
    struct convert<Eigen::MatrixX4i> : ConvertEigenMatrix<int, Eigen::Dynamic, 4> {};

    template<>
    struct convert<Eigen::MatrixX4f> : ConvertEigenMatrix<float, Eigen::Dynamic, 4> {};

    template<>
    struct convert<Eigen::MatrixX4d> : ConvertEigenMatrix<double, Eigen::Dynamic, 4> {};

    template<>
    struct convert<Eigen::Matrix23f> : ConvertEigenMatrix<float, 2, 3> {};

    template<>
    struct convert<Eigen::Matrix23d> : ConvertEigenMatrix<double, 2, 3> {};

    template<>
    struct convert<Eigen::Matrix34f> : ConvertEigenMatrix<float, 3, 4> {};

    template<>
    struct convert<Eigen::Matrix34d> : ConvertEigenMatrix<double, 3, 4> {};

    template<typename T, int Size = Eigen::Dynamic>
    struct ConvertEigenVector {
        static Node
        encode(const Eigen::Vector<T, Size>& rhs) {
            Node node(NodeType::Sequence);
            if (Size == Eigen::Dynamic) {
                for (int i = 0; i < rhs.size(); ++i) { node.push_back(rhs[i]); }
            } else {
                for (int i = 0; i < Size; ++i) { node.push_back(rhs[i]); }
            }
            return node;
        }

        static bool
        decode(const Node& node, Eigen::Vector<T, Size>& rhs) {
            if (!node.IsSequence()) { return false; }
            if (Size == Eigen::Dynamic) {
                rhs.resize(node.size());
                for (int i = 0; i < rhs.size(); ++i) { rhs[i] = node[i].as<T>(); }
            } else {
                for (int i = 0; i < Size; ++i) { rhs[i] = node[i].as<T>(); }
            }
            return true;
        }
    };

    template<>
    struct convert<Eigen::VectorXd> : ConvertEigenVector<double> {};

    template<>
    struct convert<Eigen::Vector2d> : ConvertEigenVector<double, 2> {};

    template<>
    struct convert<Eigen::Vector3d> : ConvertEigenVector<double, 3> {};

    template<>
    struct convert<Eigen::Vector4d> : ConvertEigenVector<double, 4> {};

    template<>
    struct convert<Eigen::VectorXf> : ConvertEigenVector<float> {};

    template<>
    struct convert<Eigen::Vector2f> : ConvertEigenVector<float, 2> {};

    template<>
    struct convert<Eigen::Vector3f> : ConvertEigenVector<float, 3> {};

    template<>
    struct convert<Eigen::Vector4f> : ConvertEigenVector<float, 4> {};

    template<>
    struct convert<Eigen::VectorXi> : ConvertEigenVector<int> {};

    template<>
    struct convert<Eigen::Vector2i> : ConvertEigenVector<int, 2> {};

    template<>
    struct convert<Eigen::Vector3i> : ConvertEigenVector<int, 3> {};

    template<>
    struct convert<Eigen::Vector4i> : ConvertEigenVector<int, 4> {};

    template<>
    struct convert<Eigen::VectorXl> : ConvertEigenVector<long> {};

    template<>
    struct convert<Eigen::Vector2l> : ConvertEigenVector<long, 2> {};

    template<typename... Args>
    struct convert<std::tuple<Args...>> {
        static Node
        encode(const std::tuple<Args...>& rhs) {
            Node node(NodeType::Sequence);
            std::apply(
                [&node](const Args&... args) {
                    (node.push_back(convert<Args>::encode(args)), ...);
                },
                rhs);
            return node;
        }

        static bool
        decode(const Node& node, std::tuple<Args...>& rhs) {
            if (!node.IsSequence()) { return false; }
            if (node.size() != sizeof...(Args)) { return false; }
            std::apply([&node](Args&... args) { (convert<Args>::decode(node, args) && ...); }, rhs);
            return true;
        }
    };

    template<typename T>
    struct convert<std::optional<T>> {
        static Node
        encode(const std::optional<T>& rhs) {
            if (rhs) { return convert<T>::encode(*rhs); }
            return Node(NodeType::Null);
        }

        static bool
        decode(const Node& node, std::optional<T>& rhs) {
            if (node.Type() != NodeType::Null) {
                T value;
                if (convert<T>::decode(node, value)) {
                    rhs = value;
                    return true;
                }
                return false;
            }
            rhs = std::nullopt;
            return true;
        }
    };

    template<typename T>
    struct convert<std::shared_ptr<T>> {
        static Node
        encode(const std::shared_ptr<T>& rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node& node, std::shared_ptr<T>& rhs) {
            if (node.IsNull()) {
                rhs = nullptr;
                return true;
            }
            auto value = std::make_shared<T>();
            if (convert<T>::decode(node, *value)) {
                rhs = value;
                return true;
            }
            return false;
        }
    };

    template<typename T>
    struct convert<std::unique_ptr<T>> {
        static Node
        encode(const std::unique_ptr<T>& rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node& node, std::unique_ptr<T>& rhs) {
            if (node.IsNull()) {
                rhs = nullptr;
                return true;
            }
            auto value = std::make_unique<T>();
            if (convert<T>::decode(node, *value)) {
                rhs = std::move(value);
                return true;
            }
            return false;
        }
    };

    template<typename KeyType, typename ValueType>
    struct convert<std::unordered_map<KeyType, ValueType>> {
        static Node
        encode(const std::unordered_map<KeyType, ValueType>& rhs) {
            Node node(NodeType::Map);
            for (const auto& [key, value]: rhs) {
                node[convert<KeyType>::encode(key)] = convert<ValueType>::encode(value);
            }
            return node;
        }

        static bool
        decode(const Node& node, std::unordered_map<KeyType, ValueType>& rhs) {
            if (!node.IsMap()) { return false; }
            for (auto it = node.begin(); it != node.end(); ++it) {
                KeyType key;
                ValueType value;
                if (convert<KeyType>::decode(it->first, key) &&
                    convert<ValueType>::decode(it->second, value)) {
                    rhs[key] = value;
                } else {
                    return false;
                }
            }
            return true;
        }
    };

    template<typename Period>
    struct convert<std::chrono::duration<int64_t, Period>> {
        static Node
        encode(const std::chrono::duration<int64_t, Period>& rhs) {
            return Node(rhs.count());
        }

        static bool
        decode(const Node& node, std::chrono::duration<int64_t, Period>& rhs) {
            if (!node.IsScalar()) { return false; }
            rhs = std::chrono::duration<int64_t, Period>(node.as<int64_t>());
            return true;
        }
    };

#ifdef ERL_USE_OPENCV

    template<>
    struct convert<cv::Scalar> {
        static Node
        encode(const cv::Scalar& rhs) {
            Node node(NodeType::Sequence);
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            return node;
        }

        static bool
        decode(const Node& node, cv::Scalar& rhs) {
            if (!node.IsSequence()) { return false; }
            rhs[0] = node[0].as<double>();
            rhs[1] = node[1].as<double>();
            rhs[2] = node[2].as<double>();
            rhs[3] = node[3].as<double>();
            return true;
        }
    };

#endif
}  // namespace YAML

inline std::ostream&
operator<<(std::ostream& out, const erl::common::YamlableBase& yaml) {
    out << yaml.AsYamlString();
    return out;
}
