#pragma once

#include "boost.hpp"
#include "eigen.hpp"
#include "factory_pattern.hpp"
#include "logging.hpp"
#include "opencv.hpp"
#include "reflection.hpp"
#include "ros.hpp"
#include "template_helper.hpp"
#include "version_check.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <type_traits>

#ifdef ERL_USE_ABSL
    #include <absl/container/flat_hash_map.h>
#endif

// https://yaml.org/spec/1.2.2/
// https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started

template<>
struct YAML::convert<std::filesystem::path> {
    static Node
    encode(const std::filesystem::path &path) {
        return Node(path.string());
    }

    static bool
    decode(const Node &node, std::filesystem::path &path) {
        path = node.as<std::string>();
        return true;
    }
};

namespace erl::common {

    struct YamlableBase {
        using Factory = FactoryPattern<YamlableBase>;

        YamlableBase() = default;
        YamlableBase(const YamlableBase &) = default;
        YamlableBase &
        operator=(const YamlableBase &) = default;
        YamlableBase(YamlableBase &&) = default;
        YamlableBase &
        operator=(YamlableBase &&) = default;

        virtual ~YamlableBase() = default;

        static constexpr std::size_t
        GetSchemaSize() {
            return 0;
        }

        template<typename Derived>
        static bool
        Register(const std::string &yamlable_type = "") {
            return Factory::GetInstance().Register<Derived>(yamlable_type, [] {
                return std::make_shared<Derived>();
            });
        }

        template<typename Derived>
        static std::shared_ptr<Derived>
        Create(const std::string &yamlable_type) {
            return std::dynamic_pointer_cast<Derived>(Factory::GetInstance().Create(yamlable_type));
        }

        [[nodiscard]] virtual bool
        PostDeserialization() {
            return true;
        }

        [[nodiscard]] bool
        operator==(const YamlableBase &other) const;

        [[nodiscard]] bool
        operator!=(const YamlableBase &other) const;

        [[nodiscard]] virtual bool
        FromYamlNode(const YAML::Node &node) = 0;

        [[nodiscard]] bool
        FromYamlString(const std::string &yaml_string);

        [[nodiscard]] virtual YAML::Node
        AsYamlNode() const = 0;

        [[nodiscard]] virtual std::string
        AsYamlString() const;

        [[nodiscard]] bool
        FromYamlFile(
            const std::string &yaml_file,
            const std::string &base_config_field = "__base__");

        void
        AsYamlFile(const std::string &yaml_file) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);

        /**
         *
         * @param args Command line arguments.
         * @return true if successful.
         */
        bool
        FromCommandLine(const std::vector<std::string> &args);

        /**
         * Load parameters from command line arguments. This function calls
         * FromCommandLine(int, const char**).
         * @param argc Number of command line arguments.
         * @param argv Command line arguments.
         * @return true if successful.
         */
        bool
        FromCommandLine(int argc, char *argv[]);

        /**
         * Load parameters from command line arguments. This function builds connection between
         * command line arguments and YAML
         * @param argc Number of command line arguments.
         * @param argv Command line arguments.
         * @return true if successful.
         */
        bool
        FromCommandLine(int argc, const char *argv[]);

#ifdef ERL_USE_BOOST

        virtual void
        FromCommandLineImpl(
            program_options::ProgramOptionsData & /*po_data*/,
            const std::string & /*prefix*/) {}

#endif

#ifdef ERL_ROS_VERSION_1
        /**
         * Load parameters from a ROS1 NodeHandle. A default implementation is provided by
         * the Yamlable<T> derived class. Not all the members are guaranteed to be specified by the
         * user in the ROS parameter server. The implementation should handle the default values.
         */
        [[nodiscard]] virtual bool
        LoadFromRos1(ros::NodeHandle &nh, const std::string &prefix) {
            (void) nh;
            (void) prefix;
            return true;
        };
#endif

#ifdef ERL_ROS_VERSION_2
        [[nodiscard]] virtual bool
        LoadFromRos2(rclcpp::Node *node, const std::string &prefix) {
            (void) node;
            (void) prefix;
            return true;
        }
#endif
    };

    enum class UnknownFieldPolicy {
        kIgnore = 0,
        kMerge = 1,
        kWarn = 2,
        kError = 3,
    };

    /**
     * Update destination YAML node with source YAML node.
     * @param src source YAML node
     * @param dst destination YAML node
     * @param unknown_field_policy policy for handling unknown fields
     */
    void
    UpdateYamlNode(const YAML::Node &src, YAML::Node &dst, UnknownFieldPolicy unknown_field_policy);

    namespace yaml_helper {
        // Helper template functions/structs for YAML encoding and decoding of Yamlable types. These
        // functions should be nested inside Yamlable<T> to reduce binary size. If we put these
        // functions inside Yamlable<T>, each instantiation of Yamlable<T> will have its own copy of
        // these functions, leading to code bloat. For example, T1 and T2 may have the same base
        // class Base, but Yamlable<T1> and Yamlable<T2> will each have their own copy of
        // EncodeBase<Base> and DecodeBase<Base>. By putting these functions in a separate
        // namespace, we ensure that there is only one copy of these functions for all
        // instantiations of Yamlable<T>. So, the binary size is reduced and the compile time is
        // also reduced.

        // 1. Interaction with YAML::Node

        // encoding Base derived from YamlableBase
        template<typename B>
        std::enable_if_t<
            std::is_base_of_v<YamlableBase, B> && !std::is_same_v<YamlableBase, B>,
            YAML::Node>
        EncodeBase(const B &obj) {
            return B::ConvertImpl::encode(obj);
        }

        // encoding other base types
        template<typename B>
        std::enable_if_t<
            !std::is_base_of_v<YamlableBase, B> || std::is_same_v<YamlableBase, B>,
            YAML::Node>
        EncodeBase(const B &obj) {
            (void) obj;
            return {};
        }

        // encoding smart pointers to YamlableBase derived types
        template<typename M>
        std::enable_if_t<
            is_smart_ptr<M>::value && std::is_base_of_v<YamlableBase, typename M::element_type>>
        EncodeMember(YAML::Node &node, const char *name, const M &member) {
            if (member == nullptr) {
                node[name] = YAML::Node(YAML::NodeType::Null);
                return;
            }
            node[name] = member->AsYamlNode();
        }

        // encoding smart pointers to non-YamlableBase types
        template<typename M>
        std::enable_if_t<
            is_smart_ptr<M>::value && !std::is_base_of_v<YamlableBase, typename M::element_type>>
        EncodeMember(YAML::Node &node, const char *name, const M &member) {
            node[name] = member;
        }

        // encoding YamlableBase derived types
        template<typename M>
        std::enable_if_t<std::is_base_of_v<YamlableBase, M>>
        EncodeMember(YAML::Node &node, const char *name, const M &member) {
            node[name] = member.AsYamlNode();
        }

        // encoding other types
        template<typename M>
        std::enable_if_t<
            !is_smart_ptr<M>::value &&            // not a smart pointer
            !is_weak_ptr<M>::value &&             // not a weak pointer
            !std::is_pointer_v<M> &&              // not a raw pointer
            !std::is_base_of_v<YamlableBase, M>>  // not YamlableBase derived
        EncodeMember(YAML::Node &node, const char *name, const M &member) {
            node[name] = member;
        }

        // decoding a base class derived from YamlableBase
        template<typename B>
        std::enable_if_t<
            std::is_base_of_v<YamlableBase, B> && !std::is_same_v<YamlableBase, B>,
            bool>
        DecodeBase(const YAML::Node &node, B &obj) {
            return B::ConvertImpl::decode(node, obj);
        }

        // decoding a base class that is not derived from YamlableBase
        template<typename B>
        std::enable_if_t<
            !std::is_base_of_v<YamlableBase, B> || std::is_same_v<YamlableBase, B>,
            bool>
        DecodeBase(const YAML::Node & /*node*/, B &obj) {
            (void) obj;
            return true;
        }

        // default implementation for handling no polymorphism
        template<typename M, typename = void>
        struct NoPolymorphism {

            static void
            run(M & /*member*/, const std::string & /*member_type*/) {}
        };

        // specialization for smart pointers without polymorphism
        template<typename M>
        struct NoPolymorphism<M, std::enable_if_t<is_smart_ptr<M>::value>> {
            static void
            run(M &member, const std::string & /*member_type*/) {
                if (member != nullptr) { return; }
                member = std::make_shared<typename M::element_type>();
            }
        };

        // default specialization for handling polymorphism
        template<typename M, typename = void>
        struct HandlePolymorphism : NoPolymorphism<M> {};

        // specialization for smart pointers to YamlableBase derived types with polymorphism
        template<typename M>
        struct HandlePolymorphism<M, std::enable_if_t<is_smart_ptr<M>::value>> {

            static void
            run(M &member, const std::string &member_type) {
                using ElementType = typename M::element_type;
                member = ElementType::template Create<ElementType>(member_type);
            }
        };

        // decoding smart pointers to YamlableBase derived types
        template<typename M>
        std::enable_if_t<
            is_smart_ptr_v<M> && std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        DecodeMember(const YAML::Node &node, M &member, const std::string &type, const bool poly) {
            if (node.IsNull()) {
                member = nullptr;
                return true;
            }
            if (poly) {
                HandlePolymorphism<M>::run(member, type);
            } else {
                NoPolymorphism<M>::run(member, "");
            }
            return member->FromYamlNode(node);
        }

        // decoding smart pointers to non-YamlableBase types
        template<typename M>
        std::enable_if_t<
            is_smart_ptr_v<M> && !std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        DecodeMember(
            const YAML::Node &node,
            M &member,
            const std::string & /*type*/ = "",
            const bool /*poly*/ = false) {
            member = node.as<M>();
            return true;
        }

        // decoding YamlableBase derived types
        template<typename M>
        std::enable_if_t<std::is_base_of_v<YamlableBase, M>, bool>
        DecodeMember(
            const YAML::Node &node,
            M &member,
            const std::string & /*type*/ = "",
            const bool /*poly*/ = false) {
            return member.FromYamlNode(node);
        }

        // decoding floating point types with special handling for infinity
        template<typename M>
        std::enable_if_t<std::is_floating_point_v<M>, bool>
        DecodeMember(
            const YAML::Node &node,
            M &member,
            const std::string & /*type*/ = "",
            const bool /*poly*/ = false) {
#if ERL_CHECK_VERSION_GE(   \
    YAML_CPP_VERSION_MAJOR, \
    YAML_CPP_VERSION_MINOR, \
    YAML_CPP_VERSION_PATCH, \
    0,                      \
    6,                      \
    3)
            if (node.as<std::string>() == "inf") {  // support .inf but not inf
                member = std::numeric_limits<M>::infinity();
            } else {
                member = node.as<M>();
            }
#else
            auto str = node.as<std::string>();
            if (str == ".inf") {
                // YAML 0.6.2 fails to parse "inf" and ".inf".
                member = std::numeric_limits<M>::infinity();
            } else {
                // support inf but not .inf
                member = static_cast<M>(std::stod(node.as<std::string>()));
            }
#endif
            return true;
        }

        // decoding other types
        template<typename M>
        std::enable_if_t<
            !is_smart_ptr_v<M> &&                       // not a smart pointer
                !is_weak_ptr_v<M> &&                    // not a weak pointer
                !std::is_pointer_v<M> &&                // not a raw pointer
                !std::is_base_of_v<YamlableBase, M> &&  // not YamlableBase derived
                !std::is_floating_point_v<M>,           // not floating point
            bool>
        DecodeMember(
            const YAML::Node &node,
            M &member,
            const std::string & /*type*/ = "",
            const bool /*poly*/ = false) {
            member = node.as<M>();
            return true;
        }

        // We need a helper for decode, because 'if' is a statement
        // and can't be used directly in a fold *expression*.
        template<typename T_Class, typename T_MemberInfo>
        bool
        DecodeMemberDispatch(const YAML::Node &node, T_Class &obj, T_MemberInfo &info) {
            if (!node[info.name]) {
                ERL_WARN("{} not found in YAML node during decoding.", info.name);
                return false;
            }

            // Get the type of the member (e.g., int, std::string)
            using M = std::remove_const_t<std::remove_reference_t<decltype(obj.*(info.ptr))>>;

            try {
                if (info.type_ptr == nullptr) {
                    return DecodeMember<M>(node[info.name], obj.*(info.ptr), "", false);
                }

                // decode the member
                return DecodeMember<M>(
                    node[info.name],
                    obj.*(info.ptr),
                    obj.*(info.type_ptr),
                    true);

            } catch (std::exception &e) {
                ERL_WARN("Failed to decode member {}: {}", info.name, e.what());
                return false;
            }
        }

        // 2. Interaction with command line

#ifdef ERL_USE_BOOST

        // decoding smart pointers to YamlableBase derived types
        template<typename M>
        static std::enable_if_t<
            is_smart_ptr<M>::value && std::is_base_of_v<YamlableBase, typename M::element_type>>
        LoadMemberFromCommandLine(
            program_options::ProgramOptionsData &po_data,
            const std::string &option_name,
            M &member,
            const std::string &type,
            const bool poly) {

            if (poly) {
                HandlePolymorphism<M>::run(member, type);
            } else {
                NoPolymorphism<M>::run(member, "");
            }
            member->FromCommandLineImpl(po_data, option_name);
        }

        // decoding smart pointers to non-YamlableBase types
        template<typename M>
        std::enable_if_t<
            is_smart_ptr_v<M> && !std::is_base_of_v<YamlableBase, typename M::element_type>>
        LoadMemberFromCommandLine(
            program_options::ProgramOptionsData &po_data,
            const std::string &option_name,
            M &member,
            const std::string & /*type*/,
            const bool /*poly*/) {
            using ElementType = typename M::element_type;
            member = std::make_shared<ElementType>();
            try {
                po_data.GetOptionParser<ElementType>(option_name, member.get())->Run();
            } catch (std::exception &e) { po_data.RecordError(option_name, e.what()); }
        }

        // decoding YamlableBase derived types
        template<typename M>
        std::enable_if_t<std::is_base_of_v<YamlableBase, M>>
        LoadMemberFromCommandLine(
            program_options::ProgramOptionsData &po_data,
            const std::string &option_name,
            M &member,
            const std::string & /*type*/,
            const bool /*poly*/) {
            member.FromCommandLineImpl(po_data, option_name);
        }

        // decoding other types
        template<typename M>
        std::enable_if_t<
            !is_smart_ptr_v<M> &&                 // not a smart pointer
            !is_weak_ptr_v<M> &&                  // not a weak pointer
            !std::is_pointer_v<M> &&              // not a raw pointer
            !std::is_base_of_v<YamlableBase, M>>  // not YamlableBase derived
        LoadMemberFromCommandLine(
            program_options::ProgramOptionsData &po_data,
            const std::string &option_name,
            M &member,
            const std::string & /*type*/,
            const bool /*poly*/) {
            try {
                auto parser = po_data.GetOptionParser<M>(option_name, &member);
                parser->Run();
            } catch (std::exception &e) { po_data.RecordError(option_name, e.what()); }
        }

        template<typename T_Class, typename T_MemberInfo>
        void
        LoadMemberFromCommandLineDispatch(
            program_options::ProgramOptionsData &po_data,
            const std::string &prefix,
            T_Class &obj,
            T_MemberInfo &info) {

            // Get the type of the member (e.g., int, std::string)
            using M = std::remove_const_t<std::remove_reference_t<decltype(obj.*(info.ptr))>>;

            std::string option_name = program_options::GetBoostOptionName(prefix, info.name);
            if (info.type_ptr == nullptr) {
                LoadMemberFromCommandLine<M>(po_data, option_name, obj.*(info.ptr), "", false);
                return;
            }

            // decode the member
            LoadMemberFromCommandLine<M>(
                po_data,
                option_name,
                obj.*(info.ptr),
                obj.*(info.type_ptr),
                true);
        }

#endif

        // 3. Interaction with ROS1 NodeHandle

#ifdef ERL_ROS_VERSION_1

        // loading smart pointers to YamlableBase derived types
        template<typename M>
        static std::enable_if_t<
            is_smart_ptr<M>::value && std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        LoadMemberFromRos1(
            ros::NodeHandle &nh,
            const std::string &param_path,
            M &member,
            const std::string &type,
            bool poly) {

            if (poly) {
                HandlePolymorphism<M>::run(member, type);
            } else {
                NoPolymorphism<M>::run(member, "");
            }
            return member->LoadFromRos1(nh, param_path);
        }

        // loading smart pointers to non-YamlableBase types
        template<typename M>
        static std::enable_if_t<
            is_smart_ptr<M>::value && !std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        LoadMemberFromRos1(
            ros::NodeHandle &nh,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            using ElementType = typename M::element_type;
            member = std::make_shared<ElementType>();
            using namespace erl::common::ros_params;
            LoadRos1Param<M>::Run(nh, param_path, *member);
            return true;
        }

        // loading YamlableBase derived types
        template<typename M>
        std::enable_if_t<std::is_base_of_v<YamlableBase, M>, bool>
        LoadMemberFromRos1(
            ros::NodeHandle &nh,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            return member.LoadFromRos1(nh, param_path);
        }

        template<typename M>
        std::enable_if_t<
            !is_smart_ptr_v<M> &&                     // not a smart pointer
                !is_weak_ptr_v<M> &&                  // not a weak pointer
                !std::is_pointer_v<M> &&              // not a raw pointer
                !std::is_base_of_v<YamlableBase, M>,  // not YamlableBase derived
            bool>
        LoadMemberFromRos1(
            ros::NodeHandle &nh,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            using namespace erl::common::ros_params;
            LoadRos1Param<M>::Run(nh, param_path, member);
            return true;
        }

        template<typename T_Class, typename T_MemberInfo>
        bool
        LoadMemberFromRos1Dispatch(
            ros::NodeHandle &nh,
            const std::string &prefix,
            T_Class &obj,
            T_MemberInfo &&info) {

            // Get the type of the member (e.g., int, std::string)
            using M = std::remove_const_t<std::remove_reference_t<decltype(obj.*(info.ptr))>>;
            using namespace erl::common::ros_params;

            try {
                std::string &&param_path = GetRos1ParamPath(prefix, info.name);
                if (info.type_ptr == nullptr) {
                    return LoadMemberFromRos1<M>(nh, param_path, obj.*(info.ptr), "", false);
                }

                // decode the member
                return LoadMemberFromRos1<M>(
                    nh,
                    param_path,
                    obj.*(info.ptr),
                    obj.*(info.type_ptr),
                    true);

            } catch (std::exception &e) {
                ERL_WARN("Failed to load member {}: {}", info.name, e.what());
                return false;
            }
        }
#endif  // ERL_ROS_VERSION_1

        // 4. Interaction with ROS2 Node

#ifdef ERL_ROS_VERSION_2

        // loading smart pointers to YamlableBase derived types
        template<typename M>
        static std::enable_if_t<
            is_smart_ptr<M>::value && std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        LoadMemberFromRos2(
            rclcpp::Node *node,
            const std::string &param_path,
            M &member,
            const std::string &type,
            bool poly) {

            if (poly) {
                HandlePolymorphism<M>::run(member, type);
            } else {
                NoPolymorphism<M>::run(member, "");
            }
            return member->LoadFromRos2(node, param_path);
        }

        // loading smart pointers to non-YamlableBase types
        template<typename M>
        static std::enable_if_t<
            is_smart_ptr<M>::value && !std::is_base_of_v<YamlableBase, typename M::element_type>,
            bool>
        LoadMemberFromRos2(
            rclcpp::Node *node,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            using ElementType = typename M::element_type;
            member = std::make_shared<ElementType>();
            using namespace erl::common::ros_params;
            LoadRos2Param<M>::Run(node, param_path, *member);
            return true;
        }

        // loading YamlableBase derived types
        template<typename M>
        std::enable_if_t<std::is_base_of_v<YamlableBase, M>, bool>
        LoadMemberFromRos2(
            rclcpp::Node *node,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            return member.LoadFromRos2(node, param_path);
        }

        template<typename M>
        std::enable_if_t<
            !is_smart_ptr_v<M> &&                     // not a smart pointer
                !is_weak_ptr_v<M> &&                  // not a weak pointer
                !std::is_pointer_v<M> &&              // not a raw pointer
                !std::is_base_of_v<YamlableBase, M>,  // not YamlableBase derived
            bool>
        LoadMemberFromRos2(
            rclcpp::Node *node,
            const std::string &param_path,
            M &member,
            const std::string & /*type*/ = "",
            bool /*poly*/ = false) {
            using namespace erl::common::ros_params;
            LoadRos2Param<M>::Run(node, param_path, member);
            return true;
        }

        template<typename T_Class, typename T_MemberInfo>
        bool
        LoadMemberFromRos2Dispatch(
            rclcpp::Node *node,
            const std::string &prefix,
            T_Class &obj,
            T_MemberInfo &&info) {

            // Get the type of the member (e.g., int, std::string)
            using M = std::remove_const_t<std::remove_reference_t<decltype(obj.*(info.ptr))>>;
            using namespace erl::common::ros_params;

            try {
                std::string &&param_path = GetRos2ParamPath(prefix, info.name);
                if (info.type_ptr == nullptr) {
                    return LoadMemberFromRos2<M>(node, param_path, obj.*(info.ptr), "", false);
                }

                // decode the member
                return LoadMemberFromRos2<M>(
                    node,
                    param_path,
                    obj.*(info.ptr),
                    obj.*(info.type_ptr),
                    true);

            } catch (std::exception &e) {
                ERL_WARN("Failed to load member {}: {}", info.name, e.what());
                return false;
            }
        }
#endif  // ERL_ROS_VERSION_2
    }  // namespace yaml_helper

    template<typename T, typename Base = YamlableBase>
    struct Yamlable : Base {

        [[nodiscard]] bool
        FromYamlNode(const YAML::Node &node) override {
            return YAML::convert<T>::decode(node, *static_cast<T *>(this));
        }

        [[nodiscard]] YAML::Node
        AsYamlNode() const override {
            return YAML::convert<T>::encode(*static_cast<const T *>(this));
        }

        static constexpr std::size_t
        GetSchemaSize() {
            if (std::is_base_of_v<YamlableBase, Base> && !std::is_same_v<YamlableBase, Base>) {
                return schema_size_v<T> + Base::GetSchemaSize();
            }
            return schema_size_v<T>;
        }

        /**
         * Template specialization for YAML conversion of erl::common::Yamlable types.
         */
        struct ConvertImpl {

            // main encode function
            static YAML::Node
            encode(const T &obj) {
                using namespace yaml_helper;

                YAML::Node node = EncodeBase<Base>(static_cast<const Base &>(obj));
                std::apply(
                    [&](const auto &...member_info) {
                        ((EncodeMember<std::remove_const_t<
                              std::remove_reference_t<decltype(obj.*(member_info.ptr))>>>(
                             node,
                             member_info.name,
                             obj.*(member_info.ptr))),
                         ...);
                    },
                    T::Schema);
                return node;
            }

            // main decode function
            static bool
            decode(const YAML::Node &node, T &obj) {
                using namespace yaml_helper;

                if (!node.IsMap()) { return false; }
                if (!DecodeBase<Base>(node, obj)) { return false; }
                bool success = true;
                std::apply(
                    [&](const auto &...member_info) {
                        success &= (DecodeMemberDispatch(node, obj, member_info) && ...);
                    },
                    T::Schema);
                success &= obj.PostDeserialization();  // call post deserialization hook
                return success;
            }
        };

#ifdef ERL_USE_BOOST
        void
        FromCommandLineImpl(program_options::ProgramOptionsData &po_data, const std::string &prefix)
            override {

            // add options from base class first
            if (std::is_base_of_v<YamlableBase, Base> && !std::is_same_v<YamlableBase, Base>) {
                Base::FromCommandLineImpl(po_data, prefix);
                if (!po_data.Successful()) { return; }
            }

            using namespace yaml_helper;

            std::apply(
                [&](const auto &...member_info) {
                    (LoadMemberFromCommandLineDispatch(
                         po_data,
                         prefix,
                         *dynamic_cast<T *>(this),
                         member_info),
                     ...);
                },
                T::Schema);
        }
#endif

#ifdef ERL_ROS_VERSION_1
        [[nodiscard]] bool
        LoadFromRos1(ros::NodeHandle &nh, const std::string &prefix) override {
            using namespace yaml_helper;
            using namespace erl::common::ros_params;

            if (prefix.empty() && std::is_same_v<YamlableBase, Base>) {
                // if at the top level, check for config_file first
                std::string config_file;
                nh.param<std::string>(GetRos1ParamPath(prefix, "config_file"), config_file, "");
                try {
                    if (!config_file.empty()) {  // load the config from the file first
                        if (!this->FromYamlFile(config_file)) {
                            ROS_FATAL("Failed to load %s", config_file.c_str());
                            return false;
                        }
                    }
                } catch (const std::exception &e) {
                    ROS_FATAL("Failed to parse %s: %s", config_file.c_str(), e.what());
                    return false;
                }
            }

            // if Base is not YamlableBase, load its parameters from ROS1 first
            if (std::is_base_of_v<YamlableBase, Base> && !std::is_same_v<YamlableBase, Base>) {
                if (!Base::LoadFromRos1(nh, prefix)) { return false; }
            }

            bool success = true;
            std::apply(
                [&](const auto &...member_info) {
                    // ((load each member), ...);
                    success &=
                        (LoadMemberFromRos1Dispatch(
                             nh,
                             prefix,
                             *reinterpret_cast<T *>(this),
                             member_info) &&
                         ...);
                },
                T::Schema);
            success &= this->PostDeserialization();  // call post deserialization hook
            return success;
        }
#endif

#ifdef ERL_ROS_VERSION_2
        [[nodiscard]] bool
        LoadFromRos2(rclcpp::Node *node, const std::string &prefix) override {
            using namespace yaml_helper;
            using namespace erl::common::ros_params;

            if (prefix.empty() && std::is_same_v<YamlableBase, Base>) {
                // if at the top level, check for config_file first
                std::string config_file;
                std::string param_path = GetRos2ParamPath(prefix, "config_file");
                node->declare_parameter<std::string>(param_path, config_file);
                node->get_parameter_or<std::string>(param_path, config_file, config_file);

                try {
                    if (!config_file.empty()) {  // load the config from the file first
                        if (!this->FromYamlFile(config_file)) {
                            RCLCPP_FATAL(
                                node->get_logger(),
                                "Failed to load %s",
                                config_file.c_str());
                            return false;
                        }
                    }
                } catch (const std::exception &e) {
                    RCLCPP_FATAL(
                        node->get_logger(),
                        "Failed to parse %s: %s",
                        config_file.c_str(),
                        e.what());
                    return false;
                }
            }

            // if Base is not YamlableBase, load its parameters from ROS2 first
            if (std::is_base_of_v<YamlableBase, Base> && !std::is_same_v<YamlableBase, Base>) {
                if (!Base::LoadFromRos2(node, prefix)) { return false; }
            }

            bool success = true;
            std::apply(
                [&](const auto &...member_info) {
                    // ((load each member), ...);
                    success &=
                        (LoadMemberFromRos2Dispatch(
                             node,
                             prefix,
                             *reinterpret_cast<T *>(this),
                             member_info) &&
                         ...);
                },
                T::Schema);
            success &= this->PostDeserialization();  // call post deserialization hook
            return success;
        }
#endif
    };

    template<typename T>
    std::vector<T>
    LoadYamlSequenceFromFile(const std::string &path, const bool multi_nodes = false) {
        if (multi_nodes) {
            // each node is an element in the sequence
            const std::vector<YAML::Node> nodes = YAML::LoadAllFromFile(path);
            std::vector<T> objs;
            objs.reserve(nodes.size());
            for (const auto &node: nodes) { objs.push_back(node.as<T>()); }
            return objs;
        }

        return YAML::LoadFile(path).as<std::vector<T>>();
    }

    template<typename T, int N>
    struct EnumYamlConvert {
        static constexpr std::array<EnumMemberInfo<T>, N> EnumSchema = MakeEnumSchema<T, N>();

        static YAML::Node
        encode(const T &enum_value) {
            auto it = std::find_if(
                EnumSchema.begin(),
                EnumSchema.end(),
                [&](const EnumMemberInfo<T> &info) { return info.value == enum_value; });
            ERL_ASSERTM(
                it != EnumSchema.end(),
                "Cannot find enum value: {}",
                static_cast<int>(enum_value));
            return YAML::Node(it->name);
        }

        static bool
        decode(const YAML::Node &node, T &enum_value) {
            if (!node.IsScalar()) { return false; }
            auto enum_str = node.as<std::string>();
            auto it = std::find_if(
                EnumSchema.begin(),
                EnumSchema.end(),
                [&](const EnumMemberInfo<T> &info) { return enum_str == info.name; });
            if (it == EnumSchema.end()) {
                ERL_WARN("{} is not a valid enum string.", enum_str);
                return false;
            }
            enum_value = it->value;
            return true;
        }
    };
}  // namespace erl::common

/**
 * General template specialization for YAML conversion of erl::common::Yamlable types. This makes
 * the conversion of all Yamlable types work automatically. For other types, we still need to
 * provide the specialization separately. To make this template work correctly for a type T derived
 * from erl::common::YamlableBase, T must define a static constexpr member `Schema` using
 * MakeScheme<T>. e.g.
 *
 * static constexpr auto Schema = MakeSchema<T>(
 *      MemberInfo<T, int>{"attr1", &T::attr1},
 *      MemberInfo<T, std::string>{"attr2", &T::attr2},
 * );
 *
 * @tparam T Type derived from erl::common::YamlableBase.
 */
template<typename T>
struct YAML::convert : public T::ConvertImpl {};

/**
 * Template macro to define YAML conversion for enum types. To use this macro, just add the line
 * after defining the enum type. A template specialization of MakeEnumSchema<T, N>() must also
 * be defined. e.g.
 *
 * template<>
 * constexpr std::array<EnumMemberInfo<Color>, 2>
 * MakeEnumSchema<Color, 2>() {
 *      return {
 *          EnumMemberInfo<Color>{"red", Color::kRed},
 *          EnumMemberInfo<Color>{"blue", Color::kBlue},
 *      };
 * }
 *
 * Macro is available as well to define the enum schema:
 * ERL_REFLECT_ENUM_SCHEMA(
 *     Color,
 *     2,
 *     ERL_REFLECT_ENUM_MEMBER("red", Color::kRed),
 *     ERL_REFLECT_ENUM_MEMBER("blue", Color::kBlue));
 *
 * ERL_PARSE_ENUM(T, N) can be used to define the conversion with YAML, Boost Program Options, fmt
 * and ROS param.
 *
 * @param T The enum type.
 * @param N Number of enum members.
 * @relatedalso ERL_PARSE_ENUM
 */
#define ERL_ENUM_YAML_CONVERT(T, N)                                       \
    template<>                                                            \
    struct YAML::convert<T> : public erl::common::EnumYamlConvert<T, N> { \
        static_assert(std::is_enum_v<T>, #T " must be an enum type.");    \
    };

namespace YAML {

    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct convert<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {

        using T = Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>;

        static Node
        encode(const T &rhs) {
            Node node(NodeType::Sequence);
            const int rows = Rows_ == Eigen::Dynamic ? rhs.rows() : Rows_;
            const int cols = Cols_ == Eigen::Dynamic ? rhs.cols() : Cols_;

            for (int i = 0; i < rows; ++i) {
                Node row_node(NodeType::Sequence);
                for (int j = 0; j < cols; ++j) { row_node.push_back(rhs(i, j)); }
                node.push_back(row_node);
            }

            return node;
        }

        static bool
        decode(const Node &node, T &rhs) {
            if (node.IsNull() && (Rows_ == Eigen::Dynamic || Cols_ == Eigen::Dynamic)) {
                return true;
            }
            if (!node.IsSequence()) { return false; }
            if (!node[0].IsSequence()) { return false; }

            int rows = Rows_ == Eigen::Dynamic ? node.size() : Rows_;
            int cols = Cols_ == Eigen::Dynamic ? node[0].size() : Cols_;
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
                const auto &row_node = node[i];
                for (int j = 0; j < cols; ++j) { rhs(i, j) = row_node[j].as<Scalar_>(); }
            }

            return true;
        }
    };

    template<typename Type, int Size>
    struct convert<Eigen::Vector<Type, Size>> {
        using T = Eigen::Vector<Type, Size>;

        static Node
        encode(const T &rhs) {
            Node node(NodeType::Sequence);
            if (Size == Eigen::Dynamic) {
                for (int i = 0; i < rhs.size(); ++i) { node.push_back(rhs[i]); }
            } else {
                for (int i = 0; i < Size; ++i) { node.push_back(rhs[i]); }
            }
            return node;
        }

        static bool
        decode(const Node &node, T &rhs) {
            if (!node.IsSequence()) { return false; }
            if (Size == Eigen::Dynamic) {
                rhs.resize(node.size());
                for (int i = 0; i < rhs.size(); ++i) { rhs[i] = node[i].as<Type>(); }
            } else {
                for (int i = 0; i < Size; ++i) { rhs[i] = node[i].as<Type>(); }
            }
            return true;
        }
    };

    template<typename... Args>
    struct convert<std::tuple<Args...>> {
        static Node
        encode(const std::tuple<Args...> &rhs) {
            Node node(NodeType::Sequence);
            std::apply(
                [&node](const Args &...args) {
                    (node.push_back(convert<Args>::encode(args)), ...);
                },
                rhs);
            return node;
        }

        static bool
        decode(const Node &node, std::tuple<Args...> &rhs) {
            if (!node.IsSequence()) { return false; }
            if (node.size() != sizeof...(Args)) { return false; }
            std::apply([&node](Args &...args) { (convert<Args>::decode(node, args) && ...); }, rhs);
            return true;
        }
    };

    template<typename T>
    struct convert<std::optional<T>> {
        static Node
        encode(const std::optional<T> &rhs) {
            if (rhs) { return convert<T>::encode(*rhs); }
            return Node(NodeType::Null);
        }

        static bool
        decode(const Node &node, std::optional<T> &rhs) {
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
        encode(const std::shared_ptr<T> &rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node &node, std::shared_ptr<T> &rhs) {
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
        encode(const std::unique_ptr<T> &rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node &node, std::unique_ptr<T> &rhs) {
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
        encode(const std::unordered_map<KeyType, ValueType> &rhs) {
            Node node(NodeType::Map);
            for (const auto &[key, value]: rhs) {
                node[convert<KeyType>::encode(key)] = convert<ValueType>::encode(value);
            }
            return node;
        }

        static bool
        decode(const Node &node, std::unordered_map<KeyType, ValueType> &rhs) {
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
        encode(const std::chrono::duration<int64_t, Period> &rhs) {
            return Node(rhs.count());
        }

        static bool
        decode(const Node &node, std::chrono::duration<int64_t, Period> &rhs) {
            if (!node.IsScalar()) { return false; }
            rhs = std::chrono::duration<int64_t, Period>(node.as<int64_t>());
            return true;
        }
    };

#ifdef ERL_USE_ABSL

    template<typename KeyType, typename ValueType>
    struct convert<absl::flat_hash_map<KeyType, ValueType>> {
        static Node
        encode(const absl::flat_hash_map<KeyType, ValueType> &rhs) {
            Node node(NodeType::Map);
            for (const auto &[key, value]: rhs) {
                node[convert<KeyType>::encode(key)] = convert<ValueType>::encode(value);
            }
            return node;
        }

        static bool
        decode(const Node &node, absl::flat_hash_map<KeyType, ValueType> &rhs) {
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

#endif

#ifdef ERL_USE_OPENCV

    template<>
    struct convert<cv::Scalar> {
        static Node
        encode(const cv::Scalar &rhs) {
            Node node(NodeType::Sequence);
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            return node;
        }

        static bool
        decode(const Node &node, cv::Scalar &rhs) {
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

inline std::ostream &
operator<<(std::ostream &out, const erl::common::YamlableBase &yaml) {
    out << yaml.AsYamlString();
    return out;
}
