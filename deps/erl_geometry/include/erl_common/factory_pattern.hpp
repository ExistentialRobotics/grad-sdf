#pragma once

#include "logging.hpp"
#include "string_utils.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace erl::common {

    template<typename Base, bool UniquePtr = false, bool RawPtr = false, typename... Args>
    class FactoryPattern {
    public:
        using BaseType = Base;
        using ArgsType = std::tuple<Args...>;
        using InitFunction = std::function<std::conditional_t<
            RawPtr,
            Base *,
            std::conditional_t<UniquePtr, std::unique_ptr<Base>, std::shared_ptr<Base>>>(Args...)>;

    private:
        std::map<std::string, InitFunction> m_class_id_mapping_ = {};
        FactoryPattern() = default;

    public:
        FactoryPattern(const FactoryPattern &) = delete;
        FactoryPattern(FactoryPattern &&) = delete;
        FactoryPattern &
        operator=(const FactoryPattern &) = delete;
        FactoryPattern &
        operator=(FactoryPattern &&) = delete;

        static FactoryPattern &
        GetInstance() {
            static FactoryPattern instance;
            return instance;
        }

        template<typename Derived>
        std::enable_if_t<std::is_base_of_v<Base, Derived>, bool>
        Register(InitFunction init_function) {
            return Register<Derived>("", init_function);
        }

        template<typename Derived>
        std::enable_if_t<std::is_base_of_v<Base, Derived>, bool>
        Register(std::string class_id, InitFunction init_function) {
            if (class_id.empty()) { class_id = type_name<Derived>(); }
            if (m_class_id_mapping_.count(class_id)) {
                ERL_WARN(
                    "{} is already registered to the factory.",
                    class_id,
                    type_name<FactoryPattern>());
                return false;
            }
            m_class_id_mapping_[class_id] = init_function;
            ERL_DEBUG("{} is registered.", class_id);
            return true;
        }

        std::shared_ptr<Base>
        Create(const std::string &class_id, Args... args) {
            const auto it = m_class_id_mapping_.find(class_id);
            if (it == m_class_id_mapping_.end()) {
                ERL_WARN(
                    "Unknown class id {} in the factory {}. Here are the registered class ids:",
                    class_id,
                    type_name<FactoryPattern>());
                for (const auto &pair: m_class_id_mapping_) { ERL_WARN("  - {}", pair.first); }
                return nullptr;
            }
            return it->second(args...);
        }

        [[nodiscard]] bool
        IsRegistered(const std::string &class_id) const {
            return m_class_id_mapping_.count(class_id);
        }

        template<typename Derived>
        [[nodiscard]] bool
        IsRegistered() const {
            return IsRegistered(demangle(typeid(Derived).name()));
        }

        void
        PrintRegisteredClasses() const {
            ERL_INFO("Registered classes in the factory {}:", type_name<FactoryPattern>());
            for (const auto &pair: m_class_id_mapping_) { ERL_INFO("  - {}", pair.first); }
        }
    };

}  // namespace erl::common
