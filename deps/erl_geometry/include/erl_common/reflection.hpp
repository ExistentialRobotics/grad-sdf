#pragma once

#include <string>

template<typename T, typename Member>
struct MemberInfo {
    using MemberType = Member;

    const char *name = "";
    Member T::*ptr = nullptr;
    std::string T::*type_ptr = nullptr;  // field to get the type of the member
};

template<typename T, typename... Members>
constexpr auto
MakeSchema(MemberInfo<T, Members>... members) {
    return std::make_tuple(members...);
}

template<typename T>
using static_schema_type = decltype(T::Schema);

template<typename Default, typename AlwaysVoid, template<typename...> typename Op, typename... Args>
struct HasSchema {
    using value_t = std::false_type;
    using type = Default;
};

template<typename Default, template<typename...> typename Op, typename... Args>
struct HasSchema<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template<typename T>
using has_static_schema_t = typename HasSchema<void, void, static_schema_type, T>::type;

template<typename T>
constexpr bool has_static_schema_v = HasSchema<void, void, static_schema_type, T>::value_t::value;

template<typename T, bool exists>
struct SchemaSize {

    static constexpr std::size_t
    CalculateSchemaSize() {
        std::size_t size = 0;
        std::apply(
            [&size](const auto &...member_info) {
                size +=
                    (sizeof(typename std::remove_reference_t<decltype(member_info)>::MemberType) +
                     ...);
            },
            T::Schema);
        return size;
    }

    static constexpr std::size_t size = CalculateSchemaSize();
};

template<typename T>
struct SchemaSize<T, false> {
    static constexpr std::size_t size = 0;
};

/**
 * Get the sum of sizes of all T's members defined in the schema. If T does not have a schema,
 * the size is 0. This is useful for checking if any members are forgotten in the schema if we
 * can know the total size of T's members from other means. Currently, sizeof(T) includes padding
 * bytes, so this schema size may not equal to sizeof(T).
 * @tparam T Type to get schema size for.
 */
template<typename T>
constexpr std::size_t schema_size_v = SchemaSize<T, has_static_schema_v<T>>::size;

template<typename T>
struct EnumMemberInfo {
    const char *name = "";
    T value;
};

/**
 * Specialization should be provided for each enum type T.
 * @tparam T Enum type.
 * @tparam N Number of enum members.
 * @return array of EnumMemberInfo<T> of size N.
 */
template<typename T, int N>
constexpr std::array<EnumMemberInfo<T>, N>
MakeEnumSchema();

#define ERL_REFLECT_SCHEMA(T, ...) static constexpr auto Schema = MakeSchema<T>(__VA_ARGS__)
#define ERL_REFLECT_MEMBER(T, x) \
    MemberInfo<T, decltype(T::x)> { #x, &T::x, nullptr }
#define ERL_REFLECT_MEMBER_POLY(T, x, t) \
    MemberInfo<T, decltype(T::x)> { #x, &T::x, &T::t }
#define ERL_REFLECT_ENUM_SCHEMA(T, N, ...)                              \
    template<>                                                          \
    constexpr std::array<EnumMemberInfo<T>, N> MakeEnumSchema<T, N>() { \
        return {__VA_ARGS__};                                           \
    }
#define ERL_REFLECT_ENUM_MEMBER(name, val) \
    EnumMemberInfo<decltype(val)> { name, val }
