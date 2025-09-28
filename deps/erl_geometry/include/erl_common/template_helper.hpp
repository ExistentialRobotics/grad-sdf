#pragma once

#include "logging.hpp"
#include "string_utils.hpp"

#include <memory>

/// Check if T is an instantiation of the template `Class`. For example,
/// `is_instantiation<shared_ptr, T>` is true if `T == shared_ptr<U>` where U can be anything.
template<template<typename...> class Class, typename T>
struct is_instantiation : std::false_type {};

template<template<typename...> class Class, typename... Us>
struct is_instantiation<Class, Class<Us...>> : std::true_type {};

/// Check if T is std::shared_ptr<U> where U can be anything
template<typename T>
using IsSharedPtr = is_instantiation<std::shared_ptr, T>;

/// Check if T is std::unique_ptr<U> where U can be anything
template<typename T>
using IsUniquePtr = is_instantiation<std::unique_ptr, T>;

/// Check if T is std::weak_ptr<U> where U can be anything
template<typename T>
using IsWeakPtr = is_instantiation<std::weak_ptr, T>;

/// Check if T is smart pointer (std::shared_ptr, std::unique_ptr)
template<typename T>
using IsSmartPtr = std::disjunction<IsSharedPtr<T>, IsUniquePtr<T>>;

/// assert if the pointer is null
template<typename T, typename... Args>
T
NotNull(T ptr, const bool fatal, const std::string &msg, Args &&...args) {
    if (ptr == nullptr) {
        if (fatal) { erl::common::Logging::Fatal(msg, std::forward<Args>(args)...); }
        erl::common::Logging::Error(msg, std::forward<Args>(args)...);
    }
    return ptr;
}

template<typename T>
bool
CheckRuntimeType([[maybe_unused]] const T *ptr, const bool debug_only = false) {
    if (debug_only) {
#ifdef NDEBUG
        return true;  // in release mode, we don't check the type
#endif
    }
    const std::string runtime_type = type_name(*ptr);
    const std::string compile_time_type = type_name<T>();
    const bool same = runtime_type == compile_time_type;
    ERL_DEBUG_WARN_ONCE_COND(
        !same,
        "Runtime type {} does not match compile time type {}. This may cause memory "
        "corruption if the two types have different memory sizes.",
        runtime_type,
        compile_time_type);
    return same;
}

template<typename T1, typename T2>
struct Zip {
    using type = std::pair<const T1 &, const T2 &>;

    Zip(const std::vector<T1> &v1, const std::vector<T2> &v2)
        : m_v1_(v1.data()),
          m_v2_(v2.data()),
          m_size_(std::min(v1.size(), v2.size())) {}

    struct Iterator {
        explicit Iterator(const Zip *zip, const std::size_t index = 0)
            : m_zip_(zip),
              m_index_(index) {}

        T1 &
        first() {
            return m_zip_->m_v1_[m_index_];
        }

        const T1 &
        first() const {
            return m_zip_->m_v1_[m_index_];
        }

        T2 &
        second() {
            return m_zip_->m_v2_[m_index_];
        }

        const T2 &
        second() const {
            return m_zip_->m_v2_[m_index_];
        }

        const Iterator &
        operator++() {  // prefix increment, i.e. ++it
            ++m_index_;
            if (m_index_ >= m_zip_->m_size_) { m_index_ = m_zip_->m_size_; }
            return *this;
        }

        Iterator
        operator++(int) {  // postfix increment, i.e. it++
            std::size_t index = m_index_;
            ++m_index_;
            return {m_zip_, index};
        }

        [[nodiscard]] bool
        operator==(const Iterator &other) const {
            return m_index_ == other.m_index_ && m_zip_ == other.m_zip_;
        }

        [[nodiscard]] bool
        operator!=(const Iterator &other) const {
            return !(*this == other);
        }

    private:
        Zip *m_zip_;
        std::size_t m_index_;
    };

    Iterator
    begin() const {
        return {this, 0};
    }

    Iterator
    end() const {
        return {this, m_size_};
    }

private:
    T1 *m_v1_;
    T2 *m_v2_;
    std::size_t m_size_;
};
