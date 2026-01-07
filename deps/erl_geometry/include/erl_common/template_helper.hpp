#pragma once

#include "logging.hpp"
#include "string_utils.hpp"

#include <algorithm>
#include <memory>
#include <type_traits>

template<typename T>
struct is_shared_ptr : std::false_type {};

template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

template<typename T>
struct is_unique_ptr : std::false_type {};

template<typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

template<typename T>
struct is_weak_ptr : std::false_type {};

template<typename T>
struct is_weak_ptr<std::weak_ptr<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_weak_ptr_v = is_weak_ptr<T>::value;

template<typename T>
struct is_smart_ptr : std::disjunction<is_shared_ptr<T>, is_unique_ptr<T>> {};

template<typename T>
inline constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;

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

#ifndef NDEBUG
    #define CHECKED_AT(t, index) t.at(index)
#else
    #define CHECKED_AT(t, index) t[index]  // NOLINT(*-pro-bounds-constant-array-index)
#endif

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
        : m_v1_(v1.data()), m_v2_(v2.data()), m_size_(std::min(v1.size(), v2.size())) {}

    struct Iterator {
        explicit Iterator(const Zip *zip, const std::size_t index = 0)
            : m_zip_(zip), m_index_(index) {}

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
            m_index_ = std::min(m_index_, m_zip_->m_size_);
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
