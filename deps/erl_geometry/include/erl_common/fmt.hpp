#pragma once
#ifndef FMT_HEADER_ONLY
    #define FMT_HEADER_ONLY
#endif
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#if FMT_VERSION >= 60200  // 6.2.0
    #include <fmt/os.h>
#endif
#if FMT_VERSION >= 90000
    #include <fmt/std.h>
#endif

// Eigen support for fmt library

#include <Eigen/Core>

#if FMT_VERSION >= 100200

    // Use the new nested formatter with fmt >= 10.2.0.
    // This support nested Eigen types as well as padding/format specifiers.
    #include <type_traits>

template<typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
    : fmt::nested_formatter<typename T::Scalar> {
    auto
    format(T const &a, format_context &ctx) const {
        return this->write_padded(ctx, [&](auto out) {
            for (Eigen::Index ir = 0; ir < a.rows(); ir++) {
                out = fmt::format_to(out, "{}", this->nested(a(ir, 0)));
                for (Eigen::Index ic = 1; ic < a.cols(); ic++) {
                    out = fmt::format_to(out, ", {}", this->nested(a(ir, ic)));
                }
                if (ir + 1 < a.rows()) { out = fmt::format_to(out, "\n"); }
            }
            return out;
        });
    }
};

template<typename Derived>
struct fmt::
    is_range<Derived, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>, char>>
    : std::false_type {};

#else

    #include <type_traits>

template<typename Derived>
struct fmt::formatter<
    Derived,
    std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>, char>> {

private:
    fmt::formatter<typename Derived::Scalar, char> m_underlying_;

public:
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return m_underlying_.parse(ctx);
    }

    template<typename FormatContext>
    auto
    format(const Derived &mat, FormatContext &ctx) {
        auto out = ctx.out();

        for (Eigen::Index row = 0; row < mat.rows(); ++row) {
            out = m_underlying_.format(mat.coeff(row, 0), ctx);
            for (Eigen::Index col = 1; col < mat.cols(); ++col) {
                out = fmt::format_to(out, ", ");
                out = m_underlying_.format(mat.coeff(row, col), ctx);
            }

            if (row < mat.rows() - 1) { out = fmt::format_to(out, "\n"); }
        }

        return out;
    }
};

template<typename Derived>
struct fmt::is_range<
    Derived,
    std::enable_if_t<std::is_base_of<Eigen::DenseBase<Derived>, Derived>::value, char>>
    : std::false_type {};

#endif
