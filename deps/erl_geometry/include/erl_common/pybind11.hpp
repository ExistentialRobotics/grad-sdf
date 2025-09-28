#pragma once

#include "eigen.hpp"

#include <pybind11/pybind11.h>  // must be included first
// https://github.com/pybind/pybind11/blob/d2e7e8c68711d1ebfb02e2f20bd1cb3bfc5647c0/docs/basics.rst#L81-L86

namespace py = pybind11;

// NUMPY
#include <pybind11/numpy.h>

// EIGEN
#include <pybind11/eigen.h>

// clang-format off
/**
 * NumPy to Eigen behavior
 * Type                                    | Accept c-style | Accept f-style | require writable | Posted to compatible different Ref types |
 * py::EigenDRef<const Eigen::MatrixXd>    | ref            | ref            | no               | temp copy                                |
 * py::EigenDRef<Eigen::MatrixXd>          | ref            | ref            | yes              | compile time error                       |
 * Eigen::Ref<const Eigen::MatrixXd>       | copy           | ref            | no               | temp copy                                |
 * Eigen::Ref<const ERMatrix<double>>      | copy           | ref            | no               | temp copy                                |
 * Eigen::Ref<Eigen::MatrixXd>             | no             | ref            | yes              | compile time error                       |
 * Eigen::Ref<ERMatrix<double>>            | ref            | no             | yes              | compile time error                       |
 *
 * Eigen::MatrixXd::conservativeResize can be used to replace std::vector<gpis::Point<T, Dim>>
 */
// clang-format on

#ifdef ERL_USE_OPENCV
    #include "pybind11_opencv.hpp"
#endif

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#define ERL_PYBIND_WRAP_NAME_PROPERTY_AS_READONLY(py_cls, cls, name, property) \
    py_cls.def_property_readonly(name, [](const cls& obj) { return obj.property; })
#define ERL_PYBIND_WRAP_PROPERTY_AS_READONLY(py_cls, cls, property) \
    ERL_PYBIND_WRAP_NAME_PROPERTY_AS_READONLY(py_cls, cls, #property, property)

namespace PYBIND11_NAMESPACE {
    template<typename T>
    using SupportedByNumpy = detail::any_of<detail::is_pod_struct<T>, std::is_arithmetic<T>>;

    template<typename T, int Dim>
    array_t<T>
    cast_to_array(const Eigen::MatrixX<Eigen::Vector<T, Dim>>& mat) {
        static_assert(Dim >= 1, "Dim must be greater than or equal to 1.");
        array_t<T> out({mat.rows(), mat.cols(), static_cast<long>(Dim)});
        for (long i = 0; i < mat.rows(); ++i) {
            for (long j = 0; j < mat.cols(); ++j) {
                const Eigen::Vector<T, Dim>& vec = mat(i, j);
                for (int k = 0; k < Dim; ++k) { out.mutable_at(i, j, k) = vec[k]; }
            }
        }
        return out;
    }

    template<typename T>
    class RawPtrWrapper {
        T* m_ptr_ = nullptr;

    public:
        RawPtrWrapper() = default;

        explicit RawPtrWrapper(T* ptr)
            : m_ptr_(ptr) {}

        T&
        operator*() const {
            return *m_ptr_;
        }

        T*
        operator->() const {
            return m_ptr_;
        }

        T&
        operator[](std::size_t i) const {
            return m_ptr_[i];
        }

        T*
        get() const {
            return m_ptr_;
        }
    };

    namespace detail {
        template<typename Iterator>
        struct iterator_self_access {
            using result_type = Iterator&;

            result_type
            operator()(Iterator& it) const {
                return it;
            }
        };
    }  // namespace detail

#if PYBIND11_VERSION_MAJOR >= 2 && PYBIND11_VERSION_MINOR >= 12
    template<
        return_value_policy Policy = return_value_policy::reference_internal,
        typename Iterator,
        typename Sentinel,
        typename ValueType = typename detail::iterator_self_access<Iterator>::result_type,
        typename... Extra>
    typing::Iterator<ValueType>
    wrap_iterator(Iterator first, Sentinel last, Extra&&... extra) {
        return detail::make_iterator_impl<
            detail::iterator_self_access<Iterator>,
            Policy,
            Iterator,
            Sentinel,
            ValueType,
            Extra...>(
            std::forward<Iterator>(first),
            std::forward<Sentinel>(last),
            std::forward<Extra>(extra)...);
    }
#else
    template<
        return_value_policy Policy = return_value_policy::reference_internal,
        typename Iterator,
        typename Sentinel,
        typename ValueType = typename detail::iterator_self_access<Iterator>::result_type,
        typename... Extra>
    auto
    wrap_iterator(Iterator /* first */, Sentinel /* last */, Extra&&... /* extra */) {
        py::print("pybind11 version is too old, please update to 2.12 or later.");
        return py::none();
    }
#endif
}  // namespace PYBIND11_NAMESPACE

PYBIND11_DECLARE_HOLDER_TYPE(T, RawPtrWrapper<T>);
