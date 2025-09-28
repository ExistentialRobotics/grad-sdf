#pragma once

#include "logging.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef ERL_USE_ABSL
    #include <absl/hash/hash.h>
#endif

// https://stackoverflow.com/questions/4433950/overriding-functions-from-dynamic-libraries
// https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
extern "C" {
[[maybe_unused]] int
mkl_serv_intel_cpu_true();
}

namespace Eigen {

#if !EIGEN_VERSION_AT_LEAST(3, 4, 0)  // the eigen version is older
    template<typename T>
    using MatrixX = Matrix<T, Dynamic, Dynamic, ColMajor>;
    template<typename T>
    using Matrix2X = Matrix<T, 2, Dynamic, ColMajor>;
    template<typename T>
    using Matrix3X = Matrix<T, 3, Dynamic, ColMajor>;
    template<typename T>
    using Matrix4X = Matrix<T, 4, Dynamic, ColMajor>;
    template<typename T>
    using Matrix2 = Matrix<T, 2, 2, ColMajor>;
    template<typename T>
    using Matrix3 = Matrix<T, 3, 3, ColMajor>;
    template<typename T>
    using Matrix4 = Matrix<T, 4, 4, ColMajor>;
    template<typename T, int Size>
    using Vector = Matrix<T, Size, 1, ColMajor>;
    template<typename T>
    using Vector2 = Vector<T, 2>;
    template<typename T>
    using Vector3 = Vector<T, 3>;
    template<typename T>
    using Vector4 = Vector<T, 4>;
    template<typename T>
    using VectorX = Vector<T, Dynamic>;
#endif

    // MATRIX
    using MatrixXl = MatrixX<long>;
    using MatrixXb = MatrixX<bool>;
    using MatrixX8U = MatrixX<uint8_t>;
    using Matrix6Xd = Matrix<double, 6, Dynamic>;
    using Matrix6Xf = Matrix<float, 6, Dynamic>;
    using Matrix2Xl = Matrix<long, 2, Dynamic>;
    using Matrix3Xl = Matrix<long, 3, Dynamic>;
    using Matrix23d = Matrix<double, 2, 3>;
    using Matrix23f = Matrix<float, 2, 3>;
    using Matrix24d = Matrix<double, 2, 4>;
    using Matrix24f = Matrix<float, 2, 4>;
    using Matrix34d = Matrix<double, 3, 4>;
    using Matrix34f = Matrix<float, 3, 4>;

    template<typename T, int Rows, int Cols>
    using RMatrix = Matrix<T, Rows, Cols, RowMajor>;

    using RMatrix23d = RMatrix<double, 2, 3>;
    using RMatrix2Xd = RMatrix<double, 2, Dynamic>;

    template<typename T>
    using Scalar = Matrix<T, 1, 1>;
    using Scalari = Scalar<int>;
    using Scalard = Scalar<double>;
    using Scalarf = Scalar<float>;

    // VECTOR
    using Vector2l = Vector2<long>;
    using Vector3l = Vector3<long>;
    using VectorXl = VectorX<long>;
    using VectorXb = VectorX<bool>;
    using VectorX8U = VectorX<uint8_t>;

#ifdef ERL_USE_ABSL
    template<typename H>
    H
    AbslHashValue(H state, const Eigen::Vector2i& v) {
        return H::combine(std::move(state), v[0], v[1]);
    }

    template<typename H>
    H
    AbslHashValue(H state, const Eigen::Vector3i& v) {
        return H::combine(std::move(state), v[0], v[1], v[2]);
    }

    template<typename H>
    H
    AbslHashValue(H state, const Eigen::Vector4i& v) {
        return H::combine(std::move(state), v[0], v[1], v[2], v[3]);
    }

    template<typename H, typename T, int Rows, int Cols>
    H
    AbslHashValue(H h, const Matrix<T, Rows, Cols>& mat) {
        auto data = mat.data();
        if (Rows == Dynamic || Cols == Dynamic) {
            for (int i = 0; i < mat.size(); ++i) { h = H::combine(std::move(h), data[i]); }
        } else {
            for (int i = 0; i < Rows * Cols; ++i) { h = H::combine(std::move(h), data[i]); }
        }
        return h;
    }
#endif
}  // namespace Eigen

namespace erl::common {

    template<typename T, int Rows, int Cols>
    Eigen::MatrixX<T>
    DownsampleEigenMatrix(const Eigen::Matrix<T, Rows, Cols>& mat, int row_stride, int col_stride) {
        ERL_ASSERTM(
            row_stride > 0 && col_stride > 0,
            "DownsampleEigenMatrix: row_stride and col_stride must be positive.");
        if (row_stride == 1 && col_stride == 1) { return mat; }  // no downsampling

        Eigen::MatrixX<T> downsampled(
            mat.rows() / row_stride + (mat.rows() % row_stride != 0),
            mat.cols() / col_stride + (mat.cols() % col_stride != 0));
        for (long c = 0; c < downsampled.cols(); ++c) {
            T* out = downsampled.col(c).data();
            const T* in = mat.col(c * col_stride).data();
            for (long r = 0; r < downsampled.rows(); ++r) { out[r] = in[r * row_stride]; }
        }
        return downsampled;
    }

    template<typename T, int Rows1, int Cols1, int Rows2, int Cols2>
    bool
    SafeEigenMatrixEqual(
        const Eigen::Matrix<T, Rows1, Cols1>& lhs,
        const Eigen::Matrix<T, Rows2, Cols2>& rhs) {
        static_assert(Rows1 == Eigen::Dynamic || Rows2 == Eigen::Dynamic || Rows1 == Rows2);
        static_assert(Cols1 == Eigen::Dynamic || Cols2 == Eigen::Dynamic || Cols1 == Cols2);
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) { return false; }
        if (lhs.size() == 0) { return true; }
        // we use `std::memcmp` to compare the data of the two matrices.
        // it is safe because the data is contiguous and the size of the data is the same.
        return std::memcmp(lhs.data(), rhs.data(), sizeof(T) * lhs.size()) == 0;
    }

    template<typename T>
    bool
    SafeEigenMatrixRefEqual(
        const Eigen::Ref<const Eigen::MatrixX<T>>& lhs,
        const Eigen::Ref<const Eigen::MatrixX<T>>& rhs) {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) { return false; }
        return lhs == rhs;
    }

    template<typename T1, typename T2>
    bool
    SafeEigenMapEqual(const Eigen::Map<T1>& lhs, const Eigen::Map<T2>& rhs) {
        if (sizeof(typename T1::Scalar) != sizeof(typename T2::Scalar)) { return false; }
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) { return false; }
        return std::memcmp(lhs.data(), rhs.data(), sizeof(typename T1::Scalar) * lhs.size()) == 0;
    }

    template<typename T>
    bool
    SafeSparseEigenMatrixEqual(
        const Eigen::SparseMatrix<T>& lhs,
        const Eigen::SparseMatrix<T>& rhs) {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) { return false; }
        if (lhs.nonZeros() != rhs.nonZeros()) { return false; }
        for (int i = 0; i < lhs.outerSize(); ++i) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(lhs, i); it; ++it) {
                if (it.value() != rhs.coeff(it.row(), it.col())) { return false; }
            }
        }
        return true;
    }

    // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    enum class EigenTextFormat {
        kDefaultFmt = 0,
        kCommaInitFmt = 1,
        kCleanFmt = 2,
        kOctaveFmt = 3,
        kNumpyFmt = 4,
        kCsvFmt = 5,
    };

    Eigen::IOFormat
    GetEigenTextFormat(EigenTextFormat format);

    template<typename T>
    void
    SaveEigenMatrixToTextFile(
        const std::string& file_path,
        const Eigen::Ref<const Eigen::MatrixX<T>>& matrix,
        EigenTextFormat format = EigenTextFormat::kDefaultFmt);

    /**
     * Load Eigen matrix from text file.
     * @tparam T Type of the matrix elements.
     * @tparam Rows Number of rows of the matrix. Use Eigen::Dynamic for dynamic size.
     * @tparam Cols Number of columns of the matrix. Use Eigen::Dynamic for dynamic size.
     * @tparam RowMajor Storage order of the matrix. Use Eigen::ColMajor for column-major order,
     * Eigen::RowMajor for row-major order.
     * @param file_path Path to the text file.
     * @param format Format of the text file.
     * @param transpose Whether the returned matrix is the transpose of the matrix in the file.
     * @return Eigen matrix loaded from the text file.
     */
    template<
        typename T,
        int Rows = Eigen::Dynamic,
        int Cols = Eigen::Dynamic,
        int RowMajor = Eigen::ColMajor>
    Eigen::Matrix<T, Rows, Cols, RowMajor>
    LoadEigenMatrixFromTextFile(
        const std::string& file_path,
        EigenTextFormat format = EigenTextFormat::kDefaultFmt,
        bool transpose = false);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    [[nodiscard]] bool
    SaveEigenMapToBinaryStream(
        std::ostream& s,
        const Eigen::Map<const Eigen::Matrix<T, Rows, Cols>>& matrix);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    [[nodiscard]] bool
    SaveEigenMatrixToBinaryStream(std::ostream& s, const Eigen::Matrix<T, Rows, Cols>& matrix);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    [[nodiscard]] bool
    SaveEigenMatrixToBinaryFile(
        const std::string& file_path,
        const Eigen::Matrix<T, Rows, Cols>& matrix);

    template<typename T, int Rows, int Cols>
    [[nodiscard]] bool
    SaveVectorOfEigenMatricesToBinaryStream(
        std::ostream& s,
        const std::vector<Eigen::Matrix<T, Rows, Cols>>& matrices);

    template<
        typename T,
        int Rows1,
        int Cols1,
        int Rows2 = Eigen::Dynamic,
        int Cols2 = Eigen::Dynamic>
    [[nodiscard]] bool
    SaveEigenMatrixOfEigenMatricesToBinaryStream(
        std::ostream& s,
        const Eigen::Matrix<Eigen::Matrix<T, Rows1, Cols1>, Rows2, Cols2>& matrix_of_matrices);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    [[nodiscard]] bool
    LoadEigenMatrixFromBinaryStream(std::istream& s, Eigen::Matrix<T, Rows, Cols>& matrix);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    Eigen::Matrix<T, Rows, Cols>
    LoadEigenMatrixFromBinaryFile(const std::string& file_path);

    template<typename T = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
    [[nodiscard]] bool
    LoadEigenMapFromBinaryStream(std::istream& s, Eigen::Map<Eigen::Matrix<T, Rows, Cols>> matrix);

    template<typename T = double, int Rows, int Cols>
    [[nodiscard]] bool
    LoadVectorOfEigenMatricesFromBinaryStream(
        std::istream& s,
        std::vector<Eigen::Matrix<T, Rows, Cols>>& matrices);

    template<
        typename T,
        int Rows1,
        int Cols1,
        int Rows2 = Eigen::Dynamic,
        int Cols2 = Eigen::Dynamic>
    [[nodiscard]] bool
    LoadEigenMatrixOfEigenMatricesFromBinaryStream(
        std::istream& s,
        Eigen::Matrix<Eigen::Matrix<T, Rows1, Cols1>, Rows2, Cols2>& matrix_of_matrices);

    template<EigenTextFormat Format, typename Matrix>
    std::string
    EigenToString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToDefaultFmtString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToCommaInitFmtString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToCleanFmtString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToOctaveFmtString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToNumPyFmtString(const Matrix& matrix);

    template<typename Matrix>
    std::string
    EigenToCsvFmtString(const Matrix& matrix);
}  // namespace erl::common

#include "eigen.tpp"
