#pragma once

#include <fstream>

namespace erl::common {
    template<typename T>
    void
    SaveEigenMatrixToTextFile(
        const std::string &file_path,
        const Eigen::Ref<const Eigen::MatrixX<T>> &matrix,
        const EigenTextFormat format) {
        std::ofstream ofs(file_path);
        if (!ofs.is_open()) { ERL_FATAL("Could not open file {}", file_path); }
        ofs << matrix.format(GetEigenTextFormat(format));
        ofs.close();
    }

    template<typename T, int Rows, int Cols, int RowMajor>
    Eigen::Matrix<T, Rows, Cols, RowMajor>
    LoadEigenMatrixFromTextFile(
        const std::string &file_path,
        const EigenTextFormat format,
        const bool transpose) {
        std::vector<T> data;
        std::ifstream ifs(file_path);

        if (!ifs.is_open()) { ERL_FATAL("Could not open file {}", file_path); }

        std::string row_string;
        std::string entry_string;
        int cols = 0;
        int rows = 0;

        char delim = ',';
        switch (format) {
            case EigenTextFormat::kDefaultFmt:
                delim = ' ';
                break;
            case EigenTextFormat::kCommaInitFmt:
            case EigenTextFormat::kCleanFmt:
            case EigenTextFormat::kOctaveFmt:
            case EigenTextFormat::kNumpyFmt:
            case EigenTextFormat::kCsvFmt:
                delim = ',';
                break;
        }

        while (std::getline(ifs, row_string)) {
            std::stringstream row_stream(row_string);
            int row_cols = 0;
            while (std::getline(row_stream, entry_string, delim)) {
                if (entry_string.empty()) { continue; }
                if (entry_string == ";" || entry_string == "]" || entry_string == "[") { continue; }
                data.push_back(T(std::stod(entry_string)));
                if (rows == 0) {
                    cols++;
                } else {
                    row_cols++;
                }
            }
            if (rows == 0) {
                ERL_ASSERTM(
                    Cols == Eigen::Dynamic || cols == Cols,
                    "Number of columns in file does not match template parameter. Expected {}, got "
                    "{}",
                    Cols,
                    cols);
            } else {
                ERL_ASSERTM(
                    cols == row_cols,
                    "Invalid matrix file: row {} has {} columns, expected {}",
                    rows,
                    row_cols,
                    cols);
            }
            rows++;
        }
        ifs.close();

        if (rows == 0 || cols == 0) {
            ERL_WARN("Reading empty matrix from file {}.", file_path);
            return {};
        }

        // the loaded data is in row-major order
        const bool order_compatible = (RowMajor == Eigen::RowMajor && !transpose) ||
                                      (RowMajor == Eigen::ColMajor && transpose);

        // check if the number of rows and columns matches the template parameters
        if (transpose) {
            // the returned matrix shape should be (cols, rows)
            ERL_ASSERTM(
                Rows == Eigen::Dynamic || cols == Rows,
                "Number of rows in file does not match template parameter. Expected {}, got {}",
                Rows,
                cols);
            ERL_ASSERTM(
                Cols == Eigen::Dynamic || rows == Cols,
                "Number of columns in file does not match template parameter. Expected {}, got {}",
                Cols,
                rows);
        } else {
            // the returned matrix shape should be (rows, cols)
            ERL_ASSERTM(
                Rows == Eigen::Dynamic || rows == Rows,
                "Number of rows in file does not match template parameter. Expected {}, got {}",
                Rows,
                rows);
            ERL_ASSERTM(
                Cols == Eigen::Dynamic || cols == Cols,
                "Number of columns in file does not match template parameter. Expected {}, got {}",
                Cols,
                cols);
        }

        if (order_compatible) {
            // copy the data into the matrix directly and return it
            Eigen::Matrix<T, Rows, Cols, RowMajor> matrix(
                transpose ? cols : rows,
                transpose ? rows : cols);
            std::copy(data.begin(), data.end(), matrix.data());
            return matrix;
        }

        // if we reach here, the matrix is not in the expected order
        Eigen::Matrix<T, Rows, Cols, !RowMajor> matrix(
            transpose ? cols : rows,
            transpose ? rows : cols);
        std::copy(data.begin(), data.end(), matrix.data());
        return matrix;
    }

    template<typename T, int Rows, int Cols>
    bool
    SaveEigenMapToBinaryStream(
        std::ostream &s,
        const Eigen::Map<const Eigen::Matrix<T, Rows, Cols>> &matrix) {
        const long matrix_size = matrix.size();
        s.write(reinterpret_cast<const char *>(&matrix_size), sizeof(long));
        if (matrix_size == 0) { return s.good(); }
        const long matrix_shape[2] = {matrix.rows(), matrix.cols()};
        s.write(reinterpret_cast<const char *>(matrix_shape), 2 * sizeof(long));
        s.write(reinterpret_cast<const char *>(matrix.data()), matrix.size() * sizeof(T));
        return s.good();
    }

    template<typename T>
    bool
    SaveSparseEigenMatrixToBinaryStream(std::ostream &s, const Eigen::SparseMatrix<T> &matrix) {
        const long rows = matrix.rows();
        const long cols = matrix.cols();
        const long non_zeros = matrix.nonZeros();
        s.write(reinterpret_cast<const char *>(&rows), sizeof(long));
        s.write(reinterpret_cast<const char *>(&cols), sizeof(long));
        s.write(reinterpret_cast<const char *>(&non_zeros), sizeof(long));
        if (rows == 0 || cols == 0 || non_zeros == 0) { return s.good(); }
        long cnt = 0;
        for (long k = 0; k < matrix.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, k); it; ++it) {
                T value = it.value();
                long row = it.row();
                long col = it.col();
                s.write(reinterpret_cast<const char *>(&row), sizeof(long));
                s.write(reinterpret_cast<const char *>(&col), sizeof(long));
                s.write(reinterpret_cast<const char *>(&value), sizeof(T));
                ++cnt;
            }
        }
        ERL_WARN_COND(
            non_zeros != cnt,
            "Non-zero count mismatch. Expected {}, got {}",
            non_zeros,
            cnt);
        return s.good();
    }

    template<typename T, int Rows, int Cols>
    bool
    SaveEigenMatrixToBinaryStream(std::ostream &s, const Eigen::Matrix<T, Rows, Cols> &matrix) {
        return SaveEigenMapToBinaryStream<T, Rows, Cols>(
            s,
            Eigen::Map<const Eigen::Matrix<T, Rows, Cols>>(
                matrix.data(),
                matrix.rows(),
                matrix.cols()));
    }

    template<typename T, int Rows, int Cols>
    bool
    SaveEigenMatrixToBinaryFile(
        const std::string &file_path,
        const Eigen::Matrix<T, Rows, Cols> &matrix) {
        std::ofstream ofs(file_path, std::ios::binary);
        if (!ofs.is_open()) { ERL_FATAL("Could not open file {}", file_path); }
        const bool success = SaveEigenMatrixToBinaryStream(ofs, matrix);
        ofs.close();
        return success;
    }

    template<typename T, int Rows, int Cols>
    bool
    SaveVectorOfEigenMatricesToBinaryStream(
        std::ostream &s,
        const std::vector<Eigen::Matrix<T, Rows, Cols>> &matrices) {
        const std::size_t num_matrices = matrices.size();
        s.write(reinterpret_cast<const char *>(&num_matrices), sizeof(std::size_t));
        if (Rows != Eigen::Dynamic && Cols != Eigen::Dynamic) {
            return SaveEigenMapToBinaryStream<T, Rows * Cols, Eigen::Dynamic>(
                s,
                Eigen::Map<const Eigen::Matrix<T, Rows * Cols, Eigen::Dynamic>>(
                    matrices.data()->data(),
                    Rows * Cols,
                    static_cast<long>(num_matrices)));
        }
        for (const auto &matrix: matrices) {
            if (!SaveEigenMatrixToBinaryStream<T, Rows, Cols>(s, matrix)) { return false; }
        }
        return s.good();
    }

    template<typename T, int Rows1, int Cols1, int Rows2, int Cols2>
    bool
    SaveEigenMatrixOfEigenMatricesToBinaryStream(
        std::ostream &s,
        const Eigen::Matrix<Eigen::Matrix<T, Rows1, Cols1>, Rows2, Cols2> &matrix_of_matrices) {
        const long rows = matrix_of_matrices.rows();
        const long cols = matrix_of_matrices.cols();
        s.write(reinterpret_cast<const char *>(&rows), sizeof(long));
        s.write(reinterpret_cast<const char *>(&cols), sizeof(long));
        if (rows == 0 || cols == 0) {
            ERL_WARN("Writing empty matrix to stream.");
            return s.good();
        }
        if (Rows1 != Eigen::Dynamic && Cols1 != Eigen::Dynamic) {
            // for performance and smaller file, storage for fixed size matrices is assumed to be
            // contiguous
            return SaveEigenMapToBinaryStream<T, Rows1 * Cols1, Eigen::Dynamic>(
                s,
                Eigen::Map<const Eigen::Matrix<T, Rows1 * Cols1, Eigen::Dynamic>>(
                    matrix_of_matrices.data()->data(),
                    Rows1 * Cols1,
                    rows * cols));
        }
        // Rows1 == Eigen::Dynamic or Cols1 == Eigen::Dynamic
        // warning: the storage of the matrix_of_matrices may not be contiguous
        for (long j = 0; j < cols; j++) {
            for (long i = 0; i < rows; i++) {
                if (!SaveEigenMatrixToBinaryStream<T, Rows1, Cols1>(s, matrix_of_matrices(i, j))) {
                    return false;
                }
            }
        }
        return s.good();
    }

    template<typename T, int Rows, int Cols>
    bool
    LoadEigenMatrixFromBinaryStream(std::istream &s, Eigen::Matrix<T, Rows, Cols> &matrix) {
        long matrix_size = 0;
        s.read(reinterpret_cast<char *>(&matrix_size), sizeof(long));
        if (matrix_size == 0) {
            if constexpr (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
                // ERL_WARN("Reading empty matrix from stream.");
                if (Rows != Eigen::Dynamic) {
                    matrix.resize(Rows, 0);
                } else {
                    if (Cols != Eigen::Dynamic) {
                        matrix.resize(0, Cols);
                    } else {
                        matrix.resize(0, 0);
                    }
                }
                return true;
            } else if (matrix_size != Rows * Cols) {
                ERL_WARN("Matrix size mismatch. Expected {}, got {}", Rows * Cols, matrix_size);
                return false;
            }
        }

        long matrix_shape[2];
        s.read(reinterpret_cast<char *>(matrix_shape), 2 * sizeof(long));
        if (Rows != Eigen::Dynamic && matrix_shape[0] != Rows) {
            ERL_WARN(
                "Number of rows in file does not match template parameter. Expected {}, got {}",
                Rows,
                matrix_shape[0]);
            return false;
        }
        if (Cols != Eigen::Dynamic && matrix_shape[1] != Cols) {
            ERL_WARN(
                "Number of columns in file does not match template parameter. Expected {}, got {}",
                Cols,
                matrix_shape[1]);
            return false;
        }
        if (matrix_size != matrix_shape[0] * matrix_shape[1]) {
            ERL_WARN(
                "Matrix size mismatch. Expected {}, got {}",
                matrix_size,
                matrix_shape[0] * matrix_shape[1]);
            return false;
        }

        if constexpr (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
            if (matrix.rows() != matrix_shape[0] || matrix.cols() != matrix_shape[1]) {
                matrix.resize(matrix_shape[0], matrix_shape[1]);
            }
        }
        s.read(reinterpret_cast<char *>(matrix.data()), static_cast<long>(matrix_size * sizeof(T)));
        if (!s.good()) {
            ERL_WARN("Error reading matrix from stream.");
            return false;
        }
        return s.good();
    }

    template<typename T>
    bool
    LoadSparseEigenMatrixFromBinaryStream(std::istream &s, Eigen::SparseMatrix<T> &matrix) {
        long rows, cols, non_zeros;
        s.read(reinterpret_cast<char *>(&rows), sizeof(long));
        s.read(reinterpret_cast<char *>(&cols), sizeof(long));
        s.read(reinterpret_cast<char *>(&non_zeros), sizeof(long));
        matrix = Eigen::SparseMatrix<T>(rows, cols);
        if (rows == 0 || cols == 0 || non_zeros == 0) { return s.good(); }
        std::vector<Eigen::Triplet<T>> triplets;
        triplets.reserve(non_zeros);
        for (long i = 0; i < non_zeros; ++i) {
            T value;
            long row, col;
            s.read(reinterpret_cast<char *>(&row), sizeof(long));
            s.read(reinterpret_cast<char *>(&col), sizeof(long));
            s.read(reinterpret_cast<char *>(&value), sizeof(T));
            triplets.emplace_back(row, col, value);
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return s.good();
    }

    template<typename T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols>
    LoadEigenMatrixFromBinaryFile(const std::string &file_path) {
        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open()) { ERL_FATAL("Could not open file {}", file_path); }
        Eigen::Matrix<T, Rows, Cols> matrix;
        ERL_ASSERTM(
            LoadEigenMatrixFromBinaryStream(ifs, matrix),
            "Error reading matrix from file.");
        ifs.close();
        return matrix;
    }

    template<typename T, int Rows, int Cols>
    bool
    LoadEigenMapFromBinaryStream(std::istream &s, Eigen::Map<Eigen::Matrix<T, Rows, Cols>> matrix) {
        long matrix_size = 0;
        s.read(reinterpret_cast<char *>(&matrix_size), sizeof(long));
        if (matrix_size == 0) {
            if constexpr (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
                ERL_WARN("Reading empty matrix from stream.");
            }
            return false;
        }

        long matrix_shape[2];
        s.read(reinterpret_cast<char *>(matrix_shape), 2 * sizeof(long));
        if (Rows != Eigen::Dynamic && matrix_shape[0] != Rows) {
            ERL_WARN(
                "Number of rows in file does not match template parameter. Expected {}, got {}",
                Rows,
                matrix_shape[0]);
            return false;
        }
        if (Cols != Eigen::Dynamic && matrix_shape[1] != Cols) {
            ERL_WARN(
                "Number of columns in file does not match template parameter. Expected {}, got {}",
                Cols,
                matrix_shape[1]);
            return false;
        }
        if (matrix_size != matrix_shape[0] * matrix_shape[1]) {
            ERL_WARN(
                "Matrix size mismatch. Expected {}, got {}",
                matrix_size,
                matrix_shape[0] * matrix_shape[1]);
            return false;
        }
        if (matrix.rows() != matrix_shape[0]) {
            ERL_WARN("Matrix rows mismatch. Expected {}, got {}", matrix_shape[0], matrix.rows());
            return false;
        }
        if (matrix.cols() != matrix_shape[1]) {
            ERL_WARN("Matrix cols mismatch. Expected {}, got {}", matrix_shape[1], matrix.cols());
            return false;
        }

        s.read(reinterpret_cast<char *>(matrix.data()), static_cast<long>(matrix_size * sizeof(T)));
        if (!s.good()) {
            ERL_WARN("Error reading matrix from stream.");
            return false;
        }
        return s.good();
    }

    template<typename T, int Rows, int Cols>
    bool
    LoadVectorOfEigenMatricesFromBinaryStream(
        std::istream &s,
        std::vector<Eigen::Matrix<T, Rows, Cols>> &matrices) {
        std::size_t num_matrices = 0;
        s.read(reinterpret_cast<char *>(&num_matrices), sizeof(std::size_t));
        if (num_matrices == 0) { return s.good(); }
        matrices.resize(num_matrices);

        if (Rows != Eigen::Dynamic && Cols != Eigen::Dynamic) {
            return LoadEigenMapFromBinaryStream<T, Rows * Cols, Eigen::Dynamic>(
                s,
                Eigen::Map<Eigen::Matrix<T, Rows * Cols, Eigen::Dynamic>>(
                    matrices.data()->data(),
                    Rows * Cols,
                    static_cast<long>(num_matrices)));
        }

        for (auto &matrix: matrices) {
            if (!LoadEigenMatrixFromBinaryStream<T, Rows, Cols>(s, matrix)) { return false; }
        }
        return s.good();
    }

    template<typename T, int Rows1, int Cols1, int Rows2, int Cols2>
    bool
    LoadEigenMatrixOfEigenMatricesFromBinaryStream(
        std::istream &s,
        Eigen::Matrix<Eigen::Matrix<T, Rows1, Cols1>, Rows2, Cols2> &matrix_of_matrices) {
        long rows, cols;
        s.read(reinterpret_cast<char *>(&rows), sizeof(long));
        s.read(reinterpret_cast<char *>(&cols), sizeof(long));
        if (rows == 0 || cols == 0) {
            ERL_WARN("Reading empty matrix from stream.");
            return s.good();
        }
        if (Rows2 == Eigen::Dynamic || Cols2 == Eigen::Dynamic) {
            matrix_of_matrices.resize(rows, cols);
        }
        if (Rows1 != Eigen::Dynamic && Cols1 != Eigen::Dynamic) {
            // for performance and smaller file, storage for fixed size matrices is assumed to be
            // contiguous
            return LoadEigenMapFromBinaryStream<T, Rows1 * Cols1, Eigen::Dynamic>(
                s,
                Eigen::Map<Eigen::Matrix<T, Rows1 * Cols1, Eigen::Dynamic>>(
                    matrix_of_matrices.data()->data(),
                    Rows1 * Cols1,
                    rows * cols));
        }
        // Rows1 == Eigen::Dynamic or Cols1 == Eigen::Dynamic
        // warning: the storage of the matrix_of_matrices may not be contiguous
        for (long j = 0; j < cols; j++) {
            for (long i = 0; i < rows; i++) {
                if (!LoadEigenMatrixFromBinaryStream<T, Rows1, Cols1>(
                        s,
                        matrix_of_matrices(i, j))) {
                    return false;
                }
            }
        }
        return s.good();
    }

    template<EigenTextFormat Format, typename Matrix>
    std::string
    EigenToString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(Format));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToDefaultFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kDefaultFmt));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToCommaInitFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kCommaInitFmt));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToCleanFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kCleanFmt));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToOctaveFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kOctaveFmt));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToNumPyFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kNumpyFmt));
        return ss.str();
    }

    template<typename Matrix>
    std::string
    EigenToCsvFmtString(const Matrix &matrix) {
        std::stringstream ss;
        ss << matrix.format(GetEigenTextFormat(EigenTextFormat::kCsvFmt));
        return ss.str();
    }

    template<typename IndexType, int Dim>
    std::enable_if_t<Dim == 2 || Dim == 3, Eigen::Matrix<IndexType, Dim, Eigen::Dynamic>>
    GetGridNeighborOffsets(bool include_diagonal) {
        Eigen::Matrix<IndexType, Dim, Eigen::Dynamic> offsets;
        if (!include_diagonal) {
            offsets.setZero(Dim, Dim << 1);
            for (int i = 0; i < Dim; ++i) {
                int j = i << 1;
                offsets(i, j) = 1;
                offsets(i, j + 1) = -1;
            }
            return offsets;
        }

        const int num_neighbors = std::pow(3, Dim) - 1;
        offsets.setZero(Dim, num_neighbors);
        int idx = 0;
        for (int i = 0; i < num_neighbors + 1; ++i) {
            int j = i;
            IndexType *p = offsets.col(idx).data();
            bool is_zero_offset = true;
            for (int k = 0; k < Dim; ++k) {
                p[k] = static_cast<IndexType>(j % 3) - 1;
                if (p[k] != 0) { is_zero_offset = false; }
                j /= 3;
            }
            if (!is_zero_offset) { ++idx; }
        }
        return offsets;
    }
}  // namespace erl::common
