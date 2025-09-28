#include "erl_geometry/marching_squares.hpp"

#include <absl/container/flat_hash_map.h>

/**
 * @brief Implementation of Marching Square Algorithm.
 * @see https://ieeexplore.ieee.org/document/1219671
 * @see https://en.wikipedia.org/wiki/Marching_squares
 *
 * The cell number v is based on the clockwise sum of the following pattern:
 *  8 --- 4
 *  |     |
 *  1 --- 2
 *
 * The coordinate system is (0, 0) at top left, x-axis downwards, y-axis rightwards.
 *
 * The edge of a cell is
 * -------------------------> U
 * | 3:(0, 0)  1   2:(1, 0)
 * |       ########
 * |     0 #      # 2
 * |       ########
 * | 0:(0, 1)  3   1:(1, 1)
 * V
 *
 * There are 16 kinds of contour lines. For each case, we need to consider different edges to do
 * sub-pixel computation.
 * |  v  |    pattern   | edges to consider |
 * |  0  | [0, 0; 0, 0] | None              |
 * |  1  | [0, 0; 1, 0] | 0, 3              |
 * |  2  | [0, 0; 0, 1] | 2, 3              |
 * |  3  | [0, 0; 1, 1] | 0, 2              |
 * |  4  | [0, 1; 0, 0] | 1, 2              |
 * |  5  | [0, 1; 1, 0] | 0, 3 and 1, 2     |
 * |  6  | [0, 1; 0, 1] | 1, 3              |
 * |  7  | [0, 1; 1, 1] | 0, 1              |
 * |  8  | [1, 0; 0, 0] | 0, 1              |
 * |  9  | [1, 0; 1, 0] | 1, 3              |
 * |  10 | [1, 0; 0, 1] | 0, 1 and 2, 3     |
 * |  11 | [1, 0; 1, 1] | 1, 2              |
 * |  12 | [1, 1; 0, 0] | 0, 2              |
 * |  13 | [1, 1; 1, 0] | 2, 3              |
 * |  14 | [1, 1; 0, 1] | 0, 3              |
 * |  15 | [1, 1; 1, 1] | None              |
 */

namespace erl::geometry {

    const std::array<MarchingSquares::Edge, 5> MarchingSquares::kBaseEdgeTable = {
        0, 0, 0, 1,  // edge 0
        0, 0, 1, 0,  // edge 1
        1, 0, 1, 1,  // edge 2
        0, 1, 1, 1,  // edge 3
        4, 4, 4, 4   // None
    };

    const std::array<std::array<MarchingSquares::Edge, 4>, 16> MarchingSquares::kEdgePairTable = {
        kBaseEdgeTable[4], kBaseEdgeTable[4], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[2], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[2], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[1], kBaseEdgeTable[2], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[3], kBaseEdgeTable[1], kBaseEdgeTable[2],  // and 1, 2
        kBaseEdgeTable[1], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[1], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[1], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[1], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[1], kBaseEdgeTable[2], kBaseEdgeTable[3],  // and 2, 3
        kBaseEdgeTable[1], kBaseEdgeTable[2], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[2], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[2], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[0], kBaseEdgeTable[3], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
        kBaseEdgeTable[4], kBaseEdgeTable[4], kBaseEdgeTable[4], kBaseEdgeTable[4],  //
    };  // 16 x 4 Edge, 16 x 16 long

    const int MarchingSquares::kUniqueEdgeIndexTable[16][5] = {
        {-1, -1, -1, -1, -1},  // 0
        {0, 3, -1, -1, -1},    // 1
        {2, 3, -1, -1, -1},    // 2
        {0, 2, -1, -1, -1},    // 3
        {1, 2, -1, -1, -1},    // 4
        {0, 3, 1, 2, -1},      // 5
        {1, 3, -1, -1, -1},    // 6
        {0, 1, -1, -1, -1},    // 7
        {0, 1, -1, -1, -1},    // 8
        {1, 3, -1, -1, -1},    // 9
        {0, 1, 2, 3, -1},      // 10
        {1, 2, -1, -1, -1},    // 11
        {0, 2, -1, -1, -1},    // 12
        {2, 3, -1, -1, -1},    // 13
        {0, 3, -1, -1, -1},    // 14
        {-1, -1, -1, -1, -1}   // 15
    };

    const int MarchingSquares::kLineVertexIndexTable[16][5] = {
        {-1, -1, -1, -1, -1},  // 0
        {0, 1, -1, -1, -1},    // 1
        {0, 1, -1, -1, -1},    // 2
        {0, 1, -1, -1, -1},    // 3
        {0, 1, -1, -1, -1},    // 4
        {0, 1, 2, 3, -1},      // 5
        {0, 1, -1, -1, -1},    // 6
        {0, 1, -1, -1, -1},    // 7
        {0, 1, -1, -1, -1},    // 8
        {0, 1, -1, -1, -1},    // 9
        {0, 1, 2, 3, -1},      // 10
        {0, 1, -1, -1, -1},    // 11
        {0, 1, -1, -1, -1},    // 12
        {0, 1, -1, -1, -1},    // 13
        {0, 1, -1, -1, -1},    // 14
        {-1, -1, -1, -1, -1}   // 15
    };

    const int MarchingSquares::kEdgeVertexIndexTable[4][2] = {
        {0, 3},  // edge 0
        {3, 2},  // edge 1
        {2, 1},  // edge 2
        {1, 0}   // edge 3
    };

    const int MarchingSquares::kSquareVertexCodes[4][2] = {
        {0, 1},  // vertex 0
        {1, 1},  // vertex 1
        {1, 0},  // vertex 2
        {0, 0}   // vertex 3
    };

    const int MarchingSquares::kSquareEdgeCodes[4][3] = {
        {0, 0, 2},  // edge 0, (u, v, 0), where 2 means v
        {0, 0, 1},  // edge 1, (u, v, 1), where 1 means u
        {1, 0, 2},  // edge 2, (u + 1, v, 0), where 2 means v
        {0, 1, 1}   // edge 3, (u, v + 1, 1), where 1 means u
    };

    template<typename Dtype>
    int
    CalculateVertexConfigIndexImpl(const Dtype *vertex_values, Dtype iso_value) {
        return (vertex_values[0] <= iso_value ? 1 : 0) | (vertex_values[1] <= iso_value ? 2 : 0) |
               (vertex_values[2] <= iso_value ? 4 : 0) | (vertex_values[3] <= iso_value ? 8 : 0);
    }

    template<typename Dtype>
    void
    MarchingSingleSquareImpl(
        const Eigen::Ref<const Eigen::Matrix<Dtype, 2, 4>> &vertex_coords,
        const Eigen::Ref<const Eigen::Vector<Dtype, 4>> &values,
        const Dtype iso_value,
        std::vector<Eigen::Vector2<Dtype>> &vertices,
        std::vector<Eigen::Vector2i> &lines) {

        vertices.clear();
        lines.clear();

        const int config = CalculateVertexConfigIndexImpl<Dtype>(values.data(), iso_value);
        if (config <= 0 || config >= 15) { return; }

        auto interpolate = [&](const MarchingSquares::Edge &e) -> Eigen::Vector2<Dtype> {
            constexpr Dtype kEpsilon = 1e-6f;
            long idx1 = e.v1x | (e.v1y << 1);
            long idx2 = e.v2x | (e.v2y << 1);

            const Dtype *v1 = vertex_coords.col(idx1).data();
            const Dtype *v2 = vertex_coords.col(idx2).data();

            const Dtype val_diff = values[idx1] - values[idx2];
            if (std::abs(val_diff) >= kEpsilon) {
                const Dtype t = (values[idx1] - iso_value) / val_diff;
                return {v1[0] + (v2[0] - v1[0]) * t, v1[1] + (v2[1] - v1[1]) * t};
            }
            return {(v1[0] + v2[0]) * 0.5f, (v1[1] + v2[1]) * 0.5f};
        };

        // compute edge intersections
        vertices.emplace_back(interpolate(MarchingSquares::kEdgePairTable[config][0]));
        vertices.emplace_back(interpolate(MarchingSquares::kEdgePairTable[config][1]));
        lines.emplace_back(0, 1);
        if (config == 5 || config == 10) {
            vertices.emplace_back(interpolate(MarchingSquares::kEdgePairTable[config][2]));
            vertices.emplace_back(interpolate(MarchingSquares::kEdgePairTable[config][3]));
            lines.emplace_back(2, 3);
        }
    }

    static void
    SortLinesToObjects(Eigen::Matrix2Xi &lines_to_vertices, Eigen::Matrix2Xi &objects_to_lines) {
        const long num_lines = lines_to_vertices.cols();
        // estimated maximum number of objects
        objects_to_lines.setConstant(2, num_lines + 1, -1);
        objects_to_lines(0, 0) = 0;

        bool reverse = false;
        int num_objects = 0;
        for (int line_idx = 0; line_idx < num_lines; ++line_idx) {
            int vertex_idx = lines_to_vertices(1, line_idx);
            const int next_line_idx = line_idx + 1;

            // find the next line that is connected to the current line.
            int next_connected_line_idx = next_line_idx;
            for (; next_connected_line_idx < num_lines; ++next_connected_line_idx) {
                if (lines_to_vertices(0, next_connected_line_idx) == vertex_idx ||
                    lines_to_vertices(1, next_connected_line_idx) == vertex_idx) {
                    break;
                }
            }

            if (next_connected_line_idx != num_lines) {
                // find a line.
                // extend: swap the found line to the next_line position.
                lines_to_vertices.col(next_line_idx)
                    .swap(lines_to_vertices.col(next_connected_line_idx));
                if (lines_to_vertices(1, next_line_idx) == vertex_idx) {
                    std::swap(
                        lines_to_vertices(0, next_line_idx),
                        lines_to_vertices(1, next_line_idx));
                }
            } else {
                // reverse the line sequence of the current object, try to extend it from the
                // other end
                const int obj_begin_line_idx = objects_to_lines(0, num_objects);
                vertex_idx = lines_to_vertices(0, obj_begin_line_idx);

                // find the next line that is connected to the current line.
                next_connected_line_idx = next_line_idx;
                for (; next_connected_line_idx < num_lines; ++next_connected_line_idx) {
                    if (lines_to_vertices(0, next_connected_line_idx) == vertex_idx ||
                        lines_to_vertices(1, next_connected_line_idx) == vertex_idx) {
                        break;
                    }
                }

                if (next_connected_line_idx != num_lines) {
                    // reverse the sequence of object's vertices
                    reverse = true;
                    auto block = lines_to_vertices.block(
                        0,
                        obj_begin_line_idx,
                        2,
                        next_line_idx - obj_begin_line_idx);
                    // reverse each column: swap line start and end
                    block.colwise().reverseInPlace();
                    // reverse each row: reverse line order
                    block.rowwise().reverseInPlace();

                    // extend: swap the found line to the next_line position.
                    lines_to_vertices.col(next_line_idx)
                        .swap(lines_to_vertices.col(next_connected_line_idx));
                    if (lines_to_vertices(1, next_line_idx) == vertex_idx) {
                        // reverse the vertex order of the line.
                        std::swap(
                            lines_to_vertices(0, next_line_idx),
                            lines_to_vertices(1, next_line_idx));
                    }
                } else {  // both ends cannot be extended anymore
                    if (reverse) {
                        // recover the original vertex order
                        auto block = lines_to_vertices.block(
                            0,
                            obj_begin_line_idx,
                            2,
                            next_line_idx - obj_begin_line_idx);
                        // reverse each column: swap line start and end
                        block.colwise().reverseInPlace();
                        // reverse each row: reverse line order
                        block.rowwise().reverseInPlace();
                        reverse = false;
                    }
                    objects_to_lines(1, num_objects++) = next_line_idx;
                    objects_to_lines(0, num_objects) = next_line_idx;
                }
            }
        }

        if (objects_to_lines(0, num_objects) == -1) {
            objects_to_lines(1, num_objects++) = static_cast<int>(num_lines);
        }
        objects_to_lines.conservativeResize(2, num_objects);
    }

    template<typename Dtype>
    void
    MarchingSquareImpl(
        const Eigen::Ref<const Eigen::MatrixX<Dtype>> &img,
        const Dtype iso_value,
        Eigen::Matrix2X<Dtype> &vertices,
        Eigen::Matrix2Xi &lines_to_vertices,
        Eigen::Matrix2Xi &objects_to_lines) {

        const long img_height = img.rows();
        const long img_width = img.cols();
        // binary mGoalMask of img <= iso_value
        auto b_mat = Eigen::MatrixXb(img_height, img_width);
        using MS = MarchingSquares;
        std::vector<MS::Edge> edges;
        absl::flat_hash_map<MS::Edge, int> unique_edges;
        edges.reserve(img_height * img_width);
        unique_edges.reserve(img_height * img_width);

        // 1. compute the first row of b_mat
        for (long y = 0; y < img_width; y++) { b_mat(0, y) = img(0, y) <= iso_value; }

        // 2. compute vMat
        //      a. compute x+1 row of b_mat
        //      b. compute v, Update edges, unique_edges and lines_to_vertices
        auto get_edge_index = [&](const MS::Edge &e) -> int {
            // assign value to `e` only when it is a new key
            auto [map_pair, is_new_edge] = unique_edges.try_emplace(e, edges.size());
            if (is_new_edge) { edges.push_back(e); }
            auto &[edge, edge_index] = *map_pair;
            return edge_index;
        };

        int num_lines = 0;
        for (long v = 0; v < img_height - 1; v++) {
            b_mat(v + 1, 0) = img(v + 1, 0) <= iso_value;
            for (long u = 0; u < img_width - 1; u++) {
                b_mat(v + 1, u + 1) = img(v + 1, u + 1) <= iso_value;

                if (const int val = b_mat(v, u) << 3 | b_mat(v, u + 1) << 2 |
                                    b_mat(v + 1, u + 1) << 1 | b_mat(v + 1, u);
                    val > 0 && val < 15) {
                    const auto &[e1v1x, e1v1y, e1v2x, e1v2y] = MS::kEdgePairTable[val][0];
                    int idx_1 = get_edge_index({u + e1v1x, v + e1v1y, u + e1v2x, v + e1v2y});

                    const auto &[e2v1x, e2v1y, e2v2x, e2v2y] = MS::kEdgePairTable[val][1];
                    int idx_2 = get_edge_index({u + e2v1x, v + e2v1y, u + e2v2x, v + e2v2y});

                    if (lines_to_vertices.cols() == num_lines) {
                        lines_to_vertices.conservativeResize(2, 2 * num_lines + 1);
                    }
                    lines_to_vertices.col(num_lines++) << idx_1, idx_2;

                    if (val == 5 || val == 10) {
                        const auto &[e3v1x, e3v1y, e3v2x, e3v2y] = MS::kEdgePairTable[val][2];
                        int idx_3 = get_edge_index({u + e3v1x, v + e3v1y, u + e3v2x, v + e3v2y});

                        const auto &[e4v1x, e4v1y, e4v2x, e4v2y] = MS::kEdgePairTable[val][3];
                        int idx_4 = get_edge_index({u + e4v1x, v + e4v1y, u + e4v2x, v + e4v2y});

                        if (lines_to_vertices.cols() == num_lines) {
                            lines_to_vertices.conservativeResize(2, 2 * num_lines + 1);
                        }
                        lines_to_vertices.col(num_lines++) << idx_3, idx_4;
                    }
                }
            }
        }
        lines_to_vertices.conservativeResize(2, num_lines);

        // 3. compute sub-pixel vertex coordinate by interpolation
        vertices.resize(2, static_cast<ssize_t>(edges.size()));
        for (ssize_t i = 0; i < static_cast<ssize_t>(edges.size()); ++i) {
            const auto &[v1x, v1y, v2x, v2y] = edges[i];
            const Dtype w1 = std::abs(img(v1y, v1x) - iso_value);
            const Dtype w2 = std::abs(img(v2y, v2x) - iso_value);
            const Dtype a = w2 / (w1 + w2);
            // clang-format off
            vertices.col(i) << static_cast<Dtype>(v1x) * a + static_cast<Dtype>(v2x) * (1.0f - a),
                               static_cast<Dtype>(v1y) * a + static_cast<Dtype>(v2y) * (1.0f - a);
            // clang-format on
        }

        // 4. find objects
        SortLinesToObjects(lines_to_vertices, objects_to_lines);
    }

    int
    MarchingSquares::CalculateVertexConfigIndex(
        const double *vertex_values,
        const double iso_value) {
        return CalculateVertexConfigIndexImpl<double>(vertex_values, iso_value);
    }

    int
    MarchingSquares::CalculateVertexConfigIndex(const float *vertex_values, const float iso_value) {
        return CalculateVertexConfigIndexImpl<float>(vertex_values, iso_value);
    }

    void
    MarchingSquares::SingleSquare(
        const Eigen::Ref<const Eigen::Matrix<double, 2, 4>> &vertex_coords,
        const Eigen::Ref<const Eigen::Vector<double, 4>> &values,
        const double iso_value,
        std::vector<Eigen::Vector2d> &vertices,
        std::vector<Eigen::Vector2i> &lines) {
        MarchingSingleSquareImpl<double>(vertex_coords, values, iso_value, vertices, lines);
    }

    void
    MarchingSquares::SingleSquare(
        const Eigen::Ref<const Eigen::Matrix<float, 2, 4>> &vertex_coords,
        const Eigen::Ref<const Eigen::Vector<float, 4>> &values,
        const float iso_value,
        std::vector<Eigen::Vector2f> &vertices,
        std::vector<Eigen::Vector2i> &lines) {
        MarchingSingleSquareImpl<float>(vertex_coords, values, iso_value, vertices, lines);
    }

    void
    MarchingSquares::Run(
        const Eigen::Ref<const Eigen::MatrixXd> &img,
        const double iso_value,
        Eigen::Matrix2Xd &vertices,
        Eigen::Matrix2Xi &lines_to_vertices,
        Eigen::Matrix2Xi &objects_to_lines) {
        MarchingSquareImpl<double>(img, iso_value, vertices, lines_to_vertices, objects_to_lines);
    }

    void
    MarchingSquares::Run(
        const Eigen::Ref<const Eigen::MatrixXf> &img,
        const float iso_value,
        Eigen::Matrix2Xf &vertices,
        Eigen::Matrix2Xi &lines_to_vertices,
        Eigen::Matrix2Xi &objects_to_lines) {
        MarchingSquareImpl<float>(img, iso_value, vertices, lines_to_vertices, objects_to_lines);
    }

}  // namespace erl::geometry
