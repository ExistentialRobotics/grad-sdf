#pragma once

#include "erl_common/eigen.hpp"

namespace erl::geometry {

    struct MarchingSquares {
        struct Edge {
            long v1x, v1y, v2x, v2y;

            bool
            operator==(const Edge &other) const {
                return v1x == other.v1x && v1y == other.v1y && v2x == other.v2x && v2y == other.v2y;
            }

            template<typename H>
            friend H
            AbslHashValue(H h, const Edge &e) {
                return H::combine(std::move(h), e.v1x, e.v1y, e.v2x, e.v2y);
            }
        };

        struct HashEdge {
            std::size_t
            operator()(const Edge &e) const noexcept {
                constexpr std::hash<long> long_hash;
                std::size_t &&h_1 = long_hash(e.v1x);
                std::size_t &&h_2 = long_hash(e.v1y);
                std::size_t &&h_3 = long_hash(e.v2x);
                std::size_t &&h_4 = long_hash(e.v2y);
                return h_1 ^ h_2 << 1 ^ h_3 << 1 ^ h_4 << 1;
            }
        };

        static const std::array<Edge, 5> kBaseEdgeTable;
        static const std::array<std::array<Edge, 4>, 16> kEdgePairTable;
        static const int kUniqueEdgeIndexTable[16][5];
        static const int kLineVertexIndexTable[16][5];
        static const int kEdgeVertexIndexTable[4][2];
        static const int kSquareVertexCodes[4][2];
        static const int kSquareEdgeCodes[4][3];

        static const int *
        GetEdgeIndices(int config) {
            if (config <= 0 || config >= 15) { return nullptr; }
            return kUniqueEdgeIndexTable[config];
        }

        static const int *
        GetVertexIndices(int config) {
            if (config <= 0 || config >= 15) { return nullptr; }
            return kLineVertexIndexTable[config];
        }

        static const int *
        GetUniqueEdgeIndices(int config) {
            if (config <= 0 || config >= 15) { return nullptr; }
            return kUniqueEdgeIndexTable[config];
        }

        static const int *
        GetEdgeVertexIndices(int edge_index) {
            if (edge_index < 0 || edge_index >= 4) { return nullptr; }
            return kEdgeVertexIndexTable[edge_index];
        }

        static const int *
        GetVertexCode(int vertex_index) {
            if (vertex_index < 0 || vertex_index >= 4) { return nullptr; }
            return kSquareVertexCodes[vertex_index];
        }

        static const int *
        GetEdgeCode(int edge_index) {
            if (edge_index < 0 || edge_index >= 4) { return nullptr; }
            return kSquareEdgeCodes[edge_index];
        }

        static int
        CalculateVertexConfigIndex(const double *vertex_values, double iso_value);

        static int
        CalculateVertexConfigIndex(const float *vertex_values, float iso_value);

        static void
        SingleSquare(
            const Eigen::Ref<const Eigen::Matrix<double, 2, 4>> &vertex_coords,
            const Eigen::Ref<const Eigen::Vector<double, 4>> &values,
            double iso_value,
            std::vector<Eigen::Vector2d> &vertices,
            std::vector<Eigen::Vector2i> &lines);

        static void
        SingleSquare(
            const Eigen::Ref<const Eigen::Matrix<float, 2, 4>> &vertex_coords,
            const Eigen::Ref<const Eigen::Vector<float, 4>> &values,
            float iso_value,
            std::vector<Eigen::Vector2f> &vertices,
            std::vector<Eigen::Vector2i> &lines);

        static void
        Run(const Eigen::Ref<const Eigen::MatrixXd> &img,
            double iso_value,
            Eigen::Matrix2Xd &vertices,
            Eigen::Matrix2Xi &lines_to_vertices,
            Eigen::Matrix2Xi &objects_to_lines);

        static void
        Run(const Eigen::Ref<const Eigen::MatrixXf> &img,
            float iso_value,
            Eigen::Matrix2Xf &vertices,
            Eigen::Matrix2Xi &lines_to_vertices,
            Eigen::Matrix2Xi &objects_to_lines);

        static void
        SortLinesToObjects(Eigen::Matrix2Xi &lines_to_vertices, Eigen::Matrix2Xi &objects_to_lines);
    };
}  // namespace erl::geometry
