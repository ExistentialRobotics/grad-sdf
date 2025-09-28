#include "erl_common/pybind11.hpp"
#include "erl_geometry/marching_cubes.hpp"

/**
 * unroll the cubes into matrices for easier processing in Python.
 * @param cubes the input cubes, each element is a list of valid cubes in a sub-volume.
 * @param cube_coords output, each column is the (x,y,z) index of a cube
 * @param cube_config_indices output, the configuration index of each cube
 * @param cube_edge_indices output, for each cube, the start and end index of its edges in
 * edge_coords
 * @param edge_coords output, each column is the (x,y,z) index of an edge
 */
void
CubesToMatrices(
    const std::vector<std::vector<erl::geometry::MarchingCubes::ValidCube>> &cubes,
    Eigen::Matrix3Xi &cube_coords,
    Eigen::VectorXi &cube_config_indices,
    Eigen::Matrix2Xi &cube_edge_indices,
    Eigen::Matrix3Xi &edge_coords) {

    int n_cubes = 0;
    int n_edges = 0;
    for (const auto &vec_cubes: cubes) {
        n_cubes += vec_cubes.size();
        for (const auto &cube: vec_cubes) { n_edges += static_cast<int>(cube.edges.size()); }
    }

    cube_coords.resize(3, n_cubes);
    cube_config_indices.resize(n_cubes);
    cube_edge_indices.resize(2, n_cubes);
    edge_coords.resize(3, n_edges);
    n_cubes = 0;
    n_edges = 0;
    for (const auto &vec_cubes: cubes) {
        for (const auto &cube: vec_cubes) {
            cube_coords.col(n_cubes) = cube.coords;
            cube_config_indices[n_cubes] = cube.cfg_index;
            cube_edge_indices.col(n_cubes) << n_edges,
                n_edges + static_cast<int>(cube.edges.size());
            ++n_cubes;
            for (const auto &edge: cube.edges) {
                edge_coords.col(n_edges) = edge;
                ++n_edges;
            }
        }
    }
}

/**
 * reconstruct the cubes from matrices.
 * @param cube_coords input, each column is the (x,y,z) index of a cube
 * @param cube_config_indices input, the configuration index of each cube
 * @param cube_edge_indices input, for each cube, the start and end index of its edges in
 * edge_coords
 * @param edge_coords input, each column is the (x,y,z) index of an edge
 * @param cubes output, each element is a vector of valid cubes.
 */
void
MatricesToCubes(
    const Eigen::Matrix3Xi &cube_coords,
    const Eigen::VectorXi &cube_config_indices,
    const Eigen::Matrix2Xi &cube_edge_indices,
    const Eigen::Matrix3Xi &edge_coords,
    std::vector<std::vector<erl::geometry::MarchingCubes::ValidCube>> &cubes) {

    ERL_ASSERTM(
        cube_edge_indices(1, cube_edge_indices.cols() - 1) <= edge_coords.cols(),
        "edge_coords has fewer columns than expected.");

    cubes.clear();
    if (cube_coords.cols() == 0) { return; }
    cubes.resize(1);
    std::vector<erl::geometry::MarchingCubes::ValidCube> &cubes0 = cubes[0];
    cubes0.resize(cube_coords.cols());

    for (long i = 0; i < cube_coords.cols(); ++i) {
        erl::geometry::MarchingCubes::ValidCube &cube = cubes0[i];
        cube.coords = cube_coords.col(i);
        cube.cfg_index = cube_config_indices[i];
        int start_edge_index = cube_edge_indices(0, i);
        int end_edge_index = cube_edge_indices(1, i);
        cube.edges.reserve(end_edge_index - start_edge_index);
        for (int j = start_edge_index; j < end_edge_index; ++j) {
            cube.edges.emplace_back(edge_coords.col(j));
        }
    }
}

/**
 * Convert the mesh represented as std::vector to Eigen matrices, which are then converted to numpy
 * arrays in Python. When the vectors are huge, this function can save the overhead of converting
 * std::vector to Python list, which requires the GIL and is slow.
 * @param vertices input, the vector of vertices.
 * @param triangles input, the vector of triangles.
 * @param triangle_normals input, the vector of triangle normals.
 * @return a tuple of (vertices, triangles, triangle_normals) in Eigen matrix format.
 */
std::tuple<Eigen::Matrix3Xd, Eigen::Matrix3Xi, Eigen::Matrix3Xd>
MeshToMatrices(
    const std::vector<Eigen::Vector3d> &vertices,
    const std::vector<Eigen::Vector3i> &triangles,
    const std::vector<Eigen::Vector3d> &triangle_normals) {
    Eigen::Matrix3Xd vertices_eigen = Eigen::Map<const Eigen::Matrix3Xd>(
        vertices.data()->data(),
        3,
        static_cast<long>(vertices.size()));
    Eigen::Matrix3Xi triangles_eigen = Eigen::Map<const Eigen::Matrix3Xi>(
        triangles.data()->data(),
        3,
        static_cast<long>(triangles.size()));
    Eigen::Matrix3Xd triangle_normals_eigen = Eigen::Map<const Eigen::Matrix3Xd>(
        triangle_normals.data()->data(),
        3,
        triangle_normals.size());
    return std::make_tuple(vertices_eigen, triangles_eigen, triangle_normals_eigen);
}

void
BindMarchingCubes(const py::module &m) {
    using MC = erl::geometry::MarchingCubes;
    py::class_<MC> mc(m, "MarchingCubes");

    py::class_<MC::ValidCube>(mc, "ValidCube")
        .def(py::init<>())
        .def_readwrite("coords", &MC::ValidCube::coords)
        .def_readwrite("cfg_index", &MC::ValidCube::cfg_index)
        .def_readwrite("edges", &MC::ValidCube::edges);

    mc.def(py::init<>())
        .def_static(
            "get_vertex_offsets",
            []() {
                py::array_t<int> arr({8, 3});
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        arr.mutable_at(i, j) = MC::kCubeVertexCodes[i][j];
                    }
                }
                return arr;
            })
        .def_static(
            "single_cube",
            [](const Eigen::Ref<const Eigen::Matrix<double, 3, 8>> &vertex_coords,
               const Eigen::Ref<const Eigen::Vector<double, 8>> &grid_values,
               const double iso_value) {
                std::vector<Eigen::Vector3d> vertices;
                std::vector<Eigen::Vector3i> triangles;
                std::vector<Eigen::Vector3d> face_normals;
                MC::SingleCube(
                    vertex_coords,
                    grid_values,
                    iso_value,
                    vertices,
                    triangles,
                    face_normals);
                return py::make_tuple(vertices, triangles, face_normals);
            },
            py::arg("vertex_coords"),
            py::arg("grid_values"),
            py::arg("iso_value"))
        .def_static(
            "calculate_cube_cfg_index",
            [](const Eigen::Vector<double, 8> &vertex_values, const double iso_value) {
                return MC::CalculateVertexConfigIndex(vertex_values.data(), iso_value);
            },
            py::arg("vertex_values"),
            py::arg("iso_value"))
        .def_static(
            "collect_valid_cubes",
            [](const Eigen::Ref<const Eigen::Vector3i> &grid_shape,
               const Eigen::Ref<const Eigen::VectorXd> &grid_values,
               const std::optional<Eigen::VectorXb> &mask,
               const double iso_value,
               const bool row_major,
               const bool parallel) {
                ERL_ASSERTM(
                    grid_shape[0] > 1 && grid_shape[1] > 1 && grid_shape[2] > 1,
                    "grid_shape should be at least (2,2,2).");
                ERL_ASSERTM(
                    grid_values.size() == static_cast<long>(grid_shape.prod()),
                    "grid_values has incorrect size.");
                if (mask.has_value()) {
                    ERL_ASSERTM(
                        mask->size() == static_cast<long>(grid_shape.prod()),
                        "mask has incorrect size.");
                }
                const auto valid_cubes = mask.has_value() ? MC::CollectValidCubesWithMask(
                                                                grid_shape,
                                                                grid_values,
                                                                mask.value(),
                                                                iso_value,
                                                                row_major,
                                                                parallel)
                                                          : MC::CollectValidCubes(
                                                                grid_shape,
                                                                grid_values,
                                                                iso_value,
                                                                row_major,
                                                                parallel);
                Eigen::Matrix3Xi cube_coords;
                Eigen::VectorXi cube_config_indices;
                Eigen::Matrix2Xi cube_edge_indices;
                Eigen::Matrix3Xi edge_coords;
                CubesToMatrices(
                    valid_cubes,
                    cube_coords,
                    cube_config_indices,
                    cube_edge_indices,
                    edge_coords);
                return std::make_tuple(
                    cube_coords,
                    cube_config_indices,
                    cube_edge_indices,
                    edge_coords);
            },
            py::arg("grid_shape"),
            py::arg("grid_values"),
            py::arg("mask") = py::none(),
            py::arg("iso_value") = 0.0,
            py::arg("row_major") = true,
            py::arg("parallel") = false,
            py::call_guard<py::gil_scoped_release>())
        .def_static(
            "process_valid_cubes",
            [](const Eigen::Matrix3Xi &cube_coords,
               const Eigen::VectorXi &cube_config_indices,
               const Eigen::Matrix2Xi &cube_edge_indices,
               const Eigen::Matrix3Xi &edge_coords,
               const Eigen::Ref<const Eigen::Vector3d> &coords_min,
               const Eigen::Ref<const Eigen::Vector3d> &grid_res,
               const Eigen::Ref<const Eigen::Vector3i> &grid_shape,
               const Eigen::Ref<const Eigen::VectorXd> &grid_values,
               const double iso_value,
               const bool row_major,
               const bool parallel) {
                ERL_ASSERTM(
                    cube_coords.cols() == cube_config_indices.size(),
                    "cube_coords and cube_config_indices have different number of cubes.");
                ERL_ASSERTM(
                    cube_coords.cols() == cube_edge_indices.cols(),
                    "cube_coords and cube_edge_indices have different number of cubes.");
                ERL_ASSERTM(
                    grid_shape[0] > 1 && grid_shape[1] > 1 && grid_shape[2] > 1,
                    "grid_shape should be at least (2,2,2).");
                ERL_ASSERTM(
                    grid_values.size() == static_cast<long>(grid_shape.prod()),
                    "grid_values has incorrect size.");

                std::vector<std::vector<erl::geometry::MarchingCubes::ValidCube>> valid_cubes;
                MatricesToCubes(
                    cube_coords,
                    cube_config_indices,
                    cube_edge_indices,
                    edge_coords,
                    valid_cubes);

                std::vector<Eigen::Vector3d> vertices;
                std::vector<Eigen::Vector3i> triangles;
                std::vector<Eigen::Vector3d> triangle_normals;
                MC::ProcessValidCubes(
                    valid_cubes,
                    coords_min,
                    grid_res,
                    grid_shape,
                    grid_values,
                    iso_value,
                    row_major,
                    parallel,
                    vertices,
                    triangles,
                    triangle_normals);

                return MeshToMatrices(vertices, triangles, triangle_normals);
            },
            py::arg("cube_coords"),
            py::arg("cube_config_indices"),
            py::arg("cube_edge_indices"),
            py::arg("edge_coords"),
            py::arg("coords_min"),
            py::arg("grid_res"),
            py::arg("grid_shape"),
            py::arg("grid_values"),
            py::arg("iso_value"),
            py::arg("row_major") = true,
            py::arg("parallel") = false,
            py::call_guard<py::gil_scoped_release>())
        .def_static(
            "run",
            [](const Eigen::Ref<const Eigen::Vector3d> &coords_min,
               const Eigen::Ref<const Eigen::Vector3d> &grid_res,
               const Eigen::Ref<const Eigen::Vector3i> &grid_shape,
               const Eigen::Ref<const Eigen::VectorXd> &grid_values,
               const std::optional<Eigen::VectorXb> &mask,
               const float iso_value,
               const bool row_major,
               const bool parallel) {
                ERL_ASSERTM(
                    grid_shape[0] > 1 && grid_shape[1] > 1 && grid_shape[2] > 1,
                    "grid_shape should be at least (2,2,2).");
                ERL_ASSERTM(
                    grid_values.size() == static_cast<long>(grid_shape.prod()),
                    "grid_values has incorrect size.");
                std::vector<Eigen::Vector3d> vertices;
                std::vector<Eigen::Vector3i> triangles;
                std::vector<Eigen::Vector3d> face_normals;
                if (mask.has_value()) {
                    ERL_ASSERTM(
                        mask->size() == static_cast<long>(grid_shape.prod()),
                        "mask has incorrect size.");
                    MC::RunWithMask(
                        coords_min,
                        grid_res,
                        grid_shape,
                        grid_values,
                        mask.value(),
                        iso_value,
                        row_major,
                        parallel,
                        vertices,
                        triangles,
                        face_normals);
                } else {
                    MC::Run(
                        coords_min,
                        grid_res,
                        grid_shape,
                        grid_values,
                        iso_value,
                        row_major,
                        parallel,
                        vertices,
                        triangles,
                        face_normals);
                }
                return MeshToMatrices(vertices, triangles, face_normals);
            },
            py::arg("coords_min"),
            py::arg("grid_res"),
            py::arg("grid_shape"),
            py::arg("grid_values"),
            py::arg("mask") = py::none(),
            py::arg("iso_value") = 0.0f,
            py::arg("row_major") = true,
            py::arg("parallel") = false,
            py::call_guard<py::gil_scoped_release>());
}
