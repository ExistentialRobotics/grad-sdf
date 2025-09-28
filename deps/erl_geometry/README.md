erl_geometry
============

A C++/CUDA library of various algorithms, datasets and utilities for studying geometry.
This folder only contains the functionalities required by grad-SDF. For the full library, please
visit [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry).

This lite version contains:
- Semi-sparse Octree [CPU]
- Semi-sparse Quadtree [CPU]
- Marching Cubes [CPU]
- Marching Squares [CPU]
- Morton Codes [CPU/GPU]
- Voxel search in Octree/Quadtree [CPU/GPU]
- Python bindings for all of the above

## Dependencies
- C++17
- CUDA 12 (Tested)
- CMake 3.26+
- Eigen 3.4+
- Abseil-cpp
- YAML-CPP
- pybind11
- PyTorch 2.7+ (or older versions with C++11 ABI enabled)

## Installation

```shell
sudo apt install \
    cmake \
    g++ \
    ccache \
    git \
    libeigen3-dev \
    libyaml-cpp-dev \
    libabsl-dev \
    python3-dev \
    python3-pip \
    pybind11-dev \
    pipenv

# cd to grad-sdf root folder
# activate grad-sdf's pipenv environment
pipenv shell
# cd to this folder

# install as a python package
pip install --no-build-isolation --verbose .

# OR build without python bindings
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
```

## Usage

grad_sdf contains code that uses the semi-sparse octree, marching cubes, etc. in various places.
For example:
- [octree, morton code and find voxel indices](../../grad_sdf/semi_sparse_octree_v2.py)
- [marching cubes](../../grad_sdf/evaluator_base.py)
