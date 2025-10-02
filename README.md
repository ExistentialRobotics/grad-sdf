$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
=============

<p align="center">
<a href="https://github.com/ExistentialRobotics/grad-sdf/releases"><img src="https://img.shields.io/github/v/release/ExistentialRobotics/grad-sdf?label=version" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#run-nabla-sdf"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
<!-- <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a> -->
<a href="https://github.com/ExistentialRobotics/grad-sdf/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</p>

This repository contains the code for the paper: [place_holder].

TODO: one line description of the paper.

TODO: figures, GIFs, videos, etc.

## Installation

### Prerequisites

- Ubuntu (24.04 tested) / Arch Linux
- Python 3.12 (3.10, 3.11 should also work)
- CUDA 12 (tested with 12.9)

### Steps

0. Clone the repository

    ```bash
    git clone --recursive https://github.com/ExistentialRobotics/grad-sdf.git
    cd grad-sdf
    ```

1. Setup pipenv environment

    ```bash
    pip install pipenv  # or sudo apt install pipenv
    pipenv install
    pipenv shell --verbose
    ```

2. Install system dependencies
    - For Ubuntu
    ```bash
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
        pybind11-dev
    ```
    - For Arch Linux
    ```bash
    sudo pacman -S --needed \
        cmake \
        gcc \
        ccache \
        git \
        eigen \
        yaml-cpp \
        abseil-cpp \
        python \
        python-pip \
        pybind11
    ```

3. Install other dependencies

    ```bash
    cd deps/tinycudann
    cmake . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j`nproc`
    cd bindings/torch
    python setup.py install
    cd ../../../..

    cd deps/sparse_octree
    python setup.py install
    cd ../..

    cd deps/erl_geometry
    pip install --no-build-isolation --verbose .
    cd ../..
    # for Arch Linux
    # CXX=/usr/bin/g++-14 pip install --no-build-isolation --verbose .
    ```

## Prepare Dataset

## Run $\nabla$-SDF

## Docker

## Citation

## Acknowledgement

- We develop our key frame selection strategy based on [H2-Mapping]()
- We create the GUI based on [Open3D](http://www.open3d.org/) with inspirations from [PIN-SLAM]()
