$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
=============

<p align="center">
<a href="https://github.com/ExistentialRobotics/grad-sdf/releases"><img src="https://img.shields.io/github/v/release/ExistentialRobotics/grad-sdf?label=version" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#run-nabla-sdf"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
<!-- <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a> -->
<a href="https://github.com/ExistentialRobotics/grad-sdf/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</p>

This repository contains the code for the paper: $\nabla$-SDF.

$\nabla$-SDF is a hybrid SDF reconstruction framework that combines gradient-augmented octree interpolation with an implicit neural residual to achieve efficient, continuous non-truncated, and highly accurate Euclidean SDF mapping..

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
    If you use other virtual environment tools, you can also install the dependencies by
    ```bash
    pip install -r requirements.txt
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
    pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

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

Download NICE-SLAM Replica Dataset
```bash
bash scripts/download_replica.sh
```
Run the following commands to preprocess the Replica dataset:

```bash
python grad_sdf/dataset/replica_obb_rotation.py \
    --dataset-dir data/Replica \
    --output-dir data/Replica_preprocessed

python grad_sdf/dataset/replica_augment_views.py \
    --original-dir data/Replica_preprocessed \
    --output-dir data/Replica_preprocessed \
    # --scenes room0   # (optional) process a specific scene
```
## Run $\nabla$-SDF

Example training on scene room0
```bash
python grad_sdf/trainer.py --config configs/v2/replica_room0.yaml
```
Example training on scene room0 with GUI
```bash
python grad_sdf/gui_trainer.py --gui-config configs/v2/gui.yaml --trainer-config configs/v2/replica_room0.yaml --gt-mesh-path data/Replica_preprocessed/room0_mesh.ply --apply-offset-to-gt-mesh --copy-scene-bound-to-gui
```
## Docker

## Citation

## Acknowledgement

- We develop our key frame selection strategy based on [H2-Mapping]()
- We create the GUI based on [Open3D](http://www.open3d.org/) with inspirations from [PIN-SLAM]()
