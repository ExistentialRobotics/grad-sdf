$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
=============

<p align="center">
<a href="https://github.com/ExistentialRobotics/grad-sdf/releases"><img src="https://img.shields.io/github/v/release/ExistentialRobotics/grad-sdf?label=version" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#run-nabla-sdf"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
<a href="https://github.com/ExistentialRobotics/grad-sdf?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
<!-- <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a> -->
<a href="https://github.com/ExistentialRobotics/grad-sdf/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</p>

This repository contains the code for the paper: **$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual**.

$\nabla$-SDF is a hybrid SDF reconstruction framework that combines gradient-augmented octree interpolation with an implicit neural residual to achieve efficient, continuous non-truncated, and highly accurate Euclidean SDF mapping..

<div align="center">
  <img src="assets/grad-sdf.gif" width="600" alt="SDF Mapping Demo">
</div>

## Installation

### Prerequisites

- Ubuntu (24.04 tested) / Arch Linux
- Python 3.12 (3.10, 3.11 should also work)
- CUDA (tested with CUDA 12.8 and PyTorch 2.8.0)

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
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable

    cd deps/tiny-cuda-nn
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

4. Install $\nabla$-SDF
    ```bash
    pip install --no-build-isolation -e .
    ```

## Prepare Dataset

Download Replica Dataset (only mesh, camera parameter and trajectory) at [One-Drive Link](https://ucsdcloud-my.sharepoint.com/my?id=%2Fpersonal%2Fzhdai%5Fucsd%5Fedu%2FDocuments%2FPublicShare%2Fgrad%2Dsdf%2Freplica%2Etar%2Egz&parent=%2Fpersonal%2Fzhdai%5Fucsd%5Fedu%2FDocuments%2FPublicShare%2Fgrad%2Dsdf&ga=1) and put it under path "data/Replica"

Run the following commands to preprocess the Replica dataset:

The script [`grad_sdf/dataset/replica_obb_rotation.py`](grad_sdf/dataset/replica_obb_rotation.py) is used to rotate mesh and trajectory to better match octree.
```bash
python grad_sdf/dataset/replica_obb_rotation.py \
    --dataset-dir data/Replica \
    --output-dir data/Replica_preprocessed
```
copy camera parameter to preprocessed data folder
```bash
cp data/Replica/cam_params.json data/Replica_preprocessed
```
The script [`grad_sdf/dataset/replica_augment_views.py`](grad_sdf/dataset/replica_augment_views.py) is used to augment the Replica dataset with additional virtual camera views (e.g., upward-looking frames) to improve spatial coverage for training.
```bash
python grad_sdf/dataset/replica_augment_views.py \
    --original-dir data/Replica_preprocessed \
    --output-dir data/Replica_preprocessed \
    # --scenes room0  # (optional) Process specific scenes only. If not set, process all scenes. \
    # --interval 50  # (optional, default=50) Insert upward-looking frames every n frames. \
    # --n-rolls-per-insertion 10 # (optional, default=10) Number of roll rotations per insertion. \
    # --keep-existing  # (optional) Keep existing RGBD data.
```
## Run $\nabla$-SDF

### Example: Training on Replica Scene *room0*

Run the following command to start training on the Replica dataset scene **room0**:

```bash
python grad_sdf/trainer.py  --config configs/v2/replica_room0.yaml
```

### Run GUI Trainer

The GUI trainer allows interactive visualization and monitoring of the training process, including SDF slice, octree structure, and camera poses.

```bash
python grad_sdf/gui_trainer.py \
    --gui-config configs/v2/gui.yaml \
    --trainer-config configs/v2/replica_room0.yaml \
    --gt-mesh-path data/Replica_preprocessed/room0_mesh.ply \
    --apply-offset-to-gt-mesh \
    --copy-scene-bound-to-gui
```

## Docker

### 1. Build the image
First, build the Docker image (make sure you are in the project root):


Use the following command to start a container with GPU, X11 display, and device access enabled:
```bash
./docker/build.bash
```
This script will create the Docker image `erl/grad_sdf:24.04`.

### 2. Run the container
Use the following command to start a container with GPU, X11 display, and device access enabled:
```bash
docker run --privileged --restart always -t \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME:$HOME:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --workdir /workspace \
    --gpus all \
    --runtime=nvidia \
    -e DISPLAY \
    --net=host \
    --detach \
    --hostname container-grad_sdf \
    --add-host=container-grad_sdf:127.0.0.1 \
    --name grad_sdf \
    erl/grad_sdf:24.04 \
    bash -l
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{dai2025nablasdf,
      title={{$\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual}},
      author={Zhirui Dai and Qihao Qian and Tianxing Fan and Nikolay Atanasov},
      year={2025},
      eprint={2510.18999},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.18999},
}
```

## Acknowledgement

- We develop our key frame selection strategy based on [H2-Mapping](https://github.com/Robotics-STAR-Lab/H2-Mapping).
- We create the GUI based on [Open3D](http://www.open3d.org/) with inspirations from [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).
