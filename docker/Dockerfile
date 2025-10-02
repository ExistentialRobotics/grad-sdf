FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG TZ=America/Los_Angeles
ARG LOCALE=en_US.UTF-8
ARG APT_MIRROR=""

ENV TZ=${TZ}
ENV LANG=${LOCALE}
ENV LANGUAGE=${LOCALE}
ENV LC_ALL=${LOCALE}

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup &&\
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|$APT_MIRROR|g" /etc/apt/sources.list; \
    fi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt update && apt install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    sudo \
    apt-utils \
    software-properties-common \
    lsb-release \
    ca-certificates \
    wget \
    curl \
    mesa-utils \
    libgl1 \
    libglx-mesa0 \
    x11-apps \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    g++ \
    ccache \
    git \
    libeigen3-dev \
    libyaml-cpp-dev \
    libabsl-dev \
    pybind11-dev

COPY . /workspace
WORKDIR /workspace
ENV CMAKE_CUDA_ARCHITECTURES="86;89;90;120"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
ENV TCNN_CUDA_ARCHITECTURES="86;89;90;120"
# for pytorch3d
ENV FORCE_CUDA=1
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install -r requirements.txt
RUN . .venv/bin/activate && \
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable --verbose
RUN . .venv/bin/activate && cd deps/tiny-cuda-nn && \
    rm -rf build || true && \
    cmake . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j`nproc` && \
    cd bindings/torch && \
    python3 setup.py install --verbose
RUN . .venv/bin/activate && \
    cd deps/sparse_octree && \
    rm -rf build || true && \
    python3 setup.py install --verbose
RUN . .venv/bin/activate && \
    cd deps/erl_geometry && \
    rm -rf build || true && \
    pip install --no-build-isolation --verbose .
RUN . .venv/bin/activate && \
    pip install -e . --verbose
RUN chown -R 1000:1000 /workspace
RUN userdel -r -f ubuntu
RUN DEBIAN_FRONTEND=noninteractive apt install zsh -y

# clean
RUN apt clean && rm -rf /var/lib/apt/lists/* && \
    if [ -n "$APT_MIRROR" ]; then \
        mv /etc/apt/sources.list.backup /etc/apt/sources.list; \
    fi
