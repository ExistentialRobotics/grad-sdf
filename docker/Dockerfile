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

WORKDIR /workspace
RUN git clone --recursive https://github.com/ExistentialRobotics/grad-sdf.git && \
    cd grad-sdf && \
    python -m venv .venv && \
    . .venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install -e .

RUN userdel -r -f ubuntu

# clean
RUN apt clean && rm -rf /var/lib/apt/lists/* && \
    if [ -n "$APT_MIRROR" ]; then \
        mv /etc/apt/sources.list.backup /etc/apt/sources.list; \
    fi
