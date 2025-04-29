# Base image

FROM debian:12.5

# Install system dependencies in one step to reduce layers
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        tabix \
        libreadline-dev \
        libcairo2-dev \
        git \
        procps \
        g++ \
        python3 \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Miniconda
ENV CONDA_DIR=/opt/conda
ENV MINICONDA_HOME=~/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    mkdir -p $MINICONDA_HOME && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O $MINICONDA_HOME/miniconda.sh && \
    bash $MINICONDA_HOME/miniconda.sh -b -u -p $CONDA_DIR && \
    rm $MINICONDA_HOME/miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Put conda in PATH so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Python dependencies via pip
RUN pip3 install "tensorflow<2" 
RUN pip3 install "csbdeep[tf1]" 
RUN pip3 install "stardist[tf1]"
