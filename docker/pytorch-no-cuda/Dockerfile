FROM ubuntu:16.04

ARG PYTHON_VERSION=3.6.8
ARG PYTORCH_VERSION=1.0.0
ARG VISION_VERSION=0.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
         curl \
         ca-certificates \
         bzip2 \
         screen \
         git \
         vim && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y pytorch-cpu=$PYTORCH_VERSION torchvision-cpu=$VISION_VERSION -c pytorch && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN pip install tensorflow tensorboard tensorboardX

WORKDIR /workspace
RUN chmod -R a+w /workspace
