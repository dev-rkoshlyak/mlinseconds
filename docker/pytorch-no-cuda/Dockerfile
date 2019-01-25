FROM ubuntu:16.04

ARG PYTHON_VERSION=3.6.8
ARG PYTORCH_VERSION=0.4.0
ARG VISION_VERSION=0.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
         curl \
         ca-certificates \
         bzip2 \
         git \
         vim && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y pytorch-cpu=$PYTORCH_VERSION torchvision-cpu=$VISION_VERSION -c pytorch && \
     /opt/conda/bin/conda clean -ya && \
     pip install tensorboard tensorboardX

ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace
