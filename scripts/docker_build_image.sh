#!/bin/sh
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. >/dev/null 2>&1 && pwd )"
echo Project dir: $PROJECT_DIR
docker build -t pytorch -f $PROJECT_DIR/docker/pytorch-no-cuda/Dockerfile .
