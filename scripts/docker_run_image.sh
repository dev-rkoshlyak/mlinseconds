#!/bin/sh
docker run -it -p 6006:6006 -v ~/ml/mlinseconds:/mlinseconds -w /mlinseconds --rm pytorch
