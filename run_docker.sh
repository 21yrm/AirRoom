#!/bin/bash

docker run --rm --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/home/appuser/AirRoom \
    -e HF_HOME=/home/appuser/AirRoom/cache/huggingface \
    -e TRANSFORMERS_CACHE=/home/appuser/AirRoom/cache/huggingface \
    -e TORCH_HOME=/home/appuser/AirRoom/cache/torchhub \
    --shm-size=8gb \
    --env="DISPLAY" \
    --workdir /home/appuser/AirRoom \
    -it airroom:latest \
    bash -c "cd third_party/SemanticSAM/semantic_sam/body/encoder/ops && sh make.sh && cd /home/appuser/AirRoom && bash"
