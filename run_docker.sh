#!/bin/bash

sudo docker run \
            -p 5678:5678 \
            -p 5000:5000 \
            -v ./:/content \
            --rm \
            --gpus all \
            --shm-size=32gb \
            -it \
            nazarenus/occlusion-gaussians:0.1