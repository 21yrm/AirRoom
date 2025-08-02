#!/bin/bash

docker build --build-arg USER_ID=$UID -t airroom:latest -f Dockerfile .
