#!/bin/bash
nvidia-docker build --memory-swap 0 -t wave2letter:test .
nvidia-docker push wave2letter:test
