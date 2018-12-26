#!/bin/bash
docker build --memory-swap 0 -t wave2letter:cblas .
docker tag wave2letter:cblas alexht/wav2letter:cblas
docker push alexht/wav2letter:cblas
