#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Running Inference for Lab 3 (Qwen)..."
python src/predict.py --experiment lab3