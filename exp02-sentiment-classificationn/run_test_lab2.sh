#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Running Inference for Lab 2 (BERT)..."
python src/predict.py --experiment lab2