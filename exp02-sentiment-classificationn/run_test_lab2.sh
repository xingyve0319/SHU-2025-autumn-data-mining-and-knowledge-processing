#!/bin/bash

# 添加当前目录到 Python 路径，防止找不到 src
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Running Inference for Lab 2 (BERT)..."
python src/predict.py --experiment lab2