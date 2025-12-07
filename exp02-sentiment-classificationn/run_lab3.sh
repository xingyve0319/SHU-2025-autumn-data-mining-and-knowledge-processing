#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 设置Python路径
export PYTHONPATH="$(pwd):$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1

# 运行Python脚本
python src/lab3_qwen-sentential-classifier/main.py