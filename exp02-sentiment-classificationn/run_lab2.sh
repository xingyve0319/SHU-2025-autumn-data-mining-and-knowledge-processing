#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 设置Python路径
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 运行Python脚本
python src/lab2_bert-sentential-classifer/main.py
