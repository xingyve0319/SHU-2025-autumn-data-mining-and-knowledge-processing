# src/utils/setup.py
import argparse
import logging
import os
import random
import numpy as np
import torch

def set_hf_mirrors():
    """
    设置Hugging Face镜像
    """
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'

def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Sentiment Classification Training")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Num epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def setup_logging(save_dir, model_name="model"):
    """
    配置全局日志
    参数:
        save_dir: 日志保存目录
        model_name: 模型名称 (用于生成日志文件名，如 bert.log)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 处理模型名称中的非法字符 (例如 'Qwen/Qwen2.5' -> 'Qwen_Qwen2.5')
    # 2. 拼接日志文件名
    safe_name = model_name.replace('/', '_')
    log_filename = f"{safe_name}.log"
    
    # 清空之前的 handlers (避免重复打印)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置格式和文件输出
    logging.basicConfig(
        filename=os.path.join(save_dir, log_filename), # 使用动态文件名
        filemode='a', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # 添加控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging.getLogger(__name__)