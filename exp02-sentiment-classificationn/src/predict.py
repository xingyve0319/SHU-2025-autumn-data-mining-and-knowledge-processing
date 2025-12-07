import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config.config import cfg
from src.utils import SentimentClassifier, SentimentDataset,DataLoader as DataLoaderClass,setup_logging

# 配置日志
logger = logging.getLogger(__name__)

def load_model_weights(model, model_path):
    """
    安全加载模型权重：
    1. 自动去除 DataParallel 的 'module.' 前缀
    2. 自动修正 Lab2 (self.bert) 到 Lab3 (self.encoder) 的变量名差异
    """

    
    state_dict = torch.load(model_path, map_location='cpu')
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1. 处理多卡训练的前缀 'module.'
        if k.startswith('module.'):
            k = k[7:]
            
        # 2. 处理变量名不一致问题
        # 旧权重是 'bert.embeddings...'，新代码期望 'encoder.embeddings...'
        if k.startswith('bert.'):
            k = k.replace('bert.', 'encoder.', 1)
            
        new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    return model

def predict(experiment_name):
    # 1. 根据实验名称 (lab2/lab3) 获取配置
    if experiment_name == 'lab2':
        config = cfg.lab2
    elif experiment_name == 'lab3':
        config = cfg.lab3
    else:
        raise ValueError("Experiment must be 'lab2' or 'lab3'")
    setup_logging(config['result_dir'], "predict")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running inference for {experiment_name} on {device}")
    
    # 2. 加载 Tokenizer
    logger.info(f"Loading tokenizer: {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 3. 加载模型架构
    logger.info("Initializing model...")
    model = SentimentClassifier(config['model_name'], cfg.num_classes)
    # 确保 pad_token_id 对齐
    model.encoder.config.pad_token_id = tokenizer.pad_token_id
    
    # 4. 加载训练好的权重
    model_path = config['model_save_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train first!")
        
    logger.info(f"Loading weights from {model_path}...")
    model = load_model_weights(model, model_path)
    model.to(device)
    model.eval()
    
   # 5. 加载测试数据 (使用你的 DataLoader)
    test_path = cfg.test_path
    logger.info(f"Loading test data from {test_path}...")
    
    # 初始化数据加载器
    raw_loader = DataLoaderClass(config)
    
    # 尝试从配置中获取测试集行数限制，如果没有则为 None (读取全部)
    # 你可以在 yaml 的 lab3 下面加一个 test_nrows: 100 来测试
    test_nrows = config.get('test_nrows', None)
    if test_nrows:
        logger.info(f"Note: Only reading first {test_nrows} rows based on config.")

    # 使用 load_csv 读取处理好的文本和标签
    # texts 是 "Title. Content" 格式, labels 是 0/1
    texts, labels = raw_loader.load_csv(test_path, nrows=test_nrows)

    # 构造 Dataset
    # load_csv 返回的 labels 已经是 0/1 了，可以直接用于计算准确率
    test_dataset = SentimentDataset(texts, labels, tokenizer, cfg.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # 6. 推理循环
    all_preds = []
    correct_predictions = 0
    total_predictions = 0

    logger.info("Starting inference...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # 现在 test 也有 label 了

            # 针对 4060/2080 使用混合精度加速
            if experiment_name == 'lab3':
                # Qwen 推荐使用 bfloat16 (4060) 或 float16 (2080)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    outputs = model(input_ids, attention_mask)
            else:
                # BERT 普通推理
                outputs = model(input_ids, attention_mask)
            
            # 获取预测类别 (argmax)
            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            all_preds.extend(preds.cpu().numpy())

    # 打印最终 Test 准确率
    test_acc = correct_predictions.double() / total_predictions
    logger.info(f"Test Set Accuracy: {test_acc:.4f}")
            
    # 7. 保存结果
    final_preds = np.array(all_preds) + 1
    
    df_out = pd.DataFrame({
        'predicted_label': final_preds
    })
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
    filename = f'test_predictions_{timestamp}.csv'   
    output_file = os.path.join(config['result_dir'], filename)

    df_out.to_csv(output_file, index=False)
    logger.info(f"Done! Predictions saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, choices=['lab2', 'lab3'], help="lab2 or lab3")
    args = parser.parse_args()
    
    predict(args.experiment)