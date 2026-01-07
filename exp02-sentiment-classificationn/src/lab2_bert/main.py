import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer,get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

from src.config.config import cfg

from src.utils import (
    SentimentDataset, DataLoader as DataLoaderClass, SentimentClassifier,
    set_hf_mirrors, plot_training_history, plot_confusion_matrix,
    set_seed, parse_args, setup_logging, calculate_metrics
)

logger = logging.getLogger(__name__)

def train():
    config = cfg.lab2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 初始化
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    model = SentimentClassifier(config['model_name'], cfg.num_classes)
    model.to(device)
    
    # 3. 数据加载
    data_loader = DataLoaderClass(cfg)
    train_texts, train_labels = data_loader.load_csv(cfg.train_path)
    val_texts, val_labels = data_loader.load_csv(cfg.dev_path)
    
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, cfg.max_seq_length)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, cfg.max_seq_length)
    
    # 使用字典访问参数 config['batch_size']
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    
    # 4. 优化器
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * config['num_epochs'])
    
    # 5. 训练循环
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = 0.0
    best_cm = None

    logger.info("Starting Lab 2 (BERT) Training...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        val_loss, val_acc, val_f1, val_auc, val_cm = calculate_metrics(model, val_loader, device)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        logger.info(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_cm = val_cm
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            model.save_model(config['model_save_path'])
            logger.info(f" Best Model Saved (F1: {best_f1:.4f})")
    
    # 训练结束，生成可视化图表
    logger.info("Generating training plots...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_dir = os.path.join(config['result_dir'], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    plot_path = os.path.join(save_dir, "training_history.png")
    plot_training_history(history['train_loss'], history['val_loss'], history['val_acc'], history['val_f1'], plot_path)
    
    if best_cm is not None:
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix(best_cm, classes=['Negative', 'Positive'], save_path=cm_path)

    logger.info(f"All result plots saved to directory: {save_dir}")
    
    return model

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    setup_logging(cfg.lab2['result_dir'], "bert")
    set_hf_mirrors()
    
    if args.batch_size: cfg.lab2['batch_size'] = args.batch_size
    if args.lr: cfg.lab2['learning_rate'] = args.lr
    if args.epochs: cfg.lab2['num_epochs'] = args.epochs
    
    train()