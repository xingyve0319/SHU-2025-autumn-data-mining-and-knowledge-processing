# src/utils/evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(model, data_loader, device):
    """
    计算所有评估指标: Accuracy, F1, AUC, Confusion Matrix
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []  # 存正类的概率，用于计算 AUC
    
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算概率 (Softmax)
            probs = F.softmax(outputs, dim=1)
            
            # 获取预测类别
            _, preds = torch.max(outputs, dim=1)
            
            # 收集数据
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # 假设 index 1 是正面情感 (positive)
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    # --- 计算指标  ---
    avg_loss = total_loss / len(data_loader)
    
    acc = accuracy_score(all_targets, all_preds)
    
    # F1-score: 如果是二分类，average通常选 'binary' 或 'weighted'
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # AUC-ROC: 如果类别只有1个会导致报错，加个try-catch
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
        
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    return avg_loss, acc, f1, auc, cm