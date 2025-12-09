import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os
import random

def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device):
    model.eval()
    total_loss=0
    preds, labels = [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() # 累加 Loss
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    return avg_loss, acc, f1, cm, preds, labels

def plot_confusion_matrix(cm, save_path, class_names=["Negative","Positive"]):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()
