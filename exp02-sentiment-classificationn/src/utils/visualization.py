# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, classes, save_path):
    """
    绘制并保存混淆矩阵 
    """
    plt.switch_backend('Agg')
    plt.figure(figsize=(8, 6))
    
    # 使用 Seaborn 画热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion Matrix saved to: {os.path.abspath(save_path)}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")
    finally:
        plt.close()

def plot_training_history(train_losses, val_losses, val_accuracies, val_f1s, save_path):
    """
    更新：增加 F1 Score 的曲线绘制
    """
    if not train_losses:
        return

    plt.switch_backend('Agg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 改为 1行3列 的图
    plt.figure(figsize=(18, 5))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Accuracy [cite: 50]
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 3. F1 Score [cite: 51]
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_f1s, 'm-', label='Val F1')
    plt.title('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()