# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def plot_training_history(train_losses, val_losses, val_accuracies, val_f1s, save_path):
    """
    画训练历史图
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
