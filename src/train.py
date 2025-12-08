import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import load_csv_no_header, build_vocab, TextDataset
from src.model import TextCNN
from src.utils import clean_text, tokenize


# =============================
# 固定随机种子
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# 保存/加载词表
# =============================
def save_vocab(stoi, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)


def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================
# 主训练流程
# =============================
def train_model(filter_size=3):

    print(f"\n====== 开始训练 TextCNN（filter_size={filter_size}）======")
    set_seed(42)

    # 加载数据
    dev_df = load_csv_no_header("data/dev.csv")
    test_df = load_csv_no_header("data/test.csv")

    vocab_path = "vocab.json"

    # -----------------------------
    # 构建或加载词表
    # -----------------------------
    if not os.path.exists(vocab_path):
        print("构建 vocab.json ...")

        all_tokens = [
            tokenize(clean_text(str(row["text"])))
            for _, row in dev_df.iterrows()
        ]

        stoi = build_vocab(all_tokens, vocab_size=50000)
        save_vocab(stoi, vocab_path)

        print("✔ vocab 构建完毕并已保存。")

    else:
        print("加载已有 vocab.json ...")
        stoi = load_vocab(vocab_path)
        stoi = {k: int(v) for k, v in stoi.items()}   # JSON 转 int

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_ds = TextDataset(dev_df, stoi, max_len=200)
    test_ds = TextDataset(test_df, stoi, max_len=200)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 初始化模型
    # -----------------------------
    model = TextCNN(
        vocab_size=len(stoi),
        embed_dim=128,
        filter_size=filter_size,
        num_filters=100
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("训练数据数量：", len(train_ds))
    print("测试数据数量：", len(test_ds))

    # 训练历史记录
    history = {"loss": [], "acc": [], "f1": []}

    best_f1 = 0
    save_path = f"models/textcnn_fs{filter_size}.pt"

    # =============================
    # 训练循环
    # =============================
    for epoch in range(1, 8):
        print(f"\n===== Epoch {epoch} =====")
        model.train()

        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"[{epoch}] step {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # -----------------------------
        # 测试评估
        # -----------------------------
        model.eval()
        all_pred, all_true = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)

                pred = torch.argmax(logits, dim=1).cpu().numpy()

                all_pred.extend(pred)
                all_true.extend(y.numpy())

        acc = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)

        print(f"Epoch {epoch} 平均损失：{avg_loss:.4f}")
        print(f"测试集：acc={acc:.4f}, f1={f1:.4f}")

        # 记录历史
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        history["f1"].append(f1)

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"✔ 保存最好模型到：{save_path}")

    print("\n训练结束")
    print(f"最好模型 f1={best_f1:.4f}")

    # =============================
    # 绘制 Training History
    # =============================
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(history["loss"]) + 1), history["loss"], marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(history["acc"]) + 1), history["acc"], marker='o', color='g')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(history["f1"]) + 1), history["f1"], marker='o', color='r')
    plt.title("Test F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"figures/training_history_fs{filter_size}.png", dpi=150)
    plt.close()

    print(f"✔ 保存 Training History 图到 figures/training_history_fs{filter_size}.png")

    # =============================
    # 混淆矩阵
    # =============================
    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(f"figures/confusion_matrix_fs{filter_size}.png", dpi=150)
    plt.close()

    print(f"✔ 保存 Confusion Matrix 图到 figures/confusion_matrix_fs{filter_size}.png")


# =============================
# 主入口
# =============================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    train_model(filter_size=3)
    train_model(filter_size=5)
