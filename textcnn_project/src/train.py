import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import load_csv, build_vocab, TextDataset
from src.model import TextCNN
from src.utils import evaluate, plot_history, plot_confusion_matrix

# ----------------------------
# 配置路径
# ----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# 超参数
# ----------------------------
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MAX_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 加载数据
# ----------------------------
train_texts, train_labels = load_csv(os.path.join(DATA_DIR, "dev.csv"))
test_texts, test_labels = load_csv(os.path.join(DATA_DIR, "test.csv"))

vocab = build_vocab(train_texts)

train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
test_dataset = TextDataset(test_texts, test_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 初始化模型
# ----------------------------
model = TextCNN(
    vocab_size=len(vocab),
    embed_dim=200,           # embedding维度
    num_classes=2,
    kernel_sizes=[2,3,4,5], # 卷积 kernel
    num_channels=200,        # 卷积通道数
    dropout=0.5
).to(DEVICE)

# ----------------------------
# 计算 class weight
# ----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1]),   # 注意这里必须是 numpy.ndarray
    y=train_labels
)
weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# 训练
# ----------------------------
history = {'loss': [], 'acc': [], 'f1': []}

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc, f1, cm, _, _ = evaluate(model, test_loader, DEVICE)
    history['loss'].append(avg_loss)
    history['acc'].append(acc)
    history['f1'].append(f1)

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Test Acc={acc:.4f}, F1={f1:.4f}")

# ----------------------------
# 保存模型
# ----------------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "textcnn.pth"))

# ----------------------------
# 保存训练历史图
# ----------------------------
plot_history(history, os.path.join(FIGURES_DIR, "training_history.png"))

# ----------------------------
# 保存 confusion matrix
# ----------------------------
_, _, cm, _, _ = evaluate(model, test_loader, DEVICE)
plot_confusion_matrix(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))

print("训练完成，模型和图像已保存！")
