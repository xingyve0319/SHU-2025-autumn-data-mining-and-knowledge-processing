import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

from src.dataset import load_csv, build_vocab, TextDataset
from src.model import TextCNN
from src.utils import evaluate, plot_confusion_matrix,set_seed
set_seed()
from src.visualization import plot_training_history

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
train_texts, train_labels = load_csv(os.path.join(DATA_DIR, "train.csv"))
dev_texts, dev_labels = load_csv(os.path.join(DATA_DIR, "dev.csv"))
test_texts, _ = load_csv(os.path.join(DATA_DIR, "test.csv"))  # 不需要标签

vocab = build_vocab(train_texts)

train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
dev_dataset = TextDataset(dev_texts, dev_labels, vocab, MAX_LEN)
test_dataset = TextDataset(test_texts, [0] * len(test_texts), vocab, MAX_LEN)  # 为了预测，标签设置为0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
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
    classes=np.array([0, 1]),  # 注意这里必须是 numpy.ndarray
    y=train_labels
)
weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# 训练
# ----------------------------
history = {'loss': [], 'acc': [], 'f1': [],'dev_loss':[]}

for epoch in range(1, EPOCHS + 1):
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
    dev_loss,acc, f1, cm, _, _ = evaluate(model, dev_loader, DEVICE)  # 在dev数据上进行评估
    history['loss'].append(avg_loss)
    history['dev_loss'].append(dev_loss)
    history['acc'].append(acc)
    history['f1'].append(f1)

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Dev Acc={acc:.4f}, F1={f1:.4f}")

# ----------------------------
# 保存模型
# ----------------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "textcnn.pth"))

# ----------------------------
# 保存训练历史图
# ----------------------------
plot_training_history(
    history['loss'],      # 对应 train_losses
    history['dev_loss'],  # 对应 dev_losses
    history['acc'],       # 对应 dev_acc
    history['f1'],        # 对应 dev_f1
    os.path.join(FIGURES_DIR, "training_history.png")
)

# ----------------------------
# 保存 confusion matrix
# ----------------------------
_,_, _, cm, _, _ = evaluate(model, dev_loader, DEVICE)  # 使用dev数据生成混淆矩阵
plot_confusion_matrix(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))

# ----------------------------
# 预测并保存结果
# ----------------------------
model.eval()  # 切换到评估模式
predictions = []
true_labels = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy()+1)
        true_labels.extend(y.numpy()+1)
acc = accuracy_score(true_labels, predictions)
print(f"测试集准确率 (Test Accuracy): {acc:.4f}")
# 保存预测结果
prediction_df = pd.DataFrame(predictions, columns=["prediction"])
prediction_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

print("训练完成，模型和图像已保存，预测结果已保存！")
