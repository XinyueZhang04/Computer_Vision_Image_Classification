# train.py
# ============================================
# Machine Vision Final Project
# Dataset: Teacher-provided dataset
# Train set: train-test-car + train-test-bottle + train-test-fruits (merged)
# Task: 15-class image classification
# ============================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# =====================
# 参数配置
# =====================
TRAIN_DIR = "final_dataset/train"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 自定义数据集（OpenCV）
# =====================
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.encoder = LabelEncoder()

        classes = sorted(os.listdir(root_dir))
        self.encoder.fit(classes)

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(cls)

        self.labels = self.encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# =====================
# 加载数据
# =====================
dataset = CustomDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(dataset.encoder.classes_)
print("Classes:", dataset.encoder.classes_)

# =====================
# 模型（迁移学习）
# =====================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =====================
# 训练
# =====================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss:.4f}  Acc: {acc:.2f}%")

# =====================
# 保存模型
# =====================
torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.encoder.classes_
}, "model.pth")

print("✅ Training finished, model saved as model.pth")
 