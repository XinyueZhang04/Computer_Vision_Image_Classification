import os
import cv2
import torch
import numpy as np
from torchvision import models
import torch.nn as nn

# -------------------------
# 配置
# -------------------------
TEST_DIR = "D:/test111/final_dataset/test"  # 绝对路径更安全
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 小类 -> 大类映射
# -------------------------
SUBCLASS_TO_SUPERCLASS = {
    # Fruits
    "A": "train-test-fruits",
    "B": "train-test-fruits",
    "G": "train-test-fruits",
    "L": "train-test-fruits",
    "M": "train-test-fruits",
    "O": "train-test-fruits",

    # Car
    "auto rickshaw": "train-test-car",
    "bike motor": "train-test-car",
    "bycycle": "train-test-car",
    "car": "train-test-car",
    "cng": "train-test-car",
    "taxi": "train-test-car",
    "truck": "train-test-car",

    # Bottle
    "Maaza": "train-test-bottle",
    "Slice": "train-test-bottle"
}

# 小类 -> 可读名称
FRUITS_NAME_MAP = {
    "A": "Apple",
    "B": "Broccoli",
    "G": "Grape",
    "L": "Lemon",
    "M": "Mango",
    "O": "Orange"
}

# -------------------------
# 加载模型
# -------------------------
checkpoint = torch.load("D:/test111/model.pth", map_location=DEVICE)
class_names = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# -------------------------
# 推理 + 统计
# -------------------------
total = 0
correct_subclass = 0
correct_superclass = 0

print("开始推理测试集...\n")

for true_subclass in os.listdir(TEST_DIR):
    sub_path = os.path.join(TEST_DIR, true_subclass)
    if not os.path.isdir(sub_path):
        continue

    true_superclass = SUBCLASS_TO_SUPERCLASS.get(true_subclass, "Unknown")

    for img_name in os.listdir(sub_path):
        img_path = os.path.join(sub_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_subclass = class_names[pred_idx]

        pred_superclass = SUBCLASS_TO_SUPERCLASS.get(pred_subclass, "Unknown")
        readable_name = FRUITS_NAME_MAP.get(pred_subclass, pred_subclass)

        total += 1
        if pred_subclass == true_subclass:
            correct_subclass += 1
        if pred_superclass == true_superclass:
            correct_superclass += 1

        print(f"{img_name:30s} | GT Super: {true_superclass:15s} | GT Sub: {true_subclass:15s} "
              f"| Pred Super: {pred_superclass:15s} | Pred Sub: {pred_subclass:10s} | Name: {readable_name}")

# -------------------------
# 输出准确率
# -------------------------
if total > 0:
    accuracy_subclass = correct_subclass / total * 100
    accuracy_superclass = correct_superclass / total * 100
else:
    accuracy_subclass = accuracy_superclass = 0

print("\n----------------------------------")
print(f"Total images: {total}")
print(f"Sub-class correct: {correct_subclass}")
print(f"Sub-class accuracy: {accuracy_subclass:.2f}%")
print(f"Super-class correct: {correct_superclass}")
print(f"Super-class accuracy: {accuracy_superclass:.2f}%")
