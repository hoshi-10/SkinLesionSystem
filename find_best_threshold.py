
from pathlib import Path

import torch
import numpy as np

from src.dataset import SkinLesionDataset
from src.model import UNet


# 1. 基础配置
device = torch.device("cpu")
model_path = Path("checkpoints") / "best_unet.pth"

if not model_path.exists():
    raise FileNotFoundError(f"找不到模型：{model_path}")

# 2. 加载验证集
dataset = SkinLesionDataset(split="val")

# 3. 加载模型
model = UNet().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.eval()

# 4. 不同候选阈值
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = {threshold: [] for threshold in thresholds}

# 5. 遍历验证集
with torch.no_grad():
    for image, mask in dataset:
        output = model(image.unsqueeze(0).to(device))
        probability = torch.sigmoid(output).squeeze().cpu()

        mask = mask.squeeze()

        for threshold in thresholds:
            prediction = (probability >= threshold).float()

            intersection = (prediction * mask).sum()
            total = prediction.sum() + mask.sum()

            dice = (2 * intersection + 1) / (total + 1)
            results[threshold].append(dice.item())

# 6. 输出不同阈值的平均 Dice
print("\n========== 验证集阈值对比 ==========")

best_threshold = None
best_dice = -1

for threshold in thresholds:
    mean_dice = np.mean(results[threshold])

    print(f"阈值：{threshold:.1f} | 平均 Dice：{mean_dice:.4f}")

    if mean_dice > best_dice:
        best_dice = mean_dice
        best_threshold = threshold

print("\n========== 最佳阈值 ==========")
print(f"最佳阈值：{best_threshold:.1f}")
print(f"验证集平均 Dice：{best_dice:.4f}")
