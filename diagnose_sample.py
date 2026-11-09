
from pathlib import Path

import torch

from src.dataset import SkinLesionDataset
from src.model import UNet


# 1. 加载模型
device = torch.device("cpu")
model_path = Path("checkpoints") / "best_unet.pth"

if not model_path.exists():
    raise FileNotFoundError(f"模型不存在：{model_path}")

model = UNet().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.eval()


# 2. 加载第 6 张测试图片
dataset = SkinLesionDataset(split="test")

if len(dataset) < 6:
    raise ValueError("测试集不足 6 张图片")

image, mask = dataset[5]

# 3. 获取模型预测概率
with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    probability = torch.sigmoid(output).squeeze().cpu()

mask = mask.squeeze()

# 4. 输出真实掩码和预测概率
print("\n========== 第 6 张图片诊断 ==========")

print(f"真实病变区域占比：{mask.mean().item():.4%}")
print(f"预测概率最小值：{probability.min().item():.4f}")
print(f"预测概率最大值：{probability.max().item():.4f}")
print(f"预测概率平均值：{probability.mean().item():.4f}")

# 5. 检查不同阈值下的预测情况
print("\n========== 不同阈值对比 ==========")

for threshold in [0.3, 0.5, 0.7, 0.9]:

    prediction = (probability >= threshold).float()

    intersection = (prediction * mask).sum()
    total = prediction.sum() + mask.sum()

    dice = (2 * intersection + 1) / (total + 1)

    print(f"\n阈值：{threshold}")
    print(f"预测病变区域占比：{prediction.mean().item():.4%}")
    print(f"Dice：{dice.item():.4f}")

print("\n诊断完成！")
