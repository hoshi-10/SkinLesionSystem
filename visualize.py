
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import SkinLesionDataset
from src.model import UNet


# 1. 基础配置
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "checkpoints" / "best_unet_aug.pth"
OUTPUT_DIR = ROOT / "results" / "predictions"
CSV_PATH = ROOT / "results" / "dice_scores.csv"

DEVICE = torch.device("cpu")


# 2. 检查模型文件
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. 加载测试集和模型
dataset = SkinLesionDataset(split="test")

if len(dataset) == 0:
    raise ValueError("测试集为空！")

model = UNet().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

scores = []

# 4. 遍历全部测试图片
for index in range(len(dataset)):

    image, mask = dataset[index]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(DEVICE))
        prediction = (torch.sigmoid(output) >= 0.6).float()#修改阈值，从0.5到0.6！

    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    # 计算单张图片的 Dice
    intersection = np.logical_and(
        prediction == 1, mask == 1
    ).sum()

    total = prediction.sum() + mask.sum()
    dice = (2 * intersection + 1) / (total + 1)

    scores.append(dice)

    # 绘制三图对比
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction | Dice: {dice:.4f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    image_path = OUTPUT_DIR / f"sample_{index + 1:03d}.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"完成 {index + 1}/{len(dataset)}，Dice：{dice:.4f}")


# 5. 保存全部 Dice 记录
with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample Index", "Dice"])

    for index, score in enumerate(scores, start=1):
        writer.writerow([index, f"{score:.4f}"])

# 6. 输出统计结果
print("\n========== 全部测试图片评估完成 ==========")
print("测试图片数量：", len(scores))
print(f"平均 Dice：{np.mean(scores):.4f}")
print(f"最高 Dice：{np.max(scores):.4f}")
print(f"最低 Dice：{np.min(scores):.4f}")
print("预测图片目录：", OUTPUT_DIR)
print("Dice 记录文件：", CSV_PATH)
