
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import SkinLesionDataset
from src.model import UNet


# =========================
# 1. 基础配置
# =========================

device = torch.device("cpu")

BATCH_SIZE = 2
THRESHOLD = 0.6

# 两个模型的保存路径
OLD_MODEL_PATH = Path("checkpoints") / "best_unet.pth"
NEW_MODEL_PATH = Path("checkpoints") / "best_unet_aug.pth"


# =========================
# 2. 加载测试集
# =========================

test_dataset = SkinLesionDataset(split="test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("测试集数量：", len(test_dataset))


# =========================
# 3. 定义损失函数
# =========================

criterion = nn.BCEWithLogitsLoss()


# =========================
# 4. 定义 Dice 指标
# =========================

def calculate_dice(output, mask):

    # 将模型输出转换为概率
    prediction = torch.sigmoid(output)

    # 使用统一的 0.6 阈值
    prediction = (prediction >= THRESHOLD).float()

    # 每张图片单独计算交集
    intersection = (prediction * mask).sum(dim=(1, 2, 3))

    # 每张图片单独计算 Dice
    dice = (
        2 * intersection + 1
    ) / (
        prediction.sum(dim=(1, 2, 3))
        + mask.sum(dim=(1, 2, 3))
        + 1
    )

    return dice


# =========================
# 5. 定义模型测试函数
# =========================

def evaluate_model(model_path, model_name):

    print(f"\n正在测试：{model_name}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}"
        )

    # 创建模型
    model = UNet().to(device)

    # 加载模型参数
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            # 模型预测
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)

            # 每张图片分别计算 Dice
            dice = calculate_dice(outputs, masks)

            total_loss += loss.item()

            total_dice += dice.sum().item()
            total_images += images.size(0)

    avg_loss = total_loss / len(test_loader)
    avg_dice = total_dice / total_images

    print(f"{model_name} 测试 Loss：{avg_loss:.4f}")
    print(f"{model_name} 测试 Dice：{avg_dice:.4f}")

    return avg_loss, avg_dice


# =========================
# 6. 分别测试新旧模型
# =========================

print("\n========== 开始新旧模型对比 ==========")

old_loss, old_dice = evaluate_model(
    OLD_MODEL_PATH,
    "旧模型（无数据增强）"
)

new_loss, new_dice = evaluate_model(
    NEW_MODEL_PATH,
    "新模型（有数据增强）"
)


# =========================
# 7. 输出对比结果
# =========================

print("\n========== 新旧模型对比结果 ==========")

print(f"旧模型测试 Loss：{old_loss:.4f}")
print(f"新模型测试 Loss：{new_loss:.4f}")

print(f"旧模型测试 Dice：{old_dice:.4f}")
print(f"新模型测试 Dice：{new_dice:.4f}")

print(f"\nDice 变化：{new_dice - old_dice:+.4f}")

if new_dice > old_dice:
    print("新模型在测试集上的 Dice 更高。")
elif new_dice < old_dice:
    print("旧模型在测试集上的 Dice 更高。")
else:
    print("新旧模型的测试 Dice 相同。")

print("\n测试完成！")
