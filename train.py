
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
EPOCHS = 20
LEARNING_RATE = 0.0001

# 验证时使用之前实验选出的阈值
THRESHOLD = 0.6

# 模型保存目录
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 新模型保存路径，不覆盖原模型
BEST_MODEL_PATH = SAVE_DIR / "best_unet_aug.pth"


# =========================
# 2. 加载训练集和验证集
# =========================

train_dataset = SkinLesionDataset(split="train")
val_dataset = SkinLesionDataset(split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("训练集数量：", len(train_dataset))
print("验证集数量：", len(val_dataset))


# =========================
# 3. 创建模型、损失函数和优化器
# =========================

model = UNet().to(device)

# 二分类分割任务的损失函数
criterion = nn.BCEWithLogitsLoss()

# Adam 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# 4. 定义 Dice 评价指标
# =========================

def calculate_dice(output, mask):

    # 将模型输出转换为概率
    prediction = torch.sigmoid(output)

    # 使用 0.6 作为分割阈值
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
# 5. 开始训练
# =========================

best_dice = 0.0

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # 清空上一轮的梯度
        optimizer.zero_grad()

        # 模型预测
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, masks)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)


    # =========================
    # 6. 验证模型
    # =========================

    model.eval()

    total_val_loss = 0.0
    total_dice = 0.0
    total_val_images = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            # 每张图片分别计算 Dice
            dice = calculate_dice(outputs, masks)

            total_val_loss += loss.item()

            total_dice += dice.sum().item()
            total_val_images += images.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_dice = total_dice / total_val_images

    print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")
    print(f"训练损失：{avg_train_loss:.4f}")
    print(f"验证损失：{avg_val_loss:.4f}")
    print(f"验证 Dice：{avg_dice:.4f}")


    # =========================
    # 7. 保存表现最好的模型
    # =========================

    if avg_dice > best_dice:

        best_dice = avg_dice

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print("已保存当前最佳模型！")


print("\n训练结束！")
print("最佳验证 Dice：", round(best_dice, 4))
print("模型保存位置：", BEST_MODEL_PATH)
