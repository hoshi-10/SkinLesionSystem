
import torch
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset


# 1. 加载训练集
train_dataset = SkinLesionDataset(split="train")

# 2. 检查单张图片和掩码
image, mask = train_dataset[0]

print("\n========== 单张样本检查 ==========")
print("原图形状：", image.shape)
print("掩码形状：", mask.shape)
print("原图数据类型：", image.dtype)
print("掩码数据类型：", mask.dtype)
print("原图像素范围：", image.min().item(), image.max().item())
print("掩码像素值：", torch.unique(mask).tolist())

# 3. 使用 DataLoader 批量读取
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

images, masks = next(iter(train_loader))

print("\n========== 批量数据检查 ==========")
print("批量原图形状：", images.shape)
print("批量掩码形状：", masks.shape)

print("\n数据加载测试完成！")
