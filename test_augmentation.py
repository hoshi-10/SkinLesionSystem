
import torch

from src.dataset import SkinLesionDataset


train_dataset = SkinLesionDataset(split="train")
val_dataset = SkinLesionDataset(split="val")
test_dataset = SkinLesionDataset(split="test")

print("\n========== 数据增强检查 ==========")

print("训练集数量：", len(train_dataset))
print("验证集数量：", len(val_dataset))
print("测试集数量：", len(test_dataset))

print("\n========== 训练集样本检查 ==========")

image, mask = train_dataset[0]

print("图片形状：", image.shape)
print("掩码形状：", mask.shape)
print("图片类型：", image.dtype)
print("掩码类型：", mask.dtype)
print("掩码像素值：", torch.unique(mask).tolist())

print("\n========== 数据增强配置 ==========")
print("训练集增强：", train_dataset.augment)
print("验证集增强：", val_dataset.augment)
print("测试集增强：", test_dataset.augment)

print("\n检查完成！")
