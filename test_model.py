
import torch

from src.dataset import SkinLesionDataset
from src.model import UNet


# 1. 设置设备
device = torch.device("cpu")

# 2. 加载一组真实图片
dataset = SkinLesionDataset(split="train")

image, mask = dataset[0]

# 增加批次维度
# 原来：[3, 256, 256]
# 现在：[1, 3, 256, 256]
image = image.unsqueeze(0).to(device)

# 3. 创建 U-Net 模型
model = UNet().to(device)

# 切换到测试模式
model.eval()

# 4. 输入图片，获得模型预测结果
with torch.no_grad():
    output = model(image)

    # 将模型输出转换成 0~1 的概率
    prediction = torch.sigmoid(output)

# 5. 打印测试结果
print("\n========== U-Net 模型测试 ==========")

print("输入图片形状：", image.shape)
print("真实掩码形状：", mask.unsqueeze(0).shape)
print("模型输出形状：", output.shape)
print("预测概率形状：", prediction.shape)

print("\n输入数据类型：", image.dtype)
print("模型输出数据类型：", output.dtype)

print("\n预测概率最小值：", prediction.min().item())
print("预测概率最大值：", prediction.max().item())

# 6. 检查输入输出尺寸
if output.shape == mask.unsqueeze(0).shape:
    print("\n测试通过！输入与输出尺寸一致。")
else:
    print("\n测试失败！输入与输出尺寸不一致。")

# 7. 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())

print("模型参数量：", total_params)
