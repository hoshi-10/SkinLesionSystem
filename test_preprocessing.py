
from PIL import Image
import numpy as np

from src.preprocessing import preprocess_image


# 创建一张 600×400 的测试图片
test_image = Image.new(
    "RGB",
    (600, 400),
    (200, 100, 50)
)

# 调用预处理函数
result = preprocess_image(test_image)

print("========== 预处理测试 ==========")

print("原始图片尺寸：", test_image.size)

print("预处理后形状：", result.shape)

print("数据类型：", result.dtype)

print("最小像素值：", result.min())

print("最大像素值：", result.max())

print("预处理结果是否存在异常值：", np.isnan(result).any())

print("\n测试完成！")
