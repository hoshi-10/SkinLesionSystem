
import numpy as np
from PIL import Image


def preprocess_image(image):
    """
    图片预处理，与训练集 dataset.py 保持一致。

    1. 转换为 RGB
    2. 保持宽高比例缩放
    3. 居中填充黑色背景
    4. 调整为 256×256
    5. 转换为 NumPy 数组
    6. 像素归一化到 0～1

    返回：
        shape: (256, 256, 3)
        dtype: float32
    """

    # =========================
    # 1. 统一颜色格式
    # =========================

    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效！")

    # =========================
    # 2. 保持比例缩放
    # =========================

    image_size = 256

    # 与 dataset.py 使用相同的缩放比例计算方式
    scale = image_size / max(width, height)

    # 与 dataset.py 一样使用 round() 四舍五入
    new_width = round(width * scale)
    new_height = round(height * scale)

    # 防止极端尺寸导致缩放结果为 0
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    # 使用与 dataset.py 相同的双线性插值
    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    # =========================
    # 3. 创建黑色画布并居中填充
    # =========================

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        (0, 0, 0)
    )

    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2

    canvas.paste(image, (left, top))

    # =========================
    # 4. 转换为 NumPy 数组
    # =========================

    image_array = np.array(
        canvas,
        dtype=np.float32
    )

    # =========================
    # 5. 像素归一化
    # =========================

    image_array = image_array / 255.0

    return image_array
