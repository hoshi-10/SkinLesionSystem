
import streamlit as st
import numpy as np
import torch

from PIL import Image
from pathlib import Path

from src.preprocessing import preprocess_image
from src.model import UNet


# =========================
# 1. 网页基本设置
# =========================

st.set_page_config(
    page_title="皮肤病变图像智能分割系统",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 皮肤病变图像智能分割系统")

st.markdown("""
本系统基于 U-Net 深度学习模型，
实现皮肤镜图像的病变区域分割与可视化。
""")

st.divider()


# =========================
# 2. 基础配置
# =========================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "checkpoints" / "best_unet_aug.pth"

DEVICE = torch.device("cpu")

THRESHOLD = 0.6


# =========================
# 3. 加载模型
# =========================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_PATH}"
        )

    model = UNet().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# =========================
# 4. 图片上传
# =========================

st.subheader("📷 上传皮肤镜图片")

uploaded_file = st.file_uploader(
    "上传皮肤镜图片",
    type=["bmp", "jpg", "jpeg", "png"],
    help="支持 BMP、JPG、JPEG、PNG 格式"
)


# =========================
# 5. 图片预测
# =========================

if uploaded_file is not None:

    try:

        # 读取原始图片
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # 图片预处理
        processed_array = preprocess_image(
            original_image
        )

        # NumPy 数组转换为 PyTorch 张量
        image_tensor = torch.from_numpy(
            processed_array.transpose(2, 0, 1).copy()
        )

        # 增加批次维度
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # 加载模型
        model = load_model()

        # 模型预测
        with torch.no_grad():

            output = model(image_tensor)

            probability = torch.sigmoid(output)

            prediction = (
                probability >= THRESHOLD
            ).float()

        # 转换预测结果为 NumPy 数组
        probability_array = (
            probability.squeeze().cpu().numpy()
        )

        mask_array = (
            prediction.squeeze().cpu().numpy()
        )

        # =========================
        # 6. 生成叠加图
        # =========================

        processed_uint8 = (
            processed_array * 255
        ).astype(np.uint8)

        overlay_array = processed_uint8.copy()

        # 使用红色标记预测出的病变区域
        overlay_array[mask_array == 1] = (
            overlay_array[mask_array == 1] * 0.5
            + np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)

        # 计算预测区域占比
        lesion_ratio = (
            np.sum(mask_array == 1)
            / mask_array.size
            * 100
        )

        # =========================
        # 7. 展示分割结果
        # =========================

        st.success("图片分割完成！")

        st.divider()

        st.subheader("🖼️ 分割结果")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### 原始图片")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 预测掩码")

            st.image(
                mask_array,
                clamp=True,
                use_container_width=True
            )

        with col3:

            st.markdown("### 分割叠加图")

            st.image(
                overlay_array,
                use_container_width=True
            )

        # =========================
        # 8. 展示预测数据
        # =========================

        st.divider()

        st.subheader("📊 分割数据")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "原始图片尺寸",
                f"{original_image.width} × {original_image.height}"
            )

        with col2:
            st.metric(
                "预测图片尺寸",
                "256 × 256"
            )

        with col3:
            st.metric(
                "预测病变区域占比",
                f"{lesion_ratio:.2f}%"
            )

        st.caption(
            f"模型：U-Net | 阈值：{THRESHOLD} | 设备：CPU"
        )

    except Exception as e:

        st.error(f"图片处理或模型预测失败：{e}")


# =========================
# 9. 使用说明
# =========================

st.divider()

st.caption(
    "本系统仅用于学习和科研实验，不提供医学诊断，"
    "分割结果不能替代专业医生的判断。"
)
