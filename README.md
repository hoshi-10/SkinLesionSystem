# 基于 U-Net 深度学习的皮肤病变图像分割系统（医学图像处理）

## 1. 项目简介

本项目基于深度学习技术，设计并实现了一个皮肤病变图像分割系统。系统采用 U-Net 卷积神经网络，对皮肤镜图像中的病变区域进行自动分割，并通过可视化方式展示分割结果。

项目使用 Python、PyTorch 和 Streamlit 开发，包含数据预处理、数据增强、模型训练、模型评估和网页端预测等功能。

## 2. 主要功能

- **图像预处理**：将输入图像转换为 RGB 格式，并统一调整为 256×256 大小。
- **数据增强**：通过随机水平翻转、垂直翻转和旋转等方式扩充训练数据。
- **模型训练**：使用 U-Net 网络训练皮肤病变图像分割模型。
- **模型评估**：使用 Dice 系数评价模型分割效果。
- **图像预测**：支持上传 BMP、JPG、JPEG、PNG 格式的皮肤镜图像。
- **结果可视化**：展示原始图像、预测掩码和分割叠加效果。

## 3. 技术栈

| 技术 | 用途 |
|---|---|
| Python 3.11 | 项目开发语言 |
| PyTorch | 深度学习模型构建与训练 |
| U-Net | 图像语义分割网络 |
| NumPy | 图像数组处理 |
| Pillow | 图像读取与处理 |
| Matplotlib | 实验结果可视化 |
| Streamlit | 网页交互界面 |

## 4. 项目结构

```text
SkinLesionSystem/
├── app.py                       # Streamlit 网页入口
├── train.py                     # 模型训练
├── test.py                      # 模型测试
├── visualize.py                 # 分割结果可视化
├── split_dataset.py             # 数据集划分
├── check_dataset.py             # 数据集检查
├── inspect_masks.py             # 掩码检查
├── find_best_threshold.py       # 最优阈值搜索
├── diagnose_sample.py           # 低分样本分析
├── find_sample.py               # 样本查找
├── test_*.py                    # 各模块测试
├── requirements.txt             # 项目依赖
├── .gitignore                   # Git 忽略配置
├── checkpoints/
│   └── best_unet_aug.pth        # 最终模型权重
├── results/
│   ├── dice_scores.csv          # 测试集 Dice 结果
│   └── segmentation_comparison.png
└── src/
    ├── dataset.py               # 数据集加载
    ├── model.py                 # U-Net 模型定义
    └── preprocessing.py         # 图像预处理
```

## 5. 实验数据与结果

项目使用 PH² 皮肤镜图像数据集，共 200 组图像及对应病变掩码。

数据集划分：
- 训练集：140 组
- 验证集：30 组
- 测试集：30 组

当前实验结果：
- 最佳验证集 Dice：0.9174
- 测试集平均 Dice：0.8858
- 最优预测阈值：0.6

实验结果会受到数据划分、训练参数和模型权重等因素影响。

## 6. 环境配置

建议使用 Python 3.11。

克隆项目后，在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

## 7. 数据集准备

本仓库不包含 PH² 原始数据集。请自行获取并按照项目代码要求放置数据。

数据集划分文件及图像路径需要根据本机数据集实际存放位置配置。完成数据准备后，再运行数据集划分和检查脚本。

## 8. 运行系统

在项目根目录执行：

```bash
streamlit run app.py
```

启动后根据终端提示，在浏览器中打开系统页面，上传皮肤镜图像即可查看预测分割结果。

## 9. 模型说明

本项目采用 U-Net 编码器—解码器结构，通过跳跃连接融合不同尺度的图像特征，实现皮肤病变区域的像素级分割。

训练过程中使用 BCEWithLogitsLoss 作为损失函数，并通过验证集 Dice 指标选择模型权重。

## 10. 注意事项

- `dataset/` 目录中的数据集文件未上传至 GitHub，需要自行准备。
- `checkpoints/best_unet_aug.pth` 为当前项目使用的模型权重。
- 模型输出仅用于课程实验和技术研究，不构成医学诊断结果。
