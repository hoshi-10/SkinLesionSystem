
from pathlib import Path
from PIL import Image

# PH2 数据集路径
DATASET_DIR = Path(
    r"D:\SkinLesionProject\dataset\PH2Dataset\PH2 Dataset images"
)

total_cases = 0
original_count = 0
mask_count = 0
missing_original = []
missing_mask = []
size_errors = []

# 遍历所有病例
for case_dir in sorted(DATASET_DIR.iterdir()):

    if not case_dir.is_dir():
        continue

    total_cases += 1

    # 查找原始图片
    original_dir = case_dir / f"{case_dir.name}_Dermoscopic_Image"

    # 查找病变分割掩码
    mask_dir = case_dir / f"{case_dir.name}_lesion"

    original_files = list(original_dir.glob("*")) if original_dir.exists() else []
    mask_files = list(mask_dir.glob("*")) if mask_dir.exists() else []

    original_files = [
        f for f in original_files
        if f.suffix.lower() in [".bmp", ".jpg", ".jpeg", ".png"]
    ]

    mask_files = [
        f for f in mask_files
        if f.suffix.lower() in [".bmp", ".jpg", ".jpeg", ".png"]
    ]

    if original_files:
        original_count += 1
    else:
        missing_original.append(case_dir.name)

    if mask_files:
        mask_count += 1
    else:
        missing_mask.append(case_dir.name)

    # 检查原图与掩码尺寸
    if original_files and mask_files:
        with Image.open(original_files[0]) as img:
            original_size = img.size

        with Image.open(mask_files[0]) as mask:
            mask_size = mask.size

        if original_size != mask_size:
            size_errors.append(
                (case_dir.name, original_size, mask_size)
            )

# 输出检查报告
print("=" * 40)
print("PH2 数据集检查报告")
print("=" * 40)

print(f"病例总数：{total_cases}")
print(f"存在原始图片的病例：{original_count}")
print(f"存在分割掩码的病例：{mask_count}")

print("\n缺少原图的病例：", missing_original)
print("缺少掩码的病例：", missing_mask)
print("图片尺寸不一致：", size_errors)

print("\n检查完成！")
