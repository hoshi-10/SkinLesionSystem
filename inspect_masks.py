
from pathlib import Path
from PIL import Image
import numpy as np

DATASET_DIR = Path(
    r"D:\SkinLesionProject\dataset\PH2Dataset\PH2 Dataset images"
)

valid_extensions = [".bmp", ".jpg", ".jpeg", ".png"]

total = 0
binary_masks = 0
empty_masks = 0
invalid_masks = []
preview_saved = False

for case_dir in sorted(DATASET_DIR.iterdir()):

    if not case_dir.is_dir():
        continue

    case_name = case_dir.name

    original_dir = case_dir / f"{case_name}_Dermoscopic_Image"
    mask_dir = case_dir / f"{case_name}_lesion"

    originals = [
        f for f in original_dir.glob("*")
        if f.suffix.lower() in valid_extensions
    ]

    masks = [
        f for f in mask_dir.glob("*")
        if f.suffix.lower() in valid_extensions
    ]

    if len(originals) != 1 or len(masks) != 1:
        invalid_masks.append(case_name)
        continue

    with Image.open(originals[0]) as img:
        original = img.convert("RGB").copy()

    with Image.open(masks[0]) as mask_img:
        mask = np.array(mask_img.convert("L"))

    total += 1

    unique_values = np.unique(mask)

    # 检查是否为二值掩码
    if len(unique_values) <= 2:
        binary_masks += 1

    # 检查是否为空白掩码
    if np.max(mask) == 0:
        empty_masks += 1

    # 保存一个样本用于人工查看
    if not preview_saved:
        mask_rgb = Image.fromarray(mask).convert("RGB")

        canvas = Image.new(
            "RGB",
            (original.width * 2, original.height)
        )

        canvas.paste(original, (0, 0))
        canvas.paste(mask_rgb, (original.width, 0))

        preview_path = (
            Path(r"D:\SkinLesionProject\dataset")
            / "ph2_sample_preview.png"
        )

        canvas.save(preview_path)
        preview_saved = True

        print("样本病例：", case_name)
        print("原图文件：", originals[0].name)
        print("掩码文件：", masks[0].name)
        print("掩码像素值：", unique_values[:20])

print("\n========== 掩码检查报告 ==========")
print("成功检查病例：", total)
print("二值掩码数量：", binary_masks)
print("空白掩码数量：", empty_masks)
print("文件异常病例：", invalid_masks)

print("\n检查完成！")
if preview_saved:
    print("样本对比图：D:\\SkinLesionProject\\dataset\\ph2_sample_preview.png")
