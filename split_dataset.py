
from pathlib import Path
import random
import csv

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = Path(
    r"D:\SkinLesionProject\dataset\PH2Dataset\PH2 Dataset images"
)

# 划分结果保存位置
OUTPUT_FILE = PROJECT_DIR / "dataset" / "split.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
VALID_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}

# 固定随机种子，保证每次划分结果一致
random.seed(42)

# 收集所有原图和对应掩码
samples = []
errors = []

for case_dir in sorted(DATASET_DIR.iterdir()):
    if not case_dir.is_dir():
        continue

    case_name = case_dir.name

    original_dir = case_dir / f"{case_name}_Dermoscopic_Image"
    mask_dir = case_dir / f"{case_name}_lesion"

    originals = [
        f for f in original_dir.glob("*")
        if f.suffix.lower() in VALID_EXTENSIONS
    ]

    masks = [
        f for f in mask_dir.glob("*")
        if f.suffix.lower() in VALID_EXTENSIONS
    ]

    if len(originals) != 1 or len(masks) != 1:
        errors.append(case_name)
        continue

    samples.append({
        "case_id": case_name,
        "image_path": str(originals[0]),
        "mask_path": str(masks[0]),
    })

# 有异常就先停止，避免生成不完整的划分结果
if errors:
    print("发现文件异常病例：", errors)
    print("请先检查数据集，暂不生成划分文件。")
    raise SystemExit

# 打乱样本顺序
random.shuffle(samples)

# 按照 70% / 15% / 15% 划分
total = len(samples)
train_end = int(total * 0.7)
val_end = int(total * 0.85)

for index, sample in enumerate(samples):
    if index < train_end:
        sample["split"] = "train"
    elif index < val_end:
        sample["split"] = "val"
    else:
        sample["split"] = "test"

# 写入 CSV 文件
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["case_id", "image_path", "mask_path", "split"]
    )
    writer.writeheader()
    writer.writerows(samples)

# 输出检查报告
train_count = sum(s["split"] == "train" for s in samples)
val_count = sum(s["split"] == "val" for s in samples)
test_count = sum(s["split"] == "test" for s in samples)

print("========== 数据集划分报告 ==========")
print("总样本数：", total)
print("训练集：", train_count)
print("验证集：", val_count)
print("测试集：", test_count)
print("总计：", train_count + val_count + test_count)
print("\n划分文件已保存：", OUTPUT_FILE)
print("数据集划分完成！")
