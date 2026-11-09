import csv

csv_path = "dataset/split.csv"

with open(csv_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

test_rows = [row for row in rows if row["split"] == "test"]

# sample_018 是测试集中的第 18 个样本
sample = test_rows[17]

print("病例编号：", sample["case_id"])
print("原始图片路径：", sample["image_path"])
print("真实掩码路径：", sample["mask_path"])