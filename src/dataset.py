
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class SkinLesionDataset(torch.utils.data.Dataset):

    def __init__(self, split="train", image_size=256):
        self.project_dir = Path(__file__).resolve().parent.parent
        csv_path = self.project_dir / "dataset" / "split.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"找不到数据集划分文件：{csv_path}")

        self.image_size = image_size
        self.augment = split == "train"
        self.samples = []

        with open(csv_path, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == split:
                    self.samples.append(row)

        if not self.samples:
            raise ValueError(f"{split} 数据集为空！")

        print(f"{split} 数据集加载完成，共 {len(self.samples)} 组")

    def resize_and_pad(self, image, is_mask=False):
        width, height = image.size
        scale = self.image_size / max(width, height)

        new_width = round(width * scale)
        new_height = round(height * scale)

        resampling = (
            Image.Resampling.NEAREST
            if is_mask
            else Image.Resampling.BILINEAR
        )

        image = image.resize((new_width, new_height), resampling)

        if is_mask:
            canvas = Image.new(
                "L", (self.image_size, self.image_size), 0
            )
        else:
            canvas = Image.new(
                "RGB", (self.image_size, self.image_size), (0, 0, 0)
            )

        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2

        canvas.paste(image, (left, top))
        return canvas

    def augment_image(self, image, mask):
        # 随机水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        # 随机垂直翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        # 随机旋转 0、90、180 或 270 度
        k = torch.randint(0, 4, (1,)).item()

        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])

        return image, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"图片或掩码不存在：{image_path}，{mask_path}"
            )

        with Image.open(image_path) as image:
            image = self.resize_and_pad(image.convert("RGB"))

        with Image.open(mask_path) as mask:
            mask = self.resize_and_pad(
                mask.convert("L"), is_mask=True
            )

        image_array = np.array(image, dtype=np.float32) / 255.0
        mask_array = np.array(mask, dtype=np.uint8)
        mask_array = (mask_array > 0).astype(np.float32)

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        mask_tensor = torch.from_numpy(
            mask_array[np.newaxis, :, :].copy()
        )

        # 只对训练集进行数据增强
        if self.augment:
            image_tensor, mask_tensor = self.augment_image(
                image_tensor, mask_tensor
            )

        return image_tensor, mask_tensor
