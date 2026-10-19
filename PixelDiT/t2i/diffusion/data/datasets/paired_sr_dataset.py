"""Paired LR/HR dataset for the minimal PixelDiT super-resolution baseline."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from diffusion.data.builder import DATASETS


@DATASETS.register_module()
class PairedSRDataset(Dataset):
    """Load same-named images from ``HR/X4`` and ``LR/X4`` directories.

    The HR crop is the diffusion target in [-1, 1].  The corresponding native
    LR crop remains in [0, 1] for frozen DINO conditioning.  Cropping is done
    on the LR grid, so each pair stays geometrically aligned.
    """

    def __init__(
        self,
        hr_dir: str | None = None,
        lr_dir: str | None = None,
        resolution: int = 1024,
        image_size: int | None = None,
        scale: int = 4,
        random_flip: bool = True,
        max_samples: int | None = None,
        extra: dict | None = None,
        **_: object,
    ) -> None:
        # ``DataConfig`` preserves task-specific values under ``extra`` when
        # the normal pyrallis training entry point parses a YAML file.
        extra = extra or {}
        hr_dir = hr_dir or extra.get("hr_dir")
        lr_dir = lr_dir or extra.get("lr_dir")
        scale = extra.get("scale", scale)
        random_flip = extra.get("random_flip", random_flip)
        max_samples = extra.get("max_samples", max_samples)
        if hr_dir is None or lr_dir is None:
            raise ValueError("PairedSRDataset requires hr_dir and lr_dir (directly or in data.extra).")
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.resolution = int(image_size if image_size is not None else resolution)
        self.scale = int(scale)
        self.random_flip = bool(random_flip)
        if self.resolution % self.scale:
            raise ValueError("resolution must be divisible by scale")
        self.lr_crop = self.resolution // self.scale

        self.samples = [
            (hr_path, self.lr_dir / hr_path.name)
            for hr_path in sorted(self.hr_dir.iterdir())
            if hr_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and (self.lr_dir / hr_path.name).is_file()
        ]
        if not self.samples:
            raise FileNotFoundError(f"No same-named HR/LR pairs in {self.hr_dir} and {self.lr_dir}")
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        self.ori_imgs_nums = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        hr_path, lr_path = self.samples[index]
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        lr_w, lr_h = lr.size
        if lr_w < self.lr_crop or lr_h < self.lr_crop:
            raise ValueError(f"{lr_path} is smaller than {self.lr_crop}x{self.lr_crop}")
        left = random.randint(0, lr_w - self.lr_crop)
        top = random.randint(0, lr_h - self.lr_crop)
        lr = lr.crop((left, top, left + self.lr_crop, top + self.lr_crop))
        hr = hr.crop(
            (
                left * self.scale,
                top * self.scale,
                (left + self.lr_crop) * self.scale,
                (top + self.lr_crop) * self.scale,
            )
        )
        if hr.size != (self.resolution, self.resolution):
            raise ValueError(f"Misaligned pair {hr_path}, {lr_path}: got HR size {hr.size}")
        if self.random_flip and random.random() < 0.5:
            hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        hr_tensor = pil_to_tensor(hr).float().div_(127.5).sub_(1.0)
        lr_tensor = pil_to_tensor(lr).float().div_(255.0)
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }
        # Positions 0--7 preserve the original t2i tuple convention.  Position
        # 8 carries LR and is consumed only when model.lr_conditioner is set.
        return hr_tensor, "", torch.ones(1, 1, 1, dtype=torch.int16), data_info, index, "", "", "", lr_tensor
