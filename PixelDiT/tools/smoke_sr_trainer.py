"""Verify that the production PixDiTTrainer accepts LR directly in SR mode."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "t2i"))

from diffusion.model.trainer import PixDiTTrainer


cfg = OmegaConf.load(ROOT / "t2i/configs/PixelDiT_SR_DINOv2B_x4.yaml")
model = PixDiTTrainer(
    image_size=cfg.model.image_size,
    caption_channels=cfg.model.extra.txt_embed_dim,
    model_max_length=cfg.model.extra.txt_max_length,
    extra=OmegaConf.to_container(cfg.model.extra, resolve=True),
    lr_conditioner=OmegaConf.to_container(cfg.model.lr_conditioner, resolve=True),
    freeze_pixeldit_backbone=True,
).cuda().bfloat16().eval()
lr = torch.rand(1, 3, 64, 64, device="cuda", dtype=torch.bfloat16)
x = torch.randn(1, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
t = torch.full((1,), 0.5, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    result = model(x, t, lr=lr)["x"]
print(f"trainer SR forward: LR {tuple(lr.shape)} -> output {tuple(result.shape)} ({result.dtype})")
