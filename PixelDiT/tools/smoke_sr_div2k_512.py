"""Run the SR wrapper on a real 128 -> 512 DIV2K pair using pretrained PixDiT."""
from __future__ import annotations

import sys
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "t2i"))

from diffusion.data.datasets.paired_sr_dataset import PairedSRDataset
from diffusion.model.trainer import PixDiTTrainer


cfg = OmegaConf.load(ROOT / "t2i/configs/PixelDiT_SR_DINOv2B_x4.yaml")
dataset = PairedSRDataset(**OmegaConf.to_container(cfg.data, resolve=True))
hr, _, _, _, _, _, _, _, lr = dataset[0]

model = PixDiTTrainer(
    image_size=cfg.model.image_size,
    caption_channels=cfg.model.extra.txt_embed_dim,
    model_max_length=cfg.model.extra.txt_max_length,
    extra=OmegaConf.to_container(cfg.model.extra, resolve=True),
    lr_conditioner=OmegaConf.to_container(cfg.model.extra.lr_conditioner, resolve=True),
    freeze_pixeldit_backbone=cfg.model.extra.freeze_pixeldit_backbone,
)
checkpoint = torch.load(cfg.model.load_from, map_location="cpu", weights_only=False)
core_state = {
    key.removeprefix("core."): value
    for key, value in checkpoint["state_dict"].items()
    if key.startswith("core.")
}
model.core.load_state_dict(core_state, strict=True)
del checkpoint, core_state
gc.collect()
print("Loaded pretrained PixelDiT weights; releasing checkpoint copy.", flush=True)
model.cuda().bfloat16().eval()

hr = hr.unsqueeze(0).cuda().bfloat16()
lr = lr.unsqueeze(0).cuda().bfloat16()
timestep = torch.full((1,), 0.5, device="cuda", dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
print("Running frozen-DINO conditioned 512x512 forward...", flush=True)
with torch.inference_mode():
    prediction = model(hr, timestep, lr=lr)["x"]
peak_gib = torch.cuda.max_memory_allocated() / 1024**3
print(f"DIV2K pair: LR {tuple(lr.shape)} -> target {tuple(hr.shape)}")
print(f"SR model output: {tuple(prediction.shape)}, {prediction.dtype}; peak={peak_gib:.2f} GiB")
