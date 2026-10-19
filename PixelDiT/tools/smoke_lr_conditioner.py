"""End-to-end shape smoke test for DINO-LR conditioning and PixelDiT.

Run from the PixelDiT repository after copying this file there:
  python tools/smoke_lr_conditioner.py --lr-image /path/to/lr.png
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "t2i"))

from t2i.diffusion.model.lr_conditioner import FrozenDINOv2BConditioner
import pixdit_core.pixeldit_t2i as pixeldit_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-image", required=True)
    parser.add_argument(
        "--dino-repo",
        default="/mnt/data/home/sqkang/projects_yj/oppo学习/OVSR/vosr_repo/preset/ckpts/torch_cache/facebookresearch_dinov2_main",
    )
    parser.add_argument(
        "--dino-weights",
        default="/mnt/data/home/sqkang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
    )
    parser.add_argument(
        "--pixeldit-weights", default=str(ROOT / "weights/pixeldit_t2i_v1.pth")
    )
    parser.add_argument(
        "--config", default=str(ROOT / "t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml")
    )
    parser.add_argument("--test-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA.")
    device = torch.device("cuda")

    lr = pil_to_tensor(Image.open(args.lr_image).convert("RGB")).float().div_(255).unsqueeze(0)
    conditioner = FrozenDINOv2BConditioner(args.dino_repo, args.dino_weights).to(device).eval()
    with torch.inference_mode():
        visual_tokens = conditioner(lr.to(device))
    print(f"DINO LR -> PixelDiT condition: {tuple(visual_tokens.shape)}, {visual_tokens.dtype}")

    # We only need the generated conditioning tensor for this no-gradient test;
    # release the frozen DINO model before loading the 1.3B PixelDiT checkpoint.
    visual_tokens = visual_tokens.to(dtype=torch.bfloat16)
    del conditioner
    gc.collect()
    torch.cuda.empty_cache()

    cfg = OmegaConf.load(args.config)
    model = pixeldit_module.PixDiT_T2I(**OmegaConf.to_container(cfg.model.extra)).to(device)
    checkpoint = torch.load(args.pixeldit_weights, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("core."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("core.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(dtype=torch.bfloat16)

    x = torch.randn(1, 3, args.test_size, args.test_size, device=device, dtype=torch.bfloat16)
    t = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(x, t, visual_tokens)
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"PixelDiT output: {tuple(output.shape)}, {output.dtype}; peak allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
