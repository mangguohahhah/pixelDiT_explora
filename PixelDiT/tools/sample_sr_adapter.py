"""Sample a 512px SR image from an LR image and a trained PixelDiT adapter."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "t2i"))

from diffusion import DPMS
from diffusion.model.lr_conditioner import FrozenDINOv2BConditioner
import pixdit_core.pixeldit_t2i as pixeldit_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "t2i/configs/PixelDiT_SR_DINOv2B_x4.yaml"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--lr-image", default="/mnt/data/home/sqkang/datasets/DIV2K/S3Diff_test/LR/00000.png")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "t2i_experiments/sr_samples/div2k_00000_step.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    conditioner_cfg = OmegaConf.to_container(cfg.model.extra.lr_conditioner, resolve=True)
    conditioner = FrozenDINOv2BConditioner(**conditioner_cfg).to(device=device, dtype=torch.bfloat16).eval()
    adapter_state = torch.load(args.adapter, map_location="cpu", weights_only=False)
    conditioner.adapter.load_state_dict(adapter_state["adapter"], strict=True)

    core = pixeldit_module.PixDiT_T2I(
        **{
            key: value
            for key, value in OmegaConf.to_container(cfg.model.extra, resolve=True).items()
            if key not in {"lr_conditioner", "freeze_pixeldit_backbone"}
        }
    ).to(device=device, dtype=torch.bfloat16)
    pretrained = torch.load(cfg.model.load_from, map_location="cpu", weights_only=False)
    missing, unexpected = core.load_state_dict(
        {key.removeprefix("core."): value for key, value in pretrained["state_dict"].items() if key.startswith("core.")},
        strict=False,
    )
    expected_missing = {"lr_pixel_gates", "lr_pixel_embedder.proj.weight", "lr_pixel_embedder.proj.bias"}
    if unexpected or set(missing) != expected_missing:
        raise RuntimeError(f"Unexpected pretrained checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if "lr_pixel_embedder" not in adapter_state:
        raise ValueError("Adapter checkpoint predates local LR pixel conditioning; train a new local-condition adapter first.")
    core.lr_pixel_embedder.load_state_dict(adapter_state["lr_pixel_embedder"], strict=True)
    if "lr_pixel_gates" in adapter_state:
        core.lr_pixel_gates.data.copy_(adapter_state["lr_pixel_gates"])
    elif "lr_pixel_gate" in adapter_state:
        # Legacy local-condition checkpoint: reuse its learned gate for every
        # PiT block, matching its prior strength at the first repeated pass.
        core.lr_pixel_gates.data.copy_(
            adapter_state["lr_pixel_gate"].unsqueeze(0).expand_as(core.lr_pixel_gates)
        )
    else:
        raise ValueError("Adapter checkpoint has no LR pixel gate parameters.")
    del pretrained
    gc.collect()
    core.eval()

    lr = (
        pil_to_tensor(Image.open(args.lr_image).convert("RGB"))
        .float()
        .div_(255)
        .unsqueeze(0)
        .to(device=device, dtype=torch.bfloat16)
    )
    with torch.inference_mode():
        condition = conditioner(lr)
        uncondition = torch.zeros_like(condition)
        noise = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size, device=device, dtype=torch.bfloat16)

        def model_fn(x, timestep, y, **_):
            return core(x.to(torch.bfloat16), timestep.to(torch.bfloat16), y.to(torch.bfloat16), lr=lr)

        solver = DPMS(
            model_fn,
            condition=condition,
            uncondition=uncondition,
            guidance_type="classifier-free",
            cfg_scale=1.0,
            model_type="flow",
            model_kwargs={},
            schedule="FLOW",
            interval_guidance=[0, 1],
        )
        sample = solver.sample(
            noise,
            steps=args.steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=cfg.scheduler.flow_shift,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(sample, output, normalize=True, value_range=(-1, 1))
    print(f"Saved {output} from {args.lr_image} with adapter step={adapter_state['step']}.")


if __name__ == "__main__":
    main()
