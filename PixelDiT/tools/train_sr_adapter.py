"""Memory-conscious adapter-only SR training for PixelDiT.

This script intentionally avoids the generic T2I launcher, which retains full
FP32 model/checkpoint copies while initializing distributed text-to-image
training.  PixelDiT and DINO remain frozen; only the LR-token adapter learns.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "t2i"))

from diffusion import Scheduler
from diffusion.data.datasets.paired_sr_dataset import PairedSRDataset
from diffusion.model.lr_conditioner import FrozenDINOv2BConditioner
import pixdit_core.pixeldit_t2i as pixeldit_module


class CoreForScheduler(torch.nn.Module):
    """Match Scheduler's ``timestep=`` keyword to PixelDiT's ``t`` argument."""

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor, lr: torch.Tensor | None = None, **_: object
    ) -> torch.Tensor:
        dtype = next(self.core.parameters()).dtype
        return self.core(x.to(dtype), timestep.to(dtype), y.to(dtype), lr=lr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "t2i/configs/PixelDiT_SR_DINOv2B_x4.yaml"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--init-adapter", default=None, help="Optional prior adapter checkpoint to continue from.")
    parser.add_argument("--output", default=str(ROOT / "t2i_experiments/sr_div2k_512_adapter"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1)
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    dataset = PairedSRDataset(**OmegaConf.to_container(cfg.data, resolve=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    conditioner_cfg = OmegaConf.to_container(cfg.model.extra.lr_conditioner, resolve=True)
    conditioner = FrozenDINOv2BConditioner(**conditioner_cfg).to(device=device, dtype=torch.bfloat16)
    conditioner.train()
    if args.init_adapter:
        previous = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        conditioner.adapter.load_state_dict(previous["adapter"], strict=True)
        print(f"Initialized DINO adapter from {args.init_adapter} (step {previous['step']}).", flush=True)

    core = pixeldit_module.PixDiT_T2I(
        **{
            key: value
            for key, value in OmegaConf.to_container(cfg.model.extra, resolve=True).items()
            if key not in {"lr_conditioner", "freeze_pixeldit_backbone"}
        }
    ).to(device=device, dtype=torch.bfloat16)
    checkpoint = torch.load(cfg.model.load_from, map_location="cpu", weights_only=False)
    core_state = {
        key.removeprefix("core."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("core.")
    }
    missing, unexpected = core.load_state_dict(core_state, strict=False)
    expected_missing = {"lr_pixel_gates", "lr_pixel_embedder.proj.weight", "lr_pixel_embedder.proj.bias"}
    if unexpected or set(missing) != expected_missing:
        raise RuntimeError(f"Unexpected pretrained checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del checkpoint, core_state
    gc.collect()
    core.requires_grad_(False).eval()
    if not core.use_lr_pixel_cond:
        raise RuntimeError("Set model.extra.use_lr_pixel_cond=true for local LR conditioning training.")
    core.lr_pixel_embedder.requires_grad_(True)
    core.lr_pixel_gates.requires_grad_(True)
    if args.init_adapter and "lr_pixel_embedder" in previous:
        core.lr_pixel_embedder.load_state_dict(previous["lr_pixel_embedder"], strict=True)
        if "lr_pixel_gates" in previous:
            core.lr_pixel_gates.data.copy_(previous["lr_pixel_gates"])
        elif "lr_pixel_gate" in previous:
            # Upgrade the earlier one-time injection checkpoint: begin each
            # PiT block with the learned legacy gate.
            core.lr_pixel_gates.data.copy_(
                previous["lr_pixel_gate"].unsqueeze(0).expand_as(core.lr_pixel_gates)
            )
        print("Initialized local LR pixel adapter from prior checkpoint.", flush=True)
    model_for_loss = CoreForScheduler(core)

    optimizer = torch.optim.AdamW(
        list(conditioner.adapter.parameters())
        + list(core.lr_pixel_embedder.parameters())
        + [core.lr_pixel_gates],
        lr=args.lr,
        weight_decay=0.0,
    )
    scheduler = Scheduler(
        str(cfg.scheduler.train_sampling_steps),
        noise_schedule=cfg.scheduler.noise_schedule,
        predict_flow_v=cfg.scheduler.predict_flow_v,
        learn_sigma=cfg.scheduler.learn_sigma,
        pred_sigma=cfg.scheduler.pred_sigma,
        flow_shift=cfg.scheduler.flow_shift,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = sum(p.numel() for p in conditioner.adapter.parameters()) + sum(
        p.numel() for p in core.lr_pixel_embedder.parameters()
    ) + core.lr_pixel_gates.numel()
    print(f"Training {trainable:,} SR-condition parameters on {len(dataset)} pairs.", flush=True)

    data_iter = iter(loader)
    progress = tqdm(
        range(1, args.steps + 1),
        desc="PixelDiT-SR adapter",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        hr, lr = batch[0].to(device, dtype=torch.bfloat16, non_blocking=True), batch[8].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )
        timestep = torch.randint(0, cfg.scheduler.train_sampling_steps, (hr.shape[0],), device=device).long()
        tokens = conditioner(lr)
        loss = scheduler.training_losses(
            model_for_loss, hr, timestep, model_kwargs={"y": tokens, "lr": lr}
        )["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conditioner.adapter.parameters(), 0.1)
        optimizer.step()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        progress.set_postfix(loss=f"{loss.item():.5f}", peak=f"{peak:.2f}GiB")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_path = output_dir / f"adapter_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "adapter": conditioner.adapter.state_dict(),
                    "lr_pixel_embedder": core.lr_pixel_embedder.state_dict(),
                    "lr_pixel_gates": core.lr_pixel_gates.detach().cpu(),
                    "config": conditioner_cfg,
                },
                checkpoint_path,
            )
            progress.write(f"Saved adapter checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
