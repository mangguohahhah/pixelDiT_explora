"""Frozen DINOv2-B visual conditioner for a minimal PixelDiT SR baseline.

The module turns an LR image in [0, 1] into the same token interface that
PixelDiT-T2I receives from its text encoder: ``[batch, tokens, 2304]``.
Only ``adapter`` is trainable.  DINOv2 itself is loaded fully from local files
so the implementation does not make a Hugging Face or Torch Hub network call.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDINOv2BConditioner(nn.Module):
    """Encode an LR image as 16x16 PixelDiT condition tokens.

    Args:
        dino_repo: Local checkout containing DINOv2's ``hubconf.py``.
        dino_weights: Local ``dinov2_vitb14_pretrain.pth`` state dict.
        layer_index: Zero-based transformer block whose patch features are used.
        input_size: DINO input side length; 448 yields a 32x32 patch grid.
        output_grid: Spatial token grid passed to PixelDiT.
        output_dim: PixelDiT T2I condition width (2304 for Gemma-2B tokens).
    """

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dino_repo: str | Path,
        dino_weights: str | Path,
        *,
        layer_index: int = 8,
        input_size: int = 448,
        output_grid: int = 16,
        output_dim: int = 2304,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.input_size = input_size
        self.output_grid = output_grid

        self.dino = torch.hub.load(
            str(dino_repo), "dinov2_vitb14", source="local", pretrained=False
        )
        state = torch.load(dino_weights, map_location="cpu", weights_only=False)
        self.dino.load_state_dict(state, strict=True)
        self.dino.eval()
        self.dino.requires_grad_(False)

        self.adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, output_dim),
        )
        self.register_buffer("mean", torch.tensor(self.imagenet_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.imagenet_std).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the frozen feature extractor in inference mode during training."""
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def _extract_dino_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        # This follows DINOv2.forward_features up to the requested block and
        # drops the class token, matching the existing VOSR DINO extraction.
        x = self.dino.prepare_tokens_with_masks(image, masks=None)
        for block_index, block in enumerate(self.dino.blocks):
            x = block(x)
            if block_index == self.layer_index:
                return x[:, 1:]
        raise ValueError(
            f"layer_index={self.layer_index} is outside DINO's "
            f"{len(self.dino.blocks)} transformer blocks"
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Return condition tokens with shape ``[B, output_grid**2, output_dim]``."""
        if lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError(f"Expected LR image [B, 3, H, W], received {tuple(lr.shape)}")

        image = F.interpolate(
            lr.float(),
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.mean) / self.std
        # ``model.to(dtype=...)`` may cast the frozen encoder together with the
        # diffusion model.  Keep the image dtype aligned with DINO in either
        # float32 smoke tests or mixed-precision training.
        image = image.to(dtype=next(self.dino.parameters()).dtype)
        features = self._extract_dino_patch_tokens(image)

        token_count = features.shape[1]
        feature_grid = math.isqrt(token_count)
        if feature_grid * feature_grid != token_count:
            raise RuntimeError(f"DINO returned a non-square token sequence: {token_count}")
        features = features.transpose(1, 2).reshape(
            features.shape[0], features.shape[2], feature_grid, feature_grid
        )
        features = F.interpolate(
            features.float(),
            size=(self.output_grid, self.output_grid),
            mode="bilinear",
            align_corners=False,
        )
        features = features.flatten(2).transpose(1, 2)
        return self.adapter(features.to(dtype=next(self.adapter.parameters()).dtype))
