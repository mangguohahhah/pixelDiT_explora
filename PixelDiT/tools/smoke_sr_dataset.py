from __future__ import annotations

import sys

sys.path.insert(0, "t2i")

from diffusion.data.datasets.paired_sr_dataset import PairedSRDataset


dataset = PairedSRDataset(
    "/mnt/data/home/sqkang/datasets/data_4k/test4k/HR/X4",
    "/mnt/data/home/sqkang/datasets/data_4k/test4k/LR/X4",
    resolution=1024,
)
sample = dataset[0]
print(f"pairs={len(dataset)}")
print(f"hr={tuple(sample[0].shape)}, range=({sample[0].min():.3f}, {sample[0].max():.3f})")
print(f"lr={tuple(sample[8].shape)}, range=({sample[8].min():.3f}, {sample[8].max():.3f})")
