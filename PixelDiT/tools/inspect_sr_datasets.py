from __future__ import annotations

from pathlib import Path

from PIL import Image


CANDIDATES = {
    "DIV2K S3Diff": (
        "/mnt/data/home/sqkang/datasets/DIV2K/S3Diff_test/HR",
        "/mnt/data/home/sqkang/datasets/DIV2K/S3Diff_test/LR",
    ),
    "DIV2K official": (
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/DIV2K_V2_val/gt",
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/DIV2K_V2_val/lq",
    ),
    "4klsdb eval": (
        "/mnt/data/home/sqkang/datasets/4klsdb_subset/eval_4k_x4/gt",
        "/mnt/data/home/sqkang/datasets/4klsdb_subset/eval_4k_x4/lq",
    ),
    "DRealSR crop128": (
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_HR",
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_LR",
    ),
    "RealSR crop128": (
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR",
        "/mnt/data/home/sqkang/datasets/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR",
    ),
}

exts = {".png", ".jpg", ".jpeg", ".webp"}
for name, (hr_dir, lr_dir) in CANDIDATES.items():
    hr = Path(hr_dir)
    lr = Path(lr_dir)
    if not hr.is_dir() or not lr.is_dir():
        print(f"{name}: missing")
        continue
    hr_files = {p.name: p for p in hr.iterdir() if p.suffix.lower() in exts}
    lr_files = {p.name: p for p in lr.iterdir() if p.suffix.lower() in exts}
    names = sorted(hr_files.keys() & lr_files.keys())
    if not names:
        print(f"{name}: HR={len(hr_files)}, LR={len(lr_files)}, same-named=0")
        continue
    sample = names[0]
    with Image.open(hr_files[sample]) as image:
        hr_size = image.size
    with Image.open(lr_files[sample]) as image:
        lr_size = image.size
    print(f"{name}: pairs={len(names)}, example={sample}, HR={hr_size}, LR={lr_size}")
