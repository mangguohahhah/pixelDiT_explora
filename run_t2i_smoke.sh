#!/bin/bash
set -e
cd '/mnt/data/home/sqkang/projects_yj/oppo学习/pixelDiT_explora'
export PATH=/mnt/data/home/sqkang/miniforge3/envs/pixeldit_cvpr2026/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
curl -fL -C - --connect-timeout 20 --retry 10 --retry-delay 5 https://hf-mirror.com/nvidia/PixelDiT-1300M-1024px/resolve/main/pixeldit_t2i_v1.pth -o weights/pixeldit_t2i_v1.pth
python -c "from huggingface_hub import snapshot_download; snapshot_download('Efficient-Large-Model/gemma-2-2b-it', allow_patterns=['*.json','*.safetensors','*.model'], max_workers=2)"
printf '%s\n' 'A fluffy white cat sitting by a sunny window, detailed photography, warm natural light.' > t2i_smoke_prompts.txt
cd PixelDiT/t2i
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml --model_path ../../weights/pixeldit_t2i_v1.pth --txt_file ../../t2i_smoke_prompts.txt --custom_height 1024 --custom_width 1024 --bs 1 --step 25 --cfg_scale 2.75 --seed 2025 --work_dir ../outputs/t2i_smoke
echo 'SUCCESS: text-to-image inference completed'
