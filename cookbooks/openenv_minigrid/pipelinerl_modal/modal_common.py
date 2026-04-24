from __future__ import annotations

from pathlib import Path
from typing import Iterable

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/app"
HF_CACHE_DIR = "/root/.cache/huggingface"
DATA_DIR = "/vol/data"
CKPT_DIR = "/vol/checkpoints"

HF_CACHE_VOLUME = modal.Volume.from_name("nanolong-hf-cache", create_if_missing=True)
DATA_VOLUME = modal.Volume.from_name("nanolong-data", create_if_missing=True)
CKPT_VOLUME = modal.Volume.from_name("nanolong-checkpoints", create_if_missing=True)

GPU_1 = "B200:1"
GPU_2 = "B200:2"
GPU_4 = "B200:4"
GPU_8 = "B200:8"

TRAIN_PACKAGES = [
    # Keep training-side dependencies separate from the vLLM serving image. Qwen3.5
    # currently wants very new Transformers/vLLM stacks, while TRL's documented vLLM
    # integration supports a narrower vLLM range. Splitting train and serve images keeps
    # that seam manageable.
    "torch>=2.6",
    "torchvision>=0.21.0",
    "accelerate>=1.10.0",
    "datasets>=4.1.0",
    "peft>=0.15.0",
    "trl>=0.28.0",
    "wandb>=0.19.0",
    "httpx>=0.28.1",
    "pydantic>=2.10.0",
    "sentencepiece>=0.2.0",
    "safetensors>=0.5.0",
    "einops>=0.8.0",
    "packaging>=24.0",
    "transformers @ git+https://github.com/huggingface/transformers.git@main",
]

TASK_APP_PACKAGES = [
    "fastapi>=0.115.12",
    "uvicorn>=0.34.2",
    "httpx>=0.28.1",
    "pydantic>=2.10.0",
]


def _de_dupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out



def training_image(*extra_packages: str, include_flash_attn: bool = False) -> modal.Image:
    """CUDA 12.8 base image for B200 training on Modal.

    We do not force FlashAttention here because Blackwell kernel support can lag behind
    the latest PyTorch/Transformers wheels. If you have a known-good build, set
    include_flash_attn=True or add it yourself.
    """

    packages = _de_dupe([*TRAIN_PACKAGES, *extra_packages])
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "git",
            "curl",
            "build-essential",
            "ninja-build",
            "libgl1",
            "libglib2.0-0",
        )
        .pip_install(*packages)
    )
    if include_flash_attn:
        image = image.pip_install("flash-attn>=2.7.4.post1", extra_options="--no-build-isolation")
    return image.add_local_dir(PROJECT_ROOT.as_posix(), remote_path=REMOTE_ROOT)



def task_app_image(*extra_packages: str) -> modal.Image:
    packages = _de_dupe([*TASK_APP_PACKAGES, *extra_packages])
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*packages)
        .add_local_dir(PROJECT_ROOT.as_posix(), remote_path=REMOTE_ROOT)
    )



def volume_mounts(extra: dict[str, modal.Volume] | None = None) -> dict[str, modal.Volume]:
    mounts = {
        HF_CACHE_DIR: HF_CACHE_VOLUME,
        DATA_DIR: DATA_VOLUME,
        CKPT_DIR: CKPT_VOLUME,
    }
    if extra:
        mounts.update(extra)
    return mounts
