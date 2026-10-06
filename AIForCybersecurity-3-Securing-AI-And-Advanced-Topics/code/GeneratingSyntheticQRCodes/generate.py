"""
Sample the trained generator and write PNG files for visual inspection.

Cyber angle (course): synthetic QR-looking images are dual-use — data
augmentation for detectors, or a way to probe how brittle vision models are.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import config
from qr_dataset import ensure_directories


def generate_and_save_qr_codes(
    generator: Sequential,
    n_samples: int = config.N_GENERATE,
    latent_dim: int = config.LATENT_DIM,
    out_dir: Path = config.GENERATED_DIR,
) -> Path:
    """
    Draw noise → G → rescale → save PNGs.

    Inputs:
      generator  — trained (or checkpoint-loaded) Sequential G
      n_samples  — how many fakes to write
      latent_dim — must match G's Dense input_dim
      out_dir    — destination folder
    Outputs:
      out_dir Path after writing qr_code_0.png … qr_code_{n-1}.png
    Why rescale 0.5 * x + 0.5:
      G outputs tanh ∈ [-1, 1]; displays and PNG writers expect [0, 1] or [0, 255].
    """
    ensure_directories(out_dir)

    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated = generator.predict(noise, verbose=0)
    generated = 0.5 * generated + 0.5  # [-1, 1] → [0, 1]

    for i in range(n_samples):
        plt.imshow(np.clip(generated[i], 0.0, 1.0))
        plt.axis("off")
        plt.savefig(out_dir / f"qr_code_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    return out_dir


def display_image(path: Path, title: str | None = None) -> None:
    """
    Load one PNG and show it with matplotlib (interactive backends only).

    Inputs:
      path  — e.g. qr_codes/train/qr_code_0.png or .../generated/...
      title — optional plot title
    Outputs:
      None (opens a figure window / notebook inline display).
    Why:
      Side-by-side visual check: real QR vs G sample. Early epochs often look
      noisy — expected until G/D balance improves.
    """
    img = load_img(path, target_size=config.IMG_SIZE)
    arr = img_to_array(img) / 255.0
    plt.imshow(arr)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
