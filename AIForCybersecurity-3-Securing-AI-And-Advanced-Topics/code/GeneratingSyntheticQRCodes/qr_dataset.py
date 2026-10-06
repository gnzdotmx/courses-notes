"""
Build a small QR-code dataset and turn PNG files into float tensors for the GAN.

Pipeline role
  Real QR images are the "p_data" samples the discriminator learns from.
  Without a clean train set (and a held-out val set for sanity checks), G has
  nothing meaningful to imitate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import qrcode
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import config


def ensure_directories(*dirs: Path) -> None:
    """
    Create output folders if missing.

    Inputs:
      dirs — one or more Path objects (train / val / generated / checkpoints).
    Outputs:
      None (side effect: directories exist on disk).
    Why:
      Later steps assume these paths exist; exist_ok avoids crashes on re-runs.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_qr_code(data: str, filename: Path) -> None:
    """
    Encode a string into a QR PNG.

    Inputs:
      data     — payload embedded in the code (lab uses "Message {i}").
      filename — destination path ending in .png.
    Outputs:
      None (writes an image file).
    Why:
      Gives the GAN a reproducible "real" distribution of QR patterns.
      version=1 keeps modules few; box_size/border set visual scale before resize.
    """
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)


def build_qr_dataset(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    n_train: int = config.N_TRAIN,
    n_val: int = config.N_VAL,
) -> Tuple[int, int]:
    """
    Materialize train/val QR PNGs on disk.

    Inputs:
      train_dir, val_dir — folders for the split.
      n_train, n_val     — counts (lab: 80 + 20 = 100).
    Outputs:
      (n_train_files, n_val_files) after writing.
    Why:
      Split mirrors supervised practice: train updates weights; val is a
      held-out check that files exist / shapes look right (not used in the
      lab's adversarial loop, but useful for inspection).
    """
    ensure_directories(train_dir, val_dir)

    for i in range(n_train):
        generate_qr_code(f"Message {i}", train_dir / f"qr_code_{i}.png")

    # Val messages continue the index so payloads do not collide with train.
    for i in range(n_val):
        generate_qr_code(f"Message {i + n_train}", val_dir / f"qr_code_{i}.png")

    return len(list(train_dir.glob("*.png"))), len(list(val_dir.glob("*.png")))


def load_images(
    image_dir: Path,
    target_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """
    Load every image in a directory into one float32 array (not yet GAN-scaled).

    Inputs:
      image_dir   — folder of PNGs.
      target_size — (H, W) resize so all samples match G/D tensor shapes.
    Outputs:
      ndarray of shape (N, H, W, C) with pixel values in ~[0, 255].
    Why:
      Keras models need fixed spatial size; resizing QR PNGs to 100×100 matches
      the lab generator's final Conv2DTranspose output.
    """
    images = []
    for filename in sorted(os.listdir(image_dir)):
        path = image_dir / filename
        if not path.is_file():
            continue
        img = load_img(path, target_size=target_size)
        images.append(img_to_array(img))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return np.array(images, dtype=np.float32)


def normalize_for_gan(images: np.ndarray) -> np.ndarray:
    """
    Map pixels from [0, 255] → [-1, 1].

    Inputs:
      images — array from load_images (approx. uint8 range as float).
    Outputs:
      Same shape, values in [-1, 1].
    Why:
      Generator ends with tanh (range [-1, 1]). Matching scales keeps D's
      real/fake comparison numerically consistent and training more stable.
    """
    return (images - 127.5) / 127.5


def load_train_val(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end: load PNGs and normalize for GAN training.

    Inputs:
      train_dir, val_dir — image folders.
    Outputs:
      (train_images, val_images), each shape (N, 100, 100, 3), range [-1, 1].
    """
    train = normalize_for_gan(load_images(train_dir))
    val = normalize_for_gan(load_images(val_dir))
    return train, val
