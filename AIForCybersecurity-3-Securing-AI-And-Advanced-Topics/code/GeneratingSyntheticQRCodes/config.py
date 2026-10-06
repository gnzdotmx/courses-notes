"""
Central hyperparameters and paths for the synthetic QR-code GAN lab.

Why a config module?
  Keeps magic numbers out of model/train code so you can tune epochs, image
  size, or latent dim in one place and see how G/D capacity must stay matched.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to this package directory so `python main.py` is stable)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "qr_codes"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_DIR = PACKAGE_DIR / "gan_checkpoints"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Lab: 100 QR codes total → 80 train / 20 val (same message space, held-out files).
N_TRAIN = 80
N_VAL = 20
# QR raster size after load/resize; must match G output and D input_shape.
IMG_SIZE = (100, 100)
IMG_CHANNELS = 3

# ---------------------------------------------------------------------------
# GAN
# ---------------------------------------------------------------------------
# Dimensionality of noise z ~ N(0,1). Generator maps R^{LATENT_DIM} → image.
LATENT_DIM = 100
# Lab default is 5000; lower for a quick local smoke test (expect noisier QR).
EPOCHS = 5000
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 100
# Adam for the combined GAN (lab values): small LR + beta_1=0.5 stabilizes G/D.
GAN_LR = 0.0002
GAN_BETA_1 = 0.5
# How many synthetic QR images to write after training.
N_GENERATE = 10
