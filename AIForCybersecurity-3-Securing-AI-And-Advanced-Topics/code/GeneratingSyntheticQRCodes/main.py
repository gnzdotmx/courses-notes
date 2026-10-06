"""
Entry point: run the Mod04 synthetic QR-code GAN lab as a script.

Lab flow (mirrors the Coursera notebook)
  1. Create folders + generate 80 train / 20 val QR PNGs
  2. Load & normalize images to [-1, 1]
  3. Build G, D, and combined GAN
  4. Alternate D / G training (optional; can load a checkpoint instead)
  5. Sample G and save synthetic QR images under qr_codes/generated/

Usage (from this directory):
  python -m pip install -r requirements.txt
  python main.py                      # full pipeline, config.EPOCHS
  python main.py --epochs 200         # shorter smoke run
  python main.py --data-only          # only write real QR PNGs
  python main.py --skip-train         # load latest G checkpoint, then sample
  python main.py --n-generate 5

Why split into modules?
  Each file owns one stage (data / models / train / sample) so you can study
  or swap one piece without scrolling a single notebook cell dump.
"""

from __future__ import annotations

import argparse
import sys

from tensorflow.keras.models import load_model

import config
from generate import generate_and_save_qr_codes
from models import (
    build_discriminator,
    build_gan,
    build_generator,
    compile_discriminator,
)
from qr_dataset import build_qr_dataset, ensure_directories, load_train_val
from train import latest_generator_checkpoint, train_gan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    CLI knobs for local experiments without editing source.

    Inputs:
      argv — optional argument list (defaults to sys.argv[1:]).
    Outputs:
      Namespace with epochs, batch_size, flags, n_generate.
    """
    p = argparse.ArgumentParser(
        description="Train a GAN to synthesize QR-code-like images (Mod04 lab)."
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Training epochs (lab default {config.EPOCHS}; use fewer to smoke-test).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Minibatch size for D and G updates.",
    )
    p.add_argument(
        "--n-generate",
        type=int,
        default=config.N_GENERATE,
        help="How many synthetic QR PNGs to write after training / load.",
    )
    p.add_argument(
        "--data-only",
        action="store_true",
        help="Only create train/val QR PNGs; skip model build and training.",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load the latest generator_epoch_*.h5 and sample.",
    )
    p.add_argument(
        "--summaries",
        action="store_true",
        help="Print Keras model.summary() for G, D, and GAN.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Orchestrate the full lab pipeline.

    Inputs:
      argv — CLI args (see parse_args).
    Outputs:
      Process exit code 0 on success, 1 on recoverable user error.
    """
    args = parse_args(argv)

    ensure_directories(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.GENERATED_DIR,
        config.CHECKPOINT_DIR,
    )

    # --- Stage 1: real QR dataset (p_data for D) ---
    n_train, n_val = build_qr_dataset()
    print(f"Training images generated: {n_train}")
    print(f"Validation images generated: {n_val}")

    if args.data_only:
        print(f"Data-only done. PNGs under {config.DATA_DIR}")
        return 0

    # --- Stage 2: tensors for training ---
    # Shapes should be (80, 100, 100, 3) and (20, 100, 100, 3) with lab defaults.
    train_images, val_images = load_train_val()
    print(f"Training images shape: {train_images.shape}")
    print(f"Validation images shape: {val_images.shape}")

    # --- Stage 3: models ---
    generator = build_generator()
    discriminator = compile_discriminator(build_discriminator())
    # Combined G→D graph is only for --summaries (training uses GradientTape).
    gan = build_gan(generator, discriminator) if args.summaries else None

    if args.summaries:
        generator.summary()
        discriminator.summary()
        gan.summary()

    # --- Stage 4: train or load checkpoint ---
    if args.skip_train:
        ckpt = latest_generator_checkpoint()
        if ckpt is None:
            print(
                f"No checkpoint in {config.CHECKPOINT_DIR}. "
                "Run without --skip-train first.",
                file=sys.stderr,
            )
            return 1
        print(f"Loading generator from {ckpt}")
        generator = load_model(ckpt)
    else:
        print(
            f"Training GAN for {args.epochs} epochs "
            f"(batch_size={args.batch_size})…"
        )
        train_gan(
            generator,
            discriminator,
            train_images,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        # Final weights (in addition to interval checkpoints)
        generator.save(config.CHECKPOINT_DIR / "generator_final.h5")
        discriminator.save(config.CHECKPOINT_DIR / "discriminator_final.h5")

    # --- Stage 5: sample synthetic QR-looking images ---
    out = generate_and_save_qr_codes(generator, n_samples=args.n_generate)
    print(f"Wrote {args.n_generate} synthetic images to {out}")
    print(
        "Tip: compare a real file "
        f"({config.TRAIN_DIR / 'qr_code_0.png'}) with "
        f"({config.GENERATED_DIR / 'qr_code_0.png'}). "
        "Early / short runs often look noisy until G/D balance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
