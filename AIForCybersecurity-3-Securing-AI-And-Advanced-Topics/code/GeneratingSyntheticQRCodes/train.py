"""
Alternating GAN training loop (discriminator step, then generator step).

Why GradientTape (not notebook train_on_batch)?
  Keras 3 / TF 2.20 often breaks the old pattern of compile() + toggle
  discriminator.trainable + train_on_batch on a combined Model. Explicit
  tapes match the same D-then-G logic and work on current TensorFlow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

import config
from qr_dataset import ensure_directories


def train_gan(
    generator: Sequential,
    discriminator: Sequential,
    train_images: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    latent_dim: int = config.LATENT_DIM,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    checkpoint_interval: int = config.CHECKPOINT_INTERVAL,
    gan_lr: float = config.GAN_LR,
    gan_beta_1: float = config.GAN_BETA_1,
) -> None:
    """
    Run the lab training loop (D update, then G update each epoch).

    Inputs:
      generator, discriminator — models from models.py
      train_images — real QR tensors, shape (N, H, W, C), range [-1, 1]
      epochs, batch_size, latent_dim — training knobs
      checkpoint_dir / interval — where / how often to save .h5 weights
      gan_lr, gan_beta_1 — Adam settings for G (lab: 2e-4, 0.5)
    Outputs:
      None (side effects: prints progress; writes generator/discriminator
      checkpoints). In-memory weights are updated on the passed models.
    Why this order each epoch:
      1) Sample real minibatch + G(noise) fakes
      2) Train D with labels 1 / 0  → better detector
      3) Train G so D(G(z)) → 1     → fool D (D weights not updated)
    """
    ensure_directories(checkpoint_dir)
    n_real = train_images.shape[0]
    bce = tf.keras.losses.BinaryCrossentropy()
    # Separate optimizers: D uses default Adam; G uses lab Adam(2e-4, β1=0.5).
    d_optimizer = tf.keras.optimizers.Adam()
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=gan_lr, beta_1=gan_beta_1)

    for epoch in range(epochs):
        idx = np.random.randint(0, n_real, batch_size)
        real_images = tf.convert_to_tensor(train_images[idx], dtype=tf.float32)
        noise = tf.random.normal((batch_size, latent_dim))

        # --- Discriminator: learn "real=1, fake=0" ---
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            pred_real = discriminator(real_images, training=True)
            pred_fake = discriminator(fake_images, training=True)
            d_loss_real = bce(tf.ones_like(pred_real), pred_real)
            d_loss_fake = bce(tf.zeros_like(pred_fake), pred_fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            # Rough accuracy for the same printout shape as the notebook.
            d_acc = 0.5 * (
                tf.reduce_mean(tf.cast(pred_real > 0.5, tf.float32))
                + tf.reduce_mean(tf.cast(pred_fake < 0.5, tf.float32))
            )
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        # --- Generator: want D(G(z)) ≈ 1; only G variables get gradients ---
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            pred_fake = discriminator(fake_images, training=False)
            g_loss = bce(tf.ones_like(pred_fake), pred_fake)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(
                f"{epoch} [D loss: {float(d_loss):.4f} | D acc: {float(d_acc):.4f}] "
                f"[G loss: {float(g_loss):.4f}]"
            )

        if epoch % checkpoint_interval == 0:
            generator.save(checkpoint_dir / f"generator_epoch_{epoch}.h5")
            discriminator.save(checkpoint_dir / f"discriminator_epoch_{epoch}.h5")


def latest_generator_checkpoint(
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
) -> Optional[Path]:
    """
    Find the highest-epoch generator_*.h5 if any exist.

    Inputs:
      checkpoint_dir — folder written by train_gan.
    Outputs:
      Path to a generator checkpoint, or None.
    Why:
      Lets main.py skip a long train and only sample from a saved G.
    """
    checkpoints = sorted(checkpoint_dir.glob("generator_epoch_*.h5"))
    return checkpoints[-1] if checkpoints else None
