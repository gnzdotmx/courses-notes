"""
Generator, discriminator, and combined GAN (Keras 3–friendly).

Mental model (same as course notes)
  G: z ~ N(0,1)  →  fake QR image
  D: image       →  P(real)
  GAN: freeze D, train G so D(G(z)) → 1  (fool the discriminator)
"""

from __future__ import annotations

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import config


def build_generator(
    latent_dim: int = config.LATENT_DIM,
    img_size: int = config.IMG_SIZE[0],
) -> Sequential:
    """
    Map latent noise to a synthetic RGB QR image.

    Inputs (at call / predict time):
      z — shape (batch, latent_dim), typically N(0, 1).
    Outputs:
      Fake images — shape (batch, img_size, img_size, 3), tanh ∈ [-1, 1].
    Why this shape:
      Dense → Reshape(25,25,256) then two stride-2 Conv2DTranspose:
        25 → 50 → 100, matching the lab's 100×100 QR tensors.
      LeakyReLU keeps gradients alive for negative pre-activations (better than
      plain ReLU for G). BatchNorm stabilizes the upsampling stack.
      Final tanh aligns with [-1, 1] normalization of real images.
    Why Input(...) first:
      Keras 3 Sequential models are unbuilt until called; an explicit Input
      defines the graph so G can be wired into the combined GAN.
    """
    assert img_size == 100, "Lab architecture is fixed at 100×100 output"

    model = Sequential(name="generator")
    model.add(Input(shape=(latent_dim,)))
    # Project noise into a spatial feature volume the transpose-convs can upsample.
    model.add(Dense(256 * 25 * 25))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((25, 25, 256)))

    # Upsample 25×25 → 50×50
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())

    # Upsample 50×50 → 100×100
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())

    # RGB image in [-1, 1]
    model.add(Conv2DTranspose(3, (4, 4), activation="tanh", padding="same"))
    return model


def build_discriminator(
    img_size: int = config.IMG_SIZE[0],
    channels: int = config.IMG_CHANNELS,
) -> Sequential:
    """
    Binary classifier: real QR vs generator fake.

    Inputs:
      Image batch — shape (batch, img_size, img_size, channels), values [-1, 1].
    Outputs:
      Scores — shape (batch, 1), sigmoid ∈ (0, 1) ≈ P(real).
    Why:
      Conv stack learns local QR structure (modules / edges). Stride-2 layers
      downsample so Flatten → Dense(1) is cheap. Dropout reduces D overfitting
      so G still gets a useful learning signal (if D is perfect too early, G stalls).
    """
    model = Sequential(name="discriminator")
    model.add(Input(shape=(img_size, img_size, channels)))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model


def compile_discriminator(discriminator: Sequential) -> Sequential:
    """
    Attach binary cross-entropy + Adam for standalone D updates.

    Inputs:
      Uncompiled discriminator Sequential.
    Outputs:
      Same model, compiled (mutates in place; returns for chaining).
    Why binary_crossentropy:
      Labels are 1 (real) / 0 (fake); BCE is the standard D objective.
    """
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return discriminator


def build_gan(
    generator: Sequential,
    discriminator: Sequential,
    latent_dim: int = config.LATENT_DIM,
    lr: float = config.GAN_LR,
    beta_1: float = config.GAN_BETA_1,
) -> Model:
    """
    Wire G → D into one trainable graph used only to update G.

    Inputs:
      generator, discriminator — already built; D should already be compiled
      for its own train_on_batch calls.
      latent_dim — noise vector size (must match G).
    Outputs:
      Model(noise → D(G(noise))), compiled with BCE + Adam(lr, beta_1).
    Why freeze D:
      When we call gan.train_on_batch, we want gradients to flow into G only.
      D stays fixed during that step so G learns to increase D(G(z)).
      D is still updated separately in the training loop on real/fake batches.
    Why a fresh Input (not generator.input):
      Keras 3 may leave Sequential.input undefined until the model is called;
      building the GAN functionally always defines the graph.
    """
    discriminator.trainable = False

    z = Input(shape=(latent_dim,), name="gan_noise")
    fake_img = generator(z)
    validity = discriminator(fake_img)
    gan = Model(z, validity, name="gan")
    gan.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr, beta_1=beta_1),
    )
    return gan
