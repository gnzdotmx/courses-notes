"""Caption model definition and training helpers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from captioning.config import TrainingConfig
from captioning.paths import RunArtifacts
from captioning.text_processing import create_sequences

if TYPE_CHECKING:
    from captioning.timing import TrainingTimer


def define_model(config: TrainingConfig, vocab_size: int, max_length: int) -> Model:
    inputs1 = Input(shape=(config.image_feature_dim,))
    fe1 = Dropout(config.dropout)(inputs1)
    fe2 = Dense(config.dense_units, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, config.embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(config.dropout)(se1)
    se3 = LSTM(config.lstm_units)(se2)

    merged = Add()([fe2, se3])
    decoder = Dense(config.dense_units, activation="relu")(merged)
    outputs = Dense(vocab_size, activation="softmax")(decoder)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=config.learning_rate),
    )
    return model


def model_vocab_matches(model: Model, expected_vocab_size: int) -> bool:
    for layer in model.layers:
        if hasattr(layer, "units") and layer.units == expected_vocab_size:
            return True
    return False


def data_generator(
    descriptions: dict[str, list[str]],
    images: dict[str, object],
    tokenizer,
    maxlen: int,
    vocab_size: int,
):
    while True:
        for key, description in descriptions.items():
            if key not in images or images[key] is None:
                continue
            photo = images[key]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, maxlen, description, photo, vocab_size
            )
            yield (in_img, in_seq), out_word


def train_model(
    model: Model,
    train_description: dict[str, list[str]],
    train_features: dict[str, object],
    tokenizer,
    maxlen: int,
    vocab_size: int,
    config: TrainingConfig,
    artifacts: RunArtifacts,
    timer: TrainingTimer | None = None,
) -> Model:
    steps = config.steps_per_epoch or len(train_description)
    output_signature = (
        (
            tf.TensorSpec(shape=(None, config.image_feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, maxlen), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32),
    )

    print(f"Training for {config.epochs} epoch(s), {steps} steps/epoch ...")
    if timer is not None:
        print("  ETA legend updates after each epoch (avg of finished epochs × remaining).")
    for epoch in range(config.epochs):
        if timer is not None:
            timer.print_epoch_start(epoch, total_epochs=config.epochs)
        epoch_start = time.perf_counter()
        gen = lambda: data_generator(  # noqa: E731
            train_description, train_features, tokenizer, maxlen, vocab_size
        )
        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        model.fit(
            dataset,
            steps_per_epoch=steps,
            epochs=1,
            verbose=config.verbose,
        )
        model.save(artifacts.model_path)
        epoch_elapsed = time.perf_counter() - epoch_start
        if timer is not None:
            timer.record_epoch(epoch, epoch_elapsed, total_epochs=config.epochs)
        else:
            print(f"Epoch {epoch + 1} of {config.epochs} completed")
    return model


def load_or_train_model(
    config: TrainingConfig,
    artifacts: RunArtifacts,
    vocab_size: int,
    maxlen: int,
    train_description: dict[str, list[str]],
    train_features: dict[str, object],
    tokenizer,
    *,
    should_train: bool,
    timer: TrainingTimer | None = None,
) -> Model:
    model_path = artifacts.model_path
    if model_path.is_file() and not should_train:
        candidate = load_model(model_path)
        if model_vocab_matches(candidate, vocab_size):
            print(f"Loaded existing model from {model_path}")
            return candidate
        raise ValueError(
            f"{model_path.name} vocabulary does not match tokenizer ({vocab_size}). "
            "Retrain with: python main.py train --force-retrain -y"
        )

    if not should_train:
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            f"Run: python main.py train --datasets <id>"
        )

    print(f"Training caption model (prefix: {artifacts.prefix})...")
    model = define_model(config, vocab_size, maxlen)
    model = train_model(
        model,
        train_description,
        train_features,
        tokenizer,
        maxlen,
        vocab_size,
        config,
        artifacts,
        timer=timer,
    )
    print(f"Model saved to {model_path}")
    return model
