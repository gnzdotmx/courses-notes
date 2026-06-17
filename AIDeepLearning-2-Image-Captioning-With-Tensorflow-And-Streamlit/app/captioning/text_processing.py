"""Caption cleaning, tokenization, and sequence building."""

from __future__ import annotations

import re
from pickle import dump, load
from typing import Any

from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from captioning.config import TrainingConfig
from captioning.paths import RunArtifacts

CAPTION_CLEAN_RE = re.compile(r"\w*[^a-z\s<>]+\w*")
MIN_VOCAB_SIZE = 500

# Post-padding: required for Embedding(mask_zero=True) + cuDNN LSTM on GPU/Metal.
CAPTION_PADDING = "post"


def pad_caption_sequence(sequence: list[int], maxlen: int):
    """Right-pad a token sequence to fixed length (zeros on the right)."""
    return pad_sequences([sequence], maxlen=maxlen, padding=CAPTION_PADDING)[0]


def wrap_caption(raw: str) -> str:
    return f"<start> {raw.strip()} <end>"


def clean_caption(caption: str) -> str:
    text = caption.lower()
    return CAPTION_CLEAN_RE.sub("", text)


def clean_descriptions(description: dict[str, list[str]]) -> dict[str, list[str]]:
    cleaned: dict[str, list[str]] = {}
    for image_id, captions in description.items():
        cleaned[image_id] = [clean_caption(c) for c in captions]
    return cleaned


def fit_tokenizer(
    train_description: dict[str, list[str]],
    *,
    num_words: int | None,
) -> Tokenizer:
    captions: list[str] = []
    for caps in train_description.values():
        captions.extend(caps)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(captions)
    return tokenizer


def compute_maxlen(train_description: dict[str, list[str]]) -> int:
    lengths = [
        len(caption.split())
        for caps in train_description.values()
        for caption in caps
    ]
    if not lengths:
        raise ValueError("No captions found to compute max length.")
    return max(lengths)


def load_or_create_tokenizer(
    train_description: dict[str, list[str]],
    config: TrainingConfig,
    artifacts: RunArtifacts,
    *,
    refit: bool,
) -> Tokenizer:
    path = artifacts.tokenizer_path
    if path.is_file() and not refit:
        with path.open("rb") as handle:
            tokenizer: Tokenizer = load(handle)
        vocab_size = len(tokenizer.word_index) + 1
        if vocab_size < MIN_VOCAB_SIZE:
            print(f"Tokenizer too small ({vocab_size}); refitting.")
            refit = True
        else:
            print(f"Tokenizer loaded from {path.name}; vocabulary size: {vocab_size}")
            return tokenizer

    tokenizer = fit_tokenizer(train_description, num_words=config.num_words)
    with path.open("wb") as handle:
        dump(tokenizer, handle)
    print(f"Tokenizer saved to {path.name}; vocabulary size: {len(tokenizer.word_index) + 1}")
    return tokenizer


def load_or_create_maxlen(
    train_description: dict[str, list[str]],
    config: TrainingConfig,
    artifacts: RunArtifacts,
    *,
    refit: bool,
) -> int:
    if config.max_caption_length is not None:
        return config.max_caption_length

    path = artifacts.maxlen_path
    if path.is_file() and not refit:
        with path.open("rb") as handle:
            maxlen: int = load(handle)
        print(f"Loaded maxlen from {path.name}: {maxlen}")
        return maxlen

    maxlen = compute_maxlen(train_description)
    with path.open("wb") as handle:
        dump(maxlen, handle)
    print(f"Saved maxlen to {path.name}: {maxlen}")
    return maxlen


def create_sequences(
    tokenizer: Tokenizer,
    maxlen: int,
    description_list: list[str],
    photo: Any,
    vocab_size: int,
):
    x1, x2, y = [], [], []
    for description in description_list:
        seq = tokenizer.texts_to_sequences([description])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_caption_sequence(in_seq, maxlen)
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            x1.append(photo)
            x2.append(in_seq)
            y.append(out_seq)
    return array(x1), array(x2), array(y)
