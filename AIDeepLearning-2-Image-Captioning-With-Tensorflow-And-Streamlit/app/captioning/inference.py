"""Caption generation and evaluation."""

from __future__ import annotations

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from captioning.text_processing import pad_caption_sequence


def word_for_id(integer: int, tokenizer) -> str | None:
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo, max_length: int) -> str:
    photo = np.asarray(photo)
    if photo.ndim == 1:
        photo = photo.reshape(1, -1)

    end_token_id = tokenizer.word_index.get("<end>")
    in_text = "<start>"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_caption_sequence(sequence, max_length)
        sequence = sequence.reshape(1, -1)
        y_hat = int(np.argmax(model.predict([photo, sequence], verbose=0)))
        if end_token_id is not None and y_hat == end_token_id:
            break
        word = word_for_id(y_hat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word in ("<end>", "end"):
            break
    return in_text


def evaluate_model(model, descriptions, images, tokenizer, max_length: int) -> None:
    actual, predicted = [], []
    for key, description in descriptions.items():
        if key not in images or images[key] is None:
            continue
        yhat = generate_caption(model, tokenizer, images[key], max_length)
        references = [d.split() for d in description]
        actual.append(references)
        predicted.append(yhat.split())

    if not actual:
        print("No evaluation samples available.")
        return

    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print("BLEU-4: %f" % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
