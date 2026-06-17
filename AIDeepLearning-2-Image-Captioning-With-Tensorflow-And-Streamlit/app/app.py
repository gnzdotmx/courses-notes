"""Streamlit UI for image captioning (upload image → generated caption)."""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pickle import load
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

from captioning.inference import generate_caption as generate_caption_core
from captioning.paths import (
    RunArtifacts,
    display_name_for_prefix,
    list_model_prefixes,
)

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_vgg():
    return load_model(APP_DIR / "vgg16_model.h5", compile=False)


@st.cache_resource
def load_caption_bundle(model_prefix: str):
    """Load model + tokenizer + maxlen for a trained prefix."""
    artifacts = RunArtifacts.for_prefix(model_prefix)
    if not artifacts.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {artifacts.model_path}")
    if not artifacts.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {artifacts.tokenizer_path}")
    if not artifacts.maxlen_path.is_file():
        raise FileNotFoundError(f"Maxlen not found: {artifacts.maxlen_path}")

    model = load_model(artifacts.model_path, compile=False)
    with artifacts.tokenizer_path.open("rb") as handle:
        tokenizer = load(handle)
    with artifacts.maxlen_path.open("rb") as handle:
        max_length = load(handle)
    return model, tokenizer, max_length, artifacts


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def image_to_features(vgg_model, pil_image):
    """Resize, preprocess, and run VGG16 feature extractor (same steps as main.py)."""
    pil_image = pil_image.resize((224, 224))
    arr = tf.keras.utils.img_to_array(pil_image)
    batch = np.expand_dims(arr, axis=0)
    batch = preprocess_input(batch)
    return vgg_model.predict(batch, verbose=0)


def generate_caption(caption_model, vgg_model, tokenizer, pil_image, max_length):
    """Run VGG16 + caption model; return text without <start>/<end> tokens."""
    photo = image_to_features(vgg_model, pil_image)
    raw = generate_caption_core(caption_model, tokenizer, photo, max_length)
    words = raw.split()
    if words and words[0] == "<start>":
        words = words[1:]
    if words and words[-1] in ("<end>", "end"):
        words = words[:-1]
    return " ".join(words)


def main():
    prefixes = list_model_prefixes()
    if not prefixes:
        st.error(
            "No trained models found. Train first, e.g.\n\n"
            "`python main.py train --datasets flickr8k --epochs 10`"
        )
        return

    st.title("Image Captioning")
    default_prefix = prefixes[-1]
    model_prefix = st.sidebar.selectbox(
        "Trained model",
        prefixes,
        index=prefixes.index(default_prefix),
        format_func=display_name_for_prefix,
    )

    try:
        caption_model, tokenizer, max_length, artifacts = load_caption_bundle(model_prefix)
        vgg_model = load_vgg()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.sidebar.caption(
        f"Files: `{artifacts.model_path.name}`, `{artifacts.tokenizer_path.name}`"
    )

    menu = ["Image", "About"]
    choice = st.sidebar.selectbox("Main", menu)

    if choice == "Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            st.write(
                {
                    "filename": image_file.name,
                    "filetype": image_file.type,
                    "filesize": image_file.size,
                }
            )
            pil_image = load_image(image_file)
            st.image(pil_image, width=250)

            with st.spinner("Generating caption..."):
                caption = generate_caption(
                    caption_model, vgg_model, tokenizer, pil_image, max_length
                )
            st.write(caption)
    else:
        st.subheader("About")
        st.write(
            "Upload an image to generate a caption. Each training run is saved under a "
            "prefix (e.g. `flickr8k`, `flickr30k`, `flickr8k_flickr30k`). "
            "Per-dataset feature caches (`features_<dataset>.dump`) are kept when you "
            "train on another dataset."
        )


if __name__ == "__main__":
    main()
