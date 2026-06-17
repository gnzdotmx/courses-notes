"""VGG16 feature extraction and caching."""

from __future__ import annotations

import os
from pickle import dump, load
from typing import Any

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from captioning.paths import VGG_PATH, features_path_for_dataset


def build_vgg_extractor() -> Model:
    if VGG_PATH.is_file():
        return load_model(VGG_PATH, compile=False)

    base = VGG16()
    base.layers.pop()
    model = Model(inputs=base.inputs, outputs=base.layers[-1].output)
    model.save(VGG_PATH)
    print(f"VGG16 feature model saved to {VGG_PATH}")
    return model


def extract_features(images_dir: str, vgg_model: Model) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in sorted(os.listdir(images_dir)):
        if name.startswith("."):
            continue
        lower = name.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        filepath = os.path.join(images_dir, name)
        if not os.path.isfile(filepath):
            continue
        image = load_img(filepath, target_size=(224, 224))
        array_image = img_to_array(image)
        batch = array_image.reshape((1, *array_image.shape))
        batch = preprocess_input(batch)
        feature = vgg_model.predict(batch, verbose=0)
        image_id = name.rsplit(".", 1)[0]
        features[image_id] = feature
    return features


def _load_dataset_features(dataset_id: str) -> dict[str, Any]:
    path = features_path_for_dataset(dataset_id)
    with path.open("rb") as handle:
        raw = load(handle)
    prefixed: dict[str, Any] = {}
    prefix = f"{dataset_id}:"
    for key, value in raw.items():
        if str(key).startswith(prefix):
            prefixed[str(key)] = value
        else:
            prefixed[f"{dataset_id}:{key}"] = value
    return prefixed


def _save_dataset_features(dataset_id: str, features: dict[str, Any]) -> None:
    path = features_path_for_dataset(dataset_id)
    with path.open("wb") as handle:
        dump(features, handle)
    print(f"Saved {len(features)} features to {path}")


def load_or_extract_features(
    image_roots: list[tuple[str, str]],
    *,
    datasets_to_extract: set[str],
) -> dict[str, Any]:
    """
    Merge per-dataset feature caches; extract only datasets listed in datasets_to_extract.

    Each dataset's features are stored in features_<dataset_id>.dump and are not removed
    when you train on a different dataset later.
    """
    merged: dict[str, Any] = {}
    vgg_model = None

    for dataset_id, images_dir in image_roots:
        cache_path = features_path_for_dataset(dataset_id)
        must_extract = dataset_id in datasets_to_extract or not cache_path.is_file()

        if not must_extract:
            partial = _load_dataset_features(dataset_id)
            merged.update(partial)
            print(f"Loaded {len(partial)} features from {cache_path.name}")
            continue

        if vgg_model is None:
            vgg_model = build_vgg_extractor()

        print(f"Extracting features for {dataset_id} from {images_dir} ...")
        raw = extract_features(images_dir, vgg_model)
        prefixed = {f"{dataset_id}:{image_id}": vector for image_id, vector in raw.items()}
        _save_dataset_features(dataset_id, prefixed)
        merged.update(prefixed)
        print(f"  -> {len(prefixed)} images")

    return merged


def feature_for_split(
    features: dict[str, Any],
    image_id: str,
) -> Any:
    value = features.get(image_id)
    if value is None:
        return None
    return value[0] if hasattr(value, "__getitem__") else value
