# Extract features

VGG16-based image feature extraction and per-dataset caching.

## Why extract features?

Raw images are high-dimensional. A **pretrained CNN** maps each 224×224 image to a compact vector that summarizes content. Training the caption model on these vectors is faster and more stable than training on pixels (transfer learning).

See [AI concepts: Why a CNN?](ai-concepts.md#why-a-cnn) for the theory.

## CLI

```bash
# From app/
python main.py extract-features --datasets flickr8k -y

# From repo root
make extract-features DATASETS=flickr8k YES=1
```

`extract-features` runs the feature phase only (`config.train=False`, `config.extract_features=True`). `train` also extracts when caches are missing.

## Module: `captioning/features.py`

### `build_vgg_extractor()`

- If `app/vgg16_model.h5` exists → load it.
- Else build `VGG16()`, remove the classification head (`base.layers.pop()`), save the feature model.

The saved model outputs a **1000-dimensional** vector in this course setup (`image_feature_dim=1000` in config). Standard VGG16 “no top” is often 4096-d; this project matches the course backbone: keep `image_feature_dim` aligned with actual output.

### `extract_features(images_dir, vgg_model)`

For each image file (`.jpg`, `.jpeg`, `.png`, `.webp`):

1. `load_img(..., target_size=(224, 224))`
2. `img_to_array` → batch shape `(1, 224, 224, 3)`
3. `preprocess_input`: ImageNet normalization expected by VGG16
4. `vgg_model.predict` → feature vector
5. Store in dict `image_id → vector`

### `load_or_extract_features(image_roots, datasets_to_extract)`

Merges caches across datasets:

- Loads `features_<dataset_id>.dump` when present and not forced to re-extract.
- Otherwise runs `extract_features`, prefixes keys as `{dataset_id}:{image_id}`, saves pickle.

**Important:** Feature files for dataset A are **not** deleted when you train on dataset B.

### `feature_for_split(features, image_id)`

Returns the flat vector used by the caption model (handles `(1, dim)` vs 1-D).

## Artifacts

| File | Content |
|------|---------|
| `vgg16_model.h5` | Shared VGG16 feature extractor (built once) |
| `features_flickr8k.dump` | Pickled `{flickr8k:id → vector}` |
| `features_flickr30k.dump` | Same for Flickr30k |

## Pipeline integration (`captioning/pipeline.py`)

After `load_selected_datasets`:

1. `datasets_needing_feature_extract`: ids without cache (or all if `--force-retrain`).
2. `load_or_extract_features`: timed as phase `"features"`.
3. `_build_feature_index`: filters merged features into `train_features` / `test_features` by split keys.

## When to re-extract

- First run on a dataset.
- `--force-retrain` (re-extracts features for datasets in `--datasets` after confirmation).
- Manual delete of a `features_*.dump` file or `make clean`.
