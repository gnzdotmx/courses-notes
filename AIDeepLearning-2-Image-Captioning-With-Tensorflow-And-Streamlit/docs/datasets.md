# Datasets

How images and captions are collected, parsed, cleaned, and merged for training.

## Two data types

Image captioning is **multimodal**:

| Type | Role | In code |
|------|------|---------|
| **Images** | Visual input | `images_dir` per loader; later → VGG16 feature vectors |
| **Captions** | Supervised text target | Token files → `description[image_id] = [caption, …]` |

Each caption aligns to an image via **image_id** (filename stem, e.g. `123` from `123.jpg`).

## Registry (`caption_datasets/registry.py`)

`DATASET_REGISTRY` lists every known dataset id with metadata: downloadable, trainable, size hints, Kaggle handles, and when to use each.

**Implemented loaders** (wired in `_LOADER_MODULES`):

- `flickr8k` → `caption_datasets/flickr8k.py`
- `flickr30k` → `caption_datasets/flickr30k.py`
- `custom` → `caption_datasets/custom.py`

Others (`coco`, `visual_genome`, `vizwiz`, `nocaps`, `laion_*`) are documented for reference only.

### `load_selected_datasets(config)`

For each id in `config.datasets`:

1. Verify `trainable` in registry.
2. Call the loader → `LoadedDataset`.
3. `clean_descriptions()`: lowercase, strip non-alpha tokens.
4. `prefix_descriptions()`: prefix keys as `{dataset_id}:{image_id}` so merged runs do not collide.
5. `merge_descriptions()`: combine train/test dicts across datasets.

Returns `(train_description, test_description, image_roots)` where `image_roots` is `[(dataset_id, Path), …]` for feature extraction.

## Flickr8k loader (`caption_datasets/flickr8k.py`)

1. **Download** text + image archives via `kaggle_download.download_dataset`.
2. **Locate** `Flickr8k.token.txt`, train/test image list files, and `Images/` folder.
3. **Parse** token file: each line `image.jpg caption words…` → `wrap_caption()` adds `<start>` / `<end>`.
4. **Split** with `utils.loaders.load_image_ids`: keeps only ids listed in train/test files.

Kaggle handles: `youssefaboelnasr/flickr8k-text`, `adityajn105/flickr8k`.

## Flickr30k loader (`caption_datasets/flickr30k.py`)

Same pattern as Flickr8k with multiple Kaggle mirror fallbacks (`adityajn105/flickr30k`, etc.).

## Custom loader (`caption_datasets/custom.py`)

Local paths from CLI:

- `--custom-images-dir`: folder of image files
- `--custom-captions-file`: Flickr-style lines, pipe/tab pairs, or CSV `filename,caption`
- Optional `--custom-train-list` / `--custom-test-list`

## Kaggle download helper (`caption_datasets/kaggle_download.py`)

Wraps `kagglehub.dataset_download` with caching under `~/.cache/kagglehub/`. Provides `find_file` / `find_directory` to tolerate slightly different archive layouts.

## Train/test split (`utils/loaders.py`)

```python
load_image_ids(description, filepath)
```

Reads one filename per line, strips extension to get `image_id`, copies caption lists from the full `description` dict.

## Caption cleaning (`captioning/text_processing.py`)

- `wrap_caption(raw)` → `"<start> … <end>"`
- `clean_caption`: lowercase; regex removes tokens with non-alphabetic characters (keeps `<start>`, `<end>`)
- `clean_descriptions`: applies per image

## Multiple datasets in one run

Training on `flickr8k,flickr30k` produces prefix `flickr8k_flickr30k` and prefixed image ids like `flickr8k:12345`. Feature caches stay **per dataset** (`features_flickr8k.dump`, `features_flickr30k.dump`) so later single-dataset runs reuse them.
