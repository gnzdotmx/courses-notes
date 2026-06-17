# Module reference

Quick map from files to responsibilities.

## Layout

```
app/
  main.py                 # CLI entry
  app.py                  # Streamlit UI
  captioning/             # Core ML pipeline
  caption_datasets/       # Dataset loaders
  utils/                  # Shared helpers
```

## Entry points

| File | Purpose |
|------|---------|
| `main.py` | `python main.py <command>`; default command `train` |
| `app.py` | Streamlit upload UI |
| `captioning/cli.py` | Argparse, subcommands, Streamlit subprocess |
| `captioning/pipeline.py` | `run_pipeline`: orchestrates all training phases |

## Configuration & paths

| File | Purpose |
|------|---------|
| `captioning/config.py` | `TrainingConfig` dataclass + validation |
| `captioning/paths.py` | `APP_DIR`, `RunArtifacts`, feature paths, model registry |
| `captioning/artifacts.py` | Confirm/delete caches; `prepare_training_artifacts` |
| `captioning/cleanup.py` | Interactive `clean` / `release-memory` command |

## Data

| File | Purpose |
|------|---------|
| `caption_datasets/registry.py` | Dataset catalog, `load_selected_datasets` |
| `caption_datasets/flickr8k.py` | Flickr8k Kaggle loader |
| `caption_datasets/flickr30k.py` | Flickr30k Kaggle loader |
| `caption_datasets/custom.py` | Local images + caption file |
| `caption_datasets/kaggle_download.py` | `kagglehub` wrapper, file discovery |
| `caption_datasets/base.py` | `DatasetSpec`, `LoadedDataset`, merge helpers |
| `utils/loaders.py` | `load_image_ids` for train/test lists |

## Features & vision

| File | Purpose |
|------|---------|
| `captioning/features.py` | VGG16 build/load, extract, per-dataset cache |

## Text & sequences

| File | Purpose |
|------|---------|
| `captioning/text_processing.py` | Clean, tokenize, pad, `create_sequences` |

## Model & training

| File | Purpose |
|------|---------|
| `captioning/model_builder.py` | `define_model`, `data_generator`, `train_model` |
| `captioning/timing.py` | `TrainingTimer`, epoch ETA, stats JSON |

## Inference & evaluation

| File | Purpose |
|------|---------|
| `captioning/inference.py` | `generate_caption`, `evaluate_model` (BLEU) |

## CLI command → code path

| Command | Main modules |
|---------|----------------|
| `list-datasets` | `registry.list_datasets` |
| `train` | `pipeline.run_pipeline` (train=True) |
| `extract-features` | `pipeline.run_pipeline` (train=False, extract only) |
| `evaluate` | `pipeline.run_pipeline` (evaluate=True) |
| `streamlit` | `cli.run_streamlit_app` → `app.py` |
| `clean` | `cleanup.run_cleanup` |

## Key variables (runtime)

| Name | Type | Meaning |
|------|------|---------|
| `description` | `dict[str, list[str]]` | All image_id → captions |
| `train_description` / `test_description` | same | Split subsets |
| `train_features` / `test_features` | `dict[str, vector]` | VGG16 features per id |
| `tokenizer` | Keras `Tokenizer` | `word_index`, `texts_to_sequences` |
| `vocab_size` | `int` | `len(word_index) + 1` |
| `maxlen` | `int` | Padded sequence length |

## Related docs

- [Setup](setup.md): environment
- [Datasets](datasets.md): loaders
- [Extract features](extract-features.md): VGG16
- [Training](training.md): fit loop
- [Evaluation](evaluation.md): BLEU
- [Streamlit](streamlit.md): UI
- [AI concepts](ai-concepts.md): CNN, LSTM theory
