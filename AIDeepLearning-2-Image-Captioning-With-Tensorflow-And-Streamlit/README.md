# Image Captioning with TensorFlow and Streamlit

> **Course Reference**: [Image Captioning with Tensorflow and Streamlit](https://www.coursera.org/learn/image-captioning-tensorflow-streamlit) by EDUCBA on Coursera
>
> **Disclaimer**: These are personal notes and customized code implementations based on the course materials. The code examples have been adapted, extended, and documented for learning purposes. This repository is not affiliated with or endorsed by EDUCBA or Coursera.

This repository implements an end-to-end **image captioning** system: given a photograph, it generates a short natural-language description. The model follows the classic encoder–decoder pattern from the course. A **VGG16** CNN encodes each image into a fixed feature vector, and an **LSTM** decoder predicts the caption word by word, fusing image context with the partial sentence at every step.

The project is organized as a small training and deployment pipeline rather than a single notebook. You can download Flickr8k or Flickr30k (or supply your own image–caption pairs), extract and cache CNN features, train one or more caption models from the CLI or Makefile, evaluate with **BLEU** on a held-out test set, and try the results in a **Streamlit** upload UI. Trained artifacts are kept on disk so you can compare runs (e.g. Flickr8k vs Flickr30k) and switch models in the demo without retraining.

**Developer & course notes** (code walkthrough, CNN/LSTM, modules): see [`docs/`](docs/README.md).

---

## Quick start

1. Python 3.12 venv + `make install` (see [Setup](#setup) below).
2. Kaggle credentials at `~/.kaggle/kaggle.json` for downloadable datasets.
3. `make train EPOCHS=10` from the repo root.
4. `make streamlit` to open the upload UI.

---

## Project layout

```
app/
  main.py                 # CLI (train / evaluate / streamlit / …)
  app.py                  # Streamlit demo
  captioning/             # Training pipeline, model, inference
  caption_datasets/       # Dataset registry and loaders
docs/                     # Technical documentation
Makefile                  # Shortcuts from repo root
```

---

## Setup

**Prerequisites:** Python **3.9–3.12** (not 3.13+; no TensorFlow wheels yet), `pip install -r requirements.txt`, and [Kaggle API credentials](https://www.kaggle.com/docs/api) for `flickr8k` / `flickr30k`.

### macOS (Apple Silicon)

If `pip install tensorflow` fails, use Python 3.12:

```bash
brew install python@3.12    # if needed
make venv
source .venv/bin/activate
make install
make check
```

**Optional GPU:** `make gpu-metal` (pins `tensorflow==2.19.1` + `tensorflow-metal==1.2.0`). Do not mix Metal 1.2.0 with TF 2.20.

If Metal breaks imports, `pip uninstall -y tensorflow-metal` and use CPU.

---

## CLI usage

Run from `app/` (artifacts are written there) or use the **Makefile** from the repo root.

```bash
cd app
python main.py list-datasets
```

### Makefile shortcuts

From the **repository root**:

```bash
make help
make venv
make install
make check
make gpu-metal             # optional Apple Silicon GPU

make list-datasets
make train                 # DATASETS=flickr8k EPOCHS=10
make train DATASETS=flickr8k EPOCHS=30
make train-retrain DATASETS=flickr8k,flickr30k
make train-demo            # flickr8k,flickr30k, epochs=30, YES=1
make evaluate YES=1
make evaluate-demo
make extract-features YES=1

make streamlit
make streamlit PORT=8502
make streamlit-headless
make streamlit-app

make clean
make clean-dry-run
```

| Make target | Equivalent |
|-------------|------------|
| `venv` | `python3.12 -m venv .venv` |
| `gpu-metal` | pinned TF + Metal install |
| `train` | `cd app && python main.py train …` |
| `train-demo` | `train` with `flickr8k,flickr30k`, 30 epochs, `-y` |
| `evaluate` / `extract-features` | same CLI under `app/` |
| `streamlit` | `cd app && python main.py streamlit` |
| `streamlit-app` | `cd app && streamlit run app.py` |
| `clean` | interactive artifact cleanup |

Set `YES=1` on `train`, `evaluate`, or `extract-features` to pass `-y`.

### Commands

| Command | Purpose |
|---------|---------|
| `list-datasets` | Show dataset ids, download availability, and when to use each |
| `train` | Download (if needed), extract features, fit tokenizer, train, save artifacts |
| `extract-features` | Build or refresh `features_<id>.dump` only |
| `evaluate` | BLEU on test split (needs `model_<prefix>.keras`) |
| `streamlit` | Launch upload UI; sidebar picks trained model |
| `clean` | Free disk space interactively (alias: `release-memory`) |

### Clean / release memory

```bash
python main.py clean --dry-run
python main.py clean
python main.py clean -y
```

Prompts per item: trained models, feature caches, `vgg16_model.h5`, `models_registry.json`, Kaggle caches. Type `q` to abort.

### Streamlit options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8501` | HTTP port |
| `--address` | `localhost` | Bind address (`0.0.0.0` for LAN) |
| `--headless` | off | Do not open a browser tab |

Requires `vgg16_model.h5` and at least one trained bundle (`model_<prefix>.keras`, `tokenize_<prefix>.dump`, `maxlen_<prefix>.dump`).

### Training parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | `flickr8k` | Comma-separated ids; merged with prefixed image ids |
| `--epochs` | `10` | Training epochs |
| `--num-words` | (all) | Cap tokenizer vocabulary |
| `--max-caption-length` | (from data) | Override padded sequence length |
| `--embedding-dim` | `256` | Word embedding size |
| `--lstm-units` | `256` | LSTM hidden units |
| `--dense-units` | `256` | Fusion dense size |
| `--dropout` | `0.5` | Dropout on branches |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--image-feature-dim` | `1000` | Must match VGG16 output |
| `--steps-per-epoch` | (train size) | Override fit steps |
| `--force-retrain` | off | Rebuild this run's model (and optionally re-extract features) |
| `-y`, `--yes` | off | Skip deletion confirmations |
| `--custom-images-dir` | (none) | For `--datasets custom` |
| `--custom-captions-file` | (none) | Token or CSV captions |
| `--custom-train-list` / `--custom-test-list` | (none) | Optional split lists |

### Cached artifacts

Written under `app/`:

| File | Role |
|------|------|
| `features_<dataset>.dump` | VGG16 features per dataset (kept across runs) |
| `model_<prefix>.keras` | Caption model (`prefix` = sorted dataset ids) |
| `tokenize_<prefix>.dump` | Tokenizer |
| `maxlen_<prefix>.dump` | Max caption length |
| `vgg16_model.h5` | Shared VGG16 extractor |
| `models_registry.json` | Index of trained runs |
| `training_stats_<prefix>.json` | Timing summary after train/evaluate |

**Multiple models:** Train Flickr8k → `model_flickr8k.keras`; later Flickr30k → `model_flickr30k.keras` without removing the first. Streamlit lists all bundles in the sidebar.

**`--force-retrain`:** Only targets the current prefix and selected datasets' feature files.

### Datasets

| Id | Download | Train | When to use |
|----|----------|-------|-------------|
| `flickr8k` | Yes (Kaggle) | Yes | Course baseline (~8k images) |
| `flickr30k` | Yes ([Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k)) | Yes | Richer captions (~31k); accept rules on first download |
| `custom` | No | Yes | Your images + caption file |
| `coco`, `visual_genome`, `vizwiz` | No | No | Use official exports + `custom` |
| `nocaps` | No | No | Evaluation benchmark only |
| `laion_400m`, `laion_5b` | No | No | Reference only (web-scale) |

**Custom formats:** Flickr-style lines, pipe/tab pairs, or CSV `filename,caption`.

**Flickr30k:** Accept dataset rules on [adityajn105/flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) before first download.

---

## Documentation

| Topic | Doc |
|-------|-----|
| Overview & flow | [docs/README.md](docs/README.md) |
| Environment & config | [docs/setup.md](docs/setup.md) |
| Dataset loaders | [docs/datasets.md](docs/datasets.md) |
| VGG16 features | [docs/extract-features.md](docs/extract-features.md) |
| Training pipeline | [docs/training.md](docs/training.md) |
| BLEU evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Streamlit app | [docs/streamlit.md](docs/streamlit.md) |
| CNN, LSTM, trade-offs | [docs/ai-concepts.md](docs/ai-concepts.md) |
| File map | [docs/module-reference.md](docs/module-reference.md) |
