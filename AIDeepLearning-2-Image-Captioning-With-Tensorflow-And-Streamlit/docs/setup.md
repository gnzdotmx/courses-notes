# Setup

How the project environment is configured and which modules own setup logic.

## Requirements

- **Python 3.9–3.12** (TensorFlow has no pip wheels for 3.13+ on macOS arm64 yet).
- Dependencies in [`requirements.txt`](../requirements.txt): TensorFlow, Streamlit, `kagglehub`, NumPy, Pillow, NLTK, OpenCV headless.

## Recommended workflow

From the repo root:

```bash
make venv          # python3.12 -m venv .venv (if missing)
source .venv/bin/activate
make install       # pip install -r requirements.txt
make check         # verify TensorFlow import
```

On **Apple Silicon**, optional GPU after CPU works:

```bash
make gpu-metal     # tensorflow==2.19.1 + tensorflow-metal==1.2.0
```

Pin these versions: Metal 1.2.0 does not load with TensorFlow 2.20.

## Kaggle API

Downloadable datasets (`flickr8k`, `flickr30k`) use [kagglehub](https://github.com/Kaggle/kagglehub). Place credentials at `~/.kaggle/kaggle.json` ([Kaggle API docs](https://www.kaggle.com/docs/api)).

For **Flickr30k**, accept the dataset rules on Kaggle while logged in before the first download.

## Code involved

| Piece | Module | Role |
|-------|--------|------|
| CLI entry | `app/main.py` | Dispatches to `captioning.cli` |
| Argument parsing | `captioning/cli.py` | Subcommands, flags, Streamlit launcher |
| Config dataclass | `captioning/config.py` | `TrainingConfig`: hyperparameters, paths, validation |
| Paths | `captioning/paths.py` | `APP_DIR`, artifact filenames, registry |
| Makefile | [`Makefile`](../Makefile) | Root-level shortcuts (`venv`, `install`, `train`, …) |

### `TrainingConfig` (`captioning/config.py`)

Frozen dataclass holding one run's settings: `datasets`, `epochs`, embedding/LSTM/dense sizes, dropout, learning rate, custom paths, and flags (`force_retrain`, `skip_confirm`, `train`, `evaluate`, `extract_features`).

`validate()` enforces sensible ranges and requires `--custom-images-dir` + `--custom-captions-file` when `custom` is in `--datasets`.

### Lazy imports

`list-datasets` and `streamlit` avoid loading TensorFlow until needed. `train` / `evaluate` / `extract-features` import `captioning.pipeline`, which pulls in TensorFlow and Keras.

## Security notes

- Custom dataset paths must exist; resolved with `pathlib`: no arbitrary URL downloads.
- Kaggle access only through the official `kagglehub` client.
