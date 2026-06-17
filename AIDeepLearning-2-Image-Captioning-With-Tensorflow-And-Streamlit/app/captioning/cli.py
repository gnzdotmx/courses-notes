"""Command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from captioning.config import TrainingConfig
from captioning.paths import APP_DIR
from caption_datasets.registry import list_datasets

STREAMLIT_APP = APP_DIR / "app.py"


def _comma_list(value: str) -> tuple[str, ...]:
    items = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list.")
    return tuple(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image captioning: download data, extract features, train, evaluate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="Show datasets and when to use them.")

    train = sub.add_parser("train", help="Run the full training pipeline.")
    _add_train_args(train)

    extract = sub.add_parser("extract-features", help="Only extract/cache VGG16 features.")
    _add_train_args(extract, features_only=True)

    evaluate = sub.add_parser("evaluate", help="Evaluate BLEU on the test split.")
    _add_train_args(evaluate, evaluate_only=True)

    streamlit_cmd = sub.add_parser(
        "streamlit",
        help="Launch the Streamlit captioning UI (app.py).",
    )
    streamlit_cmd.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    streamlit_cmd.add_argument(
        "--address",
        default="localhost",
        help="Bind address (default: localhost).",
    )
    streamlit_cmd.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a browser (useful on servers).",
    )

    clean = sub.add_parser(
        "clean",
        aliases=["release-memory"],
        help="Interactively free disk space (pick trained models, features, Kaggle caches, etc.).",
    )
    clean.add_argument(
        "--dry-run",
        action="store_true",
        help="List deletable items only; do not prompt or delete.",
    )
    clean.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the final confirmation after selecting items (still prompts per item).",
    )

    return parser


def _add_train_args(
    parser: argparse.ArgumentParser,
    *,
    features_only: bool = False,
    evaluate_only: bool = False,
) -> None:
    parser.add_argument(
        "--datasets",
        type=_comma_list,
        default="flickr8k",
        help="Comma-separated dataset ids (e.g. flickr8k,flickr30k,custom).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--num-words",
        type=int,
        default=None,
        help="Cap tokenizer vocabulary (Keras Tokenizer num_words).",
    )
    parser.add_argument(
        "--max-caption-length",
        type=int,
        default=None,
        help="Override maximum caption token length.",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--image-feature-dim",
        type=int,
        default=1000,
        help="Must match VGG16 head output in vgg16_model.h5.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--custom-images-dir",
        type=Path,
        default=None,
        help="Directory of images for --datasets custom.",
    )
    parser.add_argument(
        "--custom-captions-file",
        type=Path,
        default=None,
        help="Caption file (token or CSV) for --datasets custom.",
    )
    parser.add_argument("--custom-train-list", type=Path, default=None)
    parser.add_argument("--custom-test-list", type=Path, default=None)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Delete model/tokenizer artifacts and train from scratch.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts when removing cached files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory for model.dump artifacts (default: app/).",
    )

    if features_only:
        parser.set_defaults(train=False, evaluate=False, extract_features=True)
    elif evaluate_only:
        parser.set_defaults(train=False, evaluate=True, extract_features=False)
    else:
        parser.set_defaults(train=True, evaluate=False, extract_features=True)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        datasets=args.datasets,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_words=args.num_words,
        max_caption_length=args.max_caption_length,
        image_feature_dim=args.image_feature_dim,
        steps_per_epoch=args.steps_per_epoch,
        custom_images_dir=args.custom_images_dir,
        custom_captions_file=args.custom_captions_file,
        custom_train_list=args.custom_train_list,
        custom_test_list=args.custom_test_list,
        work_dir=args.work_dir,
        force_retrain=args.force_retrain,
        skip_confirm=args.yes,
        extract_features=getattr(args, "extract_features", True),
        train=getattr(args, "train", True),
        evaluate=getattr(args, "evaluate", False),
    )


def run_streamlit_app(*, port: int, address: str, headless: bool) -> None:
    """Start Streamlit with app/ as the working directory (artifact paths)."""
    if not STREAMLIT_APP.is_file():
        raise FileNotFoundError(f"Streamlit app not found: {STREAMLIT_APP}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.port",
        str(port),
        "--server.address",
        address,
    ]
    if headless:
        cmd.append("--server.headless")
        cmd.append("true")

    print(f"Starting Streamlit: {' '.join(cmd)}")
    print(f"Working directory: {APP_DIR}")
    raise SystemExit(subprocess.call(cmd, cwd=APP_DIR))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-datasets":
        list_datasets()
        return

    if args.command == "streamlit":
        run_streamlit_app(
            port=args.port,
            address=args.address,
            headless=args.headless,
        )
        return

    if args.command in ("clean", "release-memory"):
        from captioning.cleanup import run_cleanup

        run_cleanup(skip_confirm=args.yes, dry_run=args.dry_run)
        return

    config = config_from_args(args)
    from captioning.pipeline import run_pipeline  # lazy: avoids loading TensorFlow for list-datasets

    run_pipeline(config)
