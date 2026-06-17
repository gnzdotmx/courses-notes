"""End-to-end training pipeline."""

from __future__ import annotations

from captioning.artifacts import (
    datasets_needing_feature_extract,
    prepare_training_artifacts,
)
from captioning.config import TrainingConfig
from captioning.features import feature_for_split, load_or_extract_features
from captioning.inference import evaluate_model, generate_caption
from captioning.model_builder import load_or_train_model
from captioning.paths import APP_DIR, RunArtifacts, register_model_run
from captioning.text_processing import load_or_create_maxlen, load_or_create_tokenizer
from captioning.timing import TrainingTimer
from caption_datasets.registry import load_selected_datasets


def _build_feature_index(
    train_description: dict[str, list[str]],
    test_description: dict[str, list[str]],
    features: dict,
) -> tuple[dict[str, object], dict[str, object]]:
    train_features: dict[str, object] = {}
    for image_id in train_description:
        vector = feature_for_split(features, image_id)
        if vector is not None:
            train_features[image_id] = vector

    test_features: dict[str, object] = {}
    for image_id in test_description:
        vector = feature_for_split(features, image_id)
        if vector is not None:
            test_features[image_id] = vector

    missing_train = len(train_description) - len(train_features)
    if missing_train:
        print(f"Warning: {missing_train} train images have no extracted features.")

    return train_features, test_features


def run_pipeline(config: TrainingConfig) -> None:
    config.validate()
    dataset_ids = list(config.datasets)
    artifacts = RunArtifacts.for_datasets(dataset_ids)
    timer = TrainingTimer(run_label=artifacts.prefix)
    stats_path = APP_DIR / f"training_stats_{artifacts.prefix}.json"
    timer.notes["stats_file"] = str(stats_path)
    timer.notes["datasets"] = ", ".join(dataset_ids)
    timer.notes["epochs"] = config.epochs

    print(f"Run artifact prefix: {artifacts.prefix}")
    print(f"Timing started at {timer.started_at_wall}")

    need_extract = datasets_needing_feature_extract(dataset_ids, force=False)
    if config.force_retrain:
        need_extract = set(dataset_ids)

    with timer.phase("artifact_checks"):
        ok = prepare_training_artifacts(
            artifacts,
            dataset_ids,
            force_retrain=config.force_retrain,
            skip_confirm=config.skip_confirm,
            force_feature_datasets=frozenset(dataset_ids) if config.force_retrain else frozenset(),
            will_train=config.train,
        )
    if not ok:
        timer.notes["aborted"] = "artifact_checks declined or cancelled"
        timer.print_summary()
        return

    with timer.phase("load_datasets"):
        train_description, test_description, image_roots = load_selected_datasets(config)
    timer.notes["train_images"] = len(train_description)
    timer.notes["test_images"] = len(test_description)

    with timer.phase("features"):
        features = load_or_extract_features(
            [(prefix, path) for prefix, path in image_roots],
            datasets_to_extract=need_extract,
        )

    with timer.phase("prepare_splits"):
        train_features, test_features = _build_feature_index(
            train_description, test_description, features
        )

    refit_artifacts = config.force_retrain or not artifacts.model_path.is_file()
    with timer.phase("tokenizer"):
        tokenizer = load_or_create_tokenizer(
            train_description, config, artifacts, refit=refit_artifacts
        )
    vocab_size = len(tokenizer.word_index) + 1
    with timer.phase("maxlen"):
        maxlen = load_or_create_maxlen(
            train_description, config, artifacts, refit=refit_artifacts
        )
    print(f"Vocabulary size: {vocab_size}, max caption length: {maxlen}")
    timer.notes["vocab_size"] = vocab_size
    timer.notes["maxlen"] = maxlen

    should_train = config.train and (
        config.force_retrain or not artifacts.model_path.is_file()
    )
    if should_train:
        with timer.phase("caption_training"):
            model = load_or_train_model(
                config,
                artifacts,
                vocab_size,
                maxlen,
                train_description,
                train_features,
                tokenizer,
                should_train=True,
                timer=timer,
            )
    else:
        timer.notes["caption_training"] = "skipped (loaded existing model)"
        with timer.phase("load_model"):
            model = load_or_train_model(
                config,
                artifacts,
                vocab_size,
                maxlen,
                train_description,
                train_features,
                tokenizer,
                should_train=False,
                timer=timer,
            )

    model.summary()

    if should_train or artifacts.model_path.is_file():
        register_model_run(artifacts.prefix, dataset_ids)

    if config.evaluate:
        if not test_description:
            print("No test split available for evaluation.")
        else:
            with timer.phase("evaluation"):
                evaluate_model(model, test_description, test_features, tokenizer, maxlen)

    if test_description:
        with timer.phase("sample_caption"):
            sample_id = next(iter(test_description))
            if sample_id in test_features:
                caption = generate_caption(
                    model, tokenizer, test_features[sample_id], maxlen
                )
                print(f"Sample ({sample_id}): {caption}")

    timer.save(stats_path)
    timer.print_summary()
