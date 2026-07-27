from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class GroupClassPredictions:
    classes: tuple[str, ...]
    labels: tuple[int, ...]
    predictions: tuple[int, ...]
    probabilities: np.ndarray
    cluster_keys: tuple[tuple[str, str], ...]
    unit_keys: tuple[tuple[str, str, str], ...]
    crop_counts: tuple[int, ...]


def aggregate_group_class_probabilities(
    manifest_rows: list[dict[str, str]],
    labels: list[int],
    probabilities: np.ndarray,
    classes: list[str] | tuple[str, ...],
) -> GroupClassPredictions:
    classes = tuple(classes)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(classes)):
        raise ValueError(
            "Shape probability tidak cocok: "
            f"{probabilities.shape} != {(len(labels), len(classes))}"
        )
    if len(manifest_rows) != len(labels):
        raise ValueError("Jumlah manifest dan prediksi tidak sama.")
    required = {"dataset", "group_id", "visual_label"}
    if not manifest_rows or required.difference(manifest_rows[0]):
        raise ValueError(
            "Manifest group-primary wajib memiliki dataset, group_id, "
            "dan visual_label."
        )

    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (row, label) in enumerate(zip(manifest_rows, labels)):
        actual = row["visual_label"]
        if actual not in classes:
            raise ValueError(f"Visual label tidak dikenal: {actual}")
        if classes[label] != actual:
            raise ValueError(
                f"Urutan label prediksi tidak cocok manifest pada indeks {index}."
            )
        grouped[(row["dataset"], row["group_id"], actual)].append(index)

    unit_keys = tuple(sorted(grouped))
    unit_labels = []
    unit_probabilities = []
    crop_counts = []
    for key in unit_keys:
        indices = grouped[key]
        unit_labels.append(classes.index(key[2]))
        unit_probabilities.append(probabilities[indices].mean(axis=0))
        crop_counts.append(len(indices))
    matrix = np.asarray(unit_probabilities, dtype=np.float64)
    predictions = matrix.argmax(axis=1)
    return GroupClassPredictions(
        classes=classes,
        labels=tuple(unit_labels),
        predictions=tuple(int(value) for value in predictions),
        probabilities=matrix,
        cluster_keys=tuple((dataset, group_id) for dataset, group_id, _ in unit_keys),
        unit_keys=unit_keys,
        crop_counts=tuple(crop_counts),
    )


def classification_metrics(
    labels: list[int] | tuple[int, ...] | np.ndarray,
    predictions: list[int] | tuple[int, ...] | np.ndarray,
    classes: list[str] | tuple[str, ...],
) -> dict:
    classes = tuple(classes)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    indices = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indices,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        # Use the frozen class set. sklearn's balanced_accuracy_score only
        # averages classes present in one bootstrap draw, which would change
        # the estimand whenever a rare class is absent from that draw.
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "worst_class_f1": float(f1.min()),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(classes)
        },
    }


def stratified_cluster_bootstrap_indices(
    cluster_keys: tuple[tuple[str, str], ...],
    *,
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    if iterations <= 0:
        raise ValueError("Bootstrap iterations harus positif.")
    indices_by_cluster: dict[tuple[str, str], list[int]] = defaultdict(list)
    clusters_by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, key in enumerate(cluster_keys):
        indices_by_cluster[key].append(index)
    for key in sorted(indices_by_cluster):
        clusters_by_dataset[key[0]].append(key)

    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = []
        for dataset in sorted(clusters_by_dataset):
            clusters = clusters_by_dataset[dataset]
            draws = generator.integers(0, len(clusters), size=len(clusters))
            for draw in draws:
                selected.extend(indices_by_cluster[clusters[int(draw)]])
        samples.append(np.asarray(selected, dtype=np.int64))
    return samples


def bootstrap_metric_samples(
    labels: tuple[int, ...],
    predictions: dict[str, tuple[int, ...]],
    classes: tuple[str, ...],
    cluster_keys: tuple[tuple[str, str], ...],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    bootstrap_indices = stratified_cluster_bootstrap_indices(
        cluster_keys, iterations=iterations, seed=seed
    )
    label_array = np.asarray(labels, dtype=np.int64)
    prediction_arrays = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in predictions.items()
    }
    output = {
        name: {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "worst_class_f1": [],
        }
        for name in predictions
    }
    for indices in bootstrap_indices:
        sampled_labels = label_array[indices]
        for name, values in prediction_arrays.items():
            metrics = classification_metrics(
                sampled_labels, values[indices], classes
            )
            for metric in output[name]:
                output[name][metric].append(metrics[metric])
    return output


def interval(values: list[float], confidence: float = 0.95) -> dict:
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence harus berada pada (0, 1).")
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(array, alpha)),
        "upper": float(np.quantile(array, 1.0 - alpha)),
    }
