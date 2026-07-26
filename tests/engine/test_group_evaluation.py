from __future__ import annotations

import numpy as np

from bilinear_lmmd.engine.group_evaluation import (
    aggregate_group_class_probabilities,
    classification_metrics,
    stratified_cluster_bootstrap_indices,
)


def test_group_class_aggregation_averages_crops_once_per_source_class() -> None:
    classes = ("a", "b", "c")
    rows = [
        {"dataset": "d1", "group_id": "g1", "visual_label": "a"},
        {"dataset": "d1", "group_id": "g1", "visual_label": "a"},
        {"dataset": "d1", "group_id": "g1", "visual_label": "b"},
        {"dataset": "d2", "group_id": "g2", "visual_label": "c"},
    ]
    labels = [0, 0, 1, 2]
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.3, 0.7, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    bundle = aggregate_group_class_probabilities(
        rows, labels, probabilities, classes
    )

    assert bundle.unit_keys == (
        ("d1", "g1", "a"),
        ("d1", "g1", "b"),
        ("d2", "g2", "c"),
    )
    assert bundle.crop_counts == (2, 1, 1)
    assert np.allclose(bundle.probabilities[0], [0.6, 0.4, 0.0])
    assert bundle.predictions == (0, 1, 2)


def test_balanced_accuracy_uses_frozen_class_set() -> None:
    metrics = classification_metrics([0, 0], [0, 0], ("a", "b", "c"))

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0 / 3.0
    assert metrics["macro_f1"] == 1.0 / 3.0


def test_dataset_stratified_cluster_bootstrap_is_deterministic() -> None:
    keys = (
        ("d1", "g1"),
        ("d1", "g1"),
        ("d1", "g2"),
        ("d2", "g3"),
        ("d2", "g4"),
    )

    first = stratified_cluster_bootstrap_indices(
        keys, iterations=5, seed=123
    )
    second = stratified_cluster_bootstrap_indices(
        keys, iterations=5, seed=123
    )

    assert len(first) == 5
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(len(indices) >= 4 for indices in first)
