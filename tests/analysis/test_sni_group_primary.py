from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bilinear_lmmd.analysis.sni_group_primary import (
    analyze_sni_group_primary,
)


CLASSES = ("class_a", "class_b", "class_c")


def _write_manifest(root: Path) -> list[dict[str, str]]:
    rows = [
        {
            "crop_path": "source/val/class_a/a1.jpg",
            "visual_label": "class_a",
            "dataset": "adrian_detection",
            "group_id": "g1",
        },
        {
            "crop_path": "source/val/class_a/a2.jpg",
            "visual_label": "class_a",
            "dataset": "adrian_detection",
            "group_id": "g1",
        },
        {
            "crop_path": "source/val/class_b/b1.jpg",
            "visual_label": "class_b",
            "dataset": "adrian_detection",
            "group_id": "g2",
        },
        {
            "crop_path": "source/val/class_c/c1.jpg",
            "visual_label": "class_c",
            "dataset": "faruq_segmentation",
            "group_id": "g3",
        },
    ]
    manifest = root / "manifests" / "val.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "audit.json").write_text(
        json.dumps({"status": "complete", "test_locked": True}),
        encoding="utf-8",
    )
    (root / "ontology.json").write_text(
        json.dumps({"visual_classes": list(CLASSES)}),
        encoding="utf-8",
    )
    return rows


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probabilities: list[list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "path",
        "actual",
        "predicted",
        "correct",
        *[f"prob::{name}" for name in CLASSES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, values in zip(rows, probabilities):
            prediction = CLASSES[max(range(len(CLASSES)), key=values.__getitem__)]
            writer.writerow(
                {
                    "path": f"/content/data/{row['crop_path']}",
                    "actual": row["visual_label"],
                    "predicted": prediction,
                    "correct": int(prediction == row["visual_label"]),
                    **{
                        f"prob::{name}": values[index]
                        for index, name in enumerate(CLASSES)
                    },
                }
            )


def test_group_primary_analysis_emits_group_and_cluster_evidence(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifest"
    rows = _write_manifest(manifest_root)
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_predictions(
        baseline,
        rows,
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
        ],
    )
    _write_predictions(
        candidate,
        rows,
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ],
    )

    output = tmp_path / "output"
    report = analyze_sni_group_primary(
        manifest_root,
        {"BASE": baseline, "CAND": candidate},
        output,
        bootstrap_iterations=40,
        bootstrap_seed=123,
    )

    assert report["split"] == "val"
    assert report["test_opened"] is False
    assert report["group_class_units"] == 3
    assert report["source_clusters"] == 3
    assert report["group_class_metrics"]["CAND"]["macro_f1"] == 1.0
    assert (
        report["comparisons"]["BASE_vs_CAND"]["macro_f1"]["delta"] > 0
    )
    assert (output / "group_primary_report.json").is_file()
    assert (output / "group_predictions_BASE.csv").is_file()
    assert (output / "group_predictions_CAND.csv").is_file()


def test_group_primary_analysis_keeps_test_locked(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Test terkunci"):
        analyze_sni_group_primary(
            tmp_path,
            {"BASE": tmp_path / "predictions.csv"},
            tmp_path / "output",
            split="test",
        )
