from __future__ import annotations

import csv
import json
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_sni_v3_jiao_screening import (
    MODEL_CONFIGS,
    STAGE_MODELS,
    _aggregate_decision,
    validate_v3_manifest,
)


def _report(
    baseline: str,
    candidate: str,
    macro_delta: float,
    worst_delta: float,
    adrian_delta: float = 0.0,
    faruq_delta: float = 0.0,
) -> dict:
    return {
        "comparisons": {
            f"{baseline}_vs_{candidate}": {
                "macro_f1": {"delta": macro_delta},
                "worst_class_f1": {"delta": worst_delta},
            }
        },
        "domain_group_class_metrics": {
            baseline: {
                "adrian_detection": {"macro_f1": 0.7},
                "faruq_segmentation": {"macro_f1": 0.7},
            },
            candidate: {
                "adrian_detection": {
                    "macro_f1": 0.7 + adrian_delta
                },
                "faruq_segmentation": {
                    "macro_f1": 0.7 + faruq_delta
                },
            },
        },
    }


def test_v3_jiao_models_freeze_group_primary_training_contract() -> None:
    assert STAGE_MODELS["mechanism"] == ("S3J0", "S3J1")
    assert STAGE_MODELS["benchmark"] == ("S3B0",)
    assert set(MODEL_CONFIGS) == {"S3J0", "S3J1", "S3B0"}

    for path in MODEL_CONFIGS.values():
        cfg = load_config(path)
        assert cfg["model"]["num_classes"] == 15
        assert (
            cfg["data"]["manifest_weight_column"]
            == "train_weight_group_class_equal"
        )
        assert (
            cfg["evaluation"]["selection_metric"]
            == "source_group_class_macro_f1"
        )


def test_jiao_decision_requires_macro_tail_and_each_domain() -> None:
    passing = {
        42: _report("S3J0", "S3J1", 0.02, -0.005, 0.01, 0.0)
    }
    assert (
        _aggregate_decision(passing, "S3J0", "S3J1")["decision"]
        == "PASS"
    )

    harmed_domain = {
        42: _report("S3J0", "S3J1", 0.02, 0.0, -0.011, 0.02)
    }
    assert (
        _aggregate_decision(harmed_domain, "S3J0", "S3J1")["decision"]
        == "FAIL"
    )

    harmed_tail = {
        42: _report("S3J0", "S3J1", 0.02, -0.011, 0.02, 0.02)
    }
    assert (
        _aggregate_decision(harmed_tail, "S3J0", "S3J1")["decision"]
        == "FAIL"
    )


def test_three_seed_decision_requires_two_positive_seeds() -> None:
    reports = {
        42: _report("B", "C", 0.02, 0.0),
        123: _report("B", "C", 0.02, 0.0),
        2026: _report("B", "C", -0.01, 0.0),
    }

    result = _aggregate_decision(reports, "B", "C")

    assert result["decision"] == "PASS"
    assert result["required_positive_seeds"] == 2


def test_validate_v3_manifest_checks_independent_group_readiness(
    tmp_path: Path,
) -> None:
    classes = tuple(f"class_{index:02d}" for index in range(15))
    root = tmp_path / "manifest"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    (root / "audit.json").write_text(
        json.dumps({"status": "complete", "test_locked": True}),
        encoding="utf-8",
    )
    (root / "ontology.json").write_text(
        json.dumps({"visual_classes": list(classes)}),
        encoding="utf-8",
    )
    train_rows = [
        {
            "dataset": "adrian_detection",
            "group_id": f"train_{index}",
            "visual_label": name,
            "crop_path": f"source/train/{name}/{index}.jpg",
            "train_weight_group_class_equal": "1.0",
        }
        for index, name in enumerate(classes)
    ]
    val_rows = []
    for class_name in classes:
        for dataset in ("adrian_detection", "faruq_segmentation"):
            for index in range(10):
                val_rows.append(
                    {
                        "dataset": dataset,
                        "group_id": f"{class_name}_{dataset}_{index}",
                        "visual_label": class_name,
                        "crop_path": (
                            f"source/val/{class_name}/"
                            f"{dataset}_{index}.jpg"
                        ),
                    }
                )
    for path, rows in (
        (manifests / "train_weighted.csv", train_rows),
        (manifests / "val.csv", val_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    readiness = validate_v3_manifest(root)

    assert readiness["val_source_groups"] == 300
    assert set(readiness["val_dataset_groups"].values()) == {150}
