from __future__ import annotations

import csv
import json
from pathlib import Path

from bilinear_lmmd.analysis.sni_v2_failure import analyze_sni_v2_failure


CLASSES = ("class_a", "class_b", "class_c")


def _write_manifest(root: Path) -> list[dict[str, str]]:
    rows = []
    specification = (
        ("class_a", "a1.jpg", "adrian_detection", "g1", "s1"),
        ("class_a", "a2.jpg", "adrian_detection", "g1", "s1"),
        ("class_b", "b1.jpg", "adrian_detection", "g2", "s2"),
        ("class_b", "b2.jpg", "faruq_segmentation", "g3", "s3"),
        ("class_c", "c1.jpg", "faruq_segmentation", "g4", "s4"),
        ("class_c", "c2.jpg", "faruq_segmentation", "g5", "s5"),
    )
    for label, filename, dataset, group, source in specification:
        rows.append(
            {
                "crop_path": f"source/val/{label}/{filename}",
                "visual_label": label,
                "dataset": dataset,
                "group_id": group,
                "source_identity": source,
            }
        )
    manifest = root / "manifests" / "val.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "complete",
        "test_locked": True,
        "label_design": {"visual_v2_classes": list(CLASSES)},
        "statistical_readiness": {
            "weak_classes": {
                "val": {
                    "class_b": {
                        "enough_samples": False,
                        "enough_groups": False,
                    }
                }
            }
        },
    }
    (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return rows


def _write_predictions(
    path: Path,
    manifest_rows: list[dict[str, str]],
    predicted: list[str],
) -> None:
    path.parent.mkdir(parents=True)
    columns = ["path", "actual", "predicted", "correct", *[
        f"prob::{name}" for name in CLASSES
    ]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, prediction in zip(manifest_rows, predicted):
            probabilities = {
                f"prob::{name}": 0.8 if name == prediction else 0.1
                for name in CLASSES
            }
            writer.writerow(
                {
                    "path": f"/content/data/{row['crop_path']}",
                    "actual": row["visual_label"],
                    "predicted": prediction,
                    "correct": int(prediction == row["visual_label"]),
                    **probabilities,
                }
            )


def test_sni_v2_failure_audit_joins_groups_and_domains(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifest"
    rows = _write_manifest(manifest_root)
    gap_path = tmp_path / "gap" / "predictions.csv"
    multires_path = tmp_path / "multires" / "predictions.csv"
    _write_predictions(
        gap_path,
        rows,
        ["class_a", "class_a", "class_b", "class_b", "class_c", "class_a"],
    )
    _write_predictions(
        multires_path,
        rows,
        ["class_a", "class_a", "class_a", "class_a", "class_c", "class_a"],
    )

    output = tmp_path / "output"
    report = analyze_sni_v2_failure(
        manifest_root,
        {"S2G": gap_path, "S2MR": multires_path},
        output,
    )

    assert report["selection_split"] == "val"
    assert report["test_opened"] is False
    assert report["sample_count"] == 6
    assert report["group_class_unit_count"] == 5
    assert report["outcomes"]["gap_only_correct"] == 2
    assert report["crop_metrics"]["S2MR"]["macro_f1"] < report["crop_metrics"]["S2G"]["macro_f1"]
    assert report["most_harmed_classes"][0]["class"] == "class_b"
    assert report["most_harmed_classes"][0]["weak_groups"] is True
    assert set(report["domain_metrics"]) == {
        "adrian_detection",
        "faruq_segmentation",
    }
    for name in (
        "sni_v2_failure_audit.json",
        "class_diagnostics.csv",
        "domain_metrics.csv",
        "class_domain_metrics.csv",
        "confusion_pairs.csv",
        "sample_outcomes.csv",
    ):
        assert (output / name).is_file()

