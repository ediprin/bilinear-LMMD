from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from bilinear_lmmd.data.preparation.prepare_sni_classification_v3 import (
    VISUAL_CLASSES,
    allocate_source_balanced_groups,
    prepare_sni_classification_v3,
)
from bilinear_lmmd.data.sni_ontology import SNI_CLASSES


def _allocation_rows() -> list[dict[str, str]]:
    rows = []
    for dataset, groups in (
        ("adrian_detection", 104),
        ("faruq_segmentation", 80),
    ):
        for group_index in range(groups):
            group_id = f"{dataset}-{group_index}"
            class_name = VISUAL_CLASSES[group_index % len(VISUAL_CLASSES)]
            repetitions = (
                200
                if dataset == "adrian_detection" and group_index >= 100
                else 1
            )
            for _ in range(repetitions):
                rows.append(
                    {
                        "group_id": group_id,
                        "dataset": dataset,
                        "visual_label": class_name,
                    }
                )
    return rows


def test_source_balanced_allocator_is_deterministic_and_resists_dense_groups():
    rows = _allocation_rows()
    first, _ = allocate_source_balanced_groups(
        rows,
        seed=42,
        trials=128,
        min_eval_groups_per_class=1,
        min_eval_groups_per_dataset=10,
    )
    second, _ = allocate_source_balanced_groups(
        rows,
        seed=42,
        trials=128,
        min_eval_groups_per_class=1,
        min_eval_groups_per_dataset=10,
    )
    assert first == second

    # Count each source group once rather than once per crop.
    unique_counts = {
        split: Counter() for split in ("train", "val", "test")
    }
    for group_id, split in first.items():
        dataset = group_id.rsplit("-", 1)[0]
        unique_counts[split][dataset] += 1
    assert unique_counts["val"]["adrian_detection"] >= 15
    assert unique_counts["test"]["adrian_detection"] >= 15
    assert unique_counts["val"]["faruq_segmentation"] >= 12
    assert unique_counts["test"]["faruq_segmentation"] >= 12


def _source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "sni-v1"
    rows = []
    index = 0
    for archive_split in ("train", "val", "test"):
        for dataset in ("adrian_detection", "faruq_segmentation"):
            for class_name in SNI_CLASSES:
                relative = (
                    Path("source")
                    / archive_split
                    / class_name
                    / f"{index}.jpg"
                )
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (index % 255, 10, 20)).save(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "archive_split": archive_split,
                        "generated_split": archive_split,
                        "group_id": f"group-{index}",
                        "source_identity": f"source-{index}",
                        "source_file": f"/raw/{index}.jpg",
                        "source_sha256": f"source-hash-{index}",
                        "image_id": index,
                        "annotation_id": index,
                        "original_class": class_name,
                        "canonical_class": class_name,
                        "bbox_x": 1,
                        "bbox_y": 1,
                        "bbox_width": 6,
                        "bbox_height": 6,
                        "crop_width": 8,
                        "crop_height": 8,
                        "crop_sha256": f"crop-hash-{index}",
                        "crop_path": relative.as_posix(),
                    }
                )
                index += 1
    with (root / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "SNI_instance_crop_v1",
                "classes": list(SNI_CLASSES),
                "output_crops": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_prepare_v3_reassigns_manifests_without_copying_images(tmp_path: Path):
    source = _source_dataset(tmp_path)
    output = tmp_path / "v3"
    audit = prepare_sni_classification_v3(
        source,
        output,
        seed=42,
        trials=256,
        min_eval_samples_per_class=1,
        min_eval_groups_per_class=1,
        min_eval_groups_per_dataset=1,
        max_single_group_fraction_per_class=1.0,
    )

    assert audit["status"] == "complete"
    assert audit["integrity"]["generated_cross_split_groups"] == 0
    assert audit["statistical_readiness"]["split_gate"] == "PASS"
    assert audit["training_authorized"] is False
    assert not (output / "source").exists()

    with (output / "manifests/all.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["generated_split"])
    assert all(len(splits) == 1 for splits in group_splits.values())

    with (output / "manifests/train_weighted.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        train_rows = list(csv.DictReader(handle))
    assert train_rows
    assert all(row["train_weight_group_equal"] for row in train_rows)
    assert all(row["train_weight_group_class_equal"] for row in train_rows)
