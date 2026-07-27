from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import WeightedRandomSampler

from bilinear_lmmd.data.loaders import build_loaders


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_sni_v2_manifest_loader_uses_weighted_train_only(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    manifest_root = tmp_path / "classification-v2"
    rows = []
    for split in ("train", "val"):
        for index, label in enumerate(("class_a", "class_b")):
            relative = Path("source") / split / label / f"{index}.jpg"
            path = image_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (index * 100, 50, 20)).save(path)
            rows.append(
                {
                    "crop_path": relative.as_posix(),
                    "visual_label": label,
                    "train_weight_inverse_sqrt": 2.0 if label == "class_a" else 0.5,
                    "generated_split": split,
                }
            )

    train = [row for row in rows if row["generated_split"] == "train"]
    val = [row for row in rows if row["generated_split"] == "val"]
    _write_csv(manifest_root / "manifests" / "train_weighted.csv", train)
    _write_csv(manifest_root / "manifests" / "val.csv", val)
    (manifest_root / "audit.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (manifest_root / "ontology.json").write_text(
        json.dumps({"visual_classes": ["class_a", "class_b"]}),
        encoding="utf-8",
    )

    loaders = build_loaders(
        {
            "dataset_format": "sni_manifest_v2",
            "root": str(image_root),
            "manifest_root": str(manifest_root),
            "manifest_weight_column": "train_weight_inverse_sqrt",
            "image_size": 24,
            "batch_size": 2,
            "workers": 0,
            "rotation_angles": [0],
            "augmentation_mode": "paper",
        },
        require_target=False,
    )

    assert loaders.classes == ["class_a", "class_b"]
    assert isinstance(loaders.source_train.sampler, WeightedRandomSampler)
    assert loaders.source_train.dataset.sample_weights == [2.0, 0.5]
    assert loaders.source_val.dataset.sample_weights is None
    images, targets = next(iter(loaders.source_val))
    assert tuple(images.shape) == (2, 3, 24, 24)
    assert targets.tolist() == [0, 1]
