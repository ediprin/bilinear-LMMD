from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as TF

from bilinear_lmmd.data.attribute_features import segment_bean


@dataclass(frozen=True)
class DomainLoaders:
    source_train: DataLoader
    source_val: DataLoader
    target_train: DataLoader | None
    target_val: DataLoader | None
    classes: list[str]


class ManifestImageDataset(Dataset):
    """Image dataset backed by a leakage-audited CSV manifest."""

    def __init__(
        self,
        image_root: Path,
        manifest_path: Path,
        classes: list[str],
        transform=None,
        weight_column: str | None = None,
    ):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Manifest kosong: {manifest_path}")
        required = {"crop_path", "visual_label"}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(
                f"Kolom manifest kurang pada {manifest_path}: {sorted(missing)}"
            )

        self.root = str(image_root)
        self.manifest_path = str(manifest_path)
        self.classes = list(classes)
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }
        self.transform = transform
        self.target_transform = None
        self.loader = default_loader
        self.rows = rows
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []
        self.sample_weights: list[float] | None = (
            [] if weight_column is not None else None
        )

        resolved_root = image_root.resolve()
        for row in rows:
            label = row["visual_label"]
            if label not in self.class_to_idx:
                raise ValueError(f"Label manifest tidak dikenal: {label}")
            path = (image_root / row["crop_path"]).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"Path crop keluar dari image_root: {row['crop_path']}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(f"Crop manifest tidak ditemukan: {path}")
            target = self.class_to_idx[label]
            self.samples.append((str(path), target))
            self.targets.append(target)
            if self.sample_weights is not None:
                raw_weight = row.get(weight_column or "", "")
                if raw_weight == "":
                    raise ValueError(
                        f"Bobot {weight_column!r} kosong pada {manifest_path}"
                    )
                weight = float(raw_weight)
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"Bobot sampel tidak valid: {weight}")
                self.sample_weights.append(weight)

        observed = {label for _, label in self.samples}
        expected = set(range(len(self.classes)))
        if observed != expected:
            absent = [self.classes[index] for index in sorted(expected - observed)]
            raise ValueError(
                f"Manifest {manifest_path.name} tidak mencakup kelas: {absent}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class SameClassPairDataset:
    """Add one independently augmented positive image to an ImageFolder sample."""

    def __init__(self, dataset: datasets.ImageFolder):
        self.dataset = dataset
        self.samples = dataset.samples
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.indices_by_class: dict[int, list[int]] = {}
        for index, target in enumerate(self.targets):
            self.indices_by_class.setdefault(int(target), []).append(index)
        singleton = [
            self.classes[target]
            for target, indices in self.indices_by_class.items()
            if len(indices) < 2
        ]
        if singleton:
            raise ValueError(
                "Category consistency membutuhkan minimal dua sampel train per "
                f"kelas; singleton={singleton}."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        candidates = self.indices_by_class[int(target)]
        positive_index = random.choice(candidates)
        while positive_index == index:
            positive_index = random.choice(candidates)
        positive_path, positive_target = self.samples[positive_index]
        positive = self.dataset.loader(positive_path)
        if self.dataset.transform is not None:
            positive = self.dataset.transform(positive)
        if self.dataset.target_transform is not None:
            positive_target = self.dataset.target_transform(positive_target)
        if int(positive_target) != int(target):
            raise RuntimeError("Positive pair berasal dari kelas yang berbeda.")
        return image, positive, target


class DiscreteRotation:
    """Choose one paper-aligned rotation without creating duplicate files."""

    def __init__(self, angles: list[float]):
        self.angles = angles or [0]

    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.rotate(image, random.choice(self.angles), fill=255)


class ObjectCentricCrop:
    """Crop one bean from a light background without feeding a mask to CNN."""

    def __init__(self, margin_fraction: float = 0.10, segmentation_size: int = 256):
        if not 0.0 <= margin_fraction <= 1.0:
            raise ValueError("object_crop_margin harus berada di rentang [0, 1].")
        if segmentation_size < 32:
            raise ValueError("segmentation_size minimal 32 piksel.")
        self.margin_fraction = margin_fraction
        self.segmentation_size = segmentation_size

    @staticmethod
    def _background_color(rgb: np.ndarray) -> tuple[int, int, int]:
        border = np.concatenate(
            (rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0
        )
        return tuple(np.median(border, axis=0).round().astype(np.uint8))

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        rgb_uint8 = np.asarray(image)
        segmentation_image = image.copy()
        segmentation_image.thumbnail(
            (self.segmentation_size, self.segmentation_size),
            Image.Resampling.BILINEAR,
        )
        segmentation_rgb = np.asarray(segmentation_image, dtype=np.float32) / 255.0
        mask = segment_bean(segmentation_rgb)
        rows, columns = np.nonzero(mask)
        scale_x = image.width / segmentation_image.width
        scale_y = image.height / segmentation_image.height
        top = math.floor(rows.min() * scale_y)
        bottom = math.ceil((rows.max() + 1) * scale_y)
        left = math.floor(columns.min() * scale_x)
        right = math.ceil((columns.max() + 1) * scale_x)
        margin = round(max(bottom - top, right - left) * self.margin_fraction)
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(image.width, right + margin)
        bottom = min(image.height, bottom + margin)
        crop = image.crop((left, top, right, bottom))

        side = max(crop.size)
        square = Image.new("RGB", (side, side), self._background_color(rgb_uint8))
        offset = ((side - crop.width) // 2, (side - crop.height) // 2)
        square.paste(crop, offset)
        return square


def _transforms(
    image_size: int,
    train: bool,
    rotation_angles: list[float],
    object_crop: bool = False,
    object_crop_margin: float = 0.10,
    augmentation_mode: str = "standard",
):
    if augmentation_mode not in {"standard", "paper"}:
        raise ValueError("augmentation_mode harus 'standard' atau 'paper'.")
    object_transforms = (
        [ObjectCentricCrop(object_crop_margin)] if object_crop else []
    )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if augmentation_mode == "paper":
        return transforms.Compose(
            object_transforms
            + [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if train:
        return transforms.Compose(
            object_transforms
            + [
                DiscreteRotation(rotation_angles),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        object_transforms
        + [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_image_transform(
    image_size: int,
    train: bool = False,
    rotation_angles: list[float] | None = None,
    object_crop: bool = False,
    object_crop_margin: float = 0.10,
    augmentation_mode: str = "standard",
):
    """Public transform factory for evaluation tools outside DomainLoaders."""

    return _transforms(
        image_size=image_size,
        train=train,
        rotation_angles=rotation_angles or [0],
        object_crop=object_crop,
        object_crop_margin=object_crop_margin,
        augmentation_mode=augmentation_mode,
    )


def build_loaders(cfg: dict, require_target: bool = True) -> DomainLoaders:
    dataset_format = str(cfg.get("dataset_format", "image_folder"))
    if dataset_format == "sni_manifest_v2":
        if require_target:
            raise ValueError(
                "SNI manifest v2 internal hanya mendukung source-only training."
            )
        image_root = Path(cfg["root"])
        manifest_value = cfg.get("manifest_root")
        if not manifest_value:
            raise ValueError("data.manifest_root wajib untuk SNI manifest v2.")
        manifest_root = Path(manifest_value)
        audit_path = manifest_root / "audit.json"
        ontology_path = manifest_root / "ontology.json"
        if not audit_path.is_file() or not ontology_path.is_file():
            raise FileNotFoundError(
                "SNI v2 belum lengkap: audit.json dan ontology.json wajib ada."
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "complete":
            raise ValueError("Audit SNI manifest v2 belum berstatus complete.")
        ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
        classes = list(ontology.get("visual_classes", ()))
        if not classes:
            raise ValueError("ontology.json tidak memiliki visual_classes.")

        image_size = int(cfg["image_size"])
        rotation_angles = [
            float(angle) for angle in cfg.get("rotation_angles", [0])
        ]
        object_crop = bool(cfg.get("object_crop", False))
        object_crop_margin = float(cfg.get("object_crop_margin", 0.10))
        augmentation_mode = str(cfg.get("augmentation_mode", "standard"))
        train_split = str(cfg.get("train_split", "train"))
        val_split = str(cfg.get("val_split", "val"))
        weight_column = cfg.get("manifest_weight_column")
        train_manifest = (
            manifest_root
            / "manifests"
            / ("train_weighted.csv" if weight_column else f"{train_split}.csv")
        )
        val_manifest = manifest_root / "manifests" / f"{val_split}.csv"
        train_dataset = ManifestImageDataset(
            image_root,
            train_manifest,
            classes,
            transform=_transforms(
                image_size,
                train=True,
                rotation_angles=rotation_angles,
                object_crop=object_crop,
                object_crop_margin=object_crop_margin,
                augmentation_mode=augmentation_mode,
            ),
            weight_column=str(weight_column) if weight_column else None,
        )
        val_dataset = ManifestImageDataset(
            image_root,
            val_manifest,
            classes,
            transform=_transforms(
                image_size,
                train=False,
                rotation_angles=rotation_angles,
                object_crop=object_crop,
                object_crop_margin=object_crop_margin,
                augmentation_mode=augmentation_mode,
            ),
        )
        workers = int(cfg.get("workers", 4))
        loader_kwargs = {
            "batch_size": int(cfg["batch_size"]),
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": workers > 0,
        }
        sampler = (
            WeightedRandomSampler(
                train_dataset.sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
            if train_dataset.sample_weights is not None
            else None
        )
        return DomainLoaders(
            source_train=DataLoader(
                train_dataset,
                shuffle=sampler is None,
                sampler=sampler,
                drop_last=True,
                **loader_kwargs,
            ),
            source_val=DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            ),
            target_train=None,
            target_val=None,
            classes=classes,
        )
    if dataset_format != "image_folder":
        raise ValueError(
            "data.dataset_format harus 'image_folder' atau 'sni_manifest_v2'."
        )

    root = Path(cfg["root"])
    train_split = cfg.get("train_split", "train")
    val_split = cfg.get("val_split", "val")
    image_size = int(cfg["image_size"])

    source_paths = {
        "source_train": root / cfg["source"] / train_split,
        "source_val": root / cfg["source"] / val_split,
    }
    target_paths = {
        "target_train": root / cfg["target"] / train_split,
        "target_val": root / cfg["target"] / val_split,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_dir()]
    target_exists = all(path.is_dir() for path in target_paths.values())
    if require_target and not target_exists:
        missing.extend(str(path) for path in target_paths.values() if not path.is_dir())
    if missing:
        raise FileNotFoundError(
            "Folder dataset belum lengkap:\n- " + "\n- ".join(missing)
        )

    paths = dict(source_paths)
    if target_exists:
        paths.update(target_paths)
    rotation_angles = [float(angle) for angle in cfg.get("rotation_angles", [0])]
    object_crop = bool(cfg.get("object_crop", False))
    object_crop_margin = float(cfg.get("object_crop_margin", 0.10))
    augmentation_mode = str(cfg.get("augmentation_mode", "standard"))
    datasets_by_split = {
        name: datasets.ImageFolder(
            path,
            transform=_transforms(
                image_size,
                train=name.endswith("train"),
                rotation_angles=rotation_angles,
                object_crop=object_crop,
                object_crop_margin=object_crop_margin,
                augmentation_mode=augmentation_mode,
            ),
        )
        for name, path in paths.items()
    }
    expected = datasets_by_split["source_train"].class_to_idx
    for name, dataset in datasets_by_split.items():
        if dataset.class_to_idx != expected:
            raise ValueError(
                f"Pemetaan kelas {name} berbeda. Closed-set UDA mensyaratkan "
                "nama folder kelas source dan target identik."
            )

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("workers", 4)),
        "pin_memory": True,
    }
    loaders = {
        name: DataLoader(
            dataset,
            shuffle=name.endswith("train"),
            drop_last=name.endswith("train"),
            **loader_kwargs,
        )
        for name, dataset in datasets_by_split.items()
    }
    return DomainLoaders(
        source_train=loaders["source_train"],
        source_val=loaders["source_val"],
        target_train=loaders.get("target_train"),
        target_val=loaders.get("target_val"),
        classes=datasets_by_split["source_train"].classes,
    )
