from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from bilinear_lmmd.analysis.confusion_pairs import sample_identity
from bilinear_lmmd.engine.group_evaluation import (
    aggregate_group_class_probabilities,
    bootstrap_metric_samples,
    classification_metrics,
    interval,
)


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "worst_class_f1",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV kosong: {path}")
    return rows


def _indexed(rows: list[dict[str, str]], path_column: str) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        identity = sample_identity(row[path_column])
        if identity in output:
            raise ValueError(f"Identity duplikat: {identity}")
        output[identity] = row
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_sni_group_primary(
    manifest_root: Path,
    prediction_paths: dict[str, Path],
    output_dir: Path,
    *,
    split: str = "val",
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260727,
    allow_test: bool = False,
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Group-primary analysis hanya mendukung val/test.")
    if split == "test" and not allow_test:
        raise PermissionError(
            "Test terkunci. --allow-test hanya boleh dipakai setelah protokol "
            "secara eksplisit membuka test."
        )
    if not prediction_paths:
        raise ValueError("Minimal satu prediction path diperlukan.")

    audit_path = manifest_root / "audit.json"
    ontology_path = manifest_root / "ontology.json"
    manifest_path = manifest_root / "manifests" / f"{split}.csv"
    if not audit_path.is_file() or not ontology_path.is_file():
        raise FileNotFoundError("Manifest root v3 belum lengkap.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or not audit.get("test_locked"):
        raise ValueError("Audit v3 wajib complete dan test-locked.")
    classes = tuple(ontology["visual_classes"])

    manifest_table = _indexed(_read_csv(manifest_path), "crop_path")
    prediction_tables = {
        name: _indexed(_read_csv(path), "path")
        for name, path in prediction_paths.items()
    }
    identities = sorted(manifest_table)
    for name, table in prediction_tables.items():
        if set(table) != set(identities):
            raise ValueError(
                f"Identity manifest/prediction berbeda untuk {name}."
            )
    manifest_rows = [manifest_table[identity] for identity in identities]
    labels = [classes.index(row["visual_label"]) for row in manifest_rows]

    crop_metrics = {}
    bundles = {}
    group_metrics = {}
    domain_metrics = {}
    group_prediction_rows = {}
    for name, table in prediction_tables.items():
        rows = [table[identity] for identity in identities]
        if any(
            row["actual"] != manifest["visual_label"]
            for row, manifest in zip(rows, manifest_rows)
        ):
            raise ValueError(f"Actual label prediction tidak cocok untuk {name}.")
        probability_columns = [f"prob::{class_name}" for class_name in classes]
        missing = [column for column in probability_columns if column not in rows[0]]
        if missing:
            raise ValueError(f"Probability column kurang untuk {name}: {missing}")
        probabilities = np.asarray(
            [
                [float(row[column]) for column in probability_columns]
                for row in rows
            ],
            dtype=np.float64,
        )
        crop_predictions = probabilities.argmax(axis=1)
        crop_metrics[name] = classification_metrics(
            labels, crop_predictions, classes
        )
        bundle = aggregate_group_class_probabilities(
            manifest_rows, labels, probabilities, classes
        )
        bundles[name] = bundle
        group_metrics[name] = classification_metrics(
            bundle.labels, bundle.predictions, classes
        )
        domain_metrics[name] = {}
        for dataset in sorted({key[0] for key in bundle.unit_keys}):
            indices = np.asarray(
                [
                    index
                    for index, key in enumerate(bundle.unit_keys)
                    if key[0] == dataset
                ],
                dtype=np.int64,
            )
            domain_metrics[name][dataset] = classification_metrics(
                np.asarray(bundle.labels)[indices],
                np.asarray(bundle.predictions)[indices],
                classes,
            )
        group_prediction_rows[name] = [
            {
                "dataset": dataset,
                "group_id": group_id,
                "actual": actual,
                "predicted": classes[bundle.predictions[index]],
                "correct": int(
                    classes[bundle.predictions[index]] == actual
                ),
                "crop_count": bundle.crop_counts[index],
                **{
                    f"prob::{class_name}": float(
                        bundle.probabilities[index, class_index]
                    )
                    for class_index, class_name in enumerate(classes)
                },
            }
            for index, (dataset, group_id, actual) in enumerate(bundle.unit_keys)
        ]

    first_bundle = bundles[next(iter(bundles))]
    for name, bundle in bundles.items():
        if bundle.unit_keys != first_bundle.unit_keys:
            raise RuntimeError(f"Group unit berubah pada model {name}.")
        if bundle.labels != first_bundle.labels:
            raise RuntimeError(f"Group actual berubah pada model {name}.")
    bootstrap = bootstrap_metric_samples(
        first_bundle.labels,
        {
            name: bundle.predictions
            for name, bundle in bundles.items()
        },
        classes,
        first_bundle.cluster_keys,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    confidence_intervals = {
        name: {
            metric: interval(values)
            for metric, values in metric_samples.items()
        }
        for name, metric_samples in bootstrap.items()
    }
    comparisons = {}
    for baseline, candidate in combinations(prediction_paths, 2):
        key = f"{baseline}_vs_{candidate}"
        comparisons[key] = {}
        for metric in METRICS:
            deltas = (
                np.asarray(bootstrap[candidate][metric])
                - np.asarray(bootstrap[baseline][metric])
            )
            comparisons[key][metric] = {
                "baseline": group_metrics[baseline][metric],
                "candidate": group_metrics[candidate][metric],
                "delta": (
                    group_metrics[candidate][metric]
                    - group_metrics[baseline][metric]
                ),
                "bootstrap_delta": interval(deltas.tolist()),
                "probability_improved": float((deltas > 0).mean()),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in group_prediction_rows.items():
        _write_csv(output_dir / f"group_predictions_{name}.csv", rows)
    report = {
        "schema_version": 1,
        "analysis": "SNI v3 source-photograph x class evaluation",
        "split": split,
        "test_opened": split == "test",
        "primary_unit": "dataset x source group x visual class",
        "group_class_units": len(first_bundle.labels),
        "source_clusters": len(set(first_bundle.cluster_keys)),
        "bootstrap": {
            "method": "dataset-stratified source-group cluster bootstrap",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
        "group_class_metrics": group_metrics,
        "group_class_confidence_intervals": confidence_intervals,
        "crop_metrics_secondary": crop_metrics,
        "domain_group_class_metrics": domain_metrics,
        "comparisons": comparisons,
        "inputs": {
            "manifest_root": str(manifest_root),
            "predictions": {
                name: str(path) for name, path in prediction_paths.items()
            },
        },
    }
    (output_dir / "group_primary_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\n=== SNI GROUP-PRIMARY EVALUATION ===")
    print(
        f"Split={split} | units={len(first_bundle.labels):,} | "
        f"source_groups={len(set(first_bundle.cluster_keys)):,}"
    )
    for name in prediction_paths:
        group = group_metrics[name]
        crop = crop_metrics[name]
        print(
            f"{name}: GroupMacro={group['macro_f1']:.2%} "
            f"GroupWorst={group['worst_class_f1']:.2%} "
            f"CropMacro={crop['macro_f1']:.2%}"
        )
    for name, metrics in comparisons.items():
        print(
            f"{name}: DeltaGroupMacro={metrics['macro_f1']['delta']:+.2%} "
            f"P(>0)={metrics['macro_f1']['probability_improved']:.3f}"
        )
    print("SAVED:", output_dir / "group_primary_report.json")
    return report


def _prediction_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Gunakan CODE=/path/predictions.csv")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("CODE dan path tidak boleh kosong.")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SNI predictions per source photograph x class."
    )
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument(
        "--prediction",
        required=True,
        action="append",
        type=_prediction_argument,
        help="Repeatable CODE=/path/predictions.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    prediction_paths = dict(args.prediction)
    if len(prediction_paths) != len(args.prediction):
        raise ValueError("Kode prediction harus unik.")
    analyze_sni_group_primary(
        args.manifest_root,
        prediction_paths,
        args.output_dir,
        split=args.split,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        allow_test=args.allow_test,
    )


if __name__ == "__main__":
    main()
