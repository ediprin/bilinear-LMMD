from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from bilinear_lmmd.analysis.confusion_pairs import sample_identity


MODEL_CODES = ("S2G", "S2MR")
REQUIRED_MANIFEST_COLUMNS = {
    "crop_path",
    "visual_label",
    "dataset",
    "group_id",
    "source_identity",
}
REQUIRED_PREDICTION_COLUMNS = {"path", "actual", "predicted", "correct"}


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV kosong: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Kolom kurang pada {path}: {sorted(missing)}")
    return rows


def _indexed_rows(
    rows: list[dict[str, str]], path_column: str
) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = sample_identity(row[path_column])
        if identity in table:
            raise ValueError(f"Identity duplikat: {identity}")
        table[identity] = row
    return table


def _align(
    manifest_path: Path,
    prediction_paths: dict[str, Path],
) -> tuple[list[str], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    if set(prediction_paths) != set(MODEL_CODES):
        raise ValueError(f"Prediction wajib untuk {MODEL_CODES}.")
    manifest = _indexed_rows(
        _read_csv(manifest_path, REQUIRED_MANIFEST_COLUMNS), "crop_path"
    )
    predictions = {
        code: _indexed_rows(
            _read_csv(path, REQUIRED_PREDICTION_COLUMNS), "path"
        )
        for code, path in prediction_paths.items()
    }
    reference = set(manifest)
    for code, table in predictions.items():
        if set(table) != reference:
            raise ValueError(
                f"Identity {code} tidak sejajar dengan manifest: "
                f"missing={len(reference.difference(table))}, "
                f"extra={len(set(table).difference(reference))}"
            )
    identities = sorted(reference)
    manifest_rows = [manifest[identity] for identity in identities]
    prediction_rows = {
        code: [predictions[code][identity] for identity in identities]
        for code in MODEL_CODES
    }
    for position, manifest_row in enumerate(manifest_rows):
        expected = manifest_row["visual_label"]
        for code in MODEL_CODES:
            actual = prediction_rows[code][position]["actual"]
            if actual != expected:
                raise ValueError(
                    f"Ground truth tidak cocok untuk {identities[position]}: "
                    f"manifest={expected}, {code}={actual}"
                )
    return identities, manifest_rows, prediction_rows


def _classes(prediction_rows: list[dict[str, str]]) -> tuple[str, ...]:
    names = sorted(
        {
            row["actual"]
            for row in prediction_rows
        }
        | {
            row["predicted"]
            for row in prediction_rows
        }
    )
    if not names:
        raise ValueError("Tidak ada kelas prediction.")
    return tuple(names)


def _metric_bundle(
    actual: list[str], predicted: list[str], classes: tuple[str, ...]
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _class_f1(
    actual: list[str], predicted: list[str], class_name: str
) -> float:
    return float(
        f1_score(
            actual,
            predicted,
            labels=[class_name],
            average="macro",
            zero_division=0,
        )
    )


def _probability_columns(
    rows: list[dict[str, str]], classes: tuple[str, ...]
) -> tuple[str, ...] | None:
    columns = tuple(f"prob::{name}" for name in classes)
    return columns if all(column in rows[0] for column in columns) else None


def _group_class_predictions(
    manifest_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    classes: tuple[str, ...],
) -> tuple[list[str], list[str], int]:
    probability_columns = _probability_columns(prediction_rows, classes)
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (manifest, prediction) in enumerate(
        zip(manifest_rows, prediction_rows)
    ):
        key = (
            manifest["dataset"],
            manifest["group_id"],
            prediction["actual"],
        )
        grouped[key].append(index)

    actual: list[str] = []
    predicted: list[str] = []
    for (_, _, actual_name), indices in sorted(grouped.items()):
        actual.append(actual_name)
        if probability_columns is not None:
            probabilities = np.asarray(
                [
                    [
                        float(prediction_rows[index][column])
                        for column in probability_columns
                    ]
                    for index in indices
                ],
                dtype=np.float64,
            )
            predicted.append(classes[int(probabilities.mean(axis=0).argmax())])
        else:
            votes = Counter(prediction_rows[index]["predicted"] for index in indices)
            predicted.append(sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0])
    return actual, predicted, len(grouped)


def _outcome(actual: str, baseline: str, candidate: str) -> str:
    baseline_correct = baseline == actual
    candidate_correct = candidate == actual
    if baseline_correct and candidate_correct:
        return "both_correct"
    if baseline_correct:
        return "gap_only_correct"
    if candidate_correct:
        return "multires_only_correct"
    return "both_wrong"


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Tidak boleh menulis CSV kosong: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_sni_v2_failure(
    manifest_root: Path,
    prediction_paths: dict[str, Path],
    output_dir: Path,
) -> dict:
    audit_path = manifest_root / "audit.json"
    manifest_path = manifest_root / "manifests" / "val.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit SNI v2 tidak ditemukan: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or not audit.get("test_locked"):
        raise ValueError("Audit SNI v2 wajib complete dan test-locked.")

    identities, manifests, predictions = _align(
        manifest_path, prediction_paths
    )
    observed_classes = set(_classes(predictions["S2G"]))
    classes = tuple(audit["label_design"]["visual_v2_classes"])
    if observed_classes != set(classes):
        raise ValueError("Isi kelas prediction tidak cocok dengan audit.")

    actual = [row["actual"] for row in predictions["S2G"]]
    predicted = {
        code: [row["predicted"] for row in predictions[code]]
        for code in MODEL_CODES
    }
    crop_metrics = {
        code: _metric_bundle(actual, predicted[code], classes)
        for code in MODEL_CODES
    }
    group_metrics = {}
    group_unit_count = None
    for code in MODEL_CODES:
        group_actual, group_predicted, current_count = _group_class_predictions(
            manifests, predictions[code], classes
        )
        if group_unit_count is None:
            group_unit_count = current_count
        elif group_unit_count != current_count:
            raise RuntimeError("Jumlah group/class unit berubah antar-model.")
        group_metrics[code] = _metric_bundle(
            group_actual, group_predicted, classes
        )

    weak = audit["statistical_readiness"]["weak_classes"].get("val", {})
    class_rows: list[dict] = []
    sample_rows: list[dict] = []
    outcomes = Counter()
    for index, identity in enumerate(identities):
        outcome = _outcome(
            actual[index],
            predicted["S2G"][index],
            predicted["S2MR"][index],
        )
        outcomes[outcome] += 1
        sample_rows.append(
            {
                "identity": identity,
                "actual": actual[index],
                "dataset": manifests[index]["dataset"],
                "group_id": manifests[index]["group_id"],
                "source_identity": manifests[index]["source_identity"],
                "gap_prediction": predicted["S2G"][index],
                "multires_prediction": predicted["S2MR"][index],
                "outcome": outcome,
            }
        )

    for class_name in classes:
        indices = [
            index for index, value in enumerate(actual) if value == class_name
        ]
        class_outcomes = Counter(sample_rows[index]["outcome"] for index in indices)
        class_rows.append(
            {
                "class": class_name,
                "samples": len(indices),
                "groups": len({manifests[index]["group_id"] for index in indices}),
                "source_identities": len(
                    {manifests[index]["source_identity"] for index in indices}
                ),
                "adrian_samples": sum(
                    manifests[index]["dataset"] == "adrian_detection"
                    for index in indices
                ),
                "faruq_samples": sum(
                    manifests[index]["dataset"] == "faruq_segmentation"
                    for index in indices
                ),
                "gap_f1": _class_f1(actual, predicted["S2G"], class_name),
                "multires_f1": _class_f1(
                    actual, predicted["S2MR"], class_name
                ),
                "delta_f1": (
                    _class_f1(actual, predicted["S2MR"], class_name)
                    - _class_f1(actual, predicted["S2G"], class_name)
                ),
                "gap_only_correct": class_outcomes["gap_only_correct"],
                "multires_only_correct": class_outcomes["multires_only_correct"],
                "both_wrong": class_outcomes["both_wrong"],
                "weak_samples": not weak.get(class_name, {}).get(
                    "enough_samples", True
                ),
                "weak_groups": not weak.get(class_name, {}).get(
                    "enough_groups", True
                ),
            }
        )

    domain_rows: list[dict] = []
    class_domain_rows: list[dict] = []
    domain_metrics: dict[str, dict] = {}
    for dataset in sorted({row["dataset"] for row in manifests}):
        indices = [
            index
            for index, manifest in enumerate(manifests)
            if manifest["dataset"] == dataset
        ]
        domain_actual = [actual[index] for index in indices]
        present_classes = tuple(sorted(set(domain_actual)))
        row: dict[str, object] = {
            "dataset": dataset,
            "samples": len(indices),
            "groups": len({manifests[index]["group_id"] for index in indices}),
            "classes_present": len(present_classes),
        }
        domain_metrics[dataset] = {}
        for code in MODEL_CODES:
            values = _metric_bundle(
                domain_actual,
                [predicted[code][index] for index in indices],
                present_classes,
            )
            domain_metrics[dataset][code] = values
            row[f"{code.lower()}_accuracy"] = values["accuracy"]
            row[f"{code.lower()}_macro_f1"] = values["macro_f1"]
        row["delta_macro_f1"] = (
            domain_metrics[dataset]["S2MR"]["macro_f1"]
            - domain_metrics[dataset]["S2G"]["macro_f1"]
        )
        domain_rows.append(row)

        for class_name in present_classes:
            class_indices = [
                index for index in indices if actual[index] == class_name
            ]
            class_domain_rows.append(
                {
                    "dataset": dataset,
                    "class": class_name,
                    "samples": len(class_indices),
                    "groups": len(
                        {manifests[index]["group_id"] for index in class_indices}
                    ),
                    "gap_f1": _class_f1(
                        domain_actual,
                        [predicted["S2G"][index] for index in indices],
                        class_name,
                    ),
                    "multires_f1": _class_f1(
                        domain_actual,
                        [predicted["S2MR"][index] for index in indices],
                        class_name,
                    ),
                }
            )
            class_domain_rows[-1]["delta_f1"] = (
                class_domain_rows[-1]["multires_f1"]
                - class_domain_rows[-1]["gap_f1"]
            )

    confusion_rows: list[dict] = []
    for code in MODEL_CODES:
        pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, (actual_name, predicted_name) in enumerate(
            zip(actual, predicted[code])
        ):
            if actual_name != predicted_name:
                pairs[(actual_name, predicted_name)].append(index)
        for (actual_name, predicted_name), indices in pairs.items():
            confusion_rows.append(
                {
                    "model": code,
                    "actual": actual_name,
                    "predicted": predicted_name,
                    "errors": len(indices),
                    "groups": len(
                        {manifests[index]["group_id"] for index in indices}
                    ),
                    "adrian_errors": sum(
                        manifests[index]["dataset"] == "adrian_detection"
                        for index in indices
                    ),
                    "faruq_errors": sum(
                        manifests[index]["dataset"] == "faruq_segmentation"
                        for index in indices
                    ),
                }
            )
    confusion_rows.sort(
        key=lambda row: (-int(row["errors"]), row["model"], row["actual"])
    )

    domain_macro_values = {
        code: [
            domain_metrics[dataset][code]["macro_f1"]
            for dataset in sorted(domain_metrics)
        ]
        for code in MODEL_CODES
    }
    domain_gap = {
        code: max(values) - min(values)
        for code, values in domain_macro_values.items()
    }
    harmed_classes = sorted(
        class_rows, key=lambda row: float(row["delta_f1"])
    )
    report = {
        "schema_version": 1,
        "analysis": "SNI v2 multiresolution failure audit",
        "selection_split": "val",
        "test_opened": False,
        "diagnostic_only": True,
        "sample_count": len(identities),
        "group_class_unit_count": group_unit_count,
        "classes": list(classes),
        "crop_metrics": crop_metrics,
        "group_class_metrics": group_metrics,
        "outcomes": dict(outcomes),
        "domain_metrics": domain_metrics,
        "domain_macro_gap": domain_gap,
        "large_domain_association": {
            code: value >= 0.10 for code, value in domain_gap.items()
        },
        "weak_validation_classes": sorted(weak),
        "most_harmed_classes": [
            {
                "class": row["class"],
                "delta_f1": row["delta_f1"],
                "samples": row["samples"],
                "groups": row["groups"],
                "weak_samples": row["weak_samples"],
                "weak_groups": row["weak_groups"],
            }
            for row in harmed_classes[:5]
        ],
        "inputs": {
            "manifest": str(manifest_path),
            "predictions": {
                code: str(prediction_paths[code]) for code in MODEL_CODES
            },
        },
        "limitations": [
            "Validation seed 42 only; diagnostic, not model-selection evidence.",
            "Domain association does not prove shortcut learning or label error.",
            "Group/class aggregation reduces crop pseudoreplication but does not create new independent images.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sni_v2_failure_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_rows(output_dir / "class_diagnostics.csv", class_rows)
    _write_rows(output_dir / "domain_metrics.csv", domain_rows)
    _write_rows(output_dir / "class_domain_metrics.csv", class_domain_rows)
    _write_rows(output_dir / "confusion_pairs.csv", confusion_rows)
    _write_rows(output_dir / "sample_outcomes.csv", sample_rows)

    print("\n=== SNI V2 FAILURE AUDIT (VALIDATION SEED 42) ===")
    for level, metrics in (
        ("CROP", crop_metrics),
        ("GROUP/CLASS", group_metrics),
    ):
        print(f"\n{level}")
        for code in MODEL_CODES:
            print(
                f"{code:4s} Accuracy={metrics[code]['accuracy']:.2%} "
                f"Macro={metrics[code]['macro_f1']:.2%}"
            )
    print("\nOUTCOME:", dict(outcomes))
    print("\nDOMAIN")
    for row in domain_rows:
        print(
            f"{row['dataset']:20s} n={row['samples']:4d} "
            f"GAP={row['s2g_macro_f1']:.2%} "
            f"MultiRes={row['s2mr_macro_f1']:.2%} "
            f"Delta={row['delta_macro_f1']:+.2%}"
        )
    print("\nMOST HARMED")
    for row in harmed_classes[:5]:
        print(
            f"{row['class']:32s} DeltaF1={row['delta_f1']:+.2%} "
            f"n={row['samples']} groups={row['groups']} "
            f"weak={row['weak_samples'] or row['weak_groups']}"
        )
    print("\nTest dibuka: False")
    print("SAVED:", output_dir / "sni_v2_failure_audit.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-training audit for failed SNI v2 multiresolution."
    )
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analyze_sni_v2_failure(
        args.manifest_root,
        {
            "S2G": args.baseline_predictions,
            "S2MR": args.candidate_predictions,
        },
        args.output_dir,
    )


if __name__ == "__main__":
    main()
