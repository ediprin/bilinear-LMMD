from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from bilinear_lmmd.data.preparation.prepare_sni_classification_v2 import (
    ATTRIBUTE_NAMES,
    DATASETS,
    SPLITS,
    VISUAL_CLASSES,
    _normalized_weights,
    _read_manifest,
    _validate_rows,
    _write_csv,
    family_label,
    partial_attributes,
    size_label,
    visual_label,
)
from bilinear_lmmd.data.sni_ontology import SNI_CLASSES


DEFAULT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class GroupSummary:
    group_id: str
    datasets: tuple[str, ...]
    crop_count: int
    dataset_crop_counts: Counter[str]
    class_crop_counts: Counter[str]


def _largest_remainder_counts(
    total: int,
    ratios: tuple[float, float, float],
) -> dict[str, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(SPLITS)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def _group_summaries(rows: list[dict[str, str]]) -> dict[str, GroupSummary]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    summaries = {}
    for group_id, group_rows in grouped.items():
        summaries[group_id] = GroupSummary(
            group_id=group_id,
            datasets=tuple(sorted({row["dataset"] for row in group_rows})),
            crop_count=len(group_rows),
            dataset_crop_counts=Counter(row["dataset"] for row in group_rows),
            class_crop_counts=Counter(row["visual_label"] for row in group_rows),
        )
    return summaries


def _candidate_assignment(
    summaries: dict[str, GroupSummary],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Allocate exact source-group ratios inside stable visual strata.

    Most source photographs contain one visual class. Stratifying those groups
    by class guarantees that an otherwise rare class is not accidentally absent
    from a held-out split. Dense multi-object photographs are stratified
    together by domain signature so that they are distributed across all three
    splits instead of being forced into train by a unique multilabel signature.
    Dataset balance for the single-class strata is selected by the multi-trial
    objective rather than hard-coding the old archive split.
    """

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for group_id, summary in summaries.items():
        if len(summary.class_crop_counts) == 1:
            class_name = next(iter(summary.class_crop_counts))
            stratum = ("single_class", class_name)
        else:
            stratum = ("multilabel", *summary.datasets)
        strata[stratum].append(group_id)
    generator = random.Random(seed)
    assignments: dict[str, str] = {}
    for signature in sorted(strata):
        group_ids = sorted(strata[signature])
        generator.shuffle(group_ids)
        quotas = _largest_remainder_counts(len(group_ids), ratios)
        cursor = 0
        for split in SPLITS:
            stop = cursor + quotas[split]
            for group_id in group_ids[cursor:stop]:
                assignments[group_id] = split
            cursor = stop
    if set(assignments) != set(summaries):
        raise RuntimeError("Candidate assignment tidak mencakup seluruh grup.")
    return assignments


def _allocation_statistics(
    summaries: dict[str, GroupSummary],
    assignments: dict[str, str],
) -> dict:
    crop_counts = Counter()
    group_counts = Counter()
    dataset_crop_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    dataset_group_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    class_crop_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    class_group_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    class_group_crop_counts: dict[str, dict[str, list[int]]] = {
        split: defaultdict(list) for split in SPLITS
    }
    largest_group_crops = Counter()

    for group_id, summary in summaries.items():
        split = assignments[group_id]
        crop_counts[split] += summary.crop_count
        group_counts[split] += 1
        largest_group_crops[split] = max(
            largest_group_crops[split], summary.crop_count
        )
        dataset_crop_counts[split].update(summary.dataset_crop_counts)
        for dataset in summary.datasets:
            dataset_group_counts[split][dataset] += 1
        class_crop_counts[split].update(summary.class_crop_counts)
        for class_name, count in summary.class_crop_counts.items():
            class_group_counts[split][class_name] += 1
            class_group_crop_counts[split][class_name].append(count)

    return {
        "crop_counts": crop_counts,
        "group_counts": group_counts,
        "dataset_crop_counts": dataset_crop_counts,
        "dataset_group_counts": dataset_group_counts,
        "class_crop_counts": class_crop_counts,
        "class_group_counts": class_group_counts,
        "class_group_crop_counts": class_group_crop_counts,
        "largest_group_crops": largest_group_crops,
    }


def _relative_squared_error(observed: int, target: float) -> float:
    return ((observed - target) / max(target, 1.0)) ** 2


def _candidate_rank(
    statistics: dict,
    *,
    ratios: tuple[float, float, float],
    min_eval_groups_per_class: int,
    min_eval_groups_per_dataset: int,
) -> tuple[int, float]:
    total_crops = sum(statistics["crop_counts"].values())
    total_groups = sum(statistics["group_counts"].values())
    dataset_crop_totals = Counter()
    dataset_group_totals = Counter()
    class_crop_totals = Counter()
    class_group_totals = Counter()
    for split in SPLITS:
        dataset_crop_totals.update(statistics["dataset_crop_counts"][split])
        dataset_group_totals.update(statistics["dataset_group_counts"][split])
        class_crop_totals.update(statistics["class_crop_counts"][split])
        class_group_totals.update(statistics["class_group_counts"][split])

    hard_deficit = 0
    for split in ("val", "test"):
        for class_name in VISUAL_CLASSES:
            total = class_group_totals[class_name]
            if total >= 2 * min_eval_groups_per_class + 1:
                hard_deficit += max(
                    0,
                    min_eval_groups_per_class
                    - statistics["class_group_counts"][split][class_name],
                )
        for dataset in DATASETS:
            total = dataset_group_totals[dataset]
            if total >= 2 * min_eval_groups_per_dataset + 1:
                hard_deficit += max(
                    0,
                    min_eval_groups_per_dataset
                    - statistics["dataset_group_counts"][split][dataset],
                )

    score = 0.0
    for split, ratio in zip(SPLITS, ratios):
        score += _relative_squared_error(
            statistics["crop_counts"][split], total_crops * ratio
        )
        score += 4.0 * _relative_squared_error(
            statistics["group_counts"][split], total_groups * ratio
        )
        for dataset in DATASETS:
            score += 1.5 * _relative_squared_error(
                statistics["dataset_crop_counts"][split][dataset],
                dataset_crop_totals[dataset] * ratio,
            )
            score += 5.0 * _relative_squared_error(
                statistics["dataset_group_counts"][split][dataset],
                dataset_group_totals[dataset] * ratio,
            )
        for class_name in VISUAL_CLASSES:
            score += 1.5 * _relative_squared_error(
                statistics["class_crop_counts"][split][class_name],
                class_crop_totals[class_name] * ratio,
            )
            score += 8.0 * _relative_squared_error(
                statistics["class_group_counts"][split][class_name],
                class_group_totals[class_name] * ratio,
            )
            class_total = statistics["class_crop_counts"][split][class_name]
            group_values = statistics["class_group_crop_counts"][split][
                class_name
            ]
            if class_total and group_values:
                concentration = max(group_values) / class_total
                score += 20.0 * max(0.0, concentration - 0.25) ** 2
    return hard_deficit, score


def allocate_source_balanced_groups(
    rows: list[dict[str, str]],
    *,
    seed: int,
    trials: int = 512,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    min_eval_groups_per_class: int = 20,
    min_eval_groups_per_dataset: int = 50,
) -> tuple[dict[str, str], dict]:
    if trials <= 0:
        raise ValueError("Jumlah allocation trials harus positif.")
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Rasio split harus berjumlah satu.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Semua rasio split harus positif.")
    summaries = _group_summaries(rows)
    if len(summaries) < len(SPLITS):
        raise ValueError("Jumlah grup sumber tidak cukup untuk tiga split.")

    best_assignment = None
    best_statistics = None
    best_rank = None
    best_trial = None
    progress_every = max(1, trials // 8)
    for trial in range(trials):
        candidate = _candidate_assignment(
            summaries,
            ratios=ratios,
            seed=seed + trial * 104_729,
        )
        statistics = _allocation_statistics(summaries, candidate)
        rank = _candidate_rank(
            statistics,
            ratios=ratios,
            min_eval_groups_per_class=min_eval_groups_per_class,
            min_eval_groups_per_dataset=min_eval_groups_per_dataset,
        )
        candidate_rank = (*rank, trial)
        if best_rank is None or candidate_rank < best_rank:
            best_assignment = candidate
            best_statistics = statistics
            best_rank = candidate_rank
            best_trial = trial
        if trials >= 64 and (
            (trial + 1) % progress_every == 0 or trial + 1 == trials
        ):
            print(
                f"SPLIT SEARCH {trial + 1}/{trials} | "
                f"best_deficit={best_rank[0]} "
                f"best_objective={best_rank[1]:.6f}",
                flush=True,
            )
    if best_assignment is None or best_statistics is None:
        raise RuntimeError("Tidak ada candidate split yang dihasilkan.")
    return best_assignment, {
        "trials": trials,
        "selected_trial": best_trial,
        "hard_deficit": best_rank[0],
        "objective": best_rank[1],
        "statistics": best_statistics,
    }


def _normalized_row_weights(raw: list[float]) -> list[float]:
    mean = sum(raw) / len(raw)
    if mean <= 0:
        raise ValueError("Bobot train harus memiliki mean positif.")
    return [value / mean for value in raw]


def _training_weights(rows: list[dict]) -> dict[str, dict[str, float]]:
    class_counts = Counter(row["visual_label"] for row in rows)
    inverse_sqrt = _normalized_weights(
        class_counts, method="inverse_sqrt", beta=0.9999
    )
    group_counts = Counter(row["group_id"] for row in rows)
    group_class_counts = Counter(
        (row["group_id"], row["visual_label"]) for row in rows
    )
    class_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        class_groups[row["visual_label"]].add(row["group_id"])

    group_raw = [1.0 / group_counts[row["group_id"]] for row in rows]
    group_class_raw = [
        1.0
        / (
            len(class_groups[row["visual_label"]])
            * group_class_counts[(row["group_id"], row["visual_label"])]
        )
        for row in rows
    ]
    group_weights = _normalized_row_weights(group_raw)
    group_class_weights = _normalized_row_weights(group_class_raw)
    by_path = {}
    for row, group_weight, group_class_weight in zip(
        rows, group_weights, group_class_weights
    ):
        by_path[row["crop_path"]] = {
            "train_weight_inverse_sqrt": inverse_sqrt[row["visual_label"]],
            "train_weight_group_equal": group_weight,
            "train_weight_group_class_equal": group_class_weight,
        }
    return by_path


def _enrich_rows(source_rows: list[dict[str, str]]) -> list[dict]:
    enriched = []
    for row in source_rows:
        class_name = row["canonical_class"]
        width = float(row["bbox_width"])
        height = float(row["bbox_height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"BBox non-positif pada {row['crop_path']}")
        enriched.append(
            {
                "crop_path": row["crop_path"],
                "generated_split": row["generated_split"],
                "v1_generated_split": row["generated_split"],
                "dataset": row["dataset"],
                "group_id": row["group_id"],
                "source_identity": row["source_identity"],
                "flat_label_v1": class_name,
                "visual_label": visual_label(class_name),
                "family_label": family_label(class_name),
                "size_label_metadata": size_label(class_name),
                "size_visual_supervision_allowed": 0,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "bbox_aspect_ratio": max(width, height) / min(width, height),
                "crop_sha256": row["crop_sha256"],
                **{
                    f"attr_{name}": value
                    for name, value in partial_attributes(class_name).items()
                },
            }
        )
    return enriched


def _split_statistics(rows: list[dict]) -> dict:
    class_counts = Counter(row["visual_label"] for row in rows)
    class_groups: dict[str, set[str]] = defaultdict(set)
    dataset_counts = Counter(row["dataset"] for row in rows)
    dataset_groups: dict[str, set[str]] = defaultdict(set)
    group_counts = Counter(row["group_id"] for row in rows)
    group_class_counts = Counter(
        (row["group_id"], row["visual_label"]) for row in rows
    )
    for row in rows:
        class_groups[row["visual_label"]].add(row["group_id"])
        dataset_groups[row["dataset"]].add(row["group_id"])
    max_class_group_share = {}
    for class_name in VISUAL_CLASSES:
        total = class_counts[class_name]
        values = [
            count
            for (group_id, label), count in group_class_counts.items()
            if label == class_name
        ]
        max_class_group_share[class_name] = max(values, default=0) / max(
            total, 1
        )
    return {
        "crops": len(rows),
        "source_groups": len(group_counts),
        "visual_class_counts": dict(sorted(class_counts.items())),
        "visual_class_group_counts": {
            name: len(class_groups[name]) for name in VISUAL_CLASSES
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "dataset_group_counts": {
            dataset: len(dataset_groups[dataset]) for dataset in DATASETS
        },
        "largest_group_crops": max(group_counts.values()),
        "largest_group_fraction": max(group_counts.values()) / len(rows),
        "max_group_fraction_by_class": max_class_group_share,
    }


def _ontology() -> dict:
    return {
        "schema_version": 3,
        "visual_classes": list(VISUAL_CLASSES),
        "flat_to_visual": {
            class_name: visual_label(class_name) for class_name in SNI_CLASSES
        },
        "flat_to_family": {
            class_name: family_label(class_name) for class_name in SNI_CLASSES
        },
        "flat_to_size_metadata": {
            class_name: size_label(class_name) for class_name in SNI_CLASSES
        },
        "partial_attribute_values": {
            class_name: partial_attributes(class_name)
            for class_name in SNI_CLASSES
        },
        "attribute_names": list(ATTRIBUTE_NAMES),
        "unknown_attribute_value": -1,
        "size_visual_supervision_allowed": False,
    }


def prepare_sni_classification_v3(
    input_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    trials: int = 512,
    min_eval_samples_per_class: int = 50,
    min_eval_groups_per_class: int = 20,
    min_eval_groups_per_dataset: int = 50,
    max_single_group_fraction_per_class: float = 0.25,
    metadata_only: bool = False,
) -> dict:
    if not 0.0 < max_single_group_fraction_per_class <= 1.0:
        raise ValueError(
            "Maximum single-group fraction per class harus berada pada (0, 1]."
        )
    source_rows, source_audit = _read_manifest(input_root)
    if metadata_only:
        integrity = {
            "crop_files_checked": 0,
            "source_complete_audit_trusted": True,
            "metadata_only": True,
        }
    else:
        integrity = {
            **_validate_rows(input_root, source_rows),
            "source_complete_audit_trusted": False,
            "metadata_only": False,
        }
    enriched = _enrich_rows(source_rows)
    assignments, allocation = allocate_source_balanced_groups(
        enriched,
        seed=seed,
        trials=trials,
        min_eval_groups_per_class=min_eval_groups_per_class,
        min_eval_groups_per_dataset=min_eval_groups_per_dataset,
    )
    for row in enriched:
        row["generated_split"] = assignments[row["group_id"]]

    split_rows = {
        split: [row for row in enriched if row["generated_split"] == split]
        for split in SPLITS
    }
    train_weights = _training_weights(split_rows["train"])
    for row in enriched:
        if row["generated_split"] == "train":
            row.update(train_weights[row["crop_path"]])
        else:
            row["train_weight_inverse_sqrt"] = ""
            row["train_weight_group_equal"] = ""
            row["train_weight_group_class_equal"] = ""

    statistics = {
        split: _split_statistics(rows) for split, rows in split_rows.items()
    }
    weak_classes = {}
    weak_datasets = {}
    concentration_failures = {}
    for split in ("val", "test"):
        weak_classes[split] = {
            class_name: {
                "samples": statistics[split]["visual_class_counts"].get(
                    class_name, 0
                ),
                "groups": statistics[split][
                    "visual_class_group_counts"
                ].get(class_name, 0),
            }
            for class_name in VISUAL_CLASSES
            if statistics[split]["visual_class_counts"].get(class_name, 0)
            < min_eval_samples_per_class
            or statistics[split]["visual_class_group_counts"].get(
                class_name, 0
            )
            < min_eval_groups_per_class
        }
        weak_datasets[split] = {
            dataset: statistics[split]["dataset_group_counts"].get(dataset, 0)
            for dataset in DATASETS
            if statistics[split]["dataset_group_counts"].get(dataset, 0)
            < min_eval_groups_per_dataset
        }
        concentration_failures[split] = {
            class_name: share
            for class_name, share in statistics[split][
                "max_group_fraction_by_class"
            ].items()
            if share > max_single_group_fraction_per_class
        }

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in enriched:
        group_splits[row["group_id"]].add(row["generated_split"])
    cross_split_groups = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    assignment_digest = hashlib.sha256(
        "\n".join(
            f"{group_id}={assignments[group_id]}"
            for group_id in sorted(assignments)
        ).encode("utf-8")
    ).hexdigest()
    split_gate = (
        "PASS"
        if not any(weak_classes.values())
        and not any(weak_datasets.values())
        and not any(concentration_failures.values())
        and not cross_split_groups
        else "FAIL"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "manifests" / "all.csv", enriched)
    for split, rows in split_rows.items():
        _write_csv(output_root / "manifests" / f"{split}.csv", rows)
    _write_csv(
        output_root / "manifests" / "train_weighted.csv",
        split_rows["train"],
    )
    (output_root / "ontology.json").write_text(
        json.dumps(_ontology(), indent=2), encoding="utf-8"
    )
    audit = {
        "status": "complete",
        "protocol": "SNI_classification_source_group_balanced_v3",
        "source_protocol": source_audit["protocol"],
        "input_root": str(input_root),
        "input_crops": len(enriched),
        "integrity": {
            **integrity,
            "generated_cross_split_groups": len(cross_split_groups),
        },
        "label_design": {
            "visual_v2_classes": list(VISUAL_CLASSES),
            "visual_v2_num_classes": len(VISUAL_CLASSES),
            "size_policy": "same frozen 15-class visual target as v2",
        },
        "split_design": {
            "seed": seed,
            "ratios": dict(zip(SPLITS, DEFAULT_RATIOS)),
            "allocation_trials": trials,
            "selected_trial": allocation["selected_trial"],
            "objective": allocation["objective"],
            "hard_deficit": allocation["hard_deficit"],
            "assignment_sha256": assignment_digest,
            "policy": (
                "exact source-group quotas per dataset signature; candidate "
                "selection balances crop, dataset, class, source-group, and "
                "within-class group concentration"
            ),
        },
        "split_statistics": statistics,
        "training_balance": {
            "files_copied_deleted_or_reencoded": False,
            "available_weight_columns": [
                "train_weight_inverse_sqrt",
                "train_weight_group_equal",
                "train_weight_group_class_equal",
            ],
            "recommended_for_next_controlled_ablation": (
                "freeze one weighting rule before training; do not select it "
                "using test"
            ),
        },
        "statistical_readiness": {
            "minimum_eval_samples_per_class": min_eval_samples_per_class,
            "minimum_eval_groups_per_class": min_eval_groups_per_class,
            "minimum_eval_groups_per_dataset": min_eval_groups_per_dataset,
            "maximum_single_group_fraction_per_class": (
                max_single_group_fraction_per_class
            ),
            "weak_classes": weak_classes,
            "weak_datasets": weak_datasets,
            "concentration_failures": concentration_failures,
            "split_gate": split_gate,
        },
        "evaluation_policy": {
            "primary": "source-group/class Macro-F1",
            "secondary": "crop-level Macro-F1",
            "uncertainty": "cluster bootstrap by source group",
        },
        "test_locked": True,
        "training_authorized": False,
        "next_required_action": (
            "Review split gate and per-class/source-domain group counts. "
            "Training remains blocked until the protocol freezes the split "
            "and one train weighting rule."
        ),
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("\n=== SNI SOURCE-GROUP-BALANCED V3 ===")
    print("Input crops :", f"{len(enriched):,}")
    for split in SPLITS:
        row = statistics[split]
        print(
            f"{split:5s}: crops={row['crops']:,} "
            f"groups={row['source_groups']:,} "
            f"domain_groups={row['dataset_group_counts']}"
        )
    print("Weak val classes :", len(weak_classes["val"]))
    print("Weak test classes:", len(weak_classes["test"]))
    print("Split gate       :", split_gate)
    print("Training authorized: False")
    print("SAVED:", output_root / "audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a no-copy SNI v3 manifest with source-group-balanced splits."
        )
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--min-eval-samples-per-class", type=int, default=50)
    parser.add_argument("--min-eval-groups-per-class", type=int, default=20)
    parser.add_argument("--min-eval-groups-per-dataset", type=int, default=50)
    parser.add_argument(
        "--max-single-group-fraction-per-class", type=float, default=0.25
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Trust the complete v1 audit and skip checking 31k crop files. "
            "Use only for split construction; training still needs image root."
        ),
    )
    args = parser.parse_args()
    prepare_sni_classification_v3(
        args.input_root,
        args.output_root,
        seed=args.seed,
        trials=args.trials,
        min_eval_samples_per_class=args.min_eval_samples_per_class,
        min_eval_groups_per_class=args.min_eval_groups_per_class,
        min_eval_groups_per_dataset=args.min_eval_groups_per_dataset,
        max_single_group_fraction_per_class=(
            args.max_single_group_fraction_per_class
        ),
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
