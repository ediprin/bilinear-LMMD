from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bilinear_lmmd.analysis.sni_group_primary import (
    analyze_sni_group_primary,
)
from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model


MODEL_CONFIGS = {
    "S3J0": Path(
        "configs/sni_v3/S3J0_swin_tiny_gap_group_primary.yaml"
    ),
    "S3J1": Path(
        "configs/sni_v3/S3J1_swin_hssam_group_primary.yaml"
    ),
    "S3B0": Path(
        "configs/sni_v3/S3B0_efficientnetv2_gap_group_primary.yaml"
    ),
}
STAGE_MODELS = {
    "mechanism": ("S3J0", "S3J1"),
    "benchmark": ("S3B0",),
}
STAGE_COMPARISON = {
    "mechanism": ("S3J0", "S3J1"),
    "benchmark": ("S3B0", "S3J1"),
}
REPORT_FILES = (
    "metrics.json",
    "confusion_matrix.csv",
    "predictions.csv",
)


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return (
            len(json.loads(history_path.read_text(encoding="utf-8")))
            >= epochs
        )
    except (OSError, json.JSONDecodeError):
        return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_v3_manifest(manifest_root: Path) -> dict:
    audit_path = manifest_root / "audit.json"
    ontology_path = manifest_root / "ontology.json"
    required_manifests = {
        "train": manifest_root / "manifests" / "train_weighted.csv",
        "val": manifest_root / "manifests" / "val.csv",
    }
    if not audit_path.is_file() or not ontology_path.is_file():
        raise FileNotFoundError("audit.json/ontology.json v3 tidak lengkap.")
    missing = [
        str(path) for path in required_manifests.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Manifest v3 belum lengkap: {missing}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    classes = tuple(ontology.get("visual_classes", ()))
    if audit.get("status") != "complete" or audit.get("test_locked") is not True:
        raise ValueError("Audit v3 harus complete dan test-locked.")
    if len(classes) != 15:
        raise ValueError(f"V3.1 membutuhkan 15 visual class, didapat {len(classes)}.")

    train_rows = _read_csv(required_manifests["train"])
    val_rows = _read_csv(required_manifests["val"])
    required_columns = {
        "dataset",
        "group_id",
        "visual_label",
        "crop_path",
    }
    for name, rows in (("train", train_rows), ("val", val_rows)):
        if not rows or required_columns.difference(rows[0]):
            raise ValueError(f"Kolom manifest {name} tidak lengkap.")
    if "train_weight_group_class_equal" not in train_rows[0]:
        raise ValueError("Bobot train_weight_group_class_equal belum tersedia.")
    if any(not row["train_weight_group_class_equal"] for row in train_rows):
        raise ValueError("Bobot group-class train kosong.")

    class_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    dataset_groups: dict[str, set[str]] = defaultdict(set)
    for row in val_rows:
        class_groups[row["visual_label"]].add(
            (row["dataset"], row["group_id"])
        )
        dataset_groups[row["dataset"]].add(row["group_id"])
    weak_classes = {
        name: len(class_groups[name])
        for name in classes
        if len(class_groups[name]) < 20
    }
    weak_datasets = {
        name: len(groups)
        for name, groups in dataset_groups.items()
        if len(groups) < 50
    }
    if weak_classes or weak_datasets:
        raise ValueError(
            "Manifest tidak memenuhi kesiapan group-primary: "
            f"classes={weak_classes}, datasets={weak_datasets}"
        )
    return {
        "classes": classes,
        "train_crops": len(train_rows),
        "val_crops": len(val_rows),
        "val_source_groups": len(
            {
                (row["dataset"], row["group_id"])
                for row in val_rows
            }
        ),
        "val_class_groups": {
            name: len(class_groups[name]) for name in classes
        },
        "val_dataset_groups": {
            name: len(groups) for name, groups in dataset_groups.items()
        },
    }


def _model_audits() -> dict:
    output = {}
    for code, path in MODEL_CONFIGS.items():
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        output[code] = {
            "backbone": cfg["model"]["backbone"],
            "head": cfg["model"]["head"],
            "loss": cfg["training"]["classification_loss"],
            "selection_metric": cfg["evaluation"]["selection_metric"],
            "train_weight": cfg["data"]["manifest_weight_column"],
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    return output


def _restore(
    artifact_repo: str,
    artifact_namespace: str,
    code: str,
    seed: int,
    run_dir: Path,
    report_dir: Path,
) -> None:
    run_path = f"{artifact_namespace}/outputs/{code}_seed{seed}"
    report_path = f"{artifact_namespace}/crop_reports/{code}_seed{seed}"
    restored = restore_artifacts(
        artifact_repo, run_path, run_dir, overwrite=False
    )
    restored += restore_artifacts(
        artifact_repo,
        report_path,
        report_dir,
        filenames=REPORT_FILES,
        overwrite=False,
    )
    print(f"HF RESTORE {code} seed {seed}: {len(restored)} file", flush=True)


def _train_and_evaluate(
    code: str,
    seed: int,
    data_root: Path,
    manifest_root: Path,
    output_root: Path,
    artifact_repo: str | None,
    artifact_namespace: str,
    artifact_sync_every: int,
) -> None:
    config_path = MODEL_CONFIGS[code]
    epochs = int(load_config(config_path)["training"]["epochs"])
    run_dir = output_root / "outputs" / f"{code}_seed{seed}"
    report_dir = output_root / "crop_reports" / f"{code}_seed{seed}"
    artifact_run_path = f"{artifact_namespace}/outputs/{code}_seed{seed}"
    artifact_report_path = (
        f"{artifact_namespace}/crop_reports/{code}_seed{seed}"
    )
    if artifact_repo:
        _restore(
            artifact_repo,
            artifact_namespace,
            code,
            seed,
            run_dir,
            report_dir,
        )
    if _training_complete(run_dir, epochs):
        print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
    else:
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.engine.train",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--data-root",
                str(data_root),
                "--manifest-root",
                str(manifest_root),
                "--output-dir",
                str(run_dir),
                "--resume",
                *(
                    [
                        "--artifact-repo",
                        artifact_repo,
                        "--artifact-path",
                        artifact_run_path,
                        "--artifact-sync-every",
                        str(artifact_sync_every),
                        "--artifact-required",
                    ]
                    if artifact_repo
                    else []
                ),
            ]
        )
    if not (report_dir / "predictions.csv").is_file():
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.engine.evaluate_checkpoint",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--domain",
                "source",
                "--split",
                "val",
                "--data-root",
                str(data_root),
                "--manifest-root",
                str(manifest_root),
                "--output-dir",
                str(report_dir),
            ]
        )
    if artifact_repo:
        sync_artifacts(
            artifact_repo,
            artifact_report_path,
            report_dir,
            filenames=REPORT_FILES,
            commit_message=f"SNI v3.1 evaluation {code} seed {seed}",
        )


def _positive_requirement(seed_count: int) -> int:
    return max(1, math.ceil(seed_count * 2.0 / 3.0))


def _aggregate_decision(
    reports: dict[int, dict],
    baseline: str,
    candidate: str,
) -> dict:
    key = f"{baseline}_vs_{candidate}"
    macro_deltas = []
    worst_deltas = []
    domain_deltas: dict[str, list[float]] = defaultdict(list)
    per_seed = {}
    for seed, report in sorted(reports.items()):
        comparison = report["comparisons"][key]
        macro = float(comparison["macro_f1"]["delta"])
        worst = float(comparison["worst_class_f1"]["delta"])
        macro_deltas.append(macro)
        worst_deltas.append(worst)
        per_seed[seed] = {
            "macro_f1": macro,
            "worst_class_f1": worst,
        }
        domains = report["domain_group_class_metrics"]
        for dataset in sorted(domains[baseline]):
            delta = (
                float(domains[candidate][dataset]["macro_f1"])
                - float(domains[baseline][dataset]["macro_f1"])
            )
            domain_deltas[dataset].append(delta)
            per_seed[seed][f"domain::{dataset}::macro_f1"] = delta

    macro_array = np.asarray(macro_deltas)
    worst_array = np.asarray(worst_deltas)
    required = _positive_requirement(len(reports))
    criteria = {
        "group_macro_mean_improved": float(macro_array.mean()) > 0.0,
        "group_macro_required_seeds": int((macro_array > 0.0).sum())
        >= required,
        "group_worst_mean_preserved": float(worst_array.mean()) >= -0.01,
        "each_domain_macro_preserved": all(
            float(np.mean(values)) >= -0.01
            for values in domain_deltas.values()
        ),
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "required_positive_seeds": required,
        "per_seed": per_seed,
        "summary": {
            "group_macro_delta_mean": float(macro_array.mean()),
            "group_macro_delta_std": float(macro_array.std(ddof=0)),
            "group_worst_delta_mean": float(worst_array.mean()),
            "group_worst_delta_std": float(worst_array.std(ddof=0)),
            "domain_macro_delta_mean": {
                dataset: float(np.mean(values))
                for dataset, values in domain_deltas.items()
            },
        },
    }


def run_sni_v3_jiao_screening(
    data_root: Path,
    manifest_root: Path,
    output_root: Path,
    seeds: list[int],
    *,
    stage: str = "mechanism",
    artifact_repo: str | None = None,
    artifact_namespace: str = "sni-v3-jiao-group-primary-v1",
    artifact_sync_every: int = 1,
    bootstrap_iterations: int = 2000,
) -> dict:
    if stage not in STAGE_MODELS:
        raise ValueError(f"Stage harus salah satu dari {tuple(STAGE_MODELS)}.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seed harus tidak kosong dan unik.")
    if artifact_sync_every <= 0:
        raise ValueError("artifact_sync_every harus positif.")
    readiness = validate_v3_manifest(manifest_root)
    if artifact_repo:
        ensure_artifact_repo(artifact_repo, private=True)
    audits = _model_audits()
    print("=== SNI V3.1 JIAO FAIL-FAST ===", flush=True)
    print(
        f"Stage={stage} | seeds={seeds} | test locked | "
        f"group units from {readiness['val_source_groups']:,} source images",
        flush=True,
    )
    for code in (*STAGE_MODELS[stage],):
        row = audits[code]
        print(
            f"{code}: {row['backbone']} + {row['head']} | "
            f"params={row['parameters']:,}",
            flush=True,
        )

    if stage == "benchmark":
        mechanism_path = output_root / "decisions" / "mechanism.json"
        if not mechanism_path.is_file():
            raise FileNotFoundError(
                "Stage benchmark memerlukan keputusan mechanism."
            )
        mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
        if mechanism["decision"]["decision"] != "PASS":
            raise RuntimeError("Mechanism FAIL; benchmark tidak diizinkan.")

    for code in STAGE_MODELS[stage]:
        for seed in seeds:
            _train_and_evaluate(
                code,
                seed,
                data_root,
                manifest_root,
                output_root,
                artifact_repo,
                artifact_namespace,
                artifact_sync_every,
            )

    baseline, candidate = STAGE_COMPARISON[stage]
    group_reports = {}
    for seed in seeds:
        prediction_paths = {
            baseline: (
                output_root
                / "crop_reports"
                / f"{baseline}_seed{seed}"
                / "predictions.csv"
            ),
            candidate: (
                output_root
                / "crop_reports"
                / f"{candidate}_seed{seed}"
                / "predictions.csv"
            ),
        }
        missing = [str(path) for path in prediction_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Prediksi pembanding belum tersedia: {missing}"
            )
        group_reports[seed] = analyze_sni_group_primary(
            manifest_root,
            prediction_paths,
            output_root / "group_reports" / f"seed{seed}" / stage,
            split="val",
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=20260727 + seed,
        )
    decision = _aggregate_decision(
        group_reports, baseline, candidate
    )
    report = {
        "protocol": "SNI v3.1 Jiao Swin-HSSAM fail-fast",
        "stage": stage,
        "seeds": seeds,
        "selection_split": "val",
        "test_opened": False,
        "primary_unit": "dataset x source photograph x visual class",
        "readiness": readiness,
        "model_audits": audits,
        "comparison": f"{baseline}_vs_{candidate}",
        "decision": decision,
    }
    destination = output_root / "decisions" / f"{stage}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if artifact_repo:
        sync_artifacts(
            artifact_repo,
            f"{artifact_namespace}/decisions",
            destination.parent,
            filenames=(destination.name,),
            commit_message=f"SNI v3.1 Jiao {stage} decision",
        )
    summary = decision["summary"]
    print("\n=== PUTUSAN SNI V3.1 JIAO ===")
    print("Comparison:", report["comparison"])
    print(
        f"Group Macro delta: "
        f"{summary['group_macro_delta_mean']:+.2%}"
    )
    print(
        f"Group Worst delta: "
        f"{summary['group_worst_delta_mean']:+.2%}"
    )
    print("Domain delta:", summary["domain_macro_delta_mean"])
    print("DECISION:", decision["decision"], decision["criteria"])
    print("TEST DIBUKA: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNI v3.1 group-primary Jiao fail-fast screening"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_MODELS),
        default="mechanism",
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    parser.add_argument("--artifact-repo")
    parser.add_argument(
        "--artifact-namespace",
        default="sni-v3-jiao-group-primary-v1",
    )
    parser.add_argument("--artifact-sync-every", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    run_sni_v3_jiao_screening(
        args.data_root,
        args.manifest_root,
        args.output_root,
        args.seeds,
        stage=args.stage,
        artifact_repo=args.artifact_repo,
        artifact_namespace=args.artifact_namespace,
        artifact_sync_every=args.artifact_sync_every,
        bootstrap_iterations=args.bootstrap_iterations,
    )


if __name__ == "__main__":
    main()
