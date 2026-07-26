from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "S2G": Path("configs/sni_v2/S2G_efficientnetv2_gap_weighted.yaml"),
    "S2MR": Path(
        "configs/sni_v2/S2MR_efficientnetv2_multiresolution_weighted.yaml"
    ),
}
SCREENING_SEEDS = (42,)
CONFIRMATION_SEEDS = (123, 2026)


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _bottom_three(report: dict) -> float:
    scores = sorted(float(row["f1"]) for row in report["per_class"].values())
    return statistics.mean(scores[:3])


def _paired_summary(baseline: list[float], candidate: list[float]) -> dict:
    deltas = [new - old for old, new in zip(baseline, candidate)]
    return {
        "baseline_mean": statistics.mean(baseline),
        "candidate_mean": statistics.mean(candidate),
        "delta_mean": statistics.mean(deltas),
        "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "improved_seeds": sum(delta > 0.0 for delta in deltas),
        "total_seeds": len(deltas),
        "deltas": deltas,
    }


def _metrics_path(output_root: Path, code: str, seed: int) -> Path:
    return output_root / "val_reports" / f"{code}_seed{seed}" / "metrics.json"


def _restore(
    repo: str,
    namespace: str,
    code: str,
    seed: int,
    run_dir: Path,
    report_dir: Path,
) -> None:
    restored = restore_artifacts(
        repo,
        f"{namespace}/outputs/{code}_seed{seed}",
        run_dir,
        overwrite=False,
    )
    restored += restore_artifacts(
        repo,
        f"{namespace}/val_reports/{code}_seed{seed}",
        report_dir,
        filenames=("metrics.json", "confusion_matrix.csv", "predictions.csv"),
        overwrite=False,
    )
    print(f"HF RESTORE {code} seed {seed}: {len(restored)} file", flush=True)


def _evaluate(
    checkpoint: Path,
    report_dir: Path,
    image_root: Path,
    manifest_root: Path,
) -> None:
    if report_dir.joinpath("metrics.json").is_file():
        print(f"SKIP evaluasi lengkap: {report_dir.name}", flush=True)
        return
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "bilinear_lmmd.engine.evaluate_checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--domain",
            "source",
            "--split",
            "val",
            "--data-root",
            str(image_root),
            "--manifest-root",
            str(manifest_root),
            "--output-dir",
            str(report_dir),
        ]
    )


def _compare(output_root: Path, seeds: list[int]) -> dict:
    baseline_paths = [_metrics_path(output_root, "S2G", seed) for seed in seeds]
    candidate_paths = [_metrics_path(output_root, "S2MR", seed) for seed in seeds]
    result = aggregate(baseline_paths, candidate_paths)
    baseline_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths
    ]
    candidate_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in candidate_paths
    ]
    result["summary"]["bottom3_class_f1"] = _paired_summary(
        [_bottom_three(report) for report in baseline_reports],
        [_bottom_three(report) for report in candidate_reports],
    )
    for key, label in (
        ("macro_f1", "Macro-F1 "),
        ("hard_class_f1", "Hard-F1  "),
        ("bottom3_class_f1", "Bottom-3 F1"),
        ("worst_class_f1", "Worst-F1 "),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def _decision(summary: dict, confirmation: bool) -> dict:
    macro = summary["macro_f1"]
    hard = summary["hard_class_f1"]
    criteria = {
        "macro_f1_improved": float(macro["delta_mean"]) > 0.0,
        "hard_f1_improved": float(hard["delta_mean"]) > 0.0,
        "bottom3_f1_preserved": float(
            summary["bottom3_class_f1"]["delta_mean"]
        ) >= -0.01,
    }
    if confirmation:
        criteria["macro_improved_both_seeds"] = (
            int(macro["improved_seeds"]) == len(CONFIRMATION_SEEDS)
        )
        criteria["hard_improved_at_least_one_seed"] = (
            int(hard["improved_seeds"]) >= 1
        )
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _audit_inputs(manifest_root: Path) -> dict:
    audit_path = manifest_root / "audit.json"
    ontology_path = manifest_root / "ontology.json"
    if not audit_path.is_file() or not ontology_path.is_file():
        raise FileNotFoundError("SNI classification v2 belum lengkap.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("Audit SNI classification v2 belum complete.")
    if int(audit["label_design"]["visual_v2_num_classes"]) != 15:
        raise ValueError("Protokol dikunci pada 15 kelas visual.")
    if len(ontology.get("visual_classes", ())) != 15:
        raise ValueError("Urutan kelas ontology.json tidak cocok.")
    if not bool(audit.get("test_locked")):
        raise ValueError("Audit wajib mengunci test.")
    return audit


def _audit_models() -> dict:
    rows = {}
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        rows[code] = {
            "head": cfg["model"]["head"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "out_indices": list(cfg["model"]["out_indices"]),
        }
    return rows


def run_sni_v2_multiresolution(
    image_root: Path,
    manifest_root: Path,
    output_root: Path,
    seeds: list[int],
    *,
    stage: str,
    hf_repo: str | None = None,
    hf_namespace: str = "sni-classification-v2-multiresolution",
    hf_sync_every: int = 1,
) -> dict:
    expected = SCREENING_SEEDS if stage == "screen" else CONFIRMATION_SEEDS
    if tuple(seeds) != expected:
        raise ValueError(f"Stage {stage} dikunci pada seed {list(expected)}.")
    _audit_inputs(manifest_root)
    audits = _audit_models()
    screening_report = output_root / "val_reports" / "screen_seed42.json"
    if stage == "confirm":
        if not screening_report.is_file():
            raise FileNotFoundError("Confirmation membutuhkan report screening.")
        previous = json.loads(screening_report.read_text(encoding="utf-8"))
        if previous.get("decision", {}).get("decision") != "PASS":
            raise RuntimeError("Confirmation tidak diizinkan karena screening FAIL.")

    if hf_repo:
        ensure_artifact_repo(hf_repo, private=True)
    print("=== SNI V2 GAP vs MULTIRESOLUTION ===", flush=True)
    print(f"Stage: {stage} | Seeds: {seeds} | Test locked", flush=True)
    print("Training balance: inverse-sqrt weighted sampler", flush=True)
    for code, row in audits.items():
        print(
            f"{code}: head={row['head']} params={row['parameters']:,}",
            flush=True,
        )

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = output_root / "val_reports" / f"{code}_seed{seed}"
            if hf_repo:
                _restore(
                    hf_repo, hf_namespace, code, seed, run_dir, report_dir
                )
            if not _training_complete(run_dir, epochs):
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "bilinear_lmmd.engine.train",
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--data-root",
                    str(image_root),
                    "--manifest-root",
                    str(manifest_root),
                    "--output-dir",
                    str(run_dir),
                    "--resume",
                ]
                if hf_repo:
                    command.extend(
                        [
                            "--artifact-repo",
                            hf_repo,
                            "--artifact-path",
                            f"{hf_namespace}/outputs/{code}_seed{seed}",
                            "--artifact-sync-every",
                            str(hf_sync_every),
                            "--artifact-required",
                        ]
                    )
                _run(command)
            else:
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            _evaluate(
                run_dir / "best.pt", report_dir, image_root, manifest_root
            )
            if hf_repo:
                sync_artifacts(
                    hf_repo,
                    f"{hf_namespace}/val_reports/{code}_seed{seed}",
                    report_dir,
                    filenames=tuple(
                        path.name for path in report_dir.iterdir() if path.is_file()
                    ),
                    commit_message=f"Evaluate {code} seed {seed} on SNI v2 val",
                )

    comparison = _compare(output_root, seeds)
    decision = _decision(
        comparison["summary"], confirmation=stage == "confirm"
    )
    all_seed_comparison = None
    if stage == "confirm":
        print("\n=== AGREGAT DESKRIPTIF SEMUA SEED 42/123/2026 ===")
        all_seed_comparison = _compare(
            output_root, [*SCREENING_SEEDS, *CONFIRMATION_SEEDS]
        )
    report = {
        "protocol": "SNI_v2_GAP_vs_multiresolution",
        "stage": stage,
        "seeds": seeds,
        "all_reported_seeds": (
            [*SCREENING_SEEDS, *CONFIRMATION_SEEDS]
            if stage == "confirm"
            else seeds
        ),
        "selection_split": "val",
        "test_opened": False,
        "models": audits,
        "comparison": comparison["summary"],
        "all_seed_comparison": (
            all_seed_comparison["summary"]
            if all_seed_comparison is not None
            else None
        ),
        "decision": decision,
    }
    destination = (
        output_root
        / "val_reports"
        / ("screen_seed42.json" if stage == "screen" else "confirmation.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hf_repo:
        sync_artifacts(
            hf_repo,
            f"{hf_namespace}/val_reports",
            destination.parent,
            filenames=(destination.name,),
            commit_message=f"SNI v2 multiresolution {stage} decision",
        )
    print("DECISION:", decision["decision"], decision["criteria"])
    print("TEST DIBUKA: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only SNI v2 GAP versus multiresolution."
    )
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    parser.add_argument("--hf-repo")
    parser.add_argument(
        "--hf-namespace", default="sni-classification-v2-multiresolution"
    )
    parser.add_argument("--hf-sync-every", type=int, default=1)
    args = parser.parse_args()
    run_sni_v2_multiresolution(
        args.image_root,
        args.manifest_root,
        args.output_root,
        args.seeds,
        stage=args.stage,
        hf_repo=args.hf_repo,
        hf_namespace=args.hf_namespace,
        hf_sync_every=args.hf_sync_every,
    )


if __name__ == "__main__":
    main()
