from __future__ import annotations

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_sni_v2_multiresolution import (
    CONFIRMATION_SEEDS,
    MODEL_CONFIGS,
    SCREENING_SEEDS,
    _decision,
)
from bilinear_lmmd.modeling.models import build_model


def test_sni_v2_models_differ_only_in_head_representation() -> None:
    configs = {code: load_config(path) for code, path in MODEL_CONFIGS.items()}
    assert set(configs) == {"S2G", "S2MR"}
    for cfg in configs.values():
        assert cfg["model"]["backbone"] == "tf_efficientnetv2_b0.in1k"
        assert cfg["model"]["num_classes"] == 15
        assert cfg["data"]["dataset_format"] == "sni_manifest_v2"
        assert (
            cfg["data"]["manifest_weight_column"]
            == "train_weight_inverse_sqrt"
        )
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        assert model is not None
    assert configs["S2G"]["model"]["head"] == "gap"
    assert configs["S2MR"]["model"]["head"] == "sni_multiresolution_flat"


def test_sni_v2_screening_gate_requires_macro_hard_and_bottom_three() -> None:
    summary = {
        "macro_f1": {"delta_mean": 0.01, "improved_seeds": 1},
        "hard_class_f1": {"delta_mean": 0.02, "improved_seeds": 1},
        "bottom3_class_f1": {"delta_mean": -0.005},
    }
    assert _decision(summary, confirmation=False)["decision"] == "PASS"
    summary["bottom3_class_f1"]["delta_mean"] = -0.02
    assert _decision(summary, confirmation=False)["decision"] == "FAIL"


def test_sni_v2_confirmation_uses_only_unseen_confirmation_seeds() -> None:
    assert SCREENING_SEEDS == (42,)
    assert CONFIRMATION_SEEDS == (123, 2026)
    summary = {
        "macro_f1": {"delta_mean": 0.01, "improved_seeds": 2},
        "hard_class_f1": {"delta_mean": 0.01, "improved_seeds": 1},
        "bottom3_class_f1": {"delta_mean": 0.0},
    }
    assert _decision(summary, confirmation=True)["decision"] == "PASS"
    summary["macro_f1"]["improved_seeds"] = 1
    assert _decision(summary, confirmation=True)["decision"] == "FAIL"
