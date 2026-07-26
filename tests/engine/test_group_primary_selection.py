from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bilinear_lmmd.engine.train import evaluate, selection_score


def test_selection_score_supports_crop_and_group_primary_metrics() -> None:
    metrics = {
        "macro_f1": 0.7,
        "source_group_class": {"macro_f1": 0.8},
    }

    assert selection_score(metrics, "macro_f1") == 0.7
    assert (
        selection_score(metrics, "source_group_class_macro_f1") == 0.8
    )


def test_group_primary_selection_requires_group_metrics() -> None:
    with pytest.raises(ValueError, match="memerlukan manifest"):
        selection_score(
            {"macro_f1": 0.7},
            "source_group_class_macro_f1",
        )


class _ManifestDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {
                "dataset": "d1",
                "group_id": "g1",
                "visual_label": "a",
            },
            {
                "dataset": "d1",
                "group_id": "g1",
                "visual_label": "a",
            },
            {
                "dataset": "d2",
                "group_id": "g2",
                "visual_label": "b",
            },
        ]
        self.labels = [0, 0, 1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.tensor([index], dtype=torch.float32), self.labels[index]


class _LookupModel(nn.Module):
    def forward(self, images: torch.Tensor):
        indices = images[:, 0].to(dtype=torch.long)
        table = torch.tensor(
            [[3.0, 0.0], [0.0, 2.0], [0.0, 3.0]],
            device=images.device,
        )
        output = type("Output", (), {})()
        output.logits = table[indices]
        output.expert_logits = None
        return output


def test_evaluate_attaches_source_group_class_metrics() -> None:
    metrics = evaluate(
        _LookupModel(),
        DataLoader(_ManifestDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
        ["a", "b"],
        {},
        source_group_class=True,
    )

    assert metrics["source_group_class"]["unit_count"] == 2
    assert metrics["source_group_class"]["source_group_count"] == 2
    assert metrics["source_group_class"]["macro_f1"] == 1.0
