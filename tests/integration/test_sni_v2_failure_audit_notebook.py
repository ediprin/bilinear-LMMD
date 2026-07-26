from __future__ import annotations

import json
from pathlib import Path


def test_sni_v2_failure_audit_notebook_is_no_training_validation_only() -> None:
    path = Path("notebooks/sni_v2_failure_audit_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    assert "bilinear_lmmd.analysis.sni_v2_failure" in source
    assert "('S2G', 'S2MR')" in source
    assert "f'{code}_seed42'" in source
    assert "predictions.csv" in source
    assert "engine.train" not in source
    assert "source/test" not in source
    assert "test.csv" not in source
