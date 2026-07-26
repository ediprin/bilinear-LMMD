import json
from pathlib import Path


def test_sni_source_balanced_v3_notebook_is_no_training_metadata_only():
    path = Path("notebooks/sni_source_balanced_v3_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )

    assert "prepare_sni_classification_v3" in source
    assert "--metadata-only" in source
    assert "engine.train" not in source
    assert "evaluate_checkpoint" not in source
    assert "--evaluation-split" not in source
    assert "Jangan training" in source
