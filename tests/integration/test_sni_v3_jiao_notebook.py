from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/sni_v3_jiao_group_primary_colab.ipynb")


def test_sni_v3_jiao_notebook_is_resumable_and_fail_fast() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "agent/sni-instance-crops" in source
    assert "classification-v3-source-balanced" in source
    assert "sni-v3-jiao-group-primary-v1" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "'--manifest-root'" in source
    assert "'--evaluation-split', 'val'" in source
    assert "run_sni_v3_jiao_screening" in source
    assert "run_stage('mechanism', ('S3J0', 'S3J1'))" in source
    assert "mechanism['decision']['decision'] == 'PASS'" in source
    assert "source/test" not in source
    assert "time.sleep(60)" in source
