"""Independently verify frozen manifests, all input hashes and executed notebook outputs."""

import base64
import hashlib
import json
import xml.etree.ElementTree as ET

import nbformat
import pandas as pd
from PIL import Image

from scripts.audit_data import validate_manifests
from scripts.prepare_assets import ROOT, sha256


def verify():
    protocol = json.loads((ROOT / "configs/protocol-v1.json").read_text())
    frozen = protocol["data_audit"]
    report_dir = ROOT / "reports/task04"
    summary = json.loads((report_dir / "audit_summary.json").read_text())
    assert protocol["trained"] is False and summary["trained"] is False
    assert summary["archive_inventory_verified"]
    assert (
        summary["audit_policy_sha256"]
        == frozen["audit_policy_sha256"]
        == sha256(ROOT / "configs/audit-v1.json")
    )
    assert (
        summary["audit_code_sha256"]
        == frozen["audit_code_sha256"]
        == sha256(ROOT / "scripts/audit_data.py")
    )
    assert (
        summary["reference"]["sha256"]
        == frozen["reference_metadata_sha256"]
        == sha256(ROOT / "data/raw/reference/UTKFace_aligned.json")
    )
    assert summary["manifest_sha256"] == frozen["manifest_sha256"]
    for filename, digest in frozen["manifest_sha256"].items():
        assert sha256(ROOT / "data/manifests" / filename) == digest
    selected = pd.read_csv(ROOT / "data/manifests/selected.csv")
    audit = pd.read_csv(ROOT / "data/manifests/audit.csv")
    validate_manifests(selected, protocol["dataset"]["max_total_samples"])
    assert selected.split.value_counts().to_dict() == frozen["split_sizes"]
    assert len(audit) == summary["files_audited"]
    assert set(audit.loc[audit.selected, "sample_id"]) == set(selected.sample_id)
    assert sum(summary["exclusion_counts"].values()) == len(audit)
    for partition in ["train", "val_select", "val_calibration", "test"]:
        part = pd.read_csv(ROOT / f"data/manifests/{partition}.csv")
        assert set(part.sample_id) == set(selected.loc[selected.split == partition, "sample_id"])
    raw_dir = (ROOT / "data/raw/UTKFace").resolve()
    paths = {(ROOT / path).resolve() for path in audit.relative_path}
    assert paths == {path.resolve() for path in raw_dir.iterdir() if path.is_file()}
    for number, row in enumerate(audit.itertuples(), 1):
        path = (ROOT / row.relative_path).resolve()
        assert path.is_relative_to(raw_dir)
        assert sha256(path) == row.image_sha256
        if number % 5000 == 0:
            print(f"Rechecked original hashes {number}/{len(audit)}", flush=True)
    registry = json.loads((ROOT / "configs/assets.json").read_text())
    assert sha256(ROOT / registry["dataset"]["path"]) == summary["raw_archive_sha256"]
    findings = json.loads((report_dir / "eda_findings.json").read_text())
    assert len(findings) >= 5
    png_hashes = set()
    for finding in findings:
        assert len(finding["interpretation"]) > 100
        path = report_dir / "figures" / finding["filename"]
        with Image.open(path) as image:
            image.verify()
        png_hashes.add(sha256(path))
    notebook = nbformat.read(ROOT / "notebooks/04_auditoria_eda.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    embedded = set()
    markdown_outputs = 0
    all_text = []
    for cell in code_cells:
        for output in cell.outputs:
            assert output.output_type != "error"
            data = output.get("data", {})
            if "image/png" in data:
                embedded.add(hashlib.sha256(base64.b64decode(data["image/png"])).hexdigest())
            markdown_outputs += "text/markdown" in data
            all_text.append(str(output.get("text", "")))
    assert embedded == png_hashes and markdown_outputs >= len(findings)
    assert "manifests identicos" in " ".join(all_text)
    assert not any(
        "C:\\Users\\" in text or "Traceback (most recent call last)" in text for text in all_text
    )
    tests = ET.parse(ROOT / "artifacts/private/task04-tests.xml").getroot()
    suites = list(tests.iter("testsuite"))
    assert (
        suites
        and sum(
            int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
            for suite in suites
        )
        == 0
    )
    result = {
        "task": "04",
        "completed": True,
        "files_verified": len(audit),
        "images_decoded_by_full_audit": summary["images_decoded"],
        "manifest_files_verified": len(frozen["manifest_sha256"]),
        "selected": len(selected),
        "split_sizes": frozen["split_sizes"],
        "no_detected_hash_or_group_overlap": True,
        "subject_exclusive_verified": False,
        "notebook_code_cells_executed": len(code_cells),
        "notebook_errors": 0,
        "figure_files_and_embedded_outputs_match": len(embedded),
        "interpreted_figures": len(findings),
        "unit_tests_passed": sum(int(suite.attrib["tests"]) for suite in suites),
        "trained": False,
        "features_extracted": False,
        "commit_created": False,
        "push_executed": False,
        "next_task": "05_frozen_FaRL_features",
    }
    (report_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    verify()
