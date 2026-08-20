from pathlib import Path

import pytest

from chest_ct_portfolio.dataset import audit_dataset, canonical_class_name


def test_canonical_class_names():
    assert canonical_class_name("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib") == "adenocarcinoma"
    assert canonical_class_name("large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa") == "large_cell_carcinoma"
    assert canonical_class_name("normal") == "normal"
    assert canonical_class_name("squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa") == "squamous_cell_carcinoma"


def _make_dataset(root: Path):
    classes = [
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
        "normal",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
    ]
    for split in ("train", "valid", "test"):
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            (d / "image.png").write_bytes(b"not-a-real-image")
            (d / "notes.txt").write_text("ignored")


def test_audit_dataset_counts_images_only(tmp_path):
    _make_dataset(tmp_path)
    audit = audit_dataset(tmp_path)
    assert audit.split_totals == {"train": 4, "valid": 4, "test": 4}
    assert audit.total_images == 12


def test_audit_rejects_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_dataset(tmp_path)


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        canonical_class_name("mystery")
