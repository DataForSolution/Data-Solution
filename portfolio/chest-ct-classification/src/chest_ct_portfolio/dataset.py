"""Dataset-contract checks for the public chest CT experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CANONICAL_CLASSES = (
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma",
)
EXPECTED_SPLIT_TOTALS: Mapping[str, int] = {
    "train": 613,
    "valid": 72,
    "test": 315,
}


@dataclass(frozen=True)
class DatasetAudit:
    root: Path
    split_totals: dict[str, int]
    class_totals: dict[str, dict[str, int]]

    @property
    def total_images(self) -> int:
        return sum(self.split_totals.values())


def canonical_class_name(folder_name: str) -> str:
    """Normalize the verbose historical folder names to four stable classes."""
    name = folder_name.strip().lower().replace("-", ".").replace("_", ".")
    if "adenocarcinoma" in name:
        return "adenocarcinoma"
    if "large.cell" in name or "largecell" in name:
        return "large_cell_carcinoma"
    if "squamous.cell" in name or "squamouscell" in name:
        return "squamous_cell_carcinoma"
    if name == "normal" or ".normal." in f".{name}.":
        return "normal"
    raise ValueError(f"unrecognized class folder: {folder_name!r}")


def _count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def audit_dataset(
    data_root: str | Path,
    *,
    enforce_expected_totals: bool = False,
) -> DatasetAudit:
    """Validate train/valid/test structure and return deterministic counts.

    ``data_root`` should be the directory containing the three supplied split
    folders (typically the dataset's ``Data`` directory).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    split_totals: dict[str, int] = {}
    class_totals: dict[str, dict[str, int]] = {}

    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")

        counts: dict[str, int] = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            canonical = canonical_class_name(class_dir.name)
            if canonical in counts:
                raise ValueError(
                    f"multiple folders map to class {canonical!r} in split {split!r}"
                )
            counts[canonical] = _count_images(class_dir)

        missing = sorted(set(CANONICAL_CLASSES) - set(counts))
        extra = sorted(set(counts) - set(CANONICAL_CLASSES))
        if missing or extra:
            raise ValueError(
                f"class mismatch in {split}: missing={missing}, extra={extra}"
            )

        class_totals[split] = counts
        split_totals[split] = sum(counts.values())

        if enforce_expected_totals and split_totals[split] != EXPECTED_SPLIT_TOTALS[split]:
            raise ValueError(
                f"unexpected {split} image count: {split_totals[split]} "
                f"!= {EXPECTED_SPLIT_TOTALS[split]}"
            )

    return DatasetAudit(root=root, split_totals=split_totals, class_totals=class_totals)
