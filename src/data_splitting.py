"""Frozen, group-disjoint test holdout and development-only experiments.

No imputation, encoding, scaling or dimensionality reduction is performed here.
Run from the project root: python -m src.data_splitting
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from . import config, preprocessing


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_split_plan(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    validation_sizes: tuple = config.EXPERIMENT_VALIDATION_SIZES,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """First outer fold is test; first 2/3/4 inner folds are validation.

    Ratios refer to rows approximately, since a duplicate group is indivisible.
    The fold choice is fixed in advance, never selected using model scores.
    """
    if len(frame) != len(metadata) or metadata.processed_row.tolist() != list(range(len(frame))):
        raise ValueError("Metadata phai cung thu tu voi du lieu, processed_row = 0..n-1.")
    if metadata.duplicate_group.isna().any():
        raise ValueError("duplicate_group khong duoc thieu.")
    if not 0 < test_size < 1:
        raise ValueError("test_size phai nam trong (0, 1).")
    outer_folds = round(1 / test_size)
    if outer_folds < 2 or not np.isclose(1 / outer_folds, test_size):
        raise ValueError("test_size phai la nghich dao nguyen, vi du 0.2 = 1/5.")
    sizes = tuple(validation_sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Danh sach validation_sizes phai khac rong va khong trung.")
    for fraction in sizes:
        if not 0 < fraction < 1 or not np.isclose(fraction * 10, round(fraction * 10)):
            raise ValueError("Ty le validation phai la boi 0.1 trong (0, 1).")
    y = frame["Response"].to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Can du hai lop Response 0 va 1.")
    groups = metadata.duplicate_group.to_numpy()

    def check_enough_groups(rows, folds):
        for label in (0, 1):
            if len(np.unique(groups[rows][y[rows] == label])) < folds:
                raise ValueError(f"Lop {label} khong du nhom cho {folds} folds.")

    check_enough_groups(np.arange(len(frame)), outer_folds)
    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    development, test = next(outer.split(np.zeros(len(y)), y, groups))
    check_enough_groups(development, 10)
    inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_id = np.full(len(frame), -1, dtype=int)
    for fold, (_, local_validation) in enumerate(
        inner.split(np.zeros(len(development)), y[development], groups[development])
    ):
        fold_id[development[local_validation]] = fold

    def verify_partition(left, right, parent):
        if set(left) & set(right) or set(left) | set(right) != set(parent):
            raise ValueError("Tap chia khong phan hoach du cac dong.")
        if set(groups[left]) & set(groups[right]):
            raise ValueError("Phat hien nhom trung qua hai tap.")
        if any(set(np.unique(y[rows])) != {0, 1} for rows in (left, right)):
            raise ValueError("Mot tap chia bi mat lop Response.")

    verify_partition(development, test, np.arange(len(frame)))
    experiments = {}
    for fraction in sizes:
        count = round(fraction * 10)
        validation = development[fold_id[development] < count]
        train = development[fold_id[development] >= count]
        verify_partition(train, validation, development)
        name = f"{100 - round(fraction * 100)}_{round(fraction * 100)}"
        experiments[name] = {"train": train, "validation": validation}
    return {"development": development, "test": test,
            "development_fold": fold_id, "experiments": experiments}


def export_splits(
    raw: pd.DataFrame,
    output_dir: Path | None = None,
    test_size: float = config.TEST_SIZE,
    validation_sizes: tuple = config.EXPERIMENT_VALIDATION_SIZES,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Create once; subsequent calls verify hashes and never rewrite split files.

    Changed input/config or damaged files cause an error. A new dataset version
    needs an explicitly different output directory, not a silent test reshuffle.
    """
    directory = Path(output_dir) if output_dir is not None else config.SPLITS_DIR
    frame, metadata = preprocessing.create_handoff(raw)
    specification = {
        "raw_sha256": _sha256(_csv_bytes(raw)),
        "cleaned_sha256": _sha256(_csv_bytes(frame)),
        "metadata_sha256": _sha256(_csv_bytes(metadata)),
        "seed": seed, "test_size": test_size,
        "validation_sizes": list(validation_sizes),
        "strategy": "first_outer_stratified_group_fold_then_first_inner_deciles_v1",
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["specification"] != specification:
            raise ValueError("Du lieu/cau hinh da doi; khong tu dong thay tap test da chot.")
        for name, checksum in manifest["files_sha256"].items():
            path = directory / name
            if not path.is_file() or _sha256(path.read_bytes()) != checksum:
                raise ValueError(f"File chia tap bi thieu/thay doi: {name}; khong ghi de.")
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Thu muc splits co file nhung thieu manifest; khong ghi de.")

    plan = build_split_plan(frame, metadata, test_size, validation_sizes, seed)
    files = {}
    summary = []

    def add_partition(name, rows, parent_rows):
        part = frame.iloc[rows]
        files[name + ".csv"] = _csv_bytes(part)
        summary.append({
            "partition": name, "rows": len(part), "parent_rows": parent_rows,
            "fraction_of_parent": len(part) / parent_rows,
            "response_0": int((part.Response == 0).sum()),
            "response_1": int((part.Response == 1).sum()),
            "response_1_rate": float(part.Response.mean()),
            "groups": int(metadata.iloc[rows].duplicate_group.nunique()),
        })

    add_partition("development", plan["development"], len(frame))
    add_partition("test", plan["test"], len(frame))
    # Source-column snapshot, including ID; never use ID as a model feature.
    test_source_rows = metadata.iloc[plan["test"]].source_row.to_numpy()
    files["test_raw.csv"] = _csv_bytes(raw.iloc[test_source_rows])
    assignments = metadata.copy()
    assignments["partition"] = "development"
    assignments.loc[plan["test"], "partition"] = "test"
    assignments["development_fold"] = plan["development_fold"]
    files["assignments.csv"] = _csv_bytes(assignments)
    for name, experiment in plan["experiments"].items():
        for role, rows in experiment.items():
            add_partition(f"{role}_{name}", rows, len(plan["development"]))
    manifest = {
        "specification": specification, "sklearn_version": sklearn.__version__,
        "features": frame.drop(columns="Response").columns.tolist(),
        "target": "Response", "imputed": False, "encoded": False,
        "scaled": False, "dimensionality_reduced": False,
        "files_sha256": {name: _sha256(content) for name, content in files.items()},
        "summary": summary,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    manifest = export_splits(preprocessing.load_raw())
    summary = pd.DataFrame(manifest["summary"])
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES_DIR / "data_split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Test snapshots unchanged: no imputation, encoding, scaling or PCA.")
