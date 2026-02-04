from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# train_test_split
def multilabel_train_val_split(
    targets_by_filename: dict,
    val_ratio: float = 0.33,
    seed: int = 42
):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    all_fnames = sorted(targets_by_filename.keys())

    all_classes = sorted({
        label
        for t in targets_by_filename.values()
        for label in t["labels"]
    })
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    Y = np.zeros((len(all_fnames), len(all_classes)), dtype=int)
    for i, fname in enumerate(all_fnames):
        for label in set(targets_by_filename[fname]["labels"]):
            Y[i, class_to_idx[label]] = 1

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(all_fnames, Y))

    train_fnames = [all_fnames[i] for i in train_idx]
    val_fnames   = [all_fnames[i] for i in val_idx]
    
    return sorted(train_fnames), sorted(val_fnames)

# 원본 디렉토리 참조, fnames: split된 파일명 리스트, Path: 실제 이미지경로
def fnames_to_paths(image_dir: Path, fnames: list[str]) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    
    paths = [image_dir / f for f in fnames]

    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[fnames_to_paths] {image_dir} 에 존재하지 않는 파일이 있습니다. "
            f"예: {missing[:10]}"
        )

    return paths

def save_split(cache_dir: Path, train_fnames: list[str], val_fnames: list[str],
               val_ratio: float, seed: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    out = {
        "train": train_fnames, 
        "val": val_fnames, 
        "val_ratio": val_ratio, 
        "seed": seed}
    
    split_path = cache_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[save_split] saved: {split_path}")




