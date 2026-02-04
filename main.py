from train_test_split import multilabel_train_val_split
from Dataset.Dataset_load import TRAIN_IMG_DIR, TEST_IMG_DIR, ANNOTATION_DIR, CACHE_DIR, DATA_ROOT
import json
from pathlib import Path

if __name__ == "__main__":

    gt_cache_path = CACHE_DIR / "targets_by_filename.json"
    with open(gt_cache_path, "r", encoding="utf-8") as f:
        targets_by_filename = json.load(f)

    train_fnames, val_fnames = multilabel_train_val_split(
        targets_by_filename,
        val_ratio=0.33,
        seed=42
    )
    print("train:", len(train_fnames))
    print("val  :", len(val_fnames))
