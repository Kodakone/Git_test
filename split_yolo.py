from __future__ import annotations
from pathlib import Path
import json
import shutil
import yaml
from PIL import Image
from collections import defaultdict

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from Dataset.Dataset_load import TRAIN_IMG_DIR, DATA_ROOT, CACHE_DIR

# 캐시 불러오기 ---------------------------------- 
gt_cache_path  = CACHE_DIR / "targets_by_filename.json"
cat_cache_path = CACHE_DIR / "categoryid_to_name.json"

with open(gt_cache_path , "r", encoding="utf-8") as f:
    targets_by_filename = json.load(f)

    targets_by_filename = defaultdict(
        lambda: {"boxes": [], "labels": []}, targets_by_filename
    )

with open(cat_cache_path, "r", encoding="utf-8") as f:
    categoryid_to_name = json.load(f)

categoryid_to_name = {int(k): v for k, v in categoryid_to_name.items()}

# 유틸 : 욜로라벨 저장 ----------------------------------
def save_yolo_label(txt_path: Path, boxes_xyxy, cls_ids, img_w: int, img_h: int):
    """
    boxes_xyxy: list[tuple[x1,y1,x2,y2]]
    cls_ids: list[int]  (YOLO cls id: 0..nc-1)
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        # clip
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue

        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        # normalize
        cx /= img_w
        cy /= img_h
        bw /= img_w
        bh /= img_h

        lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

# train_test_split ----------------------------------
def multilabel_train_val_split(
    targets_by_filename: dict,
    categoryid_to_name: dict,
    val_ratio: float = 0.33,
    seed: int = 42
):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    all_fnames = sorted(targets_by_filename.keys())

    # split 위한 멀티라벨 행렬은 "catid 기준"으로 만들어도 됨
    all_cat_ids = sorted({
        int(label)
        for t in targets_by_filename.values()
        for label in t["labels"]
    })
    class_to_idx = {catid: i for i, catid in enumerate(all_cat_ids)}

    Y = np.zeros((len(all_fnames), len(all_cat_ids)), dtype=int)
    for i, fname in enumerate(all_fnames):
        for catid  in set(targets_by_filename[fname]["labels"]):
            if catid in class_to_idx:
                Y[i, class_to_idx[catid]] = 1

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(all_fnames, Y))

    train_fnames = sorted([all_fnames[i] for i in train_idx])
    val_fnames   = sorted([all_fnames[i] for i in val_idx])
    
    return train_fnames, val_fnames

# train+val 기준 매핑생성 ----------------------------------
def build_yolo_class_mapping(
    targets_by_filename: dict,
    categoryid_to_name: dict,
    train_fnames: list[str],
    val_fnames: list[str],
):
    split_fnames = train_fnames + val_fnames

    all_cat_ids = sorted({
        int(label)
        for f in split_fnames
        for label in targets_by_filename[f]["labels"]
    })

    catid_to_cls = {catid: i for i, catid in enumerate(all_cat_ids)}  # ex. 1900: 0, 2483: 1,...
    cls_to_catid = {i: catid for catid, i in catid_to_cls.items()}  # ex. 0: 1900, 1: 2483,...

    # YOLO cls_id 순서(list) names
    cls_to_name = []
    missing = []
    for catid in all_cat_ids:
        if catid not in categoryid_to_name:
            missing.append(catid)
            cls_to_name.append(f"UNKNOWN_{catid}")
        else:
            cls_to_name.append(categoryid_to_name[catid])  # ex. '보령부스파정 5mg', '뮤테란캡슐 100mg',...

    if missing:
        print(f"[mapping] WARNING: categoryid_to_name에 없는 catid {len(missing)}개"
              f"(예: {missing[:10]})")

    return all_cat_ids, catid_to_cls, cls_to_catid, cls_to_name

# split.json 저장 ----------------------------------
def save_split(cache_dir: Path, train_fnames: list[str], val_fnames: list[str],
               val_ratio: float, seed: int, mapping: dict | None = None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    out = {
        "train": train_fnames, 
        "val": val_fnames, 
        "val_ratio": val_ratio, 
        "seed": seed
    }
    if mapping is not None:
        out["mapping"] = mapping 

    split_path = cache_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[save_split] saved: {split_path}")

# 이미지 복사 ----------------------------------
def copy_images(src_img_dir: Path, dst_img_dir: Path, fnames: list[str]):
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    for fname in fnames:
        src = src_img_dir / fname
        dst = dst_img_dir / fname

        if not src.exists():
            missing += 1
            print(f"[WARN] image missing: {src}")
            continue

        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"[copy_images] done. missing: {missing}/{len(fnames)}")

# 라벨 변환+저장 ----------------------------------
def save_split_labels_yolo(
    output_root: Path,
    img_dir: Path,
    targets_by_filename: dict,
    catid_to_cls: dict,
    fnames: list[str],
    split: str,  # "train" or "val"
):
    out_lbl_dir = output_root / "labels" / split
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for fname in fnames:
        img_path = img_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"missing image for label save: {img_path}")

        # 이미지 크기 가져오기
        with Image.open(img_path) as im:
            w, h = im.size

        tgt = targets_by_filename[fname]
        boxes = [tuple(b) for b in tgt["boxes"]]
        catids = [int(c) for c in tgt["labels"]]

        # catid -> yolo cls id
        keep = [c in catid_to_cls for c in catids]
        if any(not k for k in keep):
            boxes = [b for b, k in zip(boxes, keep) if k]
            catids = [c for c, k in zip(catids, keep) if k]

        cls_ids = [catid_to_cls[c] for c in catids]

        out_txt = out_lbl_dir / (Path(fname).stem + ".txt")
        save_yolo_label(out_txt, boxes, cls_ids, w, h)

# dataset.yaml 생성 ----------------------------------
def write_dataset_yaml(output_root: Path, names: list[str]):
    names_dict = {i: n for i, n in enumerate(names)}
    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": names_dict,
    }
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("[dataset.yaml] saved:", yaml_path)
    print("[dataset.yaml] nc:", len(names))

def build_yolo_dataset(
    OUTPUT_ROOT: Path,
    IMG_DIR: Path,
    targets_by_filename: dict,
    categoryid_to_name: dict,
    val_ratio: float = 0.33,
    seed: int = 42,
    clean_output: bool = True,
):
    if clean_output and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    # 폴더 생성
    (OUTPUT_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 1) split
    train_fnames, val_fnames = multilabel_train_val_split(
        targets_by_filename=targets_by_filename,
        categoryid_to_name=categoryid_to_name,
        val_ratio=val_ratio,
        seed=seed
    )

    # 2) split 이후 매핑 생성 (train+val 기준)
    all_cat_ids, catid_to_cls, cls_to_catid, cls_to_name = build_yolo_class_mapping(
        targets_by_filename=targets_by_filename,
        categoryid_to_name=categoryid_to_name,
        train_fnames=train_fnames,
        val_fnames=val_fnames,
    )

    # 3) split.json 저장 (+매핑 저장 추천)
    save_split(
        cache_dir=OUTPUT_ROOT,
        train_fnames=train_fnames,
        val_fnames=val_fnames,
        val_ratio=val_ratio,
        seed=seed,
        mapping={
            "all_cat_ids": all_cat_ids,
            "catid_to_cls": catid_to_cls,
            "cls_to_catid": cls_to_catid,
            "cls_to_name": cls_to_name,
        }
    )

    # 4) 이미지 복사
    copy_images(IMG_DIR, OUTPUT_ROOT / "images" / "train", train_fnames)
    copy_images(IMG_DIR, OUTPUT_ROOT / "images" / "val",   val_fnames)

    # 5) 라벨 변환 저장 (YOLO cls_id)
    save_split_labels_yolo(
        output_root=OUTPUT_ROOT,
        img_dir=OUTPUT_ROOT / "images" / "train",  # 복사된 이미지 기준으로 w,h 얻음
        targets_by_filename=targets_by_filename,
        catid_to_cls=catid_to_cls,
        fnames=train_fnames,
        split="train",
    )
    save_split_labels_yolo(
        output_root=OUTPUT_ROOT,
        img_dir=OUTPUT_ROOT / "images" / "val",
        targets_by_filename=targets_by_filename,
        catid_to_cls=catid_to_cls,
        fnames=val_fnames,
        split="val",
    )

    # 6) dataset.yaml 생성 (names는 cls_to_name)
    write_dataset_yaml(OUTPUT_ROOT, cls_to_name)

    print("[DONE]")
    print("train:", len(train_fnames), "val:", len(val_fnames))
    print("nc:", len(cls_to_name))
    return train_fnames, val_fnames, catid_to_cls, cls_to_name    

# ============================================================================
# v1 scaffold 만들기(v0 복사+증강본 추가용) ----------------------------------
def make_aug_scaffold(v0_root: Path, v1_root: Path, cls_to_name: list[str], force: bool = False):
    if v1_root.exists() and any(v1_root.iterdir()) and not force:
        print(f"[v1] already exists and not empty: {v1_root}")
        print("[v1] scaffold step skipped (set force=True to override).")
        return
    
    if force and v1_root.exists():
        shutil.rmtree(v1_root)

    for sub in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (v1_root / sub).mkdir(parents=True, exist_ok=True)

    # 이미지/라벨복사
    def copy_all(src: Path, dst: Path):
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dst / p.name)

    # train/val 이미지/라벨 복사
    copy_all(v0_root / "images/train", v1_root / "images/train")
    copy_all(v0_root / "labels/train", v1_root / "labels/train")
    copy_all(v0_root / "images/val",   v1_root / "images/val")
    copy_all(v0_root / "labels/val",   v1_root / "labels/val")
    
    # v1용 dataset.yaml 생성
    write_dataset_yaml(v1_root, cls_to_name)
    print("v1 dataset scaffold created (no augmentation)")

# 이후 증강이미지는 yolo_dataset_aug에 저장해야함(원본복사본 포함되어있음).
# yolo_dataset에는 원본만 있음.

# 디버그 유틸 ----------------------------------
def compute_max_class_id(labels_root: Path) -> int:
    max_id = -1
    for p in list((labels_root / "train").glob("*.txt")) + list((labels_root / "val").glob("*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls > max_id:
                max_id = cls
    return max_id

# 실행! ----------------------------------
V0_ROOT  = DATA_ROOT / "yolo_dataset"
IMG_DIR = TRAIN_IMG_DIR

train_fnames, val_fnames, catid_to_cls, cls_to_name = build_yolo_dataset(
    OUTPUT_ROOT=V0_ROOT ,
    IMG_DIR=IMG_DIR,
    targets_by_filename=targets_by_filename,
    categoryid_to_name=categoryid_to_name,
    val_ratio=0.33,
    seed=42,
    clean_output=True,
)
print("train images:", len(list((V0_ROOT /'images/train').glob('*.png'))))
print("val images  :", len(list((V0_ROOT /'images/val').glob('*.png'))))
print("train labels:", len(list((V0_ROOT /'labels/train').glob('*.txt'))))
print("val labels  :", len(list((V0_ROOT /'labels/val').glob('*.txt'))))

max_id = compute_max_class_id(V0_ROOT  / "labels")
print("max class_id:", max_id, "| expected:", len(cls_to_name) - 1)

# ===========================================================
# v1 scaffold (최초 1회만) ----------------------------------
V1_ROOT = DATA_ROOT / "yolo_dataset_aug"

make_aug_scaffold(
    v0_root=V0_ROOT,
    v1_root=V1_ROOT,
    cls_to_name=cls_to_name,
    force=False,   # 기본 False: 이미 존재하면 스킵
)
print("aug_train images:", len(list((V1_ROOT /'images/train').glob('*.png'))))
print("aug_val images  :", len(list((V1_ROOT /'images/val').glob('*.png'))))
print("aug_train labels:", len(list((V1_ROOT /'labels/train').glob('*.txt'))))
print("aug_val labels  :", len(list((V1_ROOT /'labels/val').glob('*.txt'))))
