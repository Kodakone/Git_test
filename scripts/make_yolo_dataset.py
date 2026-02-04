# scripts/make_yolo_dataset.py
import json
import shutil
from pathlib import Path
from PIL import Image

from Dataset.Dataset_load import DATA_ROOT, CACHE_DIR

# ----- inputs (from cache) -----
SPLITS_JSON = CACHE_DIR / "splits_rareaware.json"
GT_JSON = CACHE_DIR / "targets_by_filename.json"
MASTER_JSON = CACHE_DIR / "master_classes.json"

# ----- output dataset root -----
OUT_ROOT = DATA_ROOT.parent / "Data_yolo"   # D:/.../ProjA/Data_yolo (Data 옆)
OUT_IMAGES_TRAIN = OUT_ROOT / "images" / "train"
OUT_IMAGES_VAL = OUT_ROOT / "images" / "val"
OUT_LABELS_TRAIN = OUT_ROOT / "labels" / "train"
OUT_LABELS_VAL = OUT_ROOT / "labels" / "val"
OUT_YAML = OUT_ROOT / "dataset.yaml"

# ----- source images -----
SRC_TRAIN_IMG_DIR = DATA_ROOT / "train_images"  # 원본 train_images(232장)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_dirs():
    for p in [OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL]:
        p.mkdir(parents=True, exist_ok=True)


def build_catid_to_yoloidx(master_list):
    # master_classes.json: [{"yolo_idx":0,"category_id":1900,"name":"..."}, ...]
    return {int(x["category_id"]): int(x["yolo_idx"]) for x in master_list}


def coco_xyxy_to_yolo_xywhn(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0

    # normalize to 0..1
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)


def write_yolo_label(txt_path: Path, yolo_lines: list[str]):
    txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def process_split(split_name: str, file_list: list[str], targets_by_filename: dict, catid2yolo: dict):
    if split_name == "train":
        out_img_dir, out_lbl_dir = OUT_IMAGES_TRAIN, OUT_LABELS_TRAIN
    else:
        out_img_dir, out_lbl_dir = OUT_IMAGES_VAL, OUT_LABELS_VAL

    n_imgs = 0
    n_labels = 0

    for fname in file_list:
        src_img = SRC_TRAIN_IMG_DIR / fname
        if not src_img.exists():
            # split에 있는데 이미지가 없으면 스킵 (경고)
            print(f"[WARN] image missing: {src_img}")
            continue

        # 이미지 복사
        dst_img = out_img_dir / fname
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # 이미지 크기 읽기 (정규화용)
        with Image.open(src_img) as im:
            img_w, img_h = im.size

        # 라벨 만들기
        d = targets_by_filename.get(fname, {"boxes": [], "labels": []})
        boxes = d.get("boxes", [])
        labels = d.get("labels", [])
        yolo_lines = []

        for box_xyxy, cid in zip(boxes, labels):
            cid = int(cid)
            if cid not in catid2yolo:
                continue
            cls = catid2yolo[cid]
            xc, yc, w, h = coco_xyxy_to_yolo_xywhn(box_xyxy, img_w, img_h)

            # bbox가 이미지 밖으로 너무 나가거나 0이면 스킵
            if w <= 0 or h <= 0:
                continue
            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                # 살짝 벗어난 케이스는 clamp 하고 싶으면 여기서 처리 가능
                continue

            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # txt 저장 (이미지와 같은 stem)
        txt_name = Path(fname).with_suffix(".txt").name
        write_yolo_label(out_lbl_dir / txt_name, yolo_lines)

        n_imgs += 1
        n_labels += len(yolo_lines)

    print(f"[{split_name}] images: {n_imgs}, total boxes written: {n_labels}")


def write_dataset_yaml(master_list):
    # names는 yolo_idx 순으로 정렬된 클래스명 리스트
    master_sorted = sorted(master_list, key=lambda x: int(x["yolo_idx"]))
    names = [x["name"] for x in master_sorted]

    # Ultralytics YOLO 형식
    # path는 dataset.yaml이 있는 폴더 기준 상대/절대 모두 가능
    content = [
        f"path: {OUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        content.append(f"  {i}: {n}")

    OUT_YAML.write_text("\n".join(content) + "\n", encoding="utf-8")
    print("saved:", OUT_YAML)


def main():
    # 체크: cache 파일 존재
    for p in [SPLITS_JSON, GT_JSON, MASTER_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    ensure_dirs()

    splits = load_json(SPLITS_JSON)
    targets_by_filename = load_json(GT_JSON)
    master_list = load_json(MASTER_JSON)
    catid2yolo = build_catid_to_yoloidx(master_list)

    train_files = splits["train"]
    val_files = splits["val"]

    # 생성
    process_split("train", train_files, targets_by_filename, catid2yolo)
    process_split("val", val_files, targets_by_filename, catid2yolo)
    write_dataset_yaml(master_list)

    # sanity
    print("DONE. dataset root:", OUT_ROOT)


if __name__ == "__main__":
    main()


