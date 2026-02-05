import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# ============================================================
# 0) 설정
# ============================================================
random.seed(42)

# 배경(base)로 쓰면 안 되는 "금지 CID"
BAN_BASE_CIDS = {3351}

# - REPO_ROOT/augment_targets.txt 에 CID를 적어두면 됨
TARGET_CIDS_FILE = "augment_targets.txt"

# 목표 인스턴스 수(부족한 만큼만 증강)
TARGET_MIN_INST = 30

# 폭발 방지
MAX_AUG_PER_CLASS = 200
MAX_TOTAL_AUG = 2000

# 합성 파라미터
BASE_TRIES = 80
POS_TRIES = 140
OCC_MARGIN = 10

# 가시성
USE_CLAHE_ON_OBJ = True
MIN_CONTRAST_DELTA = 8.0
FEATHER_SIGMA = 0.8
CORE_ERODE = 5

# donor clean 필터(겹친 알약은 donor에서 제외)
DONOR_MAX_IOU_TH = 0.05

# val도 YOLO로 변환/복사할지(증강은 안 함)
EXPORT_VAL_YOLO = True


# ============================================================
# 1) 경로 로드 (.env: LOG_FILE_PATH)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DATA_ROOT = os.getenv("LOG_FILE_PATH")
if not DATA_ROOT:
    raise ValueError("LOG_FILE_PATH가 .env에 설정되어 있지 않습니다.")
DATA_ROOT = Path(DATA_ROOT)
print("[DATA_ROOT]", DATA_ROOT)


# ============================================================
# 2) train/val split 자동 탐색
# - 팀원이 새로 준 데이터 구조가 달라도, 아래 후보들만 맞으면 자동으로 잡힘
# ============================================================
def _find_first_dir(root: Path, candidates):
    for rel in candidates:
        p = root / rel
        if p.exists() and p.is_dir():
            return p
    return None


def _collect_jsons(p: Path):
    if p is None or not p.exists():
        return []
    if p.is_file() and p.suffix.lower() == ".json":
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*.json")])
    return []


def resolve_split_paths(root: Path):
    # 이미지 폴더 후보들(필요하면 여기만 추가하면 됨)
    train_img = _find_first_dir(
        root,
        [
            "train_images",
            "images/train",
            "train/images",
            "train",
        ],
    )
    val_img = _find_first_dir(
        root,
        [
            "val_images",
            "valid_images",
            "images/val",
            "val/images",
            "val",
        ],
    )

    # 어노테이션 폴더/파일 후보들
    # - 폴더일 수도 있고, instances_train.json 같은 단일 파일일 수도 있음
    train_ann_dir = _find_first_dir(
        root,
        [
            "train_annotations",
            "annotations/train",
            "train/annotations",
            "annotations",
        ],
    )
    val_ann_dir = _find_first_dir(
        root,
        [
            "val_annotations",
            "valid_annotations",
            "annotations/val",
            "val/annotations",
            "annotations",
        ],
    )

    train_jsons = _collect_jsons(train_ann_dir)
    val_jsons = _collect_jsons(val_ann_dir)

    # annotations 폴더 하나에 train/val json이 같이 있을 경우를 대비해 이름 기반 필터
    def _filter_split(jsons, split_kw):
        if not jsons:
            return []
        # "train" / "val" 키워드가 파일명에 들어 있으면 그걸 우선
        picked = [p for p in jsons if split_kw in p.stem.lower()]
        return picked if picked else jsons

    train_jsons = _filter_split(train_jsons, "train")
    val_jsons = _filter_split(val_jsons, "val")

    if train_img is None:
        raise RuntimeError(
            "train 이미지 폴더를 찾지 못했습니다. (DATA_ROOT 아래 train_images 등 확인)"
        )
    if not train_jsons:
        raise RuntimeError(
            "train annotation(json)을 찾지 못했습니다. (DATA_ROOT 아래 train_annotations 등 확인)"
        )

    return train_img, train_jsons, val_img, val_jsons


TRAIN_IMG_DIR, TRAIN_JSONS, VAL_IMG_DIR, VAL_JSONS = resolve_split_paths(DATA_ROOT)

print("[TRAIN_IMG_DIR]", TRAIN_IMG_DIR)
print("[TRAIN_JSONS]", len(TRAIN_JSONS), "files")
print("[VAL_IMG_DIR]", VAL_IMG_DIR)
print("[VAL_JSONS]", len(VAL_JSONS), "files")


def list_images(img_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        [p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


train_img_files = list_images(TRAIN_IMG_DIR)
val_img_files = list_images(VAL_IMG_DIR) if VAL_IMG_DIR else []

print("train images:", len(train_img_files))
print("val images  :", len(val_img_files))


# ============================================================
# 3) COCO json -> targets_by_filename + categoryid_to_name
# - 캐시를 DATA_ROOT 별로 분리해서 "새 데이터 받았는데 옛 캐시가 섞이는" 문제를 막음
# ============================================================
def sha1_text(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


DATA_TAG = sha1_text(str(DATA_ROOT.resolve()))
CACHE_DIR = REPO_ROOT / "cache" / f"copypaste_{DATA_TAG}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_targets(img_files, json_paths):
    img_set = set(img_files)
    targets = defaultdict(lambda: {"boxes": [], "labels": []})
    catmap = {}

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            coco = json.load(f)

        for c in coco.get("categories", []):
            catmap[int(c["id"])] = c.get("name", str(c["id"]))

        imageid_to_fname = {}
        for img in coco.get("images", []):
            fname = img.get("file_name")
            if fname in img_set:
                imageid_to_fname[int(img["id"])] = fname

        if not imageid_to_fname:
            continue

        for ann in coco.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in imageid_to_fname:
                continue
            fname = imageid_to_fname[int(img_id)]
            cid = int(ann.get("category_id"))

            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            targets[fname]["boxes"].append([x1, y1, x2, y2])
            targets[fname]["labels"].append(cid)

    return targets, catmap


def cache_key(img_dir: Path, json_paths):
    # json 경로 + mtime/size로 대략적인 변경 감지
    sig = []
    for p in json_paths:
        st = p.stat()
        sig.append((str(p.resolve()), st.st_mtime_ns, st.st_size))
    payload = {"img_dir": str(img_dir.resolve()), "json_sig": sig}
    return sha1_text(json.dumps(payload, ensure_ascii=False))


def load_or_build_split(split_name, img_dir, img_files, json_paths):
    key = cache_key(img_dir, json_paths)
    gt_path = CACHE_DIR / f"{split_name}_targets_{key}.json"
    cat_path = CACHE_DIR / f"{split_name}_catmap_{key}.json"

    if gt_path.exists() and cat_path.exists():
        print(f"[CACHE HIT] {split_name} -> {gt_path.name}")
        targets_raw = json.loads(gt_path.read_text(encoding="utf-8"))
        targets = defaultdict(lambda: {"boxes": [], "labels": []}, targets_raw)

        catmap = json.loads(cat_path.read_text(encoding="utf-8"))
        catmap = {int(k): v for k, v in catmap.items()}
        return targets, catmap

    print(f"[CACHE MISS] {split_name} -> build from json ({len(json_paths)})")
    targets, catmap = build_targets(img_files, json_paths)

    gt_path.write_text(json.dumps(dict(targets), ensure_ascii=False), encoding="utf-8")
    cat_path.write_text(json.dumps(catmap, ensure_ascii=False), encoding="utf-8")
    return targets, catmap


targets_train, catmap_train = load_or_build_split(
    "train", TRAIN_IMG_DIR, train_img_files, TRAIN_JSONS
)
targets_val, catmap_val = (
    defaultdict(lambda: {"boxes": [], "labels": []}),
    {},
)  # default
if VAL_IMG_DIR and VAL_JSONS:
    targets_val, catmap_val = load_or_build_split(
        "val", VAL_IMG_DIR, val_img_files, VAL_JSONS
    )

# category map은 train/val 합쳐서(클래스 인덱스 일관성)
categoryid_to_name = {}
categoryid_to_name.update(catmap_train)
categoryid_to_name.update(catmap_val)

cat_ids_sorted = sorted([int(k) for k in categoryid_to_name.keys()])
catid_to_idx = {cid: i for i, cid in enumerate(cat_ids_sorted)}


# ============================================================
# 4) TARGET_CIDS 로드 (파일)
# ============================================================
def load_cids_from_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"TARGET CID 파일이 없습니다: {path}")

    txt = path.read_text(encoding="utf-8")
    # 숫자만 뽑아서 CID 목록으로 사용(쉼표/공백/줄바꿈/주석 상관없이)
    cids = [int(x) for x in re.findall(r"\d+", txt)]
    # 중복 제거(순서 유지)
    out = []
    seen = set()
    for x in cids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


TARGET_CIDS = load_cids_from_file(REPO_ROOT / TARGET_CIDS_FILE)
if not TARGET_CIDS:
    raise ValueError("TARGET_CIDS가 비어있습니다. augment_targets.txt를 확인하세요.")

print("[TARGET_CIDS]", len(TARGET_CIDS))


# ============================================================
# 5) 출력 폴더 (YOLO 표준 구조 추천)
# ============================================================
OUT_ROOT = DATA_ROOT / "copypaste_yolo"
OUT_TRAIN_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_TRAIN_LBL_DIR = OUT_ROOT / "labels" / "train"
OUT_VAL_IMG_DIR = OUT_ROOT / "images" / "val"
OUT_VAL_LBL_DIR = OUT_ROOT / "labels" / "val"

for p in [OUT_TRAIN_IMG_DIR, OUT_TRAIN_LBL_DIR, OUT_VAL_IMG_DIR, OUT_VAL_LBL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# catid_to_idx 저장
(OUT_ROOT / "catid_to_idx.json").write_text(
    json.dumps(catid_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_ROOT / "categoryid_to_name.json").write_text(
    json.dumps(categoryid_to_name, ensure_ascii=False, indent=2), encoding="utf-8"
)


# ============================================================
# 6) YOLO 라벨 쓰기
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def write_yolo_label(txt_path: Path, boxes, labels, img_w, img_h):
    lines = []
    for box, cid in zip(boxes, labels):
        cid = int(cid)
        if cid not in catid_to_idx:
            continue
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        cls = catid_to_idx[cid]
        lines.append(
            f"{cls} {cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 7) 원본 복사 + YOLO 라벨 변환 (train / val)
# ============================================================
print("\n== [A] Export TRAIN originals -> YOLO ==")
for fname in train_img_files:
    src = TRAIN_IMG_DIR / fname
    img = cv2.imread(str(src))
    if img is None:
        continue
    h, w = img.shape[:2]
    shutil.copy2(src, OUT_TRAIN_IMG_DIR / fname)

    t = targets_train[fname]  # 없으면 빈
    write_yolo_label(
        OUT_TRAIN_LBL_DIR / f"{Path(fname).stem}.txt", t["boxes"], t["labels"], w, h
    )

if EXPORT_VAL_YOLO and VAL_IMG_DIR and val_img_files:
    print("\n== [B] Export VAL originals -> YOLO (no augmentation) ==")
    for fname in val_img_files:
        src = VAL_IMG_DIR / fname
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        shutil.copy2(src, OUT_VAL_IMG_DIR / fname)

        t = targets_val[fname]
        write_yolo_label(
            OUT_VAL_LBL_DIR / f"{Path(fname).stem}.txt", t["boxes"], t["labels"], w, h
        )


# ============================================================
# 8) Copy-Paste 증강 유틸
# ============================================================
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def build_clean_donor_pool(targets_by_filename, max_iou_th=0.05):
    """
    donor 후보 중에서 "같은 이미지 내 다른 객체와 거의 안 겹치는" 객체만 사용
    -> 겹쳐서 모양 깨진 알약(마스크 이상)의 확률을 줄임
    """
    donor_pool = {}
    for fname, t in targets_by_filename.items():
        boxes = t["boxes"]
        labels = t["labels"]
        n = len(boxes)
        if n == 0:
            continue
        for i in range(n):
            bi = boxes[i]
            mx = 0.0
            for j in range(n):
                if i == j:
                    continue
                mx = max(mx, box_iou(bi, boxes[j]))
            if mx <= max_iou_th:
                cid = int(labels[i])
                donor_pool.setdefault(cid, []).append((fname, bi))
    return donor_pool


def grabcut_pill_mask(obj_bgr, iters=5):
    """
    donor crop에서 GrabCut으로 알약 마스크를 뽑음
    """
    h, w = obj_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(1, w - 4), max(1, h - 4))

    try:
        cv2.grabCut(
            obj_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT
        )
    except:
        return None

    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )

    # morphology로 구멍/찌그러짐 완화
    k = max(3, (min(h, w) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (h * w * 0.15):
        return None

    clean = np.zeros_like(m)
    cv2.drawContours(clean, [cnt], -1, 255, -1)
    clean = cv2.GaussianBlur(clean, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return clean


def enhance_obj_contrast(obj_bgr, mask_u8):
    # 마스크 내부만 CLAHE 적용 (가시성↑)
    lab = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    enh = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out = obj_bgr.copy()
    idx = mask_u8 > 80
    out[idx] = enh[idx]
    return out


def alpha_blend(base_roi, obj_roi, mask_u8):
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    return (
        alpha * obj_roi.astype(np.float32) + (1 - alpha) * base_roi.astype(np.float32)
    ).astype(np.uint8)


def paste_core_plus_feather(
    base_bgr, obj_bgr, mask_u8, x1, y1, feather_sigma=0.8, core_erode=5
):
    """
    - 코어는 통째로 붙여서 '알약이 흐려보이는 문제' 감소
    - 경계만 feather 블렌딩해서 '티나는 붙이기' 감소
    """
    out = base_bgr.copy()
    h, w = obj_bgr.shape[:2]
    roi = out[y1 : y1 + h, x1 : x1 + w]

    m_bin = (mask_u8 > 80).astype(np.uint8) * 255
    k = max(3, core_erode | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(m_bin, kernel, iterations=1)
    border = cv2.subtract(m_bin, core)

    core_idx = core > 0
    roi[core_idx] = obj_bgr[core_idx]

    if feather_sigma and feather_sigma > 0:
        border_f = cv2.GaussianBlur(border, (0, 0), feather_sigma)
        roi = alpha_blend(roi, obj_bgr, border_f)

    out[y1 : y1 + h, x1 : x1 + w] = roi
    return out


def build_integral_occupancy(h, w, boxes, margin=10):
    occ = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = clamp(int(x1) - margin, 0, w - 1)
        y1 = clamp(int(y1) - margin, 0, h - 1)
        x2 = clamp(int(x2) + margin, 1, w)
        y2 = clamp(int(y2) + margin, 1, h)
        occ[y1:y2, x1:x2] = 1
    integ = cv2.integral(occ)
    return integ


def rect_sum(integ, x1, y1, x2, y2):
    # integ: (h+1, w+1)
    return integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1]


def find_free_position_with_visibility(
    integ, base_bgr, obj_bgr, mask_u8, tries=140, min_delta=8.0
):
    bh, bw = base_bgr.shape[:2]
    oh, ow = obj_bgr.shape[:2]
    if oh >= bh or ow >= bw:
        return None

    # donor 객체 평균 밝기(마스크 내부)
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    obj_gray = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2GRAY)
    obj_mean = (
        float(obj_gray[mask_u8 > 80].mean())
        if np.any(mask_u8 > 80)
        else float(obj_gray.mean())
    )

    for _ in range(tries):
        x1 = random.randint(0, bw - ow)
        y1 = random.randint(0, bh - oh)
        x2 = x1 + ow
        y2 = y1 + oh

        # (1) 겹침 방지(occupancy)
        if rect_sum(integ, x1, y1, x2, y2) > 0:
            continue

        # (2) 가시성: base 배경 밝기와 donor 밝기 차이가 너무 작으면 스킵
        roi_mean = float(base_gray[y1:y2, x1:x2].mean())
        if abs(roi_mean - obj_mean) < min_delta:
            continue

        return (x1, y1, x2, y2)
    return None


# ============================================================
# 9) 증강 계획(부족한 만큼만 + TARGET_CIDS만)
# ============================================================
orig_counts = Counter()
for fname, t in targets_train.items():
    for cid in t["labels"]:
        orig_counts[int(cid)] += 1

need_items = []
for cid in TARGET_CIDS:
    cur = int(orig_counts.get(cid, 0))
    deficit = max(0, int(TARGET_MIN_INST) - cur)
    if deficit <= 0:
        continue
    k = min(deficit, int(MAX_AUG_PER_CLASS))
    need_items.append((cid, k))

need_items = sorted(need_items, key=lambda x: x[1], reverse=True)
max_total_planned = sum(k for _, k in need_items)
MAX_TOTAL_AUG = min(int(MAX_TOTAL_AUG), max_total_planned)

print("\n=== [Target Aug Plan] ===")
for cid, k in need_items:
    print(
        f"- cid={cid:<6} | orig={orig_counts.get(cid, 0):<5} -> +{k:<4} | {categoryid_to_name.get(cid, 'NA')}"
    )
if not need_items:
    print("증강 필요없습니다(모든 타겟 클래스가 이미 목표치 이상).")
    print("원본 변환만 수행했고 종료합니다.")
    raise SystemExit(0)


# ============================================================
# 10) 중복/금지 배경 방지용 base 후보 만들기
#  - "같은 알약이 있는 이미지에 같은 알약 복제 금지" + "3351 포함 배경 금지"
# ============================================================
labels_set_by_fname = {
    fname: set(int(x) for x in targets_train[fname]["labels"])
    for fname in train_img_files
}


def is_banned_base(fname: str):
    # base 이미지에 BAN_BASE_CIDS 중 하나라도 있으면 금지
    return not labels_set_by_fname[fname].isdisjoint(BAN_BASE_CIDS)


base_candidates_by_cid = {}
for cid in cat_ids_sorted:
    # ✅ (중복 방지) base에 이미 cid가 있으면 제외
    # ✅ (배경 금지) base에 3351 등 금지 CID가 있으면 제외
    base_candidates_by_cid[cid] = [
        f
        for f in train_img_files
        if (cid not in labels_set_by_fname[f]) and (not is_banned_base(f))
    ]


# ============================================================
# 11) donor pool(겹침 없는 깨끗한 알약만)
# ============================================================
print("\n== [C] Build donor pool (clean only) ==")
donor_pool = build_clean_donor_pool(targets_train, max_iou_th=DONOR_MAX_IOU_TH)
print("donor classes:", len(donor_pool))


# ============================================================
# 12) Copy-Paste 증강 실행 (train에만 저장)
# ============================================================
print("\n== [D] Generate copy-paste augmented images (TRAIN only) ==")

aug_created_by_cid = Counter()
aug_created_total = 0

for cid, k in need_items:
    if aug_created_total >= MAX_TOTAL_AUG:
        break

    if cid not in donor_pool or len(donor_pool[cid]) == 0:
        print(f"[SKIP] donor 없음: cid={cid} | {categoryid_to_name.get(cid, 'NA')}")
        continue

    base_cands = base_candidates_by_cid.get(cid, [])
    if not base_cands:
        print(
            f"[SKIP] base 후보 없음(중복/금지배경): cid={cid} | {categoryid_to_name.get(cid, 'NA')}"
        )
        continue

    made = 0
    while made < k and aug_created_total < MAX_TOTAL_AUG:
        donor_fname, donor_box = random.choice(donor_pool[cid])
        donor_img = cv2.imread(str(TRAIN_IMG_DIR / donor_fname))
        if donor_img is None:
            continue

        dx1, dy1, dx2, dy2 = map(int, donor_box)
        dx1 = clamp(dx1, 0, donor_img.shape[1] - 1)
        dy1 = clamp(dy1, 0, donor_img.shape[0] - 1)
        dx2 = clamp(dx2, 1, donor_img.shape[1])
        dy2 = clamp(dy2, 1, donor_img.shape[0])

        obj = donor_img[dy1:dy2, dx1:dx2]
        if obj.size == 0:
            continue
        oh, ow = obj.shape[:2]
        if oh < 12 or ow < 12:
            continue

        obj_mask = grabcut_pill_mask(obj, iters=5)
        if obj_mask is None:
            continue

        if USE_CLAHE_ON_OBJ:
            obj = enhance_obj_contrast(obj, obj_mask)

        placed = False
        for __ in range(BASE_TRIES):
            base_fname = random.choice(base_cands)
            base_img = cv2.imread(str(TRAIN_IMG_DIR / base_fname))
            if base_img is None:
                continue

            bh, bw = base_img.shape[:2]
            integ = build_integral_occupancy(
                bh, bw, targets_train[base_fname]["boxes"], margin=OCC_MARGIN
            )

            pos = find_free_position_with_visibility(
                integ,
                base_img,
                obj,
                obj_mask,
                tries=POS_TRIES,
                min_delta=MIN_CONTRAST_DELTA,
            )
            if pos is None:
                continue

            px1, py1, px2, py2 = pos
            base_img_aug = paste_core_plus_feather(
                base_img,
                obj,
                obj_mask,
                px1,
                py1,
                feather_sigma=FEATHER_SIGMA,
                core_erode=CORE_ERODE,
            )

            new_boxes = [b[:] for b in targets_train[base_fname]["boxes"]] + [
                [px1, py1, px2, py2]
            ]
            new_labels = targets_train[base_fname]["labels"][:] + [cid]

            aug_name = f"cp_vis_nodup_{cid}_{aug_created_total:06d}_{Path(base_fname).stem}.png"
            cv2.imwrite(str(OUT_TRAIN_IMG_DIR / aug_name), base_img_aug)
            write_yolo_label(
                OUT_TRAIN_LBL_DIR / f"{Path(aug_name).stem}.txt",
                new_boxes,
                new_labels,
                bw,
                bh,
            )

            placed = True
            break

        if not placed:
            continue

        made += 1
        aug_created_by_cid[cid] += 1
        aug_created_total += 1

    print(f"[DONE] cid={cid} created={made}/{k} | {categoryid_to_name.get(cid, 'NA')}")


print(f"\n✅ 증강 생성 완료: {aug_created_total}장")
print("OUT_ROOT:", OUT_ROOT)


# ============================================================
# 13) 클래스별 요약(증강으로 새로 붙인 개수)
# ============================================================
def is_augmented_file(stem: str):
    return (
        stem.startswith("cp_vis_nodup_")
        or stem.startswith("cp_")
        or stem.startswith("cp_vis_")
        or stem.startswith("cpvis_")
    )


def extract_cid_from_aug_stem(stem: str):
    tokens = stem.split("_")
    for t in tokens:
        if t.isdigit():
            return int(t)
    m = re.search(r"_(\d+)_", "_" + stem + "_")
    return int(m.group(1)) if m else None


def parse_yolo_txt_counts(txt_path: Path):
    c = Counter()
    if not txt_path.exists():
        return c
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        try:
            cls = int(parts[0])
            c[cls] += 1
        except:
            pass
    return c


def summarize_per_class(lbl_dir: Path):
    original_inst = Counter()
    final_inst = Counter()
    augmented_pasted_by_cid = Counter()
    aug_imgs = 0

    for txt in sorted(lbl_dir.glob("*.txt")):
        stem = txt.stem
        cnts = parse_yolo_txt_counts(txt)
        final_inst.update(cnts)

        if is_augmented_file(stem):
            aug_imgs += 1
            cid = extract_cid_from_aug_stem(stem)
            if cid is not None:
                augmented_pasted_by_cid[cid] += 1
        else:
            original_inst.update(cnts)

    rows = []
    for cid, idx in sorted(catid_to_idx.items(), key=lambda x: x[1]):
        name = str(categoryid_to_name.get(cid, f"cat_{cid}"))
        ori = int(original_inst.get(idx, 0))
        fin = int(final_inst.get(idx, 0))
        pasted = int(augmented_pasted_by_cid.get(cid, 0))
        rows.append((idx, cid, name, pasted, ori, fin, fin - ori))
    return rows, aug_imgs


rows, aug_imgs = summarize_per_class(OUT_TRAIN_LBL_DIR)

print("\n=== 증강 결과 요약(TRAIN) ===")
print(f"- 증강 이미지 총 개수: {aug_imgs}")
print(
    "yolo_idx | cat_id | class_name | aug_pasted | orig_inst | final_inst | added_inst"
)
for yolo_idx, cid, name, pasted, ori, fin, added in rows:
    if cid in set(TARGET_CIDS):  # 타겟만 눈에 띄게 보고 싶으면 이 조건 유지
        print(
            f"{yolo_idx:>7} | {cid:>6} | {name[:24]:<24} | {pasted:>10} | {ori:>9} | {fin:>10} | {added:>10}"
        )

# YOLO yaml 힌트 출력
names_list = [categoryid_to_name[cid] for cid in cat_ids_sorted]
dataset_yaml = {
    "path": str(OUT_ROOT),
    "train": "images/train",
    "val": "images/val"
    if (VAL_IMG_DIR and EXPORT_VAL_YOLO)
    else "images/train",  # val 없으면 train로라도
    "names": names_list,
}
(OUT_ROOT / "dataset.yaml").write_text(
    json.dumps(dataset_yaml, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n[dataset.yaml written]", OUT_ROOT / "dataset.yaml")
