# coding: utf-8
#copy-paste2.py (split_yolo 실행 후 전용)

from __future__ import annotations

import os
import re
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# ============================================================
# 0) 설정(필요한 것만 수정)
# ============================================================
random.seed(42)

# 배경(base)로 쓰면 안 되는 "금지 cat_id" (예: 3351 들어간 이미지는 base에서 제외)
BAN_BASE_CIDS = {3351, 3483, 16548}

# 증강할 cat_id 목록 파일(레포 루트 기준)
TARGET_CIDS_FILE = "augment_targets.txt"

# 목표 인스턴스 수(현재 v1(train) 기준으로 부족한 만큼만 증강)
TARGET_MIN_INST = 20

# 폭발 방지
MAX_AUG_PER_CLASS = 100
MAX_TOTAL_AUG = 2000

# 합성 파라미터
BASE_TRIES = 80
POS_TRIES = 140
OCC_MARGIN = 10


# =========================
# base(배경) 선택을 "희소 클래스 우선"으로 할지
# - copy-paste는 base 이미지의 기존 라벨도 그대로 복제되므로,
#   base 선택을 잘못하면 "이미 많은 클래스"가 더 많이 늘어날 수 있음
# =========================
BASE_SELECT_MODE = "rarity"      # "uniform" 또는 "rarity"
BASE_RARITY_ALPHA = 0.1         # 클수록 희소 클래스가 포함된 base를 더 선호 (1.0~4.0 권장)
BASE_PENALIZE_OBJCOUNT = True   # 객체(박스) 많은 base는 복제 부작용이 커서 덜 뽑기
BASE_SCORE_EPS = 1e-9
# 가시성(visibility)
USE_CLAHE_ON_OBJ = True
MIN_CONTRAST_DELTA = 8.0
FEATHER_SIGMA = 0.8
CORE_ERODE = 5

# donor clean 필터(겹친 알약은 donor에서 제외)
DONOR_MAX_IOU_TH = 0.05

# 출력 증강 이미지 파일명 prefix
AUG_PREFIX = "cp_vis_nodup_"


# ============================================================
# 1) DATA_ROOT / REPO_ROOT 잡기
# - split_yolo는 Dataset_load을 쓰지만, 여기선 .env(LOG_FILE_PATH)를 기본으로 사용
# - .env가 없으면 Dataset.Dataset_load.DATA_ROOT를 fallback
# ============================================================

def find_repo_root(start: Path) -> Path:
    """start에서 위로 올라가며 레포 루트로 보이는 폴더를 찾는다.

    우선순위(먼저 발견되는 것):
    - .env (LOG_FILE_PATH를 쓰기 때문에 가장 신뢰)
    - .git
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / ".env").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "Dataset").exists():
            return p
    return start


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)

DATA_ROOT = None
if load_dotenv is not None and (REPO_ROOT / ".env").exists():
    load_dotenv(REPO_ROOT / ".env")
    env_root = os.getenv("LOG_FILE_PATH")
    if env_root:
        DATA_ROOT = Path(env_root)

if DATA_ROOT is None:
    # fallback: split_yolo가 사용한 Dataset_load에서 DATA_ROOT 가져오기
    try:
        from Dataset.Dataset_load import DATA_ROOT as _DATA_ROOT  # type: ignore

        DATA_ROOT = Path(_DATA_ROOT)
    except Exception as e:
        raise RuntimeError(
            "DATA_ROOT를 찾지 못했습니다. (.env의 LOG_FILE_PATH를 설정하거나 Dataset/Dataset_load.py를 확인하세요.)"
        ) from e

print("[DATA_ROOT]", DATA_ROOT)

YOLO_PARENT = DATA_ROOT.parent

V0_ROOT = YOLO_PARENT / "yolo_dataset"       # split_yolo 출력 (원본 YOLO)
V1_ROOT = YOLO_PARENT / "yolo_dataset_aug"   # split_yolo가 만들어준 scaffold (여기에 증강본 추가)

V0_TRAIN_IMG_DIR = V0_ROOT / "images" / "train"
V0_TRAIN_LBL_DIR = V0_ROOT / "labels" / "train"
V0_VAL_IMG_DIR = V0_ROOT / "images" / "val"
V0_VAL_LBL_DIR = V0_ROOT / "labels" / "val"

V1_TRAIN_IMG_DIR = V1_ROOT / "images" / "train"
V1_TRAIN_LBL_DIR = V1_ROOT / "labels" / "train"
V1_VAL_IMG_DIR = V1_ROOT / "images" / "val"
V1_VAL_LBL_DIR = V1_ROOT / "labels" / "val"

for p in [V0_TRAIN_IMG_DIR, V0_TRAIN_LBL_DIR]:
    if not p.exists():
        raise FileNotFoundError(
            f"split_yolo 결과가 없습니다: {p}\n먼저 split_yolo.py를 실행해서 yolo_dataset을 만들어주세요."
        )


# ============================================================
# 2) v1 scaffold가 없으면 v0를 복사해서 만들기(안전장치)
# - split_yolo.py가 이미 만들었으면 스킵
# ============================================================

def _dir_nonempty(d: Path) -> bool:
    return d.exists() and any(d.iterdir())


def _copy_all_files(src: Path, dst: Path, exts: set[str] | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out = dst / p.name
        if not out.exists():
            shutil.copy2(p, out)


if not _dir_nonempty(V1_TRAIN_IMG_DIR) or not _dir_nonempty(V1_TRAIN_LBL_DIR):
    print("[v1 scaffold] yolo_dataset_aug 가 비어있어서 v0에서 복사합니다.")
    for sub in [
        V1_TRAIN_IMG_DIR,
        V1_TRAIN_LBL_DIR,
        V1_VAL_IMG_DIR,
        V1_VAL_LBL_DIR,
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    _copy_all_files(V0_TRAIN_IMG_DIR, V1_TRAIN_IMG_DIR)
    _copy_all_files(V0_TRAIN_LBL_DIR, V1_TRAIN_LBL_DIR, exts={".txt"})

    if V0_VAL_IMG_DIR.exists():
        _copy_all_files(V0_VAL_IMG_DIR, V1_VAL_IMG_DIR)
    if V0_VAL_LBL_DIR.exists():
        _copy_all_files(V0_VAL_LBL_DIR, V1_VAL_LBL_DIR, exts={".txt"})

    # dataset.yaml / split.json도 복사(있으면)
    for fn in ["dataset.yaml", "split.json"]:
        s = V0_ROOT / fn
        if s.exists() and not (V1_ROOT / fn).exists():
            shutil.copy2(s, V1_ROOT / fn)


# ============================================================
# 3) split.json에서 mapping 로드 (cat_id <-> yolo cls_id)
# ============================================================

def load_mapping_from_split_json(split_json_path: Path):
    if not split_json_path.exists():
        raise FileNotFoundError(f"split.json이 없습니다: {split_json_path}")

    data = json.loads(split_json_path.read_text(encoding="utf-8"))
    mapping = data.get("mapping", {})

    # json 저장 시 dict key가 str로 바뀌므로 복원
    catid_to_cls_raw = mapping.get("catid_to_cls", {})
    cls_to_catid_raw = mapping.get("cls_to_catid", {})
    cls_to_name = mapping.get("cls_to_name", [])

    catid_to_cls = {int(k): int(v) for k, v in catid_to_cls_raw.items()}
    cls_to_catid = {int(k): int(v) for k, v in cls_to_catid_raw.items()}

    # category 이름은 split_yolo에서 cls_to_name(list)로 저장해둠
    cls_to_name = list(cls_to_name)

    return catid_to_cls, cls_to_catid, cls_to_name


catid_to_cls, cls_to_catid, cls_to_name = load_mapping_from_split_json(V0_ROOT / "split.json")

print("[mapping] nc:", len(cls_to_name))


# ============================================================
# 4) TARGET_CIDS 로드 (augment_targets.txt)
# ============================================================

def load_cids_from_file(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"TARGET CID 파일이 없습니다: {path}")

    txt = path.read_text(encoding="utf-8")
    cids = [int(x) for x in re.findall(r"\d+", txt)]

    out: list[int] = []
    seen = set()
    for x in cids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_target_file(repo_root: Path, data_root: Path, filename: str) -> Path:
    cands = [
        repo_root / filename,
        Path.cwd() / filename,
        data_root / filename,
        (data_root / "yolo_dataset" / filename),
        (data_root.parents[1] / "augment" / filename),
    ]
    for p in cands:
        if p.exists():
            return p
    # 못 찾으면 후보 경로를 전부 보여주고 FileNotFoundError
    msg = "\n".join([f"- {p}" for p in cands])
    raise FileNotFoundError(
        f"TARGET CID 파일이 없습니다: {filename}\n아래 위치 중 한 곳에 파일을 두거나, 경로를 수정하세요:\n{msg}"
    )


TARGET_FILE = resolve_target_file(REPO_ROOT, DATA_ROOT, TARGET_CIDS_FILE)
print("[TARGET_FILE]", TARGET_FILE)

TARGET_CIDS = load_cids_from_file(TARGET_FILE)
if not TARGET_CIDS:
    raise ValueError("TARGET_CIDS가 비어있습니다. augment_targets.txt를 확인하세요.")

# mapping에 없는 target은 경고(증강 불가)
unknown_targets = [c for c in TARGET_CIDS if c not in catid_to_cls]
if unknown_targets:
    print("[WARN] split.json mapping에 없는 cat_id가 포함됨(증강 불가, 앞 20개):", unknown_targets[:20])

TARGET_CIDS = [c for c in TARGET_CIDS if c in catid_to_cls]
print("[TARGET_CIDS]", len(TARGET_CIDS))


# ============================================================
# 5) 유틸: YOLO txt <-> abs box 변환
# ============================================================

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def list_images(img_dir: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_yolo_label_as_abs(
    txt_path: Path,
    img_w: int,
    img_h: int,
):
    """YOLO txt를 읽어서 abs xyxy boxes + cls_ids 반환"""
    boxes: list[list[float]] = []
    cls_ids: list[int] = []

    if not txt_path.exists():
        return boxes, cls_ids

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
        except Exception:
            continue

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # clip
        x1 = float(max(0.0, min(x1, img_w - 1.0)))
        y1 = float(max(0.0, min(y1, img_h - 1.0)))
        x2 = float(max(0.0, min(x2, img_w - 1.0)))
        y2 = float(max(0.0, min(y2, img_h - 1.0)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        cls_ids.append(cls)

    return boxes, cls_ids


def write_yolo_label_from_abs(
    txt_path: Path,
    boxes_xyxy: list[list[float]],
    cls_ids: list[int],
    img_w: int,
    img_h: int,
):
    """abs xyxy + cls_ids -> YOLO txt 저장"""
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue

        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        # normalize
        cx /= float(img_w)
        cy /= float(img_h)
        bw /= float(img_w)
        bh /= float(img_h)

        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ============================================================
# 6) train targets를 YOLO 라벨에서 재구성
# - donor/base 선택/occupancy에 필요
# ============================================================

def build_targets_from_yolo(img_dir: Path, lbl_dir: Path):
    targets = defaultdict(lambda: {"boxes": [], "cls": [], "cat": []})

    fnames = list_images(img_dir)
    for fname in fnames:
        img_path = img_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes, cls_ids = read_yolo_label_as_abs(lbl_dir / f"{Path(fname).stem}.txt", w, h)
        cat_ids = [cls_to_catid.get(int(c), -1) for c in cls_ids]

        targets[fname]["boxes"] = boxes
        targets[fname]["cls"] = cls_ids
        targets[fname]["cat"] = cat_ids

    return targets


print("\n== [Load targets from v0 train YOLO] ==")
targets_train_v0 = build_targets_from_yolo(V0_TRAIN_IMG_DIR, V0_TRAIN_LBL_DIR)
train_img_files = sorted(list(targets_train_v0.keys()))
print("train images (v0):", len(train_img_files))


# ============================================================
# 7) 현재(v1) 기준 클래스 분포 계산 -> 부족분만 증강
# - 재실행해도 과도하게 늘어나는 걸 줄이기 위해 v1 라벨을 기준으로 계산
# ============================================================

def count_cat_instances_in_labels(labels_dir: Path) -> Counter:
    cnt = Counter()
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls = int(line.split()[0])
            except Exception:
                continue
            cat = cls_to_catid.get(cls, None)
            if cat is not None:
                cnt[int(cat)] += 1
    return cnt


orig_counts_v1 = count_cat_instances_in_labels(V1_TRAIN_LBL_DIR)

need_items: list[tuple[int, int]] = []  # (cat_id, need)
for cat_id in TARGET_CIDS:
    cur = int(orig_counts_v1.get(cat_id, 0))
    deficit = max(0, int(TARGET_MIN_INST) - cur)
    if deficit <= 0:
        continue
    need_items.append((cat_id, min(deficit, int(MAX_AUG_PER_CLASS))))

need_items.sort(key=lambda x: x[1], reverse=True)
planned_total = sum(k for _, k in need_items)
MAX_TOTAL_AUG = min(int(MAX_TOTAL_AUG), planned_total)

print("\n=== [Target Aug Plan] (based on v1 current counts) ===")
for cat_id, k in need_items:
    name = cls_to_name[catid_to_cls[cat_id]] if cat_id in catid_to_cls else "NA"
    print(f"- cat_id={cat_id:<6} | cur={orig_counts_v1.get(cat_id,0):<5} -> +{k:<4} | {name}")

if not need_items:
    print("증강 필요없습니다(모든 타겟 클래스가 이미 목표치 이상). 종료합니다.")
    raise SystemExit(0)


# ============================================================
# 8) base 후보 만들기 (중복/금지 배경 방지)
# ============================================================
labels_set_by_fname = {
    fname: set(int(x) for x in targets_train_v0[fname]["cat"] if int(x) >= 0)
    for fname in train_img_files
}


def is_banned_base(fname: str) -> bool:
    # base 이미지에 BAN_BASE_CIDS 중 하나라도 있으면 금지
    return not labels_set_by_fname[fname].isdisjoint(BAN_BASE_CIDS)


base_candidates_by_cat = {}
for cat_id in TARGET_CIDS:
    base_candidates_by_cat[cat_id] = [
        f
        for f in train_img_files
        if (cat_id not in labels_set_by_fname[f]) and (not is_banned_base(f))
    ]



# ============================================================
# 8.5) base(배경) "희소 클래스 우선" 가중치 준비
# - base 이미지 안의 클래스들이 현재(v1)에서 희소할수록 base 선택 확률을 높임
# - 목적: copy-paste로 base의 기존 라벨이 같이 복제되는 부작용을 줄이기
# ============================================================
import math

# v1의 현재 분포(orig_counts_v1)를 이용해 "희소도"를 계산한다.
# (v0만 보면 이미 증강이 진행된 상황을 반영 못해서 v1 기준이 안전함)
label_freq_for_score = orig_counts_v1  # Counter(cat_id -> inst count)

def _compute_base_score(fname: str) -> float:
    # 이 base 이미지에 들어있는 클래스 set (cat_id 기준)
    cats = labels_set_by_fname.get(fname, set())
    if not cats:
        score = 1.0  # 라벨 없는 배경이면 복제 부작용이 적어 점수 높게
    else:
        inv = [1.0 / (float(label_freq_for_score.get(c, 0)) + 1.0) for c in cats]
        score = float(sum(inv) / max(1, len(inv)))

    if BASE_PENALIZE_OBJCOUNT:
        # 객체가 많을수록 (1) 빈 공간 찾기 어렵고 (2) 복제되는 라벨이 많아 부작용이 커서 페널티
        n_inst = len(targets_train_v0[fname]["cat"])
        score *= 1.0 / math.sqrt(max(1, n_inst))

    return max(score, 1e-12)

# base별 점수(한 번만 계산)
base_score = {f: _compute_base_score(f) for f in train_img_files}

def _weighted_choice(items: list[str], weights: list[float]) -> str:
    total = float(sum(weights))
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += float(w)
        if acc >= r:
            return it
    return items[-1]

def pick_base_from_candidates(base_cands: list[str]) -> str:
    """base 후보 중 하나를 선택한다.
    - uniform: 균등 랜덤
    - rarity: 희소 클래스(base_score↑)가 포함된 이미지를 가중 랜덤으로 더 자주 선택
    """
    if BASE_SELECT_MODE != "rarity":
        return random.choice(base_cands)

    ws = [((base_score.get(f, 1e-12) + BASE_SCORE_EPS) ** float(BASE_RARITY_ALPHA)) for f in base_cands]
    return _weighted_choice(base_cands, ws)

print(f"[BASE_SELECT_MODE] {BASE_SELECT_MODE} | alpha={BASE_RARITY_ALPHA} | penalize_objcount={BASE_PENALIZE_OBJCOUNT}")

# ============================================================
# 9) donor pool (겹침 없는 깨끗한 객체만)
# ============================================================

def box_iou(a, b) -> float:
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
    """donor 후보 중에서 같은 이미지 내 다른 객체와 거의 안 겹치는 것만 사용"""
    donor_pool: dict[int, list[tuple[str, list[float]]]] = {}

    for fname, t in targets_by_filename.items():
        boxes = t["boxes"]
        cat_ids = t["cat"]
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
                cat_id = int(cat_ids[i])
                if cat_id < 0:
                    continue
                donor_pool.setdefault(cat_id, []).append((fname, bi))

    return donor_pool


print("\n== [Build donor pool from v0 train] ==")
donor_pool = build_clean_donor_pool(targets_train_v0, max_iou_th=DONOR_MAX_IOU_TH)
print("donor classes:", len(donor_pool))


# ============================================================
# 10) 마스크/블렌딩 유틸
# ============================================================

def grabcut_pill_mask(obj_bgr: np.ndarray, iters: int = 5):
    """donor crop에서 GrabCut으로 마스크 추출"""
    h, w = obj_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(1, w - 4), max(1, h - 4))

    try:
        cv2.grabCut(obj_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

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


def enhance_obj_contrast(obj_bgr: np.ndarray, mask_u8: np.ndarray):
    """마스크 내부에만 CLAHE 적용"""
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


def alpha_blend(base_roi: np.ndarray, obj_roi: np.ndarray, mask_u8: np.ndarray):
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    return (alpha * obj_roi.astype(np.float32) + (1.0 - alpha) * base_roi.astype(np.float32)).astype(np.uint8)


def paste_core_plus_feather(
    base_bgr: np.ndarray,
    obj_bgr: np.ndarray,
    mask_u8: np.ndarray,
    x1: int,
    y1: int,
    feather_sigma: float = 0.8,
    core_erode: int = 5,
):
    """코어는 선명하게, 경계만 feather 블렌딩"""
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


def build_integral_occupancy(h: int, w: int, boxes, margin: int = 10):
    occ = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = clamp(int(x1) - margin, 0, w - 1)
        y1 = clamp(int(y1) - margin, 0, h - 1)
        x2 = clamp(int(x2) + margin, 1, w)
        y2 = clamp(int(y2) + margin, 1, h)
        occ[y1:y2, x1:x2] = 1
    return cv2.integral(occ)


def rect_sum(integ, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1])


def mean_gray_under_mask(gray: np.ndarray, mask_u8: np.ndarray) -> float:
    m = mask_u8 > 80
    if not np.any(m):
        return float(gray.mean())
    return float(gray[m].mean())


def find_free_position_with_visibility(
    integ,
    base_bgr: np.ndarray,
    obj_bgr: np.ndarray,
    mask_u8: np.ndarray,
    tries: int = 140,
    min_delta: float = 8.0,
):
    bh, bw = base_bgr.shape[:2]
    oh, ow = obj_bgr.shape[:2]
    if oh >= bh or ow >= bw:
        return None

    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    obj_gray = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2GRAY)

    obj_mean = mean_gray_under_mask(obj_gray, mask_u8)

    for _ in range(tries):
        x1 = random.randint(0, bw - ow)
        y1 = random.randint(0, bh - oh)
        x2 = x1 + ow
        y2 = y1 + oh

        # (1) 기존 객체와 겹치면 스킵
        if rect_sum(integ, x1, y1, x2, y2) > 0:
            continue

        # (2) 가시성: 배경이랑 밝기 차가 너무 작으면 스킵
        roi_gray = base_gray[y1:y2, x1:x2]
        roi_mean = mean_gray_under_mask(roi_gray, mask_u8)
        if abs(roi_mean - obj_mean) < min_delta:
            continue

        return (x1, y1, x2, y2)

    return None


# ============================================================
# 11) 증강 이미지 인덱스(재실행 대비)
# ============================================================

def next_aug_index(img_dir: Path, prefix: str) -> int:
    max_idx = -1
    for p in img_dir.glob(f"{prefix}*.png"):
        m = re.search(r"_(\d{6})_", p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


start_idx = next_aug_index(V1_TRAIN_IMG_DIR, AUG_PREFIX)


# ============================================================
# 12) Copy-Paste 증강 실행 (v0에서 donor/base, v1에 저장)
# ============================================================
print("\n== [Augment] Generate copy-paste images -> v1/train ==")

aug_created_by_cat = Counter()
aug_created_total = 0

for cat_id, k in need_items:
    if aug_created_total >= MAX_TOTAL_AUG:
        break

    if cat_id not in donor_pool or len(donor_pool[cat_id]) == 0:
        print(f"[SKIP] donor 없음: cat_id={cat_id}")
        continue

    base_cands = base_candidates_by_cat.get(cat_id, [])
    if not base_cands:
        print(f"[SKIP] base 후보 없음(중복/금지배경): cat_id={cat_id}")
        continue

    made = 0
    while made < k and aug_created_total < MAX_TOTAL_AUG:
        donor_fname, donor_box = random.choice(donor_pool[cat_id])
        donor_img = cv2.imread(str(V0_TRAIN_IMG_DIR / donor_fname))
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
        for _ in range(BASE_TRIES):
            base_fname = pick_base_from_candidates(base_cands)
            base_img = cv2.imread(str(V0_TRAIN_IMG_DIR / base_fname))
            if base_img is None:
                continue

            bh, bw = base_img.shape[:2]
            base_boxes = targets_train_v0[base_fname]["boxes"]
            base_cls = targets_train_v0[base_fname]["cls"]

            integ = build_integral_occupancy(bh, bw, base_boxes, margin=OCC_MARGIN)

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
            out_img = paste_core_plus_feather(
                base_img,
                obj,
                obj_mask,
                px1,
                py1,
                feather_sigma=FEATHER_SIGMA,
                core_erode=CORE_ERODE,
            )

            # 라벨: base 라벨 + 새 객체 라벨
            new_boxes = [b[:] for b in base_boxes] + [[float(px1), float(py1), float(px2), float(py2)]]
            new_cls = list(base_cls) + [int(catid_to_cls[cat_id])]

            idx = start_idx + aug_created_total
            aug_name = f"{AUG_PREFIX}{cat_id}_{idx:06d}_{Path(base_fname).stem}.png"

            cv2.imwrite(str(V1_TRAIN_IMG_DIR / aug_name), out_img)
            write_yolo_label_from_abs(
                V1_TRAIN_LBL_DIR / f"{Path(aug_name).stem}.txt",
                new_boxes,
                new_cls,
                bw,
                bh,
            )

            placed = True
            break

        if not placed:
            continue

        made += 1
        aug_created_by_cat[cat_id] += 1
        aug_created_total += 1

    print(f"[DONE] cat_id={cat_id} created={made}/{k} | cls={catid_to_cls[cat_id]} | {cls_to_name[catid_to_cls[cat_id]]}")


print(f"\n✅ 증강 생성 완료: {aug_created_total}장")
print("V1_ROOT:", V1_ROOT)


# ============================================================
# 13) 요약
# - created: 타겟별로 몇 장 만들었는지
# - counts : 증강 후(v1) "전체 클래스" 인스턴스 분포 (타겟은 * 표시)
# ============================================================

print("=== [Summary] created images per target cat_id ===")
for cat_id in TARGET_CIDS:
    if cat_id in aug_created_by_cat:
        cls = catid_to_cls.get(cat_id, -1)
        name = cls_to_name[cls] if (0 <= cls < len(cls_to_name)) else "NA"
        print(f"- cat_id={cat_id:<6} +{aug_created_by_cat[cat_id]:<4} | cls={cls:<3} | {name}")

# v1 기준 최신 분포 (전체)
print ("=== [v1 counts after augmentation] (ALL classes, * = target) ===")
new_counts_v1 = count_cat_instances_in_labels(V1_TRAIN_LBL_DIR)
target_set = set(TARGET_CIDS)

# yolo cls 순서로 출력(학습에서 보는 cls 인덱스와 동일한 순서)
all_cat_ids_in_yolo_order = sorted(catid_to_cls.keys(), key=lambda c: catid_to_cls[c])

print("mark | cls | cat_id | count | class_name")
for cat_id in all_cat_ids_in_yolo_order:
    cls = catid_to_cls.get(cat_id, -1)
    name = cls_to_name[cls] if (0 <= cls < len(cls_to_name)) else "NA"
    mark = "*" if cat_id in target_set else " "
    cnt = int(new_counts_v1.get(cat_id, 0))
    print(f" {mark}   | {cls:>3} | {cat_id:>6} | {cnt:>5} | {name}")

# imbalance 체크용(옵션): top/bottom 일부만 보기
items = [(catid_to_cls[c], c, int(new_counts_v1.get(c, 0))) for c in all_cat_ids_in_yolo_order]
items_desc = sorted(items, key=lambda x: x[2], reverse=True)

print("=== [Top 20 classes by count] (* = target) ===")
for cls, cat_id, cnt in items_desc[:20]:
    mark = "*" if cat_id in target_set else " "
    name = cls_to_name[cls] if (0 <= cls < len(cls_to_name)) else "NA"
    print(f" {mark} cls={cls:<3} cat_id={cat_id:<6} count={cnt:<6} | {name}")

nonzero = [it for it in items_desc if it[2] > 0]
items_asc_nonzero = sorted(nonzero, key=lambda x: x[2])

print("=== [Bottom 20 (non-zero) classes by count] (* = target) ===")
for cls, cat_id, cnt in items_asc_nonzero[:20]:
    mark = "*" if cat_id in target_set else " "
    name = cls_to_name[cls] if (0 <= cls < len(cls_to_name)) else "NA"
    print(f" {mark} cls={cls:<3} cat_id={cat_id:<6} count={cnt:<6} | {name}")
