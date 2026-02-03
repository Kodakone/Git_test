import os
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드
load_dotenv()

# 레포루트(Code_it-Basic-Project)
REPO_ROOT = Path(__file__).resolve().parents[1]

# .env를 정확히 지정해서 로드
load_dotenv(REPO_ROOT / ".env")

# .env에서 데이터 루트 가져오기
DATA_ROOT = os.getenv("LOG_FILE_PATH")
if DATA_ROOT is None:
    raise ValueError("LOG_FILE_PATH가 .env에 설정되어 있지 않습니다.")

DATA_ROOT = Path(DATA_ROOT)
print('data root :', DATA_ROOT)

TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test_images"
ANNOTATION_DIR = DATA_ROOT / "train_annotations"
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------

# 경로지정 ---------------------------
from Dataset.Dataset_load import TRAIN_IMG_DIR, TEST_IMG_DIR, ANNOTATION_DIR, CACHE_DIR, REPO_ROOT, DATA_ROOT

print(TRAIN_IMG_DIR)
print(len(os.listdir(TRAIN_IMG_DIR)))
print(TEST_IMG_DIR)
print(len(os.listdir(TEST_IMG_DIR)))
print(ANNOTATION_DIR)
print(len(os.listdir(ANNOTATION_DIR)))

# 파일 불러오기 ---------------------------
# png 이미지파일
train_img_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR)if f.lower().endswith(".png")])
print(f"image 파일 개수: {len(train_img_files)}") #232

# 모든 annotaion json의 전체경로 리스트
json_files = []  
for root, dirs, files in os.walk(ANNOTATION_DIR):
    for file in files:
        if file.lower().endswith(".json"):
            json_files.append(os.path.join(root, file))

print("총 json 개수:", len(json_files)) #763


# ground truth역할 ---------------------------
# 학습은 이미지기준으로 해야하므로 json에 흩어진 정보를 이미지 기준으로 모음
# json훑어 이미지파일명(file_name)기준 해당이미지에 대응되는 박스/라벨로 딕셔너리 만들기
import json
from collections import defaultdict

train_img_set = set(train_img_files)

# 이미지 한장당 객체 여러개를 담기위한 구조
# ex. "image1.png": {"boxes": [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],"labels": [1, 1, 1, ...]}
targets_by_filename = defaultdict(lambda: {"boxes": [], "labels": []})

# 카테고리 매핑할 딕셔너리
categoryid_to_name = {}

for jp in json_files:
    with open(jp, "r", encoding='utf-8') as f:
        coco = json.load(f)

    # 카테고리id-> name 매핑
    for c in coco.get("categories", []):
        categoryid_to_name[c["id"]] = c.get("name", str(c["id"])) #ex. {1900:'보령부스파정',...}

    # image_id → file_name 매핑 (train에 있는 것만)
    # annotations에는 image_id만 있는데 실제 이미지파일명은 images에 있음
    imageid_to_fname = {}
    for img in coco.get("images", []):
        fname = img.get("file_name")
        if fname in train_img_set:               #훈련폴더에 실제 있는 파일만
            imageid_to_fname[img["id"]] = fname  #ex. {34: "34.png",  ...}

    if not imageid_to_fname:
        continue         # 이 json에 train_images 폴더에 있는 이미지가 없을 때

    # annotations: image_id별로 boxes/labels 모으기
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")      # ex.34
        if img_id not in imageid_to_fname:
            continue

        fname = imageid_to_fname[img_id]  # ex. image_id:34 이면 "34.png"찾음
        cat_id = ann.get("category_id")   # ex. 1900

        # 좌표변환
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = x,  y,  x + w,  y + h   # (x_min, y_min, x_max, y_max)

        targets_by_filename[fname]["boxes"].append([x1, y1, x2, y2])
        targets_by_filename[fname]["labels"].append(cat_id)

print("train 이미지 수:", len(train_img_files))
print("annotation이 붙은 이미지 수:", len(targets_by_filename))

# 샘플 확인
sample_name = next(iter(targets_by_filename))
print("샘플 이미지:", sample_name)
print("박스 개수:", len(targets_by_filename[sample_name]["boxes"]))
print("라벨:", targets_by_filename[sample_name]["labels"])

# 캐시저장 ---------------------------
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# gound truth cache
gt_cache_path = CACHE_DIR / "targets_by_filename.json"

if gt_cache_path.exists():  #있으면 읽고
    with open(gt_cache_path, "r", encoding="utf-8") as f:
        targets_by_filename = json.load(f)

    targets_by_filename = defaultdict(
        lambda: {"boxes": [], "labels": []},
        targets_by_filename
    )
    print("loaded:", gt_cache_path)

else:                    # 없으면 만들기
    with open(gt_cache_path, "w", encoding="utf-8") as f:
        json.dump(dict(targets_by_filename), f)
    print("saved:", gt_cache_path)

# category id cache
cat_cache_path = CACHE_DIR / "categoryid_to_name.json"

if cat_cache_path.exists():
    with open(cat_cache_path, "r", encoding="utf-8") as f:
        categoryid_to_name = json.load(f)

    categoryid_to_name = {int(k): v for k, v in categoryid_to_name.items()}
    print("loaded:", cat_cache_path)

else:
    with open(cat_cache_path, "w", encoding="utf-8") as f:
        json.dump(categoryid_to_name, f, ensure_ascii=False)
    print("saved:", cat_cache_path)



#==================copy-paste============================
# (개선사항 적용 버전)
# 1) 가시성 개선: seamlessClone 대신 "코어는 그대로 복사 + 경계만 feather"
# 2) 중복 방지: base 이미지에 이미 같은 category_id(같은 알약)가 있으면 그 이미지에는 붙이지 않음
# 3) 겹침 방지: occupancy(integral)로 기존 bbox 영역은 절대 침범하지 않음
# ============================================================

import random
import shutil
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
import json

random.seed(42)

# -------------------------
# 출력 폴더(원본 + 증강 이미지/라벨 저장)
# -------------------------
OUT_IMG_DIR = DATA_ROOT / "train_images_copypaste"
OUT_LBL_DIR = DATA_ROOT / "train_labels_copypaste"
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 증강 목표/제한
# -------------------------
TARGET_PER_CLASS = 30      # 클래스당 목표 인스턴스 수
MAX_AUG_PER_CLASS = 80     # 클래스별 최대 증강 수
MAX_TOTAL_AUG = 3000       # 전체 증강 최대
BASE_TRIES = 60            # base 이미지 갈아끼우는 시도
POS_TRIES = 120            # base 내 위치 랜덤 시도
OCC_MARGIN = 10            # 기존 bbox 주변 여백(겹침 방지 강화)

# 가시성 관련
USE_CLAHE_ON_OBJ = True    # donor 알약 대비 살리기
MIN_CONTRAST_DELTA = 8.0   # base 배경과 너무 비슷한 위치면 붙이지 않음(가시성↑)
FEATHER_SIGMA = 0.8        # 경계 feather 정도
CORE_ERODE = 5             # 코어 영역 크기(클수록 내부 픽셀 유지↑)

# -------------------------
# cat_id -> yolo class index
# -------------------------
cat_ids_sorted = sorted([int(k) for k in categoryid_to_name.keys()])
catid_to_idx = {cid: i for i, cid in enumerate(cat_ids_sorted)}
(CACHE_DIR / "catid_to_idx.json").write_text(
    json.dumps(catid_to_idx, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# -------------------------
# 유틸
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def write_yolo_label(txt_path: Path, boxes, labels, img_w, img_h):
    # YOLO: cls cx cy w h (normalized)
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
        lines.append(f"{cls} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

# -------------------------
# (1) donor 중 "가려진/겹친 알약" 제거 -> 모양 깨짐 방지
# -------------------------
def build_clean_donor_pool(targets_by_filename, max_iou_th=0.05):
    donor_pool = {}
    for fname, t in targets_by_filename.items():
        boxes = t["boxes"]
        labels = t["labels"]
        n = len(boxes)
        if n == 0:
            continue

        # 각 객체가 같은 이미지 내 다른 객체와 얼마나 겹치는지 체크
        for i in range(n):
            bi = boxes[i]
            mx = 0.0
            for j in range(n):
                if i == j:
                    continue
                mx = max(mx, box_iou(bi, boxes[j]))

            # 겹침이 작으면 clean donor로 사용
            if mx <= max_iou_th:
                cid = int(labels[i])
                donor_pool.setdefault(cid, []).append((fname, bi))
    return donor_pool

# -------------------------
# (2) base에서 빈 공간 찾기: occupancy + integral
# -------------------------
def build_integral_occupancy(h, w, boxes, margin=10):
    occ = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        cv2.rectangle(occ, (x1, y1), (x2, y2), 255, -1)
    integ = cv2.integral(occ)  # (h+1, w+1)
    return integ

def region_sum(integ, x1, y1, x2, y2):
    return integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1]

def mean_gray_under_mask(bgr, mask_u8, thr=80):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = mask_u8 > thr
    if m.sum() == 0:
        return float(gray.mean())
    return float(gray[m].mean())

def find_free_position_with_visibility(integ, base_bgr, obj_bgr, obj_mask_u8, tries=120, min_delta=8.0):
    bh, bw = base_bgr.shape[:2]
    oh, ow = obj_bgr.shape[:2]
    if ow >= bw or oh >= bh:
        return None

    obj_mean = mean_gray_under_mask(obj_bgr, obj_mask_u8)

    for _ in range(tries):
        x1 = random.randint(0, bw - ow)
        y1 = random.randint(0, bh - oh)
        x2 = x1 + ow
        y2 = y1 + oh

        # 1) 겹침 완전 금지
        if region_sum(integ, x1, y1, x2, y2) != 0:
            continue

        # 2) 가시성: 배경과 너무 비슷하면 스킵
        if min_delta > 0:
            roi = base_bgr[y1:y2, x1:x2]
            base_mean = mean_gray_under_mask(roi, obj_mask_u8)
            if abs(obj_mean - base_mean) < float(min_delta):
                continue

        return (x1, y1, x2, y2)

    return None

# -------------------------
# (3) GrabCut으로 donor 알약 마스크 근사
# -------------------------
def grabcut_pill_mask(obj_bgr, iters=5):
    h, w = obj_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (2, 2, max(1, w-4), max(1, h-4))
    try:
        cv2.grabCut(obj_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    except:
        return None

    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = max(3, (min(h, w)//20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (h*w*0.15):
        return None

    clean = np.zeros_like(m)
    cv2.drawContours(clean, [cnt], -1, 255, -1)

    # 경계 부드럽게
    clean = cv2.GaussianBlur(clean, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return clean

# -------------------------
# (4) 가시성 개선: 코어 그대로 + 경계 feather
# -------------------------
def alpha_blend(base_roi, obj_roi, mask_u8):
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    return (alpha * obj_roi.astype(np.float32) + (1 - alpha) * base_roi.astype(np.float32)).astype(np.uint8)

def paste_core_plus_feather(base_bgr, obj_bgr, mask_u8, x1, y1, feather_sigma=0.8, core_erode=5):
    out = base_bgr.copy()
    h, w = obj_bgr.shape[:2]
    roi = out[y1:y1+h, x1:x1+w]

    m_bin = (mask_u8 > 80).astype(np.uint8) * 255

    # 코어/경계 분리
    k = max(3, core_erode | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(m_bin, kernel, iterations=1)
    border = cv2.subtract(m_bin, core)

    # 1) 코어는 그대로 복사(가시성 유지)
    core_idx = core > 0
    roi[core_idx] = obj_bgr[core_idx]

    # 2) 경계만 feather 블렌딩
    if feather_sigma and feather_sigma > 0:
        border_f = cv2.GaussianBlur(border, (0, 0), feather_sigma)
        roi = alpha_blend(roi, obj_bgr, border_f)

    out[y1:y1+h, x1:x1+w] = roi
    return out

def enhance_obj_contrast(obj_bgr, mask_u8):
    # 마스크 내부만 CLAHE로 대비 살리기(가시성↑, 각인 학습에도 도움)
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

# ============================================================
# 실행부 A) 원본 복사 + YOLO 라벨 생성
# ============================================================
print("\n== [A] Copy originals + write YOLO labels ==")

for fname in train_img_files:
    src = TRAIN_IMG_DIR / fname
    if not src.exists():
        continue

    img = cv2.imread(str(src))
    if img is None:
        continue
    h, w = img.shape[:2]

    # 원본 이미지 복사
    dst = OUT_IMG_DIR / fname
    if not dst.exists():
        shutil.copy2(src, dst)

    # 라벨 저장(없으면 빈 txt)
    t = targets_by_filename[fname]  # defaultdict: 없으면 빈 리스트
    write_yolo_label(OUT_LBL_DIR / f"{Path(fname).stem}.txt", t["boxes"], t["labels"], w, h)

# ============================================================
# 실행부 B) 중복 방지용: 이미지별 포함 클래스 set 만들기
# ============================================================
labels_set_by_fname = {
    fname: set(int(x) for x in targets_by_filename[fname]["labels"])
    for fname in train_img_files
}

base_candidates_by_cid = {}
for cid in cat_ids_sorted:
    # base 이미지에 이미 cid가 있으면 제외 (중복 방지)
    base_candidates_by_cid[cid] = [f for f in train_img_files if cid not in labels_set_by_fname[f]]

# ============================================================
# 실행부 C) 부족 클래스 계산 + clean donor pool
# ============================================================
print("\n== [C] Build donor pool (clean only) + compute deficits ==")

donor_pool = build_clean_donor_pool(targets_by_filename, max_iou_th=0.05)

# 전체 인스턴스 카운트
all_instances = []
for fname, t in targets_by_filename.items():
    for box, cid in zip(t["boxes"], t["labels"]):
        all_instances.append((fname, box, int(cid)))

cnt = Counter([cid for _, _, cid in all_instances])
print("클래스 수:", len(cat_ids_sorted), "전체 인스턴스:", sum(cnt.values()))

need = {}
for cid in cat_ids_sorted:
    c = cnt.get(cid, 0)
    deficit = max(0, TARGET_PER_CLASS - c)
    if deficit > 0:
        need[cid] = min(deficit, MAX_AUG_PER_CLASS)

need_items = sorted(need.items(), key=lambda x: x[1], reverse=True)
print("증강 대상 클래스 수:", len(need_items))

# ============================================================
# 실행부 D) Copy-Paste 증강 생성
# ============================================================
print("\n== [D] Generate copy-paste augmented images ==")

aug_created = 0

for cid, k in need_items:
    if aug_created >= MAX_TOTAL_AUG:
        break

    # donor 없으면 스킵
    if cid not in donor_pool or len(donor_pool[cid]) == 0:
        continue

    # 중복 방지: 이 cid가 없는 base 후보가 없으면 스킵
    base_cands = base_candidates_by_cid.get(cid, [])
    if not base_cands:
        continue

    for _ in range(k):
        if aug_created >= MAX_TOTAL_AUG:
            break

        # ----- donor 선택 -----
        donor_fname, donor_box = random.choice(donor_pool[cid])
        donor_path = TRAIN_IMG_DIR / donor_fname
        donor_img = cv2.imread(str(donor_path))
        if donor_img is None:
            continue

        dx1, dy1, dx2, dy2 = map(int, donor_box)
        dx1 = clamp(dx1, 0, donor_img.shape[1]-1)
        dy1 = clamp(dy1, 0, donor_img.shape[0]-1)
        dx2 = clamp(dx2, 1, donor_img.shape[1])
        dy2 = clamp(dy2, 1, donor_img.shape[0])

        obj = donor_img[dy1:dy2, dx1:dx2]
        if obj.size == 0:
            continue
        oh, ow = obj.shape[:2]
        if oh < 12 or ow < 12:
            continue

        # GrabCut 마스크
        obj_mask = grabcut_pill_mask(obj, iters=5)
        if obj_mask is None:
            continue

        # 가시성 추가 강화(선택)
        if USE_CLAHE_ON_OBJ:
            obj = enhance_obj_contrast(obj, obj_mask)

        # ----- base 선택: cid가 없는 이미지에서만 -----
        placed = False
        for __ in range(BASE_TRIES):
            base_fname = random.choice(base_cands)
            base_path = TRAIN_IMG_DIR / base_fname
            base_img = cv2.imread(str(base_path))
            if base_img is None:
                continue
            bh, bw = base_img.shape[:2]

            # base occupancy integral
            base_boxes = targets_by_filename[base_fname]["boxes"]
            integ = build_integral_occupancy(bh, bw, base_boxes, margin=OCC_MARGIN)

            # 빈 공간 + 가시성 좋은 위치 찾기
            pos = find_free_position_with_visibility(
                integ,
                base_img,
                obj,
                obj_mask,
                tries=POS_TRIES,
                min_delta=MIN_CONTRAST_DELTA
            )
            if pos is None:
                continue

            px1, py1, px2, py2 = pos

            # 가시성 개선 합성(코어 유지 + 경계 feather)
            base_img_aug = paste_core_plus_feather(
                base_img,
                obj,
                obj_mask,
                px1,
                py1,
                feather_sigma=FEATHER_SIGMA,
                core_erode=CORE_ERODE
            )

            # 라벨 업데이트
            new_boxes = [b[:] for b in targets_by_filename[base_fname]["boxes"]] + [[px1, py1, px2, py2]]
            new_labels = targets_by_filename[base_fname]["labels"][:] + [cid]

            aug_name = f"cp_vis_nodup_{cid}_{aug_created:06d}_{Path(base_fname).stem}.png"
            cv2.imwrite(str(OUT_IMG_DIR / aug_name), base_img_aug)
            write_yolo_label(OUT_LBL_DIR / f"{Path(aug_name).stem}.txt", new_boxes, new_labels, bw, bh)

            aug_created += 1
            placed = True
            break

        if not placed:
            continue

print(f"\n증강 생성 완료: {aug_created}장")
print(f"이미지 폴더: {OUT_IMG_DIR}")
print(f"라벨 폴더 : {OUT_LBL_DIR}")
