from pathlib import Path
import json
import shutil
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import random

import numpy as np
from dataloader.dataset_load import TRAIN_IMG_DIR, DATA_ROOT, CACHE_DIR
import albumentations as A 
import cv2
import numpy as np

print("Albumentations:", A.__version__)
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)

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

# catid_to_cls 로드 ---------------------------------- 
SPLIT_JSON = Path(DATA_ROOT).parent / "yolo_dataset" / "split.json"
with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split = json.load(f)

train_fnames = split["train"]
catid_to_cls = {int(k): int(v) for k, v in split["mapping"]["catid_to_cls"].items()}
cls_to_catid = {int(k): int(v) for k, v in split["mapping"]["cls_to_catid"].items()}

# 함수정의  ---------------------------------- 
def xyxy_to_yolo_line(x1, y1, x2, y2, cls, img_w, img_h):  # 욜로형식(class cx cy w h)
    # clamp (이미지밖으로 나간 좌표 보정 - 데이터가 좀 깨져있어도 학습에서 터지지 않게)
    x1 = max(0.0, min(x1, img_w-1))
    x2 = max(0.0, min(x2, img_w-1))
    y1 = max(0.0, min(y1, img_h-1))
    y2 = max(0.0, min(y2, img_h-1))

    bw = max(0.0, x2 - x1)  # 박스너비
    bh = max(0.0, y2 - y1)  # 박스높이
    cx = x1 + bw/2          # 욜로는 좌상단(x1,y1)아닌 중심좌표(cx,cy)를 씀
    cy = y1 + bh/2

    # 욜로포맷 문자열 생성(0~1 정규화)
    return f"{cls} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"

def save_yolo_label(label_path: Path, boxes_xyxy, cls_ids, img_w, img_h):
    # 이미지 한장에 대한 욜로라벨파일(.txt) 생성
    label_path.parent.mkdir(parents=True, exist_ok=True)  # label_path없으면 에러
    lines = []
    for (x1,y1,x2,y2), c_id in zip(boxes_xyxy, cls_ids): # bbox와 category id 하나씩 처리
        # c_id : 0,1,2...
        lines.append(xyxy_to_yolo_line(x1,y1,x2,y2,int(c_id),img_w,img_h))
    label_path.write_text("\n".join(lines), encoding="utf-8")

def read_image_cv2(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return img_bgr

def write_image_cv2(path: Path, img_bgr): 
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")

def sanitize_bboxes_labels(bboxes, labels, w, h):
    """pascal_voc bbox(x1,y1,x2,y2)를 이미지 경계로 정리.
    - 값이 0~1 범위로 '정규화'된 것처럼 보이면 픽셀로 스케일링
    - 이미지 밖/이상치/너무 작은 박스 제거
    """
    clean_boxes = []
    clean_labels = []
    for b, lab in zip(bboxes, labels):
        if b is None or len(b) != 4:  #bbox없거나 길이4아니면 스킵
            continue
        try:
            x1, y1, x2, y2 = map(float, b)
        except Exception:
            continue

        # #(이 로직 제거)
        # # (1) 전부 0~1이면 정규화 좌표로 보고 픽셀로 변환
        # if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
        #     x1, x2 = x1 * w, x2 * w
        #     y1, y2 = y1 * h, y2 * h

        # (2) 너무 큰 이상치(예: 60000 같은 값) 제거/클리핑 전에 컷
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 10000:
            # 이런 값은 거의 확실히 잘못된 bbox라서 버림
            continue

        # (3) 좌표 순서 보정
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # (4) 이미지 경계로 클립
        x1 = max(0.0, min(x1, w - 1.0))
        x2 = max(0.0, min(x2, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        y2 = max(0.0, min(y2, h - 1.0))

        # (5) 너무 작은 박스 제거
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        clean_boxes.append([x1, y1, x2, y2])
        clean_labels.append(int(lab))

    return clean_boxes, clean_labels

# 증강기법 ---------------------------------- 
# - 색상(Hue/Sat/Value) 변화는 제외 (약 색상 중요)
# - bbox가 이미지 밖으로 튀는 경우를 대비해서 'clip' 옵션이 있으면 켭니다.

import inspect

# 증강 설정값 (원하면 여기만 조절)
#AUG_PER_IMAGE = 10              # train 이미지 1장당 추가 생성 개수(원본 제외).

# BboxParams 옵션 호환 처리 (알부멘테이션 버전에 따라 인자명이 다를 수 있음)
bp_kwargs = dict(
    format="pascal_voc",              # [x_min, y_min, x_max, y_max] (픽셀)
    label_fields=["class_labels"],
    min_area=1,
    min_visibility=0.0,
)
sig = inspect.signature(A.BboxParams).parameters
if "clip" in sig:
    bp_kwargs["clip"] = True  # bbox를 이미지 경계로 클리핑
if "filter_lost_elements" in sig:
    bp_kwargs["filter_lost_elements"] = True
if "check_each_transform" in sig:
    bp_kwargs["check_each_transform"] = False  # 매 변환마다 엄격 체크를 줄임(대신 아래에서 직접 정리)

bbox_params = A.BboxParams(**bp_kwargs)

# 증강기법: 밝기/대비 + 작은 회전 + scale + 약한 blur
aug_tf = A.Compose(
    [
        A.Affine(
            scale=(0.90, 1.10),       # ±10%
            rotate=(-10, 10),         # ±10도
            translate_percent=None,   # 이동은 안 함 (원하면 추가 가능)
            p=0.7                     # affine 적용 확률(원하면 1.0으로)
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,    # ±10%
            contrast_limit=0.10,      # ±10%
            p=0.8
        ),
        A.GaussianBlur(
            blur_limit=(3, 3),        # 매우 약하게 (3x3 고정)
            p=0.1                    # blur 10% 확률
        ),
        # 또는 blur를 조금 더 자연스럽게(약하게) 하고 싶으면 아래로 교체 가능:
        # A.Blur(blur_limit=3, p=0.01),
    ],
    bbox_params=bbox_params,
)

# 증강규칙 ---------------------------------- 
# 희귀클래스 들어간 이미지 목록뽑고 그 이미지들만 우선적으로 많이 증강(단, 45번이랑 같이 등장하는 클래스 3개는 제외) - 연쇄적으로 그 클래스도 늘어날테니
# 중위클래스는 조건부로 증강 - 희귀클래스와 함께 등장하지 않는 이미지들은 한번 증강
# 153번 등장하는 클래스는 증강에서 제외
class_count = defaultdict(int)
for fname, target in targets_by_filename.items():
    for label in target["labels"]:
        class_count[int(label)] += 1

target_freqs = {3, 5, 6, 9} # 8번등장 제외
exclude_rare3 = {34597, 31885, 27777}  #항상 3483(45)와 함께등장하는 3회등장클래스
top_cnt = max(class_count.values())
mid_min, mid_max = 12, 45

rare_classes = {cid for cid, cnt in class_count.items() if cnt in target_freqs}
rare_safe_set = set(rare_classes) - set(exclude_rare3)  # 희귀클래스 갯수확인
rare_notsafe_set = rare_classes & exclude_rare3         # 3483(45)와 함께등장하는 3회등장클래스 갯수확인

top_class_set = {cid for cid, cnt in class_count.items() if cnt == top_cnt}  #상위클래스
mid_class_set = {cid for cid, cnt in class_count.items() if mid_min <= cnt <= mid_max}  # 중위클래스

# 확인용
print("rare_classes (3/5/6/9):", len(rare_classes))
print("rare_classes_safe:", len(rare_safe_set))
print("rare_classes_notsafe:", rare_notsafe_set)
print("top_class:", top_class_set)
print("mid_classes:", len(mid_class_set))

safe_img_set = set()     # 희귀(3/5/6/9 - exclude_rare3 제외)
notsafe_img_set = set()  # exclude_rare3 중 하나라도 들어간 이미지
mid_img_set = set()      # 중위(12~45) 클래스가 들어간 이미지
top_img_set = set()      # 상위(153) 클래스가 들어간 이미지

safe_img_to_hits = {}    # 디버깅: 이미지별 포함된 safe 희귀 클래스들

# train만 불러오기 ---------------------------------- 
for fname in train_fnames:
    labs = set(int(x) for x in targets_by_filename[fname]["labels"]) # 0,1,2,3....

    # top(153) 포함 이미지 (증강 제외)
    if labs & top_class_set:
        top_img_set.add(fname)

    # 중위(12~45) 포함 이미지(단독증강x)
    if labs & mid_class_set:
        mid_img_set.add(fname)

    # notsafe 희귀(exclude_rare3) 포함 이미지(증강 제외)
    if labs & exclude_rare3:
        notsafe_img_set.add(fname)

    # 희귀 포함 이미지 (증강 대상)
    hits = sorted(labs & rare_safe_set)
    if hits:
        safe_img_set.add(fname)
        safe_img_to_hits[fname] = hits

# 희귀클래스 증강대상 이미지
aug_img_set = safe_img_set - notsafe_img_set - top_img_set

# 중위클래스 조건부증강대상 이미지
cond_mid_img_set = mid_img_set - safe_img_set - notsafe_img_set - top_img_set
# 중위 클래스 조건부증강할 때 어떤 중위클래스 객체와 exclude_rare3 객체가 한 이미지에 있을 수 있으니

print("safe_img_set:", len(safe_img_set))      # 38
print("notsafe_img_set:", len(notsafe_img_set))   # 6
print("mid_img_set:", len(mid_img_set))       # 149
print("top_img_set:", len(top_img_set))       # 102
print("aug_img_set:", len(aug_img_set))       # 36
print("cond_mid_img_set:", len(cond_mid_img_set))  # 11

# 증강횟수 설정/계산함수  ---------------------------------- 
FREQ_TO_AUG = {3: 10, 5: 7, 6: 6, 9: 5}  # ex. 3번등장->10번증강
COND_MID_AUG_NUM = 1  # 중위 클래스 조건부 증강 횟수

def get_aug_count(filename, labels, class_count):
    # 희귀클래스 증강대상
    if filename in aug_img_set:
        current_safe_hits = set(labels) & rare_safe_set

        # 만약 3번등장/9번등장 같은이미지에 나오면 [8,4]중 무슨 값으로 증강할래
        candidates = [FREQ_TO_AUG.get(class_count.get(cid), 0) for cid in current_safe_hits]
        return max(candidates) if candidates else 0

    # 중위클래스 조건부 증강대상
    elif filename in cond_mid_img_set:
        return COND_MID_AUG_NUM

    # 그 외(증강 제외)
    else:
        return 0

# 브레이크설정; 안전장치 ---------------------------------- 
# 목표 bbox도달시 조기중단 & 이미지당 최대 증강 상한설정
# 한 이미지에서 4장, 나머지는 다른 이미에서 찾아서..?
MAX_AUG_PER_IMAGE = 4   # 이미지당 증강 상한
TARGET_RARE_BBOX = 40   # 희귀 클래스 목표 bbox
TARGET_MID_BBOX  = 80  # 중위 클래스 목표 bbox

def all_targets_met(cur, target):
    return all(cur[c] >= t for c, t in target.items())

target_bbox_count = {}  # 목표딕셔너리

for cid in rare_safe_set:
    target_bbox_count[cid] = TARGET_RARE_BBOX

for cid in mid_class_set:
    target_bbox_count[cid] = TARGET_MID_BBOX

cur_bbox_count = defaultdict(int) # 현재까지 생성된 bbox카운트 추적(원본+누적)
for fname in train_fnames:
    for cid in targets_by_filename[fname]["labels"]:
        cid = int(cid)
        cur_bbox_count[cid] += 1

# 아직도 부족한 클래스
def get_underfilled_mid_set(cur_bbox_count, target_bbox_count, mid_class_set):
    return {
        cid for cid in mid_class_set
        if cid in target_bbox_count and cur_bbox_count[cid] < target_bbox_count[cid]
    }
underfilled_mid_set = get_underfilled_mid_set(cur_bbox_count, target_bbox_count, mid_class_set)
print("underfilled_mid_set:", len(underfilled_mid_set))

# 증강본이 아직 필요한지 판단(목표에 못미친 클래스에 bbox를 하나라도 추가해주는가)
def helps_any_underfilled_class(catids, cur_bbox_count, target_bbox_count):
    for cid in catids:
        if cid in target_bbox_count and cur_bbox_count[cid] < target_bbox_count[cid]:
            return True
    return False

# 증강시작 전 확인
V0_ROOT = Path(DATA_ROOT).parent / "yolo_dataset"
IMG_DIR0 = V0_ROOT / "images" / "train"
LBL_DIR0 = V0_ROOT / "labels" / "train"

V1_ROOT = Path(DATA_ROOT).parent / "yolo_dataset_aug"
IMG_DIR1 = V1_ROOT / "images" / "train"
LBL_DIR1 = V1_ROOT / "labels" / "train"
print("IMG_DIR:", IMG_DIR0)
print("exists:", IMG_DIR0.exists())
print("num src imgs:", len(list(IMG_DIR0.glob('*.png'))) + len(list(IMG_DIR0.glob('*.jpg'))))
print("catid_to_cls sample:", list(catid_to_cls.items())[:5], "...")
print("cls_to_catid sample:", list(cls_to_catid.items())[:5], "...")

# 증강시작 ===================================================
# train: 증강
AUG_PREFIXES = ("rare_", "mid_")
original_set = set(train_fnames)  # split.json 기준

for p in IMG_DIR0.glob("*.png"):
    if p.name not in original_set:
        p.unlink()
for p in LBL_DIR0.glob("*.txt"):
    if p.stem + ".png" not in original_set:
        p.unlink()

attempted = 0   # 증강 시도 수
saved = 0       # 실제 저장 수
skipped_empty = 0
skipped_no_safe = 0

random.seed(42)

print("Start augmentation...")

for fname in tqdm(sorted(train_fnames), desc="TRAIN preprocess"):
    if fname.startswith(AUG_PREFIXES):
        continue

    img_path = IMG_DIR0 / fname

    if not img_path.exists():
        continue
    img_bgr = read_image_cv2(img_path)
    h, w = img_bgr.shape[:2]

    tgt = targets_by_filename[fname]
    bboxes = [tuple(b) for b in tgt["boxes"]]
    catids = [int(cid) for cid in tgt["labels"]]  

    # 변환 전에 bbox 정리 (여기서 이상치 제거)
    clean_bboxes, clean_catids = sanitize_bboxes_labels(bboxes, catids, w, h)
    if len(clean_bboxes) == 0:
        continue

    clean_cls_ids = [catid_to_cls[c] for c in clean_catids]

    # 일반 증강 (이미지별 증강량 결정)
    # aug_tf 적용후 유효한 bbox 1개이상존재시 저장 _aug1,_aug2..
    raw_aug_n = get_aug_count(fname, clean_catids, class_count)
    aug_n = min(raw_aug_n, MAX_AUG_PER_IMAGE)
    if aug_n <= 0:
        continue

    is_safe_src = (len(set(clean_catids) & rare_safe_set) > 0) # safe인지 확인

    for k in range(aug_n):
        try: 
            transformed = aug_tf(image=img_bgr, bboxes=clean_bboxes, class_labels=clean_cls_ids)
        except ValueError as e:
            # 혹시라도 bbox 처리 중 에러나면 스킵
            # print(f"Aug Error {fname}: {e}")
            continue    
        imgA = transformed["image"]
        bbA  = transformed["bboxes"]
        clA  = transformed["class_labels"]

        attempted += 1
        hA, wA = imgA.shape[:2]

        final_bboxes, final_cls_ids = sanitize_bboxes_labels(bbA, clA, wA, hA)

        if len(final_bboxes) == 0:
            skipped_empty += 1
            continue  # bbox 다 날아가면 skip (bbox가 안남으면 그 증강이미지 버림)
        # 날아감 ex.회전했는데 객체가 이미지밖으로 나가 박스가 작아질 때 (w<1,h<1조건걸림)

        final_catids = [cls_to_catid[int(c)] for c in final_cls_ids]
        if is_safe_src and (len(set(final_catids) & rare_safe_set) == 0):  #safe없으면 적용x
            skipped_no_safe += 1
            continue
        
        final_catids_set = set(final_catids)
        # cond_mid는 부족한 mid 클래스에 실제로 도움되는 경우만 저장
        if fname in cond_mid_img_set:
            if not any((cid in underfilled_mid_set) for cid in final_catids_set):
                continue
        else:
            # rare 증강(또는 그 외)은 기존처럼: 목표 미달 클래스에 도움될 때만 저장
            if not helps_any_underfilled_class(final_catids_set, cur_bbox_count, target_bbox_count):
                continue

        # prefix
        if fname in aug_img_set:
            prefix = "rare_"
        elif fname in cond_mid_img_set:
            prefix = "mid_"
        else:
            prefix = ""

        suffix = f"_aug{k+1}"
        stem = Path(fname).stem
        ext  = Path(fname).suffix

        out_imgA = V1_ROOT / "images" / "train" / f"{prefix}{stem}{suffix}{ext}"
        out_lblA = V1_ROOT / "labels" / "train" / f"{prefix}{stem}{suffix}.txt"

        if out_imgA.exists() or out_lblA.exists():
            continue

        write_image_cv2(out_imgA, imgA)
        save_yolo_label(out_lblA, final_bboxes, final_cls_ids, wA, hA)

        saved += 1
        for cid in final_catids:  # 증강이미지용
            if cid in target_bbox_count:
                cur_bbox_count[cid] += 1
        if saved % 50 == 0:
            underfilled_mid_set = get_underfilled_mid_set(cur_bbox_count, target_bbox_count, mid_class_set)

        if all_targets_met(cur_bbox_count, target_bbox_count):
            print("All targets met. Stopping early.")
            break

print("생성 완료")
print("AUG attempted:", attempted)
print("saved:", saved)
print("skip empty bbox:", skipped_empty)
print("skip no safe:", skipped_no_safe)
print("---")

# 현재 V1 폴더 현황
train_imgs = len(list(IMG_DIR1.glob('*.png'))) + len(list(IMG_DIR1.glob('*.jpg')))
train_lbls = len(list((V1_ROOT/'labels/train').glob('*.txt')))
print(f"V1 Train Images: {train_imgs}")  # (원본+증강)
print(f"V1 Train Labels: {train_lbls}")  # (원본+증강)

aug_imgs = len(list(IMG_DIR1.glob('rare_*'))) + len(list(IMG_DIR1.glob('mid_*')))
print("Augmented images only:", aug_imgs)


# 확인용 -----------------------------------
# 최종 v1기준 전체 클래스 등장빈도
from collections import defaultdict

LABEL_DIR = V1_ROOT / "labels" / "train"

class_freq = defaultdict(int)

for txt_path in LABEL_DIR.glob("*.txt"):
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cls_id = int(line.split()[0])  # YOLO format: cls cx cy w h
            class_freq[cls_id] += 1

print("=== Class frequency (bbox count) ===")
for cls_id, cnt in sorted(class_freq.items(), key=lambda x: x[1], reverse=True):
    catid = cls_to_catid.get(cls_id, None)
    cname = categoryid_to_name.get(catid, "UNKNOWN")
    print(f"class {cls_id:>2} | catid {catid:>6} | {cname:<20} : {cnt}")

print("=== Rare class check ===")  # 증강이 실제 희귀클래스 분포 개선했는가(>40)
for cid in sorted(rare_safe_set):
    cls = catid_to_cls[cid]
    print(
        f"catid {cid:>6} | class {cls:>2} | "
        f"{categoryid_to_name.get(cid, 'UNKNOWN'):<20} : "
        f"{class_freq.get(cls, 0)} bboxes"
    )
    
yaml_path = V1_ROOT / "dataset.yaml"
assert yaml_path.exists(), "dataset.yaml 이 없습니다. split_yolo.py를 먼저 실행하세요."
print("dataset.yaml OK:", yaml_path)
