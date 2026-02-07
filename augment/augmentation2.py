import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # Code_it-Basic-Project
sys.path.insert(0, str(REPO_ROOT))
from dataloader.dataset_load import DATA_ROOT, CACHE_DIR # 윗줄들과 순서바뀌면 에러

import json
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import albumentations as A
import cv2
import numpy as np
import inspect

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
# 희귀클래스 들어간 이미지 목록뽑고 그 이미지들만 우선적으로 많이 증강
# 중위클래스는 조건부로 증강 - 희귀클래스와 함께 등장하지 않는 이미지들은 한번 증강
# 157번 등장하는 클래스는 증강에서 제외
class_count = defaultdict(int)
for fname, target in targets_by_filename.items():
    for label in target["labels"]:
        class_count[int(label)] += 1

target_freqs = {3,6,8,9} 
top_min = 150
mid_min, mid_max = 5, 45
always_with_3351 = {16688, 31863, 18357, 18147, 22074, 20014, 33880, 19232, 3832}
head_catids = {3351} 

rare_classes = {cid for cid, cnt in class_count.items() if cnt in target_freqs}
rare_safe_set = set(rare_classes)                         # 희귀클래스 갯수확인

top_class_set = {cid for cid, cnt in class_count.items() if cnt >= top_min}  #상위클래스
mid_class_set = {cid for cid, cnt in class_count.items() if mid_min <= cnt <= mid_max}  # 중위클래스

# 확인용
print("rare_classes:", len(rare_classes))
print("rare_classes_safe:", len(rare_safe_set))
print("top_class:", top_class_set)
print("mid_classes:", len(mid_class_set))

safe_img_set = set()     # 희귀(3/6/8/9 - exclude_rare3 제외안함)
mid_img_set = set()      # 중위(12~45) 클래스가 들어간 이미지
top_img_set = set()      # 상위(153) 클래스가 들어간 이미지

safe_img_to_hits = {}    # 디버깅: 이미지별 포함된 safe 희귀 클래스들

# train만 불러오기 ----------------------------------
for fname in train_fnames:
    labs = set(int(x) for x in targets_by_filename[fname]["labels"]) # 0,1,2,3....

    # top(157) 포함 이미지 (증강 제외)
    if labs & top_class_set:
        top_img_set.add(fname)

    # 중위(12~45) 포함 이미지(단독증강x)
    if labs & mid_class_set:
        mid_img_set.add(fname)

    # 희귀 포함 이미지 (증강 대상)
    hits = sorted(labs & rare_safe_set)
    if hits:
        safe_img_set.add(fname)
        safe_img_to_hits[fname] = hits

# 희귀클래스 증강대상 이미지
aug_img_set = safe_img_set

# 중위클래스 조건부증강대상 이미지(희귀클래스와 함께 등장하지 않은 이미지들만 대상)
cond_mid_img_set = mid_img_set - safe_img_set - top_img_set

print("safe_img_set:", len(safe_img_set))      
print("mid_img_set:", len(mid_img_set))       
print("top_img_set:", len(top_img_set))       
print("aug_img_set:", len(aug_img_set))      
print("cond_mid_img_set:", len(cond_mid_img_set)) 

# 증강횟수 설정/계산함수  ----------------------------------
FREQ_TO_AUG = {3: 6, 6: 3, 8: 2, 9: 2}  # ex. 3번등장->6번증강
COND_MID_AUG_NUM = 1  # 중위 클래스 조건부 증강 횟수

def get_aug_count(filename, labels, class_count):
    # 희귀클래스 증강대상
    if filename in aug_img_set:
        freqs = [class_count.get(cid, 10**9) for cid in set(labels)]
        min_freq = min(freqs)
        # 만약 3번등장/9번등장 같은이미지에 나오면 [6,2]중 무슨 값으로 증강할래
        return FREQ_TO_AUG.get(min_freq, 0)

    # 중위클래스 조건부 증강대상
    elif filename in cond_mid_img_set:
        return COND_MID_AUG_NUM

    # 그 외(증강 제외)
    else:
        return 0

# 브레이크설정; 안전장치 ----------------------------------
# 목표 bbox도달시 조기중단 & 이미지당 최대 증강 상한설정(cap)
MAX_AUG_PER_IMAGE_RARE_Q1 = 8  # <=3회 클래스용
MAX_AUG_PER_IMAGE_RARE_Q2 = 3  # 6~9회 클래스용
MAX_AUG_PER_IMAGE_MID  = 2     # cond_mid용 
TARGET_RARE_BBOX = 10   # 희귀 클래스 현재 bbox 개수 < n이면 증강허용
TARGET_MID_BBOX  = 18   # 중위 클래스 현재 bbox 개수 >= n이면 증강필요없음

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
OUTPUT_ROOT = V1_ROOT

print("IMG_DIR:", IMG_DIR0)
print("exists:", IMG_DIR0.exists())
print("num src imgs:", len(list(IMG_DIR0.glob('*.png'))) + len(list(IMG_DIR0.glob('*.jpg'))))
print("catid_to_cls sample:", list(catid_to_cls.items())[:5], "...")
print("cls_to_catid sample:", list(cls_to_catid.items())[:5], "...")

# 증강시작 ===================================================
# train: 증강
AUG_PREFIXES = ("rare_", "mid_")
# for prefix in AUG_PREFIXES:
#     for p in IMG_DIR0.glob(f"{prefix}*.png"):
#         p.unlink()
#     for p in LBL_DIR0.glob(f"{prefix}*.txt"):
#         p.unlink()

attempted = 0   # 증강 시도 수
saved = 0       # 실제 저장 수
skipped_empty = 0
skipped_no_safe = 0
MAX_ADD_3351_IMGS = 5     # 3351 포함 증강본은 최대 5장까지만 허용
added_3351_imgs = 0       # rare가 3351클래스 등장이미지 얼마나 올리나 추적용
seen_rare_with_head = 0   # 3351클래스와 등장하는 이미지추적용
skipped_rare_due_to_head = 0  # 3351증가돼서 증강실패한 이미지갯수 추적용
skipped_mid_due_to_head = 0

aug_bbox_per_class = Counter()  # 클래스별 실제생성 증강박스
aug_per_image = Counter()       # 이미지별 실제 증강횟수

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

    if fname in aug_img_set:
        # 이 이미지에 포함된 클래스들 중 "가장 희귀한 클래스의 빈도"로 tier 결정
        min_freq = min(class_count[c] for c in set(clean_catids))
        if min_freq <= 3:
            cap = MAX_AUG_PER_IMAGE_RARE_Q1
        elif min_freq <= 9:
            cap = MAX_AUG_PER_IMAGE_RARE_Q2
        else:
            cap = 0
        aug_n = min(raw_aug_n, cap)
        
    elif fname in cond_mid_img_set:
        aug_n = min(raw_aug_n, MAX_AUG_PER_IMAGE_MID)
        
    else:
        aug_n = 0
    
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
        # 날아가는 예)회전했는데 객체가 이미지밖으로 나가 박스가 작아질 때 (w<1,h<1조건걸림)

        final_catids = [cls_to_catid[int(c)] for c in final_cls_ids]
        final_catids_set = set(final_catids)

        if is_safe_src and (len(final_catids_set & rare_safe_set) == 0):  #safe없으면 적용x
            skipped_no_safe += 1
            continue

        # 도움된다 = 증강될 때 아직 목표에 못미친 클래스의 bbox를 최소 하나이상 증가시키는가
        # 목표 = TARGET_RARE_BBOX, TARGET_MID_BBOX
        # cond_mid는 부족한 mid 클래스에 실제로 도움되는 경우만 저장
        if fname in cond_mid_img_set:
            if not any((cid in underfilled_mid_set) for cid in final_catids_set):
                continue

            # mid 증강 결과에 head(3351)나 3351-강결합 클래스가 포함되면 저장하지 않음
            if (final_catids_set & head_catids) or (final_catids_set & always_with_3351):
                skipped_mid_due_to_head += 1
                continue

        elif fname in aug_img_set:
            # rare 목표(TARGET_RARE_BBOX) 미달 클래스에 도움될 될 때만 저장
            if not any((cid in rare_safe_set and cur_bbox_count[cid] < target_bbox_count[cid]) for cid in final_catids_set):
                continue

            # 3351 cap 초과해 실제 저장 실패수 파악
            if 3351 in final_catids_set:
                if added_3351_imgs >= MAX_ADD_3351_IMGS:
                    skipped_rare_due_to_head += 1  # cap 때문에 저장 실패한 횟수
                    continue

            # head(3351)가 포함된 rare수 파악
            if (final_catids_set & head_catids) or (final_catids_set & always_with_3351):
                seen_rare_with_head += 1
        else:
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

        out_imgA = OUTPUT_ROOT / "images" / "train" / f"{prefix}{stem}{suffix}{ext}"
        out_lblA = OUTPUT_ROOT / "labels" / "train" / f"{prefix}{stem}{suffix}.txt"

        if out_imgA.exists() or out_lblA.exists():
            continue

        write_image_cv2(out_imgA, imgA)
        save_yolo_label(out_lblA, final_bboxes, final_cls_ids, wA, hA)

        # debug count-----
        aug_per_image[fname] += 1

        for cid in final_catids:       # bbox 기준 등장
            aug_bbox_per_class[cid] += 1
        # -----

        saved += 1
        if 3351 in final_catids_set:
            added_3351_imgs += 1

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
print("added_3351_imgs:", added_3351_imgs)
print("seen_rare_with_head:", seen_rare_with_head)
print("skipped rare due to head:", skipped_rare_due_to_head)
print("skipped mid due to head/cooccur:", skipped_mid_due_to_head)
underfilled_mid_set_after = get_underfilled_mid_set(cur_bbox_count, target_bbox_count, mid_class_set)
print("underfilled_mid_set AFTER:", len(underfilled_mid_set_after))
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

LABEL_DIR1 = V1_ROOT / "labels" / "train"

class_freq = defaultdict(int)

for txt_path in LABEL_DIR1.glob("*.txt"):
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cls_id = int(line.split()[0])  # YOLO format: cls cx cy w h
            class_freq[cls_id] += 1

# # debug count -----
# print("\n[Class-wise augmentation summary]")
# print("cid | name | orig_bbox_cnt | aug_bbox_cnt") 
# # 원본bbox갯수,증강이미지중 클래스등장이미지수, 증강으로 추가된 bbox수
# check_list = sorted(set(rare_safe_set) | set(mid_class_set))
# for cid in check_list:
#     cname = categoryid_to_name.get(cid, "UNK")
#     orig_cnt = class_count[cid]
#     aug_box_cnt = aug_bbox_per_class[cid] 
    
#     # 증강된 것만 출력하거나, 원래 개수가 적은 것 위주로 출력
#     if orig_cnt < 50 or aug_box_cnt > 0:
#          print(
#             f"{cid:<6d} | {cname[:20]:<20s} | "
#             f"{orig_cnt:4d} | "
#             f"{aug_box_cnt:6d}"
#         )
# # -----

print("\n=== Class frequency (bbox count) ===")
for cls_id, cnt in sorted(class_freq.items(), key=lambda x: x[1], reverse=True):
    catid = cls_to_catid.get(cls_id, None)
    if catid in rare_safe_set:
        continue
    cname = categoryid_to_name.get(catid, "UNKNOWN")
    print(f"class {cls_id:>2} | catid {catid:>6} | {cname:<20} : {cnt}")

print("\n=== Rare class check ===")  # 희귀클래스 분포 
for cid in sorted(rare_safe_set):
    cls = catid_to_cls[cid]
    print(
        f"catid {cid:>6} | class {cls:>2} | "
        f"{categoryid_to_name.get(cid, 'UNKNOWN'):<20} : "
        f"{class_freq.get(cls, 0)} bboxes"
    )

# png/txt 1:1 일치체크
IMG_DIR = Path("yolo_dataset_aug/images/train")
LBL_DIR = Path("yolo_dataset_aug/labels/train")

img_stems = {p.stem for p in IMG_DIR.glob("*.png")}
lbl_stems = {p.stem for p in LBL_DIR.glob("*.txt")}
only_img = sorted(img_stems - lbl_stems)
only_lbl = sorted(lbl_stems - img_stems)

print(f"\nonly image: {len(only_img)}")
for s in only_img[:20]:
    print("  ", s)
print(f"only label: {len(only_lbl)}")
for s in only_lbl[:20]:
    print("  ", s)

yaml_path = V1_ROOT / "dataset.yaml"
assert yaml_path.exists(), "dataset.yaml 이 없습니다. split_yolo.py를 먼저 실행하세요."
print("dataset.yaml OK:", yaml_path)
