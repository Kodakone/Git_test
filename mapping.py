import os
from pathlib import Path
from dotenv import load_dotenv

# 경로지정 ---------------------------
from Dataset.Dataset_load import TRAIN_IMG_DIR, TEST_IMG_DIR, ANNOTATION_DIR, CACHE_DIR, DATA_ROOT

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
REPO_ROOT = Path(__file__).resolve().parent
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