import os
from pathlib import Path

from dotenv import load_dotenv

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
print("data root :", DATA_ROOT)

TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test_images"
ANNOTATION_DIR = DATA_ROOT / "train_annotations"
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
