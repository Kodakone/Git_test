import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 환경변수 가져오기 
base_path = os.getenv("LOG_FILE_PATH") # env 경로 중 택 1

# 경로 연결 - Train / Test / Annotation
train_path = os.path.join(base_path, "train_images")  
test_path = os.path.join(base_path, "test_images")
anno_path = os.path.join(base_path, 'train_annotations')

# 결과 경로 출력
print(f"Dataset 경로 확인: {base_path}")

