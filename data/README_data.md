0206 수정

기존에는 .gitignore에서 최상위 폴더인 data/ 를 지정해 전부 무시하도록 했습니다.

그러나, data 폴더를 따로 만들어서 github에 드러나도록 작업하는 것이 더 깔끔하고,
각자 local 폴더 이름이나 지정이 다르게 되어 있어 획일화가 힘들어
Repo 재 정리 및 통일을 위해 다음과 같이 수정을 요청합니다.

BASIC PROJECT
├── config/                       # 설정 파일: requirements, precommit 등 전부
├── data/                         # 데이터 모음 (git: x)
│   ├── raw/                        # 원본 data 기입
│   │    ├── train_images/            # train 이미지 (.png)
│   │    ├── train_annotations/       # train annotation (.json)
│   │    └── test_images/             # test 이미지 (.png)
│   ├── yolo_dataset/               # YOLO data 원본 (.ymal)
│   └── yolo_dataset_aug/           # 증강 data 기입 (원본 + 전체 + 희소 + Copy_Paste)
├── dataloader/                   # 데이터 불러오기 관련 folder
│   ├── dataset_load.py             # .env 파일 경로 지정
│   ├── mapping.py                  # image & annotation 매핑 (gt)
│   └── split_yolo.py               # train / valid split & .ymal 변형 (gt)
├── model/                        # Model

현재 구조와 같이, .gitignore에 raw/ 내부 파일은 인식 못하도록 수정 작업 진행하였으므로
raw 내부에 원본 Data들 넣어 주신 뒤, 작업하고 계신 Data의 경로(.env)를 수정해주세요.
