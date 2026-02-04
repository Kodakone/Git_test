# main.py
import json
from collections import Counter, defaultdict
from pathlib import Path
import random

from Dataset.Dataset_load import CACHE_DIR

GT_JSON = CACHE_DIR / "targets_by_filename.json"
CAT_JSON = CACHE_DIR / "categoryid_to_name.json"

OUT_MASTER = CACHE_DIR / "master_classes.json"
OUT_SPLITS = CACHE_DIR / "splits_rareaware.json"



def load_cache():
    if not GT_JSON.exists() or not CAT_JSON.exists():
        raise FileNotFoundError(
            "cache가 없습니다. 먼저 `python mapping.py`를 실행해 cache를 생성하세요.\n"
            f"- {GT_JSON}\n- {CAT_JSON}"
        )

    targets = json.loads(GT_JSON.read_text(encoding="utf-8"))
    cat = json.loads(CAT_JSON.read_text(encoding="utf-8"))
    cat = {int(k): v for k, v in cat.items()}  # keys might be strings
    return targets, cat


def build_master_classes(cat_id_to_name: dict[int, str]):
    """
    YOLO는 class index가 0..N-1 이어야 하므로
    cat_id 정렬 기준으로 index를 부여해 '마스터 클래스 리스트'를 만든다.
    """
    cat_ids = sorted(cat_id_to_name.keys())
    master = []
    for idx, cid in enumerate(cat_ids):
        master.append({"yolo_idx": idx, "category_id": cid, "name": cat_id_to_name[cid]})
    return master


def extract_multilabels(targets_by_filename: dict):
    """
    이미지별 멀티라벨(등장한 category_id의 집합)을 만든다.
    split 규칙에 활용.
    """
    labels_by_file = {}
    for fname, d in targets_by_filename.items():
        labels = d.get("labels", [])
        labels_by_file[fname] = sorted(set(int(x) for x in labels))
    return labels_by_file


def split_strategy_masterlist(labels_by_file: dict[str, list[int]], seed=42, val_ratio=0.33):
    rng = random.Random(seed)
    files = list(labels_by_file.keys())
    rng.shuffle(files)

    n_total = len(files)
    target_val = int(n_total * val_ratio)

    # class -> files
    class_to_files = defaultdict(list)
    for f, labs in labels_by_file.items():
        for c in set(labs):
            class_to_files[c].append(f)

    class_count = {c: len(v) for c, v in class_to_files.items()}
    eligible_classes = [c for c, cnt in class_count.items() if cnt >= 2]  # 둘 다 보장 가능한 클래스만

    train_set, val_set = set(), set()

    # 1) 일단 val을 target_val까지 채우는 기본 배치(랜덤)
    for f in files:
        if len(val_set) < target_val:
            val_set.add(f)
        else:
            train_set.add(f)

    def has_class(split_set, c):
        return any(f in split_set for f in class_to_files[c])

    # 2) 보장: eligible 클래스가 train/val 둘 다에 존재하도록 "필요 시 이동"
    # val에 없으면 train에서 하나 빼서 val로
    for c in eligible_classes:
        if not has_class(val_set, c):
            # c를 가진 파일 중 train에 있는 걸 하나 골라 val로 이동
            candidates = [f for f in class_to_files[c] if f in train_set]
            rng.shuffle(candidates)
            if candidates:
                f_move = candidates[0]
                train_set.remove(f_move)
                val_set.add(f_move)

        if not has_class(train_set, c):
            # c를 가진 파일 중 val에 있는 걸 하나 골라 train으로 이동
            candidates = [f for f in class_to_files[c] if f in val_set]
            rng.shuffle(candidates)
            if candidates:
                f_move = candidates[0]
                val_set.remove(f_move)
                train_set.add(f_move)

    # 3) val 크기 다시 target로 보정 (이동 때문에 깨졌을 수 있음)
    # val이 너무 크면 train으로 옮기고, 너무 작으면 train에서 가져옴
    val_list = list(val_set)
    train_list = list(train_set)
    rng.shuffle(val_list)
    rng.shuffle(train_list)

    while len(val_set) > target_val:
        f = val_list.pop()
        val_set.remove(f)
        train_set.add(f)

    while len(val_set) < target_val and train_set:
        f = train_list.pop()
        train_set.remove(f)
        val_set.add(f)

    return sorted(train_set), sorted(val_set)




def summarize_split(labels_by_file, train_files, val_files):
    def count_classes(files):
        c = Counter()
        for f in files:
            c.update(labels_by_file[f])
        return c

    train_c = count_classes(train_files)
    val_c = count_classes(val_files)

    all_classes = sorted(set(train_c.keys()) | set(val_c.keys()))
    missing_in_train = [cid for cid in all_classes if train_c.get(cid, 0) == 0]
    missing_in_val = [cid for cid in all_classes if val_c.get(cid, 0) == 0]

    summary = {
        "n_total": len(labels_by_file),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "missing_in_train": missing_in_train,
        "missing_in_val": missing_in_val,
        "train_class_counts_top10": train_c.most_common(10),
        "val_class_counts_top10": val_c.most_common(10),
    }
    return summary


def main():
    targets, cat = load_cache()

    master = build_master_classes(cat)
    OUT_MASTER.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUT_MASTER)

    labels_by_file = extract_multilabels(targets)

    # ✅ 여기만 네 split 규칙으로 교체하면 됨
    train_files, val_files = split_strategy_masterlist(labels_by_file, seed=42, val_ratio=0.33)

    summary = summarize_split(labels_by_file, train_files, val_files)

    OUT_SPLITS.write_text(
        json.dumps({"train": train_files, "val": val_files, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved:", OUT_SPLITS)
    print("summary:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
