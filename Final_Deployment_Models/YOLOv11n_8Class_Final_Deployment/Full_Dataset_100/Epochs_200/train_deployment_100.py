from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil

# ============================================================
# YOLOv11n 8-Class 100% Deployment Retraining
# 自動偵測圖片數量
# 全部資料用於 Training，不切 Validation
# ============================================================

ROOT = Path(__file__).resolve().parent

CLASS_NAMES = {
    0: "dog",
    1: "cat",
    2: "bear",
    3: "pig",
    4: "monkey",
    5: "person",
    6: "iguana",
    7: "deer"
}

# 訓練參數
EPOCHS = 200
BATCH = 8
IMGSZ = 640
WORKERS = 2
SEED = 42

# YOLOv11n 原有專屬設定
CLS = 0.75

DATA_YAML = ROOT / "data.yaml"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train_all"
TRAIN_LABELS_DIR = LABELS_DIR / "train_all"

PROJECT_DIR = ROOT / "runs"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# 執行後自動填入
DATASET_COUNTS = {}


def get_source_images():
    return sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def validate_label(content, label_path):
    for line in content.splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            raise ValueError(
                f"❌ Label class_id 格式錯誤：{label_path}"
            )

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"❌ 發現不存在的 class_id={class_id}\n"
                f"Label：{label_path}\n"
                f"目前允許 class_id：0～7"
            )


def prepare_100_percent_dataset():
    global DATASET_COUNTS

    print("\n===== 建立 8-Class 100% Training Dataset =====")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"❌ 找不到 images：{IMAGES_DIR}"
        )

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"❌ 找不到 labels：{LABELS_DIR}"
        )

    image_files = get_source_images()

    if len(image_files) == 0:
        raise ValueError(
            f"❌ images 資料夾中沒有圖片：{IMAGES_DIR}"
        )

    positive_images = []
    background_images = []
    stems = set()

    for image in image_files:
        if image.stem in stems:
            raise ValueError(
                f"❌ 發現同名圖片：{image.stem}"
            )

        stems.add(image.stem)

        label = LABELS_DIR / f"{image.stem}.txt"

        if label.exists():
            content = label.read_text(
                encoding="utf-8"
            ).strip()

            if content:
                validate_label(
                    content,
                    label
                )

                positive_images.append(image)

            else:
                # 空白 txt = 背景負樣本
                background_images.append(image)

        else:
            # 沒有 txt = 背景負樣本
            background_images.append(image)

    positive_count = len(positive_images)
    background_count = len(background_images)
    total_count = len(image_files)

    DATASET_COUNTS = {
        "positive": positive_count,
        "background": background_count,
        "total": total_count
    }

    print(f"正樣本：{positive_count}")
    print(f"背景負樣本：{background_count}")
    print(f"總圖片數：{total_count}")

    if positive_count == 0:
        raise ValueError(
            "❌ 沒有找到任何正樣本"
        )

    # ========================================================
    # 只建立 train_all，不切 Train / Validation
    # ========================================================

    for folder in [
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR
    ]:
        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # 全部圖片複製進 train_all
    # ========================================================

    for image in image_files:
        shutil.copy2(
            image,
            TRAIN_IMAGES_DIR / image.name
        )

        source_label = (
            LABELS_DIR / f"{image.stem}.txt"
        )

        destination_label = (
            TRAIN_LABELS_DIR / f"{image.stem}.txt"
        )

        if source_label.exists():
            shutil.copy2(
                source_label,
                destination_label
            )

        else:
            # 背景圖沒有 txt 時，自動建立空白 label
            destination_label.touch()

    print("\n✅ 100% Training Dataset 建立完成")

    print(
        f"Train："
        f"{positive_count} 正樣本 + "
        f"{background_count} 背景 = "
        f"{total_count}"
    )

    print("Validation：不切分")


def create_data_yaml():
    data = {
        "path": ROOT.as_posix(),

        "train": "images/train_all",

        # 保留原本 100% Retraining 架構
        # 實際訓練使用 val=False
        "val": "images/train_all",

        "names": CLASS_NAMES
    }

    with open(
        DATA_YAML,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print(
        f"✅ data.yaml 建立完成：{DATA_YAML}"
    )


def check_dataset():
    print("\n===== 8-Class 100% Dataset 檢查 =====")

    images = sorted([
        f for f in TRAIN_IMAGES_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    positive = 0
    background = 0
    image_stems = set()

    for image in images:
        if image.stem in image_stems:
            raise ValueError(
                f"❌ 發現重複圖片名稱：{image.stem}"
            )

        image_stems.add(
            image.stem
        )

        label = (
            TRAIN_LABELS_DIR
            / f"{image.stem}.txt"
        )

        if not label.exists():
            raise FileNotFoundError(
                f"❌ 找不到 Label：{label}"
            )

        content = label.read_text(
            encoding="utf-8"
        ).strip()

        if content:
            validate_label(
                content,
                label
            )

            positive += 1

        else:
            background += 1

    label_stems = {
        f.stem
        for f in TRAIN_LABELS_DIR.glob("*.txt")
    }

    extra_labels = (
        label_stems - image_stems
    )

    if extra_labels:
        raise ValueError(
            f"❌ 發現沒有對應圖片的 Label："
            f"{sorted(extra_labels)}"
        )

    total = positive + background

    print(f"正樣本：{positive}")
    print(f"背景負樣本：{background}")
    print(f"總數：{total}")

    if positive != DATASET_COUNTS["positive"]:
        raise ValueError(
            f"❌ 正樣本數量錯誤，"
            f"原始 {DATASET_COUNTS['positive']}，"
            f"train_all {positive}"
        )

    if background != DATASET_COUNTS["background"]:
        raise ValueError(
            f"❌ 背景圖數量錯誤，"
            f"原始 {DATASET_COUNTS['background']}，"
            f"train_all {background}"
        )

    if total != DATASET_COUNTS["total"]:
        raise ValueError(
            f"❌ 總圖片數錯誤，"
            f"原始 {DATASET_COUNTS['total']}，"
            f"train_all {total}"
        )

    print(
        "✅ 8-Class 100% Dataset 檢查完成"
    )

    print(
        f"Training：{total} 張"
    )

    print(
        "Validation：0 張"
    )


def train_100_percent():
    print(
        "\n===== YOLOv11n 8-Class 100% Retraining ====="
    )

    print(
        f"Training Images："
        f"{DATASET_COUNTS['total']}"
    )

    print(
        f"Positive Images："
        f"{DATASET_COUNTS['positive']}"
    )

    print(
        f"Background Images："
        f"{DATASET_COUNTS['background']}"
    )

    print("Classes：8")
    print(f"Epochs：{EPOCHS}")
    print(f"Batch：{BATCH}")
    print(f"Image Size：{IMGSZ}")
    print(f"cls：{CLS}")
    print(f"Seed：{SEED}")
    print("Validation：False")

    model = YOLO(
        "yolo11n.pt"
    )

    model.train(
        data=DATA_YAML.as_posix(),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        cache=False,
        amp=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,

        # YOLOv11n 原有專屬參數
        cls=CLS,

        seed=SEED,
        deterministic=True,

        # 100% Retraining
        val=False,
        patience=0,

        project=PROJECT_DIR.as_posix(),
        name="YOLOv11n_8Class_Deployment_100",
        exist_ok=False,
        save=True,
        plots=True
    )

    if (
        hasattr(model, "trainer")
        and model.trainer is not None
        and hasattr(
            model.trainer,
            "save_dir"
        )
    ):
        save_dir = Path(
            model.trainer.save_dir
        )

        print(
            "\n🎯 YOLOv11n 8-Class "
            "100% Retraining 完成"
        )

        print(
            f"結果：{save_dir}"
        )

        print(
            f"Deployment last.pt："
            f"{save_dir / 'weights' / 'last.pt'}"
        )

        print(
            f"best.pt："
            f"{save_dir / 'weights' / 'best.pt'}"
        )


if __name__ == "__main__":
    prepare_100_percent_dataset()
    check_dataset()
    create_data_yaml()
    train_100_percent()