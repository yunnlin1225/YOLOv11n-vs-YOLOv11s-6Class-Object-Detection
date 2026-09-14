from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil

# ============================================================
# YOLOv11s 100% Deployment Retraining
# 1050 張全部用於 Training，不切 Validation
# ============================================================

ROOT = Path(__file__).resolve().parent

CLASS_NAMES = {
    0: "dog",
    1: "cat",
    2: "bear",
    3: "pig",
    4: "monkey",
    5: "person"
}

# 訓練參數
EPOCHS = 100
BATCH = 8
IMGSZ = 640
WORKERS = 2
SEED = 42

# YOLOv11s 二次優化最佳參數
CLS = 0.745
BOX = 8.0

EXPECTED_POSITIVE = 900
EXPECTED_BACKGROUND = 150
EXPECTED_TOTAL = 1050

DATA_YAML = ROOT / "data.yaml"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train_all"
TRAIN_LABELS_DIR = LABELS_DIR / "train_all"

PROJECT_DIR = ROOT / "runs"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_source_images():
    return sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def prepare_100_percent_dataset():
    print("\n===== 建立 100% Training Dataset =====")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"❌ 找不到 images：{IMAGES_DIR}"
        )

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"❌ 找不到 labels：{LABELS_DIR}"
        )

    image_files = get_source_images()

    if len(image_files) != EXPECTED_TOTAL:
        raise ValueError(
            f"❌ 原始 images 應有 {EXPECTED_TOTAL} 張，"
            f"目前找到 {len(image_files)} 張"
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
                positive_images.append(image)
            else:
                background_images.append(image)
        else:
            # 沒有 txt 的圖片視為背景負樣本
            background_images.append(image)

    print(f"正樣本：{len(positive_images)}")
    print(f"背景圖：{len(background_images)}")
    print(f"總數：{len(image_files)}")

    if len(positive_images) != EXPECTED_POSITIVE:
        raise ValueError(
            f"❌ 正樣本應為 {EXPECTED_POSITIVE} 張，"
            f"目前 {len(positive_images)} 張"
        )

    if len(background_images) != EXPECTED_BACKGROUND:
        raise ValueError(
            f"❌ 背景圖應為 {EXPECTED_BACKGROUND} 張，"
            f"目前 {len(background_images)} 張"
        )

    # 只重建 train_all，不動原本 80/20 的 train / val
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
            # 背景圖若沒有 txt，自動建立空白 label
            destination_label.touch()

    print("✅ 100% Training Dataset 建立完成")
    print("Train：900 正樣本 + 150 背景 = 1050")
    print("Validation：不切分")


def create_data_yaml():
    data = {
        "path": ROOT.as_posix(),
        "train": "images/train_all",

        # Ultralytics Detection YAML 需要 val 欄位。
        # 100% Retraining 不使用獨立 Validation，
        # 因此此處只為符合格式而指向 train_all。
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

    print(f"✅ data.yaml 建立完成：{DATA_YAML}")


def check_dataset():
    print("\n===== 100% Dataset 檢查 =====")

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

        image_stems.add(image.stem)

        label = (
            TRAIN_LABELS_DIR / f"{image.stem}.txt"
        )

        if not label.exists():
            raise FileNotFoundError(
                f"❌ 找不到 Label：{label}"
            )

        content = label.read_text(
            encoding="utf-8"
        ).strip()

        if content:
            positive += 1
        else:
            background += 1

    label_stems = {
        f.stem
        for f in TRAIN_LABELS_DIR.glob("*.txt")
    }

    extra_labels = label_stems - image_stems

    if extra_labels:
        raise ValueError(
            f"❌ 發現沒有對應圖片的 Label："
            f"{sorted(extra_labels)}"
        )

    total = positive + background

    print(f"正樣本：{positive}")
    print(f"背景圖：{background}")
    print(f"總數：{total}")

    if positive != EXPECTED_POSITIVE:
        raise ValueError(
            f"❌ 正樣本應為 {EXPECTED_POSITIVE}，"
            f"目前 {positive}"
        )

    if background != EXPECTED_BACKGROUND:
        raise ValueError(
            f"❌ 背景圖應為 {EXPECTED_BACKGROUND}，"
            f"目前 {background}"
        )

    if total != EXPECTED_TOTAL:
        raise ValueError(
            f"❌ 總數應為 {EXPECTED_TOTAL}，"
            f"目前 {total}"
        )

    print("✅ 100% Dataset 檢查完成")
    print("Training：1050 張")
    print("Validation：0 張")


def train_100_percent():
    print("\n===== YOLOv11s 100% Retraining =====")
    print(f"Training Images：{EXPECTED_TOTAL}")
    print(f"Epochs：{EPOCHS}")
    print(f"Batch：{BATCH}")
    print(f"Image Size：{IMGSZ}")
    print(f"cls：{CLS}")
    print(f"box：{BOX}")
    print(f"Seed：{SEED}")
    print("Validation：False")

    model = YOLO("yolo11s.pt")

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
        cls=CLS,
        box=BOX,
        seed=SEED,
        deterministic=True,

        # 100% Retraining 專用
        val=False,
        patience=0,

        project=PROJECT_DIR.as_posix(),
        name="YOLOv11s_Deployment_100",
        exist_ok=False,
        save=True,
        plots=True
    )

    if (
        hasattr(model, "trainer")
        and model.trainer is not None
        and hasattr(model.trainer, "save_dir")
    ):
        save_dir = Path(
            model.trainer.save_dir
        )

        print("\n🎯 YOLOv11s 100% Retraining 完成")
        print(f"結果：{save_dir}")

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