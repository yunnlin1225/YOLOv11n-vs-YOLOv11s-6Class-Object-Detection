from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
import random

# ============================================================
# YOLOv11s 8-Class Final Model
# 自動偵測數量 + 自動 80% Train + 20% Validation
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

# YOLOv11s 原有專屬設定
CLS = 0.745
BOX = 8.0

# 自動切分比例
TRAIN_RATIO = 0.80

# 下面數值會由程式自動偵測並建立
EXPECTED = {}
EXPECTED_POSITIVE = 0
EXPECTED_BACKGROUND = 0
EXPECTED_TOTAL = 0

DATA_YAML = ROOT / "data.yaml"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
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


def split_dataset():
    global EXPECTED
    global EXPECTED_POSITIVE
    global EXPECTED_BACKGROUND
    global EXPECTED_TOTAL

    print("\n===== 自動偵測並切分 8-Class Final Dataset =====")

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
            f"❌ images 資料夾內沒有找到任何圖片：{IMAGES_DIR}"
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
            # 沒有 label 的圖片視為背景負樣本
            background_images.append(image)

    # =========================================================
    # 自動偵測資料數量
    # =========================================================

    EXPECTED_POSITIVE = len(positive_images)
    EXPECTED_BACKGROUND = len(background_images)
    EXPECTED_TOTAL = len(image_files)

    print("\n===== 原始 Dataset 自動偵測結果 =====")
    print(f"正樣本：{EXPECTED_POSITIVE}")
    print(f"背景圖：{EXPECTED_BACKGROUND}")
    print(f"總數：{EXPECTED_TOTAL}")

    # =========================================================
    # 自動計算 80 / 20
    # =========================================================

    train_positive_count = int(
        EXPECTED_POSITIVE * TRAIN_RATIO
    )

    val_positive_count = (
        EXPECTED_POSITIVE - train_positive_count
    )

    train_background_count = int(
        EXPECTED_BACKGROUND * TRAIN_RATIO
    )

    val_background_count = (
        EXPECTED_BACKGROUND - train_background_count
    )

    train_total = (
        train_positive_count
        + train_background_count
    )

    val_total = (
        val_positive_count
        + val_background_count
    )

    EXPECTED = {
        "train": {
            "positive": train_positive_count,
            "background": train_background_count,
            "total": train_total
        },
        "val": {
            "positive": val_positive_count,
            "background": val_background_count,
            "total": val_total
        }
    }

    print("\n===== 自動 80% / 20% 分配結果 =====")

    print(
        f"Train："
        f"{train_positive_count} 正樣本 + "
        f"{train_background_count} 背景 = "
        f"{train_total}"
    )

    print(
        f"Validation："
        f"{val_positive_count} 正樣本 + "
        f"{val_background_count} 背景 = "
        f"{val_total}"
    )

    # 固定 Seed，確保每次切分一致
    rng = random.Random(SEED)

    positive_images = positive_images.copy()
    background_images = background_images.copy()

    rng.shuffle(positive_images)
    rng.shuffle(background_images)

    # 80% Train / 20% Validation
    train_positive = positive_images[
        :train_positive_count
    ]

    val_positive = positive_images[
        train_positive_count:
    ]

    train_background = background_images[
        :train_background_count
    ]

    val_background = background_images[
        train_background_count:
    ]

    # 清除舊 train / val
    for folder in [
        IMAGES_DIR / "train",
        IMAGES_DIR / "val",
        LABELS_DIR / "train",
        LABELS_DIR / "val"
    ]:
        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    def copy_data(images, split):
        image_destination = IMAGES_DIR / split
        label_destination = LABELS_DIR / split

        for image in images:
            shutil.copy2(
                image,
                image_destination / image.name
            )

            source_label = (
                LABELS_DIR / f"{image.stem}.txt"
            )

            destination_label = (
                label_destination / f"{image.stem}.txt"
            )

            if source_label.exists():
                shutil.copy2(
                    source_label,
                    destination_label
                )
            else:
                # 背景圖沒有 txt 時，自動建立空白 label
                destination_label.touch()

    copy_data(train_positive, "train")
    copy_data(train_background, "train")

    copy_data(val_positive, "val")
    copy_data(val_background, "val")

    print("\n✅ 自動切分完成")

    print(
        f"Train："
        f"{train_positive_count} 正樣本 + "
        f"{train_background_count} 背景 = "
        f"{train_total}"
    )

    print(
        f"Validation："
        f"{val_positive_count} 正樣本 + "
        f"{val_background_count} 背景 = "
        f"{val_total}"
    )


def get_images(folder):
    return sorted([
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def create_data_yaml():
    data = {
        "path": ROOT.as_posix(),
        "train": "images/train",
        "val": "images/val",
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


def check_directories():
    required_dirs = [
        IMAGES_DIR / "train",
        IMAGES_DIR / "val",
        LABELS_DIR / "train",
        LABELS_DIR / "val"
    ]

    for folder in required_dirs:
        if not folder.exists():
            raise FileNotFoundError(
                f"❌ 找不到資料夾：{folder}"
            )

    print("✅ Train / Validation 資料夾皆存在")


def check_split(split):
    images_folder = IMAGES_DIR / split
    labels_folder = LABELS_DIR / split

    images = get_images(images_folder)

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
            labels_folder / f"{image.stem}.txt"
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
        for f in labels_folder.glob("*.txt")
    }

    extra_labels = label_stems - image_stems

    if extra_labels:
        raise ValueError(
            f"❌ {split} 發現沒有對應圖片的 Label："
            f"{sorted(extra_labels)}"
        )

    total = positive + background
    expected = EXPECTED[split]

    print(
        f"{split.upper()}："
        f"正樣本 {positive}、"
        f"背景 {background}、"
        f"總數 {total}"
    )

    if positive != expected["positive"]:
        raise ValueError(
            f"❌ {split} 正樣本應為 "
            f"{expected['positive']}，目前 {positive}"
        )

    if background != expected["background"]:
        raise ValueError(
            f"❌ {split} 背景圖應為 "
            f"{expected['background']}，目前 {background}"
        )

    if total != expected["total"]:
        raise ValueError(
            f"❌ {split} 總數應為 "
            f"{expected['total']}，目前 {total}"
        )

    return positive, background, total


def check_dataset():
    print("\n===== 8-Class Final Dataset 檢查 =====")

    train = check_split("train")
    val = check_split("val")

    total_positive = train[0] + val[0]
    total_background = train[1] + val[1]
    total_images = train[2] + val[2]

    if total_positive != EXPECTED_POSITIVE:
        raise ValueError(
            f"❌ 正樣本總數應為 {EXPECTED_POSITIVE}，"
            f"目前 {total_positive}"
        )

    if total_background != EXPECTED_BACKGROUND:
        raise ValueError(
            f"❌ 背景圖總數應為 {EXPECTED_BACKGROUND}，"
            f"目前 {total_background}"
        )

    if total_images != EXPECTED_TOTAL:
        raise ValueError(
            f"❌ 總圖片數應為 {EXPECTED_TOTAL}，"
            f"目前 {total_images}"
        )

    print("✅ 8-Class Final Dataset 檢查完成")
    print(f"Train：{train[2]} 張")
    print(f"Validation：{val[2]} 張")
    print(f"Total：{total_images} 張")


def train_final():
    print("\n===== YOLOv11s 8-Class Final Training =====")

    print(f"Epochs：{EPOCHS}")
    print(f"Batch：{BATCH}")
    print(f"Image Size：{IMGSZ}")
    print("Classes：8")
    print(f"cls：{CLS}")
    print(f"box：{BOX}")
    print(f"Seed：{SEED}")

    model = YOLO(
        "yolo11s.pt"
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

        # YOLOv11s 原有專屬設定
        cls=CLS,
        box=BOX,

        seed=SEED,
        deterministic=True,

        project=PROJECT_DIR.as_posix(),
        name="YOLOv11s_8Class_Final",
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

        print(
            "\n🎯 YOLOv11s 8-Class Final Model 訓練完成"
        )

        print(f"結果：{save_dir}")

        print(
            f"best.pt："
            f"{save_dir / 'weights' / 'best.pt'}"
        )

        print(
            f"last.pt："
            f"{save_dir / 'weights' / 'last.pt'}"
        )


if __name__ == "__main__":
    split_dataset()
    check_directories()
    check_dataset()
    create_data_yaml()
    train_final()