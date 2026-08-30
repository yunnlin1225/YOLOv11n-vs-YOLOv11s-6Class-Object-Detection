from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import shutil
import yaml
import time
import gc
import torch

# ==========================
# 基本設定
# ==========================

ROOT = Path(r"C:\Users\user\Desktop\YOLOv11s_6Categories(2)")

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

# 固定亂數種子
SEED = 42

# 預期背景圖數量
EXPECTED_BACKGROUND = 150

# ==========================
# 建立五折資料
# ==========================

def create_folds():
    images_dir = ROOT / "images"
    labels_dir = ROOT / "labels"

    # 檢查資料夾
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 資料夾：{images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到 labels 資料夾：{labels_dir}")

    # 讀取所有圖片
    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(images_dir.glob(ext))

    image_files = sorted(image_files)

    print(f"\n共找到 {len(image_files)} 張圖片")

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"在 {images_dir} 找不到任何圖片，請確認路徑與檔名是否正確！"
        )

    # 分離正樣本與背景圖
    positive_images = []
    background_images = []

    for img in image_files:
        label = labels_dir / f"{img.stem}.txt"

        if label.exists():
            content = label.read_text(encoding="utf-8").strip()

            if content:
                positive_images.append(img)
            else:
                background_images.append(img)
        else:
            background_images.append(img)

    print(f"正樣本圖片：{len(positive_images)} 張")
    print(f"背景圖片：{len(background_images)} 張")
    print(f"總圖片數：{len(image_files)} 張")

    # 檢查數量
    if len(positive_images) != 900:
        raise ValueError(
            f"\n❌ 正樣本數量應為 900 張，目前偵測到 {len(positive_images)} 張！"
        )

    if len(background_images) != EXPECTED_BACKGROUND:
        raise ValueError(
            f"\n❌ 背景圖數量應為 {EXPECTED_BACKGROUND} 張，"
            f"目前偵測到 {len(background_images)} 張！\n"
            "請檢查 labels 中的空白 txt 或缺少 txt 的圖片。"
        )

    if len(image_files) != 1050:
        raise ValueError(
            f"\n❌ 總圖片數應為 1050 張，目前偵測到 {len(image_files)} 張！"
        )

    print("\n✅ 資料數量確認正確：900 張正樣本 + 150 張背景 = 1050 張")

    # 原本 900 張正樣本 KFold
    positive_kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    # 150 張背景圖 KFold
    background_kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    positive_splits = list(positive_kf.split(positive_images))
    background_splits = list(background_kf.split(background_images))

    # 建立 Fold 1 ~ Fold 5
    for fold in range(1, 6):
        print(f"\n===== 建立 Fold {fold} =====")

        fold_dir = ROOT / f"fold{fold}"

        # 刪除舊 Fold，避免沿用以前 900 張資料
        if fold_dir.exists():
            print(f"⚠ Fold {fold} 已存在，刪除舊資料並重新建立...")
            shutil.rmtree(fold_dir)

        for p in [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val"
        ]:
            (fold_dir / p).mkdir(parents=True, exist_ok=True)

        positive_train_idx, positive_val_idx = positive_splits[fold - 1]
        background_train_idx, background_val_idx = background_splits[fold - 1]

        # Train 正樣本
        for i in positive_train_idx:
            img = positive_images[i]
            label = labels_dir / f"{img.stem}.txt"

            shutil.copy2(
                img,
                fold_dir / "images/train" / img.name
            )

            shutil.copy2(
                label,
                fold_dir / "labels/train" / label.name
            )

        # Validation 正樣本
        for i in positive_val_idx:
            img = positive_images[i]
            label = labels_dir / f"{img.stem}.txt"

            shutil.copy2(
                img,
                fold_dir / "images/val" / img.name
            )

            shutil.copy2(
                label,
                fold_dir / "labels/val" / label.name
            )

        # Train 背景
        for i in background_train_idx:
            img = background_images[i]
            label = labels_dir / f"{img.stem}.txt"

            shutil.copy2(
                img,
                fold_dir / "images/train" / img.name
            )

            destination_label = (
                fold_dir / "labels/train" / f"{img.stem}.txt"
            )

            if label.exists():
                shutil.copy2(label, destination_label)
            else:
                destination_label.touch()

        # Validation 背景
        for i in background_val_idx:
            img = background_images[i]
            label = labels_dir / f"{img.stem}.txt"

            shutil.copy2(
                img,
                fold_dir / "images/val" / img.name
            )

            destination_label = (
                fold_dir / "labels/val" / f"{img.stem}.txt"
            )

            if label.exists():
                shutil.copy2(label, destination_label)
            else:
                destination_label.touch()

        # 顯示本 Fold 數量
        train_positive_count = len(positive_train_idx)
        val_positive_count = len(positive_val_idx)
        train_background_count = len(background_train_idx)
        val_background_count = len(background_val_idx)

        train_total = train_positive_count + train_background_count
        val_total = val_positive_count + val_background_count

        print(f"Fold {fold} 正樣本 Train：{train_positive_count}")
        print(f"Fold {fold} 背景 Train：{train_background_count}")
        print(f"Fold {fold} Train 總數：{train_total}")

        print(f"Fold {fold} 正樣本 Val：{val_positive_count}")
        print(f"Fold {fold} 背景 Val：{val_background_count}")
        print(f"Fold {fold} Val 總數：{val_total}")

        # 產生 YAML
        yaml_file = ROOT / f"fold{fold}.yaml"

        data = {
            "path": fold_dir.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES
        }

        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False
            )

        print(f"✔ YAML 完成：fold{fold}.yaml")

    print("\n✅ 五折資料建立完成！")

# ==========================
# 五折訓練
# ==========================

def train_5fold():
    print(
        f"\n🚀 開始 6 類別 + {EXPECTED_BACKGROUND} 張背景圖五折訓練"
    )
    print(f"資料夾：{ROOT.name}")
    print(f"每折：{EPOCHS} 輪")

    for fold in range(1, 6):
        print("\n====================")
        print(f"🔥 Fold {fold} / 5")
        print("====================")

        model = YOLO("yolo11s.pt")
        yaml_path = (ROOT / f"fold{fold}.yaml").as_posix()

        model.train(
            data=yaml_path,
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

            # 原本最佳超參數
            cls=0.745,
            box=8.0,

            # 固定實驗
            seed=SEED,
            deterministic=True,

            project="YOLOv11_6Class_5Fold_BG150",
            name=f"Fold_{fold}",
            exist_ok=True
        )

        # 清理 GPU
        print("🧊 清理 GPU 記憶體...")

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("❄ 冷卻 60 秒...")
        time.sleep(60)

    print("\n🎯 6 類別 + 150 張背景圖五折訓練全部完成！")

# ==========================
# 主程式
# ==========================

if __name__ == "__main__":
    create_folds()
    train_5fold()