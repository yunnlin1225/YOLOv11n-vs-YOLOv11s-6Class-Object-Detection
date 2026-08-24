from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import shutil
import yaml
import time
import gc
import torch

# ==========================
# 基本設定（已更新路徑與條件）
# ==========================

# 💡 資料夾路徑已更新
ROOT = Path(r"C:\Users\user\Desktop\YOLOv11s_Bear_Cat_Dog")

# 類別名稱對照
CLASS_NAMES = {
    0: "dog",
    1: "cat",
    2: "bear"
}

# 訓練參數
EPOCHS = 100      # 跑滿 100 輪
BATCH = 8         # RTX 2060 穩健值
IMGSZ = 640
WORKERS = 2       # Windows 系統建議維持 2，若卡死請改 0

# ==========================
# 建立五折資料
# ==========================

def create_folds():
    images_dir = ROOT / "images"
    labels_dir = ROOT / "labels"

    if all((ROOT / f"fold{f}").exists() for f in range(1, 6)):
        print("\n⚡ 偵測到五折資料夾已存在，跳過資料複製步驟。")
        return

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(images_dir.glob(ext))

    image_files = sorted(image_files)
    print(f"\n找到 {len(image_files)} 張圖片")

    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {images_dir} 找不到任何圖片，請確認路徑與檔名是否正確！")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), start=1):

        print(f"\n===== 建立 Fold {fold} =====")
        fold_dir = ROOT / f"fold{fold}"

        for p in ["images/train", "images/val", "labels/train", "labels/val"]:
            (fold_dir / p).mkdir(parents=True, exist_ok=True)

        # Train
        for i in train_idx:
            img = image_files[i]
            shutil.copy2(img, fold_dir / "images/train" / img.name)

            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, fold_dir / "labels/train" / label.name)

        # Validation
        for i in val_idx:
            img = image_files[i]
            shutil.copy2(img, fold_dir / "images/val" / img.name)

            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, fold_dir / "labels/val" / label.name)

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

        print(f"✔ YAML完成：fold{fold}.yaml")

# ==========================
# 五折訓練
# ==========================

def train_5fold():
    print(f"\n🚀 開始五折訓練（資料夾：{ROOT.name}，每折跑滿 {EPOCHS} 輪）")

    for fold in range(1, 6):
        print(f"\n====================")
        print(f"🔥 Fold {fold}")
        print(f"====================")

        model = YOLO("yolo11s.pt")
        yaml_path = (ROOT / f"fold{fold}.yaml").as_posix()

        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=0,
            workers=WORKERS,
            cache=True,          # 300張圖很小，開啟 cache 可以大幅加速訓練
            amp=True,
            lr0=0.005,
            project="YOLOv11_5Fold",
            name=f"Fold_{fold}",
            exist_ok=True
        )

        # ==========================
        # 🔥 防過熱與釋放 VRAM 強化區
        # ==========================
        print("🧊 清理 GPU 記憶體...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("❄ 冷卻 60 秒...")
        time.sleep(60)

    print("\n🎯 五折訓練全部完成！")

# ==========================
# 主程式
# ==========================

if __name__ == "__main__":
    create_folds()
    train_5fold()