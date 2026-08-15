from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import shutil
import yaml
import time
import gc
import torch

# ==========================
# 基本設定（已更新 6 類別）
# ==========================

# 💡 請確認 ROOT 指向你合併好「6個類別」總資料集的根目錄
# 該目錄下需包含 images/ 與 labels/ 兩個資料夾
ROOT = Path(r"C:\Users\user\Desktop\YOLOv11s_6Categories(2)")

# 💡 1. 更新為完整的 6 個類別對照表
CLASS_NAMES = {
    0: "dog",
    1: "cat",
    2: "bear",
    3: "pig",
    4: "monkey",
    5: "person"
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

    # 如果已經建立過 fold1~fold5，自動跳過
    if all((ROOT / f"fold{f}").exists() for f in range(1, 6)):
        print("\n⚡ 偵測到五折資料夾已存在，跳過資料複製步驟。")
        return

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(images_dir.glob(ext))

    image_files = sorted(image_files)
    print(f"\n共找到 {len(image_files)} 張圖片（包含 6 類別）")

    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {images_dir} 找不到任何圖片，請確認路徑與檔名是否正確！")

    # 使用固定隨機種子 42 確保每折切分公平
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), start=1):

        print(f"\n===== 建立 Fold {fold} =====")
        fold_dir = ROOT / f"fold{fold}"

        for p in ["images/train", "images/val", "labels/train", "labels/val"]:
            (fold_dir / p).mkdir(parents=True, exist_ok=True)

        # Train 複製
        for i in train_idx:
            img = image_files[i]
            shutil.copy2(img, fold_dir / "images/train" / img.name)

            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, fold_dir / "labels/train" / label.name)

        # Validation 複製
        for i in val_idx:
            img = image_files[i]
            shutil.copy2(img, fold_dir / "images/val" / img.name)

            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, fold_dir / "labels/val" / label.name)

        # 產生 YAML 檔
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

        print(f"✔ YAML完成：fold{fold}.yaml (已寫入 6 個類別)")

# ==========================
# 五折訓練
# ==========================

def train_5fold():
    print(f"\n🚀 開始 6 類別五折訓練（資料夾：{ROOT.name}，每折 {EPOCHS} 輪）")

    for fold in range(1, 6):
        print(f"\n====================")
        print(f"🔥 Fold {fold} / 5")
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
            cache=False,         # 💡 2. 6類別總圖片增加，建議設 False 避免記憶體/VRAM 爆滿
            amp=True,
            optimizer="auto",    # 💡 自動選擇最適優化器
            cos_lr=True,
            close_mosaic=10,
            cls=0.745,            # 💡 分類損失權重 (保持 0.745 穩住高 Precision)
            box=8.0,              # 🔥【新增】調高定位損失權重至 8.0（預設為 7.5），強化框的精準度以補齊 Recall 至 9 成！
            project="YOLOv11_6Class_5Fold",  # 專案目錄名稱
            name=f"Fold_{fold}",
            exist_ok=True
        )

        # ==========================
        # 🔥 防過熱與釋放 VRAM 區
        # ==========================
        print("🧊 清理 GPU 記憶體...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("❄ 冷卻 60 秒...")
        time.sleep(60)

    print("\n🎯 6 類別五折訓練全部完成！")

# ==========================
# 主程式
# ==========================

if __name__ == "__main__":
    create_folds()
    train_5fold()