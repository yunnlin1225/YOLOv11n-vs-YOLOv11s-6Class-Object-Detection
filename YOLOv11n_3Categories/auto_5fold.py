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

ROOT = Path(r"C:\Users\user\Desktop\YOLOv11n_Bear_Cat_Dog")

# 類別名稱對照 (0:狗, 1:貓, 2:熊)
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
# 💡 自動修正熊的標記檔案 (bear_1 到 bear_107)
# ==========================
def fix_bear_labels():
    labels_dir = ROOT / "labels"
    if not labels_dir.exists():
        return

    # 自動抓取所有 bear_ 開頭的標記檔
    bear_files = list(labels_dir.glob("bear_*.txt"))
    if len(bear_files) == 0:
        return

    print(f"\n🔧 偵測到 {len(bear_files)} 個熊的標記檔案，正在檢查並將類別 0 修正為 2...")
    modify_count = 0

    for txt_file in bear_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        has_changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # 如果第一個數字是 '0'，自動改成 '2'
                if parts[0] == "0":
                    parts[0] = "2"
                    has_changed = True
                new_lines.append(" ".join(parts) + "\n")
            else:
                new_lines.append(line)

        if has_changed:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            modify_count += 1

    if modify_count > 0:
        print(f"✔ 標記修正完成！共修正了 {modify_count} 個熊的檔案。")
    else:
        print("⚡ 熊的標記檔案先前已修正過，無需重複變更。")

# ==========================
# 建立五折資料
# ==========================

def create_folds():
    # 在複製與切分資料前，先執行標記自動修正
    fix_bear_labels()

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

        # 確保宣告為 yolo11n.pt
        model = YOLO("yolo11n.pt")
        yaml_path = (ROOT / f"fold{fold}.yaml").as_posix()

        model.train(
            model="yolo11n.pt",  # 💡 關鍵修改：明確指定每一折都從 yolo11n.pt 的權重開始，避免交叉污染
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