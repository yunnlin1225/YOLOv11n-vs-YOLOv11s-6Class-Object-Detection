import gc
from pathlib import Path
import shutil
import time
import torch
from ultralytics import YOLO
import yaml
from sklearn.model_selection import KFold

# ==========================
# 基本設定（已更新路徑加上 (1)）
# ==========================

# 💡 指向正確的 YOLOv11n 6類別資料集根目錄
ROOT = Path(r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)')

# 6 個類別對照表
CLASS_NAMES = {
    0: 'dog',
    1: 'cat',
    2: 'bear',
    3: 'pig',
    4: 'monkey',
    5: 'person',
}

# 訓練參數
EPOCHS = 100  # 跑滿 100 輪
BATCH = 8  # RTX 2060 穩健值
IMGSZ = 640
WORKERS = 2  # Windows 系統建議維持 2


# ==========================
# 建立五折資料
# ==========================


def create_folds():
  images_dir = ROOT / 'images'
  labels_dir = ROOT / 'labels'

  # 如果已經建立過 fold1~fold5，自動跳過
  if all((ROOT / f'fold{f}').exists() for f in range(1, 6)):
    print('\n⚡ 偵測到五折資料夾已存在，跳過資料複製步驟。')
    return

  image_files = []
  for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
    image_files.extend(images_dir.glob(ext))

  image_files = sorted(image_files)
  print(f'\n共找到 {len(image_files)} 張圖片（包含 6 類別）')

  if len(image_files) == 0:
    raise FileNotFoundError(
        f'在 {images_dir} 找不到任何圖片，請確認路徑與檔名是否正確！'
    )

  # 使用固定隨機種子 42 確保每折切分公平
  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), start=1):

    print(f'\n===== 建立 Fold {fold} =====')
    fold_dir = ROOT / f'fold{fold}'

    for p in ['images/train', 'images/val', 'labels/train', 'labels/val']:
      (fold_dir / p).mkdir(parents=True, exist_ok=True)

    # Train 複製
    for i in train_idx:
      img = image_files[i]
      shutil.copy2(img, fold_dir / 'images/train' / img.name)

      label = labels_dir / f'{img.stem}.txt'
      if label.exists():
        shutil.copy2(label, fold_dir / 'labels/train' / label.name)

    # Validation 複製
    for i in val_idx:
      img = image_files[i]
      shutil.copy2(img, fold_dir / 'images/val' / img.name)

      label = labels_dir / f'{img.stem}.txt'
      if label.exists():
        shutil.copy2(label, fold_dir / 'labels/val' / label.name)

    # 產生 YAML 檔
    yaml_file = ROOT / f'fold{fold}.yaml'

    data = {
        'path': fold_dir.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASS_NAMES,
    }

    with open(yaml_file, 'w', encoding='utf-8') as f:
      yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f'✔ YAML完成：fold{fold}.yaml (已寫入 6 個類別)')


# ==========================
# 五折訓練
# ==========================


def train_5fold():
  print(
      f'\n🚀 開始 YOLOv11n 6 類別五折訓練（資料夾：{ROOT.name}，每折'
      f' {EPOCHS} 輪）'
  )

  for fold in range(1, 6):
    print('\n====================')
    print(f'🔥 Fold {fold} / 5 (YOLOv11n)')
    print('====================')

    # 使用 YOLOv11n 權重
    model = YOLO('yolo11n.pt')
    yaml_path = (ROOT / f'fold{fold}.yaml').as_posix()

    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        cache=False,
        amp=True,
        optimizer='auto',
        cos_lr=True,
        close_mosaic=10,
        cls=0.75,
        project='YOLOv11n_6Class_5Fold',
        name=f'Fold_{fold}',
        exist_ok=True,
    )

    # ==========================
    # 釋放 VRAM 與冷卻
    # ==========================
    print('🧊 清理 GPU 記憶體...')
    del model
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    print('❄ 冷卻 60 秒...')
    time.sleep(60)

  print('\n🎯 YOLOv11n 6 類別五折訓練全部完成！')


# ==========================
# 主程式
# ==========================

if __name__ == '__main__':
  create_folds()
  train_5fold()