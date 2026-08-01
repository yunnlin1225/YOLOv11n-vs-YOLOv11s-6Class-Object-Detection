# YOLOv11s_Bear_Cat_Dog

## 專案介紹

本專案使用 YOLOv11s 物件偵測模型，建立熊（Bear）、貓（Cat）及狗（Dog）三類動物辨識模型，並採用五折交叉驗證（5-Fold Cross Validation）進行模型訓練與效能評估，以提升模型的可靠性與泛化能力。

## 類別編號（Class ID）

| 類別編號 | 類別名稱 |
|---------|---------|
| 0 | Dog（狗） |
| 1 | Cat（貓） |
| 2 | Bear（熊） |

## 專案內容

- 模型：YOLOv11s
- 訓練方式：5-Fold Cross Validation
- 影像尺寸：640 × 640
- 深度學習框架：Ultralytics YOLO

## 資料夾說明

- **Fold1 ~ Fold5**：存放五折交叉驗證各折（Fold1～Fold5）的訓練結果，每一折皆包含獨立的資料集設定檔、模型權重及訓練成果。

- **weights**：存放各折訓練完成後所產生的模型權重。
- **README.md**：本專案說明文件，包含專案介紹、類別編號、資料夾架構、檔案用途及模型相關資訊。

## 權重檔說明

- **best.pt**：驗證集表現最佳的模型權重。
- **last.pt**：最後一個 Epoch 儲存的模型權重。

## auto_5fold.py
五折交叉驗證（5-Fold Cross Validation）主程式，負責自動建立各折資料集、產生對應的 `data.yaml` 設定檔，並執行 YOLOv11s 模型訓練，最後儲存各折的訓練結果與模型權重。

## data.yaml
各折（Fold1～Fold5）的資料集設定檔，記錄 YOLOv11s 訓練所需的資料路徑、類別數（nc）及類別名稱（names）。

## 專案目的

透過 YOLOv11s 建立 Bear、Cat、Dog 三類動物物件偵測模型，比較五折交叉驗證的訓練成果，分析模型辨識效能，作為物件偵測研究與專題成果展示。
