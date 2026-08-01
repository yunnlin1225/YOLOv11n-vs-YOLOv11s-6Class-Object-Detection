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

- **Fold1 ~ Fold5**：各折訓練結果。
- **weights**：存放模型權重（best.pt、last.pt）。
- **README.md**：專案說明文件。

## 權重檔說明

- **best.pt**：驗證集表現最佳的模型權重。
- **last.pt**：最後一個 Epoch 儲存的模型權重。

## 專案目的

透過 YOLOv11s 建立 Bear、Cat、Dog 三類動物物件偵測模型，比較五折交叉驗證的訓練成果，分析模型辨識效能，作為物件偵測研究與專題成果展示。
