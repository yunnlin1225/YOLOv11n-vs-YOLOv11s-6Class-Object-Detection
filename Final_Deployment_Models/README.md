# 🚀 Final Deployment Models
### YOLOv11n / YOLOv11s 最終部署模型

## 📌 1. 最終模型說明

本資料夾整理本研究完成模型訓練、五折交叉驗證與模型優化後所建立的 **YOLOv11n 與 YOLOv11s 最終部署模型**。

前期研究透過 **5-Fold Cross Validation** 評估模型在不同資料切分下的效能與穩定性，並經過模型參數調整、六類別資料集擴充及背景負樣本優化，逐步完成最終模型訓練設定。

在最終部署階段，YOLOv11n 與 YOLOv11s 分別採用 **80% Training / 20% Validation** 與 **100% Full Training** 兩種訓練方式建立模型，作為後續模型效能評估與實際系統部署使用。

## 📊 2. 最終訓練資料集（Final Training Dataset）

最終部署模型使用經過前期研究與模型優化後所建立的 **1050 張影像資料集**進行訓練。

資料集組成如下：

- 🐶 **Dog：150 張**
- 🐱 **Cat：150 張**
- 🐻 **Bear：150 張**
- 🐷 **Pig：150 張**
- 🐒 **Monkey：150 張**
- 👤 **Person：150 張**
- 🌲 **Background Negative Samples：150 張**

其中六個正式偵測類別共包含 **900 張目標樣本**，並加入 **150 張背景負樣本**，使最終訓練資料集共計 **1050 張影像**。

背景負樣本包含白天與夜間的森林、山景、草地、灌木、岩石及溪流等自然環境，用於增加模型對無目標物件背景影像的學習。

> **注意：Background Negative Samples 並非第 7 個偵測類別，最終 YOLO 模型仍維持 6 個正式偵測類別。**

## 🔀 3. 最終訓練方式（Final Training Strategy）

最終部署模型沿用前期 **5-Fold Cross Validation 與模型優化實驗**所選定的最佳訓練參數，並分別以兩種方式進行最終訓練。

### 📊 80% Training / 20% Validation

將 1050 張資料依 **80% Training / 20% Validation** 進行切分，保留驗證資料評估模型的 Precision、Recall、mAP50 與 mAP50-95。

### 🚀 100% Full Training

使用 **100% 完整資料集**進行訓練，使全部 1050 張影像皆參與最終模型建立。

此版本的模型設定會參考前期 **5-Fold Cross Validation、模型優化結果，以及 80/20 訓練版本的 Precision、Recall、mAP50 與 mAP50-95 評估結果**，作為最終模型建立與後續部署的依據。

> **80/20 版本主要用於模型效能評估；100% 版本則根據前期實驗與評估結果，使用完整資料集建立最終部署模型。**

## 📂 4. 模型與檔案架構（Model and File Structure）

最終部署模型依 YOLOv11n 與 YOLOv11s 分別整理，並於各模型中區分 **80% Training / 20% Validation** 與 **100% Full Training** 兩種最終訓練方式。

```text
Final_Deployment_Models/
│
├── README.md
│
├── YOLOv11n_Final_Deployment/
│   ├── README.md
│   ├── Train_Val_80_20/
│   │   ├── train_final.py
│   │   ├── args.yaml
│   │   ├── weights/
│   │   └── results/
│   │
│   └── Full_Training_100/
│       ├── train_final.py
│       ├── args.yaml
│       ├── weights/
│       └── results/
│
└── YOLOv11s_Final_Deployment/
    ├── README.md
    ├── Train_Val_80_20/
    │   ├── train_final.py
    │   ├── args.yaml
    │   ├── weights/
    │   └── results/
    │
    └── Full_Training_100/
        ├── train_final.py
        ├── args.yaml
        ├── weights/
        └── results/
```
各模型資料夾中的 `README.md` 用於記錄 YOLOv11n 與 YOLOv11s 的最終訓練設定與結果；`weights/` 保存模型權重，`results/` 則整理相關訓練與評估結果。

---

## 🎯 5. 最終部署用途（Final Deployment Purpose）

最終建立的 YOLOv11n 與 YOLOv11s 模型將作為 **黑熊智慧警示系統的物件偵測模組**，負責從攝影機影像或影片中辨識以下六個正式偵測類別：

- 🐶 Dog
- 🐱 Cat
- 🐻 Bear
- 🐷 Pig
- 🐒 Monkey
- 👤 Person

模型偵測後可提供 **物件類別、數量、信心分數及位置**等資訊，供整體系統後續進行分析與處理。

其中以 **Bear（黑熊）** 為主要警示目標，透過即時物件偵測提供視覺感知資訊，作為後續風險判斷與警示機制的重要依據。

最終模型將應用於自然環境中的野生動物監測與黑熊警示情境，作為整體黑熊智慧警示系統的前端視覺辨識模型。
