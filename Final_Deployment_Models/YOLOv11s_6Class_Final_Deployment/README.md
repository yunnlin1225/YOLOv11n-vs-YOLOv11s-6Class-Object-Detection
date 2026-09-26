# 🚀 YOLOv11s Final Deployment
### YOLOv11s 最終部署模型

## 📌 1. 模型說明（Model Overview）

本階段為 YOLOv11s 經過前期模型訓練、**5-Fold Cross Validation** 與多階段模型優化後所建立的最終部署版本。

最終模型使用包含六類別目標樣本與背景負樣本的 **1050 張影像資料集**進行訓練，並沿用前期實驗與五折交叉驗證結果所選定的模型訓練參數。

為兼顧模型效能評估與實際部署需求，本階段分別建立以下兩種最終模型：

- 📊 **80% Training / 20% Validation：**用於評估最終模型的整體偵測效能。
- 🚀 **100% Full Training：**使用完整資料集進行訓練，作為後續實際部署使用的模型版本。

兩種訓練方式皆以 YOLOv11s 為基礎，作為最終模型建立與部署階段的實驗成果。

---

## ⚙️ 2. 最終訓練參數（Final Training Parameters）

YOLOv11s 最終部署模型的訓練參數，主要根據前期 **5-Fold Cross Validation 與模型優化實驗結果**進行設定，並將選定的參數應用於 80/20 與 100% 兩種最終訓練方式。

主要訓練參數如下：

| 訓練參數 | 設定值 |
|:---:|:---:|
| Model | YOLOv11s |
| Epochs | 100 |
| Batch Size | 8 |
| Image Size | 640 |
| Optimizer | Auto |
| AMP | True |
| Cosine LR | True |
| Close Mosaic | 10 |
| cls | 0.745 |
| box | 8.0 |

其中 **cls=0.745、box=8.0** 為前期模型優化與五折交叉驗證後所選定的參數組合，並作為 YOLOv11s 最終模型訓練的主要設定。

> 80/20 與 100% 版本皆以相同的主要模型參數進行訓練，使兩種最終訓練方式具有一致的模型設定基準。

---

## 📊 3. 80% Training / 20% Validation

為評估 YOLOv11s 最終模型的偵測效能，本階段將 **1050 張影像資料集**依 80% Training / 20% Validation 進行切分。

資料配置如下：

- 🏋️ **Training：840 張**
- 📊 **Validation：210 張**
- 📁 **Total：1050 張**

此版本保留 20% 的資料作為 Validation，用於評估模型訓練完成後的整體表現。

### 📈 模型評估結果（Evaluation Results）

| Precision | Recall | mAP50 | mAP50-95 |
|:---:|:---:|:---:|:---:|
| **91.9%** | **94.6%** | **95.8%** | **82.0%** |

整體結果顯示，YOLOv11s 在 80% Training / 20% Validation 設定下，四項主要評估指標皆維持良好表現，作為最終模型效能分析的重要參考依據。

整體結果顯示，YOLOv11s 在 80/20 資料切分下，四項主要評估指標皆維持良好表現，作為最終模型效能分析的重要參考依據。

---

## 🚀 4. 100% Full Training

完成 80% Training / 20% Validation 的模型效能評估後，進一步使用 **1050 張完整資料集**進行 YOLOv11s 最終模型訓練。

此版本將全部資料投入 Training，不另外保留 20% 作為 Validation，使六類別目標樣本與背景負樣本皆完整參與模型訓練。

資料組成如下：

- 🎯 **六類別目標樣本：900 張**
- 🌲 **背景負樣本：150 張**
- 📁 **Training：1050 張（100%）**

100% Full Training 沿用前期 **5-Fold Cross Validation、模型優化實驗與 80/20 最終模型評估**所確認的訓練設定，在充分利用現有資料的情況下建立 YOLOv11s 完整訓練版本。

### 📈 模型結果（Model Results）

| Precision | Recall | mAP50 | mAP50-95 |
|:---:|:---:|:---:|:---:|
| **99.0%** | **98.4%** | **99.3%** | **93.7%** |

100% Full Training 使用完整 1050 張影像進行模型建立，並作為後續黑熊智慧警示系統實際部署所使用的 YOLOv11s 模型版本。

此版本主要作為後續 **黑熊智慧警示系統實際部署**所使用的 YOLOv11s 物件偵測模型。

> **80/20 版本主要用於模型效能評估；100% Full Training 則用於完整資料訓練與最終部署。**

---

## ✅ 5. 最終模型說明（Final Model Summary）

YOLOv11s 最終部署模型建立於前期 **5-Fold Cross Validation、模型優化與最終 80/20 效能評估**的實驗基礎上，並使用確認後的訓練設定完成最終模型建立。

其中，**80% Training / 20% Validation** 版本主要用於觀察模型的 Precision、Recall、mAP50 與 mAP50-95；**100% Full Training** 版本則使用全部 1050 張影像進行訓練，作為最終部署版本。

最終 YOLOv11s 模型負責辨識 **Dog、Cat、Bear、Pig、Monkey、Person** 六個正式類別，並作為整體 **黑熊智慧警示系統的物件偵測模型**，提供後續系統進行分析與警示所需的視覺辨識資訊。
