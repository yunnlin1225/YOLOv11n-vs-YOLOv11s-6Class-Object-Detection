## 專案介紹 (Project Introduction)

本專案以 Ultralytics YOLOv11s 物件偵測模型為基礎，建置熊（Bear）、貓（Cat）及狗（Dog）三類動物物件偵測系統，並針對原始模型進行優化，以提升模型整體辨識效能與泛化能力。透過蒐集與標註三類動物影像資料，建立符合 YOLO 格式之資料集，並結合深度學習技術進行模型訓練，使模型能夠準確辨識影像中的目標物件及其位置。

本次研究將資料集擴充至每個類別 150 張影像，共計 450 張影像，並採用五折交叉驗證（5-Fold Cross Validation）進行模型訓練與效能評估。透過增加樣本數量及模型訓練優化，可提升模型於不同場景下之辨識穩定性與泛化能力，降低資料分割所造成的評估偏差。

此外，本專案建置完整的自動化五折訓練流程，包含資料集切分、`data.yaml` 設定檔建立、YOLOv11s 模型訓練、模型權重儲存及各折訓練成果管理，使整體訓練流程更具一致性、可重現性與管理效率，亦方便後續進行模型測試、效能分析及成果展示。

## 資料集資訊 (Dataset Information)

本專案使用自行蒐集與標註之動物影像資料集，共包含三個類別，每個類別皆使用 150 張影像進行模型訓練與五折交叉驗證。本資料集應用於優化後的 YOLOv11s 物件偵測模型，以評估模型對熊（Bear）、貓（Cat）及狗（Dog）三類動物之辨識效能。

| 類別編號 (Class ID) | 類別 (Class) | 中文名稱 | 影像數量 (Images) |
|:-------------------:|:------------:|:--------:|------------------:|
| 0 | Dog | 狗 | 150 |
| 1 | Cat | 貓 | 150 |
| 2 | Bear | 熊 | 150 |
| **Total** | - | **總計** | **450** |

## 專案內容  (Project Information)

- 模型：YOLOv11s
- 訓練方式：5-Fold Cross Validation
- 影像尺寸：640 × 640
- 深度學習框架：Ultralytics YOLO

## 資料夾說明  (Folder Structure)

- **Fold1 ~ Fold5**：存放五折交叉驗證各折（Fold1～Fold5）的訓練結果，每一折皆包含獨立的資料集設定檔、模型權重及訓練成果。

- **weights**：存放各折訓練完成後所產生的模型權重。
- **README.md**：本專案說明文件，包含專案介紹、類別編號、資料夾架構、檔案用途及模型相關資訊。

## 權重檔說明  (Model Weights)

- **best.pt**：驗證集表現最佳的模型權重。
- **last.pt**：最後一個 Epoch 儲存的模型權重。

### auto_5fold.py  (5-Fold Cross Validation Training Script)
五折交叉驗證（5-Fold Cross Validation）主程式，負責自動建立各折資料集、產生對應的 `data.yaml` 設定檔，並執行 YOLOv11s 模型訓練，最後儲存各折的訓練結果與模型權重。

### data.yaml  (Dataset Configuration File)
各折（Fold1～Fold5）的資料集設定檔，記錄 YOLOv11s 訓練所需的資料路徑、類別數（nc）及類別名稱（names）。

## 訓練參數 (Training Parameters)

本專案採用 YOLOv11n 預訓練模型進行五折交叉驗證（5-Fold Cross Validation），模型訓練參數如下：

| 參數 | 設定值 | 說明 |
|------|--------|------|
| Model | YOLOv11n (`yolo11s.pt`) | 使用 Ultralytics YOLOv11s 預訓練模型 |
| Epochs | 100 | 每一折訓練 100 個 Epoch |
| Batch Size | 8 | 每次訓練載入 8 張影像 |
| Image Size | 640 × 640 | 輸入影像尺寸 |
| Workers | 2 | Windows 系統資料載入執行緒數 |
| Device | GPU（device=0） | 使用 NVIDIA GPU 進行訓練 |
| Cache | True | 將資料集快取至記憶體，加快訓練速度 |
| AMP | True | 啟用 Automatic Mixed Precision，加速訓練並降低 GPU 記憶體使用量 |
| Initial Learning Rate | 0.005 | 初始學習率（lr0） |
| Cross Validation | 5-Fold | 採用五折交叉驗證 |
| Random Seed | 42 | 固定資料切分結果，提升實驗可重現性 |

### 參數設定說明 (Parameter Configuration)

本專案延續前一版本之訓練參數設定，包含 YOLOv11s 模型、Epoch、Batch Size、Image Size、Learning Rate、Workers、Cache 及 AMP 等參數皆維持一致，並將資料集由每類別 100 張影像擴增至每類別 150 張影像（總計由 300 張增加至 450 張）。本次優化主要著重於資料集擴充，以提升資料多樣性及模型泛化能力，為確保實驗比較具有一致性，其餘訓練參數皆維持與前一版本相同。

採用相同的訓練參數可有效控制實驗變因（Control Variables），避免因超參數調整而影響模型效能，使實驗結果主要反映資料集擴充對模型辨識能力之影響，進而更客觀地分析模型於 Precision、Recall、mAP@50 及 mAP@50-95 等評估指標之改善情形。

## 研究目標 (Research Objectives)

本專案以優化後的 YOLOv11s 物件偵測模型為核心，建置熊（Bear）、貓（Cat）及狗（Dog）三類動物物件偵測系統，並採用五折交叉驗證（5-Fold Cross Validation）進行模型訓練與效能評估，以提升模型的穩定性、辨識能力及泛化能力。

本專案期望透過增加資料集樣本數、模型參數調整及訓練流程優化，使模型達成以下效能目標：

- Precision（精確率）達 **90% 以上**
- Recall（召回率）達 **90% 以上**
- mAP@50 達 **90% 以上**
- mAP@50-95 達 **80% 以上**

 ## YOLOv11s 五折交叉驗證結果 (YOLOv11s 5-Fold Cross Validation Results)

本專案採用五折交叉驗證（5-Fold Cross Validation）評估 YOLOv11s 模型於 Bear、Cat 與 Dog 三類動物物件偵測之效能。各折訓練完成後，以 Precision、Recall、mAP@50 及 mAP@50-95 作為模型效能評估指標，結果如下：

| Fold | Precision | Recall | mAP@50 | mAP@50-95 |
|------|----------:|--------:|--------:|----------:|
| Fold 1 | 0.830 | 0.830 | 0.890 | 0.650 |
| Fold 2 | 0.880 | 0.870 | 0.910 | 0.710 |
| Fold 3 | 0.910 | 0.900 | 0.910 | 0.690 |
| Fold 4 | 0.890 | 0.920 | 0.930 | 0.680 |
| Fold 5 | 0.880 | 0.830 | 0.910 | 0.700 |
| **Average** | **0.878** | **0.870** | **0.910** | **0.686** |

### 結果分析 (Performance Analysis)

由五折交叉驗證結果可知，YOLOv11s 模型於五次訓練中皆展現穩定的物件偵測效能。其中，第 4 折（Fold 4）於 Precision、Recall 及 mAP@50 表現最佳，分別達 **0.890**、**0.920** 及 **0.930**；而 mAP@50-95 則以第 2 折（Fold 2）表現最佳，達 **0.710**。

整體平均效能為 Precision **0.878**、Recall **0.870**、mAP@50 **0.910** 及 mAP@50-95 **0.686**，顯示在資料集擴充至每類別 150 張影像（總計 450 張）後，YOLOv11s 模型能維持良好的辨識能力與穩定性，並在 mAP@50 指標突破 **91.0%**，可作為後續模型比較、效能分析及研究成果展示之依據。

### 研究目標達成情形 (Research Objective Evaluation)

本專案以 Precision、Recall、mAP@50 及 mAP@50-95 作為模型效能評估指標，並與研究目標進行比較，結果如下：

| 評估指標 | 研究目標 | 實際結果 | 達成情形 |
|----------|---------:|---------:|:--------:|
| Precision | ≥ 0.90 | 0.878 | 未達成 ✗ |
| Recall | ≥ 0.90 | 0.870 | 未達成 ✗ |
| mAP@50 | ≥ 0.90 | 0.910 | 已達成 ✓ |
| mAP@50-95 | ≥ 0.80 | 0.686 | 未達成 ✗ |

由實驗結果可知，本專案成功達成 **mAP@50 達 90% 以上**之研究目標，顯示模型具有良好的物件偵測能力。然而，Precision、Recall 及 mAP@50-95 尚未達到預期目標，顯示模型在定位精度及整體辨識效能方面仍有提升空間。未來可透過持續擴充資料集、增加資料多樣性、調整模型超參數或進行資料增強（Data Augmentation）等方式，進一步提升模型效能。
