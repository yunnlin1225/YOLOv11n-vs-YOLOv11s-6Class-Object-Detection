## 專案介紹 (Project Introduction)

本專案為六類別物件偵測研究之一部分，以 **Ultralytics YOLOv11s** 物件偵測模型為核心，建立包含熊（Bear）、貓（Cat）、狗（Dog）、人（Person）、猴子（Monkey）及豬（Pig）之六類別物件偵測模型。

本研究延續 YOLOv11n 六類別模型實驗，使用相同資料集與五折交叉驗證（5-Fold Cross Validation）方式進行 YOLOv11s 模型訓練與效能評估。為進一步提升模型整體表現，本研究針對 **Classification Loss Weight（`cls`）與 Bounding Box Loss Weight（`box`）** 進行多組超參數測試，並依據研究設定之效能目標選擇適合的參數組合。

研究主要目的為比較 **YOLOv11n 與 YOLOv11s** 在相同六類別資料集下的 Precision、Recall、mAP50 及 mAP50-95 表現，分析不同模型規模於目標辨識與 Bounding Box 定位能力上的差異，並作為後續最終模型選擇與實際部署之依據。


## 研究目標 (Research Objectives)

本研究旨在建立 **YOLOv11s 六類別目標偵測模型**，並透過五折交叉驗證與超參數調整，評估模型的偵測效能、穩定性及泛化能力。

主要研究目標如下：

1. **建立六類別目標偵測模型**

   * 使用 YOLOv11s 進行狗（dog）、貓（cat）、熊（bear）、豬（pig）、猴（monkey）及人（person）六類目標的訓練與偵測。

2. **提升模型整體偵測效能**

   * Precision（P）達到 **90% 以上**。
   * Recall（R）達到 **90% 以上**。
   * mAP50 達到 **90% 以上**。
   * mAP50-95 達到 **75% 以上**。

3. **驗證模型穩定性與泛化能力**

   * 採用 **5-Fold Cross Validation** 進行模型訓練與驗證。
   * 比較各 Fold 的 Precision、Recall、mAP50 及 mAP50-95，評估模型於不同資料切分下的穩定性。

4. **進行模型超參數調整**

   * 針對 Classification Loss Weight（`cls`）與 Bounding Box Loss Weight（`box`）進行多組參數測試。
   * 比較不同參數組合對 Precision、Recall、mAP50 及 mAP50-95 的影響。
   * 以四項研究效能目標作為參數選擇依據，選擇能兼顧分類與定位表現的參數組合。

5. **作為 YOLOv11n 與 YOLOv11s 比較依據**

   * 比較兩種模型在相同六類別資料集下的偵測效能，作為後續最終模型選擇與部署之依據。

  
## 類別編號 (Class ID)

本專案共包含 **6 個目標偵測類別**，YOLOv11s 使用與 YOLOv11n 相同的類別編號與資料標註設定。

| Class ID | 類別名稱 | Class Name |
| -------: | ---- | ---------- |
|        0 | 狗    | dog        |
|        1 | 貓    | cat        |
|        2 | 熊    | bear       |
|        3 | 豬    | pig        |
|        4 | 猴    | monkey     |
|        5 | 人    | person     |

## 資料集資訊 (Dataset Information)

本研究使用包含 **900 張影像**的六類別目標偵測資料集，涵蓋狗（dog）、貓（cat）、熊（bear）、豬（pig）、猴（monkey）及人（person）六個目標類別，每個類別皆包含 **150 張影像**，以維持資料類別分布的均衡。

### 資料集規模

| 類別     | Class ID | 英文名稱          |    影像數量 |
| ------ | -------: | ------------- | ------: |
| 狗      |        0 | dog           |     150 |
| 貓      |        1 | cat           |     150 |
| 熊      |        2 | bear          |     150 |
| 豬      |        3 | pig           |     150 |
| 猴      |        4 | monkey        |     150 |
| 人      |        5 | person        |     150 |
| **總計** |  **0–5** | **6 Classes** | **900** |

### 資料標註

本研究使用 **Label Studio** 進行影像 Bounding Box 標註，並採用 YOLO 格式儲存標註資料：

`Class ID  Center X  Center Y  Width  Height`

其中座標與 Bounding Box 尺寸皆經過正規化（Normalization），數值範圍介於 **0～1**。

YOLOv11s 使用與 YOLOv11n **相同的資料集、類別定義及標註資料**，以維持後續模型效能比較的一致性與公平性。

## 訓練參數 (Training Parameters)

本研究採用 **YOLOv11s** 作為六類別目標偵測模型，並使用五折交叉驗證進行訓練。五個 Fold 均採用相同訓練條件，以確保實驗結果具有一致性與可比較性。

其中，Classification Loss（`cls`）與 Box Loss（`box`）經多組超參數測試後，最終選定 **`cls=0.745`、`box=8.0`** 作為模型訓練設定。

### 模型與訓練設定

| 訓練參數                        |          設定值 | 說明                                |
| --------------------------- | -----------: | --------------------------------- |
| Model                       |     YOLOv11s | YOLOv11 Small 模型                  |
| Pretrained Weights          | `yolo11s.pt` | 使用預訓練權重初始化                        |
| Classes                     |            6 | dog、cat、bear、pig、monkey、person    |
| Epochs                      |          100 | 每個 Fold 最大訓練 100 Epochs           |
| Batch Size                  |            8 | 每次迭代處理 8 張影像                      |
| Image Size                  |    640 × 640 | 模型輸入影像尺寸                          |
| Device                      |        GPU 0 | 使用 NVIDIA GPU 訓練                  |
| Workers                     |            2 | 資料載入程序數                           |
| Cache                       |        False | 不使用資料快取                           |
| AMP                         |         True | 啟用 Automatic Mixed Precision      |
| Optimizer                   |         Auto | 由 Ultralytics 自動選擇最佳化器            |
| Cosine LR                   |         True | 使用 Cosine Learning Rate Scheduler |
| Close Mosaic                |           10 | 最後 10 個 Epoch 關閉 Mosaic           |
| Classification Loss (`cls`) |    **0.745** | 經超參數測試後選定的分類損失權重                  |
| Box Loss (`box`)            |      **8.0** | 經超參數測試後選定的定位損失權重                  |
| Cross Validation            |       5-Fold | 五折交叉驗證                            |
| Random State                |           42 | 固定資料切分隨機種子                        |


### 五折交叉驗證設定

本研究使用 `KFold` 進行五折交叉驗證，設定：

* `n_splits = 5`
* `shuffle = True`
* `random_state = 42`

每個 Fold 約使用 **80% 資料作為訓練集、20% 作為驗證集**，並分別進行獨立模型訓練。

### 訓練策略

本研究使用 YOLOv11s 預訓練權重 `yolo11s.pt` 進行模型初始化，並啟用 **Automatic Mixed Precision（AMP）** 與 **Cosine Learning Rate Scheduler**，同時於最後 10 個 Epoch 關閉 Mosaic Data Augmentation。

經多組超參數測試後，本研究最終將 **Classification Loss Weight（`cls`）設定為 0.745**，並將 **Bounding Box Loss Weight（`box`）設定為 8.0**，作為 YOLOv11s 五折交叉驗證的最終訓練設定。詳細超參數測試結果與選擇依據將於後續「超參數調整」章節說明。

每完成一個 Fold 的訓練後，程式會釋放模型與 GPU 記憶體，並等待 **60 秒**後再進行下一個 Fold，以降低長時間連續訓練造成的硬體負載。


## 資料夾說明 (Folder Structure)

本專案依照資料集建立、五折交叉驗證、模型訓練、評估與推論流程進行檔案配置，以維持實驗流程的清晰性與可重現性。

```text
YOLOv11s_6Categories/
│
├── auto_5fold.py
├── eval_conf.py
├── inference.py
│
├── fold1/
├── fold1.yaml
├── fold2/
├── fold2.yaml
├── fold3/
├── fold3.yaml
├── fold4/
├── fold4.yaml
├── fold5/
├── fold5.yaml
│
├── YOLOv11_6Class_5Fold/
├── yolo11s.pt
│
└── README.md
```

### 主要檔案說明

| 檔案 / 資料夾                    | 功能說明                             |
| --------------------------- | -------------------------------- |
| `auto_5fold.py`             | 建立五折資料集並執行 YOLOv11s 五折交叉驗證訓練     |
| `eval_conf.py`              | 評估 Confidence Threshold 設定下的模型效能 |
| `inference.py`              | 使用訓練完成的模型進行影像推論                  |
| `fold1/` ～ `fold5/`         | 五折交叉驗證所建立的獨立資料集                  |
| `fold1.yaml` ～ `fold5.yaml` | 各 Fold 對應的 YOLO 資料集設定檔           |
| `YOLOv11_6Class_5Fold/`     | 儲存各 Fold 的訓練結果與模型權重              |
| `yolo11s.pt`                | YOLOv11s 預訓練模型權重                 |
| `README.md`                 | 專案研究方法、實驗結果與分析說明                 |

### 五折資料結構

每個 Fold 均包含獨立的訓練集與驗證集：

```text
fold1/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`fold2/` 至 `fold5/` 採用相同資料結構，並透過對應的 `.yaml` 設定檔提供模型訓練所需的資料路徑與六類別資訊。

## 權重檔說明 (Model Weights)

本專案以 **YOLOv11s 預訓練權重 `yolo11s.pt`** 作為模型初始化基礎，並透過五折交叉驗證分別進行模型訓練。各 Fold 訓練完成後，由 Ultralytics YOLO 自動保存對應的模型權重。

| 權重檔       | 說明                                      |
| --------- | --------------------------------------- |
| `best.pt` | 該 Fold 驗證過程中表現最佳的模型權重，主要用於後續模型評估與推論     |
| `last.pt` | 該 Fold 最後一個 Epoch 所保存的模型權重，可用於訓練狀態保存與續訓 |

### 五折模型權重

```text
Fold_1/
└── weights/
    ├── best.pt
    └── last.pt

Fold_2/
└── weights/
    ├── best.pt
    └── last.pt

Fold_3/
└── weights/
    ├── best.pt
    └── last.pt

Fold_4/
└── weights/
    ├── best.pt
    └── last.pt

Fold_5/
└── weights/
    ├── best.pt
    └── last.pt
```

各 Fold 的 **`best.pt`** 作為主要評估權重，用於後續模型效能分析、Confidence Threshold 評估及影像推論。

## YOLOv11s 五折交叉驗證結果 (YOLOv11s 5-Fold Cross Validation Results)

本研究經多組超參數測試後，最終選定 **`cls=0.745`、`box=8.0`** 作為 YOLOv11s 模型訓練設定，並以此設定進行 **5-Fold Cross Validation**，評估模型於不同資料切分下的偵測效能與穩定性。

### 五折交叉驗證結果

| Fold          | Precision (P) | Recall (R) |      mAP50 |   mAP50-95 |
| ------------- | ------------: | ---------: | ---------: | ---------: |
| Fold 1        |        0.9100 |     0.8740 |     0.9370 |     0.7850 |
| Fold 2        |        0.9030 |     0.9100 |     0.9400 |     0.7390 |
| Fold 3        |        0.8840 |     0.9270 |     0.9290 |     0.8050 |
| Fold 4        |        0.9220 |     0.8850 |     0.9160 |     0.7520 |
| Fold 5        |        0.8850 |     0.9130 |     0.9430 |     0.7980 |
| **5-Fold 平均** |    **0.9008** | **0.9018** | **0.9330** | **0.7758** |

五折平均結果顯示，YOLOv11s 的 **Precision 為 90.08%、Recall 為 90.18%、mAP50 為 93.30%、mAP50-95 為 77.58%**，四項指標皆達成本研究設定之效能目標。

整體結果顯示，最終參數設定下的 YOLOv11s 具備良好的六類別目標偵測能力，各組超參數測試結果與最終參數選擇依據，將於下一章「超參數調整」進一步說明。

## 超參數調整 (Hyperparameter Tuning)

為提升 YOLOv11s 的整體偵測效能，本研究針對 Classification Loss Weight（`cls`）與 Bounding Box Loss Weight（`box`）進行超參數調整，共測試 **4 組參數組合**，並以 Precision、Recall、mAP50 及 mAP50-95 作為評估依據。

### 超參數測試結果

|     `cls` |   `box` |  Precision |     Recall |      mAP50 |   mAP50-95 |
| --------: | ------: | ---------: | ---------: | ---------: | ---------: |
|     0.740 |     7.5 |     0.9000 |     0.8898 |     0.9286 |     0.7706 |
|     0.750 |     7.5 |     0.8988 | **0.9150** |     0.9320 | **0.7760** |
|     0.745 |     7.5 | **0.9232** |     0.8852 | **0.9388** |     0.7744 |
| **0.745** | **8.0** | **0.9008** | **0.9018** | **0.9330** | **0.7758** |

### 調整結果

前三組參數雖在部分指標具有較高表現，但 Precision 與 Recall 未能同時達到 **90%**。因此，本研究持續調整 `cls` 與 `box` 的損失權重，以改善分類與 Bounding Box 定位之間的效能平衡。

當參數設定為 **`cls=0.745`、`box=8.0`** 時，Precision 為 **90.08%**、Recall 為 **90.18%**、mAP50 為 **93.30%**、mAP50-95 為 **77.58%**，四項指標皆達成本研究設定之效能目標。

因此，本研究最終採用 **`cls=0.745`、`box=8.0`** 作為 YOLOv11s 的訓練參數設定，並用於後續模型評估、推論與最終模型建立。

> **超參數測試結果顯示，最終設定並非單純追求單一指標最高值，而是選擇能同時達成 Precision、Recall、mAP50 與 mAP50-95 四項研究目標的參數組合。**

## 效能分析 (Performance Analysis)

YOLOv11s 在最終參數設定 **`cls=0.745`、`box=8.0`** 下，五折平均 Precision 為 **90.08%**、Recall 為 **90.18%**、mAP50 為 **93.30%**、mAP50-95 為 **77.58%**，四項指標皆達成本研究設定之效能目標。

### 研究目標達成情況

| 評估指標      |  研究目標 |       五折平均 |  結果 |
| --------- | ----: | ---------: | :-: |
| Precision | ≥ 90% | **90.08%** |  ✅  |
| Recall    | ≥ 90% | **90.18%** |  ✅  |
| mAP50     | ≥ 90% | **93.30%** |  ✅  |
| mAP50-95  | ≥ 75% | **77.58%** |  ✅  |

其中，Fold 4 具有最高 Precision（92.20%），Fold 3 具有最高 Recall（92.70%）與 mAP50-95（80.50%），Fold 5 則具有最高 mAP50（94.30%）。

整體而言，YOLOv11s 在不同資料切分下皆維持良好的偵測表現。雖然各 Fold 間存在一定程度的效能差異，但五折平均結果均達成研究目標，顯示最終參數設定具有良好的六類別目標偵測能力。

> **綜合五折交叉驗證結果，經超參數調整後的 YOLOv11s 已達成 Precision、Recall、mAP50 及 mAP50-95 四項研究效能目標。**


# Confidence Threshold 調整

為進一步評估信心度門檻（Confidence Threshold）對 YOLOv11s 模型偵測效能的影響，本研究在完成模型超參數調整後，進一步針對 Confidence Threshold 進行測試。

原模型使用 **Confidence Threshold = 0.25**，並將門檻提高至 **0.26** 進行五折交叉驗證。實驗過程中其餘驗證條件保持一致，其中影像輸入尺寸（imgsz）維持 **640**，NMS IoU Threshold 維持 **0.60**，以確保不同 Confidence Threshold 間具有一致的比較基準。

## Confidence Threshold = 0.26 五折驗證結果

| 折數（Fold）       | Precision (P) | Recall (R) |      mAP50 |   mAP50-95 |
| -------------- | ------------: | ---------: | ---------: | ---------: |
| Fold 1         |        0.9133 |     0.8740 |     0.8824 |     0.7388 |
| Fold 2         |        0.9033 |     0.9111 |     0.9061 |     0.7103 |
| Fold 3         |        0.9111 |     0.9039 |     0.9016 |     0.7774 |
| Fold 4         |        0.9215 |     0.8860 |     0.8792 |     0.7280 |
| Fold 5         |        0.8884 |     0.9075 |     0.8924 |     0.7634 |
| **五折平均（Mean）** |    **0.9075** | **0.8965** | **0.8923** | **0.7436** |

## Confidence Threshold 效能比較

將原本的 **Confidence Threshold = 0.25** 與調整後的 **Confidence Threshold = 0.26** 進行比較：

| Confidence Threshold | Precision (P) | Recall (R) |      mAP50 |   mAP50-95 |
| -------------------: | ------------: | ---------: | ---------: | ---------: |
|             **0.25** |    **0.9008** | **0.9018** | **0.9330** | **0.7758** |
|                 0.26 |        0.9075 |     0.8965 |     0.8923 |     0.7436 |

## Confidence Threshold 效能分析

當 Confidence Threshold 由 **0.25 提升至 0.26** 後，Precision 由 **0.9008 提升至 0.9075**，增加 **0.0067**，顯示提高信心度門檻可以降低部分低信心度預測，使模型的預測結果更加保守，進而降低誤判情況。

然而，Recall 則由 **0.9018 降低至 0.8965**，下降 **0.0053**。這表示隨著 Confidence Threshold 提高，部分原本可以被正確偵測的目標可能因信心度低於門檻而遭到排除，因此造成漏檢增加。

在整體偵測效能方面，mAP50 由 **0.9330 降低至 0.8923**，下降 **0.0407**；mAP50-95 則由 **0.7758 降低至 0.7436**，下降 **0.0322**。

本研究所設定的模型效能目標為：

* Precision ≥ 0.90
* Recall ≥ 0.90
* mAP50 ≥ 0.90
* mAP50-95 ≥ 0.75

在 **Confidence Threshold = 0.25** 時，四項指標皆能同時達成研究設定目標。

相較之下，當 Confidence Threshold 提升至 **0.26** 時，雖然 Precision 提升至 **0.9075**，但 Recall 降低至 **0.8965**、mAP50 降低至 **0.8923**、mAP50-95 降低至 **0.7436**，其中三項指標未達研究設定標準。

因此，提高 Confidence Threshold 至 0.26 雖然能進一步提升 Precision，但會造成 Recall 與整體 mAP 指標下降。綜合 Precision、Recall、mAP50 與 mAP50-95 四項指標後，**Confidence Threshold = 0.25 能取得較佳的整體偵測效能與指標平衡，因此維持 0.25 作為 YOLOv11s 模型的 Confidence Threshold。**


