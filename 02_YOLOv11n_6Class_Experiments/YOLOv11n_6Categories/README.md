## 專案介紹 (Project Introduction)

本專案採用 **YOLOv11n** 建立六類別目標偵測模型，偵測類別包含狗（dog）、貓（cat）、熊（bear）、豬（pig）、猴（monkey）及人（person）。

YOLOv11n 為較輕量化的模型版本，本研究透過 **5-Fold Cross Validation** 進行模型訓練與驗證，並以 Precision、Recall、mAP50 及 mAP50-95 作為主要效能評估指標，以分析模型於不同資料切分下的偵測表現與穩定性。

經初始模型五折交叉驗證後，部分評估指標未達本研究設定之效能目標，因此本階段未進一步進行 Confidence Threshold 調整、測試影像推論與最終部署模型建立，而是依據實驗結果進一步進行模型效能優化。

待模型達成研究效能目標後，再進行 Confidence Threshold 評估、實際影像推論及最終部署模型建立。

## 資料集與類別 (Dataset & Classes)

本研究使用包含 **900 張影像**的六類別目標偵測資料集，共包含狗（dog）、貓（cat）、熊（bear）、豬（pig）、猴（monkey）及人（person）六個類別，每個類別各包含 **150 張影像**。

| Class ID | 類別 | Class Name | 影像數量 |
|---:|---|---|---:|
| 0 | 狗 | dog | 150 |
| 1 | 貓 | cat | 150 |
| 2 | 熊 | bear | 150 |
| 3 | 豬 | pig | 150 |
| 4 | 猴 | monkey | 150 |
| 5 | 人 | person | 150 |
| **總計** | **6 類別** | **6 Classes** | **900** |

影像資料採用 YOLO Bounding Box 格式進行標註，並透過 **5-Fold Cross Validation** 將資料分為不同的訓練集與驗證集，以進行模型效能評估。

## 訓練參數 (Training Parameters)

本研究首先採用 **YOLOv11n** 建立六類別目標偵測模型，並透過 **5-Fold Cross Validation** 進行模型訓練與效能評估。五個 Fold 均使用相同訓練設定，以確保實驗結果具有一致性與可比較性。

此階段為 YOLOv11n 的初始模型實驗，主要用於評估模型在現有訓練設定下的基本偵測效能。完成五折交叉驗證後，再依據實驗結果分析未達研究目標之指標，作為後續模型調整與效能優化之依據。

### 模型與訓練設定

| 訓練參數                          |          設定值 | 說明                                |
| ----------------------------- | -----------: | --------------------------------- |
| Model                         |     YOLOv11n | YOLOv11 Nano 輕量化模型                |
| Pretrained Weights            | `yolo11n.pt` | 使用預訓練權重初始化                        |
| Classes                       |            6 | dog、cat、bear、pig、monkey、person    |
| Epochs                        |          100 | 每個 Fold 最大訓練 100 Epochs           |
| Batch Size                    |            8 | 每次迭代處理 8 張影像                      |
| Image Size                    |    640 × 640 | 模型輸入影像尺寸                          |
| Device                        |        GPU 0 | 使用 NVIDIA GPU 進行訓練                |
| Workers                       |            2 | 資料載入程序數                           |
| Cache                         |        False | 不使用資料快取                           |
| AMP                           |         True | 啟用 Automatic Mixed Precision      |
| Initial Learning Rate (`lr0`) |        0.005 | 初始學習率                             |
| Cosine LR                     |         True | 使用 Cosine Learning Rate Scheduler |
| Close Mosaic                  |           10 | 最後 10 個 Epoch 關閉 Mosaic           |
| Cross Validation              |       5-Fold | 採用五折交叉驗證                          |
| Random State                  |           42 | 固定資料切分隨機種子                        |

### 訓練策略

模型以 `yolo11n.pt` 預訓練權重進行初始化，並使用固定的五折資料切分進行獨立訓練。訓練過程啟用 AMP 與 Cosine Learning Rate Scheduler，並於最後 10 個 Epoch 關閉 Mosaic Data Augmentation。

每完成一個 Fold 後，程式會釋放模型與 GPU 記憶體，並等待 **60 秒**後再進行下一 Fold，以降低長時間連續訓練造成的硬體負載。

## YOLOv11n 五折交叉驗證結果 (YOLOv11n 5-Fold Cross Validation Results)

為評估 YOLOv11n 初始模型於六類別目標偵測任務中的效能，本研究採用 **5-Fold Cross Validation** 進行模型訓練與驗證，並以 Precision、Recall、mAP50 及 mAP50-95 作為主要評估指標。

### 五折交叉驗證結果

| Fold | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Fold 1 | 0.901 | 0.861 | 0.925 | 0.770 |
| Fold 2 | 0.956 | 0.887 | 0.957 | 0.786 |
| Fold 3 | 0.898 | 0.935 | 0.954 | 0.818 |
| Fold 4 | 0.908 | 0.866 | 0.899 | 0.730 |
| Fold 5 | 0.882 | 0.932 | 0.938 | 0.787 |
| **5-Fold 平均** | **0.909** | **0.896** | **0.935** | **0.778** |

五折平均結果顯示，YOLOv11n 初始模型的 Precision 為 **90.9%**、Recall 為 **89.6%**、mAP50 為 **93.5%**、mAP50-95 為 **77.8%**。

此結果作為 YOLOv11n 初始模型之效能評估依據，後續將進一步分析各項指標與研究目標之差異，作為模型效能改善與後續調整之依據。

## 效能分析 (Performance Analysis)

YOLOv11n 初始模型五折平均結果如下：

| 評估指標 | 5-Fold 平均 | 研究目標 | 結果 |
|---|---:|---:|:---:|
| Precision | 90.9% | ≥ 90% | ✅ 達標 |
| Recall | 89.6% | ≥ 90% | ❌ 未達標 |
| mAP50 | 93.5% | ≥ 90% | ✅ 達標 |
| mAP50-95 | 77.8% | ≥ 75% | ✅ 達標 |

結果顯示，YOLOv11n 初始模型的 **Precision、mAP50 與 mAP50-95 均已達成本研究設定之效能目標**，顯示模型在分類準確度、整體偵測效能及 Bounding Box 定位能力方面已有良好表現。

然而，Recall 五折平均為 **89.6%**，尚未達到本研究設定之 **90% 目標**，代表模型在目標檢出能力方面仍具有改善空間，可能存在部分目標未被成功偵測的情況。

因此，本研究未直接進入 Confidence Threshold 評估、測試影像推論與最終部署模型建立階段，而是先依據初始模型的實驗結果進一步進行模型效能改善，使四項主要評估指標皆能達成研究目標後，再進行後續實驗。

## 結論 (Conclusion)

YOLOv11n 初始模型經五折交叉驗證後，平均 Precision 為 **90.9%**、Recall 為 **89.6%**、mAP50 為 **93.5%**、mAP50-95 為 **77.8%**。

其中 Precision、mAP50 與 mAP50-95 已達本研究設定之效能目標，但 Recall 尚未達到 **90%** 的設定標準，顯示模型在目標檢出能力方面仍具有進一步改善空間。

因此，本階段暫不進行 Confidence Threshold 調整、測試影像推論與最終部署模型建立，而是先依據本次五折交叉驗證結果進行模型效能改善。

> **待模型四項主要評估指標皆達成研究目標後，再進一步進行 Confidence Threshold 評估、實際影像推論及最終部署模型建立。**
