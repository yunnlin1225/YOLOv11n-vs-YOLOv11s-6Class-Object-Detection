# YOLOv11n 優化模型影片測試
(YOLOv11n Optimized Video Test)

## 影片測試目的 (Video Test Objective)

本階段使用優化後的 **YOLOv11n 六類別目標偵測模型**進行實際影片推論測試，進一步觀察模型於連續動態影像中的實際偵測表現。

相較於單張影像推論，影片測試可進一步觀察模型在連續影格中的**目標辨識、偵測穩定性、誤檢、漏檢及 Bounding Box 定位情況**，以評估模型面對實際動態場景時的表現。

此外，本階段亦透過不同推論設定進行測試與比較，觀察 **Confidence Threshold、IoU Threshold 與 Image Size (`imgsz`)** 對影片偵測結果的影響。

由於 YOLOv11n 屬於較輕量化的模型，本階段亦將其影片推論結果作為後續實際部署與 YOLOv11s 比較的重要依據。

> **影片測試結果將作為 YOLOv11n 後續推論設定選擇，以及與 YOLOv11s 比較與最終部署評估之參考。**

## 測試模型與權重 (Test Model & Weights)

本階段影片測試採用 **YOLOv11n 優化模型 Fold 3** 所產生的最佳權重檔 `best.pt` 進行影片推論。

| 項目 | 設定 |
| --- | --- |
| Model | YOLOv11n |
| Classes | 6 |
| Model Version | Optimized |
| Selected Fold | **Fold 3** |
| Weight | `best.pt` |
| Test Type | Video Inference |

### Fold 3 選擇依據

五折交叉驗證的主要目的為評估模型於不同資料切分下的穩定性與泛化能力，而實際影片推論則需選定單一模型權重進行測試。

因此，本階段由五個 Fold 中選擇 **Fold 3 的 `best.pt`** 作為 YOLOv11n 優化模型的影片測試權重，並透過實際影片觀察模型於動態場景中的偵測表現。

Fold 3 權重主要用於後續影片推論、推論參數比較及實際偵測效果評估，五折平均結果仍作為 YOLOv11n 整體模型效能的主要研究評估依據。

> **本階段使用的 Fold 3 `best.pt` 為影片測試權重，並非後續使用完整六類別資料集重新訓練建立的最終部署模型。**

## 影片測試參數 (Video Test Parameters)

本階段使用 YOLOv11n 優化模型 **Fold 3 `best.pt`** 進行影片推論，並以相同測試影片比較三組不同 Image Size 設定，觀察輸入影像尺寸對實際影片偵測結果的影響。

### 測試設定

| Test | Confidence | Image Size | IoU | 說明 |
| --- | ---: | ---: | ---: | --- |
| Test 1 | **0.24** | **896** | **0.70** | 較低 Image Size |
| Test 2 | **0.24** | **960** | **0.70** | 提高 Image Size |
| Test 3 | **0.24** | **1024** | **0.70** | 最高 Image Size |

三組測試皆使用相同的 **YOLOv11n Fold 3 `best.pt`** 與相同測試影片，並固定 Confidence Threshold 為 **0.24**、IoU Threshold 為 **0.70**，僅調整 Image Size (`imgsz`)。

此測試方式可在其他推論條件維持一致的情況下，較直接觀察不同輸入影像尺寸對模型實際影片偵測表現的影響。

### Confidence Threshold 設定依據

根據 YOLOv11n 優化模型 **Fold 3 的 F1-Confidence Curve**，模型的 F1 Score 在 Confidence 約 **0.24** 附近具有較佳表現，因此本階段將 **conf=0.24** 固定作為三組影片測試的 Confidence Threshold。

透過固定 Confidence Threshold，可避免不同 Confidence 設定同時影響實驗結果，使本階段主要聚焦於 Image Size 的比較。

### Image Size 調整

本階段分別測試：

- `imgsz=896`
- `imgsz=960`
- `imgsz=1024`

透過逐步提高輸入影像尺寸，觀察 YOLOv11n 在目標辨識、誤檢、漏檢、Bounding Box 定位及連續影格偵測穩定性上的變化。

同時亦可評估提高 Image Size 是否確實能改善實際影片偵測效果，作為後續推論設定與最終部署模型參數選擇的參考。

> **本階段固定 `conf=0.24` 與 `iou=0.70`，主要比較 `imgsz=896、960、1024` 三種設定對 YOLOv11n 實際影片偵測表現的影響。**

## 影片測試重點 (Video Test Focus)

本階段透過實際影片進行 YOLOv11n 優化模型推論，並固定 **Confidence Threshold = 0.24** 與 **IoU Threshold = 0.70**，主要比較不同 Image Size 對影片偵測結果的影響。

影片測試主要觀察以下項目：

1. **目標辨識能力**
   - 觀察模型是否能正確辨識 Dog、Cat、Bear、Pig、Monkey 及 Person 六個類別。
   - 確認不同 Image Size 下是否出現類別辨識錯誤。

2. **連續影格偵測穩定性**
   - 觀察目標於影片連續影格中是否能維持穩定偵測。
   - 比較 Bounding Box 是否出現頻繁消失、重新出現或跳動。

3. **誤檢與漏檢**
   - 觀察模型是否將其他物件錯誤辨識為目標類別。
   - 比較 `imgsz=896、960、1024` 下的誤檢與漏檢情況。

4. **Bounding Box 定位表現**
   - 觀察不同 Image Size 下偵測框的位置與穩定性。
   - 評估提高輸入影像尺寸是否改善目標定位表現。

5. **Image Size 影響**
   - 比較 `imgsz=896、960、1024` 三組設定。
   - 觀察提高輸入影像尺寸是否能實際改善 YOLOv11n 的影片偵測效果。

6. **部署參考**
   - 綜合偵測效果與不同 Image Size 的實際表現，選擇較適合 YOLOv11n 後續影片推論的設定。
   - 測試結果將作為後續完整六類別資料重新訓練及最終部署模型建立的參考。
  
## 影片測試結果 (Video Test Results)

本研究使用 **3 部不同測試影片**進行 YOLOv11n 優化模型的實際影片推論，並分別使用三組不同 Image Size 進行測試，共完成 **9 組影片推論實驗**。

三組測試皆固定使用 **Confidence Threshold = 0.24**、**IoU Threshold = 0.70**，並分別比較 `imgsz=896`、`960` 與 `1024` 對實際影片偵測結果的影響。

經三組設定於實際影片中的比較後，本研究選擇 **Image Size = 1024** 作為 YOLOv11n 本階段的主要影片推論設定。

### 最終影片推論設定

| 項目 | 設定 |
| --- | --- |
| Model | YOLOv11n |
| Model Version | Optimized |
| Weight | Fold 3 `best.pt` |
| Confidence Threshold | **0.24** |
| IoU Threshold | **0.70** |
| Image Size | **1024** |

### 代表性影片測試

為避免同時展示 9 組相似的影片結果造成內容過於冗長，本說明檔主要展示最終選定參數 **conf=0.24、iou=0.70、imgsz=1024** 下的 3 部不同測試影片。

三部影片皆使用相同模型權重與推論參數進行測試，用以觀察 YOLOv11n 優化模型於不同動態影像與場景中的實際偵測表現。

### Video 1

https://github.com/user-attachments/assets/7e5dcfa0-d234-436d-a2c5-9a88d203fa3d

### Video 2

https://github.com/user-attachments/assets/05bebdec-4409-48ca-8c7b-212488936553

### Video 3

https://github.com/user-attachments/assets/fe039bca-6c61-4ecd-85a0-166641e96865

> **其餘 Image Size 設定主要作為推論參數比較實驗，本說明檔則以 `imgsz=1024` 的三部代表性影片作為主要展示結果。**

## 不同推論設定比較 (Inference Setting Comparison)

本階段針對 YOLOv11n 優化模型進行三組影片推論參數測試，並固定 Confidence Threshold 與 IoU Threshold，主要比較不同 Image Size 對實際影片偵測結果的影響。

| Test | Confidence | Image Size | IoU | 調整重點 |
| --- | ---: | ---: | ---: | --- |
| Test 1 | 0.24 | 896 | 0.70 | 較低輸入影像尺寸 |
| Test 2 | 0.24 | 960 | 0.70 | 提高輸入影像尺寸 |
| **Test 3** | **0.24** | **1024** | **0.70** | **最高輸入影像尺寸，最終選定設定** |

### Test 1：conf=0.24、imgsz=896

首先以 Image Size = **896** 進行影片推論，作為三組 Image Size 比較的起始設定，觀察 YOLOv11n 於實際動態影像中的基本偵測表現。

### Test 2：conf=0.24、imgsz=960

第二組將 Image Size 由 **896 提高至 960**，其餘推論參數維持不變，用以觀察提高輸入影像尺寸後，對目標辨識、Bounding Box 定位及連續影格偵測穩定性的影響。

### Test 3：conf=0.24、imgsz=1024

第三組進一步將 Image Size 提高至 **1024**，並維持 Confidence Threshold = **0.24**、IoU Threshold = **0.70**。

經實際影片測試後，此設定在三組 Image Size 中呈現較符合本研究需求的整體偵測表現，因此選定作為 YOLOv11n 後續代表性影片展示的主要推論設定。

### 最終設定選擇

經三組 Image Size 實際影片推論比較後，本研究最終選擇 **Test 3**：

- **Confidence Threshold：0.24**
- **IoU Threshold：0.70**
- **Image Size：1024**
- **Weight：Fold 3 `best.pt`**

本階段結果顯示，在固定 Confidence Threshold 與 IoU Threshold 的情況下，不同 Image Size 會影響 YOLOv11n 於實際影片中的偵測表現，其中 `imgsz=1024` 為三組設定中整體表現較符合本研究需求的設定。

> **因此，本階段最終採用 `conf=0.24`、`iou=0.70`、`imgsz=1024` 作為 YOLOv11n 優化模型的主要影片推論設定。**

## 最終部署模型規劃 (Final Deployment Model Plan)

本階段使用 YOLOv11n 優化模型 **Fold 3 `best.pt`** 進行影片推論與 Image Size 比較，主要用於觀察模型在實際動態影像中的偵測表現，**並非最終部署模型**。

完成五折交叉驗證、模型優化、Confidence Threshold 分析及影片測試後，將整合前期實驗所選定的模型訓練與推論設定，使用完整的 **900 張六類別資料集**重新訓練 YOLOv11n，建立最終部署模型。

整體流程如下：

**5-Fold 模型評估與參數調整**  
→ **Fold 3 影片推論測試**  
→ **確認模型與推論設定**  
→ **完整 900 張六類別資料重新訓練**  
→ **YOLOv11n Final Deployment Model**

> **Fold 3 `best.pt` 作為部署前的影片測試權重；最終部署模型將使用完整六類別資料集重新訓練建立。**

## 結論 (Conclusion)

本階段完成 YOLOv11n 優化模型的實際影片推論測試，並以 **Fold 3 `best.pt`** 作為測試權重，在固定 Confidence Threshold 與 IoU Threshold 的情況下，比較不同 Image Size 對實際影片偵測結果的影響。

經 `imgsz=896`、`960` 與 `1024` 三組設定比較後，最終採用 **Confidence Threshold = 0.24、IoU Threshold = 0.70、Image Size = 1024** 作為 YOLOv11n 目前主要的影片推論設定，並使用此設定進行 3 部不同測試影片的實際偵測展示。

### 實際影片偵測限制

經實際影片測試後，YOLOv11n 雖能完成六類別目標偵測，但在部分動態場景中仍觀察到類別辨識穩定性不足的情況。

其中較明顯的問題為 **Class Misclassification（類別誤判）**。在部分測試影片中，實際目標為 **Bear**，但模型於部分連續影格中可能將其錯誤辨識為其他類別，使同一目標在影片過程中出現類別短暫切換的現象。

實際測試中較常觀察到 Bear 被誤判為 **Person** 或 **Pig** 的情況，部分影片亦曾出現 **Monkey**、**Cat** 等其他類別的誤判。由於本階段主要透過影片進行實際偵測效果觀察，並未針對各誤判類別的發生次數進行量化統計，因此主要將此現象視為 YOLOv11n 類別辨識穩定性的實際觀察結果。

此外，當目標受到遮擋、距離改變或場景較複雜時，亦可能出現 **Bounding Box 短暫消失或偵測不連續**的情況。

雖然本階段將 Image Size 提高至 `1024` 後，在三組測試設定中呈現較佳的整體影片偵測表現，但仍無法完全消除 Bear 被誤判為其他類別的情況。

> **因此，YOLOv11n 目前主要限制在於動態影片中的類別辨識穩定性，特別是 Bear 類別於部分影格中可能發生 Person 或 Pig 的誤判，此問題將作為後續最終部署模型建立與 YOLOv11s 比較的重要評估項目。**
