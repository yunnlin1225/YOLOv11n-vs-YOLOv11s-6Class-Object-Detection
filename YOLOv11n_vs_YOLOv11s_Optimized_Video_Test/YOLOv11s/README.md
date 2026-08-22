# YOLOv11s 優化模型影片測試
(YOLOv11s Optimized Video Test)

## 影片測試目的 (Video Test Objective)

本階段使用優化後的 **YOLOv11s 六類別目標偵測模型**進行實際影片推論測試，進一步觀察模型於連續動態影像中的實際偵測表現。

相較於單張影像推論，影片測試可進一步觀察模型在連續影格中的**目標辨識、偵測穩定性、誤檢、漏檢及 Bounding Box 定位情況**，以評估模型面對實際動態場景時的表現。

此外，本階段亦透過不同推論設定進行測試與比較，觀察 **Confidence Threshold、IoU Threshold 與 Image Size (`imgsz`)** 對影片偵測結果的影響。

> **影片測試結果將作為 YOLOv11s 後續推論設定選擇，以及與 YOLOv11n 比較與最終部署評估之參考。**

---

## 測試模型與權重 (Test Model & Weights)

本階段影片測試採用 **YOLOv11s 優化模型 Fold 3** 所產生的最佳權重檔 `best.pt` 進行影片推論。

| 項目 | 設定 |
| --- | --- |
| Model | YOLOv11s |
| Classes | 6 |
| Model Version | Optimized |
| Selected Fold | **Fold 3** |
| Weight | `best.pt` |
| Test Type | Video Inference |

### Fold 3 選擇依據

五折交叉驗證的主要目的為評估模型於不同資料切分下的穩定性與泛化能力，而實際影片推論則需選定單一模型權重進行測試。

因此，本階段由五個 Fold 中選擇 **Fold 3 的 `best.pt`** 作為 YOLOv11s 優化模型的影片測試權重，並於相同測試影片下觀察模型的實際動態偵測表現。

> **Fold 3 權重主要用於影片推論與實際偵測效果評估，五折平均結果仍作為 YOLOv11s 整體模型效能的主要研究評估依據。**

---

## 影片測試參數 (Video Test Parameters)

本階段使用 YOLOv11s 優化模型 **Fold 3 `best.pt`** 進行影片推論，並以相同測試影片比較三組不同推論設定，觀察 Confidence Threshold 與 Image Size 對實際影片偵測結果的影響。

### 測試設定

| Test | Confidence | Image Size | IoU | 說明 |
|---|---:|---:|---:|---|
| Test 1 | **0.25** | **800** | **0.70** | 基準設定 |
| Test 2 | **0.32** | **800** | **0.70** | 依 Fold 3 F1-Confidence Curve 調整 Confidence |
| Test 3 | **0.32** | **960** | **0.70** | 維持 Confidence，進一步提高輸入影像尺寸 |

三組測試皆使用相同的 **YOLOv11s Fold 3 `best.pt`** 與相同測試影片，並固定 IoU Threshold 為 **0.70**，主要調整 `conf` 與 `imgsz`，以比較不同推論設定對影片偵測結果的影響。

### Confidence Threshold 調整依據

Test 1 先以 **conf=0.25、imgsz=800** 作為影片推論的基準設定。

在觀察 YOLOv11s 優化模型 **Fold 3 的 F1-Confidence Curve** 後，發現模型的 F1 Score 在 Confidence 約 **0.32** 附近具有較佳表現，因此 Test 2 將 Confidence Threshold 由 **0.25 調整至 0.32**。

此調整以 Fold 3 模型的 F1-Confidence Curve 作為參數選擇依據，希望在 Precision 與 Recall 之間取得較適當的平衡，並進一步觀察提高 Confidence Threshold 後對影片中誤檢與漏檢情況的影響。

### Image Size 調整

Test 2 維持 **imgsz=800**，主要觀察 Confidence Threshold 由 0.25 提高至 0.32 後的變化。

Test 3 則維持 **conf=0.32、iou=0.70**，並將 Image Size 由 **800 提高至 960**，進一步觀察較高輸入解析度對目標辨識、Bounding Box 定位及連續影格偵測穩定性的影響。

> **三組測試將從目標辨識、誤檢、漏檢、Bounding Box 及連續影格穩定性等方面進行比較，作為 YOLOv11s 後續影片推論與部署參數選擇之依據。**

---

## 影片測試重點 (Video Test Focus)

本階段透過實際影片進行 YOLOv11s 優化模型推論，除了觀察模型是否能正確辨識六個目標類別外，亦針對模型於連續影格中的實際偵測表現進行觀察。

影片測試主要評估以下項目：

1. **目標辨識能力**
   - 觀察模型是否能正確辨識 Dog、Cat、Bear、Pig、Monkey 及 Person 六個類別。
   - 確認是否出現類別辨識錯誤的情況。

2. **連續影格偵測穩定性**
   - 觀察同一目標於影片連續影格中是否能維持穩定偵測。
   - 記錄 Bounding Box 是否出現頻繁消失、重新出現或明顯跳動的情況。

3. **誤檢情況 (False Positive)**
   - 觀察模型是否將背景、其他物件或非目標物件錯誤辨識為六個目標類別。
   - 比較不同推論參數設定下的誤檢變化。

4. **漏檢情況 (False Negative)**
   - 觀察影片中實際存在的目標是否出現未被模型偵測的情況。
   - 比較提高 Confidence Threshold 後是否造成漏檢增加。

5. **Bounding Box 定位表現**
   - 觀察偵測框是否能合理涵蓋目標物件。
   - 觀察目標移動或場景變化時 Bounding Box 的穩定性。

6. **Confidence Score 表現**
   - 觀察模型於連續影格中的 Confidence Score 變化。
   - 比較 `conf=0.25` 與 `conf=0.32` 對實際偵測結果的影響。

7. **Image Size 影響**
   - 比較 `imgsz=800` 與 `imgsz=960` 的影片偵測結果。
   - 觀察提高輸入影像尺寸後，對目標辨識、定位與偵測穩定性的影響。

8. **實際部署可行性**
   - 綜合模型的辨識能力、偵測穩定性、誤檢與漏檢情況，評估 YOLOv11s 優化模型於實際動態影像環境中的應用表現。
   - 測試結果將作為後續模型比較及最終部署設定選擇之參考。

---

## 影片測試結果 (Video Test Results)

本研究使用 **3 部不同測試影片**進行 YOLOv11s 優化模型的實際影片推論，並分別測試三組不同推論參數，共完成 **9 組影片推論實驗**。

經比較三組參數於實際影片中的目標辨識、誤檢、漏檢、Bounding Box 定位及連續影格偵測穩定性後，最終選擇 **Confidence Threshold = 0.32、Image Size = 800、IoU Threshold = 0.70** 作為本階段表現較佳的影片推論設定。

### 最終影片推論設定

| 項目 | 設定 |
|---|---:|
| Model | YOLOv11s |
| Model Version | Optimized |
| Weight | Fold 3 `best.pt` |
| Confidence Threshold | **0.32** |
| IoU Threshold | **0.70** |
| Image Size | **800** |

### 代表性影片測試

為避免同時展示 9 組相似的影片結果造成內容過於冗長，本說明檔主要展示最終選定參數 **conf=0.32、iou=0.70、imgsz=800** 下的 3 部不同測試影片。

三部影片皆使用相同模型權重與推論參數進行測試，用以觀察 YOLOv11s 優化模型於不同動態影像與場景中的實際偵測表現。

### Video 1

https://github.com/user-attachments/assets/a44167d7-1a82-4070-a5fa-12e433984dd8

### Video 2

https://github.com/user-attachments/assets/0366a76a-6f01-4413-8aec-413306bd2c0d

### Video 3

https://github.com/user-attachments/assets/8b9c3243-5d46-4fb8-b0a5-41b3ef8c7054

> **其餘參數組合主要作為推論參數比較實驗，本說明檔則以最終選定設定之三部代表性影片作為主要展示結果。**

---

## 不同推論設定比較 (Inference Setting Comparison)

本階段針對 YOLOv11s 優化模型進行三組影片推論參數測試，並使用相同模型權重與測試影片進行比較。

| Test | Confidence | Image Size | IoU | 調整重點 |
| --- | ---: | ---: | ---: | --- |
| Test 1 | 0.25 | 800 | 0.70 | 基準設定 |
| **Test 2** | **0.32** | **800** | **0.70** | **F1 Curve 調整後之最終設定** |
| Test 3 | 0.32 | 960 | 0.70 | 提高輸入影像尺寸進行比較 |

### Test 1：conf=0.25、imgsz=800

首先以 Confidence Threshold = **0.25**、Image Size = **800** 作為影片推論的基準設定，用於觀察模型在較低 Confidence Threshold 下的實際偵測情況。

### Test 2：conf=0.32、imgsz=800

根據 YOLOv11s 優化模型 **Fold 3 的 F1-Confidence Curve**，將 Confidence Threshold 由 **0.25 提高至 0.32**，Image Size 維持 800。

實際影片測試後，此設定在三組測試中呈現較符合本研究需求的整體偵測表現，因此作為後續影片展示的主要推論設定。

### Test 3：conf=0.32、imgsz=960

第三組維持 Confidence Threshold = **0.32** 與 IoU Threshold = **0.70**，並將 Image Size 由 **800 提高至 960**。

此組主要用於確認提高輸入影像尺寸是否能進一步改善實際影片偵測效果。經比較後，提高至 `imgsz=960` 並未呈現較 Test 2 更佳的整體影片偵測表現，因此未作為最終設定。

### 最終設定選擇

經三組推論參數於實際影片中的比較後，本研究最終選擇 **Test 2** 作為 YOLOv11s 優化模型的主要影片推論設定：

- **Confidence Threshold：0.32**
- **IoU Threshold：0.70**
- **Image Size：800**
- **Weight：Fold 3 `best.pt`**

此結果亦顯示，提高 Image Size 並不代表實際影片偵測效果必然提升，因此本研究最終保留 `imgsz=800`，並搭配由 Fold 3 F1-Confidence Curve 所選定的 `conf=0.32`。

> **綜合實際影片偵測表現，本階段最終採用 `conf=0.32`、`iou=0.70`、`imgsz=800` 作為 YOLOv11s 優化模型的主要影片推論設定。**

---

## 最終部署模型規劃 (Final Deployment Model Plan)

本階段使用 YOLOv11s 優化模型 **Fold 3 `best.pt`** 進行影片推論與參數比較，主要用於觀察模型在實際動態影像中的偵測表現，**並非最終部署模型**。

完成五折交叉驗證、超參數調整、Confidence Threshold 分析及影片測試後，將整合前期實驗所選定的模型訓練與推論設定，使用完整的 **900 張六類別資料集**重新訓練 YOLOv11s，建立最終部署模型。

整體流程如下：

**5-Fold 模型評估與參數調整**  
→ **Fold 3 影片推論測試**  
→ **確認模型與推論設定**  
→ **完整 900 張六類別資料重新訓練**  
→ **YOLOv11s Final Deployment Model**

> **Fold 3 `best.pt` 作為部署前的影片測試權重；最終部署模型將使用完整六類別資料集重新訓練建立。**

---

## 結論 (Conclusion)

本階段完成 YOLOv11s 優化模型的實際影片推論測試，並以 **Fold 3 `best.pt`** 作為測試權重，比較不同 Confidence Threshold 與 Image Size 對影片偵測結果的影響。

經三組推論設定比較後，最終採用 **Confidence Threshold = 0.32、IoU Threshold = 0.70、Image Size = 800** 作為 YOLOv11s 目前主要的影片推論設定，並使用此設定於 3 部不同測試影片進行實際偵測展示。

本階段的 Fold 3 模型主要作為影片測試與推論參數評估使用，**並非最終部署模型**。後續將整合五折交叉驗證、模型優化及影片測試結果，使用完整 **900 張六類別資料集**重新訓練 YOLOv11s，建立最終部署模型。

> **影片測試結果將作為 YOLOv11s Final Deployment Model 建立與後續實際部署測試的重要參考。**

