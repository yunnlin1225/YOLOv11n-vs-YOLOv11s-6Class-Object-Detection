# YOLOv11s 第二次優化
### YOLOv11s Second Optimization with Background Negative Samples

## 📌 優化說明

本階段為 YOLOv11s 六類別模型的第二次優化，主要針對模型在自然環境中的背景誤判問題進行改善。

透過新增背景負樣本（Background Negative Samples），強化模型對森林、山景及其他自然環境背景的辨識能力，以降低樹木、岩石、陰影及草叢等背景被誤判為目標物件的情況。

## 📊 資料集調整

原始資料集共有 **900 張目標樣本**，包含 6 個正式偵測類別，每個類別各 150 張圖片。

第二次優化新增 **150 張背景負樣本（Background Negative Samples）**，使資料集總數增加至 **1050 張圖片**。

> **注意：Background 並非第 7 個偵測類別，模型仍維持 6 個正式偵測類別。**
