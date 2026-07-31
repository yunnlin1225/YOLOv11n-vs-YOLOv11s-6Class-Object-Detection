# YOLOv11n-vs-YOLOv11s-6Class-Object-Detection
本專題以 YOLOv11n 與 YOLOv11s 為核心模型，建立六類別物件偵測系統，包含 Bear、Cat、Dog、Person、Monkey 與 Pig。資料集由自行蒐集之影像建立，並完成物件標註與資料前處理，採用五折交叉驗證（5-Fold Cross Validation）進行模型訓練與評估，以降低資料切分造成的偏差，提升實驗結果的可信度。研究中比較兩種模型在 Precision、Recall、mAP50 與 mAP50-95 等指標上的表現，分析模型於不同類別的偵測能力與定位精度，作為模型效能評估依據，並提供未來多類別物件偵測系統之研究與應用參考。
