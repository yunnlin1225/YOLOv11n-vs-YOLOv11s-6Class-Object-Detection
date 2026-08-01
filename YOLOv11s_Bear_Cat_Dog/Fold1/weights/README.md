# weights
本資料夾存放第一折（Fold1）YOLOv11s 模型訓練完成後所產生的權重檔。

## 檔案說明

- **best.pt**
  - 訓練過程中在驗證集（Validation Set）表現最佳的模型權重。
  - 建議用於模型測試、推論及最終成果展示。

- **last.pt**
  - 訓練完成最後一個 Epoch 所儲存的模型權重。
  - 可用於繼續訓練（Resume Training）或作為訓練完成的紀錄。
