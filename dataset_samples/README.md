# Dataset Samples｜資料集樣本與標註說明

本資料夾用於展示本研究所使用之六類別目標偵測資料，包含影像資料（Images）以及對應的 YOLO 格式標註檔（Labels）。

本研究共包含六個目標類別：

| 類別     | 英文名稱          |    影像數量 | Label 數量 |
| ------ | ------------- | ------: | -------: |
| 熊      | Bear          |     150 |      150 |
| 貓      | Cat           |     150 |      150 |
| 狗      | Dog           |     150 |      150 |
| 猴      | Monkey        |     150 |      150 |
| 人      | Person        |     150 |      150 |
| 豬      | Pig           |     150 |      150 |
| **總計** | **6 Classes** | **900** |  **900** |

最終共整理：

* **900 張影像**
* **900 個 YOLO `.txt` 標註檔**
* 每一張影像皆有對應的 Label 檔案

---

## 1. 資料夾結構

為使資料內容更容易閱讀與確認，本研究依六個類別分別整理影像與標註檔。

```text
dataset_samples/
│
├── images/
│   ├── bear/
│   ├── cat/
│   ├── dog/
│   ├── monkey/
│   ├── person/
│   └── pig/
│
├── labels/
│   ├── bear/
│   ├── cat/
│   ├── dog/
│   ├── monkey/
│   ├── person/
│   └── pig/
│
└── README.md
```

其中：

* `images/`：存放六類別影像
* `labels/`：存放與影像對應的 YOLO 標註檔
* `README.md`：說明資料集結構與整理流程

影像與 Label 之間以相同檔名進行對應，例如：

```text
images/bear/bear_001.jpg
labels/bear/bear_001.txt
```

---

# 2. 本機資料來源

原始影像與 Label 在整理前分別集中存放於：

```text
C:\Users\user\Desktop\images
```

以及：

```text
C:\Users\user\Desktop\labels
```

由於六種類別原本存放於同一個資料夾，因此透過檔案名稱中的類別名稱進行分類，再將資料複製至 GitHub Repository 對應的資料夾。

---

# 3. 確認 Git Repository 狀態

在進行資料整理前，先進入本機已 Clone 的 GitHub Repository：

```bat
cd /d "C:\Users\user\Desktop\YOLOv11n-vs-YOLOv11s-6Class-Object-Detection"
```

確認目前 Git 狀態：

```bash
git status
```

若顯示：

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

代表：

* 目前位於 `main` Branch
* 本機版本與 GitHub 遠端版本一致
* 沒有尚未 Commit 的檔案
* 可以開始進行資料整理

---

# 4. 建立 Images 類別資料夾

首先於 `dataset_samples/images/` 中建立六個類別資料夾。

以 Bear 為例：

```bat
mkdir dataset_samples\images\bear
```

其他類別依序建立：

```bat
mkdir dataset_samples\images\cat
mkdir dataset_samples\images\dog
mkdir dataset_samples\images\monkey
mkdir dataset_samples\images\person
mkdir dataset_samples\images\pig
```

---

# 5. 分類並複製影像

原始影像集中存放於：

```text
C:\Users\user\Desktop\images
```

因此使用檔名中的類別名稱進行篩選。

### Bear

```bat
copy "C:\Users\user\Desktop\images\*bear*" "dataset_samples\images\bear\"
```

### Cat

```bat
copy "C:\Users\user\Desktop\images\*cat*" "dataset_samples\images\cat\"
```

### Dog

```bat
copy "C:\Users\user\Desktop\images\*dog*" "dataset_samples\images\dog\"
```

### Monkey

```bat
copy "C:\Users\user\Desktop\images\*monkey*" "dataset_samples\images\monkey\"
```

### Person

```bat
copy "C:\Users\user\Desktop\images\*person*" "dataset_samples\images\person\"
```

### Pig

```bat
copy "C:\Users\user\Desktop\images\*pig*" "dataset_samples\images\pig\"
```

---

# 6. 個別檢查每個類別的影像數量

每完成一個類別的複製後，先個別確認該類別的影像數量。

由於資料集中可能同時包含：

```text
.jpg
.jpeg
.png
```

因此檢查時需將不同影像格式納入。

例如 Bear：

```bat
dir /b dataset_samples\images\bear\*.jpg dataset_samples\images\bear\*.jpeg dataset_samples\images\bear\*.png 2>nul | find /c /v ""
```

若結果為：

```text
150
```

代表 Bear 類別影像數量正確。

其餘類別依相同方式進行檢查。

---

# 7. 六類影像最終完整性確認

六個類別全部整理完成後，再一次檢查所有類別的影像數量：

```bat
for %c in (bear cat dog monkey person pig) do @echo ===== %c ===== & @dir /b dataset_samples\images\%c\*.jpg dataset_samples\images\%c\*.jpeg dataset_samples\images\%c\*.png 2>nul | find /c /v ""
```

實際確認結果：

```text
===== bear =====
150
===== cat =====
150
===== dog =====
150
===== monkey =====
150
===== person =====
150
===== pig =====
150
```

因此：

```text
150 × 6 = 900 Images
```

確認六個類別共包含 **900 張影像**。

---

# 8. 建立 Labels 類別資料夾

影像整理完成後，使用相同結構建立 Labels 資料夾。

### Bear

```bat
mkdir dataset_samples\labels\bear
```

### Cat

```bat
mkdir dataset_samples\labels\cat
```

### Dog

```bat
mkdir dataset_samples\labels\dog
```

### Monkey

```bat
mkdir dataset_samples\labels\monkey
```

### Person

```bat
mkdir dataset_samples\labels\person
```

### Pig

```bat
mkdir dataset_samples\labels\pig
```

---

# 9. 分類並複製 YOLO Labels

原始 YOLO Labels 集中存放於：

```text
C:\Users\user\Desktop\labels
```

每個標註檔皆為 `.txt` 格式。

### Bear

```bat
copy "C:\Users\user\Desktop\labels\*bear*.txt" "dataset_samples\labels\bear\"
```

### Cat

```bat
copy "C:\Users\user\Desktop\labels\*cat*.txt" "dataset_samples\labels\cat\"
```

### Dog

```bat
copy "C:\Users\user\Desktop\labels\*dog*.txt" "dataset_samples\labels\dog\"
```

### Monkey

```bat
copy "C:\Users\user\Desktop\labels\*monkey*.txt" "dataset_samples\labels\monkey\"
```

### Person

```bat
copy "C:\Users\user\Desktop\labels\*person*.txt" "dataset_samples\labels\person\"
```

### Pig

```bat
copy "C:\Users\user\Desktop\labels\*pig*.txt" "dataset_samples\labels\pig\"
```

---

# 10. 個別檢查每個類別的 Label 數量

每完成一個類別的 Label 複製後，先個別確認 `.txt` 數量是否為 150。

### Bear

```bat
dir /b dataset_samples\labels\bear\*.txt | find /c /v ""
```

### Cat

```bat
dir /b dataset_samples\labels\cat\*.txt | find /c /v ""
```

### Dog

```bat
dir /b dataset_samples\labels\dog\*.txt | find /c /v ""
```

### Monkey

```bat
dir /b dataset_samples\labels\monkey\*.txt | find /c /v ""
```

### Person

```bat
dir /b dataset_samples\labels\person\*.txt | find /c /v ""
```

### Pig

```bat
dir /b dataset_samples\labels\pig\*.txt | find /c /v ""
```

每個類別均確認為：

```text
150
```

完成單一類別確認後，再進行下一類別的資料整理與 Git Commit。

---

# 11. 六類 Labels 最終完整性確認

六個類別全部完成後，再一次檢查所有 `.txt` 標註檔：

```bat
for %c in (bear cat dog monkey person pig) do @echo ===== %c ===== & @dir /b dataset_samples\labels\%c\*.txt 2>nul | find /c /v ""
```

實際確認結果：

```text
===== bear =====
150
===== cat =====
150
===== dog =====
150
===== monkey =====
150
===== person =====
150
===== pig =====
150
```

因此：

```text
150 × 6 = 900 Labels
```

確認六類別共包含 **900 個 YOLO `.txt` 標註檔**。

---

# 12. Git Commit 與 Push 流程

為避免一次提交大量影像造成上傳失敗，同時讓 Git Commit History 更清楚，本研究採用「一個類別一個 Commit」的方式進行版本管理。

例如 Bear Images：

```bash
git add dataset_samples/images/bear/
```

建立 Commit：

```bash
git commit -m "Add bear dataset samples"
```

推送至 GitHub：

```bash
git push origin main
```

Bear Labels：

```bash
git add dataset_samples/labels/bear/
```

建立 Commit：

```bash
git commit -m "Add bear dataset labels"
```

再推送：

```bash
git push origin main
```

其他類別依照相同方式依序完成：

```text
Bear
Cat
Dog
Monkey
Person
Pig
```

---

# 13. Git 遠端版本衝突處理

若 GitHub Repository 在本機 Push 前已經存在新的 Commit，可能出現：

```text
! [rejected] main -> main (fetch first)
```

以及：

```text
Updates were rejected because the remote contains work that you do not have locally.
```

此情況代表 GitHub 遠端 Repository 有較新的內容，本機需先同步遠端版本。

使用：

```bash
git pull --rebase origin main
```

將遠端最新 Commit 整合至本機後，再次執行：

```bash
git push origin main
```

使用 `rebase` 可以在保留遠端最新內容的同時，將本機尚未推送的 Commit 接續於最新版本之後，使 Commit History 維持較清楚的線性結構。

---

# 14. 最終資料完整性確認

完成所有影像與標註檔整理後，最終資料如下：

| Class     |  Images |  Labels | Status         |
| --------- | ------: | ------: | -------------- |
| Bear      |     150 |     150 | ✅              |
| Cat       |     150 |     150 | ✅              |
| Dog       |     150 |     150 | ✅              |
| Monkey    |     150 |     150 | ✅              |
| Person    |     150 |     150 | ✅              |
| Pig       |     150 |     150 | ✅              |
| **Total** | **900** | **900** | **✅ Complete** |

最終確認：

```text
6 Classes
900 Images
900 YOLO Labels
```

各類別影像與 Label 數量一致，可作為後續 YOLOv11 系列模型訓練、交叉驗證與模型效能比較之資料基礎。

---

# 15. YOLO Label 格式說明

本研究使用 YOLO Object Detection 標註格式，每一張影像皆具有對應的 `.txt` 檔。

基本格式如下：

```text
class_id x_center y_center width height
```

其中：

| 欄位         | 說明                    |
| ---------- | --------------------- |
| `class_id` | 目標類別編號                |
| `x_center` | Bounding Box 中心點 X 座標 |
| `y_center` | Bounding Box 中心點 Y 座標 |
| `width`    | Bounding Box 寬度       |
| `height`   | Bounding Box 高度       |

座標值皆經過正規化處理，數值範圍介於 `0` 至 `1`。

本研究六類別編號為：

| Class ID | Class  |
| -------: | ------ |
|        0 | Dog    |
|        1 | Cat    |
|        2 | Bear   |
|        3 | Pig    |
|        4 | Monkey |
|        5 | Person |

---

# 16. 資料整理目的

本資料夾除了保存研究所使用的影像與 YOLO Labels，也透過明確的資料夾結構與 Git Commit 紀錄，呈現完整的資料管理流程。

透過六類別分層管理，可以：

* 清楚確認各類別資料數量
* 快速比對影像與 Label
* 降低資料遺漏或分類錯誤的可能性
* 保留資料整理與更新歷程
* 提升研究專案的可讀性與可重現性
* 方便後續進行模型訓練、驗證與部署

本資料集將作為後續 **YOLOv11n 與 YOLOv11s 六類別目標偵測模型比較研究**之資料基礎，並進一步評估不同模型在偵測準確度、定位能力、運算效率與實際部署上的表現。

