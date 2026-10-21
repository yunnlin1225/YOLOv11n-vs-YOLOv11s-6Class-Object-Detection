# Dataset Samples｜資料集樣本與標註說明

本資料夾用於展示本研究所使用之 **8-Class（八類別）最終物件偵測資料集**，包含影像資料（Images）以及對應的 YOLO 格式標註檔（Labels）。

本研究由前期六類別逐步擴充至八類別，最終資料集包含八個正式偵測類別及背景負樣本。

| Class ID | 類別 | 英文名稱 | 影像數量 | Label 數量 |
|---|---|---|---:|---:|
| 0 | 狗 | Dog | 200 | 200 |
| 1 | 貓 | Cat | 200 | 200 |
| 2 | 台灣黑熊 | Bear | 400 | 400 |
| 3 | 豬 | Pig | 200 | 200 |
| 4 | 台灣獼猴 | Monkey | 200 | 200 |
| 5 | 人 | Person | 200 | 200 |
| 6 | 綠鬣蜥 | Iguana | 400 | 400 |
| 7 | 台灣水鹿 | Deer | 200 | 200 |
| — | 背景負樣本 | Background | 200 | 200（空白標註檔） |
| **總計** | **8 Classes + Background** | | **2,200** | **2,200** |

最終共整理：

- **2,000 張目標影像**
- **200 張背景負樣本**
- **2,200 張影像**
- **2,000 個目標影像 YOLO `.txt` 標註檔**
- **200 個背景空白 `.txt` 標註檔**

> **注意：** Background 並非第九個偵測類別，不設定 Class ID。背景影像對應的 `.txt` 為空白檔案，不包含任何 Bounding Box 標註。

---

## 1. 資料夾結構

為使資料內容更容易閱讀與確認，本研究依八個正式偵測類別及背景負樣本，分別整理影像與標註檔。

```text
dataset_samples/
│
├── images/
│   ├── bear/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── iguana/
│   ├── monkey/
│   ├── person/
│   ├── pig/
│   └── background/
│
├── labels/
│   ├── bear/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── iguana/
│   ├── monkey/
│   ├── person/
│   ├── pig/
│   └── background/
│
└── README.md
```

其中：

- `images/`：存放八類別影像及背景負樣本
- `labels/`：存放與影像對應的 YOLO 標註檔
- `README.md`：說明資料集結構與整理流程

影像與 Label 之間以相同檔名進行對應，例如：

```text
images/bear/bear_001.jpg
labels/bear/bear_001.txt
```

背景負樣本亦使用相同檔名進行對應，但 Label 內容為空白。

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

由於八個正式偵測類別及背景負樣本原本集中存放於來源資料夾，因此透過檔案名稱中的類別名稱進行分類，再將資料複製至 GitHub Repository 對應的資料夾。

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

- 目前位於 `main` Branch
- 本機版本與 GitHub 遠端版本一致
- 沒有尚未 Commit 的檔案
- 可以開始進行資料整理

---

# 4. 建立 Images 類別資料夾

首先於 `dataset_samples/images/` 中建立八個正式偵測類別及背景負樣本資料夾。

以 Bear 為例：

```bat
mkdir dataset_samples\images\bear
```

其他類別依序建立：

```bat
mkdir dataset_samples\images\cat
mkdir dataset_samples\images\deer
mkdir dataset_samples\images\dog
mkdir dataset_samples\images\iguana
mkdir dataset_samples\images\monkey
mkdir dataset_samples\images\person
mkdir dataset_samples\images\pig
mkdir dataset_samples\images\background
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

### Deer

```bat
copy "C:\Users\user\Desktop\images\*deer*" "dataset_samples\images\deer\"
```

### Dog

```bat
copy "C:\Users\user\Desktop\images\*dog*" "dataset_samples\images\dog\"
```

### Iguana

```bat
copy "C:\Users\user\Desktop\images\*iguana*" "dataset_samples\images\iguana\"
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

### Background

```bat
copy "C:\Users\user\Desktop\images\*background*" "dataset_samples\images\background\"
```

> **注意：** 上述指令以檔名包含對應英文類別名稱為前提。若原始影像命名方式不同，應依實際檔名調整篩選條件。

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
400
```

代表 Bear 類別的影像數量符合最終資料集規劃。

各類別預期數量如下：

| Class | Images |
|---|---:|
| Bear | 400 |
| Cat | 200 |
| Deer | 200 |
| Dog | 200 |
| Iguana | 400 |
| Monkey | 200 |
| Person | 200 |
| Pig | 200 |
| Background | 200 |

其餘類別依相同方式進行檢查。

---

# 7. 八類影像最終完整性確認

八個正式偵測類別及背景負樣本全部整理完成後，再一次檢查所有類別的影像數量：

```bat
for %c in (bear cat deer dog iguana monkey person pig background) do @echo ===== %c ===== & @dir /b dataset_samples\images\%c\*.jpg dataset_samples\images\%c\*.jpeg dataset_samples\images\%c\*.png 2>nul | find /c /v ""
```

最終資料集預期數量：

```text
===== bear =====
400
===== cat =====
200
===== deer =====
200
===== dog =====
200
===== iguana =====
400
===== monkey =====
200
===== person =====
200
===== pig =====
200
===== background =====
200
```

因此：

```text
8 Classes = 2,000 Images
Background = 200 Images

Total = 2,200 Images
```

確認八個正式偵測類別共包含 **2,000 張目標影像**，加上 **200 張背景負樣本**，最終資料集共計 **2,200 張影像**。

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

### Deer

```bat
mkdir dataset_samples\labels\deer
```

### Dog

```bat
mkdir dataset_samples\labels\dog
```

### Iguana

```bat
mkdir dataset_samples\labels\iguana
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

### Background

```bat
mkdir dataset_samples\labels\background
```

背景負樣本的 Label 檔案不包含任何物件標註，應保持空白。

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

### Deer

```bat
copy "C:\Users\user\Desktop\labels\*deer*.txt" "dataset_samples\labels\deer\"
```

### Dog

```bat
copy "C:\Users\user\Desktop\labels\*dog*.txt" "dataset_samples\labels\dog\"
```

### Iguana

```bat
copy "C:\Users\user\Desktop\labels\*iguana*.txt" "dataset_samples\labels\iguana\"
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

### Background

若原始背景負樣本已有對應的空白 `.txt` 檔案，可使用：

```bat
copy "C:\Users\user\Desktop\labels\*background*.txt" "dataset_samples\labels\background\"
```

若背景影像尚未建立空白 Label，則需依照影像檔名建立對應的空白 `.txt` 檔案。

---

# 10. 個別檢查每個類別的 Label 數量

每完成一個類別的 Label 複製後，先個別確認 `.txt` 數量是否符合預期。

### Bear

```bat
dir /b dataset_samples\labels\bear\*.txt | find /c /v ""
```

### Cat

```bat
dir /b dataset_samples\labels\cat\*.txt | find /c /v ""
```

### Deer

```bat
dir /b dataset_samples\labels\deer\*.txt | find /c /v ""
```

### Dog

```bat
dir /b dataset_samples\labels\dog\*.txt | find /c /v ""
```

### Iguana

```bat
dir /b dataset_samples\labels\iguana\*.txt | find /c /v ""
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

### Background

```bat
dir /b dataset_samples\labels\background\*.txt | find /c /v ""
```

各類別預期 Label 數量如下：

| Class | Labels |
|---|---:|
| Bear | 400 |
| Cat | 200 |
| Deer | 200 |
| Dog | 200 |
| Iguana | 400 |
| Monkey | 200 |
| Person | 200 |
| Pig | 200 |
| Background | 200 |

完成單一類別確認後，再進行下一類別的資料整理與 Git Commit。

---

# 11. 八類 Labels 最終完整性確認

八個正式偵測類別及背景負樣本全部完成後，再一次檢查所有 `.txt` 標註檔：

```bat
for %c in (bear cat deer dog iguana monkey person pig background) do @echo ===== %c ===== & @dir /b dataset_samples\labels\%c\*.txt 2>nul | find /c /v ""
```

最終資料集預期數量：

```text
===== bear =====
400
===== cat =====
200
===== deer =====
200
===== dog =====
200
===== iguana =====
400
===== monkey =====
200
===== person =====
200
===== pig =====
200
===== background =====
200
```

因此：

```text
8 Classes = 2,000 Target Labels
Background = 200 Empty Labels

Total = 2,200 Label Files
```

確認八個正式偵測類別共有 **2,000 個 YOLO 目標標註檔**，另有 **200 個背景空白標註檔**。

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
Deer
Dog
Iguana
Monkey
Person
Pig
Background
```

背景負樣本亦依相同流程上傳。

例如 Background Images：

```bash
git add dataset_samples/images/background/
git commit -m "Add background dataset samples"
git push origin main
```

Background Labels：

```bash
git add dataset_samples/labels/background/
git commit -m "Add background dataset labels"
git push origin main
```

透過分類提交，可保留各類別資料整理與更新的版本紀錄。

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

若發生檔案衝突，需先解決衝突並完成 rebase，才能繼續 Push。

---

# 14. 最終資料完整性確認

完成所有影像與標註檔整理後，最終八類別資料集的規劃數量如下：

| Class | Images | Labels | Status |
|---|---:|---:|---|
| Bear | 400 | 400 | ✅ Complete |
| Cat | 200 | 200 | ✅ Complete |
| Deer | 200 | 200 | ✅ Complete |
| Dog | 200 | 200 | ✅ Complete |
| Iguana | 400 | 400 | ✅ Complete |
| Monkey | 200 | 200 | ✅ Complete |
| Person | 200 | 200 | ✅ Complete |
| Pig | 200 | 200 | ✅ Complete |
| Background | 200 | 200 | ✅ Complete |
| **Total** | **2,200** | **2,200** | **✅ Complete** |

最終資料集規模：

```text
8 Classes
2,000 Target Images
200 Background Images

2,200 Total Images
2,200 YOLO Label Files
```

其中 2,200 個 Label 檔案包含 2,000 個目標影像標註檔，以及 200 個背景空白標註檔。

各類別影像與 Label 數量確認一致後，仍需檢查檔名配對及標註內容，確保資料可作為後續 YOLO11 系列模型訓練、交叉驗證與模型效能比較的基礎。

---

# 15. YOLO Label 格式說明

本研究使用 YOLO Object Detection 標註格式，每一張目標影像皆具有對應的 `.txt` 檔案。

基本格式如下：

```text
class_id x_center y_center width height
```

其中：

| 欄位 | 說明 |
|---|---|
| `class_id` | 目標類別編號 |
| `x_center` | Bounding Box 中心點 X 座標 |
| `y_center` | Bounding Box 中心點 Y 座標 |
| `width` | Bounding Box 寬度 |
| `height` | Bounding Box 高度 |

座標值皆經過正規化處理，數值範圍介於 `0` 至 `1`。

本研究最終八類別編號為：

| Class ID | Class |
|---:|---|
| 0 | Dog |
| 1 | Cat |
| 2 | Bear |
| 3 | Pig |
| 4 | Monkey |
| 5 | Person |
| 6 | Iguana |
| 7 | Deer |

背景負樣本不設定 Class ID，對應的 YOLO `.txt` 檔案為空白，不包含任何物件標註資訊。

---

# 16. 資料整理目的

本資料夾除了保存研究所使用的影像與 YOLO Labels，也透過明確的資料夾結構與 Git Commit 紀錄，呈現完整的資料管理流程。

透過八類別及背景負樣本的分層管理，可以：

- 清楚確認各類別資料數量
- 快速比對影像與 Label
- 降低資料遺漏或分類錯誤的可能性
- 保留資料整理與更新歷程
- 提升研究專案的可讀性與可重現性
- 方便後續進行模型訓練、驗證與部署

本研究由前期六類別資料集逐步擴充至八類別，最終建立包含 **2,000 張目標影像及 200 張背景負樣本，共 2,200 張影像**的資料集。

本資料集將作為 **YOLO11n 與 YOLO11s 八類別物件偵測模型**訓練、五折交叉驗證及模型效能比較的資料基礎，並以 **YOLO11n 八類別的 `Train_Val_80_20` 模型**作為 WildGuard AI 的主要部署版本。
