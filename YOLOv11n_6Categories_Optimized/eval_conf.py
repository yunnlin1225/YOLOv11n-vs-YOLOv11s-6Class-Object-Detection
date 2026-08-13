import numpy as np
from ultralytics import YOLO

# 1. 5 折最佳權重檔 (best.pt) 完整路徑 (更新為 YOLOv11n)
model_paths = [
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_1\weights\best.pt',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_2\weights\best.pt',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_3\weights\best.pt',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_4\weights\best.pt',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_5\weights\best.pt',
]

# 2. 5 折對應數據配置文件 (.yaml) 完整路徑 (更新為 YOLOv11n)
data_paths = [
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\fold1.yaml',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\fold2.yaml',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\fold3.yaml',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\fold4.yaml',
    r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\fold5.yaml',
]

# 3. 測試信心度門檻 (設定為 0.26)
target_conf = 0.26

if __name__ == '__main__':
    p_list = []
    r_list = []
    map50_list = []
    map50_95_list = []

    print(f'🚀 開始以 conf={target_conf} (標準乾淨驗證設定) 評估 5 折模型數據...\n')

    for i in range(5):
        # 載入模型
        model = YOLO(model_paths[i])

        # 指定 project 與 name，進行標準評估
        metrics = model.val(
            data=data_paths[i],
            conf=target_conf,
            iou=0.60,         # 💡 恢復標準 NMS IoU 門檻
            imgsz=640,        # 💡 恢復與訓練階段一致的 640 原生解析度
            split='val',
            workers=0,
            project='eval_results',
            name=f'conf_0.26_fold{i+1}',
        )

        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map

        p_list.append(p)
        r_list.append(r)
        map50_list.append(map50)
        map50_95_list.append(map50_95)

        print(
            f'Fold {i+1} -> Precision: {p:.4f} | Recall: {r:.4f} | mAP50:'
            f' {map50:.4f} | mAP50-95: {map50_95:.4f}'
        )

    # 計算平均
    mean_p = np.mean(p_list)
    mean_r = np.mean(r_list)
    mean_map50 = np.mean(map50_list)
    mean_map50_95 = np.mean(map50_95_list)

    print('\n' + '=' * 60)
    print(f'🎯 5-Fold 重新評估終局成績 (conf = {target_conf}):')
    print(
        f'平均 Precision (P): {mean_p:.4f} '
        + ('🟢 (達標 > 0.90)' if mean_p >= 0.90 else '🔴 (未達標)')
    )
    print(
        f'平均 Recall (R)   : {mean_r:.4f} '
        + ('🟢 (達標 > 0.90)' if mean_r >= 0.90 else '🔴 (未達標)')
    )
    print(f'平均 mAP50       : {mean_map50:.4f}')
    print(f'平均 mAP50-95    : {mean_map50_95:.4f}')
    print('=' * 60)