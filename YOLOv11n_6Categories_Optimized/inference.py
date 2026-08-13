import os
from ultralytics import YOLO

# ==========================================
# 1. 基本設定 (YOLOv11n)
# ==========================================
# 填入你的 YOLOv11n 最佳模型權重路徑 (例如 Fold 1 的 best.pt)
MODEL_PATH = r'C:\Users\user\Desktop\YOLOv11n_6Categories(1)\runs\detect\YOLOv11n_6Class_5Fold\Fold_1\weights\best.pt'

# 官方標準預設門檻設定 (P、R、mAP50 🟢達標 > 0.90)
CONF_THRES = 0.25  # 信心度門檻預設值 (Confidence Threshold)
IOU_THRES = 0.60   # 非極大值抑制重疊門檻預設值 (NMS IoU Threshold)
IMGSZ = 640        # 推論影像解析度預設值 (Image Size)

# 測試目標 (可填入圖片路徑、影片路徑或包含多張圖片的資料夾路徑)
INPUT_SOURCE = r'C:\Users\user\Desktop\test_img.jpg'


def run_inference(
    model_path, source_path, conf=0.25, iou=0.60, imgsz=640, save_txt=False
):
    """使用訓練好的 YOLO 模型對圖片或影片進行辨識推論

    :param model_path: best.pt 權重檔路徑
    :param source_path: 輸入來源 (圖片檔/影片檔/圖片資料夾)
    :param conf: 信心度門檻
    :param iou: NMS IoU 門檻
    :param imgsz: 影像解析度
    :param save_txt: 是否儲存辨識結果的文字檔 (txt 座標與類別)
    """
    if not os.path.exists(model_path):
        print(f'❌ 錯誤：找不到模型權重檔案：{model_path}')
        return

    if not os.path.exists(source_path):
        print(f'❌ 錯誤：找不到輸入來源檔案：{source_path}')
        return

    print('🚀 正在載入 YOLO 模型...')
    model = YOLO(model_path)

    print(
        f'🔍 開始進行推論 (設定 conf={conf}, iou={iou}, imgsz={imgsz}, 來源:'
        f' {source_path})...'
    )

    # 執行推論
    results = model.predict(
        source=source_path,
        conf=conf,   # 設定 0.25 預設門檻
        iou=iou,     # 設定 0.60 預設門檻
        imgsz=imgsz, # 設定 640 預設解析度
        save=True,   # 自動繪製邊界框並儲存標記後的影像/影片
        save_txt=save_txt,  # 是否另外導出 .txt 格式的標籤結果
        save_conf=True,     # 導出 .txt 時是否包含信心度數值
        line_width=2,       # 繪製邊界框的線條粗細
        show_labels=True,   # 顯示類別名稱
        show_conf=True,     # 顯示信心度百分比
    )

    # 取得儲存結果的路徑
    save_dir = results[0].save_dir
    print('\n' + '=' * 60)
    print('✅ 推論完成！')
    print(f'📂 自動畫框後的結果已儲存至：{save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    # 執行推論
    run_inference(
        model_path=MODEL_PATH,
        source_path=INPUT_SOURCE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMGSZ,
        save_txt=False,  # 若需要導出 txt 標註檔，可改為 True
    )
