import cv2
import os
import torch
import numpy as np
from tkinter import Tk, filedialog
from ultralytics import YOLO

# ======== TẠO THƯ MỤC LƯU ẢNH KÝ TỰ =========
output_folder = "D:/KyTu_Tach_YOLO"
os.makedirs(output_folder, exist_ok=True)
# ======== LOAD MÔ HÌNH YOLOv8 SEGMENTATION =========
model_path = r"D:/Bien_So/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# ======== HÀM TÁCH VÀ LƯU ẢNH KÝ TỰ =========
def tach_ky_tu_bang_yolo(duong_dan_anh):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"Không mở được ảnh tại: {duong_dan_anh}")
        return

    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy toạ độ (x1, y1, x2, y2)

    if len(boxes) == 0:
        print("Không phát hiện được ký tự nào.")
        return

    # Sắp xếp: trước theo trục y, sau theo x (từ trên xuống dưới, trái sang phải)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    print(f"Tổng số ký tự phát hiện: {len(boxes)}")

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        ky_tu = img[y1:y2, x1:x2]

        ten_anh = os.path.join(output_folder, f"ky_tu_{idx+1}.jpg")
        cv2.imwrite(ten_anh, ky_tu)
        cv2.imshow(f"Ký tự {idx+1}", ky_tu)

    cv2.imshow("Ảnh gốc", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======== CHƯƠNG TRÌNH CHÍNH =========
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    duong_dan_anh = filedialog.askopenfilename(title="Chọn ảnh biển số", filetypes=[("Ảnh", "*.jpg *.png *.jpeg")])

    if duong_dan_anh:
        print(f"Đã chọn ảnh: {duong_dan_anh}")
        tach_ky_tu_bang_yolo(duong_dan_anh)
    else:
        print("Bạn chưa chọn ảnh, thoát chương trình.")
