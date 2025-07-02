import cv2
import os
import torch
import numpy as np
from tkinter import Tk, filedialog
from ultralytics import YOLO
from model_ocr import predict_character  # Bạn cần có sẵn file model_ocr.py với hàm predict_character
from utils import warp_polygon_to_upright  # Hàm warp polygon, tôi sẽ gửi luôn nếu bạn cần

# ================== CẤU HÌNH ===================
segment_model_path = r"D:/Bien_So/runs/segment/bien_so_seg3/weights/best.pt"
detect_model_path = r"D:/Bien_So/runs/detect/train/weights/best.pt"
ocr_model_path = r"D:/Bien_So/model_ocr.pth"

# ================== LOAD MÔ HÌNH ===================
segment_model = YOLO(segment_model_path)
detect_model = YOLO(detect_model_path)

# ================== TIỆN ÍCH ===================
def nhị_phan_hoa(anh_goc):
    gray = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2GRAY) if len(anh_goc.shape) == 3 else anh_goc
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# ================== QUY TRÌNH CHÍNH ===================
def xu_ly_bien_so(duong_dan_anh):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print("Không mở được ảnh.")
        return

    # -------- Segment biển số ---------
    results = segment_model(img)
    if len(results[0].masks.xy) == 0:
        print("Không tìm thấy biển số.")
        return

    mask_points = results[0].masks.xy[0]  # Lấy polygon đầu tiên
    mask_points = np.int32(mask_points)

    # Warp về chính diện
    warped_plate = warp_polygon_to_upright(img, mask_points)
    if warped_plate is None:
        print("Lỗi warp biển số.")
        return

    # -------- Detect ký tự trên biển số đã chính diện ---------
    detect_results = detect_model(warped_plate)
    boxes = detect_results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("Không phát hiện ký tự.")
        return

    # Sắp xếp ký tự từ trái sang phải
    boxes = sorted(boxes, key=lambda b: b[0])

    bien_so_text = ""
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        char_img = warped_plate[y1:y2, x1:x2]

        # Resize và nhị phân hóa
        char_img = cv2.resize(char_img, (28, 28))
        char_img = nhị_phan_hoa(char_img)

        # Nhận diện ký tự
        ky_tu = predict_character(char_img, ocr_model_path)
        bien_so_text += ky_tu

    # -------- Hiển thị kết quả ---------
    print(f"Biển số: {bien_so_text}")

    # Vẽ khung polygon biển số lên ảnh gốc
    cv2.polylines(img, [mask_points], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img, bien_so_text, (mask_points[0][0], mask_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Kết quả", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================== CHẠY ===================
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Chọn ảnh biển số", filetypes=[("Ảnh", "*.jpg *.png *.jpeg")])
    if path:
        xu_ly_bien_so(path)
    else:
        print("Bạn chưa chọn ảnh.")
