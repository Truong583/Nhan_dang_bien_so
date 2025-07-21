import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import math
import time
import re

SEG_MODEL_PATH = "runs/segment/bien_so_seg3/weights/best.pt"
OCR_MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRESHOLD_PLATE = 0.7  # Giảm ngưỡng để phát hiện tốt hơn trong điều kiện mờ
CONF_THRESHOLD_CHAR = 0.7  # Giảm ngưỡng cho OCR

# Tạo thư mục output nếu chưa có
import os

# ===== Tiền xử lý ảnh để xử lý ảnh mờ =====
def preprocess_image(image):
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tăng cường độ tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Giảm nhiễu
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # Tăng cường cạnh
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    # Chuyển lại sang ảnh màu
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# ===== Hàm xoay ảnh căn chỉnh biển số (cải tiến cho ảnh mờ) =====
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tăng cường cạnh bằng Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Tìm góc xoay bằng Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta * 180 / np.pi) - 90
            break
    else:
        # Nếu Hough Transform thất bại, dùng minAreaRect
        coords = cv2.findNonZero(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        if coords is not None:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ===== Hậu xử lý định dạng biển số Việt Nam =====
def validate_and_correct_bien_so(text):
    text = text.replace(" ", "").replace(".", "")
    # Sửa lỗi OCR phổ biến
    corrections = {"B": "8", "O": "0", "I": "1", "Z": "2", "S": "5"}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    # Kiểm tra định dạng
    pattern = r"^\d{2}[A-Z]\d{3,5}$"
    if re.match(pattern, text):
        if len(text) <= 8:
            return text[:3] + "-" + text[3:]  # Ví dụ: 51A-12345
        else:
            return text[:3] + "-" + text[3:6] + "." + text[6:]  # Ví dụ: 51A-123.45
    return text


# ===== Hàm OCR Biển số =====
def ocr_bien_so(image, ocr_model):
    results = ocr_model(image, verbose=False)
    if not results or not results[0].boxes:
        return ""

    char_list = []
    for r in results:
        for box in r.boxes:
            if float(box.conf) < CONF_THRESHOLD_CHAR:
                continue
            x1, y1, _, _ = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            label = ocr_model.names[cls_id]
            char_list.append({'label': label, 'x1': x1, 'y1': y1})

    if not char_list:
        return ""

    # Sắp xếp ký tự theo tọa độ x (phù hợp với biển số một hoặc hai dòng)
    sorted_chars = sorted(char_list, key=lambda c: c['x1'])
    bien_so_text = "".join([c['label'] for c in sorted_chars])
    return validate_and_correct_bien_so(bien_so_text)


# ===== Hàm xử lý từng khung hình =====
def process_frame(frame, seg_model, ocr_model, best_conf, best_frame):
    # Tiền xử lý ảnh để cải thiện chất lượng
    frame_processed = preprocess_image(frame)
    results = seg_model(frame_processed, verbose=False)
    if not results or not results[0].boxes:
        return frame, best_conf, best_frame

    for box in results[0].boxes:
        if float(box.conf) < CONF_THRESHOLD_PLATE:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bien_so_crop = frame[y1:y2, x1:x2].copy()
        bien_so_crop = preprocess_image(bien_so_crop)  # Tiền xử lý vùng biển số
        bien_so_crop = deskew(bien_so_crop)

        bien_so_text = ocr_bien_so(bien_so_crop, ocr_model)

        # Vẽ khung và hiển thị kết quả
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if bien_so_text:
            label_size, _ = cv2.getTextSize(bien_so_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, bien_so_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return frame, best_conf, best_frame


# ===== Main toàn bộ quy trình =====
def main():
    print("Đang tải model...")
    seg_model = YOLO(SEG_MODEL_PATH)
    ocr_model = YOLO(OCR_MODEL_PATH)
    print("Tải model thành công!")

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not video_path:
        print("Bạn chưa chọn video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    print("Bắt đầu xử lý... Nhấn 'q' để thoát.")
    best_conf = 0
    best_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        frame_count += 1
        if frame_count % 3 == 0:  # Xử lý mỗi khung thứ 3 để tăng tốc
            frame, best_conf, best_frame = process_frame(frame, seg_model, ocr_model, best_conf, best_frame)

        cv2.imshow("Nhận diện biển số", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Giảm waitKey để tăng tốc
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn thành.")


if __name__ == "__main__":
    main()