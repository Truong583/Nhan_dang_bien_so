import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Tắt GPU, bắt PaddleOCR chạy CPU

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tkinter import Tk, filedialog

def process_plate(crop):
    """Xử lý ảnh biển số: chuyển sang grayscale, lọc nhiễu, cân bằng histogram, nhị phân."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def correct_rotation(mask, crop):
    """Xoay thẳng ảnh biển số theo góc từ contour của mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        rect = cv2.minAreaRect(contours[0])
        angle = rect[-1]
        if angle < -45:
            angle += 90
        h, w = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return crop

def main():
    # 1. Chọn file ảnh
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title='Chọn ảnh biển số')

    if not file_path:
        print("Chưa chọn ảnh, thoát.")
        return

    # 2. Load mô hình YOLOv8 segmentation (đường dẫn thay đổi theo của bạn)
    model = YOLO('runs/segment/bien_so_seg3/weights/best.pt')

    # 3. Chạy dự đoán trên ảnh
    results = model(file_path)[0]

    # 4. Đọc ảnh gốc
    image = cv2.imread(file_path)
    annotated_image = image.copy()

    # 5. Khởi tạo OCR PaddleOCR chạy trên CPU
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    # 6. Duyệt qua từng mask và bbox của biển số
    for mask, box in zip(results.masks.data, results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]

        # Chuyển mask từ tensor sang numpy, resize về crop
        mask_np = mask.cpu().numpy() * 255
        mask_resized = cv2.resize(mask_np, (crop.shape[1], crop.shape[0]))
        mask_bin = mask_resized.astype(np.uint8)

        # Xoay thẳng biển số
        aligned = correct_rotation(mask_bin, crop)

        # Xử lý ảnh biển số (nhị phân)
        processed = process_plate(aligned)

        # OCR nhận diện
        result = ocr.predict(processed)

        text = ""
        conf_total = 0
        count = 0
        for line in result:
            for word in line:
                text += word[1][0] + " "
                conf_total += word[1][1]
                count += 1
        conf_avg = (conf_total / count) if count > 0 else 0

        # Vẽ bbox và ghi kết quả lên ảnh
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{text.strip()} ({conf_avg*100:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 7. Hiển thị ảnh kết quả
    cv2.imshow("Biển số và OCR", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
