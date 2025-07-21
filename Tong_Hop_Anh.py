import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import numpy as np

# Cấu hình đường dẫn mô hình và ngưỡng độ tin cậy
SEG_MODEL_PATH = "runs/segment/bien_so_seg3/weights/best.pt"
OCR_MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRESHOLD_PLATE = 0.6
CONF_THRESHOLD_CHAR = 0.5

def chon_anh_gui():
    """Mở hộp thoại để chọn ảnh từ máy tính."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh biển số",
        filetypes=[("Ảnh", "*.jpg *.png *.jpeg")]
    )
    return file_path

def ocr_bien_so(bien_so_crop, ocr_model):
    """Nhận dạng ký tự trên ảnh biển số đã cắt."""
    ocr_results = ocr_model(bien_so_crop, verbose=False)

    if not ocr_results or not ocr_results[0].boxes:
        return "", bien_so_crop

    char_list = []
    for r in ocr_results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < CONF_THRESHOLD_CHAR:
                continue

            x1_char, y1_char, x2_char, y2_char = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            label = ocr_model.names[cls_id]
            char_list.append({'label': label, 'x1': x1_char, 'y1': y1_char})

    if not char_list:
        return "", bien_so_crop

    y_coords = [char['y1'] for char in char_list]
    median_y = np.median(y_coords)

    dong_1 = sorted([c for c in char_list if c['y1'] < median_y], key=lambda c: c['x1'])
    dong_2 = sorted([c for c in char_list if c['y1'] >= median_y], key=lambda c: c['x1'])

    bien_so_text_1 = "".join([c['label'] for c in dong_1])
    bien_so_text_2 = "".join([c['label'] for c in dong_2])

    final_text = bien_so_text_1 + bien_so_text_2

    return final_text, bien_so_crop

def main():
    print("Dang tai mo hinh...")

    try:
        seg_model = YOLO(SEG_MODEL_PATH)
        ocr_model = YOLO(OCR_MODEL_PATH)
        print("Tai mo hinh thanh cong.")
    except Exception as e:
        print("Khong the tai mo hinh. Loi:", e)
        return

    duong_dan_anh = chon_anh_gui()
    if not duong_dan_anh:
        print("Chua chon anh. Ket thuc chuong trinh.")
        return

    img = cv2.imread(duong_dan_anh)
    if img is None:
        print("Khong the doc anh.")
        return

    results = seg_model(img, verbose=False)

    if not results or not results[0].boxes:
        print("Khong tim thay bien so trong anh.")
        cv2.imshow("Anh goc - Khong tim thay bien so", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < CONF_THRESHOLD_PLATE:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bien_so_crop = img[y1:y2, x1:x2].copy()

        bien_so_text, _ = ocr_bien_so(bien_so_crop, ocr_model)

        if bien_so_text:
            print("Bien so nhan dang duoc:", bien_so_text)

            # Ve khung bao quanh bien so
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Tinh toan kich thuoc text
            label_size, _ = cv2.getTextSize(bien_so_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            y1_label = max(y1, label_size[1] + 10)

            # Ve nen text
            cv2.rectangle(img, (x1, y1_label - label_size[1] - 10),
                          (x1 + label_size[0], y1_label - 10), (0, 255, 0), cv2.FILLED)

            # Ve text len anh
            cv2.putText(img, bien_so_text, (x1, y1_label - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hien thi ket qua
    cv2.imshow("Ket qua nhan dang bien so", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
