import cv2
import torch
from ultralytics import YOLO
from tkinter import Tk, filedialog
import numpy as np

# ------------------------- CẤU HÌNH -------------------------
seg_model_path = r'D:/Bien_So/runs/segment/bien_so_seg3/weights/best.pt'
ocr_model_path = r'D:/Bien_So/runs/detect/train/weights/best.pt'

# ------------------------- BẢNG ÁNH XẠ NHẦM (BÊN TRÁI -> NHÃN HIỂN THỊ, BÊN PHẢI -> KÍ TỰ THỰC TẾ) -------------------------
# Theo quy luật bạn cung cấp
label_mapping = {
    0: 'T', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8',
    10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H',
    19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'S', 25: 'T', 26: 'U', 27: 'V',
    28: 'X', 29: 'Y', 30: 'Z'
}

# Hàm ánh xạ chính xác
def map_label(cls_id):
    return label_mapping.get(cls_id, '?')

# ------------------------- CHỌN ẢNH -------------------------
Tk().withdraw()
img_path = filedialog.askopenfilename(title="Chọn ảnh chứa biển số")
if not img_path:
    print("Không chọn ảnh.")
    exit()

img = cv2.imread(img_path)
if img is None:
    print("Không đọc được ảnh.")
    exit()

# ------------------------- NHẬN DIỆN BIỂN SỐ -------------------------
seg_model = YOLO(seg_model_path)
results = seg_model(img, conf=0.5)

plates = []
for mask in results[0].masks.xy:
    pts = np.array(mask, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    plate_crop = img[y:y+h, x:x+w]
    plates.append((plate_crop, (x, y, w, h)))

if not plates:
    print("Không tìm thấy biển số.")
    exit()

# ------------------------- NHẬN DẠNG KÝ TỰ -------------------------
ocr_model = YOLO(ocr_model_path)

for idx, (plate_crop, (px, py, pw, ph)) in enumerate(plates):
    ocr_res = ocr_model(plate_crop, conf=0.25)[0]
    chars = []

    if ocr_res.boxes is None or len(ocr_res.boxes.cls) == 0:
        print(f"Không phát hiện ký tự ở biển số {idx+1}")
        continue

    for box, cls in zip(ocr_res.boxes.xyxy.cpu(), ocr_res.boxes.cls.cpu()):
        x1, y1, x2, y2 = map(int, box)
        char = map_label(int(cls))
        chars.append((x1, char, (x1, y1, x2, y2)))

    chars.sort()  # Sắp xếp từ trái qua phải
    plate_str = ''.join([c[1] for c in chars])

    # Vẽ ký tự lên biển số cắt ra
    for _, char, (x1, y1, x2, y2) in chars:
        cv2.rectangle(plate_crop, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(plate_crop, char, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị kết quả lên ảnh chính
    cv2.putText(img, plate_str, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
    print(f"[Biển số {idx+1}] Kết quả: {plate_str}")

# ------------------------- HIỂN THỊ KẾT QUẢ -------------------------
cv2.imshow("Kết quả nhận dạng biển số", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
