import torch
import cv2
from tkinter import Tk, filedialog
from pathlib import Path
import shutil

# Thư mục lưu kết quả cố định
output_dir = Path('D:/Bien_So/anh_bien_so')
output_dir.mkdir(parents=True, exist_ok=True)

# Ẩn cửa sổ chính Tkinter
root = Tk()
root.withdraw()

# Hộp thoại chọn ảnh từ máy
image_path = filedialog.askopenfilename(
    title='Chọn ảnh biển số cần phân tích',
    filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')]
)

if image_path:
    print(f"Đã chọn ảnh: {image_path}")

    # Đường dẫn mô hình
    model_path = 'D:/Bien_So/LP_ocr_nano_62.pt'

    # Load mô hình từ GitHub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')

    # Đọc ảnh
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Nhận diện
    results = model(img_rgb)

    # In kết quả ra terminal
    results.print()

    # Hiển thị ảnh kết quả
    results.show()

    # Nhận diện xong lưu tạm trong runs/detect/exp
    results.save()

    # Tìm ảnh kết quả mới nhất
    latest_dir = Path('runs/detect')
    latest_exp = sorted(latest_dir.glob('exp*'), key=lambda x: x.stat().st_mtime, reverse=True)[0]

    for file in latest_exp.iterdir():
        if file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # Lưu về đúng thư mục bạn muốn, giữ tên gốc + _ketqua
            dest_path = output_dir / f"{Path(image_path).stem}_ketqua{file.suffix}"
            shutil.move(str(file), dest_path)
            print(f"Đã lưu ảnh kết quả tại: {dest_path}")

else:
    print("Bạn chưa chọn ảnh nào.")
