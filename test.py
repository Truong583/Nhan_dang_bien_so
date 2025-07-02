import cv2

# Đường dẫn ảnh và file nhãn
image_path = r'D:\Bien_So\OCR\images\train\0.jpg'  # sửa tên ảnh phù hợp
label_path = r'D:\Bien_So\OCR\labels\train\0.txt'  # sửa tên nhãn phù hợp

# Bảng tên lớp theo đúng file YAML
names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']

# Đọc ảnh
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Đọc file nhãn
with open(label_path, 'r') as f:
    lines = f.readlines()

# Vẽ từng nhãn
for line in lines:
    class_id, x_center, y_center, width, height = map(float, line.strip().split())
    # Chuyển về tọa độ góc
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # Lấy ký tự tương ứng
    label = names[int(class_id)]

    # Vẽ khung và nhãn
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Hiển thị ảnh
cv2.imshow('Label Check', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
