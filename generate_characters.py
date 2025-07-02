from PIL import Image, ImageDraw, ImageFont
import os


def generate_character_images(char, output_dir, num_images=50, font_path="arial.ttf", size=(28, 28)):
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.truetype(font_path, size=20)  # Điều chỉnh size font cho phù hợp

    for i in range(num_images):
        # Tạo ảnh nền đen
        image = Image.new('L', size, color=0)  # 'L' là grayscale, 0 là đen
        draw = ImageDraw.Draw(image)

        # Lấy kích thước văn bản bằng textbbox
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        text_width = right - left
        text_height = bottom - top

        # Vẽ ký tự trắng
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        draw.text((x, y), char, fill=255, font=font)  # 255 là trắng

        # Lưu ảnh
        image.save(os.path.join(output_dir, f"{char}_{i:03d}.png"))


# Tạo ảnh bổ sung cho các lớp M, Y, Z
train_dir = r"D:\Bien_So\char_data\train"
val_dir = r"D:\Bien_So\char_data\val"
font_path = r"C:\Windows\Fonts\arial.ttf"  # Thay bằng font biển số Việt Nam nếu có

for char in ['M', 'Y', 'Z']:
    generate_character_images(char, os.path.join(train_dir, char), num_images=50)
    generate_character_images(char, os.path.join(val_dir, char), num_images=15)