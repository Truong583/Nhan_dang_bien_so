import os
import random
import shutil #Dùng để copy file


data_goc = r'D:/Bien_So/Character dataset'
data_huan_luyen = r'D:/Bien_So/char_data'
train_ty_le = 0.8

os.makedirs(f"{data_huan_luyen}/train",exist_ok=True) #Thư mục tồn tại thì bỏ qua ko , báo lỗi
os.makedirs(f"{data_huan_luyen}/val",exist_ok=True)

for label in os.listdir(data_goc):
    src_dir = os.path.join(data_goc, label)
    imgs = os.listdir(src_dir)
    random.shuffle(imgs) #Xáo trộn ngẫu nhiên

    tach_idx = int(len(imgs) * train_ty_le)
    train_imgs = imgs[:tach_idx]
    val_imgs = imgs[tach_idx:]

    os.makedirs(f"{data_huan_luyen}/train/{label}", exist_ok=True)
    os.makedirs(f"{data_huan_luyen}/val/{label}", exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(data_huan_luyen,'train',label, img))

    for img in val_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(data_huan_luyen,'val',label, img))

print("Đã xong , được lưu tại D:/Bien_So/char_data ")
