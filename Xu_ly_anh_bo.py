import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure
from tkinter import Tk, filedialog
import os

# ================== HÀM TÁCH KÝ TỰ CHÍNH XÁC ==================
def tach_ky_tu(anh_nhi_phan, anh_goc, thu_muc_luu):
    ket_qua = []

    anh_dao_mau = cv2.bitwise_not(anh_nhi_phan)
    anh_dao_mau = cv2.medianBlur(anh_dao_mau, 3)

    nhan = measure.label(anh_dao_mau, connectivity=2, background=0)
    chieu_cao_bien_so = anh_nhi_phan.shape[0]
    nguong_chieu_cao = chieu_cao_bien_so * 0.25  # Chỉ lấy ký tự cao trên 25% biển số

    for label in np.unique(nhan):
        if label == 0:
            continue

        mask = np.zeros(anh_dao_mau.shape, dtype="uint8")
        mask[nhan == label] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(c) / float(w * h)
            height_ratio = h / chieu_cao_bien_so

            if 0.1 < aspect_ratio < 1.2 and 0.15 < solidity <= 1.0 and 0.25 < height_ratio < 1.0 and h > nguong_chieu_cao:
                ky_tu_crop = anh_goc[y:y+h, x:x+w]
                ky_tu_nhi_phan = anh_dao_mau[y:y+h, x:x+w]

                if np.mean(ky_tu_nhi_phan) < 127:
                    ky_tu_nhi_phan = cv2.bitwise_not(ky_tu_nhi_phan)

                trang = cv2.countNonZero(ky_tu_nhi_phan)
                tong_dien_tich = ky_tu_nhi_phan.shape[0] * ky_tu_nhi_phan.shape[1]
                tile_mau = trang / tong_dien_tich

                if 0.15 < tile_mau < 0.95:
                    ket_qua.append((ky_tu_crop, x))

    ket_qua = sorted(ket_qua, key=lambda k: k[1])

    for idx, (ky_tu_mau, _) in enumerate(ket_qua):
        duong_dan = os.path.join(thu_muc_luu, f"ky_tu_{idx+1}.jpg")
        cv2.imwrite(duong_dan, ky_tu_mau)

    return [k[0] for k in ket_qua]

# ================== HÀM XỬ LÝ ẢNH BIỂN SỐ ==================
def xu_ly_bien_so(duong_dan_anh):
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        print(f"Không đọc được ảnh tại: {duong_dan_anh}")
        return

    hsv = cv2.cvtColor(anh, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)

    T = threshold_local(v_equalized, 15, offset=10, method="gaussian")
    anh_nhi_phan = (v_equalized > T).astype("uint8") * 255

    cv2.imshow("Ảnh gốc", anh)
    cv2.imshow("Ảnh nhị phân", anh_nhi_phan)

    thu_muc_luu = "D:/ky_tu_xuat_ra"
    os.makedirs(thu_muc_luu, exist_ok=True)

    ds_ky_tu = tach_ky_tu(anh_nhi_phan, anh, thu_muc_luu)

    print(f"Đã tách {len(ds_ky_tu)} ký tự. Ảnh lưu tại {thu_muc_luu}")

    for idx, ky_tu in enumerate(ds_ky_tu):
        cv2.imshow(f"Ký tự {idx+1}", ky_tu)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================== CHƯƠNG TRÌNH CHÍNH ==================
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    duong_dan_anh = filedialog.askopenfilename(title="Chọn ảnh biển số", filetypes=[("Ảnh", "*.jpg *.png *.jpeg")])

    if duong_dan_anh:
        print(f"Đã chọn ảnh: {duong_dan_anh}")
        xu_ly_bien_so(duong_dan_anh)
    else:
        print("Bạn chưa chọn ảnh, thoát chương trình.")
