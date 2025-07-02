import cv2
import numpy as np

def warp_polygon_to_upright(img, polygon):
    """
    Warp vùng polygon (dạng 4 điểm) về chính diện để dễ xử lý
    :param img: Ảnh gốc
    :param polygon: List 4 điểm [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    :return: Ảnh sau khi warp chính diện
    """
    if len(polygon) != 4:
        print("Cần đúng 4 điểm để warp.")
        return None

    pts_src = np.array(polygon, dtype=np.float32)

    # Tính kích thước đầu ra (giả sử biển số có thể hình chữ nhật tự do)
    width_top = np.linalg.norm(pts_src[0] - pts_src[1])
    width_bottom = np.linalg.norm(pts_src[2] - pts_src[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(pts_src[0] - pts_src[3])
    height_right = np.linalg.norm(pts_src[1] - pts_src[2])
    max_height = int(max(height_left, height_right))

    # Điểm đích sau khi warp
    pts_dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Ma trận biến đổi
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped
