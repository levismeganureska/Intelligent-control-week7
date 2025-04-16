import cv2
import numpy as np

# Inisialisasi StereoBM
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # numDisparities harus kelipatan 16

def compute_depth(left_image, right_image):
    """Menghitung peta kedalaman dari dua gambar stereo"""
    grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Hitung peta kedalaman (disparity map)
    disparity = stereo.compute(grayL, grayR)

    # Normalisasi untuk ditampilkan
    disp = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp = np.uint8(disp)

    return disp

# Capture video dari dua kamera
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture(1)

if not cam_left.isOpened() or not cam_right.isOpened():
    print("Gagal membuka salah satu kamera.")
    exit()

while True:
    retL, frameL = cam_left.read()
    retR, frameR = cam_right.read()

    if not retL or not retR:
        break

    # Hitung kedalaman (depth map)
    depth_map = compute_depth(frameL, frameR)

    # Tampilkan hasil
    cv2.imshow("Kamera Kiri", frameL)
    cv2.imshow("Kamera Kanan", frameR)
    cv2.imshow("Peta Kedalaman", depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()