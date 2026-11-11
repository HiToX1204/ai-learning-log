import numpy as np
import cv2

img = np.zeros((100, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
img[50, 100] = [0, 0, 255]        # 正中间一个红点
print("img.shape =", img.shape)
# ↓ 下面四步你写
img2=img.transpose(2, 0, 1)
img3=img2.transpose(1, 2, 0)
print("img2.shape =", img2.shape)
print("img3.shape =", img3.shape)
print(np.array_equal(img, img3))