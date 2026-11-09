import numpy as np
import cv2

img = np.zeros((100, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
cv2.rectangle(img, (70, 10), (100, 40), (150, 150, 150), -1)
cv2.rectangle(img, (130, 10), (160, 40), (100, 100, 100), -1)
arr = np.array([[1, 2, 3], [4, 5, 6]])
mask = arr > 3
print(mask)
print(arr[mask])  # 输出大于3的元素
print(arr[mask].mean())
print(arr[mask].shape)
print(arr[mask].size)
print(img[20, 20])
print(img[20, 80])
print(img[20, 140])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray[20, 20],gray[20, 80], gray[20, 140])
count=(gray > 128).sum()
print(count)
for t in range(0, 256, 32):
    count = (gray > t).sum()
    ratio = count / gray.size
    print(f"阈值 {t:3d}: {count:6d} 个像素, 占比 {ratio:.3f}")
    t_otsu, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu 选的阈值:", t_otsu)
print("二值化后亮的像素:", (binary > 0).sum())