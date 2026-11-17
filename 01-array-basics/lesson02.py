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
# x = np.array([10, 20, 30])
# print(x[None, :, None].shape)

# a = np.ones((2, 3))
# print(a * np.array([10, 20, 30]))
# ── 2. 加一层"左暗右亮"的光照 ─────────────────────────
grad  = np.linspace(0.4, 1.3, 200)                       # (200,) 每列的亮度系数
img2  = np.clip(img * grad[None, :, None], 0, 255).astype(np.uint8)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ── 3. 对比：同样阈值 128，加渐变前后 ──────────────────
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("加渐变前 (gray1 > 128).sum() =", (gray1 > 128).sum())
print("加渐变后 (gray2 > 128).sum() =", (gray2 > 128).sum())
print("亮块左端 / 右端灰度 =", gray2[20, 10], gray2[20, 40])
print("img2.shape =", img2.shape)