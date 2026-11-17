import cv2
import numpy as np

# 1) 三块灰度的原图
img = np.zeros((100, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
cv2.rectangle(img, (70, 10), (100, 40), (150, 150, 150), -1)
cv2.rectangle(img, (130, 10), (160, 40), (100, 100, 100), -1)

# 2) 渐变系数：拆成三步
coefficients = np.linspace(0.4, 1.3, 200)      # (200,)
g = coefficients[None, :, None]                # (1, 200, 1)
img2 = np.clip(img * g, 0, 255).astype(np.uint8)

# 3) 对比
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print("原图 (gray1 > 128).sum()  =", (gray1 > 128).sum())
print("渐变后 (gray2 > 128).sum() =", (gray2 > 128).sum())
print("亮块左端/右端 =", gray2[20, 10], gray2[20, 40])