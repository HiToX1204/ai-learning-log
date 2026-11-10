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
edges1 = cv2.Canny(gray1, 30, 90)
edges2 = cv2.Canny(gray2, 30, 90)

print("原图边缘像素 =", (edges1 > 0).sum())
print("渐变图边缘像素 =", (edges2 > 0).sum())