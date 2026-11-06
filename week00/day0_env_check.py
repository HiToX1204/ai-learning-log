import numpy as np
import cv2

print("numpy", np.__version__)
print("cv2  ", cv2.__version__)

# 1. 自己在内存里造一张"图"，不依赖任何文件
img = np.zeros((100, 200, 3), dtype=np.uint8)      # 高100 宽200 三通道
cv2.rectangle(img, (20, 20), (80, 90), (0, 255, 0), -1)   # 画一个绿色实心矩形

print("img.shape =", img.shape)
print("img.dtype =", img.dtype)

# 2. 三通道 -> 单通道灰度（这就是你 OpenCV 项目里的第一步）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray.shape =", gray.shape)

# 3. 用 NumPy 做阈值统计，不用 cv2
mask = gray > 100
print("亮像素个数 =", mask.sum())
print("亮像素占比 =", round(mask.sum() / gray.size, 4))

# 4. axis 到底指什么：沿哪个轴压缩，那个轴就消失
print("mask.sum(axis=0).shape =", mask.sum(axis=0).shape, "  <- 沿行压缩，剩下列")
print("mask.sum(axis=1).shape =", mask.sum(axis=1).shape, "  <- 沿列压缩，剩下行")
