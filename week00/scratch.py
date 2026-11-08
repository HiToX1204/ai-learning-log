# import numpy as np
# a=np.array([[1,2,3],[4,5,6],])
# print(a.shape)
# print (a.sum(axis=0))
# print(a.sum(axis=1))
# print(a[:,2])
# img=np.zeros((2,3,3), dtype=np.uint8)
# img[0,0]=[10,20,30]
# img[1,2]=[0,0,255]
# print(img[0,0,2])
# print(img[:,:,2].shape)
# print(img[:,:,2])
import cv2
import numpy as np
img = np.full((60, 100, 3), 255, dtype=np.uint8)   # 60行 100列 白图
cv2.rectangle(img, (10, 10), (40, 30), (255, 0, 0), -1)   # 蓝色实心矩形
print(img[20, 20])      # 看第20行第20列那个像素的三个数
img = np.zeros((50, 80,3), dtype=np.uint8)
print(img.shape)          # 期望 (50, 80, 3)
import cv2                 # 放文件最上面也可以
cv2.rectangle(img, (10, 10), (39, 49), (0, 0, 255), -1)
red = img[:, :, 2]         # 抽出红层
print(red.sum())           # 顺手看看红色总量
print((red > 0).sum())     # 数有多少个像素的红色 > 0
