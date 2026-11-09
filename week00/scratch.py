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
img = np.zeros((50, 80, 3), dtype=np.uint8)
print(img.shape)          # 期望 (50, 80, 3)               # 放文件最上面也可以
cv2.rectangle(img, (10, 10), (39, 49), (0, 0, 255), -1)
print(img[10, 10])    # 应该是 [0, 0, 255]
red = img[:, :, 2]         # 抽出红层
print(red.sum())           # 顺手看看红色总量
print((red > 0).sum())     # 数有多少个像素的红色 > 0
col_counts=img[:, :, 2].sum(axis=0)  # 每一列的红色总量
print(col_counts)  
print(np.argmax(col_counts))  # 红色最多的列