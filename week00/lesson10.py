import numpy as np
img=np.random.randint(0,256,(100,200),dtype=np.uint8)
col_mean = img.mean(axis=0)      # 形状 (200,)
print(col_mean.shape)
col_max = col_mean.argmax()        # 形状 (200,)
print(col_max)
print((img>128).mean())