import pandas as pd

# df = pd.DataFrame({
#     "frame":      [1, 2, 3, 4, 5, 6],
#     "brightness": [180, 92, 210, 65, 150, 88],
#     "detected":   [True, False, True, False, True, False],
# })

# # print(df)
# # print("shape =", df.shape)
# # print("columns =", list(df.columns))
# # print(df["brightness"].mean())
# # print(df[df["detected"]])
# # print(df["brightness"].max())
# df["lighting"] = ["day", "night", "day", "night", "day", "night"]

# # print(df.groupby("lighting")["brightness"].mean())
# # print(df.groupby("lighting")["detected"].mean())
# df.to_csv("week00/demo.csv", index=False)
# df2 = pd.read_csv("week00/demo.csv")

# print(df2)
# print(df2.dtypes)
# print(df2.isnull().sum())
# 造 12 行数据：3 种光照 × 4 帧，brightness 和 detected 你自己定
df = pd.DataFrame({
    "frame":      range(12),
    "lighting":   ["day","dusk","night"] * 4,
    "brightness": [180, 92, 100, 200, 150, 88, 120, 75, 88, 194, 160, 95],      # 你填
    "detected":   [True, True, False, True, True, False, True, False, False, True, False, False],      # 你填，让 day 全 True、night 全 False、dusk 一半
})
print(df)
print(df.groupby("lighting")[ "brightness" ].mean())
print(df.groupby("lighting")[ "detected" ].mean())
print(df.groupby("lighting")[ "detected" ].mean().idxmin())
df.to_csv("week00/demo2.csv", index=False)
df2 = pd.read_csv("week00/demo2.csv")
print(df2.shape[0] == df.shape[0])
# 1. 按 lighting 分组，算 brightness 均值和 detected 检出率
# 2. 找出检出率最低的那一组
# 3. 存成 CSV（不写索引），再读回来，确认读回来的行数一致