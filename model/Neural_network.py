# import torch
import pandas as pd


file_path = '../trots_2013-2022.parquet'

# 读取Parquet文件
df = pd.read_parquet(file_path)

# 查看DataFrame的前几行
print(df.head())
