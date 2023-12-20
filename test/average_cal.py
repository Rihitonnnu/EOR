import pandas as pd

# Excelファイルを読み込む
df = pd.read_excel('../data/kawanishi/eor_base.xlsx', sheet_name='Sheet')

# カラムごとの平均値を計算する
average = df.mean()

print(average)
