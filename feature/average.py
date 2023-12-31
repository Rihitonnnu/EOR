import pandas as pd


df = pd.read_excel('../data/kawanishi/eor.xlsx', sheet_name='Sheet')
# カラムごとの平均値を計算する
average = df.mean()

#averageのlefe_eyeとright_eyeの値を取得する
average_left_eye = average.iloc[0]
average_right_eye = average.iloc[1]

print(average_left_eye,average_right_eye)
