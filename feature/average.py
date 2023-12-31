import pandas as pd


df = pd.read_excel('../data/kawanishi/20231231/eor_16.xlsx', sheet_name='Sheet')

#excelのカラムから110を超える値を削除する
df = df[df['left_eye_opening_rate'] < 110]
df = df[df['right_eye_opening_rate'] < 110]

# カラムごとの平均値を計算する
average = df.mean()

#averageのlefe_eyeとright_eyeの値を取得する
average_left_eye = average.iloc[0]
average_right_eye = average.iloc[1]

print(average_left_eye,average_right_eye)
