import eor_webcam
import os
import datetime

name='kawanishi'

# 現在の日時を取得する
now = datetime.datetime.now()

try:
    # 現在の年月日を含めてディレクトリを作成する
    os.mkdir('../data/{}/{}'.format(name,now.strftime('%Y%m%d')))
except FileExistsError:
    pass

eor_webcam.EORWebcam(name).run()
