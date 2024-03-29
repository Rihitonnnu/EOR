import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import socket
import struct
import openpyxl
import datetime

class EORWebcam:
    def __init__(self,name):
        # ファイル作成のための名前
        self.name=name

        self.now = datetime.datetime.now()

        self.cnt=0

        # MediaPipeのFaceMeshモデルを初期化する
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

        # MediaPipeのDrawingSpec（描画設定）を初期化する
        mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # カメラを開く
        self.cap = cv2.VideoCapture(0)

        # 現在のカメラの解像度を取得
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'frame_width: {self.frame_width}')
        print(f'frame_height: {self.frame_height}')

        # data/hoge/eor_base.xlsxを読み込む
        self.df_base = pd.read_excel('../data/{}/eor_base.xlsx'.format(self.name), sheet_name='Sheet')
        # カラムごとの平均値を計算する
        self.average_base = self.df_base.mean()

        #self.averageのlefe_eyeとright_eyeの値を取得する
        self.average_left_eye_base = self.average_base.iloc[0]
        self.average_right_eye_base = self.average_base.iloc[1]

        self.df_close = pd.read_excel('../data/{}/pfh_close.xlsx'.format(self.name), sheet_name='Sheet')
        # カラムごとの平均値を計算する
        self.average_close = self.df_close.mean()

        #self.averageのlefe_eyeとright_eyeの値を取得する
        self.average_left_eye_close = self.average_close.iloc[0]
        self.average_right_eye_close = self.average_close.iloc[1]

        #excelファイルを作成し、1行目にcntとleft_eye_opening_rateとright_eye_opening_rateを書き込む
        self.wb = openpyxl.Workbook()
        self.sheet = self.wb.active
        self.sheet['A1'] = 'cnt'
        self.sheet['B1'] = 'left_eye_opening_rate'
        self.sheet['C1'] = 'right_eye_opening_rate'

        # 動画ファイルの保存設定
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # mp4ファイルで出力
        self.output_file = cv2.VideoWriter('../data/{}/video/{}.mp4'.format(self.name,self.now.strftime('%Y%m%d%H%M%S')), fourcc, 30.0, (640, 480))
    
    def udp_send(self,data,server_ip,server_port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # dataの中身はboolean型なので、int型に変換する
        sock.sendto(struct.pack('?',data), (server_ip, server_port))

    # EORがしきい値を超えているかどうかの判定
    def eor_judge(self,path):
        df = pd.read_excel(path, sheet_name='Sheet')
        #excelのカラムから110を超える値を削除する
        df = df[df['left_eye_opening_rate'] < 105]
        df = df[df['right_eye_opening_rate'] < 105]

        # カラムごとの平均値を計算する
        average = df.mean()

        #averageのlefe_eyeとright_eyeの値を取得する
        average_left_eye = average.iloc[0]
        average_right_eye = average.iloc[1]

        print(average_left_eye,average_right_eye)

        if average_left_eye <= 80 or average_right_eye <= 80:
            return True
        
        return False

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # BGR画像をRGBに変換する
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 顔のランドマークを検出する
                results = self.face_mesh.process(rgb_image)

                # 検出されたランドマークを描画し、まぶたの距離を計算する
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w, c = frame.shape
                        for id, lm in enumerate(face_landmarks.landmark):
                            # 上まぶたと下まぶたのランドマークを描画する
                            if id in [159, 145, 386, 374]:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                # 線で描画する
                                cv2.line(frame, (cx - 20, cy),
                                        (cx + 20, cy), (0, 255, 0), 1)

                        # 左目の上まぶたと下まぶたの距離を計算する
                        left_eye_top = np.array(
                            [face_landmarks.landmark[159].x * w, face_landmarks.landmark[159].y * h])
                        left_eye_bottom = np.array(
                            [face_landmarks.landmark[145].x * w, face_landmarks.landmark[145].y * h])
                        left_eye_distance = np.linalg.norm(
                            left_eye_top - left_eye_bottom)

                        # 右目の上まぶたと下まぶたの距離を計算する
                        right_eye_top = np.array(
                            [face_landmarks.landmark[386].x * w, face_landmarks.landmark[386].y * h])
                        right_eye_bottom = np.array(
                            [face_landmarks.landmark[374].x * w, face_landmarks.landmark[374].y * h])
                        right_eye_distance = np.linalg.norm(
                            right_eye_top - right_eye_bottom)
                        
                        # 左目の目頭と目尻の距離を計算する
                        left_eye_start = np.array(
                            [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
                        left_eye_end = np.array(
                            [face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h])
                        left_eye_corner_distance = np.linalg.norm(
                            left_eye_start - left_eye_end)

                        # 右目の目頭と目尻の距離を計算する
                        right_eye_start = np.array(
                            [face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h])
                        right_eye_end = np.array(
                            [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
                        right_eye_corner_distance = np.linalg.norm(
                            right_eye_start - right_eye_end)

                        # 目頭と目尻の距離を1としたときの上まぶたと下まぶたの距離を計算する
                        # left_eye_distance = left_eye_distance / left_eye_corner_distance
                        # right_eye_distance = right_eye_distance / right_eye_corner_distance

                        # print((left_eye_distance-self.average_left_eye_close)/left_eye_corner_distance/(self.average_left_eye_base-self.average_left_eye_close))
                        print('左は{}'.format((left_eye_distance-self.average_left_eye_close)/left_eye_corner_distance))

                        a=(left_eye_distance-self.average_left_eye_close)/left_eye_corner_distance
                        b=(self.average_left_eye_base-self.average_left_eye_close)/left_eye_corner_distance
                        # 開眼率の計算(任意時間の開眼度-開眼時の開眼度/覚醒時の開眼度-閉眼時の開眼度*100)
                        left_eye_opening_rate=a/b

                        right_eye_opening_rate=right_eye_distance/self.average_right_eye_base*100

                        # openCVのputText関数を使って、画面に左右の開眼率を表示する
                        cv2.putText(frame, f'{left_eye_opening_rate*100:.2f}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
                        cv2.putText(frame, f'{right_eye_opening_rate:.2f}', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
                        
                        # excelファイルに開眼率を書き込む
                        self.sheet.append([self.cnt,left_eye_opening_rate,right_eye_opening_rate])

                        # 日時をファイル名にする
                        # self.wb.save('../data/{}/{}/eor_{}.xlsx'.format(self.name,now.strftime('%Y%m%d'),now.strftime('%H%m%s')))

                        self.cnt+=1

                # 画像を表示する
                cv2.imshow('MediaPipe FaceMesh', frame)
                self.output_file.write(frame)

                # 1分間経過したらもしくは'q'キーが押されたらループを終了する
                if cv2.waitKey(5) & 0xFF == ord('q') or self.cnt==1800:
                    path='../data/{}/{}/{}.xlsx'.format(self.name,self.now.strftime('%Y%m%d'),self.now.strftime('%H%M%S'))
                    self.wb.save(path)

                    is_sleepy=self.eor_judge(path)
                    print(is_sleepy)
                    self.udp_send(is_sleepy,'127.0.0.1',2002)

                    # self.sheetの中身をリセットする
                    self.wb = openpyxl.Workbook()
                    self.sheet = self.wb.active
                    self.sheet['A1'] = 'cnt'
                    self.sheet['B1'] = 'left_eye_opening_rate'
                    self.sheet['C1'] = 'right_eye_opening_rate'

                    # 眠くなったらループを終了する
                    if is_sleepy:
                        break

                    self.cnt=0
        
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

        finally:
            # カメラを解放し、OpenCVのウィンドウを閉じる
            self.cap.release()
            self.output_file.release()
            cv2.destroyAllWindows()

# Create an instance of the WebcamFaceMesh class and run the program
# eor_webcom = EORWebcam('kawanishi')
# eor_webcom.run()

