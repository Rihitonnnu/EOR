import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

class EORWebcam:
    def __init__(self):
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

        # data/kawanishi/eor_base.xlsxを読み込む
        self.df = pd.read_excel('../data/kawanishi/eor_base.xlsx', sheet_name='Sheet')
        # カラムごとの平均値を計算する
        self.average = self.df.mean()

        #self.averageのlefe_eyeとright_eyeの値を取得する
        self.average_left_eye_distance = self.average[0]
        self.average_right_eye_distance = self.average[1]

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
                        left_eye_distance = left_eye_distance / left_eye_corner_distance
                        right_eye_distance = right_eye_distance / right_eye_corner_distance

                        test1=left_eye_distance/self.average_left_eye_distance
                        test2=right_eye_distance/self.average_right_eye_distance

                        # openCVのputText関数を使って、画面にtest1,test2を表示する
                        cv2.putText(frame, f'{test1:.2f}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
                        cv2.putText(frame, f'{test2:.2f}', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)


                # 画像を表示する
                cv2.imshow('MediaPipe FaceMesh', frame)

                # 'q'キーが押されたらループを終了する
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

        finally:
            # カメラを解放し、OpenCVのウィンドウを閉じる
            self.cap.release()
            cv2.destroyAllWindows()

    # def eye_corner_detection(self):
    #     if self.cap.isOpened():
    #         ret, frame = self.cap.read()

    #         # BGR画像をRGBに変換する
    #         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         # 顔のランドマークを検出する
    #         results = self.face_mesh.process(rgb_image)

    #         # 目頭と目尻のピクセル単位の距離を計算する
    #         if results.multi_face_landmarks:
    #             for face_landmarks in results.multi_face_landmarks:
    #                 h, w, c = frame.shape

    #                 # 左目の目頭と目尻の距離を計算する
    #                 left_eye_start = np.array(
    #                     [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
    #                 left_eye_end = np.array(
    #                     [face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h])
    #                 left_eye_distance = np.linalg.norm(
    #                     left_eye_start - left_eye_end)

    #                 # 右目の目頭と目尻の距離を計算する
    #                 right_eye_start = np.array(
    #                     [face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h])
    #                 right_eye_end = np.array(
    #                     [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
    #                 right_eye_distance = np.linalg.norm(
    #                     right_eye_start - right_eye_end)
                    
    #                 # 目頭と目尻の距離を返却する
    #                 return left_eye_distance, right_eye_distance

# Create an instance of the WebcamFaceMesh class and run the program
eor_webcom = EORWebcam()
eor_webcom.run()
