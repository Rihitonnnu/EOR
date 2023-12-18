import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのFaceMeshモデルを初期化する
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# MediaPipeのDrawingSpec（描画設定）を初期化する
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# カメラを開く
cap = cv2.VideoCapture(0)

# 現在のカメラの解像度を取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'frame_width: {frame_width}')
print(f'frame_height: {frame_height}')

# カメラの解像度を設定
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # BGR画像をRGBに変換する
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 顔のランドマークを検出する
        results = face_mesh.process(rgb_image)

        # 目頭と目尻のランドマークを描画する
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape
                for id, lm in enumerate(face_landmarks.landmark):
                    # 目頭と目尻のランドマークを描画する
                    if id in [33, 133, 362, 263]:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # 丸で描画する
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), 1)

        # 画像を表示する
        cv2.imshow('MediaPipe FaceMesh', frame)

        # 'q'キーが押されたらループを終了する
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"エラーが発生しました: {str(e)}")

finally:
    # カメラを解放し、OpenCVのウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
