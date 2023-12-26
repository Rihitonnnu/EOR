import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp

# MediaPipe FaceMeshの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# カメラの初期化
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR画像をRGBに変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔のランドマークを検出
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                # 目頭と目尻のランドマークの3次元座標を取得
                # 左目の目尻
                if id == 33:
                    x0, y0, z0 = lm.x, lm.y, lm.z
                    # ランドマークを描画
                    h, w, c = frame.shape
                    cv2.circle(frame, (int(x0 * w), int(y0 * h)), 1, (0, 0, 255), -1)
                    # 座標をopencvでウィンドウに表示
                    # cv2.putText(frame, f'({x0:.3f}, {y0:.3f}, {z0:.3f})', (int(x0 * w), int(y0 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # print(f'Landmark {id}: ({x0}, {y0}, {z0})')
                
                # 右目の目尻
                # elif id == 263:
                #     x1, y1, z1 = lm.x, lm.y, lm.z
                #     # ランドマークを描画
                #     h, w, c = frame.shape
                #     cv2.circle(frame, (int(x1 * w), int(y1 * h)), 1, (0, 0, 255), -1)
                #     # 座標をopencvでウィンドウに表示
                #     # cv2.putText(frame, f'({x1:.3f}, {y1:.3f}, {z1:.3f})', (int(x1 * w), int(y1 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                #     # print(f'Landmark {id}: ({x1}, {y1}, {z1})')
                
                # 左目の目頭
                elif id == 133:
                    x2, y2, z2 = lm.x, lm.y, lm.z
                    # ランドマークを描画
                    h, w, c = frame.shape
                    cv2.circle(frame, (int(x2 * w), int(y2 * h)), 1, (0, 0, 255), -1)
                    # 座標をopencvでウィンドウに表示
                    # cv2.putText(frame, f'({x2:.3f}, {y2:.3f}, {z2:.3f})', (int(x2 * w), int(y2 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # print(f'Landmark {id}: ({x2}, {y2}, {z2})')
                
                #　x0,y0,z0とx2,y2,z2のユークリッド距離を計算
                
                # # 右目の目頭
                # elif id == 362:
                #     x3, y3, z3 = lm.x, lm.y, lm.z
                #     # ランドマークを描画
                #     h, w, c = frame.shape
                #     # cv2.circle(frame, (int(x3 * w), int(y3 * h)), 1, (0, 0, 255), -1)
                #     # 座標をopencvでウィンドウに表示
                #     # cv2.putText(frame, f'({x3:.3f}, {y3:.3f}, {z3:.3f})', (int(x3 * w), int(y3 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                #     # print(f'Landmark {id}: ({x3}, {y3}, {z3})')

    # 画像を表示
    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
