import cv2
import face_recognition

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 捕获一帧视频
    ret, frame = video_capture.read()

    # 将视频帧转换成RGB色彩（face_recognition使用RGB）
    rgb_frame = frame[:, :, ::-1]

    # 查找视频帧中所有面部的位置
    face_locations = face_recognition.face_locations(rgb_frame)

    # 在每个面部周围绘制一个方框
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
video_capture.release()
cv2.destroyAllWindows()
