import cv2
import torch
import numpy as np
import mediapipe as mp
import os

# 初始化 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 视频输入
video_path = 'C:/Users/15811/Desktop/123/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频输出设置
output_path = 'C:/Users/15811/Desktop/123/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# 骨架数据保存路径
skeleton_output_dir = 'C:/Users/15811/Desktop/skeleton/train/1'
if not os.path.exists(skeleton_output_dir):
    os.makedirs(skeleton_output_dir)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每 8 帧处理一次
    if frame_idx % 8 == 0:
        # 只处理视频的下半部分，同时忽略左边300像素和右边300像素
        lower_half = frame[frame_height//2:, 300:frame_width-300]

        # 使用 YOLOv5 检测人物
        results = model(lower_half)
        detections = results.xyxy[0]  # 获取检测结果

        # 计算距离视频顶部最近的人物
        min_distance = float('inf')
        nearest_person = None

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf > 0.5:  # class 0 is for person in COCO dataset, conf > 0.5
                distance = y1  # 距离视频顶部的距离
                if distance < min_distance:
                    min_distance = distance
                    nearest_person = (int(x1), int(y1), int(x2), int(y2))

        # 如果没有找到最近的人物，框出可以框出的人
        if not nearest_person and len(detections) > 0:
            detection = detections[0]
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf > 0.2:  # class 0 is for person in COCO dataset, conf > 0.2
                nearest_person = (int(x1), int(y1), int(x2), int(y2))

        if nearest_person:
            x1, y1, x2, y2 = nearest_person
            # 因为只处理下半部分且忽略左右两边，需要将坐标还原到原始帧的位置
            x1 += 300
            x2 += 300
            y1 += frame_height // 2
            y2 += frame_height // 2
            person_roi = frame[y1:y2, x1:x2]

            # 使用 MediaPipe 检测骨骼
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)

            if result.pose_landmarks:
                landmarks = []
                for landmark in result.pose_landmarks.landmark:
                    x = int(x1 + landmark.x * (x2 - x1))
                    y = int(y1 + landmark.y * (y2 - y1))
                    landmarks.append([x, y])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # 保存骨架数据
                landmarks = np.array(landmarks)
                # 添加一维度使其成为 (num_landmarks, num_coordinates, 1) 的三维数组
                landmarks = landmarks[:, :, np.newaxis]
                skeleton_output_path = os.path.join(skeleton_output_dir, f'frame_{frame_idx:04d}.npy')
                np.save(skeleton_output_path, landmarks)

                # 绘制骨骼连接
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = result.pose_landmarks.landmark[start_idx]
                    end_point = result.pose_landmarks.landmark[end_idx]
                    start_coords = (int(x1 + start_point.x * (x2 - x1)), int(y1 + start_point.y * (y2 - y1)))
                    end_coords = (int(x1 + end_point.x * (x2 - x1)), int(y1 + end_point.y * (y2 - y1)))
                    cv2.line(frame, start_coords, end_coords, (0, 255, 0), 2)

            # 用矩形框出人物
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 在每一帧上绘制红色水平中线
    midline = int(frame_height / 2)
    cv2.line(frame, (0, midline), (frame_width, midline), (0, 0, 255), 2)

    # 显示和保存视频帧
    cv2.imshow('Frame', frame)
    out.write(frame)

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()