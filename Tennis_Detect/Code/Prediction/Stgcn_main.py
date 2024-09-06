import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import torch.nn as nn

# 确认训练时的模型定义
class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(STGCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = x.mean(dim=-1)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x

# 加载训练好的模型
model_path = 'C:/Users/15811/Desktop/Skeleton/models/model_epoch_1.pth'  # 替换为你的模型路径
model_stgcn = STGCN(in_channels=2, num_classes=2)

# 加载状态字典并跳过缺失的键
state_dict = torch.load(model_path)
model_stgcn.load_state_dict(state_dict, strict=False)
model_stgcn.eval()

# 初始化 YOLOv5 模型
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 视频输入
video_path = 'C:/Users/15811/Desktop/skeleton/input/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频输出设置
output_path = 'C:/Users/15811/Desktop/skeleton/output/output_video_with_predictions.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编解码器
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

frame_idx = 0
batch_size = 5
frames = []

# 躯干及下半身的关键点索引
trunk_and_lower_body_indices = [
    0,  # NOSE
    11, 12,  # LEFT_SHOULDER, RIGHT_SHOULDER
    23, 24,  # LEFT_HIP, RIGHT_HIP
    25, 26,  # LEFT_KNEE, RIGHT_KNEE
    27, 28  # LEFT_ANKLE, RIGHT_ANKLE
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    if len(frames) < batch_size:
        continue

    predictions = []
    boxes = []
    skeletons = []

    for frame in frames:
        # 只处理视频的下半部分，同时忽略左边300像素和右边300像素
        lower_half = frame[frame_height//2:, 300:frame_width-300]

        # 使用 YOLOv5 检测人物
        results = model_yolo(lower_half)
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
                    x = landmark.x
                    y = landmark.y
                    landmarks.append([x, y])

                # 预测动作
                landmarks = np.array(landmarks)
                landmarks = landmarks / [1, 1]  # 标准化
                if landmarks.shape[0] == 33:  # 骨架数据的关节点数
                    landmarks = landmarks.transpose(1, 0)  # 转换为 (num_coordinates, num_joints)
                    landmarks = np.expand_dims(landmarks, axis=0)  # 添加一维度使其成为 (1, num_coordinates, num_joints)
                    landmarks = np.expand_dims(landmarks, axis=0)  # 添加一维度使其成为 (1, 1, num_coordinates, num_joints)
                    landmarks = np.repeat(landmarks, 2, axis=1)  # 重复以添加另一个通道
                    input_tensor = torch.tensor(landmarks, dtype=torch.float32)  # 转换为 torch 张量 (1, 2, num_coordinates, num_joints)
                    with torch.no_grad():
                        output = model_stgcn(input_tensor)
                        _, preds = torch.max(output, 1)
                        prediction = preds.item()
                        predictions.append(prediction)
                        boxes.append((x1, y1, x2, y2))
                        skeletons.append(result.pose_landmarks)

    # 对批次中的预测结果进行投票
    if predictions:
        prediction_counts = np.bincount(predictions)
        majority_prediction = np.argmax(prediction_counts)
        movement_threshold = 0.03  # 可以根据需要调整这个阈值
        total_movement = 0

        if len(skeletons) > 1:
            num_joints = len(trunk_and_lower_body_indices)
            for i in range(1, len(skeletons)):
                for idx in trunk_and_lower_body_indices:
                    total_movement += np.linalg.norm(
                        np.array([skeletons[i].landmark[idx].x, skeletons[i].landmark[idx].y]) -
                        np.array([skeletons[i-1].landmark[idx].x, skeletons[i-1].landmark[idx].y])
                    )

            avg_movement = total_movement / ((len(skeletons) - 1) * num_joints)
        else:
            avg_movement = 0

        if majority_prediction == 1:
            if avg_movement > movement_threshold:
                label = 'Moving Hitting'
            else:
                label = 'Standing Hitting'
        else:
            if avg_movement > movement_threshold:
                label = 'Moving'
            else:
                label = 'Standing'

        for frame, box, skeleton in zip(frames, boxes, skeletons):
            x1, y1, x2, y2 = box
            # 绘制人物的矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 在矩形框上方绘制标签
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # 绘制骨骼连接
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = skeleton.landmark[start_idx]
                end_point = skeleton.landmark[end_idx]
                start_coords = (int(x1 + start_point.x * (x2 - x1)), int(y1 + start_point.y * (y2 - y1)))
                end_coords = (int(x1 + end_point.x * (x2 - x1)), int(y1 + end_point.y * (y2 - y1)))
                cv2.line(frame, start_coords, end_coords, (0, 255, 0), 2)

            out.write(frame)

    frames = []

cap.release()
out.release()
cv2.destroyAllWindows()
