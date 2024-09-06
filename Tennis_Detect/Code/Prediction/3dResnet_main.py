import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import os
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

model_opponent = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
model_opponent.blocks[-1].proj = nn.Linear(model_opponent.blocks[-1].proj.in_features, 2)
model_opponent.load_state_dict(torch.load('D:/Tennis_Detect/Model/3dResnet50/Opponent_Best_Model.pth'))
model_opponent = model_opponent.to('cuda')
model_opponent.eval()

model_player = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
model_player.blocks[-1].proj = nn.Linear(model_player.blocks[-1].proj.in_features, 2)
model_player.load_state_dict(torch.load('D:/Tennis_Detect/Model/3dResnet50/Ellie_Best_Model.pth'))
model_player = model_player.to('cuda')
model_player.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def Is_Moving(current_box, prev_box, threshold=10):
    x1_current, y1_current, x2_current, y2_current = current_box
    x1_prev, y1_prev, x2_prev, y2_prev = prev_box

    current_center = np.array([(x1_current + x2_current) / 2, (y1_current + y2_current) / 2])
    prev_center = np.array([(x1_prev + x2_prev) / 2, (y1_prev + y2_prev) / 2])

    movement_distance = np.linalg.norm(current_center - prev_center)

    if movement_distance > threshold:
        return True
    return False

def filter_hitting_data(hitting_frames, min_duration=2):
    filtered_data = []
    current_hitting = []
    
    for i in range(1, len(hitting_frames)):
        if hitting_frames[i] - hitting_frames[i - 1] == 1:
            current_hitting.append(hitting_frames[i - 1])
        else:
            if len(current_hitting) >= min_duration:
                current_hitting.append(hitting_frames[i - 1])
                filtered_data.append(current_hitting)
            current_hitting = []
    
    if len(current_hitting) >= min_duration:
        current_hitting.append(hitting_frames[-1])
        filtered_data.append(current_hitting)
    
    return filtered_data

def calculate_intervals(hitting_data):
    intervals = []
    for i in range(1, len(hitting_data)):
        intervals.append(hitting_data[i][0] - hitting_data[i - 1][-1])
    return intervals

def process_video(input_video_path, output_video_path, yolo_model, model_opponent, model_player):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frames_buffer_opponent = []
    frames_buffer_player = []
    buffer_size = 16

    prev_box_opponent = None
    prev_box_player = None
    movement_label_opponent = "Stationary"
    movement_label_player = "Stationary"
    frame_count = 0

    opponent_hitting_frames = []
    player_hitting_frames = []

    opponent_votes = []
    player_votes = []

    current_label_opponent = "Stationary"
    current_label_player = "Stationary"

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            h_half = frame.shape[0] // 2
            cropped_frame_opponent = frame[:h_half, :, :]
            cropped_frame_player = frame[h_half:, :, :]

            results_opponent = yolo_model(cropped_frame_opponent)
            detections_opponent = results_opponent.xyxy[0].cpu().numpy()

            results_player = yolo_model(cropped_frame_player)
            detections_player = results_player.xyxy[0].cpu().numpy()

            if len(detections_opponent) > 0:
                best_detection_opponent = None
                for detection in detections_opponent:
                    x1, y1, x2, y2, conf, cls = map(int, detection)
                    if y2 < h_half // 2:
                        continue
                    if best_detection_opponent is None or conf > best_detection_opponent[4]:
                        best_detection_opponent = detection
                
                if best_detection_opponent is not None:
                    x1, y1, x2, y2, conf, cls = map(int, best_detection_opponent)

                    processed_frame = transform(cropped_frame_opponent).unsqueeze(0).to('cuda')

                    frames_buffer_opponent.append(processed_frame)
                    if len(frames_buffer_opponent) < buffer_size:
                        prev_box_opponent = (x1, y1, x2, y2)
                        continue
                    
                    inputs = torch.stack(frames_buffer_opponent, dim=2).squeeze(0)
                    outputs = model_opponent(inputs.unsqueeze(0))
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)
                    label = 'Hitting' if preds.item() == 1 else 'Other'
                    prob = probs[0, preds.item()].item()

                    # 投票机制
                    opponent_votes.append(label)
                    if len(opponent_votes) == 5:
                        final_label = max(set(opponent_votes), key=opponent_votes.count)
                        current_label_opponent = final_label
                        opponent_votes = []

                    if current_label_opponent == 'Hitting':
                        opponent_hitting_frames.append(frame_idx)
                        current_box = (x1, y1, x2, y2)
                        if Is_Moving(current_box, prev_box_opponent):
                            movement_label_opponent = "Moving Hitting"
                        else:
                            movement_label_opponent = "Stationary Hitting"
                        prev_box_opponent = current_box
                        display_label = f'{movement_label_opponent} ({prob:.2f})'
                    else:
                        if prev_box_opponent is not None and frame_count % 16 == 0:
                            current_box = (x1, y1, x2, y2)
                            if Is_Moving(current_box, prev_box_opponent):
                                movement_label_opponent = "Moving"
                            else:
                                movement_label_opponent = "Stationary"
                            prev_box_opponent = current_box
                        display_label = movement_label_opponent

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    frames_buffer_opponent.pop(0)

            if len(detections_player) > 0:
                best_detection_player = detections_player[detections_player[:, 4].argmax()]
                x1, y1, x2, y2, conf, cls = map(int, best_detection_player)
                y1 += h_half
                y2 += h_half

                processed_frame = transform(cropped_frame_player).unsqueeze(0).to('cuda')

                frames_buffer_player.append(processed_frame)
                if len(frames_buffer_player) < buffer_size:
                    prev_box_player = (x1, y1, x2, y2)
                    continue
                
                inputs = torch.stack(frames_buffer_player, dim=2).squeeze(0)
                outputs = model_player(inputs.unsqueeze(0))
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                label = 'Hitting' if preds.item() == 1 else 'Other'
                prob = probs[0, preds.item()].item()

                player_votes.append(label)
                if len(player_votes) == 5:
                    final_label = max(set(player_votes), key=player_votes.count)
                    current_label_player = final_label
                    player_votes = []

                if current_label_player == 'Hitting':
                    player_hitting_frames.append(frame_idx)
                    current_box = (x1, y1, x2, y2)
                    if Is_Moving(current_box, prev_box_player):
                        movement_label_player = "Moving Hitting"
                    else:
                        movement_label_player = "Stationary Hitting"
                    prev_box_player = current_box
                    display_label = f'{movement_label_player} ({prob:.2f})'
                else:
                    if prev_box_player is not None and frame_count % 16 == 0:
                        current_box = (x1, y1, x2, y2)
                        if Is_Moving(current_box, prev_box_player):
                            movement_label_player = "Moving"
                        else:
                            movement_label_player = "Stationary"
                        prev_box_player = current_box
                    display_label = movement_label_player

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                frames_buffer_player.pop(0)

            out.write(frame)
    
    cap.release()
    out.release()
    
    filtered_opponent_hitting = filter_hitting_data(opponent_hitting_frames)
    filtered_player_hitting = filter_hitting_data(player_hitting_frames)

    opponent_intervals = calculate_intervals(filtered_opponent_hitting)
    player_intervals = calculate_intervals(filtered_player_hitting)

    with open('D:/Tennis_Detect/stats.txt', 'w') as f:
        f.write(f"Opponent hitting count: {len(filtered_opponent_hitting)}\n")
        f.write(f"Player hitting count: {len(filtered_player_hitting)}\n")
        f.write(f"Opponent hitting frames: {filtered_opponent_hitting}\n")
        f.write(f"Player hitting frames: {filtered_player_hitting}\n")
        f.write(f"Opponent hitting intervals: {opponent_intervals}\n")
        f.write(f"Player hitting intervals: {player_intervals}\n")
    
    print(f"Processed video saved at {output_video_path}")
    print(f"Statistics saved at D:/Tennis_Detect/stats.txt")

input_video_path = 'D:/Test.mp4'
output_video_path = 'D:/Tennis_Detect/demo_combined.mp4'

process_video(input_video_path, output_video_path, yolo_model, model_opponent, model_player)
