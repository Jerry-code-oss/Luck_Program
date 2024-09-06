import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 使用yolov5s模型
yolo_model.eval()

model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)
model.load_state_dict(torch.load('D:/Tennis_Detect/Model/3dResnet50/Ellie_Best_Model.pth'))  # 替换为你的模型路径
model = model.to('cuda')
model.eval()

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

def process_video(input_video_path, output_video_path, yolo_model, model):
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

    frames_buffer = []
    buffer_size = 16 

    prev_box = None 
    movement_label = "Stationary"
    frame_count = 0

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            h_half = frame.shape[0] // 2
            cropped_frame = frame[h_half:, :, :]

            results = yolo_model(cropped_frame)
            detections = results.xyxy[0].cpu().numpy() 

            if len(detections) > 0:
                best_detection = detections[detections[:, 4].argmax()]
                x1, y1, x2, y2, conf, cls = map(int, best_detection)

                processed_frame = transform(cropped_frame).unsqueeze(0).to('cuda')

                frames_buffer.append(processed_frame)
                if len(frames_buffer) < buffer_size:
                    prev_box = (x1, y1, x2, y2)
                    continue
                
                inputs = torch.stack(frames_buffer, dim=2).squeeze(0) 
                outputs = model(inputs.unsqueeze(0))
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                label = 'Hitting' if preds.item() == 1 else 'Other'
                prob = probs[0, preds.item()].item()

                if label == 'Hitting':
                    current_box = (x1, y1, x2, y2)
                    if Is_Moving(current_box, prev_box):
                        movement_label = "Moving Hitting"
                    else:
                        movement_label = "Stationary Hitting"
                    prev_box = current_box

                    display_label = f'{movement_label} ({prob:.2f})'

                elif label == 'Other':
                    if prev_box is not None and frame_count % 16 == 0:
                        current_box = (x1, y1, x2, y2)
                        if Is_Moving(current_box, prev_box):
                            movement_label = "Moving"
                        else:
                            movement_label = "Stationary"
                        prev_box = current_box
                    display_label = movement_label

                cv2.rectangle(frame, (x1, y1 + h_half), (x2, y2 + h_half), (0, 255, 0), 2)
                cv2.putText(frame, display_label, (x1, y1 + h_half - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                frames_buffer.pop(0)

            out.write(frame)
    
    cap.release()
    out.release()
    print(f"saved{output_video_path}")

input_video_path = 'D:\Test.mp4'
output_video_path = 'D:/Tennis_Detect/demo.mp4'

process_video(input_video_path, output_video_path, yolo_model, model)