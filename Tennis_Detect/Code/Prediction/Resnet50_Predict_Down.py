import cv2
import torch
import torchvision.models as models
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True).to(device)
pose_model = models.resnet50(pretrained=False)
num_features = pose_model.fc.in_features
pose_model.fc = torch.nn.Linear(num_features, 2)
model_weights = torch.load('D:/Tennis_Detect/model_epoch_10.pth', map_location=device)
pose_model.load_state_dict(model_weights)
pose_model = pose_model.to(device)
pose_model.eval()
cap = cv2.VideoCapture('D:/Tennis_Detect/(1)PicsetVideo.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter('Pytorch_Predicted_Video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
progress_bar = tqdm(total=total_frames, desc='Processing Video', unit='frame')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    half_frame = frame[height//2:901, 0:width]
    results = yolov5_model(half_frame)

    for *xyxy, conf, cls in results.xyxy[0]:
        xmin, ymin, xmax, ymax = map(int, xyxy)
        ymin += height // 2
        ymax += height // 2

        person_crop = frame[ymin:ymax, xmin:xmax]
        if person_crop.size == 0:
            continue
        person_tensor = torch.from_numpy(person_crop).permute(2, 0, 1).float().to(device)
        person_tensor = torch.unsqueeze(person_tensor, 0)

        with torch.no_grad():
            pose_prob = torch.softmax(pose_model(person_tensor), dim=1)
            pose_label = torch.argmax(pose_prob)

        label = f"Stand: {pose_prob[0][0]:.2f}" if pose_label == 0 else f"Hit: {pose_prob[0][1]:.2f}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    out.write(frame)
    progress_bar.update(1) 

cap.release()
out.release()
progress_bar.close()
print("completed!")
