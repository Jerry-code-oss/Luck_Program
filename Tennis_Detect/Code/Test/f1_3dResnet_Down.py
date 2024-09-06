import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torchvision import transforms
import cv2
import os
from torch.utils.data import DataLoader, Dataset


class TennisDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_paths = []
        self.labels = []

        for label in ['0', '1']:
            folder_path = os.path.join(video_folder, label)
            for video_name in sorted(os.listdir(folder_path)):
                video_path = os.path.join(folder_path, video_name)
                self.video_paths.append(video_path)
                self.labels.append(int(label)) 

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                #cropped_frame = frame[h//2:, :, :]  # 裁剪down half
                cropped_frame = frame[:h//2, :, :]  # 裁剪up half
                frames.append(cropped_frame)
            
            cap.release()

            if len(frames) < 16:
                frames = frames * (16 // len(frames)) + frames[:16 % len(frames)]
            else:
                step = len(frames) // 16
                frames = frames[::step][:16]

            frames = [cv2.resize(frame, (224, 224)) for frame in frames]

            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames]).permute(1, 0, 2, 3)

            if self.transform:
                frames = self.transform(frames)

            return frames, label

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None, None
      

def load_model(model_path, device):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score: {f1:.4f}")
    return f1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_dataset = TennisDataset(video_folder='D:\Tennis_Detect\Datasets\Group(3) Video\Totaldata\Opponent\Test')
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

model_path = 'D:/Tennis_Detect/Model/3dResnet50/Opponent_Best_Model.pth'
model = load_model(model_path, device)

evaluate_model(model, val_loader, device)
