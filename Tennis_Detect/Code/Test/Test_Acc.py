import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.video_list = [os.path.join(video_dir, fname) for fname in os.listdir(video_dir)]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        frames = torch.stack(frames)
        label = int(os.path.basename(self.video_dir))
        return frames, label

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            videos, labels = data
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    model1_path = 'path/to/model1.pth'
    model2_path = 'path/to/model2.pth'
    test1_dir_0 = 'path/to/test1/0'
    test1_dir_1 = 'path/to/test1/1'
    test2_dir_0 = 'path/to/test2/0'
    test2_dir_1 = 'path/to/test2/1'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)) 
    ])

    test1_dataset_0 = VideoDataset(test1_dir_0, transform=transform)
    test1_dataset_1 = VideoDataset(test1_dir_1, transform=transform)
    test2_dataset_0 = VideoDataset(test2_dir_0, transform=transform)
    test2_dataset_1 = VideoDataset(test2_dir_1, transform=transform)

    test1_dataset = torch.utils.data.ConcatDataset([test1_dataset_0, test1_dataset_1])
    test2_dataset = torch.utils.data.ConcatDataset([test2_dataset_0, test2_dataset_1])

    test1_loader = DataLoader(test1_dataset, batch_size=1, shuffle=False)
    test2_loader = DataLoader(test2_dataset, batch_size=1, shuffle=False)

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    accuracy1 = evaluate_model(model1, test1_loader)
    accuracy2 = evaluate_model(model2, test2_loader)

    print(f'Model 1 Accuracy: {accuracy1 * 100:.2f}%')
    print(f'Model 2 Accuracy: {accuracy2 * 100:.2f}%')

if __name__ == "__main__":
    main()
