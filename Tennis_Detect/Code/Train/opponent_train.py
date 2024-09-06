import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
                cropped_frame = frame[:h//2, :, :]  # 裁剪上半部分
                frames.append(cropped_frame)
            
            cap.release()

            if len(frames) < 30:
                frames = frames * (30 // len(frames)) + frames[:30 % len(frames)]
            else:
                step = len(frames) // 30
                frames = frames[::step][:30]

            frames = [cv2.resize(frame, (224, 224)) for frame in frames]

            frames = [transforms.ToTensor()(frame) for frame in frames]

            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            frames = torch.stack(frames).permute(1, 0, 2, 3)

            return frames, label

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return torch.zeros((3, 30, 224, 224)), -1

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        torch.save(model.state_dict(), f'model_epoch_{epoch}_{epoch_acc:.4f}.pth')

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, device):
    model.eval()
    running_corrects = 0
    for inputs, labels in tqdm(dataloaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Test Acc: {acc:.4f}')
    return acc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TennisDataset(video_folder='D:\\Tennis_Detect\\Datasets\\Group(3) Video\\Totaldata\\Opponent\\Trial', transform=data_transforms)
val_dataset = TennisDataset(video_folder='D:\\Tennis_Detect\\Datasets\\Group(3) Video\\Totaldata\\Opponent\\Test', transform=data_transforms)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0),  # 减小批处理大小
    'val': DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)  # 减小批处理大小
}

model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)

model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=50)

test_model(model, dataloaders, device)