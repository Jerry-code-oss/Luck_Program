import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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

def random_horizontal_flip(skeleton_data, p=0.5):
    if np.random.rand() < p:
        skeleton_data[:, 0] = 1 - skeleton_data[:, 0]
    return skeleton_data

def random_rotation(skeleton_data, angle_range=10):
    angle = np.random.uniform(-angle_range, angle_range)
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    skeleton_data[:, :2] = np.dot(skeleton_data[:, :2], rotation_matrix)
    return skeleton_data

class SkeletonDataset(Dataset):
    def __init__(self, data_dir, frame_width, frame_height, transform=None):
        self.data_dir = data_dir
        self.data_files = []
        self.labels = []
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.transform = transform

        for label in ['0', '1']:
            class_dir = os.path.join(data_dir, label)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npy'):
                    self.data_files.append(os.path.join(class_dir, file_name))
                    self.labels.append(int(label))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        label = self.labels[idx]
        skeleton_data = np.load(data_file)
        
        # 归一化
        skeleton_data = skeleton_data / [self.frame_width, self.frame_height]

        if self.transform:
            skeleton_data = self.transform(skeleton_data)

        return torch.tensor(skeleton_data, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)

def custom_transform(skeleton_data):
    skeleton_data = random_horizontal_flip(skeleton_data)
    skeleton_data = random_rotation(skeleton_data)
    return skeleton_data

frame_width = 1280  # 替换为你的视频宽度
frame_height = 720  # 替换为你的视频高度

train_data_dir = 'C:/Users/15811/Desktop/Skeleton/train'  # 替换为你的训练数据路径
val_data_dir = 'C:/Users/15811/Desktop/Skeleton/test'  # 替换为你的验证数据路径
test_data_dir = 'C:/Users/15811/Desktop/Skeleton/test'  # 替换为你的测试数据路径

train_dataset = SkeletonDataset(train_data_dir, frame_width, frame_height, transform=custom_transform)
val_dataset = SkeletonDataset(val_data_dir, frame_width, frame_height)
test_dataset = SkeletonDataset(test_data_dir, frame_width, frame_height)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = STGCN(in_channels=2, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

model_save_dir = 'C:/Users/15811/Desktop/Skeleton/models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=10, min_delta=0.01)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        scheduler.step()
        
        model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Training complete')

def evaluate_model(model, test_loader):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    accuracy = corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=25)
evaluate_model(model, test_loader)
