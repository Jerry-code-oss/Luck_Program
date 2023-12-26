import cv2
import torch
from torchvision import transforms

# yolo5s 预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# 定义视频源，0 通常是电脑的内置摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为模型的输入格式
    img = transforms.ToTensor()(frame)
    img = torch.unsqueeze(img, 0)

    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(img)

    # 绘制边框和标签
    for prediction in predictions[0]:
        if prediction[5] == 0 and prediction[4] >= 0.5:  # 类别 0 通常为人类
            x1, y1, x2, y2 = map(int, prediction[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Real-time Object Detection', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()