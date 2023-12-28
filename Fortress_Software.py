import cv2
import torch
import numpy as np

class Yolo_Bracket:
    def __init__(self) -> None:
        pass
    def Load_Model():
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def Capture_Video(model):
        cap = cv2.VideoCapture(0)
        while True:
            # 读取摄像头的一帧
            ret, frame = cap.read()
            if not ret:
                break
            # 将图像转换为YOLO模型需要的格式
            results = model(frame)
            # 解析模型返回的结果
            for i in range(len(results.pred[0])):
                if results.names[int(results.pred[0][i][5])] == 'person':
                    xmin, ymin, xmax, ymax = results.pred[0][i][:4].int()
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            # 显示处理后的图像
            cv2.imshow('frame', frame)
            # 如果按下'q'键，退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    Yolo_Bracket.Capture_Video(Yolo_Bracket.Load_Model())