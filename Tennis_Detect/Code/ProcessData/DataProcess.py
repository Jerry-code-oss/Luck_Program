import cv2
import os
import numpy as np
from yolov5 import YOLOv5
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import re
from moviepy.editor import VideoFileClip
import math
class PicProcess():
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s'):
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
    def Sequence2pic():
        vc = cv2.VideoCapture(r"D:/Tennis_Detect/Datasets(Vid2Pic)/RaW/202.mp4")
        n = 1  # count
        timeF = 10  # pic per fps
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        i = 0
        while rval:
            rval, frame = vc.read()
            if (n % timeF == 0):  # per timeF fps store
                i += 1
                print(i)
                cv2.imwrite(r"D:/Tennis_Detect/Datasets(Vid2Pic)/RaW/2pic{}.jpg".format(i), frame)  # store
            n = n + 1
            cv2.waitKey(1)
        vc.release()

    def pic2HalfGrey():
        #path = "C:\\Users\\25573\\Desktop\\Program\\Tennis_Player_Detect\\Datasets(Vid2Pic)"
        #files = os.listdir(path)
        i=int(1)
        for i in range(3996):
            if  os.path.exists("D:/Tennis_Detect/Datasets(Vid2Pic)/RaW/2pic"+str(i)+".jpg") == False:
                next
            else:
                img = Image.open("D:/Tennis_Detect/Datasets(Vid2Pic)/RaW/2pic"+str(i)+".jpg")
                width, height = img.size
                #img = img.convert('L')
                box = (0,height/2,width,986)
                img = img.crop(box)
                img.save("D:/Tennis_Detect/Datasets(Vid2Pic)/After_Cut/Rf2Pic"+str(i)+".jpg")
                print("pic"+str(i)+"complete refine!")
    def Cut_Out_Athele(self, input_folder, output_folder):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for img_file in Path(input_folder).glob('*.jpg'):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            results = self.model(image)
            for i, det in enumerate(results.xyxy[0]):
                if results.names[int(det[5])] == 'person':
                    xmin, ymin, xmax, ymax = map(int, det[:4])
                    cropped_img = image[ymin:ymax, xmin:xmax]
                    save_path = os.path.join(output_folder, f'{img_file.stem}_person_{i}.jpg')
                    cv2.imwrite(save_path, cropped_img)
    def Pic2Vie(image_folder,video_name,fps=20):
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            video.release()

    def split_video(video_path, segment_length, output_folder):
        clip = VideoFileClip(video_path)
        duration = clip.duration
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        total_segments = int(math.floor(duration / segment_length))
        for i in range(total_segments):
            start_time = i * segment_length
            end_time = start_time + segment_length
            subclip = clip.subclip(start_time, end_time)
            output_path = os.path.join(output_folder, f"segment_{i + 1}.mp4")
            subclip.write_videofile(output_path, codec="libx264")
        if duration % segment_length > 0:
            start_time = total_segments * segment_length
            subclip = clip.subclip(start_time, duration)
            output_path = os.path.join(output_folder, f"segment_{total_segments + 1}.mp4")
            subclip.write_videofile(output_path, codec="libx264")

    def track_and_crop_person(input_video_path, output_video_path, model_size='yolov5s'):
        model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=True)
        if torch.cuda.is_available():
            model.cuda()
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_size = (300, 300)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, output_size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            lower_half_frame = frame[height//2:height, 0:width]
            results = model(lower_half_frame)
            predictions = results.xyxy[0]
            first_person_found = False
            for *box, conf, cls_id in predictions:
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box)
                    if not first_person_found:
                        person_crop = lower_half_frame[y1:y2, x1:x2]
                        person_crop_resized = cv2.resize(person_crop, output_size)
                        out.write(person_crop_resized)
                        first_person_found = True
                        break
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    #PicProcess.pic2HalfGrey()

    #PicProcess.Pic2Vie("D:/Tennis_Detect/Datasets(Vid2Pic)/(1)Raw_picset","D:/Tennis_Detect/Datasets(Vid2Pic)/(1)PicsetVideo.avi")

    #input_folder = 'D:/Tennis_Detect/Datasets(Vid2Pic)/After_Cut/Temp'
    #output_folder = 'D:/Tennis_Detect/Datasets(Vid2Pic)/After_Cut/Temp1'
    #yolo = PicProcess()
    #yolo.Cut_Out_Athele(input_folder, output_folder)

    video_path = 'D:\Tennis_Detect\Datasets\Group(3) Video\Totaldata\Opponent/no hit.mp4'
    segment_length = 5  # 每个片段的长度
    output_folder = 'D:/Tennis_Detect/Datasets/Group(3) Video/Totaldata/Opponent/Test/'
    PicProcess.split_video(video_path, segment_length, output_folder)

    #PicProcess.track_and_crop_person('D:/Tennis_Detect/Datasets/Group(3) Video/Totaldata/Trial/0.mp4', 'D:/Tennis_Detect/Datasets/Group(3) Video/Totaldata/Trial/0_Cut.mp4', 'yolov5s')
