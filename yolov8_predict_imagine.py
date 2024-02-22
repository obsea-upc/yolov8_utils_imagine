from ultralytics import YOLO
import os
from rich import print
from tqdm import tqdm

# Load a pretrained YOLOv8n model
model = YOLO('./weigths/18sp_2301img_xlarge_lr0_000375_1920_workers6.pt')

path = '/home/polba/sarti/datasets/temp_train/images/'
im_list = [path + im for im in os.listdir(path)]

for path in tqdm(im_list):
    model.predict(path, save=True, save_txt=True, imgsz=1920, conf=0.25, line_width=2)
