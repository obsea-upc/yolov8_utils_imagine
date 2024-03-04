from ultralytics import YOLO
import os
from rich import print
from tqdm import tqdm

# Load a pretrained YOLOv8n model
model = YOLO('./weigths/18sp_2301img_xlarge_lr0_000375_1920_workers6.pt')

path = '/home/polba/sarti/datasets/temp_train/images/'
im_list = [path + im for im in os.listdir(path)]

# new OBSEA imgsz=2688 × 1520
for path in tqdm(im_list):
    model.predict(path, save=True, save_txt=True, imgsz=2688, conf=0.5, line_width=2)
