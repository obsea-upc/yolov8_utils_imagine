from ultralytics import YOLO, settings
import os
from rich import print
from tqdm import tqdm

# Load a pretrained YOLOv8n model
model = YOLO('./weigths/19sp_2538img_17425annots_xlarge_lr0_000375_1280_seed_1714_workers16_autobatch.pt')

path = '/home/polba/sarti/datasets/temp_train/images/'
im_list = [path + im for im in os.listdir(path)]
print(settings)
# new OBSEA imgsz=2688 × 1520
for path in tqdm(im_list):
    model.predict(path, save=True, save_txt=True, imgsz=1920, conf=0.5, line_width=2)
    # model.predict(path, save=True, save_txt=True, imgsz=2688, conf=0.5, line_width=2)

# ## video infeerence
# path = '/home/polba/sarti/datasets/Videos_ML/instagram_1_original.mp4'
# model.predict(path, save=True, save_txt=True, imgsz=1920, conf=0.5, line_width=2)
