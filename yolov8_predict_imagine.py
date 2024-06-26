from ultralytics import YOLO, settings
import os
from rich import print
from tqdm import tqdm

# Load a pretrained YOLOv8n model
# model = YOLO('./weigths/21sp_3946img_24653annots_xlarge_lr0_000375_2560_seed_1998_workers0_autobatch.pt')
# model = YOLO('./weigths/multiclass_fish_segmentation_large.pt')
# model = YOLO('./weigths/21sp_4122img_26467annots_extensive_lr0_000375_2560_seed_1998_workers0_batch1.pt')
model = YOLO('./weigths/21sp_3946img_24653annots_xlarge_lr0_000375_2560_seed_1998_workers0_autobatch.pt')
print(model.names)

path = '/home/polba/sarti/datasets/temp_train/images/'
path = '/home/polba/Pictures/'
im_list = [path + im for im in os.listdir(path)]
# new OBSEA imgsz=2688 × 1520
for path in tqdm(im_list):
    model.predict(path, save=True, save_txt=True, imgsz=1920, conf=0.5, line_width=2)
    # model.predict(path, save=True, save_txt=True, imgsz=2688, conf=0.5, line_width=2)
#
# ## video infeerence
# path = '/home/polba/sarti/datasets/Videos_ML/instagram_1_original.mp4'
# model.predict(path, save=True, save_txt=True, imgsz=1920, conf=0.5, line_width=2)
