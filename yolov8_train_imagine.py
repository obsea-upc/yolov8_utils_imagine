import time
from ultralytics import YOLO, settings
from rich import print as print
import rich
import os
import subprocess as sp

datasets_list = ['19sp_2538img_17425annots']
# models_list = [['yolov8n.pt', 'nano'], ['yolov8s.pt', 'small'], ['yolov8m.pt', 'medium'],
#                ['yolov8l.pt', 'large'], ['yolov8x.pt', 'xlarge']]
models_list = [['yolov8n.pt', 'nano'], ['yolov8l.pt', 'large'], ['yolov8x.pt', 'xlarge']]
# lrs_list = [0.000375, 0.00075]
lrs_list = [0.000375]
# img_shapes_list = [[640, 360], [768, 432], [960, 540], [1280, 720], [1920, 1080]]
img_shapes_list = [[640, 360], [960, 540], [1920, 1080]]
seed_list = [86, 1714, 1998, 1999]

for dataset in datasets_list:
    for model in models_list:
        for lr in lrs_list:
            for img_shp in img_shapes_list:
                for seed in seed_list:
                    train_done = False
                    for workers in range(8, 0, -2):
                        if train_done:
                            pass
                        else:

                            settings.update({'clearml': True, 'mlflow': True})

                            print(f'Yolo v8 Settings: {settings}')
                            command = f'python3 yolov8_train_imagine_arg.py {dataset} {model[1]} {lr} {img_shp[0]} {img_shp[1]} {workers} {seed}'
                            print(f"[red] -------- RUNNING TRAINING AS SUBPROCESS -------")
                            print(command)

                            train_proc = sp.Popen(command.split(' '))
                            print(type(train_proc))

                            while train_proc.poll() is None:
                                print(f"[purple]still running")
                                import time
                                time.sleep(15)

                            if train_proc.poll() == 0:
                                print('[green]Finish correctly!')
                            elif train_proc.poll() == -9:
                                print("[purple]Program got killed! err -9, not memory RAM")
                            else:
                                print(f"[purple]Program ended for any other reason {train_proc.poll()}")
