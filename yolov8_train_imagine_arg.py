from ultralytics import YOLO
from rich import print
import os
import argparse
from clearml import Task

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Choose dataset into the path: './ultralitycs/datasets/", type=str)
parser.add_argument("yolo_model", help="Choose between one the options:\n\t- nano\n\t- small\n\t- medium\n\t- large\n\t- xlarge") #, type=str)
parser.add_argument("lr", help="Choose learning rate") #, type=float)
# parser.add_argument("img_shp", help="Choose image shape. Ex: '[1920, 1080]'", type=list)
parser.add_argument("img_shp_x", help="Choose image shape x.", type=int)
parser.add_argument("img_shp_y", help="Choose image shape y.", type=int)
parser.add_argument("workers", help="Choose num of workers", type=int)
parser.add_argument("seed", help="Choose num of seed", type=int)
args = parser.parse_args()

dict_pt_yolo = {
                'nano': 'yolov8n.pt',
                'small': 'yolov8s.pt',
                'medium': 'yolov8m.pt',
                'large': 'yolov8l.pt',
                'xlarge': 'yolov8x.pt'
               }

dataset = args.dataset
model_name = args.yolo_model
model_pt = dict_pt_yolo[model_name]
lr = float(args.lr)
img_shp_x = args.img_shp_x
img_shp_y = args.img_shp_y
img_shp = [img_shp_x, img_shp_y]
workers = args.workers
seed = args.seed


def execution_train_complete(folder):
    path = './runs/detection' + folder
    if os.path.isfile(path + 'confusion_matrix.png'):
        return True
    else:
        if os.path.exists(path):
            os.remove(path)
        return False


name = f'/{dataset}_{model_name}_lr{lr}_{img_shp[0]}_seed_{seed}_workers{workers}_autobatch/'.replace('.', '_')

if not execution_train_complete(name):

    task = Task.init(project_name='OBSEA', task_name=f'{name}')
    task.set_parameter('model', model_pt[:-3])

    # Load the model.
    model_yolov8 = YOLO(model_pt)

    args = dict(model=model_pt,
                # IMagine Docker Server
                # cfg='/srv/yolov8_ws/ultralytics/yolov8_utils_imagine/da.yaml',
                # data=f'/srv/yolov8_ws/ultralytics/datasets/{dataset}/data.yaml',
                # Works in Local
                cfg='/home/polba/workspace/yolov8/ultralytics/yolov8_utils_imagine/da.yaml',
                data=f'/home/polba/workspace/yolov8/ultralytics/datasets/{dataset}/data.yaml',
                epochs=200,
                patience=200,
                batch=-1,
                lr0=lr,
                workers=workers,
                imgsz=img_shp[0],
                name='.' + name,
                seed=seed
                )

    task.connect(args)
    results = model_yolov8.train(**args)
