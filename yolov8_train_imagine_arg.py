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
args = parser.parse_args()

dict_pt_yolo = {
                'nano': 'yolov8n.pt',
                'small': 'yolov8s.pt',
                'medium': 'yolov8m.pt',
                'large': 'yolov8l.pt',
                'xlarge': 'yolov8x.pt'
               }

dataset = args.dataset
print(dataset)

model_name = args.yolo_model
print(model_name)

model_pt = dict_pt_yolo[model_name]
print(model_pt)

lr = float(args.lr)
print(lr, type(lr))

img_shp_x = args.img_shp_x
img_shp_y = args.img_shp_y
img_shp = [img_shp_x, img_shp_y]
print(img_shp)

workers = args.workers
print(workers)


def execution_train_complete(folder):
    path = './runs/detection' + folder
    if os.path.isfile(path + 'confusion_matrix.png'):
        return True
    else:
        if os.path.exists(path):
            os.remove(path)
        return False
        

# try:

print(f'[blue]Try with workers={workers}')
name = f'/{dataset}/{model_name}/lr{lr}/{img_shp[0]}/workers{workers}/autobatch/'.replace('.', '_')

if not execution_train_complete(name):

    task = Task.init(project_name='OBSEA', task_name=f'{name}')
    task.set_parameter('model', model_pt[:-3])

    # Load the model.
    model_yolov8 = YOLO(model_pt)

    args = dict(model=model_pt,
                # cfg='/srv/yolov8_ws/ultralytics/yolov8_utils_imagine/da.yaml',
                cfg='/home/polba/workspace/yolov8/ultralytics/yolov8_utils_imagine/da.yaml',
                data=f'/home/polba/workspace/yolov8/ultralytics/datasets/{dataset}/data.yaml',
                # data=f'/srv/yolov8_ws/ultralytics/datasets/{dataset}/data.yaml',
                epochs=1,
                patience=1,
                batch=-1,
                lr0=lr,
                workers=workers,
                imgsz=img_shp[0],
                name='.' + name
                )
    # args = dict(model=model_pt,
    #             # cfg='/srv/yolov8_ws/ultralytics/ultralytics/cfg/da.yaml',
    #             cfg='/home/polba/workspace/demo_serv_IMagine/ultralytics/ultralytics/cfg/da.yaml',
    #             data=f'/home/polba/workspace/demo_serv_IMagine/ultralytics/datasets/{dataset}/data.yaml',
    #             # data=f'/srv/yolov8_ws/ultralytics/datasets/{dataset}/data.yaml',
    #             epochs=200,
    #             patience=200,
    #             batch=-1,
    #             lr0=lr,
    #             workers=workers,
    #             imgsz=img_shp[0],
    #             name='.' + name
    #             )

    task.connect(args)

    # Training.
    results = model_yolov8.train(**args)
#     results = model_yolov8.train(
#         model=args['model'],
#         cfg=args['cfg'],
#         data=args['data'],
#         epochs=args['epochs'],
#         patience=args['patience'],
#         batch=args['batch'],
#         lr0=args['lr0'],
#         workers=args['workers'],
#         imgsz=args['imgsz'],
#         name=args['name']
#     )
# # except:
#     print(f'[red]Error with workers={workers}')
# python3 yolov8_train_imagine_arg.py 10sp_508img nano 0.000375 1920 1080 0

