from rich import print
import os
from ultralytics import YOLO
import json


path = "/home/polba/workspace/yolov8_ws/ultralytics/yolov8_utils_imagine/runs/detect/"
print(os.path.isfile("/srv/yolov8_ws/ultralytics/results_validation.json"))
if os.path.isfile(path + "results_validation.json"):
    f = open("results_validation.json")
    dict = json.load(f)
    f.close()
else:
    dict = {}
for num_sp_num_img in os.listdir(path):
    print(num_sp_num_img)
    if os.path.isdir(path + num_sp_num_img) and num_sp_num_img[0] == "1":
        for model in os.listdir(path + num_sp_num_img + "/"):
            for lr in os.listdir(path + num_sp_num_img + "/" + model + "/"):
                for res in os.listdir(path + num_sp_num_img + "/" + model + "/" + lr + "/"):
                    print(path + num_sp_num_img + "/" + model + "/" + lr + "/" + res + "/workers0/autobatch/weights/")
                    if num_sp_num_img + "/" + model + "/" + lr + "/" + res in dict.keys():
                        print('[green]Done')
                        pass
                    else:
                        # Load a model
                        model_yolo = YOLO(path + num_sp_num_img + "/" + model + "/" + lr + "/" + res + "/workers0/autobatch/weights/best.pt")  # load a custom model
    
                        # Validate the model
                        metrics = model_yolo.val()  # no arguments needed, dataset and settings remembered
                        # metrics.box.map    # map50-95
                        # metrics.box.map50  # map50
                        # metrics.box.map75  # map75
                        # metrics.box.maps   # a list contains map50-95 of each category
                        # # print(metrics)
                        # print(metrics.box)
                        # print(metrics.box.map)
                        # print("--------------------------------------------------------------------")
                        # print(metrics.box.map50)
                        # print("--------------------------------------------------------------------")
                        # print(metrics.box.map75)
                        # print("--------------------------------------------------------------------")
                        # print(metrics.box.maps)
                        # print("--------------------------------------------------------------------")
                        dict[num_sp_num_img + "/" + model + "/" + lr + "/" + res] = {
                            "names": metrics.names,
                            "map50": metrics.box.ap50.tolist(),
                            "map50-95": metrics.box.ap.tolist(),
                            "confusion_matrix": metrics.box.all_ap.tolist(),
                            "map50_all": metrics.box.map50,
                            "map75_all": metrics.box.map75,
                            "map95_all": metrics.box.map,
                            "mp": metrics.box.mp.tolist(),
                            "mr": metrics.box.mr.tolist(),
                            "p": metrics.box.p.tolist(),
                            "r": metrics.box.r.tolist()  
                            # "metrics": metrics, 
                            # "metrics_box": metrics.box
                        }
                        # print(dict[num_sp_num_img + "/" + model + "/" + lr + "/" + res])
                        # print(json.dump(dict))
    
                        # with open("./results_validation.json", "w", encoding="utf-8") as f:
                        #     json.dumps(dict, f, ensure_ascii=False)
                        with open("./results_validation.json", "w") as f:
                            json.dump(dict, f)
                        del model_yolo
                        del metrics


