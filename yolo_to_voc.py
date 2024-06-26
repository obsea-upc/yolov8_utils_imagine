import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join
from rich import print
import shutil

ROOT = '/home/polba/sarti/datasets/temp_train/'
YOLO_CLASSES = ('Chromis chromis', 'Chromis chromis (back)', 'Coris julis', 'Dentex dentex', 'Diplodus cervinus',
                'Diplodus puntazzo', 'Diplodus sargus', 'Diplodus vulgaris', 'Diver', 'Epinephelus costae',
                'Epinephelus marginatus', 'Mullus surmuletus', 'Muraena helena', 'Aetomylaeus bovinus', 'Oblada melanura',
                'Parablennius gattorugine', 'Sarpa salpa', 'Seriola dumerili', 'Serranus cabrilla', 'Sparus aurata',
                'Symphodus mediterraneus')

## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


# converts coco into xml
def xml_transform(root, classes):  
    class_path = join(root, 'labels_txt/')
    ids = list()
    list_txt = os.listdir(class_path)
    
    check = '.DS_Store' in list_txt
    if check == True:
        list_txt.remove('.DS_Store')
        
    ids = [x.split('.')[0] for x in list_txt]

    annopath = join(root, 'labels_txt', '%s.txt')
    imgpath = join(root, 'images', '%s.jpg')
    imgpath_jpeg = join(root, 'images', '%s.jpeg')
    imgpath_png = join(root, 'images', '%s.png')

    os.makedirs(join(root, 'outputs'), exist_ok=True)
    outpath = join(root, 'outputs', '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i] 
        if img_id == "classes":
            continue
        if os.path.exists(outpath % img_id):
            continue
        print(imgpath % img_id)
        # Copy img to outpath
        try:
            shutil.copyfile(imgpath % img_id, join(root, 'outputs', f'{img_id}.jpg'))
            img = cv2.imread(imgpath % img_id)

        except:
            print('[orange][WARN] The extension is not .jpg')

        try:
            shutil.copyfile(imgpath_jpeg % img_id, join(root, 'outputs', f'{img_id}.jpeg'))
            img = cv2.imread(imgpath_jpeg % img_id)

        except:
            print('[orange][WARN] The extension is not .jpeg')

        # try:
        #     shutil.copyfile(imgpath_png % img_id, join(root, 'outputs', f'{img_id}.png'))
        #     img = cv2.imread(imgpath_png % img_id)

        # except:
        #     print('[orange][WARN] The extension is not .png')

        height, width, channels = img.shape # pega tamanhos e canais das images

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'
        img_name = img_id + '.jpg'
    
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'Coco database'
        
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):

            # Number of predictions == label_norm
            label_norm= np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]
                
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[3])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)
                
        print(xml)  
        f = open(outpath % img_id, "wb")
        #f = open(os.path.join(outpath, img_id), "w")
        #os.remove(target)
        f.write(xml)
        f.close()     
       

xml_transform(ROOT, YOLO_CLASSES)
