from PIL import Image
import os
import numpy as np
import shutil
maps_path = '/home/zh/Documents/workspace/scene/candidate'

image_list = []

for root, dirs, files in os.walk(maps_path):
    if files:
        for name in files:
            img_path = os.path.join(root, name)
            trav_map = np.array(Image.open(img_path))
            area = np.sum(trav_map==255)/10000
            image_list.append([name,np.sum(trav_map==255)/10000])
            if area<=40:
                shutil.copyfile(img_path,'/home/zh/Documents/workspace/scene/val/easy/'+name)
            if 40<area<=60:
                shutil.copyfile(img_path,'/home/zh/Documents/workspace/scene/val/middle/'+name)
            if area>60:
                shutil.copyfile(img_path,'/home/zh/Documents/workspace/scene/val/hard/'+name)
   
image_list.sort(key=lambda a:a[-1])
