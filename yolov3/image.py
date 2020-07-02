import json
from json import encoder
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
from model import *
import glob
""" import glob
import numpy as np
import os
from io import BytesIO """

def generate_mapping_classes(coco_classes, path_to_instance_val):
    val_json_categories = {}
    instances_classes = []
    mapping_list = []

    with open(path_to_instance_val) as f:
        d = json.load(f)
        val_json_categories = d['categories']
        
    for category in val_json_categories:
        instances_classes.append(category['name'])

    for val1, val2 in zip(coco_classes, instances_classes):
        mapping_list.append({val1:val2})

    return mapping_list

# return matching for one classe
def get_mapped(mapping_list, classe_name):

    for map in mapping_list:
        if(map.get(classe_name)):
            return map.get(classe_name)

def save_json(path_to_file, result):
    df = pd.DataFrame(result)
    # print(df)
    df.to_json(path_to_file,orient='records')

def gen_result(path_to_instance_val,image_id,category_name,maping_classes):
    
    val_json_categories = {}
    val_json_ann = {}
    categoy_id = None
    
    category_name = get_mapped(maping_classes, category_name)
    
    with open(path_to_instance_val) as f:
        d = json.load(f)
        val_json_categories = d['categories']
        val_json_ann = d['annotations']

    for category in val_json_categories:
        if (category['name']==category_name):
            categoy_id = category['id']
    return categoy_id 

def get_data(path_to_instance_val, path_to_images):
    inferenced_instance_val_2017 = []
    inferences_images = []    
    val_json_images = {}
    val_json_ann = {}
    
    with open(path_to_instance_val) as f:
        d = json.load(f)
        val_json_images = d['images']
        val_json_ann = d['annotations']

    for filename in glob.glob(path_to_images):
        im=Image.open(filename) #os.path.basename(your_path)
        #image_list.append({os.path.basename(filename):im})
        image_name = os.path.basename(filename)
        for image in val_json_images:
            if(image['file_name'] == image_name):
                #response = requests.get(image['coco_url'])
                #img = Image.open(BytesIO(response.content))
                inferences_images ={"image_id":image['id'],"image":im}    
                yield inferences_images

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    #image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data