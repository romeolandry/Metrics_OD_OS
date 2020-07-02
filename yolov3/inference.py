
#from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import numpy as np
import os
from io import BytesIO

import tensorflow as tf

from timeit import default_timer as timer

from keras import backend as K
sess = K.get_session()

from image import *
from model import *
from model import _get_class


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataDir= '/home/kamgo/Donnees/Master_projekt/Data/val2017'
model_path= '../model_data/yolov3_OT.h5'
imageTpye = 'jpg'
imageFiles ='%s/*.%s'%(dataDir,imageTpye)
dataType='val2017'
annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
annFile_result = '%s/annotations/instances_predicted_result_%s.json'%(dataDir,dataType)


anchors_path= '../model_data/yolo_anchors.txt'
classes_path= '../model_data/coco_classes.txt'

def run_inference ():
    result = []
    yolo_model,boxes, scores, classes, input_image_shape  = generate(model_path,anchors_path,classes_path)
    maping_classes = generate_mapping_classes(_get_class(classes_path), annFile)

    for inferences_images in get_data(annFile,imageFiles):
        start = timer()
        image = inferences_images['image']
        image_data = preprocess(inferences_images['image'])


        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })    

        for i, c in reversed(list(enumerate(out_classes))):

            predicted_class = _get_class(classes_path)[c]
            #print(predicted_class)
            box = out_boxes[i]
            #print(box) 
            score = out_scores[i]
            #print(score)
            category_id = gen_result(annFile,inferences_images['image_id'],predicted_class,maping_classes)
            #print('predicted category: {}'.format(category_id))
            
            top, left, bottom, right = box
            
            # the coordinated is relatif to image in yolo
            
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))       
            
            bbox=[]
            bbox.append(top)
            bbox.append(left)
            bbox.append(bottom)
            bbox.append(right)
            print({"image_id":inferences_images['image_id'],"category_id":category_id,"bbox":bbox,"score":score})
            result.append({"image_id":inferences_images['image_id'],"category_id":category_id,"bbox":bbox,"score":score})    
            #print(label, (left, top), (right, bottom))
        end = timer()
        print(end - start)
    save_json(annFile_result, result)

if __name__ == "__main__":
    run_inference()