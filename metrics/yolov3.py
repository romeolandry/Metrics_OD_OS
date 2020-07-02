import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

def main():

    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print ('Running demo for {0} results.'.format(annType))

    resFile = '/home/kamgo/Donnees/Master_projekt/Data/val2017 (1)/val2017/annotations/instances_result_yolov3_O_val2017.json'
    dataDir='/home/kamgo/Donnees/Master_projekt/Data/val2017 (1)/val2017'
    dataType='val2017'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)    

    #initialize COCO ground truth api
    cocoGt=COCO(annFile)

    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    main()