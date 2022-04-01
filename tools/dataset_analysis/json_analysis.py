import pycocotools
from pycocotools.coco import COCO


ann_train_file = '/shared/xjd/DataSets/transmission_line_detection/data1_add_dxyw.json'
coco_train = COCO(ann_train_file)
print(len(coco_train.dataset['categories']))
print(len(coco_train.dataset['images']))
print(len(coco_train.dataset['annotations']))
