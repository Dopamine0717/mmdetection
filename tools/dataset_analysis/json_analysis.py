import pycocotools
from pycocotools.coco import COCO


ann_train_file = '/data/DataSets/transmission_line_detection/train_7dirs.json'
coco_train = COCO(ann_train_file)
print(len(coco_train.dataset['categories']))
print(len(coco_train.dataset['images']))
print(len(coco_train.dataset['annotations']))
