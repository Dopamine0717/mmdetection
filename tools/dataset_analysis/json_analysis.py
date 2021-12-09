import pycocotools
from pycocotools.coco import COCO


ann_train_file = '/shared/xjd/chenchao/dataset/coco_custom/transmission_line_detection/annotations/instances_test.json'
coco_train = COCO(ann_train_file)
print(len(coco_train.dataset['categories']))
print(len(coco_train.dataset['images']))
print(len(coco_train.dataset['annotations']))
