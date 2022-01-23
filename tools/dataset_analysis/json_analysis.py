import pycocotools
from pycocotools.coco import COCO


ann_train_file = '/shared/xjd/DataSets/transmission_line_detection/test_6cates_3490.json'
coco_train = COCO(ann_train_file)
print(len(coco_train.dataset['categories']))
print(len(coco_train.dataset['images']))
print(len(coco_train.dataset['annotations']))
