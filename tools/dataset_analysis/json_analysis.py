import pycocotools
from pycocotools.coco import COCO


ann_train_file = '/shared/xjd/chenchao/mmdetection/work_dirs_semi_supervision/train14000/train14000.json'
coco_train = COCO(ann_train_file)
print(len(coco_train.dataset['categories']))
print(len(coco_train.dataset['images']))
print(len(coco_train.dataset['annotations']))
