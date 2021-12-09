# The new config inherits a base config to highlight the necessary modification
_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        num_classes=1))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='data/balloon/train/',
        classes=classes,
        ann_file='data/balloon/train/annotation_coco.json'),
    val=dict(
        img_prefix='data/balloon/val/',
        classes=classes,
        ann_file='data/balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='data/balloon/val/',
        classes=classes,
        ann_file='data/balloon/val/annotation_coco.json'))
