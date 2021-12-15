# --coding=utf-8--
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../common/mstrain_3x_coco.py'
]

model = dict(
        bbox_head=dict(
            num_classes=5,
            anchor_generator=dict(ratios=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
            )
            )

dataset_type = 'COCODataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        img_prefix='data/transmission_line_detection/train',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/train16462.json'),
    val=dict(
        img_prefix='data/transmission_line_detection/test',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/test16462.json'),
    test=dict(
        img_prefix='data/transmission_line_detection/test',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/test16462.json'))

load_from = 'checkpoints/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=3)
evaluation = dict(interval=2, metric='bbox')
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
