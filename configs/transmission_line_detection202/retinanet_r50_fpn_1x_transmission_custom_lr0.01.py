# --coding=utf-8--

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

model = dict(
        bbox_head=dict(
            num_classes=5,
            anchor_generator=dict(ratios=[0.2, 0.5, 1.0, 2.0, 5.0])
            )
            )


load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=0.001,
    step=[30, 40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5, metric='bbox')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


dataset_type = 'COCODataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        img_prefix='data/tranmission_16462/transmission_line_detection/train',
        classes=classes,
        ann_file='data/tranmission_16462/transmission_line_detection/train16462.json'),
    val=dict(
        img_prefix='data/tranmission_16462/transmission_line_detection/test',
        classes=classes,
        ann_file='data/tranmission_16462/transmission_line_detection/test16462.json'),
    test=dict(
        img_prefix='data/tranmission_16462/transmission_line_detection/test',
        classes=classes,
        ann_file='data/tranmission_16462/transmission_line_detection/test16462.json'))

