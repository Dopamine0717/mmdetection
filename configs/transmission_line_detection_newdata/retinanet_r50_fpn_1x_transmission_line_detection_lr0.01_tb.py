# --coding=utf-8--

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

model = dict(
        bbox_head=dict(
            num_classes=5,
            anchor_generator=dict(ratios=[0.2, 0.5, 1.0, 2.0, 5.0])
            )
            )

dataset_type = 'COCODataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        img_prefix='data/transmission_line_detection/images16462',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/train16462.json'),
    val=dict(
        img_prefix='data/transmission_line_detection/images16462',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/test16462.json'),
    test=dict(
        img_prefix='data/transmission_line_detection/images16462',
        classes=classes,
        ann_file='/shared/xjd/DataSets/transmission_line_detection/test16462.json'))



# load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0005*4, weight_decay=0.01)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001,
    step=[16, 18])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1, metric='bbox')
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

train_pipeline = [
    dict(type='Resize', img_scale=(2666,1600), keep_ratio=True)
]
