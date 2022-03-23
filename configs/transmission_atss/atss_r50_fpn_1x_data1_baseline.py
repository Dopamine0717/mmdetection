# --coding=utf-8--

_base_ = '../atss/atss_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(num_classes=5))

load_from = 'checkpoints/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=400,
#     warmup_ratio=0.001,
#     step=[12, 16])
# runner = dict(type='EpochBasedRunner', max_epochs=20)
# checkpoint_config = dict(interval=5)
# evaluation = dict(interval=1, metric='bbox')
# log_config = dict(
#     interval=20,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook')
#     ])

data_root = '/shared/xjd/DataSets/transmission_line_detection/'
dataset_type = 'OurDataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'instances_train.json',    # data1:instances_train.json  data2:instances_train30462.json
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'instances_test.json',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'instances_test.json',
        classes=classes,
        pipeline=test_pipeline))
