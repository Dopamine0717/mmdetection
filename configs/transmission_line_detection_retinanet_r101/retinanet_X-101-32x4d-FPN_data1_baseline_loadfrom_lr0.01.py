# --coding=utf-8--

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    bbox_head=dict(
        num_classes=5,    # TODO:注意修改
        anchor_generator=dict(ratios=[0.25, 0.6, 1.0, 1.8, 3.0]),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0)
        ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.3),    # TODO:注意修改
        max_per_img=100)
            )


load_from = 'checkpoints/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1, metric='bbox', classwise=True, 
    jsonfile_prefix='work_dirs_r101/retinanet_X-101-32x4d-FPN_data1_baseline_loadfrom_lr0.01/evaluation')

log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

data_root = '/shared/xjd/DataSets/transmission_line_detection/'
dataset_type = 'OurDataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 900), keep_ratio=True),
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
        img_scale=(1200, 900),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
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
