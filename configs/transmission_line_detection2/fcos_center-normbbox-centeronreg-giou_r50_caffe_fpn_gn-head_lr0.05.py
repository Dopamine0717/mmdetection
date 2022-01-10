_base_ = '../fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    bbox_head=dict(
        num_classes=5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))

# dataset settings
data_root = '/shared/xjd/DataSets/transmission_line_detection/'
dataset_type = 'OurDataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
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
    samples_per_gpu=16,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'instances_train.json',
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

load_from = 'checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth'
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=0.001,
    step=[8, 14])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1, metric='bbox')
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])