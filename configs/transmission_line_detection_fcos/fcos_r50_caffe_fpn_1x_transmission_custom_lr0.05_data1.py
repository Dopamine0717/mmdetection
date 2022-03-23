# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=5,  # 注意
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
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
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

dataset_type = 'OurDataset'
data_root = '/data/DataSets/transmission_line_detection/'
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
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_train.json',  # data1
        # '/data/DataSets/transmission_line_detection/instances_train30462.json',  # data2
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    val=dict(
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_test.json',
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    test=dict(
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_test.json',  
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),  #img_scale
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    persistent_workers=True)
evaluation = dict(interval=1, metric='bbox')  # 注意根据epoch修改interval

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,  # warmup_iters
    warmup_ratio=0.001,
    step=[12, 16])  #Lr Schuster 注意修改
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)  # 注意根据epoch修改interval
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'  # checkpoint
resume_from = None  # None
workflow = [('train', 1)]
classes = ('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao', 'YanHuo')
work_dir = 'work_dir_luo/fcos_lr0.05_epoch20'  # workdir
gpu_ids = range(0, 4)
