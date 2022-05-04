# --coding=utf-8--

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale='dynamic')

model = dict(
    bbox_head=dict(
        num_classes=6,    # TODO:注意修改
        anchor_generator=dict(ratios=[0.25, 0.6, 1.0, 1.8, 3.0]),
        reg_decoded_bbox=True,    # 使用GIoUI时注意添加
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)
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


load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1600,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1, metric='bbox', classwise=True, 
    jsonfile_prefix='work_dirs_fp16/retinanet_train_7dirs_dynamic_fp16_lr0.01_bs6_GIoU_MixUp/evaluation')

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

data_root = '/data/DataSets/transmission_line_detection/'
dataset_type = 'OurDataset'
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo", "GanTa")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
  
# MixUp and Mosaic
img_scale = (1200, 900)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_7dirs.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# img_scale = (1200, 900)
# train_pipeline = [dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#                   dict(type='RandomAffine',scaling_ratio_range=(0.1, 2),border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#                   dict(type='MixUp',img_scale=img_scale,ratio_range=(0.8, 1.6),pad_val=114.0),
#                   dict(type='PhotoMetricDistortion',brightness_delta=32,contrast_range=(0.5, 1.5),saturation_range=(0.5, 1.5),hue_delta=18),
#                   dict(type='RandomFlip', flip_ratio=0.5),
#                   dict(type='Resize', keep_ratio=True),
#                   dict(type='Pad', pad_to_square=True, pad_val=114.0),
#                   dict(type='Normalize', **img_norm_cfg),
#                   dict(type='DefaultFormatBundle'),
#                   dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

# train_dataset = dict( 
#     type='MultiImageMixDataset', 
#     dataset=dict( 
#         type=dataset_type, 
#         ann_file=data_root + 'train_7dirs.json', 
#         img_prefix=data_root, 
#         pipeline=[ 
#             dict(type='CustomLoadImageFromFile'), 
#             dict(type='LoadAnnotations', with_bbox=True) 
#         ], 
#         filter_empty_gt=False, 
#     ), 
#     pipeline=train_pipeline,
#     dynamic_scale=img_scale) 

# # train_dataset = dict(
# #     type='MultiImageMixDataset',
# #     dataset=dict(
# #         type=dataset_type,
# #         ann_file=data_root + 'train_7dirs.json',
# #         img_prefix=data_root,
# #         classes=classes,
# #         pipeline=[
# #             dict(type='CustomLoadImageFromFile', to_float32=True),
# #             dict(type='LoadAnnotations', with_bbox=True)],
# #         filter_empty_gt=False,),
# #     pipeline=train_pipeline,
# #     dynamic_scale=img_scale)


# test_pipeline = [
#     dict(type='CustomLoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1200, 900),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset,
    # train=dict(
    #     type=dataset_type,
    #     img_prefix=data_root,
    #     ann_file=data_root + 'train_7dirs.json',    # data1:instances_train.json  data2:instances_train30462.json
    #     classes=classes,
    #     pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'test_7dirs.json',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + 'test_7dirs.json',
        classes=classes,
        pipeline=test_pipeline))

# initial
# train_pipeline = [
#     dict(type='CustomLoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1200, 900), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='MixUp', img_scale=(1200, 900)),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='CustomLoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1200, 900),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# 
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=8,
#     persistent_workers=True,
#     train=dict(
#         type=dataset_type,
#         img_prefix=data_root,
#         ann_file=data_root + 'train_7dirs.json',    # data1:instances_train.json  data2:instances_train30462.json
#         classes=classes,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         img_prefix=data_root,
#         ann_file=data_root + 'test_7dirs.json',
#         classes=classes,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         img_prefix=data_root,
#         ann_file=data_root + 'test_7dirs.json',
#         classes=classes,
#         pipeline=test_pipeline))