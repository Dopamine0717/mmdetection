# model settings
model = dict(
    type='FCOS',  # 分类器类型
    backbone=dict(
        type='ResNet',  # 主干网络类型
        depth=50,  # 主干网网络深度， ResNet 一般有18, 34, 50, 101, 152 可以选择
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(0, 1, 2, 3),  # 输出的特征图输出索引。越远离输入图像，索引越大
        frozen_stages=1,  # 网络微调时，冻结网络的stage（训练时不执行反相传播算法），若num_stages=4，backbone包含stem 与 4 个 stages。frozen_stages为-1时，不冻结网络； 为0时，冻结 stem； 为1时，冻结 stem 和 stage1； 为4时，冻结整个backbone
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',  # 颈网络类型
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,  # 颈网络neck的输出通道数
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=5,  # 注意  输出类别数，这与数据集的类别数一致
        in_channels=256,  # 输入通道数，这与 neck 的输出通道一致
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(   # 损失函数配置信息
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
img_norm_cfg = dict(   #图像归一化配置，用来归一化输入的图像。
    mean=[103.530, 116.280, 123.675],   # 预训练里用于预训练主干网络模型的平均值。
    std=[1.0, 1.0, 1.0],   # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=False)   # 是否反转通道
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),   # 读取图片
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),  # 归一化
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # 决定数据中哪些键应该传递给检测器的流程
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
            dict(type='Collect', keys=['img']),    # test 时不传递 gt_label和gt_bboxes
        ])
]
data = dict(
    samples_per_gpu=6,  # 单个 GPU 的 Batch size
    workers_per_gpu=8,  # 单个 GPU 的 线程数
    train=dict(   # 训练数据信息
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_train.json',  # data1
        # '/data/DataSets/transmission_line_detection/instances_train30462.json',  # data2
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=train_pipeline,
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    val=dict(   # 验证数据集信息
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_test.json',
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=test_pipeline,
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    test=dict(
        type='OurDataset',
        ann_file=
        '/data/DataSets/transmission_line_detection/instances_test.json',  
        img_prefix='/data/DataSets/transmission_line_detection/',
        pipeline=test_pipeline,
        classes=('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao',
                 'YanHuo')),
    persistent_workers=True)
evaluation = dict(   # evaluation hook 的配置
    interval=1,   # 注意根据epoch修改  # 验证期间的间隔，单位为 epoch 或者 iter， 取决于 runner 类型。
    metric='bbox')    # 验证期间使用的指标。

optimizer = dict(type='SGD',   # 优化器类型
                lr=0.02,   # 优化器的学习率
                momentum=0.9,   # 动量(Momentum)
                weight_decay=0.0001)   # 权重衰减系数(weight decay)
optimizer_config = dict(grad_clip=None)  # 大多数方法不使用梯度限制(grad_clip)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,  # warmup_iters
    warmup_ratio=0.001,
    step=[12, 16])  #Lr Schuster 注意修改
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)   # 注意根据epoch修改  # 保存的间隔是 1，单位会根据 runner 不同变动，可以为 epoch 或者 iter
log_config = dict(
    interval=50,  # 打印日志的间隔， 单位 iters
    hooks=[dict(type='TextLoggerHook'),  # 用于记录训练过程的文本记录器(logger)
           dict(type='TensorboardLoggerHook')])  # 同样支持 Tensorboard 日志
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')   # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'  # checkpoint
resume_from = None  # None  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]   # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。
classes = ('DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao', 'YanHuo')
work_dir = 'work_dir_luo/fcos_lr0.02_epoch20'  # workdir
gpu_ids = range(0, 4)
