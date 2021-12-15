# --coding=utf-8--

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'


model = dict(
        bbox_head=dict(
            num_classes=5,
            anchor_generator=dict(ratios=[0.2, 0.5, 1.0, 2.0, 5.0])
            )
        # test_cfg=dict(
        #     nms_pre=1000,
        #     min_bbox_size=0,
        #     score_thr=0.05,
        #     nms=dict(type='soft_nms', iou_threshold=0.5),
        #     max_per_img=100)
            )


dataset_type = 'COCODataset'
# classes = ("DiaoChe", "ShiGongJiXie", "YanHuo", "TaDiao", "DaoXianYiWu")
classes = ("DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo")
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    persistent_workers=True,
    train=dict(
        img_prefix='data/transmission_30462/train/',
        classes=classes,
        ann_file='data/transmission_30462/annotations/instances_train30462.json'),
    val=dict(
        img_prefix='data/transmission_30462/train/',
        classes=classes,
        ann_file='data/transmission_30462/annotations/instances_train30462.json'),
    test=dict(
        img_prefix='data/transmission_30462/test/',
        classes=classes,
        ann_file='data/transmission_30462/annotations/instances_test30462.json'))


# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0005*4, weight_decay=0.01)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[60, 80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='bbox')
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# train_pipeline = [
#     dict(type='Resize', img_scale=[(1333, 800),(2666,1600)], keep_ratio=True)
# ]
