from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from mmcv import Config
import numpy as np


# Specify the path to model config and checkpoint file
config_file = 'configs/RetinaNet_GanTa/retinanet_r50_fpn_1x_baseline.py'
checkpoint_file = 'GanTa/baseline/epoch_100.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# 可以检测一个文件夹中的所有图片
cfg = Config.fromfile(config_file)


img_path = 'data/failed_image'
img_list = os.listdir(img_path)
for img_file in img_list:
    img = os.path.join(img_path, img_file)
    result = inference_detector(model, img)
    img_save_path = os.path.join('data/test', img_file)
    # score_thr=cfg.model.test_cfg.score_thr
    model.show_result(img, result, score_thr=0.3, out_file=img_save_path)
    print(f'{os.path.basename(img_file)} finished!')





