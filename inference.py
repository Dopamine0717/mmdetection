from importlib_metadata import re
from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from mmcv import Config
import numpy as np

def dig2text(result, class_names):
    # if isinstance(result, tuple):
    #     bbox_result, segm_result = result
    #     if isinstance(segm_result, tuple):
    #         segm_result = segm_result[0]  # ms rcnn
    # else:
    #     bbox_result, segm_result = result, None

    bbox_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        print(label_text)

class_names = ['DaoXianYiWu', 'DiaoChe', 'ShiGongJiXie', 'TaDiao', 'YanHuo']


# Specify the path to model config and checkpoint file
# config_file = 'configs/transmission_line_detection3/retinanet_r50_fpn_1x_transmission_data2_softnms0.7.py'
# checkpoint_file = 'work_dirs3/data2_softnms0.7/latest.pth'
config_file = 'configs/transmission_line_detection_add_GanTa/retinanet_r50_fpn_1x_transmission_data1_softnms0.7_bbox_weight2.py'
checkpoint_file = 'work_dirs_add_GanTa/data1_softnms0.7_bbox_weight2/epoch_20.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'data/failed_image/5.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# # model.show_result(img, result)
# # or save the visualization results to image files
# img_save = os.path.join('data/detect_result', os.path.basename(img))
# model.show_result(img, result, out_file=img_save)

# 可以检测一个文件夹中的所有图片
cfg = Config.fromfile(config_file)


img_path = 'data/failed_image'
img_list = os.listdir(img_path)
for img_file in img_list:
    img = os.path.join(img_path, img_file)
    result = inference_detector(model, img)
    # dig2text(result, class_names)
    img_save_path = os.path.join('data/test_GanTa', img_file)
    # score_thr=cfg.model.test_cfg.score_thr
    model.show_result(img, result, score_thr=0.3, out_file=img_save_path)
    print(f'{os.path.basename(img_file)} finished!')





