# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import argparse
from mmcv.image import imread, imwrite

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the COCO dataset')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    parser.add_argument(
        '--output_dir',
        default='work_dirs_semi_supervision/balloon/semi_supervision/browse_dataset/',
        type=str,
        help='Visual image save path')
    parser.add_argument(
        '--only_bbox', 
        default=True, 
        help='whether to only visualize the bbox label(segmentation label will be ignored)')
    parser.add_argument(
        '--show_all', 
        default=True, 
        help='whether to show all categories, if not, use "category_name" to specified the category which need to visualize')
    parser.add_argument(
        '--category_name', 
        default='bicycle', 
        type=str,
        help='specified the category which need to visualize')
    
    parser.add_argument(
        '--data-root',    # 保存的json文件所在的文件夹路径
        default='work_dirs_semi_supervision/balloon/semi_supervision/',
        type=str,
        help='Visual image save path')
    parser.add_argument(
        '--ann-file',    # 保存的json文件
        default='test_result2json.bbox.json',
        type=str,
        help='Visual image save path')
    parser.add_argument(
        '--img-prefix',    # json文件对应的图片所在的文件夹路径
        default='data/balloon/val/',
        type=str,
        help='Visual image save path')
    
    args = parser.parse_args()
    return args

def showBBox(coco, anns, label_box=True, is_filling=True):
    """
    show bounding box of annotations or predictions
    anns: loadAnns() annotations or predictions subject to coco results format
    label_box: show background of category labels or not
    """
    if len(anns) == 0:
        return 0
    ax = plt.gca()    
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
    for ann in anns:
        c = image2color[ann['category_id']]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)
        if label_box:
            label_bbox = dict(facecolor=c)
        else:
            label_bbox = None
        if 'score' in ann:
            ax.text(bbox_x, bbox_y, '%s: %.2f' % (coco.loadCats(ann['category_id'])[0]['name'], ann['score']),
                    color='white', bbox=label_bbox)
        else:
            ax.text(bbox_x, bbox_y, '%s' % (coco.loadCats(ann['category_id'])[0]['name']), color='white',
                    bbox=label_bbox)
    if is_filling:
        # option for filling bounding box
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_coco(data_root, ann_file, img_prefix, only_bbox=True, show_all=True, category_name='bicycle', output_dir=None):
    example_coco = COCO(ann_file)
    print('图片总数：{}'.format(len(example_coco.getImgIds())))
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

    if show_all:
        category_ids = []
    else:
        category_ids = example_coco.getCatIds(category_name)
    image_ids = example_coco.getImgIds(catIds=category_ids)

    save_dir = output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(image_ids)):
        plt.figure()
        image_data = example_coco.loadImgs(image_ids[i])[0]
        path = os.path.join(img_prefix, image_data['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)
        if only_bbox:
            showBBox(example_coco, annotations)
        else:
            example_coco.showAnns(annotations)
        
        # output_dir = '/data/chenchao/MMDetection/customed_op_result/visualize_coco/' + image_data['file_name']
        # TODO:形式还需要优化
        output_dir = save_dir + image_data['file_name']
        plt.savefig(output_dir)
        plt.close()    # 不加的话会报错，打开的数量太多。。。
        # plt.show()



if __name__ == '__main__':
    args = parse_args()

    # data_root = 'data/balloon/'
    # ann_file = data_root + 'train/annotation_coco.json'
    # img_prefix = data_root + 'train/'
    
    # data_root = 'work_dirs_semi_supervision/balloon/semi_supervision/'
    # ann_file = data_root + 'test_result2json.bbox.json'
    # img_prefix = 'data/balloon/val/'
    
    data_root = args.data_root
    ann_file = data_root + args.ann_file
    img_prefix = args.img_prefix
    show_coco(data_root, ann_file, img_prefix, only_bbox=args.only_bbox, show_all=args.show_all, category_name=args.category_name, output_dir=args.output_dir)

    # # voc转化为coco后显示
    # data_root = '/home/pi/dataset/VOCdevkit/'
    # ann_file = data_root + 'annotations/voc0712_trainval.json'
    # img_prefix = data_root
    # show_coco(data_root, ann_file, img_prefix)
